"""SPLADE sparse retriever for Mandol memory units."""

from __future__ import annotations

import logging
import os
import pickle
import threading
import traceback
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union, TYPE_CHECKING

import numpy as np
import torch
from safetensors.torch import load_file as safetensors_load_file, save_file as safetensors_save_file

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    njit = None

from .retrieval_interface import BaseRetriever, RetrievalInterface, RetrievalMethod, RetrievalResult
from .query_bundle import QueryBundle
from .cuda_stream_utils import get_thread_local_cuda_stream
from ..core.memory_unit import MemoryUnit
from ..utils.model_manager import global_model_manager
from ..utils.logging_config import create_module_logger

if TYPE_CHECKING:
    from ..storage.duckdb_operator import DuckDBOperator

logger = create_module_logger("splade_retriever")


if NUMBA_AVAILABLE:
    @njit(fastmath=True, nogil=True, cache=True)
    def _numba_materialize_candidate_query_matrix(
        cand_ids: np.ndarray,
        sorted_doc_ids: np.ndarray,
        doc_indptr: np.ndarray,
        flat_tids: np.ndarray,
        flat_weights: np.ndarray,
        sorted_q_tids: np.ndarray,
        q_order: np.ndarray,
        vocab_size: int,
        mat: np.ndarray,
    ) -> int:
        hit_count = 0
        n_docs = sorted_doc_ids.size
        n_qterm = sorted_q_tids.size
        for row in range(cand_ids.size):
            int_id = cand_ids[row]

            lo = 0
            hi = n_docs
            doc_pos = -1
            while lo < hi:
                mid = (lo + hi) // 2
                mid_value = sorted_doc_ids[mid]
                if mid_value < int_id:
                    lo = mid + 1
                elif mid_value > int_id:
                    hi = mid
                else:
                    doc_pos = mid
                    break
            if doc_pos < 0:
                continue

            start = doc_indptr[doc_pos]
            end = doc_indptr[doc_pos + 1]
            for cursor in range(start, end):
                tid = flat_tids[cursor]
                if tid < 0 or tid >= vocab_size:
                    continue

                q_lo = 0
                q_hi = n_qterm
                q_pos = -1
                while q_lo < q_hi:
                    q_mid = (q_lo + q_hi) // 2
                    q_value = sorted_q_tids[q_mid]
                    if q_value < tid:
                        q_lo = q_mid + 1
                    elif q_value > tid:
                        q_hi = q_mid
                    else:
                        q_pos = q_mid
                        break
                if q_pos >= 0:
                    mat[row, q_order[q_pos]] = flat_weights[cursor]
                    hit_count += 1
        return hit_count

    @njit(fastmath=True, nogil=True, cache=True)
    def _numba_fill_term_matrix_scratch(
        doc_ids: np.ndarray,
        weights: np.ndarray,
        row_lookup: np.ndarray,
        stamps: np.ndarray,
        generation: int,
        capacity: int,
        column: int,
        mat: np.ndarray,
    ) -> int:
        hits = 0
        for idx in range(doc_ids.size):
            doc_id = doc_ids[idx]
            if doc_id >= 0 and doc_id < capacity and stamps[doc_id] == generation:
                row = row_lookup[doc_id]
                mat[row, column] = weights[idx]
                hits += 1
        return hits

    @njit(fastmath=True, nogil=True, cache=True)
    def _numba_fill_term_matrix_searchsorted(
        doc_ids: np.ndarray,
        weights: np.ndarray,
        sorted_ids: np.ndarray,
        order: np.ndarray,
        column: int,
        mat: np.ndarray,
    ) -> int:
        hits = 0
        n_sorted = sorted_ids.size
        for idx in range(doc_ids.size):
            doc_id = doc_ids[idx]
            lo = 0
            hi = n_sorted
            pos = -1
            while lo < hi:
                mid = (lo + hi) // 2
                mid_value = sorted_ids[mid]
                if mid_value < doc_id:
                    lo = mid + 1
                elif mid_value > doc_id:
                    hi = mid
                else:
                    pos = mid
                    break
            if pos >= 0:
                mat[order[pos], column] = weights[idx]
                hits += 1
        return hits

    @njit(nogil=True, cache=True)
    def _numba_accumulate_scores_scratch(
        doc_ids: np.ndarray,
        weights: np.ndarray,
        row_lookup: np.ndarray,
        stamps: np.ndarray,
        generation: int,
        capacity: int,
        query_weight: float,
        scores: np.ndarray,
    ) -> int:
        hits = 0
        for idx in range(doc_ids.size):
            doc_id = doc_ids[idx]
            if doc_id >= 0 and doc_id < capacity and stamps[doc_id] == generation:
                row = row_lookup[doc_id]
                scores[row] += weights[idx] * query_weight
                hits += 1
        return hits

    @njit(nogil=True, cache=True)
    def _numba_accumulate_scores_searchsorted(
        doc_ids: np.ndarray,
        weights: np.ndarray,
        sorted_ids: np.ndarray,
        order: np.ndarray,
        query_weight: float,
        scores: np.ndarray,
    ) -> int:
        hits = 0
        n_sorted = sorted_ids.size
        for idx in range(doc_ids.size):
            doc_id = doc_ids[idx]
            lo = 0
            hi = n_sorted
            pos = -1
            while lo < hi:
                mid = (lo + hi) // 2
                mid_value = sorted_ids[mid]
                if mid_value < doc_id:
                    lo = mid + 1
                elif mid_value > doc_id:
                    hi = mid
                else:
                    pos = mid
                    break
            if pos >= 0:
                scores[order[pos]] += weights[idx] * query_weight
                hits += 1
        return hits
else:
    _numba_materialize_candidate_query_matrix = None
    _numba_fill_term_matrix_scratch = None
    _numba_fill_term_matrix_searchsorted = None
    _numba_accumulate_scores_scratch = None
    _numba_accumulate_scores_searchsorted = None


class SPLADERetriever(BaseRetriever):
    """Sparse neural retriever backed by SPLADE token-weight postings.

    SPLADERetriever stores sparse token postings keyed by stable Mandol integer
    IDs. The same IDs are used by memory-space filters and persisted sparse
    indexes, which allows candidate restriction without rebuilding embeddings.
    """

    _DENSE_LOCAL_LOOKUP_INITIAL_CAPACITY = 1_000_001
    _DENSE_LOCAL_LOOKUP_MAX_RATIO = 256
    _STATIC_ROW_SLICE_MAX_FRACTION = 0.25

    def __init__(
        self,
        retrieval_source: RetrievalInterface,
        default_text_field: str = "text_content",
        model_name: str = "naver/splade-v3",
    ):
        """Initialize the SPLADE retriever.

        Args:
            retrieval_source: SemanticMap-like source that provides memory units
                and filtering APIs.
            default_text_field: Raw-data field used as the preferred text body.
            model_name: SPLADE model identifier resolved by the global model
                manager.
        """
        self.retrieval_source = retrieval_source
        self.default_text_field = default_text_field
        self.model_name = model_name

        
        # token_id -> {int_id: weight}
        self.inverted_index: Dict[int, Dict[int, float]] = {}
        
        self.doc_postings: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self._posting_arrays: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self._dirty_tokens: Set[int] = set()
        self._dirty_token_appends: Dict[int, Set[int]] = {}
        self._dirty_token_rebuilds: Set[int] = set()
        self.total_docs: int = 0
        self._index_built: bool = False

        self._local_uid_to_int_id: Dict[str, int] = {}
        self._local_int_id_to_uid: Dict[int, str] = {}
        self._local_next_int_id: int = 0

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._use_gpu = torch.cuda.is_available()
        self._gpu_materialize_threshold: int = 1_048_576
        self._pinned_h2d_cache = threading.local()
        self._materialize_scratch = threading.local()
        self._candidate_lookup_scratch = threading.local()
        self._flat_doc_postings_dirty = True
        self._flat_doc_int_ids = np.empty(0, dtype=np.int64)
        self._flat_doc_indptr = np.zeros(1, dtype=np.int64)
        self._flat_doc_tids = np.empty(0, dtype=np.int32)
        self._flat_doc_weights = np.empty(0, dtype=np.float32)

        
        self._static_mode = False
        self._static_matrix = None
        self._static_int_ids: List[int] = []
        self._static_int_id_array: Optional[np.ndarray] = None

        
        try:
            logger.info(f"Loading SPLADE model through the global model manager: {model_name}")
            self.model = global_model_manager.get_splade_model(model_name)
        except Exception as e:
            logger.error(f"Failed to load SPLADE model: {e}")
            self.model = None

        self.vocab_size: int = self._detect_vocab_size()
        logger.info(f"Detected SPLADE vocabulary size: {self.vocab_size}")

    
    
    

    @property
    def _row_to_uid(self) -> List[str]:
        return [uid for uid in (self._internal_id_to_uid(int_id) for int_id in self.doc_postings.keys()) if uid]

    @property
    def _uid_to_row(self) -> Dict[str, int]:
        result: Dict[str, int] = {}
        for row, int_id in enumerate(self.doc_postings.keys()):
            uid = self._internal_id_to_uid(int_id)
            if uid is not None:
                result[uid] = row
        return result

    
    

    def _mapping_owner(self):
        source = self.retrieval_source
        if hasattr(source, "_get_or_create_int_id"):
            return source
        semantic_map = getattr(source, "semantic_map", None)
        if semantic_map is not None and hasattr(semantic_map, "_get_or_create_int_id"):
            return semantic_map
        return None

    def _legacy_uid_to_int_id_map(self) -> Optional[Dict[str, int]]:
        source = self.retrieval_source
        mapping = getattr(source, "_uid_to_internal_faiss_id", None)
        if isinstance(mapping, dict):
            return mapping
        semantic_map = getattr(source, "semantic_map", None)
        mapping = getattr(semantic_map, "_uid_to_internal_faiss_id", None)
        if isinstance(mapping, dict):
            return mapping
        return None

    def _get_or_create_internal_id(self, uid: str) -> int:
        owner = self._mapping_owner()
        if owner is not None:
            return int(owner._get_or_create_int_id(uid))
        uid = str(uid)
        legacy_mapping = self._legacy_uid_to_int_id_map()
        if legacy_mapping is not None:
            existing = legacy_mapping.get(uid)
            if existing is not None:
                return int(existing)
            int_id = len(legacy_mapping) + 1
            legacy_mapping[uid] = int_id
            return int_id
        existing = self._local_uid_to_int_id.get(uid)
        if existing is not None:
            return existing
        int_id = self._local_next_int_id
        self._local_next_int_id += 1
        self._local_uid_to_int_id[uid] = int_id
        self._local_int_id_to_uid[int_id] = uid
        return int_id

    def _uid_to_internal_id(self, uid: str, create: bool = False) -> Optional[int]:
        owner = self._mapping_owner()
        if owner is not None:
            if create:
                return int(owner._get_or_create_int_id(uid))
            mapping = owner._get_uid_to_int_id_map() if hasattr(owner, "_get_uid_to_int_id_map") else getattr(owner, "_uid_to_int_id", {})
            int_id = mapping.get(str(uid))
            return int(int_id) if int_id is not None else None
        legacy_mapping = self._legacy_uid_to_int_id_map()
        if legacy_mapping is not None:
            if create:
                return self._get_or_create_internal_id(uid)
            int_id = legacy_mapping.get(str(uid))
            return int(int_id) if int_id is not None else None
        if create:
            return self._get_or_create_internal_id(uid)
        int_id = self._local_uid_to_int_id.get(str(uid))
        return int(int_id) if int_id is not None else None

    def _uids_to_internal_ids(self, uids: List[str], create: bool = False) -> List[int]:
        int_ids: List[int] = []
        for uid in uids:
            int_id = self._uid_to_internal_id(uid, create=create)
            if int_id is not None:
                int_ids.append(int_id)
        return int_ids

    def _normalise_space_names_for_owner(self, owner: Any, space_names: List[str]) -> List[str]:
        memory_spaces = getattr(owner, "memory_spaces", None)
        normalized: List[str] = []
        for name in space_names:
            if (
                isinstance(memory_spaces, dict)
                and isinstance(name, str)
                and name in memory_spaces
            ):
                normalized.append(name)
            elif isinstance(name, str) and name.startswith("ms:"):
                normalized.append(name[3:])
            else:
                normalized.append(str(name))
        return normalized

    def _get_space_filter_internal_ids(self, space_names: List[str]) -> Optional[np.ndarray]:
        """Return cached global int ids for a pure space filter when the source supports it."""
        owners = [self._mapping_owner(), getattr(self.retrieval_source, "semantic_map", None)]
        seen_owner_ids: Set[int] = set()
        for owner in owners:
            if owner is None or id(owner) in seen_owner_ids:
                continue
            seen_owner_ids.add(id(owner))
            getter = getattr(owner, "_get_space_filter_int_ids", None)
            if not callable(getter):
                continue
            try:
                normalized_names = self._normalise_space_names_for_owner(owner, space_names)
                int_ids = getter(normalized_names)
                if int_ids is None:
                    continue
                return np.ascontiguousarray(np.asarray(int_ids, dtype=np.int64))
            except Exception as exc:
                logger.debug(f"SPLADE cached space int-id filter unavailable: {exc}")
        return None

    def _internal_id_to_uid(self, int_id: int) -> Optional[str]:
        owner = self._mapping_owner()
        if owner is not None:
            mapping = owner._get_int_id_to_uid_map() if hasattr(owner, "_get_int_id_to_uid_map") else getattr(owner, "_int_id_to_uid", {})
            return mapping.get(int(int_id))
        legacy_mapping = self._legacy_uid_to_int_id_map()
        if legacy_mapping is not None:
            target = int(int_id)
            for uid, mapped_int_id in legacy_mapping.items():
                if int(mapped_int_id) == target:
                    return str(uid)
        return self._local_int_id_to_uid.get(int(int_id))

    def _current_uid_to_int_id_map(self) -> Dict[str, int]:
        owner = self._mapping_owner()
        if owner is not None and hasattr(owner, "_get_uid_to_int_id_map"):
            return {str(uid): int(int_id) for uid, int_id in owner._get_uid_to_int_id_map().items()}
        legacy_mapping = self._legacy_uid_to_int_id_map()
        if legacy_mapping is not None:
            return {str(uid): int(int_id) for uid, int_id in legacy_mapping.items()}
        return dict(self._local_uid_to_int_id)

    def _restore_mapping_from_payload(self, mapping: Optional[Dict[str, Any]]) -> None:
        if not mapping:
            return
        owner = self._mapping_owner()
        if owner is not None and hasattr(owner, "_set_uid_int_mapping") and not owner._get_uid_to_int_id_map():
            owner._set_uid_int_mapping(mapping)
            return
        legacy_mapping = self._legacy_uid_to_int_id_map()
        if legacy_mapping is not None and not legacy_mapping:
            legacy_mapping.update({str(uid): int(int_id) for uid, int_id in mapping.items()})
            return
        if owner is None:
            uid_to_int_id: Dict[str, int] = {}
            int_id_to_uid: Dict[int, str] = {}
            next_int_id = 0
            for uid_raw, int_id_raw in mapping.items():
                uid = str(uid_raw)
                int_id = int(int_id_raw)
                uid_to_int_id[uid] = int_id
                int_id_to_uid[int_id] = uid
                if int_id >= next_int_id:
                    next_int_id = int_id + 1
            self._local_uid_to_int_id = uid_to_int_id
            self._local_int_id_to_uid = int_id_to_uid
            self._local_next_int_id = next_int_id

    def _get_unit_by_internal_id(self, int_id: int) -> Optional[MemoryUnit]:
        uid = self._internal_id_to_uid(int_id)
        if uid is None:
            return None
        return self.retrieval_source.get_unit(uid) if hasattr(self.retrieval_source, "get_unit") else None

    def _uid_exists(self, uid: str) -> bool:
        if hasattr(self.retrieval_source, "get_unit") and self.retrieval_source.get_unit(uid) is not None:
            return True
        memory_units = getattr(self.retrieval_source, "memory_units", None)
        if isinstance(memory_units, dict) and uid in memory_units:
            return True
        semantic_map = getattr(self.retrieval_source, "semantic_map", None)
        return bool(semantic_map is not None and uid in getattr(semantic_map, "memory_units", {}))

    def _coerce_doc_key_to_int_id(self, doc_key: Any) -> Optional[int]:
        if isinstance(doc_key, (int, np.integer)):
            return int(doc_key)
        if isinstance(doc_key, str):
            mapped = self._uid_to_internal_id(doc_key, create=False)
            if mapped is not None:
                return mapped
            if doc_key.lstrip("-").isdigit() and not self._uid_exists(doc_key):
                return int(doc_key)
            return self._get_or_create_internal_id(doc_key)
        return None

    def get_method_type(self) -> RetrievalMethod:
        return RetrievalMethod.SPLADE

    
    

    def _detect_vocab_size(self) -> int:
        """Detect vocab size."""
        default_size = 30522
        if not self.model:
            return default_size
        try:
            if hasattr(self.model, "tokenizer") and hasattr(self.model.tokenizer, "vocab_size"):
                return int(self.model.tokenizer.vocab_size)
            if hasattr(self.model, "model") and hasattr(self.model.model, "config"):
                return int(getattr(self.model.model.config, "vocab_size", default_size))
            if hasattr(self.model, "config") and hasattr(self.model.config, "vocab_size"):
                return int(self.model.config.vocab_size)
        except Exception as e:
            logger.warning(f"Failed to detect SPLADE vocabulary size; using default {default_size}: {e}")
        return default_size

    
    
    

    @staticmethod
    def _extract_sparse_dict(unit: MemoryUnit) -> Optional[Dict[int, float]]:
        """Extract sparse dict."""
        sparse_vec = None
        if unit.sparse_embedding is not None:
            sparse_vec = unit.sparse_embedding
        elif unit.raw_data and "splade" in unit.raw_data:
            sparse_vec = unit.raw_data["splade"]

        if not sparse_vec:
            return None
        if not isinstance(sparse_vec, dict):
            return None
        return sparse_vec

    
    

    def _ensure_dynamic_array_state(self) -> None:
        if not hasattr(self, "_posting_arrays"):
            self._posting_arrays = {}
        if not hasattr(self, "_dirty_tokens"):
            self._dirty_tokens = set()
        if not hasattr(self, "_dirty_token_appends"):
            self._dirty_token_appends = {}
        if not hasattr(self, "_dirty_token_rebuilds"):
            self._dirty_token_rebuilds = set()
        if not hasattr(self, "_candidate_lookup_scratch"):
            self._candidate_lookup_scratch = threading.local()

    def _mark_tokens_dirty_for_append(self, token_to_doc_ids: Dict[int, Set[int]]) -> None:
        self._ensure_dynamic_array_state()
        for token_id, doc_ids in token_to_doc_ids.items():
            if not doc_ids:
                continue
            token_id = int(token_id)
            self._dirty_tokens.add(token_id)
            self._dirty_token_appends.setdefault(token_id, set()).update(doc_ids)

    def _mark_tokens_dirty_for_rebuild(self, token_ids: Iterable[int]) -> None:
        self._ensure_dynamic_array_state()
        token_set = {int(token_id) for token_id in token_ids}
        if not token_set:
            return
        self._dirty_tokens.update(token_set)
        self._dirty_token_rebuilds.update(token_set)

    def _clear_dirty_token_state(self, token_id: int) -> None:
        token_id = int(token_id)
        self._dirty_tokens.discard(token_id)
        self._dirty_token_appends.pop(token_id, None)
        self._dirty_token_rebuilds.discard(token_id)

    def _refresh_posting_arrays_for_tokens(self, token_ids: Iterable[int]) -> None:
        """Refresh posting arrays for tokens."""
        self._ensure_dynamic_array_state()
        for tid_raw in token_ids:
            tid = int(tid_raw)
            chain = self.inverted_index.get(tid)
            if not chain:
                self._posting_arrays.pop(tid, None)
                self._clear_dirty_token_state(tid)
                continue

            existing_arrays = self._posting_arrays.get(tid)
            append_doc_ids = self._dirty_token_appends.get(tid)
            needs_full_rebuild = (
                tid in self._dirty_token_rebuilds
                or existing_arrays is None
                or not append_doc_ids
            )
            if not needs_full_rebuild:
                ids, weights = existing_arrays
                new_doc_ids = [doc_id for doc_id in append_doc_ids if doc_id in chain]
                if new_doc_ids:
                    new_ids = np.asarray(new_doc_ids, dtype=np.int64)
                    if ids.size == 0 or int(new_ids.min()) > int(ids[-1]):
                        order = np.argsort(new_ids)
                        new_ids = np.ascontiguousarray(new_ids[order], dtype=np.int64)
                        new_weights = np.asarray([chain[int(doc_id)] for doc_id in new_ids], dtype=np.float32)
                        self._posting_arrays[tid] = (
                            np.ascontiguousarray(np.concatenate((ids, new_ids)), dtype=np.int64),
                            np.ascontiguousarray(np.concatenate((weights, new_weights)), dtype=np.float32),
                        )
                        self._clear_dirty_token_state(tid)
                        continue

            ids = np.fromiter(chain.keys(), dtype=np.int64, count=len(chain))
            weights = np.fromiter(chain.values(), dtype=np.float32, count=len(chain))
            if ids.size > 1:
                order = np.argsort(ids)
                ids = np.ascontiguousarray(ids[order], dtype=np.int64)
                weights = np.ascontiguousarray(weights[order], dtype=np.float32)
            else:
                ids = np.ascontiguousarray(ids, dtype=np.int64)
                weights = np.ascontiguousarray(weights, dtype=np.float32)
            self._posting_arrays[tid] = (ids, weights)
            self._clear_dirty_token_state(tid)

    def _rebuild_posting_arrays(self) -> None:
        self._ensure_dynamic_array_state()
        self._posting_arrays.clear()
        self._refresh_posting_arrays_for_tokens(self.inverted_index.keys())
        self._dirty_tokens.clear()
        self._dirty_token_appends.clear()
        self._dirty_token_rebuilds.clear()

    def _get_posting_arrays(self, tid: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        self._ensure_dynamic_array_state()
        tid = int(tid)
        if tid in self._dirty_tokens:
            self._refresh_posting_arrays_for_tokens([tid])
            return self._posting_arrays.get(tid)
        arrays = self._posting_arrays.get(tid)
        if arrays is not None:
            return arrays
        chain = self.inverted_index.get(tid)
        if not chain:
            return None
        self._refresh_posting_arrays_for_tokens([tid])
        return self._posting_arrays.get(tid)

    def _build_candidate_lookup(self, int_id_array: np.ndarray) -> Tuple[str, Any]:
        """Build candidate lookup."""
        self._ensure_dynamic_array_state()
        if int_id_array.size == 0:
            return "empty", None
        max_id = int(int_id_array.max())
        required_capacity = max_id + 1
        dense_limit = max(65_536, int_id_array.size * self._DENSE_LOCAL_LOOKUP_MAX_RATIO)
        if 0 <= max_id and required_capacity <= dense_limit:
            values, stamps, generation = self._get_candidate_lookup_scratch(required_capacity)
            values[int_id_array] = np.arange(int_id_array.size, dtype=np.intp)
            stamps[int_id_array] = generation
            return "scratch", (values, stamps, generation, required_capacity)

        order = np.argsort(int_id_array)
        return "searchsorted", (
            np.ascontiguousarray(int_id_array[order], dtype=np.int64),
            order.astype(np.intp, copy=False),
        )

    def _get_candidate_lookup_scratch(self, required_capacity: int) -> Tuple[np.ndarray, np.ndarray, int]:
        scratch = self._candidate_lookup_scratch
        values = getattr(scratch, "values", None)
        stamps = getattr(scratch, "stamps", None)
        generation = getattr(scratch, "generation", 0)
        required_capacity = max(int(required_capacity), self._DENSE_LOCAL_LOOKUP_INITIAL_CAPACITY)
        capacity = 1 << (required_capacity - 1).bit_length()
        if values is None or stamps is None or values.size < capacity or stamps.size < capacity:
            values = np.empty(capacity, dtype=np.intp)
            stamps = np.zeros(capacity, dtype=np.uint32)
            generation = 0
            scratch.values = values
            scratch.stamps = stamps
        generation += 1
        if generation >= np.iinfo(np.uint32).max:
            stamps.fill(0)
            generation = 1
        scratch.generation = generation
        return values, stamps, generation

    def _map_posting_ids_to_candidate_rows(self, doc_ids: np.ndarray, lookup: Tuple[str, Any]) -> np.ndarray:
        mode, payload = lookup
        mapped = np.full(doc_ids.size, -1, dtype=np.intp)
        if doc_ids.size == 0 or mode == "empty":
            return mapped
        if mode == "scratch":
            values, stamps, generation, capacity = payload
            in_bounds = (doc_ids >= 0) & (doc_ids < capacity)
            if np.any(in_bounds):
                candidates = doc_ids[in_bounds]
                active = stamps[candidates] == generation
                if np.any(active):
                    mapped[np.flatnonzero(in_bounds)[active]] = values[candidates[active]]
            return mapped

        sorted_ids, order = payload
        positions = np.searchsorted(sorted_ids, doc_ids)
        valid = positions < sorted_ids.size
        if np.any(valid):
            valid_positions = positions[valid]
            valid_indices = np.flatnonzero(valid)
            equal_mask = sorted_ids[valid_positions] == doc_ids[valid_indices]
            if np.any(equal_mask):
                mapped[valid_indices[equal_mask]] = order[valid_positions[equal_mask]]
        return mapped

    def _ingest_doc(
        self,
        int_id: int,
        sparse_vec: Dict[int, float],
        update_arrays: bool = True,
        touched_tokens: Optional[List[int]] = None,
    ) -> int:
        """Ingest doc."""
        tids = np.fromiter(sparse_vec.keys(), dtype=np.int32, count=len(sparse_vec))
        weights = np.fromiter(sparse_vec.values(), dtype=np.float32, count=len(sparse_vec))
        if tids.size == 0:
            return 0

        positive_mask = weights > 0
        if not np.any(positive_mask):
            return 0

        tids = np.ascontiguousarray(tids[positive_mask], dtype=np.int32)
        weights = np.ascontiguousarray(weights[positive_mask], dtype=np.float32)

        for tid, weight in zip(tids, weights):
            tid = int(tid)
            chain = self.inverted_index.get(tid)
            if chain is None:
                chain = {}
                self.inverted_index[tid] = chain
            chain[int_id] = float(weight)

        changed_tokens = [int(tid) for tid in tids]
        if touched_tokens is not None:
            touched_tokens.extend(changed_tokens)
        if update_arrays:
            self._refresh_posting_arrays_for_tokens(changed_tokens)

        self.doc_postings[int_id] = (tids, weights)
        self.total_docs += 1
        self._flat_doc_postings_dirty = True
        return int(tids.size)

    def _evict_doc(
        self,
        int_id: int,
        update_arrays: bool = True,
        touched_tokens: Optional[List[int]] = None,
    ) -> bool:
        """Evict doc."""
        postings = self.doc_postings.pop(int_id, None)
        if postings is None:
            return False
        tids, _ = postings
        changed_tokens = [int(tid) for tid in tids]
        for tid in tids:
            chain = self.inverted_index.get(int(tid))
            if chain is None:
                continue
            chain.pop(int_id, None)
            if not chain:
                self.inverted_index.pop(int(tid), None)
        if touched_tokens is not None:
            touched_tokens.extend(changed_tokens)
        if update_arrays:
            self._refresh_posting_arrays_for_tokens(changed_tokens)
        self.total_docs -= 1
        self._flat_doc_postings_dirty = True
        return True

    
    

    def build_index(self, units: Optional[List[MemoryUnit]] = None) -> None:
        """Build index."""
        self._invalidate_static_index()

        if units is None:
            if hasattr(self.retrieval_source, "get_all_units"):
                units = self.retrieval_source.get_all_units()
            else:
                logger.warning("Retrieval source does not support get_all_units; full SPLADE index cannot be built.")
                return

        self.inverted_index.clear()
        self.doc_postings.clear()
        self._ensure_dynamic_array_state()
        self._posting_arrays.clear()
        self._dirty_tokens.clear()
        self._dirty_token_appends.clear()
        self._dirty_token_rebuilds.clear()
        self.total_docs = 0
        self._index_built = True
        self._flat_doc_postings_dirty = True

        if not units:
            logger.info(" SPLADE inverted index reset because the input unit list is empty.")
            return

        added = 0
        nnz_total = 0
        for unit in units:
            int_id = self._get_or_create_internal_id(unit.uid)
            if int_id in self.doc_postings:
                continue
            sparse_vec = self._extract_sparse_dict(unit)
            if not sparse_vec:
                continue
            n = self._ingest_doc(int_id, sparse_vec, update_arrays=False)
            if n > 0:
                added += 1
                nnz_total += n

        self._rebuild_posting_arrays()

        logger.info(
            f" SPLADE inverted index built: docs={self.total_docs}, "
            f"vocab={len(self.inverted_index)}, nnz={nnz_total} (added {added})"
        )

    def add_units(self, new_units: List[MemoryUnit]) -> bool:
        """Add units."""
        if not new_units:
            return True
        try:
            added = 0
            touched_tokens: List[int] = []
            token_to_doc_ids: Dict[int, Set[int]] = {}
            for unit in new_units:
                int_id = self._get_or_create_internal_id(unit.uid)
                if int_id in self.doc_postings:
                    continue
                sparse_vec = self._extract_sparse_dict(unit)
                if not sparse_vec:
                    continue
                if self._ingest_doc(int_id, sparse_vec, update_arrays=False, touched_tokens=touched_tokens) > 0:
                    for token_id, weight in sparse_vec.items():
                        token_id = int(token_id)
                        if 0 <= token_id < self.vocab_size and float(weight) > 0:
                            token_to_doc_ids.setdefault(token_id, set()).add(int_id)
                    added += 1
            self._index_built = True
            if added:
                self._mark_tokens_dirty_for_append(token_to_doc_ids)
                self._invalidate_static_index()
            if added:
                logger.info(
                    f" SPLADE incremental attach: added {added} documents, total={self.total_docs}, "
                    f"vocab={len(self.inverted_index)}"
                )
            return True
        except Exception as e:
            logger.error(f"SPLADE incremental attach failed: {e}")
            logger.debug(traceback.format_exc())
            return False

    def remove_uids(self, uids: List[str]) -> int:
        """Remove uids."""
        if not uids:
            return 0
        removed = 0
        touched_tokens: List[int] = []
        for int_id in self._uids_to_internal_ids(uids, create=False):
            if self._evict_doc(int_id, update_arrays=False, touched_tokens=touched_tokens):
                removed += 1
        if removed:
            self._mark_tokens_dirty_for_rebuild(touched_tokens)
            self._invalidate_static_index()
            logger.info(
                f"SPLADE evicted {removed} documents, total={self.total_docs}, "
                f"vocab={len(self.inverted_index)}"
            )
        return removed

    
    

    def _ensure_index_built(self) -> None:
        if self._index_built:
            return
        logger.info("SPLADE index is not built; triggering automatic build.")
        self.build_index()

    def _invalidate_static_index(self) -> None:
        """Invalidate static index."""
        self._static_mode = False
        self._static_matrix = None
        self._static_int_ids = []
        self._static_int_id_array = None

    def build_freeze_index(self) -> bool:
        """Build freeze index."""
        self._ensure_index_built()
        if self.total_docs == 0 or not self.doc_postings:
            logger.warning("SPLADE static index build skipped because the dynamic index is empty.")
            self._invalidate_static_index()
            return False

        try:
            static_int_ids = sorted(int(int_id) for int_id in self.doc_postings.keys())
            indptr = np.zeros(len(static_int_ids) + 1, dtype=np.int64)
            indices_parts: List[np.ndarray] = []
            data_parts: List[np.ndarray] = []
            nnz = 0

            for row, int_id in enumerate(static_int_ids):
                postings = self.doc_postings.get(int_id)
                if postings is None:
                    indptr[row + 1] = nnz
                    continue
                tids, weights = postings
                valid_mask = (tids >= 0) & (tids < self.vocab_size) & (weights > 0)
                if np.any(valid_mask):
                    row_indices = np.ascontiguousarray(tids[valid_mask], dtype=np.int64)
                    row_data = np.ascontiguousarray(weights[valid_mask], dtype=np.float32)
                    order = np.argsort(row_indices)
                    row_indices = row_indices[order]
                    row_data = row_data[order]
                    indices_parts.append(row_indices)
                    data_parts.append(row_data)
                    nnz += int(row_indices.size)
                indptr[row + 1] = nnz

            if nnz == 0:
                logger.warning("SPLADE static index build skipped because there are no valid non-zero entries.")
                self._invalidate_static_index()
                return False

            indices = np.concatenate(indices_parts).astype(np.int64, copy=False)
            data = np.concatenate(data_parts).astype(np.float32, copy=False)
            crow_indices = torch.from_numpy(indptr)
            col_indices = torch.from_numpy(indices)
            values = torch.from_numpy(data)
            matrix = torch.sparse_csr_tensor(
                crow_indices,
                col_indices,
                values,
                size=(len(static_int_ids), self.vocab_size),
                dtype=torch.float32,
            )
            if self._use_gpu:
                matrix = matrix.to(self._device)
                backend = "torch/cuda sparse_csr"
            else:
                backend = "torch/cpu sparse_csr"

            self._static_matrix = matrix
            self._static_int_ids = static_int_ids
            self._static_int_id_array = np.asarray(static_int_ids, dtype=np.int64)
            self._static_mode = True
            logger.info(
                f"SPLADE static acceleration index built: shape={tuple(matrix.shape)}, "
                f"nnz={nnz}, backend={backend}"
            )
            return True
        except Exception as e:
            logger.error(f"SPLADE static index build failed: {e}")
            logger.debug(traceback.format_exc())
            self._invalidate_static_index()
            return False

    def _search_static(
        self,
        query: Union[str, QueryBundle],
        top_k: int,
        candidate_uids: Optional[List[str]],
        space_names: Optional[List[str]],
    ) -> Optional[List[RetrievalResult]]:
        """Search static."""
        try:
            if isinstance(query, QueryBundle):
                query_dict = query.get_or_compute_splade(self._encode_query_to_sparse_dict)
            else:
                query_dict = self._encode_query_to_sparse_dict(query)
            if not query_dict:
                return []

            active_items = [
                (int(tid), float(weight))
                for tid, weight in query_dict.items()
                if 0 <= int(tid) < self.vocab_size and float(weight) > 0
            ]
            if not active_items:
                return []

            space_filter_ids: Optional[np.ndarray] = None
            if space_names and candidate_uids is None:
                space_filter_ids = self._get_space_filter_internal_ids(space_names)
            if space_names and space_filter_ids is None:
                space_uids = set(self._get_uids_from_spaces(space_names))
                if candidate_uids is None:
                    candidate_uids = list(space_uids)
                else:
                    candidate_uids = list(set(candidate_uids) & space_uids)
            row_indices: Optional[np.ndarray] = None
            if space_filter_ids is not None:
                row_indices = self._static_rows_from_internal_ids(space_filter_ids)
                if row_indices.size == 0:
                    return []
            elif candidate_uids is not None:
                filter_ids = np.asarray(self._uids_to_internal_ids(candidate_uids, create=False), dtype=np.int64)
                row_indices = self._static_rows_from_internal_ids(filter_ids)
                if row_indices.size == 0:
                    return []

            matrix = self._static_matrix
            if matrix is None:
                return None
            device = matrix.device
            with torch.no_grad():
                q_dense = torch.zeros((self.vocab_size, 1), dtype=torch.float32, device=device)
                q_indices = torch.as_tensor([tid for tid, _ in active_items], dtype=torch.long, device=device)
                q_values = torch.as_tensor([weight for _, weight in active_items], dtype=torch.float32, device=device)
                q_dense.index_add_(0, q_indices, q_values.unsqueeze(1))
                source_int_ids = self._static_int_id_array
                if source_int_ids is None or source_int_ids.size != len(self._static_int_ids):
                    source_int_ids = np.asarray(self._static_int_ids, dtype=np.int64)
                    self._static_int_id_array = source_int_ids

                use_row_slice = (
                    row_indices is not None
                    and row_indices.size <= max(1, int(len(self._static_int_ids) * self._STATIC_ROW_SLICE_MAX_FRACTION))
                )
                if use_row_slice:
                    row_t = torch.as_tensor(row_indices, dtype=torch.long, device=device)
                    search_matrix = self._slice_static_csr_rows(matrix, row_t)
                    scores_t = torch.sparse.mm(search_matrix, q_dense).squeeze(1)
                    source_int_ids = source_int_ids[row_indices]
                else:
                    scores_t = torch.sparse.mm(matrix, q_dense).squeeze(1)
                    if row_indices is not None:
                        row_t = torch.as_tensor(row_indices, dtype=torch.long, device=device)
                        scores_t = scores_t.index_select(0, row_t)
                        source_int_ids = source_int_ids[row_indices]
                real_k = min(top_k, scores_t.shape[0])
                if real_k <= 0:
                    return []
                top_values_t, top_indices_t = torch.topk(scores_t, real_k)
                top_values = top_values_t.detach().cpu().numpy()
                top_indices = top_indices_t.detach().cpu().numpy()

            results: List[RetrievalResult] = []
            for value, local_idx in zip(top_values, top_indices):
                score = float(value)
                if score <= 0:
                    continue
                int_id = int(source_int_ids[int(local_idx)])
                unit = self._get_unit_by_internal_id(int_id)
                if unit is None:
                    continue
                results.append(RetrievalResult(unit, score, RetrievalMethod.SPLADE, {"splade_score": score}))
            return results
        except Exception as e:
            logger.warning(f"SPLADE static fast path failed; falling back to dynamic retrieval: {e}")
            logger.debug(traceback.format_exc())
            return None

    def _encode_query_to_sparse_dict(self, query_text: str) -> Dict[int, float]:
        """Encode query to sparse dict."""
        if not self.model:
            logger.error("SPLADE model is not initialized; query encoding is unavailable.")
            return {}
        if not hasattr(self.model, "encode_query"):
            logger.error("SPLADE model does not expose encode_query.")
            return {}
        try:
            query_vec_raw = self.model.encode_query(query_text)
        except Exception as e:
            logger.error(f"SPLADE encode_query failed: {e}")
            return {}

        if isinstance(query_vec_raw, dict):
            return {int(k): float(v) for k, v in query_vec_raw.items() if float(v) > 0}

        if isinstance(query_vec_raw, torch.Tensor):
            try:
                with torch.no_grad():
                    if query_vec_raw.is_sparse:
                        coalesced = query_vec_raw.coalesce()
                        idx = coalesced.indices().detach().cpu().numpy()
                        val = coalesced.values().detach().cpu().numpy()
                        if idx.ndim == 2:
                            idx = idx[0]
                        return {int(i): float(v) for i, v in zip(idx.flatten(), val.flatten()) if float(v) > 0}
                    cpu_t = query_vec_raw.detach().cpu()
                    nz = torch.nonzero(cpu_t).flatten()
                    if len(nz) == 0:
                        return {}
                    vals = cpu_t[nz].numpy()
                    idxs = nz.numpy()
                    return {int(i): float(v) for i, v in zip(idxs, vals) if float(v) > 0}
            except Exception as e:
                logger.error(f"SPLADE tensor conversion failed: {e}")
                return {}

        return {}

    def _materialize_candidate_query_matrix(
        self,
        cand_list: List[int],
        q_tids: List[int],
        n_qterm: int,
    ) -> Optional[np.ndarray]:
        """Materialize candidate query matrix."""
        if not cand_list or n_qterm <= 0:
            return None

        cand_ids = np.asarray(cand_list, dtype=np.int64)
        arrays_by_column: List[Tuple[int, np.ndarray, np.ndarray]] = []
        for column, tid_raw in enumerate(q_tids):
            pair = self._get_posting_arrays(int(tid_raw))
            if pair is None:
                continue
            doc_ids, weights = pair
            if doc_ids.size == 0:
                continue
            arrays_by_column.append((column, doc_ids, weights))
        if not arrays_by_column:
            return None

        lookup = self._build_candidate_lookup(cand_ids)
        mat = self._get_materialize_scratch(len(cand_list), n_qterm)
        hit_count = 0
        mode, payload = lookup
        for column, doc_ids, weights in arrays_by_column:
            if NUMBA_AVAILABLE and _numba_fill_term_matrix_scratch is not None:
                if mode == "scratch":
                    values, stamps, generation, capacity = payload
                    hit_count += _numba_fill_term_matrix_scratch(
                        doc_ids, weights, values, stamps, generation, capacity, column, mat
                    )
                elif mode == "searchsorted":
                    sorted_ids, order = payload
                    hit_count += _numba_fill_term_matrix_searchsorted(
                        doc_ids, weights, sorted_ids, order, column, mat
                    )
                continue

            mapped = self._map_posting_ids_to_candidate_rows(doc_ids, lookup)
            valid = mapped != -1
            if np.any(valid):
                mat[mapped[valid], column] = weights[valid]
                hit_count += int(np.count_nonzero(valid))

        return mat if hit_count else None

    def _candidate_ids_from_posting_arrays(
        self,
        arrays: List[Tuple[np.ndarray, np.ndarray]],
        filter_set: Optional[Set[int]],
    ) -> np.ndarray:
        if not arrays:
            return np.empty(0, dtype=np.int64)
        if len(arrays) == 1:
            cand_ids = np.unique(arrays[0][0].astype(np.int64, copy=False))
        else:
            cand_ids = np.unique(np.concatenate([pair[0] for pair in arrays]).astype(np.int64, copy=False))
        if filter_set is not None:
            if not filter_set or cand_ids.size == 0:
                return np.empty(0, dtype=np.int64)
            filter_ids = np.fromiter(filter_set, dtype=np.int64, count=len(filter_set))
            cand_ids = cand_ids[np.isin(cand_ids, filter_ids, assume_unique=False)]
        return np.ascontiguousarray(cand_ids, dtype=np.int64)

    def _filter_posting_arrays_by_candidate_ids(
        self,
        arrays: List[Tuple[np.ndarray, np.ndarray]],
        filter_set: Optional[Set[int]],
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        if filter_set is None:
            return arrays
        if not filter_set or not arrays:
            return []

        filter_ids = np.fromiter(filter_set, dtype=np.int64, count=len(filter_set))
        if filter_ids.size == 0:
            return []

        max_posting_id = -1
        for posting_ids, _ in arrays:
            if posting_ids.size:
                max_posting_id = max(max_posting_id, int(posting_ids[-1]))
        max_filter_id = int(filter_ids.max()) if filter_ids.size else -1
        required_capacity = max(max_posting_id, max_filter_id) + 1
        dense_limit = max(65_536, filter_ids.size * self._DENSE_LOCAL_LOOKUP_MAX_RATIO, self.total_docs * 4)

        filtered_arrays: List[Tuple[np.ndarray, np.ndarray]] = []
        if 0 < required_capacity <= dense_limit:
            _, stamps, generation = self._get_candidate_lookup_scratch(required_capacity)
            valid_filter_ids = filter_ids[(filter_ids >= 0) & (filter_ids < required_capacity)]
            if valid_filter_ids.size == 0:
                return [(ids[:0], values[:0]) for ids, values in arrays]
            stamps[valid_filter_ids] = generation
            for posting_ids, values in arrays:
                if posting_ids.size == 0:
                    filtered_arrays.append((posting_ids, values))
                    continue
                in_bounds = (posting_ids >= 0) & (posting_ids < required_capacity)
                keep = np.zeros(posting_ids.size, dtype=bool)
                if np.any(in_bounds):
                    in_bound_positions = np.flatnonzero(in_bounds)
                    keep[in_bound_positions] = stamps[posting_ids[in_bounds]] == generation
                filtered_arrays.append((posting_ids[keep], values[keep]))
            return filtered_arrays

        for posting_ids, values in arrays:
            keep = np.isin(posting_ids, filter_ids, assume_unique=False)
            filtered_arrays.append((posting_ids[keep], values[keep]))
        return filtered_arrays

    def _static_rows_from_internal_ids(self, filter_ids: np.ndarray) -> np.ndarray:
        if filter_ids.size == 0 or not self._static_int_ids:
            return np.empty(0, dtype=np.int64)
        static_ids = self._static_int_id_array
        if static_ids is None or static_ids.size != len(self._static_int_ids):
            static_ids = np.asarray(self._static_int_ids, dtype=np.int64)
            self._static_int_id_array = static_ids
        positions = np.searchsorted(static_ids, filter_ids)
        valid = positions < static_ids.size
        if not np.any(valid):
            return np.empty(0, dtype=np.int64)
        valid_positions = positions[valid]
        valid_filter_ids = filter_ids[valid]
        matched = static_ids[valid_positions] == valid_filter_ids
        if not np.any(matched):
            return np.empty(0, dtype=np.int64)
        return np.unique(valid_positions[matched].astype(np.int64, copy=False))

    def _slice_static_csr_rows(self, matrix: torch.Tensor, row_indices: torch.Tensor) -> torch.Tensor:
        row_indices = row_indices.to(device=matrix.device, dtype=torch.long)
        if row_indices.numel() == 0:
            return torch.sparse_csr_tensor(
                torch.zeros(1, dtype=torch.int64, device=matrix.device),
                torch.empty(0, dtype=torch.int64, device=matrix.device),
                torch.empty(0, dtype=matrix.dtype, device=matrix.device),
                size=(0, matrix.shape[1]),
                dtype=matrix.dtype,
                device=matrix.device,
            )

        crow = matrix.crow_indices()
        starts = crow.index_select(0, row_indices)
        ends = crow.index_select(0, row_indices + 1)
        lengths = ends - starts
        new_crow = torch.empty(row_indices.numel() + 1, dtype=crow.dtype, device=matrix.device)
        new_crow[0] = 0
        new_crow[1:] = torch.cumsum(lengths, dim=0)
        total_nnz = int(new_crow[-1].item())
        if total_nnz == 0:
            return torch.sparse_csr_tensor(
                new_crow,
                torch.empty(0, dtype=matrix.col_indices().dtype, device=matrix.device),
                torch.empty(0, dtype=matrix.values().dtype, device=matrix.device),
                size=(row_indices.numel(), matrix.shape[1]),
                dtype=matrix.dtype,
                device=matrix.device,
            )

        row_offsets = torch.repeat_interleave(new_crow[:-1], lengths)
        starts_repeated = torch.repeat_interleave(starts, lengths)
        within_row = torch.arange(total_nnz, dtype=crow.dtype, device=matrix.device) - row_offsets
        gather_indices = starts_repeated + within_row
        return torch.sparse_csr_tensor(
            new_crow,
            matrix.col_indices().index_select(0, gather_indices),
            matrix.values().index_select(0, gather_indices),
            size=(row_indices.numel(), matrix.shape[1]),
            dtype=matrix.dtype,
            device=matrix.device,
        )

    def _score_candidates_accumulator(
        self,
        cand_ids: np.ndarray,
        q_weights: np.ndarray,
        arrays: List[Tuple[np.ndarray, np.ndarray]],
    ) -> Optional[np.ndarray]:
        if cand_ids.size == 0 or q_weights.size == 0 or not arrays:
            return None
        lookup = self._build_candidate_lookup(cand_ids)
        scores = np.zeros(cand_ids.size, dtype=np.float32)
        hit_count = 0
        mode, payload = lookup
        for q_weight, (doc_ids, weights) in zip(q_weights, arrays):
            if NUMBA_AVAILABLE and _numba_accumulate_scores_scratch is not None:
                if mode == "scratch":
                    values, stamps, generation, capacity = payload
                    hit_count += _numba_accumulate_scores_scratch(
                        doc_ids, weights, values, stamps, generation, capacity, np.float32(q_weight), scores
                    )
                elif mode == "searchsorted":
                    sorted_ids, order = payload
                    hit_count += _numba_accumulate_scores_searchsorted(
                        doc_ids, weights, sorted_ids, order, np.float32(q_weight), scores
                    )
                continue

            mapped = self._map_posting_ids_to_candidate_rows(doc_ids, lookup)
            valid = mapped != -1
            if np.any(valid):
                scores[mapped[valid]] += weights[valid] * np.float32(q_weight)
                hit_count += int(np.count_nonzero(valid))

        return scores if hit_count else None

    def _get_materialize_scratch(self, rows: int, cols: int) -> np.ndarray:
        cells = max(1, int(rows) * int(cols))
        capacity = 1 << (cells - 1).bit_length()
        buffer = getattr(self._materialize_scratch, "buffer", None)
        if buffer is None or buffer.size < cells:
            buffer = np.empty(capacity, dtype=np.float32)
            self._materialize_scratch.buffer = buffer
        mat = buffer[:cells].reshape(rows, cols)
        mat.fill(0.0)
        return mat

    def _ensure_flat_doc_posting_arrays(self) -> bool:
        if not self._flat_doc_postings_dirty:
            return self._flat_doc_int_ids.size > 0
        if not self.doc_postings:
            self._flat_doc_int_ids = np.empty(0, dtype=np.int64)
            self._flat_doc_indptr = np.zeros(1, dtype=np.int64)
            self._flat_doc_tids = np.empty(0, dtype=np.int32)
            self._flat_doc_weights = np.empty(0, dtype=np.float32)
            self._flat_doc_postings_dirty = False
            return False

        sorted_doc_ids = np.asarray(sorted(self.doc_postings.keys()), dtype=np.int64)
        indptr = np.zeros(sorted_doc_ids.size + 1, dtype=np.int64)
        tids_parts: List[np.ndarray] = []
        weights_parts: List[np.ndarray] = []
        cursor = 0
        for row, int_id in enumerate(sorted_doc_ids):
            tids, weights = self.doc_postings[int(int_id)]
            tids_arr = np.asarray(tids, dtype=np.int32)
            weights_arr = np.asarray(weights, dtype=np.float32)
            tids_parts.append(tids_arr)
            weights_parts.append(weights_arr)
            cursor += int(tids_arr.size)
            indptr[row + 1] = cursor

        self._flat_doc_int_ids = sorted_doc_ids
        self._flat_doc_indptr = indptr
        self._flat_doc_tids = np.concatenate(tids_parts).astype(np.int32, copy=False)
        self._flat_doc_weights = np.concatenate(weights_parts).astype(np.float32, copy=False)
        self._flat_doc_postings_dirty = False
        return self._flat_doc_int_ids.size > 0

    def _materialize_candidate_query_matrix_numba(
        self,
        cand_list: List[int],
        q_tids: List[int],
        n_qterm: int,
    ) -> Optional[np.ndarray]:
        if not self._ensure_flat_doc_posting_arrays():
            return None
        if not cand_list or n_qterm <= 0:
            return None

        cand_ids = np.asarray(cand_list, dtype=np.int64)
        q_tid_arr = np.asarray(q_tids, dtype=np.int32)
        if q_tid_arr.size != n_qterm:
            return None
        q_order = np.argsort(q_tid_arr).astype(np.int64, copy=False)
        sorted_q_tids = q_tid_arr[q_order]
        mat = self._get_materialize_scratch(len(cand_list), n_qterm)
        hit_count = _numba_materialize_candidate_query_matrix(
            cand_ids,
            self._flat_doc_int_ids,
            self._flat_doc_indptr,
            self._flat_doc_tids,
            self._flat_doc_weights,
            sorted_q_tids,
            q_order,
            self.vocab_size,
            mat,
        )
        return mat if hit_count else None

    def _numpy_to_device_tensor(self, array: np.ndarray) -> torch.Tensor:
        source_array = array if array.flags.c_contiguous else np.ascontiguousarray(array)
        source_tensor = torch.from_numpy(source_array)
        if not self._use_gpu or self._device.type != "cuda":
            return source_tensor.to(self._device)

        try:
            buffers = getattr(self._pinned_h2d_cache, "buffers", None)
            if buffers is None:
                buffers = {}
                self._pinned_h2d_cache.buffers = buffers

            cache_key = (tuple(source_array.shape), source_tensor.dtype)
            pinned_tensor = buffers.get(cache_key)
            if pinned_tensor is None:
                if len(buffers) >= 4:
                    buffers.clear()
                pinned_tensor = torch.empty(source_array.shape, dtype=source_tensor.dtype, pin_memory=True)
                buffers[cache_key] = pinned_tensor

            pinned_tensor.copy_(source_tensor)
            return pinned_tensor.to(self._device, non_blocking=True)
        except Exception as e:
            logger.debug(f"SPLADE pinned H2D fallback to pageable transfer: {e}")
            return source_tensor.to(self._device, non_blocking=True)

    def search(
        self,
        query: Union[str, QueryBundle],
        top_k: int = 10,
        candidate_uids: Optional[List[str]] = None,
        space_names: Optional[List[str]] = None,
        **kwargs,
    ) -> List[RetrievalResult]:
        """Search."""
        if not self.model:
            logger.error("SPLADE model is not initialized; retrieval is unavailable.")
            return []
        if top_k <= 0:
            return []

        if isinstance(candidate_uids, np.ndarray):
            candidate_uids = candidate_uids.tolist()
        if space_names is None and "ms_names" in kwargs:
            space_names = kwargs.get("ms_names")
        if isinstance(space_names, np.ndarray):
            space_names = space_names.tolist()

        if self._static_mode and self._static_matrix is not None:
            static_results = self._search_static(query, top_k, candidate_uids, space_names)
            if static_results is not None:
                return static_results

        self._ensure_index_built()
        if self.total_docs == 0 or not self.inverted_index:
            return []

        try:
            if isinstance(query, QueryBundle):
                query_dict = query.get_or_compute_splade(self._encode_query_to_sparse_dict)
            else:
                query_dict = self._encode_query_to_sparse_dict(query)
            if not query_dict:
                return []

            
            
            
            space_filter_ids: Optional[np.ndarray] = None
            if space_names and candidate_uids is None:
                space_filter_ids = self._get_space_filter_internal_ids(space_names)
                if space_filter_ids is not None and space_filter_ids.size == 0:
                    return []
            if space_names and space_filter_ids is None:
                space_uids = set(self._get_uids_from_spaces(space_names))
                if candidate_uids is None:
                    candidate_uids = list(space_uids)
                else:
                    candidate_uids = list(set(candidate_uids) & space_uids)
            if space_filter_ids is not None:
                filter_set: Optional[Set[int]] = set(int(int_id) for int_id in space_filter_ids.tolist())
            else:
                filter_set = (
                    set(self._uids_to_internal_ids(candidate_uids, create=False))
                    if candidate_uids is not None else None
                )
            if filter_set is not None and not filter_set:
                return []

            q_tids: List[int] = []
            q_weights: List[float] = []
            query_arrays: List[Tuple[np.ndarray, np.ndarray]] = []
            for tid, w in query_dict.items():
                tid_i = int(tid)
                if tid_i < 0 or tid_i >= self.vocab_size:
                    continue
                arrays = self._get_posting_arrays(tid_i)
                if arrays is None:
                    continue
                q_tids.append(tid_i)
                q_weights.append(float(w))
                query_arrays.append(arrays)
            if not q_tids:
                return []

            if filter_set is not None:
                query_arrays = self._filter_posting_arrays_by_candidate_ids(query_arrays, filter_set)
            cand_ids = self._candidate_ids_from_posting_arrays(query_arrays, None)
            if cand_ids.size == 0:
                return []

            n_cand = int(cand_ids.size)
            n_qterm = len(q_tids)
            q_arr = np.asarray(q_weights, dtype=np.float32)

            cell_count = n_cand * n_qterm
            if self._use_gpu and cell_count >= self._gpu_materialize_threshold:
                cand_list = [int(int_id) for int_id in cand_ids.tolist()]
                mat = self._materialize_candidate_query_matrix(cand_list, q_tids, n_qterm)
                if mat is None:
                    return []
                with torch.no_grad():
                    stream = get_thread_local_cuda_stream(self._device)
                    if stream is not None:
                        with torch.cuda.stream(stream):
                            mat_t = self._numpy_to_device_tensor(mat)
                            q_t = self._numpy_to_device_tensor(q_arr)
                            scores_t = mat_t @ q_t  # (n_cand,)
                            real_k = min(top_k, n_cand)
                            top_v, top_i = torch.topk(scores_t, real_k)
                        stream.synchronize()
                    else:
                        mat_t = self._numpy_to_device_tensor(mat)
                        q_t = self._numpy_to_device_tensor(q_arr)
                        scores_t = mat_t @ q_t  # (n_cand,)
                        real_k = min(top_k, n_cand)
                        top_v, top_i = torch.topk(scores_t, real_k)
                    top_values = top_v.detach().cpu().numpy()
                    top_indices = top_i.detach().cpu().numpy()
            else:
                scores = self._score_candidates_accumulator(cand_ids, q_arr, query_arrays)
                if scores is None:
                    return []
                real_k = min(top_k, n_cand)
                if n_cand > real_k:
                    part = np.argpartition(scores, -real_k)[-real_k:]
                    order = np.argsort(scores[part])[::-1]
                    top_indices = part[order]
                else:
                    top_indices = np.argsort(scores)[::-1]
                top_values = scores[top_indices]

            results: List[RetrievalResult] = []
            for v, i in zip(top_values, top_indices):
                score = float(v)
                if score <= 0:
                    continue
                int_id = int(cand_ids[int(i)])
                unit = self._get_unit_by_internal_id(int_id)
                if unit is None:
                    continue
                results.append(
                    RetrievalResult(unit, score, RetrievalMethod.SPLADE, {"splade_score": score})
                )
            return results

        except Exception as e:
            logger.error(f"SPLADE retrieval failed: {e}")
            logger.debug(traceback.format_exc())
            return []

    
    
    

    _SAVE_FILE = "splade_index.pkl"

    def save_index(self, directory: str) -> bool:
        """Save index."""
        if not self._index_built:
            logger.warning("SPLADE index has not been built; save skipped.")
            return False
        try:
            os.makedirs(directory, exist_ok=True)
            path = os.path.join(directory, self._SAVE_FILE)
            payload = {
                "version": "3.0_global_int_id",
                "doc_id_type": "global_int_id",
                "model_name": self.model_name,
                "vocab_size": self.vocab_size,
                "total_docs": self.total_docs,
                "uid_to_int_id": self._current_uid_to_int_id_map(),
                "inverted_index": self.inverted_index,
                "doc_postings": self.doc_postings,
            }

            static_matrix_file = "splade_static_matrix.safetensors"
            if self._static_mode and self._static_matrix is not None:
                try:
                    static_matrix = self._static_matrix
                    if static_matrix.layout != torch.sparse_csr:
                        raise ValueError(f"Unsupported static matrix layout: {static_matrix.layout}")

                    payload["static_meta"] = {
                        "version": "1.0_sparse_csr_safetensors",
                        "layout": "sparse_csr",
                        "matrix_file": static_matrix_file,
                        "int_ids": [int(int_id) for int_id in self._static_int_ids],
                        "shape": tuple(int(dim) for dim in static_matrix.shape),
                    }
                    safetensors_save_file(
                        {
                            "crow_indices": static_matrix.crow_indices().detach().cpu().contiguous(),
                            "col_indices": static_matrix.col_indices().detach().cpu().contiguous(),
                            "values": static_matrix.values().detach().cpu().contiguous(),
                        },
                        os.path.join(directory, static_matrix_file),
                    )
                    logger.info(f"SPLADE static acceleration matrix saved: {static_matrix_file}")
                except Exception as e:
                    payload.pop("static_meta", None)
                    logger.warning(f"Failed to save SPLADE static acceleration matrix; saving dynamic index only: {e}")

            with open(path, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            nnz = sum(int(tids.size) for tids, _ in self.doc_postings.values())
            logger.info(
                f"SPLADE inverted index saved: {path} (docs={self.total_docs}, "
                f"vocab={len(self.inverted_index)}, nnz={nnz})"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to save SPLADE index: {e}")
            return False

    def load_index(self, directory: str) -> bool:
        """Load index."""
        path = os.path.join(directory, self._SAVE_FILE)
        if not os.path.exists(path):
            if os.path.exists(os.path.join(directory, "splade_matrix.npz")):
                logger.warning(
                    f"Detected splade_matrix.npz from an incompatible SPLADE index format; "
                    f"returning False to trigger a rebuild: {directory}"
                )
            else:
                logger.warning(f"SPLADE index file does not exist: {path}")
            return False

        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)

            saved_vocab = int(payload.get("vocab_size", self.vocab_size))
            if saved_vocab != self.vocab_size:
                logger.error(
                    f"SPLADE index vocabulary mismatch: saved={saved_vocab}, current={self.vocab_size}"
                )
                return False

            self._restore_mapping_from_payload(payload.get("uid_to_int_id"))
            self.inverted_index = self._normalize_inverted_index(payload.get("inverted_index") or {})
            self.doc_postings = self._normalize_doc_postings(payload.get("doc_postings") or {})
            self.total_docs = int(payload.get("total_docs", len(self.doc_postings)))
            self._index_built = True
            self._flat_doc_postings_dirty = True
            self._rebuild_posting_arrays()
            self._invalidate_static_index()

            static_meta = payload.get("static_meta")
            if isinstance(static_meta, dict):
                static_matrix_file = static_meta.get("matrix_file", "splade_static_matrix.safetensors")
                static_matrix_path = os.path.join(directory, str(static_matrix_file))
                if os.path.exists(static_matrix_path):
                    try:
                        tensors = safetensors_load_file(static_matrix_path)
                        shape = tuple(int(dim) for dim in static_meta["shape"])
                        matrix = torch.sparse_csr_tensor(
                            tensors["crow_indices"].to(dtype=torch.int64).contiguous(),
                            tensors["col_indices"].to(dtype=torch.int64).contiguous(),
                            tensors["values"].to(dtype=torch.float32).contiguous(),
                            size=shape,
                            dtype=torch.float32,
                        ).to(self._device)

                        self._static_matrix = matrix
                        self._static_int_ids = [int(int_id) for int_id in static_meta.get("int_ids", [])]
                        self._static_int_id_array = np.asarray(self._static_int_ids, dtype=np.int64)
                        self._static_mode = True
                        logger.info(
                            f"SPLADE static acceleration matrix restored: shape={tuple(matrix.shape)}, "
                            f"backend={matrix.device}"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to restore SPLADE static acceleration matrix; falling back to dynamic postings: {e}")
                        self._invalidate_static_index()
                else:
                    logger.warning(f"SPLADE static metadata exists but the matrix file is missing: {static_matrix_path}")

            if self.total_docs != len(self.doc_postings):
                logger.warning(
                    f"SPLADE index integrity warning: total_docs={self.total_docs} != "
                    f"doc_postings={len(self.doc_postings)}; total_docs has been corrected."
                )
                self.total_docs = len(self.doc_postings)

            nnz = sum(int(tids.size) for tids, _ in self.doc_postings.values())
            logger.info(
                f"SPLADE inverted index loaded: docs={self.total_docs}, "
                f"vocab={len(self.inverted_index)}, nnz={nnz}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load SPLADE index: {e}")
            logger.debug(traceback.format_exc())
            self.inverted_index = {}
            self.doc_postings = {}
            self.total_docs = 0
            self._index_built = False
            self._invalidate_static_index()
            return False

    def _normalize_doc_postings(
        self,
        raw_postings: Dict[Any, Union[Dict[int, float], Tuple[np.ndarray, np.ndarray]]]
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """Normalize doc postings."""
        normalized: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        for doc_key, postings in raw_postings.items():
            int_id = self._coerce_doc_key_to_int_id(doc_key)
            if int_id is None:
                continue
            if isinstance(postings, tuple) and len(postings) == 2:
                tids_raw, weights_raw = postings
                tids = np.asarray(tids_raw, dtype=np.int32)
                weights = np.asarray(weights_raw, dtype=np.float32)
            elif isinstance(postings, dict):
                tids = np.fromiter(postings.keys(), dtype=np.int32, count=len(postings))
                weights = np.fromiter(postings.values(), dtype=np.float32, count=len(postings))
            else:
                continue

            if tids.size == 0 or weights.size == 0:
                continue
            if tids.size != weights.size:
                logger.warning(f"SPLADE doc_postings skipped a malformed entry: doc_key={doc_key}")
                continue

            positive_mask = weights > 0
            if not np.any(positive_mask):
                continue

            normalized[int_id] = (
                np.ascontiguousarray(tids[positive_mask], dtype=np.int32),
                np.ascontiguousarray(weights[positive_mask], dtype=np.float32),
            )
        return normalized

    def _normalize_inverted_index(self, raw_index: Dict[Any, Dict[Any, float]]) -> Dict[int, Dict[int, float]]:
        """Normalize inverted index."""
        normalized: Dict[int, Dict[int, float]] = {}
        for raw_tid, postings in (raw_index or {}).items():
            try:
                tid = int(raw_tid)
            except (TypeError, ValueError):
                continue
            term_postings: Dict[int, float] = {}
            for doc_key, weight in postings.items():
                int_id = self._coerce_doc_key_to_int_id(doc_key)
                if int_id is not None:
                    term_postings[int_id] = float(weight)
            if term_postings:
                normalized[tid] = term_postings
        return normalized

    
    
    

    def swap_out_to_db(
        self,
        db_operator: "DuckDBOperator",
        target_uids: Optional[List[str]] = None,
        evict: bool = True,
    ) -> int:
        """Persist SPLADE vectors to DuckDB and optionally evict them from memory.

        Args:
            db_operator: Connected DuckDB storage operator.
            target_uids: Optional public UIDs to persist. When omitted, all
                in-memory SPLADE documents are considered.
            evict: Whether to remove persisted vectors from the in-memory index.

        Returns:
            Number of sparse documents written.
        """
        if target_uids is None:
            int_ids_list = list(self.doc_postings.keys())
        else:
            int_ids_list = [int_id for int_id in self._uids_to_internal_ids(target_uids, create=False) if int_id in self.doc_postings]
        if not int_ids_list:
            return 0

        updates: List[Dict] = []
        for int_id in int_ids_list:
            uid = self._internal_id_to_uid(int_id)
            if uid is None:
                continue
            postings = self.doc_postings.get(int_id)
            if postings is None:
                continue
            tids_arr, weights_arr = postings
            if tids_arr.size == 0 or weights_arr.size == 0:
                continue
            indices = [int(tid) for tid in tids_arr.tolist()]
            values = [float(weight) for weight in weights_arr.tolist()]
            updates.append({"uid": uid, "splade_indices": indices, "splade_values": values})

        if not updates:
            return 0

        written = db_operator.update_sparse_vectors_batch(updates)

        if evict:
            self.remove_uids([u["uid"] for u in updates])

        logger.info(
            f"swap_out_to_db: written={written}, evicted={evict}, "
            f"remaining L1 docs={self.total_docs}"
        )
        return written

    def swap_in_from_db(
        self,
        db_operator: "DuckDBOperator",
        uids: List[str],
    ) -> int:
        """Swap in from db."""
        if not uids:
            return 0

        rows = db_operator.read_sparse_vectors_batch(uids)
        if not rows:
            logger.info("swap_in_from_db: no matching DuckDB rows.")
            return 0

        loaded = 0
        touched_tokens: List[int] = []
        for uid, indices, values in rows:
            if not indices or not values:
                continue
            int_id = self._get_or_create_internal_id(uid)
            if int_id in self.doc_postings:
                self._evict_doc(int_id, update_arrays=False, touched_tokens=touched_tokens)
            sparse_dict = {int(t): float(w) for t, w in zip(indices, values)}
            if self._ingest_doc(int_id, sparse_dict, update_arrays=False, touched_tokens=touched_tokens) > 0:
                loaded += 1

        self._index_built = True
        if loaded:
            self._mark_tokens_dirty_for_rebuild(touched_tokens)
            self._invalidate_static_index()
        logger.info(
            f"swap_in_from_db: loaded={loaded}, current L1 docs={self.total_docs}, "
            f"vocab={len(self.inverted_index)}"
        )
        return loaded

    
    

    def get_index_stats(self) -> Dict[str, Union[int, float, bool, str]]:
        nnz = sum(int(tids.size) for tids, _ in self.doc_postings.values())
        return {
            "index_built": self._index_built,
            "total_docs": self.total_docs,
            "vocab_size_active": len(self.inverted_index),
            "vocab_size_model": self.vocab_size,
            "nnz": nnz,
            "avg_doc_nnz": (nnz / self.total_docs) if self.total_docs else 0.0,
            "use_gpu": self._use_gpu,
            "device": str(self._device),
            "model_name": self.model_name,
            "static_mode": self._static_mode,
            "static_matrix_shape": tuple(self._static_matrix.shape) if self._static_matrix is not None else None,
        }

    def cleanup(self) -> None:
        """Release associated resources."""
        self.inverted_index.clear()
        self.doc_postings.clear()
        if hasattr(self, "_posting_arrays"):
            self._posting_arrays.clear()
        self.total_docs = 0
        self._index_built = False
        self._invalidate_static_index()
