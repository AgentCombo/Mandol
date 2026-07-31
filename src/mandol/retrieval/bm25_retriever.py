"""BM25 lexical retrieval over Mandol memory units.

The retriever stores token postings keyed by the stable UID-to-int-id mapping
shared with SemanticMap, supports candidate and MemorySpace filters, and can
materialize optional static sparse matrices for high-throughput benchmark
queries. spaCy is used when available, with regex and jieba fallbacks.
"""

import logging
import math
import os
import re
import pickle
import traceback
import threading
from collections import Counter
from importlib import import_module
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union, TYPE_CHECKING

import numpy as np
import torch
import jieba
from safetensors.torch import load_file as safetensors_load_file, save_file as safetensors_save_file

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    njit = None

from .retrieval_interface import BaseRetriever, RetrievalInterface, RetrievalMethod, RetrievalResult
from .query_bundle import QueryBundle
from .cuda_stream_utils import get_thread_local_cuda_stream, to_device_with_thread_local_pinned
from ..core.memory_unit import MemoryUnit
from ..utils.model_manager import global_model_manager
from ..utils.logging_config import create_module_logger

if TYPE_CHECKING:
    pass

logger = create_module_logger("bm25_retriever")


if NUMBA_AVAILABLE:
    @njit(fastmath=True, nogil=True, cache=True)
    def _numba_fill_tf_column_scratch(
        global_ids: np.ndarray,
        tfs: np.ndarray,
        values: np.ndarray,
        stamps: np.ndarray,
        generation: int,
        capacity: int,
        column: int,
        tf_matrix: np.ndarray,
    ) -> int:
        hits = 0
        for idx in range(global_ids.size):
            gid = global_ids[idx]
            if gid >= 0 and gid < capacity and stamps[gid] == generation:
                row = values[gid]
                tf_matrix[row, column] = tfs[idx]
                hits += 1
        return hits

    @njit(fastmath=True, nogil=True, cache=True)
    def _numba_fill_tf_column_searchsorted(
        global_ids: np.ndarray,
        tfs: np.ndarray,
        sorted_ids: np.ndarray,
        order: np.ndarray,
        column: int,
        tf_matrix: np.ndarray,
    ) -> int:
        hits = 0
        n_sorted = sorted_ids.size
        for idx in range(global_ids.size):
            gid = global_ids[idx]
            lo = 0
            hi = n_sorted
            pos = -1
            while lo < hi:
                mid = (lo + hi) // 2
                mid_value = sorted_ids[mid]
                if mid_value < gid:
                    lo = mid + 1
                elif mid_value > gid:
                    hi = mid
                else:
                    pos = mid
                    break
            if pos >= 0:
                tf_matrix[order[pos], column] = tfs[idx]
                hits += 1
        return hits

    @njit(nogil=True, cache=True)
    def _numba_accumulate_bm25_scratch(
        global_ids: np.ndarray,
        tfs: np.ndarray,
        values: np.ndarray,
        stamps: np.ndarray,
        generation: int,
        capacity: int,
        doc_lengths: np.ndarray,
        term_weight: np.float32,
        k1: np.float32,
        b: np.float32,
        avgdl: np.float32,
        scores: np.ndarray,
    ) -> int:
        hits = 0
        norm_left = np.float32(1.0) - b
        k1_plus = k1 + np.float32(1.0)
        for idx in range(global_ids.size):
            global_id = global_ids[idx]
            if global_id >= 0 and global_id < capacity and stamps[global_id] == generation:
                row = values[global_id]
                tf = np.float32(tfs[idx])
                doc_len = np.float32(doc_lengths[row])
                denom = tf + k1 * (norm_left + b * (doc_len / avgdl))
                if denom > np.float32(0.0):
                    scores[row] += (tf * k1_plus / denom) * term_weight
                    hits += 1
        return hits

    @njit(nogil=True, cache=True)
    def _numba_accumulate_bm25_searchsorted(
        global_ids: np.ndarray,
        tfs: np.ndarray,
        sorted_ids: np.ndarray,
        order: np.ndarray,
        doc_lengths: np.ndarray,
        term_weight: np.float32,
        k1: np.float32,
        b: np.float32,
        avgdl: np.float32,
        scores: np.ndarray,
    ) -> int:
        hits = 0
        n_sorted = sorted_ids.size
        norm_left = np.float32(1.0) - b
        k1_plus = k1 + np.float32(1.0)
        for idx in range(global_ids.size):
            global_id = global_ids[idx]
            lo = 0
            hi = n_sorted
            pos = -1
            while lo < hi:
                mid = (lo + hi) // 2
                mid_value = sorted_ids[mid]
                if mid_value < global_id:
                    lo = mid + 1
                elif mid_value > global_id:
                    hi = mid
                else:
                    pos = mid
                    break
            if pos >= 0:
                row = order[pos]
                tf = np.float32(tfs[idx])
                doc_len = np.float32(doc_lengths[row])
                denom = tf + k1 * (norm_left + b * (doc_len / avgdl))
                if denom > np.float32(0.0):
                    scores[row] += (tf * k1_plus / denom) * term_weight
                    hits += 1
        return hits


class BM25Retriever(BaseRetriever):
    """Lexical retriever backed by BM25 postings over MemoryUnit text.

    BM25Retriever maintains stable UID-to-integer-ID mappings so lexical
    postings can be shared with candidate filters and saved indexes. It can use
    spaCy lemmatization when available and falls back to regex or jieba
    tokenization without changing retrieval APIs.
    """

    _CHINESE_PATTERN = re.compile(r'[\u4e00-\u9fff]')
    _FALLBACK_TOKEN_PATTERN = re.compile(r'\b\w+\b')
    _DENSE_LOCAL_LOOKUP_INITIAL_CAPACITY = 1_000_001
    _DENSE_LOCAL_LOOKUP_MAX_RATIO = 256
    _STATIC_ROW_SLICE_MAX_FRACTION = 0.25
    _dense_lookup_scratch = threading.local()
    default_spacy_model = "en_core_web_lg"

    def __init__(
        self,
        retrieval_source: RetrievalInterface,
        default_text_field: str = "text_content",
        k1: float = 1.5,
        b: float = 0.75,
        use_jieba: bool = False,
        custom_stop_words: Optional[Set[str]] = None,
        spacy_model_name: Optional[str] = None,
        auto_install_spacy_model: bool = True,
    ):
        """Initialize the BM25 retriever.

        Args:
            retrieval_source: SemanticMap-like source that exposes memory units
                and memory-space filtering APIs.
            default_text_field: Raw-data field used as the preferred text body.
            k1: BM25 term-frequency saturation parameter.
            b: BM25 document-length normalization parameter.
            use_jieba: Whether to use jieba tokenization for Chinese text.
            custom_stop_words: Optional stop-word set merged into defaults.
            spacy_model_name: Optional spaCy model used for lemmatization.
            auto_install_spacy_model: Whether the model manager may download the
                requested spaCy model when it is missing.
        """
        self.retrieval_source = retrieval_source
        self.default_text_field = default_text_field
        self.k1 = k1
        self.b = b
        self.use_jieba = use_jieba
        self.stop_words = self._initialize_stop_words(custom_stop_words)
        self.spacy_model_name = spacy_model_name or self.default_spacy_model
        self.auto_install_spacy_model = auto_install_spacy_model

        
        self._index_built: bool = False
        self.total_docs: int = 0
        self.total_doc_length: int = 0
        self.doc_lengths: Dict[int, int] = {}           # int_id -> doc_len
        self.dfs: Dict[str, int] = {}                   # term -> DF
        self.inverted_index: Dict[str, Dict[int, int]] = {}  # term -> {int_id: tf}
        self._posting_arrays: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._dirty_terms: Set[str] = set()
        self._dirty_term_appends: Dict[str, Set[int]] = {}
        self._dirty_term_rebuilds: Set[str] = set()
        
        # Reverse postings are required for incremental removal and L1 eviction.
        self.doc_postings: Dict[int, Dict[str, int]] = {}

        self._local_uid_to_int_id: Dict[str, int] = {}
        self._local_int_id_to_uid: Dict[int, str] = {}
        self._local_next_int_id: int = 0

        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._use_gpu = torch.cuda.is_available()
        self._gpu_materialize_threshold = 4096

        # Static CSR acceleration is invalidated by any posting or mapping update.
        self._static_mode = False
        self._static_matrix = None
        self._static_int_ids: List[int] = []
        self._static_int_id_array: Optional[np.ndarray] = None
        self._static_vocab: Dict[str, int] = {}
        self._static_idf = None
        self._static_doc_lengths = None
        self._static_avgdl: float = 1.0

        self.nlp = self._initialize_spacy_model(
            self.spacy_model_name,
            auto_install=self.auto_install_spacy_model,
        )

    def _initialize_spacy_model(self, model_name: str, auto_install: bool):
        """Load the BM25 lemmatizer lazily and fall back without blocking input."""
        try:
            spacy_module = import_module("spacy")
        except ImportError:
            logger.warning(
                "Mandol BM25: spaCy package is not installed; using regex "
                "tokenization."
            )
            return None

        disabled_components = ["parser", "ner", "textcat", "senter"]

        def _loader():
            try:
                return spacy_module.load(model_name, disable=disabled_components)
            except OSError as exc:
                if not auto_install:
                    logger.warning(
                        "Mandol BM25: spaCy model %s is not installed and "
                        "automatic download is disabled; using regex tokenization.",
                        model_name,
                    )
                    raise RuntimeError(
                        f"spaCy model {model_name} is unavailable and automatic "
                        "download is disabled"
                    ) from exc

            first_message = (
                f"Mandol BM25: spaCy model {model_name} is not installed."
            )
            second_message = (
                f"Mandol BM25: downloading {model_name} for English "
                "lemmatization and higher BM25 recall..."
            )
            print(first_message, flush=True)
            print(second_message, flush=True)
            logger.warning(first_message)
            logger.info(second_message)

            try:
                spacy_cli = import_module("spacy.cli")
                spacy_cli.download(model_name)
            except (Exception, SystemExit) as exc:
                raise RuntimeError(
                    f"failed to download spaCy model {model_name}: {exc}"
                ) from exc

            try:
                return spacy_module.load(model_name, disable=disabled_components)
            except Exception as exc:
                raise RuntimeError(
                    f"spaCy model {model_name} could not be loaded after download: {exc}"
                ) from exc

        try:
            nlp = global_model_manager.get_or_load_model(
                model_type="spacy",
                model_name=model_name,
                loader_func=_loader,
            )
        except Exception as exc:
            logger.warning(
                "Mandol BM25: spaCy lemmatization is unavailable (%s); using "
                "regex tokenization.",
                exc,
            )
            return None

        if nlp is None:
            logger.warning(
                "Mandol BM25: spaCy model %s returned no pipeline; using regex "
                "tokenization.",
                model_name,
            )
            return None

        logger.info(
            "Mandol BM25: spaCy lemmatization enabled with model %s.",
            model_name,
        )
        return nlp


    def _mapping_owner(self):
        source = self.retrieval_source
        if hasattr(source, "_get_or_create_int_id"):
            return source
        semantic_map = getattr(source, "semantic_map", None)
        if semantic_map is not None and hasattr(semantic_map, "_get_or_create_int_id"):
            return semantic_map
        return None

    def _get_or_create_internal_id(self, uid: str) -> int:
        """Return the shared global int-id, or allocate a local one as fallback."""
        owner = self._mapping_owner()
        if owner is not None:
            return int(owner._get_or_create_int_id(uid))
        uid = str(uid)
        existing = self._local_uid_to_int_id.get(uid)
        if existing is not None:
            return existing
        int_id = self._local_next_int_id
        self._local_next_int_id += 1
        self._local_uid_to_int_id[uid] = int_id
        self._local_int_id_to_uid[int_id] = uid
        return int_id

    def _uid_to_internal_id(self, uid: str, create: bool = False) -> Optional[int]:
        """Translate a public UID to the int-id used by postings and filters."""
        owner = self._mapping_owner()
        if owner is not None:
            if create:
                return int(owner._get_or_create_int_id(uid))
            mapping = owner._get_uid_to_int_id_map() if hasattr(owner, "_get_uid_to_int_id_map") else getattr(owner, "_uid_to_int_id", {})
            int_id = mapping.get(str(uid))
            return int(int_id) if int_id is not None else None
        if create:
            return self._get_or_create_internal_id(uid)
        int_id = self._local_uid_to_int_id.get(str(uid))
        return int(int_id) if int_id is not None else None

    def _uids_to_internal_ids(self, uids: List[str], create: bool = False) -> List[int]:
        """Vectorized UID-to-int-id translation that silently drops missing UIDs."""
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
        """Return cached global int-ids for pure MemorySpace filters.

        The SemanticMap cache is versioned by space membership and UID mapping
        updates. BM25 can reuse it only when no explicit candidate UID list is
        present; combined filters still use the intersection path.
        """
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
                logger.debug(f"BM25 cached space int-id filter unavailable: {exc}")
        return None

    def _internal_id_to_uid(self, int_id: int) -> Optional[str]:
        owner = self._mapping_owner()
        if owner is not None:
            mapping = owner._get_int_id_to_uid_map() if hasattr(owner, "_get_int_id_to_uid_map") else getattr(owner, "_int_id_to_uid", {})
            return mapping.get(int(int_id))
        return self._local_int_id_to_uid.get(int(int_id))

    def _current_uid_to_int_id_map(self) -> Dict[str, int]:
        owner = self._mapping_owner()
        if owner is not None and hasattr(owner, "_get_uid_to_int_id_map"):
            return {str(uid): int(int_id) for uid, int_id in owner._get_uid_to_int_id_map().items()}
        return dict(self._local_uid_to_int_id)

    def _restore_mapping_from_payload(self, mapping: Optional[Dict[str, Any]]) -> None:
        """Restore persisted UID/int-id mappings without overwriting active owners."""
        if not mapping:
            return
        owner = self._mapping_owner()
        if owner is not None and hasattr(owner, "_set_uid_int_mapping") and not owner._get_uid_to_int_id_map():
            owner._set_uid_int_mapping(mapping)
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
        if hasattr(self.retrieval_source, "_unit_exists"):
            return bool(self.retrieval_source._unit_exists(uid))
        semantic_map = getattr(self.retrieval_source, "semantic_map", None)
        if semantic_map is not None and hasattr(semantic_map, "_unit_exists"):
            return bool(semantic_map._unit_exists(uid))
        memory_units = getattr(self.retrieval_source, "memory_units", None)
        if isinstance(memory_units, dict) and uid in memory_units:
            return True
        return bool(
            hasattr(self.retrieval_source, "get_unit")
            and self.retrieval_source.get_unit(uid) is not None
        )

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


    def _initialize_stop_words(self, custom_stop_words: Optional[Set[str]] = None) -> Set[str]:
        """Build the English/Chinese stop-word set used before posting updates."""
        english_stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'am', 'be', 'been', 'being', 'have', 'has', 'had', 'having',
            'do', 'does', 'did', 'doing', 'will', 'would', 'could', 'should', 'may', 'might', 'can',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me',
            'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their',
            'myself', 'yourself', 'himself', 'herself', 'itself', 'ourselves', 'yourselves', 'themselves',
            'here', 'there', 'where', 'everywhere', 'somewhere', 'nowhere', 'above', 'below', 'under',
            'over', 'between', 'among', 'through', 'into', 'onto', 'upon', 'within', 'without',
            'very', 'quite', 'rather', 'really', 'too', 'so', 'such', 'much', 'many', 'more', 'most',
            'less', 'least', 'few', 'little', 'some', 'any', 'all', 'both', 'each', 'every', 'either',
            'neither', 'another', 'other', 'same', 'different',
            'um', 'uh', 'oh', 'ah', 'yeah', 'yes', 'no', 'okay', 'ok', 'well', 'like', 'just',
            'actually', 'basically', 'literally', 'obviously', 'definitely', 'probably', 'maybe',
            'perhaps', 'suppose', 'guess', 'think', 'know', 'see', 'look', 'listen', 'hear',
            'however', 'therefore', 'moreover', 'furthermore', 'nevertheless', 'nonetheless',
            'meanwhile', 'otherwise', 'besides', 'instead', 'rather', 'though', 'although',
            'because', 'since', 'as', 'if', 'unless', 'whether', 'while', 'whereas',
        }
        chinese_stop_words = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很',
            '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这', '当', '下', '想', '出',
            '那', '里', '以', '时候', '把', '行', '比', '但是', '他', '她', '它', '我们', '你们', '他们', '她们',
            '它们', '这个', '那个', '这些', '那些', '什么', '怎么', '为什么', '哪里', '什么时候', '多少', '怎样',
            '如何', '或者', '还是', '然后', '所以', '因为', '虽然', '但是', '如果', '只要', '嗯', '啊', '哦',
            '呃', '额', '这样', '那样', '好的', '知道', '明白', '对', '不对', '可以', '不可以', '应该', '可能',
        }
        all_stop_words = english_stop_words | chinese_stop_words
        if custom_stop_words:
            all_stop_words |= custom_stop_words
        logger.info(f"BM25 stop-word set initialized with {len(all_stop_words)} terms.")
        return all_stop_words


    def _is_chinese(self, text: str) -> bool:
        return bool(self._CHINESE_PATTERN.search(text))

    def get_method_type(self) -> RetrievalMethod:
        return RetrievalMethod.BM25

    def _preprocess_text(self, text: str) -> List[str]:
        """Tokenize text with spaCy, jieba, or regex fallback."""
        if not text or not isinstance(text, str):
            return []
        text = text.strip().lower()
        if not text:
            return []
        try:
            if self.use_jieba and self._is_chinese(text):
                words = list(jieba.cut(text))
                result = []
                for w in words:
                    w = w.strip()
                    if not w or w in self.stop_words:
                        continue
                    if self._is_chinese(w):
                        result.append(w)
                    elif w.isdigit():
                        result.append(w)
                    else:
                        if not w[0].isalnum():
                            continue
                        if self.nlp:
                            token_doc = self.nlp(w)
                            if len(token_doc) > 0:
                                lemma = token_doc[0].lemma_.lower()
                                if lemma not in self.stop_words:
                                    result.append(lemma)
                            else:
                                result.append(w)
                        else:
                            if len(w) > 1:
                                result.append(w)
                return result

            if self.nlp:
                doc = self.nlp(text)
                return [
                    token.lemma_.lower()
                    for token in doc
                    if not token.is_stop and not token.is_punct and not token.is_space
                    and (len(token.lemma_) > 1 or token.lemma_.isdigit())
                    and token.lemma_.lower() not in self.stop_words
                ]

            # Regex tokenization keeps BM25 usable when spaCy is not configured.
            words = self._FALLBACK_TOKEN_PATTERN.findall(text)
            return [w for w in words if w not in self.stop_words and (len(w) > 1 or w.isdigit())]

        except Exception as e:
            logger.warning(f"BM25 preprocessing failed: {e}")
            return []

    def _extract_text_from_unit(self, unit: MemoryUnit) -> str:
        if not unit or not unit.raw_data:
            return ""
        text = unit.raw_data.get(self.default_text_field, "")
        if not text:
            for field in ("content", "description", "summary", "title", "message"):
                val = unit.raw_data.get(field)
                if val and isinstance(val, str):
                    text = val
                    break
        if not text:
            text = str(unit.raw_data)
        return text


    def _batch_tokenize(self, units: List[MemoryUnit]) -> List[Tuple[str, List[str]]]:
        """Tokenize memory units while batching spaCy work when possible."""
        n = len(units)
        raw_texts: List[str] = []
        uid_list: List[str] = []
        english_indices: List[int] = []
        english_texts: List[str] = []
        chinese_indices: List[int] = []

        for idx, unit in enumerate(units):
            text = self._extract_text_from_unit(unit)
            raw_texts.append(text)
            uid_list.append(unit.uid)
            if self.use_jieba and self._is_chinese(text):
                chinese_indices.append(idx)
            else:
                english_indices.append(idx)
                english_texts.append(text)

        doc_tokens: List[Optional[List[str]]] = [None] * n

        if english_indices and self.nlp:
            count = len(english_texts)
            spacy_processes = 1
            if count > 1000:
                cpu_count = os.cpu_count() or 1
                env_limit = os.getenv("MANDOL_BM25_SPACY_MAX_PROCESSES")
                if env_limit:
                    try:
                        cpu_count = max(1, int(env_limit))
                    except ValueError:
                        logger.warning(f"Ignoring invalid MANDOL_BM25_SPACY_MAX_PROCESSES={env_limit!r}")
                spacy_processes = max(1, cpu_count)
            logger.info(
                f"Processing {count} English documents "
                f"(spaCy batch mode, processes={spacy_processes})."
            )
            try:
                for doc, original_idx in zip(
                    self.nlp.pipe(english_texts, batch_size=2000, n_process=spacy_processes),
                    english_indices,
                ):
                    tokens = [
                        token.lemma_.lower()
                        for token in doc
                        if not token.is_stop and not token.is_punct and not token.is_space
                        and token.lemma_.lower() not in self.stop_words
                    ]
                    doc_tokens[original_idx] = tokens
            except Exception as e:
                logger.error(f"spaCy batch processing failed; falling back to per-document preprocessing: {e}")
                for idx in english_indices:
                    doc_tokens[idx] = self._preprocess_text(raw_texts[idx])

        if chinese_indices:
            use_jieba_parallel = os.name == "posix" and self.use_jieba
            mode = "native parallel" if use_jieba_parallel else "sequential"
            logger.info(f"Processing {len(chinese_indices)} Chinese or mixed documents with jieba ({mode}).")
            parallel_enabled = False
            try:
                if use_jieba_parallel:
                    try:
                        jieba.enable_parallel()
                        parallel_enabled = True
                    except Exception as exc:
                        logger.warning(f"Failed to enable jieba parallel mode; falling back to sequential tokenization: {exc}")
                for idx in chinese_indices:
                    doc_tokens[idx] = self._preprocess_text(raw_texts[idx])
            finally:
                if parallel_enabled:
                    try:
                        jieba.disable_parallel()
                    except Exception as exc:
                        logger.warning(f"Failed to disable jieba parallel mode: {exc}")

        for idx in range(n):
            if doc_tokens[idx] is None:
                doc_tokens[idx] = self._preprocess_text(raw_texts[idx])

        return list(zip(uid_list, doc_tokens))  # type: ignore[arg-type]

    

    def _ingest_document(self, int_id: int, tokens: List[str], update_arrays: bool = True) -> None:
        """Insert or replace one document in BM25 postings."""
        touched_terms: Set[str] = set()
        if int_id in self.doc_lengths:
            touched_terms.update(self._evict_document(int_id, update_arrays=False))

        doc_len = len(tokens)
        self.doc_lengths[int_id] = doc_len
        self.total_docs += 1
        self.total_doc_length += doc_len

        tf_counter = Counter(tokens)

        
        self.doc_postings[int_id] = tf_counter

        for term, tf in tf_counter.items():
            if term not in self.inverted_index:
                self.inverted_index[term] = {}
            self.inverted_index[term][int_id] = tf
            self.dfs[term] = self.dfs.get(term, 0) + 1
            touched_terms.add(term)

        if update_arrays:
            self._refresh_posting_arrays_for_terms(touched_terms)

    def _evict_document(self, int_id: int, update_arrays: bool = True) -> List[str]:
        """Remove one document from postings and return terms that changed."""
        if int_id not in self.doc_lengths:
            return []
        doc_len = self.doc_lengths.pop(int_id)
        self.total_docs -= 1
        self.total_doc_length -= doc_len

        terms = self.doc_postings.pop(int_id, None)
        if not terms:
            return []
        touched_terms = list(terms.keys())
        for term in touched_terms:
            chain = self.inverted_index.get(term)
            if chain is not None:
                chain.pop(int_id, None)
                if not chain:
                    self.inverted_index.pop(term, None)
            new_df = self.dfs.get(term, 1) - 1
            if new_df <= 0:
                self.dfs.pop(term, None)
            else:
                self.dfs[term] = new_df
        if update_arrays:
            self._refresh_posting_arrays_for_terms(touched_terms)
        return touched_terms

    def _ensure_dynamic_array_state(self) -> None:
        """Create dynamic posting-array state for older restored indexes."""
        if not hasattr(self, "_posting_arrays"):
            self._posting_arrays = {}
        if not hasattr(self, "_dirty_terms"):
            self._dirty_terms = set()
        if not hasattr(self, "_dirty_term_appends"):
            self._dirty_term_appends = {}
        if not hasattr(self, "_dirty_term_rebuilds"):
            self._dirty_term_rebuilds = set()

    def _mark_terms_dirty_for_append(self, term_to_doc_ids: Dict[str, Set[int]]) -> None:
        """Mark terms that can be refreshed by append-only posting updates."""
        self._ensure_dynamic_array_state()
        for term, doc_ids in term_to_doc_ids.items():
            if not doc_ids:
                continue
            self._dirty_terms.add(term)
            self._dirty_term_appends.setdefault(term, set()).update(doc_ids)

    def _mark_terms_dirty_for_rebuild(self, terms: Iterable[str]) -> None:
        """Mark terms whose posting arrays require full reconstruction."""
        self._ensure_dynamic_array_state()
        term_set = set(terms)
        if not term_set:
            return
        self._dirty_terms.update(term_set)
        self._dirty_term_rebuilds.update(term_set)

    def _clear_dirty_term_state(self, term: str) -> None:
        self._dirty_terms.discard(term)
        self._dirty_term_appends.pop(term, None)
        self._dirty_term_rebuilds.discard(term)

    def _refresh_posting_arrays_for_terms(self, terms: Iterable[str]) -> None:
        """Refresh sorted NumPy posting arrays for changed terms.

        Append-only updates avoid rebuilding an entire term chain when new int
        ids are greater than the existing tail. Rebuilds are used for removals
        and out-of-order updates so searchsorted/Numba paths remain valid.
        """
        self._ensure_dynamic_array_state()
        for term in terms:
            chain = self.inverted_index.get(term)
            if not chain:
                self._posting_arrays.pop(term, None)
                self._clear_dirty_term_state(term)
                continue

            existing_arrays = self._posting_arrays.get(term)
            append_doc_ids = self._dirty_term_appends.get(term)
            needs_full_rebuild = (
                term in self._dirty_term_rebuilds
                or existing_arrays is None
                or not append_doc_ids
            )
            if not needs_full_rebuild:
                ids, tfs = existing_arrays
                new_doc_ids = [doc_id for doc_id in append_doc_ids if doc_id in chain]
                if new_doc_ids:
                    new_ids = np.asarray(new_doc_ids, dtype=np.int64)
                    if ids.size == 0 or int(new_ids.min()) > int(ids[-1]):
                        order = np.argsort(new_ids)
                        new_ids = np.ascontiguousarray(new_ids[order], dtype=np.int64)
                        new_tfs = np.asarray([chain[int(doc_id)] for doc_id in new_ids], dtype=np.float32)
                        self._posting_arrays[term] = (
                            np.ascontiguousarray(np.concatenate((ids, new_ids)), dtype=np.int64),
                            np.ascontiguousarray(np.concatenate((tfs, new_tfs)), dtype=np.float32),
                        )
                        self._clear_dirty_term_state(term)
                        continue

            ids = np.fromiter(chain.keys(), dtype=np.int64, count=len(chain))
            tfs = np.fromiter(chain.values(), dtype=np.float32, count=len(chain))
            if ids.size > 1:
                order = np.argsort(ids)
                ids = np.ascontiguousarray(ids[order], dtype=np.int64)
                tfs = np.ascontiguousarray(tfs[order], dtype=np.float32)
            else:
                ids = np.ascontiguousarray(ids, dtype=np.int64)
                tfs = np.ascontiguousarray(tfs, dtype=np.float32)
            self._posting_arrays[term] = (ids, tfs)
            self._clear_dirty_term_state(term)

    def _rebuild_posting_arrays(self) -> None:
        """Rebuild all posting arrays after full index construction or load."""
        self._ensure_dynamic_array_state()
        self._posting_arrays.clear()
        self._refresh_posting_arrays_for_terms(self.inverted_index.keys())
        self._dirty_terms.clear()
        self._dirty_term_appends.clear()
        self._dirty_term_rebuilds.clear()

    def _get_posting_arrays(self, term: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        self._ensure_dynamic_array_state()
        if term in self._dirty_terms:
            self._refresh_posting_arrays_for_terms([term])
            return self._posting_arrays.get(term)
        arrays = self._posting_arrays.get(term)
        if arrays is not None:
            return arrays
        chain = self.inverted_index.get(term)
        if not chain:
            return None
        self._refresh_posting_arrays_for_terms([term])
        return self._posting_arrays.get(term)

    def _build_local_lookup(self, int_id_array: np.ndarray) -> Tuple[str, Any]:
        """Build an int-id to local-row lookup for scoring one query.

        Dense scratch arrays are faster for compact id ranges; sorted ids with
        searchsorted avoid allocating huge arrays for sparse id spaces.
        """
        if int_id_array.size == 0:
            return "empty", None
        max_id = int(int_id_array.max())
        dense_limit = max(65_536, int_id_array.size * self._DENSE_LOCAL_LOOKUP_MAX_RATIO)
        required_capacity = max_id + 1
        if 0 <= max_id and required_capacity <= dense_limit:
            values, stamps, generation = self._get_dense_lookup_scratch(required_capacity)
            values[int_id_array] = np.arange(int_id_array.size, dtype=np.intp)
            stamps[int_id_array] = generation
            return "scratch", (values, stamps, generation, required_capacity)

        order = np.argsort(int_id_array)
        return "searchsorted", (np.ascontiguousarray(int_id_array[order], dtype=np.int64), order.astype(np.intp, copy=False))

    def _get_dense_lookup_scratch(self, required_capacity: int) -> Tuple[np.ndarray, np.ndarray, int]:
        values = getattr(self._dense_lookup_scratch, "values", None)
        stamps = getattr(self._dense_lookup_scratch, "stamps", None)
        generation = getattr(self._dense_lookup_scratch, "generation", 0)
        required_capacity = max(int(required_capacity), self._DENSE_LOCAL_LOOKUP_INITIAL_CAPACITY)
        capacity = 1 << (required_capacity - 1).bit_length()
        if values is None or stamps is None or values.size < capacity or stamps.size < capacity:
            values = np.empty(capacity, dtype=np.intp)
            stamps = np.zeros(capacity, dtype=np.uint32)
            generation = 0
            self._dense_lookup_scratch.values = values
            self._dense_lookup_scratch.stamps = stamps
        generation += 1
        if generation >= np.iinfo(np.uint32).max:
            stamps.fill(0)
            generation = 1
        self._dense_lookup_scratch.generation = generation
        return values, stamps, generation

    def _map_global_ids_to_local_indices(self, global_ids: np.ndarray, lookup: Tuple[str, Any]) -> np.ndarray:
        mode, payload = lookup
        mapped = np.full(global_ids.size, -1, dtype=np.intp)
        if global_ids.size == 0 or mode == "empty":
            return mapped
        if mode == "dense":
            local_map = payload
            in_bounds = (global_ids >= 0) & (global_ids < local_map.size)
            if np.any(in_bounds):
                mapped[in_bounds] = local_map[global_ids[in_bounds]]
            return mapped
        if mode == "scratch":
            values, stamps, generation, capacity = payload
            in_bounds = (global_ids >= 0) & (global_ids < capacity)
            if np.any(in_bounds):
                candidates = global_ids[in_bounds]
                active = stamps[candidates] == generation
                if np.any(active):
                    mapped[np.flatnonzero(in_bounds)[active]] = values[candidates[active]]
            return mapped

        sorted_ids, order = payload
        positions = np.searchsorted(sorted_ids, global_ids)
        valid = positions < sorted_ids.size
        if np.any(valid):
            valid_positions = positions[valid]
            valid_indices = np.flatnonzero(valid)
            equal_mask = sorted_ids[valid_positions] == global_ids[valid_indices]
            if np.any(equal_mask):
                mapped[valid_indices[equal_mask]] = order[valid_positions[equal_mask]]
        return mapped

    def _candidate_ids_from_posting_arrays(
        self,
        arrays: List[Tuple[np.ndarray, np.ndarray]],
        filter_set: Optional[Set[int]],
    ) -> np.ndarray:
        """Collect unique document ids touched by query postings and filters."""
        if not arrays:
            return np.empty(0, dtype=np.int64)
        if len(arrays) == 1:
            candidate_ids = np.unique(arrays[0][0].astype(np.int64, copy=False))
        else:
            candidate_ids = np.unique(
                np.concatenate([posting_ids for posting_ids, _ in arrays]).astype(np.int64, copy=False)
            )
        if filter_set is not None:
            if not filter_set or candidate_ids.size == 0:
                return np.empty(0, dtype=np.int64)
            filter_ids = np.fromiter(filter_set, dtype=np.int64, count=len(filter_set))
            candidate_ids = candidate_ids[np.isin(candidate_ids, filter_ids, assume_unique=False)]
        return np.ascontiguousarray(candidate_ids, dtype=np.int64)

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
            _, stamps, generation = self._get_dense_lookup_scratch(required_capacity)
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

    def _materialize_tf_matrix_from_posting_arrays(
        self,
        candidate_ids: np.ndarray,
        posting_arrays: List[Tuple[np.ndarray, np.ndarray]],
    ) -> np.ndarray:
        """Materialize a candidate-term TF matrix for the GPU scoring path."""
        tf_matrix = np.zeros((candidate_ids.size, len(posting_arrays)), dtype=np.float32)
        local_lookup = self._build_local_lookup(candidate_ids)
        for column, (global_ids, tfs) in enumerate(posting_arrays):
            if NUMBA_AVAILABLE and local_lookup[0] == "scratch":
                values, stamps, generation, capacity = local_lookup[1]
                _numba_fill_tf_column_scratch(
                    global_ids, tfs, values, stamps, generation, capacity, column, tf_matrix,
                )
            elif NUMBA_AVAILABLE and local_lookup[0] == "searchsorted":
                sorted_ids, order = local_lookup[1]
                _numba_fill_tf_column_searchsorted(
                    global_ids, tfs, sorted_ids, order, column, tf_matrix,
                )
            else:
                mapped_indices = self._map_global_ids_to_local_indices(global_ids, local_lookup)
                valid_mask = mapped_indices != -1
                if np.any(valid_mask):
                    tf_matrix[mapped_indices[valid_mask], column] = tfs[valid_mask]
        return tf_matrix

    def _score_candidates_accumulator(
        self,
        candidate_ids: np.ndarray,
        doc_len_array: np.ndarray,
        idf_array: np.ndarray,
        qtf_array: np.ndarray,
        posting_arrays: List[Tuple[np.ndarray, np.ndarray]],
        avgdl: float,
    ) -> np.ndarray:
        """Score candidates directly from posting arrays without materializing TF."""
        scores = np.zeros(candidate_ids.size, dtype=np.float32)
        if candidate_ids.size == 0 or not posting_arrays:
            return scores

        local_lookup = self._build_local_lookup(candidate_ids)
        k1 = np.float32(self.k1)
        b = np.float32(self.b)
        avgdl32 = np.float32(avgdl)
        for idf, qtf, (global_ids, tfs) in zip(idf_array, qtf_array, posting_arrays):
            term_weight = np.float32(np.float32(idf) * np.float32(qtf))
            if NUMBA_AVAILABLE and local_lookup[0] == "scratch":
                values, stamps, generation, capacity = local_lookup[1]
                _numba_accumulate_bm25_scratch(
                    global_ids, tfs, values, stamps, generation, capacity,
                    doc_len_array, term_weight, k1, b, avgdl32, scores,
                )
            elif NUMBA_AVAILABLE and local_lookup[0] == "searchsorted":
                sorted_ids, order = local_lookup[1]
                _numba_accumulate_bm25_searchsorted(
                    global_ids, tfs, sorted_ids, order,
                    doc_len_array, term_weight, k1, b, avgdl32, scores,
                )
            else:
                mapped_indices = self._map_global_ids_to_local_indices(global_ids, local_lookup)
                valid_mask = mapped_indices != -1
                if np.any(valid_mask):
                    rows = mapped_indices[valid_mask]
                    tf_values = tfs[valid_mask].astype(np.float32, copy=False)
                    denom = tf_values + self.k1 * (
                        1.0 - self.b + self.b * (doc_len_array[rows] / avgdl)
                    )
                    scores[rows] += (tf_values * (self.k1 + 1.0) / denom) * term_weight
        return scores

    def _results_from_scores(
        self,
        final_scores: np.ndarray,
        int_id_list: List[int],
        top_k: int,
    ) -> List[RetrievalResult]:
        real_k = min(top_k, len(int_id_list))
        if real_k <= 0:
            return []
        if len(int_id_list) > real_k:
            top_indices = np.argpartition(final_scores, -real_k)[-real_k:]
            top_indices = top_indices[np.argsort(final_scores[top_indices])[::-1]]
        else:
            top_indices = np.argsort(final_scores)[::-1]

        results: List[RetrievalResult] = []
        for idx in top_indices:
            score = float(final_scores[int(idx)])
            if score <= 1e-6:
                continue
            int_id = int_id_list[int(idx)]
            unit = self._get_unit_by_internal_id(int_id)
            if unit:
                results.append(RetrievalResult(
                    unit, score, self.get_method_type(), {"bm25_score": score},
                ))
        return results

    def remove_uids(self, uids: List[str]) -> int:
        """Remove documents by UID while preserving shared UID/int-id mappings."""
        if not uids:
            return 0

        
        present_count = 0
        touched_terms: Set[str] = set()
        for int_id in self._uids_to_internal_ids(uids, create=False):
            if int_id not in self.doc_lengths:
                continue
            touched_terms.update(self._evict_document(int_id, update_arrays=False))
            present_count += 1

        if present_count:
            self._mark_terms_dirty_for_rebuild(touched_terms)
            self._invalidate_static_index()
            logger.info(
                f"remove_uids: removed {present_count} documents, "
                f"L1 remaining docs={self.total_docs}, vocab={len(self.inverted_index)}"
            )
        return present_count

    def build_index(self, units: Optional[List[MemoryUnit]] = None) -> None:
        """Build a complete BM25 inverted index from memory units.

        Args:
            units: Optional units to index. When omitted, units are read from
                the retrieval source.

        Notes:
            Existing postings, dirty-term state, and static acceleration
            matrices are discarded before rebuilding.
        """
        self._invalidate_static_index()

        if units is None:
            if hasattr(self.retrieval_source, 'get_all_units'):
                units = self.retrieval_source.get_all_units()
            elif hasattr(self.retrieval_source, 'memory_units'):
                units = list(self.retrieval_source.memory_units.values())
            else:
                units = []

        if not units:
            logger.warning("No units are available for BM25 index construction.")
            self.total_docs = 0
            self.total_doc_length = 0
            self.doc_lengths.clear()
            self.dfs.clear()
            self.inverted_index.clear()
            self._posting_arrays.clear()
            self._dirty_terms.clear()
            self._dirty_term_appends.clear()
            self._dirty_term_rebuilds.clear()
            self.doc_postings.clear()
            self._index_built = False
            return

        logger.info(f"Building BM25 inverted index for {len(units)} documents.")

        self.total_docs = 0
        self.total_doc_length = 0
        self.doc_lengths.clear()
        self.dfs.clear()
        self.inverted_index.clear()
        self._posting_arrays.clear()
        self._dirty_terms.clear()
        self._dirty_term_appends.clear()
        self._dirty_term_rebuilds.clear()
        self.doc_postings.clear()

        tokenized = self._batch_tokenize(units)

        
        for uid, tokens in tokenized:
            int_id = self._get_or_create_internal_id(uid)
            self._ingest_document(int_id, tokens, update_arrays=False)

        self._rebuild_posting_arrays()

        self._index_built = True
        vocab_size = len(self.inverted_index)
        avgdl = self.total_doc_length / max(1, self.total_docs)
        logger.info(
            f"BM25 inverted index built: docs={self.total_docs}, "
            f"vocab={vocab_size}, avgdl={avgdl:.1f}"
        )

    def add_units(self, new_units: List[MemoryUnit]) -> bool:
        """Incrementally add units and mark affected posting arrays dirty."""
        if not new_units:
            return True

        if not self._index_built:
            logger.info("BM25 index is not built; falling back to build_index.")
            self.build_index(new_units)
            return self._index_built

        try:
            tokenized = self._batch_tokenize(new_units)
            added = 0
            term_to_doc_ids: Dict[str, Set[int]] = {}
            for uid, tokens in tokenized:
                int_id = self._get_or_create_internal_id(uid)
                if int_id not in self.doc_lengths:
                    self._ingest_document(int_id, tokens, update_arrays=False)
                    for term in set(tokens):
                        term_to_doc_ids.setdefault(term, set()).add(int_id)
                    added += 1

            if added:
                self._mark_terms_dirty_for_append(term_to_doc_ids)
                self._invalidate_static_index()

            logger.info(f" BM25 incremental update complete: added {added}, total {self.total_docs}")
            return True

        except Exception as e:
            logger.error(f"BM25 incremental update failed: {e}")
            return False


    def _ensure_index_built(self) -> None:
        if not self._index_built:
            self.build_index()

    def _invalidate_static_index(self) -> None:
        """Invalidate the optional static CSR acceleration matrix."""
        self._static_mode = False
        self._static_matrix = None
        self._static_int_ids = []
        self._static_int_id_array = None
        self._static_vocab = {}
        self._static_idf = None
        self._static_doc_lengths = None
        self._static_avgdl = 1.0

    def build_freeze_index(self) -> bool:
        """Build a static sparse CSR matrix for repeated BM25 queries."""
        self._ensure_index_built()
        if self.total_docs == 0 or not self.doc_lengths or not self.inverted_index:
            logger.warning("BM25 static index build skipped because the dynamic index is empty.")
            self._invalidate_static_index()
            return False

        try:
            int_ids = sorted(int(int_id) for int_id in self.doc_lengths.keys())
            vocab_terms = sorted(str(term) for term in self.inverted_index.keys())
            row_by_int_id = {int_id: row for row, int_id in enumerate(int_ids)}
            col_by_term = {term: col for col, term in enumerate(vocab_terms)}

            avgdl = self.total_doc_length / max(1, self.total_docs)
            if avgdl == 0:
                avgdl = 1.0

            total_docs = max(1, self.total_docs)
            row_entries: List[List[Tuple[int, float]]] = [[] for _ in int_ids]
            nnz = 0

            for term, postings in self.inverted_index.items():
                term_key = str(term)
                col = col_by_term.get(term_key)
                if col is None:
                    continue
                df = self.dfs.get(term_key, 0)
                idf = math.log((total_docs - df + 0.5) / (df + 0.5) + 1.0)
                if idf <= 0:
                    continue

                for int_id_raw, tf_raw in postings.items():
                    int_id = int(int_id_raw)
                    row = row_by_int_id.get(int_id)
                    if row is None:
                        continue
                    tf = float(tf_raw)
                    if tf <= 0:
                        continue
                    doc_len = float(self.doc_lengths.get(int_id, 0))
                    denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / avgdl))
                    if denom <= 0:
                        continue
                    weight = (tf * (self.k1 + 1.0) / denom) * idf
                    if weight <= 0:
                        continue
                    row_entries[row].append((col, float(weight)))
                    nnz += 1

            if nnz == 0:
                logger.warning("BM25 static index build skipped because there are no valid non-zero entries.")
                self._invalidate_static_index()
                return False

            indptr = np.zeros(len(int_ids) + 1, dtype=np.int64)
            indices_parts: List[np.ndarray] = []
            data_parts: List[np.ndarray] = []
            cursor = 0
            for row, entries in enumerate(row_entries):
                if entries:
                    entries.sort(key=lambda item: item[0])
                    row_indices = np.fromiter((col for col, _ in entries), dtype=np.int64, count=len(entries))
                    row_data = np.fromiter((weight for _, weight in entries), dtype=np.float32, count=len(entries))
                    indices_parts.append(row_indices)
                    data_parts.append(row_data)
                    cursor += len(entries)
                indptr[row + 1] = cursor

            indices = np.concatenate(indices_parts).astype(np.int64, copy=False)
            data = np.concatenate(data_parts).astype(np.float32, copy=False)
            matrix = torch.sparse_csr_tensor(
                torch.from_numpy(indptr),
                torch.from_numpy(indices),
                torch.from_numpy(data),
                size=(len(int_ids), len(vocab_terms)),
                dtype=torch.float32,
            )
            if self._use_gpu:
                matrix = matrix.to(self._device)
                backend = "torch/cuda sparse_csr"
            else:
                backend = "torch/cpu sparse_csr"

            self._static_matrix = matrix
            self._static_int_ids = int_ids
            self._static_int_id_array = np.asarray(int_ids, dtype=np.int64)
            self._static_vocab = col_by_term
            self._static_idf = None
            self._static_doc_lengths = None
            self._static_avgdl = float(avgdl)
            self._static_mode = True

            logger.info(
                f"BM25 static acceleration index built: shape={tuple(matrix.shape)}, "
                f"nnz={nnz}, backend={backend}"
            )
            return True
        except Exception as e:
            logger.error(f"BM25 static index build failed: {e}", exc_info=True)
            self._invalidate_static_index()
            return False

    def _search_static(
        self,
        query: Union[str, QueryBundle],
        top_k: int,
        candidate_uids: Optional[List[str]],
        space_names: Optional[List[str]],
    ) -> Optional[List[RetrievalResult]]:
        """Use the optional static CSR matrix for BM25 scoring.

        Returns ``None`` when the static path cannot run, allowing callers to
        fall back to the dynamic posting-array implementation. Candidate UID and
        MemorySpace filters are converted to static row ids before scoring.
        """
        try:
            matrix = self._static_matrix
            if matrix is None or not isinstance(matrix, torch.Tensor):
                return None

            if isinstance(query, QueryBundle):
                query_words = query.get_or_compute_bm25_tokens(self._preprocess_text)
            else:
                query_words = self._preprocess_text(query)
            if not query_words:
                return []

            query_tf = Counter(query_words)
            active_items = [
                (self._static_vocab[term], float(tf))
                for term, tf in query_tf.items()
                if term in self._static_vocab and float(tf) > 0
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

            device = matrix.device
            with torch.no_grad():
                q_dense = torch.zeros((len(self._static_vocab), 1), dtype=torch.float32, device=device)
                q_indices = torch.as_tensor([col for col, _ in active_items], dtype=torch.long, device=device)
                q_values = torch.as_tensor([tf for _, tf in active_items], dtype=torch.float32, device=device)
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
                if score <= 1e-6:
                    continue
                int_id = int(source_int_ids[int(local_idx)])
                unit = self._get_unit_by_internal_id(int_id)
                if unit:
                    results.append(RetrievalResult(unit, score, self.get_method_type(), {"bm25_score": score}))
            return results
        except Exception as e:
            logger.warning(f"BM25 static fast path failed; falling back to dynamic retrieval: {e}")
            logger.debug(traceback.format_exc())
            return None

    def search(
        self,
        query: Union[str, QueryBundle],
        top_k: int = 10,
        candidate_uids: Optional[List[str]] = None,
        space_names: Optional[List[str]] = None,
        **kwargs,
    ) -> List[RetrievalResult]:
        """Retrieve BM25-ranked memory units.

        Args:
            query: Raw query text or QueryBundle with cached BM25 tokens.
            top_k: Maximum number of results to return.
            candidate_uids: Optional candidate UID set. When provided together
                with ``space_names``, BM25 uses the intersection of both filters.
            space_names: Optional MemorySpace names used to restrict retrieval.
            **kwargs: Accepts ``ms_names`` as a backward-compatible alias.

        Returns:
            RetrievalResult objects with ``metadata["bm25_score"]``.

        Notes:
            Pure space filters can reuse SemanticMap's cached int-id filter.
            Combined candidate and space filters are resolved through UID
            intersection to preserve existing semantics.
        """
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
        if self.total_docs == 0:
            return []

        
        
        space_filter_ids: Optional[np.ndarray] = None
        if space_names and candidate_uids is None:
            space_filter_ids = self._get_space_filter_internal_ids(space_names)
            if space_filter_ids is not None and space_filter_ids.size == 0:
                return []
        if space_names and space_filter_ids is None:
            try:
                space_uids = set(self._get_uids_from_spaces(space_names))
                if candidate_uids is None:
                    candidate_uids = list(space_uids)
                else:
                    candidate_uids = list(set(candidate_uids) & space_uids)
            except Exception as e:
                logger.warning(f"BM25 failed to resolve UIDs from spaces: {e}")
                return []

        if isinstance(query, QueryBundle):
            query_words = query.get_or_compute_bm25_tokens(self._preprocess_text)
        else:
            query_words = self._preprocess_text(query)

        if not query_words:
            return []

        query_tf = Counter(query_words)

        candidate_set: Optional[Set[int]]
        if space_filter_ids is not None:
            candidate_set = set(int(int_id) for int_id in space_filter_ids.tolist())
        else:
            candidate_set = (
                set(self._uids_to_internal_ids(candidate_uids, create=False))
                if candidate_uids is not None else None
            )
        if candidate_set is not None and not candidate_set:
            return []

        query_terms: List[str] = []
        query_posting_arrays: List[Tuple[np.ndarray, np.ndarray]] = []
        for term in query_tf.keys():
            posting_arrays = self._get_posting_arrays(term)
            if posting_arrays is None:
                continue
            query_terms.append(term)
            query_posting_arrays.append(posting_arrays)
        if not query_terms:
            return []

        if candidate_set is not None:
            query_posting_arrays = self._filter_posting_arrays_by_candidate_ids(
                query_posting_arrays, candidate_set,
            )
        candidate_ids = self._candidate_ids_from_posting_arrays(query_posting_arrays, None)
        if candidate_ids.size == 0:
            return []

        int_id_list = [int(int_id) for int_id in candidate_ids.tolist()]
        candidate_count = int(candidate_ids.size)
        query_count = len(query_terms)

        avgdl = self.total_doc_length / max(1, self.total_docs)
        if avgdl == 0:
            avgdl = 1.0
        N = self.total_docs

        doc_len_array = np.fromiter(
            (self.doc_lengths.get(int_id, 0) for int_id in int_id_list),
            dtype=np.float32,
            count=candidate_count,
        )
        idf_array = np.empty(query_count, dtype=np.float32)
        qtf_array = np.empty(query_count, dtype=np.float32)

        for j, term in enumerate(query_terms):
            df = self.dfs.get(term, 0)
            idf_array[j] = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
            qtf_array[j] = query_tf[term]

        cell_count = candidate_count * query_count

        try:
            if self._use_gpu and cell_count >= self._gpu_materialize_threshold:
                tf_matrix = self._materialize_tf_matrix_from_posting_arrays(
                    candidate_ids, query_posting_arrays,
                )
                return self._search_gpu(
                    tf_matrix, doc_len_array, idf_array, qtf_array, avgdl,
                    int_id_list, top_k,
                )
            else:
                final_scores = self._score_candidates_accumulator(
                    candidate_ids, doc_len_array, idf_array, qtf_array,
                    query_posting_arrays, avgdl,
                )
                return self._results_from_scores(final_scores, int_id_list, top_k)
        except Exception as e:
            logger.error(f"BM25 retrieval failed: {e}", exc_info=True)
            return []

    def _search_gpu(
        self,
        tf_matrix: np.ndarray,
        doc_len_array: np.ndarray,
        idf_array: np.ndarray,
        qtf_array: np.ndarray,
        avgdl: float,
        int_id_list: List[int],
        top_k: int,
    ) -> List[RetrievalResult]:
        """Score a materialized TF matrix on CUDA when candidate volume is large."""
        device = self._device
        k1 = self.k1
        b = self.b

        stream = get_thread_local_cuda_stream(device)

        def compute_topk():
            tf_t = to_device_with_thread_local_pinned(torch.from_numpy(tf_matrix), device, "bm25_tf")          # (M, Q)
            dl_t = to_device_with_thread_local_pinned(torch.from_numpy(doc_len_array), device, "bm25_dl")      # (M,)
            idf_t = to_device_with_thread_local_pinned(torch.from_numpy(idf_array), device, "bm25_idf")        # (Q,)
            qtf_t = to_device_with_thread_local_pinned(torch.from_numpy(qtf_array), device, "bm25_qtf")        # (Q,)

            dl_norm = dl_t.unsqueeze(1) / avgdl                       # (M, 1)
            denom = tf_t + k1 * (1.0 - b + b * dl_norm)              # (M, Q)
            numer = tf_t * (k1 + 1.0)                                # (M, Q)
            term_scores = (numer / denom) * idf_t.unsqueeze(0) * qtf_t.unsqueeze(0)  # (M, Q)
            final_scores = term_scores.sum(dim=1)                     # (M,)
            real_k = min(top_k, final_scores.shape[0])
            return torch.topk(final_scores, real_k)

        with torch.no_grad():
            if stream is not None:
                with torch.cuda.stream(stream):
                    top_values, top_indices = compute_topk()
                stream.synchronize()
            else:
                top_values, top_indices = compute_topk()

        top_values_cpu = top_values.cpu().numpy()
        top_indices_cpu = top_indices.cpu().numpy()

        results: List[RetrievalResult] = []
        for i in range(len(top_indices_cpu)):
            score = float(top_values_cpu[i])
            if score <= 1e-6:
                continue
            int_id = int_id_list[int(top_indices_cpu[i])]
            unit = self._get_unit_by_internal_id(int_id)
            if unit:
                results.append(RetrievalResult(
                    unit, score, self.get_method_type(), {"bm25_score": score},
                ))
        return results

    def _search_cpu(
        self,
        tf_matrix: np.ndarray,
        doc_len_array: np.ndarray,
        idf_array: np.ndarray,
        qtf_array: np.ndarray,
        avgdl: float,
        int_id_list: List[int],
        top_k: int,
    ) -> List[RetrievalResult]:
        """Score a materialized TF matrix on CPU."""
        k1 = self.k1
        b = self.b
        M = tf_matrix.shape[0]

        # (M, 1)
        dl_norm = doc_len_array[:, np.newaxis] / avgdl
        denom = tf_matrix + k1 * (1.0 - b + b * dl_norm)                 # (M, Q)
        numer = tf_matrix * (k1 + 1.0)                                   # (M, Q)
        term_scores = (numer / denom) * idf_array[np.newaxis, :] * qtf_array[np.newaxis, :]  # (M, Q)
        final_scores = term_scores.sum(axis=1)                            # (M,)

        real_k = min(top_k, M)
        if M > real_k:
            top_indices = np.argpartition(final_scores, -real_k)[-real_k:]
            top_indices = top_indices[np.argsort(final_scores[top_indices])[::-1]]
        else:
            top_indices = np.argsort(final_scores)[::-1]

        results: List[RetrievalResult] = []
        for idx in top_indices:
            score = float(final_scores[idx])
            if score <= 1e-6:
                continue
            int_id = int_id_list[int(idx)]
            unit = self._get_unit_by_internal_id(int_id)
            if unit:
                results.append(RetrievalResult(
                    unit, score, self.get_method_type(), {"bm25_score": score},
                ))
        return results


    def save_index(self, directory: str) -> bool:
        """Persist BM25 dynamic postings and optional static CSR acceleration."""
        if not self._index_built or self.total_docs == 0:
            logger.warning("BM25 index has not been built; save skipped.")
            return False

        try:
            os.makedirs(directory, exist_ok=True)
            data = {
                "inverted_index": self.inverted_index,
                "doc_postings": self.doc_postings,
                "dfs": self.dfs,
                "doc_lengths": self.doc_lengths,
                "total_docs": self.total_docs,
                "total_doc_length": self.total_doc_length,
                "k1": self.k1,
                "b": self.b,
                "doc_id_type": "global_int_id",
                "uid_to_int_id": self._current_uid_to_int_id_map(),
                "version": "3.0_global_int_id",
            }

            static_matrix_file = "bm25_static_matrix.safetensors"
            if self._static_mode and self._static_matrix is not None:
                try:
                    static_matrix = self._static_matrix
                    if static_matrix.layout != torch.sparse_csr:
                        raise ValueError(f"Unsupported static matrix layout: {static_matrix.layout}")

                    data["static_meta"] = {
                        "version": "1.0_sparse_csr_safetensors",
                        "layout": "sparse_csr",
                        "matrix_file": static_matrix_file,
                        "int_ids": [int(int_id) for int_id in self._static_int_ids],
                        "vocab": dict(self._static_vocab),
                        "avgdl": float(self._static_avgdl),
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
                    logger.info(f"BM25 static acceleration matrix saved: {static_matrix_file}")
                except Exception as e:
                    data.pop("static_meta", None)
                    logger.warning(f"Failed to save BM25 static acceleration matrix; saving dynamic index only: {e}")

            path = os.path.join(directory, "bm25_index.pkl")
            with open(path, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

            size_mb = os.path.getsize(path) / (1024 * 1024)
            logger.info(
                f"BM25 index saved to {directory} "
                f"(docs={self.total_docs}, vocab={len(self.inverted_index)}, "
                f"size={size_mb:.2f} MB)"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to save BM25 index: {e}")
            return False

    def load_index(self, directory: str) -> bool:
        """Load a persisted BM25 index if its format is compatible.

        Legacy metadata-only formats return ``False`` so the caller can rebuild
        from MemoryUnits rather than using incomplete UID/int-id mappings.
        """
        new_path = os.path.join(directory, "bm25_index.pkl")
        legacy_path = os.path.join(directory, "bm25_metadata.pkl")

        target_path = None
        if os.path.exists(new_path):
            target_path = new_path
        elif os.path.exists(legacy_path):
            logger.warning(
                f"Detected an incompatible BM25 index format ({legacy_path}); "
                f"the index will be rebuilt on first retrieval."
            )
            return False
        else:
            logger.warning(f"BM25 index file does not exist: {directory}")
            return False

        try:
            with open(target_path, "rb") as f:
                data = pickle.load(f)

            self._restore_mapping_from_payload(data.get("uid_to_int_id"))
            self.inverted_index = self._normalize_inverted_index(data["inverted_index"])
            self.dfs = data["dfs"]
            self.doc_lengths = self._normalize_doc_lengths(data["doc_lengths"])
            self.total_docs = int(data.get("total_docs", len(self.doc_lengths)))
            self.total_doc_length = int(data.get("total_doc_length", sum(self.doc_lengths.values())))
            if "k1" in data:
                self.k1 = data["k1"]
            if "b" in data:
                self.b = data["b"]

            
            self.doc_postings = self._normalize_doc_postings(data.get("doc_postings") or {})
            if not self.doc_postings and self.inverted_index:
                logger.info("BM25 index is missing doc_postings; rebuilding the reverse index from postings.")
                self._rebuild_doc_postings_from_inverted()
            if self.total_docs != len(self.doc_lengths):
                self.total_docs = len(self.doc_lengths)
            if self.total_doc_length != sum(self.doc_lengths.values()):
                self.total_doc_length = sum(self.doc_lengths.values())
            self._rebuild_posting_arrays()

            self._index_built = True
            self._invalidate_static_index()
            avgdl = self.total_doc_length / max(1, self.total_docs)

            static_meta = data.get("static_meta")
            if isinstance(static_meta, dict):
                static_matrix_file = static_meta.get("matrix_file", "bm25_static_matrix.safetensors")
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
                        self._static_vocab = {
                            str(term): int(col)
                            for term, col in (static_meta.get("vocab") or {}).items()
                        }
                        self._static_idf = None
                        self._static_doc_lengths = None
                        self._static_avgdl = float(static_meta.get("avgdl", avgdl if avgdl else 1.0))
                        self._static_mode = True
                        logger.info(
                            f"BM25 static acceleration matrix restored: shape={tuple(matrix.shape)}, "
                            f"backend={matrix.device}"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to restore BM25 static acceleration matrix; falling back to dynamic postings: {e}")
                        self._invalidate_static_index()
                else:
                    logger.warning(f"BM25 static metadata exists but the matrix file is missing: {static_matrix_path}")

            logger.info(
                f"BM25 index loaded from disk: {directory} "
                f"(docs={self.total_docs}, vocab={len(self.inverted_index)}, "
                f"avgdl={avgdl:.1f})"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load BM25 index; it will be rebuilt: {e}")
            self._index_built = False
            self.inverted_index.clear()
            self._posting_arrays.clear()
            self.dfs.clear()
            self.doc_lengths.clear()
            self.doc_postings.clear()
            self.total_docs = 0
            self.total_doc_length = 0
            self._invalidate_static_index()
            return False

    # Storage operations must preserve transactional consistency.

    def _rebuild_doc_postings_from_inverted(self) -> None:
        """Rebuild doc postings from inverted."""
        rebuilt: Dict[int, Dict[str, int]] = {}
        for term, postings in self.inverted_index.items():
            for int_id, tf in postings.items():
                int_id = int(int_id)
                terms = rebuilt.get(int_id)
                if terms is None:
                    terms = {}
                    rebuilt[int_id] = terms
                terms[term] = int(tf)
        self.doc_postings = rebuilt
        logger.info(
            f"_rebuild_doc_postings: rebuilt reverse postings from inverted_index, "
            f"docs={len(self.doc_postings)}"
        )

    def _normalize_inverted_index(self, raw_index: Dict[str, Dict[Any, int]]) -> Dict[str, Dict[int, int]]:
        normalized: Dict[str, Dict[int, int]] = {}
        for term, postings in (raw_index or {}).items():
            term_postings: Dict[int, int] = {}
            for doc_key, tf in postings.items():
                int_id = self._coerce_doc_key_to_int_id(doc_key)
                if int_id is None:
                    continue
                term_postings[int_id] = int(tf)
            if term_postings:
                normalized[str(term)] = term_postings
        return normalized

    def _normalize_doc_lengths(self, raw_lengths: Dict[Any, int]) -> Dict[int, int]:
        normalized: Dict[int, int] = {}
        for doc_key, doc_len in (raw_lengths or {}).items():
            int_id = self._coerce_doc_key_to_int_id(doc_key)
            if int_id is not None:
                normalized[int_id] = int(doc_len)
        return normalized

    def _normalize_doc_postings(self, raw_postings: Dict[Any, Dict[str, int]]) -> Dict[int, Dict[str, int]]:
        normalized: Dict[int, Dict[str, int]] = {}
        for doc_key, postings in (raw_postings or {}).items():
            int_id = self._coerce_doc_key_to_int_id(doc_key)
            if int_id is None or not isinstance(postings, dict):
                continue
            normalized[int_id] = {str(term): int(tf) for term, tf in postings.items()}
        return normalized

    def get_index_stats(self) -> Dict[str, Union[int, float, bool]]:
        """Return index stats."""
        avgdl = self.total_doc_length / max(1, self.total_docs)
        return {
            "index_built": self._index_built,
            "total_docs": self.total_docs,
            "vocab_size": len(self.inverted_index),
            "reverse_index_docs": len(self.doc_postings),
            "avgdl": avgdl,
            "total_doc_length": self.total_doc_length,
            "use_gpu": self._use_gpu,
            "device": str(self._device),
            "k1": self.k1,
            "b": self.b,
            "static_mode": self._static_mode,
            "static_matrix_shape": tuple(self._static_matrix.shape) if self._static_matrix is not None else None,
        }
