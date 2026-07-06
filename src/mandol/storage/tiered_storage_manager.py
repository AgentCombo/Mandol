"""Tiered L1/L2 storage manager with delegate callbacks."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import inspect
import threading
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np

from ..core.memory_unit import MemoryUnit
from ..utils.logging_config import create_module_logger

logger = create_module_logger("storage.tiered_storage_manager")


@dataclass
class TieredEvictionResult:
    requested_count: int
    selected_uids: List[str]
    persisted_count: int
    removed_count: int
    error: Optional[str] = None


class TieredStorageManager:
    """Coordinate L1 memory and L2 DuckDB paging via callbacks.

    The manager owns policy and I/O orchestration only. It never mutates FAISS,
    SPLADE, BM25, SemanticMap, or SemanticGraph internals directly. L1 mutation is
    delegated to callbacks registered by SemanticMap/SemanticGraph.
    """

    REQUIRED_CALLBACKS = ("get_l1_data_cb", "remove_from_l1_cb", "add_to_l1_cb")

    def __init__(
        self,
        duckdb_operator: Any,
        callbacks: Dict[str, Callable[..., Any]],
        max_capacity: int = 100_000,
        high_watermark: float = 0.85,
        low_watermark: float = 0.70,
        l1_mutation_lock: Optional[threading.RLock] = None,
    ) -> None:
        missing = [name for name in self.REQUIRED_CALLBACKS if name not in callbacks]
        if missing:
            raise ValueError(f"TieredStorageManager missing callbacks: {missing}")
        if max_capacity <= 0:
            raise ValueError("max_capacity must be positive")

        self.duckdb_operator = duckdb_operator
        self.callbacks = callbacks
        self.max_capacity = int(max_capacity)
        self.high_watermark = high_watermark
        self.low_watermark = low_watermark
        self.high_watermark_count = self._resolve_watermark(high_watermark)
        self.low_watermark_count = self._resolve_watermark(low_watermark)
        if self.low_watermark_count >= self.high_watermark_count:
            raise ValueError("low_watermark must be lower than high_watermark")

        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tiered-storage")
        self.eviction_lock = threading.Lock()
        self.L1_mutation_lock = l1_mutation_lock or threading.RLock()
        self._eviction_future: Optional[Future] = None

    def _resolve_watermark(self, value: float) -> int:
        if 0 < value <= 1:
            return max(1, int(self.max_capacity * value))
        return int(value)

    def is_eviction_running(self) -> bool:
        with self.eviction_lock:
            return self._eviction_future is not None and not self._eviction_future.done()

    def check_and_trigger_eviction(self, current_size: int) -> Optional[Future]:
        """Submit a background eviction job when L1 reaches the high watermark."""
        if current_size < self.high_watermark_count:
            return None

        with self.eviction_lock:
            if self._eviction_future is not None and not self._eviction_future.done():
                return self._eviction_future

            requested_count, selected_uids, l1_batch_data = self._prepare_eviction_payload(
                int(current_size), None
            )
            if requested_count <= 0 or not selected_uids:
                self._eviction_future = Future()
                self._eviction_future.set_result(
                    TieredEvictionResult(requested_count, selected_uids, 0, 0)
                )
            else:
                self._eviction_future = self.executor.submit(
                    self._persist_prepared_eviction,
                    requested_count,
                    selected_uids,
                    l1_batch_data,
                )
            self._eviction_future.add_done_callback(self._log_eviction_completion)
            return self._eviction_future

    def evict_once(self, current_size: int, count: Optional[int] = None) -> TieredEvictionResult:
        """Run one eviction cycle synchronously. Useful for legacy explicit swap calls."""
        requested_count, selected_uids, l1_batch_data = self._prepare_eviction_payload(
            int(current_size), count
        )
        return self._persist_prepared_eviction(requested_count, selected_uids, l1_batch_data)

    def handle_page_fault_batch(self, missing_uids: List[str]) -> List[MemoryUnit]:
        """Batch load missing UIDs from L2 and attach them back to L1."""
        unique_uids = self._dedupe(missing_uids)
        if not unique_uids:
            return []

        payload = self._swap_in_for_page_fault(unique_uids)
        recovered_units: List[MemoryUnit] = list(payload.get("units", []) or [])
        if not recovered_units:
            return []

        dense_matrix = payload.get("dense_matrix")
        uid_order = payload.get("uid_order") or [unit.uid for unit in recovered_units]
        splade_dicts = self._splade_dicts_from_payload(payload, uid_order)

        with self.L1_mutation_lock:
            self.callbacks["add_to_l1_cb"](recovered_units, dense_matrix, splade_dicts)

        return recovered_units

    def shutdown(self, wait: bool = True) -> None:
        self.executor.shutdown(wait=wait)

    def _swap_in_for_page_fault(self, unique_uids: List[str]) -> Dict[str, Any]:
        swap_in = self.duckdb_operator.swap_in
        try:
            if "include_splade_csr" in inspect.signature(swap_in).parameters:
                return swap_in(unique_uids, include_splade_csr=False)
        except (TypeError, ValueError):
            pass
        return swap_in(unique_uids)

    def _evict_to_low_watermark(
        self,
        current_size: int,
        explicit_count: Optional[int],
    ) -> TieredEvictionResult:
        requested_count, selected_uids, l1_batch_data = self._prepare_eviction_payload(
            current_size, explicit_count
        )
        return self._persist_prepared_eviction(requested_count, selected_uids, l1_batch_data)

    def _prepare_eviction_payload(
        self,
        current_size: int,
        explicit_count: Optional[int],
    ) -> tuple[int, List[str], Dict[str, Any]]:
        requested_count = self._requested_eviction_count(current_size, explicit_count)
        if requested_count <= 0:
            return 0, [], {}

        try:
            with self.L1_mutation_lock:
                l1_batch_data = self.callbacks["get_l1_data_cb"](requested_count)
            selected_uids = self._extract_uids(l1_batch_data)
            return requested_count, selected_uids, l1_batch_data
        except Exception as exc:
            logger.error("Tiered eviction payload preparation failed: %s", exc, exc_info=True)
            return requested_count, [], {"_prepare_error": str(exc)}

    def _persist_prepared_eviction(
        self,
        requested_count: int,
        selected_uids: List[str],
        l1_batch_data: Dict[str, Any],
    ) -> TieredEvictionResult:
        if requested_count <= 0 or not selected_uids:
            error = l1_batch_data.get("_prepare_error") if isinstance(l1_batch_data, dict) else None
            return TieredEvictionResult(requested_count, selected_uids, 0, 0, error=error)

        try:
            persisted_count = int(self.duckdb_operator.swap_out(selected_uids, l1_batch_data) or 0)
            persisted_uids = selected_uids[:persisted_count]
            if not persisted_uids:
                return TieredEvictionResult(requested_count, selected_uids, persisted_count, 0)

            with self.L1_mutation_lock:
                removed = self.callbacks["remove_from_l1_cb"](persisted_uids)
            removed_count = int(removed if removed is not None else len(persisted_uids))
            return TieredEvictionResult(requested_count, selected_uids, persisted_count, removed_count)
        except Exception as exc:
            logger.error("Tiered eviction failed: %s", exc, exc_info=True)
            return TieredEvictionResult(requested_count, selected_uids, 0, 0, error=str(exc))

    def _requested_eviction_count(self, current_size: int, explicit_count: Optional[int]) -> int:
        if explicit_count is not None:
            return max(0, min(int(explicit_count), current_size))
        return max(0, current_size - self.low_watermark_count)

    def _extract_uids(self, l1_batch_data: Dict[str, Any]) -> List[str]:
        uid_order = l1_batch_data.get("uid_order")
        if uid_order:
            return self._dedupe(uid_order)
        units = l1_batch_data.get("units") or []
        return self._dedupe([unit.uid for unit in units if hasattr(unit, "uid")])

    @staticmethod
    def _dedupe(uids: Iterable[str]) -> List[str]:
        seen = set()
        result = []
        for uid in uids:
            if uid is None:
                continue
            uid_str = str(uid)
            if uid_str not in seen:
                seen.add(uid_str)
                result.append(uid_str)
        return result

    @staticmethod
    def _splade_dicts_from_payload(payload: Dict[str, Any], uid_order: List[str]) -> Dict[str, Dict[int, float]]:
        indices_rows = payload.get("splade_indices") if isinstance(payload, dict) else None
        values_rows = payload.get("splade_values") if isinstance(payload, dict) else None
        if indices_rows is not None and values_rows is not None:
            if hasattr(indices_rows, "to_pylist"):
                indices_rows = indices_rows.to_pylist()
            if hasattr(values_rows, "to_pylist"):
                values_rows = values_rows.to_pylist()
            splade_dicts: Dict[str, Dict[int, float]] = {}
            for uid, indices, values in zip(uid_order, indices_rows, values_rows):
                if indices is None or values is None:
                    continue
                if len(indices) == 0 or len(values) == 0:
                    continue
                splade_dicts[str(uid)] = {int(idx): float(value) for idx, value in zip(indices, values)}
            return splade_dicts

        splade_csr = payload.get("splade_csr") if isinstance(payload, dict) else None
        if splade_csr is None:
            return {}
        splade_dicts: Dict[str, Dict[int, float]] = {}
        for row_index, uid in enumerate(uid_order):
            if row_index >= getattr(splade_csr, "shape", (0,))[0]:
                break
            row = splade_csr.getrow(row_index)
            if getattr(row, "nnz", 0) <= 0:
                continue
            indices = np.asarray(row.indices, dtype=np.int64)
            values = np.asarray(row.data, dtype=np.float32)
            splade_dicts[str(uid)] = {int(idx): float(value) for idx, value in zip(indices, values)}
        return splade_dicts

    @staticmethod
    def _log_eviction_completion(future: Future) -> None:
        try:
            result = future.result()
            if result.error:
                logger.warning("Tiered eviction completed with error: %s", result.error)
            elif result.removed_count:
                logger.info(
                    "Tiered eviction completed: selected=%s persisted=%s removed=%s",
                    len(result.selected_uids),
                    result.persisted_count,
                    result.removed_count,
                )
        except Exception as exc:
            logger.error("Tiered eviction future failed: %s", exc, exc_info=True)
