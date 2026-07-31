"""Tiered L1/L2 storage manager with delegate callbacks."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
import threading
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional

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
    """Coordinate resident payload caching and RocksDB paging via callbacks.

    The manager owns payload eviction policy and I/O orchestration only.
    Retrieval indexes and graph state remain resident and are never mutated by
    this class.
    """

    REQUIRED_CALLBACKS = ("get_l1_data_cb", "remove_from_l1_cb", "add_to_l1_cb")

    def __init__(
        self,
        payload_store: Any,
        callbacks: Dict[str, Callable[..., Any]],
        max_capacity: int = 100_000,
        high_watermark: float = 0.85,
        low_watermark: float = 0.70,
        l1_mutation_lock: Optional[threading.RLock] = None,
    ) -> None:
        if payload_store is None:
            raise ValueError("TieredStorageManager requires a payload store")
        missing = [name for name in self.REQUIRED_CALLBACKS if name not in callbacks]
        if missing:
            raise ValueError(f"TieredStorageManager missing callbacks: {missing}")
        if max_capacity <= 0:
            raise ValueError("max_capacity must be positive")

        self.payload_store = payload_store
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
        self._quiesce_depth = 0
        self._shutdown = False

    def _resolve_watermark(self, value: float) -> int:
        if 0 < value <= 1:
            return max(1, int(self.max_capacity * value))
        return int(value)

    def is_eviction_running(self) -> bool:
        with self.eviction_lock:
            return self._eviction_future is not None and not self._eviction_future.done()

    def check_and_trigger_eviction(self, current_size: int) -> Optional[Future]:
        """Submit a background eviction job when L1 reaches the high watermark."""
        with self.eviction_lock:
            if (
                self._shutdown
                or self._quiesce_depth > 0
                or current_size < self.high_watermark_count
            ):
                return None
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

    def _evict_once(self, current_size: int, count: Optional[int] = None) -> TieredEvictionResult:
        """Run one synchronous eviction cycle for internal maintenance."""
        requested_count, selected_uids, l1_batch_data = self._prepare_eviction_payload(
            int(current_size), count
        )
        return self._persist_prepared_eviction(requested_count, selected_uids, l1_batch_data)

    def wait_for_idle(self) -> None:
        """Wait for the currently submitted eviction, if any, to complete."""
        with self.eviction_lock:
            future = self._eviction_future
        if future is None:
            return
        result = future.result()
        if result.error:
            raise RuntimeError(f"Tiered eviction failed: {result.error}")

    @contextmanager
    def quiesce(
        self,
        current_size_cb: Optional[Callable[[], int]] = None,
    ) -> Iterator[None]:
        """Pause eviction scheduling while a graph snapshot is created.

        Existing work is awaited without holding ``eviction_lock``. Normal
        asynchronous scheduling resumes when the outermost quiescence scope
        exits.
        """
        with self.eviction_lock:
            if self._shutdown:
                raise RuntimeError("Tiered storage manager is closed.")
            self._quiesce_depth += 1
        try:
            self.wait_for_idle()
            yield
        finally:
            should_resume = False
            with self.eviction_lock:
                self._quiesce_depth -= 1
                should_resume = self._quiesce_depth == 0 and not self._shutdown
            if should_resume and current_size_cb is not None:
                try:
                    self.check_and_trigger_eviction(int(current_size_cb()))
                except Exception as exc:
                    logger.error(
                        "Failed to resume tiered eviction scheduling: %s",
                        exc,
                        exc_info=True,
                    )

    def handle_page_fault_batch(
        self,
        missing_uids: List[str],
    ) -> List[MemoryUnit]:
        """Batch-load missing payloads and publish them to the resident cache."""
        unique_uids = self._dedupe(missing_uids)
        if not unique_uids:
            return []

        recovered_units: List[MemoryUnit] = list(
            self.payload_store.swap_in(unique_uids) or []
        )
        if not recovered_units:
            return []

        with self.L1_mutation_lock:
            self.callbacks["add_to_l1_cb"](recovered_units)

        return recovered_units

    def shutdown(self, wait: bool = True) -> None:
        with self.eviction_lock:
            if self._shutdown:
                return
            self._shutdown = True
            future = self._eviction_future
        if wait and future is not None:
            result = future.result()
            if result.error:
                logger.error("Tiered eviction failed during shutdown: %s", result.error)
        self.executor.shutdown(wait=wait)

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
            persisted_count = int(self.payload_store.swap_out(selected_uids, l1_batch_data) or 0)
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
