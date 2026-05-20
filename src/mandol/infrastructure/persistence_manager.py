"""Persistence manager for loading and saving complete MemorySystem state.

Orchestrates periodic auto-save, dirty-triggered debounced saves, and
full state restoration from JSON. Thread-safe via daemon worker thread.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import numpy as np

if TYPE_CHECKING:
    from ..application.memory_system import MemorySystem

from .json_persistence import (
    CURRENT_VERSION,
    JsonPersistenceEngine,
    PersistenceError,
    SaveResult,
    VerificationResult,
)

logger = logging.getLogger(__name__)


class PersistenceManager:
    def __init__(
        self,
        storage_root: str,
        system: "MemorySystem",
        auto_save_interval: int = 300,
    ):
        self._root = str(storage_root)
        self._system = system
        self._auto_save_interval = int(auto_save_interval)
        self._engine = JsonPersistenceEngine(self._root)
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._pending_save = False
        self._save_lock = threading.Lock()
        self._timer_task: Optional[threading.Thread] = None
        self._stop_timer = threading.Event()

    @property
    def storage_root(self) -> str:
        return self._root

    def save_full(self) -> SaveResult:
        start_time = datetime.now(timezone.utc)
        errors: List[str] = []

        try:
            self._engine.ensure_directories()

            units = self._system.semantic_map.list_units()
            spaces = self._system.semantic_map.list_spaces()
            edges = self._extract_graph_edges()
            sessions = self._system._session_manager.get_sessions()

            pending_session_ids = self._get_pending_session_ids()

            self._engine.save_units(units)
            self._engine.save_spaces(spaces)
            self._engine.save_graph(edges)
            self._engine.save_sessions(sessions)

            self._engine.save_processed_state(
                insertion_order=self._system._insertion_order.copy(),
                processed_session_ids=self._system._processed_session_ids.copy(),
                processed_similarity_pairs=self._system._processed_similarity_pairs.copy(),
                last_async_reasoning=getattr(self._system, '_last_async_reasoning', ''),
                async_check_scheduled=getattr(self._system, '_async_check_scheduled', False),
            )

            build_state = "partial" if pending_session_ids else "complete"
            stats = self._engine.get_storage_stats()

            self._engine.save_manifest(
                build_state=build_state,
                pending_session_ids=list(pending_session_ids),
                stats=stats,
            )

            self._system._dirty = False

            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()

            logger.info(f"Persistence save completed in {duration:.2f}s: {stats}")

            return SaveResult(
                success=True,
                saved_at=end_time.isoformat(),
                stats=stats,
                errors=errors,
            )

        except Exception as e:
            errors.append(str(e))
            logger.error(f"Persistence save failed: {e}")
            return SaveResult(
                success=False,
                saved_at=datetime.now(timezone.utc).isoformat(),
                stats={},
                errors=errors,
            )

    def schedule_save(self, delay_seconds: int = 5) -> None:
        with self._save_lock:
            if self._pending_save:
                return
            self._pending_save = True
            self._executor.submit(self._delayed_save, delay_seconds)

    def _delayed_save(self, delay: int) -> None:
        time.sleep(delay)
        # Skip save if sync build is in progress
        if getattr(self._system, '_auto_save_paused', False):
            with self._save_lock:
                self._pending_save = False
            return
        try:
            self.save_full()
        except Exception as e:
            logger.error(f"Delayed save failed: {e}")
        finally:
            with self._save_lock:
                self._pending_save = False

    def load_full(self) -> Tuple[List[Any], List[Any], List[Tuple[str, str, str, Dict[str, Any]]], List[Any], List[str], Set[str], Set[Tuple[str, str]], str, bool]:
        try:
            manifest = self._engine.load_manifest()
            if manifest is None:
                logger.warning("No manifest found, returning empty state")
                return [], [], [], [], [], set(), set()

            version = manifest.get("version", "0.9")
            if version != CURRENT_VERSION:
                logger.warning(f"Version mismatch: expected {CURRENT_VERSION}, got {version}")

            units = self._engine.load_units()
            spaces = self._engine.load_spaces()
            edges = self._engine.load_graph()
            sessions = self._engine.load_sessions()
            insertion_order, processed_session_ids, processed_similarity_pairs, last_async_reasoning, async_check_scheduled = self._engine.load_processed_state()

            build_state = manifest.get("build_state", "complete")
            pending_session_ids = manifest.get("pending_session_ids", [])

            if build_state == "partial" and pending_session_ids:
                logger.warning(f"Found {len(pending_session_ids)} sessions with incomplete build")

            return (
                units,
                spaces,
                edges,
                sessions,
                insertion_order,
                processed_session_ids,
                processed_similarity_pairs,
                last_async_reasoning,
                async_check_scheduled,
            )

        except PersistenceError as e:
            logger.error(f"Failed to load persisted state: {e}")
            raise

    def rebuild_indexes(self, units: List[Any]) -> None:
        try:
            embedder = self._system.semantic_map._embedder
            if embedder is None:
                logger.warning("No embedder available, skipping vector index rebuild")
                return

            from ..domain.types import Uid
            items: List[Tuple[Uid, np.ndarray]] = []
            for u in units:
                if u.embedding is not None:
                    emb = np.asarray(u.embedding, dtype=np.float32).reshape(-1)
                    items.append((Uid(str(u.uid)), emb))

            if items:
                self._system._abi.rebuild(items)
                logger.info(f"Rebuilt vector index with {len(items)} vectors")

        except Exception as e:
            logger.error(f"Failed to rebuild indexes: {e}")
            raise

    def verify_integrity(self) -> VerificationResult:
        errors: List[str] = []
        warnings: List[str] = []

        try:
            manifest = self._engine.load_manifest()
            if manifest is None:
                errors.append("No manifest found")
                return VerificationResult(valid=False, errors=errors, warnings=warnings)

            units = self._engine.load_units()
            spaces = self._engine.load_spaces()
            edges = self._engine.load_graph()

            unit_uids = {str(u.uid) for u in units}
            for i, (source, target, rel_type, _) in enumerate(edges):
                if source not in unit_uids:
                    errors.append(f"Edge {i}: source '{source}' not found in units")
                if target not in unit_uids:
                    errors.append(f"Edge {i}: target '{target}' not found in units")

            for space in spaces:
                for uid in space.unit_uids:
                    if str(uid) not in unit_uids:
                        errors.append(f"Space '{space.name}': unit '{uid}' not found in units")

            return VerificationResult(
                valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
            )

        except PersistenceError as e:
            errors.append(f"Persistence error: {e}")
            return VerificationResult(valid=False, errors=errors, warnings=warnings)

    def start_auto_save(self) -> None:
        def _auto_save_loop():
            while not self._stop_timer.is_set():
                self._stop_timer.wait(self._auto_save_interval)
                if self._stop_timer.is_set():
                    break
                if self._system._dirty:
                    logger.info("Auto-save triggered")
                    self.schedule_save(delay_seconds=0)

        self._timer_task = threading.Thread(target=_auto_save_loop, daemon=True)
        self._timer_task.start()
        logger.info(f"Auto-save started with interval {self._auto_save_interval}s")

    def stop_auto_save(self) -> None:
        if self._timer_task is not None:
            self._stop_timer.set()
            self._timer_task.join(timeout=5)
            self._timer_task = None
            logger.info("Auto-save stopped")

    def _extract_graph_edges(self) -> List[Tuple[str, str, str, Dict[str, Any]]]:
        edges: List[Tuple[str, str, str, Dict[str, Any]]] = []

        try:
            graph = self._system._graph_store
            for source, target, rel_type, properties in graph.get_all_edges():
                edges.append((source, target, rel_type, properties))
        except Exception as e:
            logger.warning(f"Failed to extract graph edges: {e}")

        return edges

    def _get_pending_session_ids(self) -> Set[str]:
        pending = set()
        for s in self._system._session_manager.get_sessions():
            if s.session_id not in self._system._processed_session_ids:
                pending.add(s.session_id)
        return pending


class MemorySystemStateLoader:
    def __init__(self, persistence_manager: PersistenceManager):
        self._pm = persistence_manager

    def load_into_system(self, system: "MemorySystem") -> Optional[List[str]]:
        units, spaces, edges, sessions, insertion_order, processed_session_ids, processed_similarity_pairs, last_async_reasoning, async_check_scheduled = self._pm.load_full()

        for unit in units:
            existing = system.semantic_map.get_unit(unit.uid)
            if existing is None:
                system.semantic_map.add_unit(
                    unit,
                    space_names=[str(s) for s in unit.metadata.get("spaces", [])],
                    ensure_embedding=False,
                )

        for space in spaces:
            existing = system.semantic_map.get_space(space.name)
            if existing is None:
                system.semantic_map.create_space(space.name)

        for source, target, rel_type, properties in edges:
            try:
                system.graph.add_relationship(
                    source_uid=source,
                    target_uid=target,
                    relationship_name=rel_type,
                    **properties,
                )
            except Exception as e:
                logger.debug(f"Edge already exists or invalid: {e}")

        system._insertion_order = insertion_order
        system._processed_session_ids = processed_session_ids
        system._processed_similarity_pairs = processed_similarity_pairs
        system._last_async_reasoning = last_async_reasoning
        system._async_check_scheduled = async_check_scheduled

        system._session_manager.reset()
        for session in sessions:
            system._session_manager._sessions.append(session)

        self._pm.rebuild_indexes(units)

        manifest = self._pm._engine.load_manifest()
        pending_session_ids = []
        if manifest and manifest.get("build_state") == "partial":
            pending_session_ids = manifest.get("pending_session_ids", [])
            if pending_session_ids:
                logger.warning(f"System loaded with {len(pending_session_ids)} pending sessions to rebuild")

        # Resume async check if pending units and flag were saved
        if async_check_scheduled:
            with system._pending_lock:
                if len(system._pending_units) >= 2:
                    system._async_check_scheduled = True
                    system._executor.submit(system._do_async_check)
                else:
                    system._async_check_scheduled = False

        return pending_session_ids
