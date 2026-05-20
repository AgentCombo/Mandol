"""Persistence service for saving and loading memory system state.

Manages JSON-based serialization of SemanticMapService and config snapshot,
including session tracking, edge-graph persistence, and file I/O.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from ...domain.memory_unit import MemoryUnit
from ...domain.types import SpaceName, Uid

logger = logging.getLogger(__name__)


@dataclass
class SaveResult:
    """Result of a save operation.

    Attributes:
        success: Whether the save completed without critical errors.
        saved_at: ISO 8601 timestamp of when the save completed.
        stats: Dictionary of save statistics (e.g. unit_count, space_count).
        errors: List of non-critical error messages encountered during saving.
    """
    success: bool
    saved_at: str
    stats: Dict[str, Any]
    errors: List[str]


@dataclass
class LoadResult:
    """Result of a load operation.

    Attributes:
        success: Whether the load completed without errors.
        loaded_from: Path the state was loaded from.
        stats: Dict of loading statistics.
        errors: List of error messages encountered.
    """
    success: bool
    loaded_from: str
    stats: Dict[str, Any]
    errors: List[str]


class MemoryPersistenceService:
    """Handles save/load of MemorySystem state to JSON files.

    Delegates to JsonPersistenceEngine for file I/O and manages serialization
    of units, spaces, edges, sessions, processed state tracking, and config
    snapshots.

    Args:
        semantic_map: SemanticMapService for unit serialization.
        graph_store: Underlying GraphStore for edge extraction.
        naming: SpaceNamingPolicy for space name resolution.
        root: Root SpaceName for the system.
        config: MemorySystemConfig for config serialization.
        session_manager: SessionManager for session state.
        abi: Additional build info dict or None.
    """

    def __init__(
        self,
        semantic_map,
        graph_store,
        naming,
        root: SpaceName,
        config,
        session_manager,
        abi,
    ):
        self._semantic_map = semantic_map
        self._graph_store = graph_store
        self._naming = naming
        self._root = root
        self._cfg = config
        self._session_manager = session_manager
        self._abi = abi

        self._dirty = False
        self._insertion_order: List[str] = []
        self._processed_session_ids: Set[str] = set()
        self._processed_similarity_pairs: Set[Tuple[str, str]] = set()
        self._pending_lock = None
        self._pending_units = None
        self._pending_events = None
        self._pending_entities = None
        self._all_events = None
        self._all_entities = None
        self._last_async_reasoning: str = ""
        self._async_check_scheduled: bool = False

    def attach_state(
        self,
        *,
        insertion_order: List[str],
        processed_session_ids: Set[str],
        processed_similarity_pairs: Set[Tuple[str, str]],
        pending_lock,
        pending_units,
        pending_events: list,
        pending_entities: list,
        all_events: list,
        all_entities: list,
        last_async_reasoning: str = "",
        async_check_scheduled: bool = False,
        dirty: bool = False,
    ) -> None:
        """Attach runtime mutable state from MemorySystem for persistence.

        Binds references to the shared mutable containers so that save/load
        can serialise and restore the full system state.

        Args:
            insertion_order: Ordered list of unit UIDs as inserted.
            processed_session_ids: Set of session IDs already processed.
            processed_similarity_pairs: Set of (uid1, uid2) pairs with edges.
            pending_lock: Threading lock guarding pending buffers.
            pending_units: List of pending MemoryUnits awaiting processing.
            pending_events: List of pending extracted events.
            pending_entities: List of pending extracted entities.
            all_events: List of all extracted events across sessions.
            all_entities: List of all extracted entities across sessions.
            last_async_reasoning: Current async reasoning chain latest value.
            async_check_scheduled: Dirty-flag state for async check.
            dirty: Whether the system has unsaved changes.
        """
        self._insertion_order = insertion_order
        self._processed_session_ids = processed_session_ids
        self._processed_similarity_pairs = processed_similarity_pairs
        self._pending_lock = pending_lock
        self._pending_units = pending_units
        self._pending_events = pending_events
        self._pending_entities = pending_entities
        self._all_events = all_events
        self._all_entities = all_entities
        self._last_async_reasoning = last_async_reasoning
        self._async_check_scheduled = async_check_scheduled

    def save(self, storage_path: Optional[str] = None, persistence=None) -> "SaveResult":
        """Persist the memory system state to disk.

        If *storage_path* is provided, saves directly to that directory.
        If *persistence* is provided, delegates to its ``save_full`` method.
        One of the two must be supplied.

        Args:
            storage_path: Target directory path for the JSON save.
            persistence: PersistenceManager instance (when auto-save is enabled).

        Returns:
            SaveResult indicating success, timestamp, stats, and any errors.

        Raises:
            ValueError: If neither storage_path nor persistence is provided.
        """
        if storage_path is not None:
            return self._save_to_path(storage_path)

        if persistence is not None:
            return persistence.save_full()

        raise ValueError(
            "storage_path is required when persistence is not enabled. "
            "Call save('/path/to/dir') to save to a specific directory."
        )

    def _save_to_path(self, storage_path: str) -> "SaveResult":
        from ...infrastructure.json_persistence import (
            JsonPersistenceEngine,
        )
        from dataclasses import asdict

        start_time = datetime.now(timezone.utc)
        errors: List[str] = []

        try:
            engine = JsonPersistenceEngine(storage_path)
            engine.ensure_directories()

            units = self._semantic_map.list_units()
            spaces = self._semantic_map.list_spaces()
            edges = self._extract_graph_edges()
            sessions = self._session_manager.get_sessions()

            engine.save_units(units)
            engine.save_spaces(spaces)
            engine.save_graph(edges)
            engine.save_sessions(sessions)

            engine.save_processed_state(
                insertion_order=self._insertion_order.copy(),
                processed_session_ids=self._processed_session_ids.copy(),
                processed_similarity_pairs=self._processed_similarity_pairs.copy(),
                last_async_reasoning=self._last_async_reasoning if isinstance(self._last_async_reasoning, str) else "",
                async_check_scheduled=self._async_check_scheduled if isinstance(self._async_check_scheduled, bool) else False,
            )

            engine.save_config({
                "memory_system_config": asdict(self._cfg),
                "root": self._root,
            })

            pending_session_ids = self._get_pending_session_ids()
            build_state = "partial" if pending_session_ids else "complete"
            stats = engine.get_storage_stats()

            engine.save_manifest(
                build_state=build_state,
                pending_session_ids=list(pending_session_ids),
                stats=stats,
            )

            self._dirty = False

            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()
            logger.info(f"Save completed in {duration:.2f}s: {stats}")

            return SaveResult(
                success=True,
                saved_at=end_time.isoformat(),
                stats=stats,
                errors=errors,
            )

        except Exception as e:
            errors.append(str(e))
            logger.error(f"Save failed: {e}")
            return SaveResult(
                success=False,
                saved_at=datetime.now(timezone.utc).isoformat(),
                stats={},
                errors=errors,
            )

    def _load(self, storage_path: str) -> LoadResult:
        from ...infrastructure.json_persistence import JsonPersistenceEngine

        start_time = datetime.now(timezone.utc)
        errors: List[str] = []

        try:
            engine = JsonPersistenceEngine(storage_path)

            manifest = engine.load_manifest()
            if manifest is None:
                raise ValueError(f"No valid manifest found in {storage_path}")

            units = engine.load_units()
            spaces = engine.load_spaces()
            edges = engine.load_graph()
            sessions = engine.load_sessions()
            insertion_order, processed_session_ids, processed_similarity_pairs, last_async_reasoning, async_check_scheduled = engine.load_processed_state()

            self._reset_state()

            store = self._semantic_map.get_store()
            for unit in units:
                store.upsert_units([unit])

            for space in spaces:
                store.upsert_spaces([space])

            for source, target, rel_type, properties in edges:
                try:
                    self._graph_store.upsert_relationship(
                        Uid(str(source)), Uid(str(target)), str(rel_type), dict(properties)
                    )
                except Exception as e:
                    logger.debug(f"Failed to restore edge {source}->{target}: {e}")

            self._session_manager.reset()
            for session in sessions:
                self._session_manager._sessions.append(session)

            self._insertion_order = insertion_order
            self._processed_session_ids = processed_session_ids
            self._processed_similarity_pairs = processed_similarity_pairs
            self._last_async_reasoning = last_async_reasoning
            self._async_check_scheduled = async_check_scheduled

            self._rebuild_vector_index(units)

            self._dirty = False

            stats = {
                "unit_count": len(units),
                "space_count": len(spaces),
                "edge_count": len(edges),
                "session_count": len(sessions),
            }

            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()
            logger.info(f"Load completed in {duration:.2f}s: {stats}")

            return LoadResult(
                success=True,
                loaded_from=storage_path,
                stats=stats,
                errors=errors,
            )

        except Exception as e:
            errors.append(str(e))
            logger.error(f"Load failed: {e}")
            return LoadResult(
                success=False,
                loaded_from=storage_path,
                stats={},
                errors=errors,
            )

    def load(
        self,
        storage_path: str,
        *,
        embedder=None,
        reranker=None,
        llm_provider=None,
        config_cls=None,
    ) -> Any:
        """Reconstruct a MemorySystem from a saved state on disk.

        Loads the config snapshot, instantiates a new MemorySystem, then
        restores all units, spaces, edges, sessions, and processed state.

        Args:
            storage_path: Directory path containing the saved JSON state.
            embedder: Optional EmbeddingProvider override for the new system.
            reranker: Optional Reranker override for the new system.
            llm_provider: Optional LLMProvider override for the new system.
            config_cls: Factory callable (e.g. MemorySystem) to create the instance.

        Returns:
            The reconstructed MemorySystem with all state restored.
        """
        from ...infrastructure.json_persistence import JsonPersistenceEngine

        engine = JsonPersistenceEngine(storage_path)
        config_data = engine.load_config()

        config = type(self._cfg)()
        root = None

        if config_data is not None:
            cfg_dict = config_data.get("memory_system_config", {})
            if cfg_dict:
                try:
                    config = type(self._cfg)(**cfg_dict)
                except TypeError:
                    logger.warning("Saved config is incompatible, using defaults")
            root = config_data.get("root", None)

        system = config_cls(
            config=config,
            embedder=embedder,
            reranker=reranker,
            llm_provider=llm_provider,
        )

        if root is not None:
            system._root = root
        self._load(storage_path)
        return system

    def _extract_graph_edges(self) -> List[Tuple[str, str, str, Dict[str, Any]]]:
        edges: List[Tuple[str, str, str, Dict[str, Any]]] = []
        try:
            for source, target, rel_type, properties in self._graph_store.get_all_edges():
                edges.append((str(source), str(target), str(rel_type), properties))
        except Exception as e:
            logger.warning(f"Failed to extract graph edges: {e}")
        return edges

    def _get_pending_session_ids(self) -> Set[str]:
        pending: Set[str] = set()
        for s in self._session_manager.get_sessions():
            if s.session_id not in self._processed_session_ids:
                pending.add(s.session_id)
        return pending

    def _reset_state(self) -> None:
        self._semantic_map.get_store().clear()
        self._graph_store.clear()

        self._abi.rebuild([])

        self._semantic_map._dirty_units.clear()
        self._semantic_map._dirty_spaces.clear()
        self._semantic_map._deleted_units.clear()

        if self._pending_lock is not None:
            with self._pending_lock:
                self._pending_units.clear()
                self._pending_events.clear()
                self._pending_entities.clear()
                self._all_events.clear()
                self._all_entities.clear()

        self._processed_session_ids.clear()
        self._processed_similarity_pairs.clear()
        self._insertion_order.clear()

        self._session_manager.reset()

        self._last_async_reasoning = ""
        self._async_check_scheduled = False

        self._dirty = False

    def _rebuild_vector_index(self, units: List[MemoryUnit]) -> None:
        items: List[Tuple[Uid, np.ndarray]] = []
        space_items: Dict[SpaceName, List[Tuple[Uid, np.ndarray]]] = {}

        for u in units:
            if u.embedding is not None:
                uid = Uid(str(u.uid))
                emb = np.asarray(u.embedding, dtype=np.float32).reshape(-1)
                if emb.shape[0] != self._abi.dim():
                    continue
                items.append((uid, emb))

                for space_name in u.metadata.get("spaces", []):
                    sn = SpaceName(str(space_name))
                    space_items.setdefault(sn, []).append((uid, emb))

        self._abi.rebuild(items)
