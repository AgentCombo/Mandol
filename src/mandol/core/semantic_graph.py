import logging
from datetime import datetime
import os
import pickle
import uuid
import orjson
from typing import TYPE_CHECKING, Dict, Any, Optional, List, Set, Tuple, Union
from collections import Counter
import numpy as np
import rustworkx as rx

# from ..retrieval.graph_context_expander import GraphContextExpander


# from .retrieval_interface import RetrievalMethod

if TYPE_CHECKING:
    from ..retrieval.advance_retriever import MultiRetriever
    from ..retrieval.retrieval_interface import RetrievalMethod
    from ..storage.rocksdb_payload_store import RocksDBPayloadStore

from .semantic_map import SemanticMap
from .memory_unit import MemoryUnit
from .memory_space import MemorySpace
from ..utils.logging_config import create_module_logger

logger = create_module_logger("semantic_graph")


class SemanticGraph:
    """Unified graph layer for Mandol memory.

    SemanticGraph combines a SemanticMap-backed retrieval layer with a rustworkx
    directed multigraph. It stores memory units, memory spaces, and explicit
    relationships while preserving stable public UID to rustworkx node-index
    mappings for persistence and graph traversal.
    """

    def __init__(self, semantic_map_instance: Optional[SemanticMap] = None):
        """Initialize the graph layer.

        Args:
            semantic_map_instance: Optional SemanticMap to attach. When omitted,
                a new SemanticMap is created with its default configuration.
        """
        self.semantic_map: SemanticMap = (
            semantic_map_instance if semantic_map_instance else SemanticMap()
        )
        self.semantic_map._parent_semantic_graph = self

        self.rx_graph: rx.PyDiGraph = rx.PyDiGraph(multigraph=True)

        # rustworkx uses integer node indices internally; Mandol exposes stable
        # public UIDs and persists this bidirectional mapping with the graph.
        self._uid_to_index: Dict[str, int] = {}  
        self._index_to_uid: Dict[int, str] = {}  

        logger.info("SemanticGraph initialized with the rustworkx backend.")

        # self._graph_expander: Optional[GraphContextExpander] = None

        
        self._payload_store: Optional["RocksDBPayloadStore"] = None
        self._tiered_storage_config: Optional[Dict[str, Any]] = None
        self._closed = False
        self._modified_relationships = (
            set()
        )
        self._deleted_relationships = set()
        self._modified_units = set()
        self._deleted_units = set()

        self._max_nodes_in_memory = 100000
        self._nodes_access_counts = {}
        self._nodes_last_accessed = {}
        self._nodes_dirty_flag = set()

        self._max_relationships_in_memory = 100000
        self._relationship_cache = (
            {}
        )  # Hot relationship properties mirrored from the rustworkx edge layer.
        self._relationships_access_counts = {}
        self._relationships_last_accessed = {}

        self._multi_retriever: Optional[MultiRetriever] = None
        self.tiered_storage_manager = None
        
        
        self._index_loading_root: Optional[str] = None
        self._high_level_memory_build_state: Dict[str, Any] = {}

    def set_high_level_memory_build_state(self, state: Optional[Dict[str, Any]]) -> None:
        """Attach resumable high-level-memory build metadata to this graph and its map."""
        self._high_level_memory_build_state = dict(state or {})
        if hasattr(self.semantic_map, "set_high_level_memory_build_state"):
            self.semantic_map.set_high_level_memory_build_state(self._high_level_memory_build_state)

    def get_high_level_memory_build_state(self) -> Dict[str, Any]:
        """Return persisted high-level-memory build metadata."""
        state = getattr(self, "_high_level_memory_build_state", {}) or {}
        if not state and hasattr(self.semantic_map, "get_high_level_memory_build_state"):
            state = self.semantic_map.get_high_level_memory_build_state()
        return dict(state)

    def update_high_level_memory_build_state(self, **updates: Any) -> Dict[str, Any]:
        """Update and return high-level-memory build metadata."""
        state = self.get_high_level_memory_build_state()
        state.update(updates)
        self.set_high_level_memory_build_state(state)
        return state

    
    
    
    @property
    def nx_graph(self):
        """Return the compatibility graph handle."""
        logger.warning("nx_graph is deprecated; use rx_graph instead.")
        return self.rx_graph
    
    
    
    
    def has_node(self, uid: str) -> bool:
        """Return whether node is available."""
        return uid in self._uid_to_index
    
    def has_edge(self, source_uid: str, target_uid: str) -> bool:
        """Return whether edge is available."""
        if source_uid not in self._uid_to_index or target_uid not in self._uid_to_index:
            return False
        src_idx = self._uid_to_index[source_uid]
        tgt_idx = self._uid_to_index[target_uid]
        try:
            return len(self.rx_graph.get_all_edge_data(src_idx, tgt_idx)) > 0
        except Exception:  # rustworkx.NoEdgeBetweenNodes
            return False
    
    def get_node_data(self, uid: str) -> Optional[Dict[str, Any]]:
        """Return node data."""
        if uid not in self._uid_to_index:
            return None
        idx = self._uid_to_index[uid]
        return self.rx_graph[idx]
    
    def get_edge_data(self, source_uid: str, target_uid: str) -> List[Dict[str, Any]]:
        """Return edge data."""
        if source_uid not in self._uid_to_index or target_uid not in self._uid_to_index:
            return []
        src_idx = self._uid_to_index[source_uid]
        tgt_idx = self._uid_to_index[target_uid]
        try:
            return self.rx_graph.get_all_edge_data(src_idx, tgt_idx)
        except Exception:  # rustworkx.NoEdgeBetweenNodes
            return []
    
    def get_successors(self, uid: str) -> List[str]:
        """Return successors."""
        if uid not in self._uid_to_index:
            return []
        idx = self._uid_to_index[uid]
        result = []
        for _, tgt_idx, _ in self.rx_graph.out_edges(idx):
            tgt_uid = self._index_to_uid.get(tgt_idx, "")
            if tgt_uid and tgt_uid not in result:
                result.append(tgt_uid)
        return result
    
    def get_predecessors(self, uid: str) -> List[str]:
        """Return predecessors."""
        if uid not in self._uid_to_index:
            return []
        idx = self._uid_to_index[uid]
        result = []
        for src_idx, _, _ in self.rx_graph.in_edges(idx):
            src_uid = self._index_to_uid.get(src_idx, "")
            if src_uid and src_uid not in result:
                result.append(src_uid)
        return result

    
    def connect_to_l2(
        self,
        l2_base_path: Optional[str] = None,
        max_capacity: Optional[int] = None,
        high_watermark: float = 0.85,
        low_watermark: float = 0.70,
    ) -> bool:
        """Enable RocksDB-backed automatic payload paging.

        Args:
            l2_base_path: Directory where RocksDB should be opened.
                When omitted, a timestamped directory under ``./l2_database`` is
                created.
            max_capacity: Optional resident payload-cache capacity.
            high_watermark: Resident-cache eviction trigger.
            low_watermark: Resident-cache target after eviction.

        Returns:
            True when RocksDB and automatic paging are ready; False on
            initialization failure. A failed initialization leaves the graph
            usable with resident in-memory payloads.
        """
        self._ensure_open()
        if l2_base_path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            short_id = uuid.uuid4().hex[:8]
            l2_base_path = os.path.join(
                ".", "l2_database", f"graph_{ts}_{short_id}"
            )

        resolved_base_path = os.path.realpath(os.path.abspath(l2_base_path))
        rocksdb_path = os.path.join(resolved_base_path, "payloads.rocksdb")
        effective_capacity = int(
            max_capacity if max_capacity is not None else self._max_nodes_in_memory
        )
        requested_config = {
            "rocksdb_path": rocksdb_path,
            "max_capacity": effective_capacity,
            "high_watermark": high_watermark,
            "low_watermark": low_watermark,
        }

        if self.tiered_storage_manager is not None:
            if self._tiered_storage_config == requested_config:
                return True
            raise RuntimeError(
                "RocksDB tiered storage is already connected with a different "
                "path or capacity/watermark configuration."
            )

        payload_store = None
        try:
            os.makedirs(resolved_base_path, exist_ok=True)

            from ..storage.rocksdb_payload_store import RocksDBPayloadStore

            payload_store = RocksDBPayloadStore(db_path=rocksdb_path)
            if not payload_store.is_connected:
                logger.error("RocksDB payload-store connection failed.")
                payload_store.close()
                return False

            self.enable_tiered_storage(
                payload_store,
                max_capacity=effective_capacity,
                high_watermark=high_watermark,
                low_watermark=low_watermark,
            )
            self._tiered_storage_config = requested_config
            logger.info(
                "RocksDB-backed tiered payload storage connected: %s",
                rocksdb_path,
            )
            return True
        except Exception as e:
            logger.error(f"Failed to initialize L2 storage: {e}")
            self._reset_tiered_storage_after_failure(payload_store)
            return False

    def _reset_tiered_storage_after_failure(self, payload_store=None) -> None:
        """Release a partially initialized store without changing payloads."""
        manager = getattr(self.semantic_map, "tiered_storage_manager", None)
        if manager is not None:
            manager.shutdown(wait=True)
        map_store = getattr(self.semantic_map, "_external_storage", None)
        stores = {store for store in (payload_store, self._payload_store, map_store) if store}
        for store in stores:
            store.close()
        self.semantic_map.tiered_storage_manager = None
        self.semantic_map._external_storage = None
        self.semantic_map._storage_uids = None
        self._payload_store = None
        self.tiered_storage_manager = None
        self._tiered_storage_config = None

    def enable_tiered_storage(
        self,
        payload_store,
        max_capacity: Optional[int] = None,
        high_watermark: float = 0.85,
        low_watermark: float = 0.70,
    ):
        """Attach graph-aware RocksDB payload callbacks to SemanticMap."""
        self._ensure_open()
        if self.tiered_storage_manager is not None:
            raise RuntimeError("RocksDB tiered storage is already enabled.")
        if max_capacity is not None:
            self._max_nodes_in_memory = int(max_capacity)
        self.tiered_storage_manager = self.semantic_map.enable_tiered_storage(
            payload_store=payload_store,
            max_capacity=self._max_nodes_in_memory,
            high_watermark=high_watermark,
            low_watermark=low_watermark,
            callbacks={
                "get_l1_data_cb": self._get_l1_data_for_tiered_swap_out,
                "remove_from_l1_cb": self._remove_from_l1_for_tiered_swap,
                "add_to_l1_cb": self._add_to_l1_from_tiered_swap,
            },
        )
        self._payload_store = payload_store
        return self.tiered_storage_manager

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("SemanticGraph is closed.")

    def close(self) -> None:
        """Finish background paging and close the RocksDB payload store.

        Closing ends this graph's lifecycle and does not materialize cold
        payloads into resident memory. Repeated calls are safe.
        """
        if self._closed:
            return
        try:
            self.semantic_map._close_tiered_storage()
        finally:
            self._payload_store = None
            self.tiered_storage_manager = None
            self._tiered_storage_config = None
            self._closed = True

    def _trigger_tiered_eviction_if_needed(self) -> None:
        manager = getattr(self, "tiered_storage_manager", None)
        if manager is not None:
            manager.check_and_trigger_eviction(len(self.semantic_map.memory_units))

    def _get_l1_data_for_tiered_swap_out(self, count: int) -> Dict[str, Any]:
        return self.semantic_map._get_l1_data_for_tiered_swap_out(count)

    def _remove_from_l1_for_tiered_swap(self, uids: List[str]) -> int:
        removed_count = self.semantic_map._remove_from_l1_for_tiered_swap(uids)
        for uid in uids:
            self._nodes_access_counts.pop(uid, None)
            self._nodes_last_accessed.pop(uid, None)
            self._nodes_dirty_flag.discard(uid)
            self._modified_units.discard(uid)
            if uid in self._uid_to_index:
                idx = self._uid_to_index[uid]
                node_attrs = dict(self.rx_graph[idx] or {})
                node_attrs["uid"] = uid
                node_attrs["ghost"] = True
                node_attrs["swapped_out_at"] = str(datetime.now())
                self.rx_graph[idx] = node_attrs
        return removed_count

    def _add_to_l1_from_tiered_swap(self, recovered_units) -> int:
        restored_count = self.semantic_map._add_to_l1_from_tiered_swap(recovered_units)
        now = datetime.now()
        timestamp = now.timestamp()
        for unit in recovered_units:
            node_attrs = {
                "uid": unit.uid,
                "ghost": False,
                "updated": str(now),
                **{
                    key: value
                    for key, value in unit.raw_data.items()
                    if isinstance(value, (str, int, float, bool))
                },
            }
            if unit.uid not in self._uid_to_index:
                idx = self.rx_graph.add_node(node_attrs)
                self._uid_to_index[unit.uid] = idx
                self._index_to_uid[idx] = unit.uid
            else:
                idx = self._uid_to_index[unit.uid]
                existing = dict(self.rx_graph[idx] or {})
                existing.update(node_attrs)
                self.rx_graph[idx] = existing
            self._nodes_access_counts[unit.uid] = self._nodes_access_counts.get(unit.uid, 0) + 1
            self._nodes_last_accessed[unit.uid] = timestamp
        return restored_count

    def add_unit(
        self,
        unit: MemoryUnit,
        explicit_content_for_embedding: Optional[Any] = None,
        content_type_for_embedding: Optional[str] = None,
        space_names: Optional[List[str]] = None,
        index_update_mode: str = "incremental",
        generate_sparse_embedding: bool = True,  
        **legacy_kwargs,
    ):
        """Add a memory unit to the graph and the underlying SemanticMap.

        Args:
            unit: MemoryUnit to insert or update.
            explicit_content_for_embedding: Optional content used to generate
                the embedding. If omitted, SemanticMap infers content from the
                unit payload.
            content_type_for_embedding: Content type used by the embedding
                path, such as ``"text"`` or ``"image_path"``.
            space_names: Logical memory spaces that should contain this unit.
            index_update_mode: SemanticMap index update strategy, such as
                ``"incremental"``, ``"none"``, or ``"rebuild"``.
            generate_sparse_embedding: Whether SemanticMap should generate a
                SPLADE sparse embedding during insertion.
            **legacy_kwargs: Backward-compatible aliases accepted by older code.
        """
        self._ensure_open()
        if len(self.semantic_map.memory_units) >= self._max_nodes_in_memory:
            self._trigger_tiered_eviction_if_needed()

        if "rebuild_semantic_map_index_immediately" in legacy_kwargs:
            legacy_value = bool(legacy_kwargs.pop("rebuild_semantic_map_index_immediately"))
            index_update_mode = "incremental" if legacy_value else "none"
            logger.warning(
                "rebuild_semantic_map_index_immediately is deprecated; use index_update_mode instead."
            )
        if legacy_kwargs:
            raise TypeError(
                f"add_unit() got unexpected keyword arguments: {list(legacy_kwargs.keys())}"
            )

        self.semantic_map.add_unit(
            unit,
            explicit_content_for_embedding,
            content_type_for_embedding,
            space_names=space_names,
            index_update_mode=index_update_mode,
            generate_sparse_embedding=generate_sparse_embedding,  
        )

        node_attrs = {
            "uid": unit.uid,
            "spaces": space_names if space_names else [],
            **{
                k: v
                for k, v in unit.raw_data.items()
                if isinstance(v, (str, int, float, bool))
            },
        }
        
        if unit.uid not in self._uid_to_index:
            node_attrs["created"] = str(datetime.now())
            idx = self.rx_graph.add_node(node_attrs)
            self._uid_to_index[unit.uid] = idx
            self._index_to_uid[idx] = unit.uid
            logger.debug(f"Node '{unit.uid}' added to the rustworkx graph (index={idx}, spaces={space_names}).")
        else:
            
            node_attrs["updated"] = str(datetime.now())
            idx = self._uid_to_index[unit.uid]
            self.rx_graph[idx] = node_attrs
            logger.debug(f"Node '{unit.uid}' attributes updated (index={idx}, spaces={space_names}).")

        self._nodes_access_counts[unit.uid] = (
            self._nodes_access_counts.get(unit.uid, 0) + 1
        )
        self._nodes_last_accessed[unit.uid] = datetime.now().timestamp()
        self._nodes_dirty_flag.add(unit.uid)
        self._modified_units.add(unit.uid)

        

    def batch_add_units(
        self,
        units: List[MemoryUnit],
        batch_size: int = 32,
        space_names: Optional[List[str]] = None,
        content_type_for_embedding: Optional[str] = None,
        explicit_contents_for_embedding: Optional[List[Any]] = None,
        content_types_for_embedding: Optional[List[Optional[str]]] = None,
        per_unit_space_names: Optional[List[Optional[List[str]]]] = None,
        index_update_mode: str = "incremental",
        generate_sparse_embedding: bool = True,
        sparse_model_name: str = "naver/splade-v3",
        show_progress: bool = True,
        **legacy_kwargs,
    ) -> Dict[str, Any]:
        """Add memory units in batches and synchronize graph nodes.

        Args:
            units: Memory units to insert or update.
            batch_size: Batch size used by SemanticMap embedding generation.
            space_names: Shared memory spaces applied to every unit.
            content_type_for_embedding: Shared content type for embedding
                generation.
            explicit_contents_for_embedding: Optional per-unit embedding
                contents aligned with ``units``.
            content_types_for_embedding: Optional per-unit content types aligned
                with ``units``.
            per_unit_space_names: Optional per-unit memory-space lists aligned
                with ``units``.
            index_update_mode: SemanticMap index update strategy.
            generate_sparse_embedding: Whether to generate SPLADE sparse
                embeddings for inserted units.
            sparse_model_name: SPLADE model name passed to SemanticMap.
            show_progress: Whether embedding backends may display progress.
            **legacy_kwargs: Backward-compatible aliases accepted by older code.

        Returns:
            A statistics dictionary covering SemanticMap insertion, graph node
            updates, and elapsed time.
        """
        self._ensure_open()
        import time
        
        if not units:
            logger.warning("batch_add_units: no units to add")
            return {
                "total": 0, "added": 0, "skipped": 0,
                "embedding_generated": 0, "sparse_generated": 0,
                "graph_nodes_added": 0, "graph_nodes_updated": 0,
                "duration": 0.0
            }
        
        logger.info(f" SemanticGraph starting batch insertion for {len(units)} units...")
        start_time = time.time()

        if "rebuild_index" in legacy_kwargs:
            legacy_value = bool(legacy_kwargs.pop("rebuild_index"))
            index_update_mode = "rebuild" if legacy_value else "none"
            logger.warning("rebuild_index is deprecated; use index_update_mode instead.")
        if legacy_kwargs:
            raise TypeError(
                f"batch_add_units() got unexpected keyword arguments: {list(legacy_kwargs.keys())}"
            )

        unit_count = len(units)
        if explicit_contents_for_embedding is not None and len(explicit_contents_for_embedding) != unit_count:
            raise ValueError("explicit_contents_for_embedding length must match units")
        if content_types_for_embedding is not None and len(content_types_for_embedding) != unit_count:
            raise ValueError("content_types_for_embedding length must match units")
        if per_unit_space_names is not None and len(per_unit_space_names) != unit_count:
            raise ValueError("per_unit_space_names length must match units")

        def spaces_for_index(index: int) -> List[str]:
            if per_unit_space_names is not None:
                item_spaces = per_unit_space_names[index]
                return list(item_spaces) if item_spaces else []
            return list(space_names) if space_names else []
        
        if len(self.semantic_map.memory_units) + len(units) >= self._max_nodes_in_memory:
            self._trigger_tiered_eviction_if_needed()
        
        map_stats = self.semantic_map.batch_add_units(
            units=units,
            batch_size=batch_size,
            space_names=space_names,
            content_type_for_embedding=content_type_for_embedding,
            explicit_contents_for_embedding=explicit_contents_for_embedding,
            content_types_for_embedding=content_types_for_embedding,
            per_unit_space_names=per_unit_space_names,
            index_update_mode=index_update_mode,
            generate_sparse_embedding=generate_sparse_embedding,
            sparse_model_name=sparse_model_name,
            show_progress=show_progress
        )
        
        logger.info(" Synchronizing units into the rustworkx graph...")
        graph_nodes_added = 0
        graph_nodes_updated = 0
        current_time = datetime.now()
        current_timestamp = current_time.timestamp()
        
        for index, unit in enumerate(units):
            if not isinstance(unit, MemoryUnit):
                continue
            
            if not self.semantic_map._unit_exists(unit.uid):
                continue
            
            node_attrs = {
                "uid": unit.uid,
                "spaces": spaces_for_index(index),
                **{
                    k: v
                    for k, v in unit.raw_data.items()
                    if isinstance(v, (str, int, float, bool))
                },
            }
            
            if unit.uid not in self._uid_to_index:
                node_attrs["created"] = str(current_time)
                idx = self.rx_graph.add_node(node_attrs)
                self._uid_to_index[unit.uid] = idx
                self._index_to_uid[idx] = unit.uid
                graph_nodes_added += 1
            else:
                node_attrs["updated"] = str(current_time)
                idx = self._uid_to_index[unit.uid]
                self.rx_graph[idx] = node_attrs
                graph_nodes_updated += 1
            
            self._nodes_access_counts[unit.uid] = (
                self._nodes_access_counts.get(unit.uid, 0) + 1
            )
            self._nodes_last_accessed[unit.uid] = current_timestamp
            self._nodes_dirty_flag.add(unit.uid)
            self._modified_units.add(unit.uid)
        
        stats = {
            **map_stats,
            "graph_nodes_added": graph_nodes_added,
            "graph_nodes_updated": graph_nodes_updated,
            "duration": time.time() - start_time
        }
        
        logger.info(
            f" SemanticGraph batch insert complete: "
            f"added {stats['added']}, graph nodes (created {graph_nodes_added}, updated {graph_nodes_updated}), "
            f"elapsed {stats['duration']:.2f}s"
        )
        
        return stats

    def add_relationship(
        self,
        source_uid: str,
        target_uid: str,
        relationship_name: str,
        bidirectional: bool = False,
        **kwargs: Any,
    ):
        """Add relationship."""

        def get_node_id(uid):
            if self.semantic_map._unit_exists(uid):
                return uid, "memory_unit"
            elif uid in self.semantic_map.memory_spaces:
                return f"ms:{uid}", "memory_space"
            elif isinstance(uid, str) and uid.startswith("ms:"):
                ms_name = uid[3:]
                if ms_name in self.semantic_map.memory_spaces:
                    return uid, "memory_space"
            return uid, None

        src_id, src_type = get_node_id(source_uid)
        tgt_id, tgt_type = get_node_id(target_uid)

        src_exists = (
            src_type == "memory_unit" and self.semantic_map._unit_exists(src_id)
        ) or (
            src_type == "memory_space" and src_id[3:] in self.semantic_map.memory_spaces
        )
        tgt_exists = (
            tgt_type == "memory_unit" and self.semantic_map._unit_exists(tgt_id)
        ) or (
            tgt_type == "memory_space" and tgt_id[3:] in self.semantic_map.memory_spaces
        )
        if not src_exists:
            logger.error(f"Source node '{source_uid}' does not exist; cannot add relationship.")
            return False
        if not tgt_exists:
            logger.error(f"Target node '{target_uid}' does not exist; cannot add relationship.")
            return False

        if src_id not in self._uid_to_index:
            if src_type == "memory_unit":
                node_attrs = {
                    "uid": src_id,
                    "type": "memory_unit",
                    "created": str(datetime.now()),
                }
                idx = self.rx_graph.add_node(node_attrs)
                self._uid_to_index[src_id] = idx
                self._index_to_uid[idx] = src_id
            elif src_type == "memory_space":
                ms = self.semantic_map.memory_spaces.get(src_id[3:])
                if ms is not None:
                    node_attrs = {
                        "name": ms.name,
                        "type": "memory_space",
                        "created": str(datetime.now()),
                    }
                    idx = self.rx_graph.add_node(node_attrs)
                    self._uid_to_index[src_id] = idx
                    self._index_to_uid[idx] = src_id
                    
        if tgt_id not in self._uid_to_index:
            if tgt_type == "memory_unit":
                node_attrs = {
                    "uid": tgt_id,
                    "type": "memory_unit",
                    "created": str(datetime.now()),
                }
                idx = self.rx_graph.add_node(node_attrs)
                self._uid_to_index[tgt_id] = idx
                self._index_to_uid[idx] = tgt_id
            elif tgt_type == "memory_space":
                ms = self.semantic_map.memory_spaces.get(tgt_id[3:])
                if ms is not None:
                    node_attrs = {
                        "name": ms.name,
                        "type": "memory_space",
                        "created": str(datetime.now()),
                    }
                    idx = self.rx_graph.add_node(node_attrs)
                    self._uid_to_index[tgt_id] = idx
                    self._index_to_uid[idx] = tgt_id

        def filter_gml_attrs(attrs):
            return {
                k: (str(v) if not isinstance(v, (str, int, float, bool)) else v)
                for k, v in attrs.items()
                if v is not None
            }

        edge_attributes = {
            "type": relationship_name,
            "created": str(datetime.now()),
            **filter_gml_attrs(kwargs),
        }
        
        
        src_idx = self._uid_to_index[src_id]
        tgt_idx = self._uid_to_index[tgt_id]
        self.rx_graph.add_edge(src_idx, tgt_idx, edge_attributes)

        # Keep the external relationship cache in sync with the graph edge.
        self.swap_in_relationship(src_id, tgt_id, relationship_name, kwargs)

        self._modified_relationships.add((src_id, tgt_id, relationship_name))

        # self.rel_types.add(relationship_name)

        logger.info(
            f"Added relationship '{relationship_name}' from '{src_id}' to '{tgt_id}'."
        )

        if bidirectional:
            self.rx_graph.add_edge(tgt_idx, src_idx, edge_attributes)
            self.swap_in_relationship(tgt_id, src_id, relationship_name, kwargs)
            self._modified_relationships.add((tgt_id, src_id, relationship_name))
            logger.info(
                f"Added bidirectional relationship '{relationship_name}' from '{tgt_id}' to '{src_id}'."
            )

        return True
    
    def get_relationship(
        self, source_uid: str, target_uid: str, relationship_type: str
        ) -> Optional[Dict[str, Any]]:
        """Return relationship properties from the cache or resident graph."""
        try:
            # The cache stores public UIDs rather than rustworkx node indices.
            rel_key = (source_uid, target_uid, relationship_type)
            if rel_key in self._relationship_cache:
                self._relationships_access_counts[rel_key] = (
                    self._relationships_access_counts.get(rel_key, 0) + 1
                )
                self._relationships_last_accessed[rel_key] = datetime.now().timestamp()
                return self._relationship_cache[rel_key]

            if source_uid in self._uid_to_index and target_uid in self._uid_to_index:
                src_idx = self._uid_to_index[source_uid]
                tgt_idx = self._uid_to_index[target_uid]
                
                try:
                    edge_data_list = self.rx_graph.get_all_edge_data(src_idx, tgt_idx)
                except Exception:  # rustworkx.NoEdgeBetweenNodes
                    edge_data_list = []
                
                for edge_data in edge_data_list:
                    if edge_data and edge_data.get("type") == relationship_type:
                        relationship_properties = {
                            k: v
                            for k, v in edge_data.items()
                            if k not in ["type", "created", "updated"]
                        }

                        # Mirror the graph edge in the relationship cache.
                        self.swap_in_relationship(
                            source_uid,
                            target_uid,
                            relationship_type,
                            relationship_properties,
                        )

                        return relationship_properties

            return None

        except Exception as e:
            logger.error(
                f"Relationship lookup failed ({source_uid} -[{relationship_type}]-> {target_uid}): {e}"
            )
            return None

    def delete_unit(
        self, uid: str, rebuild_semantic_map_index_immediately: bool = False
    ):
        """Remove unit."""
        self.semantic_map.delete_unit(
            uid, rebuild_index_immediately=rebuild_semantic_map_index_immediately
        )

        if uid in self._uid_to_index:
            idx = self._uid_to_index[uid]
            
            for _, tgt_idx, edge_data in self.rx_graph.out_edges(idx):
                if edge_data:
                    rel_type = edge_data.get("type", "RELATED_TO")
                    tgt_uid = self._index_to_uid.get(tgt_idx, "")
                    if tgt_uid:
                        self._deleted_relationships.add((uid, tgt_uid, rel_type))

            for src_idx, _, edge_data in self.rx_graph.in_edges(idx):
                if edge_data:
                    rel_type = edge_data.get("type", "RELATED_TO")
                    src_uid = self._index_to_uid.get(src_idx, "")
                    if src_uid:
                        self._deleted_relationships.add((src_uid, uid, rel_type))

            self.rx_graph.remove_node(idx)
            del self._uid_to_index[uid]
            del self._index_to_uid[idx]
            logger.info(f"Node '{uid}' (index: {idx}) and its relationships were removed from the rustworkx graph.")
        else:
            logger.warning(f"Attempted to remove missing node '{uid}' from the rustworkx graph.")

        self._deleted_units.add(uid)
        if uid in self._modified_units:
            self._modified_units.remove(uid)
        if uid in self._nodes_access_counts:
            del self._nodes_access_counts[uid]
        if uid in self._nodes_last_accessed:
            del self._nodes_last_accessed[uid]
        if uid in self._nodes_dirty_flag:
            self._nodes_dirty_flag.remove(uid)

    def delete_relationship(
        self, source_uid: str, target_uid: str, relationship_name: Optional[str] = None
    ):
        """Remove relationship."""
        if source_uid not in self._uid_to_index or target_uid not in self._uid_to_index:
            logger.warning(f"Node '{source_uid}' and/or '{target_uid}' is not in the graph.")
            return False
        
        src_idx = self._uid_to_index[source_uid]
        tgt_idx = self._uid_to_index[target_uid]
        
        try:
            edge_data_list = self.rx_graph.get_all_edge_data(src_idx, tgt_idx)
        except Exception:  # rustworkx.NoEdgeBetweenNodes
            edge_data_list = []
        if not edge_data_list:
            logger.warning(f"No direct edge exists between node '{source_uid}' and '{target_uid}'.")
            return False

        if relationship_name:
            edge_found = False
            
            for edge_idx in list(self.rx_graph.edge_indices_from_endpoints(src_idx, tgt_idx)):
                edge_data = self.rx_graph.get_edge_data_by_index(edge_idx)
                if edge_data and edge_data.get("type") == relationship_name:
                    self.rx_graph.remove_edge_from_index(edge_idx)
                    self._deleted_relationships.add(
                        (source_uid, target_uid, relationship_name)
                    )

                    # Remove matching cache metadata for the deleted edge.
                    rel_key = (source_uid, target_uid, relationship_name)
                    if rel_key in self._relationship_cache:
                        del self._relationship_cache[rel_key]
                    if rel_key in self._relationships_access_counts:
                        del self._relationships_access_counts[rel_key]
                    if rel_key in self._relationships_last_accessed:
                        del self._relationships_last_accessed[rel_key]

                    logger.info(
                        f"Deleted relationship '{relationship_name}' from '{source_uid}' to '{target_uid}'."
                    )
                    edge_found = True
                    break
            
            if not edge_found:
                logger.warning(
                    f"Relationship '{relationship_name}' from '{source_uid}' to '{target_uid}' was not found."
                )
                return False
            return True
        else:
            for edge_idx in list(self.rx_graph.edge_indices_from_endpoints(src_idx, tgt_idx)):
                edge_data = self.rx_graph.get_edge_data_by_index(edge_idx)
                if edge_data:
                    rel_type = edge_data.get("type", "RELATED_TO")
                    self._deleted_relationships.add((source_uid, target_uid, rel_type))
                self.rx_graph.remove_edge_from_index(edge_idx)

            logger.info(f"Deleted all direct relationships from '{source_uid}' to '{target_uid}'.")
            return True

    def get_unit(self, uid: str) -> Optional[MemoryUnit]:
        """Return a unit, automatically paging in a cold payload if needed."""
        self._ensure_open()
        unit = self.semantic_map.get_unit(uid)
        if unit:
            self._nodes_access_counts[uid] = self._nodes_access_counts.get(uid, 0) + 1
            self._nodes_last_accessed[uid] = datetime.now().timestamp()
            return unit

        logger.warning(f"Node '{uid}' is missing from both memory and external storage.")
        return None
    
    def get_units_by_spaces(
        self, 
        space_names: List[str], 
        mode: str = "union",
        recursive: bool = True
    ) -> List[MemoryUnit]:
        """Delegate MemorySpace set queries to the underlying SemanticMap.

        Args:
            space_names: MemorySpace names, with or without the ``ms:`` prefix.
            mode: Set operation over the selected spaces.
            recursive: Whether child MemorySpaces contribute units.

        Returns:
            Memory units selected by the requested space operation.
        """
        return self.semantic_map.get_units_by_spaces(space_names, mode, recursive)

    def get_space_statistics(self) -> Dict[str, Any]:
        """Return MemorySpace membership and index statistics from SemanticMap."""
        return self.semantic_map.get_space_statistics()
    

    
    
    

    def create_memory_space_in_map(self, space_name: str) -> MemorySpace:
        """Create or return a MemorySpace in the underlying SemanticMap."""
        return self.semantic_map.create_memory_space(space_name)

    def add_unit_to_space_in_map(self, unit_or_uid, space_name: str):
        """Attach a unit to a SemanticMap MemorySpace."""
        self.semantic_map.add_unit_to_space(unit_or_uid, space_name)

    def remove_unit_from_space_in_map(self, unit_or_uid, space_name: str):
        """Remove a unit's membership from a SemanticMap MemorySpace."""
        self.semantic_map.remove_unit_from_space(unit_or_uid, space_name)

    def get_all_units(self) -> List[MemoryUnit]:
        """Return all L1 units currently available in SemanticMap."""
        return self.semantic_map.get_all_units()

    def cluster_nodes(self, method: str = "leiden", **kwargs) -> Dict[int, List[str]]:
        """Cluster graph nodes with the configured community method."""
        from ..cluster import cluster_nodes

        return cluster_nodes(self, method=method, **kwargs)

    def get_units_in_memory_space(
        self, ms_names: Union[str, List[str]], recursive: bool = True
    ) -> List[MemoryUnit]:
        """Return units in memory space."""
        if isinstance(ms_names, str):
            ms_names = [ms_names]

        return self.semantic_map.get_units_in_memory_space(ms_names, recursive=recursive)

    def get_multi_retriever(self) -> 'MultiRetriever':
        """Return multi retriever."""
        if self._multi_retriever is not None:
            return self._multi_retriever

        from ..retrieval.advance_retriever import MultiRetriever
        from ..retrieval.retrieval_interface import RetrievalMethod

        self._multi_retriever = MultiRetriever(self)

        retrieval_indices_dir = getattr(self, '_index_loading_root', None)
        
        if not retrieval_indices_dir and hasattr(self, 'storage_path') and self.storage_path:
             retrieval_indices_dir = os.path.join(self.storage_path, "retrieval_indices")

        if retrieval_indices_dir and os.path.exists(retrieval_indices_dir):
            logger.info(f" Restoring retriever indexes from disk: {retrieval_indices_dir}")
            
            
            bm25_path = os.path.join(retrieval_indices_dir, "bm25")
            if os.path.exists(bm25_path):
                try:
                    
                    self._multi_retriever._ensure_retriever_loaded(RetrievalMethod.BM25)
                    
                    bm25_retriever = self._multi_retriever.retrievers.get(RetrievalMethod.BM25)
                    
                    
                    if bm25_retriever and not getattr(bm25_retriever, '_index_built', False):
                        if bm25_retriever.load_index(bm25_path):
                            logger.info(" BM25 index loaded successfully; skipping prebuild.")
                        else:
                            logger.warning(" BM25 disk index load failed; it will be rebuilt later.")
                except Exception as e:
                    logger.warning(f"Error while loading BM25 index: {e}")

            
            splade_path = os.path.join(retrieval_indices_dir, "splade")
            if os.path.exists(splade_path):
                try:
                    
                    self._multi_retriever._ensure_retriever_loaded(RetrievalMethod.SPLADE)
                    
                    splade_retriever = self._multi_retriever.retrievers.get(RetrievalMethod.SPLADE)
                    
                    
                    if splade_retriever and not getattr(splade_retriever, '_index_built', False):
                        if splade_retriever.load_index(splade_path):
                            logger.info(" SPLADE index loaded successfully; skipping prebuild.")
                        else:
                            logger.warning(" SPLADE disk index load failed; it will be rebuilt later.")
                except Exception as e:
                    logger.warning(f"Error while loading SPLADE index: {e}")
            
            
            self._index_loading_root = None
        else:
            logger.debug("No persisted retriever index directory found; indexes will be built on first retrieval.")

        return self._multi_retriever

    def build_freeze_indexes(self) -> Dict[str, bool]:
        """Build freeze indexes."""
        logger.info("SemanticGraph is building static acceleration indexes globally...")
        multi_retriever = self.get_multi_retriever()
        if multi_retriever is None:
            logger.warning("MultiRetriever is not initialized; cannot build static acceleration indexes.")
            return {}
        return multi_retriever.build_freeze_indexes()
    
    # def get_multi_retriever(self):
    #     if self._multi_retriever is None:
    #         try:
    #             from ..retrieval.advance_retriever import MultiRetriever
    #             self._multi_retriever = MultiRetriever(self)
    #         except Exception as e:
    #             import traceback
    #             return None
    #     return self._multi_retriever

    def search_hybrid_in_graph(self, query: str, **kwargs):
        """Retrieve hybrid in graph."""
        multi_retriever = self.get_multi_retriever()
        if multi_retriever:
            try:
                return multi_retriever.search_hybrid(query, **kwargs)
            except Exception as e:
                logger.warning(f"Hybrid retrieval failed: {e}")
        
        # Dense retrieval remains available when the hybrid retriever fails.
        return self.semantic_map.search_similarity_by_text(query, **kwargs)

    def search_similarity_in_graph(
        self,
        query_text: Optional[str] = None,
        query_embedding: Optional[np.ndarray] = None,
        query_image_path: Optional[str] = None,
        top_k: int = 5,
        ms_names: Optional[List[str]] = None,
        recursive: bool = True,
        return_score: bool = False,
        candidate_units: Optional[List[Any]] = None,  
        ):
        normed = None
        if ms_names:
            normed = [(n[3:] if n.startswith("ms:") else n) for n in ms_names]

        
        candidate_uids = None
        if candidate_units is not None:
            candidate_uids = []
            for u in candidate_units:
                if hasattr(u, "uid"):
                    candidate_uids.append(u.uid)
                else:
                    candidate_uids.append(str(u))

        if query_text is not None:
            results = self.semantic_map.search_similarity_by_text(
                query_text, top_k, normed, candidate_uids
            )
        elif query_embedding is not None:
            results = self.semantic_map.search_similarity_by_vector(
                query_embedding, top_k, normed, candidate_uids
            )
        elif query_image_path is not None:
            results = self.semantic_map.search_similarity_by_image(
                query_image_path, top_k, normed, candidate_uids
            )
        else:
            logger.warning(
                "Provide one of query_text, query_embedding, or query_image_path for similarity search."
            )
            return []

        if return_score:
            return results
        else:
            return [unit for unit, _ in results]

    def search_graph_nodes(self, 
                        query: str, 
                        top_k: int = 10,
                        search_method: str = "hybrid",
                        **kwargs) -> List[Tuple[MemoryUnit, float]]:
        """Search graph nodes through the graph traversal retriever.

        Args:
            query: Text query used by semantic or hybrid graph retrieval.
            top_k: Maximum number of nodes to return.
            search_method: ``hybrid``, ``semantic``, or ``fulltext``.
            **kwargs: Additional backend-specific retrieval options.

        Returns:
            Ranked memory units and scores, or dense-search results when the
            graph retriever is unavailable.
        """
        multi_retriever = self.get_multi_retriever()
        if not multi_retriever:
            logger.warning("MultiRetriever is unavailable.")
            return []
        
        if RetrievalMethod.GRAPH_TRAVERSAL not in multi_retriever.retrievers:
            logger.warning("Graph retriever is unavailable; falling back to semantic search.")
            return self.search_similarity_in_graph(query_text=query, top_k=top_k, **kwargs)
        
        graph_retriever = multi_retriever.retrievers[RetrievalMethod.GRAPH_TRAVERSAL]
        
        if search_method == "hybrid":
            return graph_retriever.hybrid_node_search(query, top_k, **kwargs)
        elif search_method == "semantic":
            return graph_retriever._node_similarity_search(query, top_k, **kwargs)
        elif search_method == "fulltext":
            return graph_retriever._node_fulltext_search(query, top_k, **kwargs)
        else:
            return graph_retriever.hybrid_node_search(query, top_k, **kwargs)

    def search_graph_relations(self,
                            seed_nodes: Optional[List[str]] = None,
                            relation_types: Optional[List[str]] = None,
                            max_depth: int = 2,
                            limit: int = 50) -> List[Tuple[str, str, Dict[str, Any]]]:
        """Retrieve graph edges from selected seed nodes or the full graph.

        Args:
            seed_nodes: Optional source UIDs used as traversal roots.
            relation_types: Optional relation type filter.
            max_depth: Traversal depth when seed nodes are provided.
            limit: Maximum number of edges to return.

        Returns:
            Tuples of ``(source_uid, target_uid, edge_attributes)``.
        """
        multi_retriever = self.get_multi_retriever()
        if not multi_retriever:
            return []
        
        if RetrievalMethod.GRAPH_TRAVERSAL not in multi_retriever.retrievers:
            logger.warning("Graph retriever is unavailable.")
            return []
        
        graph_retriever = multi_retriever.retrievers[RetrievalMethod.GRAPH_TRAVERSAL]
        
        if seed_nodes:
            return graph_retriever.edge_bfs_search(
                origin_node_uids=seed_nodes,
                max_depth=max_depth,
                relation_types=relation_types,
                limit=limit
            )
        else:
            edges = []
            for edge_idx in self.rx_graph.edge_indices():
                src_idx, tgt_idx = self.rx_graph.get_edge_endpoints_by_index(edge_idx)
                data = self.rx_graph.get_edge_data_by_index(edge_idx)
                if relation_types and data.get('type') not in relation_types:
                    continue
                
                src_uid = self._index_to_uid.get(src_idx, "")
                tgt_uid = self._index_to_uid.get(tgt_idx, "")
                if src_uid and tgt_uid:
                    edges.append((src_uid, tgt_uid, data))
                if len(edges) >= limit:
                    break
            return edges

    def get_node_neighbors(self,
                        node_uid: str,
                        max_depth: int = 2,
                        include_semantic: bool = True,
                        include_structural: bool = True,
                        similarity_threshold: float = 0.7) -> Dict[str, List[MemoryUnit]]:
        """Return semantic and structural neighbors for a graph node.

        Semantic neighbors come from similarity search over the node content.
        Structural neighbors come from graph traversal up to ``max_depth``.
        """
        multi_retriever = self.get_multi_retriever()
        if not multi_retriever or RetrievalMethod.GRAPH_TRAVERSAL not in multi_retriever.retrievers:
            return {"semantic": [], "structural": []}
        
        graph_retriever = multi_retriever.retrievers[RetrievalMethod.GRAPH_TRAVERSAL]
        unit = self.get_unit(node_uid)
        
        if not unit:
            return {"semantic": [], "structural": []}
        
        neighbors_dict = graph_retriever.get_relevant_nodes(
            seed_nodes=[unit],
            similarity_threshold=similarity_threshold,
            max_results=20,
            include_semantic=include_semantic,
            include_structural=include_structural
        )
        
        return {
            "all_neighbors": neighbors_dict.get(node_uid, []),
            "semantic": [],
            "structural": []
        }

    
    

    def swap_in_relationship(
        self, source_uid: str, target_uid: str, relationship_type: str, properties: dict
    ):
        """Store relationship properties in the in-memory hot cache."""
        rel_key = (source_uid, target_uid, relationship_type)
        self._relationship_cache[rel_key] = properties
        self._relationships_access_counts[rel_key] = 1
        self._relationships_last_accessed[rel_key] = datetime.now().timestamp()

        # Evict cold relationship entries when the hot cache exceeds its budget.
        if len(self._relationship_cache) > self._max_relationships_in_memory:
            self.swap_out_relationship(int(self._max_relationships_in_memory * 0.2))

    def swap_out_relationship(self, count: int = 100, strategy: str = "LRU"):
        """Evict relationship-cache entries while graph edges remain resident."""
        if not self._relationship_cache:
            logger.debug("Relationship cache is empty; nothing to swap out.")
            return

        if strategy == "LRU":
            sorted_rels = sorted(
                self._relationship_cache.keys(),
                key=lambda k: self._relationships_last_accessed.get(k, 0),
            )
        elif strategy == "LFU":
            sorted_rels = sorted(
                self._relationship_cache.keys(),
                key=lambda k: self._relationships_access_counts.get(k, 0),
            )
        else:
            logger.warning(f"Unsupported swap-out strategy: {strategy}, using LRU")
            sorted_rels = sorted(
                self._relationship_cache.keys(),
                key=lambda k: self._relationships_last_accessed.get(k, 0),
            )

        removed_count = 0
        for rel_key in sorted_rels[: min(count, len(sorted_rels))]:
            self._relationship_cache.pop(rel_key, None)
            self._relationships_access_counts.pop(rel_key, None)
            self._relationships_last_accessed.pop(rel_key, None)
            removed_count += 1

        logger.info(
            "Evicted %d relationship-cache entries with %s; rustworkx edges "
            "remain resident.",
            removed_count,
            strategy,
        )

    def get_all_relations_with_samples(self, samples_per_type: int = 2) -> dict:
        rel_samples = {}
        seen_pairs = set()
        
        for edge_idx in self.rx_graph.edge_indices():
            src_idx, tgt_idx = self.rx_graph.get_edge_endpoints_by_index(edge_idx)
            data = self.rx_graph.get_edge_data_by_index(edge_idx)
            
            
            u = self._index_to_uid.get(src_idx, "")
            v = self._index_to_uid.get(tgt_idx, "")
            if not u or not v:
                continue
                
            rel_type = data.get("type")
            if not rel_type:
                continue
            if rel_type not in rel_samples:
                rel_samples[rel_type] = []
            if len(rel_samples[rel_type]) >= samples_per_type:
                continue
            if (v, u, rel_type) in seen_pairs:
                continue
            if v in self._uid_to_index and u in self._uid_to_index:
                v_idx = self._uid_to_index[v]
                u_idx = self._uid_to_index[u]
                try:
                    rev_edge_data_list = self.rx_graph.get_all_edge_data(v_idx, u_idx)
                except Exception:  # rustworkx.NoEdgeBetweenNodes
                    rev_edge_data_list = []
                has_reverse = any(ed.get("type") == rel_type for ed in rev_edge_data_list)
                if has_reverse:
                    rel_samples[rel_type].append(f"{u} <-> {v}")
                    seen_pairs.add((u, v, rel_type))
                    seen_pairs.add((v, u, rel_type))
                    continue
            rel_samples[rel_type].append(f"{u} -> {v}")
            seen_pairs.add((u, v, rel_type))
        return rel_samples

    
    def get_unit_data(self, uid: str) -> Optional[MemoryUnit]:
        """Compatibility alias for ``get_unit``."""
        return self.get_unit(uid)

    def build_semantic_map_index(self):
        """Build semantic map index."""
        self.semantic_map.build_index()

    def rebuild_all_indexes(self):
        """Rebuild all indexes."""
        if hasattr(self.semantic_map, "rebuild_all_indexes"):
            self.semantic_map.rebuild_all_indexes()
        else:
            self.semantic_map.build_index()

    def get_all_relations(self) -> List[str]:
        """Return all relations."""
        rel_types = set()
        for edge_idx in self.rx_graph.edge_indices():
            data = self.rx_graph.get_edge_data_by_index(edge_idx)
            rel_type = data.get("type")
            if rel_type:
                rel_types.add(rel_type)
        return list(rel_types)

    def traverse_explicit_nodes(
        self,
        uid: str,
        relationship_type: Optional[str] = None,
        direction: str = "successors",
        ms_names: Optional[List[str]] = None,
    ) -> List[MemoryUnit]:
        """Traverse explicit nodes."""
        if uid not in self._uid_to_index:
            logger.warning(f"Node '{uid}' is not in the graph; cannot traverse.")
            return []

        idx = self._uid_to_index[uid]
        neighbor_ids: Set[str] = set()
        
        if direction == "successors":
            for _, tgt_idx, edge_data in self.rx_graph.out_edges(idx):
                successor_uid = self._index_to_uid.get(tgt_idx, "")
                if not successor_uid:
                    continue
                if relationship_type:
                    if edge_data and edge_data.get("type") == relationship_type:
                        neighbor_ids.add(successor_uid)
                else:
                    neighbor_ids.add(successor_uid)
        elif direction == "predecessors":
            for src_idx, _, edge_data in self.rx_graph.in_edges(idx):
                predecessor_uid = self._index_to_uid.get(src_idx, "")
                if not predecessor_uid:
                    continue
                if relationship_type:
                    if edge_data and edge_data.get("type") == relationship_type:
                        neighbor_ids.add(predecessor_uid)
                else:
                    neighbor_ids.add(predecessor_uid)
        elif direction == "all":
            all_neighbor_uids_temp = set()
            
            for _, tgt_idx, _ in self.rx_graph.out_edges(idx):
                neighbor_uid = self._index_to_uid.get(tgt_idx, "")
                if neighbor_uid:
                    all_neighbor_uids_temp.add(neighbor_uid)
            
            for src_idx, _, _ in self.rx_graph.in_edges(idx):
                neighbor_uid = self._index_to_uid.get(src_idx, "")
                if neighbor_uid:
                    all_neighbor_uids_temp.add(neighbor_uid)

            for neighbor_uid in all_neighbor_uids_temp:
                if neighbor_uid not in self._uid_to_index:
                    continue
                neighbor_idx = self._uid_to_index[neighbor_uid]
                passes_filter = False
                if not relationship_type:
                    passes_filter = True
                else:
                    for edge_idx in self.rx_graph.edge_indices_from_endpoints(idx, neighbor_idx):
                        edge_data = self.rx_graph.get_edge_data_by_index(edge_idx)
                        if edge_data and edge_data.get("type") == relationship_type:
                            passes_filter = True
                            break
                    if not passes_filter:
                        for edge_idx in self.rx_graph.edge_indices_from_endpoints(neighbor_idx, idx):
                            edge_data = self.rx_graph.get_edge_data_by_index(edge_idx)
                            if edge_data and edge_data.get("type") == relationship_type:
                                passes_filter = True
                                break
                if passes_filter:
                    neighbor_ids.add(neighbor_uid)
        else:
            logger.warning(
                f"Invalid traversal direction: '{direction}'. Expected 'successors', 'predecessors', or 'all'."
            )
            return []

        if ms_names:
            ms_units = set()
            for name in ms_names:
                if name.startswith("ms:"):
                    space_name = name[3:]
                else:
                    space_name = name
                space = self.semantic_map.get_memory_space(space_name)
                if space:
                    space_uids = space.get_all_unit_uids(recursive=True)
                    ms_units.update(space_uids)
                else:
                    logger.warning(f"Memory space '{space_name}' was not found and was ignored.")
            neighbor_ids.intersection_update(ms_units)

        results: List[MemoryUnit] = []
        for nid in neighbor_ids:
            unit = self.get_unit(nid)
            if unit:
                results.append(unit)
        return results

    def traverse_implicit_nodes(
        self, uid: str, k: int = 5, ms_names: Optional[List[str]] = None
    ) -> List[Tuple[MemoryUnit, float]]:
        """Traverse implicit nodes."""
        start_unit = self.get_unit(uid)
        if not start_unit or start_unit.embedding is None:
            logger.warning(f"Node '{uid}' does not exist or has no embedding; cannot run implicit traversal.")
            return []

        normed = None
        if ms_names:
            normed = [(n[3:] if n.startswith("ms:") else n) for n in ms_names]
        similar_units_with_scores = self.semantic_map.search_similarity_by_vector(
            start_unit.embedding,
            k=k + 1,  
            ms_names=normed,
        )

        results: List[Tuple[MemoryUnit, float]] = []
        for unit, score in similar_units_with_scores:
            if unit.uid != uid:
                results.append((unit, score))
            if len(results) >= k:
                break
        return results

    def display_graph_summary(self):
        """Display graph summary."""
        num_map_units = len(self.semantic_map.memory_units)
        num_map_indexed = (
            self.semantic_map.faiss_index.ntotal if self.semantic_map.faiss_index else 0
        )
        num_map_spaces = len(self.semantic_map.memory_spaces)

        # Graph and SemanticMap counts are reported separately for diagnostics.
        num_graph_nodes = self.rx_graph.num_nodes()
        num_graph_edges = self.rx_graph.num_edges()

        num_dirty_nodes = len(self._nodes_dirty_flag)
        num_modified_units = len(self._modified_units)
        num_deleted_units = len(self._deleted_units)
        num_modified_relationships = len(self._modified_relationships)
        num_deleted_relationships = len(self._deleted_relationships)

        summary = (
            f"--- SemanticGraph Summary ---\n"
            f"SemanticMap:\n"
            f"  - memory units: {num_map_units}\n"
            f"  - indexed vectors: {num_map_indexed}\n"
            f"  - memory spaces: {num_map_spaces} ({list(self.semantic_map.memory_spaces.keys())})\n"
            f"Rustworkx Graph:\n"
            f"  - graph nodes: {num_graph_nodes}\n"
            f"  - edges (relationships): {num_graph_edges}\n"
            f"  - mapping size: {len(self._uid_to_index)}\n"
            f"memory management:\n"
            f"  - dirty graph nodes: {num_dirty_nodes}\n"
            f"  - modified units pending sync: {num_modified_units}\n"
            f"  - deleted units pending sync: {num_deleted_units}\n"
            f"  - modified relationships pending sync: {num_modified_relationships}\n"
            f"  - deleted relationships pending sync: {num_deleted_relationships}\n"
            f"external storage connection:\n"
            f"  - RocksDB payload store: {'connected' if self._payload_store and self._payload_store.is_connected else 'not connected'}\n"
            f"---------------------------\n"
        )
        print(summary)
        logger.info(summary.replace("\n", " | "))

    
    # Compatibility helpers retained for the public Mandol API.
    

    def get_all_memory_space_names(self) -> List[str]:
        """Return all memory space names."""
        return self.semantic_map.get_all_memory_space_names()

    def get_memory_space_structures(self) -> List[dict]:
        """Return serialized MemorySpace hierarchy from SemanticMap."""
        return self.semantic_map.get_memory_space_structures()

    def deduplicate_units(self, units: List[MemoryUnit]) -> List[MemoryUnit]:
        """Deduplicate units."""
        return self.semantic_map.deduplicate_units(units)

    def filter_memory_units(
        self,
        candidate_units: Optional[List[MemoryUnit]] = None,
        filter_condition: Optional[dict] = None,
        ms_names: Optional[List[str]] = None,
        recursive: bool = True,
    ) -> List[MemoryUnit]:
        normed = None
        if ms_names:
            normed = [(n[3:] if n.startswith("ms:") else n) for n in ms_names]
        return self.semantic_map.filter_memory_units(
            candidate_units=candidate_units,
            filter_condition=filter_condition,
            ms_names=normed,
            recursive=recursive,
        )

    def get_explicit_neighbors(
        self,
        uids: List[str],
        rel_type: Optional[str] = None,
        ms_names: Optional[List[str]] = None,
        direction: str = "successors",
        recursive: bool = True,
    ) -> List[MemoryUnit]:
        """Return explicit neighbors."""
        result = []
        for uid in uids:
            if uid not in self._uid_to_index:
                continue
            idx = self._uid_to_index[uid]
            
            
            neighbor_uids = []
            if direction == "successors":
                for _, tgt_idx, edge_data in self.rx_graph.out_edges(idx):
                    n_uid = self._index_to_uid.get(tgt_idx, "")
                    if n_uid:
                        neighbor_uids.append((n_uid, edge_data))
            elif direction == "predecessors":
                for src_idx, _, edge_data in self.rx_graph.in_edges(idx):
                    n_uid = self._index_to_uid.get(src_idx, "")
                    if n_uid:
                        neighbor_uids.append((n_uid, edge_data))
            else:
                for _, tgt_idx, edge_data in self.rx_graph.out_edges(idx):
                    n_uid = self._index_to_uid.get(tgt_idx, "")
                    if n_uid:
                        neighbor_uids.append((n_uid, edge_data))
                for src_idx, _, edge_data in self.rx_graph.in_edges(idx):
                    n_uid = self._index_to_uid.get(src_idx, "")
                    if n_uid:
                        neighbor_uids.append((n_uid, edge_data))
                        
            for n, edge_data in neighbor_uids:
                if rel_type:
                    if not edge_data or edge_data.get("type") != rel_type:
                        continue
                unit = self.semantic_map.get_unit(n)
                if unit is not None:
                    result.append(unit)

        if ms_names:
            ms_units = set(
                u.uid
                for u in self.get_units_in_memory_space(ms_names, recursive=recursive)
            )
            result = [u for u in result if u.uid in ms_units]
        return result

    def get_implicit_neighbors(
        self,
        uids: List[str],
        top_k: int = 5,
        ms_names: Optional[List[str]] = None,
        recursive: bool = True,
    ) -> List[MemoryUnit]:
        """Return implicit neighbors."""
        if isinstance(uids, str):
            uids = [uids]
        all_results = []
        seen = set()
        for uid in uids:
            unit = self.semantic_map.get_unit(uid)
            if not unit or unit.embedding is None:
                continue
            
            if ms_names:
                units_with_scores = self.semantic_map.search_similarity_by_vector(
                    unit.embedding, k=top_k + 1, ms_names=ms_names
                )
            else:
                units_with_scores = self.semantic_map.search_similarity_by_vector(
                    unit.embedding, k=top_k + 1
                )
            for neighbor, _ in units_with_scores:
                if neighbor.uid != uid and neighbor.uid not in seen:
                    seen.add(neighbor.uid)
                    all_results.append(neighbor)
                if len(all_results) >= top_k * len(uids):
                    break
        return all_results

    
    
    
    
    

    def save_graph(
        self,
        directory_path: str,
        freeze_retrievers: bool = False,
        force_rebuild_retrievers: bool = False,
        retriever_methods_to_build: Optional[List[Any]] = None,
        build_sparse_vectors: bool = True,
    ):
        """Persist the graph, SemanticMap, L2 state, and retrieval indexes.

        Args:
            directory_path: Output directory for graph metadata, SemanticMap
                state, rustworkx graph data, L2 storage, and retrieval indexes.
            freeze_retrievers: Whether to materialize static acceleration
                matrices for initialized retrievers.
            force_rebuild_retrievers: Whether auxiliary retriever indexes should
                be rebuilt before saving.
            retriever_methods_to_build: Optional retrieval-method subset to
                build and persist. If omitted, the default BM25/SPLADE policy is
                used.
            build_sparse_vectors: Whether missing SPLADE vectors should be
                generated before SemanticMap persistence.

        Notes:
            The method waits for tiered eviction work submitted before the
            snapshot. Concurrent graph mutation from another user thread is
            not supported while a snapshot is being created.
        """
        self._ensure_open()
        manager = self.tiered_storage_manager
        if manager is None:
            return self._save_graph_snapshot(
                directory_path,
                freeze_retrievers=freeze_retrievers,
                force_rebuild_retrievers=force_rebuild_retrievers,
                retriever_methods_to_build=retriever_methods_to_build,
                build_sparse_vectors=build_sparse_vectors,
            )

        with manager.quiesce(lambda: len(self.semantic_map.memory_units)):
            return self._save_graph_snapshot(
                directory_path,
                freeze_retrievers=freeze_retrievers,
                force_rebuild_retrievers=force_rebuild_retrievers,
                retriever_methods_to_build=retriever_methods_to_build,
                build_sparse_vectors=build_sparse_vectors,
            )

    def _save_graph_snapshot(
        self,
        directory_path: str,
        freeze_retrievers: bool = False,
        force_rebuild_retrievers: bool = False,
        retriever_methods_to_build: Optional[List[Any]] = None,
        build_sparse_vectors: bool = True,
    ):
        """Write graph and payload state while tiered eviction is quiescent."""
        os.makedirs(directory_path, exist_ok=True)
        logger.info(f"Saving SemanticGraph to sandbox directory: {directory_path}")

        
        semantic_map_dir = os.path.join(directory_path, "semantic_map_data")
        self.semantic_map._save_map_for_graph_snapshot(
            semantic_map_dir,
            build_sparse_vectors=build_sparse_vectors,
        )

        
        graph_path = os.path.join(directory_path, "rx_graph.pkl")
        try:
            graph_data = {
                "rx_graph": self.rx_graph,
                "_uid_to_index": self._uid_to_index,
                "_index_to_uid": self._index_to_uid,
            }
            with open(graph_path, "wb") as f:
                pickle.dump(graph_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Rustworkx graph saved (Pickle): {os.path.getsize(graph_path)/1024/1024:.2f} MB")
        except Exception as e:
            logger.error(f"Failed to save graph structure: {e}")
            
            try:
                edge_list_path = os.path.join(directory_path, "rx_edges.json")
                edges = []
                for edge_idx in self.rx_graph.edge_indices():
                    src_idx, tgt_idx = self.rx_graph.get_edge_endpoints_by_index(edge_idx)
                    data = self.rx_graph.get_edge_data_by_index(edge_idx)
                    u = self._index_to_uid.get(src_idx, "")
                    v = self._index_to_uid.get(tgt_idx, "")
                    if u and v:
                        edges.append({"u": u, "v": v, "d": data})
                with open(edge_list_path, "wb") as f:
                    f.write(orjson.dumps(edges))
                logger.warning(f"Fell back to edge-list-only graph save: {edge_list_path}")
            except Exception as e2:
                logger.error(f"Fallback graph save also failed: {e2}")

        
        retrieval_indices_dir = os.path.join(directory_path, "retrieval_indices")
        os.makedirs(retrieval_indices_dir, exist_ok=True)
        
        retriever_save_status = {}
        static_index_status = {}
        multi_retriever = self.get_multi_retriever()
        
        if multi_retriever is not None:
            try:
                from ..retrieval.retrieval_interface import RetrievalMethod
                
                if retriever_methods_to_build is None:
                    retriever_methods = [RetrievalMethod.BM25]
                    if build_sparse_vectors:
                        retriever_methods.append(RetrievalMethod.SPLADE)
                else:
                    retriever_methods = list(retriever_methods_to_build)

                logger.info(f"Calling build_all_indexes for retriever indexes: {[method.value for method in retriever_methods]}")
                build_result = multi_retriever.build_all_indexes(
                    methods_to_build=retriever_methods,
                    force_rebuild=force_rebuild_retrievers
                )
                logger.info(f"build_all_indexes complete: {build_result}")

                if freeze_retrievers:
                    logger.info("Building retriever static acceleration matrices (freeze_retrievers=True)...")
                    static_index_status = self.build_freeze_indexes()
                    logger.info(f"Retriever static acceleration matrix build status: {static_index_status}")
                
                
                if RetrievalMethod.BM25 in retriever_methods and RetrievalMethod.BM25 in multi_retriever.retrievers:
                    bm25_retriever = multi_retriever.retrievers[RetrievalMethod.BM25]
                    if hasattr(bm25_retriever, 'save_index'):
                        bm25_dir = os.path.join(retrieval_indices_dir, "bm25")
                        if bm25_retriever.save_index(bm25_dir):
                            retriever_save_status["bm25"] = True
                            logger.info("BM25 index saved")
                        else:
                            retriever_save_status["bm25"] = False
                
                
                if RetrievalMethod.SPLADE in retriever_methods and RetrievalMethod.SPLADE in multi_retriever.retrievers:
                    splade_retriever = multi_retriever.retrievers[RetrievalMethod.SPLADE]
                    if hasattr(splade_retriever, 'save_index'):
                        splade_dir = os.path.join(retrieval_indices_dir, "splade")
                        if splade_retriever.save_index(splade_dir):
                            retriever_save_status["splade"] = True
                            logger.info("SPLADE index saved")
                        else:
                            retriever_save_status["splade"] = False
                            
            except Exception as e:
                logger.warning(f"Exception while saving retriever indexes: {e}")

        l2_relative_path = "l2_database/payloads.rocksdb"
        target_rocksdb_path = os.path.join(directory_path, l2_relative_path)
        rocksdb_enabled = bool(
            self._payload_store is not None
            and self._payload_store.is_connected
        )
        paging_manager = self.semantic_map.tiered_storage_manager

        if rocksdb_enabled:
            destination_path = os.path.abspath(target_rocksdb_path)
            try:
                self._payload_store.copy_to(destination_path)
                logger.info(
                    "RocksDB payload checkpoint ready: %s", destination_path
                )
            except Exception as e:
                logger.error(f"RocksDB payload checkpoint copy failed: {e}")

        
        state = {
            "version": "3.0_rocksdb",
            "saved_time": datetime.now().isoformat(),
            "management": {
                "_nodes_access_counts": self._nodes_access_counts,
                "_nodes_last_accessed": self._nodes_last_accessed,
                "_nodes_dirty_flag": list(self._nodes_dirty_flag),
                "_modified_units": list(self._modified_units),
                "_deleted_units": list(self._deleted_units),
                "_modified_relationships": [list(r) for r in self._modified_relationships],
                "_deleted_relationships": [list(r) for r in self._deleted_relationships],
            },
            "relationship_cache_meta": {
                f"{k[0]}|{k[1]}|{k[2]}": {
                    "props": v, 
                    "acc": self._relationships_access_counts.get(k, 0),
                    "last": self._relationships_last_accessed.get(k, 0)
                }
                for k, v in self._relationship_cache.items()
            },
            "retriever_indices_saved": retriever_save_status,
            "retrieval": {
                "indices_saved": retriever_save_status,
                "frozen_matrices_saved": static_index_status,
            },
            "l2_storage": {
                "enabled": rocksdb_enabled,
                "backend": "rocksdb" if rocksdb_enabled else None,
                "rocksdb_relative_path": (
                    l2_relative_path if rocksdb_enabled else None
                ),
                "rocksdb_path": (
                    self._payload_store.db_path if rocksdb_enabled else None
                ),
                "max_capacity": (
                    paging_manager.max_capacity if paging_manager is not None else None
                ),
                "high_watermark": (
                    paging_manager.high_watermark if paging_manager is not None else None
                ),
                "low_watermark": (
                    paging_manager.low_watermark if paging_manager is not None else None
                ),
            },
            "high_level_memory_build": self.get_high_level_memory_build_state(),
        }
        
        state_path = os.path.join(directory_path, "graph_state.json")
        with open(state_path, "wb") as f:
            f.write(orjson.dumps(state, option=orjson.OPT_INDENT_2))
            
        logger.info("SemanticGraph sandbox save complete.")

    @classmethod
    def load_graph(cls, directory_path: str, **kwargs) -> "SemanticGraph":
        """Load a graph saved by ``save_graph``.

        Args:
            directory_path: Directory containing ``graph_state.json`` and the
                saved SemanticMap and graph artifacts.
            **kwargs: Constructor overrides forwarded to ``SemanticMap.load_map``.

        Returns:
            A reconstructed SemanticGraph instance.
        """
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory does not exist: {directory_path}")

        state_path = os.path.join(directory_path, "graph_state.json")
        if not os.path.exists(state_path):
            raise FileNotFoundError(f"SemanticGraph metadata file not found: {state_path}")

        logger.info(f"Loading SemanticGraph from: {directory_path}")

        with open(state_path, "rb") as f:
            state = orjson.loads(f.read())

        legacy_placement = state.get("l2_storage", {}).get("mode")
        if legacy_placement == "store_only":
            raise ValueError(
                "This checkpoint uses the removed store_only payload placement "
                "and cannot be loaded by this release."
            )

        
        map_dir = os.path.join(directory_path, "semantic_map_data")
        loaded_map = SemanticMap.load_map(map_dir, **kwargs)

        instance = cls(semantic_map_instance=loaded_map)
        instance.set_high_level_memory_build_state(
            state.get("high_level_memory_build")
            or loaded_map.get_high_level_memory_build_state()
            if hasattr(loaded_map, "get_high_level_memory_build_state")
            else state.get("high_level_memory_build", {})
        )

        
        rx_graph_path = os.path.join(directory_path, "rx_graph.pkl")
        
        if os.path.exists(rx_graph_path):
            try:
                with open(rx_graph_path, "rb") as f:
                    graph_data = pickle.load(f)
                instance.rx_graph = graph_data["rx_graph"]
                instance._uid_to_index = graph_data["_uid_to_index"]
                instance._index_to_uid = graph_data["_index_to_uid"]
                logger.info(f"Loaded rustworkx graph: {instance.rx_graph.num_nodes()} nodes, {instance.rx_graph.num_edges()} edges")
            except Exception as e:
                logger.error(f"Failed to load rustworkx pickle: {e}; rebuilding empty graph nodes from SemanticMap.")
                
                for uid in instance.semantic_map.memory_units:
                    if uid not in instance._uid_to_index:
                        idx = instance.rx_graph.add_node({"uid": uid})
                        instance._uid_to_index[uid] = idx
                        instance._index_to_uid[idx] = uid
        else:
            logger.warning("Graph file rx_graph.pkl was not found; initializing an empty graph.")

        
        retrieval_indices_dir = os.path.join(directory_path, "retrieval_indices")
        if os.path.exists(retrieval_indices_dir):
            
            instance._index_loading_root = retrieval_indices_dir
            logger.info(f"Detected retriever index directory: {retrieval_indices_dir}, it will be loaded when MultiRetriever is first requested")
        else:
            instance._index_loading_root = None

        
        try:
            mgmt = state.get("management", {})
            instance._nodes_access_counts = mgmt.get("_nodes_access_counts", {})
            instance._nodes_last_accessed = mgmt.get("_nodes_last_accessed", {})
            instance._nodes_dirty_flag = set(mgmt.get("_nodes_dirty_flag", []))
            instance._modified_units = set(mgmt.get("_modified_units", []))
            instance._deleted_units = set(mgmt.get("_deleted_units", []))
            
            instance._modified_relationships = set(tuple(r) for r in mgmt.get("_modified_relationships", []))
            instance._deleted_relationships = set(tuple(r) for r in mgmt.get("_deleted_relationships", []))
            
            # Restore hot-cache metadata saved with public UID triples.
            cache_meta = state.get("relationship_cache_meta", {})
            for k_str, v in cache_meta.items():
                parts = k_str.split("|")
                if len(parts) == 3:
                    key = (parts[0], parts[1], parts[2])
                    instance._relationship_cache[key] = v.get("props", {})
                    instance._relationships_access_counts[key] = v.get("acc", 0)
                    instance._relationships_last_accessed[key] = v.get("last", 0)
                    
        except Exception as e:
            logger.warning(f"Failed to load management state: {e}")

        try:
            l2_info = state.get("l2_storage", {})
            persistent_payloads_enabled = bool(l2_info.get("enabled")) or (
                legacy_placement == "tiered_cache"
            )
            if persistent_payloads_enabled:
                relative_path = l2_info.get("rocksdb_relative_path")
                resolved_db_path = (
                    os.path.join(directory_path, relative_path)
                    if relative_path
                    else None
                )

                if resolved_db_path and os.path.isdir(resolved_db_path):
                    logger.info(
                        "Restoring RocksDB payload store from: %s", relative_path
                    )
                    l2_base = os.path.dirname(resolved_db_path)
                    paging_kwargs = {
                        key: l2_info[key]
                        for key in (
                            "max_capacity",
                            "high_watermark",
                            "low_watermark",
                        )
                        if l2_info.get(key) is not None
                    }
                    connected = instance.connect_to_l2(
                        l2_base_path=l2_base,
                        **paging_kwargs,
                    )
                    if not connected:
                        logger.warning(
                            "RocksDB payload restore failed; cold checkpoint "
                            "payloads are unavailable in the loaded graph."
                        )
                else:
                    logger.warning(
                        "graph_state.json records RocksDB payload storage, but "
                        "the checkpoint directory is missing; skipping restore."
                    )
        except Exception as e:
            logger.warning(f"Failed to restore L2 storage connection (non-fatal): {e}")

        return instance

    
    
    

    def units_union(self, *args):
        """Delegate first-seen union of unit collections to SemanticMap."""
        return self.semantic_map.units_union(*args)

    def units_intersection(self, *args):
        """Delegate UID-based intersection of unit collections to SemanticMap."""
        return self.semantic_map.units_intersection(*args)

    def units_difference(self, arg1, arg2):
        """Delegate UID-based difference of unit collections to SemanticMap."""
        return self.semantic_map.units_difference(arg1, arg2)

    def aggregate_results(
        self, memory_units: List[MemoryUnit]
    ) -> Dict[MemoryUnit, int]:
        counter = Counter(memory_units)
        return dict(counter)

    def add_memory_space_in_map(self, space: "MemorySpace"):
        """Add memory space in map."""
        return self.semantic_map.add_memory_space(space)

    
    
    
    
    def build_sparse_embeddings(
        self,
        units: Optional[List[MemoryUnit]] = None,
        # text_field: str = "text_content",
        model_name: str = "naver/splade-v3",
        batch_size: int = 32,
        force_rebuild: bool = False,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """Build sparse embeddings."""
        return self.semantic_map.build_sparse_embeddings(
            units=units,
            # text_field=text_field,
            model_name=model_name,
            batch_size=batch_size,
            force_rebuild=force_rebuild,
            show_progress=show_progress
        )
