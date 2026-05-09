"""Graph service for building and traversing semantic relationships.

Provides CRUD operations for explicit edges in the knowledge/event graph and
synthesizes implicit neighbors via embedding similarity search. Supports BFS
expansion for multi-hop graph traversal.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..domain.memory_unit import MemoryUnit
from ..domain.types import Embedding, SpaceName, Uid
from ..ports.graph_store import GraphStore
from .semantic_map import SemanticMapService


class SemanticGraphService:
    """Builds and queries the semantic relationship graph.

    Wraps a GraphStore with higher-level methods that coordinate with the
    SemanticMapService to ensure graph consistency. Provides both explicit
    edge CRUD and implicit neighbor discovery via vector search.

    Args:
        semantic_map: The SemanticMapService used for unit storage and search.
        graph_store: The underlying GraphStore for relationship persistence.
    """

    def __init__(self, *, semantic_map: SemanticMapService, graph_store: GraphStore):
        self.semantic_map = semantic_map
        self._graph = graph_store

    def get_graph_store(self) -> GraphStore:
        """Return the underlying GraphStore instance."""
        return self._graph

    def add_unit(
        self,
        unit: MemoryUnit,
        *,
        space_names: Optional[Sequence[Union[str, SpaceName]]] = None,
        ensure_embedding: bool = True,
        rebuild_index_immediately: bool = False,
    ) -> None:
        """Add a unit to the semantic map and optionally index it.

        Delegates directly to the underlying SemanticMapService.

        Args:
            unit: The MemoryUnit to add.
            space_names: Optional sequence of space names to assign the unit to.
            ensure_embedding: If True, compute an embedding before storing.
            rebuild_index_immediately: If True, rebuild the vector index after insertion.
        """
        self.semantic_map.add_unit(
            unit,
            space_names=space_names,
            ensure_embedding=ensure_embedding,
            rebuild_index_immediately=rebuild_index_immediately,
        )

    def delete_unit(self, uid: Union[str, Uid]) -> None:
        """Remove a unit and all its inbound/outbound relationships.

        Args:
            uid: The UID (or string) of the unit to delete.
        """
        u = Uid(str(uid))
        for neighbor_uid in self._graph.get_neighbors(u, direction="out"):
            self._graph.delete_relationship(u, neighbor_uid)
        for neighbor_uid in self._graph.get_neighbors(u, direction="in"):
            self._graph.delete_relationship(neighbor_uid, u)
        if hasattr(self._graph, "_g"):
            try:
                self._graph._g.remove_node(u)
            except Exception:
                pass
        self.semantic_map.delete_unit(uid)

    def add_relationship(
        self,
        source_uid: Union[str, Uid],
        target_uid: Union[str, Uid],
        relationship_name: str,
        **properties: Any,
    ) -> None:
        """Create a named directed edge between two units.

        Requires both source and target to exist in the semantic map, unless
        the source UID starts with \"ms:\" (system-level aggregate units).

        Args:
            source_uid: UID of the source unit.
            target_uid: UID of the target unit.
            relationship_name: Name of the relationship type (e.g. RELATED_TO).
            **properties: Arbitrary key-value properties stored on the edge.

        Raises:
            KeyError: If either unit is not found in the semantic map.
        """
        s = Uid(str(source_uid))
        t = Uid(str(target_uid))
        if self.semantic_map.get_unit(s) is None and not str(s).startswith("ms:"):
            raise KeyError(f"source unit not found: {s}")
        if self.semantic_map.get_unit(t) is None and not str(t).startswith("ms:"):
            raise KeyError(f"target unit not found: {t}")
        self._graph.upsert_relationship(s, t, str(relationship_name), dict(properties))

    def get_relationship(
        self, source_uid: Union[str, Uid], target_uid: Union[str, Uid], relationship_name: str
    ) -> Optional[Dict[str, Any]]:
        """Look up a specific named edge between two units.

        Args:
            source_uid: Source UID.
            target_uid: Target UID.
            relationship_name: Name of the relationship type.

        Returns:
            The edge properties dict, or None if no edge exists.
        """
        return self._graph.get_relationship(
            Uid(str(source_uid)), Uid(str(target_uid)), str(relationship_name)
        )

    def delete_relationship(
        self,
        source_uid: Union[str, Uid],
        target_uid: Union[str, Uid],
        relationship_name: Optional[str] = None,
    ) -> None:
        """Remove a relationship edge.

        If relationship_name is None, all edges between the two nodes are
        deleted.

        Args:
            source_uid: Source UID.
            target_uid: Target UID.
            relationship_name: Specific relationship to delete, or None to delete all.
        """
        self._graph.delete_relationship(
            Uid(str(source_uid)), Uid(str(target_uid)), relationship_name
        )

    def get_explicit_neighbors(
        self,
        uids: Sequence[Union[str, Uid]],
        *,
        rel_type: Optional[str] = None,
        direction: str = "out",
    ) -> List[MemoryUnit]:
        """Get neighbors reachable via explicit edges from a set of nodes.

        Args:
            uids: Sequence of source UIDs.
            rel_type: Optional filter to only return neighbors with this relationship type.
            direction: Graph traversal direction, \"out\" (default) or \"in\".

        Returns:
            List of MemoryUnits that are direct neighbors via explicit edges.
        """
        out: List[MemoryUnit] = []
        seen: set[Uid] = set()
        for u in uids:
            uid = Uid(str(u))
            for n in self._graph.get_neighbors(uid, rel_type=rel_type, direction=direction):
                if n in seen:
                    continue
                seen.add(n)
                unit = self.semantic_map.get_unit(n)
                if unit is not None:
                    out.append(unit)
        return out

    def get_implicit_neighbors(
        self,
        uids: Sequence[Union[str, Uid]],
        *,
        top_k: int = 10,
    ) -> List[Tuple[MemoryUnit, float]]:
        """Find neighbors via embedding similarity (no explicit edges needed).

        Computes the mean embedding of the given units and searches for the
        top_k most similar units in the vector index.

        Args:
            uids: Sequence of source UIDs whose embeddings are averaged.
            top_k: Number of nearest neighbors to return.

        Returns:
            List of (MemoryUnit, similarity_score) tuples.
        """
        queries: List[Embedding] = []
        for u in uids:
            unit = self.semantic_map.get_unit(u)
            if unit is None or unit.embedding is None:
                continue
            queries.append(np.asarray(unit.embedding, dtype=np.float32).reshape(-1))
        if not queries:
            return []
        q = np.mean(np.stack(queries, axis=0), axis=0)
        return self.semantic_map.search_by_vector(q, top_k=top_k)

    def get_units_in_spaces(
        self,
        space_names: Sequence[Union[str, SpaceName]],
        *,
        mode: str = "union",
        recursive: bool = True,
    ) -> List[MemoryUnit]:
        """Get all units in the given spaces.

        Delegates to SemanticMapService.get_units_in_spaces.

        Args:
            space_names: Sequence of space names.
            mode: \"union\" (default) or \"intersection\".
            recursive: If True, include units from child spaces.

        Returns:
            List of MemoryUnits in the specified spaces.
        """
        return self.semantic_map.get_units_in_spaces(
            space_names, mode=mode, recursive=recursive
        )

    def bfs_expand_units(
        self,
        seeds: Sequence[MemoryUnit],
        *,
        per_seed: int = 3,
        hops: int = 1,
        rel_type: Optional[str] = None,
    ) -> List[MemoryUnit]:
        """Expand a set of seed units via BFS on the explicit graph.

        Traverses outgoing and incoming edges up to the specified number of
        hops, collecting units until the per_seed budget is exhausted.

        Args:
            seeds: The seed MemoryUnits to expand from.
            per_seed: Maximum units to collect per seed node.
            hops: Maximum number of BFS hops (depth).
            rel_type: Optional relationship type filter.

        Returns:
            List of MemoryUnits discovered through BFS expansion.
        """
        if not seeds or per_seed <= 0 or hops <= 0:
            return []

        results: List[MemoryUnit] = []
        seen: set[Uid] = set()
        queue: List[Tuple[Uid, int]] = []

        for u in seeds:
            uid = Uid(str(u.uid))
            if uid in seen:
                continue
            seen.add(uid)
            queue.append((uid, 0))

        while queue and len(results) < per_seed * max(1, len(seeds)):
            uid, depth = queue.pop(0)
            if depth >= hops:
                continue

            neighbors = []
            try:
                neighbors.extend(
                    self._graph.get_neighbors(uid, rel_type=rel_type, direction="out")
                )
                neighbors.extend(
                    self._graph.get_neighbors(uid, rel_type=rel_type, direction="in")
                )
            except Exception:
                neighbors = []

            for n in neighbors:
                if n in seen:
                    continue
                seen.add(n)
                unit = self.semantic_map.get_unit(n)
                if unit is not None:
                    results.append(unit)
                    if len(results) >= per_seed * max(1, len(seeds)):
                        break
                queue.append((n, depth + 1))

        return results

    def flush(self) -> None:
        """Persist all pending changes to the underlying stores."""
        self._graph.flush()
        self.semantic_map.flush()
