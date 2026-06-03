"""Graph service for building and traversing semantic relationships.

Provides CRUD operations for explicit edges in the knowledge/event graph and
synthesizes implicit neighbors via embedding similarity search. Supports BFS
expansion for multi-hop graph traversal.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..domain.memory_unit import MemoryUnit
from ..domain.types import Embedding, SpaceName, Uid
from ..ports.graph_store import GraphStore
from .semantic_map import SemanticMapService

if TYPE_CHECKING:
    from ..retrieval.types import (
        CorefTraceResult,
        EntityInvolvementResult,
        EntitySubgraphResult,
        EvidenceChainResult,
        SummaryEvidenceChainResult,
    )

logger = logging.getLogger(__name__)


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
        out_neighbors = list(self._graph.get_neighbors(u, direction="out"))
        in_neighbors = list(self._graph.get_neighbors(u, direction="in"))
        for neighbor_uid in out_neighbors:
            self._graph.delete_relationship(u, neighbor_uid)
        for neighbor_uid in in_neighbors:
            self._graph.delete_relationship(neighbor_uid, u)
        if hasattr(self._graph, "_g"):
            try:
                self._graph._g.remove_node(u)
            except (AttributeError, RuntimeError, KeyError):
                pass
        logger.debug(
            "Deleted unit %s with %d out-edges and %d in-edges",
            u, len(out_neighbors), len(in_neighbors),
        )
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
        logger.debug("Edge added: %s -[%s]-> %s", s, relationship_name, t)

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
        queue: deque[Tuple[Uid, int]] = deque()

        for u in seeds:
            uid = Uid(str(u.uid))
            if uid in seen:
                continue
            seen.add(uid)
            queue.append((uid, 0))

        while queue and len(results) < per_seed * max(1, len(seeds)):
            uid, depth = queue.popleft()
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
            except (AttributeError, RuntimeError, KeyError):
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

        logger.debug(
            "BFS expand: %d seeds × per_seed=%d hops=%d → %d results",
            len(seeds), per_seed, hops, len(results),
        )
        return results

    def flush(self) -> None:
        """Persist all pending changes to the underlying stores."""
        self._graph.flush()
        self.semantic_map.flush()

    # ------------------------------------------------------------------
    # Public query methods
    # ------------------------------------------------------------------

    def get_edges_of_unit(
        self,
        uid: Union[str, Uid],
        *,
        rel_type: Optional[str] = None,
        direction: str = "all",
    ) -> List[Dict[str, Any]]:
        """Get all relationship edges for a given node.

        Uses O(degree) adjacency lookups via the graph store rather than
        scanning all edges.

        Args:
            uid: The unit UID whose edges are requested.
            rel_type: Optional filter to only return edges of this
                relationship type.
            direction: ``"out"`` for outgoing edges, ``"in"`` for incoming,
                ``"all"`` (default) for both.

        Returns:
            List of edge dicts, each containing:
                - ``"source"``: source UID string
                - ``"target"``: target UID string
                - ``"type"``: relationship name string
                - ``"properties"``: edge properties dict
        """
        uid_obj = Uid(str(uid))
        edges: List[Dict[str, Any]] = []

        if direction in ("out", "all"):
            for neighbor in self._graph.get_neighbors(uid_obj, rel_type=rel_type, direction="out"):
                rel_data = self._graph.get_relationship(uid_obj, neighbor, rel_type) if rel_type else None
                edges.append({
                    "source": str(uid_obj),
                    "target": str(neighbor),
                    "type": rel_type or rel_data.get("rel_type", "UNKNOWN") if rel_data else (rel_type or "UNKNOWN"),
                    "properties": dict(rel_data) if rel_data else {},
                })

        if direction in ("in", "all"):
            for neighbor in self._graph.get_neighbors(uid_obj, rel_type=rel_type, direction="in"):
                rel_data = self._graph.get_relationship(neighbor, uid_obj, rel_type) if rel_type else None
                edges.append({
                    "source": str(neighbor),
                    "target": str(uid_obj),
                    "type": rel_type or rel_data.get("rel_type", "UNKNOWN") if rel_data else (rel_type or "UNKNOWN"),
                    "properties": dict(rel_data) if rel_data else {},
                })

        return edges

    def get_node_neighbors(
        self,
        node_uid: Union[str, Uid],
        *,
        max_depth: int = 2,
        include_semantic: bool = True,
        include_structural: bool = True,
        similarity_threshold: float = 0.7,
    ) -> Dict[str, List[MemoryUnit]]:
        """Combined structural and semantic neighbor query.

        Queries both explicit graph edges and implicit (embedding-based)
        neighbors, returning the results keyed by category.

        Args:
            node_uid: The unit UID whose neighbors are requested.
            max_depth: Maximum depth for structural BFS expansion.
            include_semantic: If True, include semantic (implicit) neighbors.
            include_structural: If True, include structural (explicit) neighbors.
            similarity_threshold: Minimum similarity score [0, 1] for
                semantic neighbors.

        Returns:
            Dict with keys ``"structural"`` and/or ``"semantic"``, each
            mapping to a list of MemoryUnits. The dict only contains keys
            for which the corresponding ``include_*`` flag was True.
        """
        result: Dict[str, List[MemoryUnit]] = {}

        if include_structural:
            unit = self.semantic_map.get_unit(node_uid)
            if unit is not None and max_depth > 1:
                # Use a single BFS expansion to cover all hops, avoiding
                # double-traversal of depth-1 neighbors.
                expanded = self.bfs_expand_units(
                    seeds=[unit],
                    per_seed=20,
                    hops=max_depth,
                )
                seen_uids: set[Uid] = set()
                structural_neighbors: List[MemoryUnit] = []
                for u in expanded:
                    uid = Uid(str(u.uid))
                    if uid not in seen_uids:
                        structural_neighbors.append(u)
                        seen_uids.add(uid)
            else:
                structural_neighbors = self.get_explicit_neighbors([node_uid])
            result["structural"] = structural_neighbors

        if include_semantic:
            implicit = self.get_implicit_neighbors([node_uid], top_k=10)
            semantic_neighbors = [
                unit for unit, score in implicit if score >= similarity_threshold
            ]
            result["semantic"] = semantic_neighbors

        return result

    def search_graph_relations(
        self,
        seed_nodes: Optional[List[str]] = None,
        relation_types: Optional[List[str]] = None,
        max_depth: int = 2,
        limit: int = 50,
    ) -> List[Tuple[str, str, Dict[str, Any]]]:
        """Search graph relations with BFS from seed nodes.

        Performs a breadth-first traversal starting from each seed node,
        collecting (source, target, properties) tuples for every edge
        traversed.

        Args:
            seed_nodes: Optional list of UID strings to use as seeds.
                If None, all units in the semantic map are used.
            relation_types: Optional list of relationship type strings
                to traverse. If None, all types are traversed.
            max_depth: Maximum BFS depth per seed.
            limit: Maximum total number of edge tuples to return.

        Returns:
            List of ``(source_uid_str, target_uid_str, properties_dict)``
            tuples. The properties dict contains at least a ``"rel_type"``
            key.
        """
        results: List[Tuple[str, str, Dict[str, Any]]] = []
        seen_pairs: set = set()

        if seed_nodes is None:
            all_units = self.semantic_map.list_units()
            seed_nodes = [str(u.uid) for u in all_units]

        rel_types: List[Optional[str]] = relation_types if relation_types else [None]

        for seed_str in seed_nodes:
            source_uid = Uid(seed_str)
            queue: deque[Tuple[Uid, int]] = deque([(source_uid, 0)])
            visited: set = {source_uid}

            while queue and len(results) < limit:
                current, depth = queue.popleft()
                if depth >= max_depth:
                    continue

                for rt in rel_types:
                    for neighbor in self._graph.get_neighbors(
                        current, rel_type=rt, direction="out"
                    ):
                        # Include rel_type in pair_key to avoid dropping
                        # multi-type edges between the same source-target pair.
                        pair_key = (str(current), str(neighbor), rt)
                        if pair_key in seen_pairs:
                            continue
                        seen_pairs.add(pair_key)

                        # When no specific rel_type filter is provided,
                        # resolve the actual type and properties from the
                        # relationship stored in the graph.
                        if rt:
                            rel = self._graph.get_relationship(current, neighbor, rt)
                            props: Dict[str, Any] = dict(rel) if rel else {"rel_type": rt}
                        else:
                            props = self._resolve_edge_properties(current, neighbor)

                        results.append((str(current), str(neighbor), props))
                        if len(results) >= limit:
                            break

                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, depth + 1))

                    if len(results) >= limit:
                        break

            if len(results) >= limit:
                break

        return results[:limit]

    def _resolve_edge_properties(
        self,
        source: Uid,
        target: Uid,
    ) -> Dict[str, Any]:
        """Resolve the actual relationship type and properties for an edge.

        When no explicit relation_types filter is provided, this inspects
        the graph store for every relationship type registered between the
        two nodes.

        Args:
            source: Source node UID.
            target: Target node UID.

        Returns:
            Properties dict with at least ``"rel_type"`` set to the
            discovered relationship name, or ``"UNKNOWN"`` if none found.
        """
        for source_uid, target_uid, etype, props in self._graph.get_all_edges():
            if source_uid == source and target_uid == target:
                result: Dict[str, Any] = dict(props) if props else {}
                result["rel_type"] = etype
                return result
        return {"rel_type": "UNKNOWN"}

    def retrieve_entity_subgraph(
        self,
        query: str,
        *,
        max_depth: int = 2,
        rel_types: Optional[List[str]] = None,
        top_k: int = 10,
    ) -> "EntitySubgraphResult":
        """Retrieve an entity relationship panorama centered on one entity.

        Searches for an entity by text, then performs BFS expansion to
        discover neighbors and relationships.

        Args:
            query: Text query to locate the center entity.
            max_depth: Maximum BFS depth for neighbor expansion.
            rel_types: Optional list of relationship types to traverse.
                If None, all types are traversed.
            top_k: Maximum number of neighbors to collect.

        Returns:
            EntitySubgraphResult with center_entity, neighbors,
            relationships, and depth_map.
        """
        from ..retrieval.types import EntitySubgraphResult, RelationshipInfo

        search_results = self.semantic_map.search_by_text(query, top_k=1, space_names=None)
        if not search_results:
            return EntitySubgraphResult()

        center_unit, _ = search_results[0]
        center_uid = Uid(str(center_unit.uid))

        depth_map: Dict[str, int] = {str(center_uid): 0}
        relationships: List[RelationshipInfo] = []
        neighbors: List[MemoryUnit] = []
        seen: set = {center_uid}

        rtypes: List[Optional[str]]
        if rel_types is None:
            rtypes = [None]
        else:
            rtypes = list(rel_types)

        queue: deque[Tuple[Uid, int]] = deque([(center_uid, 0)])

        while queue and len(neighbors) < top_k:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue

            for rt in rtypes:
                if len(neighbors) >= top_k:
                    break
                for n_uid in self._graph.get_neighbors(
                    current, rel_type=rt, direction="out"
                ):
                    if n_uid in seen:
                        continue
                    seen.add(n_uid)

                    unit = self.semantic_map.get_unit(n_uid)
                    if unit is not None:
                        neighbors.append(unit)
                        depth_map[str(n_uid)] = depth + 1

                    relation_props: Dict[str, Any] = {}
                    if rt:
                        rel_data = self._graph.get_relationship(current, n_uid, rt)
                        if rel_data:
                            relation_props = dict(rel_data)

                    relationships.append(
                        RelationshipInfo(
                            source_uid=current,
                            target_uid=n_uid,
                            rel_type=rt or "UNKNOWN",
                            properties=relation_props,
                        )
                    )

                    queue.append((n_uid, depth + 1))

                    if len(neighbors) >= top_k:
                        break

        return EntitySubgraphResult(
            center_entity=center_unit,
            neighbors=neighbors,
            relationships=relationships,
            depth_map=depth_map,
        )

    def trace_evidence(
        self,
        uid: Union[str, Uid],
        *,
        max_depth: int = 2,
        top_k: int = 10,
    ) -> "EvidenceChainResult":
        """Top-down evidence tracing along EVIDENCED_BY edges.

        Starting from the source unit, performs BFS along outgoing
        ``EVIDENCED_BY`` edges to discover evidence units.

        Args:
            uid: The source UID to trace evidence from.
            max_depth: Maximum BFS depth for evidence tracing.
            top_k: Maximum number of evidence units to return.

        Returns:
            EvidenceChainResult with source, evidence list, and depth_map.
        """
        from ..retrieval.types import EvidenceChainResult

        source_unit = self.semantic_map.get_unit(uid)
        if source_unit is None:
            return EvidenceChainResult()

        uid_obj = Uid(str(uid))
        evidence: List[MemoryUnit] = []
        depth_map: Dict[str, int] = {str(uid_obj): 0}
        seen: set = {uid_obj}
        queue: deque[Tuple[Uid, int]] = deque([(uid_obj, 0)])

        while queue and len(evidence) < top_k:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue

            for neighbor in self._graph.get_neighbors(
                current, rel_type="EVIDENCED_BY", direction="out"
            ):
                if neighbor in seen:
                    continue
                seen.add(neighbor)

                unit = self.semantic_map.get_unit(neighbor)
                if unit is not None:
                    evidence.append(unit)
                    if len(evidence) >= top_k:
                        break

                depth_map[str(neighbor)] = depth + 1
                queue.append((neighbor, depth + 1))

        return EvidenceChainResult(
            source=source_unit,
            evidence=evidence,
            depth_map=depth_map,
        )

    def trace_coref(
        self,
        uid: Union[str, Uid],
        *,
        max_depth: int = 2,
        top_k: int = 10,
    ) -> "CorefTraceResult":
        """Bottom-up coreference resolution.

        Follows ``COREF`` edges from the source to discover canonical
        entities/events, then traces ``EVIDENCED_BY`` edges in reverse
        to build the full coreference chain.

        Args:
            uid: The source UID to trace coreference from.
            max_depth: Maximum BFS depth (currently unused; reserved
                for future multi-hop coref resolution).
            top_k: Maximum number of coref chain units to return.

        Returns:
            CorefTraceResult with canonical entities, events, and the
            coref chain.
        """
        from ..retrieval.types import CorefTraceResult

        del max_depth  # reserved for future multi-hop coref resolution

        source_unit = self.semantic_map.get_unit(uid)
        if source_unit is None:
            return CorefTraceResult()

        uid_obj = Uid(str(uid))

        # Step 1: Find canonical entities/events via COREF edges.
        canonical_neighbors = self.get_explicit_neighbors(
            [uid_obj], rel_type="COREF", direction="out"
        )

        canonical_entities: List[MemoryUnit] = []
        canonical_events: List[MemoryUnit] = []

        for unit in canonical_neighbors:
            is_event = False
            unit_spaces = unit.metadata.get("spaces", [])
            if unit_spaces:
                for space in unit_spaces:
                    if "event" in str(space).lower():
                        is_event = True
                        break
            if is_event:
                canonical_events.append(unit)
            else:
                canonical_entities.append(unit)

        # Step 2: Build coref chain — for each canonical, find dialogue
        # units that reference it via EVIDENCED_BY (inbound).
        coref_chain: List[MemoryUnit] = []
        chain_seen: set = set()

        for canon in canonical_entities + canonical_events:
            canon_uid = Uid(str(canon.uid))
            if canon_uid not in chain_seen:
                chain_seen.add(canon_uid)
                coref_chain.append(canon)

            ev_neighbors = self.get_explicit_neighbors(
                [canon_uid], rel_type="EVIDENCED_BY", direction="in"
            )
            for u in ev_neighbors:
                u_uid = Uid(str(u.uid))
                if u_uid not in chain_seen:
                    chain_seen.add(u_uid)
                    coref_chain.append(u)
                    if len(coref_chain) >= top_k:
                        break
            if len(coref_chain) >= top_k:
                break

        return CorefTraceResult(
            source=source_unit,
            canonical_entities=canonical_entities[:top_k],
            canonical_events=canonical_events[:top_k],
            coref_chain=coref_chain[:top_k],
        )

    def retrieve_summary_evidence_chain(
        self,
        uid: Union[str, Uid],
        *,
        include_entities: bool = True,
        include_events: bool = True,
        top_k: int = 10,
    ) -> "SummaryEvidenceChainResult":
        """Summary-to-evidence chain with entity/event resolution.

        Traces ``EVIDENCED_BY`` edges from the summary to find supporting
        evidence, then resolves each evidence unit via ``COREF`` upward
        to discover related canonical entities and events.

        Args:
            uid: The summary unit UID.
            include_entities: If True, include related entity MemoryUnits.
            include_events: If True, include related event MemoryUnits.
            top_k: Maximum number of results per category.

        Returns:
            SummaryEvidenceChainResult with summary, evidence units,
            related entities, and related events.
        """
        from ..retrieval.types import SummaryEvidenceChainResult

        summary_unit = self.semantic_map.get_unit(uid)
        if summary_unit is None:
            return SummaryEvidenceChainResult()

        uid_obj = Uid(str(uid))

        # Step 1: Trace EVIDENCED_BY from summary to get evidence units.
        evidence_units = self.get_explicit_neighbors(
            [uid_obj], rel_type="EVIDENCED_BY", direction="out"
        )

        related_entities: List[MemoryUnit] = []
        related_events: List[MemoryUnit] = []

        # Step 2: For each evidence unit, trace COREF upward.
        for ev_unit in evidence_units[:top_k]:
            ev_uid = Uid(str(ev_unit.uid))
            coref_neighbors = self.get_explicit_neighbors(
                [ev_uid], rel_type="COREF", direction="out"
            )
            for cn in coref_neighbors:
                is_event = False
                cn_spaces = cn.metadata.get("spaces", [])
                if cn_spaces:
                    for space in cn_spaces:
                        if "event" in str(space).lower():
                            is_event = True
                            break
                if is_event:
                    if include_events and len(related_events) < top_k:
                        related_events.append(cn)
                else:
                    if include_entities and len(related_entities) < top_k:
                        related_entities.append(cn)

        return SummaryEvidenceChainResult(
            summary=summary_unit,
            evidence_units=evidence_units[:top_k],
            related_entities=related_entities[:top_k],
            related_events=related_events[:top_k],
        )

    def retrieve_entity_involvement(
        self,
        query: str,
        *,
        role: Optional[str] = None,
        top_k: int = 20,
    ) -> "EntityInvolvementResult":
        """Find all events involving an entity, with optional role filtering.

        Searches for the entity by text, then discovers events via
        ``INVOLVES`` edges. Optionally traces causal chains for each
        discovered event.

        Args:
            query: Text query to locate the entity.
            role: Optional role filter (e.g. ``"participant"``,
                ``"location"``, ``"organizer"``, ``"victim"``). If None,
                all roles are included.
            top_k: Maximum number of events to return.

        Returns:
            EntityInvolvementResult with entity, events list (each as a
            ``(MemoryUnit, role_str)`` tuple), and causal chains.
        """
        from ..retrieval.types import EntityInvolvementResult

        search_results = self.semantic_map.search_by_text(query, top_k=1, space_names=None)
        if not search_results:
            return EntityInvolvementResult()

        entity_unit, _ = search_results[0]
        entity_uid = Uid(str(entity_unit.uid))

        # Find events involving this entity via INVOLVES edges (incoming).
        event_neighbors = self.get_explicit_neighbors(
            [entity_uid], rel_type="INVOLVES", direction="in"
        )

        events: List[Tuple[MemoryUnit, str]] = []
        for ev_unit in event_neighbors:
            ev_uid = Uid(str(ev_unit.uid))
            # Attempt to read the role from the edge properties.
            role_str = "participant"
            rel_data = self._graph.get_relationship(ev_uid, entity_uid, "INVOLVES")
            if rel_data:
                role_str = str(
                    rel_data.get("role", rel_data.get("subtype", "participant"))
                )

            if role is not None and role_str != role:
                continue

            events.append((ev_unit, role_str))
            if len(events) >= top_k:
                break

        events = events[:top_k]

        # Optionally trace causal chains for each event.
        causal_chains: List[List[MemoryUnit]] = []
        for ev_unit, _ in events[:5]:
            chain_units = self.bfs_expand_units(
                seeds=[ev_unit], per_seed=3, hops=1, rel_type="CAUSES"
            )
            if chain_units:
                chain = [ev_unit] + chain_units
                causal_chains.append(chain)

        return EntityInvolvementResult(
            entity=entity_unit,
            events=events,
            causal_chains=causal_chains,
        )
