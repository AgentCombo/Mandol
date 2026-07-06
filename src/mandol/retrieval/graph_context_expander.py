from typing import Dict, List, Optional, Tuple, Any, Set, TYPE_CHECKING
import numpy as np
from collections import defaultdict, deque
import rustworkx as rx
from dataclasses import dataclass

if TYPE_CHECKING:
    from ..core.semantic_graph import SemanticGraph
    from ..core.memory_unit import MemoryUnit

from ..utils.logging_config import create_module_logger

logger = create_module_logger("graph_context_expander")


@dataclass
class GraphContext:
    """Expanded graph context returned for a set of seed units."""
    seed_units: List["MemoryUnit"]
    expanded_units: List["MemoryUnit"]
    relationships: List[Dict[str, Any]]
    paths: List[List[str]]
    context_metadata: Dict[str, Any]


@dataclass
class PathInfo:
    """Path metadata between graph nodes."""
    path_nodes: List[str]
    path_edges: List[Dict[str, Any]]
    path_score: float
    path_type: str


class GraphContextExpander:
    """Expand retrieval results with rustworkx graph neighborhoods and paths."""

    def __init__(self, semantic_graph: "SemanticGraph"):
        """Initialize the graph context expander."""
        self.semantic_graph = semantic_graph
        self.semantic_map = semantic_graph.semantic_map
        logger.info("GraphContextExpander initialized with rustworkx backend.")

    def expand_context_by_hops(
        self,
        uids: List[str],
        n_hops: int = 1,
        include_edges: bool = True,
        relation_filter: Optional[List[str]] = None,
        max_neighbors_per_hop: int = 10,
        semantic_threshold: float = 0.7,
        space_names: Optional[List[str]] = None,
    ) -> GraphContext:
        """Expand context by hops."""
        seed_units: List["MemoryUnit"] = []

        try:
            seed_indices = self._uids_to_indices(uids, require_unit=True)
            seed_index_set = set(seed_indices)
            seed_units = self._indices_to_units(seed_indices)

            if not seed_units:
                return GraphContext([], [], [], [], {"error": "no_valid_seed_units"})

            candidate_indices = self._candidate_indices_from_spaces(space_names)
            expanded_indices: Set[int] = set(seed_indices)
            current_layer_indices: Set[int] = set(seed_indices)
            all_relationships: List[Dict[str, Any]] = []
            hop_info: Dict[str, Any] = {}

            for hop in range(n_hops):
                next_layer_indices: Set[int] = set()
                hop_relationships: List[Dict[str, Any]] = []

                semantic_neighbors_by_source = self._get_semantic_neighbor_indices_batch(
                    current_layer_indices,
                    threshold=semantic_threshold,
                    max_count=max_neighbors_per_hop // 2,
                    candidate_indices=candidate_indices,
                )

                for source_idx in current_layer_indices:
                    structural_neighbors = self._get_structural_neighbor_entries(
                        source_idx,
                        relation_filter=relation_filter,
                        max_count=max_neighbors_per_hop,
                        candidate_indices=candidate_indices,
                        include_relationship=include_edges,
                    )

                    for neighbor_idx, relationship in structural_neighbors:
                        if neighbor_idx in expanded_indices:
                            continue

                        next_layer_indices.add(neighbor_idx)
                        expanded_indices.add(neighbor_idx)

                        if include_edges:
                            relationship["hop"] = hop + 1
                            relationship["traversal_source_uid"] = self._index_to_uid(source_idx)
                            relationship["traversal_target_uid"] = self._index_to_uid(neighbor_idx)
                            hop_relationships.append(relationship)

                    for neighbor_idx, score in semantic_neighbors_by_source.get(source_idx, []):
                        if neighbor_idx in expanded_indices:
                            continue

                        next_layer_indices.add(neighbor_idx)
                        expanded_indices.add(neighbor_idx)

                        if include_edges:
                            source_uid = self._index_to_uid(source_idx)
                            target_uid = self._index_to_uid(neighbor_idx)
                            hop_relationships.append({
                                "type": "SEMANTIC_SIMILAR",
                                "direction": "bidirectional",
                                "properties": {
                                    "connection_type": "semantic",
                                    "score": score,
                                },
                                "source_uid": source_uid,
                                "target_uid": target_uid,
                                "hop": hop + 1,
                                "traversal_source_uid": source_uid,
                                "traversal_target_uid": target_uid,
                            })

                all_relationships.extend(hop_relationships)
                hop_info[f"hop_{hop + 1}"] = {
                    "new_nodes": self._indices_to_uids(next_layer_indices),
                    "relationships_count": len(hop_relationships),
                }

                current_layer_indices = next_layer_indices
                if not current_layer_indices:
                    break

            expanded_units = self._indices_to_units(expanded_indices - seed_index_set)
            context_metadata = {
                "expansion_method": "n_hops",
                "n_hops": n_hops,
                "total_expanded": len(expanded_units),
                "hop_breakdown": hop_info,
                "relation_filter": relation_filter,
            }

            if space_names:
                context_metadata["space_filter"] = space_names
                context_metadata["space_restricted"] = True

            return GraphContext(
                seed_units=seed_units,
                expanded_units=expanded_units,
                relationships=all_relationships,
                paths=[],
                context_metadata=context_metadata,
            )

        except Exception as e:
            logger.error(f"Hop expansion failed: {e}")
            return GraphContext(seed_units, [], [], [], {"error": str(e)})

    def find_paths(
        self,
        source_uids: List[str],
        target_type: Optional[str] = None,
        target_uids: Optional[List[str]] = None,
        max_path_length: int = 5,
        path_limit: int = 10,
        relation_filter: Optional[List[str]] = None,
        path_scoring_method: str = "semantic",
    ) -> List[PathInfo]:
        """Find paths."""
        try:
            source_indices = self._uids_to_indices(source_uids)
            target_indices: Set[int] = set()

            if target_uids:
                target_indices.update(self._uids_to_indices(target_uids))
            elif target_type:
                target_indices.update(self._find_indices_by_type(target_type))

            if not source_indices or not target_indices:
                logger.warning("No valid source or target nodes were found.")
                return []

            relation_filter_set = set(relation_filter or [])
            all_paths: List[PathInfo] = []

            for source_idx in source_indices:
                for target_idx in target_indices:
                    try:
                        simple_paths_indices = rx.all_simple_paths(
                            self.semantic_graph.rx_graph,
                            source_idx,
                            target_idx,
                            cutoff=max_path_length,
                        )

                        path_count = 0
                        for path_indices in simple_paths_indices:
                            if path_count >= path_limit:
                                break

                            path_edges = self._path_edges_for_indices(
                                path_indices,
                                relation_filter_set=relation_filter_set,
                            )
                            if path_edges is None:
                                continue

                            path_nodes = self._indices_to_uids(path_indices)
                            if len(path_nodes) != len(path_indices):
                                continue

                            path_score = self._calculate_path_score(
                                path_nodes,
                                path_edges,
                                path_scoring_method,
                            )
                            all_paths.append(PathInfo(
                                path_nodes=path_nodes,
                                path_edges=path_edges,
                                path_score=path_score,
                                path_type="simple_path",
                            ))
                            path_count += 1

                    except Exception as e:
                        logger.warning(
                            f"Path search failed {self._index_to_uid(source_idx)} -> "
                            f"{self._index_to_uid(target_idx)}: {e}"
                        )
                        continue

            all_paths.sort(key=lambda x: x.path_score, reverse=True)
            return all_paths[:path_limit]

        except Exception as e:
            logger.error(f"Path search failed: {e}")
            return []

    def find_shortest_paths(
        self,
        source_uids: List[str],
        target_uids: List[str],
        weight_attribute: Optional[str] = None,
    ) -> List[PathInfo]:
        """Find shortest paths."""
        shortest_paths: List[PathInfo] = []

        try:
            source_indices = self._uids_to_indices(source_uids)
            target_indices = self._uids_to_indices(target_uids)

            for source_idx in source_indices:
                for target_idx in target_indices:
                    try:
                        if weight_attribute:
                            def weight_fn(edge_data):
                                return edge_data.get(weight_attribute, 1.0) if edge_data else 1.0
                        else:
                            def weight_fn(_edge_data):
                                return 1.0

                        paths_dict = rx.dijkstra_shortest_paths(
                            self.semantic_graph.rx_graph,
                            source_idx,
                            target=target_idx,
                            weight_fn=weight_fn,
                        )

                        if target_idx not in paths_dict:
                            continue

                        path_indices = paths_dict[target_idx]
                        path_edges = self._path_edges_for_indices(path_indices)
                        if path_edges is None:
                            continue

                        path_nodes = self._indices_to_uids(path_indices)
                        if len(path_nodes) != len(path_indices):
                            continue

                        path_weight = len(path_nodes) - 1
                        shortest_paths.append(PathInfo(
                            path_nodes=path_nodes,
                            path_edges=path_edges,
                            path_score=1.0 / (1.0 + path_weight),
                            path_type="shortest_path",
                        ))

                    except Exception as e:
                        logger.warning(
                            f"Shortest-path calculation failed {self._index_to_uid(source_idx)} -> "
                            f"{self._index_to_uid(target_idx)}: {e}"
                        )
                        continue

            return shortest_paths

        except Exception as e:
            logger.error(f"Shortest-path search failed: {e}")
            return []

    def find_semantic_paths(
        self,
        source_uids: List[str],
        query: str,
        max_path_length: int = 4,
        path_limit: int = 5,
    ) -> List[PathInfo]:
        """Find semantic paths."""
        try:
            source_indices = self._uids_to_indices(source_uids)
            target_indices = self._get_semantic_target_indices(query, top_k=50)

            if not source_indices or not target_indices:
                return []

            semantic_paths: List[PathInfo] = []

            for source_idx in source_indices:
                for target_idx in target_indices:
                    if source_idx == target_idx:
                        continue

                    paths = self._semantic_bfs_paths_by_indices(
                        source_idx,
                        target_idx,
                        semantic_target_indices=target_indices,
                        max_length=max_path_length,
                    )

                    for path_nodes, path_edges, semantic_score in paths:
                        semantic_paths.append(PathInfo(
                            path_nodes=path_nodes,
                            path_edges=path_edges,
                            path_score=semantic_score,
                            path_type="semantic_path",
                        ))

                    if len(semantic_paths) >= path_limit * 3:
                        break

            semantic_paths.sort(key=lambda x: x.path_score, reverse=True)
            return semantic_paths[:path_limit]

        except Exception as e:
            logger.error(f"Semantic path search failed: {e}")
            return []

    def get_relation_summary(self, uids: List[str]) -> Dict[str, Any]:
        """Return relation summary."""
        try:
            relation_stats = defaultdict(int)
            edge_details: List[Dict[str, Any]] = []

            for idx in self._uids_to_indices(uids):
                uid = self._index_to_uid(idx)
                if uid is None:
                    continue

                for edge_source_idx, edge_target_idx, edge_data in self.semantic_graph.rx_graph.out_edges(idx):
                    relation_type = self._edge_type(edge_data)
                    relation_stats[relation_type] += 1
                    edge_details.append({
                        "source": self._index_to_uid(edge_source_idx),
                        "target": self._index_to_uid(edge_target_idx),
                        "type": relation_type,
                        "direction": "outgoing",
                    })

                for edge_source_idx, edge_target_idx, edge_data in self.semantic_graph.rx_graph.in_edges(idx):
                    relation_type = self._edge_type(edge_data)
                    relation_stats[f"{relation_type}_incoming"] += 1
                    edge_details.append({
                        "source": self._index_to_uid(edge_source_idx),
                        "target": self._index_to_uid(edge_target_idx),
                        "type": relation_type,
                        "direction": "incoming",
                    })

            return {
                "relation_counts": dict(relation_stats),
                "total_edges": len(edge_details),
                "unique_relations": len(set(detail["type"] for detail in edge_details)),
                "edge_details": edge_details,
            }

        except Exception as e:
            logger.error(f"Relationship-summary generation failed: {e}")
            return {}

    def enrich_retrieval_results(
        self,
        retrieval_results: List[Tuple["MemoryUnit", float]],
        expansion_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Enrich retrieval results."""
        if not retrieval_results:
            return {"original_results": [], "enriched_context": None}

        config = {
            "expand_hops": 1,
            "max_neighbors": 5,
            "find_inter_paths": True,
            "semantic_threshold": 0.7,
            "include_relation_summary": True,
        }
        if expansion_config:
            config.update(expansion_config)

        try:
            seed_uids = [unit.uid for unit, _ in retrieval_results]
            enriched_results: Dict[str, Any] = {
                "original_results": retrieval_results,
                "seed_count": len(seed_uids),
                "enrichment_config": config,
            }

            if config["expand_hops"] > 0:
                graph_context = self.expand_context_by_hops(
                    uids=seed_uids,
                    n_hops=config["expand_hops"],
                    max_neighbors_per_hop=config["max_neighbors"],
                    semantic_threshold=config["semantic_threshold"],
                )
                enriched_results["expanded_context"] = graph_context

            if config["find_inter_paths"] and len(seed_uids) > 1:
                inter_paths: List[PathInfo] = []
                for i, source_uid in enumerate(seed_uids):
                    for target_uid in seed_uids[i + 1:]:
                        inter_paths.extend(self.find_shortest_paths([source_uid], [target_uid]))
                enriched_results["inter_result_paths"] = inter_paths

            if config["include_relation_summary"]:
                enriched_results["relation_summary"] = self.get_relation_summary(seed_uids)

            subgraph_indices = set(self._uids_to_indices(seed_uids))
            if "expanded_context" in enriched_results:
                expanded_uids = [unit.uid for unit in enriched_results["expanded_context"].expanded_units]
                subgraph_indices.update(self._uids_to_indices(expanded_uids))

            subgraph_edge_count = 0
            for idx in subgraph_indices:
                for _edge_source_idx, edge_target_idx, _edge_data in self.semantic_graph.rx_graph.out_edges(idx):
                    if edge_target_idx in subgraph_indices:
                        subgraph_edge_count += 1

            subgraph_node_count = len(subgraph_indices)
            if subgraph_node_count > 1:
                max_edges = subgraph_node_count * (subgraph_node_count - 1)
                density = subgraph_edge_count / max_edges if max_edges > 0 else 0.0
            else:
                density = 0.0

            enriched_results["subgraph_stats"] = {
                "nodes": subgraph_node_count,
                "edges": subgraph_edge_count,
                "density": density,
            }

            logger.info(
                f"Retrieval-result enrichment completed: original={len(retrieval_results)}, "
                f"expanded_nodes={len(enriched_results.get('expanded_context', GraphContext([], [], [], [], {})).expanded_units)}"
            )
            return enriched_results

        except Exception as e:
            logger.error(f"Retrieval-result enrichment failed: {e}")
            return {
                "original_results": retrieval_results,
                "enriched_context": None,
                "error": str(e),
            }

    
    

    def _uid_to_index(self, uid: str) -> Optional[int]:
        return self.semantic_graph._uid_to_index.get(uid)

    def _index_to_uid(self, idx: int) -> Optional[str]:
        return self.semantic_graph._index_to_uid.get(idx)

    def _uids_to_indices(self, uids: List[str], require_unit: bool = False) -> List[int]:
        indices: List[int] = []
        for uid in uids:
            idx = self._uid_to_index(uid)
            if idx is None:
                continue
            if require_unit and self.semantic_graph.get_unit(uid) is None:
                continue
            indices.append(idx)
        return indices

    def _indices_to_uids(self, indices) -> List[str]:
        uids: List[str] = []
        for idx in indices:
            uid = self._index_to_uid(idx)
            if uid is not None:
                uids.append(uid)
        return uids

    def _indices_to_units(self, indices) -> List["MemoryUnit"]:
        units: List["MemoryUnit"] = []
        for uid in self._indices_to_uids(indices):
            unit = self.semantic_graph.get_unit(uid)
            if unit is not None:
                units.append(unit)
        return units

    def _candidate_indices_from_spaces(self, space_names: Optional[List[str]]) -> Optional[Set[int]]:
        if not space_names:
            return None

        try:
            candidate_units = self.semantic_graph.get_units_in_memory_space(
                space_names,
                recursive=True,
            )
            candidate_indices = {
                idx
                for idx in (self._uid_to_index(unit.uid) for unit in candidate_units)
                if idx is not None
            }
            logger.debug(f"Graph expansion candidate scope: {len(candidate_indices)} nodes.")
            return candidate_indices
        except Exception as e:
            logger.warning(f"Failed to resolve candidate nodes from spaces: {e}")
            return None

    def _edge_type(self, edge_data: Optional[Dict[str, Any]]) -> str:
        if not isinstance(edge_data, dict) or not edge_data:
            return "UNKNOWN"
        return edge_data.get("type", "UNKNOWN")

    def _edge_properties(self, edge_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(edge_data, dict) or not edge_data:
            return {}
        return {key: value for key, value in edge_data.items() if key != "type"}

    def _relationship_from_edge(
        self,
        edge_source_idx: int,
        edge_target_idx: int,
        edge_data: Optional[Dict[str, Any]],
        direction: str,
    ) -> Dict[str, Any]:
        return {
            "type": self._edge_type(edge_data),
            "direction": direction,
            "properties": self._edge_properties(edge_data),
            "source_uid": self._index_to_uid(edge_source_idx),
            "target_uid": self._index_to_uid(edge_target_idx),
        }

    def _get_structural_neighbor_entries(
        self,
        source_idx: int,
        relation_filter: Optional[List[str]] = None,
        max_count: int = 10,
        candidate_indices: Optional[Set[int]] = None,
        include_relationship: bool = True,
    ) -> List[Tuple[int, Dict[str, Any]]]:
        """Get structural neighbor entries."""
        if max_count <= 0:
            return []

        entries: List[Tuple[int, Dict[str, Any]]] = []
        seen_neighbors: Set[int] = set()
        relation_contains = set(relation_filter).__contains__ if relation_filter else None
        candidate_contains = candidate_indices.__contains__ if candidate_indices is not None else None
        seen_contains = seen_neighbors.__contains__
        seen_add = seen_neighbors.add
        append_entry = entries.append
        rx_graph = self.semantic_graph.rx_graph
        index_to_uid = self.semantic_graph._index_to_uid

        def consume_edges(edge_iterable, direction: str, incoming: bool) -> bool:
            for edge_source_idx, edge_target_idx, edge_data in edge_iterable:
                neighbor_idx = edge_source_idx if incoming else edge_target_idx

                if candidate_contains is not None and not candidate_contains(neighbor_idx):
                    continue
                if seen_contains(neighbor_idx):
                    continue

                edge_type = None
                if relation_contains is not None:
                    edge_type = self._edge_type(edge_data)
                    if not relation_contains(edge_type):
                        continue

                if include_relationship:
                    if edge_type is None:
                        edge_type = self._edge_type(edge_data)
                    properties = self._edge_properties(edge_data)
                    relationship = {
                        "type": edge_type,
                        "direction": direction,
                        "properties": properties,
                        "source_uid": index_to_uid.get(edge_source_idx),
                        "target_uid": index_to_uid.get(edge_target_idx),
                    }
                else:
                    relationship = {}

                append_entry((neighbor_idx, relationship))
                seen_add(neighbor_idx)
                if len(entries) >= max_count:
                    return True
            return False

        try:
            if consume_edges(rx_graph.out_edges(source_idx), "outgoing", False):
                return entries
            consume_edges(rx_graph.in_edges(source_idx), "incoming", True)
        except Exception as exc:
            logger.warning(f"Structural-neighbor traversal failed for idx={source_idx}: {exc}")

        return entries

    def _candidate_faiss_ids_from_graph_indices(self, candidate_indices: Optional[Set[int]]) -> Optional[np.ndarray]:
        if candidate_indices is None:
            return None
        if not candidate_indices:
            return np.empty((0,), dtype=np.int64)

        uid_to_int_id = self.semantic_map._get_uid_to_int_id_map()
        index_to_uid = self.semantic_graph._index_to_uid
        faiss_ids = [
            int_id
            for idx in candidate_indices
            for uid in (index_to_uid.get(idx),)
            for int_id in (uid_to_int_id.get(uid),)
            if int_id is not None
        ]
        if not faiss_ids:
            return np.empty((0,), dtype=np.int64)
        return np.ascontiguousarray(np.asarray(faiss_ids, dtype=np.int64))

    def _build_faiss_search_params(self, faiss, candidate_faiss_ids: Optional[np.ndarray]):
        if candidate_faiss_ids is None:
            return None, None, None
        if candidate_faiss_ids.size == 0:
            return None, None, candidate_faiss_ids

        candidate_faiss_ids = np.ascontiguousarray(candidate_faiss_ids, dtype=np.int64)
        selector = None
        selector_errors = []

        if hasattr(faiss, "IDSelectorBatch"):
            try:
                selector = faiss.IDSelectorBatch(candidate_faiss_ids)
            except Exception as exc:
                selector_errors.append(f"IDSelectorBatch(array): {exc}")
                try:
                    selector = faiss.IDSelectorBatch(
                        int(candidate_faiss_ids.size),
                        faiss.swig_ptr(candidate_faiss_ids),
                    )
                except Exception as exc2:
                    selector_errors.append(f"IDSelectorBatch(ptr): {exc2}")

        if selector is None and hasattr(faiss, "IDSelectorArray"):
            try:
                selector = faiss.IDSelectorArray(
                    int(candidate_faiss_ids.size),
                    faiss.swig_ptr(candidate_faiss_ids),
                )
            except Exception as exc:
                selector_errors.append(f"IDSelectorArray(ptr): {exc}")
                try:
                    selector = faiss.IDSelectorArray(candidate_faiss_ids)
                except Exception as exc2:
                    selector_errors.append(f"IDSelectorArray(array): {exc2}")

        if selector is None:
            logger.warning("FAISS IDSelector creation failed; using exact candidate-space fallback: %s", "; ".join(selector_errors))
            return None, None, candidate_faiss_ids

        try:
            faiss_index_type = str(getattr(self.semantic_map, "faiss_index_type", ""))
            search_params = faiss.SearchParametersIVF() if "IVF" in faiss_index_type else faiss.SearchParameters()
            search_params.sel = selector
            return search_params, selector, candidate_faiss_ids
        except Exception as exc:
            logger.warning(f"FAISS SearchParameters creation failed; using exact candidate-space fallback: {exc}")
            return None, None, candidate_faiss_ids

    def _semantic_neighbors_by_candidate_matrix(
        self,
        query_indices: List[int],
        query_matrix: np.ndarray,
        candidate_indices: Set[int],
        threshold: float,
        max_count: int,
    ) -> Dict[int, List[Tuple[int, float]]]:
        candidate_graph_indices: List[int] = []
        candidate_embeddings: List[np.ndarray] = []

        for candidate_idx in candidate_indices:
            uid = self._index_to_uid(candidate_idx)
            if uid is None:
                continue
            unit = self.semantic_graph.get_unit(uid)
            if unit is None or unit.embedding is None:
                continue
            embedding = np.asarray(unit.embedding, dtype=np.float32)
            if embedding.ndim != 1 or embedding.shape[0] != self.semantic_map.embedding_dim:
                continue
            candidate_graph_indices.append(candidate_idx)
            candidate_embeddings.append(embedding)

        neighbors_by_source: Dict[int, List[Tuple[int, float]]] = {idx: [] for idx in query_indices}
        if not candidate_embeddings:
            return neighbors_by_source

        try:
            candidate_matrix = np.stack(candidate_embeddings, axis=0).astype(np.float32, copy=False)
            candidate_norms = np.linalg.norm(candidate_matrix, axis=1, keepdims=True)
            valid_mask = candidate_norms[:, 0] > 1e-6
            if not np.any(valid_mask):
                return neighbors_by_source
            candidate_matrix = candidate_matrix[valid_mask]
            candidate_graph_indices = [idx for idx, valid in zip(candidate_graph_indices, valid_mask.tolist()) if valid]
            candidate_matrix = candidate_matrix / np.maximum(candidate_norms[valid_mask], 1e-12)

            top_window = min(candidate_matrix.shape[0], max_count + 1)
            for row_idx, source_idx in enumerate(query_indices):
                scores = candidate_matrix @ query_matrix[row_idx]
                if top_window < scores.shape[0]:
                    top_indices = np.argpartition(scores, -top_window)[-top_window:]
                    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
                else:
                    top_indices = np.argsort(scores)[::-1]

                source_neighbors = neighbors_by_source[source_idx]
                seen_neighbors: Set[int] = set()
                for candidate_position in top_indices:
                    neighbor_idx = candidate_graph_indices[int(candidate_position)]
                    if neighbor_idx == source_idx or neighbor_idx in seen_neighbors:
                        continue
                    normalized_score = (float(scores[int(candidate_position)]) + 1.0) / 2.0
                    normalized_score = max(0.0, min(1.0, normalized_score))
                    if normalized_score < threshold:
                        continue
                    source_neighbors.append((neighbor_idx, normalized_score))
                    seen_neighbors.add(neighbor_idx)
                    if len(source_neighbors) >= max_count:
                        break
            return neighbors_by_source
        except Exception as exc:
            logger.warning(f"Exact semantic fallback for candidate spaces failed: {exc}")
            return neighbors_by_source

    def _get_semantic_neighbor_indices_batch(
        self,
        source_indices: Set[int],
        threshold: float = 0.7,
        max_count: int = 5,
        candidate_indices: Optional[Set[int]] = None,
    ) -> Dict[int, List[Tuple[int, float]]]:
        """Get semantic neighbor indices batch."""
        if not source_indices or max_count <= 0 or threshold > 1.0:
            return {}

        faiss_index = getattr(self.semantic_map, "faiss_index", None)
        if faiss_index is None or getattr(faiss_index, "ntotal", 0) <= 0:
            return {}

        try:
            import faiss
        except Exception as e:
            logger.warning(f"FAISS is unavailable; skipping batched semantic-neighbor retrieval: {e}")
            return {}

        query_indices: List[int] = []
        query_embeddings: List[np.ndarray] = []

        for source_idx in source_indices:
            uid = self._index_to_uid(source_idx)
            if uid is None:
                continue
            unit = self.semantic_graph.get_unit(uid)
            if unit is None or unit.embedding is None:
                continue

            embedding = np.asarray(unit.embedding, dtype=np.float32)
            if embedding.ndim != 1 or embedding.shape[0] != self.semantic_map.embedding_dim:
                continue

            query_indices.append(source_idx)
            query_embeddings.append(embedding)

        if not query_embeddings:
            return {}

        try:
            query_matrix = np.stack(query_embeddings, axis=0).astype(np.float32, copy=False)
            zero_mask = np.linalg.norm(query_matrix, axis=1) < 1e-6
            faiss.normalize_L2(query_matrix)
            if zero_mask.any():
                query_matrix[zero_mask] = 0.0

            ntotal = getattr(faiss_index, "ntotal", 0)
            candidate_faiss_ids = self._candidate_faiss_ids_from_graph_indices(candidate_indices)
            if candidate_indices is not None:
                if candidate_faiss_ids is None or candidate_faiss_ids.size == 0:
                    return {idx: [] for idx in query_indices}
                search_k = min(int(candidate_faiss_ids.size), max_count + 1)
            else:
                search_k = min(ntotal, max_count + 1)

            search_params, _selector, _selector_ids = self._build_faiss_search_params(faiss, candidate_faiss_ids)
            if candidate_indices is not None and search_params is None:
                return self._semantic_neighbors_by_candidate_matrix(
                    query_indices,
                    query_matrix,
                    candidate_indices,
                    threshold,
                    max_count,
                )
            try:
                if search_params is not None:
                    similarities, faiss_ids = faiss_index.search(query_matrix, search_k, params=search_params)
                else:
                    similarities, faiss_ids = faiss_index.search(query_matrix, search_k)
            except Exception as search_exc:
                if candidate_indices is not None:
                    logger.warning(f"FAISS selector pre-filtering failed; using exact candidate-space fallback: {search_exc}")
                    return self._semantic_neighbors_by_candidate_matrix(
                        query_indices,
                        query_matrix,
                        candidate_indices,
                        threshold,
                        max_count,
                    )
                raise

            int_id_to_uid = self.semantic_map._get_int_id_to_uid_map()
            uid_to_index = self.semantic_graph._uid_to_index
            neighbors_by_source: Dict[int, List[Tuple[int, float]]] = {idx: [] for idx in query_indices}

            for row_idx, source_idx in enumerate(query_indices):
                seen_neighbors: Set[int] = set()

                for raw_score, internal_id in zip(similarities[row_idx], faiss_ids[row_idx]):
                    if int(internal_id) == -1:
                        continue

                    neighbor_uid = int_id_to_uid.get(int(internal_id))
                    if neighbor_uid is None:
                        continue

                    neighbor_idx = uid_to_index.get(neighbor_uid)
                    if neighbor_idx is None or neighbor_idx == source_idx:
                        continue
                    if neighbor_idx in seen_neighbors:
                        continue

                    normalized_score = (float(raw_score) + 1.0) / 2.0
                    normalized_score = max(0.0, min(1.0, normalized_score))
                    if normalized_score < threshold:
                        continue

                    neighbors_by_source[source_idx].append((neighbor_idx, normalized_score))
                    seen_neighbors.add(neighbor_idx)
                    if len(neighbors_by_source[source_idx]) >= max_count:
                        break

            return neighbors_by_source

        except Exception as e:
            logger.warning(f"Batched semantic-neighbor retrieval failed: {e}")
            return {}

    def _edge_data_between_indices(self, source_idx: int, target_idx: int) -> List[Dict[str, Any]]:
        edge_data_list: List[Dict[str, Any]] = []
        for _edge_source_idx, edge_target_idx, edge_data in self.semantic_graph.rx_graph.out_edges(source_idx):
            if edge_target_idx == target_idx:
                edge_data_list.append(edge_data or {})
        return edge_data_list

    def _path_edges_for_indices(
        self,
        path_indices: List[int],
        relation_filter_set: Optional[Set[str]] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        relation_filter_set = relation_filter_set or set()
        path_edges: List[Dict[str, Any]] = []

        for i in range(len(path_indices) - 1):
            edge_data_list = self._edge_data_between_indices(path_indices[i], path_indices[i + 1])
            if not edge_data_list:
                return None

            selected_edge = None
            for edge_data in edge_data_list:
                if not relation_filter_set or self._edge_type(edge_data) in relation_filter_set:
                    selected_edge = edge_data
                    break

            if selected_edge is None:
                return None
            path_edges.append(selected_edge)

        return path_edges

    def _get_semantic_target_indices(self, query: str, top_k: int = 50) -> Set[int]:
        """Get semantic target indices."""
        raw_results = []

        try:
            multi_retriever = self.semantic_graph.get_multi_retriever()
            if multi_retriever is not None and hasattr(multi_retriever, "search_hybrid"):
                raw_results = multi_retriever.search_hybrid(query, top_k=top_k)
            elif multi_retriever is not None and hasattr(multi_retriever, "smart_search"):
                raw_results = multi_retriever.smart_search(query, top_k=top_k)
        except Exception as e:
            logger.debug(f"Multi-backend retrieval failed; falling back to SemanticMap text-vector retrieval: {e}")
            raw_results = []

        if not raw_results:
            try:
                raw_results = self.semantic_map.search_similarity_by_text(query, k=top_k)
            except Exception as e:
                logger.debug(f"Text-vector retrieval failed: {e}")
                raw_results = []

        return self._retrieval_results_to_indices(raw_results)

    def _retrieval_results_to_indices(self, raw_results) -> Set[int]:
        if isinstance(raw_results, dict):
            raw_results = raw_results.get("results", []) or raw_results.get("data", []) or []

        indices: Set[int] = set()
        for item in raw_results:
            uid = None

            if isinstance(item, dict):
                unit = item.get("unit") or item.get("memory_unit")
                uid = item.get("uid") or getattr(unit, "uid", None)
            elif isinstance(item, (tuple, list)) and item:
                first = item[0]
                uid = first if isinstance(first, str) else getattr(first, "uid", None)
            elif hasattr(item, "unit"):
                uid = getattr(item.unit, "uid", None)
            elif hasattr(item, "uid"):
                uid = item.uid

            if uid is None:
                continue

            idx = self._uid_to_index(uid)
            if idx is not None:
                indices.add(idx)

        return indices

    def _get_structural_neighbors(
        self,
        uid: str,
        relation_filter: Optional[List[str]] = None,
        max_count: int = 10,
        candidate_node_uids: Optional[Set[str]] = None,
    ) -> List[str]:
        """Get structural neighbors."""
        source_idx = self._uid_to_index(uid)
        if source_idx is None:
            return []

        candidate_indices = None
        if candidate_node_uids is not None:
            candidate_indices = {
                idx
                for idx in (self._uid_to_index(candidate_uid) for candidate_uid in candidate_node_uids)
                if idx is not None
            }

        entries = self._get_structural_neighbor_entries(
            source_idx,
            relation_filter=relation_filter,
            max_count=max_count,
            candidate_indices=candidate_indices,
            include_relationship=False,
        )
        return self._indices_to_uids([neighbor_idx for neighbor_idx, _relationship in entries])

    def _get_semantic_neighbors(
        self,
        uid: str,
        threshold: float = 0.7,
        max_count: int = 5,
        candidate_node_uids: Optional[Set[str]] = None,
    ) -> List[str]:
        """Get semantic neighbors."""
        source_idx = self._uid_to_index(uid)
        if source_idx is None:
            return []

        candidate_indices = None
        if candidate_node_uids is not None:
            candidate_indices = {
                idx
                for idx in (self._uid_to_index(candidate_uid) for candidate_uid in candidate_node_uids)
                if idx is not None
            }

        neighbors_by_source = self._get_semantic_neighbor_indices_batch(
            {source_idx},
            threshold=threshold,
            max_count=max_count,
            candidate_indices=candidate_indices,
        )
        return self._indices_to_uids([neighbor_idx for neighbor_idx, _score in neighbors_by_source.get(source_idx, [])])

    def _get_edge_info(self, source_uid: str, target_uid: str) -> Optional[Dict[str, Any]]:
        """Get edge info."""
        source_idx = self._uid_to_index(source_uid)
        target_idx = self._uid_to_index(target_uid)
        if source_idx is None or target_idx is None:
            return None

        for edge_source_idx, edge_target_idx, edge_data in self.semantic_graph.rx_graph.out_edges(source_idx):
            if edge_target_idx == target_idx:
                return self._relationship_from_edge(edge_source_idx, edge_target_idx, edge_data, "outgoing")

        for edge_source_idx, edge_target_idx, edge_data in self.semantic_graph.rx_graph.in_edges(source_idx):
            if edge_source_idx == target_idx:
                return self._relationship_from_edge(edge_source_idx, edge_target_idx, edge_data, "incoming")

        return {
            "type": "SEMANTIC_SIMILAR",
            "direction": "bidirectional",
            "properties": {"connection_type": "semantic"},
            "source_uid": source_uid,
            "target_uid": target_uid,
        }

    def _find_indices_by_type(self, node_type: str) -> List[int]:
        matching_indices: List[int] = []
        lowered_type = node_type.lower()

        for idx, uid in self.semantic_graph._index_to_uid.items():
            try:
                node_data = self.semantic_graph.rx_graph.get_node_data(idx)
            except Exception:
                node_data = None

            if node_data and node_data.get("type") == node_type:
                matching_indices.append(idx)
                continue

            unit = self.semantic_graph.get_unit(uid)
            if unit and unit.raw_data:
                unit_type = unit.raw_data.get("type")
                if unit_type == node_type:
                    matching_indices.append(idx)
                    continue

                content = str(unit.raw_data).lower()
                if lowered_type in content:
                    matching_indices.append(idx)

        return matching_indices

    def _find_nodes_by_type(self, node_type: str) -> List[str]:
        """Find nodes by type."""
        return self._indices_to_uids(self._find_indices_by_type(node_type))

    def _calculate_path_score(
        self,
        path_nodes: List[str],
        path_edges: List[Dict[str, Any]],
        scoring_method: str,
    ) -> float:
        """Calculate path score."""
        if scoring_method == "length":
            return 1.0 / len(path_nodes)

        if scoring_method == "semantic":
            node_scores = []
            for node_uid in path_nodes:
                unit = self.semantic_graph.get_unit(node_uid)
                if unit and unit.embedding is not None:
                    node_scores.append(0.5)
            return sum(node_scores) / len(node_scores) if node_scores else 0.0

        if scoring_method == "relation_weight":
            edge_weights = [edge_data.get("weight", 1.0) for edge_data in path_edges]
            return sum(edge_weights) / len(edge_weights) if edge_weights else 0.0

        return 1.0 / len(path_nodes)

    def _semantic_bfs_paths(
        self,
        source_uid: str,
        target_uid: str,
        query: str,
        max_length: int,
    ) -> List[Tuple[List[str], List[Dict], float]]:
        """Semantic bfs paths."""
        source_idx = self._uid_to_index(source_uid)
        target_idx = self._uid_to_index(target_uid)
        if source_idx is None or target_idx is None or source_idx == target_idx:
            return []

        semantic_target_indices = self._get_semantic_target_indices(query, top_k=50)
        return self._semantic_bfs_paths_by_indices(
            source_idx,
            target_idx,
            semantic_target_indices=semantic_target_indices,
            max_length=max_length,
        )

    def _semantic_bfs_paths_by_indices(
        self,
        source_idx: int,
        target_idx: int,
        semantic_target_indices: Set[int],
        max_length: int,
    ) -> List[Tuple[List[str], List[Dict], float]]:
        if source_idx == target_idx:
            return []

        queue = deque([(source_idx, [source_idx], [], 0.0)])
        found_paths: List[Tuple[List[str], List[Dict], float]] = []

        while queue and len(found_paths) < 3:
            current_idx, path_indices, path_edges, cumulative_score = queue.popleft()
            if len(path_indices) >= max_length:
                continue

            for _edge_source_idx, neighbor_idx, edge_data in self.semantic_graph.rx_graph.out_edges(current_idx):
                if neighbor_idx in path_indices:
                    continue

                neighbor_score = 1.0 if neighbor_idx in semantic_target_indices else 0.0
                new_path_indices = path_indices + [neighbor_idx]
                new_path_edges = path_edges + [edge_data or {}]
                new_cumulative_score = cumulative_score + neighbor_score

                if neighbor_idx == target_idx:
                    path_nodes = self._indices_to_uids(new_path_indices)
                    if len(path_nodes) != len(new_path_indices):
                        continue
                    final_score = new_cumulative_score / max(1, len(new_path_indices) - 1)
                    found_paths.append((path_nodes, new_path_edges, final_score))
                else:
                    queue.append((neighbor_idx, new_path_indices, new_path_edges, new_cumulative_score))

        return found_paths
