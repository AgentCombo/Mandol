from typing import Dict, List, Optional, Tuple, Any, Set, TYPE_CHECKING
from collections import defaultdict
from .retrieval_interface import BaseRetriever, RetrievalMethod, RetrievalResult
if TYPE_CHECKING:
    from ..core.semantic_graph import SemanticGraph
from ..utils.logging_config import create_module_logger

logger = create_module_logger("graph_retriever")


class GraphRetriever(BaseRetriever):
    """Graph traversal retriever for SemanticGraph relationships and paths."""
    
    def __init__(self, semantic_graph: "SemanticGraph"):
        """Initialize graph retrieval over a SemanticGraph."""
        self.semantic_graph = semantic_graph
        self.semantic_map = semantic_graph.semantic_map

        try:
            from .graph_context_expander import GraphContextExpander
            self.expander = GraphContextExpander(semantic_graph)
        except Exception as e:
            self.expander = None
            logger.warning(f"GraphContextExpander initialization failed; path search will return empty results: {e}")
        
        self._relation_cache = {}
        self._path_cache = {}
        self._neighbor_cache = {}
        
        logger.info("GraphRetriever initialized with rustworkx backend.")
    
    def get_method_type(self) -> RetrievalMethod:
        return RetrievalMethod.GRAPH_TRAVERSAL
    
    def search(self, query: str, top_k: int = 10, 
          space_names: Optional[List[str]] = None,
          **kwargs) -> List[RetrievalResult]:
        """Search graph nodes, edges, paths, or hybrid graph evidence."""
        try:
            if space_names:
                kwargs['space_names'] = space_names
                logger.debug(f"Graph retrieval restricted to spaces: {space_names}")
            
            search_mode = kwargs.get('search_mode', 'hybrid')
            
            if search_mode == 'node':
                return self._node_search(query, top_k, **kwargs)
            elif search_mode == 'edge':
                return self._edge_search(query, top_k, **kwargs)
            elif search_mode == 'path':
                return self._path_search(query, top_k, **kwargs)
            elif search_mode == 'hybrid':
                return self._hybrid_search(query, top_k, **kwargs)
            else:
                logger.warning(f"Unknown graph search mode {search_mode}; using hybrid search.")
                return self._hybrid_search(query, top_k, **kwargs)
                
        except Exception as e:
            logger.error(f"Graph retrieval failed: {e}")
            import traceback
            logger.debug(f"Detailed error: {traceback.format_exc()}")
            return []
    
    def _node_search(self, query: str, top_k: int, **kwargs) -> List[RetrievalResult]:
        """Node search."""
        node_search_type = kwargs.get('node_search_type', 'hybrid')
        
        if node_search_type == 'semantic':
            return self._semantic_node_search(query, top_k, **kwargs)
        elif node_search_type == 'fulltext':
            return self._fulltext_node_search(query, top_k, **kwargs)
        elif node_search_type == 'hybrid':
            return self._hybrid_node_search(query, top_k, **kwargs)
        else:
            return self._hybrid_node_search(query, top_k, **kwargs)
    
    def _edge_search(self, query: str, top_k: int, **kwargs) -> List[RetrievalResult]:
        """Edge search."""
        try:
            space_names = kwargs.get('space_names')
            
            candidate_node_uids = None
            if space_names:
                try:
                    candidate_units = self.semantic_graph.get_units_in_memory_space(
                        space_names, recursive=True
                    )
                    candidate_node_uids = set(unit.uid for unit in candidate_units)
                    logger.debug(f"Edge-search candidate scope: {len(candidate_node_uids)} nodes.")
                except Exception as e:
                    logger.warning(f"Failed to resolve candidate nodes from spaces: {e}")
                    candidate_node_uids = None
            
            relation_types = kwargs.get('relation_types')
            source_nodes = kwargs.get('source_nodes')
            target_nodes = kwargs.get('target_nodes')
            edge_direction = kwargs.get('edge_direction', 'both')
            
            if candidate_node_uids:
                if source_nodes:
                    source_nodes = [uid for uid in source_nodes if uid in candidate_node_uids]
                if target_nodes:
                    target_nodes = [uid for uid in target_nodes if uid in candidate_node_uids]
            
            edge_results = self._search_edges_by_query(
                query, relation_types, source_nodes, target_nodes, 
                edge_direction, candidate_node_uids
            )
            
            retrieval_results = []
            for edge_info in edge_results[:top_k]:
                source_uid, target_uid, edge_data, edge_score = edge_info
                
                nodes_to_return = self._determine_nodes_from_edge(
                    source_uid, target_uid, edge_data, query, kwargs
                )
                
                for node_uid, node_relevance in nodes_to_return:
                    unit = self.semantic_graph.get_unit(node_uid)
                    if unit:
                        method_details = {
                            "search_mode": "edge",
                            "edge_context": {
                                "source_uid": source_uid,
                                "target_uid": target_uid,
                                "relation_type": edge_data.get('type', 'UNKNOWN'),
                                "relation_properties": {k: v for k, v in edge_data.items() if k != 'type'},
                                "edge_score": float(edge_score),
                                "node_role": "source" if node_uid == source_uid else "target",
                                "query_relevance": float(node_relevance)
                            },
                            "graph_traversal_info": {
                                "traversal_type": "edge_search",
                                "query": query,
                                "relation_filter": relation_types
                            }
                        }
                        
                        if space_names:
                            method_details["search_scope"] = space_names
                            method_details["scope_type"] = "memory_space"
                        
                        final_score = edge_score * 0.7 + node_relevance * 0.3
                        
                        result = RetrievalResult(
                            unit=unit,
                            score=float(final_score),
                            method=self.get_method_type(),
                            method_details=method_details
                        )
                        retrieval_results.append(result)
            
            retrieval_results = self._deduplicate_and_sort_results(retrieval_results)
            
            if space_names:
                logger.debug(f"Edge search completed for spaces {space_names}: {len(retrieval_results[:top_k])} results.")
            
            return retrieval_results[:top_k]
            
        except Exception as e:
            logger.error(f"Edge search failed: {e}")
            return []
    
    def _path_search(self, query: str, top_k: int, **kwargs) -> List[RetrievalResult]:
        """Path search."""
        try:
            if top_k <= 0:
                return []

            query_text = str(query)
            space_names = kwargs.get('space_names')
            
            candidate_node_uids = None
            if space_names:
                try:
                    candidate_units = self.semantic_graph.get_units_in_memory_space(
                        space_names, recursive=True
                    )
                    candidate_node_uids = set(unit.uid for unit in candidate_units)
                    logger.debug(f"Path-search candidate scope: {len(candidate_node_uids)} nodes.")
                except Exception as e:
                    logger.warning(f"Failed to resolve candidate nodes from spaces: {e}")
                    candidate_node_uids = None

            if candidate_node_uids is not None and not candidate_node_uids:
                return []
            
            start_nodes = kwargs.get('start_nodes')
            end_nodes = kwargs.get('end_nodes')
            max_path_length = kwargs.get('max_path_length', 5)
            path_relation_types = kwargs.get('path_relation_types')
            path_limit = kwargs.get('path_limit')
            if path_limit is None:
                path_limit = max(top_k * 3, 10)
            
            if candidate_node_uids:
                if start_nodes:
                    start_nodes = [uid for uid in start_nodes if uid in candidate_node_uids]
                if end_nodes:
                    end_nodes = [uid for uid in end_nodes if uid in candidate_node_uids]
            
            path_results = self._search_paths_by_query(
                query_text, start_nodes, end_nodes, max_path_length,
                path_relation_types, candidate_node_uids, path_limit=path_limit
            )
            
            retrieval_results = []
            unit_cache = {}
            get_unit = self.semantic_graph.get_unit
            append_result = retrieval_results.append
            for path_info in path_results[:top_k]:
                path_nodes, path_edges, path_score = path_info
                path_length = len(path_nodes)
                if path_length == 0:
                    continue
                
                for i, node_uid in enumerate(path_nodes):
                    if node_uid not in unit_cache:
                        unit_cache[node_uid] = get_unit(node_uid)
                    unit = unit_cache[node_uid]
                    if unit:
                        method_details = {
                            "search_mode": "path",
                            "path_context": {
                                "full_path": path_nodes,
                                "path_edges": path_edges,
                                "node_position": i,
                                "path_length": path_length,
                                "path_score": float(path_score),
                                "node_role": self._determine_node_role_in_path(i, path_length)
                            },
                            "graph_traversal_info": {
                                "traversal_type": "path_search",
                                "query": query_text,
                                "max_path_length": max_path_length,
                                "relation_filter": path_relation_types
                            }
                        }
                        
                        if space_names:
                            method_details["search_scope"] = space_names
                            method_details["scope_type"] = "memory_space"
                        
                        position_score = self._calculate_path_position_score(i, path_length)
                        final_score = path_score * position_score
                        
                        result = RetrievalResult(
                            unit=unit,
                            score=float(final_score),
                            method=self.get_method_type(),
                            method_details=method_details
                        )
                        append_result(result)
            
            retrieval_results = self._deduplicate_and_sort_results(retrieval_results)
            
            if space_names:
                logger.debug(f"Path search completed for spaces {space_names}: {len(retrieval_results[:top_k])} results.")
            
            return retrieval_results[:top_k]
            
        except Exception as e:
            logger.error(f"Path search failed: {e}")
            return []
    
    def _hybrid_search(self, query: str, top_k: int, **kwargs) -> List[RetrievalResult]:
        """HybrID search."""
        try:
            all_results = []
            
            node_weight = kwargs.get('node_weight', 0.5)
            if node_weight > 0:
                node_results = self._node_search(query, int(top_k * 1.5), **kwargs)
                for result in node_results:
                    result.score *= node_weight
                    result.method_details['search_component'] = 'node'
                all_results.extend(node_results)
            
            edge_weight = kwargs.get('edge_weight', 0.3)
            if edge_weight > 0:
                edge_results = self._edge_search(query, int(top_k * 1.0), **kwargs)
                for result in edge_results:
                    result.score *= edge_weight
                    result.method_details['search_component'] = 'edge'
                all_results.extend(edge_results)
            
            path_weight = kwargs.get('path_weight', 0.2)
            if path_weight > 0 and kwargs.get('enable_path_search', False):
                path_results = self._path_search(query, int(top_k * 0.5), **kwargs)
                for result in path_results:
                    result.score *= path_weight
                    result.method_details['search_component'] = 'path'
                all_results.extend(path_results)
            
            
            final_results = self._deduplicate_and_sort_results(all_results)
            
            for result in final_results:
                if 'graph_traversal_info' not in result.method_details:
                    result.method_details['graph_traversal_info'] = {}
                result.method_details['graph_traversal_info']['search_mode'] = 'hybrid'
                result.method_details['graph_traversal_info']['fusion_method'] = 'weighted'
            
            return final_results[:top_k]
            
        except Exception as e:
            logger.error(f"Hybrid graph search failed: {e}")
            return []
    
    def _hybrid_node_search(self, query: str, top_k: int, **kwargs) -> List[RetrievalResult]:
        """HybrID node search."""
        try:
            semantic_results = self._semantic_node_search(query, top_k * 2, **kwargs)
            
            fulltext_results = self._fulltext_node_search(query, top_k * 2, **kwargs)
            
            fused_results = self._rrf_fusion_retrieval_results([semantic_results, fulltext_results])
            
            for result in fused_results:
                result.method_details['graph_traversal_info'] = {
                    'traversal_type': 'hybrid_node_search',
                    'query': query,
                    'fusion_method': 'rrf'
                }
            
            return fused_results[:top_k]
            
        except Exception as e:
            logger.error(f"Hybrid node search failed: {e}")
            return []
    
    def _semantic_node_search(self, query: str, top_k: int, **kwargs) -> List[RetrievalResult]:
        """Semantic node search."""
        try:
            space_names = kwargs.get('space_names')
            candidate_uids = kwargs.get('candidate_uids')
            
            search_kwargs = {}
            if candidate_uids:
                search_kwargs['candidate_uids'] = candidate_uids
            elif space_names:
                search_kwargs['ms_names'] = space_names
            
            semantic_results = self.semantic_map.search_similarity_by_text(
                query, top_k, **search_kwargs
            )
            
            retrieval_results = []
            for unit, score in semantic_results:
                method_details = {
                    "search_mode": "node",
                    "node_search_type": "semantic",
                    "semantic_score": float(score),
                    "graph_traversal_info": {
                        "traversal_type": "semantic_node_search",
                        "query": query
                    }
                }
                
                if space_names:
                    method_details["search_scope"] = space_names
                    method_details["scope_type"] = "memory_space"
                
                result = RetrievalResult(
                    unit=unit,
                    score=float(score),
                    method=self.get_method_type(),
                    method_details=method_details
                )
                retrieval_results.append(result)
            
            if space_names:
                logger.debug(f"Semantic node search completed for spaces {space_names}: {len(retrieval_results)} results.")
            
            return retrieval_results
            
        except Exception as e:
            logger.error(f"Semantic node search failed: {e}")
            return []
    
    def _fulltext_node_search(self, query: str, top_k: int, **kwargs) -> List[RetrievalResult]:
        """Fulltext node search."""
        try:
            space_names = kwargs.get('space_names')
            
            
            multi_retriever = self.semantic_graph.get_multi_retriever()
            if multi_retriever and RetrievalMethod.BM25 in multi_retriever.retrievers:
                search_kwargs = dict(kwargs)
                if space_names:
                    search_kwargs['space_names'] = space_names
                
                bm25_results = multi_retriever.search_single(
                    RetrievalMethod.BM25, query, top_k, **search_kwargs
                )
                
                for result in bm25_results:
                    result.method = self.get_method_type()
                    result.method_details.update({
                        "search_mode": "node",
                        "node_search_type": "fulltext",
                        "graph_traversal_info": {
                            "traversal_type": "fulltext_node_search",
                            "query": query,
                            "delegated_to": "bm25"
                        }
                    })
                    
                    if space_names:
                        result.method_details["search_scope"] = space_names
                        result.method_details["scope_type"] = "memory_space"
                
                if space_names:
                    logger.debug(f"Full-text node search completed for spaces {space_names}: {len(bm25_results)} results.")
                
                return bm25_results
            else:
                logger.warning("BM25 retriever is unavailable; full-text node search skipped.")
                return []
                
        except Exception as e:
            logger.error(f"Full-text node search failed: {e}")
            return []
    
    
    
    
    def _search_edges_by_query(self, query: str, 
                          relation_types: Optional[List[str]] = None,
                          source_nodes: Optional[List[str]] = None,
                          target_nodes: Optional[List[str]] = None,
                          direction: str = 'both',
                          candidate_node_uids: Optional[Set[str]] = None
                          ) -> List[Tuple[str, str, Dict[str, Any], float]]:
        """Score graph edges against a query and optional endpoint filters.

        Returns tuples of ``(source_uid, target_uid, edge_data, score)``. The
        method works on public UIDs while rustworkx stores integer node indices.
        """
        edge_results = []
        
        try:
            relation_type_set = set(relation_types or [])
            source_node_set = set(source_nodes) if source_nodes else None
            target_node_set = set(target_nodes) if target_nodes else None
            candidate_contains = candidate_node_uids.__contains__ if candidate_node_uids else None
            rx_graph = self.semantic_graph.rx_graph
            index_to_uid = self.semantic_graph._index_to_uid
            get_unit = self.semantic_graph.get_unit
            calculate_edge_relevance = self._calculate_edge_query_relevance
            calculate_text_relevance = self._calculate_text_relevance

            for edge_idx in rx_graph.edge_indices():
                src_idx, tgt_idx = rx_graph.get_edge_endpoints_by_index(edge_idx)
                edge_data = rx_graph.get_edge_data_by_index(edge_idx) or {}
                
                source = index_to_uid.get(src_idx, "")
                target = index_to_uid.get(tgt_idx, "")
                
                if not source or not target:
                    continue
                
                if candidate_contains:
                    if not candidate_contains(source) or not candidate_contains(target):
                        continue
                
                edge_score = 0.0
                
                if relation_type_set:
                    edge_type = edge_data.get('type', '')
                    if edge_type not in relation_type_set:
                        continue
                
                if source_node_set is not None and source not in source_node_set:
                    continue
                if target_node_set is not None and target not in target_node_set:
                    continue
                
                edge_score += calculate_edge_relevance(edge_data, query)
                
                source_unit = get_unit(source)
                target_unit = get_unit(target)
                
                if source_unit:
                    source_relevance = calculate_text_relevance(
                        source_unit.raw_data.get('text_content', ''), query
                    )
                    edge_score += source_relevance * 0.3
                
                if target_unit:
                    target_relevance = calculate_text_relevance(
                        target_unit.raw_data.get('text_content', ''), query
                    )
                    edge_score += target_relevance * 0.3
                
                if edge_score > 0:
                    edge_results.append((source, target, edge_data, edge_score))
            
            
            edge_results.sort(key=lambda x: x[3], reverse=True)
            return edge_results
            
        except Exception as e:
            logger.error(f"Edge search failed: {e}")
            return []
    
    def _search_paths_by_query(self, query: str, 
                          start_nodes: Optional[List[str]] = None,
                          end_nodes: Optional[List[str]] = None,
                          max_length: int = 5,
                          relation_types: Optional[List[str]] = None,
                          candidate_node_uids: Optional[Set[str]] = None,
                          path_limit: Optional[int] = None,
                          ) -> List[Tuple[List[str], List[Dict], float]]:
        """Find query-relevant graph paths through the context expander.

        Returns tuples of ``(path_nodes, path_edges, path_score)``. If the
        expander is unavailable, path search fails closed with an empty list.
        """
        try:
            expander = getattr(self, "expander", None)
            if expander is None:
                logger.warning("GraphContextExpander is unavailable; path search returns empty results.")
                return []

            query_text = str(query)
            try:
                max_length = max(1, int(max_length))
            except (TypeError, ValueError):
                max_length = 5

            relation_filter = list(relation_types or [])
            relation_filter_set = set(relation_filter)
            candidate_set = set(candidate_node_uids) if candidate_node_uids is not None else None
            if candidate_set is not None and not candidate_set:
                return []

            source_uids = list(dict.fromkeys(start_nodes or []))
            if not start_nodes:
                search_kwargs = {}
                if candidate_set is not None:
                    search_kwargs["candidate_uids"] = list(candidate_set)
                seed_k = min(max(path_limit or 10, 5), 50)
                semantic_results = self.semantic_map.search_similarity_by_text(query_text, k=seed_k, **search_kwargs)
                source_uids = [unit.uid for unit, _ in semantic_results if getattr(unit, "uid", None)]
            
            if candidate_set is not None:
                source_uids = [uid for uid in source_uids if uid in candidate_set]

            has_node = self.semantic_graph.has_node
            source_uids = [uid for uid in dict.fromkeys(source_uids) if has_node(uid)]
            if not source_uids:
                return []

            target_uids = list(dict.fromkeys(end_nodes or []))
            explicit_targets = end_nodes is not None
            if candidate_set is not None:
                target_uids = [uid for uid in target_uids if uid in candidate_set]
            target_uids = [uid for uid in target_uids if has_node(uid)]
            if explicit_targets and not target_uids:
                return []

            effective_limit = max(1, int(path_limit or max(10, len(source_uids) * 5)))

            if target_uids:
                path_infos = expander.find_paths(
                    source_uids=source_uids,
                    target_uids=target_uids,
                    max_path_length=max_length,
                    path_limit=effective_limit,
                    relation_filter=relation_filter or None,
                    path_scoring_method="semantic",
                )
            else:
                path_infos = expander.find_semantic_paths(
                    source_uids=source_uids,
                    query=query_text,
                    max_path_length=max_length,
                    path_limit=effective_limit,
                )

            if not path_infos:
                return []

            best_paths: Dict[Tuple[Tuple[str, ...], Tuple[str, ...]], Tuple[List[str], List[Dict], float]] = {}
            for path_info in path_infos:
                path_nodes = list(getattr(path_info, "path_nodes", None) or [])
                path_edges = [edge if isinstance(edge, dict) else {} for edge in (getattr(path_info, "path_edges", None) or [])]
                if len(path_nodes) < 2 or len(path_edges) != len(path_nodes) - 1:
                    continue

                if candidate_set is not None and any(uid not in candidate_set for uid in path_nodes):
                    continue

                if relation_filter_set:
                    edge_types = tuple(edge.get('type', '') for edge in path_edges)
                    if any(edge_type not in relation_filter_set for edge_type in edge_types):
                        continue
                else:
                    edge_types = tuple(edge.get('type', '') for edge in path_edges)

                try:
                    path_score = float(getattr(path_info, "path_score", 0.0) or 0.0)
                except (TypeError, ValueError):
                    path_score = 0.0

                if path_score <= 0:
                    continue

                key = (tuple(path_nodes), edge_types)
                previous = best_paths.get(key)
                if previous is None or path_score > previous[2]:
                    best_paths[key] = (path_nodes, path_edges, path_score)

            path_results = list(best_paths.values())
            path_results.sort(key=lambda x: x[2], reverse=True)
            return path_results[:effective_limit]
            
        except Exception as e:
            logger.error(f"Path search failed: {e}")
            return []
    
    def _determine_nodes_from_edge(self, source_uid: str, target_uid: str, edge_data: Dict,
                                  query: str, kwargs: Dict) -> List[Tuple[str, float]]:
        """Returns: List of (node_uid, relevance_score)."""
        nodes_to_return = []
        
        return_strategy = kwargs.get('edge_node_strategy', 'both')  # 'source', 'target', 'both', 'most_relevant'
        
        if return_strategy == 'source':
            nodes_to_return.append((source_uid, 1.0))
        elif return_strategy == 'target':
            nodes_to_return.append((target_uid, 1.0))
        elif return_strategy == 'both':
            nodes_to_return.append((source_uid, 0.8))
            nodes_to_return.append((target_uid, 1.0))
        elif return_strategy == 'most_relevant':
            source_unit = self.semantic_graph.get_unit(source_uid)
            target_unit = self.semantic_graph.get_unit(target_uid)
            
            source_relevance = 0.0
            target_relevance = 0.0
            
            if source_unit:
                source_relevance = self._calculate_text_relevance(
                    source_unit.raw_data.get('text_content', ''), query
                )
            
            if target_unit:
                target_relevance = self._calculate_text_relevance(
                    target_unit.raw_data.get('text_content', ''), query
                )
            
            if source_relevance >= target_relevance:
                nodes_to_return.append((source_uid, source_relevance))
            else:
                nodes_to_return.append((target_uid, target_relevance))
        
        return nodes_to_return
    
    def _calculate_edge_query_relevance(self, edge_data: Dict, query: str) -> float:
        """Calculate edge query relevance."""
        relevance = 0.0
        
        try:
            relation_type = edge_data.get('type', '')
            if relation_type:
                type_relevance = self._calculate_text_relevance(relation_type, query)
                relevance += type_relevance * 0.5
            
            for key, value in edge_data.items():
                if isinstance(value, str) and key not in ['type', 'created', 'updated']:
                    text_relevance = self._calculate_text_relevance(value, query)
                    relevance += text_relevance * 0.3
            
            return min(relevance, 1.0)
            
        except Exception as e:
            logger.warning(f"Edge relevance calculation failed: {e}")
            return 0.0
    
    def _calculate_path_query_relevance(self, path_nodes: List[str], 
                                       path_edges: List[Dict], query: str) -> float:
        """Calculate path query relevance."""
        relevance = 0.0
        
        try:
            node_relevances = []
            for node_uid in path_nodes:
                unit = self.semantic_graph.get_unit(node_uid)
                if unit:
                    node_relevance = self._calculate_text_relevance(
                        unit.raw_data.get('text_content', ''), query
                    )
                    node_relevances.append(node_relevance)
            
            if node_relevances:
                max_node_relevance = max(node_relevances)
                avg_node_relevance = sum(node_relevances) / len(node_relevances)
                relevance += max_node_relevance * 0.6 + avg_node_relevance * 0.2
            
            edge_relevances = []
            for edge_data in path_edges:
                edge_relevance = self._calculate_edge_query_relevance(edge_data, query)
                edge_relevances.append(edge_relevance)
            
            if edge_relevances:
                avg_edge_relevance = sum(edge_relevances) / len(edge_relevances)
                relevance += avg_edge_relevance * 0.2
            
            return min(relevance, 1.0)
            
        except Exception as e:
            logger.warning(f"Path relevance calculation failed: {e}")
            return 0.0
    
    def _calculate_text_relevance(self, text: str, query: str) -> float:
        """Calculate text relevance."""
        if not text or not query:
            return 0.0
        
        try:
            text_lower = text.lower()
            query_lower = query.lower()
            query_words = query_lower.split()
            
            if not query_words:
                return 0.0
            
            matches = sum(1 for word in query_words if word in text_lower)
            relevance = matches / len(query_words)
            
            return relevance
            
        except Exception as e:
            logger.warning(f"Text relevance calculation failed: {e}")
            return 0.0
    
    def _deduplicate_and_sort_results(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Deduplicate and sort results."""
        
        uid_to_best_result = {}
        
        for result in results:
            uid = result.unit.uid
            if uid not in uid_to_best_result or result.score > uid_to_best_result[uid].score:
                uid_to_best_result[uid] = result
        
        
        deduplicated_results = list(uid_to_best_result.values())
        deduplicated_results.sort(key=lambda x: x.score, reverse=True)
        
        return deduplicated_results
    
    def _rrf_fusion_retrieval_results(self, result_lists: List[List[RetrievalResult]], k: int = 60) -> List[RetrievalResult]:
        """Run RRF fusion retrieval results."""
        rrf_scores = defaultdict(float)
        uid_to_result = {}
        
        for results in result_lists:
            for rank, result in enumerate(results, 1):
                uid = result.unit.uid
                rrf_scores[uid] += 1.0 / (k + rank)
                
                if uid not in uid_to_result or result.score > uid_to_result[uid].score:
                    uid_to_result[uid] = result
        
        fused_results = []
        for uid, rrf_score in rrf_scores.items():
            result = uid_to_result[uid]
            result.score = float(rrf_score)
            fused_results.append(result)
        
        
        fused_results.sort(key=lambda x: x.score, reverse=True)
        return fused_results
    
    def _determine_node_role_in_path(self, position: int, path_length: int) -> str:
        """Determine node role in path."""
        if position == 0:
            return "start"
        elif position == path_length - 1:
            return "end"
        else:
            return "intermediate"
    
    def _calculate_path_position_score(self, position: int, path_length: int) -> float:
        """Calculate path position score."""
        if path_length == 1:
            return 1.0
        
        if position == 0 or position == path_length - 1:
            return 1.0
        else:
            return 0.5 + 0.5 * (1.0 - position / (path_length - 1))
