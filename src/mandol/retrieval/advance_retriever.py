"""Multi-backend retrieval coordinator for Mandol.

This module orchestrates dense, BM25, SPLADE, graph, score-fusion, reranking,
and optional graph-expansion paths. It keeps backend index loading separate
from retrieval policy so reproduction scripts can run fixed retrieval
configurations without rebuilding unrelated components.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from functools import partial
import gc
import os
import threading
import time
import logging
import traceback
import asyncio
from typing import Dict, List, Optional, Set, Tuple, Union, Any
from contextlib import contextmanager

import numpy as np
import torch

from .retrieval_interface import (
    BaseRetriever, MultiRetrievalInterface, RetrievalInterface, 
    RetrievalMethod, RetrievalResult, parse_retrieval_methods, parse_weights
)
from .retrieval_utils import (
    ParallelRetrievalConfig, RetrievalSnapshot, 
    FlexibleRetrievalConfig, MultiRetrievalResults
)
from .bm25_retriever import BM25Retriever
from .splade_retriever import SPLADERetriever
from .cosine_retriever import CosineRetrieverAdapter
from .score_fusion import ScoreFusion
from .graph_context_expander import GraphContextExpander, GraphContext
from .rerank_manager import RerankerManager
from .query_bundle import QueryBundle
from ..core.memory_unit import MemoryUnit

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..core.semantic_map import SemanticMap
    from ..core.semantic_graph import SemanticGraph

from ..utils.logging_config import create_module_logger

logger = create_module_logger("advance_retriever")


class MultiRetriever(MultiRetrievalInterface):
    """Coordinate dense, lexical, sparse, graph, fusion, and reranking retrieval.

    MultiRetriever lazily loads backend retrievers for a SemanticMap or
    SemanticGraph, fuses their scores, and optionally applies reranking or graph
    expansion. It keeps backend loading and index reuse separate from retrieval
    policy so benchmark scripts can reproduce fixed configurations.
    """

    _shared_executor_lock = threading.Lock()
    _shared_executors: Dict[int, ThreadPoolExecutor] = {}
    _ASYNC_RERANK_TEXT_OFFLOAD_MIN_CANDIDATES = 1024

    def __init__(self, 
        retrieval_source: Union["SemanticGraph", "SemanticMap"], 
        preload_rerankers: bool = False,
        reranker_configs: Optional[Dict[str, str]] = None,
        reranker_manager: Optional[RerankerManager] = None):
        """Initialize a multi-backend retriever.

        Args:
            retrieval_source: SemanticMap or SemanticGraph used as the retrieval
                source.
            preload_rerankers: Whether to initialize configured rerankers during
                construction.
            reranker_configs: Optional mapping from reranker names to model IDs.
            reranker_manager: Optional externally managed reranker manager shared
                across retrievers.
        """
        from ..core.semantic_map import SemanticMap
        from ..core.semantic_graph import SemanticGraph
        
        if not isinstance(retrieval_source, (SemanticGraph, SemanticMap)):
            raise TypeError("MultiRetriever expects retrieval_source to be a SemanticGraph or SemanticMap instance.")
        
        self.retrieval_source = retrieval_source
        self.retrievers: Dict[RetrievalMethod, BaseRetriever] = {}
        
        
        self._initialized_retrievers: Set[RetrievalMethod] = set()
        
        self.parallel_config = ParallelRetrievalConfig()
        self._retrieval_lock = threading.RLock()
        self._active_snapshots: Dict[str, RetrievalSnapshot] = {}
        
        
        self.reranker_configs = reranker_configs or {
            "baai": "BAAI/bge-reranker-v2-m3",
            "qwen": "Qwen/Qwen3-Reranker-0.6B"
        }
        
        
        self.reranker_manager = reranker_manager
        
        if reranker_manager:
            ScoreFusion._reranker_manager = reranker_manager
            self._uses_external_manager = True
            logger.debug("Using an externally provided reranker manager.")
        else:
            self._uses_external_manager = False
            if preload_rerankers:
                self._preload_rerankers()
            logger.debug("Using an independent reranker manager.")
        
        
        if preload_rerankers:
            self._preload_rerankers()
        
        
        self._is_semantic_graph = isinstance(retrieval_source, SemanticGraph)
        
        
        self.graph_expander: Optional[GraphContextExpander] = None
        
        
        self.index_root_dir: Optional[str] = None
        _candidate_dir = getattr(retrieval_source, '_index_loading_root', None)
        if not _candidate_dir and hasattr(retrieval_source, 'storage_path') and retrieval_source.storage_path:
            _candidate_dir = os.path.join(retrieval_source.storage_path, "retrieval_indices")
        if _candidate_dir and os.path.isdir(_candidate_dir):
            self.index_root_dir = _candidate_dir
            logger.info(f"Detected retrieval index directory: {self.index_root_dir}")
        
        logger.info("MultiRetriever initialized in lazy-loading mode.")

    def _default_smart_search_methods(self) -> List[RetrievalMethod]:
        loaded_methods = [
            method for method in self.retrievers.keys()
            if method != RetrievalMethod.GRAPH_TRAVERSAL
        ]
        if loaded_methods:
            return loaded_methods
        return [
            RetrievalMethod.BM25,
            RetrievalMethod.COSINE_SIMILARITY,
            RetrievalMethod.SPLADE,
        ]
    
    def _ensure_retriever_loaded(self, method: RetrievalMethod) -> bool:
        """Load a backend retriever once and register it for later searches.

        Disk-backed BM25/SPLADE indexes are restored by the concrete loader
        when available. A failed load returns ``False`` so callers can keep the
        retrieval pipeline conservative instead of raising from every search.
        """
        
        if method in self._initialized_retrievers:
            return True
        
        try:
            logger.info(f"Lazy-loading retriever: {method.value}")
            
            
            if method == RetrievalMethod.BM25:
                self._load_bm25_retriever()
            elif method == RetrievalMethod.COSINE_SIMILARITY:
                self._load_cosine_retriever()
            elif method == RetrievalMethod.SPLADE:
                self._load_splade_retriever()
            elif method == RetrievalMethod.GRAPH_TRAVERSAL:
                self._load_graph_retriever()
            else:
                logger.warning(f"Unknown retriever type: {method.value}")
                return False
            
            
            if method not in self.retrievers:
                logger.error(f"Retriever {method.value} was not registered successfully.")
                return False
            
            self._initialized_retrievers.add(method)
            logger.info(f"Retriever loaded successfully: {method.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load retriever {method.value}: {e}")
            logger.debug(traceback.format_exc())
            return False
        
    def _load_bm25_retriever(self):
        """Create the BM25 backend and restore its persisted index if present."""
        if RetrievalMethod.BM25 not in self.retrievers:
            semantic_map_instance = self._get_semantic_map()
            bm25_retriever = BM25Retriever(semantic_map_instance)
            
            
            if self.index_root_dir:
                bm25_path = os.path.join(self.index_root_dir, "bm25")
                if os.path.isdir(bm25_path) and not bm25_retriever._index_built:
                    try:
                        if bm25_retriever.load_index(bm25_path):
                            logger.info("Loaded BM25 index from disk; rebuild skipped.")
                        else:
                            logger.warning("Failed to load BM25 index from disk; it will be rebuilt on demand.")
                    except Exception as e:
                        logger.warning(f"Error while loading BM25 index: {e}")
            
            self.add_retriever(bm25_retriever)
    
    def _load_cosine_retriever(self):
        """Create the dense cosine backend over the underlying SemanticMap."""
        if RetrievalMethod.COSINE_SIMILARITY not in self.retrievers:
            semantic_map_instance = self._get_semantic_map()
            cosine_retriever = CosineRetrieverAdapter(semantic_map_instance)
            self.add_retriever(cosine_retriever)
    
    def _load_splade_retriever(self):
        """Create the SPLADE backend and restore its persisted index if present."""
        if RetrievalMethod.SPLADE not in self.retrievers:
            semantic_map_instance = self._get_semantic_map()
            splade_retriever = SPLADERetriever(semantic_map_instance)
            
            
            if self.index_root_dir:
                splade_path = os.path.join(self.index_root_dir, "splade")
                if os.path.isdir(splade_path) and not splade_retriever._index_built:
                    try:
                        if splade_retriever.load_index(splade_path):
                            logger.info("Loaded SPLADE index from disk; rebuild skipped.")
                        else:
                            logger.warning("Failed to load SPLADE index from disk; it will be rebuilt on demand.")
                    except Exception as e:
                        logger.warning(f"Error while loading SPLADE index: {e}")
            
            self.add_retriever(splade_retriever)
    
    def _load_graph_retriever(self):
        """Create graph traversal retrieval when the source is SemanticGraph."""
        if not self._is_semantic_graph:
            logger.warning("Graph retriever is available only for SemanticGraph sources.")
            return
        
        if RetrievalMethod.GRAPH_TRAVERSAL not in self.retrievers:
            from .graph_retriever import GraphRetriever
            graph_retriever = GraphRetriever(self.retrieval_source)
            self.add_retriever(graph_retriever)
    
    def _get_semantic_map(self):
        """Resolve the SemanticMap shared by dense, lexical, and sparse paths."""
        from ..core.semantic_map import SemanticMap
        from ..core.semantic_graph import SemanticGraph
        
        if isinstance(self.retrieval_source, SemanticGraph):
            return self.retrieval_source.semantic_map
        elif isinstance(self.retrieval_source, SemanticMap):
            return self.retrieval_source
        else:
            raise TypeError("Unable to resolve a SemanticMap instance.")
    
    def search_single(self, method: RetrievalMethod, query: Union[str, QueryBundle], 
             top_k: int = 10, 
             space_names: Optional[List[str]] = None,
             **kwargs) -> List[RetrievalResult]:
        """Run one retrieval backend with optional MemorySpace filtering.

        Args:
            method: Backend retrieval method to execute.
            query: Raw query text or QueryBundle with cached query features.
            top_k: Maximum number of results returned by the backend.
            space_names: Optional memory-space names used to restrict search.
            **kwargs: Backend-specific options such as candidate filters.

        Returns:
            RetrievalResult objects produced by the selected backend. Returns an
            empty list if the backend cannot be loaded.
        """
        
        if isinstance(query, str):
            query = QueryBundle(query)
        
        
        if not self._ensure_retriever_loaded(method):
            logger.warning(f"Failed to load retriever {method.value}.")
            return []
        
        if method not in self.retrievers:
            logger.warning(f"Retriever {method.value} is not available.")
            return []
        
        if space_names:
            kwargs['space_names'] = space_names
            logger.debug(f"Single retriever restricted to spaces: {space_names}")
        
        return self.retrievers[method].search(query, top_k, **kwargs)

    def search_multi(self, query: Union[str, QueryBundle],
                methods: Optional[List[RetrievalMethod]] = None,
                top_k: int = 10,
                space_names: Optional[List[str]] = None,
                **kwargs) -> MultiRetrievalResults:
        """Run multiple retrieval backends and collect per-method results.

        Args:
            query: Raw query text or QueryBundle. QueryBundle lets dense and
                reranking paths reuse computed query features.
            methods: Retrieval methods to run. If omitted, loaded methods or
                Mandol's standard BM25/dense/SPLADE set are used.
            top_k: Per-backend retrieval budget before fusion.
            space_names: Optional MemorySpace filter forwarded to each backend.
            **kwargs: Backend-specific filters such as candidate_uids.

        Returns:
            MultiRetrievalResults containing results grouped by method.

        Notes:
            Missing indexes are built on demand. Dynamic inverted-index
            backends are scheduled sequentially when parallel execution would
            increase contention more than it helps.
        """
        
        if isinstance(query, str):
            query = QueryBundle(query)
        
        if methods is None:
            methods = self._default_smart_search_methods()
        
        
        available_methods = []
        for method in methods:
            if self._ensure_retriever_loaded(method):
                available_methods.append(method)
        
        if not available_methods:
            logger.warning("No retrieval methods are available.")
            return MultiRetrievalResults()
        
        # Avoid mutating LogRecord fields before other handlers process the record.
        if space_names:
            logger.debug(f"search_multi restricted to spaces: {space_names}")
        else:
            logger.debug("search_multi using global retrieval.")
        
        
        needs_index_build = []
        for method in available_methods:
            retriever = self.retrievers[method]
            if hasattr(retriever, '_index_built') and not retriever._index_built:
                needs_index_build.append(method)
        
        if needs_index_build:
            logger.warning(f"The following retriever indexes are missing and will be built: {[m.value for m in needs_index_build]}")
            for method in needs_index_build:
                try:
                    self.retrievers[method].build_index()
                except Exception as e:
                    logger.error(f"Failed to build index for {method.value}: {e}")
        
        
        if self._should_use_mixed_scheduling_for_methods(available_methods):
            result = self._search_multi_mixed(query, available_methods, top_k, space_names=space_names, **kwargs)
        elif not self._should_use_parallel_for_methods(available_methods):
            result = self._search_multi_sequential(query, available_methods, top_k, space_names=space_names, **kwargs)
        else:
            result = self._search_multi_parallel(query, available_methods, top_k, space_names=space_names, **kwargs)
        
        
        if isinstance(query, QueryBundle):
            stats = query.get_stats()
            if stats['total_cache_hits'] > 0:
                logger.debug(
                    f"QueryBundle vector-cache stats: "
                    f"cached_types={stats['cached_types']}, "
                    f"hits={stats['total_cache_hits']}, "
                    f"compute_ms={stats['total_compute_time_ms']:.1f}"
                )
        
        return result

    @classmethod
    def _get_shared_executor(cls, max_workers: int) -> ThreadPoolExecutor:
        max_workers = max(1, int(max_workers))
        with cls._shared_executor_lock:
            executor = cls._shared_executors.get(max_workers)
            if executor is None:
                executor = ThreadPoolExecutor(max_workers=max_workers)
                cls._shared_executors[max_workers] = executor
            return executor

    def _get_dense_query_compute_fn(self):
        """Return the dense-query encoder used by MMR without loading a new model."""
        retrieval_source = getattr(self, "retrieval_source", None)
        if retrieval_source is not None and hasattr(retrieval_source, "_get_text_embedding"):
            return retrieval_source._get_text_embedding

        cosine_retriever = self.retrievers.get(RetrievalMethod.COSINE_SIMILARITY)
        cosine_source = getattr(cosine_retriever, "retrieval_source", None)
        if cosine_source is not None and hasattr(cosine_source, "_get_text_embedding"):
            return cosine_source._get_text_embedding

        return None

    def _should_use_parallel_for_methods(self, methods: List[RetrievalMethod]) -> bool:
        """Decide whether all requested backends can safely run in parallel."""
        if not self.parallel_config.enable_parallel or len(methods) <= 1:
            return False

        for method in methods:
            retriever = self.retrievers.get(method)
            static_mode = getattr(retriever, "_static_mode", None)
            if static_mode is False:
                logger.debug(
                    "search_multi: %s uses dynamic inverted indexes; using sequential scheduling to reduce GIL contention.",
                    method.value,
                )
                return False
        return True

    def _partition_methods_by_dynamic_mode(
        self,
        methods: List[RetrievalMethod],
    ) -> Tuple[List[RetrievalMethod], List[RetrievalMethod]]:
        """Separate dynamic inverted-index backends from parallel-safe backends."""
        dynamic_methods: List[RetrievalMethod] = []
        parallel_methods: List[RetrievalMethod] = []
        for method in methods:
            retriever = self.retrievers.get(method)
            if getattr(retriever, "_static_mode", None) is False:
                dynamic_methods.append(method)
            else:
                parallel_methods.append(method)
        return dynamic_methods, parallel_methods

    def _should_use_mixed_scheduling_for_methods(self, methods: List[RetrievalMethod]) -> bool:
        """Return whether sequential dynamic work should be mixed with parallel work."""
        if not self.parallel_config.enable_parallel or len(methods) <= 1:
            return False
        dynamic_methods, parallel_methods = self._partition_methods_by_dynamic_mode(methods)
        return bool(dynamic_methods and parallel_methods)

    async def search_multi_async(self, query: Union[str, QueryBundle],
                methods: Optional[List[RetrievalMethod]] = None,
                top_k: int = 10,
                space_names: Optional[List[str]] = None,
                **kwargs) -> MultiRetrievalResults:
        """Run multiple backend retrievers from an async caller.

        The method uses a shared thread pool for blocking retrieval backends and
        preserves the same grouping/fallback behavior as ``search_multi``.
        """
        if isinstance(query, str):
            query = QueryBundle(query)
        
        if methods is None:
            methods = [RetrievalMethod.BM25, RetrievalMethod.COSINE_SIMILARITY]
        
        
        available_methods = []
        for method in methods:
            if self._ensure_retriever_loaded(method):
                available_methods.append(method)
        
        if not available_methods:
            logger.warning("No retrieval methods are available.")
            return MultiRetrievalResults()
        
        logger.debug(f"search_multi_async using {len(available_methods)} retrieval methods.")
        
        
        needs_index_build = []
        for method in available_methods:
            retriever = self.retrievers[method]
            if hasattr(retriever, '_index_built') and not retriever._index_built:
                needs_index_build.append(method)
        
        if needs_index_build:
            logger.warning(f"The following retriever indexes are missing and will be built: {[m.value for m in needs_index_build]}")
            loop = asyncio.get_running_loop()
            executor = self._get_shared_executor(self.parallel_config.max_workers)
            for method in needs_index_build:
                try:
                    await loop.run_in_executor(executor, self.retrievers[method].build_index)
                except Exception as e:
                    logger.error(f"Failed to build index for {method.value}: {e}")
        
        loop = asyncio.get_running_loop()
        executor = self._get_shared_executor(self.parallel_config.max_workers)

        query_text = query.query_text if isinstance(query, QueryBundle) else query

        def search_single_sync(method: RetrievalMethod, snapshot: RetrievalSnapshot) -> List[RetrievalResult]:
            try:
                return self._safe_retrieval_worker(
                    method,
                    query,
                    top_k,
                    snapshot,
                    space_names=space_names,
                    **kwargs,
                )
            except Exception as e:
                logger.error(f"Async retrieval failed ({method.value}): {e}")
                if not self.parallel_config.fallback_on_error:
                    raise
                return []

        def search_group_sync(
            group_methods: List[RetrievalMethod],
            snapshot: RetrievalSnapshot,
        ) -> Dict[RetrievalMethod, List[RetrievalResult]]:
            grouped_results: Dict[RetrievalMethod, List[RetrievalResult]] = {}
            for method in group_methods:
                grouped_results[method] = search_single_sync(method, snapshot)
            return grouped_results

        results_by_method: Dict[RetrievalMethod, List[RetrievalResult]] = {}
        with self._create_retrieval_context(query_text) as snapshot:
            if self.parallel_config.enable_parallel and len(available_methods) > 1:
                dynamic_methods, parallel_methods = self._partition_methods_by_dynamic_mode(available_methods)
            else:
                dynamic_methods, parallel_methods = available_methods, []

            task_labels: List[Any] = []
            tasks: List[asyncio.Future] = []

            if dynamic_methods:
                tasks.append(loop.run_in_executor(executor, search_group_sync, dynamic_methods, snapshot))
                task_labels.append("dynamic_group")

            for method in parallel_methods:
                tasks.append(loop.run_in_executor(executor, search_single_sync, method, snapshot))
                task_labels.append(method)

            results_list = await asyncio.gather(*tasks, return_exceptions=True)

            for label, results in zip(task_labels, results_list):
                if isinstance(results, Exception):
                    if label == "dynamic_group":
                        logger.error(f"Async dynamic-group retrieval raised an exception: {results}")
                    else:
                        logger.error(f"Async retrieval raised an exception ({label.value}): {results}")
                    if not self.parallel_config.fallback_on_error:
                        raise results
                    continue

                if label == "dynamic_group":
                    results_by_method.update(results)
                else:
                    results_by_method[label] = results

        multi_results = MultiRetrievalResults()
        for method in available_methods:
            results = results_by_method.get(method, [])
            if results:
                multi_results.add_results(results)
                logger.debug(f"Async retrieval completed for {method.value}: {len(results)} results.")
        
        
        if isinstance(query, QueryBundle):
            stats = query.get_stats()
            if stats['total_cache_hits'] > 0:
                logger.debug(
                    f"QueryBundle vector-cache stats (async): "
                    f"cached_types={stats['cached_types']}, "
                    f"hits={stats['total_cache_hits']}, "
                    f"compute_ms={stats['total_compute_time_ms']:.1f}"
                )
        
        return multi_results

    def build_all_indexes(
        self, 
        methods_to_build: Optional[List[RetrievalMethod]] = None, 
        force_rebuild: bool = False
    ) -> Dict[str, Any]:
        """Build indexes for loaded or explicitly selected retrieval backends.

        Args:
            methods_to_build: Optional subset of backends to load and index.
            force_rebuild: Whether to rebuild even when a backend reports an
                existing index.

        Returns:
            Build statistics keyed by method, including status and duration.
        """
        logger.info("Building retriever indexes.")
        
        
        if methods_to_build:
            logger.info(f"Building specified retrievers: {[m.value for m in methods_to_build]}")
            for method in methods_to_build:
                success = self._ensure_retriever_loaded(method)
                if not success:
                    logger.warning(f"Failed to load retriever {method.value}; skipping.")
        else:
            
            if not self.retrievers:
                logger.info("No retrievers are loaded; loading default retrievers.")
                default_methods = [
                    RetrievalMethod.BM25,
                    RetrievalMethod.COSINE_SIMILARITY,
                    RetrievalMethod.SPLADE
                ]
                for method in default_methods:
                    try:
                        self._ensure_retriever_loaded(method)
                    except Exception as e:
                        logger.warning(f"Failed to load default retriever {method.value}: {e}")
            else:
                logger.info("Building all loaded retrievers.")

        build_stats = {
            "total_retrievers": len(self.retrievers),
            "built_count": 0,
            "skipped_count": 0,
            "failed_count": 0,
            "total_duration": 0.0,
            "details": {}
        }
        
        start_time = time.time()
        
        for method, retriever in self.retrievers.items():
            if methods_to_build and method not in methods_to_build:
                continue

            method_name = method.value if hasattr(method, 'value') else str(method)
            method_start = time.time()
            
            try:
                if not hasattr(retriever, 'build_index'):
                    build_stats["details"][method_name] = {"status": "no_build_method"}
                    build_stats["skipped_count"] += 1
                    continue
                
                is_built = getattr(retriever, '_index_built', False)
                
                if is_built and not force_rebuild:
                    logger.info(f"{method_name} index already exists; build skipped.")
                    build_stats["details"][method_name] = {"status": "already_built"}
                    build_stats["skipped_count"] += 1
                    continue
                
                logger.info(f"Building {method_name} index.")
                retriever.build_index()
                method_duration = time.time() - method_start
                
                logger.info(f"{method_name} index built in {method_duration:.2f}s.")
                build_stats["details"][method_name] = {
                    "status": "success",
                    "duration": method_duration
                }
                build_stats["built_count"] += 1
                
            except Exception as e:
                method_duration = time.time() - method_start
                logger.error(f"Failed to build {method_name} index: {e}")
                logger.debug(traceback.format_exc())
                
                build_stats["details"][method_name] = {
                    "status": "failed",
                    "error": str(e),
                    "duration": method_duration
                }
                build_stats["failed_count"] += 1
        
        build_stats["total_duration"] = time.time() - start_time
        
        logger.info(f"Index build complete: built {build_stats['built_count']}, "
                    f"skipped {build_stats['skipped_count']}, "
                    f"failed {build_stats['failed_count']}, "
                    f"duration {build_stats['total_duration']:.2f}s")
        
        return build_stats

    def build_freeze_indexes(self) -> Dict[str, bool]:
        """Build optional static acceleration indexes for loaded retrievers."""
        results: Dict[str, bool] = {}

        for method, retriever in self.retrievers.items():
            if not hasattr(retriever, "build_freeze_index"):
                continue

            method_name = method.value
            try:
                success = bool(retriever.build_freeze_index())
                results[method_name] = success
                if success:
                    logger.info(f"{method_name} static acceleration index built.")
                else:
                    logger.error(f"Failed to build {method_name} static acceleration index.")
            except Exception as e:
                results[method_name] = False
                logger.error(f"{method_name} static acceleration index build raised an exception: {e}", exc_info=True)

        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Return backend, index, parallel scheduling, and reranker diagnostics."""
        stats = {
            "registered_retrievers": [method.value if hasattr(method, 'value') else str(method) 
                                    for method in self.retrievers.keys()],
            "initialized_retrievers": [method.value for method in self._initialized_retrievers],
            "lazy_loading_enabled": True,
            "retriever_details": {},
            "parallel_config": {
                "enabled": self.parallel_config.enable_parallel,
                "max_workers": self.parallel_config.max_workers,
                "timeout_seconds": self.parallel_config.timeout_seconds,
                "consistency_check": self.parallel_config.consistency_check,
                "fallback_on_error": self.parallel_config.fallback_on_error
            },
            "active_snapshots": len(self._active_snapshots),
            "special_components": {
                "graph_expander": getattr(self, 'graph_expander', None) is not None,
            }
        }
        
        for method, retriever in self.retrievers.items():
            method_name = method.value if hasattr(method, 'value') else str(method)
            
            retriever_info = {
                "type": type(retriever).__name__,
                "initialized": method in self._initialized_retrievers,
                "has_build_index": hasattr(retriever, 'build_index'),
                "index_built": getattr(retriever, '_index_built', False),
                "available": True
            }
            
            if method == RetrievalMethod.BM25 and hasattr(retriever, 'get_index_stats'):
                
                stats = retriever.get_index_stats()
                retriever_info.update({
                    "model_built": stats.get("index_built", False),
                    "total_documents": stats.get("total_docs", 0),
                    "vocabulary_size": stats.get("vocab_size", 0),
                    "k1_parameter": stats.get("k1", 1.5),
                    "b_parameter": stats.get("b", 0.75),
                    "use_jieba": getattr(retriever, 'use_jieba', False),
                    "stop_words_count": len(getattr(retriever, 'stop_words', set()))
                })
                
            elif method == RetrievalMethod.SPLADE and hasattr(retriever, 'model'):
                
                
                total_units = 0
                if hasattr(self.retrieval_source, 'get_all_units'):
                    try:
                        all_units = self.retrieval_source.get_all_units()
                        
                        total_units = sum(1 for unit in all_units 
                                        if unit.has_sparse_embedding())
                    except Exception as e:
                        logger.debug(f"Failed to count SPLADE units: {e}")
                
                retriever_info.update({
                    "model_name": getattr(retriever, 'model_name', 'unknown'),
                    "total_documents": total_units,
                    "embeddings_stored_in_units": True,  
                    "default_text_field": getattr(retriever, 'default_text_field', 'text_content')
                })
                
            elif method == RetrievalMethod.COSINE_SIMILARITY:
                retriever_info.update({
                    "delegates_to": "SemanticMap.search_similarity_by_text",
                    "uses_faiss": True
                })
            
            stats["retriever_details"][method_name] = retriever_info
        
        
        try:
            stats["reranker_info"] = self.get_reranker_stats()
        except Exception as e:
            logger.debug(f"Failed to get reranker stats: {e}")
            stats["reranker_info"] = {"available": False}
        
        return stats
    
    def cleanup_unused_retrievers(self, keep_methods: Optional[List[RetrievalMethod]] = None):
        """Release backend retrievers that are not needed by the active policy."""
        if keep_methods is None:
            keep_methods = []
        
        to_remove = []
        for method in list(self.retrievers.keys()):
            if method not in keep_methods:
                to_remove.append(method)
        
        for method in to_remove:
            try:
                retriever = self.retrievers[method]
                if hasattr(retriever, 'cleanup'):
                    retriever.cleanup()
                
                del self.retrievers[method]
                self._initialized_retrievers.discard(method)
                logger.info(f"Cleaned retriever: {method.value}")
            except Exception as e:
                logger.warning(f"Failed to clean retriever {method.value}: {e}")
        
        gc.collect()
        
        logger.info(f"Cleaned {len(to_remove)} unused retrievers.")
    
    def _preload_rerankers(self):
        """Optionally warm configured rerankers before the first query."""
        try:
            reranker_manager = ScoreFusion.get_reranker_manager()
            
            for reranker_type, model_name in self.reranker_configs.items():
                try:
                    logger.info(f"Preloading reranker: {reranker_type} - {model_name}")
                    reranker_manager.get_reranker(reranker_type, model_name)
                    logger.info(f"Reranker preloaded: {reranker_type}")
                except Exception as e:
                    logger.warning(f"Failed to preload reranker {reranker_type}: {e}")
                    
        except Exception as e:
            logger.warning(f"Reranker preload failed: {e}")
    
    def get_reranker_stats(self) -> Dict[str, Any]:
        """Return available reranker types and currently cached rerankers."""
        try:
            reranker_manager = ScoreFusion.get_reranker_manager()
            return {
                "available_types": reranker_manager.get_available_types(),
                "cached_rerankers": reranker_manager.get_cached_rerankers(),
                "preload_configs": self.reranker_configs
            }
        except Exception as e:
            logger.error(f"Failed to get reranker stats: {e}")
            return {"error": str(e)}
    
    def cleanup_rerankers(self):
        """Clear cached reranker wrappers managed by ScoreFusion."""
        ScoreFusion.cleanup_rerankers()
        logger.info("Reranker cache cleared.")

    def expand_context_for_results(self, 
                                  results: List[Tuple[MemoryUnit, float]],
                                  expansion_type: str = "hops",
                                  **expansion_kwargs) -> Optional[GraphContext]:
        """Expand retrieved units with graph neighborhood or path context."""
        if not self.graph_expander or not results:
            return None
        
        uids = [unit.uid for unit, _ in results]
        
        try:
            if expansion_type == "hops":
                return self.graph_expander.expand_context_by_hops(uids, **expansion_kwargs)
            elif expansion_type == "paths":
                paths = []
                for i, source_uid in enumerate(uids):
                    for target_uid in uids[i+1:]:
                        source_paths = self.graph_expander.find_shortest_paths(
                            [source_uid], [target_uid]
                        )
                        paths.extend(source_paths)
                
                seed_units = [unit for unit, _ in results]
                return GraphContext(
                    seed_units=seed_units,
                    expanded_units=[],
                    relationships=[],
                    paths=[path.path_nodes for path in paths],
                    context_metadata={"expansion_type": "inter_result_paths", "path_count": len(paths)}
                )
            else:
                logger.warning(f"Unknown expansion type: {expansion_type}")
                return None
                
        except Exception as e:
            logger.error(f"Context expansion failed: {e}")
            return None
    
    def get_graph_expansion_capabilities(self) -> Dict[str, bool]:
        """Report graph-expansion features exposed by the current source."""
        return {
            "graph_expander_available": self.graph_expander is not None,
            "supports_hop_expansion": self.graph_expander is not None,
            "supports_path_finding": self.graph_expander is not None,
            "supports_semantic_paths": self.graph_expander is not None,
            "supports_result_enrichment": self.graph_expander is not None
        }

    def smart_search(self, 
                    query: str,
                    methods: Union[str, List[str], List[RetrievalMethod]] = None,
                    top_k: int = 10,
                    
                    
                    fusion_method: str = "rrf",
                    rerank_method: Optional[str] = None,
                    weights: Optional[Union[Dict[str, float], Dict[RetrievalMethod, float]]] = None,
                    
                    enable_graph_expansion: bool = False,
                    graph_expansion_config: Optional[Dict[str, Any]] = None,
                    
                    
                    rerank_params: Optional[Dict[str, Any]] = None,
                    
                    return_detailed: bool = False,
                    
                    **kwargs) -> Union[List[Tuple[MemoryUnit, float]], Dict[str, Any]]:
        """Run fused retrieval with optional reranking and graph expansion.

        Args:
            query: Query text or QueryBundle.
            methods: Retrieval methods to enable. Defaults to loaded methods or
                Mandol's standard dense, BM25, and SPLADE set.
            top_k: Number of final results to return.
            fusion_method: Score fusion strategy.
            rerank_method: Optional reranker identifier.
            weights: Optional per-method fusion weights.
            enable_graph_expansion: Whether to enrich results with graph context.
            graph_expansion_config: Optional graph expansion settings.
            rerank_params: Optional reranker-specific parameters.
            return_detailed: Whether to include execution diagnostics.
            **kwargs: Backend-specific retrieval options.

        Returns:
            Ranked memory units, or a detailed result dictionary when requested.
        """
        if rerank_method:
            ScoreFusion.ensure_sync_rerank_allowed(rerank_method, context="MultiRetriever.smart_search")
        try:
            start_time = datetime.now()
            
            if methods is None:
                methods = self._default_smart_search_methods()
            
            parsed_methods = parse_retrieval_methods(methods)
            
            # Avoid mutating LogRecord fields before other handlers process the record.
            filtered_methods = [m for m in parsed_methods if m != RetrievalMethod.GRAPH_TRAVERSAL]
            
            if not filtered_methods:
                error_msg = "No valid retrieval methods are available."
                logger.warning(error_msg)
                return [] if not return_detailed else {"results": [], "error": error_msg}
            
            execution_plan = {
                "methods": filtered_methods,
                "fusion_method": fusion_method,
                "rerank_method": rerank_method,
                "enable_graph_expansion": enable_graph_expansion,
                "weights": parse_weights(weights, filtered_methods) if weights else None,
                "top_k": top_k
            }
            
            logger.info(f"Smart-search plan: {len(filtered_methods)} methods, "
                        f"fusion={fusion_method}, rerank={rerank_method}, "
                        f"graph_expansion={enable_graph_expansion}")

            query_bundle = query if isinstance(query, QueryBundle) else QueryBundle(query)
            query_text = query_bundle.query_text
            
            base_results = self._execute_base_retrieval(
                query=query_bundle,
                execution_plan=execution_plan,
                **kwargs
            )
            
            if not base_results:
                error_msg = "Base retrieval returned no results."
                logger.warning(error_msg)
                return [] if not return_detailed else {"results": [], "error": error_msg}
            
            
            if rerank_method:
                reranked_results = self._execute_reranking(
                    query=query_bundle if rerank_method.lower() == "mmr" else query_text,
                    base_results=base_results,
                    rerank_method=rerank_method,
                    rerank_params=rerank_params or {},
                    top_k=top_k
                )
                final_results = reranked_results
            else:
                final_results = base_results[:top_k]
            
            graph_enrichment_info = None
            if enable_graph_expansion and self.graph_expander:
                try:
                    graph_enrichment_info = self._execute_graph_expansion(
                        results=final_results,
                        graph_expansion_config=graph_expansion_config or {},
                        top_k=top_k
                    )
                    
                    if graph_enrichment_info and "expanded_results" in graph_enrichment_info:
                        final_results = graph_enrichment_info["expanded_results"]
                    
                except Exception as e:
                    logger.warning(f"Graph expansion failed; using original results: {e}")
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            if return_detailed:
                return self._build_detailed_response(
                    final_results=final_results,
                    execution_plan=execution_plan,
                    base_results=base_results,
                    graph_enrichment_info=graph_enrichment_info,
                    duration=duration,
                    **kwargs
                )
            else:
                return final_results
                
        except Exception as e:
            if ScoreFusion.is_vllm_sync_rerank_error(e):
                raise
            error_msg = f"Smart search failed: {e}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            
            if return_detailed:
                return {
                    "results": [],
                    "error": error_msg,
                    "execution_plan": execution_plan if 'execution_plan' in locals() else None
                }
            return []

    async def smart_search_async(self, 
                    query: str,
                    methods: Union[str, List[str], List[RetrievalMethod]] = None,
                    top_k: int = 10,
                    
                    
                    fusion_method: str = "rrf",
                    rerank_method: Optional[str] = None,
                    weights: Optional[Union[Dict[str, float], Dict[RetrievalMethod, float]]] = None,
                    
                    enable_graph_expansion: bool = False,
                    graph_expansion_config: Optional[Dict[str, Any]] = None,
                    
                    
                    rerank_params: Optional[Dict[str, Any]] = None,
                    
                    return_detailed: bool = False,
                    
                    **kwargs) -> Union[List[Tuple[MemoryUnit, float]], Dict[str, Any]]:
        """Run fused retrieval and optional reranking from an async caller.

        Args are the same as ``smart_search``. Blocking retrieval and reranking
        work is offloaded to the shared executor when needed, while async-capable
        rerankers are awaited directly.
        """
        try:
            start_time = datetime.now()
            
            if methods is None:
                methods = self._default_smart_search_methods()
            
            parsed_methods = parse_retrieval_methods(methods)
            filtered_methods = [m for m in parsed_methods if m != RetrievalMethod.GRAPH_TRAVERSAL]
            
            if not filtered_methods:
                error_msg = "No valid retrieval methods are available."
                logger.warning(error_msg)
                return [] if not return_detailed else {"results": [], "error": error_msg}
            
            execution_plan = {
                "methods": filtered_methods,
                "fusion_method": fusion_method,
                "rerank_method": rerank_method,
                "enable_graph_expansion": enable_graph_expansion,
                "weights": parse_weights(weights, filtered_methods) if weights else None,
                "top_k": top_k
            }
            
            logger.info(f"Async smart-search plan: {len(filtered_methods)} methods, "
                        f"fusion={fusion_method}, rerank={rerank_method}")

            query_bundle = query if isinstance(query, QueryBundle) else QueryBundle(query)
            query_text = query_bundle.query_text
            
            base_results = await self._execute_base_retrieval_async(
                query=query_bundle,
                execution_plan=execution_plan,
                **kwargs
            )
            
            if not base_results:
                error_msg = "Base retrieval returned no results."
                logger.warning(error_msg)
                return [] if not return_detailed else {"results": [], "error": error_msg}
            
            
            if rerank_method:
                reranked_results = await self._execute_reranking_async(
                    query=query_bundle if rerank_method.lower() == "mmr" else query_text,
                    base_results=base_results,
                    rerank_method=rerank_method,
                    rerank_params=rerank_params or {},
                    top_k=top_k
                )
                final_results = reranked_results
            else:
                final_results = base_results[:top_k]
            
            graph_enrichment_info = None
            if enable_graph_expansion and self.graph_expander:
                try:
                    graph_enrichment_info = await asyncio.to_thread(
                        self._execute_graph_expansion,
                        final_results,
                        graph_expansion_config or {},
                        top_k
                    )
                    
                    if graph_enrichment_info and "expanded_results" in graph_enrichment_info:
                        final_results = graph_enrichment_info["expanded_results"]
                    
                except Exception as e:
                    logger.warning(f"Graph expansion failed; using original results: {e}")
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            if return_detailed:
                return self._build_detailed_response(
                    final_results=final_results,
                    execution_plan=execution_plan,
                    base_results=base_results,
                    graph_enrichment_info=graph_enrichment_info,
                    duration=duration,
                    **kwargs
                )
            else:
                return final_results
                
        except Exception as e:
            if ScoreFusion.is_vllm_sync_rerank_error(e):
                raise
            error_msg = f"Async smart search failed: {e}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            
            if return_detailed:
                return {
                    "results": [],
                    "error": error_msg,
                    "execution_plan": execution_plan if 'execution_plan' in locals() else None
                }
            return []


    def _execute_base_retrieval(self, 
                       query: Union[str, QueryBundle],
                       execution_plan: Dict[str, Any],
                       space_names: Optional[List[str]] = None,
                       **kwargs) -> List[Tuple[MemoryUnit, float]]:
        """Run first-stage retrieval and fuse backend scores.

        The first stage intentionally retrieves a larger candidate pool than
        ``top_k`` so second-stage reranking has enough cross-backend evidence.
        """
        methods = execution_plan["methods"]
        fusion_method = execution_plan["fusion_method"]
        weights = execution_plan["weights"]
        top_k = execution_plan["top_k"]
        
        
        # search_top_k = max(top_k * 3, 50) if execution_plan["rerank_method"] else top_k * 2
        
        
        search_top_k = max(top_k * 3, 50)
        
        logger.debug(f"_execute_base_retrieval using spaces: {space_names}")
        
        multi_results = self.search_multi(
            query, 
            methods, 
            search_top_k, 
            space_names=space_names,
            **kwargs
        )
        
        if fusion_method == "rrf":
            fused_results = ScoreFusion.rrf_fusion(multi_results, k=execution_plan.get("rrf_k", 60))
        elif fusion_method == "weighted" and weights:
            fused_results = ScoreFusion.weighted_fusion(multi_results, weights)
        elif fusion_method == "average":
            fused_results = ScoreFusion.average_fusion(multi_results)
        else:
            fused_results = ScoreFusion.rrf_fusion(multi_results)
        
        logger.debug(f"Base retrieval completed with {len(fused_results)} fused results.")
        return fused_results
    
    def _execute_reranking(self, 
        query: Union[str, QueryBundle],
        base_results: List[Tuple[MemoryUnit, float]],
        rerank_method: str,
        rerank_params: Dict[str, Any],
        top_k: int) -> List[Tuple[MemoryUnit, float]]:
        """Apply the selected synchronous reranker or MMR fusion.

        vLLM-backed rerankers are guarded by ``ScoreFusion`` because those
        backends require the async path. Runtime failures fall back to the fused
        first-stage ranking unless the error is a sync/async contract violation.
        """
        ScoreFusion.ensure_sync_rerank_allowed(rerank_method, context="MultiRetriever._execute_reranking")
        try:
            query_text = query.query_text if isinstance(query, QueryBundle) else query
            
            rerank_method_lower = rerank_method.lower()
            
            
            threshold = rerank_params.get("threshold", 0.0)
            
            
            if rerank_method_lower == "mmr":
                temp_multi_results = MultiRetrievalResults()
                for unit, score in base_results:
                    result = RetrievalResult(unit, score, RetrievalMethod.HYBRID, {"base_score": score})
                    temp_multi_results.add_result(result)
                
                reranked_results = ScoreFusion.mmr_fusion(
                    temp_multi_results,
                    query,
                    lambda_param=rerank_params.get("lambda_param", 0.5),
                    top_k=top_k,
                    dense_compute_fn=rerank_params.get("dense_compute_fn") or self._get_dense_query_compute_fn(),
                )
            
            
            elif rerank_method_lower == "baai" or "baai" in rerank_method_lower:
                reranked_results = ScoreFusion.BAAI_fusion(
                    base_results,
                    query_text,
                    top_k=top_k,
                    model_name=rerank_params.get("model_name", "BAAI/bge-reranker-v2-m3"),
                    max_candidates=rerank_params.get("max_candidates", 100),
                    text_field=rerank_params.get("text_field", "text_content"),
                    max_length=rerank_params.get("max_length", 512),
                    batch_size=rerank_params.get("batch_size", 32),
                    threshold=threshold
                )
            
            
            elif rerank_method_lower == "qwen" or "qwen" in rerank_method_lower:
                reranked_results = ScoreFusion.qwen_fusion(
                    base_results,
                    query_text,
                    top_k=top_k,
                    model_name=rerank_params.get("model_name", "Qwen/Qwen3-Reranker-0.6B"),
                    max_candidates=rerank_params.get("max_candidates", 100),
                    text_field=rerank_params.get("text_field", "text_content"),
                    max_length=rerank_params.get("max_length", 512),
                    batch_size=rerank_params.get("batch_size", 32),
                    instruction=rerank_params.get("instruction", None),
                    threshold=threshold
                )
            
            
            elif rerank_method_lower == "jina" or "jina" in rerank_method_lower:
                reranked_results = ScoreFusion.jina_fusion(
                    base_results,
                    query_text,
                    top_k=top_k,
                    model_name=rerank_params.get("model_name", "jinaai/jina-reranker-v3"),
                    max_candidates=rerank_params.get("max_candidates", 100),
                    text_field=rerank_params.get("text_field", "text_content"),
                    max_length=rerank_params.get("max_length", 512),
                    batch_size=rerank_params.get("batch_size", 32),
                    threshold=threshold
                )
            
            else:
                logger.warning(f"Unknown rerank method {rerank_method}; skipping reranking.")
                reranked_results = base_results[:top_k]
            
            logger.debug(f"Reranking completed: method={rerank_method}, threshold={threshold}, results={len(reranked_results)}.")
            return reranked_results
            
        except Exception as e:
            if ScoreFusion.is_vllm_sync_rerank_error(e):
                raise
            logger.error(f"Reranking failed: {e}")
            logger.debug(traceback.format_exc())
            return base_results[:top_k]
    
    async def _execute_base_retrieval_async(self, 
                       query: Union[str, QueryBundle],
                       execution_plan: Dict[str, Any],
                       space_names: Optional[List[str]] = None,
                       **kwargs) -> List[Tuple[MemoryUnit, float]]:
        """Async wrapper for first-stage retrieval and score fusion."""
        methods = execution_plan["methods"]
        fusion_method = execution_plan["fusion_method"]
        weights = execution_plan["weights"]
        top_k = execution_plan["top_k"]
        
        # search_top_k = max(top_k * 3, 50) if execution_plan["rerank_method"] else top_k * 2
        
        
        search_top_k = max(top_k * 3, 50)
        
        logger.debug(f"_execute_base_retrieval_async using spaces: {space_names}")
        
        
        multi_results = await self.search_multi_async(
            query, 
            methods, 
            search_top_k, 
            space_names=space_names,
            **kwargs
        )
        
        if fusion_method == "rrf":
            fused_results = await asyncio.to_thread(
                ScoreFusion.rrf_fusion, multi_results, execution_plan.get("rrf_k", 60)
            )
        elif fusion_method == "weighted" and weights:
            fused_results = await asyncio.to_thread(
                ScoreFusion.weighted_fusion, multi_results, weights
            )
        elif fusion_method == "average":
            fused_results = await asyncio.to_thread(
                ScoreFusion.average_fusion, multi_results
            )
        else:
            fused_results = await asyncio.to_thread(
                ScoreFusion.rrf_fusion, multi_results
            )
        
        logger.debug(f"Async base retrieval completed with {len(fused_results)} fused results.")
        return fused_results
    
    async def _execute_reranking_async(self, 
        query: Union[str, QueryBundle],
        base_results: List[Tuple[MemoryUnit, float]],
        rerank_method: str,
        rerank_params: Dict[str, Any],
        top_k: int) -> List[Tuple[MemoryUnit, float]]:
        """Apply async-capable rerankers over the fused candidate pool."""
        try:
            query_text = query.query_text if isinstance(query, QueryBundle) else query
            rerank_method_lower = rerank_method.lower()
            threshold = rerank_params.get("threshold", 0.0)
            max_candidates = rerank_params.get("max_candidates", 100)
            text_field = rerank_params.get("text_field", "text_content")
            max_length = rerank_params.get("max_length", 512)
            batch_size = rerank_params.get("batch_size", 32)
            
            candidates = base_results[:max_candidates]
            if len(candidates) >= self._ASYNC_RERANK_TEXT_OFFLOAD_MIN_CANDIDATES:
                documents = await asyncio.to_thread(
                    self._prepare_rerank_documents,
                    candidates,
                    text_field,
                )
            else:
                documents = self._prepare_rerank_documents(candidates, text_field)
            
            
            reranker_manager = ScoreFusion.get_reranker_manager()
            
            if rerank_method_lower == "mmr":
                return await asyncio.to_thread(
                    self._execute_reranking,
                    query, base_results, rerank_method, rerank_params, top_k
                )
            
            elif rerank_method_lower == "baai" or "baai" in rerank_method_lower:
                model_name = rerank_params.get("model_name", "BAAI/bge-reranker-v2-m3")
                reranker = reranker_manager.get_reranker("baai", model_name)
                scores = await reranker.rerank_async(query_text, documents, batch_size, max_length)
            
            elif rerank_method_lower == "qwen" or "qwen" in rerank_method_lower:
                model_name = rerank_params.get("model_name", "Qwen/Qwen3-Reranker-0.6B")
                instruction = rerank_params.get("instruction", None)
                reranker = reranker_manager.get_reranker("qwen", model_name)
                scores = await reranker.rerank_async(query_text, documents, batch_size, max_length, instruction)
            
            elif rerank_method_lower == "jina" or "jina" in rerank_method_lower:
                model_name = rerank_params.get("model_name", "jinaai/jina-reranker-v3")
                reranker = reranker_manager.get_reranker("jina", model_name)
                scores = await reranker.rerank_async(query_text, documents, batch_size, max_length)
            
            
            elif rerank_method_lower == "qwen-sili" or "sili" in rerank_method_lower:
                model_name = rerank_params.get("model_name", "Qwen/Qwen3-Reranker-8B")
                reranker = reranker_manager.get_reranker("qwen-sili", model_name)
                scores = await reranker.rerank_async(query_text, documents, batch_size, max_length)
            
            elif rerank_method_lower == "qwen-dashscope" or "dashscope" in rerank_method_lower:
                model_name = rerank_params.get("model_name", "qwen3-rerank")
                instruction = rerank_params.get("instruction", None)
                reranker = reranker_manager.get_reranker("qwen-dashscope", model_name)
                scores = await reranker.rerank_async(query_text, documents, batch_size, max_length, instruction)
            
            elif rerank_method_lower == "gte-dashscope" or "gte" in rerank_method_lower:
                model_name = rerank_params.get("model_name", "gte-rerank-v2")
                reranker = reranker_manager.get_reranker("gte-dashscope", model_name)
                scores = await reranker.rerank_async(query_text, documents, batch_size, max_length)
            
            else:
                logger.warning(f"Unknown async rerank method {rerank_method}; falling back to the synchronous implementation.")
                return await asyncio.to_thread(
                    self._execute_reranking,
                    query, base_results, rerank_method, rerank_params, top_k
                )
            
            
            scored_results = list(zip(candidates, scores))
            scored_results.sort(key=lambda x: x[1], reverse=True)
            
            if threshold > 0:
                scored_results = [(r, s) for r, s in scored_results if s >= threshold]
            
            reranked_results = [(unit, score) for (unit, _), score in scored_results[:top_k]]
            
            logger.debug(f"Async reranking completed: method={rerank_method}, results={len(reranked_results)}.")
            return reranked_results
            
        except Exception as e:
            if ScoreFusion.is_vllm_sync_rerank_error(e):
                raise
            logger.error(f"Async reranking failed: {e}")
            logger.debug(traceback.format_exc())
            return base_results[:top_k]

    @staticmethod
    def _prepare_rerank_documents(
        candidates: List[Tuple[MemoryUnit, float]],
        text_field: str,
    ) -> List[str]:
        """Extract candidate text payloads for second-stage rerankers."""
        return [ScoreFusion._extract_text_content(unit, text_field) for unit, _ in candidates]
        
    def _execute_graph_expansion(self, 
                                results: List[Tuple[MemoryUnit, float]],
                                graph_expansion_config: Dict[str, Any],
                                top_k: int) -> Dict[str, Any]:
        """Enrich final results with graph neighbors or inter-result paths."""
        if not self.graph_expander:
            return {"error": "Graph expander is unavailable."}
        
        try:
            default_config = {
                "expand_hops": 1,
                "max_neighbors": 5,
                "find_inter_paths": True,
                "semantic_threshold": 0.7,
                "include_relation_summary": True
            }
            default_config.update(graph_expansion_config)
            
            enriched_info = self.graph_expander.enrich_retrieval_results(
                results, default_config
            )
            
            expanded_results = results.copy()
            
            if enriched_info and "expanded_context" in enriched_info:
                expanded_units = enriched_info["expanded_context"].expanded_units
                
                expansion_score_weight = 0.3
                expanded_entries = [
                    (unit, expansion_score_weight) 
                    for unit in expanded_units[:top_k//2]
                ]
                
                seen_uids = {unit.uid for unit, _ in results}
                for unit, score in expanded_entries:
                    if unit.uid not in seen_uids:
                        expanded_results.append((unit, score))
                        seen_uids.add(unit.uid)
                
                
                expanded_results.sort(key=lambda x: x[1], reverse=True)
                expanded_results = expanded_results[:top_k]
            
            enriched_info["expanded_results"] = expanded_results
            logger.debug(f"Graph expansion completed: original={len(results)}, expanded={len(expanded_results)}.")
            
            return enriched_info
            
        except Exception as e:
            logger.error(f"Graph expansion execution failed: {e}")
            return {"error": str(e), "expanded_results": results}

    def _build_detailed_response(self, 
                                final_results: List[Tuple[MemoryUnit, float]],
                                execution_plan: Dict[str, Any],
                                base_results: List[Tuple[MemoryUnit, float]],
                                graph_enrichment_info: Optional[Dict[str, Any]],
                                duration: float,
                                **kwargs) -> Dict[str, Any]:
        """Assemble diagnostics returned by ``smart_search(return_detailed=True)``."""
        response = {
            "results": final_results,
            "execution_plan": execution_plan,
            "statistics": {
                "final_results_count": len(final_results),
                "base_results_count": len(base_results),
                "duration_seconds": duration,
                "methods_used": [m.value for m in execution_plan["methods"]],
                "retrieval_success": True
            }
        }
        
        if graph_enrichment_info:
            response["graph_enrichment"] = graph_enrichment_info
            response["statistics"]["graph_expansion_enabled"] = execution_plan["enable_graph_expansion"]
        
        method_stats = {}
        for method in execution_plan["methods"]:
            if method in self.retrievers:
                method_stats[method.value] = {
                    "available": True,
                    "type": type(self.retrievers[method]).__name__
                }
        response["method_stats"] = method_stats
        
        response["config"] = {
            "parallel_enabled": self.parallel_config.enable_parallel,
            "graph_expansion_available": self.graph_expander is not None
        }
        
        return response

    def configure_parallel_retrieval(self, config: ParallelRetrievalConfig):
        """Replace the runtime parallel-retrieval policy."""
        self.parallel_config = config
        logger.info(f"Parallel retrieval configured: enabled={config.enable_parallel}, workers={config.max_workers}")

    @contextmanager
    def _create_retrieval_context(self, query: str):
        """Register a short-lived consistency snapshot for concurrent searches."""
        snapshot = None
        try:
            snapshot = RetrievalSnapshot(self.retrieval_source, f"query_{hash(query)}_{int(time.time() * 1000)}")
            
            with self._retrieval_lock:
                self._active_snapshots[snapshot.snapshot_id] = snapshot
            
            yield snapshot
            
        finally:
            if snapshot:
                with self._retrieval_lock:
                    self._active_snapshots.pop(snapshot.snapshot_id, None)

    def add_retriever(self, retriever: BaseRetriever):
        """Register a backend retriever."""
        self.retrievers[retriever.get_method_type()] = retriever
        logger.info(f"Added retriever: {retriever.get_method_type().value}")
    
    def remove_retriever(self, method: RetrievalMethod):
        """Unregister a backend retriever without touching persisted indexes."""
        if method in self.retrievers:
            del self.retrievers[method]
            logger.info(f"Removed retriever: {method.value}")
        
    def _search_multi_sequential(self, query: Union[str, QueryBundle], 
                            methods: List[RetrievalMethod],
                            top_k: int,
                            space_names: Optional[List[str]] = None,
                            **kwargs) -> MultiRetrievalResults:
        """Run backends one at a time under one retrieval snapshot."""
        multi_results = MultiRetrievalResults()
        query_text = query.query_text if isinstance(query, QueryBundle) else query
        
        with self._create_retrieval_context(query_text) as snapshot:
            for method in methods:
                try:
                    results = self.retrievers[method].search(
                        query, 
                        top_k, 
                        space_names=space_names,
                        **kwargs
                    )
                    
                    multi_results.add_results(results)
                    logger.debug(f"{method.value} retrieval completed: {len(results)} results.")
                    
                except Exception as e:
                    logger.error(f"{method.value} retrieval failed: {e}")
                    if not self.parallel_config.fallback_on_error:
                        raise
        
        return multi_results
    
    def _search_multi_parallel(self, query: Union[str, QueryBundle], 
                          methods: List[RetrievalMethod],
                          top_k: int,
                          space_names: Optional[List[str]] = None,
                          **kwargs) -> MultiRetrievalResults:
        """Run parallel-safe backends through the shared executor."""
        multi_results = MultiRetrievalResults()
        query_text = query.query_text if isinstance(query, QueryBundle) else query
        
        with self._create_retrieval_context(query_text) as snapshot:
            executor = self._get_shared_executor(self.parallel_config.max_workers)
            future_to_method = {
                executor.submit(
                    self._safe_retrieval_worker,
                    method, query, top_k, snapshot,
                    space_names=space_names,
                    **kwargs
                ): method
                for method in methods
            }

            for future in as_completed(future_to_method, timeout=self.parallel_config.timeout_seconds):
                method = future_to_method[future]
                try:
                    results = future.result()
                    multi_results.add_results(results)
                    logger.debug(f"Parallel retrieval completed for {method.value}: {len(results)} results.")
                except Exception as e:
                    logger.error(f"Parallel retrieval failed for {method.value}: {e}")
                    if not self.parallel_config.fallback_on_error:
                        raise
        
        return multi_results

    def _search_multi_mixed(self, query: Union[str, QueryBundle],
                            methods: List[RetrievalMethod],
                            top_k: int,
                            space_names: Optional[List[str]] = None,
                            **kwargs) -> MultiRetrievalResults:
        """Run dynamic backends sequentially and parallel-safe backends concurrently."""
        multi_results = MultiRetrievalResults()
        query_text = query.query_text if isinstance(query, QueryBundle) else query
        dynamic_methods, parallel_methods = self._partition_methods_by_dynamic_mode(methods)
        results_by_method: Dict[RetrievalMethod, List[RetrievalResult]] = {}

        with self._create_retrieval_context(query_text) as snapshot:
            executor = self._get_shared_executor(self.parallel_config.max_workers)
            future_to_method = {
                executor.submit(
                    self._safe_retrieval_worker,
                    method, query, top_k, snapshot,
                    space_names=space_names,
                    **kwargs,
                ): method
                for method in parallel_methods
            }

            for method in dynamic_methods:
                try:
                    results_by_method[method] = self._safe_retrieval_worker(
                        method, query, top_k, snapshot,
                        space_names=space_names,
                        **kwargs,
                    )
                    logger.debug(f"Mixed scheduling dynamic retrieval completed for {method.value}: {len(results_by_method[method])} results.")
                except Exception as e:
                    logger.error(f"Mixed scheduling dynamic retrieval failed for {method.value}: {e}")
                    if not self.parallel_config.fallback_on_error:
                        raise

            for future in as_completed(future_to_method, timeout=self.parallel_config.timeout_seconds):
                method = future_to_method[future]
                try:
                    results_by_method[method] = future.result()
                    logger.debug(f"Mixed scheduling parallel retrieval completed for {method.value}: {len(results_by_method[method])} results.")
                except Exception as e:
                    logger.error(f"Mixed scheduling parallel retrieval failed for {method.value}: {e}")
                    if not self.parallel_config.fallback_on_error:
                        raise

        for method in methods:
            multi_results.add_results(results_by_method.get(method, []))
        return multi_results
    
    def _safe_retrieval_worker(self, method: RetrievalMethod, query: Union[str, QueryBundle], 
                          top_k: int, snapshot: RetrievalSnapshot,
                          space_names: Optional[List[str]] = None,
                          **kwargs) -> List[RetrievalResult]:
        """Execute one backend with fallback-on-error semantics."""
        try:
            if self.parallel_config.consistency_check and not snapshot.validate_consistency():
                logger.warning(f"Retrieval context changed during {method.value}; continuing.")
            
            results = self.retrievers[method].search(
                query, 
                top_k, 
                space_names=space_names,
                **kwargs
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Retrieval failed ({method.value}): {e}")
            if not self.parallel_config.fallback_on_error:
                raise
            return []
    
        
    def _search_single_flexible(self, query: str, config: FlexibleRetrievalConfig, 
                               top_k: int, return_detailed: bool, **kwargs) -> Union[List[Tuple[MemoryUnit, float]], Dict[str, Any]]:
        """Execute a one-method FlexibleRetrievalConfig."""
        method = config.methods[0]
        
        if method not in self.retrievers:
            error_msg = f"Retriever {method.value} is not registered."
            logger.error(error_msg)
            if return_detailed:
                return {"results": [], "error": error_msg, "config": config.to_dict()}
            return []
        
        try:
            with self._create_retrieval_context(query) as snapshot:
                results = self.retrievers[method].search(query, top_k, **kwargs)
            
            final_results = [(result.unit, result.score) for result in results]
            
            if return_detailed:
                return {
                    "results": final_results,
                    "config": config.to_dict(),
                    "method_stats": {
                        method.value: {
                            "count": len(results),
                            "avg_score": np.mean([r.score for r in results]) if results else 0.0,
                            "max_score": max([r.score for r in results]) if results else 0.0,
                            "min_score": min([r.score for r in results]) if results else 0.0
                        }
                    },
                    "parallel_mode": False
                }
            return final_results
            
        except Exception as e:
            error_msg = f"Single-method retrieval failed ({method.value}): {e}"
            logger.error(error_msg)
            if return_detailed:
                return {"results": [], "error": error_msg, "config": config.to_dict()}
            return []
        
    def _search_multi_flexible(self, query: str, config: FlexibleRetrievalConfig, 
                              top_k: int, return_detailed: bool, **kwargs) -> Union[List[Tuple[MemoryUnit, float]], Dict[str, Any]]:
        """Execute a multi-method FlexibleRetrievalConfig with configured fusion."""
        search_top_k = int(top_k * config.top_k_multiplier)
        multi_results = self.search_multi(query, config.methods, search_top_k, **kwargs)
        
        if config.fusion_method == "rrf":
            fused_results = ScoreFusion.rrf_fusion(multi_results, k=config.rrf_k, top_k=top_k)
        elif config.fusion_method == "weighted":
            fused_results = ScoreFusion.weighted_fusion(
                multi_results, config.weights, normalization=config.normalization
            )
        elif config.fusion_method == "average":
            fused_results = ScoreFusion.average_fusion(multi_results)
        else:
            raise ValueError(f"Unsupported fusion method: {config.fusion_method}")
        
        
        final_results = fused_results[:top_k]
        
        if return_detailed:
            method_stats = {}
            for method in config.methods:
                method_results = multi_results.results_by_method.get(method, [])
                method_stats[method.value] = {
                    "count": len(method_results),
                    "avg_score": np.mean([r.score for r in method_results]) if method_results else 0.0,
                    "max_score": max([r.score for r in method_results]) if method_results else 0.0,
                    "min_score": min([r.score for r in method_results]) if method_results else 0.0
                }
            
            return {
                "results": final_results,
                "config": config.to_dict(),
                "method_stats": method_stats,
                "fusion_stats": {
                    "total_candidates": len(multi_results.get_union_units()),
                    "methods_used": [method.value for method in multi_results.get_methods_used()],
                    "fusion_method": config.fusion_method
                },
                "parallel_mode": self.parallel_config.enable_parallel and len(config.methods) > 1
            }
        
        return final_results
    
    def create_preset_configs(self) -> Dict[str, FlexibleRetrievalConfig]:
        """Create simple per-method and all-method retrieval presets."""
        available_methods = list(self.retrievers.keys())
        configs = {}
        
        for method in available_methods:
            config_name = f"{method.value}_only"
            configs[config_name] = FlexibleRetrievalConfig(methods=[method])
        
        if len(available_methods) >= 2:
            from itertools import combinations
            for method1, method2 in combinations(available_methods, 2):
                config_name = f"{method1.value}_{method2.value}_rrf"
                configs[config_name] = FlexibleRetrievalConfig(
                    methods=[method1, method2],
                    fusion_method="rrf"
                )
            
            if len(available_methods) >= 3:
                all_methods_str = "_".join([m.value for m in available_methods])
                configs[f"{all_methods_str}_rrf"] = FlexibleRetrievalConfig(
                    methods=available_methods,
                    fusion_method="rrf"
                )
                
                configs[f"{all_methods_str}_weighted"] = FlexibleRetrievalConfig(
                    methods=available_methods,
                    fusion_method="weighted"
                )
        
        return configs
    
    def smart_search_with_quantification(self, 
                                       query: str,
                                       methods: Union[str, List[str], List[RetrievalMethod]] = None,
                                       top_k: int = 10,
                                       rerank_method: Optional[str] = "baai",
                                       rerank_params: Optional[Dict[str, Any]] = None,
                                       **kwargs) -> Dict[str, Any]:
        """Run retrieval plus lightweight consistency-based confidence scoring.

        The quantification compares top sparse and dense UID sets before
        reranking. It does not call an LLM and does not alter retrieval indexes.
        """
        target_rerank_method = rerank_method or "baai"
        ScoreFusion.ensure_sync_rerank_allowed(
            target_rerank_method,
            context="MultiRetriever.smart_search_with_quantification",
        )

        METHOD_CATEGORY_MAPPING = {
            RetrievalMethod.BM25: "sparse",
            RetrievalMethod.SPLADE: "sparse",
            
            RetrievalMethod.COSINE_SIMILARITY: "dense",
        }

        if methods is None:
            methods = [
                RetrievalMethod.BM25, 
                RetrievalMethod.SPLADE, 
                RetrievalMethod.COSINE_SIMILARITY
            ]

        
        recall_k = max(50, top_k * 3)
        
        multi_results = self.search_multi(
            query=query,
            methods=methods,
            top_k=recall_k,
            **kwargs
        )
        
        
        sparse_uids = set()
        dense_uids = set()
        
        consistency_check_depth = 10
        
        for method, results in multi_results.results_by_method.items():
            category = METHOD_CATEGORY_MAPPING.get(method)
            
            current_top_uids = {r.unit.uid for r in results[:consistency_check_depth]}
            
            if category == "sparse":
                sparse_uids.update(current_top_uids)
            elif category == "dense":
                dense_uids.update(current_top_uids)
        
        if not sparse_uids or not dense_uids:
            consistency_score = 0.0
        else:
            intersection = sparse_uids.intersection(dense_uids)
            union = sparse_uids.union(dense_uids)
            consistency_score = len(intersection) / len(union) if union else 0.0
            
        
        fused_results = ScoreFusion.rrf_fusion(multi_results)
        
        reranked_results = []
        confidence_score = 0.0
        
        
        current_rerank_params = dict(rerank_params or {})
        
        if 'max_candidates' not in current_rerank_params:
            current_rerank_params['max_candidates'] = recall_k
        else:
            current_rerank_params['max_candidates'] = max(current_rerank_params['max_candidates'], recall_k)
        
        if fused_results:
            try:
                reranked_results = self._execute_reranking(
                    query=query,
                    base_results=fused_results,
                    rerank_method=target_rerank_method,
                    rerank_params=current_rerank_params,
                    top_k=top_k  
                )
                
                if reranked_results:
                    top_score = reranked_results[0][1]
                    confidence_score = max(0.0, min(1.0, float(top_score)))
            except Exception as e:
                if ScoreFusion.is_vllm_sync_rerank_error(e):
                    raise
                logger.warning(f"Quantified retrieval reranking failed; using fused results: {e}")
                reranked_results = fused_results[:top_k]
                confidence_score = 0.5 
        
        final_score = (0.7 * confidence_score) + (0.3 * consistency_score)
        
        if final_score >= 0.7:
            diagnosis = "High Confidence (Direct Answer)"
        elif final_score >= 0.4:
            diagnosis = "Medium Confidence (Context Expansion Recommended)"
        else:
            diagnosis = "Low Confidence (Query Rewrite Recommended)"
            
        return {
            "results": reranked_results,
            "quantification": {
                "consistency_score": round(consistency_score, 4),
                "confidence_score": round(confidence_score, 4),
                "final_score": round(final_score, 4),
                "diagnosis": diagnosis,
                "details": {
                    "sparse_set_size": len(sparse_uids),
                    "dense_set_size": len(dense_uids),
                    "intersection_count": len(sparse_uids.intersection(dense_uids))
                }
            }
        }

    async def smart_search_with_quantification_async(self,
                                       query: str,
                                       methods: Union[str, List[str], List[RetrievalMethod]] = None,
                                       top_k: int = 10,
                                       rerank_method: Optional[str] = "baai",
                                       rerank_params: Optional[Dict[str, Any]] = None,
                                       **kwargs) -> Dict[str, Any]:
        """Async variant of retrieval plus consistency-based confidence scoring."""
        method_category_mapping = {
            RetrievalMethod.BM25: "sparse",
            RetrievalMethod.SPLADE: "sparse",
            RetrievalMethod.COSINE_SIMILARITY: "dense",
        }

        if methods is None:
            methods = [
                RetrievalMethod.BM25,
                RetrievalMethod.SPLADE,
                RetrievalMethod.COSINE_SIMILARITY,
            ]

        recall_k = max(50, top_k * 3)
        multi_results = await self.search_multi_async(
            query=query,
            methods=parse_retrieval_methods(methods),
            top_k=recall_k,
            **kwargs,
        )

        sparse_uids = set()
        dense_uids = set()
        consistency_check_depth = 10

        for method, results in multi_results.results_by_method.items():
            category = method_category_mapping.get(method)
            current_top_uids = {result.unit.uid for result in results[:consistency_check_depth]}
            if category == "sparse":
                sparse_uids.update(current_top_uids)
            elif category == "dense":
                dense_uids.update(current_top_uids)

        if not sparse_uids or not dense_uids:
            consistency_score = 0.0
        else:
            intersection = sparse_uids.intersection(dense_uids)
            union = sparse_uids.union(dense_uids)
            consistency_score = len(intersection) / len(union) if union else 0.0

        fused_results = await asyncio.to_thread(ScoreFusion.rrf_fusion, multi_results)
        reranked_results = []
        confidence_score = 0.0

        target_rerank_method = rerank_method or "baai"
        current_rerank_params = dict(rerank_params or {})
        current_rerank_params["max_candidates"] = max(
            int(current_rerank_params.get("max_candidates", recall_k)),
            recall_k,
        )

        if fused_results:
            try:
                reranked_results = await self._execute_reranking_async(
                    query=query,
                    base_results=fused_results,
                    rerank_method=target_rerank_method,
                    rerank_params=current_rerank_params,
                    top_k=top_k,
                )
                if reranked_results:
                    confidence_score = max(0.0, min(1.0, float(reranked_results[0][1])))
            except Exception as exc:
                if ScoreFusion.is_vllm_sync_rerank_error(exc):
                    raise
                logger.warning(f"Async quantified retrieval reranking failed; using fused results: {exc}")
                reranked_results = fused_results[:top_k]
                confidence_score = 0.5

        final_score = (0.7 * confidence_score) + (0.3 * consistency_score)
        if final_score >= 0.7:
            diagnosis = "High Confidence (Direct Answer)"
        elif final_score >= 0.4:
            diagnosis = "Medium Confidence (Context Expansion Recommended)"
        else:
            diagnosis = "Low Confidence (Query Rewrite Recommended)"

        return {
            "results": reranked_results,
            "quantification": {
                "consistency_score": round(consistency_score, 4),
                "confidence_score": round(confidence_score, 4),
                "final_score": round(final_score, 4),
                "diagnosis": diagnosis,
                "details": {
                    "sparse_set_size": len(sparse_uids),
                    "dense_set_size": len(dense_uids),
                    "intersection_count": len(sparse_uids.intersection(dense_uids)),
                },
            },
        }

def cleanup_retrieval_resources(retriever: Optional[MultiRetriever] = None):
    """Release retriever wrappers, reranker caches, and GPU allocator state."""
    try:
        if retriever:
            retriever.cleanup_unused_retrievers()
        
        
        ScoreFusion.cleanup_rerankers()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        gc.collect()
        
        logger.info("Retrieval resources cleaned up.")
        
    except Exception as e:
        logger.warning(f"Error while cleaning retrieval resources: {e}")
        
