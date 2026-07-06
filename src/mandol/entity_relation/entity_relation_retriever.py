"""Thin entity-relation tower retriever over unified smart_search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..core.memory_space_registry import TowerSpace
from ..core.memory_unit import MemoryUnit
from ..retrieval.advance_retriever import MultiRetriever
from ..retrieval.rerank_manager import RerankerManager
from ..retrieval.score_fusion import ScoreFusion
from ..utils.logging_config import create_module_logger

logger = create_module_logger("entity_relation.entity_relation_retriever")


@dataclass
class GraphRetrievalConfig:
    semantic_methods: Optional[List[str]] = None
    fusion_method: str = "rrf"
    rerank_method: str = "baai"
    topk_similarity: int = 15
    topk_graph: int = 0
    final_topk: int = 10
    use_entity_relation: bool = False
    enable_relation_filtering: bool = True
    relation_filter_threshold: float = 0.6
    rerank_candidates: int = 50
    target_space: str = TowerSpace.GRAPH_ROOT.value

    def __post_init__(self) -> None:
        if self.semantic_methods is None:
            self.semantic_methods = ["bm25", "cosine_similarity", "splade"]


@dataclass
class EntityRelationResult:
    """Compatibility result object for older graph benchmark code."""

    unit: MemoryUnit
    score: float
    metadata: Optional[Dict[str, Any]] = None


class EntityRelationGraphRetriever:
    """Retrieval-facing graph tower wrapper.

    Entity/relation extraction and graph construction are handled by auto builders.
    Runtime retrieval is intentionally thin and delegates to smart_search with a
    graph tower memory-space filter.
    """

    def __init__(
        self,
        semantic_graph: Any,
        config: Optional[GraphRetrievalConfig] = None,
        reranker_configs: Optional[Dict[str, str]] = None,
        reranker_manager: Optional[RerankerManager] = None,
    ):
        self.semantic_graph = semantic_graph
        self.config = config or GraphRetrievalConfig()
        self.reranker_configs = reranker_configs or {
            "baai": "BAAI/bge-reranker-v2-m3",
            "qwen": "Qwen/Qwen3-Reranker-0.6B",
        }
        self.reranker_manager = reranker_manager
        self._multi_retriever: Optional[MultiRetriever] = None
        logger.info("EntityRelationGraphRetriever initialized as thin smart_search wrapper")

    def entity_relation_search(self, query: str, **kwargs) -> List[Tuple[MemoryUnit, float]]:
        top_k = kwargs.pop("final_topk", self.config.final_topk)
        top_k = kwargs.pop("top_k", top_k)
        space_names = kwargs.pop("space_names", [self.config.target_space])
        methods = kwargs.pop("methods", self.config.semantic_methods)
        fusion_method = kwargs.pop("fusion_method", self.config.fusion_method)
        rerank_method = kwargs.pop("rerank_method", self.config.rerank_method)
        kwargs.pop("topk_similarity", None)
        kwargs.pop("topk_graph", None)
        kwargs.pop("use_entity_relation", None)

        ScoreFusion.ensure_sync_rerank_allowed(
            rerank_method,
            context="EntityRelationGraphRetriever.entity_relation_search",
        )

        try:
            results = self._get_multi_retriever().smart_search(
                query=query,
                methods=methods,
                fusion_method=fusion_method,
                rerank_method=rerank_method,
                top_k=top_k,
                return_detailed=False,
                space_names=space_names,
                **kwargs,
            )
            return self._normalize_results(results)
        except Exception as exc:
            logger.error("Graph tower smart_search failed: %s", exc)
            return []

    def search(self, query: str, top_k: int = 10, **kwargs) -> List[Tuple[MemoryUnit, float]]:
        return self.entity_relation_search(query, top_k=top_k, **kwargs)

    async def entity_relation_search_async(self, query: str, **kwargs) -> List[Tuple[MemoryUnit, float]]:
        top_k = kwargs.pop("final_topk", self.config.final_topk)
        top_k = kwargs.pop("top_k", top_k)
        space_names = kwargs.pop("space_names", [self.config.target_space])
        methods = kwargs.pop("methods", self.config.semantic_methods)
        fusion_method = kwargs.pop("fusion_method", self.config.fusion_method)
        rerank_method = kwargs.pop("rerank_method", self.config.rerank_method)
        kwargs.pop("topk_similarity", None)
        kwargs.pop("topk_graph", None)
        kwargs.pop("use_entity_relation", None)

        try:
            results = await self._get_multi_retriever().smart_search_async(
                query=query,
                methods=methods,
                fusion_method=fusion_method,
                rerank_method=rerank_method,
                top_k=top_k,
                return_detailed=False,
                space_names=space_names,
                **kwargs,
            )
            return self._normalize_results(results)
        except NotImplementedError:
            raise
        except Exception as exc:
            logger.error("Graph tower async smart_search failed: %s", exc)
            return []

    async def search_async(self, query: str, top_k: int = 10, **kwargs) -> List[Tuple[MemoryUnit, float]]:
        return await self.entity_relation_search_async(query, top_k=top_k, **kwargs)

    def get_stats(self) -> Dict[str, Any]:
        stats = {
            "retriever_type": "EntityRelationGraphRetriever",
            "config": {
                "target_space": self.config.target_space,
                "semantic_methods": self.config.semantic_methods,
                "fusion_method": self.config.fusion_method,
                "rerank_method": self.config.rerank_method,
                "topk_similarity": self.config.topk_similarity,
                "topk_graph": self.config.topk_graph,
                "final_topk": self.config.final_topk,
                "use_entity_relation": self.config.use_entity_relation,
            },
            "components": {
                "multi_retriever_available": self._multi_retriever is not None,
                "entity_retriever_available": False,
                "reranker_manager": "external" if self.reranker_manager else "internal",
            },
        }
        if self._multi_retriever is not None:
            try:
                stats["multi_retriever_stats"] = self._multi_retriever.get_stats()
            except Exception as exc:
                logger.debug("Failed to get MultiRetriever stats: %s", exc)
        try:
            space = self.semantic_graph.semantic_map.get_memory_space(self.config.target_space)
            if space:
                stats["target_space_stats"] = {
                    "space_name": self.config.target_space,
                    "unit_count": len(space.get_all_unit_uids(recursive=True)),
                    "direct_units": len(space.get_unit_uids()),
                    "child_spaces": len(space.get_child_space_names()),
                }
        except Exception as exc:
            logger.debug("Failed to get graph target space stats: %s", exc)
        return stats

    def update_config(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                logger.warning("Unknown graph retriever config field: %s", key)

    def _get_multi_retriever(self) -> MultiRetriever:
        if self._multi_retriever is not None:
            return self._multi_retriever
        if hasattr(self.semantic_graph, "get_multi_retriever"):
            self._multi_retriever = self.semantic_graph.get_multi_retriever()
        else:
            self._multi_retriever = MultiRetriever(
                self.semantic_graph,
                preload_rerankers=False,
                reranker_configs=self.reranker_configs,
                reranker_manager=self.reranker_manager,
            )
        return self._multi_retriever

    @staticmethod
    def _normalize_results(results: Any) -> List[Tuple[MemoryUnit, float]]:
        if isinstance(results, dict):
            results = results.get("results", [])
        normalized: List[Tuple[MemoryUnit, float]] = []
        for item in results or []:
            if isinstance(item, tuple) and len(item) >= 2 and isinstance(item[0], MemoryUnit):
                normalized.append((item[0], float(item[1])))
            elif hasattr(item, "unit"):
                normalized.append((item.unit, float(getattr(item, "score", 0.0))))
        return normalized


def create_graph_retriever(
    semantic_graph: Any,
    topk_similarity: int = 15,
    topk_graph: int = 0,
    final_topk: int = 10,
    use_entity_relation: bool = False,
    rerank_method: str = "baai",
    target_space: str = TowerSpace.GRAPH_ROOT.value,
    reranker_manager: Optional[RerankerManager] = None,
) -> EntityRelationGraphRetriever:
    config = GraphRetrievalConfig(
        topk_similarity=topk_similarity,
        topk_graph=topk_graph,
        final_topk=final_topk,
        use_entity_relation=use_entity_relation,
        rerank_method=rerank_method,
        target_space=target_space,
    )
    return EntityRelationGraphRetriever(
        semantic_graph=semantic_graph,
        config=config,
        reranker_manager=reranker_manager,
    )


class EntityRelationRetriever(EntityRelationGraphRetriever):
    """Backward-compatible graph tower retriever facade.

    Older LoCoMo benchmark scripts construct ``EntityRelationRetriever`` with
    relation-extraction flags and expect ``search`` to return objects exposing
    ``unit`` and ``score`` attributes. The public graph tower now delegates to
    unified smart_search; this facade keeps the benchmark call shape stable.
    """

    def __init__(
        self,
        semantic_graph: Any,
        use_bert_ner: bool = True,
        enable_relation_filtering: bool = True,
        relation_filter_threshold: float = 0.6,
        topk_similarity: int = 15,
        topk_graph: int = 0,
        final_topk: int = 10,
        rerank_method: str = "baai",
        target_space: str = TowerSpace.GRAPH_ROOT.value,
        reranker_configs: Optional[Dict[str, str]] = None,
        reranker_manager: Optional[RerankerManager] = None,
        **kwargs: Any,
    ):
        del use_bert_ner
        config = kwargs.pop("config", None) or GraphRetrievalConfig(
            topk_similarity=topk_similarity,
            topk_graph=topk_graph,
            final_topk=final_topk,
            enable_relation_filtering=enable_relation_filtering,
            relation_filter_threshold=relation_filter_threshold,
            rerank_method=rerank_method,
            target_space=target_space,
        )
        super().__init__(
            semantic_graph=semantic_graph,
            config=config,
            reranker_configs=reranker_configs,
            reranker_manager=reranker_manager,
        )

    def search(self, query: str, top_k: int = 10, **kwargs) -> List[EntityRelationResult]:
        results = self.entity_relation_search(query, top_k=top_k, **kwargs)
        return [
            EntityRelationResult(unit=unit, score=score, metadata={"retriever": "entity_relation_graph"})
            for unit, score in results
        ]
