"""Thin hierarchical retriever wrapper over the unified smart_search stack."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from ..core.memory_space_registry import TowerSpace
from ..core.memory_unit import MemoryUnit
from ..retrieval.advance_retriever import MultiRetriever
from ..retrieval.retrieval_interface import BaseRetriever, RetrievalMethod, RetrievalResult
from ..retrieval.score_fusion import ScoreFusion
from ..utils.logging_config import create_module_logger

logger = create_module_logger("hierarchical.hierarchical_retriever")


class MemoryLevel(Enum):
    """Canonical hierarchical levels kept for public API compatibility."""

    OBSERVATION = "L0_Observation"
    SUMMARY = "L1_Summary"
    INSIGHT = "L2_Insight"


class SummaryType(Enum):
    """Legacy summary-type labels retained as metadata-only values."""

    EPISODIC = "episodic"
    PROCEDURAL = "procedural"
    KNOWLEDGE = "knowledge"
    EMOTIONAL = "emotional"


@dataclass
class LayerConfig:
    level: MemoryLevel
    space_name: str
    description: str = ""


@dataclass
class LayerRelationship:
    source_level: MemoryLevel
    target_level: MemoryLevel
    relationship_type: str
    direction: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class HierarchicalSearchDirection(Enum):
    TOP_DOWN = "top_down"
    BOTTOM_UP = "bottom_up"
    CROSS_LAYER = "cross_layer"
    LAYER_SPECIFIC = "layer_specific"


class HierarchicalSearchStrategy(Enum):
    CASCADING = "cascading"
    PARALLEL = "parallel"
    ADAPTIVE = "adaptive"
    FOCUSED = "focused"


@dataclass
class HierarchicalSearchConfig:
    direction: HierarchicalSearchDirection = HierarchicalSearchDirection.CROSS_LAYER
    strategy: HierarchicalSearchStrategy = HierarchicalSearchStrategy.ADAPTIVE
    layer_weights: Optional[Dict[MemoryLevel, float]] = None
    top_k_per_layer: int = 10
    fusion_method: str = "rrf"

    def __post_init__(self) -> None:
        if self.layer_weights is None:
            self.layer_weights = {
                MemoryLevel.OBSERVATION: 1.0,
                MemoryLevel.SUMMARY: 1.0,
                MemoryLevel.INSIGHT: 1.0,
            }


@dataclass
class HierarchicalSearchResult:
    query: str
    direction: HierarchicalSearchDirection
    strategy: HierarchicalSearchStrategy
    results_by_layer: Dict[MemoryLevel, List[RetrievalResult]]
    final_results: List[RetrievalResult]
    search_path: List[Dict[str, Any]] = field(default_factory=list)
    relationships_found: List[Dict[str, Any]] = field(default_factory=list)
    total_searched_units: int = 0
    execution_time: float = 0.0
    layer_statistics: Dict[MemoryLevel, Dict[str, Any]] = field(default_factory=dict)


class HierarchicalRetriever(BaseRetriever):
    """Retrieval-facing hierarchical tower wrapper.

    The historical hierarchical manager/interface generated summaries and hand-rolled
    layer expansion. Generation now lives in ``mandol.auto_builder``; this wrapper
    only routes layer-scoped retrieval into ``MultiRetriever.smart_search``.
    """

    LAYER_TO_SPACE = {
        MemoryLevel.OBSERVATION: TowerSpace.HIERARCHICAL_L0.value,
        MemoryLevel.SUMMARY: TowerSpace.HIERARCHICAL_L1.value,
        MemoryLevel.INSIGHT: TowerSpace.HIERARCHICAL_L2.value,
    }

    def __init__(
        self,
        semantic_system: Any,
        multi_retriever: Optional[MultiRetriever] = None,
        default_config: Optional[HierarchicalSearchConfig] = None,
        target_space: str = TowerSpace.HIERARCHICAL_ROOT.value,
    ):
        if hasattr(semantic_system, "semantic_system"):
            semantic_system = semantic_system.semantic_system
        self.semantic_system = semantic_system
        self.target_space = target_space
        self.multi_retriever = multi_retriever
        self.default_config = default_config or HierarchicalSearchConfig()
        self.search_stats = {"total_searches": 0, "average_execution_time": 0.0}

    def get_method_type(self) -> RetrievalMethod:
        return RetrievalMethod.HYBRID

    def search(self, query: str, top_k: int = 10, **kwargs) -> List[RetrievalResult]:
        result = self.hierarchical_search(query=query, top_k=top_k, **kwargs)
        return result.final_results

    async def search_async(self, query: str, top_k: int = 10, **kwargs) -> List[RetrievalResult]:
        result = await self.hierarchical_search_async(query=query, top_k=top_k, **kwargs)
        return result.final_results

    def hierarchical_search(
        self,
        query: str,
        config: Optional[HierarchicalSearchConfig] = None,
        top_k: int = 10,
        enable_cache: bool = True,
        rerank_method: str = "baai",
        rerank_candidates: int = 50,
        space_names: Optional[List[str]] = None,
        retrieval_methods: Optional[Union[str, List[str], List[RetrievalMethod]]] = None,
        **kwargs,
    ) -> HierarchicalSearchResult:
        del enable_cache, rerank_candidates
        start_time = time.time()
        cfg = config or self.default_config
        effective_methods = retrieval_methods or ["bm25", "cosine_similarity", "splade"]
        requested_spaces = space_names or [self.target_space]
        ScoreFusion.ensure_sync_rerank_allowed(
            rerank_method,
            context="HierarchicalRetriever.hierarchical_search",
        )

        if self.target_space in requested_spaces:
            layers = [MemoryLevel.OBSERVATION, MemoryLevel.SUMMARY, MemoryLevel.INSIGHT]
        else:
            layers = [self._layer_for_space(space) for space in requested_spaces]
            layers = [layer for layer in layers if layer is not None]
            if not layers:
                layers = [MemoryLevel.OBSERVATION, MemoryLevel.SUMMARY, MemoryLevel.INSIGHT]

        per_layer_k = max(top_k, cfg.top_k_per_layer)
        results_by_layer: Dict[MemoryLevel, List[RetrievalResult]] = {}
        for layer in layers:
            layer_space = self.LAYER_TO_SPACE[layer]
            tuples = self._smart_search(
                query=query,
                top_k=per_layer_k,
                methods=effective_methods,
                fusion_method=cfg.fusion_method,
                rerank_method=rerank_method,
                space_names=[layer_space],
                **kwargs,
            )
            results_by_layer[layer] = self._to_retrieval_results(tuples, layer)

        final_results = self._fuse_layer_results(results_by_layer, cfg, top_k)
        execution_time = time.time() - start_time
        self._update_stats(execution_time)

        return HierarchicalSearchResult(
            query=query,
            direction=cfg.direction,
            strategy=cfg.strategy,
            results_by_layer=results_by_layer,
            final_results=final_results,
            search_path=[
                {"layer": layer.value, "space_names": [self.LAYER_TO_SPACE[layer]], "count": len(results)}
                for layer, results in results_by_layer.items()
            ],
            total_searched_units=sum(len(results) for results in results_by_layer.values()),
            execution_time=execution_time,
            layer_statistics={
                layer: {"count": len(results), "space": self.LAYER_TO_SPACE[layer]}
                for layer, results in results_by_layer.items()
            },
        )

    async def hierarchical_search_async(
        self,
        query: str,
        config: Optional[HierarchicalSearchConfig] = None,
        top_k: int = 10,
        enable_cache: bool = True,
        rerank_method: str = "baai",
        rerank_candidates: int = 50,
        space_names: Optional[List[str]] = None,
        retrieval_methods: Optional[Union[str, List[str], List[RetrievalMethod]]] = None,
        **kwargs,
    ) -> HierarchicalSearchResult:
        del enable_cache, rerank_candidates
        start_time = time.time()
        cfg = config or self.default_config
        effective_methods = retrieval_methods or ["bm25", "cosine_similarity", "splade"]
        requested_spaces = space_names or [self.target_space]

        if self.target_space in requested_spaces:
            layers = [MemoryLevel.OBSERVATION, MemoryLevel.SUMMARY, MemoryLevel.INSIGHT]
        else:
            layers = [self._layer_for_space(space) for space in requested_spaces]
            layers = [layer for layer in layers if layer is not None]
            if not layers:
                layers = [MemoryLevel.OBSERVATION, MemoryLevel.SUMMARY, MemoryLevel.INSIGHT]

        per_layer_k = max(top_k, cfg.top_k_per_layer)

        async def search_layer(layer: MemoryLevel) -> Tuple[MemoryLevel, List[RetrievalResult]]:
            layer_space = self.LAYER_TO_SPACE[layer]
            tuples = await self._smart_search_async(
                query=query,
                top_k=per_layer_k,
                methods=effective_methods,
                fusion_method=cfg.fusion_method,
                rerank_method=rerank_method,
                space_names=[layer_space],
                **kwargs,
            )
            return layer, self._to_retrieval_results(tuples, layer)

        layer_pairs = await asyncio.gather(*(search_layer(layer) for layer in layers))
        results_by_layer = {layer: results for layer, results in layer_pairs}
        final_results = self._fuse_layer_results(results_by_layer, cfg, top_k)
        execution_time = time.time() - start_time
        self._update_stats(execution_time)

        return HierarchicalSearchResult(
            query=query,
            direction=cfg.direction,
            strategy=cfg.strategy,
            results_by_layer=results_by_layer,
            final_results=final_results,
            search_path=[
                {"layer": layer.value, "space_names": [self.LAYER_TO_SPACE[layer]], "count": len(results)}
                for layer, results in results_by_layer.items()
            ],
            total_searched_units=sum(len(results) for results in results_by_layer.values()),
            execution_time=execution_time,
            layer_statistics={
                layer: {"count": len(results), "space": self.LAYER_TO_SPACE[layer]}
                for layer, results in results_by_layer.items()
            },
        )

    def retrieve_single_layer_units(
        self,
        query: str,
        level: str,
        top_k: int = 5,
        return_scores: bool = True,
        retrieval_methods: Optional[List[str]] = None,
        rerank_method: str = "baai",
        **kwargs,
    ) -> List[Union[MemoryUnit, Tuple[MemoryUnit, float]]]:
        layer = self._normalize_level(level)
        ScoreFusion.ensure_sync_rerank_allowed(
            rerank_method,
            context="HierarchicalRetriever.retrieve_single_layer_units",
        )
        tuples = self._smart_search(
            query=query,
            top_k=top_k,
            methods=retrieval_methods or ["bm25", "cosine_similarity", "splade"],
            rerank_method=rerank_method,
            space_names=[self.LAYER_TO_SPACE[layer]],
            **kwargs,
        )
        return tuples if return_scores else [unit for unit, _score in tuples]

    async def retrieve_single_layer_units_async(
        self,
        query: str,
        level: str,
        top_k: int = 5,
        return_scores: bool = True,
        retrieval_methods: Optional[List[str]] = None,
        rerank_method: str = "baai",
        **kwargs,
    ) -> List[Union[MemoryUnit, Tuple[MemoryUnit, float]]]:
        layer = self._normalize_level(level)
        tuples = await self._smart_search_async(
            query=query,
            top_k=top_k,
            methods=retrieval_methods or ["bm25", "cosine_similarity", "splade"],
            rerank_method=rerank_method,
            space_names=[self.LAYER_TO_SPACE[layer]],
            **kwargs,
        )
        return tuples if return_scores else [unit for unit, _score in tuples]

    def retrieve_hierarchical_units(
        self,
        query: str,
        top_k_l0: int = 10,
        top_k_l1: int = 5,
        top_k_l2: int = 3,
        return_scores: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        layer_specs = {
            "l0": (MemoryLevel.OBSERVATION, top_k_l0),
            "l1": (MemoryLevel.SUMMARY, top_k_l1),
            "l2": (MemoryLevel.INSIGHT, top_k_l2),
        }
        output: Dict[str, Any] = {}
        for key, (layer, top_k) in layer_specs.items():
            results = self.retrieve_single_layer_units(
                query=query,
                level=layer.value,
                top_k=top_k,
                return_scores=return_scores,
                **kwargs,
            )
            output[key] = results
        return output

    async def retrieve_hierarchical_units_async(
        self,
        query: str,
        top_k_l0: int = 10,
        top_k_l1: int = 5,
        top_k_l2: int = 3,
        return_scores: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        layer_specs = {
            "l0": (MemoryLevel.OBSERVATION, top_k_l0),
            "l1": (MemoryLevel.SUMMARY, top_k_l1),
            "l2": (MemoryLevel.INSIGHT, top_k_l2),
        }

        async def retrieve_layer(key: str, layer: MemoryLevel, layer_top_k: int) -> Tuple[str, Any]:
            results = await self.retrieve_single_layer_units_async(
                query=query,
                level=layer.value,
                top_k=layer_top_k,
                return_scores=return_scores,
                **kwargs,
            )
            return key, results

        layer_results = await asyncio.gather(
            *(retrieve_layer(key, layer, layer_top_k) for key, (layer, layer_top_k) in layer_specs.items())
        )
        return {key: results for key, results in layer_results}

    def search_by_concept(self, concept: str, top_k: int = 10, **kwargs) -> HierarchicalSearchResult:
        return self.hierarchical_search(concept, top_k=top_k, **kwargs)

    async def search_by_concept_async(self, concept: str, top_k: int = 10, **kwargs) -> HierarchicalSearchResult:
        return await self.hierarchical_search_async(concept, top_k=top_k, **kwargs)

    def search_by_facts(self, facts_query: str, top_k: int = 10, **kwargs) -> HierarchicalSearchResult:
        return self.hierarchical_search(facts_query, top_k=top_k, **kwargs)

    async def search_by_facts_async(self, facts_query: str, top_k: int = 10, **kwargs) -> HierarchicalSearchResult:
        return await self.hierarchical_search_async(facts_query, top_k=top_k, **kwargs)

    def search_for_insights(self, topic: str, top_k: int = 10, **kwargs) -> HierarchicalSearchResult:
        return self.hierarchical_search(
            topic,
            top_k=top_k,
            space_names=[TowerSpace.HIERARCHICAL_L2.value],
            **kwargs,
        )

    async def search_for_insights_async(self, topic: str, top_k: int = 10, **kwargs) -> HierarchicalSearchResult:
        return await self.hierarchical_search_async(
            topic,
            top_k=top_k,
            space_names=[TowerSpace.HIERARCHICAL_L2.value],
            **kwargs,
        )

    def explore_topic(self, topic: str, top_k: int = 15, **kwargs) -> HierarchicalSearchResult:
        return self.hierarchical_search(topic, top_k=top_k, **kwargs)

    async def explore_topic_async(self, topic: str, top_k: int = 15, **kwargs) -> HierarchicalSearchResult:
        return await self.hierarchical_search_async(topic, top_k=top_k, **kwargs)

    def get_search_statistics(self) -> Dict[str, Any]:
        return dict(self.search_stats)

    def clear_cache(self) -> None:
        return None

    def _get_multi_retriever(self) -> MultiRetriever:
        if self.multi_retriever is not None:
            return self.multi_retriever
        if hasattr(self.semantic_system, "get_multi_retriever"):
            self.multi_retriever = self.semantic_system.get_multi_retriever()
        else:
            self.multi_retriever = MultiRetriever(self.semantic_system)
        return self.multi_retriever

    def _smart_search(self, **kwargs) -> List[Tuple[MemoryUnit, float]]:
        ScoreFusion.ensure_sync_rerank_allowed(
            kwargs.get("rerank_method"),
            context="HierarchicalRetriever._smart_search",
        )
        try:
            results = self._get_multi_retriever().smart_search(return_detailed=False, **kwargs)
        except Exception as exc:
            logger.warning("Hierarchical smart_search failed: %s", exc)
            return []
        return self._normalize_tuple_results(results)

    async def _smart_search_async(self, **kwargs) -> List[Tuple[MemoryUnit, float]]:
        try:
            results = await self._get_multi_retriever().smart_search_async(return_detailed=False, **kwargs)
        except NotImplementedError:
            raise
        except Exception as exc:
            logger.warning("Hierarchical async smart_search failed: %s", exc)
            return []
        return self._normalize_tuple_results(results)

    def _to_retrieval_results(
        self, results: List[Tuple[MemoryUnit, float]], layer: MemoryLevel
    ) -> List[RetrievalResult]:
        return [
            RetrievalResult(
                unit=unit,
                score=float(score),
                method=RetrievalMethod.HYBRID,
                method_details={"tower": "hierarchical", "layer": layer.value},
            )
            for unit, score in results
        ]

    def _fuse_layer_results(
        self,
        results_by_layer: Dict[MemoryLevel, List[RetrievalResult]],
        config: HierarchicalSearchConfig,
        top_k: int,
    ) -> List[RetrievalResult]:
        fused: Dict[str, RetrievalResult] = {}
        weights = config.layer_weights or {}
        for layer, results in results_by_layer.items():
            weight = float(weights.get(layer, 1.0))
            for result in results:
                weighted_score = float(result.score) * weight
                if result.unit.uid not in fused or weighted_score > fused[result.unit.uid].score:
                    fused[result.unit.uid] = RetrievalResult(
                        unit=result.unit,
                        score=weighted_score,
                        method=result.method,
                        method_details={**result.method_details, "layer_weight": weight},
                    )
        return sorted(fused.values(), key=lambda item: item.score, reverse=True)[:top_k]

    @staticmethod
    def _normalize_tuple_results(results: Any) -> List[Tuple[MemoryUnit, float]]:
        if isinstance(results, dict):
            results = results.get("results", [])
        normalized: List[Tuple[MemoryUnit, float]] = []
        for item in results or []:
            if isinstance(item, tuple) and len(item) >= 2 and isinstance(item[0], MemoryUnit):
                normalized.append((item[0], float(item[1])))
            elif hasattr(item, "unit"):
                normalized.append((item.unit, float(getattr(item, "score", 0.0))))
        return normalized

    @classmethod
    def _normalize_level(cls, level: Union[str, MemoryLevel]) -> MemoryLevel:
        if isinstance(level, MemoryLevel):
            return level
        key = str(level).strip().lower()
        mapping = {
            "l0": MemoryLevel.OBSERVATION,
            "observation": MemoryLevel.OBSERVATION,
            "l0_observation": MemoryLevel.OBSERVATION,
            "l1": MemoryLevel.SUMMARY,
            "summary": MemoryLevel.SUMMARY,
            "l1_summary": MemoryLevel.SUMMARY,
            "l2": MemoryLevel.INSIGHT,
            "insight": MemoryLevel.INSIGHT,
            "l2_insight": MemoryLevel.INSIGHT,
        }
        if key not in mapping:
            raise ValueError(f"Unknown hierarchical level: {level}")
        return mapping[key]

    @classmethod
    def _layer_for_space(cls, space_name: str) -> Optional[MemoryLevel]:
        for layer, candidate in cls.LAYER_TO_SPACE.items():
            if space_name == candidate or space_name.endswith(layer.value):
                return layer
        return None

    def _update_stats(self, execution_time: float) -> None:
        count = self.search_stats["total_searches"] + 1
        previous = self.search_stats["average_execution_time"]
        self.search_stats["total_searches"] = count
        self.search_stats["average_execution_time"] = previous + (execution_time - previous) / count


def create_hierarchical_retriever(
    semantic_system: Any,
    multi_retriever: Optional[MultiRetriever] = None,
) -> HierarchicalRetriever:
    """Create a retrieval-facing hierarchical tower wrapper."""
    return HierarchicalRetriever(semantic_system, multi_retriever=multi_retriever)
