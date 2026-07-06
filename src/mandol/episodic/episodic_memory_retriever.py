"""Thin episodic memory retriever wrapper over unified smart_search."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..core.memory_space_registry import TowerSpace
from ..core.memory_unit import MemoryUnit
from ..retrieval.advance_retriever import MultiRetriever
from ..retrieval.rerank_manager import RerankerManager
from ..retrieval.score_fusion import ScoreFusion
from ..utils.logging_config import create_module_logger

logger = create_module_logger("episodic.episodic_memory_retriever")


@dataclass
class EpisodicRetrievalConfig:
    semantic_methods: Optional[List[str]] = None
    fusion_method: str = "rrf"
    rerank_method: str = "baai"
    top_k: int = 15
    target_space: str = TowerSpace.EPISODIC_ROOT.value
    enable_time_injection: bool = True

    def __post_init__(self) -> None:
        if self.semantic_methods is None:
            self.semantic_methods = ["bm25", "cosine_similarity", "splade"]


class EpisodicMemoryRetriever:
    """Retrieval-facing episodic tower wrapper.

    All retrieval, scoring, and fusion are delegated to ``MultiRetriever.smart_search``.
    This class keeps episodic formatting helpers for prompt construction only.
    """

    def __init__(
        self,
        semantic_graph: Any,
        config: Optional[EpisodicRetrievalConfig] = None,
        reranker_configs: Optional[Dict[str, str]] = None,
        reranker_manager: Optional[RerankerManager] = None,
    ):
        self.semantic_graph = semantic_graph
        self.config = config or EpisodicRetrievalConfig()
        self.reranker_configs = reranker_configs or {
            "baai": "BAAI/bge-reranker-v2-m3",
            "qwen": "Qwen/Qwen3-Reranker-0.6B",
        }
        self.reranker_manager = reranker_manager
        self._multi_retriever: Optional[MultiRetriever] = None
        logger.info("EpisodicMemoryRetriever initialized as thin smart_search wrapper")

    def episodic_search(self, query: str, **kwargs) -> List[Tuple[MemoryUnit, float]]:
        top_k = kwargs.pop("top_k", self.config.top_k)
        space_names = kwargs.pop("space_names", [self.config.target_space])
        methods = kwargs.pop("methods", self.config.semantic_methods)
        fusion_method = kwargs.pop("fusion_method", self.config.fusion_method)
        rerank_method = kwargs.pop("rerank_method", self.config.rerank_method)

        ScoreFusion.ensure_sync_rerank_allowed(
            rerank_method,
            context="EpisodicMemoryRetriever.episodic_search",
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
            logger.error("Episodic smart_search failed: %s", exc)
            return []

    def search(self, query: str, top_k: int = 10, **kwargs) -> List[Tuple[MemoryUnit, float]]:
        return self.episodic_search(query, top_k=top_k, **kwargs)

    async def episodic_search_async(self, query: str, **kwargs) -> List[Tuple[MemoryUnit, float]]:
        top_k = kwargs.pop("top_k", self.config.top_k)
        space_names = kwargs.pop("space_names", [self.config.target_space])
        methods = kwargs.pop("methods", self.config.semantic_methods)
        fusion_method = kwargs.pop("fusion_method", self.config.fusion_method)
        rerank_method = kwargs.pop("rerank_method", self.config.rerank_method)

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
            logger.error("Episodic async smart_search failed: %s", exc)
            return []

    async def search_async(self, query: str, top_k: int = 10, **kwargs) -> List[Tuple[MemoryUnit, float]]:
        return await self.episodic_search_async(query, top_k=top_k, **kwargs)

    def build_context_with_time_injection(self, results: List[Tuple[MemoryUnit, float]]) -> str:
        if not results:
            return "No episodic memories found."
        context_parts = []
        for index, (unit, _score) in enumerate(results, 1):
            timestamp = self._extract_time(unit)
            marker = f"[{timestamp}]" if timestamp else "[time unknown]"
            context_parts.append(f"Fact {index}: {marker} {self._extract_text(unit)}")
        return "\n\n".join(context_parts)

    def build_context_with_metadata(self, results: List[Tuple[MemoryUnit, float]]) -> str:
        if not results:
            return "No episodic memories found."
        context_parts = []
        for index, (unit, score) in enumerate(results, 1):
            metadata = dict(unit.metadata or {})
            context_parts.append(
                f"Fact {index} (score={float(score):.4f}, uid={unit.uid}, metadata={metadata}): "
                f"{self._extract_text(unit)}"
            )
        return "\n".join(context_parts)

    def get_stats(self) -> Dict[str, Any]:
        stats = {
            "retriever_type": "EpisodicMemoryRetriever",
            "config": {
                "target_space": self.config.target_space,
                "semantic_methods": self.config.semantic_methods,
                "fusion_method": self.config.fusion_method,
                "rerank_method": self.config.rerank_method,
                "top_k": self.config.top_k,
            },
            "components": {"multi_retriever_available": self._multi_retriever is not None},
        }
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
            logger.debug("Failed to get episodic space stats: %s", exc)
        return stats

    def update_config(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                logger.warning("Unknown episodic config field: %s", key)

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

    @staticmethod
    def _extract_text(unit: MemoryUnit) -> str:
        raw = unit.raw_data or {}
        for field in ("text_content", "content", "message", "description", "summary", "fact"):
            value = raw.get(field)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return str(raw)

    @staticmethod
    def _extract_time(unit: MemoryUnit) -> str:
        raw = unit.raw_data or {}
        metadata = unit.metadata or {}
        for field in ("timestamp", "time", "date", "event_time", "created"):
            value = metadata.get(field) or raw.get(field)
            if value:
                return str(value)
        return ""


def create_episodic_retriever(
    semantic_graph: Any,
    top_k: int = 15,
    rerank_method: str = "baai",
    target_space: str = TowerSpace.EPISODIC_ROOT.value,
    enable_time_injection: bool = True,
    reranker_manager: Optional[RerankerManager] = None,
) -> EpisodicMemoryRetriever:
    config = EpisodicRetrievalConfig(
        top_k=top_k,
        rerank_method=rerank_method,
        target_space=target_space,
        enable_time_injection=enable_time_injection,
    )
    return EpisodicMemoryRetriever(
        semantic_graph=semantic_graph,
        config=config,
        reranker_manager=reranker_manager,
    )
