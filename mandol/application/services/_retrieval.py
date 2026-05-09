"""Memory retrieval service extracted from MemorySystem.

Performs multi-group retrieval (base/entity/event/summary) with auto-build
trigger support and Cross-Encoder reranking.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Set, Tuple

from ...domain.memory_unit import MemoryUnit
from ...domain.types import SpaceName
from ...retrieval.types import SearchHit

logger = logging.getLogger(__name__)

# Retrieval group keys for the four memory categories.
RETRIEVAL_GROUP_BASE = "base"
RETRIEVAL_GROUP_ENTITY = "entity"
RETRIEVAL_GROUP_EVENT = "event"
RETRIEVAL_GROUP_SUMMARY = "summary"


class MemoryRetrievalService:
    """Handles all retrieval operations across MemorySystem.

    Supports holistic retrieval (4-group pipeline), per-space retrieval,
    and view-based retrieval (base_memory/entity_relation/event_causal/etc.).

    Args:
        semantic_map: SemanticMapService for ANN and rerank access.
        graph: SemanticGraphService for BFS expansion access.
        naming: SpaceNamingPolicy for constructing space names.
        root: Root SpaceName.
        config: MemorySystemConfig for similarity thresholds.
    """

    def __init__(
        self,
        semantic_map,
        graph,
        naming,
        root: SpaceName,
        config,
    ):
        self._semantic_map = semantic_map
        self._graph = graph
        self._naming = naming
        self._root = root
        self._cfg = config
        self._hybrid_retriever = None

    def _get_retrieval_groups(
        self,
    ) -> Dict[str, Tuple[str, List[SpaceName]]]:
        return {
            RETRIEVAL_GROUP_BASE: (
                "base memory",
                [self._naming.base_memory(self._root)],
            ),
            RETRIEVAL_GROUP_EVENT: (
                "events",
                [self._naming.episodic_event(self._root)],
            ),
            RETRIEVAL_GROUP_ENTITY: (
                "entities",
                [self._naming.knowledge_entity(self._root)],
            ),
            RETRIEVAL_GROUP_SUMMARY: (
                "summaries and insights",
                [
                    self._naming.episodic_summary(self._root),
                    self._naming.knowledge_summary(self._root),
                    self._naming.insights(self._root),
                ],
            ),
        }

    def _are_high_level_spaces_empty(self) -> bool:
        """Check if all high-level memory spaces (entity, event, summary) are empty.

        Returns:
            True when every high-level space contains zero units, indicating
            that build_high_level has never been called or produced no output.
        """
        high_level_spaces = [
            self._naming.episodic_event(self._root),
            self._naming.knowledge_entity(self._root),
            self._naming.episodic_summary(self._root),
            self._naming.knowledge_summary(self._root),
            self._naming.insights(self._root),
        ]
        for space_name in high_level_spaces:
            units = self._semantic_map.get_units_in_spaces([space_name])
            if units:
                return False
        return True

    def _is_base_memory_empty(self) -> bool:
        """Check if the base memory space has no units.

        Returns:
            True when base_memory contains zero units, meaning no memory
            has been ingested yet.
        """
        base_space = self._naming.base_memory(self._root)
        units = self._semantic_map.get_units_in_spaces([base_space])
        return len(units) == 0

    def holistic_retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        use_rerank: bool = True,
        auto_build_if_empty: bool = True,
        build_trigger: Optional[Callable[[], None]] = None,
    ) -> List[SearchHit]:
        """Retrieve across all memory groups with optional auto-build.

        Searches base memory, entity, event, and summary spaces in
        parallel, deduplicates by UID, then optionally reranks the
        combined candidates.

        Args:
            query: Natural language search query.
            top_k: Maximum number of results to return.
            use_rerank: Enable Cross-Encoder reranking (default True).
            auto_build_if_empty: Trigger build_high_level when high-level
                spaces are empty but base memory exists.
            build_trigger: Callable that runs build_high_level (typically
                MemorySystem.build_high_level).

        Returns:
            Ranked list of SearchHit objects ordered by final_score.
        """
        if auto_build_if_empty and build_trigger is not None:
            if not self._is_base_memory_empty() and self._are_high_level_spaces_empty():
                logger.info(
                    "High-level memory spaces are empty, triggering auto-build"
                )
                build_trigger()

        groups = self._get_retrieval_groups()
        all_candidates: List[MemoryUnit] = []
        seen: Set[str] = set()

        for group_key, (group_label, space_names) in groups.items():
            group_hits = self._search_group(
                query,
                space_names,
                top_k=max(1, int(top_k) * 3),
                use_rerank=False,
            )
            for unit, _score in group_hits:
                uid = str(unit.uid)
                if uid not in seen:
                    seen.add(uid)
                    all_candidates.append(unit)

        if not all_candidates:
            return []

        if use_rerank and self._semantic_map.get_reranker() is not None:
            reranked = self._semantic_map.get_reranker().rerank(
                query, all_candidates, top_k=max(1, int(top_k))
            )
            out: List[SearchHit] = []
            for unit, rerank_score in reranked:
                out.append(
                    SearchHit(
                        unit=unit,
                        final_score=float(rerank_score),
                        scores={"rerank": float(rerank_score)},
                        ranks={},
                    )
                )
            return out

        all_candidates.sort(key=lambda u: u.metadata.get("timestamp", ""), reverse=True)
        hits: List[SearchHit] = []
        for u in all_candidates[: max(0, int(top_k))]:
            hits.append(
                SearchHit(
                    unit=u,
                    final_score=1.0,
                    scores={},
                    ranks={},
                )
            )
        return hits

    def retrieve_in_space(
        self,
        query: str,
        space_name: str,
        *,
        top_k: int = 10,
        use_rerank: bool = True,
    ) -> List[SearchHit]:
        """Retrieve within a single named space.

        Args:
            query: Natural language search query.
            space_name: Name of the space to search.
            top_k: Maximum number of results.
            use_rerank: Enable Cross-Encoder reranking.

        Returns:
            List of (MemoryUnit, score) tuples from the hybrid retriever.
        """
        return self._search_group(
            query,
            [SpaceName(space_name)],
            top_k=top_k,
            use_rerank=use_rerank,
        )

    def retrieve_by_view(
        self,
        query: str,
        view: str,
        *,
        top_k: int = 10,
        use_rerank: bool = True,
    ) -> List[SearchHit]:
        """Retrieve using a named view that maps to one or more spaces.

        Args:
            query: Natural language search query.
            view: View name — one of: base_memory, entity_relation,
                event_causal, emotional, episodic, knowledge, procedural,
                insights.
            top_k: Maximum number of results.
            use_rerank: Enable Cross-Encoder reranking.

        Returns:
            List of (MemoryUnit, score) tuples from the hybrid retriever.

        Raises:
            ValueError: If *view* is not a recognised view name.
        """
        view_space_map = {
            "base_memory": [self._naming.base_memory(self._root)],
            "entity_relation": [self._naming.knowledge_entity(self._root)],
            "event_causal": [self._naming.episodic_event(self._root)],
            "emotional": [self._naming.emotional(self._root)],
            "episodic": [self._naming.episodic_summary(self._root)],
            "knowledge": [self._naming.knowledge_summary(self._root)],
            "procedural": [self._naming.procedural(self._root)],
            "insights": [self._naming.insights(self._root)],
        }

        space_names = view_space_map.get(view)
        if space_names is None:
            raise ValueError(
                f"Unknown view: {view}. "
                f"Available views: {', '.join(view_space_map.keys())}"
            )

        return self._search_group(
            query,
            space_names,
            top_k=top_k,
            use_rerank=use_rerank,
        )

    def _search_group(
        self,
        query: str,
        space_names: List[SpaceName],
        top_k: int,
        use_rerank: bool,
    ) -> List[Tuple[MemoryUnit, float]]:
        from ...retrieval.pipeline import HybridRetriever, HybridRetrieverConfig

        if self._hybrid_retriever is None:
            self._hybrid_retriever = HybridRetriever(
                semantic_map=self._semantic_map,
                graph=self._graph,
            )
        self._hybrid_retriever._config = HybridRetrieverConfig(
            top_k=top_k,
            use_rerank=use_rerank,
        )
        results = self._hybrid_retriever.search(query, space_names=space_names)
        return [(hit.unit, hit.final_score) for hit in results]
