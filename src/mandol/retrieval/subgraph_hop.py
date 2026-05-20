"""Prototype: query → dense/BM25/sparse fusion seeds → weighted multi-hop graph expansion."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from ..application.semantic_graph import SemanticGraphService
from ..domain.coref_graph_constants import DEFAULT_REL_WEIGHTS
from ..domain.memory_unit import MemoryUnit
from ..domain.types import Uid
from .pipeline import HybridRetriever, HybridRetrieverConfig
from .types import ReasoningStep, SearchHit


@dataclass
class SubgraphHopConfig:
    """Fusion and graph expansion for cross-session / multi-hop QA prototypes."""

    base: HybridRetrieverConfig = field(default_factory=HybridRetrieverConfig)
    # Graph branch
    max_hops: int = 2
    hop_decay: float = 0.85
    rel_weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_REL_WEIGHTS))
    # How many base retrieval hits become BFS seeds
    seed_top_k: int = 5
    # final = (1 - graph_branch_weight) * norm(retrieval) + graph_branch_weight * norm(graph_boost)
    graph_branch_weight: float = 0.45


@dataclass
class SubgraphHopHit:
    """A retrieval hit enriched with a multi-hop reasoning path.

    Attributes:
        unit: The matched MemoryUnit.
        final_score: Combined retrieval + graph-boost score.
        scores: Per-signal score breakdown.
        reasoning_path: Ordered list of graph-traversal steps from seed to hit.
    """
    unit: MemoryUnit
    final_score: float
    scores: Dict[str, float]
    reasoning_path: List[ReasoningStep] = field(default_factory=list)


class SubgraphHopRetriever:
    """HybridRetriever plus weighted multi-rel-type expansion (2-hop by default)."""

    def __init__(
        self,
        *,
        hybrid: HybridRetriever,
        config: Optional[SubgraphHopConfig] = None,
    ) -> None:
        """Initialize with a hybrid retriever and optional configuration.

        Args:
            hybrid: The HybridRetriever instance for base retrieval.
            config: Optional configuration for hop count, decay, and weights.
        """
        self._hybrid = hybrid
        self._config = config or SubgraphHopConfig()

    @property
    def semantic_graph(self) -> SemanticGraphService:
        return self._hybrid.semantic_graph

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        space_names: Optional[Sequence[str]] = None,
        recursive: bool = True,
        use_rerank: bool = True,
    ) -> List[SubgraphHopHit]:
        """Run base retrieval then fuse with multi-hop graph expansion scores.

        Args:
            query: Search query text.
            top_k: Number of final results to return.
            space_names: Optional space filter for base retrieval.
            recursive: Include child spaces in the filter.
            use_rerank: Apply reranking to base retrieval.

        Returns:
            List of SubgraphHopHit objects with final fused scores.
        """
        cfg = self._config
        seeds_k = max(int(cfg.seed_top_k), int(top_k))
        base_hits = self._hybrid.search(
            query,
            top_k=seeds_k,
            space_names=space_names,
            recursive=recursive,
            use_rerank=use_rerank,
        )
        if not base_hits:
            return []

        graph_boost = self._accumulate_graph_boost(base_hits[:seeds_k])
        ret_max = max(h.final_score for h in base_hits) or 1.0
        gb_max = max((v[0] for v in graph_boost.values())) if graph_boost else 0.0
        gb_norm_denom = gb_max if gb_max > 1e-12 else 1.0

        uid_to_hit: Dict[str, SearchHit] = {str(h.unit.uid): h for h in base_hits}
        all_uids = set(uid_to_hit.keys()) | set(graph_boost.keys())

        alpha = float(cfg.graph_branch_weight)
        alpha = max(0.0, min(1.0, alpha))

        ranked: List[SubgraphHopHit] = []
        for uid in all_uids:
            unit = self.semantic_graph.semantic_map.get_unit(Uid(uid))
            if unit is None:
                continue
            hit = uid_to_hit.get(uid)
            ret_s = float(hit.final_score) / ret_max if hit else 0.0
            gb_data = graph_boost.get(uid)
            gb_raw = gb_data[0] if gb_data else 0.0
            reasoning_path = gb_data[1] if gb_data else []
            gb = float(gb_raw) / gb_norm_denom
            final = (1.0 - alpha) * ret_s + alpha * gb
            scores = dict(hit.scores) if hit else {}
            scores["retrieval_norm"] = ret_s
            scores["graph_boost"] = gb
            scores["graph_boost_raw"] = float(gb_raw)
            ranked.append(
                SubgraphHopHit(
                    unit=unit, final_score=final, scores=scores, reasoning_path=reasoning_path
                )
            )

        ranked.sort(key=lambda x: x.final_score, reverse=True)
        return ranked[: max(0, int(top_k))]

    def _accumulate_graph_boost(
        self, seed_hits: List[SearchHit]
    ) -> Dict[str, tuple[float, List[ReasoningStep]]]:
        """Max-path aggregation from multiple seeds (multiplicative edge × hop decay).

        Returns:
            Dict mapping uid -> (best_score, best_reasoning_path)
        """
        cfg = self._config
        rel_weights = cfg.rel_weights
        max_hops = max(1, int(cfg.max_hops))
        hop_decay = float(cfg.hop_decay)
        graph = self.semantic_graph
        best: Dict[str, tuple[float, List[ReasoningStep]]] = {}

        for h in seed_hits:
            seed = str(h.unit.uid)
            seed_score = max(float(h.final_score), 1e-12)
            from collections import deque

            q: deque[tuple[str, int, float, List[ReasoningStep]]] = deque()
            q.append((seed, 0, seed_score, []))
            seen_states: set[tuple[str, int]] = set()

            while q:
                uid, depth, acc, path = q.popleft()
                st = (uid, depth)
                if st in seen_states:
                    continue
                seen_states.add(st)

                prev_score, prev_path = best.get(uid, (0.0, []))
                if acc > prev_score:
                    best[uid] = (acc, path)
                if depth >= max_hops:
                    continue

                for rel, w in rel_weights.items():
                    w_eff = float(w) * hop_decay
                    for direction in ("out", "in"):
                        try:
                            neigh = graph.get_explicit_neighbors(
                                [uid], rel_type=rel, direction=direction
                            )
                        except (RuntimeError, KeyError, AttributeError):
                            neigh = []
                        for nu in neigh:
                            nid = str(nu.uid)
                            next_acc = acc * w_eff
                            if direction == "out":
                                step = ReasoningStep(
                                    source_uid=Uid(uid),
                                    target_uid=Uid(nid),
                                    rel_type=rel,
                                    rel_weight=w_eff,
                                    direction=direction,
                                )
                            else:
                                step = ReasoningStep(
                                    source_uid=Uid(nid),
                                    target_uid=Uid(uid),
                                    rel_type=rel,
                                    rel_weight=w_eff,
                                    direction=direction,
                                )
                            q.append((nid, depth + 1, next_acc, path + [step]))

        return best
