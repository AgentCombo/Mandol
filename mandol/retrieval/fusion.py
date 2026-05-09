"""Reciprocal Rank Fusion (RRF) for combining multiple ranked lists.

Merges ranked results from dense, BM25, and sparse retrieval into a single
fused ranking using the RRF formula: score = Σ 1/(k + rank).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from ..domain.memory_unit import MemoryUnit


@dataclass(slots=True)
class RankedUnit:
    """A single ranked result from one retrieval method.

    Attributes:
        unit: The MemoryUnit that was retrieved.
        score: Relevance score from the originating method.
    """
    unit: MemoryUnit
    score: float


def rrf_fusion(
    ranked_lists: Sequence[Sequence[RankedUnit]],
    *,
    k: int = 60,
    top_k: int,
    method_names: Sequence[str],
) -> List[tuple[MemoryUnit, float, Dict[str, int]]]:
    """Fuse multiple ranked lists using Reciprocal Rank Fusion.

    For each result, computes RRF score = Σ 1/(k + rank) across all
    participating methods, then sorts descending by fused score.

    Args:
        ranked_lists: Per-method list of RankedUnit results.
        k: RRF smoothing constant (default 60).
        top_k: Number of fused results to return.
        method_names: Human-readable labels for each ranked list.

    Returns:
        Sorted list of (unit, fused_score, {method: rank}) tuples.
    """
    uid_to_unit: Dict[str, MemoryUnit] = {}
    uid_to_score: Dict[str, float] = {}
    uid_to_ranks: Dict[str, Dict[str, int]] = {}

    for method_name, ranked in zip(method_names, ranked_lists):
        ordered = list(ranked)
        ordered.sort(key=lambda x: x.score, reverse=True)
        for rank, item in enumerate(ordered, start=1):
            uid = str(item.unit.uid)
            uid_to_unit[uid] = item.unit
            uid_to_score[uid] = uid_to_score.get(uid, 0.0) + 1.0 / float(int(k) + int(rank))
            ranks = uid_to_ranks.get(uid)
            if ranks is None:
                ranks = {}
                uid_to_ranks[uid] = ranks
            ranks[method_name] = int(rank)

    fused = [(uid_to_unit[uid], float(score), uid_to_ranks.get(uid, {})) for uid, score in uid_to_score.items()]
    fused.sort(key=lambda x: x[1], reverse=True)
    return fused[: max(0, int(top_k))]
