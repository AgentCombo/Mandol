"""Type definitions for the retrieval pipeline.

Defines SearchHit, ReasoningStep, and other immutable result types used
throughout the retrieval subsystem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from ..domain.memory_unit import MemoryUnit
from ..domain.types import Uid


@dataclass(slots=True)
class ReasoningStep:
    """A single step in a multi-hop graph traversal path.

    Attributes:
        source_uid: UID of the source unit in this traversal step.
        target_uid: UID of the target unit reached by this step.
        rel_type: Relationship type label (e.g. SEMANTIC_SIMILAR, COREF).
        rel_weight: Normalised weight of this relationship edge.
        direction: Traversal direction — 'outgoing' or 'incoming'.
    """
    source_uid: Uid
    target_uid: Uid
    rel_type: str
    rel_weight: float
    direction: str


@dataclass(slots=True)
class SearchHit:
    """A single result from the hybrid retrieval pipeline.

    Attributes:
        unit: The matched MemoryUnit.
        final_score: Aggregated relevance score after fusion / reranking.
        scores: Per-signal scores (e.g. dense, bm25, sparse, rerank).
        ranks: Per-signal rank positions.
        debug: Optional debug metadata for traceability.
    """
    unit: MemoryUnit
    final_score: float
    scores: Dict[str, float] = field(default_factory=dict)
    ranks: Dict[str, int] = field(default_factory=dict)
    debug: Dict[str, Any] = field(default_factory=dict)
