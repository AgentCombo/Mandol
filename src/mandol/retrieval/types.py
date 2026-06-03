"""Type definitions for the retrieval pipeline.

Defines SearchHit, ReasoningStep, CausalStep, CausalChainResult,
TimelineResult, SessionContextResult, RelationshipInfo, EntitySubgraphResult,
EvidenceChainResult, CorefTraceResult, SummaryEvidenceChainResult,
EntityInvolvementResult, ReasoningHit, and other immutable result types used
throughout the retrieval subsystem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

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


@dataclass(slots=True)
class CausalStep:
    """One link in an event causal chain.

    Attributes:
        source_uid: UID of the source event in this causal link.
        target_uid: UID of the target event reached by this link.
        causal_type: The type of causal relationship (e.g. "CAUSES", "CAUSED_BY").
        confidence: Confidence score for this causal link [0, 1].
        direction: Traversal direction — 'forward' or 'backward'.
    """
    source_uid: Uid
    target_uid: Uid
    causal_type: str
    confidence: float = 1.0
    direction: str = "forward"


@dataclass(slots=True)
class CausalChainResult:
    """Result of retrieving an event causal chain.

    Attributes:
        chain: Ordered list of CausalStep links forming the causal chain.
        root_event: The root event MemoryUnit, if one was identified.
        leaf_events: Leaf event MemoryUnit nodes at the end of the chain.
    """
    chain: list[CausalStep] = field(default_factory=list)
    root_event: Optional[MemoryUnit] = None
    leaf_events: list[MemoryUnit] = field(default_factory=list)


@dataclass(slots=True)
class TimelineResult:
    """Result of retrieving an entity timeline.

    Attributes:
        events: List of (MemoryUnit, timestamp_str) tuples ordered chronologically.
        entity: The entity MemoryUnit whose timeline was retrieved.
    """
    events: list[tuple[MemoryUnit, str]] = field(default_factory=list)
    entity: Optional[MemoryUnit] = None


@dataclass(slots=False)
class SessionContextResult:
    """Result of retrieving session context.

    NOTE: slots=False is required because this type is self-referencing
    (adjacent_sessions is a list of SessionContextResult).

    Attributes:
        session_units: MemoryUnits belonging to this session.
        session_id: Identifier for this session.
        adjacent_sessions: Adjacent SessionContextResult instances.
    """
    session_units: list[MemoryUnit] = field(default_factory=list)
    session_id: str = ""
    adjacent_sessions: list[SessionContextResult] = field(default_factory=list)


@dataclass(slots=True)
class RelationshipInfo:
    """Relationship details for an entity subgraph edge.

    Attributes:
        source_uid: UID of the source entity.
        target_uid: UID of the target entity.
        rel_type: Relationship type label.
        properties: Additional key-value properties for this relationship.
    """
    source_uid: Uid
    target_uid: Uid
    rel_type: str
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EntitySubgraphResult:
    """Result of retrieving an entity subgraph.

    Attributes:
        center_entity: The central entity MemoryUnit.
        neighbors: Neighbouring entity MemoryUnits.
        relationships: Details of all relationships in the subgraph.
        depth_map: Mapping from entity UID to its hop distance from center.
    """
    center_entity: Optional[MemoryUnit] = None
    neighbors: list[MemoryUnit] = field(default_factory=list)
    relationships: list[RelationshipInfo] = field(default_factory=list)
    depth_map: Dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class EvidenceChainResult:
    """Result of tracing evidence for a memory unit.

    Attributes:
        source: The source MemoryUnit for which evidence was traced.
        evidence: Evidence MemoryUnits ordered by evidential strength.
        depth_map: Mapping from evidence UID to its hop distance from source.
    """
    source: Optional[MemoryUnit] = None
    evidence: list[MemoryUnit] = field(default_factory=list)
    depth_map: Dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class CorefTraceResult:
    """Result of tracing coreference chains.

    Attributes:
        source: The source MemoryUnit for which coreference was traced.
        canonical_entities: Canonical entity MemoryUnits discovered.
        canonical_events: Canonical event MemoryUnits discovered.
        coref_chain: Ordered list of MemoryUnits in the coreference chain.
    """
    source: Optional[MemoryUnit] = None
    canonical_entities: list[MemoryUnit] = field(default_factory=list)
    canonical_events: list[MemoryUnit] = field(default_factory=list)
    coref_chain: list[MemoryUnit] = field(default_factory=list)


@dataclass(slots=True)
class SummaryEvidenceChainResult:
    """Result of retrieving a summary evidence chain.

    Attributes:
        summary: The summary MemoryUnit.
        evidence_units: MemoryUnits serving as evidence for the summary.
        related_entities: Entity MemoryUnits related to the summary.
        related_events: Event MemoryUnits related to the summary.
    """
    summary: Optional[MemoryUnit] = None
    evidence_units: list[MemoryUnit] = field(default_factory=list)
    related_entities: list[MemoryUnit] = field(default_factory=list)
    related_events: list[MemoryUnit] = field(default_factory=list)


@dataclass(slots=True)
class EntityInvolvementResult:
    """Result of retrieving entity involvement across events.

    Attributes:
        entity: The entity MemoryUnit whose involvement was retrieved.
        events: List of (MemoryUnit, role_str) tuples showing event participation.
        causal_chains: Causal chains the entity participates in, each as a list of MemoryUnits.
    """
    entity: Optional[MemoryUnit] = None
    events: list[tuple[MemoryUnit, str]] = field(default_factory=list)
    causal_chains: list[list[MemoryUnit]] = field(default_factory=list)


@dataclass(slots=True)
class ReasoningHit:
    """A hit from retrieve_with_reasoning_path.

    Similar to SearchHit but augmented with a reasoning path showing
    the multi-hop traversal steps that led to this result.

    Attributes:
        unit: The matched MemoryUnit.
        final_score: Aggregated relevance score after fusion / reranking.
        scores: Per-signal scores (e.g. dense, bm25, sparse, rerank).
        reasoning_path: Ordered list of ReasoningStep nodes forming the path.
    """
    unit: MemoryUnit
    final_score: float
    scores: Dict[str, float] = field(default_factory=dict)
    reasoning_path: list[ReasoningStep] = field(default_factory=list)
