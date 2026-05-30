"""Hybrid retrieval pipeline with dense, BM25, and sparse fusion.

Provides text extraction/tokenization, BM25 scoring, TF-IDF sparse retrieval,
reciprocal rank fusion (RRF), multi-hop graph expansion, and the main
HybridRetriever that orchestrates the full retrieval pipeline.
"""

from .bm25 import Bm25Retriever
from .pipeline import HybridRetriever, HybridRetrieverConfig
from .sparse import TfidfSparseRetriever
from .subgraph_hop import SubgraphHopConfig, SubgraphHopHit, SubgraphHopRetriever
from .types import (
    CausalChainResult,
    CausalStep,
    CorefTraceResult,
    EntityInvolvementResult,
    EntitySubgraphResult,
    EvidenceChainResult,
    ReasoningHit,
    ReasoningStep,
    RelationshipInfo,
    SearchHit,
    SessionContextResult,
    SummaryEvidenceChainResult,
    TimelineResult,
)

__all__ = [
    "Bm25Retriever",
    "CausalChainResult",
    "CausalStep",
    "CorefTraceResult",
    "EntityInvolvementResult",
    "EntitySubgraphResult",
    "EvidenceChainResult",
    "HybridRetriever",
    "HybridRetrieverConfig",
    "ReasoningHit",
    "ReasoningStep",
    "RelationshipInfo",
    "SearchHit",
    "SessionContextResult",
    "SubgraphHopConfig",
    "SubgraphHopHit",
    "SubgraphHopRetriever",
    "SummaryEvidenceChainResult",
    "TfidfSparseRetriever",
    "TimelineResult",
]
