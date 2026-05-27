"""Application layer for the Mandol memory system.

This package implements the core business logic that coordinates between domain
models and infrastructure adapters. It provides the main MemorySystem facade
along with services for semantic mapping, graph operations, session management,
document chunking, entity/event extraction, insight reduction, and pipeline
orchestration.
"""

from .semantic_graph import SemanticGraphService
from .semantic_map import SemanticMapService
from .legacy.mdsg_pipeline import run_mdsg_pipeline
from .legacy.multidim_semantic_graph import (
    DimensionBuilder,
    LayoutNormalizationDimension,
    MultiDimBuildContext,
    MultiDimSemanticGraphBuilder,
    SemanticSimilarityDimension,
    SpaceNamingPolicy,
)

__all__ = [
    "SemanticGraphService",
    "SemanticMapService",
    "run_mdsg_pipeline",
    "DimensionBuilder",
    "LayoutNormalizationDimension",
    "MultiDimBuildContext",
    "MultiDimSemanticGraphBuilder",
    "SemanticSimilarityDimension",
    "SpaceNamingPolicy",
]
