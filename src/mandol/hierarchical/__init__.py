"""
Hierarchical memory retrieval components.

High-level memory construction now lives in ``mandol.auto_builder``. This
package root intentionally exposes only the layer metadata and retrieval-facing
interfaces for already-built hierarchical memory spaces. Clustering helpers live
in ``mandol.cluster`` and are re-exported here for compatibility.
"""

from .hierarchical_retriever import (
    HierarchicalRetriever,
    HierarchicalSearchConfig,
    HierarchicalSearchDirection,
    HierarchicalSearchResult,
    HierarchicalSearchStrategy,
    LayerConfig,
    LayerRelationship,
    MemoryLevel,
    SummaryType,
    create_hierarchical_retriever,
)
from ..cluster import (
    analyze_cluster_characteristics,
    cluster_nodes,
    find_clusters_with_dbscan,
    find_communities_with_leiden,
    get_cluster_component_status,
    optimize_dbscan_parameters,
)

__all__ = [
    "MemoryLevel",
    "SummaryType",
    "LayerConfig",
    "LayerRelationship",
    "HierarchicalRetriever",
    "HierarchicalSearchConfig",
    "HierarchicalSearchDirection",
    "HierarchicalSearchStrategy",
    "HierarchicalSearchResult",
    "create_hierarchical_retriever",
    "cluster_nodes",
    "find_communities_with_leiden",
    "find_clusters_with_dbscan",
    "optimize_dbscan_parameters",
    "analyze_cluster_characteristics",
    "get_hierarchical_component_status",
]


def get_hierarchical_component_status():
    """Return availability for the retrieval-facing hierarchical package."""
    status = {
        "core_components": {
            "hierarchical_retriever": True,
            "hierarchical_retriever_async": True,
        },
        "generation_location": "mandol.auto_builder",
        "available_memory_levels": [level.value for level in MemoryLevel],
        "available_summary_types": [summary_type.value for summary_type in SummaryType],
        "available_search_directions": [direction.value for direction in HierarchicalSearchDirection],
        "available_search_strategies": [strategy.value for strategy in HierarchicalSearchStrategy],
    }
    status["optional_dependencies"] = get_cluster_component_status()["optional_dependencies"]
    return status
