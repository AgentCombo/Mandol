"""Clustering utilities for memory units and semantic graphs."""

from importlib.util import find_spec
from typing import Any

from .base_clusterer import (
    BaseClusterer,
    ClusterMethod,
    DBSCANClusterer,
    LeidenClusterer,
    cluster_nodes,
    create_clusterer,
    normalize_cluster_method,
)

__all__ = [
    "BaseClusterer",
    "ClusterMethod",
    "DBSCANClusterer",
    "LeidenClusterer",
    "cluster_nodes",
    "create_clusterer",
    "normalize_cluster_method",
    "find_communities_with_leiden",
    "find_clusters_with_dbscan",
    "optimize_dbscan_parameters",
    "analyze_cluster_characteristics",
    "get_cluster_component_status",
]

_DBSCAN_EXPORTS = {
    "find_clusters_with_dbscan",
    "optimize_dbscan_parameters",
    "analyze_cluster_characteristics",
}
_LEIDEN_EXPORTS = {"find_communities_with_leiden"}


def __getattr__(name: str) -> Any:
    if name in _DBSCAN_EXPORTS:
        from . import dbscan_method

        return getattr(dbscan_method, name)
    if name in _LEIDEN_EXPORTS:
        from . import leiden_method

        return getattr(leiden_method, name)
    raise AttributeError(f"module 'mandol.cluster' has no attribute {name!r}")


def get_cluster_component_status():
    """Return optional dependency availability for clustering backends."""
    return {
        "available_methods": [method.value for method in ClusterMethod],
        "optional_dependencies": {
            "leiden_algorithm": (
                find_spec("igraph") is not None and find_spec("leidenalg") is not None
            ),
            "dbscan_algorithm": find_spec("sklearn") is not None,
        },
    }
