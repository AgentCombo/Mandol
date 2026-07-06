from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import Enum
from typing import Any, Dict, List, Optional

from ..core.memory_unit import MemoryUnit


class ClusterMethod(str, Enum):
    """Supported clustering backends."""

    LEIDEN = "leiden"
    DBSCAN = "dbscan"


class BaseClusterer(ABC):
    """Unified clustering interface used by graph and memory-space callers."""

    method: ClusterMethod

    @abstractmethod
    def cluster(self, source: Any = None, **kwargs) -> Dict[int, List[str]]:
        pass


class LeidenClusterer(BaseClusterer):
    method = ClusterMethod.LEIDEN

    def cluster(self, source: Any = None, **kwargs) -> Dict[int, List[str]]:
        from .leiden_method import find_communities_with_leiden

        graph = kwargs.pop("graph", None)
        if graph is None:
            graph = _extract_graph_source(source)
        return find_communities_with_leiden(graph, **kwargs)


class DBSCANClusterer(BaseClusterer):
    method = ClusterMethod.DBSCAN

    def cluster(self, source: Any = None, **kwargs) -> Dict[int, List[str]]:
        from .dbscan_method import find_clusters_with_dbscan

        units = kwargs.pop("units", None)
        if units is None:
            units = _extract_units(source)
        return find_clusters_with_dbscan(units, **kwargs)


_CLUSTERERS = {
    ClusterMethod.LEIDEN: LeidenClusterer,
    ClusterMethod.DBSCAN: DBSCANClusterer,
}


def normalize_cluster_method(method: str | ClusterMethod) -> ClusterMethod:
    if isinstance(method, ClusterMethod):
        return method

    method_value = str(method).lower().strip()
    aliases = {
        "community": ClusterMethod.LEIDEN,
        "communities": ClusterMethod.LEIDEN,
        "density": ClusterMethod.DBSCAN,
    }
    if method_value in aliases:
        return aliases[method_value]

    try:
        return ClusterMethod(method_value)
    except ValueError as exc:
        supported = ", ".join(cluster_method.value for cluster_method in ClusterMethod)
        raise ValueError(f"未知聚类方法: {method}. 支持的方法: {supported}") from exc


def create_clusterer(method: str | ClusterMethod = ClusterMethod.LEIDEN) -> BaseClusterer:
    cluster_method = normalize_cluster_method(method)
    return _CLUSTERERS[cluster_method]()


def cluster_nodes(
    source: Any = None,
    method: str | ClusterMethod = ClusterMethod.LEIDEN,
    **kwargs,
) -> Dict[int, List[str]]:
    """
    Unified clustering entrypoint.

    Args:
        source: SemanticGraph, MemorySpace, SemanticMap, NetworkX/rustworkx graph,
            or a sequence of MemoryUnit objects depending on the selected method.
        method: ``"leiden"`` for graph community detection or ``"dbscan"`` for
            embedding-density clustering.
        **kwargs: Method-specific options, forwarded to the concrete backend.

    Returns:
        Mapping from cluster/community id to UID list. DBSCAN uses ``-1`` for noise.
    """
    return create_clusterer(method).cluster(source, **kwargs)


def _extract_graph_source(source: Any) -> Any:
    if source is None:
        raise ValueError("Leiden 聚类需要提供图对象或 SemanticGraph 实例")

    if hasattr(source, "rx_graph") and hasattr(source, "_index_to_uid"):
        return source

    if hasattr(source, "nodes") and hasattr(source, "edges"):
        return source

    if hasattr(source, "node_indices") and hasattr(source, "edge_list"):
        return source

    if hasattr(source, "nx_graph"):
        return source.nx_graph

    raise ValueError("Leiden 聚类需要 SemanticGraph、NetworkX 图或 rustworkx 图")


def _extract_units(source: Any) -> List[MemoryUnit]:
    if source is None:
        return []

    if _is_memory_unit_sequence(source):
        return list(source)

    if hasattr(source, "get_all_units"):
        return list(source.get_all_units())

    if hasattr(source, "semantic_map") and hasattr(source.semantic_map, "get_all_units"):
        return list(source.semantic_map.get_all_units())

    memory_units = getattr(source, "memory_units", None)
    if isinstance(memory_units, dict):
        return list(memory_units.values())

    raise ValueError("DBSCAN 聚类需要 MemoryUnit 列表，或支持 get_all_units() 的对象")


def _is_memory_unit_sequence(source: Any) -> bool:
    if isinstance(source, (str, bytes)) or not isinstance(source, Sequence):
        return False
    return all(isinstance(item, MemoryUnit) for item in source)
