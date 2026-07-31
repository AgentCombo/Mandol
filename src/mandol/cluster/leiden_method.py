from typing import Any, Dict, List, Optional, Tuple

from ..utils.logging_config import create_module_logger

logger = create_module_logger("cluster.leiden")


def find_communities_with_leiden(
    graph: Any,
    weights: Optional[List[float]] = None,
    directed: Optional[bool] = None,
) -> Dict[int, List[str]]:
    """Find communities with leiden."""
    node_labels, edges, is_directed = _extract_graph_components(graph, directed)
    if not node_labels:
        return {}

    try:
        import igraph as ig
        import leidenalg as la
    except ImportError as exc:
        raise ImportError("Leiden clustering requires igraph and leidenalg.") from exc

    ig_graph = ig.Graph(n=len(node_labels), directed=is_directed)
    if edges:
        ig_graph.add_edges(edges)

    partition_kwargs = {}
    if weights is not None:
        partition_kwargs["weights"] = weights

    logger.info("Running Leiden community detection")
    partition = la.find_partition(
        ig_graph,
        la.ModularityVertexPartition,
        **partition_kwargs,
    )
    logger.info(f"Community detection completed with {len(partition)} communities")

    communities: Dict[int, List[str]] = {}
    for community_id, community in enumerate(partition):
        communities[community_id] = [node_labels[node_idx] for node_idx in community]

    return communities


def _extract_graph_components(
    graph: Any,
    directed: Optional[bool],
) -> Tuple[List[str], List[Tuple[int, int]], bool]:
    if graph is None:
        return [], [], False

    if hasattr(graph, "rx_graph") and hasattr(graph, "_index_to_uid"):
        return _extract_from_semantic_graph(graph, directed)

    if hasattr(graph, "nodes") and hasattr(graph, "edges"):
        return _extract_from_networkx_graph(graph, directed)

    if hasattr(graph, "node_indices") and hasattr(graph, "edge_list"):
        return _extract_from_rustworkx_graph(graph, directed)

    raise TypeError(
        "find_communities_with_leiden requires a NetworkX graph, rustworkx graph, or SemanticGraph instance"
    )


def _extract_from_semantic_graph(
    semantic_graph: Any,
    directed: Optional[bool],
) -> Tuple[List[str], List[Tuple[int, int]], bool]:
    rx_graph = semantic_graph.rx_graph
    node_indices = list(rx_graph.node_indices())
    if not node_indices:
        return [], [], False

    index_to_local = {idx: local_idx for local_idx, idx in enumerate(node_indices)}
    node_labels = [str(semantic_graph._index_to_uid.get(idx, idx)) for idx in node_indices]
    edges = [
        (index_to_local[source_idx], index_to_local[target_idx])
        for source_idx, target_idx in rx_graph.edge_list()
        if source_idx in index_to_local and target_idx in index_to_local
    ]
    return node_labels, edges, _is_directed(rx_graph, directed)


def _extract_from_networkx_graph(
    nx_graph: Any,
    directed: Optional[bool],
) -> Tuple[List[str], List[Tuple[int, int]], bool]:
    if nx_graph.number_of_nodes() == 0:
        return [], [], False

    nodes = list(nx_graph.nodes())
    node_to_local = {node: idx for idx, node in enumerate(nodes)}
    edges = [(node_to_local[u], node_to_local[v]) for u, v in nx_graph.edges()]
    return [str(node) for node in nodes], edges, _is_directed(nx_graph, directed)


def _extract_from_rustworkx_graph(
    rx_graph: Any,
    directed: Optional[bool],
) -> Tuple[List[str], List[Tuple[int, int]], bool]:
    node_indices = list(rx_graph.node_indices())
    if not node_indices:
        return [], [], False

    index_to_local = {idx: local_idx for local_idx, idx in enumerate(node_indices)}
    node_labels = [_node_label_from_rustworkx(rx_graph, idx) for idx in node_indices]
    edges = [
        (index_to_local[source_idx], index_to_local[target_idx])
        for source_idx, target_idx in rx_graph.edge_list()
        if source_idx in index_to_local and target_idx in index_to_local
    ]
    return node_labels, edges, _is_directed(rx_graph, directed)


def _node_label_from_rustworkx(rx_graph: Any, idx: int) -> str:
    try:
        node_data = rx_graph.get_node_data(idx)
    except Exception:
        try:
            node_data = rx_graph[idx]
        except Exception:
            node_data = None

    if isinstance(node_data, dict):
        return str(node_data.get("uid") or node_data.get("name") or idx)
    if node_data is not None:
        return str(node_data)
    return str(idx)


def _is_directed(graph: Any, directed: Optional[bool]) -> bool:
    if directed is not None:
        return directed
    is_directed_fn = getattr(graph, "is_directed", None)
    if callable(is_directed_fn):
        return bool(is_directed_fn())
    return graph.__class__.__name__.lower().endswith("digraph")
