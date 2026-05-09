"""In-memory implementation of the GraphStore port.

Stores directed relationships in a NetworkX DiGraph, providing
fast in-process edge traversal and typed relationship queries.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from ..domain.types import Uid
from ..ports.graph_store import GraphStore


class InMemoryGraphStore(GraphStore):
    """Graph store backed by an in-memory NetworkX directed graph.

    Each node is a Uid; each edge carries a 'type' attribute (the
    relationship type) plus arbitrary key-value properties. Tracks
    a dirty flag for use by persistence managers.

    Attributes:
        _g: The underlying NetworkX DiGraph instance.
        _dirty: Set to True on every mutating operation; cleared by flush().
    """

    def __init__(self):
        self._g: nx.DiGraph = nx.DiGraph()
        self._dirty: bool = False

    def upsert_relationship(
        self, source: Uid, target: Uid, rel_type: str, properties: Dict[str, Any]
    ) -> None:
        """Insert or update an edge between two nodes.

        Nodes are created automatically if they do not exist.

        Args:
            source: Source node Uid.
            target: Target node Uid.
            rel_type: Relationship type string.
            properties: Key-value edge metadata (None values are stripped).
        """
        s = Uid(str(source))
        t = Uid(str(target))
        self._g.add_node(s)
        self._g.add_node(t)
        attrs = {"type": str(rel_type), **{k: v for k, v in properties.items() if v is not None}}
        self._g.add_edge(s, t, **attrs)
        self._dirty = True

    def delete_relationship(
        self, source: Uid, target: Uid, rel_type: Optional[str] = None
    ) -> None:
        """Delete an edge (or any edge if rel_type is omitted) between two nodes.

        Args:
            source: Source node Uid.
            target: Target node Uid.
            rel_type: If provided, only delete the edge with this type.
        """
        s = Uid(str(source))
        t = Uid(str(target))
        if not self._g.has_edge(s, t):
            return
        if rel_type is None:
            self._g.remove_edge(s, t)
            self._dirty = True
            return
        data = self._g.get_edge_data(s, t) or {}
        if data.get("type") == rel_type:
            self._g.remove_edge(s, t)
            self._dirty = True

    def get_relationship(
        self, source: Uid, target: Uid, rel_type: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieve edge properties for a specific relationship.

        Args:
            source: Source node Uid.
            target: Target node Uid.
            rel_type: Relationship type to look up.

        Returns:
            Properties dict (without the internal 'type' key), or None.
        """
        s = Uid(str(source))
        t = Uid(str(target))
        if not self._g.has_edge(s, t):
            return None
        data = self._g.get_edge_data(s, t) or {}
        if data.get("type") != rel_type:
            return None
        return {k: v for k, v in data.items() if k != "type"}

    def get_neighbors(
        self, uid: Uid, *, rel_type: Optional[str] = None, direction: str = "out"
    ) -> List[Uid]:
        """Return neighbors reachable via outgoing or incoming edges.

        Args:
            uid: The node whose neighbors are requested.
            rel_type: Optional filter for edge type.
            direction: \"out\" for successors, \"in\" for predecessors.

        Returns:
            List of neighbor Uids.

        Raises:
            ValueError: If direction is not \"out\" or \"in\".
        """
        u = Uid(str(uid))
        if not self._g.has_node(u):
            return []
        if direction not in {"out", "in"}:
            raise ValueError("direction must be 'out' or 'in'")

        neighbors = self._g.successors(u) if direction == "out" else self._g.predecessors(u)
        out: List[Uid] = []
        for v in neighbors:
            if rel_type is None:
                out.append(Uid(str(v)))
                continue
            data = self._g.get_edge_data(u, v) if direction == "out" else self._g.get_edge_data(v, u)
            if (data or {}).get("type") == rel_type:
                out.append(Uid(str(v)))
        return out

    def get_all_edges(self) -> List[Tuple[Uid, Uid, str, Dict[str, Any]]]:
        """Return every edge in the graph as (source, target, type, properties).

        Returns:
            List of edge tuples.
        """
        edges: List[Tuple[Uid, Uid, str, Dict[str, Any]]] = []
        for source, target, data in self._g.edges(data=True):
            edges.append((
                Uid(str(source)),
                Uid(str(target)),
                str(data.get("type", "")),
                {k: v for k, v in data.items() if k != "type"},
            ))
        return edges

    def clear(self) -> None:
        """Remove all nodes and edges from the graph."""
        self._g.clear()
        self._dirty = True

    def flush(self) -> None:
        self._dirty = False
