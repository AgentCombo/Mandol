"""In-memory implementation of the GraphStore port.

Stores directed relationships in a NetworkX MultiDiGraph, using the
relationship type as the edge key.  This mirrors the public GraphStore
identity ``(source, target, rel_type)`` and permits two differently typed
relationships between the same ordered pair of nodes.
"""

from __future__ import annotations

from typing import Any

import networkx as nx

from ..domain.types import Uid
from ..ports.graph_store import GraphStore


class InMemoryGraphStore(GraphStore):
    """Graph store backed by an in-memory NetworkX directed multigraph.

    Each node is a Uid; each edge carries a 'type' attribute (the
    relationship type) plus arbitrary key-value properties. Tracks
    a dirty flag for use by persistence managers.

    Attributes:
        _g: The underlying NetworkX MultiDiGraph instance.
        _dirty: Set to True on every mutating operation; cleared by flush().
    """

    def __init__(self):
        self._g: nx.MultiDiGraph = nx.MultiDiGraph()
        self._dirty: bool = False

    def upsert_relationship(
        self, source: Uid, target: Uid, rel_type: str, properties: dict[str, Any]
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
        relation = str(rel_type)
        attrs = {"type": relation, **{k: v for k, v in properties.items() if v is not None}}
        self._g.add_edge(s, t, key=relation, **attrs)
        self._dirty = True

    def delete_relationship(
        self, source: Uid, target: Uid, rel_type: str | None = None
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
            keys = list((self._g.get_edge_data(s, t) or {}).keys())
            for key in keys:
                self._g.remove_edge(s, t, key=key)
            self._dirty = True
            return
        relation = str(rel_type)
        if self._g.has_edge(s, t, key=relation):
            self._g.remove_edge(s, t, key=relation)
            self._dirty = True

    def get_relationship(
        self, source: Uid, target: Uid, rel_type: str
    ) -> dict[str, Any] | None:
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
        relation = str(rel_type)
        if not self._g.has_edge(s, t, key=relation):
            return None
        data = self._g.get_edge_data(s, t, key=relation) or {}
        return {k: v for k, v in data.items() if k != "type"}

    def get_neighbors(
        self, uid: Uid, *, rel_type: str | None = None, direction: str = "out"
    ) -> list[Uid]:
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
        if direction not in {"out", "in", "both"}:
            raise ValueError("direction must be 'out', 'in', or 'both'")

        relation = None if rel_type is None else str(rel_type)
        neighbors: set[Uid] = set()

        if direction in {"out", "both"}:
            for _, target, key in self._g.out_edges(u, keys=True):
                if relation is None or str(key) == relation:
                    neighbors.add(Uid(str(target)))

        if direction in {"in", "both"}:
            for source, _, key in self._g.in_edges(u, keys=True):
                if relation is None or str(key) == relation:
                    neighbors.add(Uid(str(source)))

        return sorted(neighbors, key=str)

    def get_all_edges(self) -> list[tuple[Uid, Uid, str, dict[str, Any]]]:
        """Return every edge in the graph as (source, target, type, properties).

        Returns:
            List of edge tuples.
        """
        edges: list[tuple[Uid, Uid, str, dict[str, Any]]] = []
        for source, target, key, data in self._g.edges(keys=True, data=True):
            edges.append((
                Uid(str(source)),
                Uid(str(target)),
                str(key),
                {k: v for k, v in data.items() if k != "type"},
            ))
        return sorted(edges, key=lambda edge: (str(edge[0]), str(edge[1]), edge[2]))

    def clear(self) -> None:
        """Remove all nodes and edges from the graph."""
        self._g.clear()
        self._dirty = True

    def flush(self) -> None:
        self._dirty = False
