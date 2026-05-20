"""Abstract interface for semantic graph storage.

Defines the CRUD contract for explicit relationships (edges)
between memory units in the knowledge graph layer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from ..domain.types import Uid


RelationshipKey = Tuple[Uid, Uid, str]


class GraphStore(ABC):
    """Abstract interface for storing and querying graph relationships.

    Each relationship is a directed edge from a source Uid to a target Uid
    with a rel_type label and an optional properties dictionary.
    """

    @abstractmethod
    def upsert_relationship(
        self, source: Uid, target: Uid, rel_type: str, properties: Dict[str, Any]
    ) -> None:
        """Insert or update a relationship between two units.

        Args:
            source: Source unit Uid.
            target: Target unit Uid.
            rel_type: Relationship type string (e.g., COREF, INVOLVES).
            properties: Key-value metadata for the edge.
        """
        raise NotImplementedError

    @abstractmethod
    def delete_relationship(
        self, source: Uid, target: Uid, rel_type: Optional[str] = None
    ) -> None:
        """Delete a relationship (or all relationships of any type) between two units.

        Args:
            source: Source unit Uid.
            target: Target unit Uid.
            rel_type: Specific relationship type to delete, or None to delete all types.
        """
        raise NotImplementedError

    @abstractmethod
    def get_relationship(
        self, source: Uid, target: Uid, rel_type: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieve the properties of a specific relationship.

        Args:
            source: Source unit Uid.
            target: Target unit Uid.
            rel_type: Relationship type to look up.

        Returns:
            Edge properties dict if found, None otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def get_neighbors(
        self, uid: Uid, *, rel_type: Optional[str] = None, direction: str = "out"
    ) -> List[Uid]:
        """Return the neighbors of a node in the graph.

        Args:
            uid: The node Uid whose neighbors are requested.
            rel_type: Optional filter for relationship type.
            direction: \"out\" for outgoing edges, \"in\" for incoming,
                or \"both\" for all.

        Returns:
            List of neighbor Uids.
        """
        raise NotImplementedError

    @abstractmethod
    def get_all_edges(self) -> List[Tuple[Uid, Uid, str, Dict[str, Any]]]:
        """Return all edges in the graph.

        Returns:
            List of (source, target, rel_type, properties) tuples.
        """
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """Remove all edges from the graph."""
        raise NotImplementedError

    @abstractmethod
    def flush(self) -> None:
        """Persist any pending writes to the underlying storage."""
        raise NotImplementedError
