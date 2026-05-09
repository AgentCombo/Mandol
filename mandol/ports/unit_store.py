"""Abstract interface for unit and space persistence.

Defines the contract for CRUD operations on MemoryUnits and MemorySpaces,
abstracting over the underlying storage backend (in-memory, Milvus, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List, Optional, Sequence

from ..domain.memory_space import MemorySpace
from ..domain.memory_unit import MemoryUnit
from ..domain.types import SpaceName, Uid


class UnitStore(ABC):
    """Abstract interface for persisting and retrieving memory units and spaces."""

    @abstractmethod
    def upsert_units(self, units: Sequence[MemoryUnit]) -> None:
        """Insert or update a batch of MemoryUnits.

        Args:
            units: Sequence of MemoryUnit instances to persist.
        """
        raise NotImplementedError

    @abstractmethod
    def delete_units(self, uids: Iterable[Uid]) -> None:
        """Delete units by their Uids.

        Args:
            uids: Iterable of Uids to delete. Non-existent Uids are silently ignored.
        """
        raise NotImplementedError

    @abstractmethod
    def get_unit(self, uid: Uid) -> Optional[MemoryUnit]:
        """Retrieve a single MemoryUnit by Uid.

        Args:
            uid: Unique identifier of the unit.

        Returns:
            MemoryUnit if found, None otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def get_units(self, uids: Sequence[Uid]) -> List[MemoryUnit]:
        """Retrieve multiple MemoryUnits by their Uids.

        Args:
            uids: Sequence of Uids to fetch.

        Returns:
            List of MemoryUnit instances (only those found).
        """
        raise NotImplementedError

    @abstractmethod
    def list_units(self) -> List[MemoryUnit]:
        """Return all stored MemoryUnits.

        Returns:
            List of all MemoryUnit instances in the store.
        """
        raise NotImplementedError

    @abstractmethod
    def upsert_spaces(self, spaces: Sequence[MemorySpace]) -> None:
        """Insert or update a batch of MemorySpaces.

        Args:
            spaces: Sequence of MemorySpace instances to persist.
        """
        raise NotImplementedError

    @abstractmethod
    def get_space(self, name: SpaceName) -> Optional[MemorySpace]:
        """Retrieve a single MemorySpace by name.

        Args:
            name: Unique SpaceName identifier.

        Returns:
            MemorySpace if found, None otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def list_spaces(self) -> List[MemorySpace]:
        """Return all stored MemorySpaces.

        Returns:
            List of all MemorySpace instances in the store.
        """
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """Remove all units and spaces from the store."""
        raise NotImplementedError

    @abstractmethod
    def flush(self) -> None:
        """Persist any buffered writes to the underlying storage."""
        raise NotImplementedError
