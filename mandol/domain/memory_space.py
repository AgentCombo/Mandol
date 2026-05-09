"""Memory space — a named logical container for grouping MemoryUnits.

A MemorySpace organizes units hierarchically: it holds a set of unit Uids
and may contain child sub-spaces, enabling recursive traversal for scoped
retrieval. Each space can also carry a human-readable summary with an
embedding vector for space-level similarity search.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Union

import numpy as np

from .types import Embedding, SpaceName, Uid


def _now_iso() -> str:
    """Return the current UTC time as an ISO 8601 formatted string."""
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class MemorySpace:
    """Named container that groups MemoryUnits into a logical space.

    Supports hierarchical organization via child spaces, enabling recursive
    queries that span an entire sub-tree. Optionally holds a summary text
    and embedding for space-level retrieval.

    Attributes:
        name: Unique space identifier (SpaceName).
        unit_uids: Set of Uids for all units belonging directly to this space.
        child_spaces: Set of SpaceNames for nested sub-spaces.
        summary_text: Optional human-readable summary of the space's content.
        summary_embedding: Optional dense embedding of the summary text.
        metadata: Extensible key-value metadata store.
    """

    name: SpaceName
    unit_uids: Set[Uid] = field(default_factory=set)
    child_spaces: Set[SpaceName] = field(default_factory=set)

    summary_text: Optional[str] = None
    summary_embedding: Optional[Embedding] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the space name and initialize system metadata.

        Sets creation/update timestamps and coerces the summary embedding
        to float32 if provided as a raw array.

        Raises:
            ValueError: If name is empty or not a non-blank string.
        """
        name_str = str(self.name)
        if not isinstance(name_str, str) or not name_str.strip():
            raise ValueError("MemorySpace.name must be a non-empty string")

        current_time = _now_iso()
        self.metadata.setdefault("_system_created_at", current_time)
        self.metadata["_system_updated_at"] = current_time

        if self.summary_embedding is not None and not isinstance(
            self.summary_embedding, np.ndarray
        ):
            self.summary_embedding = np.array(self.summary_embedding, dtype=np.float32)

    def touch(self) -> None:
        """Update the `_system_updated_at` timestamp to the current UTC time."""
        self.metadata["_system_updated_at"] = _now_iso()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this MemorySpace to a JSON-compatible dictionary.

        Uid and SpaceName sets are converted to sorted string lists.
        Summary embedding arrays are converted to Python lists.

        Returns:
            Dictionary with name, unit_uids, child_spaces, summary, and metadata.
        """
        return {
            "name": str(self.name),
            "unit_uids": [str(u) for u in sorted(self.unit_uids, key=lambda x: str(x))],
            "child_spaces": [
                str(s) for s in sorted(self.child_spaces, key=lambda x: str(x))
            ],
            "summary_text": self.summary_text,
            "summary_embedding": self.summary_embedding.tolist()
            if self.summary_embedding is not None
            else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemorySpace":
        """Deserialize a MemorySpace from a JSON-compatible dictionary.

        String lists are converted back to Uid/SpaceName sets. Summary
        embedding lists are converted back to float32 numpy arrays.
        System timestamps from the serialized data are preserved.

        Args:
            data: Serialized dictionary with name, unit_uids, child_spaces, etc.

        Returns:
            A fully constructed MemorySpace instance.
        """
        name = SpaceName(str(data.get("name", "")))
        unit_uids_raw = data.get("unit_uids") or []
        child_spaces_raw = data.get("child_spaces") or []

        summary_text = data.get("summary_text")
        summary_embedding = data.get("summary_embedding")
        if summary_embedding is not None:
            summary_embedding = np.asarray(summary_embedding, dtype=np.float32).reshape(-1)

        metadata = data.get("metadata") or {}
        space = cls(
            name=name,
            unit_uids={Uid(str(u)) for u in unit_uids_raw},
            child_spaces={SpaceName(str(s)) for s in child_spaces_raw},
            summary_text=summary_text,
            summary_embedding=summary_embedding,
            metadata=dict(metadata),
        )
        if isinstance(metadata, dict) and "_system_created_at" in metadata:
            space.metadata["_system_created_at"] = metadata["_system_created_at"]
        if isinstance(metadata, dict) and "_system_updated_at" in metadata:
            space.metadata["_system_updated_at"] = metadata["_system_updated_at"]
        return space

    def add_unit(self, uid: Union[Uid, str]) -> None:
        """Associate a unit with this space.

        Args:
            uid: The Uid (or plain string) of the unit to add.
        """
        self.unit_uids.add(Uid(str(uid)))
        self.touch()

    def remove_unit(self, uid: Union[Uid, str]) -> None:
        """Disassociate a unit from this space (no-op if not present).

        Args:
            uid: The Uid (or plain string) of the unit to remove.
        """
        self.unit_uids.discard(Uid(str(uid)))
        self.touch()

    def add_child_space(self, space: Union[SpaceName, str]) -> None:
        """Add a sub-space as a child of this space.

        Args:
            space: The SpaceName (or plain string) of the child to add.
        """
        self.child_spaces.add(SpaceName(str(space)))
        self.touch()

    def remove_child_space(self, space: Union[SpaceName, str]) -> None:
        """Remove a child sub-space (no-op if not present).

        Args:
            space: The SpaceName (or plain string) of the child to remove.
        """
        self.child_spaces.discard(SpaceName(str(space)))
        self.touch()

    def set_summary(self, text: Optional[str], embedding: Optional[Embedding] = None) -> None:
        """Set or clear the space's summary text and embedding.

        Args:
            text: Human-readable summary, or None to clear.
            embedding: Dense embedding vector for the summary, or None.
        """
        self.summary_text = text
        self.summary_embedding = embedding
        self.touch()

    def get_all_child_space_names(self, *, recursive: bool = True, resolver=None) -> Set[SpaceName]:
        """Return all child space names.

        If recursive is True, a resolver(space_name)->MemorySpace must be provided.
        """
        if not recursive:
            return set(self.child_spaces)
        if resolver is None:
            raise ValueError("resolver is required when recursive=True")

        visited: Set[SpaceName] = set()
        stack: List[SpaceName] = list(self.child_spaces)
        while stack:
            s = stack.pop()
            if s in visited:
                continue
            visited.add(s)
            child = resolver(s)
            if child is not None:
                stack.extend(list(child.child_spaces))
        return visited

    def get_all_unit_uids(self, *, recursive: bool = True, resolver=None) -> Set[Uid]:
        """Return all unit Uids in this space and optionally all descendants.

        Args:
            recursive: If True, include units from all child spaces recursively.
            resolver: A callable(SpaceName) -> MemorySpace, required when recursive=True.

        Returns:
            Set of Uids for all units in the requested scope.

        Raises:
            ValueError: If recursive=True but no resolver is provided.
        """
        if not recursive:
            return set(self.unit_uids)
        if resolver is None:
            raise ValueError("resolver is required when recursive=True")

        all_uids: Set[Uid] = set(self.unit_uids)
        for child_name in self.get_all_child_space_names(recursive=True, resolver=resolver):
            child = resolver(child_name)
            if child is not None:
                all_uids.update(child.unit_uids)
        return all_uids
