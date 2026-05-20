"""Atomic memory unit — the fundamental data element in the Mandol system.

Each MemoryUnit carries raw text/payload, extensible metadata, and optional
dense/sparse embedding vectors for vector-similarity retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np

from .types import Embedding, Uid


def _now_iso() -> str:
    """Return the current UTC time as an ISO 8601 formatted string."""
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class MemoryUnit:
    """Atomic unit of memory with raw data, metadata, and embedding vectors.

    Each MemoryUnit represents a single addressable piece of information
    (e.g., a dialogue turn, a conversation summary, an extracted entity).
    It carries optional dense and sparse embeddings to support hybrid
    (vector + keyword) retrieval.

    System metadata fields (`_system_created_at`, `_system_updated_at`)
    are managed automatically and excluded from the user-visible metadata
    returned by get_user_metadata().

    Attributes:
        uid: Globally unique identifier for this unit.
        raw_data: Original payload (e.g., text, structured dict).
        metadata: Extensible key-value metadata store.
        embedding: Dense vector embedding (float32 ndarray).
        sparse_embedding: Sparse vector embedding (float32 ndarray).
    """

    uid: Uid
    raw_data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[Embedding] = None
    sparse_embedding: Optional[Embedding] = None

    def __post_init__(self) -> None:
        """Validate fields and initialize system metadata after construction.

        Ensures uid is non-empty, raw_data and metadata are dicts, sets
        creation/update timestamps, and coerces embedding arrays to float32.

        Raises:
            ValueError: If uid is empty or raw_data is not a dict.
        """
        uid_str = str(self.uid)
        if not isinstance(uid_str, str) or not uid_str.strip():
            raise ValueError("MemoryUnit.uid must be a non-empty string")
        if not isinstance(self.raw_data, dict):
            raise ValueError("MemoryUnit.raw_data must be a dict")
        if self.metadata is None:
            self.metadata = {}
        if not isinstance(self.metadata, dict):
            raise ValueError("MemoryUnit.metadata must be a dict")

        current_time = _now_iso()
        self.metadata.setdefault("_system_created_at", current_time)
        self.metadata["_system_updated_at"] = current_time
        self.metadata.setdefault("timestamp", current_time)

        if self.embedding is not None and not isinstance(self.embedding, np.ndarray):
            self.embedding = np.array(self.embedding, dtype=np.float32)
        if self.sparse_embedding is not None and not isinstance(
            self.sparse_embedding, np.ndarray
        ):
            self.sparse_embedding = np.array(self.sparse_embedding, dtype=np.float32)

    def touch(self) -> None:
        """Update the `_system_updated_at` timestamp to the current UTC time."""
        self.metadata["_system_updated_at"] = _now_iso()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this MemoryUnit to a JSON-compatible dictionary.

        Embedding arrays are converted to Python lists for serialization.

        Returns:
            Dictionary with uid, raw_data, metadata, and embeddings.
        """
        return {
            "uid": str(self.uid),
            "raw_data": self.raw_data,
            "metadata": self.metadata,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "sparse_embedding": self.sparse_embedding.tolist()
            if self.sparse_embedding is not None
            else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryUnit":
        """Deserialize a MemoryUnit from a JSON-compatible dictionary.

        Embedding lists are converted back to float32 numpy arrays.
        System timestamps from the serialized data are preserved.

        Args:
            data: Serialized dictionary with uid, raw_data, metadata, and embeddings.

        Returns:
            A fully constructed MemoryUnit instance.
        """
        uid = Uid(str(data.get("uid", "")))
        raw_data = data.get("raw_data") or {}
        metadata = data.get("metadata") or {}

        emb = data.get("embedding")
        if emb is not None:
            emb = np.asarray(emb, dtype=np.float32).reshape(-1)
        sparse = data.get("sparse_embedding")
        if sparse is not None:
            sparse = np.asarray(sparse, dtype=np.float32).reshape(-1)

        unit = cls(uid=uid, raw_data=dict(raw_data), metadata=dict(metadata), embedding=emb, sparse_embedding=sparse)

        if isinstance(metadata, dict) and "_system_created_at" in metadata:
            unit.metadata["_system_created_at"] = metadata["_system_created_at"]
        if isinstance(metadata, dict) and "_system_updated_at" in metadata:
            unit.metadata["_system_updated_at"] = metadata["_system_updated_at"]
        return unit

    def get_user_metadata(self) -> Dict[str, Any]:
        """Return metadata excluding internal system keys.

        Filters out all keys starting with `_system_` so callers only see
        user-defined or application-level metadata.

        Returns:
            Shallow copy of metadata with system keys removed.
        """
        return {k: v for k, v in self.metadata.items() if not k.startswith("_system_")}

    def __hash__(self) -> int:  # pragma: no cover
        return hash(str(self.uid))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MemoryUnit):
            return False
        if str(self.uid) != str(other.uid):
            return False
        if self.raw_data != other.raw_data:
            return False

        if self.embedding is not None and other.embedding is not None:
            if not np.array_equal(self.embedding, other.embedding, equal_nan=True):
                return False

        if self.sparse_embedding is not None and other.sparse_embedding is not None:
            if not np.array_equal(
                self.sparse_embedding, other.sparse_embedding, equal_nan=True
            ):
                return False

        return self.get_user_metadata() == other.get_user_metadata()
