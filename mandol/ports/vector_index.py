"""Abstract interface for vector similarity search indices.

Defines the contract that all vector index implementations
(flat, FAISS, adaptive, etc.) must fulfill.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List, Optional, Sequence, Set, Tuple

from ..domain.types import Embedding, Uid


class VectorIndex(ABC):
    """Abstract interface for vector similarity search indices."""

    @abstractmethod
    def dim(self) -> int:
        """Return the dimensionality of vectors stored in this index.

        Returns:
            Integer number of dimensions (e.g., 768, 2560).
        """
        raise NotImplementedError

    @abstractmethod
    def upsert(self, items: Sequence[Tuple[Uid, Embedding]]) -> None:
        """Insert or update vectors in the index.

        Args:
            items: Sequence of (uid, embedding_vector) pairs to upsert.
        """
        raise NotImplementedError

    @abstractmethod
    def delete(self, uids: Iterable[Uid]) -> None:
        """Remove vectors from the index by their Uids.

        Args:
            uids: Iterable of Uids to remove. Non-existent Uids are silently ignored.
        """
        raise NotImplementedError

    @abstractmethod
    def search(self, query: Embedding, top_k: int) -> List[Tuple[Uid, float]]:
        """Search for the k-nearest neighbors of a query vector.

        Args:
            query: Query embedding vector (float32, 1-D).
            top_k: Number of nearest neighbors to return.

        Returns:
            List of (uid, similarity_score) tuples, sorted by score descending.
        """
        raise NotImplementedError

    @abstractmethod
    def search_in_space(
        self,
        query: Embedding,
        space_name: str,
        candidates: Optional[Set[Uid]],
        top_k: int,
    ) -> List[Tuple[Uid, float]]:
        """Search within a restricted candidate set for a given space.

        Args:
            query: Query embedding vector (float32, 1-D).
            space_name: Logical space name for the search scope.
            candidates: If provided, restrict search to these Uids.
            top_k: Number of nearest neighbors to return.

        Returns:
            List of (uid, similarity_score) tuples, sorted by score descending.
        """
        raise NotImplementedError

    @abstractmethod
    def rebuild(self, items: Sequence[Tuple[Uid, Embedding]]) -> None:
        """Fully rebuild the index from a fresh set of items.

        Args:
            items: Sequence of (uid, embedding_vector) pairs to index.
        """
        raise NotImplementedError
