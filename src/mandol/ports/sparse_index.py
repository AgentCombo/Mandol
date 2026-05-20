"""Abstract interface for sparse vector search indices.

Defines the contract for sparse retrieval (e.g., TF-IDF, SPLADE),
supporting per-space scoping alongside a global index.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ..domain.types import Uid


class SparseIndex(ABC):
    """Abstract interface for sparse vector similarity search indices."""

    @abstractmethod
    def upsert(self, items: Sequence[Tuple[Uid, Dict[str, float]]]) -> None:
        """Insert or update sparse vectors in the global index.

        Args:
            items: Sequence of (uid, sparse_vector_dict) pairs to index.
        """
        raise NotImplementedError

    @abstractmethod
    def upsert_to_space(self, items: Sequence[Tuple[Uid, Dict[str, float]]], space_name: str) -> None:
        """Insert or update sparse vectors in a specific space's index.

        Args:
            items: Sequence of (uid, sparse_vector_dict) pairs to index.
            space_name: Logical space name for scoped indexing.
        """
        raise NotImplementedError

    @abstractmethod
    def delete(self, uids: Iterable[Uid]) -> None:
        """Remove vectors from the global index by their Uids.

        Args:
            uids: Iterable of Uids to remove.
        """
        raise NotImplementedError

    @abstractmethod
    def delete_from_space(self, uids: Iterable[Uid], space_name: str) -> None:
        """Remove vectors from a specific space's index.

        Args:
            uids: Iterable of Uids to remove.
            space_name: Logical space name.
        """
        raise NotImplementedError

    @abstractmethod
    def search(self, query: Dict[str, float], top_k: int) -> List[Tuple[Uid, float]]:
        """Search the global sparse index.

        Args:
            query: Sparse query vector as a {token: weight} dict.
            top_k: Number of top results to return.

        Returns:
            List of (uid, similarity_score) tuples, sorted by score descending.
        """
        raise NotImplementedError

    @abstractmethod
    def search_in_space(
        self,
        query: Dict[str, float],
        space_name: str,
        candidates: Optional[Set[Uid]],
        top_k: int,
    ) -> List[Tuple[Uid, float]]:
        """Search within a restricted candidate set for a given space.

        Args:
            query: Sparse query vector as a {token: weight} dict.
            space_name: Logical space name.
            candidates: If provided, restrict search to these Uids.
            top_k: Number of top results to return.

        Returns:
            List of (uid, similarity_score) tuples, sorted by score descending.
        """
        raise NotImplementedError

    @abstractmethod
    def rebuild(self, items: Sequence[Tuple[Uid, Dict[str, float]]]) -> None:
        """Fully rebuild the global index from a fresh set of sparse vectors.

        Args:
            items: Sequence of (uid, sparse_vector_dict) pairs to index.
        """
        raise NotImplementedError

    @abstractmethod
    def rebuild_space(self, space_name: str, items: Sequence[Tuple[Uid, Dict[str, float]]]) -> None:
        """Fully rebuild a specific space's index.

        Args:
            space_name: Logical space name.
            items: Sequence of (uid, sparse_vector_dict) pairs to index.
        """
        raise NotImplementedError
