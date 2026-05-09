"""Abstract interface for BM25 lexical search indices.

Defines the contract for BM25-based keyword retrieval, supporting
per-space scoping alongside a global index.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List, Optional, Sequence, Set, Tuple

from ..domain.types import Uid


class BM25Index(ABC):
    """Abstract interface for BM25 lexical (keyword) search indices."""

    @abstractmethod
    def upsert(self, items: Sequence[Tuple[Uid, List[str]]]) -> None:
        """Insert or update tokenized documents in the global index.

        Args:
            items: Sequence of (uid, token_list) pairs to index.
        """
        raise NotImplementedError

    @abstractmethod
    def upsert_to_space(self, items: Sequence[Tuple[Uid, List[str]]], space_name: str) -> None:
        """Insert or update tokenized documents in a specific space's index.

        Args:
            items: Sequence of (uid, token_list) pairs to index.
            space_name: Logical space name for scoped indexing.
        """
        raise NotImplementedError

    @abstractmethod
    def delete(self, uids: Iterable[Uid]) -> None:
        """Remove documents from the global index by their Uids.

        Args:
            uids: Iterable of Uids to remove.
        """
        raise NotImplementedError

    @abstractmethod
    def delete_from_space(self, uids: Iterable[Uid], space_name: str) -> None:
        """Remove documents from a specific space's index.

        Args:
            uids: Iterable of Uids to remove.
            space_name: Logical space name.
        """
        raise NotImplementedError

    @abstractmethod
    def search(self, query: List[str], top_k: int) -> List[Tuple[Uid, float]]:
        """Search the global BM25 index.

        Args:
            query: Tokenized query as a list of strings.
            top_k: Number of top results to return.

        Returns:
            List of (uid, bm25_score) tuples, sorted by score descending.
        """
        raise NotImplementedError

    @abstractmethod
    def search_in_space(
        self,
        query: List[str],
        space_name: str,
        candidates: Optional[Set[Uid]],
        top_k: int,
    ) -> List[Tuple[Uid, float]]:
        """Search within a restricted candidate set for a given space.

        Args:
            query: Tokenized query as a list of strings.
            space_name: Logical space name.
            candidates: If provided, restrict search to these Uids.
            top_k: Number of top results to return.

        Returns:
            List of (uid, bm25_score) tuples, sorted by score descending.
        """
        raise NotImplementedError

    @abstractmethod
    def rebuild(self, items: Sequence[Tuple[Uid, List[str]]]) -> None:
        """Fully rebuild the global index from a fresh set of tokenized documents.

        Args:
            items: Sequence of (uid, token_list) pairs to index.
        """
        raise NotImplementedError

    @abstractmethod
    def rebuild_space(self, space_name: str, items: Sequence[Tuple[Uid, List[str]]]) -> None:
        """Fully rebuild a specific space's index.

        Args:
            space_name: Logical space name.
            items: Sequence of (uid, token_list) pairs to index.
        """
        raise NotImplementedError
