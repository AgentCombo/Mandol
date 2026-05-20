"""Abstract interface for cross-encoder reranking.

Defines the contract for reranking a list of candidate MemoryUnits
against a query to produce more accurate relevance scores.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

from ..domain.memory_unit import MemoryUnit


class Reranker(ABC):
    """Abstract interface for cross-encoder reranking providers."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        units: List[MemoryUnit],
        *,
        top_k: int = 10,
    ) -> List[Tuple[MemoryUnit, float]]:
        """Rerank a list of candidate units against a query.

        Uses a cross-encoder model to jointly encode the query and each
        candidate, producing a more accurate relevance score than
        bi-encoder similarity alone.

        Args:
            query: Natural language query string.
            units: List of candidate MemoryUnits to rerank.
            top_k: Maximum number of results to return after reranking.

        Returns:
            List of (MemoryUnit, reranked_score) tuples, sorted descending.
        """
        raise NotImplementedError
