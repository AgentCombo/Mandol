"""Adaptive (multi-tier) vector index with automatic promotion.

Starts with a brute-force flat cosine index (InMemoryCosineVectorIndex).
When the number of vectors exceeds *promote_threshold*, automatically
promotes to a FAISS-backed ANN index for faster approximate search.
"""

from __future__ import annotations

import logging
from typing import Iterable, List, Optional, Sequence, Set, Tuple


from ..domain.types import Embedding, Uid
from ..ports.vector_index import VectorIndex
from .faiss_vector_index import FaissVectorIndex
from .in_memory_vector_index import InMemoryCosineVectorIndex

logger = logging.getLogger(__name__)


class AdaptiveVectorIndex(VectorIndex):
    """Multi-tier vector index with automatic promotion to optimized backends.

    Uses a lightweight flat cosine index for small datasets and promotes
    to a FAISS ANN index once the number of vectors crosses the threshold.
    This avoids the overhead of building an ANN structure for tiny collections
    while still benefiting from fast approximate search at scale.

    Attributes:
        _promote_threshold: Number of vectors that triggers FAISS promotion.
        _flat: The current flat (brute-force) index.
        _faiss_index: The promoted FAISS index, or None before promotion.
    """

    def __init__(self, dim: int, promote_threshold: int = 100):
        self._dim = int(dim)
        self._promote_threshold = int(promote_threshold)
        self._flat = InMemoryCosineVectorIndex(self._dim)
        self._faiss_index: Optional[FaissVectorIndex] = None

    @property
    def active(self) -> VectorIndex:
        return self._faiss_index or self._flat

    def dim(self) -> int:
        return self._dim

    def upsert(self, items: Sequence[Tuple[Uid, Embedding]]) -> None:
        """Insert or update vectors, triggering FAISS promotion if threshold crossed.

        Args:
            items: Sequence of (uid, embedding_vector) pairs to upsert.
        """
        self.active.upsert(items)
        self._maybe_promote()

    def delete(self, uids: Iterable[Uid]) -> None:
        self.active.delete(uids)

    def search(self, query: Embedding, top_k: int) -> List[Tuple[Uid, float]]:
        return self.active.search(query, top_k)

    def search_in_space(
        self,
        query: Embedding,
        space_name: str,
        candidates: Optional[Set[Uid]],
        top_k: int,
    ) -> List[Tuple[Uid, float]]:
        return self.active.search_in_space(query, space_name, candidates, top_k)

    def rebuild(self, items: Sequence[Tuple[Uid, Embedding]]) -> None:
        """Fully rebuild the index, promoting to FAISS if the item count exceeds threshold.

        Args:
            items: Sequence of (uid, embedding_vector) pairs to index.
        """
        if len(items) >= self._promote_threshold:
            self._faiss_index = FaissVectorIndex(self._dim)
            logger.info(
                "Promoting to FAISS index (dim=%d, vectors=%d)",
                self._dim, len(items),
            )
        else:
            self._faiss_index = None
        self.active.rebuild(items)

    def _maybe_promote(self) -> None:
        """Promote to FAISS if the flat index has enough vectors and no promo yet."""
        # Count vectors in the flat index; each entry holds a normalized float32 array
        if self._faiss_index is not None or not hasattr(self._flat, "_vectors"):
            return

        flat_count = len(self._flat._vectors)
        if flat_count >= self._promote_threshold:
            items = list(self._flat._vectors.items())
            try:
                self._faiss_index = FaissVectorIndex(self._dim)
                self._faiss_index.rebuild(items)
                logger.info(
                    "Promoted to FAISS (dim=%d, vectors=%d)",
                    self._dim, len(items),
                )
            except ImportError:
                logger.debug(
                    "FAISS not installed — staying on flat index (dim=%d, vectors=%d)",
                    self._dim, flat_count,
                )
