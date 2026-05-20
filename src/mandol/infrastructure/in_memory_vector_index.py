"""In-memory brute-force cosine similarity vector index.

Provides exact (non-approximate) cosine similarity search using
L2-normalized vectors and numpy dot products. Used as the default
index for small collections and as the flat tier in AdaptiveVectorIndex.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from ..domain.types import Embedding, Uid
from ..ports.vector_index import VectorIndex


def _normalize(v: np.ndarray) -> np.ndarray:
    """L2-normalize a batch of vectors, guarding against zero vectors.

    Args:
        v: (N, D) or (D,) float32 array.

    Returns:
        Unit-length array of same leading shape.
    """
    v = v.astype(np.float32)
    if v.ndim == 1:
        v = v.reshape(1, -1)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    v = v / norms
    return v


class InMemoryCosineVectorIndex(VectorIndex):
    """Brute-force cosine similarity index backed by an in-memory dict.

    Normalizes all inserted vectors to unit length so that inner product
    equals cosine similarity. Suitable for small-to-medium collections
    or as the fallback tier in AdaptiveVectorIndex.

    Attributes:
        _dim: Dimensionality of stored vectors.
        _vectors: Mapping from Uid to L2-normalized float32 vector.
    """

    def __init__(self, dim: int):
        self._dim = int(dim)
        self._vectors: Dict[Uid, np.ndarray] = {}

    def dim(self) -> int:
        return self._dim

    def upsert(self, items: Sequence[Tuple[Uid, Embedding]]) -> None:
        """Insert or update normalized vectors in the index.

        Args:
            items: Sequence of (uid, embedding_vector) pairs.

        Raises:
            ValueError: If any vector dimension does not match self._dim.
        """
        for uid, emb in items:
            vec = np.asarray(emb, dtype=np.float32).reshape(-1)
            if vec.shape[0] != self._dim:
                raise ValueError("embedding dim mismatch")
            self._vectors[Uid(str(uid))] = _normalize(vec)[0]

    def delete(self, uids: Iterable[Uid]) -> None:
        for uid in uids:
            self._vectors.pop(Uid(str(uid)), None)

    def search(self, query: Embedding, top_k: int) -> List[Tuple[Uid, float]]:
        """Brute-force cosine similarity search via normalized dot products.

        Args:
            query: Query embedding vector (float32, 1-D).
            top_k: Number of nearest neighbors to return.

        Returns:
            List of (uid, similarity_score) tuples sorted descending.

        Raises:
            ValueError: If query dimension does not match self._dim.
        """
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        if q.shape[0] != self._dim:
            raise ValueError("query dim mismatch")
        if not self._vectors:
            return []
        qn = _normalize(q)[0]

        uids = list(self._vectors.keys())
        mat = np.stack([self._vectors[u] for u in uids], axis=0)
        scores = mat @ qn
        idx = np.argsort(-scores)[: max(0, int(top_k))]
        return [(uids[int(i)], float(scores[int(i)])) for i in idx]

    def search_in_space(
        self,
        query: Embedding,
        space_name: str,
        candidates: Optional[Set[Uid]],
        top_k: int,
    ) -> List[Tuple[Uid, float]]:
        """Search within a candidate set with oversampling for post-filtering.

        Args:
            query: Query embedding vector.
            space_name: Logical space name (unused here).
            candidates: Restrict results to these Uids.
            top_k: Number of results to return.

        Returns:
            Filtered list of (uid, score) pairs.
        """
        hits = self.search(query, top_k * 3 if candidates else top_k)
        if candidates is not None:
            hits = [(u, s) for u, s in hits if u in candidates]
        return hits[: max(1, int(top_k))]

    def rebuild(self, items: Sequence[Tuple[Uid, Embedding]]) -> None:
        """Clear and fully rebuild from a fresh set of vectors.

        Args:
            items: Sequence of (uid, embedding_vector) pairs.
        """
        self._vectors.clear()
        self.upsert(items)
