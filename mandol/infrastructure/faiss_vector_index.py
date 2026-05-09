"""FAISS-backed approximate nearest-neighbor vector index.

Provides fast ANN search via FAISS IndexFlatIP with L2-normalized vectors
for cosine similarity retrieval. Falls back to brute-force numpy when
FAISS is not installed.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from ..domain.types import Embedding, Uid
from ..ports.vector_index import VectorIndex

logger = logging.getLogger(__name__)


class FaissVectorIndex(VectorIndex):
    """FAISS approximate nearest-neighbor index for cosine similarity search.

    Normalizes all vectors to unit length and uses FAISS IndexFlatIP
    (inner product = cosine similarity for normalized vectors). Falls
    back to InMemoryCosineVectorIndex when FAISS is unavailable.

    Attributes:
        _dim: Dimensionality of stored vectors.
        _index: Underlying FAISS IndexFlatIP instance, built lazily.
        _vectors: Fallback flat dictionary of (Uid → normalized vector).
        _use_faiss: Whether FAISS is available and in use.
    """

    def __init__(self, dim: int):
        self._dim = int(dim)
        self._vectors: Dict[Uid, np.ndarray] = {}
        self._use_faiss = False
        self._index = None
        try:
            import faiss
            self._faiss = faiss
            self._index = faiss.IndexFlatIP(self._dim)
            self._use_faiss = True
        except ImportError:
            logger.warning("FAISS not installed, falling back to numpy brute-force.")

    def dim(self) -> int:
        return self._dim

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        """L2-normalize a vector, guarding against zero vectors.

        Args:
            v: Input vector (float32).

        Returns:
            Unit-length vector of same shape.
        """
        v = v.astype(np.float32)
        if v.ndim == 1:
            v = v.reshape(1, -1)
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        return v / norms

    def upsert(self, items: Sequence[Tuple[Uid, Embedding]]) -> None:
        """Insert or update vectors in the FAISS (or fallback) index.

        Args:
            items: Sequence of (uid, embedding_vector) pairs.

        Raises:
            ValueError: If any embedding dimension does not match self._dim.
        """
        for uid, emb in items:
            vec = np.asarray(emb, dtype=np.float32).reshape(-1)
            if vec.shape[0] != self._dim:
                raise ValueError("embedding dim mismatch")
            normalized = self._normalize(vec)[0]
            self._vectors[Uid(str(uid))] = normalized
            if self._use_faiss and self._index is not None:
                # FAISS IndexFlatIP.add expects (N, D) float32 array
                self._index.add(normalized.reshape(1, -1))

    def delete(self, uids: Iterable[Uid]) -> None:
        for uid in uids:
            self._vectors.pop(Uid(str(uid)), None)

    def search(self, query: Embedding, top_k: int) -> List[Tuple[Uid, float]]:
        """Search for k-nearest neighbors via FAISS or brute-force.

        Args:
            query: Query embedding vector (float32, 1-D).
            top_k: Number of results to return.

        Returns:
            List of (uid, score) pairs sorted by cosine similarity descending.

        Raises:
            ValueError: If query dimension does not match self._dim.
        """
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        if q.shape[0] != self._dim:
            raise ValueError("query dim mismatch")
        if not self._vectors:
            return []

        qn = self._normalize(q)[0]

        if self._use_faiss and self._index is not None:
            k = min(top_k, self._index.ntotal)
            if k <= 0:
                return []
            distances, indices = self._index.search(qn.reshape(1, -1), k)
            uids = list(self._vectors.keys())
            results: List[Tuple[Uid, float]] = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx >= 0 and idx < len(uids):
                    results.append((uids[int(idx)], float(dist)))
            return sorted(results, key=lambda x: x[1], reverse=True)

        # Brute-force fallback
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
        """Search within a restricted candidate set for a given space.

        Retrieves more candidates than needed then filters to the allowed set.

        Args:
            query: Query embedding vector.
            space_name: Logical space name (unused in this implementation).
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
        """Fully rebuild the FAISS index from a fresh set of vectors.

        Args:
            items: Sequence of (uid, embedding_vector) pairs.
        """
        self._vectors.clear()
        if self._use_faiss and self._index is not None:
            self._index.reset()
        self.upsert(items)
