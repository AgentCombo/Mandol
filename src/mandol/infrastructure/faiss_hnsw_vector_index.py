"""True approximate nearest-neighbor search using FAISS HNSW."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from ..domain.types import Embedding, Uid
from ..ports.vector_index import VectorIndex


class FaissHNSWVectorIndex(VectorIndex):
    """Cosine ANN index backed by ``faiss.IndexHNSWFlat``.

    Vectors are L2-normalized and searched with inner product.  HNSW does not
    support arbitrary updates/deletes in this adapter, so changes to an
    existing UID rebuild the in-memory acceleration structure from the
    authoritative vector dictionary.
    """

    def __init__(
        self,
        dim: int,
        *,
        m: int = 32,
        ef_construction: int = 80,
        ef_search: int = 64,
        filter_oversample: int = 4,
    ) -> None:
        try:
            import faiss
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "FaissHNSWVectorIndex requires faiss-cpu: "
                "pip install 'mandol[faiss]'"
            ) from exc
        if int(dim) <= 0:
            raise ValueError("dim must be positive")
        if int(m) <= 0:
            raise ValueError("m must be positive")
        if int(ef_construction) <= 0 or int(ef_search) <= 0:
            raise ValueError("ef_construction and ef_search must be positive")
        if int(filter_oversample) <= 0:
            raise ValueError("filter_oversample must be positive")

        self._faiss = faiss
        self._dim = int(dim)
        self._m = int(m)
        self._ef_construction = int(ef_construction)
        self._ef_search = int(ef_search)
        self._filter_oversample = int(filter_oversample)
        self._vectors: dict[Uid, np.ndarray] = {}
        self._uid_order: list[Uid] = []
        self._index = self._new_index()

    def dim(self) -> int:
        return self._dim

    @property
    def ef_search(self) -> int:
        return int(self._index.hnsw.efSearch)

    @ef_search.setter
    def ef_search(self, value: int) -> None:
        if int(value) <= 0:
            raise ValueError("ef_search must be positive")
        self._ef_search = int(value)
        self._index.hnsw.efSearch = self._ef_search

    def upsert(self, items: Sequence[tuple[Uid, Embedding]]) -> None:
        if not items:
            return
        normalized: list[tuple[Uid, np.ndarray]] = []
        rebuild_required = False
        for uid, embedding in items:
            key = Uid(str(uid))
            vector = self._normalize_one(embedding)
            if key in self._vectors:
                rebuild_required = True
            normalized.append((key, vector))
            self._vectors[key] = vector

        if rebuild_required:
            self._rebuild_index()
            return

        batch = np.stack([vector for _, vector in normalized]).astype(np.float32)
        self._uid_order.extend(uid for uid, _ in normalized)
        self._index.add(batch)

    def delete(self, uids: Iterable[Uid]) -> None:
        changed = False
        for uid in uids:
            changed = self._vectors.pop(Uid(str(uid)), None) is not None or changed
        if changed:
            self._rebuild_index()

    def search(self, query: Embedding, top_k: int) -> list[tuple[Uid, float]]:
        if not self._uid_order or int(top_k) <= 0:
            return []
        vector = self._normalize_one(query).reshape(1, -1)
        k = min(int(top_k), len(self._uid_order))
        distances, indices = self._index.search(vector, k)
        return [
            (self._uid_order[int(index)], float(distance))
            for distance, index in zip(distances[0], indices[0])
            if 0 <= int(index) < len(self._uid_order)
        ]

    def search_in_space(
        self,
        query: Embedding,
        space_name: str,
        candidates: set[Uid] | None,
        top_k: int,
    ) -> list[tuple[Uid, float]]:
        del space_name
        if candidates is None:
            return self.search(query, top_k)
        if not candidates or int(top_k) <= 0:
            return []

        total = len(self._uid_order)
        recall = min(
            total,
            max(int(top_k), int(top_k) * self._filter_oversample),
        )
        while recall > 0:
            hits = self.search(query, recall)
            filtered = [(uid, score) for uid, score in hits if uid in candidates]
            if len(filtered) >= int(top_k) or recall >= total:
                return filtered[: int(top_k)]
            recall = min(total, recall * 2)
        return []

    def rebuild(self, items: Sequence[tuple[Uid, Embedding]]) -> None:
        self._vectors = {
            Uid(str(uid)): self._normalize_one(embedding)
            for uid, embedding in items
        }
        self._rebuild_index()

    def _new_index(self):
        index = self._faiss.IndexHNSWFlat(
            self._dim,
            self._m,
            self._faiss.METRIC_INNER_PRODUCT,
        )
        index.hnsw.efConstruction = self._ef_construction
        index.hnsw.efSearch = self._ef_search
        return index

    def _rebuild_index(self) -> None:
        self._index = self._new_index()
        self._uid_order = list(self._vectors)
        if self._uid_order:
            self._index.add(
                np.stack([self._vectors[uid] for uid in self._uid_order]).astype(
                    np.float32
                )
            )

    def _normalize_one(self, value: Embedding) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
        if vector.shape[0] != self._dim:
            raise ValueError(
                f"embedding dim mismatch: expected {self._dim}, got {vector.shape[0]}"
            )
        if not np.all(np.isfinite(vector)):
            raise ValueError("embedding must contain only finite values")
        norm = float(np.linalg.norm(vector))
        return vector if norm == 0.0 else vector / norm
