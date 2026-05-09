"""TF-IDF sparse vector index with per-space adaptive promotion.

Implements cosine-similarity sparse retrieval via inverted indices.
Uses a tiered architecture: starts with flat per-space buffers and
promotes to _SparseIndexCore once the count crosses *promote_threshold*.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ..domain.types import Uid
from ..ports.sparse_index import SparseIndex


class TfidfSparseIndex(SparseIndex):
    """Sparse vector index with per-space adaptive promotion.

    Maintains a global inverted index plus per-space indexes. Spaces
    below *promote_threshold* keep their vectors in an unpromoted flat
    buffer; those that cross the threshold get promoted to a
    _SparseIndexCore for efficient cosine-similarity search.

    Attributes:
        _threshold: Minimum count to trigger space-level promotion.
        _global_index: The global _SparseIndexCore instance.
        _space_indexes: Per-space core indexes (promoted only).
        _unpromoted: Flat sparse-vector buffer for spaces below threshold.
        _uid_to_spaces: Reverse mapping: Uid → set of SpaceNames.
        _space_counts: Per-space unit count tracker.
    """

    def __init__(
        self,
        *,
        promote_threshold: int = 100,
    ) -> None:
        self._threshold = int(promote_threshold)

        self._global_index: Optional[_SparseIndexCore] = None
        self._space_indexes: Dict[str, _SparseIndexCore] = {}
        self._unpromoted: Dict[Uid, Dict[str, float]] = {}
        self._uid_to_spaces: Dict[Uid, Set[str]] = {}
        self._space_counts: Dict[str, int] = {}

    def upsert(self, items: Sequence[Tuple[Uid, Dict[str, float]]]) -> None:
        if self._global_index is None:
            self._global_index = _SparseIndexCore()

        for uid, sparse_vec in items:
            u = Uid(str(uid))
            self._global_index.upsert(u, sparse_vec)
            self._uid_to_spaces.setdefault(u, set())

        self._check_and_promote_all()

    def upsert_to_space(
        self,
        items: Sequence[Tuple[Uid, Dict[str, float]]],
        space_name: str,
    ) -> None:
        """Insert or update sparse vectors in a specific space.

        Args:
            items: Sequence of (uid, sparse_vector_dict) pairs.
            space_name: Logical space name.
        """
        if self._global_index is None:
            self._global_index = _SparseIndexCore()

        new_to_space: List[Tuple[Uid, Dict[str, float]]] = []
        for uid, sparse_vec in items:
            u = Uid(str(uid))
            self._global_index.upsert(u, sparse_vec)
            already_in = u in self._uid_to_spaces and space_name in self._uid_to_spaces[u]

            if not already_in:
                self._uid_to_spaces.setdefault(u, set()).add(space_name)
                self._space_counts[space_name] = self._space_counts.get(space_name, 0) + 1
                new_to_space.append((u, sparse_vec))
            elif space_name not in self._space_indexes:
                self._unpromoted[u] = sparse_vec
            else:
                self._space_indexes[space_name].upsert(u, sparse_vec)

        if space_name in self._space_indexes and new_to_space:
            for u, v in new_to_space:
                self._space_indexes[space_name].upsert(u, v)
        elif new_to_space:
            for u, v in new_to_space:
                self._unpromoted[u] = v

        self._check_and_promote(space_name)

    def delete(self, uids: Iterable[Uid]) -> None:
        for uid in uids:
            u = Uid(str(uid))
            if self._global_index is not None:
                self._global_index.delete(u)
            for idx in self._space_indexes.values():
                idx.delete(u)
            self._unpromoted.pop(u, None)
            self._uid_to_spaces.pop(u, None)

    def delete_from_space(self, uids: Iterable[Uid], space_name: str) -> None:
        for uid in uids:
            u = Uid(str(uid))
            if self._global_index is not None:
                self._global_index.delete(u)
            if space_name in self._space_indexes:
                self._space_indexes[space_name].delete(u)
            spaces = self._uid_to_spaces.get(u)
            if spaces and space_name in spaces:
                spaces.discard(space_name)
                self._space_counts[space_name] = max(0, self._space_counts.get(space_name, 1) - 1)
                self._unpromoted.pop(u, None)

    def search(self, query: Dict[str, float], top_k: int) -> List[Tuple[Uid, float]]:
        if self._global_index is None:
            return []
        return self._global_index.search(query, max(1, int(top_k)))

    def search_in_space(
        self,
        query: Dict[str, float],
        space_name: str,
        candidates: Optional[Set[Uid]],
        top_k: int,
    ) -> List[Tuple[Uid, float]]:
        """Search within a space via promoted index or temporary fallback.

        Args:
            query: Sparse query vector as {token: weight} dict.
            space_name: Logical space name.
            candidates: If provided, restrict results to these Uids.
            top_k: Number of top results.

        Returns:
            List of (uid, similarity_score) pairs sorted descending.
        """
        top_k = max(1, int(top_k))

        if space_name in self._space_indexes:
            hits = self._space_indexes[space_name].search(query, top_k * 2)
            if candidates is not None:
                hits = [(u, s) for u, s in hits if u in candidates]
            return hits[:top_k]

        target_uids: Set[Uid] = set()
        if candidates is not None:
            target_uids = set(candidates)
        elif self._uid_to_spaces:
            target_uids = {
                uid for uid, spaces in self._uid_to_spaces.items()
                if space_name in spaces
            }
            if not target_uids:
                return []
        else:
            return []

        vec_list = [self._unpromoted.get(u) for u in target_uids if u in self._unpromoted]
        if not vec_list:
            return []

        core = _SparseIndexCore()
        for u, v in zip(target_uids, vec_list):
            if v is not None:
                core.upsert(u, v)
        hits = core.search(query, top_k * 2)
        if candidates is not None:
            hits = [(u, s) for u, s in hits if u in candidates]
        return hits[:top_k]

    def rebuild(self, items: Sequence[Tuple[Uid, Dict[str, float]]]) -> None:
        self._global_index = _SparseIndexCore()
        self._uid_to_spaces.clear()
        self._space_counts.clear()
        self._space_indexes.clear()
        self._unpromoted.clear()

        for uid, sparse_vec in items:
            u = Uid(str(uid))
            self._global_index.upsert(u, sparse_vec)
            self._uid_to_spaces.setdefault(u, set())

    def rebuild_space(self, space_name: str, items: Sequence[Tuple[Uid, Dict[str, float]]]) -> None:
        for uid, sparse_vec in items:
            u = Uid(str(uid))
            self._uid_to_spaces.setdefault(u, set()).add(space_name)
        self._space_counts[space_name] = len(items)

        if space_name in self._space_indexes:
            self._space_indexes[space_name].clear()
            for u, v in items:
                self._space_indexes[space_name].upsert(u, v)
        elif len(items) >= self._threshold:
            core = _SparseIndexCore()
            for u, v in items:
                core.upsert(u, v)
            self._space_indexes[space_name] = core
        else:
            for u, v in items:
                self._unpromoted[u] = v

    def _check_and_promote_all(self) -> None:
        for space_name in list(self._space_counts.keys()):
            self._check_and_promote(space_name)

    def _check_and_promote(self, space_name: str) -> None:
        if space_name in self._space_indexes:
            return
        count = self._space_counts.get(space_name, 0)
        if count < self._threshold:
            return
        self._promote(space_name)

    def _promote(self, space_name: str) -> None:
        promoted_uids = {
            uid for uid, spaces in self._uid_to_spaces.items()
            if space_name in spaces
        }
        if not promoted_uids:
            return

        vec_list = [self._unpromoted.get(uid) for uid in promoted_uids]
        if not all(vec_list):
            return

        core = _SparseIndexCore()
        for uid, vec in zip(promoted_uids, vec_list):
            if vec is not None:
                core.upsert(uid, vec)
        self._space_indexes[space_name] = core

        for uid in promoted_uids:
            self._unpromoted.pop(uid, None)


class _SparseIndexCore:
    """Optimized sparse index core with inverted-index cosine similarity.

    Maintains an inverted term → posting list mapping and per-document
    L2 norms, enabling fast dot-product / cosine similarity search.
    """

    def __init__(self) -> None:
        self._inverted_index: Dict[str, List[Tuple[Uid, float]]] = {}
        self._uid_to_norm: Dict[Uid, float] = {}
        self._uid_to_vec: Dict[Uid, Dict[str, float]] = {}

    def upsert(self, uid: Uid, sparse_vec: Dict[str, float]) -> None:
        u = Uid(str(uid))
        old_vec = self._uid_to_vec.get(u)
        if old_vec is not None:
            for term in old_vec:
                if term in self._inverted_index:
                    self._inverted_index[term] = [
                        (id_, w) for id_, w in self._inverted_index[term] if id_ != u
                    ]
                    if not self._inverted_index[term]:
                        del self._inverted_index[term]

        self._uid_to_vec[u] = sparse_vec

        norm_sq = sum(v * v for v in sparse_vec.values())
        self._uid_to_norm[u] = math.sqrt(norm_sq) if norm_sq > 0 else 0.0

        for term, weight in sparse_vec.items():
            if term not in self._inverted_index:
                self._inverted_index[term] = []
            self._inverted_index[term].append((u, weight))

    def delete(self, uid: Uid) -> None:
        u = Uid(str(uid))
        old_vec = self._uid_to_vec.pop(u, None)
        if old_vec is None:
            return

        self._uid_to_norm.pop(u, None)

        for term in old_vec:
            if term in self._inverted_index:
                self._inverted_index[term] = [
                    (id_, w) for id_, w in self._inverted_index[term] if id_ != u
                ]
                if not self._inverted_index[term]:
                    del self._inverted_index[term]

    def search(self, query: Dict[str, float], top_k: int) -> List[Tuple[Uid, float]]:
        """Cosine similarity search via inverted index traversal.

        Args:
            query: Sparse query vector as {term: weight} dict.
            top_k: Number of top results to return.

        Returns:
            List of (uid, cosine_similarity) pairs sorted descending.
        """
        if not query or not self._uid_to_vec:
            return []

        q_norm_sq = sum(v * v for v in query.values())
        q_norm = math.sqrt(q_norm_sq) if q_norm_sq > 0 else 0.0
        if q_norm <= 0:
            return []

        doc_scores: Dict[Uid, float] = {}
        for term, q_weight in query.items():
            if term not in self._inverted_index:
                continue
            for doc_uid, doc_weight in self._inverted_index[term]:
                dot = q_weight * doc_weight
                if doc_uid in doc_scores:
                    doc_scores[doc_uid] += dot
                else:
                    doc_scores[doc_uid] = dot

        results: List[Tuple[Uid, float]] = []
        for doc_uid, dot_prod in doc_scores.items():
            doc_norm = self._uid_to_norm.get(doc_uid, 0.0)
            if doc_norm <= 0:
                continue
            score = dot_prod / (q_norm * doc_norm + 1e-12)
            results.append((doc_uid, float(score)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def clear(self) -> None:
        self._inverted_index.clear()
        self._uid_to_norm.clear()
        self._uid_to_vec.clear()
