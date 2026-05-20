"""Rank-BM25 lexical search index with per-space adaptive promotion.

Implements the BM25 scoring function for keyword-based retrieval.
Uses a tiered architecture: starts with a flat buffer per space and
promotes to an optimized BM25 index once the vector count crosses
*promote_threshold*.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ..domain.types import Uid
from ..ports.bm25_index import BM25Index


class RankBM25Index(BM25Index):
    """BM25 lexical index with per-space adaptive promotion to core indexes.

    Tracks a global index plus per-space indexes. Spaces with fewer than
    *promote_threshold* units keep their tokens in an unpromoted flat
    buffer; those that cross the threshold get promoted to a _BM25IndexCore
    for efficient search.

    Attributes:
        _k1: BM25 term-frequency saturation parameter.
        _b: BM25 document-length normalization parameter.
        _threshold: Minimum count to trigger space-level index promotion.
        _global_index: The global _BM25IndexCore instance.
        _space_indexes: Per-space _BM25IndexCore instances (promoted only).
        _unpromoted: Flat token buffer for spaces below threshold.
        _uid_to_spaces: Reverse mapping: Uid → set of SpaceNames.
        _space_counts: Per-space unit count tracker.
    """

    def __init__(
        self,
        *,
        k1: float = 1.5,
        b: float = 0.75,
        promote_threshold: int = 100,
    ) -> None:
        self._k1 = float(k1)         # BM25 term-frequency saturation (Robertson & Zaragoza, 2009)
        self._b = float(b)           # BM25 document-length normalization (standard value)
        self._threshold = int(promote_threshold)

        self._global_index: Optional[_BM25IndexCore] = None
        self._space_indexes: Dict[str, _BM25IndexCore] = {}
        self._unpromoted: Dict[Uid, List[str]] = {}
        self._uid_to_spaces: Dict[Uid, Set[str]] = {}
        self._space_counts: Dict[str, int] = {}

    def upsert(self, items: Sequence[Tuple[Uid, List[str]]]) -> None:
        """Insert or update tokenized documents in the global index.

        Args:
            items: Sequence of (uid, token_list) pairs to index.
        """
        if self._global_index is None:
            self._global_index = _BM25IndexCore(k1=self._k1, b=self._b)

        for uid, tokens in items:
            u = Uid(str(uid))
            self._global_index.upsert(u, tokens)
            self._uid_to_spaces.setdefault(u, set())

        self._check_and_promote_all()

    def upsert_to_space(
        self,
        items: Sequence[Tuple[Uid, List[str]]],
        space_name: str,
    ) -> None:
        """Insert or update tokenized documents in a specific space.

        Args:
            items: Sequence of (uid, token_list) pairs.
            space_name: Logical space name.
        """
        if self._global_index is None:
            self._global_index = _BM25IndexCore(k1=self._k1, b=self._b)

        new_to_space: List[Tuple[Uid, List[str]]] = []
        for uid, tokens in items:
            u = Uid(str(uid))
            self._global_index.upsert(u, tokens)
            already_in = u in self._uid_to_spaces and space_name in self._uid_to_spaces[u]

            if not already_in:
                self._uid_to_spaces.setdefault(u, set()).add(space_name)
                self._space_counts[space_name] = self._space_counts.get(space_name, 0) + 1
                new_to_space.append((u, tokens))
            elif space_name not in self._space_indexes:
                self._unpromoted[u] = tokens
            else:
                self._space_indexes[space_name].upsert(u, tokens)

        if space_name in self._space_indexes and new_to_space:
            for u, t in new_to_space:
                self._space_indexes[space_name].upsert(u, t)
        elif new_to_space:
            for u, t in new_to_space:
                self._unpromoted[u] = t

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

    def search(self, query: List[str], top_k: int) -> List[Tuple[Uid, float]]:
        if self._global_index is None:
            return []
        return self._global_index.search(query, max(1, int(top_k)))

    def search_in_space(
        self,
        query: List[str],
        space_name: str,
        candidates: Optional[Set[Uid]],
        top_k: int,
    ) -> List[Tuple[Uid, float]]:
        """Search within a space, using the promoted index or building a temp one.

        Args:
            query: Tokenized query as a list of strings.
            space_name: Logical space name.
            candidates: If provided, restrict search to these Uids.
            top_k: Number of top results.

        Returns:
            List of (uid, bm25_score) pairs sorted descending.
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

        # Build a temporary core index for the unpromoted space
        uids_list = list(target_uids)
        tokens_list = [self._unpromoted.get(u) for u in uids_list if u in self._unpromoted]
        if not tokens_list:
            return []

        core = _BM25IndexCore(k1=self._k1, b=self._b)
        for u, t in zip(uids_list, tokens_list):
            if t is not None:
                core.upsert(u, t)
        hits = core.search(query, top_k * 2)
        if candidates is not None:
            hits = [(u, s) for u, s in hits if u in candidates]
        return hits[:top_k]

    def rebuild(self, items: Sequence[Tuple[Uid, List[str]]]) -> None:
        """Completely rebuild the global index from scratch.

        Args:
            items: Sequence of (uid, token_list) pairs.
        """
        self._global_index = _BM25IndexCore(k1=self._k1, b=self._b)
        self._uid_to_spaces.clear()
        self._space_counts.clear()
        self._space_indexes.clear()
        self._unpromoted.clear()

        for uid, tokens in items:
            u = Uid(str(uid))
            self._global_index.upsert(u, tokens)
            self._uid_to_spaces.setdefault(u, set())

    def rebuild_space(self, space_name: str, items: Sequence[Tuple[Uid, List[str]]]) -> None:
        for uid, tokens in items:
            u = Uid(str(uid))
            self._uid_to_spaces.setdefault(u, set()).add(space_name)
        self._space_counts[space_name] = len(items)

        if space_name in self._space_indexes:
            self._space_indexes[space_name].clear()
            for u, t in items:
                self._space_indexes[space_name].upsert(u, t)
        elif len(items) >= self._threshold:
            core = _BM25IndexCore(k1=self._k1, b=self._b)
            for u, t in items:
                core.upsert(u, t)
            self._space_indexes[space_name] = core
        else:
            for u, t in items:
                self._unpromoted[u] = t

    def _check_and_promote_all(self) -> None:
        for space_name in list(self._space_counts.keys()):
            self._check_and_promote(space_name)

    def _check_and_promote(self, space_name: str) -> None:
        """Promote a space to a core index if its count meets the threshold."""
        if space_name in self._space_indexes:
            return
        count = self._space_counts.get(space_name, 0)
        if count < self._threshold:
            return
        self._promote(space_name)

    def _promote(self, space_name: str) -> None:
        """Build a _BM25IndexCore for the space from unpromoted buffers."""
        promoted_uids = {
            uid for uid, spaces in self._uid_to_spaces.items()
            if space_name in spaces
        }
        if not promoted_uids:
            return

        tokens_list = [self._unpromoted.get(uid) for uid in promoted_uids]
        if not all(tokens_list):
            return

        core = _BM25IndexCore(k1=self._k1, b=self._b)
        for uid, tokens in zip(promoted_uids, tokens_list):
            if tokens is not None:
                core.upsert(uid, tokens)
        self._space_indexes[space_name] = core

        for uid in promoted_uids:
            self._unpromoted.pop(uid, None)


class _BM25IndexCore:
    """Optimized BM25 index core with Okapi BM25 scoring.

    Uses standard BM25 term-frequency saturation and document-length
    normalization as defined by Robertson & Zaragoza (2009).
    """

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = float(k1)  # Term-frequency saturation parameter
        self._b = float(b)    # Document-length normalization parameter
        self._uid_list: List[Uid] = []
        self._tokenized_docs: List[List[str]] = []
        self._uid_to_idx: Dict[Uid, int] = {}

        self._N = 0
        self._avgdl = 0.0
        self._df: Dict[str, int] = {}

    def upsert(self, uid: Uid, tokens: List[str]) -> None:
        u = Uid(str(uid))
        if u in self._uid_to_idx:
            idx = self._uid_to_idx[u]
            self._tokenized_docs[idx] = tokens
        else:
            idx = len(self._uid_list)
            self._uid_to_idx[u] = idx
            self._uid_list.append(u)
            self._tokenized_docs.append(tokens)
        self._recompute_stats()

    def delete(self, uid: Uid) -> None:
        u = Uid(str(uid))
        if u not in self._uid_to_idx:
            return
        idx = self._uid_to_idx.pop(u)
        removed_tokens = self._tokenized_docs.pop(idx)
        self._uid_list.pop(idx)

        # Re-index after removal to keep uid_to_idx consistent
        for i, stored_uid in enumerate(self._uid_list):
            if stored_uid in self._uid_to_idx:
                self._uid_to_idx[stored_uid] = i

        for term in set(removed_tokens):
            if term in self._df:
                self._df[term] = max(0, self._df[term] - 1)

        self._recompute_stats()

    def search(self, query: List[str], top_k: int) -> List[Tuple[Uid, float]]:
        if not query or not self._uid_list:
            return []

        import math

        q_tf: Dict[str, int] = {}
        for term in query:
            q_tf[term] = q_tf.get(term, 0) + 1

        def idf(term: str) -> float:
            n_q = self._df.get(term, 0)
            # BM25 smoothed IDF
            return math.log((self._N - n_q + 0.5) / (n_q + 0.5) + 1.0)

        scores: List[Tuple[Uid, float]] = []
        for idx, tokens in enumerate(self._tokenized_docs):
            dl = float(len(tokens))
            s = 0.0
            tf_counter: Dict[str, int] = {}
            for term in tokens:
                tf_counter[term] = tf_counter.get(term, 0) + 1

            for term, q_count in q_tf.items():
                f = float(tf_counter.get(term, 0))
                if f <= 0:
                    continue
                term_idf = idf(term)
                # BM25 scoring formula
                denom = f + self._k1 * (1.0 - self._b + self._b * dl / (self._avgdl + 1e-12))
                s += term_idf * (f * (self._k1 + 1.0)) / (denom + 1e-12)

            if s > 0:
                scores.append((self._uid_list[idx], float(s)))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def clear(self) -> None:
        """Remove all documents and statistics."""
        self._uid_list.clear()
        self._tokenized_docs.clear()
        self._uid_to_idx.clear()
        self._N = 0
        self._avgdl = 0.0
        self._df.clear()

    def _recompute_stats(self) -> None:
        """Recalculate N, average doc length, and term document frequencies."""
        self._N = len(self._tokenized_docs)
        if self._N == 0:
            self._avgdl = 0.0
            self._df.clear()
            return

        total_len = sum(len(tokens) for tokens in self._tokenized_docs)
        self._avgdl = total_len / self._N

        df_counter: Dict[str, int] = {}
        for tokens in self._tokenized_docs:
            for term in set(tokens):
                df_counter[term] = df_counter.get(term, 0) + 1
        self._df = df_counter
