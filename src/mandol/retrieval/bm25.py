"""BM25-based sparse text retriever.

Implements the Okapi BM25 scoring function for text-based retrieval
using in-memory document statistics and IDF weighting.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Sequence

from ..domain.memory_unit import MemoryUnit
from .text import TextExtractor, Tokenizer


@dataclass(slots=True)
class ScoredUnit:
    """A BM25-scored memory unit.

    Attributes:
        unit: The MemoryUnit.
        score: BM25 relevance score.
    """
    unit: MemoryUnit
    score: float


class Bm25Retriever:
    """In-memory BM25 retrieval over candidate MemoryUnits.

    Computes IDF from the candidate set and scores each document using
    the Okapi BM25 formula with configurable k1 and b parameters.

    Args:
        text_extractor: TextExtractor for pulling text from MemoryUnits.
        tokenizer: Tokenizer for word segmentation.
        k1: BM25 term saturation parameter (default 1.5).
        b: BM25 length normalization parameter (default 0.75).
    """
    def __init__(
        self,
        *,
        text_extractor: Optional[TextExtractor] = None,
        tokenizer: Optional[Tokenizer] = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self._text_extractor = text_extractor or TextExtractor(primary_key="text_content")
        self._tokenizer = tokenizer or Tokenizer()
        self._k1 = float(k1)
        self._b = float(b)

    def search(self, query: str, candidates: Sequence[MemoryUnit], *, top_k: int) -> List[ScoredUnit]:
        """Score candidates against a query using BM25.

        Args:
            query: The search query text.
            candidates: MemoryUnits to score.
            top_k: Number of top results to return.

        Returns:
            List of ScoredUnit results sorted by score descending.
        """
        if not isinstance(query, str) or not query.strip() or not candidates:
            return []

        docs_tokens: List[List[str]] = []
        docs_lens: List[int] = []
        tf_list: List[Counter] = []
        df: Counter = Counter()

        for u in candidates:
            txt = self._text_extractor.extract(u)
            toks = self._tokenizer.tokenize(txt)
            docs_tokens.append(toks)
            docs_lens.append(len(toks))
            tf = Counter(toks)
            tf_list.append(tf)
            for term in set(toks):
                df[term] += 1

        q_tokens = self._tokenizer.tokenize(query)
        if not q_tokens:
            return []

        N = len(docs_tokens)
        avgdl = sum(docs_lens) / max(1.0, float(len(docs_lens)))

        def idf(term: str) -> float:
            n_q = df.get(term, 0)
            return math.log((N - n_q + 0.5) / (n_q + 0.5) + 1.0)

        scores: List[ScoredUnit] = []
        for idx, tf in enumerate(tf_list):
            dl = float(docs_lens[idx])
            s = 0.0
            for term in q_tokens:
                if term not in tf:
                    continue
                term_idf = idf(term)
                f = float(tf[term])
                denom = f + self._k1 * (1.0 - self._b + self._b * dl / (avgdl + 1e-12))
                s += term_idf * (f * (self._k1 + 1.0)) / (denom + 1e-12)
            if s > 0:
                scores.append(ScoredUnit(unit=candidates[idx], score=float(s)))

        scores.sort(key=lambda x: x.score, reverse=True)
        return scores[: max(0, int(top_k))]
