"""TF-IDF based sparse vector retriever.

Computes cosine similarity between TF-IDF weighted sparse vectors
for in-memory retrieval without a persistent sparse index.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from ..domain.memory_unit import MemoryUnit
from .text import TextExtractor, Tokenizer


@dataclass(slots=True)
class SparseScoredUnit:
    """A TF-IDF scored memory unit.

    Attributes:
        unit: The MemoryUnit.
        score: Cosine similarity score in [0.0, 1.0].
    """
    unit: MemoryUnit
    score: float


class TfidfSparseRetriever:
    """In-memory TF-IDF sparse vector retrieval.

    Computes TF-IDF vectors for query and candidates, then ranks by
    cosine similarity.

    Args:
        text_extractor: TextExtractor for pulling text from MemoryUnits.
        tokenizer: Tokenizer for word segmentation.
    """
    def __init__(
        self,
        *,
        text_extractor: Optional[TextExtractor] = None,
        tokenizer: Optional[Tokenizer] = None,
    ) -> None:
        self._text_extractor = text_extractor or TextExtractor(primary_key="text_content")
        self._tokenizer = tokenizer or Tokenizer()

    def search(self, query: str, candidates: Sequence[MemoryUnit], *, top_k: int) -> List[SparseScoredUnit]:
        """Score candidates by TF-IDF cosine similarity to the query.

        Args:
            query: The search query text.
            candidates: MemoryUnits to score.
            top_k: Number of top results to return.

        Returns:
            List of SparseScoredUnit sorted by score descending.
        """
        if not isinstance(query, str) or not query.strip() or not candidates:
            return []

        docs: List[List[str]] = []
        df: Counter = Counter()
        for u in candidates:
            toks = self._tokenizer.tokenize(self._text_extractor.extract(u))
            docs.append(toks)
            for t in set(toks):
                df[t] += 1

        q_toks = self._tokenizer.tokenize(query)
        if not q_toks:
            return []

        N = float(len(docs))

        def idf(term: str) -> float:
            n_q = float(df.get(term, 0))
            return math.log((N + 1.0) / (n_q + 1.0)) + 1.0

        q_tf = Counter(q_toks)
        q_vec: Dict[str, float] = {t: float(c) * idf(t) for t, c in q_tf.items()}
        q_norm = math.sqrt(sum(v * v for v in q_vec.values()))
        if q_norm <= 0:
            return []

        results: List[SparseScoredUnit] = []
        for u, toks in zip(candidates, docs):
            tf = Counter(toks)
            d_vec: Dict[str, float] = {t: float(c) * idf(t) for t, c in tf.items()}
            d_norm = math.sqrt(sum(v * v for v in d_vec.values()))
            if d_norm <= 0:
                continue

            dot = 0.0
            for t, qv in q_vec.items():
                dv = d_vec.get(t)
                if dv is None:
                    continue
                dot += qv * dv
            score = dot / (q_norm * d_norm + 1e-12)
            if score > 0:
                results.append(SparseScoredUnit(unit=u, score=float(score)))

        results.sort(key=lambda x: x.score, reverse=True)
        return results[: max(0, int(top_k))]
