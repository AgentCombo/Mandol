"""Sentence-Transformers local cross-encoder reranker.

Loads a HuggingFace CrossEncoder model and scores query-document pairs
locally without API calls. Extracts document text from MemoryUnit raw_data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from ..domain.memory_unit import MemoryUnit
from ..ports.reranker import Reranker


@dataclass(frozen=True, slots=True)
class SentenceTransformersRerankerConfig:
    """Configuration for the Sentence-Transformers cross-encoder reranker.

    Attributes:
        max_length: Maximum token length for input pairs (truncation applied).
    """

    max_length: int = 512


class SentenceTransformersCrossEncoderReranker(Reranker):
    """Cross-encoder reranker backed by a local SentenceTransformer CrossEncoder.

    Loads the model on initialization and scores each (query, document) pair
    in a single predict call. Documents are extracted from MemoryUnit.raw_data
    using *text_key* or common fallback field names.

    Attributes:
        _model_name: HuggingFace CrossEncoder model identifier.
        _device: Torch device string.
        _text_key: Preferred key in raw_data for extracting document text.
        _cfg: Max-length and other configuration options.
        _model: The loaded CrossEncoder instance.
    """

    def __init__(
        self,
        *,
        model: str,
        device: str = "cpu",
        text_key: str = "text_content",
        config: SentenceTransformersRerankerConfig | None = None,
    ) -> None:
        self._model_name = str(model)
        self._device = str(device)
        self._text_key = str(text_key)
        self._cfg = config or SentenceTransformersRerankerConfig()

        try:
            from sentence_transformers import CrossEncoder
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "sentence-transformers is required for SentenceTransformersCrossEncoderReranker"
            ) from e

        self._model = CrossEncoder(self._model_name, device=self._device, max_length=int(self._cfg.max_length))

    def rerank(
        self,
        query: str,
        units: List[MemoryUnit],
        *,
        top_k: int = 10,
    ) -> List[Tuple[MemoryUnit, float]]:
        """Score and rank candidates against a query.

        Args:
            query: Natural language query string.
            units: List of candidate MemoryUnits.
            top_k: Maximum number of results to return.

        Returns:
            List of (MemoryUnit, relevance_score) pairs sorted descending.
        """
        if not isinstance(query, str) or not query.strip() or not units:
            return []

        documents: List[str] = []
        unit_order: List[MemoryUnit] = []
        for u in units:
            if u is None or u.raw_data is None:
                continue
            raw = u.raw_data
            v = raw.get(self._text_key)
            if not isinstance(v, str) or not v.strip():
                v = raw.get("content") or raw.get("summary") or raw.get("title") or raw.get("message") or raw.get("text") or ""
            if isinstance(v, str) and v.strip():
                documents.append(v)
                unit_order.append(u)

        if not documents:
            return []

        pairs = [(query, d) for d in documents]
        scores = self._model.predict(pairs)

        out: List[Tuple[MemoryUnit, float]] = []
        for i, s in enumerate(list(scores)):
            try:
                score = float(s)
            except (ValueError, TypeError):
                continue
            out.append((unit_order[i], score))

        out.sort(key=lambda x: x[1], reverse=True)
        return out[: max(0, int(top_k))]
