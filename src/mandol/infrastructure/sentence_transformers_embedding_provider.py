"""Sentence-Transformers local embedding provider.

Loads a HuggingFace SentenceTransformer model and generates dense
embeddings locally (no API calls). Supports configurable normalization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import numpy as np

from ..domain.types import Embedding
from ..ports.embedding_provider import EmbeddingProvider


@dataclass(frozen=True, slots=True)
class SentenceTransformersEmbeddingConfig:
    """Configuration for the Sentence-Transformers embedding provider.

    Attributes:
        normalize_embeddings: Whether to L2-normalize output vectors.
    """

    normalize_embeddings: bool = True


class SentenceTransformersEmbeddingProvider(EmbeddingProvider):
    """Embedding provider backed by a local SentenceTransformer model.

    Loads the model on initialization and infers the output dimensionality.
    Optionally L2-normalizes embeddings for cosine similarity search.

    Attributes:
        _model_name: HuggingFace model identifier.
        _device: Torch device string (\"cpu\", \"cuda\", etc.).
        _cfg: Normalization and other config options.
        _model: The loaded SentenceTransformer instance.
        _dim: Inferred output dimensionality.
    """

    def __init__(
        self,
        *,
        model: str,
        device: str = "cpu",
        config: Optional[SentenceTransformersEmbeddingConfig] = None,
    ) -> None:
        self._model_name = str(model)
        self._device = str(device)
        self._cfg = config or SentenceTransformersEmbeddingConfig()

        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "sentence-transformers is required for SentenceTransformersEmbeddingProvider"
            ) from e

        self._model = SentenceTransformer(self._model_name, device=self._device)

        # Infer dimension
        try:
            dim = int(getattr(self._model, "get_sentence_embedding_dimension")())
        except (AttributeError, RuntimeError, ValueError):
            # Fallback: run a single encode to measure output shape
            v = self._model.encode(["dim_probe"], normalize_embeddings=False)
            dim = int(np.asarray(v, dtype=np.float32).reshape(1, -1).shape[1])
        self._dim = dim

    def embedding_dim(self) -> int:
        return int(self._dim)

    def embed_text(self, texts: Sequence[str], **kwargs: Any) -> List[Embedding]:
        if not texts:
            return []
        vecs = self._model.encode(
            list(texts),
            normalize_embeddings=bool(self._cfg.normalize_embeddings),
            convert_to_numpy=True,
            show_progress_bar=bool(kwargs.get("show_progress_bar", False)),
        )
        arr = np.asarray(vecs, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return [arr[i].reshape(-1) for i in range(arr.shape[0])]

    def embed_image_paths(self, image_paths: Sequence[str], **kwargs: Any) -> List[Embedding]:
        # Sentence-Transformers text models do not support image inputs; treat paths as text
        return self.embed_text([str(p) for p in image_paths], **kwargs)
