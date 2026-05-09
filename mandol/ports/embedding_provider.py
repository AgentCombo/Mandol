"""Abstract interface for text/image embedding providers.

Defines the contract for generating dense vector embeddings from
raw text or image inputs. Also includes a StaticEmbeddingProvider
convenience class for testing with fixed-dimension dummy embeddings.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Sequence

import numpy as np

from ..domain.types import Embedding


class EmbeddingProvider(ABC):
    """Abstract interface for generating dense vector embeddings."""

    @abstractmethod
    def embed_text(self, texts: Sequence[str], **kwargs: Any) -> List[Embedding]:
        """Generate embeddings for a batch of text strings.

        Args:
            texts: Sequence of input text strings.
            **kwargs: Provider-specific options (e.g., normalize, instruction).

        Returns:
            List of Embedding arrays (float32), one per input text.
        """
        raise NotImplementedError

    @abstractmethod
    def embed_image_paths(
        self, image_paths: Sequence[str], **kwargs: Any
    ) -> List[Embedding]:
        """Generate embeddings for a batch of image file paths.

        Args:
            image_paths: Sequence of image file paths.
            **kwargs: Provider-specific options.

        Returns:
            List of Embedding arrays (float32), one per input image.
        """
        raise NotImplementedError

    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the output dimensionality of the embedding model.

        Returns:
            Integer number of dimensions.
        """
        raise NotImplementedError


class StaticEmbeddingProvider(EmbeddingProvider):
    """Embedding provider that returns constant-value vectors.

    Useful for testing and benchmarking without a real embedding model.

    Attributes:
        _dim: Output vector dimensionality.
        _fill: Constant value to fill every element with (default 0.0).
    """

    def __init__(self, dim: int, *, fill: float = 0.0):
        self._dim = int(dim)
        self._fill = float(fill)

    def embed_text(self, texts: Sequence[str], **kwargs: Any) -> List[Embedding]:
        return [np.full((self._dim,), self._fill, dtype=np.float32) for _ in texts]

    def embed_image_paths(
        self, image_paths: Sequence[str], **kwargs: Any
    ) -> List[Embedding]:
        return [np.full((self._dim,), self._fill, dtype=np.float32) for _ in image_paths]

    def embedding_dim(self) -> int:
        return self._dim
