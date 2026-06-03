"""OpenAI-compatible HTTP-based embedding provider.

Sends text/image inputs to an OpenAI-compatible /embeddings endpoint
and returns dense float32 vectors. Supports environment-variable-driven
configuration via OpenAICompatibleEmbeddingConfig.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..domain.types import Embedding
from ..ports.embedding_provider import EmbeddingProvider

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class OpenAICompatibleEmbeddingConfig:
    """Configuration for the OpenAI-compatible embeddings API endpoint.

    Attributes:
        base_url: Base URL of the embeddings service.
        api_path: Path appended to base_url (e.g., \"/embeddings\").
        token_env: Name of the environment variable holding the API key.
        timeout_s: HTTP request timeout in seconds.
    """

    base_url: str = os.getenv(
        "MANDOL_EMBEDDER_BASE_URL",
        "https://api.openai.com/v1",
    )
    api_path: str = os.getenv(
        "MANDOL_EMBEDDER_API_PATH",
        "/embeddings",
    )
    token_env: str = os.getenv(
        "MANDOL_EMBEDDER_API_KEY_ENV",
        "MANDOL_EMBEDDER_API_KEY",
    )
    timeout_s: int = int(os.getenv("MANDOL_EMBEDDER_TIMEOUT_S", "60"))


class OpenAICompatibleEmbeddingProvider(EmbeddingProvider):
    """Embedding provider backed by an OpenAI-compatible HTTP API.

    Sends text batches to the /embeddings endpoint using the requests
    library. The embedding dimensionality is auto-detected on the first
    successful response.

    Attributes:
        _model: Model identifier sent in the request payload.
        _dim: Inferred embedding dimensionality (set after first embed_text).
        _config: Endpoint and timeout configuration.
        _token: Bearer token for authorization.
    """

    def __init__(
        self,
        *,
        model: str,
        dim: int = 0,
        config: Optional[OpenAICompatibleEmbeddingConfig] = None,
        token: Optional[str] = None,
    ) -> None:
        self._model = str(model)
        self._dim = int(dim)
        self._config = config or OpenAICompatibleEmbeddingConfig()
        self._token = token or os.getenv(self._config.token_env)

    def embedding_dim(self) -> int:
        if int(self._dim) <= 0:
            raise RuntimeError("embedding dim is unknown; call embed_text once to infer")
        return int(self._dim)

    def embed_text(self, texts: Sequence[str], **kwargs: Any) -> List[Embedding]:
        return self._embed(list(texts))

    def embed_image_paths(self, image_paths: Sequence[str], **kwargs: Any) -> List[Embedding]:
        # Treat image paths as input strings; compatible with multimodal embedding endpoints
        return self._embed(list(image_paths))

    def _embed(self, inputs: List[str]) -> List[Embedding]:
        """Send a batch of inputs to the embeddings endpoint.

        Automatically shards large batches into smaller chunks and
        retries transient failures with exponential back-off.

        Args:
            inputs: List of text or image-path strings.

        Returns:
            List of float32 embedding vectors.

        Raises:
            RuntimeError: If the API token is missing, HTTP status ≠ 200,
                or the response format is unexpected.
        """
        if not inputs:
            return []
        if not self._token:
            raise RuntimeError(
                f"Embedding API token is required; set env {self._config.token_env} or pass token=..."
            )

        try:
            import requests  # noqa: F401
        except Exception as e:  # pragma: no cover
            raise RuntimeError("requests is required for OpenAICompatibleEmbeddingProvider") from e

        max_batch = int(os.getenv("MANDOL_EMBEDDER_MAX_BATCH_SIZE", "64"))
        shards: List[List[str]] = []
        for i in range(0, len(inputs), max_batch):
            shards.append(inputs[i : i + max_batch])

        results: List[Embedding] = []
        for shard_idx, shard in enumerate(shards):
            results.extend(self._embed_shard_with_retry(shard, shard_idx, len(shards)))
        return results

    def _embed_shard_with_retry(
        self,
        inputs: List[str],
        shard_idx: int,
        total_shards: int,
    ) -> List[Embedding]:
        """Embed a single shard with exponential-backoff retry.

        Retries on connection errors and 5xx server errors up to
        ``max_retries`` times with exponential back-off.

        Args:
            inputs: Texts for this shard.
            shard_idx: Zero-based shard index (for logging).
            total_shards: Total number of shards (for logging).

        Returns:
            List of float32 embedding vectors for this shard.
        """
        import requests

        max_retries = int(os.getenv("MANDOL_EMBEDDER_MAX_RETRIES", "3"))
        base_delay = float(os.getenv("MANDOL_EMBEDDER_RETRY_DELAY", "2.0"))

        url = f"{self._config.base_url.rstrip('/')}{self._config.api_path}"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._token}",
            "X-Request-ID": str(uuid.uuid4()),
        }
        payload: Dict[str, Any] = {"model": self._model, "input": inputs}

        last_exc: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.post(
                    url,
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=self._config.timeout_s,
                )
                if resp.status_code >= 500:
                    raise requests.exceptions.ConnectionError(
                        f"Server error {resp.status_code}: {resp.text[:256]}"
                    )
                if resp.status_code != 200:
                    raise RuntimeError(
                        f"Embedding endpoint non-200: {resp.status_code}, body={resp.text[:512]}"
                    )

                data = resp.json().get("data")
                if not isinstance(data, list):
                    raise RuntimeError(
                        f"Unexpected embeddings response: {resp.text[:512]}"
                    )

                out: List[Embedding] = []
                for item in data:
                    emb = item.get("embedding")
                    if emb is None:
                        raise RuntimeError(f"Missing embedding in response: {resp.text[:512]}")
                    arr = np.asarray(emb, dtype=np.float32).reshape(-1)
                    if int(self._dim) <= 0:
                        self._dim = int(arr.shape[0])
                    elif arr.shape[0] != int(self._dim):
                        raise RuntimeError(
                            f"Embedding dim mismatch: got {arr.shape[0]} expected {self._dim}"
                        )
                    out.append(arr)
                return out

            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
                last_exc = exc
                if attempt < max_retries:
                    delay = base_delay * (2 ** (attempt - 1))
                    logger.warning(
                        "Embedding shard %d/%d attempt %d/%d failed: %s; retrying in %.1fs",
                        shard_idx + 1,
                        total_shards,
                        attempt,
                        max_retries,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "Embedding shard %d/%d failed after %d attempts: %s",
                        shard_idx + 1,
                        total_shards,
                        max_retries,
                        exc,
                    )

        raise RuntimeError(
            f"Embedding request failed after {max_retries} retries"
        ) from last_exc


# Backward compatible aliases
UniApiEmbeddingConfig = OpenAICompatibleEmbeddingConfig
UniApiEmbeddingProvider = OpenAICompatibleEmbeddingProvider
