"""OpenAI-compatible HTTP-based cross-encoder reranker.

Sends query-document pairs to an OpenAI-compatible /v1/rerank endpoint
and returns relevance-scored MemoryUnit lists. Supports flexible text
extraction from unit raw_data via configurable key or fallback heuristics.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..domain.memory_unit import MemoryUnit
from ..ports.reranker import Reranker

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class OpenAICompatibleRerankConfig:
    """Configuration for the OpenAI-compatible reranking API endpoint.

    Attributes:
        base_url: Base URL of the reranker service.
        api_path: Path appended to base_url (e.g., \"/v1/rerank\").
        token_env: Name of the environment variable holding the API key.
        timeout_s: HTTP request timeout in seconds.
        return_documents: Whether the API should include document text in results.
    """

    base_url: str = os.getenv(
        "MANDOL_RERANKER_BASE_URL",
        "",
    )
    api_path: str = os.getenv(
        "MANDOL_RERANKER_API_PATH",
        "/v1/rerank",
    )
    token_env: str = os.getenv(
        "MANDOL_RERANKER_API_KEY_ENV",
        "MANDOL_RERANKER_API_KEY",
    )
    timeout_s: int = int(os.getenv("MANDOL_RERANKER_TIMEOUT_S", "60"))
    return_documents: bool = os.getenv("MANDOL_RERANKER_RETURN_DOCUMENTS", "0") in {"1", "true", "True"}


class OpenAICompatibleReranker(Reranker):
    """Cross-encoder reranker backed by an OpenAI-compatible /v1/rerank endpoint.

    Extracts text from each MemoryUnit's raw_data (preferring *text_key*,
    then falling back to common field names like 'content', 'summary'),
    sends the query-document pairs to the API, and pairs results back
    with the original MemoryUnit instances.

    Attributes:
        _model: Reranker model identifier.
        _config: Endpoint and timeout configuration.
        _token: Bearer token for authorization.
        _text_key: Preferred key in raw_data for the document text.
    """

    def __init__(
        self,
        *,
        model: str = "bge-reranker-v2-m3",
        config: Optional[OpenAICompatibleRerankConfig] = None,
        token: Optional[str] = None,
        text_key: str = "text_content",
    ) -> None:
        self._model = str(model)
        self._config = config or OpenAICompatibleRerankConfig()
        self._token = token or os.getenv(self._config.token_env)
        self._text_key = str(text_key)

    def rerank(
        self,
        query: str,
        units: List[MemoryUnit],
        *,
        top_k: int = 10,
    ) -> List[Tuple[MemoryUnit, float]]:
        """Rerank candidates against a query via the remote API.

        Args:
            query: Natural language query string.
            units: List of candidate MemoryUnits.
            top_k: Maximum number of results to return.

        Returns:
            List of (MemoryUnit, relevance_score) pairs sorted descending.

        Raises:
            RuntimeError: If the API token is missing or HTTP status ≠ 200.
        """
        if not isinstance(query, str) or not query.strip() or not units:
            return []
        if not self._token:
            raise RuntimeError(
                f"Rerank API token is required; set env {self._config.token_env} or pass token=..."
            )

        documents: List[str] = []
        unit_order: List[MemoryUnit] = []
        for u in units:
            doc = self._extract_text(u)
            if isinstance(doc, str) and doc.strip():
                documents.append(doc)
                unit_order.append(u)

        if not documents:
            return []

        payload: Dict[str, Any] = {
            "model": self._model,
            "query": query,
            "documents": documents,
            "top_n": int(top_k) if top_k and top_k > 0 else 5,
            "return_documents": bool(self._config.return_documents),
        }

        try:
            import requests
        except Exception as e:  # pragma: no cover
            raise RuntimeError("requests is required for OpenAICompatibleReranker") from e

        max_retries = int(os.getenv("MANDOL_RERANKER_MAX_RETRIES", "3"))
        base_delay = float(os.getenv("MANDOL_RERANKER_RETRY_DELAY", "2.0"))

        url = f"{self._config.base_url.rstrip('/')}{self._config.api_path}"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._token}",
            "X-Request-ID": str(uuid.uuid4()),
        }

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
                        f"Rerank endpoint non-200: {resp.status_code}, body={resp.text[:512]}"
                    )

                data = resp.json()
                results = data.get("results") or []
                if not isinstance(results, list):
                    return []

                paired: List[Tuple[MemoryUnit, float]] = []
                for item in results:
                    try:
                        idx = int(item.get("index"))
                        score = float(item.get("relevance_score"))
                    except (KeyError, ValueError, TypeError):
                        continue
                    if 0 <= idx < len(unit_order):
                        paired.append((unit_order[idx], score))

                paired.sort(key=lambda x: x[1], reverse=True)
                return paired[:top_k]

            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
                last_exc = exc
                if attempt < max_retries:
                    delay = base_delay * (2 ** (attempt - 1))
                    logger.warning(
                        "Rerank attempt %d/%d failed: %s; retrying in %.1fs",
                        attempt,
                        max_retries,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "Rerank failed after %d attempts: %s",
                        max_retries,
                        exc,
                    )

        raise RuntimeError(
            f"Rerank request failed after {max_retries} retries"
        ) from last_exc

    def _extract_text(self, unit: MemoryUnit) -> str:
        """Extract a document string from a MemoryUnit's raw_data.

        Tries the configured *text_key* first, then falls back through
        common field names ('content', 'summary', etc.). As a last resort,
        concatenates all string-valued raw_data entries as key: value pairs.

        Args:
            unit: The MemoryUnit to extract text from.

        Returns:
            A non-empty string representation, or the Uid as fallback.
        """
        if unit is None or unit.raw_data is None:
            return ""
        raw = unit.raw_data

        # Prefer explicit text key
        val = raw.get(self._text_key)
        if isinstance(val, str) and val.strip():
            return val

        for k in ["content", "summary", "title", "message", "text"]:
            v = raw.get(k)
            if isinstance(v, str) and v.strip():
                return v

        parts: List[str] = []
        for k, v in raw.items():
            if k == "embedding":
                continue
            if isinstance(v, str) and v.strip():
                parts.append(f"{k}: {v}")
        return "; ".join(parts) if parts else str(unit.uid)


# Backward compatible aliases
UniApiRerankConfig = OpenAICompatibleRerankConfig
UniApiReranker = OpenAICompatibleReranker
