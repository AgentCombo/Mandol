"""OpenAI-compatible HTTP-based LLM provider.

Sends chat messages to an OpenAI-compatible /chat/completions endpoint
and returns structured LLMChatResponse objects. Supports environment-variable
or direct-parameter API key injection.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

from ..ports.llm_provider import ChatMessage, LLMChatResponse, LLMProvider
from ..application.chunker import estimate_tokens


@dataclass(frozen=True, slots=True)
class OpenAICompatibleLLMConfig:
    """Configuration for the OpenAI-compatible chat completions API.

    Attributes:
        base_url: Base URL of the LLM service.
        api_key_env: Name of the environment variable holding the API key.
        timeout_s: HTTP request timeout in seconds.
    """

    base_url: str = os.getenv("MANDOL_LLM_BASE_URL", "https://api.openai.com/v1")
    api_key_env: str = os.getenv("MANDOL_LLM_API_KEY_ENV", "MANDOL_LLM_API_KEY")
    timeout_s: int = int(os.getenv("MANDOL_LLM_TIMEOUT_S", "60"))


class OpenAICompatibleLLMProvider(LLMProvider):
    """LLM provider backed by an OpenAI-compatible chat completions API.

    Sends message sequences to the /chat/completions endpoint and wraps
    the response in an LLMChatResponse. Extra **kwargs are forwarded as
    top-level JSON fields (e.g., top_p, stop).

    Attributes:
        _model: Default model identifier.
        _base_url: API base URL (trailing slash stripped).
        _timeout_s: HTTP request timeout in seconds.
        _api_key: Bearer token for authorization.
        _api_key_env: Fallback env-var name when api_key not explicitly passed.
    """

    def __init__(
        self,
        *,
        model: str,
        config: Optional[OpenAICompatibleLLMConfig] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_s: Optional[int] = None,
    ) -> None:
        self._model = str(model)
        cfg = config or OpenAICompatibleLLMConfig()

        self._base_url = str(base_url or cfg.base_url).rstrip("/")
        self._timeout_s = int(timeout_s or cfg.timeout_s)

        key = api_key
        if key is None:
            key = os.getenv(cfg.api_key_env)
        self._api_key = key
        self._api_key_env = cfg.api_key_env

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> LLMChatResponse:
        """Send chat messages and return the model's completion.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            model: Optional model override.
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Maximum tokens in the response.
            response_format: Format spec (e.g., {\"type\": \"json_object\"}).
            **kwargs: Additional provider-specific parameters forwarded to the API.

        Returns:
            LLMChatResponse with the model's text content and raw JSON.

        Raises:
            RuntimeError: If the API key is missing, HTTP status ≠ 200,
                or the response does not contain a valid content string.
        """
        if not self._api_key:
            raise RuntimeError(
                f"LLM api key is required; set env {self._api_key_env} or pass api_key=..."
            )

        try:
            import requests
        except Exception as e:  # pragma: no cover
            raise RuntimeError("requests is required for OpenAICompatibleLLMProvider") from e

        url = f"{self._base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
            "X-Request-ID": str(uuid.uuid4()),
        }

        payload: Dict[str, Any] = {
            "model": str(model or self._model),
            "messages": list(messages),
            "temperature": float(temperature),
        }
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)
        if response_format is not None:
            payload["response_format"] = dict(response_format)

        # Forward extra kwargs as top-level API fields, skipping None values and collisions
        for k, v in kwargs.items():
            if v is None:
                continue
            if k in payload:
                continue
            payload[k] = v

        _RETRYABLE_STATUS = {429, 500, 502, 503, 504}
        _MAX_RETRIES = 3
        _BASE_DELAY = 2.0
        _logger = logging.getLogger(__name__)

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                resp = requests.post(
                    url,
                    headers=headers,
                    data=json.dumps(payload, ensure_ascii=False),
                    timeout=self._timeout_s,
                )
            except (requests.ConnectionError, requests.Timeout) as exc:
                if attempt == _MAX_RETRIES:
                    raise RuntimeError(
                        f"LLM connection failed after {_MAX_RETRIES} attempts: {exc}"
                    ) from exc
                delay = _BASE_DELAY * (2 ** (attempt - 1))
                _logger.warning(
                    "LLM connection error (attempt %d/%d), retrying in %.1fs: %s",
                    attempt, _MAX_RETRIES, delay, exc,
                )
                time.sleep(delay)
                continue

            if resp.status_code == 200:
                break
            if resp.status_code not in _RETRYABLE_STATUS or attempt == _MAX_RETRIES:
                raise RuntimeError(
                    f"LLM non-200: {resp.status_code}, body={resp.text[:512]}"
                )
            delay = _BASE_DELAY * (2 ** (attempt - 1))
            _logger.warning(
                "LLM %d (attempt %d/%d), retrying in %.1fs: %s",
                resp.status_code, attempt, _MAX_RETRIES, delay, resp.text[:120],
            )
            time.sleep(delay)

        raw = resp.json()
        try:
            choice = (raw.get("choices") or [])[0]
            msg = (choice.get("message") or {})
            content = msg.get("content")
        except (KeyError, IndexError, TypeError):
            content = None

        if not isinstance(content, str):
            content = ""

        raw_usage = raw.get("usage", {})
        if isinstance(raw_usage, dict) and raw_usage.get("total_tokens"):
            usage = {
                "prompt_tokens": raw_usage.get("prompt_tokens", 0) or 0,
                "completion_tokens": raw_usage.get("completion_tokens", 0) or 0,
                "total_tokens": raw_usage.get("total_tokens", 0) or 0,
            }
        else:
            prompt_text = " ".join(
                str(m.get("content", "")) for m in messages if isinstance(m, dict)
            )
            prompt_tokens = estimate_tokens(prompt_text)
            completion_tokens = estimate_tokens(content)
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }

        return LLMChatResponse(content=content, raw=raw, usage=usage)
