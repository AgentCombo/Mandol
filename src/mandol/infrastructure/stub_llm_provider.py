"""Stub (mock) LLM provider for testing.

Returns pre-configured responses without making any API calls.
Supports a single static response or a list consumed sequentially.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from ..ports.llm_provider import ChatMessage, LLMChatResponse, LLMProvider


@dataclass(slots=True)
class StubLLMProvider(LLMProvider):
    """Mock LLM provider that returns canned responses.

    Useful for unit testing components that depend on LLMProvider
    without incurring API costs or network dependency. When *contents*
    (plural) is provided, each chat() call returns the next string
    in the list sequentially.

    Attributes:
        content: Default static response string.
        raw: Raw response dict to attach to the LLMChatResponse.
        contents: Optional list of sequential responses (takes precedence).
        _i: Internal counter tracking position in *contents*.
    """

    content: str = ""
    raw: Optional[Dict[str, Any]] = None
    contents: Optional[List[str]] = None
    _i: int = 0

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
        """Return the next pre-configured response, ignoring all input arguments.

        Args:
            messages: Ignored.
            model: Ignored.
            temperature: Ignored.
            max_tokens: Ignored.
            response_format: Ignored.
            **kwargs: Ignored.

        Returns:
            LLMChatResponse with the canned content and raw dict.
        """
        _ = (messages, model, temperature, max_tokens, response_format, kwargs)
        if self.contents:
            idx = max(0, min(int(self._i), len(self.contents) - 1))
            self._i += 1
            return LLMChatResponse(content=str(self.contents[idx]), raw=dict(self.raw or {}))
        return LLMChatResponse(content=str(self.content), raw=dict(self.raw or {}))
