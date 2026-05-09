"""Abstract interface for LLM (Large Language Model) providers.

Defines the chat completion contract and the LLMChatResponse dataclass
used by all LLM-backed components (entity extraction, deduplication,
session detection, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence


# A single chat message as an OpenAI-compatible dictionary (role, content, etc.)
ChatMessage = Dict[str, Any]


@dataclass(frozen=True, slots=True)
class LLMChatResponse:
    """Encapsulates an LLM chat completion response.

    Attributes:
        content: The main text content of the response.
        raw: The full raw response dictionary from the provider.
    """

    content: str
    raw: Dict[str, Any]


class LLMProvider(ABC):
    """Abstract interface for LLM chat completion providers."""

    @abstractmethod
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
        """Send a sequence of chat messages and return the model's response.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            model: Optional model override (provider-specific identifier).
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Maximum tokens in the response.
            response_format: Optional format spec (e.g., {\"type\": \"json_object\"}).
            **kwargs: Additional provider-specific parameters.

        Returns:
            LLMChatResponse with the model's text content and raw response data.
        """
        raise NotImplementedError
