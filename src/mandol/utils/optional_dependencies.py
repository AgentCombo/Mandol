"""Helpers for optional runtime dependencies."""

from __future__ import annotations

from importlib.util import find_spec


def is_flash_attention_available() -> bool:
    """Return whether the optional ``flash_attn`` package can be imported."""
    try:
        return find_spec("flash_attn") is not None
    except (ImportError, ValueError):
        return False

