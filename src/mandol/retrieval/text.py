"""Text extraction and tokenization utilities.

Provides TextExtractor for pulling text content from MemoryUnit raw_data
and Tokenizer with optional jieba support for Chinese word segmentation.
"""

from __future__ import annotations

import re
from typing import List

from ..domain.memory_unit import MemoryUnit


class TextExtractor:
    """Extracts text content from a MemoryUnit's raw_data dict.

    Tries the configured primary_key first, then falls back to common
    keys (text, content, summary, title, message), and finally any
    string-valued key.

    Args:
        primary_key: Preferred key in raw_data for text extraction
            (default \"text_content\").
    """
    def __init__(self, *, primary_key: str = "text_content") -> None:
        self._primary_key = str(primary_key)

    def extract(self, unit: MemoryUnit) -> str:
        """Extract text from a memory unit.

        Args:
            unit: The MemoryUnit to extract text from.

        Returns:
            Extracted text string, or \"\" if nothing found.
        """
        raw = unit.raw_data or {}
        val = raw.get(self._primary_key)
        if isinstance(val, str) and val.strip():
            return val
        for k in ["text", "content", "summary", "title", "message"]:
            v = raw.get(k)
            if isinstance(v, str) and v.strip():
                return v
        for k, v in raw.items():
            if isinstance(v, str) and v.strip():
                return v
        return ""


class Tokenizer:
    """Tokenizes text for sparse/BM25 retrieval.

    Supports jieba for Chinese word segmentation as an optional dependency.
    When jieba is unavailable, falls back to simple whitespace splitting.

    Args:
        use_jieba: Enable jieba segmentation for Chinese text (default True).
    """
    def __init__(self, *, use_jieba: bool = True) -> None:
        self._use_jieba = bool(use_jieba)
        self._jieba = None
        if self._use_jieba:
            try:
                import jieba  # type: ignore

                self._jieba = jieba
            except ImportError:
                self._jieba = None

    def tokenize(self, text: str) -> List[str]:
        """Split text into tokens.

        Uses jieba for Chinese text if enabled and available, otherwise
        splits on whitespace.

        Args:
            text: Input text to tokenize.

        Returns:
            List of token strings.
        """
        if not isinstance(text, str):
            return []
        t = text.strip().lower()
        if not t:
            return []
        if self._jieba is not None:
            toks = [w.strip() for w in self._jieba.cut(t) if w and w.strip()]
            return [w for w in toks if len(w) > 1]

        toks = re.findall(r"\w+", t, flags=re.UNICODE)
        if toks:
            return [w for w in toks if len(w) > 1]

        compact = re.sub(r"\s+", "", t)
        if not compact:
            return []
        return [ch for ch in compact]
