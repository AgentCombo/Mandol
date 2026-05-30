"""Document chunking with token estimation.

Splits long text documents into overlapping chunks based on token counts,
using sentence boundaries for natural segmentation. Supports both tiktoken
(preferred) and a heuristic fallback for environments where tiktoken is not
available.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..domain.memory_unit import MemoryUnit
from ..domain.types import Uid

logger = logging.getLogger(__name__)

# Matches any single Chinese character in the Unicode CJK Unified Ideographs block.
CHINESE_CHARS_RE = re.compile(r"[\u4e00-\u9fff]")
# Matches sentence-ending punctuation for splitting text into sentences.
SENTENCE_ENDINGS = re.compile(r"[.!?。！？]+")


try:
    import tiktoken
    _ENCODING_CACHE: Optional[Any] = None

    def _get_encoding() -> Any:
        global _ENCODING_CACHE
        if _ENCODING_CACHE is None:
            _ENCODING_CACHE = tiktoken.get_encoding("cl100k_base")
        return _ENCODING_CACHE

    def estimate_tokens(text: str) -> int:
        """Estimate token count for a text string.

        Uses tiktoken's cl100k_base encoding when available, otherwise falls back
        to a heuristic that weights Chinese characters at 0.6, ASCII alphabetic
        characters at 0.3, and other characters at 0.4 tokens each.

        Args:
            text: The input text to estimate tokens for.

        Returns:
            Estimated token count as an integer.
        """
        encoding = _get_encoding()
        return len(encoding.encode(text, disallowed_special=()))

except ImportError:
    def estimate_tokens(text: str) -> int:
        chinese_chars = len(CHINESE_CHARS_RE.findall(text))
        english_chars = len([c for c in text if c.isalpha() and ord(c) < 128])
        other_chars = len(text) - chinese_chars - english_chars
        return int(chinese_chars * 0.6 + english_chars * 0.3 + other_chars * 0.4)


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using punctuation boundaries.

    Uses sentence-ending punctuation (., !, ?, 。，！，？) as split points
    and preserves the punctuation as part of the preceding sentence.

    Args:
        text: The input text to split.

    Returns:
        A list of sentence strings with trailing punctuation preserved.
    """
    if not text.strip():
        return []

    pattern = re.compile(r'([.!?。！？]+)')
    parts = pattern.split(text)
    sentences: List[str] = []
    current = ""

    for part in parts:
        part = part.strip()
        if not part:
            continue

        is_separator = bool(pattern.match(part)) if part else False

        if is_separator:
            current += part
            if current.strip():
                sentences.append(current.strip())
            current = ""
        else:
            current += part

    if current.strip():
        sentences.append(current.strip())

    return [s for s in sentences if s.strip()]


@dataclass
class ChunkResult:
    """Result of chunking a single document unit.

    Attributes:
        chunks: The list of chunk MemoryUnit objects produced from the document.
        parent_metadata: Metadata about the original document (uid, text, total chunks).
    """
    chunks: List[MemoryUnit]
    parent_metadata: Dict[str, Any]


class DocumentChunker:
    """Splits documents into smaller chunks based on token limits.

    Chunk boundaries are aligned with sentence endings to preserve readability.
    Optional token overlap allows context bleeding between adjacent chunks.

    Args:
        max_tokens: Maximum tokens per chunk (default 512).
        overlap_tokens: Number of tokens to overlap between chunks (default 0).
        text_key: Key in the unit's raw_data dict to extract text from.
    """

    def __init__(
        self,
        max_tokens: int = 512,
        overlap_tokens: int = 0,
        text_key: str = "text_content",
    ):
        self._max_tokens = int(max_tokens)
        self._overlap_tokens = int(overlap_tokens)
        self._text_key = str(text_key)

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    def should_chunk(self, unit: MemoryUnit) -> bool:
        """Check whether a unit needs chunking.

        Args:
            unit: The memory unit to evaluate.

        Returns:
            True if the unit's text token count exceeds max_tokens.
        """
        text = self._get_text(unit)
        if not text:
            return False
        return estimate_tokens(text) > self._max_tokens

    def chunk_unit(self, unit: MemoryUnit) -> ChunkResult:
        """Split a unit into sentence-aligned chunks.

        Sentences are accumulated until adding another would exceed max_tokens,
        at which point a chunk is emitted. Overlap from the previous chunk is
        preserved when overlap_tokens > 0.

        Args:
            unit: The memory unit to chunk.

        Returns:
            A ChunkResult containing the chunk units and parent metadata.
        """
        text = self._get_text(unit)
        if not text:
            return ChunkResult(chunks=[], parent_metadata={})

        sentences = split_into_sentences(text)
        chunks: List[MemoryUnit] = []
        current_chunk_parts: List[str] = []
        current_tokens = 0
        chunk_index = 0

        for sentence in sentences:
            sentence_tokens = estimate_tokens(sentence)

            if current_tokens + sentence_tokens > self._max_tokens and current_chunk_parts:
                chunk_text = " ".join(current_chunk_parts)
                chunk_uid = f"{unit.uid}:chunk:{chunk_index}"

                chunk_unit = MemoryUnit(
                    uid=Uid(chunk_uid),
                    raw_data={self._text_key: chunk_text},
                    metadata={
                        "type": "chunk",
                        "parent_uid": str(unit.uid),
                        "chunk_index": chunk_index,
                        "spaces": list(unit.metadata.get("spaces", [])),
                    },
                )
                chunks.append(chunk_unit)
                chunk_index += 1

                if self._overlap_tokens > 0 and current_chunk_parts:
                    overlap_parts = []
                    overlap_tokens = 0
                    for part in reversed(current_chunk_parts):
                        part_tokens = estimate_tokens(part)
                        if overlap_tokens + part_tokens > self._overlap_tokens:
                            break
                        overlap_parts.insert(0, part)
                        overlap_tokens += part_tokens
                    current_chunk_parts = overlap_parts
                    current_tokens = overlap_tokens
                else:
                    current_chunk_parts = []
                    current_tokens = 0

            current_chunk_parts.append(sentence)
            current_tokens += sentence_tokens

        if current_chunk_parts:
            chunk_text = " ".join(current_chunk_parts)
            chunk_uid = f"{unit.uid}:chunk:{chunk_index}"
            chunk_unit = MemoryUnit(
                uid=Uid(chunk_uid),
                raw_data={self._text_key: chunk_text},
                metadata={
                    "type": "chunk",
                    "parent_uid": str(unit.uid),
                    "chunk_index": chunk_index,
                    "spaces": list(unit.metadata.get("spaces", [])),
                },
            )
            chunks.append(chunk_unit)

        logger.debug(
            "Chunked unit %s: %d sentences → %d chunks (max_tokens=%d, tokens=~%d)",
            unit.uid, len(sentences), len(chunks),
            self._max_tokens, estimate_tokens(text),
        )

        parent_metadata: Dict[str, Any] = {
            "original_uid": str(unit.uid),
            "original_text": text,
            "total_chunks": len(chunks),
            "parent_type": "document",
        }

        return ChunkResult(chunks=chunks, parent_metadata=parent_metadata)

    def _get_text(self, unit: MemoryUnit) -> str:
        raw = unit.raw_data
        if isinstance(raw, dict):
            text = raw.get(self._text_key)
            if isinstance(text, str):
                return text
            for key, value in raw.items():
                if isinstance(value, str) and value.strip():
                    return value
        return ""

    def get_parent_space_name(self, unit_uid: Uid) -> str:
        """Return the space name for a parent unit's chunks.

        Args:
            unit_uid: The UID of the parent unit.

        Returns:
            A formatted space name string like \"parent:{unit_uid}\".
        """
        return f"parent:{unit_uid}"
