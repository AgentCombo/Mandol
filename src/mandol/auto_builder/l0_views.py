"""Utilities for separating L0 retrieval and inference views."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Tuple


_ORIGINAL_TEXT_FIELDS = (
    "original_content",
    "text_content",
    "content",
    "text",
    "message",
    "dialogue",
)

_TIMESTAMP_FIELDS = ("timestamp", "datetime", "date", "session_date", "created_at")


def _raw_data(unit: Any) -> Dict[str, Any]:
    return getattr(unit, "raw_data", None) or {}


def _metadata(unit: Any) -> Dict[str, Any]:
    return getattr(unit, "metadata", None) or {}


def extract_original_text(unit: Any) -> str:
    """Return the original human text for LLM inference, never enhanced_content."""
    raw = _raw_data(unit)

    for field in _ORIGINAL_TEXT_FIELDS:
        value = raw.get(field)
        if isinstance(value, str) and value.strip():
            return value

    dialogues = raw.get("dialogues") or raw.get("messages")
    if isinstance(dialogues, list):
        lines: List[str] = []
        for turn in dialogues:
            if isinstance(turn, dict):
                speaker = turn.get("speaker") or turn.get("role") or ""
                text = (
                    turn.get("text_content")
                    or turn.get("content")
                    or turn.get("text")
                    or turn.get("message")
                    or ""
                )
                if text:
                    lines.append(f"{speaker}: {text}" if speaker else str(text))
            elif turn:
                lines.append(str(turn))
        if lines:
            return "\n".join(lines)

    return ""


def extract_embedding_text(unit: Any) -> str:
    """Return the retrieval view for embeddings: enhanced text when present, else original."""
    raw = _raw_data(unit)
    enhanced = raw.get("enhanced_content")
    if isinstance(enhanced, str) and enhanced.strip():
        return enhanced
    return extract_original_text(unit)


def _group_key(unit: Any) -> str:
    meta = _metadata(unit)
    raw = _raw_data(unit)
    source_raw_uid = meta.get("source_raw_uid") or raw.get("source_raw_uid")
    if source_raw_uid:
        return str(source_raw_uid)
    if ("chunk_index" in meta or "total_chunks" in meta) and raw.get("source_uid"):
        return str(raw.get("source_uid"))
    return str(getattr(unit, "uid", ""))


def _chunk_index(unit: Any, fallback: int) -> int:
    meta = _metadata(unit)
    value = meta.get("chunk_index")
    if value is None:
        return fallback
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _has_chunk_metadata(units: Iterable[Any]) -> bool:
    for unit in units:
        meta = _metadata(unit)
        if "chunk_index" in meta or "total_chunks" in meta:
            return True
    return False


def _append_without_exact_overlap(base: str, next_text: str) -> str:
    if not base:
        return next_text
    if not next_text:
        return base

    max_overlap = min(len(base), len(next_text))
    for overlap in range(max_overlap, 0, -1):
        if base.endswith(next_text[:overlap]):
            return base + next_text[overlap:]
    return base + next_text


def _join_chunk_texts(units: List[Any]) -> str:
    merged = ""
    for unit in units:
        merged = _append_without_exact_overlap(merged, extract_original_text(unit))
    return merged


def _first_non_empty(unit: Any, fields: Tuple[str, ...]) -> str:
    raw = _raw_data(unit)
    meta = _metadata(unit)
    for field in fields:
        value = raw.get(field)
        if value:
            return str(value)
        value = meta.get(field)
        if value:
            return str(value)
    return ""


def _paragraph_prefix(units: List[Any]) -> str:
    if not units:
        return ""

    first = units[0]
    speaker = _first_non_empty(first, ("speaker", "role"))
    timestamp = _first_non_empty(first, _TIMESTAMP_FIELDS)

    if timestamp and speaker:
        return f"[{timestamp}] [{speaker}]: "
    if speaker:
        return f"[{speaker}]: "
    if timestamp:
        return f"[{timestamp}]: "
    return ""


def build_l0_inference_context(l0_units: List[Any]) -> str:
    """
    Rebuild the original logical L0 context for LLM reasoning.

    Retrieval-only fields such as enhanced_content are intentionally ignored.
    Physical chunks sharing source_raw_uid are sorted by chunk_index and joined
    without separators before a single speaker/timestamp prefix is applied.
    """
    groups: "OrderedDict[str, List[Tuple[int, Any]]]" = OrderedDict()
    for original_order, unit in enumerate(l0_units):
        key = _group_key(unit)
        groups.setdefault(key, []).append((original_order, unit))

    paragraphs: List[str] = []
    for _, entries in groups.items():
        units = [unit for _, unit in entries]
        chunked = _has_chunk_metadata(units)
        if chunked:
            ordered_units = [
                unit for _, unit in sorted(entries, key=lambda item: _chunk_index(item[1], item[0]))
            ]
            text = _join_chunk_texts(ordered_units)
        else:
            ordered_units = [unit for _, unit in sorted(entries, key=lambda item: item[0])]
            text = "\n".join(
                part for part in (extract_original_text(unit) for unit in ordered_units) if part
            )

        text = text.strip()
        if not text:
            continue

        prefix = _paragraph_prefix(ordered_units)
        paragraphs.append(f"{prefix}{text}" if prefix else text)

    return "\n\n".join(paragraphs)
