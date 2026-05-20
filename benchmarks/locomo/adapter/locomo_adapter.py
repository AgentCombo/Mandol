"""LoCoMo dataset adapter for writing dialogue samples into a Mandol graph.

Provides helper functions to load a LoCoMo sample from the JSON dataset
and insert its sessions and dialogues as memory units with PRECEDES /
FOLLOWS edges into a :class:`SemanticGraphService`.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from mandol.application.semantic_graph import SemanticGraphService
from mandol.domain.memory_unit import MemoryUnit
from mandol.domain.types import SpaceName, Uid

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LocomoAdapterConfig:
    """Configuration for the LoCoMo adapter.

    Attributes:
        dataset_path: Path to the LoCoMo JSON dataset file.
    """
    dataset_path: str


def load_locomo_sample(*, dataset_path: str, sample_id: str) -> Dict[str, Any]:
    """Load a single LoCoMo sample by ID from the dataset file.

    Args:
        dataset_path: Path to the LoCoMo JSON dataset.
        sample_id: The ``sample_id`` to look up.

    Returns:
        The sample dict.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If the dataset root is not a list.
        KeyError: If *sample_id* is not found.
    """
    p = Path(str(dataset_path))
    if not p.exists():
        raise FileNotFoundError(str(p))

    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("dataset root must be a list")

    for item in data:
        if isinstance(item, dict) and item.get("sample_id") == sample_id:
            return item

    raise KeyError(f"sample not found: {sample_id}")


def _parse_session_number(key: str) -> Optional[int]:
    """Extract the numeric session index from a key like ``"session_3"``.

    Returns:
        The session number, or ``None`` if *key* does not match the pattern.
    """
    m = re.match(r"^session_(\d+)$", str(key))
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _parse_dialogue_index(dia_id: str, fallback: int) -> int:
    """Extract the dialogue index from a ``dia_id`` like ``"D3:5"``.

    Falls back to *fallback* when parsing fails.

    Returns:
        The zero-based dialogue index within its session.
    """
    try:
        if ":" in str(dia_id):
            return int(str(dia_id).split(":")[-1])
    except Exception:
        pass
    return int(fallback)


_LOCOMO_DATETIME_FORMATS = [
    "%I:%M %p on %d %B, %Y",
    "%I:%M%p on %d %B, %Y",
    "%I:%M %p on %d %B %Y",
    "%I:%M%p on %d %B %Y",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%dT%H:%M",
]


def _parse_locomo_datetime(dt_str: str) -> Optional[str]:
    """Parse a LoCoMo session datetime string into ISO 8601 format.

    LoCoMo datetimes look like ``"11:01 am on 17 December, 2022"``.
    Returns an ISO-format string suitable for ``metadata["timestamp"]``,
    or ``None`` when parsing fails.

    Args:
        dt_str: Raw datetime string from the LoCoMo dataset.

    Returns:
        ISO 8601 datetime string or ``None``.
    """
    if not dt_str or not dt_str.strip():
        return None
    dt_str = dt_str.strip()
    for fmt in _LOCOMO_DATETIME_FORMATS:
        try:
            dt = datetime.strptime(dt_str, fmt)
            return dt.replace(tzinfo=timezone.utc).isoformat()
        except ValueError:
            continue
    logger.warning("Could not parse LoCoMo datetime: %r. All %d format(s) failed.", dt_str, len(_LOCOMO_DATETIME_FORMATS))
    return None


def write_sample_to_graph(
    *,
    graph: SemanticGraphService,
    sample: Dict[str, Any],
    base_space_name: Optional[str] = None,
    batch_embed: bool = True,
) -> str:
    """Write a LoCoMo sample into *graph*.

    Creates the following space hierarchy and units:

    - Input space: ``{sample_id}``
    - Base memory space: ``{sample_id}_base_memory``
    - Session spaces: ``{sample_id}_session_{N}`` (children of base memory)
    - Dialogue units: ``{sample_id}_dialogue_{dia_id}`` with
      ``PRECEDES`` / ``FOLLOWS`` edges between consecutive dialogues.

    When *batch_embed* is True (default), units are first inserted without
    embeddings and then a single batch call computes all embeddings at
    once, which is significantly faster for remote embedding APIs.

    Args:
        graph: The :class:`SemanticGraphService` to populate.
        sample: A LoCoMo sample dict with ``sample_id`` and ``conversation``.
        base_space_name: Override for the root space name.  Defaults to
            ``sample_id``.
        batch_embed: If True, defer embedding computation and batch-compute
            at the end.  If False, compute embeddings one-by-one during
            insertion (slower but simpler).

    Returns:
        The base root space name (usually ``sample_id``).

    Raises:
        ValueError: If required fields are missing or malformed.
    """

    sid = str(sample.get("sample_id") or "").strip()
    if not sid:
        raise ValueError("sample.sample_id is required")

    base_root = str(base_space_name or sid).strip()
    if not base_root:
        raise ValueError("base_space_name resolved to empty")

    sm = graph.semantic_map

    input_space = sm.create_space(SpaceName(base_root))
    base_space = sm.create_space(SpaceName(f"{base_root}_base_memory"))
    sm.attach_child_space(input_space.name, base_space.name, ensure_exists=True)

    conv = sample.get("conversation") or {}
    if not isinstance(conv, dict):
        raise ValueError("sample.conversation must be a dict")

    sessions: List[Tuple[int, str, List[Dict[str, Any]]]] = []
    for k, v in conv.items():
        n = _parse_session_number(k)
        if n is None:
            continue
        if not isinstance(v, list):
            continue
        dt = conv.get(f"session_{n}_date_time", "")
        sessions.append((n, str(dt or ""), [x for x in v if isinstance(x, dict)]))
    sessions.sort(key=lambda x: x[0])

    prev_last_uid: Optional[str] = None

    for sess_n, sess_dt, dialogues in sessions:
        session_space_name = f"{base_root}_session_{sess_n}"
        session_space = sm.create_space(SpaceName(session_space_name))
        sm.attach_child_space(base_space.name, session_space.name, ensure_exists=True)

        ordered: List[Tuple[int, str]] = []
        for i, d in enumerate(dialogues):
            dia_id = str(d.get("dia_id") or "").strip()
            speaker = str(d.get("speaker") or "").strip()
            text = d.get("text")
            if text is None or str(text).strip() == "":
                text = d.get("text_content") or d.get("content") or ""
            text = str(text)
            if not text.strip():
                continue

            didx = _parse_dialogue_index(dia_id, i)
            if not dia_id:
                dia_id = f"D{sess_n}:{didx}"

            unit_uid = f"{base_root}_dialogue_{dia_id}"
            unit = sm.get_unit(unit_uid)
            if unit is None:
                blip_caption = d.get("blip_caption")
                parsed_ts = _parse_locomo_datetime(sess_dt)
                unit = MemoryUnit(
                    uid=Uid(unit_uid),
                    raw_data={
                        "type": "dialogue",
                        "dia_id": dia_id,
                        "speaker": speaker,
                        "text": text,
                        "dialogue_index": didx,
                        "session_datetime": sess_dt,
                        "img_url": d.get("img_url"),
                        "blip_caption": blip_caption,
                        "query": d.get("query"),
                        "text_content": f"Dialogue {dia_id} [Time {sess_dt}]: {speaker}{f' sent {blip_caption} and' if blip_caption else ''} said: {text}",
                    },
                    metadata={
                        "unit_type": "dialogue",
                        "session_number": sess_n,
                        "dia_id": dia_id,
                        "speaker": speaker,
                        **({"timestamp": parsed_ts} if parsed_ts else {}),
                    },
                    embedding=None,
                )
                graph.add_unit(unit, space_names=[session_space.name, base_space.name], ensure_embedding=not batch_embed)
            else:
                sm.add_unit_to_space(unit.uid, session_space.name)

            ordered.append((didx, unit_uid))

        ordered.sort(key=lambda x: x[0])
        for (_, a), (_, b) in zip(ordered, ordered[1:]):
            if graph.get_relationship(a, b, "PRECEDES") is None:
                graph.add_relationship(a, b, "PRECEDES")
            if graph.get_relationship(b, a, "FOLLOWS") is None:
                graph.add_relationship(b, a, "FOLLOWS")

        if prev_last_uid and ordered:
            first_uid = ordered[0][1]
            if graph.get_relationship(prev_last_uid, first_uid, "PRECEDES") is None:
                graph.add_relationship(prev_last_uid, first_uid, "PRECEDES")
            if graph.get_relationship(first_uid, prev_last_uid, "FOLLOWS") is None:
                graph.add_relationship(first_uid, prev_last_uid, "FOLLOWS")

        if ordered:
            prev_last_uid = ordered[-1][1]

    if batch_embed:
        count = sm.batch_embed_unembedded()
        logger.info("Batch-computed embeddings for %d units", count)

    return base_root
