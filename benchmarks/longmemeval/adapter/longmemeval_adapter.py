"""LongMemEval dataset adapter for writing dialogue sessions into a Mandol graph.

Provides helper functions to load a LongMemEval sample from the JSON dataset
and insert its haystack sessions and dialogues as memory units with
PRECEDES / FOLLOWS edges into a :class:`SemanticGraphService`.

The LongMemEval dataset format:
    - ``question_id``: unique question identifier
    - ``question_type``: e.g. "single-session-user"
    - ``question``: the question text
    - ``question_date``: date of the question
    - ``answer``: ground-truth answer
    - ``answer_session_ids``: session IDs containing the answer
    - ``haystack_dates``: list of date strings, one per haystack session
    - ``haystack_session_ids``: list of session ID strings
    - ``haystack_sessions``: list of lists, each inner list is a dialogue
      with turns like ``{"role": "user"/"assistant", "content": "..."}``
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mandol.application.semantic_graph import SemanticGraphService
from mandol.domain.memory_unit import MemoryUnit
from mandol.domain.types import SpaceName, Uid

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LongMemEvalAdapterConfig:
    """Configuration for the LongMemEval adapter.

    Attributes:
        dataset_path: Path to the LongMemEval JSON dataset file.
    """
    dataset_path: str


def load_longmemeval_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """Load the full LongMemEval dataset from a JSON file.

    Args:
        dataset_path: Path to the LongMemEval JSON dataset.

    Returns:
        The list of sample dicts.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If the dataset root is not a list.
    """
    p = Path(str(dataset_path))
    if not p.exists():
        raise FileNotFoundError(str(p))

    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("dataset root must be a list")
    return data


def load_longmemeval_sample(
    *, dataset_path: str, question_id: str
) -> Dict[str, Any]:
    """Load a single LongMemEval sample by question_id from the dataset file.

    Args:
        dataset_path: Path to the LongMemEval JSON dataset.
        question_id: The ``question_id`` to look up.

    Returns:
        The sample dict.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If the dataset root is not a list.
        KeyError: If *question_id* is not found.
    """
    p = Path(str(dataset_path))
    if not p.exists():
        raise FileNotFoundError(str(p))

    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("dataset root must be a list")

    for item in data:
        if isinstance(item, dict) and item.get("question_id") == question_id:
            return item

    raise KeyError(f"sample not found: {question_id}")


def _parse_longmemeval_datetime(dt_str: str) -> Optional[str]:
    """Parse a LongMemEval datetime string into ISO 8601 format.

    LongMemEval datetimes look like ``"2023/05/30 (Tue) 23:40"``.
    Returns an ISO-format string suitable for ``metadata["timestamp"]``,
    or ``None`` when parsing fails.

    Args:
        dt_str: Raw datetime string from the LongMemEval dataset.

    Returns:
        ISO 8601 datetime string or ``None``.
    """
    if not dt_str or not dt_str.strip():
        return None
    dt_str = dt_str.strip()

    # Format: "2023/05/30 (Tue) 23:40"
    formats = [
        "%Y/%m/%d (%a) %H:%M",
        "%Y/%m/%d (%a)%H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(dt_str, fmt)
            return dt.replace(tzinfo=timezone.utc).isoformat()
        except ValueError:
            continue
    logger.warning("Could not parse LongMemEval datetime: %r", dt_str)
    return None


def _format_dialogue_text(
    session_id: str,
    didx: int,
    date_str: str,
    role: str,
    content: str,
) -> str:
    """Format a dialogue turn as a MemoryUnit text_content string.

    Uses the format: ``Dialogue D{session_id}:{didx} [Time {date}]: {role} said: {content}``
    This matches the convention used in the LoCoMo benchmark adapter.

    Args:
        session_id: The haystack session ID.
        didx: The dialogue turn index within the session.
        date_str: The date string for the session.
        role: The role of the speaker ("user" or "assistant").
        content: The dialogue content.

    Returns:
        Formatted text content string.
    """
    return (
        f"Dialogue D{session_id}:{didx} [Time {date_str}]: "
        f"{role} said: {content}"
    )


def write_sample_to_graph(
    *,
    graph: SemanticGraphService,
    sample: Dict[str, Any],
    base_space_name: Optional[str] = None,
    batch_embed: bool = True,
) -> str:
    """Write a LongMemEval sample into *graph*.

    Creates the following space hierarchy and units:

    - Input space: ``{question_id}``
    - Base memory space: ``{question_id}_base_memory``
    - Session spaces: ``{question_id}_session_{N}`` (children of base memory)
    - Dialogue units: ``{question_id}_dialogue_{hash}`` with
      ``PRECEDES`` / ``FOLLOWS`` edges between consecutive dialogues.

    When *batch_embed* is True (default), units are first inserted without
    embeddings and then a single batch call computes all embeddings at
    once, which is significantly faster for remote embedding APIs.

    Args:
        graph: The :class:`SemanticGraphService` to populate.
        sample: A LongMemEval sample dict with ``question_id`` and
            ``haystack_sessions``.
        base_space_name: Override for the root space name.  Defaults to
            ``question_id``.
        batch_embed: If True, defer embedding computation and batch-compute
            at the end.  If False, compute embeddings one-by-one during
            insertion (slower but simpler).

    Returns:
        The base root space name (usually ``question_id``).

    Raises:
        ValueError: If required fields are missing or malformed.
    """
    qid = str(sample.get("question_id") or "").strip()
    if not qid:
        raise ValueError("sample.question_id is required")

    base_root = str(base_space_name or qid).strip()
    if not base_root:
        raise ValueError("base_space_name resolved to empty")

    sm = graph.semantic_map

    input_space = sm.create_space(SpaceName(base_root))
    base_space = sm.create_space(SpaceName(f"{base_root}_base_memory"))
    sm.attach_child_space(input_space.name, base_space.name, ensure_exists=True)

    haystack_sessions: List[List[Dict[str, Any]]] = sample.get("haystack_sessions") or []
    haystack_dates: List[str] = sample.get("haystack_dates") or []
    haystack_session_ids: List[str] = sample.get("haystack_session_ids") or []

    if not isinstance(haystack_sessions, list):
        raise ValueError("sample.haystack_sessions must be a list")

    session_count = len(haystack_sessions)

    prev_last_uid: Optional[str] = None
    global_didx = 0

    for sess_idx in range(session_count):
        session_id = (
            haystack_session_ids[sess_idx]
            if sess_idx < len(haystack_session_ids)
            else f"session_{sess_idx}"
        )
        date_str = (
            haystack_dates[sess_idx]
            if sess_idx < len(haystack_dates)
            else ""
        )

        session_space_name = f"{base_root}_session_{sess_idx}"
        session_space = sm.create_space(SpaceName(session_space_name))
        sm.attach_child_space(base_space.name, session_space.name, ensure_exists=True)

        dialogues: List[Dict[str, Any]] = haystack_sessions[sess_idx]
        if not isinstance(dialogues, list):
            continue

        ordered: List[Tuple[int, str]] = []

        for didx, turn in enumerate(dialogues):
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("role") or "").strip()
            content = str(turn.get("content") or "").strip()
            if not role or not content:
                continue

            text_content = _format_dialogue_text(
                session_id=session_id,
                didx=didx,
                date_str=date_str,
                role=role,
                content=content,
            )

            unit_uid = f"{base_root}_dialogue_{global_didx}"
            parsed_ts = _parse_longmemeval_datetime(date_str)

            unit = sm.get_unit(unit_uid)
            if unit is None:
                unit = MemoryUnit(
                    uid=Uid(unit_uid),
                    raw_data={
                        "type": "dialogue",
                        "session_id": session_id,
                        "session_index": sess_idx,
                        "dialogue_index": didx,
                        "global_index": global_didx,
                        "role": role,
                        "text": content,
                        "session_datetime": date_str,
                        "text_content": text_content,
                    },
                    metadata={
                        "unit_type": "dialogue",
                        "session_number": sess_idx,
                        "session_id": session_id,
                        "role": role,
                        **({"timestamp": parsed_ts} if parsed_ts else {}),
                    },
                    embedding=None,
                )
                graph.add_unit(
                    unit,
                    space_names=[session_space.name, base_space.name],
                    ensure_embedding=not batch_embed,
                )
            else:
                sm.add_unit_to_space(unit.uid, session_space.name)

            ordered.append((didx, unit_uid))
            global_didx += 1

        # Create PRECEDES / FOLLOWS edges within the session
        ordered.sort(key=lambda x: x[0])
        for (_, a), (_, b) in zip(ordered, ordered[1:]):
            if graph.get_relationship(a, b, "PRECEDES") is None:
                graph.add_relationship(a, b, "PRECEDES")
            if graph.get_relationship(b, a, "FOLLOWS") is None:
                graph.add_relationship(b, a, "FOLLOWS")

        # Create cross-session PRECEDES / FOLLOWS edges
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
