"""Shared helpers for the unified fact pipeline.

Provides text content formatters, parsers, UID generators, JSON response
cleaners, and dialogue-matching utilities used by both the pipeline and
the cross-session coreference manager.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List, Sequence

from ...domain.memory_unit import MemoryUnit
from ...domain.types import Uid


# Regex matching "Entity {name}({type}): {description}" formatted strings.
_ENTITY_NAME_RE = re.compile(r"^Entity\s+(.+?)\(([^)]*)\)(?::\s*(.*))?$", re.DOTALL)
# Regex matching "Event {name}: {description}" formatted strings.
_EVENT_NAME_RE = re.compile(r"^Event\s+(.+?)(?::\s*(.*))?$", re.DOTALL)


def format_entity_text_content(name: str, entity_type: str, description: str = "") -> str:
    """Format entity fields into a canonical text_content string.

    Args:
        name: Entity name.
        entity_type: Entity type label (e.g. PERSON, ORGANIZATION).
        description: Optional description text.

    Returns:
        Formatted string like \"Entity name(type): description\".
    """
    if description:
        return f"Entity {name}({entity_type}): {description}"
    return f"Entity {name}({entity_type})"


def format_event_text_content(name: str, description: str = "") -> str:
    """Format event fields into a canonical text_content string.

    Args:
        name: Event name.
        description: Optional description text.

    Returns:
        Formatted string like \"Event name: description\".
    """
    if description:
        return f"Event {name}: {description}"
    return f"Event {name}"


def extract_entity_name_from_text_content(text_content: str) -> str:
    """Parse the entity name from a formatted text_content string.

    Args:
        text_content: An \"Entity {name}({type}): {description}\" string.

    Returns:
        The extracted entity name, or the raw text if parsing fails.
    """
    m = _ENTITY_NAME_RE.match(text_content.strip())
    if m:
        return m.group(1).strip()
    return text_content.strip()


def extract_entity_type_from_text_content(text_content: str) -> str:
    """Parse the entity type from a formatted text_content string.

    Args:
        text_content: An \"Entity {name}({type}): {description}\" string.

    Returns:
        The extracted entity type, or \"\" if parsing fails.
    """
    m = _ENTITY_NAME_RE.match(text_content.strip())
    if m:
        return m.group(2).strip()
    return ""


def extract_entity_desc_from_text_content(text_content: str) -> str:
    """Parse the entity description from a formatted text_content string.

    Args:
        text_content: An \"Entity {name}({type}): {description}\" string.

    Returns:
        The extracted description, or \"\" if no description is present.
    """
    m = _ENTITY_NAME_RE.match(text_content.strip())
    if m and m.group(3):
        return m.group(3).strip()
    return ""


def extract_event_name_from_text_content(text_content: str) -> str:
    """Parse the event name from a formatted text_content string.

    Args:
        text_content: An \"Event {name}: {description}\" string.

    Returns:
        The extracted event name, or the raw text if parsing fails.
    """
    m = _EVENT_NAME_RE.match(text_content.strip())
    if m:
        return m.group(1).strip()
    return text_content.strip()


def extract_event_desc_from_text_content(text_content: str) -> str:
    """Parse the event description from a formatted text_content string.

    Args:
        text_content: An \"Event {name}: {description}\" string.

    Returns:
        The extracted description, or \"\" if no description is present.
    """
    m = _EVENT_NAME_RE.match(text_content.strip())
    if m and m.group(2):
        return m.group(2).strip()
    return ""


def is_short_alias(text: str, max_words: int = 6) -> bool:
    """Check whether text is a short alias (not a full description).

    Aliases should be brief references; long strings are treated as
    descriptions not suitable for alias indexing.

    Args:
        text: The alias candidate string.
        max_words: Maximum word count for a valid alias (default 6).

    Returns:
        True if the text is non-empty and has max_words or fewer words.
    """
    if not text or not text.strip():
        return False
    word_count = len(text.strip().split())
    return word_count <= max_words


def generate_entity_uid(entity_name: str, entity_type: str = "", session_id: str = "") -> Uid:
    """Generate a deterministic UID for an entity.

    Based on entity name, type, and optional session ID with SHA-256
    hashing for uniqueness.

    Args:
        entity_name: The entity's name (max 50 chars used).
        entity_type: The entity type label.
        session_id: Optional session ID to scope the UID.

    Returns:
        A Uid of the form \"entity:{safe_name}_{hash[:8]}\".
    """
    safe_name = entity_name[:50].lower().replace(" ", "_").replace("'", "")
    hash_input = f"{entity_name.lower().strip()}|{entity_type.lower().strip()}"
    if session_id:
        hash_input += f"|{session_id}"
    short_hash = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()[:8]
    return Uid(f"entity:{safe_name}_{short_hash}")


def generate_event_uid(event_name: str, session_id: str = "") -> Uid:
    """Generate a deterministic UID for an event.

    Args:
        event_name: The event's name (max 50 chars used).
        session_id: Optional session ID to scope the UID.

    Returns:
        A Uid of the form \"event:{safe_name}_{hash[:8]}\".
    """
    safe_name = event_name[:50].lower().replace(" ", "_").replace("'", "")
    hash_input = f"{event_name.lower().strip()}"
    if session_id:
        hash_input += f"|{session_id}"
    short_hash = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()[:8]
    return Uid(f"event:{safe_name}_{short_hash}")


def parse_json_response(content: str) -> Dict[str, Any]:
    """Strip markdown fences and parse LLM response as JSON.

    Args:
        content: Raw LLM response text, possibly wrapped in ```json``` fences.

    Returns:
        Parsed dict.

    Raises:
        json.JSONDecodeError: If the content is not valid JSON or not a dict.
    """
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()

    result = json.loads(content)
    if not isinstance(result, dict):
        raise json.JSONDecodeError("Expected JSON object", content, 0)
    return result


def find_matching_dialogue_uids(
    mention_text: str,
    dialogue_units: Sequence[MemoryUnit],
) -> List[Uid]:
    """Find dialogue units whose text_content contains the mention text.

    Args:
        mention_text: The text snippet to search for.
        dialogue_units: Sequence of MemoryUnits to scan.

    Returns:
        List of UIDs of matching dialogue units (case-insensitive).
    """
    if not mention_text:
        return []
    matching = []
    mention_lower = mention_text.lower()
    for dia_unit in dialogue_units:
        dia_text = dia_unit.raw_data.get("text_content", "").lower()
        if mention_lower in dia_text:
            matching.append(dia_unit.uid)
    return matching
