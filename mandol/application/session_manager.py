"""Session detection and segmentation for conversational memory.

Uses LLM-based analysis to identify topic boundaries in continuous dialogue
streams, splitting long conversations into discrete sessions. Falls back to
midpoint splitting when LLM calls fail.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from ..domain.memory_unit import MemoryUnit
from ..domain.types import Uid
from ..ports.llm_provider import ChatMessage, LLMProvider

logger = logging.getLogger(__name__)


# System prompt instructing the LLM how to identify session boundaries
# in continuous conversations. Emphasizes conservative splitting (merge by
# default) and defines the JSON output schema for split decisions.
SESSION_SYSTEM_PROMPT = """You are a session segmentation expert. Your task is to identify conversational boundaries in a continuous dialog.

CRITICAL UNDERSTANDING:
- You are analyzing a CONTINUOUS CONVERSATION between the same people
- People naturally discuss multiple related subtopics within one conversation - this is NORMAL
- ONLY split when there is a CLEAR CONVERSATIONAL BREAK, not when topics drift slightly

SESSION DEFINITION:
A session = one complete conversational thread where participants discuss a set of related topics.

What to Split (CLEAR YES):
- Different days (explicit date/time change with new context)
- Explicit transitions: "Anyway...", "Let's talk about...", "Changing the subject..."
- One discussion clearly ends, new discussion starts with different focus

What NOT to Split (CLEAR NO):
- Same day, same conversation, even with topic variations
- Related subtopics (e.g., work → colleagues → office → commute)
- Brief mentions of other subjects
- Natural topic drift within a flowing conversation

Principles:
1. DEFAULT TO MERGING: Only split when there's a clear boundary
2. Session size: 10-25 units is typical. <8 units means over-splitting, >30 units means under-splitting
3. Time gaps: Reference only, never a split reason alone

IMPORTANT - Split Index Semantics:
- `split_at_index` means "the FIRST unit of the NEW session" (0-indexed)
- Example: 15 fragments [0]-[14], fragment [10] starts new session → split_at_index=10
- old_session = fragments[:split_at_index], new_session = fragments[split_at_index:]

MULTIPLE SPLITS - STRICT RULES:
- Only if 2+ CLEAR conversational breaks exist
- Each resulting session MUST have 8+ units
- NO adjacent indices (e.g., 12 and 13)
- If unsure about ANY split, DON'T make it

Output Format (JSON only):
{"should_split": true/false, "splits": [{"split_at_index": N, "new_session_topic": "...", "reason": "..."}]}"""

# User prompt template for session splitting. Includes the fragment content,
# current session ID, and detailed instructions on what constitutes a valid
# split boundary.
SESSION_USER_PROMPT = """Analyze these memory fragments from a continuous conversation.

Memory Fragments:
{content}

Current session number: {session_id}

ANALYSIS:
1. Read ALL fragments to understand the overall conversation flow
2. Look for CLEAR boundaries: different days, explicit transitions, new discussion starts
3. If splitting, each resulting session must be 8+ units

CRITICAL: `split_at_index` = FIRST fragment of the NEW session.
Example: [0]-[9] topic A, [10]-[14] topic B → split_at_index=10

DO NOT SPLIT FOR:
- Minor topic changes within same conversation
- Related subtopics or brief mentions
- Natural topic drift
- Adjacent indices (creates tiny sessions)

ONLY SPLIT FOR:
- Different days/times with clear new topic
- Explicit conversation transitions
- Clear breaks where one discussion ends and another begins

If uncertain, merge (should_split=false).

Output JSON only:"""


def estimate_tokens(text: str) -> int:
    """Estimate token count using a heuristic character-based model.

    Chinese characters: 0.6 tokens each.
    ASCII alphabetic: 0.3 tokens each.
    Other characters: 0.4 tokens each.

    Args:
        text: The input text to estimate.

    Returns:
        Estimated token count as an integer.
    """
    chinese_chars = len([c for c in text if "\u4e00" <= c <= "\u9fff"])
    english_chars = len([c for c in text if c.isalpha() and ord(c) < 128])
    other_chars = len(text) - chinese_chars - english_chars
    return int(chinese_chars * 0.6 + english_chars * 0.3 + other_chars * 0.4)


@dataclass
class Session:
    """A discrete conversational session.

    Attributes:
        session_id: Unique identifier for the session.
        unit_uids: Ordered list of MemoryUnit UIDs in this session.
        start_time: ISO-format timestamp of the first unit.
        end_time: ISO-format timestamp of the last unit.
        topic: Human-readable topic label for the session.
    """
    session_id: str
    unit_uids: List[Uid] = field(default_factory=list)
    start_time: str = ""
    end_time: str = ""
    topic: str = ""

    @property
    def unit_count(self) -> int:
        return len(self.unit_uids)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the session to a plain dict."""
        return {
            "session_id": self.session_id,
            "unit_uids": [str(u) for u in self.unit_uids],
            "start_time": self.start_time,
            "end_time": self.end_time,
            "topic": self.topic,
        }


@dataclass
class SessionSplitPoint:
    """A single split boundary within a session.

    Attributes:
        split_at_index: The fragment index where the new session starts (0-indexed).
        topic: Topic label for the new session.
        reason: Explanation for why this split was chosen.
    """
    split_at_index: int
    topic: str = ""
    reason: str = ""


@dataclass
class SessionSplitDecision:
    """Result of a session boundary analysis.

    Attributes:
        should_split: Whether the content should be split.
        split_at_index: Index of the first split point (for backwards compatibility).
        topic: Topic of the first new session.
        split_points: All split points when multiple splits are detected.
    """
    should_split: bool
    split_at_index: Optional[int] = None
    topic: str = ""
    split_points: List[SessionSplitPoint] = field(default_factory=list)


class SessionManager:
    """Detects and manages conversational session boundaries.

    Accumulates memory units and periodically invokes an LLM to decide whether
    a new session has started. Supports multiple split points per batch and
    validates that resulting sessions meet minimum size requirements.

    Args:
        llm_provider: The LLM provider used for split decision analysis.
        max_unit_count: Maximum units per batch before forced analysis (default 15).
        time_gap_threshold_seconds: Time gap in seconds that may suggest a boundary (default 1800).
        min_session_size: Minimum number of units a session must contain (default 8).
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        max_unit_count: int = 15,
        time_gap_threshold_seconds: int = 1800,
        min_session_size: int = 8,
    ):
        self._llm = llm_provider
        self._max_unit_count = int(max_unit_count)
        self._time_gap_threshold = int(time_gap_threshold_seconds)
        self._min_session_size = int(min_session_size)
        self._sessions: List[Session] = []

    def split_sessions(
        self,
        units: Sequence[MemoryUnit],
    ) -> List[Session]:
        """Segment a sequence of memory units into conversational sessions.

        Accumulates units into batches of max_unit_count and uses LLM analysis
        to detect topic boundaries. Multiple split points per batch are supported,
        and minimum session size is enforced.

        Args:
            units: The memory units to segment, sorted by timestamp.

        Returns:
            List of Session objects representing the detected sessions.
        """
        if not units:
            return []

        sorted_units = sorted(
            units,
            key=lambda u: u.metadata.get("timestamp", ""),
        )

        session_id_counter = len(self._sessions)
        sessions: List[Session] = []
        current_session_units: List[MemoryUnit] = []
        current_session_id = f"sess_{datetime.now(timezone.utc).strftime('%Y%m%d')}_{session_id_counter:03d}"

        for unit in sorted_units:
            current_session_units.append(unit)

            if len(current_session_units) >= self._max_unit_count:
                content_parts = []
                for i, u in enumerate(current_session_units):
                    text = u.raw_data.get("text_content", "")
                    timestamp = u.metadata.get("timestamp", "")
                    content_parts.append(f"[{i}] {timestamp}: {text}")

                decision = self._llm_split_decision(
                    content_parts,
                    current_session_id,
                )

                if decision.should_split and decision.split_points:
                    # Process multiple split points
                    remaining_units = list(current_session_units)
                    last_split_idx = 0
                    
                    for split_point in decision.split_points:
                        split_idx = split_point.split_at_index - last_split_idx
                        
                        # Validate split_idx against minimum session size
                        if split_idx < self._min_session_size:
                            logger.warning(f"Split at index {split_point.split_at_index} would create session of {split_idx} units (<{self._min_session_size} min), skipping this split")
                            continue
                        
                        if split_idx >= len(remaining_units):
                            logger.warning(f"Invalid split index {split_point.split_at_index} (relative={split_idx}) for {len(remaining_units)} units, skipping this split")
                            continue

                        session_units = remaining_units[:split_idx]
                        
                        if len(session_units) < self._min_session_size:
                            logger.warning(f"Split at index {split_point.split_at_index} produced session of {len(session_units)} units (<{self._min_session_size} min), skipping")
                            last_split_idx = split_point.split_at_index
                            continue

                        sessions.append(self._build_session(
                            current_session_id,
                            session_units,
                            split_point.topic or self._infer_topic(session_units),
                        ))

                        session_id_counter += 1
                        current_session_id = f"sess_{datetime.now(timezone.utc).strftime('%Y%m%d')}_{session_id_counter:03d}"
                        remaining_units = remaining_units[split_idx:]
                        last_split_idx = split_point.split_at_index
                    
                    current_session_units = remaining_units

        if current_session_units:
            sessions.append(self._build_session(
                current_session_id,
                current_session_units,
                self._infer_topic(current_session_units),
            ))

        self._sessions.extend(sessions)
        return sessions

    def _llm_split_decision(
        self,
        content_parts: List[str],
        session_id: str,
    ) -> SessionSplitDecision:
        content = "\n".join(content_parts)
        messages: List[ChatMessage] = [
            {"role": "system", "content": SESSION_SYSTEM_PROMPT},
            {"role": "user", "content": SESSION_USER_PROMPT.format(
                content=content[:8000],
                session_id=session_id,
            )},
        ]

        try:
            response = self._llm.chat(messages, temperature=0.1, max_tokens=512)
            decision = self._parse_split_response(response.content, len(content_parts))
            split_info = ", ".join([f"idx={sp.split_at_index}({sp.topic})" for sp in decision.split_points]) if decision.split_points else "none"
            logger.info(f"SessionManager LLM split: should_split={decision.should_split}, splits=[{split_info}]")
            return decision
        except Exception as e:
            logger.warning(f"LLM session split failed: {e}, using fallback")
            return self._fallback_split(len(content_parts))

    def _parse_split_response(
        self,
        response: str,
        content_count: int,
    ) -> SessionSplitDecision:
        try:
            data = json.loads(response)
            should_split = bool(data.get("should_split", False))
            
            # Support new "splits" array format
            splits_data = data.get("splits", [])
            split_points = []
            
            if should_split and splits_data:
                for sp in splits_data:
                    split_at = sp.get("split_at_index")
                    topic = str(sp.get("new_session_topic", ""))
                    reason = str(sp.get("reason", ""))
                    
                    if split_at is not None:
                        try:
                            split_at = int(split_at)
                            if 0 <= split_at < content_count:
                                split_points.append(SessionSplitPoint(
                                    split_at_index=split_at,
                                    topic=topic,
                                    reason=reason,
                                ))
                            else:
                                logger.warning(f"split_at_index={split_at} out of range [0, {content_count}), skipping")
                        except (ValueError, TypeError):
                            pass
            
            # Fallback to old format if no splits array
            if not split_points and should_split:
                split_at = data.get("split_at_index")
                topic = str(data.get("new_session_topic", ""))
                
                if split_at is not None:
                    try:
                        split_at = int(split_at)
                        if 0 <= split_at < content_count:
                            split_points.append(SessionSplitPoint(
                                split_at_index=split_at,
                                topic=topic,
                                reason="",
                            ))
                        else:
                            should_split = False
                    except (ValueError, TypeError):
                        should_split = False
                
                if should_split and not split_points:
                    should_split = False
            
            return SessionSplitDecision(
                should_split=should_split,
                split_at_index=split_points[0].split_at_index if split_points else None,
                topic=split_points[0].topic if split_points else "",
                split_points=split_points,
            )
        except json.JSONDecodeError:
            return SessionSplitDecision(should_split=False, split_at_index=None, topic="")

    def _fallback_split(
        self,
        content_count: int,
    ) -> SessionSplitDecision:
        split_idx = content_count // 2
        return SessionSplitDecision(
            should_split=True,
            split_at_index=split_idx,
            topic="",
        )

    def _build_session(
        self,
        session_id: str,
        units: List[MemoryUnit],
        topic: str,
    ) -> Session:
        session = Session(
            session_id=session_id,
            unit_uids=[Uid(str(u.uid)) for u in units],
            topic=topic,
        )

        if units:
            first_ts = units[0].metadata.get("timestamp", "")
            last_ts = units[-1].metadata.get("timestamp", "")
            session.start_time = first_ts
            session.end_time = last_ts

        return session

    def _infer_topic(self, units: List[MemoryUnit]) -> str:
        if not units:
            return "Untitled Session"

        first_text = units[0].raw_data.get("text_content", "")
        if len(first_text) > 80:
            return first_text[:80] + "..."
        return first_text

    def add_session(
        self,
        session_id: str,
        unit_uids: List[str],
    ) -> Session:
        """Manually register a session with the given units.

        Args:
            session_id: Unique session identifier.
            unit_uids: List of Unit UID strings.

        Returns:
            The newly created Session object.
        """
        session = Session(
            session_id=session_id,
            unit_uids=[Uid(u) for u in unit_uids],
        )
        self._sessions.append(session)
        return session

    def get_sessions(self) -> List[Session]:
        """Return all tracked sessions.

        Returns:
            Copy of the internal session list.
        """
        return list(self._sessions)

    def reset(self) -> None:
        """Clear all tracked sessions."""
        self._sessions.clear()

