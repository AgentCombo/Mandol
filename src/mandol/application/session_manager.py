"""Session detection and segmentation for conversational memory.

Uses LLM-based analysis to identify topic boundaries in continuous dialogue
streams, splitting long conversations into discrete sessions.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..domain.types import Uid
from ..ports.llm_provider import ChatMessage, LLMProvider
from ._llm_retry import retry_llm_json_call, strip_json_fences

logger = logging.getLogger(__name__)


# System prompt: semantic-first session boundaries, flat JSON
# (reasoning + boundaries + should_wait) for reliable parsing.
SESSION_SYSTEM_PROMPT = """You are an episodic memory boundary expert for conversational memory (MemCells / sessions).

**CRITICAL LANGUAGE RULE**: The `reasoning` string MUST use the SAME language as the conversation fragments below. If the fragments are mostly Chinese, write `reasoning` in Chinese; if mostly English, use English. No mixed-language reasoning.

**Core principle**: default to merging; split only on clear semantic / episodic breaks. Prefer **topic and discourse flow** over raw time gaps: a long silence alone is NOT enough unless the new messages clearly start an unrelated episode.

### Session (episode) definition
A session = one coherent thread participants could remember as a single episode (same concrete goal, problem, plan, or tightly related subtopics).

### When to add a boundary (1-based message index — see Output)
Add a boundary only on **clear** signals, including:
- **Strong topic shift**: the conversation moves to a clearly unrelated subject (e.g. deployment debugging → weekend travel plans).
- **Explicit episode handoff**: phrases like "anyway", "changing topic", "let's talk about …" that start a new substantive thread (not a brief aside).
- **Task closure + unrelated follow-up**: the closing line of a task stays in that episode; split only when the **next** messages open a genuinely new episode.
- **Time as a weak hint only**: a large gap may support a split if combined with a **clear** new topic; never split solely because timestamps are far apart.

### Do NOT split for
- Greetings, thanks, short acknowledgements ("ok", "got it") — keep with the episode they attach to.
- Light asides or "by the way" that still relate to the same thread.
- Natural drift within one meeting or one ongoing problem.
- **System / placeholder-only tail**: if the last lines are only `[image]`/`[video]` with no text, or bare "ok"/emoji with no topic — use `should_wait` instead of inventing a boundary.

### `should_wait`
Set `should_wait` to true when the **end of the batch** is too thin to fix an episode boundary responsibly:
- Only non-text placeholders or minimal replies at the end.
- System-like lines that do not carry conversational topic.
- Ambiguous tail where you cannot tell if the same episode continues.

When `should_wait` is true, set `boundaries` to [] (do not split this batch; downstream will see more context later).

### Multiple boundaries
Only if **two or more** clear breaks exist in one batch. Use semantic topic change as the sole split criterion; do not enforce a minimum segment size.

### Output format (single JSON object only)
No markdown fences, no text before or after the JSON. All strings one line (no raw line breaks inside `reasoning`).

Schema:
{
  "reasoning": "<one concise sentence summarizing all boundary decisions>",
  "boundaries": [<int>, ...],
  "should_wait": <boolean>
}

**boundaries**: each integer is a **1-based line index** taken from the prefix `[k]` in the log. Meaning: split **after** that message — the next line starts a **new** session. Example: boundaries [4] means lines [1]–[4] stay in the current session; line [5] onward start the next session.
An empty boundaries array [] means no split in this batch.

### Example A (English)
Input lines (abbreviated):
[1] ... Alice: Can you debug login?
[2] ... Bob: Checking logs.
[3] ... Bob: Found a null pointer in AuthService.
[4] ... Alice: Fixed, thanks!
[5] ... Alice: Lunch today?
[6] ... Bob: 12:30 works.

Output:
{"reasoning": "Messages 1-4 complete the bugfix episode; message 5 opens a new lunch topic.", "boundaries": [4], "should_wait": false}

### Example B (no boundary)
Output:
{"reasoning": "All lines belong to one ongoing roadmap discussion without a clear new episode.", "boundaries": [], "should_wait": false}

### Example C (non-English)
Input lines are mostly non-English about the same project. Output:
{"reasoning": "The entire conversation stays on the same requirement clarification without a new independent topic, therefore no split.", "boundaries": [], "should_wait": false}"""


SESSION_USER_PROMPT = """Current session id: {session_id}

{previous_reasoning_block}
### Memory fragments (chronological, same speakers)
Each line format: [1-based index] timestamp: text

{content}

Return exactly one JSON object as defined in the system message (no markdown, no commentary outside JSON)."""


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
    chinese_chars = len([c for c in text if "一" <= c <= "鿿"])
    english_chars = len([c for c in text if c.isalpha() and ord(c) < 128])
    other_chars = len(text) - chinese_chars - english_chars
    return int(chinese_chars * 0.6 + english_chars * 0.3 + other_chars * 0.4)


def _format_previous_reasoning_block(previous_reasoning: str, max_tokens: int = 300) -> str:
    """Format the previous reasoning for injection into the user prompt.

    Truncates from the tail if it exceeds max_tokens.

    Args:
        previous_reasoning: The reasoning string from the previous batch.
        max_tokens: Maximum token budget for the block.

    Returns:
        Empty string if previous_reasoning is empty, or a formatted block.
    """
    if not previous_reasoning.strip():
        return ""
    if estimate_tokens(previous_reasoning) > max_tokens:
        # Truncate from tail: keep last ~(max_tokens - overhead) tokens
        # Simple approach: keep last N chars proportional to max_tokens
        chars_per_token = len(previous_reasoning) / max(1, estimate_tokens(previous_reasoning))
        keep_chars = int((max_tokens - 20) * chars_per_token)
        previous_reasoning = "…(truncated) " + previous_reasoning[-keep_chars:]
    return f"### Previous batch rationale (reference only)\nThe previous batch's analysis concluded: {previous_reasoning}\n"


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
        topic: Topic of the first new session (often inferred downstream).
        split_points: All split points when multiple splits are detected.
        reasoning: One-sentence LLM rationale (new JSON schema); may be empty.
        should_wait: When True, the tail was ambiguous — no split was applied for this batch.
    """

    should_split: bool
    split_at_index: Optional[int] = None
    topic: str = ""
    split_points: List[SessionSplitPoint] = field(default_factory=list)
    reasoning: str = ""
    should_wait: bool = False


class SessionManager:
    """Detects and manages conversational session boundaries.

    Provides a single-entry LLM-based batch analysis method (analyze_batch)
    and session storage/query methods. Does NOT manage cross-batch state
    (reasoning chain, session assignment filtering, batch sizing) — that is the
    caller's responsibility (MemorySystem).

    Args:
        llm_provider: The LLM provider used for split decision analysis.
        max_unit_count: Maximum units per batch before forced analysis (default 20).
        time_gap_threshold_seconds: Time gap hint injected into the session
            detection prompt for LLM reference only. Not used for hard splitting.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        max_unit_count: int = 20,
        time_gap_threshold_seconds: int = 1800,
    ):
        self._llm = llm_provider
        self._max_unit_count = int(max_unit_count)
        self._time_gap_threshold = int(time_gap_threshold_seconds)  # Unused. Retained for backward compatibility.
        self._sessions: List[Session] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_batch(
        self,
        content_lines: List[str],
        session_id: str,
        previous_reasoning: str = "",
        on_warning: Optional[Callable[[str], None]] = None,
    ) -> SessionSplitDecision:
        """Analyze a single batch of formatted content lines for session boundaries.

        This is the single entry point for LLM-based session boundary detection.
        It assembles the system + user prompts, calls the LLM with retry, and
        parses the V2 JSON response.

        On LLM failure (retries exhausted): returns a no-split decision with a
        warning log. The caller is responsible for merging the batch into the
        current session without splitting.

        Args:
            content_lines: Formatted lines, each "[idx] timestamp: text".
            session_id: Current session ID for context.
            previous_reasoning: Reasoning from the previous batch (already
                truncated to ~300 tokens by the caller).
            on_warning: Optional callback for warning accumulation.

        Returns:
            SessionSplitDecision with boundaries, reasoning, and should_wait.
        """
        previous_block = _format_previous_reasoning_block(previous_reasoning)
        content = "\n".join(content_lines)

        combined = SESSION_SYSTEM_PROMPT + "\n\n" + SESSION_USER_PROMPT.format(
            session_id=session_id,
            previous_reasoning_block=previous_block,
            content=content,
        )
        messages: List[ChatMessage] = [
            {"role": "user", "content": combined},
        ]

        try:
            decision = retry_llm_json_call(
                self._llm,
                messages,
                lambda resp: self._parse_v2_response(resp, len(content_lines)),
                temperature=0.1,
                max_tokens=1024,
                context_label=f"session_split_{session_id}",
            )
            split_info = (
                ", ".join(
                    [
                        f"idx={sp.split_at_index}({sp.topic})"
                        for sp in decision.split_points
                    ]
                )
                if decision.split_points
                else "none"
            )
            logger.info(
                "SessionManager analyze_batch: should_split=%s should_wait=%s splits=[%s] reasoning=%.200s",
                decision.should_split,
                decision.should_wait,
                split_info,
                decision.reasoning or "",
            )
            return decision
        except json.JSONDecodeError:
            # retry_llm_json_call exhausted all retries — LLM persistently
            # returned unparseable output.
            msg = (
                f"[FALLBACK] Session split LLM returned unparseable JSON "
                f"for session {session_id} after all retries. "
                f"Returning no-split — {len(content_lines)} units "
                f"will be merged without boundary detection."
            )
            logger.error(msg)
            if on_warning:
                on_warning(msg)
        except (ConnectionError, TimeoutError, OSError, RuntimeError) as e:
            # Network, timeout, or provider-level API errors.
            msg = (
                f"[FALLBACK] Session split LLM call FAILED for session {session_id}: {e}. "
                f"Returning no-split — {len(content_lines)} units "
                f"will be merged without boundary detection."
            )
            logger.error(msg)
            if on_warning:
                on_warning(msg)
        return SessionSplitDecision(
            should_split=False,
            split_at_index=None,
            topic="",
            split_points=[],
            reasoning="",
            should_wait=False,
        )

    # ------------------------------------------------------------------
    # Session storage / query
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Internal: V2 JSON parsing
    # ------------------------------------------------------------------

    def _parse_v2_response(
        self,
        response: str,
        content_count: int,
    ) -> SessionSplitDecision:
        """Parse flat JSON: reasoning, boundaries (1-based after-index), should_wait."""
        data = json.loads(strip_json_fences(response))
        reasoning = str(data.get("reasoning", "")).replace("\n", " ").strip()
        should_wait = bool(data.get("should_wait", False))

        if should_wait:
            logger.info(
                "Session split deferred (should_wait): %.200s",
                reasoning or "(no reasoning)",
            )
            return SessionSplitDecision(
                should_split=False,
                split_at_index=None,
                topic="",
                split_points=[],
                reasoning=reasoning,
                should_wait=True,
            )

        raw_bounds = data.get("boundaries", [])
        if not isinstance(raw_bounds, list):
            raw_bounds = []

        boundaries: List[int] = []
        for b in raw_bounds:
            try:
                boundaries.append(int(b))
            except (ValueError, TypeError):
                continue

        boundaries = sorted(set(boundaries))
        split_points: List[SessionSplitPoint] = []

        # 1-based "after line b" => 0-based first index of new session == b
        max_b = content_count - 1
        for b in boundaries:
            if 1 <= b <= max_b:
                split_points.append(
                    SessionSplitPoint(
                        split_at_index=b,
                        topic="",
                        reason=reasoning,
                    ),
                )
            else:
                logger.warning(
                    "boundary %s invalid for batch size %s (valid 1..%s), skipping",
                    b,
                    content_count,
                    max_b,
                )

        should_split = len(split_points) > 0
        return SessionSplitDecision(
            should_split=should_split,
            split_at_index=split_points[0].split_at_index if split_points else None,
            topic="",
            split_points=split_points,
            reasoning=reasoning,
            should_wait=False,
        )

    # ------------------------------------------------------------------
    # Helpers (used by MemorySystem)
    # ------------------------------------------------------------------

    def build_session(
        self,
        session_id: str,
        unit_uids: List[str],
        topic: str = "",
        start_time: str = "",
        end_time: str = "",
    ) -> Session:
        """Build a Session object with the given parameters.

        Args:
            session_id: Unique session identifier.
            unit_uids: List of unit UID strings.
            topic: Human-readable topic label.
            start_time: ISO-format start timestamp.
            end_time: ISO-format end timestamp.

        Returns:
            The newly created Session.
        """
        session = Session(
            session_id=session_id,
            unit_uids=[Uid(u) for u in unit_uids],
            topic=topic,
            start_time=start_time,
            end_time=end_time,
        )
        self._sessions.append(session)
        return session
