"""Event and causal relationship extraction from conversational memory.

Uses LLM prompts to extract events and identify cause-effect relationships
between them within a given session context.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from ...domain.memory_unit import MemoryUnit
from ...ports.llm_provider import ChatMessage, LLMProvider

logger = logging.getLogger(__name__)

# Prompt for inferring causal relationships between events.
EVENT_CAUSAL_EXTRACT_PROMPT = """You are a causal relationship analysis expert. Given a list of events from a conversation session, identify causal relationships where one event causes or influences another.

**CRITICAL INSTRUCTIONS:**
- Output ONLY valid JSON format, no other text
- Focus on clear cause-effect relationships
- Cause must occur before effect chronologically
- Confidence score: 0.0 (weak) to 1.0 (strong)
- Only include relationships with confidence >= 0.6
- Event signatures must exactly match the original event descriptions
- Each event has a UID - use the UID for source tracking

**CAUSAL RELATIONSHIP TYPES TO CONSIDER:**
- direct_causal: Event A directly caused Event B (e.g., "led to", "caused", "triggered")
- indirect_causal: Event A contributed to or influenced Event B (e.g., "contributed to", "facilitated", "enabled")
- conditional_causal: Event A caused Event B under specific conditions (e.g., "when X happened, A caused B")
- temporal_causal: Event A preceded and caused Event B (e.g., "after", "following", "subsequently")
- emotional_causal: An emotional state caused a behavior or decision (e.g., "fear led to", "excitement caused")
- behavioral_causal: A decision or action caused an outcome (e.g., "decided to", "chose to", "action resulted in")

**EVENTS TO ANALYZE:**
{events_json}

**REQUIRED OUTPUT FORMAT:**
{{
  "causal_relationships": [
    {{
      "cause_event": "string - exact event description",
      "effect_event": "string - exact event description",
      "causal_type": "string - one of: direct_causal, indirect_causal, conditional_causal, temporal_causal, emotional_causal, behavioral_causal",
      "confidence_score": "float",
      "rationale": "string - brief explanation in English",
      "source_uids": ["string - UID of cause event", "string - UID of effect event"]
    }}
  ]
}}

Generate a response in JSON format. All textual content within the JSON must be in English."""

# Prompt for extracting event descriptions from conversation records.
EVENT_EXTRACTION_PROMPT = """You are an event extraction expert. Given a list of memory records, identify and extract all meaningful events mentioned.

**EVENT TYPES TO EXTRACT:**
- action_event: Specific actions taken or decisions made (e.g., "decided to migrate database", "cancelled the meeting")
- state_change: Changes in status, condition, or situation (e.g., "server went down", "status changed to active")
- communication_event: Exchanges of information (e.g., "sent email to team", "received feedback")
- temporal_event: Time-bound occurrences (e.g., "every Monday", "during Q3", "after the release")
- relationship_event: Events involving relationships (e.g., "joined the team", "left the company")

**CRITICAL INSTRUCTIONS:**
- Output ONLY valid JSON format, no other text
- Extract events that are specific enough to be meaningful
- Include timestamp when available
- Each event must have a unique UID for tracking
- Confidence score: 0.0 (weak) to 1.0 (strong)

**RECORDS TO ANALYZE:**
{records_json}

**REQUIRED OUTPUT FORMAT:**
{{
  "events": [
    {{
      "signature": "string - concise event description (max 100 chars)",
      "text": "string - full event description",
      "event_type": "string - one of: action_event, state_change, communication_event, temporal_event, relationship_event",
      "timestamp": "string - when the event occurred if known",
      "confidence": "float",
      "uid": "string - unique identifier for this event"
    }}
  ]
}}

Generate a response in JSON format. All textual content within the JSON must be in English."""

CAUSAL_TYPES = [
    "direct_causal",
    "indirect_causal",
    "conditional_causal",
    "temporal_causal",
    "emotional_causal",
    "behavioral_causal",
]


@dataclass
class ExtractedEvent:
    """An event extracted from conversational data.

    Attributes:
        signature: Concise event description (max 100 chars).
        text: Full event description.
        event_type: Category (action_event, state_change, etc.).
        timestamp: When the event occurred, if known.
        confidence: Extraction confidence in [0.0, 1.0].
        uid: Unique identifier for this event.
    """
    signature: str
    text: str
    event_type: str
    timestamp: str
    confidence: float
    uid: str


@dataclass
class CausalRelation:
    """A cause-effect relationship between two events.

    Attributes:
        cause_event: The cause event description.
        effect_event: The effect event description.
        causal_type: Type of causality (direct_causal, indirect_causal, etc.).
        confidence: Confidence score in [0.0, 1.0].
        rationale: Human-readable explanation.
        source_uids: UIDs of the events involved.
    """
    cause_event: str
    effect_event: str
    causal_type: str
    confidence: float
    rationale: str = ""
    source_uids: List[str] = field(default_factory=list)


class EventCausalExtractor:
    """Extracts events and causal relationships from conversations.

    Wraps an LLM provider with two tasks:
    1. Event extraction: identify events from raw records.
    2. Causal extraction: infer cause-effect chains between events.

    Args:
        llm_provider: LLM provider for extraction calls.
        confidence_threshold: Minimum confidence to include (default 0.6).
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        confidence_threshold: float = 0.6,
    ):
        self._llm = llm_provider
        self._threshold = float(confidence_threshold)

    def extract_events(
        self,
        records: Sequence[MemoryUnit],
    ) -> List[ExtractedEvent]:
        """Extract events from raw conversation records.

        Args:
            records: MemoryUnits with conversation text.

        Returns:
            List of ExtractedEvent objects.
        """
        if not records:
            return []

        record_list = []
        for i, r in enumerate(records):
            text = r.raw_data.get("text_content", "") or r.raw_data.get("text", "")
            if not text:
                continue
            ts = r.metadata.get("timestamp", "")
            speaker = r.metadata.get("speaker", "Unknown")
            record_list.append({
                "index": i,
                "text": text,
                "timestamp": ts,
                "speaker": speaker,
                "uid": str(r.uid),
            })

        if not record_list:
            return []

        content = json.dumps(record_list, ensure_ascii=False, indent=2)
        prompt = EVENT_EXTRACTION_PROMPT.format(records_json=content)

        messages: List[ChatMessage] = [
            {"role": "user", "content": prompt},
        ]

        try:
            response = self._llm.chat(
                messages,
                temperature=0.1,
                max_tokens=4096,
            )
            return self._parse_event_response(response.content)
        except (json.JSONDecodeError, ConnectionError, TimeoutError, OSError, RuntimeError) as e:
            logger.error("Event extraction failed: %s", e)
            return []

    def extract_causal_relations(
        self,
        events: Sequence[MemoryUnit],
        session_id: str,
    ) -> List[CausalRelation]:
        """Extract causal relationships between events.

        Args:
            events: Event MemoryUnits to analyze.
            session_id: Session identifier for context tracking.

        Returns:
            List of CausalRelation objects above the confidence threshold.
        """
        if not events or len(events) < 2:
            return []

        event_list = []
        for e in events:
            text = e.raw_data.get("text_content", "") or e.raw_data.get("text", "")
            if not text:
                continue
            signature = e.raw_data.get("signature", text[:100])
            event_list.append({
                "text": text,
                "signature": signature,
                "uid": str(e.uid),
                "timestamp": e.metadata.get("timestamp", ""),
            })

        if len(event_list) < 2:
            return []

        content = json.dumps(event_list, ensure_ascii=False, indent=2)
        prompt = EVENT_CAUSAL_EXTRACT_PROMPT.format(events_json=content)

        messages: List[ChatMessage] = [
            {"role": "user", "content": prompt},
        ]

        try:
            response = self._llm.chat(
                messages,
                temperature=0.1,
                max_tokens=2048,
            )
            return self._parse_response(response.content, event_list)
        except (json.JSONDecodeError, ConnectionError, TimeoutError, OSError, RuntimeError) as e:
            logger.error("Event causal extraction failed: %s", e)
            return []

    def _parse_event_response(
        self,
        response: str,
    ) -> List[ExtractedEvent]:
        """Parse the LLM JSON response for event extraction.

        Args:
            response: Raw LLM response string (JSON).

        Returns:
            List of ExtractedEvent objects, or empty on parse failure.
        """
        try:
            data = json.loads(response)
            events_data = data.get("events", [])

            results = []
            for item in events_data:
                signature = str(item.get("signature", "")).strip()
                text = str(item.get("text", "")).strip()
                if not signature and not text:
                    continue
                results.append(ExtractedEvent(
                    signature=signature or text[:100],
                    text=text or signature,
                    event_type=str(item.get("event_type", "action_event")),
                    timestamp=str(item.get("timestamp", "")),
                    confidence=float(item.get("confidence", 0.8)),
                    uid=str(item.get("uid", f"event_{len(results)}")),
                ))

            return results
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse event extraction response: {response[:200]}")
            return []

    def _parse_response(
        self,
        response: str,
        event_list: List[Dict[str, Any]],
    ) -> List[CausalRelation]:
        """Parse the LLM JSON response for causal relationship extraction.

        Validates that both cause and effect events exist in the event
        list and that confidence exceeds the threshold.

        Args:
            response: Raw LLM response string (JSON).
            event_list: Original event dicts for validation.

        Returns:
            List of validated CausalRelation objects.
        """
        try:
            data = json.loads(response)
            relations_data = data.get("causal_relationships", [])

            event_signatures = {e["signature"] for e in event_list}
            event_texts = {e["text"] for e in event_list}
            event_uids = {e["signature"]: e["uid"] for e in event_list}
            event_uids.update({e["text"]: e["uid"] for e in event_list})
            results = []

            for r in relations_data:
                cause = str(r.get("cause_event", ""))
                effect = str(r.get("effect_event", ""))
                causal_type = str(r.get("causal_type", "direct_causal"))
                conf = float(r.get("confidence_score", 0.0))

                if cause not in event_signatures and cause not in event_texts:
                    continue
                if effect not in event_signatures and effect not in event_texts:
                    continue
                if conf < self._threshold:
                    continue
                if causal_type not in CAUSAL_TYPES:
                    causal_type = "direct_causal"

                source_uids = []
                if cause in event_uids:
                    source_uids.append(event_uids[cause])
                if effect in event_uids:
                    source_uids.append(event_uids[effect])

                results.append(CausalRelation(
                    cause_event=cause,
                    effect_event=effect,
                    causal_type=causal_type,
                    confidence=conf,
                    rationale=str(r.get("rationale", "")),
                    source_uids=source_uids,
                ))

            return results
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse causal relation response: {response[:200]}")
            return []