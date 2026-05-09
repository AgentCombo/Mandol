"""Summary map reducer using LLM-based chunked reduction.

Reduces conversation memory into structured summaries (episodic, knowledge,
emotional, procedural) by accumulating units into token-budgeted chunks and
invoking LLM calls per chunk.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from ...domain.memory_unit import MemoryUnit
from ...domain.types import Uid
from ...ports.llm_provider import ChatMessage, LLMProvider
from ..session_manager import Session, estimate_tokens
from .summary_prompts import (
    EPISODIC_SUMMARY_PROMPT,
    PROCEDURAL_SUMMARY_PROMPT,
    KNOWLEDGE_SUMMARY_PROMPT,
    EMOTIONAL_SUMMARY_PROMPT,
    SUMMARY_TYPE_DEFINITIONS,
)

logger = logging.getLogger(__name__)

SUMMARY_REDUCE_PROMPT_TEMPLATE = """You are a memory integration expert. Merge multiple {category} summary fragments into a coherent session-level {category} summary.

Category Definition:
{SUMMARY_TYPE_DEFINITIONS}

Principles:
1. Preserve all important information, avoid information loss
2. Eliminate duplicates, merge similar viewpoints
3. Maintain chronological/logical order
4. Preserve the structured format of the {category} summary

Output Format:
{{
    "summaries": [
        {{
            "category": "{category}",
            "text": "Merged summary text",
            "key_source_uids": ["uid1", "uid2"],
            {category_fields}
        }}
    ]
}}

Generate a response in JSON format. All textual content within the JSON must be in English."""

CATEGORY_OUTPUT_FIELDS = {
    "episodic": '''"timeline": [],
    "key_people": [],
    "main_events": [],
    "location_info": [],
    "event_relationships": []''',
    "procedural": '''"process_name": [],
    "key_steps": [],
    "decision_points": [],
    "preconditions": [],
    "expected_outcomes": [],
    "optimization_opportunities": []''',
    "knowledge": '''"core_concepts": [],
    "key_facts": [],
    "techniques_methods": [],
    "prerequisites_knowledge": [],
    "related_concepts": [],
    "practical_applications": []''',
    "emotional": '''"user_preferences": [],
    "emotional_reactions": [],
    "behavioral_patterns": [],
    "satisfaction_factors": [],
    "frustration_points": [],
    "underlying_values": []''',
}

CATEGORY_PROMPTS = {
    "episodic": EPISODIC_SUMMARY_PROMPT,
    "procedural": PROCEDURAL_SUMMARY_PROMPT,
    "knowledge": KNOWLEDGE_SUMMARY_PROMPT,
    "emotional": EMOTIONAL_SUMMARY_PROMPT,
}

CATEGORIES = ["episodic", "knowledge", "emotional", "procedural"]

# Max total tokens sent to the LLM in a single summary call.
MAX_TOTAL_TOKENS = 8000
# Max number of MemoryUnit texts accumulated before a forced chunk flush.
MAX_UNITS_PER_CHUNK = 30


def _parse_list_field(value: Any) -> List[str]:
    """Parse a field that should be a list of strings.

    Handles cases where LLM returns:
    - A proper list: ["item1", "item2"]
    - A single string: "single_item"
    - A string that looks like a list: '["item1", "item2"]'
    - None or empty values
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    return [str(v) for v in parsed if v]
            except json.JSONDecodeError:
                pass
        return [stripped]
    return []


@dataclass
class SummaryResult:
    """Structured summary produced by the map-reduce pipeline.

    Attributes:
        category: Summary category — one of episodic, knowledge, emotional, procedural.
        text: Main summary text.
        key_points: Extracted key points from the summary.
        level: Granularity — 'chunk' or 'session'.
        session_id: Owning session identifier.
        source_uids: UIDs of the MemoryUnits that contributed to this summary.
        timeline: (episodic) Ordered list of timeline entries.
        key_people: (episodic) People mentioned in the summary.
        main_events: (episodic) Key events identified.
        location_info: (episodic) Location references.
        event_relationships: (episodic) Inter-event causal/temporal links.
        process_name: (procedural) Name of the described process.
        key_steps: (procedural) Steps in the process.
        decision_points: (procedural) Decision points encountered.
        preconditions: (procedural) Preconditions for the process.
        expected_outcomes: (procedural) Expected outcomes.
        optimization_opportunities: (procedural) Improvement suggestions.
        core_concepts: (knowledge) Core concepts identified.
        key_facts: (knowledge) Key factual statements.
        techniques_methods: (knowledge) Techniques and methods mentioned.
        prerequisites_knowledge: (knowledge) Prerequisite knowledge areas.
        related_concepts: (knowledge) Related conceptual links.
        practical_applications: (knowledge) Practical use cases.
        user_preferences: (emotional) User preference observations.
        emotional_reactions: (emotional) Emotional response patterns.
        behavioral_patterns: (emotional) Observed behavioral patterns.
        satisfaction_factors: (emotional) Satisfaction drivers.
        frustration_points: (emotional) Frustration triggers.
        underlying_values: (emotional) Core values inferred.
    """
    category: str
    text: str
    key_points: List[str] = field(default_factory=list)
    level: str = "chunk"
    session_id: str = ""
    source_uids: List[str] = field(default_factory=list)
    timeline: List[str] = field(default_factory=list)
    key_people: List[str] = field(default_factory=list)
    main_events: List[str] = field(default_factory=list)
    location_info: List[str] = field(default_factory=list)
    event_relationships: List[str] = field(default_factory=list)
    process_name: List[str] = field(default_factory=list)
    key_steps: List[str] = field(default_factory=list)
    decision_points: List[str] = field(default_factory=list)
    preconditions: List[str] = field(default_factory=list)
    expected_outcomes: List[str] = field(default_factory=list)
    optimization_opportunities: List[str] = field(default_factory=list)
    core_concepts: List[str] = field(default_factory=list)
    key_facts: List[str] = field(default_factory=list)
    techniques_methods: List[str] = field(default_factory=list)
    prerequisites_knowledge: List[str] = field(default_factory=list)
    related_concepts: List[str] = field(default_factory=list)
    practical_applications: List[str] = field(default_factory=list)
    user_preferences: List[str] = field(default_factory=list)
    emotional_reactions: List[str] = field(default_factory=list)
    behavioral_patterns: List[str] = field(default_factory=list)
    satisfaction_factors: List[str] = field(default_factory=list)
    frustration_points: List[str] = field(default_factory=list)
    underlying_values: List[str] = field(default_factory=list)


class SummaryMapReducer:
    """Map-reduce pipeline for generating multi-category session summaries.

    Accumulates MemoryUnits into token-budgeted chunks, calls the LLM per
    chunk per category (map phase), then pairwise-merges chunk summaries
    into a single session-level summary per category (reduce phase).

    Args:
        llm_provider: LLM provider for summary generation.
        map_chunk_tokens: Maximum tokens per LLM prompt chunk.
        prompt_tokens: Estimated overhead tokens for the prompt template.
        text_key: Key used to extract text from MemoryUnit.raw_data.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        map_chunk_tokens: int = 2560,
        prompt_tokens: int = 500,
        text_key: str = "text_content",
    ):
        self._llm = llm_provider
        self._map_chunk_tokens = int(map_chunk_tokens)
        self._prompt_tokens = int(prompt_tokens)
        self._content_limit = self._map_chunk_tokens - self._prompt_tokens
        self._text_key = str(text_key)

    def process_session(
        self,
        session: Session,
        units: Sequence[MemoryUnit],
    ) -> Dict[str, List[MemoryUnit]]:
        """Generate session-level summaries for all four categories.

        Runs the map phase (chunk-level summaries) followed by the reduce
        phase (pairwise merge into session-level summaries), then wraps
        each final summary as a MemoryUnit.

        Args:
            session: The session to summarise.
            units: All MemoryUnits available for the session.

        Returns:
            Dict mapping category name to a list containing one session-level
            summary MemoryUnit per category.
        """
        unit_map = {str(u.uid): u for u in units}
        session_units = [unit_map[str(uid)] for uid in session.unit_uids if str(uid) in unit_map]

        if not session_units:
            return {}

        chunks = self._create_chunks(session_units)
        chunk_summaries: List[Dict[str, SummaryResult]] = []

        for i, chunk_units in enumerate(chunks):
            summaries = self._map_phase(chunk_units, session.session_id, f"chunk_{i}")
            chunk_summaries.append(summaries)

        all_summaries_by_category = self._reduce_phase(chunk_summaries, session.session_id)

        summary_units: Dict[str, List[MemoryUnit]] = {
            "episodic": [],
            "knowledge": [],
            "emotional": [],
            "procedural": [],
        }

        for category, summaries in all_summaries_by_category.items():
            if not summaries:
                continue
            final = summaries[-1]
            uid = f"{session.session_id}:summary:{category}:session"
            raw_data = {
                self._text_key: final.text,
                "category": category,
                "level": "session",
            }
            metadata = {
                "type": "summary",
                "category": category,
                "level": "session",
                "session_id": session.session_id,
                "evidence_uids": final.source_uids,
            }
            if category == "episodic":
                raw_data["timeline"] = final.timeline
                raw_data["key_people"] = final.key_people
                raw_data["main_events"] = final.main_events
                raw_data["location_info"] = final.location_info
                raw_data["event_relationships"] = final.event_relationships
            elif category == "procedural":
                raw_data["process_name"] = final.process_name
                raw_data["key_steps"] = final.key_steps
                raw_data["decision_points"] = final.decision_points
                raw_data["preconditions"] = final.preconditions
                raw_data["expected_outcomes"] = final.expected_outcomes
                raw_data["optimization_opportunities"] = final.optimization_opportunities
            elif category == "knowledge":
                raw_data["core_concepts"] = final.core_concepts
                raw_data["key_facts"] = final.key_facts
                raw_data["techniques_methods"] = final.techniques_methods
                raw_data["prerequisites_knowledge"] = final.prerequisites_knowledge
                raw_data["related_concepts"] = final.related_concepts
                raw_data["practical_applications"] = final.practical_applications
            elif category == "emotional":
                raw_data["user_preferences"] = final.user_preferences
                raw_data["emotional_reactions"] = final.emotional_reactions
                raw_data["behavioral_patterns"] = final.behavioral_patterns
                raw_data["satisfaction_factors"] = final.satisfaction_factors
                raw_data["frustration_points"] = final.frustration_points
                raw_data["underlying_values"] = final.underlying_values
            for k, v in raw_data.items():
                metadata[k] = v
            unit = MemoryUnit(
                uid=Uid(uid),
                raw_data=raw_data,
                metadata=metadata,
            )
            summary_units[category].append(unit)

        return summary_units

    def _create_chunks(
        self,
        units: Sequence[MemoryUnit],
    ) -> List[List[MemoryUnit]]:
        chunks: List[List[MemoryUnit]] = []
        current_chunk: List[MemoryUnit] = []
        current_tokens = 0

        for unit in units:
            text = unit.raw_data.get(self._text_key, "")
            tokens = estimate_tokens(text)

            if current_tokens + tokens > self._content_limit and current_chunk:
                chunks.append(list(current_chunk))
                current_chunk = []
                current_tokens = 0

            current_chunk.append(unit)
            current_tokens += tokens

        if current_chunk:
            chunks.append(current_chunk)

        return chunks if chunks else [list(units)]

    def _map_phase(
        self,
        units: Sequence[MemoryUnit],
        session_id: str,
        chunk_label: str,
    ) -> Dict[str, SummaryResult]:
        results: Dict[str, SummaryResult] = {}

        for category in CATEGORIES:
            prompt_template = CATEGORY_PROMPTS[category]
            content_parts = []
            for i, unit in enumerate(units):
                text = unit.raw_data.get(self._text_key, "")
                ts = unit.metadata.get("timestamp", "")
                speaker = unit.metadata.get("speaker", "Unknown")
                uid = str(unit.uid)
                content_parts.append(
                    f"Record {i+1}(UID: {uid}):\nTime: {ts}\nSpeaker: {speaker}\nContent: {text}"
                )

            content = "\n".join(content_parts)
            prompt = prompt_template.format(records=content)

            messages: List[ChatMessage] = [
                {"role": "user", "content": prompt},
            ]

            try:
                response = self._llm.chat(
                    messages,
                    temperature=0.1,
                    max_tokens=2048,
                    response_format={"type": "json_object"},
                )
                result = self._parse_summary_response(
                    response.content, session_id, chunk_label, category, units
                )
            except Exception as e:
                logger.error(f"Summary map phase failed for {category}: {e}")
                result = self._create_fallback_summary(session_id, chunk_label, category, units)

            results[category] = result

        return results

    def _reduce_phase(
        self,
        chunk_summaries: Sequence[Dict[str, SummaryResult]],
        session_id: str,
    ) -> Dict[str, List[SummaryResult]]:
        if not chunk_summaries:
            return {}

        result: Dict[str, List[SummaryResult]] = {cat: [] for cat in CATEGORIES}

        for cat in CATEGORIES:
            cat_summaries = [cs.get(cat) for cs in chunk_summaries if cs.get(cat)]
            if not cat_summaries:
                continue

            current = list(cat_summaries)
            while len(current) > 1:
                next_round: List[SummaryResult] = []
                for i in range(0, len(current), 2):
                    if i + 1 < len(current):
                        merged = self._reduce_two(current[i], current[i + 1], session_id, cat)
                        next_round.append(merged)
                    else:
                        next_round.append(current[i])
                current = next_round

            if current:
                result[cat] = [current[0]]

        return result

    def _reduce_two(
        self,
        summary1: SummaryResult,
        summary2: SummaryResult,
        session_id: str,
        category: str,
    ) -> SummaryResult:
        if not summary1.text and not summary2.text:
            return summary1
        if not summary1.text:
            return summary2
        if not summary2.text:
            return summary1

        reduce_prompt = SUMMARY_REDUCE_PROMPT_TEMPLATE.format(
            category=category,
            SUMMARY_TYPE_DEFINITIONS=SUMMARY_TYPE_DEFINITIONS.get(category, ""),
            category_fields=CATEGORY_OUTPUT_FIELDS.get(category, ""),
        )

        messages: List[ChatMessage] = [
            {"role": "system", "content": reduce_prompt},
            {"role": "user", "content": f"""Merge the following {category} summaries into one coherent session-level summary.

Summary 1:
{self._format_summary_for_reduce(summary1, category)}

Summary 2:
{self._format_summary_for_reduce(summary2, category)}

Output JSON:"""},
        ]

        try:
            response = self._llm.chat(
                messages,
                temperature=0.1,
                max_tokens=1024,
                response_format={"type": "json_object"},
            )
            return self._parse_reduce_response(
                response.content, session_id, category, [summary1, summary2]
            )
        except Exception as e:
            logger.error(f"Summary reduce phase failed for {category}: {e}")
            return summary1

    def _format_summary_for_reduce(self, summary: SummaryResult, category: str) -> str:
        base_info = f"Text: {summary.text}\nKey Source UIDs: {summary.source_uids}"
        category_specific = ""
        if category == "episodic":
            category_specific = f"Timeline: {summary.timeline}\nKey People: {summary.key_people}\nMain Events: {summary.main_events}\nLocation Info: {summary.location_info}\nEvent Relationships: {summary.event_relationships}"
        elif category == "procedural":
            category_specific = f"Process Name: {summary.process_name}\nKey Steps: {summary.key_steps}\nDecision Points: {summary.decision_points}\nPreconditions: {summary.preconditions}\nExpected Outcomes: {summary.expected_outcomes}\nOptimization Opportunities: {summary.optimization_opportunities}"
        elif category == "knowledge":
            category_specific = f"Core Concepts: {summary.core_concepts}\nKey Facts: {summary.key_facts}\nTechniques Methods: {summary.techniques_methods}\nPrerequisites: {summary.prerequisites_knowledge}\nRelated Concepts: {summary.related_concepts}\nPractical Applications: {summary.practical_applications}"
        elif category == "emotional":
            category_specific = f"User Preferences: {summary.user_preferences}\nEmotional Reactions: {summary.emotional_reactions}\nBehavioral Patterns: {summary.behavioral_patterns}\nSatisfaction Factors: {summary.satisfaction_factors}\nFrustration Points: {summary.frustration_points}\nUnderlying Values: {summary.underlying_values}"
        return f"{base_info}\n{category_specific}"

    def _parse_summary_response(
        self,
        response: str,
        session_id: str,
        chunk_label: str,
        category: str,
        units: Sequence[MemoryUnit],
    ) -> SummaryResult:
        try:
            data = json.loads(response)
            summary_data = data.get("summary", {})
            key_source_uids = data.get("key_source_uids", [])

            if not isinstance(key_source_uids, list):
                key_source_uids = [str(u.uid) for u in units]

            result = SummaryResult(
                category=category,
                text=str(summary_data.get("text", "")) or str(data.get("summary", "")),
                level="chunk",
                session_id=session_id,
                source_uids=[str(uid) for uid in key_source_uids if isinstance(uid, str)],
            )

            if category == "episodic":
                result.timeline = _parse_list_field(summary_data.get("timeline"))
                result.key_people = _parse_list_field(summary_data.get("key_people"))
                result.main_events = _parse_list_field(summary_data.get("main_events"))
                result.location_info = _parse_list_field(summary_data.get("location_info"))
                result.event_relationships = _parse_list_field(summary_data.get("event_relationships"))
            elif category == "procedural":
                result.process_name = _parse_list_field(summary_data.get("process_name"))
                result.key_steps = _parse_list_field(summary_data.get("key_steps"))
                result.decision_points = _parse_list_field(summary_data.get("decision_points"))
                result.preconditions = _parse_list_field(summary_data.get("preconditions"))
                result.expected_outcomes = _parse_list_field(summary_data.get("expected_outcomes"))
                result.optimization_opportunities = _parse_list_field(summary_data.get("optimization_opportunities"))
            elif category == "knowledge":
                result.core_concepts = _parse_list_field(summary_data.get("core_concepts"))
                result.key_facts = _parse_list_field(summary_data.get("key_facts"))
                result.techniques_methods = _parse_list_field(summary_data.get("techniques_methods"))
                result.prerequisites_knowledge = _parse_list_field(summary_data.get("prerequisites_knowledge"))
                result.related_concepts = _parse_list_field(summary_data.get("related_concepts"))
                result.practical_applications = _parse_list_field(summary_data.get("practical_applications"))
            elif category == "emotional":
                result.user_preferences = _parse_list_field(summary_data.get("user_preferences"))
                result.emotional_reactions = _parse_list_field(summary_data.get("emotional_reactions"))
                result.behavioral_patterns = _parse_list_field(summary_data.get("behavioral_patterns"))
                result.satisfaction_factors = _parse_list_field(summary_data.get("satisfaction_factors"))
                result.frustration_points = _parse_list_field(summary_data.get("frustration_points"))
                result.underlying_values = _parse_list_field(summary_data.get("underlying_values"))

            if not result.text:
                result = self._create_fallback_summary(session_id, chunk_label, category, units)

            return result
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse summary response for {category}: {response[:200]}")
            return self._create_fallback_summary(session_id, chunk_label, category, units)

    def _parse_reduce_response(
        self,
        response: str,
        session_id: str,
        category: str,
        sources: Sequence[SummaryResult],
    ) -> SummaryResult:
        try:
            data = json.loads(response)
            summaries_data = data.get("summaries", [])
            for item in summaries_data:
                if not isinstance(item, dict):
                    continue
                if str(item.get("category", "")).lower() == category:
                    summary_data = item.get("summary", item)

                    result = SummaryResult(
                        category=category,
                        text=str(summary_data.get("text", "")),
                        level="session",
                        session_id=session_id,
                        source_uids=[s.session_id + ":" + s.category for s in sources],
                    )

                    if category == "episodic":
                        result.timeline = _parse_list_field(summary_data.get("timeline"))
                        result.key_people = _parse_list_field(summary_data.get("key_people"))
                        result.main_events = _parse_list_field(summary_data.get("main_events"))
                        result.location_info = _parse_list_field(summary_data.get("location_info"))
                        result.event_relationships = _parse_list_field(summary_data.get("event_relationships"))
                    elif category == "procedural":
                        result.process_name = _parse_list_field(summary_data.get("process_name"))
                        result.key_steps = _parse_list_field(summary_data.get("key_steps"))
                        result.decision_points = _parse_list_field(summary_data.get("decision_points"))
                        result.preconditions = _parse_list_field(summary_data.get("preconditions"))
                        result.expected_outcomes = _parse_list_field(summary_data.get("expected_outcomes"))
                        result.optimization_opportunities = _parse_list_field(summary_data.get("optimization_opportunities"))
                    elif category == "knowledge":
                        result.core_concepts = _parse_list_field(summary_data.get("core_concepts"))
                        result.key_facts = _parse_list_field(summary_data.get("key_facts"))
                        result.techniques_methods = _parse_list_field(summary_data.get("techniques_methods"))
                        result.prerequisites_knowledge = _parse_list_field(summary_data.get("prerequisites_knowledge"))
                        result.related_concepts = _parse_list_field(summary_data.get("related_concepts"))
                        result.practical_applications = _parse_list_field(summary_data.get("practical_applications"))
                    elif category == "emotional":
                        result.user_preferences = _parse_list_field(summary_data.get("user_preferences"))
                        result.emotional_reactions = _parse_list_field(summary_data.get("emotional_reactions"))
                        result.behavioral_patterns = _parse_list_field(summary_data.get("behavioral_patterns"))
                        result.satisfaction_factors = _parse_list_field(summary_data.get("satisfaction_factors"))
                        result.frustration_points = _parse_list_field(summary_data.get("frustration_points"))
                        result.underlying_values = _parse_list_field(summary_data.get("underlying_values"))

                    return result
            return sources[0] if sources else SummaryResult(
                category=category, text="", session_id=session_id
            )
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse reduce response for {category}: {response[:200]}")
            return sources[0] if sources else SummaryResult(
                category=category, text="", session_id=session_id
            )

    def _create_fallback_summary(
        self,
        session_id: str,
        chunk_label: str,
        category: str,
        units: Sequence[MemoryUnit],
    ) -> SummaryResult:
        combined_text = " ".join(
            u.raw_data.get(self._text_key, "") for u in units
        )
        return SummaryResult(
            category=category,
            text=combined_text[:500],
            level="chunk",
            session_id=session_id,
            source_uids=[str(u.uid) for u in units],
        )