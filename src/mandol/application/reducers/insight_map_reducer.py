"""Insight map reducer for extracting cross-session global insights.

Reduces accumulated memory across all sessions into high-level insights
using chunked LLM-based reduction.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Dict, List

from ...domain.memory_unit import MemoryUnit
from ...domain.types import Uid
from ...ports.llm_provider import ChatMessage, LLMProvider
from .._llm_retry import retry_llm_json_call, strip_json_fences
from ..session_manager import Session

logger = logging.getLogger(__name__)

INSIGHT_MAP_SYSTEM_PROMPT = """You are a deep analysis expert. Generate high-level insights based on the following session summaries through reflection, analysis, and reasoning.
And identify the most core summary UIDs that form this insight.

Summary information:
{summaries_text}

**Step 1: Thinking Process (no need to reflect in final output)**
1. Analyze each summary category: episodic, procedural, knowledge, emotional
2. Identify cross-category patterns and connections
3. Draw causal relationships between events or behaviors
4. Generate actionable insights based on patterns

**Step 2: Final Output**

Please return the insight analysis in the following JSON format:
{{
    "key_source_uids": ["List of the most critical source summary UIDs"],
    "insights": {{
        "pattern_recognition": ["Cross-summary patterns and trends identified"],
        "causal_relationships": ["Discovered causal relationship chains"],
        "predictive_insights": ["Predictions and trend judgments based on patterns"],
        "behavioral_characteristics": ["Deep behavioral characteristics of users or systems"],
        "optimization_recommendations": ["Specific improvement suggestions based on insights"],
        "risk_warnings": ["Potential problems and risk points"]
    }}
}}

The insights should:
1. Go beyond surface information of individual summaries
2. Discover hidden patterns and associations
3. Provide actionable recommendations
4. Base reasoning on evidence
5. Have predictive value
6. Accurately identify the most important source UIDs

Generate a response in JSON format. All textual content within the JSON must be in English."""

INSIGHT_REDUCE_SYSTEM_PROMPT = """You are an insight integration expert. Merge multiple session insights into a global insight to maintain a coherent, non-redundant global understanding.

**Step 1: Thinking Process (no need to reflect in final output)**
1. Compare insights across sessions for common themes
2. Identify which insights represent genuine patterns vs one-off observations
3. Prioritize insights with highest confidence and broadest application
4. Preserve diversity in insight types

**Step 2: Final Output**

Merge principles:
1. Preserve genuinely important insights even if from old sessions
2. Merge overlapping or redundant insights into more comprehensive ones
3. Add genuinely new insights from new sessions
4. Maintain diversity - don't over-merge everything into few points
5. Keep insight text concise but informative

Please return the merged global insight in the following JSON format:
{{
    "key_source_session_ids": ["List of session IDs that contributed to this global insight"],
    "insights": {{
        "pattern_recognition": ["Cross-session patterns and trends identified"],
        "causal_relationships": ["Discovered causal relationship chains"],
        "predictive_insights": ["Predictions and trend judgments based on patterns"],
        "behavioral_characteristics": ["Deep behavioral characteristics"],
        "optimization_recommendations": ["Specific improvement suggestions"],
        "risk_warnings": ["Potential problems and risk points"]
    }}
}}

Generate a response in JSON format. All textual content within the JSON must be in English."""

INSIGHT_TYPES = [
    "pattern_recognition",
    "causal_relationships",
    "predictive_insights",
    "behavioral_characteristics",
    "optimization_recommendations",
    "risk_warnings",
]


@dataclass
class InsightResult:
    """Structured insight produced by the insight map-reduce pipeline.

    Attributes:
        key_source_uids: UIDs of the most important source summaries.
        insights: Dict mapping insight type to a list of insight strings.
            Keys are from INSIGHT_TYPES: pattern_recognition,
            causal_relationships, predictive_insights,
            behavioral_characteristics, optimization_recommendations,
            risk_warnings.
        level: Granularity — 'session' or 'global'.
        session_id: Owning session identifier (empty for global insights).
    """
    key_source_uids: List[str]
    insights: Dict[str, List[str]]
    level: str = "session"
    session_id: str = ""


class InsightMapReducer:
    """Reduces memory into cross-session global insights via LLM.

    Accumulates MemoryUnits (typically from the insight space) into
    token-budgeted chunks and calls the LLM to extract high-level insights.

    Args:
        llm: LLM provider for insight generation.
        embedder: Optional embedding provider for insight units.
        chunk_max_tokens: Max tokens per LLM prompt chunk.
        default_text_key: Key for extracting text from raw_data.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        text_key: str = "text_content",
    ):
        self._llm = llm_provider
        self._text_key = str(text_key)

    def process_session(
        self,
        session: Session,
        summary_units: Dict[str, List[MemoryUnit]],
    ) -> List[MemoryUnit]:
        """Generate session-level insights from summary units.

        Accumulates all summary texts, calls the LLM to extract high-level
        insights, and wraps the result as a single MemoryUnit.

        Args:
            session: The session to analyse.
            summary_units: Dict mapping category name to summary MemoryUnits.

        Returns:
            List containing one insight MemoryUnit for the session.
        """
        summary_texts = []
        source_summary_uids = []

        for category, units in summary_units.items():
            for unit in units:
                text = unit.raw_data.get(self._text_key, "")
                if text:
                    uid = str(unit.uid)
                    summary_texts.append(f"[{category.upper()}] (UID: {uid})\n{text}")
                    source_summary_uids.append(uid)

        if not summary_texts:
            return []

        content = "\n\n".join(summary_texts)
        insight = self._map_phase(content, session.session_id, source_summary_uids)

        if not insight.key_source_uids:
            insight.key_source_uids = source_summary_uids[:5]

        uid = f"{session.session_id}:insight:session"
        raw_data = {
            self._text_key: json.dumps(insight.insights, ensure_ascii=False),
            "insights": insight.insights,
        }
        metadata = {
            "type": "insight",
            "category": "insights",
            "level": insight.level,
            "session_id": session.session_id,
            "key_source_uids": insight.key_source_uids,
        }
        unit = MemoryUnit(
            uid=Uid(uid),
            raw_data=raw_data,
            metadata=metadata,
        )

        return [unit]

    def _map_phase(
        self,
        content: str,
        session_id: str,
        source_summary_uids: List[str],
    ) -> InsightResult:
        summaries_text = f"Summaries to analyze:\n{content[:8000]}"
        prompt = INSIGHT_MAP_SYSTEM_PROMPT.format(summaries_text=summaries_text)

        messages: List[ChatMessage] = [
            {"role": "user", "content": prompt},
        ]

        try:
            insight = retry_llm_json_call(
                self._llm,
                messages,
                lambda resp: self._parse_insight_response(resp, session_id, source_summary_uids),
                temperature=0.1,
                max_tokens=2048,
                context_label="insight_map",
            )
        except json.JSONDecodeError:
            insight = self._create_fallback_insight(session_id, source_summary_uids)
        except Exception as e:
            logger.error(f"Insight map phase failed: {e}")
            insight = self._create_fallback_insight(session_id, source_summary_uids)

        return insight

    def _parse_insight_response(
        self,
        response: str,
        session_id: str,
        source_summary_uids: List[str],
    ) -> InsightResult:
        data = json.loads(strip_json_fences(response))

        key_source_uids = data.get("key_source_uids", [])
        if not isinstance(key_source_uids, list):
            key_source_uids = source_summary_uids[:5]

        insights_raw = data.get("insights", {})
        insights: Dict[str, List[str]] = {}

        for insight_type in INSIGHT_TYPES:
            items = insights_raw.get(insight_type, [])
            if isinstance(items, list):
                insights[insight_type] = [str(item) for item in items if item]
            else:
                insights[insight_type] = []

        for insight_type in INSIGHT_TYPES:
            if not insights.get(insight_type):
                insights[insight_type] = []

        return InsightResult(
            key_source_uids=[str(uid) for uid in key_source_uids if isinstance(uid, str)],
            insights=insights,
            level="session",
            session_id=session_id,
        )

    def _create_fallback_insight(
        self,
        session_id: str,
        source_summary_uids: List[str],
    ) -> InsightResult:
        return InsightResult(
            key_source_uids=source_summary_uids[:5] if source_summary_uids else [],
            insights={insight_type: [] for insight_type in INSIGHT_TYPES},
            level="session",
            session_id=session_id,
        )

    def reduce_insights(
        self,
        session_insights: List[MemoryUnit],
        all_session_ids: List[str],
    ) -> InsightResult:
        """Merge multiple session-level insights into a global insight.

        When only one session insight exists, its data is returned directly
        with level promoted to 'global'. Otherwise, the LLM is called to
        intelligently merge all session insights.

        Args:
            session_insights: Insight MemoryUnits from all sessions.
            all_session_ids: All session IDs contributing to the global insight.

        Returns:
            A global-level InsightResult with merged insight data.
        """
        if not session_insights:
            return InsightResult(
                key_source_uids=[],
                insights={insight_type: [] for insight_type in INSIGHT_TYPES},
                level="global",
            )

        if len(session_insights) == 1:
            unit = session_insights[0]
            insights = unit.raw_data.get("insights", {})
            if isinstance(insights, str):
                try:
                    insights = json.loads(insights)
                except json.JSONDecodeError:
                    insights = {}
            return InsightResult(
                key_source_uids=unit.metadata.get("key_source_uids", []),
                insights=insights if isinstance(insights, dict) else {},
                level="global",
            )

        existing_insights_texts = []
        for i, unit in enumerate(session_insights):
            insights = unit.raw_data.get("insights", {})
            if isinstance(insights, str):
                try:
                    insights = json.loads(insights)
                except json.JSONDecodeError:
                    insights = {}
            session_id_insight = unit.metadata.get("session_id", f"session_{i}")
            existing_insights_texts.append(
                f"Session {session_id_insight} insights:\n{json.dumps(insights, ensure_ascii=False, indent=2)}"
            )

        prompt = f"""Existing Session Insights:
{chr(10).join(existing_insights_texts)}

All session IDs: {all_session_ids}"""

        combined = INSIGHT_REDUCE_SYSTEM_PROMPT + "\n\n" + prompt
        messages: List[ChatMessage] = [
            {"role": "user", "content": combined},
        ]

        try:
            result = retry_llm_json_call(
                self._llm,
                messages,
                lambda resp: self._parse_reduce_response(resp, all_session_ids),
                temperature=0.1,
                max_tokens=2048,
                context_label="insight_reduce",
            )
            return result
        except json.JSONDecodeError:
            return self._merge_insights_fallback(session_insights, all_session_ids)
        except Exception as e:
            logger.error(f"Insight reduce failed: {e}")
            return self._merge_insights_fallback(session_insights, all_session_ids)

    def _parse_reduce_response(
        self,
        response: str,
        session_ids: List[str],
    ) -> InsightResult:
        data = json.loads(strip_json_fences(response))

        key_source_sessions = data.get("key_source_session_ids", session_ids)
        if not isinstance(key_source_sessions, list):
            key_source_sessions = session_ids

        insights_raw = data.get("insights", {})
        insights: Dict[str, List[str]] = {}

        for insight_type in INSIGHT_TYPES:
            items = insights_raw.get(insight_type, [])
            if isinstance(items, list):
                insights[insight_type] = [str(item) for item in items if item]
            else:
                insights[insight_type] = []

        return InsightResult(
            key_source_uids=list(key_source_sessions),
            insights=insights,
            level="global",
        )

    def _merge_insights_fallback(
        self,
        session_insights: List[MemoryUnit],
        session_ids: List[str],
    ) -> InsightResult:
        merged: Dict[str, List[str]] = {t: [] for t in INSIGHT_TYPES}

        for unit in session_insights:
            insights = unit.raw_data.get("insights", {})
            if isinstance(insights, str):
                try:
                    insights = json.loads(insights)
                except json.JSONDecodeError:
                    continue
            if not isinstance(insights, dict):
                continue
            for insight_type in INSIGHT_TYPES:
                items = insights.get(insight_type, [])
                if isinstance(items, list):
                    for item in items:
                        if item and item not in merged[insight_type]:
                            merged[insight_type].append(str(item))

        return InsightResult(
            key_source_uids=session_ids,
            insights=merged,
            level="global",
        )