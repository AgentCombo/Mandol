"""Global insight manager for accumulating and updating cross-session insights.

Merges new insights from each session into a persistent global insight store.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .._llm_retry import strip_json_fences
from ...domain.memory_unit import MemoryUnit
from ...domain.types import Uid
from ...ports.llm_provider import ChatMessage, LLMProvider

if TYPE_CHECKING:
    from ..session_manager import Session

logger = logging.getLogger(__name__)

INSIGHT_TYPES = [
    "pattern_recognition",
    "causal_relationships",
    "predictive_insights",
    "behavioral_characteristics",
    "optimization_recommendations",
    "risk_warnings",
]

GLOBAL_INSIGHT_MERGE_PROMPT = """You are an insight integration expert. Merge new session insights into existing global insights to maintain a coherent, non-redundant global understanding.

**Step 1: Thinking Process (no need to reflect in final output)**
1. Compare existing global insights with new session insights
2. Identify which insights are truly new vs overlapping
3. Merge similar insights while preserving diversity
4. Prioritize high-value insights across all types

**Step 2: Final Output**

Merge rules:
1. Preserve genuinely important global insights even if old
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

Generate ONLY valid JSON, without markdown code fences. All textual content within the JSON must be in English."""


@dataclass
class GlobalInsightResult:
    """Global insight accumulated across all sessions.

    Attributes:
        key_source_session_ids: Session IDs that contributed to this insight.
        insights: Dict mapping insight type to a list of insight strings.
        level: Always 'global' for this result type.
    """
    key_source_session_ids: List[str]
    insights: Dict[str, List[str]]
    level: str = "global"


class GlobalInsightManager:
    """Accumulates and updates global insights across all sessions.

    Each session's insights are appended to a persistent list, with
    periodic deduplication via InsightMapReducer. Supports retrieving
    formatted insight texts for use in LLM prompts.

    Args:
        llm_provider: LLM provider for insight reduction.
        text_key: Key in raw_data for extracting text content.
    """

    GLOBAL_INSIGHT_UID = "global_insight_v1"

    def __init__(
        self,
        llm_provider: LLMProvider,
        text_key: str = "text_content",
    ):
        """Initialize the global insight manager.

        Args:
            llm_provider: LLM provider for insight reduction calls.
            text_key: Key in raw_data for extracting text content.
        """
        self._llm = llm_provider
        self._text_key = str(text_key)

        self._lock = threading.Lock()
        self._global_insight: Optional[GlobalInsightResult] = None
        self._is_initialized = False
        self._version = 0

    def merge_and_update(
        self,
        session: "Session",
        session_insights: List[MemoryUnit],
        semantic_map: Any,
        graph: Any,
        naming: Any,
        root_space: str = "default",
    ) -> None:
        """Thread-safe merge of session insights into the global insight.

        Args:
            session: The current session object.
            session_insights: Sub-insights from the current session (not persisted).
            semantic_map: SemanticMapService instance.
            graph: SemanticGraphService instance.
            naming: SpaceNamingPolicy instance.
            root_space: Root space name.
        """
        if not session_insights:
            logger.debug(f"Session {session.session_id} has no insights to merge")
            return

        session_insight = self._extract_session_insight(session_insights)
        if not session_insight or not session_insight.insights:
            logger.debug(f"Session {session.session_id} has empty insights")
            return

        with self._lock:
            if not self._is_initialized:
                self._global_insight = GlobalInsightResult(
                    key_source_session_ids=[session.session_id],
                    insights=session_insight.insights,
                    level="global",
                )
                self._is_initialized = True
            else:
                self._incremental_merge(session.session_id, session_insight)

            self._version += 1
            self._persist_global_insight(semantic_map, graph, naming, root_space)

    def _extract_session_insight(self, session_insights: List[MemoryUnit]) -> Optional[GlobalInsightResult]:
        if not session_insights:
            return None
        unit = session_insights[0]
        insights = unit.raw_data.get("insights", {})
        if isinstance(insights, str):
            try:
                insights = json.loads(insights)
            except json.JSONDecodeError:
                insights = {}
        if not isinstance(insights, dict):
            return None
        key_source_uids = unit.metadata.get("key_source_uids", [])
        return GlobalInsightResult(
            key_source_session_ids=key_source_uids,
            insights=insights,
            level="session",
        )

    def _incremental_merge(
        self,
        session_id: str,
        session_insight: GlobalInsightResult,
    ) -> None:
        """Incrementally merge session insights into global insights (lock already held)."""
        if not self._global_insight:
            self._global_insight = GlobalInsightResult(
                key_source_session_ids=[session_id],
                insights=session_insight.insights,
                level="global",
            )
            return

        prompt = f"""Existing Global Insights:
{json.dumps(self._global_insight.insights, ensure_ascii=False, indent=2)}

New Session Insights (from session {session_id}):
{json.dumps(session_insight.insights, ensure_ascii=False, indent=2)}

All contributing session IDs: {self._global_insight.key_source_session_ids + [session_id]}"""

        combined = GLOBAL_INSIGHT_MERGE_PROMPT + "\n\n" + prompt
        messages: List[ChatMessage] = [
            {"role": "user", "content": combined},
        ]

        try:
            response = self._llm.chat(messages, temperature=0.1, max_tokens=2048)
            self._global_insight = self._parse_merge_response(
                response.content, session_id
            )
        except (json.JSONDecodeError, ConnectionError, TimeoutError, OSError, RuntimeError) as e:
            logger.error("Global insight merge failed: %s, using fallback merge", e)
            self._merge_insights_fallback(session_id, session_insight)

    def _parse_merge_response(
        self,
        response: str,
        session_id: str,
    ) -> GlobalInsightResult:
        """Parse the LLM merge response and update the global insight."""
        data = json.loads(strip_json_fences(response))

        key_source_sessions = data.get("key_source_session_ids", [])
        if not isinstance(key_source_sessions, list):
            key_source_sessions = []

        insights_raw = data.get("insights", {})
        insights: Dict[str, List[str]] = {}

        for insight_type in INSIGHT_TYPES:
            items = insights_raw.get(insight_type, [])
            if isinstance(items, list):
                insights[insight_type] = [str(item) for item in items if item]
            else:
                insights[insight_type] = []

        return GlobalInsightResult(
            key_source_session_ids=key_source_sessions,
            insights=insights,
            level="global",
        )

    def _merge_insights_fallback(
        self,
        session_id: str,
        session_insight: GlobalInsightResult,
    ) -> None:
        """Fallback merge when LLM fails: simple concatenation of insights."""
        if not self._global_insight:
            return

        merged: Dict[str, List[str]] = dict(self._global_insight.insights)

        for insight_type in INSIGHT_TYPES:
            existing = set(merged.get(insight_type, []))
            new_items = session_insight.insights.get(insight_type, [])
            merged[insight_type] = list(existing.union(set(new_items)))

        self._global_insight = GlobalInsightResult(
            key_source_session_ids=self._global_insight.key_source_session_ids + [session_id],
            insights=merged,
            level="global",
        )

    def _persist_global_insight(
        self,
        semantic_map: Any,
        graph: Any,
        naming: Any,
        root_space: str,
    ) -> None:
        """Persist the global insight into the semantic map."""
        if not self._global_insight:
            return

        insights_text = json.dumps(self._global_insight.insights, ensure_ascii=False, indent=2)

        embedding_text_parts: List[str] = []
        for itype in INSIGHT_TYPES:
            items = self._global_insight.insights.get(itype, [])
            if items:
                embedding_text_parts.append(f"{itype}: " + "; ".join(str(i) for i in items))
        embedding_text = "\n".join(embedding_text_parts) if embedding_text_parts else insights_text

        insight_unit = MemoryUnit(
            uid=Uid(self.GLOBAL_INSIGHT_UID),
            raw_data={
                self._text_key: insights_text,
                "embedding_text": embedding_text,
                "insights": self._global_insight.insights,
            },
            metadata={
                "type": "global_insight",
                "category": "insights",
                "level": "global",
                "key_source_session_ids": self._global_insight.key_source_session_ids,
                "version": self._version,
            },
        )

        insight_space = naming.insights(root_space)

        semantic_map.add_unit(
            insight_unit,
            space_names=[insight_space],
            ensure_embedding=True,
            embedding_text=embedding_text,
        )

        logger.info(f"Persisted global insight v{self._version} with sessions: {self._global_insight.key_source_session_ids}")

    @property
    def global_insight(self) -> Optional[GlobalInsightResult]:
        """Return the current global insight (for debugging)."""
        with self._lock:
            return self._global_insight

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    @property
    def version(self) -> int:
        return self._version