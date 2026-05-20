"""Prompt templates for the unified fact pipeline.

Centralizes all LLM prompt strings used by UnifiedFactPipeline and
CrossSessionCorefManager, covering entity/event extraction, matching,
relation/causal inference, cluster judging, and description merging.
"""

from .cross_session_merge import (
    CLUSTER_JUDGE_PROMPT,
    CROSS_SESSION_EVENT_MERGE_PROMPT,
    CROSS_SESSION_MERGE_PROMPT,
    EVENT_CLUSTER_JUDGE_PROMPT,
)
from .description_merge import DESCRIPTION_MERGE_PROMPT, EVENT_DESCRIPTION_MERGE_PROMPT
from .entity_extraction import ENTITY_EXTRACTION_PROMPT
from .entity_match_judge import ENTITY_MATCH_JUDGE_PROMPT
from .entity_relation import ENTITY_RELATION_PROMPT
from .event_causal import EVENT_CAUSAL_PROMPT
from .event_extraction import EVENT_EXTRACTION_PROMPT
from .event_match_judge import EVENT_MATCH_JUDGE_PROMPT

__all__ = [
    "CLUSTER_JUDGE_PROMPT",
    "CROSS_SESSION_EVENT_MERGE_PROMPT",
    "CROSS_SESSION_MERGE_PROMPT",
    "DESCRIPTION_MERGE_PROMPT",
    "ENTITY_EXTRACTION_PROMPT",
    "ENTITY_MATCH_JUDGE_PROMPT",
    "ENTITY_RELATION_PROMPT",
    "EVENT_CAUSAL_PROMPT",
    "EVENT_CLUSTER_JUDGE_PROMPT",
    "EVENT_DESCRIPTION_MERGE_PROMPT",
    "EVENT_EXTRACTION_PROMPT",
    "EVENT_MATCH_JUDGE_PROMPT",
]
