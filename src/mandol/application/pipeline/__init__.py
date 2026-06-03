"""Unified fact extraction pipeline and cross-session coreference management.

Provides the UnifiedFactPipeline for end-to-end entity/event extraction from
dialogue sessions, and the CrossSessionCorefManager for deduplication across
multiple sessions.
"""

from .unified_fact_pipeline import UnifiedFactPipeline as UnifiedFactPipeline
from .cross_session_coref_manager import CrossSessionCorefManager as CrossSessionCorefManager
