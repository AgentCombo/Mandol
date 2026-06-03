"""Memory reduction components for summarization and insight extraction.

Uses LLMs to condense conversational memory into structured summaries
(episodic, knowledge, emotional, procedural) and extracts global insights
across sessions.
"""

from .summary_map_reducer import SummaryMapReducer as SummaryMapReducer
from .insight_map_reducer import InsightMapReducer as InsightMapReducer
from .global_insight_manager import GlobalInsightManager as GlobalInsightManager
