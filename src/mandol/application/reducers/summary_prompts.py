"""LLM prompt templates for the SummaryMapReducer.

Defines prompt templates for four summary types: episodic, knowledge,
emotional, and procedural. Each prompt includes formatting placeholders
for context text and optional existing summary content.
"""

from __future__ import annotations

from typing import Any, Dict, List


# Prompt for generating episodic summaries (events/timeline).
EPISODIC_SUMMARY_PROMPT = """You are a professional memory organization expert. Please summarize the following records into an episodic memory summary with unique IDs (UIDs), extracting key time, location, character, and event information.
And identify the most core record UIDs that form this summary (preferably 3-5, can be appropriately increased or decreased if really necessary).

Original records:
{records}

Please return the episodic memory summary in the following JSON format:
{{
    "reasoning": "Brief analysis of the records and how this summary was formed",
    "key_source_uids": ["List of the most critical source record UIDs"],
    "summary": {{
        "timeline": ["Key time points organized chronologically"],
        "key_people": ["Important people involved and their roles"],
        "main_events": ["Concise descriptions of core events"],
        "location_info": ["Important locations involved"],
        "event_relationships": ["Causal relationships or associations between events"]
    }}
}}

The summary should:
1. Maintain factual accuracy without adding speculation
2. Highlight continuity in time and space
3. Reflect interactions between people
4. Be concise and highlight key points
5. Accurately identify the most important source UIDs

Generate a response in JSON format. All textual content within the JSON must be in English."""


# Prompt for generating procedural summaries (steps/processes).
PROCEDURAL_SUMMARY_PROMPT = """You are a process optimization expert. Please summarize the following records into a procedural memory summary with unique IDs (UIDs), extracting the optimal execution path, key steps, and decision points.
And identify the most core record UIDs that form this summary (preferably 3-5, can be appropriately increased or decreased if really necessary).

Original records:
{records}

Please return the procedural memory summary in the following JSON format:
{{
    "reasoning": "Brief analysis of the records and how this summary was formed",
    "key_source_uids": ["List of the most critical source record UIDs"],
    "summary": {{
        "process_name": ["Name or title of the process/flow"],
        "key_steps": ["Ordered list of critical steps in the process"],
        "decision_points": ["Key decision nodes or branching conditions"],
        "preconditions": ["Prerequisites or conditions required before starting"],
        "expected_outcomes": ["Anticipated results or outputs of the process"],
        "optimization_opportunities": ["Potential improvements or efficiency gains"]
    }}
}}

The summary should:
1. Focus on actionable, repeatable processes
2. Highlight critical success factors
3. Identify potential failure points or risks
4. Be clear and unambiguous in step sequencing
5. Accurately identify the most important source UIDs

Generate a response in JSON format. All textual content within the JSON must be in English."""


# Prompt for generating knowledge summaries (facts/concepts).
KNOWLEDGE_SUMMARY_PROMPT = """You are a knowledge organization expert. Please summarize the following records into a knowledge memory summary with unique IDs (UIDs), extracting concepts, principles, techniques, and factual information.
And identify the most core record UIDs that form this summary (preferably 3-5, can be appropriately increased or decreased if really necessary).

Original records:
{records}

Please return the knowledge memory summary in the following JSON format:
{{
    "reasoning": "Brief analysis of the records and how this summary was formed",
    "key_source_uids": ["List of the most critical source record UIDs"],
    "summary": {{
        "core_concepts": ["Fundamental concepts or principles explained"],
        "key_facts": ["Important facts or data points captured"],
        "techniques_methods": ["Specific techniques, methods, or approaches described"],
        "prerequisites_knowledge": ["Background knowledge required to understand"],
        "related_concepts": ["Connected or related concepts for further learning"],
        "practical_applications": ["Real-world applications or use cases"]
    }}
}}

The summary should:
1. Be factual and accurate without speculation
2. Capture the essence of complex information concisely
3. Highlight cause-and-effect relationships
4. Be accessible to someone with basic background knowledge
5. Accurately identify the most important source UIDs

Generate a response in JSON format. All textual content within the JSON must be in English."""


# Prompt for generating emotional summaries.
EMOTIONAL_SUMMARY_PROMPT = """You are a user experience analysis expert. Please summarize the following records into an emotional memory summary with unique IDs (UIDs), extracting user preferences, attitudes, emotional reactions, and behavioral patterns.
And identify the most core record UIDs that form this summary (preferably 3-5, can be appropriately increased or decreased if really necessary).

Original records:
{records}

Please return the emotional memory summary in the following JSON format:
{{
    "reasoning": "Brief analysis of the records and how this summary was formed",
    "key_source_uids": ["List of the most critical source record UIDs"],
    "summary": {{
        "user_preferences": ["Stated or implied user preferences and likes"],
        "emotional_reactions": ["Observed emotional responses and attitudes"],
        "behavioral_patterns": ["Recurring behavioral patterns or habits"],
        "satisfaction_factors": ["Factors contributing to satisfaction or dissatisfaction"],
        "frustration_points": ["Pain points or sources of frustration expressed"],
        "underlying_values": ["Inferred underlying values or priorities"]
    }}
}}

The summary should:
1. Be empathetic and non-judgmental
2. Distinguish between stated preferences and inferred patterns
3. Highlight both positive and negative emotional signals
4. Focus on actionable emotional insights
5. Accurately identify the most important source UIDs

Generate a response in JSON format. All textual content within the JSON must be in English."""


SUMMARY_TYPE_DEFINITIONS = {
    "episodic": "Episodic memory records specific time, location, characters, and events - concrete experiences and occurrences.",
    "procedural": "Procedural memory records processes, workflows, methods, and step-by-step procedures - how to do something.",
    "knowledge": "Knowledge memory records concepts, facts, principles, and technical information - abstract understanding.",
    "emotional": "Emotional memory records preferences, attitudes, feelings, and behavioral patterns - affective and motivational aspects.",
}


SUMMARY_PROMPTS: Dict[str, str] = {
    "episodic": EPISODIC_SUMMARY_PROMPT,
    "procedural": PROCEDURAL_SUMMARY_PROMPT,
    "knowledge": KNOWLEDGE_SUMMARY_PROMPT,
    "emotional": EMOTIONAL_SUMMARY_PROMPT,
}


def get_summary_prompt(category: str, records: List[Dict[str, Any]]) -> str:
    prompt_template = SUMMARY_PROMPTS.get(category, EPISODIC_SUMMARY_PROMPT)
    records_text = "\n\n".join(
        f"Record {i}(UID: {r.get('uid', 'unknown_uid')}):\n"
        f"Time: {r.get('metadata', {}).get('timestamp', 'Unknown time')}\n"
        f"Speaker: {r.get('metadata', {}).get('speaker', 'Unknown')}\n"
        f"Content: {r.get('text', r.get('content', ''))}"
        for i, r in enumerate(records, 1)
    )
    return prompt_template.format(records=records_text)


def get_summary_prompt_with_context(
    category: str, records: List[Dict[str, Any]], context: Dict[str, Any] = None
) -> str:
    prompt = get_summary_prompt(category, records)
    if context:
        context_str = f"\n\nAdditional context:\n{context.get('summary', '')}"
        prompt = prompt.replace("{context}", context_str)
    return prompt