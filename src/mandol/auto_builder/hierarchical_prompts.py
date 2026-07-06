# mandol/auto_builder/hierarchical_prompts.py
"""Utilities for hierarchical prompts."""
from typing import Dict, List, Optional, Any
from string import Template
from dataclasses import dataclass
from enum import Enum


class ExtractionStyle(str, Enum):
    DEFAULT = "default"
    LOCOMO = "locomo"
    LONGMEMEVAL = "longmemeval"
    
    @classmethod
    def values(cls) -> List[str]:
        return [e.value for e in cls]


class L1SummaryType(str, Enum):
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"
    KNOWLEDGE = "knowledge"
    EMOTIONAL = "emotional"
    STRUCTURED = "structured"
    
    @classmethod
    def values(cls) -> List[str]:
        return [e.value for e in cls]
    
    @classmethod
    def default_types(cls) -> List[str]:
        """Run default types."""
        return [cls.EPISODIC.value, cls.KNOWLEDGE.value]
    
    @classmethod
    def locomo_types(cls) -> List[str]:
        """Run locomo types."""
        return [cls.STRUCTURED.value]
    
    @classmethod
    def longmemeval_types(cls) -> List[str]:
        """Run longmemeval types."""
        return [cls.EPISODIC.value, cls.KNOWLEDGE.value]


class HierarchicalPromptManager:
    
    
    
    
    CONTEXTUAL_RETRIEVAL_PROMPT = """<system_instructions>
You are a Data Enrichment Expert for a RAG system. 
Your task is to rewrite a specific message from a dialogue into a **standalone, self-contained indexing string**.
The goal is to allow a vector search engine to find this message without needing the surrounding context window.

### MANDATORY TRANSFORMATION RULES (Apply Strictly):

1. **Global Context Resolution (God Mode)**:
   - You have access to the **FULL transcript** (Past and Future context).
   - Resolve ALL pronouns (it, he, she, that, they) to specific entity names.
   - If a message relies on context from 10 turns later to be understood, incorporate that future context NOW.
   - If the message is a short reaction (e.g., "I agree", "No way"), rewrite it to include exactly *what* is being agreed to or denied.

2. **Absolute Time Enforcement**:
   - The **Session Date** is provided in the context below.
   - **NEVER** use relative terms like "today", "tomorrow", "next Friday", "last week".
   - **CALCULATE** and output the specific absolute date (YYYY-MM-DD) for any event mentioned.
   - *Example*: If Session Date is 2023-01-01 and text says "I'm leaving next Friday", write "leaving on 2023-01-06".

3. **Event & State Classification**:
   - **Occurrence**: If an event is happening *now* or is a specific plan, phrase it as a factual occurrence.
   - **Reference/Recall**: If they are discussing a *past* event, phrase it as "Speaker recalls/discusses [Event] from [Date]".
   - **State Change**: If a key status changes (e.g., Job, Location, Relationship), explicitly state: "Speaker's [Attribute] changed from [Old] to [New]".

4. **Noise Filtering**:
   - If the message is pure chitchat (e.g., "Haha", "Okay") with no factual content, output a minimal string or just the intent. Focus on **Entities, Actions, Dates, and Facts**.

</system_instructions>

<document_context>
Session Reference Date: ${session_date}
Participants: ${participants}

=== BEGIN FULL TRANSCRIPT (Static Context) ===
${full_session_transcript}
=== END FULL TRANSCRIPT ===
</document_context>

<target_message_locator>
Target Speaker: ${speaker}
Target Message Content: "${message_text}"
</target_message_locator>

<output_requirement>
Output ONLY the contextualized string. Do not output explanations.
Format: [Date: YYYY-MM-DD] [Speaker Name] [Event/Fact/Action] [Details with Resolved Entities]
</output_requirement>"""

    
    
    
    
    L1_EPISODIC_PROMPT = """You are a professional memory summarization assistant. Please analyze the following conversation content and extract key **Episodic Memory**.

Conversation Content:
${content}

Please generate a concise episodic summary containing:
1. Key events and actions
2. Time points
3. Important people and objects

Summary (within 150 words):"""

    L1_PROCEDURAL_PROMPT = """You are a professional memory summarization assistant. Please analyze the following conversation content and extract **Procedural Memory**.

Conversation Content:
${content}

Please generate a procedural summary containing:
1. Operation steps and procedures
2. Methods and techniques
3. How-to guidance

Summary (within 150 words):"""

    L1_KNOWLEDGE_PROMPT = """You are a professional memory summarization assistant. Please analyze the following conversation content and extract **Knowledge Memory**.

Conversation Content:
${content}

Please generate a knowledge summary containing:
1. Core concepts and definitions
2. Factual information
3. Domain knowledge points

Summary (within 150 words):"""

    L1_EMOTIONAL_PROMPT = """You are a professional memory summarization assistant. Please analyze the following conversation content and extract **Emotional Memory**.

Conversation Content:
${content}

Please generate an emotional summary containing:
1. Emotional tone (positive/negative/neutral)
2. Key emotional expressions
3. Attitudes and feelings

Summary (within 150 words):"""

    
    L1_STRUCTURED_EXTRACTION_PROMPT = """You are a Data Extraction Specialist. Analyze this conversation session.
Your goal is to EXTRACT structured data points (Events, States, Counts) for a Knowledge Graph.

<input_context>
Session ID: ${session_id}
Session Date: ${session_date} (Reference this for absolute date calculation)
Participants: ${participants}
Transcript:
${transcript}
</input_context>

<extraction_rules>
1. **Absolute Time Resolution**: Convert relative times (e.g., "next Friday", "last week", "yesterday") to YYYY-MM-DD format using Session Date as reference.
2. **State Change Tracking**: Identify status changes (Job, Location, Relationship, Health, Mood). Format: Old -> New.
3. **Event Classification**:
   - "Occurrence": Happening NOW or Planned for future.
   - "Reference": Discussing past events.
4. **Countable Actions**: Extract items that can be counted (games played, trips taken, meals, recommendations made).
5. **Quote Extraction**: Include exact quotes that support state changes or key facts.
</extraction_rules>

<output_format>
Return ONLY valid JSON (no markdown, no explanation):
{
    "session_id": "${session_id}",
    "session_date": "${session_date}",
    "session_topic": "Brief 5-10 word topic description",
    "structured_events": [
        {
            "event_name": "Descriptive name of the event",
            "event_type": "Activity|Crisis|Milestone|Plan|Social|Health|Work",
            "date": "YYYY-MM-DD or 'unknown'",
            "date_source": "explicit|calculated|unknown",
            "is_new_occurrence": true,
            "participants": ["Name1", "Name2"],
            "location": "Location if mentioned or null",
            "supporting_quote": "Exact quote from transcript"
        }
    ],
    "state_updates": [
        {
            "entity": "Person name",
            "attribute": "Job|Location|Relationship|Health|Mood|Hobby|Goal",
            "old_value": "Previous state or 'unknown'",
            "new_value": "Current/new state",
            "change_date": "YYYY-MM-DD or 'during_session'",
            "trigger_quote": "Exact quote that reveals this change"
        }
    ],
    "countable_items": [
        {
            "category": "Game|Place|Food|Activity|Recommendation|Purchase",
            "item_name": "Specific name of the item",
            "action": "Played|Visited|Ate|Did|Recommended|Bought",
            "count": 1,
            "by_whom": "Person who performed action"
        }
    ],
    "mentioned_dates": [
        {
            "original_text": "The relative/absolute date as mentioned",
            "resolved_date": "YYYY-MM-DD",
            "context": "What this date refers to"
        }
    ],
    "key_facts": [
        {
            "fact_type": "Identity|Preference|History|Plan|Opinion",
            "subject": "Who this fact is about",
            "fact": "The factual statement",
            "supporting_quote": "Exact quote"
        }
    ]
}
</output_format>

IMPORTANT: 
- Return ONLY the JSON object, no additional text.
- If no items exist for a category, use empty array [].
- All text content must be in English."""

    
    L1_LONGMEMEVAL_PROMPT = """You are a conversational memory expert. Analyze the following conversation chunk and extract key information that would be useful for answering questions about this conversation later.

<input_context>
Context ID: ${context_id}
Session Info: ${session_info}
Conversation Chunk:
${content}
</input_context>

<extraction_focus>
1. **Speaker Information**: Who is speaking and what are their characteristics
2. **Topics Discussed**: Main subjects and themes
3. **Facts Mentioned**: Specific facts, numbers, dates, names
4. **Opinions & Preferences**: Views expressed by speakers
5. **Events & Actions**: Things that happened or are planned
6. **Relationships**: How speakers relate to each other and to mentioned entities
</extraction_focus>

<output_format>
Return a JSON object:
{
    "context_id": "${context_id}",
    "main_topics": ["topic1", "topic2"],
    "speakers": [
        {
            "name": "Speaker name",
            "role": "Role if mentioned",
            "characteristics": ["trait1", "trait2"]
        }
    ],
    "key_facts": [
        {
            "category": "personal|work|event|general",
            "fact": "The factual statement",
            "speaker": "Who stated this"
        }
    ],
    "events": [
        {
            "event": "Event description",
            "when": "Time reference",
            "participants": ["person1"]
        }
    ],
    "relationships": [
        {
            "entity1": "Person/Thing A",
            "entity2": "Person/Thing B",
            "relationship": "How they relate"
        }
    ],
    "summary": "A 2-3 sentence summary of this chunk"
}
</output_format>

Return ONLY the JSON object."""

    
    # Dataset-specific handling used by the reproduction workflow.
    
    
    
    L2_INSIGHT_PROMPT = """You are a professional deep analysis assistant. Based on the following L1 summaries, generate a **Deep Insight**.

L1 Summary Content:
${summaries}

Please generate a deep insight containing:
1. Cross-session patterns and trends
2. Deep-level connections and discoveries
3. Actionable suggestions or predictions

Insight (within 200 words):"""

    
    L2_GLOBAL_AGGREGATION_PROMPT = """You are a Data Aggregation Specialist. Given the session-level facts extracted from multiple sessions, create a GLOBAL summary.

<input_context>
Sample ID: ${sample_id}
Total Sessions: ${total_sessions}
Participants: ${participants}
Time Range: ${time_range}

Session Extractions (L1 Data):
${session_data}
</input_context>

<aggregation_rules>
1. **Global Statistics**: Compute totals across ALL sessions (total games played, total trips taken, etc.)
2. **Character Status Snapshot**: For each person, provide their LATEST status as of the final session.
3. **Master Timeline**: Combine all events into a single chronological timeline.
4. **Cross-Session Patterns**: Identify recurring topics, relationships, activities.
5. **Count Deduplication**: Do NOT double-count the same event mentioned in multiple sessions.
6. **Temporal Logic**: For any recurring events (e.g., Doctor Visits, Meetings), you MUST CALCULATE the exact time gap between them (in days or months) and include it in the output.
</aggregation_rules>

<output_format>
Return ONLY valid JSON (no markdown, no explanation):
{
    "sample_id": "${sample_id}",
    "aggregation_time": "${aggregation_time}",
    "time_range": {
        "first_session": "${first_session_date}",
        "last_session": "${last_session_date}",
        "total_sessions": ${total_sessions}
    },
    "global_statistics": {
        "total_conversations": ${total_sessions},
        "total_unique_events": 0,
        "total_state_changes": 0,
        "activity_counts": {
            "games_played": {"total": 0, "items": []},
            "places_visited": {"total": 0, "items": []},
            "foods_mentioned": {"total": 0, "items": []},
            "recommendations_made": {"total": 0, "items": []}
        },
        "topic_frequency": []
    },
    "character_status_snapshot": [
        {
            "person": "Person Name",
            "status_at_end": {
                "job": "Current job or 'unknown'",
                "location": "Current city/location or 'unknown'",
                "relationship_status": "Status or 'unknown'",
                "current_mood": "Mood or 'unknown'",
                "active_hobbies": [],
                "ongoing_goals": [],
                "health_status": "Status or 'unknown'"
            },
            "key_changes_during_timeline": []
        }
    ],
    "master_timeline": [
        {
            "date": "YYYY-MM-DD",
            "events": [
                {
                    "event": "Description of event",
                    "participants": [],
                    "source_session": "session_1"
                }
            ]
        }
    ],
    "relationship_graph": {
        "edges": [
            {
                "from": "Person A",
                "to": "Person B",
                "relationship": "friends|colleagues|family",
                "interaction_count": 10
            }
        ]
    },
    "recurring_topics": [
        {
            "topic": "Topic name",
            "occurrences": 5,
            "sessions": ["session_1", "session_3", "session_5"]
        }
    ],
    "cross_session_insights": [],
    "temporal_analysis": []
}
</output_format>

IMPORTANT: 
- Return ONLY the JSON object, no additional text.
- Compute actual totals from the input data.
- Deduplicate events that appear in multiple sessions.
- All text content must be in English."""

    
    L2_LONGMEMEVAL_AGGREGATION_PROMPT = """You are a conversation synthesis expert. Analyze the following chunk summaries and create a comprehensive overview.

<input_context>
Total Chunks: ${total_chunks}
Conversation Overview:
${chunk_summaries}
</input_context>

<synthesis_goals>
1. **Overall Narrative**: What is the main story or flow of this conversation?
2. **Key Participants**: Who are the main speakers and what are their roles?
3. **Important Information**: What facts would be essential to remember?
4. **Topic Evolution**: How do topics change throughout the conversation?
5. **Unresolved Items**: Are there any pending questions or unfinished discussions?
</synthesis_goals>

<output_format>
Return a JSON object:
{
    "narrative_summary": "A comprehensive 3-5 sentence summary",
    "main_participants": [
        {
            "name": "Name",
            "role": "Their role in the conversation",
            "key_contributions": ["contribution1", "contribution2"]
        }
    ],
    "topic_flow": [
        {"order": 1, "topic": "Topic name", "duration": "chunks 1-3"}
    ],
    "critical_facts": [
        {"fact": "Important fact", "importance": "high|medium|low"}
    ],
    "insights": [
        "Pattern or insight 1",
        "Pattern or insight 2"
    ],
    "pending_items": []
}
</output_format>

Return ONLY the JSON object."""

    
    
    
    L1_DEDUPLICATION_PROMPT = """You are a data deduplication expert. Given a list of L1 summaries that may contain duplicates or overlapping information, merge them into a deduplicated list.

<input_data>
L1 Summaries:
${summaries}
</input_data>

<deduplication_rules>
1. **Identify Duplicates**: Find summaries that describe the same event, fact, or state
2. **Merge Similar**: Combine similar entries, keeping the most complete information
3. **Preserve Unique**: Keep all unique information
4. **Maintain Structure**: Preserve the original data structure
</deduplication_rules>

<output_format>
Return a JSON object:
{
    "deduplicated_count": <number>,
    "original_count": <number>,
    "merged_items": [
        {
            "merged_from": ["id1", "id2"],
            "result": "<merged content>"
        }
    ],
    "unique_items": ["<kept as-is items>"],
    "deduplication_log": ["<explanation of what was merged>"]
}
</output_format>

Return ONLY the JSON object."""

    
    
    
    SESSION_TYPE_PREFIXES = {
        "meeting": "This is a meeting record.",
        "chat": "This is a chat conversation.",
        "task": "This is a task execution record.",
        "document": "This is document content.",
        "qa": "This is a Q&A conversation.",
        "default": "This is conversation content."
    }
    
    
    
    
    @classmethod
    def get_contextual_retrieval_prompt(cls,
                                        session_date: str,
                                        participants: List[str],
                                        full_session_transcript: str,
                                        speaker: str,
                                        message_text: str,
                                        custom_prompt: Optional[str] = None) -> str:
        """Return contextual retrieval prompt."""
        if custom_prompt:
            template = Template(custom_prompt)
        else:
            template = Template(cls.CONTEXTUAL_RETRIEVAL_PROMPT)
        
        return template.safe_substitute(
            session_date=session_date,
            participants=", ".join(participants) if isinstance(participants, list) else participants,
            full_session_transcript=full_session_transcript,
            speaker=speaker,
            message_text=message_text
        )
    
    @classmethod
    def get_l1_prompt(cls,
                      summary_type: str,
                      content: str,
                      extraction_style: str = "default",
                      session_type: str = "default",
                      session_id: Optional[str] = None,
                      session_date: Optional[str] = None,
                      participants: Optional[List[str]] = None,
                      context_id: Optional[str] = None,
                      session_info: Optional[str] = None,
                      custom_prompt: Optional[str] = None) -> str:
        """Return l1 prompt."""
        if custom_prompt:
            template = Template(custom_prompt)
            return template.safe_substitute(
                content=content,
                session_id=session_id or "",
                session_date=session_date or "",
                participants=", ".join(participants) if participants else "",
                transcript=content,
                context_id=context_id or "",
                session_info=session_info or ""
            )
        
        if extraction_style == ExtractionStyle.LOCOMO.value and summary_type == L1SummaryType.STRUCTURED.value:
            template = Template(cls.L1_STRUCTURED_EXTRACTION_PROMPT)
            return template.safe_substitute(
                session_id=session_id or "session_unknown",
                session_date=session_date or "unknown",
                participants=", ".join(participants) if participants else "Unknown",
                transcript=content
            )
        elif extraction_style == ExtractionStyle.LONGMEMEVAL.value:
            template = Template(cls.L1_LONGMEMEVAL_PROMPT)
            return template.safe_substitute(
                context_id=context_id or "chunk_unknown",
                session_info=session_info or "No session info",
                content=content
            )
        else:
            prompt_map = {
                L1SummaryType.EPISODIC.value: cls.L1_EPISODIC_PROMPT,
                L1SummaryType.PROCEDURAL.value: cls.L1_PROCEDURAL_PROMPT,
                L1SummaryType.KNOWLEDGE.value: cls.L1_KNOWLEDGE_PROMPT,
                L1SummaryType.EMOTIONAL.value: cls.L1_EMOTIONAL_PROMPT
            }
            
            base_prompt = prompt_map.get(summary_type, cls.L1_EPISODIC_PROMPT)
            
            prefix = cls.SESSION_TYPE_PREFIXES.get(session_type, "")
            if prefix:
                base_prompt = prefix + "\n\n" + base_prompt
            
            template = Template(base_prompt)
            return template.safe_substitute(content=content)
    
    @classmethod
    def get_l2_prompt(cls,
                      extraction_style: str = "default",
                      summaries: Optional[str] = None,
                      sample_id: Optional[str] = None,
                      total_sessions: Optional[int] = None,
                      participants: Optional[List[str]] = None,
                      time_range: Optional[str] = None,
                      session_data: Optional[str] = None,
                      first_session_date: Optional[str] = None,
                      last_session_date: Optional[str] = None,
                      aggregation_time: Optional[str] = None,
                      total_chunks: Optional[int] = None,
                      chunk_summaries: Optional[str] = None,
                      custom_prompt: Optional[str] = None) -> str:
        """Return l2 prompt."""
        if custom_prompt:
            template = Template(custom_prompt)
            return template.safe_substitute(
                summaries=summaries or "",
                sample_id=sample_id or "",
                total_sessions=total_sessions or 0,
                participants=", ".join(participants) if participants else "",
                time_range=time_range or "",
                session_data=session_data or "",
                first_session_date=first_session_date or "",
                last_session_date=last_session_date or "",
                aggregation_time=aggregation_time or "",
                total_chunks=total_chunks or 0,
                chunk_summaries=chunk_summaries or ""
            )
        
        if extraction_style == ExtractionStyle.LOCOMO.value:
            template = Template(cls.L2_GLOBAL_AGGREGATION_PROMPT)
            return template.safe_substitute(
                sample_id=sample_id or "unknown",
                total_sessions=total_sessions or 0,
                participants=", ".join(participants) if participants else "Unknown",
                time_range=time_range or "unknown",
                session_data=session_data or "",
                first_session_date=first_session_date or "unknown",
                last_session_date=last_session_date or "unknown",
                aggregation_time=aggregation_time or ""
            )
        elif extraction_style == ExtractionStyle.LONGMEMEVAL.value:
            template = Template(cls.L2_LONGMEMEVAL_AGGREGATION_PROMPT)
            return template.safe_substitute(
                total_chunks=total_chunks or 0,
                chunk_summaries=chunk_summaries or ""
            )
        else:
            template = Template(cls.L2_INSIGHT_PROMPT)
            return template.safe_substitute(summaries=summaries or "")
    
    @classmethod
    def get_deduplication_prompt(cls,
                                 summaries: str,
                                 custom_prompt: Optional[str] = None) -> str:
        """Return deduplication prompt."""
        if custom_prompt:
            template = Template(custom_prompt)
        else:
            template = Template(cls.L1_DEDUPLICATION_PROMPT)
        
        return template.safe_substitute(summaries=summaries)



HierarchicalPromptTemplateManager = HierarchicalPromptManager
