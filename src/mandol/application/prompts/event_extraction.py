"""Prompt 2: Event extraction with coreference resolution."""

# LLM prompt for extracting events from dialogue with coreference linking.
# Inputs: dialogue text and list of existing candidate events.
# Output: JSON with extracted events, each having name, description, linked_id, etc.
EVENT_EXTRACTION_PROMPT = """You are an expert Knowledge Graph Engineer specializing in event extraction and coreference resolution.

### Context
**Dialogue (current session)**:
{dialogue_context}

**Existing Canonical Events** (top-5 most relevant):
{existing_events}

**Current Session Entities**:
{current_entities}

### Instructions

**Step 1: Event Extraction**
Extract events with their entity relationships:
- `event_name`: canonical event name
- `description`: brief event description
- `participants`: list of entity references involved (use linked_id from current_entities)
- `location`: where the event happened, as an entity name string from Current Session Entities. Use null if no location.
- `inferred_time`: ISO 8601 date or relative time expression

**Step 2: Coreference Resolution**
For event references like "the meeting", "that project", "it" (referring to an event):
- If it refers to an existing event: set `linked_id` to that event's id
- If it refers to a new event: set `is_new` to true

**Step 3: Participant Linking**
For each participant:
- Use `linked_id` from Current Session Entities if the entity matches
- Set `role` to describe the entity's role in the event

**CRITICAL**: `location` MUST be a plain string (the entity name from Current Session Entities) or null. It must NEVER be an object/dict. `participants` entries MUST use `linked_id` from Current Session Entities.

### Few-Shot Examples

**Example 1: Basic Event Extraction**
Dialogue: "John's hometown was hit by a terrible flood"
Existing Events: []
Current Entities: [{{"id": "entity:john", "name": "John", "type": "Person"}}, {{"id": "entity:john_s_hometown", "name": "John's hometown", "type": "Place"}}]
Output:
{{
    "events": [
        {{
            "reasoning": "A flood event is described that affected John's hometown",
            "event_name": "flood",
            "linked_id": null,
            "is_new": true,
            "confidence": 0.95,
            "description": "A terrible flood hit John's hometown",
            "participants": [
                {{"mention": "John", "linked_id": "entity:john", "role": "participant"}}
            ],
            "location": "John's hometown",
            "inferred_time": null,
            "new_facts": []
        }}
    ]
}}

**Example 2: Event with Time**
Dialogue: "Mary attended the conference last Friday. She presented her research there."
Existing Events: []
Current Entities: [{{"id": "entity:mary", "name": "Mary", "type": "Person"}}, {{"id": "entity:conference", "name": "the conference", "type": "Activity"}}]
Output:
{{
    "events": [
        {{
            "reasoning": "Mary attended a conference and presented research",
            "event_name": "conference attendance",
            "linked_id": null,
            "is_new": true,
            "confidence": 0.90,
            "description": "Mary attended a conference and presented her research",
            "participants": [
                {{"mention": "Mary", "linked_id": "entity:mary", "role": "participant"}}
            ],
            "location": "the conference",
            "inferred_time": "last Friday",
            "new_facts": ["Mary presented her research"]
        }}
    ]
}}

**Example 3: Event Coreference**
Dialogue: "The meeting was postponed. It will happen next week instead."
Existing Events: [{{"id": "event:meeting_2024", "name": "meeting", "description": "A meeting"}}]
Current Entities: []
Output:
{{
    "events": [
        {{
            "reasoning": "'The meeting' refers to the existing meeting event, 'It' also refers to the same meeting",
            "event_name": "meeting",
            "linked_id": "event:meeting_2024",
            "is_new": false,
            "confidence": 0.90,
            "description": "The meeting was postponed to next week",
            "participants": [],
            "location": null,
            "inferred_time": "next week",
            "new_facts": ["was postponed", "will happen next week"]
        }}
    ]
}}

### Output Format (JSON only)
{{
    "events": [
        {{
            "reasoning": "Why this event and its coreference decision",
            "event_name": "Canonical event name",
            "linked_id": "event_id or null",
            "is_new": true,
            "confidence": 0.9,
            "description": "Event description",
            "participants": [
                {{"mention": "John", "linked_id": "entity:john", "role": "participant"}}
            ],
            "location": "West County or null",
            "inferred_time": "2024-01-15 or null",
            "new_facts": []
        }}
    ]
}}
"""
