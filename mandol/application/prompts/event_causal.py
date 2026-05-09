"""Prompt 4: Event-event causal relation extraction."""

# LLM prompt for extracting causal relationships between event pairs.
# Inputs: dialogue text and extracted events with their UIDs.
# Output: JSON with causal_relations list (cause_event, effect_event, type, confidence).
EVENT_CAUSAL_PROMPT = """You are an expert Knowledge Graph Engineer.

### Context
**Dialogue**:
{dialogue_context}

**Current Session Events**:
{current_events}

### Instructions

Extract causal relationships between events.
- A CAUSES B: event A directly causes event B
- Consider temporal order: cause should happen before effect
- Look for explicit causal language: "because", "due to", "led to", "resulted in", "caused"

### Few-Shot Examples

**Example 1: Explicit Causation**
Dialogue: "The heavy rain caused severe flooding in the area."
Current Events: [{{"id": "event:heavy_rain", "name": "heavy rain"}}, {{"id": "event:flooding", "name": "flooding"}}]
Output:
{{
    "causal_relations": [
        {{
            "cause_event": "event:heavy_rain",
            "effect_event": "event:flooding",
            "confidence": 0.95,
            "reasoning": "The dialogue explicitly states that heavy rain caused flooding"
        }}
    ]
}}

**Example 2: Implicit Causation**
Dialogue: "John missed the deadline. He had to work overtime to catch up."
Current Events: [{{"id": "event:missed_deadline", "name": "missed deadline"}}, {{"id": "event:work_overtime", "name": "work overtime"}}]
Output:
{{
    "causal_relations": [
        {{
            "cause_event": "event:missed_deadline",
            "effect_event": "event:work_overtime",
            "confidence": 0.85,
            "reasoning": "Missing the deadline caused John to work overtime to catch up"
        }}
    ]
}}

**Example 3: No Causal Relation**
Dialogue: "John went to the store. Mary read a book."
Current Events: [{{"id": "event:store_visit", "name": "store visit"}}, {{"id": "event:reading", "name": "reading"}}]
Output:
{{
    "causal_relations": []
}}

**Example 4: Chain of Events**
Dialogue: "The server crashed due to high traffic. This caused the website to go down, and users couldn't access their accounts."
Current Events: [{{"id": "event:server_crash", "name": "server crash"}}, {{"id": "event:website_down", "name": "website down"}}, {{"id": "event:access_failure", "name": "access failure"}}]
Output:
{{
    "causal_relations": [
        {{
            "cause_event": "event:server_crash",
            "effect_event": "event:website_down",
            "confidence": 0.95,
            "reasoning": "Server crash caused website to go down"
        }},
        {{
            "cause_event": "event:website_down",
            "effect_event": "event:access_failure",
            "confidence": 0.90,
            "reasoning": "Website being down caused users to not access accounts"
        }}
    ]
}}

### Output Format (JSON only)
{{
    "causal_relations": [
        {{
            "cause_event": "event:cause_id",
            "effect_event": "event:effect_id",
            "confidence": 0.85,
            "reasoning": "Why this causal relationship exists"
        }}
    ]
}}
"""
