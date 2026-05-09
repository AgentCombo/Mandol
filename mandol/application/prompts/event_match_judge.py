"""Prompt for LLM-based event match judging.

Determines whether a newly extracted event refers to the same real-world
occurrence as a known canonical event candidate.
"""

# LLM prompt for pairwise event matching with confidence scoring.
# Inputs: new event details and list of candidate canonical events.
# Output: JSON with matched boolean, canonical_id, confidence, reasoning.
EVENT_MATCH_JUDGE_PROMPT = """You are an expert in coreference resolution and event matching.

Given a newly extracted event and a list of candidate canonical events from the knowledge graph, determine whether the new event refers to the same real-world event as one of the candidates.

## New Event
Name: {new_event_name}
Description: {new_event_description}
Participants: {new_participants}
Time: {new_time}
Facts: {new_facts}

## Candidate Canonical Events
{candidates}

## Instructions
1. Compare the new event with each candidate carefully.
2. Consider: event name similarity, description overlap, participant overlap, temporal proximity, and location consistency.
3. If the new event clearly refers to the same real-world event as a candidate, select that candidate.
4. If the new event is distinct from all candidates, indicate no match.
5. Events from different sessions may describe the same real-world event from different perspectives.

## Output Format
Return a JSON object:
```json
{{
    "matched_index": <integer or null>,
    "confidence": <float between 0.0 and 1.0>,
    "canonical_name_suggestion": <string or null>,
    "reasoning": "<brief explanation>"
}}
```

- `matched_index`: 0-based index of the matched candidate, or null if no match
- `confidence`: your confidence in the match decision (0.0 = no confidence, 1.0 = absolute certainty)
- `canonical_name_suggestion`: suggested canonical name if merging (or null to keep existing)
- `reasoning`: brief explanation of your decision
"""
