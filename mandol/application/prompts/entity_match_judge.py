"""Prompt for LLM-based entity match judging.

Determines whether a newly extracted entity refers to the same real-world
entity as a known canonical entity candidate.
"""

ENTITY_MATCH_JUDGE_PROMPT = """You are an expert in coreference resolution and entity matching.

Given a newly extracted entity and a list of candidate canonical entities from the knowledge graph, determine whether the new entity refers to the same real-world entity as one of the candidates.

## New Entity
Name: {new_entity_name}
Type: {new_entity_type}
Facts: {new_facts}
Mention text: "{mention_text}"

## Candidate Canonical Entities
{candidates}

## Instructions
1. Compare the new entity with each candidate carefully.
2. Consider: name similarity, type compatibility, description overlap, and factual consistency.
3. If the new entity clearly refers to the same real-world entity as a candidate, select that candidate.
4. If the new entity is distinct from all candidates, indicate no match.
5. When uncertain but the entities could plausibly be the same, prefer matching (with lower confidence).

## Output Format
Return a JSON object:
```json
{{
    "matched_index": <integer or null>,
    "confidence": <float between 0.0 and 1.0>,
    "canonical_name_suggestion": <string or null>,
    "new_aliases": <list of strings>,
    "reasoning": "<brief explanation>"
}}
```

- `matched_index`: 0-based index of the matched candidate, or null if no match
- `confidence`: your confidence in the match decision (0.0 = no confidence, 1.0 = absolute certainty)
- `canonical_name_suggestion`: suggested canonical name if merging (or null to keep existing)
- `new_aliases`: any new aliases to add to the canonical entity
- `reasoning`: brief explanation of your decision
"""
