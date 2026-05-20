"""Prompts for cross-session entity/event merging."""

# LLM prompt for judging whether an entity cluster should be merged.
# Inputs: entity name, type, and per-session fact descriptions.
# Output: JSON with should_merge, confidence, canonical_name, reasoning.
CLUSTER_JUDGE_PROMPT = """You are an expert Knowledge Graph Engineer specializing in entity coreference resolution.

### Task
Determine whether the entities in the given cluster refer to the same real-world entity.

### Cluster Information
{cluster_info}

### Instructions

1. Analyze each entity's name, type, and description
2. Check for semantic equivalence (same entity referred by different names/descriptions)
3. Consider:
   - Name similarity (exact match, aliases, abbreviations)
   - Type consistency (must be same type to merge)
   - Description compatibility (no contradictory information)
   - Context clues from session information

4. Decision rules:
   - If ALL entities clearly refer to the same real-world entity: return `should_merge: true`
   - If ANY entity is clearly different or contradictory: return `should_merge: false`
   - If uncertain but likely same: return `should_merge: true` with lower confidence

### Few-Shot Examples

**Example 1: Same Entity, Different Names**
Cluster:
- Entity A: "John" (Person) - "John is a software engineer from West County"
- Entity B: "John Smith" (Person) - "John Smith bought a camera yesterday"
Output:
{{
    "should_merge": true,
    "confidence": 0.95,
    "canonical_name": "John Smith",
    "reasoning": "Both refer to the same person. 'John Smith' is the full name, 'John' is a short form. Descriptions are compatible."
}}

**Example 2: Different Entities, Same Name**
Cluster:
- Entity A: "Apple" (Organization) - "Apple Inc. is a technology company"
- Entity B: "Apple" (Place) - "Apple is a small town in England"
Output:
{{
    "should_merge": false,
    "confidence": 0.90,
    "canonical_name": null,
    "reasoning": "Same name but different types and clearly different real-world entities."
}}

**Example 3: Descriptive Reference Merge**
Cluster:
- Entity A: "John's hometown" (Place) - "John's hometown was hit by a flood"
- Entity B: "West County" (Place) - "West County is located in the northern region"
Output:
{{
    "should_merge": true,
    "confidence": 0.85,
    "canonical_name": "West County",
    "reasoning": "'John's hometown' is a descriptive reference that likely refers to 'West County'. Both are Place type."
}}

**Example 4: Uncertain but Likely Same**
Cluster:
- Entity A: "the company" (Organization) - "the company announced layoffs"
- Entity B: "Google" (Organization) - "Google is a tech giant"
Output:
{{
    "should_merge": false,
    "confidence": 0.60,
    "canonical_name": null,
    "reasoning": "'the company' is too vague to confidently link to 'Google' without more context."
}}

### Output Format (JSON only)
{{
    "should_merge": true|false,
    "confidence": 0.0-1.0,
    "canonical_name": "The preferred canonical name if merging, or null",
    "reasoning": "Brief explanation of the decision"
}}
"""

# LLM prompt for judging whether an event cluster should be merged.
# Output: JSON with should_merge, confidence, canonical_name, reasoning.
EVENT_CLUSTER_JUDGE_PROMPT = """You are an expert Knowledge Graph Engineer specializing in event coreference resolution.

### Task
Determine whether the events in the given cluster refer to the same real-world event.

### Cluster Information
{cluster_info}

### Instructions

1. Analyze each event's name, description, participants, and time
2. Check for semantic equivalence (same event described differently)
3. Consider:
   - Event type/nature similarity
   - Participant overlap
   - Temporal proximity
   - Location consistency
   - Description compatibility

4. Decision rules:
   - If ALL events clearly refer to the same real-world event: return `should_merge: true`
   - If ANY event is clearly different or contradictory: return `should_merge: false`
   - If uncertain but likely same: return `should_merge: true` with lower confidence

### Few-Shot Examples

**Example 1: Same Event, Different Descriptions**
Cluster:
- Event A: "the flood" - "A flood hit John's hometown"
- Event B: "the disaster" - "The disaster caused significant damage to West County"
Output:
{{
    "should_merge": true,
    "confidence": 0.90,
    "canonical_name": "the flood",
    "reasoning": "Both refer to the same flood event. 'the disaster' is a coreference to 'the flood'."
}}

**Example 2: Different Events**
Cluster:
- Event A: "the meeting" - "The meeting was held on Monday"
- Event B: "the conference" - "The conference started on Wednesday"
Output:
{{
    "should_merge": false,
    "confidence": 0.85,
    "canonical_name": null,
    "reasoning": "Different events on different days. Meeting vs conference are distinct."
}}

**Example 3: Same Event with More Details**
Cluster:
- Event A: "John bought a camera" - "John bought a camera yesterday"
- Event B: "the purchase" - "The purchase was made at Amazon"
Output:
{{
    "should_merge": true,
    "confidence": 0.95,
    "canonical_name": "John bought a camera",
    "reasoning": "'the purchase' refers to the camera buying event. Same event with additional details."
}}

### Output Format (JSON only)
{{
    "should_merge": true|false,
    "confidence": 0.0-1.0,
    "canonical_name": "The preferred canonical name if merging, or null",
    "reasoning": "Brief explanation of the decision"
}}
"""

# LLM prompt for merging entity descriptions across multiple sessions.
# Inputs: {name}, {type}, {session_facts} JSON.
# Output: JSON with merged_description, merged_aliases.
CROSS_SESSION_MERGE_PROMPT = """You are an expert Knowledge Graph Engineer.

### Task
Merge descriptions from multiple sessions into a unified canonical description.

### Context
**Entity/Event Name**: {name}
**Type**: {type}
**Session Facts**: {session_facts}

### Instructions

1. Combine all session facts into a single coherent description
2. Resolve contradictions (prefer more recent or more specific information)
3. Keep the description concise (2-4 sentences, max 120 words)
4. Maintain factual accuracy - do not hallucinate
5. Preserve important details from all sessions
6. For merged_aliases: only include SHORT alternative names or references (max 6 words each). Do NOT include full descriptions or sentences as aliases.

### Output Format (JSON only)
{{
    "merged_description": "The unified description combining all session information",
    "merged_aliases": ["short_alias1", "short_alias2"],
    "confidence": 0.95
}}
"""

# LLM prompt for merging event descriptions across multiple sessions.
# Inputs: {name}, {session_facts}, {existing_participants}, {existing_time}.
# Output: JSON with merged_description, merged_participants, merged_time.
CROSS_SESSION_EVENT_MERGE_PROMPT = """You are an expert Knowledge Graph Engineer.

### Task
Merge event descriptions from multiple sessions into a unified canonical description.

### Context
**Event Name**: {name}
**Session Facts**: {session_facts}
**Existing Participants**: {existing_participants}
**Existing Time**: {existing_time}

### Instructions

1. Merge participants from all sessions
2. Resolve time information (prefer explicit time over inferred)
3. Combine descriptions into a single coherent description (1-2 sentences, max 60 words)
4. Do not hallucinate - only use provided information

### Output Format (JSON only)
{{
    "merged_description": "The unified event description",
    "merged_participants": ["participant1", "participant2"],
    "merged_time": "ISO 8601 or relative time",
    "confidence": 0.95
}}
"""
