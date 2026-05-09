"""Task B: Description merge prompt for entities and events."""

# LLM prompt for merging entity descriptions from multiple sessions.
# Inputs: canonical name, type, and list of per-session fact descriptions.
# Output: JSON with merged_description and merged_aliases.
DESCRIPTION_MERGE_PROMPT = """You are an expert Knowledge Graph Engineer.

### Task
Merge the existing canonical description with newly extracted facts to create an updated, unified description.

### Context
**Entity/Event Name**: {name}
**Type**: {type}
**Existing Canonical Description**: {existing_description}
**Newly Extracted Facts**: {new_facts}
**Previous Session Facts**: {session_facts}

### Instructions

1. Combine all information into a single, coherent description
2. Resolve any contradictions (prefer newer information if conflicting)
3. Keep the description concise (2-4 sentences, max 120 words)
4. Maintain factual accuracy - do not hallucinate
5. Preserve important details from both existing and new information

### Few-Shot Examples

**Example 1: Entity Description Merge**
Name: John
Existing Description: John is a software engineer.
New Facts: ["bought a camera yesterday", "is from West County"]
Previous Session Facts: []
Output:
{{
    "merged_description": "John is a software engineer from West County who recently bought a camera.",
    "confidence": 0.95
}}

**Example 2: Entity Description with Contradiction**
Name: Mary
Existing Description: Mary lives in New York.
New Facts: ["moved to San Francisco last month"]
Previous Session Facts: []
Output:
{{
    "merged_description": "Mary is a person who recently moved to San Francisco. She previously lived in New York.",
    "confidence": 0.90
}}

**Example 3: Event Description Merge**
Name: flood
Existing Description: A flood hit John's hometown.
New Facts: ["happened on January 15, 2024", "caused significant damage"]
Previous Session Facts: []
Output:
{{
    "merged_description": "A flood hit John's hometown on January 15, 2024, causing significant damage.",
    "confidence": 0.95
}}

**Example 4: Cross-Session Entity Merge**
Name: West County
Existing Description: John is from West County.
New Facts: ["hit by a flood", "located in the northern region"]
Previous Session Facts: [{{"session": "sess_1", "description": "John's hometown"}}]
Output:
{{
    "merged_description": "West County is John's hometown located in the northern region. It was hit by a flood.",
    "confidence": 0.90
}}

### Output Format (JSON only)
{{
    "merged_description": "The unified description combining existing and new information",
    "confidence": 0.95
}}
"""

# LLM prompt for merging event descriptions from multiple sessions.
# Inputs: canonical name, existing participants, time, and session facts.
# Output: JSON with merged_description, merged_participants, merged_time.
EVENT_DESCRIPTION_MERGE_PROMPT = """You are an expert Knowledge Graph Engineer.

### Task
Merge the existing canonical event description with newly extracted facts to create an updated, unified description.

### Context
**Event Name**: {name}
**Existing Canonical Description**: {existing_description}
**Existing Participants**: {existing_participants}
**Existing Time**: {existing_time}
**Newly Extracted Facts**: {new_facts}
**New Participants**: {new_participants}
**New Time**: {new_time}

### Instructions

1. Merge participants: combine existing and new participants, remove duplicates
2. Resolve time: prefer explicit time over inferred, newer information over older
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
