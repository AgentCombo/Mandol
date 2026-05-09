"""Prompt 3: Entity-entity relation extraction."""

# LLM prompt for extracting semantic relationships between entity pairs.
# Inputs: dialogue text and extracted entities with their UIDs.
# Output: JSON with relations list (source, target, rel_type, confidence, reasoning).
ENTITY_RELATION_PROMPT = """You are an expert Knowledge Graph Engineer.

### Context
**Dialogue**:
{dialogue_context}

**Current Session Entities**:
{current_entities}

### Instructions

Extract relationships between entities based on the dialogue.

For descriptive references that imply relationships:
- "John's hometown" implies: John --RELATED_TO {{subtype:"hometown"}}--> Place
- "Mary's company" implies: Mary --RELATED_TO {{subtype:"works_at"}}--> Organization
- "the capital of France" implies: France --RELATED_TO {{subtype:"located_in"}}--> Place

Available subtypes for RELATED_TO:
- hometown: person's hometown
- lives_in: person's current residence
- works_at: person's workplace
- located_in: entity's location
- part_of: part-whole relationship

### Few-Shot Examples

**Example 1: Hometown Relationship**
Dialogue: "John's hometown was hit by a terrible flood"
Current Entities: [{{"id": "entity:john", "name": "John", "type": "Person"}}, {{"id": "entity:john_s_hometown", "name": "John's hometown", "type": "Place"}}]
Output:
{{
    "relations": [
        {{
            "source": "entity:john",
            "target": "entity:john_s_hometown",
            "rel_type": "RELATED_TO",
            "subtype": "hometown",
            "confidence": 0.90,
            "reasoning": "'John's hometown' implies a hometown relationship between John and the place"
        }}
    ]
}}

**Example 2: Multiple Relationships**
Dialogue: "Mary works at Google in Mountain View. She lives in San Francisco."
Current Entities: [{{"id": "entity:mary", "name": "Mary", "type": "Person"}}, {{"id": "entity:google", "name": "Google", "type": "Organization"}}, {{"id": "entity:mountain_view", "name": "Mountain View", "type": "Place"}}, {{"id": "entity:san_francisco", "name": "San Francisco", "type": "Place"}}]
Output:
{{
    "relations": [
        {{
            "source": "entity:mary",
            "target": "entity:google",
            "rel_type": "RELATED_TO",
            "subtype": "works_at",
            "confidence": 0.95,
            "reasoning": "Mary works at Google"
        }},
        {{
            "source": "entity:google",
            "target": "entity:mountain_view",
            "rel_type": "RELATED_TO",
            "subtype": "located_in",
            "confidence": 0.90,
            "reasoning": "Google is located in Mountain View"
        }},
        {{
            "source": "entity:mary",
            "target": "entity:san_francisco",
            "rel_type": "RELATED_TO",
            "subtype": "lives_in",
            "confidence": 0.95,
            "reasoning": "Mary lives in San Francisco"
        }}
    ]
}}

**Example 3: No Relationships**
Dialogue: "The weather is nice today."
Current Entities: []
Output:
{{
    "relations": []
}}

### Output Format (JSON only)
{{
    "relations": [
        {{
            "source": "entity:source_id",
            "target": "entity:target_id",
            "rel_type": "RELATED_TO",
            "subtype": "hometown|lives_in|works_at|located_in|part_of",
            "confidence": 0.9,
            "reasoning": "Why this relationship exists"
        }}
    ]
}}
"""
