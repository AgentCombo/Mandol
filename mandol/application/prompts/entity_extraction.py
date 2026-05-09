"""Prompt 1: Entity extraction with coreference resolution."""

# LLM prompt for extracting entities from dialogue with coreference linking.
# Inputs: dialogue text and list of existing candidate entities.
# Output: JSON with extracted entities, each having name, type, linked_id, etc.
ENTITY_EXTRACTION_PROMPT = """You are an expert Knowledge Graph Engineer specializing in entity extraction and coreference resolution.

### Context
**Dialogue (current session)**:
{dialogue_context}

**Existing Canonical Entities** (top-10 most relevant):
{existing_entities}

### Instructions

**Step 1: Coreference Resolution**
For each pronoun ("he", "she", "it", "they", "there") and descriptive reference
("John's hometown", "her company", "that project"):
- If it refers to an existing entity: set `linked_id` to that entity's id
- If it refers to a new entity: set `is_new` to true

**Step 2: Entity Extraction**
Extract all entities mentioned, including:
- Named entities: "John", "West County"
- Descriptive entities: "John's hometown" (as a Place entity, not just an attribute)
- Pronouns resolved: "he" -> linked to existing entity

For each entity, determine:
- `entity_type`: Person | Group | Organization | Place | Object | Activity | Concept
- `mention_text`: the exact text span used to refer to this entity

### Few-Shot Examples

**Example 1: Descriptive Reference**
Dialogue: "John's hometown was hit by a terrible flood"
Existing Entities: []
Output:
{{
    "entities": [
        {{"reasoning": "'John' is a named person entity", "entity_name": "John",
         "entity_type": "Person", "linked_id": null, "is_new": true,
         "confidence": 0.95, "mention_text": "John", "new_facts": [], "aliases": []}},
        {{"reasoning": "'John's hometown' is a descriptive reference to a place - it should be extracted as a Place entity, not just as an attribute of John",
         "entity_name": "John's hometown", "entity_type": "Place", "linked_id": null,
         "is_new": true, "confidence": 0.90, "mention_text": "John's hometown",
         "new_facts": [], "aliases": []}}
    ]
}}

**Example 2: Pronoun Resolution**
Dialogue: "He really likes it. John bought a camera yesterday."
Existing Entities: [{{"id": "entity:john", "name": "John", "type": "Person"}}]
Output:
{{
    "entities": [
        {{"reasoning": "'He' refers to John based on context",
         "entity_name": "John", "entity_type": "Person", "linked_id": "entity:john",
         "is_new": false, "confidence": 0.90, "mention_text": "He",
         "new_facts": ["bought a camera yesterday"], "aliases": ["He"]}},
        {{"reasoning": "'it' refers to the camera mentioned later in the dialogue",
         "entity_name": "camera", "entity_type": "Object", "linked_id": null,
         "is_new": true, "confidence": 0.85, "mention_text": "it",
         "new_facts": [], "aliases": []}}
    ]
}}

**Example 3: Entity Linking with Existing**
Dialogue: "Mary works at Google. She loves her job there."
Existing Entities: [{{"id": "entity:mary", "name": "Mary", "type": "Person"}}, {{"id": "entity:google", "name": "Google", "type": "Organization"}}]
Output:
{{
    "entities": [
        {{"reasoning": "'Mary' matches existing entity",
         "entity_name": "Mary", "entity_type": "Person", "linked_id": "entity:mary",
         "is_new": false, "confidence": 0.95, "mention_text": "Mary",
         "new_facts": ["works at Google"], "aliases": []}},
        {{"reasoning": "'She' refers to Mary",
         "entity_name": "Mary", "entity_type": "Person", "linked_id": "entity:mary",
         "is_new": false, "confidence": 0.95, "mention_text": "She",
         "new_facts": ["loves her job"], "aliases": ["She"]}},
        {{"reasoning": "'Google' matches existing entity",
         "entity_name": "Google", "entity_type": "Organization", "linked_id": "entity:google",
         "is_new": false, "confidence": 0.95, "mention_text": "Google",
         "new_facts": [], "aliases": []}},
        {{"reasoning": "'her job' is a descriptive reference to Mary's job",
         "entity_name": "Mary's job", "entity_type": "Activity", "linked_id": null,
         "is_new": true, "confidence": 0.80, "mention_text": "her job",
         "new_facts": [], "aliases": []}},
        {{"reasoning": "'there' refers to Google",
         "entity_name": "Google", "entity_type": "Organization", "linked_id": "entity:google",
         "is_new": false, "confidence": 0.90, "mention_text": "there",
         "new_facts": [], "aliases": []}}
    ]
}}

### Output Format (JSON only)
{{
    "entities": [
        {{
            "reasoning": "Why this entity and its coreference decision",
            "entity_name": "Canonical name",
            "entity_type": "Person|Group|Organization|Place|Object|Activity|Concept",
            "linked_id": "entity_id or null",
            "is_new": false,
            "confidence": 0.95,
            "mention_text": "Original mention span",
            "new_facts": ["fact1"],
            "aliases": ["short alternative name only, max 6 words"]
        }}
    ]
}}
"""
