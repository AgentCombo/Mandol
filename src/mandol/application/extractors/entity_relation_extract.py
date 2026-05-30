"""Entity and relationship extraction from conversational memory.

Uses LLM prompts to identify entities (person, organization, location, etc.)
and infer semantic relationships between them within a given session context.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from ...domain.memory_unit import MemoryUnit
from ...ports.llm_provider import ChatMessage, LLMProvider

logger = logging.getLogger(__name__)

# Prompt for extracting relationships between known entities.
ENTITY_RELATION_EXTRACT_PROMPT = """You are a professional entity relationship extraction expert. Given a list of entities from a conversation session, identify meaningful relationships between them.

**CRITICAL INSTRUCTIONS:**
- Output ONLY valid JSON format, no other text
- Relationships should be semantic connections meaningful in context
- Confidence score: 0.0 (weak) to 1.0 (strong)
- Only include relationships with confidence >= 0.6
- Entity texts must exactly match the original entity names
- Each entity has a UID - use the UID for source tracking

**RELATION TYPES TO CONSIDER:**
- Identity: is_a, works_as, role_is
- Ownership: owns, belongs_to, contains
- Location: located_at, lives_in, works_at, from
- Temporal: happened_at, lasted, before, after
- Participation: participated_in, experienced, completed
- Emotional: likes, loves, hates, worries_about, supports
- Causal: caused, because, therefore, affected, resulted_in
- Social: friend_of, family_of, colleague_of, neighbor_of, partner_of
- Learning/Work: studies, works_at, teaches, guides, collaborates_with

**ENTITIES TO ANALYZE:**
{entities_json}

**REQUIRED OUTPUT FORMAT:**
{{
  "relationships": [
    {{
      "head_entity": "string - exact entity text",
      "tail_entity": "string - exact entity text",
      "relation_type": "string - use relation type from the list above (e.g., works_as, located_at, likes)",
      "confidence_score": "float",
      "rationale": "string - brief explanation in English",
      "source_uids": ["string - UID of head entity", "string - UID of tail entity"]
    }}
  ]
}}

Generate a response in JSON format. All textual content within the JSON must be in English."""

# Prompt for extracting entities from raw conversation records.
ENTITY_TYPES_EXTRACTION_PROMPT = """You are a professional entity extraction expert. Given a list of memory records, identify and extract all meaningful entities mentioned.

**ENTITY TYPES TO EXTRACT:**
- person: Specific individuals by name or clear reference (e.g., "John", "the doctor", "my manager")
- organization: Companies, institutions, teams, groups (e.g., "Google", "the engineering team")
- location: Places, venues, geographical references (e.g., "San Francisco", "the office", "meeting room")
- concept: Abstract ideas, theories, principles (e.g., "machine learning", "agile methodology")
- object: Physical or digital objects, tools, products (e.g., "the laptop", "the new API")
- event: Specific occurrences, meetings, incidents (e.g., "the launch", "the Q3 meeting")
- activity: Actions, processes, workflows (e.g., "code review", "data migration")

**CRITICAL INSTRUCTIONS:**
- Output ONLY valid JSON format, no other text
- Extract entities that are specific enough to be meaningful
- Include context/description when available
- Each entity must have a unique UID for tracking
- Confidence score: 0.0 (weak) to 1.0 (strong)

**RECORDS TO ANALYZE:**
{records_json}

**REQUIRED OUTPUT FORMAT:**
{{
  "entities": [
    {{
      "text": "string - exact entity text as it appears",
      "entity_type": "string - one of: person, organization, location, concept, object, event, activity",
      "description": "string - brief context or description of the entity",
      "confidence": "float",
      "uid": "string - unique identifier for this entity (format: entity_{index})"
    }}
  ]
}}

Generate a response in JSON format. All textual content within the JSON must be in English."""

RELATION_TYPES = [
    "is_a", "works_as", "role_is",
    "owns", "belongs_to", "contains",
    "located_at", "lives_in", "works_at", "from",
    "happened_at", "before", "after", "during",
    "participated_in", "experienced", "completed",
    "likes", "loves", "hates", "worries_about", "supports",
    "caused", "because", "therefore", "affected", "resulted_in",
    "friend_of", "family_of", "colleague_of", "neighbor_of", "partner_of",
    "studies", "teaches", "guides", "collaborates_with",
]

RELATION_CATEGORIES = {
    "identity": ["is_a", "works_as", "role_is"],
    "ownership": ["owns", "belongs_to", "contains"],
    "location": ["located_at", "lives_in", "works_at", "from"],
    "temporal": ["happened_at", "before", "after", "during"],
    "participation": ["participated_in", "experienced", "completed"],
    "emotional": ["likes", "loves", "hates", "worries_about", "supports"],
    "causal": ["caused", "because", "therefore", "affected", "resulted_in"],
    "social": ["friend_of", "family_of", "colleague_of", "neighbor_of", "partner_of"],
    "learning_work": ["studies", "teaches", "guides", "collaborates_with"],
}


@dataclass
class ExtractedEntity:
    """A single entity extracted from conversation data.

    Attributes:
        text: The entity text as it appears in the source.
        entity_type: Category (person, organization, location, etc.).
        description: Brief contextual description or empty string.
        confidence: Extraction confidence in [0.0, 1.0].
        uid: Unique identifier for this entity.
    """

    text: str
    entity_type: str
    description: str
    confidence: float
    uid: str


@dataclass
class ExtractedRelation:
    """A semantic relationship between two entities.

    Attributes:
        head_entity: Source entity text.
        tail_entity: Target entity text.
        relation_type: Type of relationship (e.g. works_as, located_at).
        confidence: Confidence score in [0.0, 1.0].
        rationale: Human-readable explanation of the relationship.
        source_uids: UIDs of the entities involved.
    """

    head_entity: str
    tail_entity: str
    relation_type: str
    confidence: float
    rationale: str = ""
    source_uids: List[str] = field(default_factory=list)


class EntityRelationExtractor:
    """Extracts entities and relationships from conversational memory.

    Wraps an LLM provider with pre-defined prompts for two tasks:
    1. Entity extraction: identify entities from raw conversation records.
    2. Relationship extraction: infer semantic links between entities.

    Args:
        llm_provider: LLM provider for extraction calls.
        confidence_threshold: Minimum confidence to include a result (default 0.6).
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        confidence_threshold: float = 0.6,
    ):
        self._llm = llm_provider
        self._threshold = float(confidence_threshold)

    def extract_entities(
        self,
        records: Sequence[MemoryUnit],
    ) -> List[ExtractedEntity]:
        """Extract entities from raw conversation records.

        Args:
            records: MemoryUnits containing conversation text.

        Returns:
            List of ExtractedEntity objects.
        """
        if not records:
            return []

        record_list = []
        for i, r in enumerate(records):
            text = r.raw_data.get("text_content", "") or r.raw_data.get("text", "")
            if not text:
                continue
            ts = r.metadata.get("timestamp", "")
            speaker = r.metadata.get("speaker", "Unknown")
            record_list.append({
                "index": i,
                "text": text,
                "timestamp": ts,
                "speaker": speaker,
                "uid": str(r.uid),
            })

        if not record_list:
            return []

        content = json.dumps(record_list, ensure_ascii=False, indent=2)
        prompt = ENTITY_TYPES_EXTRACTION_PROMPT.format(records_json=content)

        messages: List[ChatMessage] = [
            {"role": "user", "content": prompt},
        ]

        try:
            response = self._llm.chat(
                messages,
                temperature=0.1,
                max_tokens=4096,
            )
            return self._parse_entity_response(response.content)
        except (json.JSONDecodeError, ConnectionError, TimeoutError, OSError, RuntimeError) as e:
            logger.error("Entity extraction failed: %s", e)
            return []

    def extract_relations(
        self,
        entities: Sequence[MemoryUnit],
        session_id: str,
    ) -> List[ExtractedRelation]:
        """Extract relationships between a set of entity units.

        Args:
            entities: Entity MemoryUnits to analyze.
            session_id: Session identifier for context tracking.

        Returns:
            List of ExtractedRelation objects above the confidence threshold.
        """
        if not entities or len(entities) < 2:
            return []

        entity_list = []
        for e in entities:
            text = e.raw_data.get("text_content", "") or e.raw_data.get("text", "")
            if not text:
                continue
            entity_type = e.metadata.get("entity_type", "unknown")
            entity_list.append({
                "text": text,
                "type": entity_type,
                "uid": str(e.uid),
            })

        if len(entity_list) < 2:
            return []

        content = json.dumps(entity_list, ensure_ascii=False, indent=2)
        prompt = ENTITY_RELATION_EXTRACT_PROMPT.format(entities_json=content)

        messages: List[ChatMessage] = [
            {"role": "user", "content": prompt},
        ]

        try:
            response = self._llm.chat(
                messages,
                temperature=0.1,
                max_tokens=2048,
            )
            return self._parse_response(response.content, entity_list)
        except (json.JSONDecodeError, ConnectionError, TimeoutError, OSError, RuntimeError) as e:
            logger.error("Entity relation extraction failed: %s", e)
            return []

    def _parse_entity_response(
        self,
        response: str,
    ) -> List[ExtractedEntity]:
        """Parse the LLM JSON response for entity extraction.

        Args:
            response: Raw LLM response string (JSON).

        Returns:
            List of ExtractedEntity objects, or empty on parse failure.
        """
        try:
            data = json.loads(response)
            entities_data = data.get("entities", [])

            results = []
            for item in entities_data:
                text = str(item.get("text", "")).strip()
                if not text:
                    continue
                results.append(ExtractedEntity(
                    text=text,
                    entity_type=str(item.get("entity_type", "unknown")),
                    description=str(item.get("description", "")),
                    confidence=float(item.get("confidence", 0.8)),
                    uid=str(item.get("uid", f"entity_{len(results)}")),
                ))

            return results
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse entity extraction response: {response[:200]}")
            return []

    def _parse_response(
        self,
        response: str,
        entity_list: List[Dict[str, Any]],
    ) -> List[ExtractedRelation]:
        """Parse the LLM JSON response for relationship extraction.

        Validates that both head and tail entities exist in the entity
        list and that confidence exceeds the threshold.

        Args:
            response: Raw LLM response string (JSON).
            entity_list: Original entity dicts for validation.

        Returns:
            List of validated ExtractedRelation objects.
        """
        try:
            data = json.loads(response)
            relations_data = data.get("relationships", [])

            entity_texts = {e["text"] for e in entity_list}
            entity_uids = {e["text"]: e["uid"] for e in entity_list}
            results = []

            for r in relations_data:
                head = str(r.get("head_entity", ""))
                tail = str(r.get("tail_entity", ""))
                rel_type = str(r.get("relation_type", ""))
                conf = float(r.get("confidence_score", 0.0))

                if head not in entity_texts or tail not in entity_texts:
                    continue
                if conf < self._threshold:
                    continue

                source_uids = []
                if head in entity_uids:
                    source_uids.append(entity_uids[head])
                if tail in entity_uids:
                    source_uids.append(entity_uids[tail])

                results.append(ExtractedRelation(
                    head_entity=head,
                    tail_entity=tail,
                    relation_type=rel_type,
                    confidence=conf,
                    rationale=str(r.get("rationale", "")),
                    source_uids=source_uids,
                ))

            return results
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse entity relation response: {response[:200]}")
            return []