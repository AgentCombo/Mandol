"""Entity and event extraction with deduplication and relationship inference.

Provides extractors that use LLMs to identify entities and events from
conversational data, deduplicate them against existing stores, and infer
relationships (entity-entity and event-causal). Also supports cross-session
deduplication via LLM-based cluster judging.
"""

from .entity_dedup import EntityDeduplicator
from .event_dedup import EventDeduplicator
from .entity_relation_extract import EntityRelationExtractor, ExtractedEntity, ExtractedRelation
from .event_causal_extract import EventCausalExtractor, ExtractedEvent, CausalRelation
