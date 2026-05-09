"""Coreference graph relationship types for unified fact pipeline.

Design philosophy:
- Abstract stable relations (core layer) vs domain-specific relations (extension layer)
- SAME_AS/SAME_EVENT removed: COREF edge covers cross-session alignment
- INVOLVES unifies all event-entity relationships
- RELATED_TO with subtype for domain-specific entity-entity relations
"""

from __future__ import annotations

from typing import FrozenSet

REL_COREF = "COREF"               # Cross-session coreference: two units refer to the same entity/event
REL_ALIAS_OF = "ALIAS_OF"         # One name is an alias/abbreviation of another
REL_CAUSES = "CAUSES"             # Event A causes Event B (directed)
REL_CAUSED_BY = "CAUSED_BY"       # Inverse of CAUSES (Event B is caused by Event A)
REL_INVOLVES = "INVOLVES"         # An event involves an entity (participant, location, organizer, victim)
REL_EVIDENCED_BY = "EVIDENCED_BY" # Canonical entity/event is supported by a dialogue unit
REL_RELATED_TO = "RELATED_TO"     # Domain-specific entity-entity relation (requires subtype)
REL_SEMANTIC_SIMILAR = "SEMANTIC_SIMILAR"  # Two units are similar by embedding cosine distance
REL_HAPPENED_IN = "HAPPENED_IN"   # An event physically occurred at a location
REL_HOMETOWN = "HOMETOWN"         # A person's hometown (Place)
REL_LIVES_IN = "LIVES_IN"         # A person's current residence (Place)
REL_LOCATED_IN = "LOCATED_IN"     # An entity is physically located within a place

# Location relation types ordered by specificity for deduplication preference
PREFERRED_LOCATION_RELS: FrozenSet[str] = frozenset({
    REL_LOCATED_IN,
    REL_LIVES_IN,
    REL_HOMETOWN,
    REL_HAPPENED_IN,
})

# Default traversal weight for each edge type (1.0 = strongest, 0.0 = weakest)
DEFAULT_REL_WEIGHTS: dict[str, float] = {
    REL_INVOLVES: 1.0,
    REL_COREF: 0.95,
    REL_ALIAS_OF: 0.9,
    REL_CAUSES: 0.7,
    REL_RELATED_TO: 0.55,
    REL_SEMANTIC_SIMILAR: 0.35,
}

# Valid subtypes for the INVOLVES relationship (entity roles in an event)
INVOLVES_SUBTYPES: FrozenSet[str] = frozenset({
    "participant",
    "location",
    "organizer",
    "victim",
})

# Valid subtypes for the RELATED_TO relationship (domain-specific entity relations)
RELATED_TO_SUBTYPES: FrozenSet[str] = frozenset({
    "hometown",
    "lives_in",
    "works_at",
    "located_in",
    "part_of",
})

# Recognized entity types for extraction and deduplication
ENTITY_TYPES: FrozenSet[str] = frozenset({
    "Person",
    "Group",
    "Organization",
    "Place",
    "Object",
    "Activity",
    "Concept",
})

# Expected schema for COREF edge properties (used for validation and documentation)
COREF_EDGE_PROPERTIES = {
    "type": REL_COREF,
    "confidence": float,
    "mention_text": str,
    "session_id": str,
    "original_description": str,
    "original_source_uid": str,
}
