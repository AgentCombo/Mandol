"""Core domain model for the Mandol memory system.

Defines the fundamental data structures used throughout the entire system:
  - MemoryUnit: atomic unit of memory carrying raw data and embeddings.
  - MemorySpace: hierarchical container for grouping units.
  - Coreference graph constants: relationship types, weights, and subtypes
    for the semantic graph layer.

Exports:
    MemoryUnit, MemorySpace, SearchResult, Uid, SpaceName, Embedding,
    and the primary relation constants (REL_COREF, REL_ALIAS_OF, etc.).
"""

from .coref_graph_constants import (
    DEFAULT_REL_WEIGHTS,
    PREFERRED_LOCATION_RELS,
    REL_ALIAS_OF,
    REL_CAUSES,
    REL_COREF,
    REL_HAPPENED_IN,
    REL_HOMETOWN,
    REL_LIVES_IN,
    REL_LOCATED_IN,
    REL_SEMANTIC_SIMILAR,
)
from .memory_space import MemorySpace
from .memory_unit import MemoryUnit
from .types import Embedding, SearchResult, SpaceName, Uid

__all__ = [
    "DEFAULT_REL_WEIGHTS",
    "Embedding",
    "MemorySpace",
    "MemoryUnit",
    "PREFERRED_LOCATION_RELS",
    "REL_ALIAS_OF",
    "REL_CAUSES",
    "REL_COREF",
    "REL_HAPPENED_IN",
    "REL_HOMETOWN",
    "REL_LIVES_IN",
    "REL_LOCATED_IN",
    "REL_SEMANTIC_SIMILAR",
    "SearchResult",
    "SpaceName",
    "Uid",
]
