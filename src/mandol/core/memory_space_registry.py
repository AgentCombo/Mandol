"""Central registry for Mandol triple-tower memory spaces."""

from __future__ import annotations

from enum import Enum
from typing import Any, Iterable, Union


class TowerSpace(str, Enum):
    """Canonical tree-shaped memory-space names for the three memory towers."""

    HIERARCHICAL_ROOT = "hierarchical_memory"
    HIERARCHICAL_L0 = "hierarchical_memory:L0_Observation"
    HIERARCHICAL_L1 = "hierarchical_memory:L1_Summary"
    HIERARCHICAL_L2 = "hierarchical_memory:L2_Insight"

    GRAPH_ROOT = "entity_relation"
    GRAPH_ENTITIES = "entity_relation:entities"
    GRAPH_MENTIONS = "entity_relation:mentions"
    GRAPH_RELATIONS = "entity_relation:relations"

    EPISODIC_ROOT = "episodic_memory"


MemorySpaceName = Union[str, TowerSpace]


class MemorySpaceRegistry:
    """Register and mount canonical tower memory spaces on a SemanticGraph or SemanticMap."""

    HIERARCHICAL_CHILDREN = (
        TowerSpace.HIERARCHICAL_L0,
        TowerSpace.HIERARCHICAL_L1,
        TowerSpace.HIERARCHICAL_L2,
    )
    GRAPH_CHILDREN = (
        TowerSpace.GRAPH_ENTITIES,
        TowerSpace.GRAPH_MENTIONS,
        TowerSpace.GRAPH_RELATIONS,
    )

    @staticmethod
    def space_name(space: MemorySpaceName) -> str:
        """Return the concrete string name for a registry enum or raw name."""
        return space.value if isinstance(space, TowerSpace) else str(space)

    @staticmethod
    def space_names(spaces: Iterable[MemorySpaceName]) -> list[str]:
        """Return concrete string names for a sequence of registry spaces."""
        return [MemorySpaceRegistry.space_name(space) for space in spaces]

    @staticmethod
    def initialize_spaces(semantic_system: Any) -> None:
        """Create canonical spaces and mount tower children under their roots."""
        space_manager = getattr(semantic_system, "semantic_map", semantic_system)
        create_space = getattr(semantic_system, "create_memory_space_in_map", None)
        if create_space is None:
            create_space = getattr(space_manager, "create_memory_space", None)
        if create_space is None or not hasattr(space_manager, "add_space_to_space"):
            raise TypeError(
                "initialize_spaces() expects a SemanticGraph or SemanticMap-like object "
                "with memory-space creation and add_space_to_space() support"
            )

        for space in TowerSpace:
            create_space(space.value)

        for child in MemorySpaceRegistry.HIERARCHICAL_CHILDREN:
            space_manager.add_space_to_space(child.value, TowerSpace.HIERARCHICAL_ROOT.value)
        for child in MemorySpaceRegistry.GRAPH_CHILDREN:
            space_manager.add_space_to_space(child.value, TowerSpace.GRAPH_ROOT.value)
