# SemanticGraphService (语义关系图 / Semantic Relationship Graph) API Reference

## Overview

`SemanticGraphService` (from `mandol.application.semantic_graph`, accessed via `system.graph`) is the core service for managing graph-structured relationships between memory units. It handles both explicit relationships (entity relations, event causality, provenance) and implicit semantic relationships (vector-similarity-based edges). Graph storage is delegated to `GraphStore` (default: `InMemoryGraphStore`).

> **Note**: The class name in code is `SemanticGraphService`. Throughout documentation it is referred to as SemanticGraph for brevity.

## Constructor

```python
SemanticGraphService(
    *,
    semantic_map: SemanticMapService,
    graph_store: GraphStore,
)
```

## Unit Management

- `add_unit(unit: MemoryUnit, *, space_names, ensure_embedding=True, rebuild_index_immediately=False)`: Add a unit (delegates to SemanticMap)
- `delete_unit(uid: Uid)`: Delete a unit and all its associated edges

## Relationship Management

- `add_relationship(source_uid: Uid, target_uid: Uid, relationship_name: str, **properties)`: Add an explicit relationship edge
- `get_relationship(source_uid: Uid, target_uid: Uid, relationship_name: str) -> Optional[Dict]`: Query relationship edge properties
- `delete_relationship(source_uid: Uid, target_uid: Uid, relationship_name: Optional[str] = None)`: Delete a relationship edge

## Graph Retrieval

- `get_explicit_neighbors(uids: List[Uid], *, rel_type=None, direction="out") -> List[MemoryUnit]`: Get explicit relationship neighbors. Direction values: `"out"`, `"in"`, `"both"`.
- `get_implicit_neighbors(uids: List[Uid], *, top_k=10) -> List[Tuple[MemoryUnit, float]]`: Get implicit semantic neighbors (vector similarity edges)
- `get_units_in_spaces(space_names: List[str], *, mode="union", recursive=True) -> List[MemoryUnit]`: Get units in specified spaces
- `bfs_expand_units(seeds: List[MemoryUnit], *, per_seed=3, hops=1, rel_type=None) -> List[MemoryUnit]`: BFS graph expansion from seed units

## Maintenance

- `flush()`: Clear all graph data
- `get_graph_store() -> GraphStore`: Get underlying graph store

## Edge Types

### Explicit Relations

| Edge Type | Description | Subtypes |
|-----------|-------------|----------|
| `RELATED_TO` | Entity relationship | `hometown`, `lives_in`, `works_at`, `located_in`, `part_of` |
| `CAUSES` | Event causality (A causes B) | — |
| `CAUSED_BY` | Event causality (B caused by A) | — |
| `INVOLVES` | Event-entity relation | `participant`, `location`, `organizer`, `victim` |
| `EVIDENCED_BY` | Provenance edge (points to original dialogue) | — |
| `COREF` | Coreference edge (cross-session entity merge) | — |
| `ALIAS_OF` | Entity alias relation | — |
| `PRECEDES` | Temporal order (A precedes B) | — |
| `FOLLOWS` | Temporal order (B follows A) | — |

### Implicit Relations

| Edge Type | Description |
|-----------|-------------|
| `SEMANTIC_SIMILAR` | Semantic similarity edge (based on vector cosine similarity) |

## Usage

```python
from mandol import MemorySystem
from mandol.domain.types import Uid

system = MemorySystem()

# Add explicit relationship
system.graph.add_relationship(
    source_uid=Uid("entity_001"),
    target_uid=Uid("entity_002"),
    relationship_name="RELATED_TO",
    subtype="works_at",
)

# Get explicit neighbors
neighbors = system.graph.get_explicit_neighbors(
    [Uid("entity_001")],
    rel_type="RELATED_TO",
    direction="out",
)

# Get implicit semantic neighbors
similar = system.graph.get_implicit_neighbors(
    [Uid("dialogue_001")],
    top_k=5,
)

# BFS graph expansion
expanded = system.graph.bfs_expand_units(
    seeds=top_units,
    per_seed=3,
    hops=1,
)
```
