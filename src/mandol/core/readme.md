# Mandol Core

`mandol.core` contains the primary in-memory data structures:

- `MemoryUnit`: raw content, metadata, dense embedding and optional sparse embedding.
- `MemorySpace`: UID-based logical membership with nested child spaces.
- `MemorySpaceRegistry`: canonical hierarchical, entity-relation and episodic spaces.
- `SemanticMap`: global FAISS index, UID/int-ID mapping, spaces and vector retrieval.
- `SemanticGraph`: rustworkx directed multigraph layered on a `SemanticMap`.

## Lightweight objects

Creating a `MemoryUnit` or using `MemorySpaceRegistry` does not initialize an
embedding model:

```python
from mandol.core.memory_unit import MemoryUnit
from mandol.core.memory_space_registry import MemorySpaceRegistry, TowerSpace

unit = MemoryUnit(
    uid="message-1",
    raw_data={"text_content": "Let's discuss the project."},
)

assert unit.text_cached == "Let's discuss the project."
assert (
    MemorySpaceRegistry.space_name(TowerSpace.HIERARCHICAL_ROOT)
    == "hierarchical_memory"
)
```

## SemanticMap and MemorySpace

`SemanticMap` initializes the selected embedding backend. It may load or
download a model during construction.

```python
from mandol.core import MemoryUnit, create_semantic_map

semantic_map = create_semantic_map(preset="qwen-0.6b")
semantic_map.add_unit(
    MemoryUnit(
        uid="message-1",
        raw_data={"text_content": "Let's discuss the project."},
    ),
    space_names=["chat:messages"],
)

units = semantic_map.get_units_by_spaces(["chat:messages"])
stats = semantic_map.get_space_statistics()
```

Spaces store references to global units rather than copies or local FAISS
indexes. Nested-space membership is recursively expanded by `SemanticMap`,
which invalidates its cached integer-ID filters when membership changes.

## SemanticGraph

Pass an existing map when both vector and graph retrieval should share the same
units and indexes:

```python
from mandol.core.semantic_graph import SemanticGraph

graph = SemanticGraph(semantic_map_instance=semantic_map)
graph.add_unit(
    MemoryUnit(uid="message-2", raw_data={"text_content": "Follow-up notes."}),
    space_names=["chat:messages"],
)
graph.add_relationship("message-1", "message-2", "followed_by")
```

`SemanticGraph()` without a map constructs a default `SemanticMap` and therefore
also initializes the default embedding model.
