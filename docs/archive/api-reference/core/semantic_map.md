# SemanticMapService (语义映射表 / Semantic Map) API Reference

## Overview

`SemanticMapService` (from `mandol.application.semantic_map`, accessed via `system.semantic_map`) is the core service for managing memory storage and vector indexing. It integrates `UnitStore` (unit storage), `AdaptiveVectorIndex` (adaptive vector index), `EmbeddingProvider` (embedding), and `Reranker` (reranking) to provide a complete infrastructure for semantic retrieval.

> **Note**: The class name in code is `SemanticMapService`. Throughout documentation it is referred to as SemanticMap for brevity.

## Constructor

```python
SemanticMapService(
    *,
    store: UnitStore,
    index: VectorIndex,
    embedder: Optional[EmbeddingProvider] = None,
    reranker: Optional[Reranker] = None,
    default_text_key: str = "text_content",
    default_image_path_key: str = "image_path",
    max_units_in_memory: int = 10000,
)
```

## Space Management

- `create_space(name) -> MemorySpace`: Create a new space (returns existing if present)
- `get_space(name) -> Optional[MemorySpace]`: Get a space by name
- `list_spaces() -> List[MemorySpace]`: List all spaces
- `add_unit_to_space(uid: Uid, space_name: str)`: Add a unit to a specified space
- `attach_child_space(parent: str, child: str, *, ensure_exists: bool = True)`: Establish parent-child space relationship
- `ensure_child_space(parent: str, child: str) -> MemorySpace`: Ensure child space exists and is attached

## Unit Management

- `add_unit(unit: MemoryUnit, *, space_names, ensure_embedding=True, rebuild_index_immediately=False, embedding_text=None, embedding_image_path=None)`: Add a memory unit
- `upsert_unit(unit: MemoryUnit, *, rebuild_index_immediately=False)`: Insert or update a unit
- `delete_unit(uid: Uid)`: Delete a unit (removes from all spaces)
- `get_unit(uid: Uid) -> Optional[MemoryUnit]`: Get a single unit
- `list_units() -> List[MemoryUnit]`: List all units

## Space Retrieval

- `get_units_in_spaces(space_names: List[str], *, mode="union", recursive=True) -> List[MemoryUnit]`: Get all units within specified spaces

## Semantic Search

- `search_by_text(query_text: str, *, top_k=10, space_names=None, recursive=True) -> List[Tuple[MemoryUnit, float]]`: Text-based semantic search
- `search_by_vector(query: np.ndarray, *, top_k=10, space_names=None, recursive=True) -> List[Tuple[MemoryUnit, float]]`: Vector-based semantic search
- `search_by_text_with_rerank(query_text: str, *, top_k=10, recall_k=None, space_names=None, recursive=True, use_rerank=True) -> List[Tuple[MemoryUnit, float]]`: Text-based semantic search with Cross-Encoder reranking
- `search_in_space(query_text: str, space_name: str, candidates=None, *, top_k=10, recall_k=None) -> List[Tuple[MemoryUnit, float]]`: Search within a specified space

## Index Maintenance

- `rebuild_index_from_store()`: Rebuild vector index from storage
- `set_embedder(embedder: EmbeddingProvider)`: Replace the embedder
- `set_reranker(reranker: Reranker)`: Replace the reranker
- `flush()`: Clear all data

## Usage

```python
from mandol import MemorySystem
from mandol.domain.types import Uid

system = MemorySystem()

# Space management
custom_space = system.semantic_map.create_space("my_custom_space")
system.semantic_map.add_unit_to_space(Uid("dialogue_001"), "my_custom_space")

# Text semantic search
results = system.semantic_map.search_by_text("Beijing", top_k=10)
for unit, score in results:
    print(f"{unit.uid}: score={score:.3f}")

# Search with reranking
results = system.semantic_map.search_by_text_with_rerank(
    "Where did Zhang San go?", top_k=5
)

# Get units in specific spaces
entities = system.semantic_map.get_units_in_spaces(
    ["root_knowledge_entity"]
)
```
