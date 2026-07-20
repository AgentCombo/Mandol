SemanticMapService Complete Reference
========================================

Responsible for unit CRUD, vector indexing, and space management.

Access
-------

Accessed through ``system.semantic_map``.

All Public Methods
-------------------

Space Management:

- ``create_space(name: str) -> MemorySpace``
- ``get_space(name: str) -> MemorySpace | None``
- ``list_spaces() -> list[MemorySpace]``
- ``delete_space(name: str, cascade: bool = False) -> None``
- ``attach_child_space(parent: str, child: str) -> None``
- ``ensure_child_space(parent: str, child: str) -> None``

Unit Management:

- ``add_unit(unit: MemoryUnit) -> None``
- ``upsert_unit(unit: MemoryUnit) -> None``
- ``delete_unit(uid: Uid) -> None``
- ``get_unit(uid: Uid) -> MemoryUnit | None``
- ``list_units() -> list[MemoryUnit]``
- ``add_unit_to_space(uid: str, space_name: str) -> None``
- ``remove_unit_from_space(uid: str, space_name: str) -> None``
- ``get_units_in_spaces(space_names: list[str]) -> list[MemoryUnit]``
- ``list_units_in_space(space_name: str) -> list[MemoryUnit]``

Retrieval:

- ``search_by_vector(embedding, top_k: int, space_names=None, recursive=True) -> list[tuple[Uid, float]]``
- ``search_by_text(text: str, top_k: int, space_names=None, recursive=True) -> list[tuple[Uid, float]]``
- ``search_by_text_with_rerank(text: str, top_k: int, recall_k=None, space_names=None, recursive=True, use_rerank=True) -> list[tuple[Uid, float]]``
- ``search_in_space(embedding, space_name: str, top_k: int) -> list[tuple[Uid, float]]``

Maintenance:

- ``rebuild_index_from_store() -> None``
- ``flush() -> None``
- ``count_units(space_name=None) -> int``

Configuration:

- ``set_embedder(provider) -> None`` / ``get_embedder()``
- ``set_reranker(provider) -> None`` / ``get_reranker()``
- ``get_store()``
- ``dim`` (property)

Usage Example
--------------

.. code-block:: python

   system.semantic_map.create_space("Knowledge Base")
   system.semantic_map.add_unit_to_space("uid_123", "Knowledge Base")
   results = system.semantic_map.search_by_text_with_rerank("query", top_k=10)
