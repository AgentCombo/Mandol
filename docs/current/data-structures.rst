Core Data Structures
====================

MemoryUnit
----------

``MemoryUnit`` is the basic memory record.

Constructor:

.. code-block:: text

   MemoryUnit(
       uid: str,
       raw_data: dict[str, Any],
       metadata: dict[str, Any] | None = None,
       embedding: np.ndarray | None = None,
       sparse_embedding: dict[int, float] | np.ndarray | None = None,
   )

Important fields:

* ``uid`` is a non-empty string.
* ``raw_data`` is a dictionary. Text extraction prefers ``text_content``,
  ``content``, ``description``, ``summary``, ``title`` and ``message``.
* ``text_cached`` is maintained from ``raw_data`` for retrieval and display.
* ``embedding`` stores dense vectors.
* ``sparse_embedding`` stores SPLADE-style sparse vectors.

MemorySpace
-----------

``MemorySpace`` is a logical tree container. It stores unit UIDs and child space
names, not full unit objects and not local FAISS indexes.

Useful methods:

* ``add_unit(unit_or_uid)``
* ``remove_unit(unit_or_uid)``
* ``add_child_space(space_or_name)``
* ``remove_child_space(space_or_name)``
* ``contains_unit(unit_or_uid, recursive=False)``
* ``get_unit_uids()``
* ``get_all_unit_uids(recursive=True)``

SemanticMap
-----------

``SemanticMap`` owns in-memory units, memory spaces, dense embeddings, optional
SPLADE vectors and a global FAISS index.

Constructor:

.. code-block:: text

   SemanticMap(
       embedding_model_name="Qwen/Qwen3-Embedding-0.6B",
       embedding_dim=None,
       faiss_index_type="IDMap,Flat",
       use_flash_attention=None,
       **kwargs,
   )

Common methods:

* ``add_unit(unit, space_names=None, index_update_mode="incremental",
  generate_sparse_embedding=True)``
* ``batch_add_units(units, batch_size=32, space_names=None,
  per_unit_space_names=None)``
* ``create_memory_space(space_name)``
* ``add_unit_to_space(unit_or_uid, space_name)``
* ``get_units_by_spaces(space_names, mode="union", recursive=True)``
* ``search_similarity_by_text(query_text, k=5, ms_names=None,
  candidate_uids=None)``
* ``search_similarity_by_vector(query_embedding, k=5, ms_names=None,
  candidate_uids=None)``
* ``get_multi_retriever()``
* ``save_map(directory_path)`` and ``load_map(directory_path)`` for resident
  map-only state; use graph snapshots when tiered paging is enabled

SemanticGraph
-------------

``SemanticGraph`` wraps a ``SemanticMap`` and adds a rustworkx directed graph
for explicit relationships and graph traversal.

Constructor:

.. code-block:: text

   SemanticGraph(semantic_map_instance: SemanticMap | None = None)

Common methods:

* ``add_unit(...)`` and ``batch_add_units(...)``
* ``add_relationship(source_uid, target_uid, relationship_name,
  bidirectional=False, **kwargs)``
* ``get_relationship(source_uid, target_uid, relationship_name=None)``
* ``delete_unit(uid)`` and ``delete_relationship(...)``
* ``search_similarity_in_graph(query_text=None, query_embedding=None,
  query_image_path=None, top_k=5, ms_names=None, return_score=False)``
* ``get_multi_retriever()``
* ``save_graph(directory_path)`` and ``load_graph(directory_path)``
* ``connect_to_l2(...)`` and ``close()`` for RocksDB tiered-paging lifecycle

Canonical tower spaces
----------------------

``MemorySpaceRegistry`` and ``TowerSpace`` define canonical names used by the
three retrieval towers:

* ``hierarchical_memory``
* ``hierarchical_memory:L0_Observation``
* ``hierarchical_memory:L1_Summary``
* ``hierarchical_memory:L2_Insight``
* ``entity_relation``
* ``entity_relation:entities``
* ``entity_relation:mentions``
* ``entity_relation:relations``
* ``episodic_memory``
