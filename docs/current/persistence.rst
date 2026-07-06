Persistence
===========

SemanticGraph snapshots
-----------------------

Use ``SemanticGraph.save_graph`` and ``SemanticGraph.load_graph`` for complete
state snapshots.

.. code-block:: python

   graph.save_graph(
       "./memory_snapshot",
       freeze_retrievers=False,
       force_rebuild_retrievers=False,
       build_sparse_vectors=False,
   )

   restored = SemanticGraph.load_graph(
       "./memory_snapshot",
       embedding_model_name="all-MiniLM-L6-v2",
       use_flash_attention=False,
   )

The saved directory contains graph state, semantic map data, optional retrieval
indices and a sandboxed DuckDB L2 copy when L2 storage is connected.

SemanticMap snapshots
---------------------

``SemanticMap.save_map`` and ``SemanticMap.load_map`` save the map layer only:

.. code-block:: python

   semantic_map.save_map("./map_snapshot", build_sparse_vectors=False)
   restored_map = SemanticMap.load_map(
       "./map_snapshot",
       embedding_model_name="all-MiniLM-L6-v2",
       use_flash_attention=False,
   )

Use map-only snapshots only when graph topology and relationship state are not
needed.

Index behavior
--------------

``save_map`` automatically builds the FAISS index if units have dense embeddings
but the index is empty. SPLADE sparse vector generation can be skipped with
``build_sparse_vectors=False`` to avoid loading the SPLADE model during a light
checkpoint.
