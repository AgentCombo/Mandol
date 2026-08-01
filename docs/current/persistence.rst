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
indices and a sandboxed RocksDB payload store when persistent storage is
enabled.

Automatic payload paging
------------------------

RocksDB is the only supported persistent payload backend in the current
implementation. Calling ``connect_to_l2`` enables automatic tiered payload paging.
Dense, BM25 and SPLADE indexes, UID mappings, MemorySpace membership and graph
topology remain resident. Cold ``MemoryUnit`` payloads are written to RocksDB
and paged back into the resident cache on demand.

.. code-block:: python

   graph.connect_to_l2(
       "./l2_database",
       max_capacity=100_000,
       high_watermark=0.85,
       low_watermark=0.70,
   )

If ``connect_to_l2`` is not called, payloads remain resident in memory and the
graph runs without persistent payload storage. This is the normal baseline
state, not a separately configured placement mode.

Tiered eviction is triggered from the existing add path. Candidate selection
and eviction scheduling occur within the add call, while RocksDB persistence
and resident-cache removal may complete asynchronously in the tiered-storage
executor. Cold-result materialization remains inside the search call that
requires the payload. Retrieval algorithms, top-k values, reranking and result
schemas are unchanged.

SemanticMap snapshots
---------------------

``SemanticMap.save_map`` saves resident map-only state. When tiered paging is
enabled, it fails closed because a map-only directory cannot include the cold
RocksDB payload catalog. Use ``SemanticGraph.save_graph`` for a complete
tiered checkpoint.

``SemanticGraph.save_graph`` pauses new eviction scheduling and waits for
already submitted eviction work before writing the map, graph and RocksDB
snapshot. Concurrent graph mutation from another user thread is not supported
during this operation.

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
