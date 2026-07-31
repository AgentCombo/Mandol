Quick Start
===========

Start with the current ``MemoryUnit``, ``SemanticMap`` and ``SemanticGraph``
APIs.

Minimal semantic graph
----------------------

This example uses the small MiniLM preset and disables realtime SPLADE
generation so first-run setup stays modest. Creating ``SemanticMap`` loads an
embedding model; if it is not cached, sentence-transformers may download it.

.. code-block:: python

   from mandol import MemoryUnit, SemanticGraph, SemanticMap

   semantic_map = SemanticMap(
       embedding_model_name="all-MiniLM-L6-v2",
       use_flash_attention=False,
   )
   graph = SemanticGraph(semantic_map_instance=semantic_map)

   graph.add_unit(
       MemoryUnit(
           uid="msg_001",
           raw_data={"text_content": "Zhang San travelled to Beijing today."},
           metadata={"timestamp": "2026-06-21T09:00:00"},
       ),
       space_names=["demo"],
       generate_sparse_embedding=False,
   )
   graph.add_unit(
       MemoryUnit(
           uid="msg_002",
           raw_data={"text_content": "He will discuss the Q2 delivery plan."},
           metadata={"timestamp": "2026-06-21T09:05:00"},
       ),
       space_names=["demo"],
       generate_sparse_embedding=False,
   )

   graph.add_relationship("msg_001", "msg_002", "NEXT")

   hits = graph.search_similarity_in_graph(
       query_text="Where did Zhang San go?",
       top_k=3,
       ms_names=["demo"],
       return_score=True,
   )

   for unit, score in hits:
       print(f"{score:.3f} {unit.uid}: {unit.text_cached}")

Multi-method retrieval
----------------------

Use ``MultiRetriever.smart_search`` for BM25, SPLADE, cosine retrieval, fusion
and optional reranking. The example avoids reranking so it does not load a
cross-encoder model.

.. code-block:: python

   from mandol.retrieval import MultiRetriever

   retriever = MultiRetriever(graph)
   results = retriever.smart_search(
       "Where did Zhang San go?",
       methods=["bm25", "cosine"],
       top_k=5,
       rerank_method=None,
       space_names=["demo"],
   )

   for unit, score in results:
       print(score, unit.uid)

Save and load
-------------

.. code-block:: python

   graph.save_graph("./memory_snapshot", build_sparse_vectors=False)

   restored = SemanticGraph.load_graph(
       "./memory_snapshot",
       embedding_model_name="all-MiniLM-L6-v2",
       use_flash_attention=False,
   )

Use ``SemanticGraph.save_graph`` for complete snapshots. ``SemanticMap.save_map``
is available for resident map-only persistence, but it does not preserve graph
topology and fails closed while tiered paging is enabled.
