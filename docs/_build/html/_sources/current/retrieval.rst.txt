Retrieval
=========

The current retrieval entry point is ``mandol.retrieval.MultiRetriever``.
Historical docs that mention ``HybridRetriever`` or
``mandol.retrieval.pipeline`` refer to an older API.

Available methods
-----------------

``RetrievalMethod`` currently includes:

* ``BM25`` / ``"bm25"``
* ``COSINE_SIMILARITY`` / ``"cosine"`` or ``"cosine_similarity"``
* ``SPLADE`` / ``"splade"``
* ``GRAPH_TRAVERSAL`` / ``"graph"``
* ``HYBRID`` / ``"hybrid"``
* ``GRAPH_CONTEXT_EXPANSION`` / ``"graph_context_expansion"``

Basic search
------------

.. code-block:: python

   from mandol.retrieval import MultiRetriever

   retriever = MultiRetriever(graph)
   results = retriever.smart_search(
       "Where did Zhang San go?",
       methods=["bm25", "cosine"],
       top_k=5,
       fusion_method="rrf",
       rerank_method=None,
       space_names=["demo"],
   )

``smart_search`` returns ``list[tuple[MemoryUnit, float]]`` by default. Pass
``return_detailed=True`` to include the execution plan, method results and
timing information.

Quantified search
-----------------

.. code-block:: python

   payload = retriever.smart_search_with_quantification(
       "Where did Zhang San go?",
       methods=["bm25", "cosine"],
       top_k=5,
       rerank_method="baai",
       space_names=["demo"],
   )

   results = payload["results"]
   metrics = payload["quantification"]

The quantification payload reports consistency between sparse and dense result
sets, a confidence score and a simple diagnosis. The current implementation
uses a reranker in this path; choose the backend and method deliberately because
local rerankers can load large models.

Async reranking with vLLM
-------------------------

If ``RERANKER_BACKEND=vllm`` is set, local neural rerankers must use async
retrieval paths:

.. code-block:: python

   results = await retriever.smart_search_async(
       "query",
       methods=["bm25", "cosine"],
       rerank_method="baai",
   )

The sync APIs intentionally raise for this backend/method combination to avoid
blocking a vLLM HTTP rerank path incorrectly.

Three-tower retrieval
---------------------

``mandol.triple_retrieval.TripleTowerRetriever`` orchestrates hierarchical,
entity-relation and episodic retrieval over already-built memory spaces.

.. code-block:: python

   from mandol.triple_retrieval import TripleTowerConfig, TripleTowerRetriever

   tower = TripleTowerRetriever(graph, config=TripleTowerConfig(final_top_k=10))
   result = tower.search("What changed in the Q2 delivery plan?")

The three tower package is retrieval-facing. Automatic high-level memory
construction is not exposed as a top-level ``MemorySystem`` API in this checkout.
