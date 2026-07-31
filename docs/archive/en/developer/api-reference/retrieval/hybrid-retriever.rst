HybridRetriever Reference
============================

Orchestrates multi-path retrieval, fusion, and expansion retrieval pipeline.

Main Methods
-------------

- ``retrieve(query: str, config: RetrievalConfig) -> list[SearchHit]``

``RetrievalConfig`` contains:

- ``groups: list[RetrievalGroup]`` — Retrieval groups (BASE/ENTITY/EVENT/SUMMARY)
- ``retriever_types: list[str]`` — Retrievers to use
- ``use_graph_expansion: bool`` — Whether to use BFS expansion
- ``use_rerank: bool`` — Whether to use Cross-Encoder reranking

Usage Example
--------------

.. code-block:: python

   # HybridRetriever is an internal component of MemorySystem
   # Used indirectly through holistic_retrieve/search
   hits = system.holistic_retrieve("query", top_k=10)
