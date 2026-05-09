Retriever Reference
=====================

.. toctree::
   :maxdepth: 1

   hybrid-retriever
   bm25-retriever
   sparse-retriever
   subgraph-retriever
   fusion

HybridRetriever — Multi-Path Retrieval Orchestration
-------------------------------------------------------

Coordinates Dense/BM25/Sparse three-way retrieval, RRF fusion, and BFS expansion.

Bm25Retriever — Keyword Retrieval
------------------------------------

Keyword-level retriever based on the BM25 algorithm.

SparseRetriever — Sparse Vector Retrieval
--------------------------------------------

Sparse vector retriever based on TF-IDF.

SubgraphHopRetriever — Graph Retriever
-----------------------------------------

Expands along graph relationships, performing BFS exploration using retrieval results as seeds.

RRFusion — Rank Fusion
-------------------------

Merges rankings from multiple retrievers using Reciprocal Rank Fusion.

Fusion formula:

::

   RRF(d) = Σ 1 / (k + rank_r(d))
