Memory Building and Retrieval Pipeline
========================================

Mandol's core pipeline has only three steps: **add memory → build high-level semantics → retrieve**. No matter what level of user you are, you need to understand this pipeline. This chapter provides three levels of pipeline descriptions — choose the depth that suits your needs.

.. toctree::
   :maxdepth: 1

   basic-flow
   detailed-flow
   architecture-flow

Pipeline Overview
-----------------

.. mermaid::

   graph LR
       A[MemoryUnit] -->|add| B[Vectorization + Indexing]
       B -->|build_high_level| C[Session Segmentation]
       C --> D[Multi-dimensional Construction]
       D --> E[Entity / Event / Summary / Relationship]
       E -->|holistic_retrieve| F[Multi-path Recall]
       F --> G[RRF Fusion + BFS Expansion + Rerank]
       G --> H[SearchHit Results]
