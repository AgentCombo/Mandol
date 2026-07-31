Retrieval Latency Optimization
=================================

Latency Breakdown
-------------------

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Stage
     - Time
     - Description
   * - Query Embedding
     - ~20ms
     - Query vectorization
   * - Dense + BM25 + Sparse
     - ~10-50ms
     - Three-way retrieval in parallel
   * - RRF Fusion
     - ~1ms
     - Rank fusion
   * - BFS Expansion
     - ~50-200ms
     - Graph expansion, biggest variable
   * - Cross-Encoder Rerank
     - ~100-300ms
     - Reranking, affected by candidate count
   * - **Total**
     - **~200-600ms**

Optimization Strategies
-------------------------

1. **Disable Rerank**: ``use_rerank=False``, latency -100~300ms

2. **Disable BFS**: ``bfs_expansion_hops=0``, latency -50~200ms

3. **Reduce candidates**: ``similarity_top_k=3``, reduces processing at each stage

4. **Use GPU**: Embedding + Rerank 3-10x faster on GPU

Ultra-Low Latency Configuration
----------------------------------

.. code-block:: yaml

   system:
     similarity_top_k: 3
     bfs_expansion_hops: 0

.. code-block:: python

   hits = system.holistic_retrieve("...", use_rerank=False, top_k=3)
