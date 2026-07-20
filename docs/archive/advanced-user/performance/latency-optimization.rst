检索延迟优化
============

延迟分解
--------

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - 阶段
     - 耗时
     - 说明
   * - Query Embedding
     - ~20ms
     - 查询向量化
   * - Dense + BM25 + Sparse
     - ~10-50ms
     - 三路检索并行
   * - RRF 融合
     - ~1ms
     - 排名融合
   * - BFS 扩展
     - ~50-200ms
     - 图扩展，最大变数
   * - Cross-Encoder Rerank
     - ~100-300ms
     - 精排，受候选数影响
   * - **总计**
     - **~200-600ms**

优化策略
--------

1. **关闭 Rerank**：``use_rerank=False``，延迟 -100~300ms

2. **关闭 BFS**：``bfs_expansion_hops=0``，延迟 -50~200ms

3. **减少候选数**：``similarity_top_k=3``，减少各阶段处理量

4. **使用 GPU**：Embedding + Rerank 在 GPU 上快 3-10x

极致低延迟配置
--------------

.. code-block:: yaml

   system:
     similarity_top_k: 3
     bfs_expansion_hops: 0

.. code-block:: python

   hits = system.holistic_retrieve("...", use_rerank=False, top_k=3)
