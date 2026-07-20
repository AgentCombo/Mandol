检索器参考
============

.. toctree::
   :maxdepth: 1

   hybrid-retriever
   bm25-retriever
   sparse-retriever
   subgraph-retriever
   fusion

HybridRetriever — 多路检索编排
--------------------------------

负责协调 Dense/BM25/Sparse 三路检索、RRF 融合和 BFS 扩展。

Bm25Retriever — 关键词检索
----------------------------

基于 BM25 算法的关键词级检索器。

SparseRetriever — 稀疏向量检索
--------------------------------

基于 TF-IDF 的稀疏向量检索器。

SubgraphHopRetriever — 图检索器
---------------------------------

沿图关系扩展，以检索结果为种子进行 BFS 探索。

RRFusion — 排名融合
--------------------

使用倒数排名融合 (Reciprocal Rank Fusion) 合并多个检索器的排名。

融合公式：

::

   RRF(d) = Σ 1 / (k + rank_r(d))
