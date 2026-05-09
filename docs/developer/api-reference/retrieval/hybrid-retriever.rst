HybridRetriever 参考
=========================

负责编排多路检索、融合和扩展的检索管线。

主要方法
--------

- ``retrieve(query: str, config: RetrievalConfig) -> list[SearchHit]``

``RetrievalConfig`` 包含：

- ``groups: list[RetrievalGroup]`` — 检索分组（BASE/ENTITY/EVENT/SUMMARY）
- ``retriever_types: list[str]`` — 使用的检索器
- ``use_graph_expansion: bool`` — 是否 BFS 扩展
- ``use_rerank: bool`` — 是否 Cross-Encoder 重排

使用示例
--------

.. code-block:: python

   # HybridRetriever 是 MemorySystem 的内部组件
   # 通过 holistic_retrieve/search 间接使用
   hits = system.holistic_retrieve("查询", top_k=10)
