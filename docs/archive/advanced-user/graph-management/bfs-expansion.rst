BFS 图扩展
==============

BFS 扩展是 ``holistic_retrieve`` 中的关键步骤，也可以单独使用进行图遍历探索。

直接使用
--------

.. code-block:: python

   seeds = [Uid("entity_张三"), Uid("event_会议")]
   expanded = system.semantic_graph.bfs_expand_units(
         seeds=seeds,
         per_seed=5,
         hops=2,
   )
   print(f"从 {len(seeds)} 个种子发现 {len(expanded)} 个相关节点")

工作原理
--------

.. mermaid::

   graph LR
       S1[种子节点 1] --> N1[1-hop 邻居 A]
       S1 --> N2[1-hop 邻居 B]
       S2[种子节点 2] --> N3[1-hop 邻居 C]
       N1 --> N4[2-hop 邻居 D]
       N3 --> N5[2-hop 邻居 E]

1. 从种子节点出发，沿所有边类型扩展
2. 第一跳取每个种子的 top ``per_seed`` 邻居
3. 第二跳从第一跳结果继续扩展
4. 返回去重后的所有发现的节点

参数调优
--------

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - 参数
     - 默认值
     - 建议范围
     - 说明
   * - ``per_seed``
     - 3
     - 1-10
     - 每个种子拉多少个邻居
   * - ``hops``
     - 1
     - 0-2
     - 扩展跳数；hops=0 不扩展

小 per_seed + 大 hops 适合多跳推理，大 per_seed + 小 hops 适合广度覆盖。

在检索管线中的位置
------------------

.. code-block::

   holistic_retrieve(query)
   ├── 1. 分组召回（四组）
   ├── 2. 每组内三路检索（Dense/BM25/Sparse）
   ├── 3. RRF 融合
   ├── 4. BFS 扩展 ← 在此步骤使用
   └── 5. 全局 Rerank
