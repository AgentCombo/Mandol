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

参数说明
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

在检索管线中的位置
------------------

.. code-block::

   holistic_retrieve(query)
   ├── 1. 分组召回（四组）
   ├── 2. 每组内三路检索（Dense/BM25/Sparse）
   ├── 3. RRF 融合
   ├── 4. BFS 扩展 ← 在此步骤使用
   └── 5. 全局 Rerank
