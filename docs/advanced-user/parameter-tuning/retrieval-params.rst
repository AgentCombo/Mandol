检索参数
========

similarity_threshold
---------------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - 参数
     - 默认值
     - 说明
   * - ``similarity_threshold``
     - 0.7
     - SEMANTIC_SIMILAR 边建立阈值，影响图密度

- 提高（0.8-0.9）：更精确，图更稀疏
- 降低（0.5-0.6）：更多边，BFS 效果更强

similarity_top_k
-----------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - 参数
     - 默认值
     - 说明
   * - ``similarity_top_k``
     - 5
     - 每组向量检索初始召回数

bfs_expansion_per_seed / bfs_expansion_hops
--------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - 参数
     - 默认值
     - 说明
   * - ``bfs_expansion_per_seed``
     - 3
     - 每个种子扩展邻居数
   * - ``bfs_expansion_hops``
     - 1
     - BFS 跳数；hops=0 关闭

.. list-table::
   :header-rows: 1
   :widths: 25 50 25

   * - 场景
     - 配置
     - 延迟
   * - 精确检索
     - per_seed=0, hops=0
     - 最低
   * - 一般
     - per_seed=3, hops=1
     - 中等
   * - 多跳推理
     - per_seed=5, hops=2
     - ~2-3x
