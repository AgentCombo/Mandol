SubgraphHopRetriever 参考
================================

以检索种子为起点的图遍历扩展器。

主要方法
--------

- ``expand(seeds: list[Uid], per_seed: int, hops: int) -> list[Uid]``

扩展策略：

1. 第一跳：从种子出发，沿所有边类型取 top ``per_seed`` 邻居
2. 后续跳：从上一跳结果继续扩展
3. 去重返回

被 ``HybridRetriever`` 调用，也可通过 ``system.semantic_graph.bfs_expand_units`` 直接使用。
