RRF 融合参考
==================

Reciprocal Rank Fusion — 融合多个检索器结果的无参数算法。

公式
----

::

   RRF(d) = Σ_{r ∈ retrievers} 1 / (k + rank_r(d))

其中：

- ``d``：候选文档
- ``rank_r(d)``：检索器 r 中文档 d 的排名（从 1 开始）
- ``k``：常数（默认 60），防止低排名项权重过高

实现
----

- ``fuse(results: dict[str, list[tuple[Uid, float]]], k=60) -> list[tuple[Uid, float]]``

融合后按 RRF 分数降序排列返回。

为什么用 RRF
------------

1. **无参数调整**：不需对不同检索器的得分做归一化
2. **异构兼容**：BM25（无界）和 Dense（0~1）等不同量纲的分数可公平竞争
3. **稳定排序**：对单路排序的轻微变化不敏感
