SparseRetriever 参考
==========================

基于 TF-IDF 的稀疏向量检索器。

主要方法
--------

- ``index(units: list[MemoryUnit]) -> None``
- ``search(query_vector, top_k: int) -> list[tuple[Uid, float]]``

与 BM25 的区别：使用 TF-IDF 权重做余弦相似度计算，结果更接近 Dense 的评分语义。
