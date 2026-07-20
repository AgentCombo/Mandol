Bm25Retriever 参考
========================

基于 BM25 算法的关键词检索器。

主要方法
--------

- ``index(units: list[MemoryUnit]) -> None`` — 构建 BM25 索引
- ``search(query: str, top_k: int) -> list[tuple[Uid, float]]``

配置
----

- ``k1``：词频饱和参数（默认 1.5）
- ``b``：长度归一化参数（默认 0.75）
