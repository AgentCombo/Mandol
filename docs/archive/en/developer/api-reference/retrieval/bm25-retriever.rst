Bm25Retriever Reference
==========================

Keyword retriever based on the BM25 algorithm.

Main Methods
-------------

- ``index(units: list[MemoryUnit]) -> None`` — Build BM25 index
- ``search(query: str, top_k: int) -> list[tuple[Uid, float]]``

Configuration
--------------

- ``k1``: Term frequency saturation parameter (default 1.5)
- ``b``: Length normalization parameter (default 0.75)
