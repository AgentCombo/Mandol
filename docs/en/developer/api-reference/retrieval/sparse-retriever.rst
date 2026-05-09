SparseRetriever Reference
============================

Sparse vector retriever based on TF-IDF.

Main Methods
-------------

- ``index(units: list[MemoryUnit]) -> None``
- ``search(query_vector, top_k: int) -> list[tuple[Uid, float]]``

Difference from BM25: Uses TF-IDF weights for cosine similarity computation, results are closer to Dense's scoring semantics.
