RRF Fusion Reference
=======================

Reciprocal Rank Fusion — A parameter-free algorithm for merging results from multiple retrievers.

Formula
--------

::

   RRF(d) = Σ_{r ∈ retrievers} 1 / (k + rank_r(d))

Where:

- ``d``: Candidate document
- ``rank_r(d)``: Rank of document d in retriever r (starting from 1)
- ``k``: Constant (default 60), prevents low-ranked items from having too much weight

Implementation
---------------

- ``fuse(results: dict[str, list[tuple[Uid, float]]], k=60) -> list[tuple[Uid, float]]``

Returns results sorted by RRF score in descending order.

Why RRF
---------

1. **No parameter tuning**: No need to normalize scores across different retrievers
2. **Heterogeneous compatibility**: BM25 (unbounded) and Dense (0~1) scores with different scales can compete fairly
3. **Stable ranking**: Not sensitive to minor changes in individual retriever rankings
