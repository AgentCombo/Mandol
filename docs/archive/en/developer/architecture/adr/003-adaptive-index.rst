ADR-003: Adaptive Vector Index
==================================

Status
-------

Accepted

Date
-----

2024-10

Context
--------

Brute-force search is faster for small amounts of memories (< 100), while FAISS indexing is faster at scale. A smooth transition is needed.

Decision
---------

Introduce ``AdaptiveVectorIndex``, which automatically switches based on data volume:
- < 100: numpy brute-force search
- >= 100: FAISS HNSW index

The threshold is controlled by ``promote_threshold`` (default 100).

Consequences
-------------

- Small dataset users get optimal performance at zero configuration cost
- Large-scale data automatically gets FAISS acceleration
- One-time index construction overhead during switching (~10ms per thousand records)
