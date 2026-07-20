SubgraphHopRetriever Reference
=================================

Graph traversal expander starting from retrieval seeds.

Main Methods
-------------

- ``expand(seeds: list[Uid], per_seed: int, hops: int) -> list[Uid]``

Expansion Strategy:

1. First hop: From seeds, take top ``per_seed`` neighbors along all edge types
2. Subsequent hops: Continue expanding from previous hop results
3. Deduplicate and return

Called by ``HybridRetriever``, also available directly via ``system.semantic_graph.bfs_expand_units``.
