# In-memory vector-seeded graph traversal

Mandol's default stores can execute the first multimodal fixed plan without
DuckDB, a model download, or an API key:

```bash
python examples/vector_seeded_in_memory.py
```

The runnable example builds this graph:

```text
session-1 -[BELONGS_TO]-> task-vector
node-best -[IN_SESSION]-> session-1
node-lower -[IN_SESSION]-> session-1
node-buggy -[IN_SESSION]-> session-1
node-best -[HAS_CHILD]-> trajectory-1 -[HAS_CHILD]-> trajectory-2
```

It then executes:

```text
VectorScan
  -> seed metadata Filter
  -> vector Top-K
  -> typed BELONGS_TO EdgeJoin
  -> typed IN_SESSION EdgeJoin
  -> node metadata Filter
  -> per-seed GroupTopK
  -> bounded path-preserving Traverse
  -> stable ProjectLimit
```

The executor reports, for every physical operator:

- physical algorithm;
- logical stage;
- input cardinality;
- output cardinality;
- exclusive elapsed time.

These observations are exposed through
`execution.metrics.operators` and are intended to become feedback for a later
cost model. The current milestone deliberately uses one fixed plan and does
not perform cost estimation, join reordering, or physical-plan enumeration.

Applications can access the executor through `MemorySystem`:

```python
execution = system.vector_seeded_graph_traversal(spec, profile=True)
plan = system.queries.explain_vector_seeded_graph_traversal(spec)
```

The in-memory graph uses `(source_uid, target_uid, rel_type)` as relationship
identity, so differently typed parallel edges between the same two nodes are
preserved.
