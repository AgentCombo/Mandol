# DuckDB unified storage: concrete implementation plan

## Deliverable boundary

The first implementation milestone establishes a correct physical baseline
before adding cost-based plan selection:

```text
SemanticMap / SemanticGraph
          |
 UnitStore + VectorIndex + GraphStore compatibility facades
          |
       DuckDBUnifiedStore
          |
 exact vector seed -> typed joins -> per-seed top-k -> recursive traversal
```

ANN/VSS persistence, lexical-index unification, arbitrary joins, and the CBO
itself remain later milestones. Exact vector ranking is the correctness
reference against which approximate plans will be measured.

## Implemented file map

| File | Responsibility |
|---|---|
| `src/mandol/infrastructure/duckdb_unified_store.py` | Schema ownership, transactions, three storage facades, unified SQL query |
| `src/mandol/ports/unified_query_store.py` | Backend-neutral unified execution port |
| `src/mandol/query/vector_seeded.py` | Backend-neutral query/result/metrics contracts |
| `tests/unit/test_duckdb_unified_store.py` | Storage conformance, rollback, compatibility, traversal integration |
| `examples/vector_seeded_duckdb.py` | Executable end-to-end example |

The database schema contains:

```text
memory_units
memory_vectors
memory_spaces
space_memberships
space_children
memory_edges
mandol_schema_meta
```

Dense vectors are normalized in `memory_vectors`, so DuckDB
`array_inner_product` exactly matches the existing in-memory cosine index,
including zero-vector behavior. Original embeddings remain in
`memory_units` for lossless `MemoryUnit` reconstruction.

## Completed milestone

- [x] One DuckDB connection, lock, nested transaction manager, and schema version
- [x] Existing `UnitStore`, `VectorIndex`, and `GraphStore` contracts
- [x] Multiple relationship types per endpoint pair
- [x] Space membership and hierarchy round trips
- [x] Exact global and space-restricted vector top-k
- [x] Scalar metadata equality pushdown
- [x] Task-to-session and session-to-node typed joins
- [x] Deterministic per-task fitness top-k
- [x] Bounded, cycle-safe, path-preserving traversal
- [x] Result completeness/truncation and elapsed-time metrics
- [x] Per-query vector/join/traversal cardinality breakdown
- [x] Actual-cardinality ASCII query diagrams
- [x] DuckDB physical `EXPLAIN`
- [x] Existing SemanticMap/SemanticGraph service compatibility tests

## Next milestone: optimizer-ready logical layer

Add immutable logical operators:

```text
UnitScan
Filter
VectorTopK
RelationJoin
GroupTopK
Expand
Project
Limit
```

The current `VectorSeededTraversalSpec` lowers directly to one fixed SQL
template. The next compiler will lower the logical tree to multiple legal SQL
templates while preserving:

- exact versus approximate result properties;
- required ordering and top-k scope;
- bag semantics and alias bindings;
- path uniqueness and maximum depth;
- policy filters that cannot be reordered across unsafe boundaries;
- provenance and result completeness.

## First bounded CBO

Enumerate only a small number of known-correct alternatives:

1. vector-first exact;
2. selective relation/metadata filter followed by candidate vector scoring;
3. oversampled ANN, filter, exact rerank;
4. vector seed, joins, early group top-k, traversal;
5. vector seed, joins, traversal, late top-k when path semantics require it.

Collect:

```text
unit and space cardinalities
metadata selectivity
vector latency and ANN recall by k/ef_search
edge count and degree quantiles by relationship type/direction
task->session and session->node fan-out
per-hop traversal frontier size and duplicate rate
estimated rows, actual rows, elapsed time, and intermediate bytes
```

Use the fixed query in this milestone as the semantic oracle and measured-cost
baseline. A plan is eligible only if its exactness, ordering, path, policy, and
completeness properties satisfy the logical request.

## Executable examples

The small deterministic example prints result paths, stage logs, and an ASCII
query diagram:

```bash
python examples/vector_seeded_duckdb.py
```

Data generation and engine execution are separate processes. First generate a
persistent data set containing task/session/node hierarchies, branching
trajectories, buggy-node filters, fitness rankings, and graph cycles:

```bash
python examples/generate_vector_seeded_dataset.py \
  --database vector_seeded_synthetic.duckdb
```

The default data set contains 3,900 units and 4,410 edges. Scale or reshape
generation independently:

```bash
python examples/generate_vector_seeded_dataset.py \
  --database vector_seeded_synthetic_large.duckdb \
  --tasks 100 \
  --sessions-per-task 4 \
  --nodes-per-session 8 \
  --branching-factor 3 \
  --trajectory-depth 3
```

Then open the existing database and run only the query engine:

```bash
python examples/run_vector_seeded_queries.py \
  --database vector_seeded_synthetic.duckdb \
  --queries 5
```

The stage breakdown reports actual row counts from the same unified SQL:

```text
vector candidates -> vector seeds
                 -> task/session rows
                 -> session/node rows
                 -> selected high-fitness nodes
                 -> traversal rows
```

With `profile=True`, DuckDB's JSON profiler attributes physical-operator CPU
time to the build subtree of each named CTE:

```text
vector_search_cpu_ms
relation_join_cpu_ms
graph_traversal_cpu_ms
other_cpu_ms
total_cpu_ms
```

These are CPU times from the same execution, not three separately rerun
queries. Overall `elapsed_ms` remains wall-clock latency. CPU time and
wall-clock latency need not be equal because profiling, result serialization,
and parallel execution have different accounting. Pass `--no-profile` to the
runner to measure the non-profiled query path; per-stage CPU values will be
reported as `n/a`.

The original combined generator/runner remains available as
`vector_seeded_duckdb_large.py`, but the two-process workflow above is the
recommended benchmark setup.

## Verification commands

```bash
pytest -q tests/unit/test_duckdb_unified_store.py
python examples/vector_seeded_duckdb.py
python examples/generate_vector_seeded_dataset.py \
  --database /tmp/mandol-synthetic.duckdb \
  --overwrite \
  --tasks 6
python examples/run_vector_seeded_queries.py \
  --database /tmp/mandol-synthetic.duckdb \
  --queries 1
ruff check src/mandol/infrastructure/duckdb_unified_store.py \
  src/mandol/query tests/unit/test_duckdb_unified_store.py \
  examples/vector_seeded_duckdb.py \
  examples/generate_vector_seeded_dataset.py \
  examples/run_vector_seeded_queries.py
pytest -q
```
