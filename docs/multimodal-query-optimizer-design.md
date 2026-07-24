# Mandol multimodal query unification and optimizer design

Date: 2026-07-23

## Executive answer

Mandol currently unifies graph, vector, and key-value (KV) search at the
**data-identity and application-composition layers**:

- A `MemoryUnit` is the canonical record. Its UID keys raw data and metadata,
  and the same object may carry dense and sparse embeddings.
- `SemanticMapService` coordinates the unit store, spaces, and vector index.
- `SemanticGraphService` stores explicit UID-to-UID edges and uses the same
  semantic map to resolve nodes and synthesize implicit semantic neighbors.
- Retrieval pipelines pass UID-backed `MemoryUnit` objects directly from
  dense/BM25/sparse retrieval into graph expansion and reranking, avoiding a
  user-visible conversion between vector and graph records.

That is a useful unification, but it is **not yet a unified query system**.
The current `main` branch has separate ports and physical stores, procedural
Python pipelines, post-filters, and no logical algebra, join operator,
statistics catalog, plan enumeration, or cost model.

Mandol does support:

- metadata predicates over units;
- union/intersection of memory-space membership;
- relationship-type and direction filters during neighbor lookup or BFS.

It does **not** currently unify relational filtering or joins with vector
search and graph traversal in one optimizable request. In particular,
`global vector top-k -> fixed-size result -> filter` can silently return fewer than `k`
eligible results, and the fixed retrieval pipeline cannot choose a different
order.

The recommended implementation is a Python-native, Cascades-inspired
optimizer introduced incrementally:

1. define a small typed logical algebra;
2. execute the existing fixed pipeline through physical operators;
3. add statistics and `EXPLAIN ANALYZE`;
4. enumerate a bounded set of vector/filter/join/traversal plan templates;
5. add adaptive filtered ANN and vector-seeded graph traversal;
6. move to a memo/rule optimizer only after the estimates are validated.

Do not embed Axiom/Velox into Mandol initially. Axiom is a useful architectural
reference, but it is a C++ execution stack and would overwhelm the current
Python ports before Mandol has a stable logical contract.

## Scope and evidence

This assessment targets repository `main` at commit `8c795ca`. The README
explicitly says that `main` is being refactored and differs from the released
paper artifact; exact reproduction lives on `paper-repro`
([`README.md:15-16`](../README.md)).

The papers were downloaded locally:

- [Mandol: An Agglomerative Agent Memory System for Long-Term
  Conversations](../papers/2606.29778.pdf), especially Sections 3.1-3.3 and 4.
- [Experience Graphs: The Data Foundation for Self-Improving
  Agents](../papers/2606.29823.pdf), especially Sections 4, 6, and 7.

This distinction matters because the Mandol paper describes in-process DuckDB
paging, whereas current `main` defaults to in-memory dictionaries, a vector
index, NetworkX, and JSON persistence.

## 1. How Mandol unifies graph, vector, and KV search

### 1.1 The paper's design

Section 3.2 of the Mandol paper (PDF page 4) describes an “agglomerative semantic data
structure” made of two cooperating structures in one address space:

- **SemanticMap**: UID-keyed raw information, metadata-based space membership,
  dense vectors, sparse vectors, and an inverted index.
- **SemanticGraph**: an adjacency list for explicit structural relationships;
  implicit semantic relationships are computed from SemanticMap indexes.
- **Shared identity**: lookup, filtering, vector search, and graph traversal
  return the same Memory Unit identifiers.
- **Atomic hybrid composition**: a dense hit can immediately become a graph
  seed and then be filtered by memory-space metadata.
- **Persistence**: the paper's active in-memory layer pages cold data to an
  embedded DuckDB backend.

The paper then uses a fixed quantitative retrieval flow:

`source routing -> parallel BM25/SPLADE/dense recall -> RRF -> selective graph expansion -> cross-encoder rerank -> denoise/conflict resolution -> MMR/token budget`

Calling these operations “atomic hybrid operators” means that they share a
record identity and can be composed without cross-database serialization. It
does not imply that the paper defines a relational algebra or a cost-based
optimizer.

### 1.2 What current `main` actually implements

The canonical unit is a dataclass containing a UID, raw-data dictionary,
metadata dictionary, dense embedding, and sparse embedding
([`memory_unit.py:23-48`](../src/mandol/domain/memory_unit.py)). A `MemorySpace`
is a named set of those UIDs, with optional hierarchy and its own summary
embedding ([`memory_space.py:25-48`](../src/mandol/domain/memory_space.py)).

The storage contracts are deliberately separate:

- `UnitStore`: unit/space CRUD and scans
  ([`unit_store.py:17-109`](../src/mandol/ports/unit_store.py)).
- `VectorIndex`: global and candidate-restricted top-k
  ([`vector_index.py:15-86`](../src/mandol/ports/vector_index.py)).
- `GraphStore`: typed edge CRUD and neighbor access
  ([`graph_store.py:18-101`](../src/mandol/ports/graph_store.py)).

`MemorySystem` wires them into one process. By default it chooses
`InMemoryUnitStore`, `AdaptiveVectorIndex`, and `InMemoryGraphStore`, passes the
unit store and vector index to `SemanticMapService`, and gives that same map to
`SemanticGraphService`
([`memory_system.py:247-298`](../src/mandol/application/memory_system.py)).

The physical defaults are:

- Python dictionaries for units/spaces
  ([`in_memory_unit_store.py:18-32`](../src/mandol/infrastructure/in_memory_unit_store.py));
- brute-force cosine search, promoted to FAISS `IndexFlatIP` (still an exact
  flat scan, despite “ANN” wording in some docstrings)
  ([`adaptive_vector_index.py:22-44`](../src/mandol/infrastructure/adaptive_vector_index.py),
  [`faiss_vector_index.py:21-44`](../src/mandol/infrastructure/faiss_vector_index.py));
- a NetworkX directed graph keyed by UIDs
  ([`in_memory_graph_store.py:17-31`](../src/mandol/infrastructure/in_memory_graph_store.py)).

On insertion, space membership is written into unit metadata and the space's
UID set; the unit is then written to the KV store and its embedding is upserted
under the same UID in the vector index
([`semantic_map.py:155-203`](../src/mandol/application/semantic_map.py)).
Explicit relationships require their endpoint UIDs to resolve in the semantic
map ([`semantic_graph.py:102-130`](../src/mandol/application/semantic_graph.py)).
Implicit neighbors average seed embeddings and invoke vector search
([`semantic_graph.py:199-226`](../src/mandol/application/semantic_graph.py)).

The result is best visualized as follows:

```text
                         shared UID
                            |
          +-----------------+-----------------+
          |                 |                 |
  UnitStore / spaces   VectorIndex       GraphStore
  raw data, metadata   UID -> vector     UID -[type]-> UID
          \                 |                 /
           \------- SemanticMap/Graph -------/
                          services
                             |
    dense + lexical + sparse -> RRF -> BFS -> rerank
```

The `HybridRetriever` implements that last line as a hard-coded flow
([`pipeline.py:66-77`](../src/mandol/retrieval/pipeline.py),
[`pipeline.py:127-186`](../src/mandol/retrieval/pipeline.py),
[`pipeline.py:215-290`](../src/mandol/retrieval/pipeline.py)). This is
application-level orchestration over common objects, not a physical query plan.

### 1.3 Paper/current-branch gaps

| Capability | Mandol paper | Current `main` |
|---|---|---|
| Canonical identity | Shared Memory Unit ID | Implemented |
| KV + vector record | SemanticMap | Implemented through separate store/index ports |
| Explicit graph | Lightweight adjacency list | NetworkX `DiGraph` |
| Implicit graph | Computed from vector index | Mean seed vector then top-k |
| Dense + sparse + keyword recall | Dense/SPLADE/BM25 | Dense/TF-IDF sparse/BM25 |
| Hybrid pipeline | Routed, fused, expanded, reranked | Fixed service pipeline |
| Query-adaptive source routing | Intent-based source selection | `holistic_retrieve` loops all four non-skipped groups |
| Cold persistence | In-process DuckDB paging | JSON persistence by default |
| Logical/physical optimizer | Not specified | Absent |

Current JSON persistence is explicit in
[`_persistence.py:149-188`](../src/mandol/application/services/_persistence.py)
and [`json_persistence.py:1-6`](../src/mandol/infrastructure/json_persistence.py).
Therefore the README's high-level DuckDB description should not be assumed to
describe this branch.

## 2. Are relation filters and joins unified?

### Short answer: filters partially; joins no

Current metadata filtering is a list interpreter over candidate
`MemoryUnit`s. It supports nested fields and `eq`, `neq`, `in`, `contains`,
`gt`, `lt`, `gte`, and `lte`, with conjunction semantics
([`semantic_map.py:603-692`](../src/mandol/application/semantic_map.py)).
Memory spaces can be combined by UID-set union or intersection
([`semantic_map.py:410-417`](../src/mandol/application/semantic_map.py)).
Graph access can constrain one edge type and one direction
([`in_memory_graph_store.py:99-130`](../src/mandol/infrastructure/in_memory_graph_store.py)),
and BFS accepts a relationship type
([`semantic_graph.py:251-318`](../src/mandol/application/semantic_graph.py)).

These are independent procedural filters. There is no:

- query AST representing all predicates;
- `Join` or `SemiJoin` logical operator;
- edge relation exposed as rows `(src_uid, dst_uid, type, properties)`;
- predicate pushdown across unit, vector, and graph operators;
- join-order or traversal-order enumeration;
- physical hash, index nested-loop, or adjacency join choice;
- cardinality/selectivity estimation;
- optimizer or `EXPLAIN`.

The public `SemanticMapService.search` calls one or more retrievers, performs
RRF in Python, and applies `candidate_uids` only after retrieval
([`semantic_map.py:694-804`](../src/mandol/application/semantic_map.py)).
That is not a relational join.

There is also a concrete correctness/performance issue. `search_by_vector`
asks the global index for exactly `top_k` and only then discards UIDs outside
the requested spaces
([`semantic_map.py:419-457`](../src/mandol/application/semantic_map.py)).
FAISS space search merely over-fetches by a fixed factor of three before
filtering ([`faiss_vector_index.py:130-153`](../src/mandol/infrastructure/faiss_vector_index.py)).
Both can under-return even when at least `k` eligible records exist.

This is why the first optimizer milestone should be **semantic correctness for
filtered top-k**, not a sophisticated join enumerator.

## 3. What “vector-seeded graph traversal” means

The Experience Graph paper's query (PDF pages 6-7) is more than Mandol's existing
“vector hit, then BFS”:

1. run ANN over task-description embeddings;
2. follow relational links from matching tasks to sessions;
3. follow session-to-node links;
4. filter for non-buggy, policy-visible nodes;
5. select high-fitness nodes;
6. expand each selected node through a variable-length optimization trajectory.

Its illustrative logical request combines vector similarity, relation joins,
structured/policy filters, ordering by vector score and fitness, and graph
traversal. Trellis says its implementation uses Axiom over Velox to plan SQL,
Cypher, and vector fragments into a common physical plan. Section 7
simultaneously identifies the missing research piece: existing cost models do
not capture the chain

`vector selectivity -> relation fan-out -> traversal cost`

or maintain cross-modal statistics for it.

There is no contradiction: a generic CBO can produce a physical plan without
having a calibrated, modality-aware cost model for every cross-modal
alternative.

The paper's early measurements (PDF page 8) also make reuse policy part of the design:
memory reached a 1.2x speedup in about five evaluated steps versus 51 for the
cold baseline and reduced buggy-node rate from 55% to 34% (`p=0.1`) and 21%
(`p=0.5`). However, `p=0.5` collapsed strategy diversity and the best cold
solution remained better. Therefore `reuse_probability`, diversity limits,
and provenance must not be hidden inside the database operator.

### What Mandol already has

`SubgraphHopRetriever` is a useful baseline. It takes hybrid retrieval hits as
seeds, performs typed, weighted, multi-hop traversal, retains a reasoning path,
and combines normalized retrieval and graph scores
([`subgraph_hop.py:16-28`](../src/mandol/retrieval/subgraph_hop.py),
[`subgraph_hop.py:69-136`](../src/mandol/retrieval/subgraph_hop.py),
[`subgraph_hop.py:138-205`](../src/mandol/retrieval/subgraph_hop.py)).

It is not the Experience Graph operation because it does not express or
optimize task-session-node joins, per-group highest-fitness selection,
structured/policy predicates, or alternative orders. It should be retained as
the fixed-plan baseline and eventually implemented as one physical plan.

## 4. Target query architecture

### 4.1 Logical algebra

Add a small internal query package:

```text
src/mandol/query/
    expressions.py    typed fields and predicate AST
    logical.py        immutable logical operators
    physical.py       executable physical operators
    properties.py     ordering, cardinality, exactness, provenance
    capabilities.py   backend capability discovery
    rules.py          equivalence and implementation rules
    stats.py          statistics catalog and feedback
    cost.py           cardinality and cost models
    optimizer.py      bounded enumerator, later memo optimizer
    executor.py       pull/iterator execution and instrumentation
    explain.py        logical/physical/actual plan rendering
```

Minimum logical operators:

- `UnitScan(spaces, recursive)`
- `UidLookup(uids)`
- `Filter(input, predicate)`
- `VectorTopK(input, query_vector, k, metric, recall_target)`
- `TextTopK(input, query, k, method)`
- `Fuse(inputs, method="rrf")`
- `EdgeScan(type, direction, edge_predicate)`
- `Expand(input, edge_pattern, min_hops, max_hops, uniqueness)`
- `Join(left, right, condition, kind)`
- `TopK(input, order, k, partition_by=None)`
- `Rerank(input, query, k)`
- `Project` and `Fetch`

Treat an edge as a first-class relation:

```text
Edge(src_uid, dst_uid, rel_type, properties)
SpaceMember(space_name, uid)
Unit(uid, raw_data, metadata, embedding, sparse_embedding)
```

This makes graph expansion an optimized sequence of adjacency joins while
allowing a specialized traversal executor to preserve paths and avoid
materializing all intermediate rows.

### 4.2 Query API

Do not start with a SQL/Cypher parser. Add a typed Python builder and compile
existing APIs into it:

```python
query = (
    Query.from_space("tasks")
    .where(field("metadata.kind") == "task")
    .vector_top_k(text=new_task, k=20, recall_target=0.95)
    .expand(rel_type="BELONGS_TO", direction="in", hops=1)
    .expand(rel_type="IN_SESSION", direction="in", hops=1)
    .where(
        (field("metadata.is_buggy") == False)
        & policy_allows(user)
    )
    .top_k(
        k=1,
        partition_by="metadata.task_id",
        order_by="-metadata.fitness_score",
    )
    .expand(rel_type="HAS_CHILD", direction="out", min_hops=0, max_hops=8)
    .limit(10)
)
```

Return a richer result:

```python
QueryResult(
    rows=[...],
    complete=True,
    approximate=True,
    estimated_recall=0.96,
    plan=...,
    metrics=...,
)
```

`complete=False` is essential when an ANN or traversal budget is exhausted
before producing `k` qualifying results.

### 4.3 Physical alternatives

The first optimizer need only consider a bounded, useful set:

- metadata/space scan -> exact vector distances -> top-k;
- ANN -> iterative post-filter -> top-k;
- backend-native inline filtered ANN;
- vector seeds -> adjacency expansion -> filter;
- vector seeds -> filter -> adjacency expansion;
- selective structural filter/join -> exact or ANN ranking;
- adjacency expansion -> vector rank;
- hash/semi-join for materialized UID sets;
- indexed nested-loop/adjacency join for small seed sets;
- fixed dense/BM25/sparse parallel recall -> RRF;
- rerank only after a bounded candidate-producing operator.

Backends should advertise capabilities rather than grow the existing abstract
base classes immediately. Optional protocols avoid breaking adapters while
`main` is still refactoring:

```python
VectorCapabilities(
    candidate_filter="exact" | "inline" | "post" | "none",
    paged_search=True,
    range_search=False,
    tunable_effort=True,
)

GraphCapabilities(
    batched_neighbors=True,
    edge_property_filter=False,
    variable_length_traversal=False,
)
```

## 5. Cardinality and cost model

### 5.1 Statistics to collect

Unit/KV statistics:

- total units and units per space/type;
- null fraction, distinct count, top-N values, and equi-depth histograms for
  frequently filtered metadata paths;
- average serialized unit size and lookup/scan latency.

Vector statistics:

- index backend, dimension, indexed count, build age, and memory;
- latency and distance-comparison curves by requested `k` and search effort;
- empirical recall against sampled exact searches;
- threshold/selectivity curves for similarity values;
- conditional filter survival among ANN ranks,
  `P(filter | ANN rank <= r)`;
- correlation buckets for common `(space/type/predicate, vector-query class)`
  combinations.

Graph/relation statistics:

- edge counts by `(rel_type, direction, source kind, target kind)`;
- degree histograms and p50/p90/p99;
- distinct source and destination UIDs;
- one- and two-hop expansion factors with deduplication;
- cycle/duplicate rate and average property size.

Execution feedback:

- estimated and actual rows per operator;
- wall/CPU time, peak memory, distance comparisons, adjacency visits;
- ANN survivors per page/probe;
- cache residency and remote-call counts for external adapters.

Use sampled `ANALYZE` plus exponentially weighted runtime feedback. Store a
schema/version fingerprint with statistics so stale estimates are detectable.

### 5.2 Cardinality estimates

Let:

- `N` be input units;
- `s_f` be structured-filter selectivity;
- `k` be requested neighbors;
- `p_r = P(filter | ANN rank <= r)` be observed filter survival;
- `b_t` be mean fan-out for edge type `t`;
- `d_h` be the deduplication factor through hop `h`;
- `q` be policy-filter survival.

Then:

```text
filtered rows       = N * s_f
ANN rows inspected  ~= min(N, k / max(p_r, epsilon)) * recall_effort
hop-1 output        ~= seeds * b_t * q * d_1
hop-h output        ~= previous * b_t(h) * q(h) * d_h
```

Do not assume predicate/vector independence when runtime feedback can estimate
`p_r` directly. Negative correlation is precisely where fixed ANN over-fetch
fails.

For multi-edge patterns, estimate joins using per-kind distinct counts and
degree distributions, then clamp estimates by known endpoint populations.
Use p90/p99 fan-out for memory-risk guards even if expected latency uses the
mean.

### 5.3 Cost objective

A practical first objective is:

```text
cost(plan) =
    wall_time
  + alpha * cpu_time
  + beta  * peak_memory
  + gamma * remote_round_trips
  + delta * model_rerank_items
  + infinite_penalty(recall < recall_target)
  + infinite_penalty(policy violation)
```

The calibrated operator models include:

```text
exact_filtered_vector =
    filter_scan(N) + distance(filtered_rows, dimension) + topk(filtered_rows)

iterative_ann_filter =
    ann(probes/pages, k) + predicate(rows_seen) + fetch(survivors)

adjacency_expand =
    neighbor_lookups(frontier) + edge_filter(edges_seen)
    + dedup(intermediate_rows) + path_materialization(output_rows)
```

Initially fit robust piecewise-linear regressions from microbenchmarks. A
simple empirical model will be more reliable than hand-tuned constants.

### 5.4 Required physical properties

Cost alone cannot define equivalence. Track:

- exact versus approximate result semantics;
- ordering and score domain;
- expected recall and search budget;
- path/provenance preservation;
- UID uniqueness;
- snapshot/version;
- policy scope;
- materialized columns.

RRF scores, vector similarities, fitness scores, and cross-encoder scores are
not interchangeable. A rule may reorder operations only when it preserves the
logical ranking contract or explicitly consumes an approximation budget.

## 6. Optimizer search strategy

### Phase-one bounded enumerator

For each recognized query shape, instantiate five to ten legal templates,
estimate every operator bottom-up, reject plans that violate required
properties, and choose the cheapest. This is sufficient to test the hard
cross-modal estimates without implementing a general optimizer.

For `Filter + VectorTopK + Expand`, enumerate:

1. `Filter -> exact vector -> TopK -> Expand`
2. `Filter -> inline filtered ANN -> Expand`
3. `ANN iterative post-filter -> Expand`
4. `ANN -> Expand -> Filter` when the filter applies to expanded nodes
5. `selective graph join -> vector rank -> Expand`
6. parallel lexical/vector seeds -> RRF -> Expand

The logical predicate's target determines legality: a predicate over task seeds
can be pushed below traversal; a predicate over descendant nodes cannot.

### Cascades-lite evolution

After template estimates are accurate:

- represent equivalent logical expressions in memo groups;
- add transformation rules for filter pushdown, join commutation/association,
  top-k pushdown under proven conditions, and expand/join conversion;
- add implementation rules for each backend capability;
- retain the cheapest expression per required physical-property set;
- cap search by group count and planning-time budget;
- cache plans using normalized shape plus statistics/capability versions.

Re-optimize at safe pipeline breakers when actual cardinality differs from the
estimate by, for example, 4x. An iterative ANN operator is a natural adaptive
boundary: it can request another page when survivor count is too low.

## 7. Implementing Experience Graph retrieval in Mandol

### 7.1 Data representation

Mandol has no first-class Task, Session, or ExplorationNode type. The least
invasive first version uses `MemoryUnit` plus typed metadata and edges:

```text
Task unit:
  metadata.kind = "task"
  raw_data.description
  embedding = task-description embedding

Session unit:
  metadata.kind = "session"
  metadata.search_algorithm, owner, timestamps

ExplorationNode unit:
  metadata.kind = "exploration_node"
  metadata.fitness_score, is_buggy, generation, policy_scope
  raw_data.analysis, artifact refs, evaluator output

Edges:
  session -[BELONGS_TO]-> task
  node    -[IN_SESSION]-> session
  parent  -[HAS_CHILD]-> child
```

This reuses the canonical UID and current graph API. Add a validation layer
for required fields, then consider dedicated domain classes only after query
semantics stabilize.

NetworkX `DiGraph` permits only one edge between a source and target, so
multiple relationship types between the same pair overwrite one another.
If experience graphs require multi-edges, migrate the in-memory backend to
`MultiDiGraph` or encode edge identity explicitly before relying on it.

### 7.2 Logical plan

Compile the paper's operation to:

```text
VectorTopK(
  Filter(UnitScan(tasks), task predicate),
  new_task_embedding,
  seed_k
)
  -> Expand(BELONGS_TO, in, 1)
  -> Expand(IN_SESSION, in, 1)
  -> Filter(not is_buggy AND policy_allows)
  -> TopK(partition_by=task_uid, order_by=fitness DESC, k=per_task)
  -> Expand(HAS_CHILD, out, 0..max_hops, path predicates)
  -> TopK(order_by=(task_similarity, fitness, path_score), k=result_k)
```

Every output should carry:

- seed task UID and vector score;
- selected session/node UIDs and fitness;
- complete edge path with directions/types;
- policy decision/version;
- approximation and truncation flags.

### 7.3 Executor details

- Add `get_neighbors_batch(uids, rel_types, direction)` to eliminate one
  application call per node. Provide a default loop implementation and an
  optimized NetworkX implementation.
- Use a frontier iterator rather than materializing every path.
- Make uniqueness explicit: node-global, edge-global, or path-local.
- Keep `(uid, depth, path-state)` when path predicates are depth-sensitive.
- Support per-seed and global expansion budgets, deterministic tie-breaking,
  cancellation, and a result-completeness flag.
- Preserve the existing `SubgraphHopRetriever` scoring as a baseline physical
  operator, then expose its traversal weights through the new plan.
- Apply access/policy predicates before any unauthorized payload is fetched or
  returned; treat them as mandatory barriers rather than freely reorderable
  performance filters.
- Keep reuse injection outside the storage operator:
  `ReusePolicy(probability, max_sources, diversity_lambda, freshness, trust)`.
  This makes the exploration/anchoring tradeoff observable and tunable.

### 7.4 First CBO decisions for this query

The useful alternatives are:

- small/selective task set: structured pre-filter then exact vector ranking;
- broad task set: ANN with inline or iterative filtering;
- few vector seeds: adjacency nested-loop into sessions/nodes;
- many vector seeds: materialize seed UIDs and hash/semi-join against edge rows;
- selective non-buggy/policy condition: push into node access before path
  expansion;
- high-degree selected node: bounded best-first/beam expansion;
- low-degree node: BFS/DFS frontier traversal.

Estimate total cost end-to-end. Choosing ANN solely because it is cheap can
create an enormous task-to-session-to-node fan-out and lose to a more selective
seed plan.

## 8. Correctness invariants

1. **Filtered top-k:** exact semantics are “filter the eligible population,
   rank it, return k.” Fixed post-filtering of k global ANN hits is not
   equivalent.
2. **Approximation contract:** every approximate operator declares recall
   target, effort limit, and whether the returned set is complete.
3. **Safe top-k pushdown:** never push `TopK` through a filter, join, expansion,
   or score transformation unless a rule proves order preservation.
4. **UID integrity:** every vector hit and edge endpoint resolves to the same
   canonical unit version or is reported as dangling/corrupt.
5. **Graph semantics:** direction, edge types, hop bounds, uniqueness, path
   predicates, and deterministic ties are explicit.
6. **Policy safety:** access filters are mandatory barriers and cannot be
   weakened by reordering, caches, or approximate search.
7. **Snapshot consistency:** a query sees a defined unit/vector/edge version;
   concurrent writes cannot mix incompatible states silently.
8. **Provenance:** vector seed, joins, selected node, and traversal path remain
   reconstructible in the result.
9. **Score semantics:** fusion and reranking consume named scores; missing
   scores do not silently default into a different scale.
10. **Resource bounds:** traversal, ANN effort, reranker candidates, and path
    materialization have enforced limits and visible truncation.

## 9. Phased implementation plan

### Phase 0 — baselines and semantic contracts (1-2 weeks)

- Freeze representative current queries and `SubgraphHopRetriever` results.
- Add exhaustive filtered-vector and reference graph traversal implementations.
- Add randomized/property tests for filters, top-k, cycles, directions, and
  dangling UIDs.
- Instrument existing operators for rows, time, distance comparisons, and edge
  visits.
- Document exact/approximate behavior and result-completeness semantics.

Exit: a golden reference can detect the current under-return behavior and
validate future plans.

### Phase 1 — logical plan and fixed physical execution (2-3 weeks)

- Add expression/query AST and logical operators.
- Add physical wrappers over current stores/retrievers.
- Compile existing `SemanticMapService.search`, `HybridRetriever`, and
  `SubgraphHopRetriever` calls into fixed equivalent plans behind a feature
  flag.
- Add `EXPLAIN` with logical and physical trees.

Exit: existing APIs produce equivalent results through the new executor.

### Phase 2 — statistics and `EXPLAIN ANALYZE` (2 weeks)

- Implement sampled statistics catalog and backend capabilities.
- Record estimated/actual rows, latency, memory, ANN pages, and graph visits.
- Build microbenchmarks and fit initial operator cost curves.

Exit: estimates are inspectable and persisted with schema/index versions.

### Phase 3 — filtered-vector optimizer (2-3 weeks)

- Implement exact prefilter, iterative ANN postfilter, and inline-filter
  alternatives when supported.
- Add cardinality, correlation, and recall-aware costing.
- Enumerate bounded filter/vector plan templates.

Exit: no silent under-return; selected plan is competitive with the best
enumerated alternative.

### Phase 4 — joins and vector-seeded traversal (3-5 weeks)

- Expose `Edge` and `SpaceMember` relations.
- Add batch adjacency, semi/hash joins, path-preserving traversal, and
  partitioned top-k by fitness.
- Add Experience Graph schema validator and query builder.
- Compile `SubgraphHopRetriever` as the baseline plan and add alternative
  vector/filter/join/traversal orders.

Exit: the Experience Graph reference query executes as one logical request,
with exact provenance and optimizer-selected physical plan.

### Phase 5 — memo optimization and adaptive execution (3-4 weeks)

- Add memo groups, transformation/implementation rules, property enforcement,
  planning budget, and plan cache.
- Add runtime cardinality feedback and safe re-optimization boundaries.
- Add robust fan-out and stale-statistics guards.

Exit: larger query shapes remain tractable and recover from material estimate
errors.

### Phase 6 — backend pushdown and scale (later)

- Add capability-specific Milvus/Neo4j/DuckDB or other embedded backends.
- Push filters, joins, and traversal only where their semantics are equivalent.
- Evaluate a columnar executor or Axiom/Velox integration only if Python
  execution becomes the measured bottleneck.

## 10. Experiment and benchmark plan

### Workloads

Generate a factorial synthetic dataset varying:

- units: `10^3`, `10^4`, `10^5`;
- structured selectivity: `0.001`, `0.01`, `0.1`, `0.5`, `1.0`;
- vector/filter correlation: positive, independent, negative;
- seed `k`: `5`, `10`, `50`;
- relationship fan-out: `1`, `4`, `16`, `64`;
- hops: `1` through `4`;
- graph shape: tree, DAG with merging, cyclic, power-law;
- hot versus cold/cache-miss execution.

Add a synthetic Experience Graph dataset with tasks, sessions, fitness,
bugginess, policies, and known root-to-leaf trajectories. Use existing
LoCoMo/LongMemEval workflows for end-to-end retrieval-quality non-regression,
but do not use QA accuracy as the only optimizer metric.

### Baselines

- current fixed `HybridRetriever`;
- current `SubgraphHopRetriever`;
- exact filter -> vector -> traversal oracle;
- fixed ANN -> filter -> traversal;
- every legal template for small query shapes;
- optimizer choice.

### Metrics

- p50/p95/p99 wall latency and optimizer overhead;
- CPU, peak RSS, distance computations, index pages/probes;
- adjacency visits, intermediate rows, dedup rate;
- result count and incomplete-result rate;
- Recall@k against exhaustive filtered top-k;
- graph node/path recall and provenance correctness;
- chosen-plan cost relative to the measured best legal plan;
- reranker calls/items and remote round trips;
- end-to-end QA quality and token count.

### Proposed acceptance criteria

These thresholds should be calibrated on the target deployment, but are useful
initial gates:

- exact plans match reference UID sets and paths across all randomized tests;
- filtered ANN Recall@10 is at least 0.95 when requested, with no silent
  under-return when enough eligible rows exist within the configured budget;
- every budget exhaustion reports `complete=False`;
- optimizer median latency is within 20% of the measured oracle plan in at
  least 90% of benchmark cells;
- no benchmark cell has a p95 regression greater than 2x versus the best fixed
  baseline without an explicit quality improvement;
- p95 planning overhead is below 2 ms for the bounded enumerator;
- estimated cardinality is within 4x of actual for at least 90% of operators
  after feedback warm-up;
- vector-seeded results exactly preserve task/session/node/path provenance and
  policy visibility;
- current tests and retrieval-quality benchmarks remain green/non-inferior.

Use repeated runs, report confidence intervals, and retain raw per-query
measurements. Plan-quality comparisons must execute every legal plan on the
same query/data snapshot.

## 11. Principal risks and mitigations

| Risk | Consequence | Mitigation |
|---|---|---|
| Treating approximate plans as relationally equivalent | Wrong or short top-k | Physical exactness/recall properties and completeness flag |
| Vector/filter correlation | Severe cardinality error | Conditional survivor statistics and iterative execution |
| Heavy-tailed graph degree | Traversal explosion | Degree quantiles, robust cost, budgets, adaptive breakers |
| Stale cross-store state | Dangling or mismatched UIDs | Snapshot/version contract and integrity checks |
| NetworkX single-edge semantics | Lost relationship types | `MultiDiGraph` or explicit edge IDs |
| Backend capability mismatch | Incorrect pushdown | Capability protocols and semantic conformance tests |
| Reuse anchoring | Less exploration/diversity | External quality-aware reuse policy and MMR/diversity caps |
| Overbuilding the optimizer | Long delay before evidence | Bounded templates before memo/Cascades |
| Python overhead dominates | Misleading CBO gains | Instrument first; optimize/batch executor after measurement |
| Paper/main divergence | Designing against absent components | Treat current `main` ports as source of truth |

## 12. Recommended first pull requests

1. **Result semantics and references**
   - `QueryResult.complete/approximate/estimated_recall`
   - exhaustive filtered-vector reference implementation
   - regression test demonstrating current fixed-overfetch under-return

2. **Plan IR and explain**
   - typed predicate AST
   - `UnitScan`, `Filter`, `VectorTopK`, `Expand`, `TopK`
   - fixed physical compilation of current vector-seeded BFS

3. **Instrumentation and statistics**
   - operator counters/timers
   - space/metadata/edge histograms
   - `EXPLAIN ANALYZE`

4. **Filtered-vector plan choice**
   - exact prefilter versus iterative postfilter
   - bounded enumerator and calibrated cost

5. **Experience Graph query**
   - generic schema validator
   - batched adjacency and path provenance
   - task-session-node joins and per-task fitness top-k

This order yields a correct, measurable optimizer nucleus before committing to
a general-purpose planner.

## External research used

- [Axiom](https://github.com/facebookincubator/axiom) demonstrates the useful
  parser/logical-plan/query-graph/physical-plan separation and
  `EXPLAIN ANALYZE`, but is not a drop-in fit for Mandol's Python execution
  layer.
- [Filtered Vector Search: State-of-the-art and Research
  Opportunities](https://research.google/pubs/filtered-vector-search-state-of-the-art-and-research-opportunities/)
  identifies pre-filtering, post-filtering, and inline filtering, and explains
  why selectivity, cardinality, and vector/filter correlation control the best
  strategy.
- [pgvector](https://github.com/pgvector/pgvector#filtering) documents the
  practical under-return problem of post-filtered ANN and uses iterative index
  scans to continue until enough results or a scan budget is reached.
- [ACORN](https://arxiv.org/abs/2403.04871) is a useful future inline-filter
  alternative based on predicate-subgraph traversal in HNSW.
- [VBASE](https://www.usenix.org/conference/osdi23/presentation/zhang-qianxi)
  motivates treating vector similarity as a database operator that can compose
  with relational operations rather than materializing an arbitrary tentative
  top-k first.

## Material Passport

- Origin Skill: `academic-research-suite/experiment-agent`
- Origin Mode: `plan`
- Origin Date: `2026-07-23`
- Verification Status: `papers-read; current-main-code-inspected; external-primary-sources-checked; tests-not-run-pytest-missing`
- Repository Commit: `8c795ca`
- Local Artifacts:
  - `papers/2606.29778.pdf`
  - `papers/2606.29778.txt`
  - `papers/2606.29823.pdf`
  - `papers/2606.29823.txt`
