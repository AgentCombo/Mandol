# Relation joins for vector-seeded graph traversal

## Material Passport

- Origin Skill: `academic-research-suite/experiment-agent`
- Origin Mode: `plan`
- Origin Date: `2026-07-23`
- Verification Status: `current-main-code-inspected; official-database-docs-checked; implementation-not-started`
- Version Label: `relation_join_plan_v2`
- Repository Commit: `8c795ca`

## Outcome

Implement one narrow relational primitive before building the cost optimizer:

```text
vector seeds
    -> typed adjacency join
    -> typed adjacency join
    -> node/policy filter
    -> per-group top-k
    -> path-preserving graph traversal
```

For the Experience Graph query this becomes:

```text
Task vector seeds
    -> Session -[BELONGS_TO]-> Task
    -> Node -[IN_SESSION]-> Session
    -> non-buggy/policy-visible nodes
    -> highest-fitness nodes per task
    -> Node -[HAS_CHILD*0..h]-> Node
```

This plan deliberately does not implement a general SQL/Cypher engine. It
creates stable row, edge, join, and traversal contracts that the later
optimizer can reorder and cost.

The recommended primary backend is a **single SQL database containing units,
vectors, spaces, and relationships**. For Mandol's embedded deployment this is
DuckDB; PostgreSQL with pgvector is the production/multi-process alternative.
Neo4j is not required. The batched `GraphStore` execution described below is a
portable reference/fallback, while a unified SQL backend should compile the
same semantics into one statement and one transaction.

## Why the existing BFS is insufficient

Mandol already converts hybrid/vector hits into graph seeds:

- `HybridRetriever` fuses retrieval results and sends the leading units to
  `bfs_expand_units`
  ([`pipeline.py:181-226`](../src/mandol/retrieval/pipeline.py)).
- `SubgraphHopRetriever` performs typed, weighted traversal and retains one
  best reasoning path
  ([`subgraph_hop.py:138-205`](../src/mandol/retrieval/subgraph_hop.py)).

BFS returns a set/list of units. A relation join must instead return bindings:

```text
{task=t1, session=s7, node=n42, task_score=0.91, path=[...]}
```

Bindings are necessary because:

- the same node may be reached from multiple tasks or sessions;
- fitness must be ordered within each task or session;
- filters may refer to different aliases;
- downstream traversal must preserve the exact task/session/node provenance;
- join multiplicity must be retained until an explicit distinct/group
  operator removes it.

Therefore relation join should not be implemented by extending
`bfs_expand_units` with more flags.

## Scope

### In scope

- typed edge records;
- batched edge access;
- inner adjacency joins over UIDs;
- node and edge predicates;
- binding and score propagation;
- partitioned top-k;
- bounded path expansion;
- exact reference vector seeding;
- execution metrics and result completeness;
- a convenience API for the Experience Graph pattern.

### Deferred until the optimizer

- arbitrary value-based or Cartesian joins;
- join reordering;
- physical hash-join selection;
- SQL or Cypher parsing;
- ANN versus exact plan selection;
- cross-modal cardinality estimation;
- backend-specific pushdown beyond batched edge lookup.

## Storage decision: Neo4j is optional, not on the critical path

There are four useful levels of “unified”:

1. one canonical UID/schema;
2. one transactional source of truth;
3. one query statement and optimizer;
4. one in-process address space.

A DuckDB backend can provide all four. PostgreSQL with pgvector provides the
first three but runs as a separate server process. A configuration that uses
Neo4j for edges and FAISS/Milvus for vectors provides only the first level
unless Mandol builds a federated planner and consistency protocol; that is the
fragmented architecture the paper intends to avoid.

Therefore:

- do not make the current Neo4j adapter a dependency of relation join;
- keep it as an explicitly federated/optional backend;
- do not block the initial PRs on Neo4j conformance;
- implement `DuckDBUnifiedStore` first, with exact vector ranking as the
  correctness path and optional VSS acceleration;
- offer `PostgresUnifiedStore` later for concurrent multi-process serving.

The `paper-repro` branch already contains the core proof of feasibility in
`src/mandol/storage/duckdb_operator.py`: one `memory_nodes` table stores JSON
and dense vectors, one `memory_edges` table stores typed relationships, VSS is
loaded for HNSW, and graph traversal uses DuckPGQ when available or a recursive
CTE fallback. The new implementation should port the model and tests, not copy
the old class wholesale.

Expose one physical backend through compatibility facades:

```python
backend = DuckDBUnifiedStore("mandol.duckdb", embedding_dim=1024)

system = MemorySystem(
    unit_store=backend.units,
    vector_index=backend.vectors,
    graph_store=backend.graph,
)

retriever = VectorSeededGraphRetriever(query_store=backend.queries)
```

All four facades share one connection/transaction and the query facade can
execute vector seeding, joins, filters, and recursion without returning to
Python between stages.

## 1. Establish the relation model

### 1.1 Make relationships first-class values

Add `src/mandol/domain/relationship.py`:

```python
@dataclass(frozen=True, slots=True)
class Relationship:
    source_uid: Uid
    target_uid: Uid
    rel_type: str
    properties: Mapping[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple[Uid, Uid, str]:
        return (self.source_uid, self.target_uid, self.rel_type)
```

The logical edge relation is:

```text
Relationship(source_uid, target_uid, rel_type, properties)
```

Its key is already implied by the existing `RelationshipKey =
(source, target, type)` declaration
([`graph_store.py:15`](../src/mandol/ports/graph_store.py)).

Initial cardinality semantics:

- one stored relationship per `(source_uid, target_uid, rel_type)`;
- two different relation types between the same endpoints are distinct rows;
- upserting the same key replaces its properties;
- join output has bag semantics—one result for every input-row/edge match;
- `Distinct` or `GroupTopK`, never the join itself, removes duplicates.

### 1.2 Correct the in-memory graph representation first

`InMemoryGraphStore` currently uses `networkx.DiGraph`
([`in_memory_graph_store.py:17-31`](../src/mandol/infrastructure/in_memory_graph_store.py)).
It can store only one edge between a source and target, so adding a second
relationship type overwrites the first. That conflicts with the declared
relationship key.

Change it to:

```python
nx.MultiDiGraph()
graph.add_edge(source, target, key=rel_type, type=rel_type, **properties)
```

Keep legacy `get_neighbors()` behavior UID-oriented and deduplicated, so
existing BFS callers do not unexpectedly see the same neighbor twice. New
relationship-reading methods retain every typed edge.

Required migration tests:

- two relation types between the same UIDs survive;
- updating one type does not change the other;
- deleting one type preserves the other;
- deleting with `rel_type=None` deletes all endpoint-pair relationships;
- JSON save/load and `get_all_edges()` preserve all typed edges.

## 2. Add a batched relationship-reader contract

Add non-abstract default methods to `GraphStore`, avoiding an immediate break
for every adapter:

```python
def get_relationships_batch(
    self,
    uids: Sequence[Uid],
    *,
    direction: Literal["out", "in"],
    rel_types: Optional[Collection[str]] = None,
) -> Mapping[Uid, Sequence[Relationship]]:
    ...
```

Semantics:

- `out`: map each requested UID to edges whose `source_uid` equals it;
- `in`: map each requested UID to edges whose `target_uid` equals it;
- return typed edges and properties, not only neighboring UIDs;
- preserve edge multiplicity and deterministic edge-key ordering;
- missing UIDs map to an empty sequence;
- reject `both` initially—callers can explicitly union `in` and `out`.

The default implementation may scan `get_all_edges()` once per batch. Override
it in:

- `InMemoryGraphStore` using `MultiDiGraph` adjacency in
  `O(sum(degree(seed)))`;
- `Neo4jGraphStore` using one `UNWIND $uids ... MATCH` query.

The current Neo4j adapter does not implement the abstract `get_all_edges()` or
`clear()` methods, so it is not presently a complete `GraphStore`
([`neo4j_graph_store.py:18-164`](../src/mandol/infrastructure/neo4j_graph_store.py)).
Complete it only when the optional federated backend is deliberately enabled;
it is not an initial milestone.

Do not reconstruct edge type/properties by calling `get_neighbors()` followed
by `get_relationship()` for each neighbor. That introduces an N+1 query pattern
and cannot recover all types when `rel_type=None`; current service code already
has to scan all edges to recover an unknown edge type
([`semantic_graph.py:520-544`](../src/mandol/application/semantic_graph.py)).

## 3. Define the intermediate row contract

Add `src/mandol/query/rows.py`:

```python
@dataclass(frozen=True, slots=True)
class BindingRow:
    bindings: Mapping[str, Uid]
    scores: Mapping[str, float] = field(default_factory=dict)
    path: tuple[Relationship, ...] = ()
    attributes: Mapping[str, Any] = field(default_factory=dict)
```

Rules:

- aliases are unique within one plan;
- rebinding an existing alias is legal only if the UID is identical;
- vector scores use a named field such as `task_similarity`;
- every adjacency join appends the matched relationship to `path`;
- operators copy-on-write rather than mutating upstream rows;
- stable ties use UID and relationship key;
- raw unit payloads are fetched only when required by projection/reranking.

The row deliberately carries UIDs rather than `MemoryUnit` objects. This
reduces repeated object copying and lets node fetches be batched.

## 4. Implement one physical join: `AdjacencyJoin`

Add `src/mandol/query/operators/adjacency_join.py`:

```python
@dataclass(frozen=True, slots=True)
class AdjacencyJoinSpec:
    input_alias: str
    output_alias: str
    direction: Literal["out", "in"]
    rel_types: tuple[str, ...]
    edge_predicate: Optional[Predicate] = None
    node_predicate: Optional[Predicate] = None
    dangling: Literal["skip", "error"] = "skip"
    batch_size: int = 256
```

Join condition:

```text
direction="out":
  input.bindings[input_alias] = edge.source_uid
  output_alias = edge.target_uid

direction="in":
  input.bindings[input_alias] = edge.target_uid
  output_alias = edge.source_uid
```

Execution:

1. consume at most `batch_size` input rows;
2. deduplicate lookup UIDs only for the storage call, not the logical rows;
3. fetch matching typed relationships in one batch;
4. collect unique output UIDs and fetch their units in one `get_units()` call;
5. evaluate edge and output-node predicates;
6. emit one extended `BindingRow` per matching edge;
7. preserve input order, then deterministic edge order;
8. record actual input rows, lookup UIDs, edges visited, units fetched, output
   rows, and elapsed time.

Start with inner joins only. A semi-join can be added later for existence
filters; left/full outer joins are unnecessary for vector-seeded traversal.

### Predicate representation

Reuse the semantics of `filter_memory_units`, but extract its nested-field
resolution and comparison operators into typed expression objects rather than
passing unvalidated dictionaries
([`semantic_map.py:603-692`](../src/mandol/application/semantic_map.py)).

Minimum predicates:

- `Field(alias, "metadata.kind") == "session"`;
- numeric comparison on fitness;
- boolean `and`, `or`, `not`;
- edge property fields;
- a non-optional policy predicate hook.

Policy predicates execute before downstream payload projection and are marked
as mandatory barriers. They must never be skipped because of an unsupported
backend pushdown.

## 5. Add the non-join operators required by the pattern

### 5.1 Exact vector seed source

Implement an exact candidate-restricted seed source first:

```python
VectorSeed(
    alias="task",
    spaces=("experience/tasks",),
    predicate=Field("task", "metadata.kind") == "task",
    query_vector=...,
    k=20,
    score_name="task_similarity",
)
```

It must:

1. obtain eligible task UIDs from space and metadata predicates;
2. compute cosine similarity only over those eligible embeddings;
3. emit top-k `BindingRow`s with deterministic ties.

This avoids inheriting current fixed post-filter behavior, which can
under-return eligible results
([`semantic_map.py:419-457`](../src/mandol/application/semantic_map.py),
[`faiss_vector_index.py:130-153`](../src/mandol/infrastructure/faiss_vector_index.py)).

The exact implementation is the correctness oracle. The optimizer can later
replace it with inline-filtered or iterative ANN without changing row
semantics.

### 5.2 Node filter

`FilterRows` batches alias UIDs through `UnitStore.get_units()`, evaluates the
predicate, and preserves bindings, scores, and paths.

### 5.3 Partitioned top-k

Implement:

```text
GroupTopK(
    partition_by=("task",),
    order_by=(
        fitness DESC,
        task_similarity DESC,
        node UID ASC,
    ),
    k=per_task,
)
```

This is what turns all non-buggy nodes from matching sessions into the
highest-fitness reuse candidates for each vector seed. Do not use a global
top-k here; that could let one task consume the entire result budget.

### 5.4 Path expansion

Implement `PathExpand` separately from relation join:

```python
PathExpand(
    input_alias="node",
    output_alias="trajectory_node",
    rel_types=("HAS_CHILD",),
    direction="out",
    min_hops=0,
    max_hops=8,
    uniqueness="node_global_per_seed",
    per_seed_limit=100,
    global_limit=1000,
)
```

It operates on `BindingRow`, preserves the task/session/selected-node aliases,
and appends every traversed edge. Its result reports `complete=False` if a hop,
per-seed, global, time, or memory budget truncates expansion.

## 6. Compose a fixed vector-seeded plan

Add `src/mandol/retrieval/vector_seeded_graph.py`:

```python
@dataclass(frozen=True, slots=True)
class VectorSeededTraversalSpec:
    seed: VectorSeedSpec
    joins: tuple[AdjacencyJoinSpec, ...]
    candidate_predicate: Predicate
    group_top_k: GroupTopKSpec
    traversal: PathExpandSpec
    final_k: int


class VectorSeededGraphRetriever:
    def search(
        self,
        query: str | Embedding,
        *,
        spec: VectorSeededTraversalSpec,
        policy_context: PolicyContext,
    ) -> QueryResult:
        ...
```

The Experience Graph convenience compiler produces:

```python
VectorSeed(alias="task", ...)
AdjacencyJoin("task", "session", direction="in",
              rel_types=("BELONGS_TO",))
AdjacencyJoin("session", "node", direction="in",
              rel_types=("IN_SESSION",))
FilterRows(alias="node",
           predicate=(is_buggy == False) & policy_allows(...))
GroupTopK(partition_by=("task",), order_by=fitness_desc, k=1)
PathExpand("node", "trajectory_node",
           direction="out", rel_types=("HAS_CHILD",), max_hops=...)
FinalTopK(order_by=(task_similarity_desc, fitness_desc), k=10)
```

### Result contract

```python
QueryResult(
    rows: list[BindingRow],
    complete: bool,
    approximate: bool,
    truncation_reasons: tuple[str, ...],
    metrics: QueryMetrics,
)
```

Every result can reconstruct:

- vector seed task and similarity;
- joined session and selected node;
- node fitness and filter/policy decision version;
- complete typed path from selected node to output node.

## 7. Package and dependency layout

```text
src/mandol/
  domain/
    relationship.py
  ports/
    graph_store.py                 batch relationship reads
    unified_query_store.py         one-statement multimodal query contract
  infrastructure/
    in_memory_graph_store.py       MultiDiGraph + batch override
    duckdb_unified_store.py        primary embedded backend
    postgres_unified_store.py      later production backend
    neo4j_graph_store.py           optional federated adapter
  query/
    __init__.py
    rows.py
    predicates.py
    result.py
    operators/
      vector_seed.py
      adjacency_join.py
      filter_rows.py
      group_topk.py
      path_expand.py
  retrieval/
    vector_seeded_graph.py
```

Keep these operators independent of `SemanticGraphService` orchestration.
They should depend on `UnitStore`, `VectorIndex`/embedder, and `GraphStore`
ports so the optimizer can instantiate them later without calling a
hard-coded retrieval pipeline.

## 8. Pull-request sequence

### PR 1 — edge semantics and storage conformance

- add `Relationship`;
- migrate in-memory graph to `MultiDiGraph`;
- preserve legacy neighbor behavior;
- complete typed-edge persistence tests;
- implement missing Neo4j `GraphStore` methods or explicitly keep that adapter
  unsupported.

Exit criterion: relationship-key semantics are consistent across CRUD,
neighbor reads, full scans, and persistence.

### PR 2 — batched relationship access

- add default `get_relationships_batch`;
- add optimized in-memory implementation;
- add adapter conformance tests and call counters;
- add deterministic ordering.

Exit criterion: batched results exactly equal a reference filter over
`get_all_edges()`.

In parallel within this PR, create the DuckDB relational schema and compatibility
facades. Neo4j work is explicitly excluded.

### PR 3 — binding rows, predicates, and adjacency join

- add `BindingRow`, typed predicates, and `AdjacencyJoin`;
- batch endpoint fetches;
- add execution counters;
- property-test against a naive nested-loop join.

Exit criterion: all randomized join outputs, including multiplicities and
paths, match the reference implementation.

### PR 4 — fixed vector-seeded traversal

- add exact `VectorSeed`, `FilterRows`, `GroupTopK`, and `PathExpand`;
- add `VectorSeededGraphRetriever`;
- compile the Experience Graph convenience query;
- compile the full operation to one parameterized DuckDB statement;
- retain current `SubgraphHopRetriever` as a behavioral/performance baseline.

Exit criterion: one API call executes vector seed -> two joins -> fitness
selection -> bounded trajectory expansion with full provenance.

### PR 5 — hardening and optimizer handoff

- add policy barriers, snapshot/version checks, cancellation, and resource
  limits;
- add `EXPLAIN ACTUAL`-style metrics;
- run synthetic fan-out/correlation benchmarks;
- document physical operator capabilities and measured cost curves.

Exit criterion: operators expose enough cardinality and runtime data for the
first cost-based plan enumerator.

## 9. Test plan

### Unit tests

- incoming and outgoing join direction;
- one and multiple relationship types;
- multiple typed edges between the same endpoint pair;
- self-loop and cycle handling;
- missing/dangling endpoint: skip and error modes;
- edge-property and node-property predicates;
- alias collision;
- empty input, empty relation, and no matches;
- input multiplicity preservation;
- deterministic ordering and tie-breaking;
- batch size boundaries;
- policy rejection before result projection;
- traversal limits and `complete=False`.

### Property tests

For random small `MultiDiGraph`s:

1. materialize the edge relation with `get_all_edges()`;
2. execute a naive nested-loop equi-join;
3. execute `AdjacencyJoin`;
4. compare full multisets of bindings and relationship keys.

For random traversal graphs, compare path outputs with a simple exhaustive
reference under the same direction, type, hop, uniqueness, and budget rules.

### Integration fixture

Create a deterministic graph:

```text
task_a <-BELONGS_TO- session_a1 <-IN_SESSION- node_a1 (fitness=.7)
                                      \------- node_a2 (fitness=.9)
task_a <-BELONGS_TO- session_a2 <-IN_SESSION- node_a3 (buggy, fitness=1.0)
task_b <-BELONGS_TO- session_b1 <-IN_SESSION- node_b1 (fitness=.8)

node_a2 -HAS_CHILD-> node_a2_1 -HAS_CHILD-> node_a2_2
node_b1 -HAS_CHILD-> node_b1_1
```

Use controlled task embeddings so `task_a` and `task_b` are known vector
seeds. Assert:

- the two joins retain task/session lineage;
- buggy `node_a3` is excluded;
- `node_a2` wins task A's partition;
- `node_b1` wins task B's partition;
- trajectory paths and vector scores are exact;
- a denied policy hides its node and descendants;
- the final global ordering follows the declared score tuple.

### Performance tests

Vary:

- seeds: 1, 10, 100;
- sessions per task: 1, 10, 100;
- nodes per session: 1, 10, 100;
- traversal fan-out: 1, 4, 16;
- hops: 1-4.

Record per operator:

- input/output rows;
- relationship batches and edges examined;
- unit batches and units fetched;
- p50/p95 latency;
- peak intermediate rows and memory;
- truncation/completeness.

## 10. Acceptance criteria

- `AdjacencyJoin` equals the naive edge-table join as a multiset for every
  randomized test.
- Multiple relation types between the same UIDs survive CRUD and persistence.
- The in-memory join performs at most
  `ceil(unique_input_uids / batch_size)` relationship-read calls per operator.
- Endpoint units are fetched once per unique UID per batch, not once per edge.
- Vector similarity, all aliases, and every traversed relationship remain in
  result provenance.
- Partitioned fitness top-k is correct with deterministic ties.
- Mandatory policy predicates cannot be disabled or bypassed.
- Every resource-limit truncation is visible through `complete=False` and a
  reason.
- Existing semantic-graph, hybrid-retriever, subgraph-hop, and persistence
  tests remain green.
- The fixed operators expose actual cardinalities and time so the future
  optimizer can calibrate costs without changing their semantics.

## 11. Optimizer handoff

Once these criteria pass, the optimizer receives stable alternatives:

```text
VectorSeed -> AdjacencyJoin -> AdjacencyJoin -> Filter -> GroupTopK -> PathExpand
```

It can then introduce and compare:

- filter tasks before exact vector ranking;
- iterative/inline filtered ANN;
- push non-buggy/policy predicates into the node join;
- adjacency nested-loop versus materialized edge hash join;
- fitness top-k before versus after safe joins;
- BFS versus best-first/beam path expansion.

The join layer must be completed first because its actual input/output
cardinalities provide the missing measurements for:

```text
vector seed count -> session fan-out -> node fan-out -> traversal cost
```
