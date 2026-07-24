# End-to-end vector-seeded graph traversal on one storage engine

Status: design mock for the proposed API; it is not runnable on current
`main` until the relation-join PRs are implemented.

## Material Passport

- Origin Skill: `academic-research-suite/deep-research`
- Origin Mode: `fact-check`
- Origin Date: `2026-07-23`
- Verification Status: `official-documentation-checked; DuckDB-SQL-executed`
- DuckDB Validation Version: `1.5.4`
- Repository Commit: `8c795ca`

## Architectural answer

Neo4j is not required. It appeared in the earlier plan only because Mandol
already has a partial Neo4j adapter. Making that adapter part of the critical
path would be the wrong default.

“Unified storage” has four increasingly strong meanings:

| Level | Meaning | DuckDB | PostgreSQL + pgvector | Neo4j + separate vector store |
|---|---|---:|---:|---:|
| 1 | Canonical UID/schema | Yes | Yes | Possible |
| 2 | One transactional source of truth | Yes | Yes | No |
| 3 | One query/optimizer for vector, joins, filters, traversal | Yes | Yes | No |
| 4 | One in-process address space | Yes | No | No |

PostgreSQL is still a unified storage/query layer even though it is a server
process: units, vectors, spaces, and edges live in one database and one
snapshot, and one SQL statement performs the whole operation. It does not
satisfy the Mandol paper's stricter “single address space” wording. DuckDB
does.

The recommended choices are:

- **Embedded Mandol/default:** DuckDB, exact vector ranking first, optional
  in-memory/rebuilt VSS index.
- **Concurrent production service:** PostgreSQL + pgvector.
- **Neo4j:** optional federated adapter only; never required by the logical
  query API.

DuckDB supports fixed-size vector arrays, HNSW through its VSS extension, and
recursive CTEs. Its documentation currently marks persistent VSS indexes
experimental and advises against production use; the index must fit in RAM.
DuckDB also limits normal read-write access to one process, although threads
inside that process can write concurrently. See the official
[VSS documentation](https://duckdb.org/docs/lts/core_extensions/vss),
[recursive CTE documentation](https://duckdb.org/docs/current/sql/query_syntax/with),
and [concurrency documentation](https://duckdb.org/docs/current/connect/concurrency).

PostgreSQL with pgvector supports exact vector ranking, persistent HNSW and
IVFFlat indexes, and iterative ANN scans for filtered queries. PostgreSQL 18
supports recursive CTE traversal with `SEARCH` and `CYCLE`, and row-level
security can enforce policy predicates inside the same statement. See
[pgvector](https://github.com/pgvector/pgvector),
[PostgreSQL recursive queries](https://www.postgresql.org/docs/current/queries-with.html),
and [PostgreSQL row security](https://www.postgresql.org/docs/18/ddl-rowsecurity.html).

DuckPGQ can provide SQL/PGQ graph syntax over DuckDB tables, but its official
community-extension page describes it as an ongoing research project. It
should be optional syntax sugar; the correctness path should use ordinary edge
tables and recursive CTEs
([DuckPGQ](https://duckdb.org/community_extensions/extensions/duckpgq.html)).

## Unified physical schema

Use normalized tables rather than a graph database beside a vector database:

```sql
CREATE TABLE memory_units (
    uid             VARCHAR PRIMARY KEY,
    raw_data        JSON NOT NULL,
    metadata        JSON NOT NULL,
    embedding       FLOAT[4],

    -- Promoted, typed columns for frequent predicates and statistics.
    unit_kind       VARCHAR NOT NULL,
    fitness_score   DOUBLE,
    is_buggy        BOOLEAN,
    tenant_id       VARCHAR NOT NULL
);

CREATE TABLE memory_spaces (
    name            VARCHAR PRIMARY KEY,
    parent_name     VARCHAR
);

CREATE TABLE space_members (
    space_name      VARCHAR NOT NULL,
    uid             VARCHAR NOT NULL,
    PRIMARY KEY (space_name, uid)
);

CREATE TABLE relationships (
    source_uid      VARCHAR NOT NULL,
    target_uid      VARCHAR NOT NULL,
    rel_type        VARCHAR NOT NULL,
    properties      JSON NOT NULL DEFAULT '{}',
    PRIMARY KEY (source_uid, target_uid, rel_type)
);

CREATE INDEX relationships_out
    ON relationships (source_uid, rel_type);
CREATE INDEX relationships_in
    ON relationships (target_uid, rel_type);
CREATE INDEX space_members_by_space
    ON space_members (space_name, uid);
```

For PostgreSQL, change `FLOAT[4]` to `vector(4)`, `JSON` to `JSONB`, and add
foreign keys and a pgvector HNSW index. For DuckDB, VSS acceleration is
optional:

```sql
INSTALL vss;
LOAD vss;
CREATE INDEX memory_embedding_hnsw
ON memory_units USING HNSW (embedding)
WITH (metric = 'cosine');
```

For durable DuckDB production data, do not enable experimental persistent
HNSW initially. Keep the canonical embeddings in `memory_units` and either:

1. use exact scans; or
2. rebuild the in-memory HNSW index at process startup.

That retains one source of truth even when the acceleration structure is
rebuildable.

## Proposed Python API: complete mock

```python
from __future__ import annotations

import numpy as np

from mandol.domain import MemoryUnit
from mandol.domain.types import Uid
from mandol.infrastructure.duckdb_unified_store import DuckDBUnifiedStore
from mandol.query.predicates import Field, PolicyAllows
from mandol.retrieval.vector_seeded_graph import (
    ExperienceTraversalSpec,
    VectorSeededGraphRetriever,
)


# One database and transaction. These are compatibility facades over the same
# DuckDB connection, not separate physical stores.
backend = DuckDBUnifiedStore(
    path="experience.duckdb",
    embedding_dim=4,
    vector_mode="exact",  # correctness baseline; optimizer may later choose VSS
)


def unit(
    uid: str,
    kind: str,
    *,
    embedding: list[float] | None = None,
    fitness: float | None = None,
    buggy: bool | None = None,
    tenant: str = "acme",
) -> MemoryUnit:
    return MemoryUnit(
        uid=Uid(uid),
        raw_data={"text_content": uid},
        metadata={
            "kind": kind,
            "fitness_score": fitness,
            "is_buggy": buggy,
            "tenant_id": tenant,
        },
        embedding=(
            np.asarray(embedding, dtype=np.float32)
            if embedding is not None
            else None
        ),
    )


with backend.transaction():
    backend.units.upsert_units(
        [
            # Vector-searchable task descriptions.
            unit("task_a", "task", embedding=[1.0, 0.0, 0.0, 0.0]),
            unit("task_b", "task", embedding=[0.8, 0.2, 0.0, 0.0]),
            unit("task_c", "task", embedding=[0.0, 1.0, 0.0, 0.0]),

            # Sessions.
            unit("session_a1", "session"),
            unit("session_a2", "session"),
            unit("session_b1", "session"),

            # Candidate roots.
            unit("node_a1", "exploration_node", fitness=0.70, buggy=False),
            unit("node_a2", "exploration_node", fitness=0.90, buggy=False),
            unit("node_a3", "exploration_node", fitness=1.00, buggy=True),
            unit("node_a4", "exploration_node", fitness=0.99, buggy=False,
                 tenant="other"),
            unit("node_b1", "exploration_node", fitness=0.80, buggy=False),

            # Descendants.
            unit("node_a2_1", "exploration_node", fitness=0.92, buggy=False),
            unit("node_a2_2", "exploration_node", fitness=0.94, buggy=False),
            unit("node_b1_1", "exploration_node", fitness=0.82, buggy=False),
        ]
    )

    backend.spaces.add_members(
        "experience/tasks",
        [Uid("task_a"), Uid("task_b"), Uid("task_c")],
    )

    # Relationships are rows in the same database.
    backend.graph.upsert_relationship(
        Uid("session_a1"), Uid("task_a"), "BELONGS_TO", {}
    )
    backend.graph.upsert_relationship(
        Uid("session_a2"), Uid("task_a"), "BELONGS_TO", {}
    )
    backend.graph.upsert_relationship(
        Uid("session_b1"), Uid("task_b"), "BELONGS_TO", {}
    )

    backend.graph.upsert_relationship(
        Uid("node_a1"), Uid("session_a1"), "IN_SESSION", {}
    )
    backend.graph.upsert_relationship(
        Uid("node_a2"), Uid("session_a1"), "IN_SESSION", {}
    )
    backend.graph.upsert_relationship(
        Uid("node_a3"), Uid("session_a2"), "IN_SESSION", {}
    )
    backend.graph.upsert_relationship(
        Uid("node_a4"), Uid("session_a2"), "IN_SESSION", {}
    )
    backend.graph.upsert_relationship(
        Uid("node_b1"), Uid("session_b1"), "IN_SESSION", {}
    )

    backend.graph.upsert_relationship(
        Uid("node_a2"), Uid("node_a2_1"), "HAS_CHILD", {}
    )
    backend.graph.upsert_relationship(
        Uid("node_a2_1"), Uid("node_a2_2"), "HAS_CHILD", {}
    )
    backend.graph.upsert_relationship(
        Uid("node_b1"), Uid("node_b1_1"), "HAS_CHILD", {}
    )


spec = ExperienceTraversalSpec(
    seed_space="experience/tasks",
    seed_k=2,
    task_predicate=Field("task", "unit_kind") == "task",

    # These compile to incoming adjacency joins:
    # task <-BELONGS_TO- session <-IN_SESSION- node
    per_task_roots=1,
    root_predicate=(
        (Field("node", "is_buggy") == False)
        & PolicyAllows(alias="node")
    ),
    root_order=(
        Field("node", "fitness_score").desc(),
        Field("node", "uid").asc(),
    ),

    traversal_rel_types=("HAS_CHILD",),
    traversal_direction="out",
    min_hops=0,
    max_hops=2,
    uniqueness="node_global_per_seed",
    per_seed_limit=20,
    global_limit=100,
    final_k=10,
)

retriever = VectorSeededGraphRetriever(
    query_store=backend.queries,
)

result = retriever.search(
    query=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    spec=spec,
    policy_context={"tenant_id": "acme"},
)

assert result.complete is True
assert result.approximate is False

for row in result.rows:
    print(
        row.bindings["task"],
        row.bindings["session"],
        row.bindings["node"],              # selected high-fitness root
        row.bindings["trajectory_node"],   # root or descendant
        row.scores["task_similarity"],
        [edge.rel_type for edge in row.path],
    )
```

Expected logical results:

```text
task_a session_a1 node_a2 node_a2   1.0000 [BELONGS_TO, IN_SESSION]
task_a session_a1 node_a2 node_a2_1 1.0000 [BELONGS_TO, IN_SESSION, HAS_CHILD]
task_a session_a1 node_a2 node_a2_2 1.0000 [BELONGS_TO, IN_SESSION, HAS_CHILD, HAS_CHILD]
task_b session_b1 node_b1 node_b1   0.9701 [BELONGS_TO, IN_SESSION]
task_b session_b1 node_b1 node_b1_1 0.9701 [BELONGS_TO, IN_SESSION, HAS_CHILD]
```

`node_a3` is removed because it is buggy. `node_a4` is removed by the tenant
policy. `node_a2` then wins task A's partition by fitness. The task vector
score and both join relationships remain attached to every trajectory row.

## The single DuckDB statement generated by that API

The concrete compiler can emit the parameterized statement below. The query
was executed successfully against the mock dataset with local DuckDB 1.5.4
and produced the five expected rows.

```sql
WITH RECURSIVE
params AS (
    SELECT
        ?::FLOAT[4] AS query_embedding,
        ?::VARCHAR AS tenant_id
),

task_seeds AS MATERIALIZED (
    SELECT
        u.uid AS task_uid,
        1.0 - array_cosine_distance(
            u.embedding,
            p.query_embedding
        ) AS task_similarity
    FROM memory_units u
    JOIN space_members sm
      ON sm.uid = u.uid
     AND sm.space_name = 'experience/tasks'
    CROSS JOIN params p
    WHERE u.unit_kind = 'task'
      AND u.embedding IS NOT NULL
      AND u.tenant_id = p.tenant_id
    ORDER BY array_cosine_distance(u.embedding, p.query_embedding), u.uid
    LIMIT 2
),

joined_nodes AS (
    SELECT
        ts.task_uid,
        ts.task_similarity,
        s.uid AS session_uid,
        n.uid AS node_uid,
        n.fitness_score,
        n.tenant_id
    FROM task_seeds ts
    JOIN relationships belongs
      ON belongs.target_uid = ts.task_uid
     AND belongs.rel_type = 'BELONGS_TO'
    JOIN memory_units s
      ON s.uid = belongs.source_uid
     AND s.unit_kind = 'session'
    JOIN relationships membership
      ON membership.target_uid = s.uid
     AND membership.rel_type = 'IN_SESSION'
    JOIN memory_units n
      ON n.uid = membership.source_uid
     AND n.unit_kind = 'exploration_node'
    CROSS JOIN params p
    WHERE n.is_buggy = false
      AND n.tenant_id = p.tenant_id
),

ranked_roots AS (
    SELECT
        *,
        row_number() OVER (
            PARTITION BY task_uid
            ORDER BY fitness_score DESC, node_uid ASC
        ) AS fitness_rank
    FROM joined_nodes
),

roots AS (
    SELECT *
    FROM ranked_roots
    WHERE fitness_rank <= 1
),

trajectory (
    task_uid,
    task_similarity,
    session_uid,
    root_node_uid,
    trajectory_node_uid,
    root_fitness,
    depth,
    node_path
) AS (
    SELECT
        task_uid,
        task_similarity,
        session_uid,
        node_uid,
        node_uid,
        fitness_score,
        0,
        list_value(node_uid)
    FROM roots

    UNION ALL

    SELECT
        tr.task_uid,
        tr.task_similarity,
        tr.session_uid,
        tr.root_node_uid,
        edge.target_uid,
        tr.root_fitness,
        tr.depth + 1,
        list_append(tr.node_path, edge.target_uid)
    FROM trajectory tr
    JOIN relationships edge
      ON edge.source_uid = tr.trajectory_node_uid
     AND edge.rel_type = 'HAS_CHILD'
    JOIN memory_units child
      ON child.uid = edge.target_uid
    CROSS JOIN params p
    WHERE tr.depth < 2
      AND child.is_buggy = false
      AND child.tenant_id = p.tenant_id
      AND NOT list_contains(tr.node_path, edge.target_uid)
)

SELECT *
FROM trajectory
ORDER BY
    task_similarity DESC,
    root_fitness DESC,
    task_uid,
    depth,
    trajectory_node_uid
LIMIT 10;
```

This is vector retrieval, two relation joins, structured/policy filtering,
partitioned fitness selection, and variable-length graph traversal over one
database snapshot. No NetworkX or Neo4j round-trip is involved.

For the first implementation, use the exact distance expression above. A
future DuckDB physical plan can replace `task_seeds` with an HNSW scan. Since
the logical query and output bindings do not change, this becomes an optimizer
decision rather than an API rewrite.

## PostgreSQL translation

The schema and joins are the same. Change the seed expression to pgvector:

```sql
SELECT
    u.uid AS task_uid,
    1.0 - (u.embedding <=> $1::vector) AS task_similarity
FROM memory_units u
JOIN space_members sm ON sm.uid = u.uid
WHERE sm.space_name = 'experience/tasks'
  AND u.unit_kind = 'task'
ORDER BY u.embedding <=> $1::vector
LIMIT $2;
```

Change the recursive path operations to PostgreSQL arrays:

```sql
tr.node_path || edge.target_uid
AND NOT edge.target_uid = ANY(tr.node_path)
```

Then enable:

```sql
CREATE EXTENSION vector;

CREATE INDEX memory_embedding_hnsw
ON memory_units
USING hnsw (embedding vector_cosine_ops);

SET hnsw.iterative_scan = strict_order;
```

PostgreSQL is the stronger choice when many agent worker processes write
concurrently, durable ANN indexes are mandatory, row-level policy enforcement
must live in the database, or replication/point-in-time recovery is required.

## What the later Mandol optimizer owns

The unified database does not eliminate the need for Mandol's multimodal
optimizer. It gives that optimizer safe physical alternatives:

- exact filtered task scan;
- global HNSW then iterative filter;
- metadata/space prefilter then exact vector ranking;
- task seeds then adjacency joins;
- selective node filtering during the second join;
- recursive CTE versus application frontier traversal;
- materialized seed CTE versus an inlined subplan.

The storage engine can optimize ordinary joins, but Mandol still needs to
model vector recall/selectivity, relation fan-out, traversal growth, reranking
cost, and quality constraints across the whole request.
