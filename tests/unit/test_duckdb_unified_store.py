"""Conformance tests for the unified DuckDB physical backend."""

from __future__ import annotations

import numpy as np
import pytest

duckdb = pytest.importorskip("duckdb")

from mandol.application import SemanticGraphService, SemanticMapService
from mandol.domain import MemorySpace, MemoryUnit, SpaceName, Uid
from mandol.infrastructure import DuckDBUnifiedStore
from mandol.ports import StaticEmbeddingProvider
from mandol.query import (
    VectorSeededTraversalSpec,
    format_query_log,
    render_query_diagram,
)


def _unit(
    uid: str,
    *,
    kind: str,
    embedding: list[float] | None = None,
    **metadata,
) -> MemoryUnit:
    return MemoryUnit(
        uid=Uid(uid),
        raw_data={"text_content": uid},
        metadata={"kind": kind, **metadata},
        embedding=(np.asarray(embedding, dtype=np.float32) if embedding is not None else None),
    )


def test_facades_round_trip_and_preserve_typed_parallel_edges(tmp_path):
    database = tmp_path / "mandol.duckdb"
    backend = DuckDBUnifiedStore(database, embedding_dim=3)

    unit = _unit("u1", kind="task", embedding=[3.0, 0.0, 0.0], owner="alice")
    other = _unit("u2", kind="node", embedding=[0.0, 1.0, 0.0])
    backend.units.upsert_units([unit, other])
    backend.units.upsert_spaces(
        [
            MemorySpace(
                name=SpaceName("tasks"),
                unit_uids={Uid("u1")},
                child_spaces={SpaceName("archive")},
                summary_text="task memories",
                summary_embedding=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            )
        ]
    )

    backend.graph.upsert_relationship(Uid("u1"), Uid("u2"), "A", {"weight": 1})
    backend.graph.upsert_relationship(Uid("u1"), Uid("u2"), "B", {"weight": 2})

    restored = backend.units.get_unit(Uid("u1"))
    assert restored is not None
    assert restored.raw_data == unit.raw_data
    assert restored.get_user_metadata() == unit.get_user_metadata()
    np.testing.assert_allclose(restored.embedding, [3.0, 0.0, 0.0])

    space = backend.units.get_space(SpaceName("tasks"))
    assert space is not None
    assert space.unit_uids == {Uid("u1")}
    assert space.child_spaces == {SpaceName("archive")}

    assert backend.vectors.search(np.asarray([1.0, 0.0, 0.0]), top_k=2)[0] == (
        Uid("u1"),
        pytest.approx(1.0),
    )
    assert backend.vectors.search_in_space(
        np.asarray([1.0, 0.0, 0.0]),
        "tasks",
        candidates=None,
        top_k=2,
    ) == [(Uid("u1"), pytest.approx(1.0))]

    assert backend.graph.get_relationship(Uid("u1"), Uid("u2"), "A") == {"weight": 1}
    assert backend.graph.get_relationship(Uid("u1"), Uid("u2"), "B") == {"weight": 2}
    assert len(backend.graph.get_all_edges()) == 2

    backend.graph.delete_relationship(Uid("u1"), Uid("u2"), "A")
    assert backend.graph.get_relationship(Uid("u1"), Uid("u2"), "A") is None
    assert backend.graph.get_relationship(Uid("u1"), Uid("u2"), "B") == {"weight": 2}
    backend.close()

    reopened = DuckDBUnifiedStore(database, embedding_dim=3)
    assert reopened.units.get_unit(Uid("u1")) is not None
    assert reopened.graph.get_relationship(Uid("u1"), Uid("u2"), "B") == {"weight": 2}
    reopened.close()


def test_transaction_rolls_back_nested_facade_writes():
    backend = DuckDBUnifiedStore(embedding_dim=2)

    with pytest.raises(RuntimeError, match="abort"), backend.transaction():
        backend.units.upsert_units([_unit("will-rollback", kind="task", embedding=[1.0, 0.0])])
        backend.graph.upsert_relationship(Uid("will-rollback"), Uid("other"), "RELATED_TO", {})
        raise RuntimeError("abort")

    assert backend.units.get_unit(Uid("will-rollback")) is None
    assert backend.graph.get_all_edges() == []
    backend.close()


def test_existing_semantic_map_and_graph_services_accept_duckdb_facades():
    backend = DuckDBUnifiedStore(embedding_dim=2)
    semantic_map = SemanticMapService(
        store=backend.units,
        index=backend.vectors,
        embedder=StaticEmbeddingProvider(dim=2, fill=1.0),
    )
    graph = SemanticGraphService(
        semantic_map=semantic_map,
        graph_store=backend.graph,
    )

    graph.add_unit(_unit("u1", kind="task", embedding=[1.0, 0.0]), space_names=["s"])
    graph.add_unit(_unit("u2", kind="node", embedding=[0.0, 1.0]), space_names=["s"])
    graph.add_relationship("u1", "u2", "RELATED_TO", weight=0.7)

    hits = semantic_map.search_by_vector(
        np.asarray([1.0, 0.0], dtype=np.float32),
        top_k=2,
        space_names=["s"],
    )
    assert [unit.uid for unit, _ in hits] == [Uid("u1"), Uid("u2")]
    assert [unit.uid for unit in graph.get_explicit_neighbors(["u1"])] == [Uid("u2")]
    backend.close()


def test_vector_seeded_relation_join_and_path_traversal_is_one_query():
    backend = DuckDBUnifiedStore(embedding_dim=2)
    units = [
        _unit("task-a", kind="task", embedding=[1.0, 0.0]),
        _unit("task-b", kind="task", embedding=[0.0, 1.0]),
        _unit("session-a", kind="session"),
        _unit("session-b", kind="session"),
        _unit(
            "node-best",
            kind="node",
            fitness_score=0.9,
            is_buggy=False,
        ),
        _unit(
            "node-lower",
            kind="node",
            fitness_score=0.4,
            is_buggy=False,
        ),
        _unit(
            "node-buggy",
            kind="node",
            fitness_score=1.0,
            is_buggy=True,
        ),
        _unit("child-1", kind="node", fitness_score=0.7, is_buggy=False),
        _unit("child-2", kind="node", fitness_score=0.6, is_buggy=False),
    ]
    backend.units.upsert_units(units)

    for source, target, relation in [
        ("session-a", "task-a", "BELONGS_TO"),
        ("session-b", "task-b", "BELONGS_TO"),
        ("node-best", "session-a", "IN_SESSION"),
        ("node-lower", "session-a", "IN_SESSION"),
        ("node-buggy", "session-a", "IN_SESSION"),
        ("node-best", "child-1", "HAS_CHILD"),
        ("child-1", "child-2", "HAS_CHILD"),
        # Cycle must not repeat a UID already in the current path.
        ("child-2", "node-best", "HAS_CHILD"),
    ]:
        backend.graph.upsert_relationship(Uid(source), Uid(target), relation, {})

    spec = VectorSeededTraversalSpec(
        query_vector=np.asarray([1.0, 0.0], dtype=np.float32),
        vector_k=1,
        seed_metadata_equals={"kind": "task"},
        node_metadata_equals={"kind": "node", "is_buggy": False},
        nodes_per_seed=1,
        min_hops=0,
        max_hops=3,
        result_limit=10,
    )
    execution = backend.queries.vector_seeded_graph_traversal(spec, profile=True)

    assert [
        (row.seed_uid, row.session_uid, row.node_uid, row.current_uid, row.depth)
        for row in execution.rows
    ] == [
        ("task-a", "session-a", "node-best", "node-best", 0),
        ("task-a", "session-a", "node-best", "child-1", 1),
        ("task-a", "session-a", "node-best", "child-2", 2),
    ]
    assert execution.rows[-1].path == ("node-best", "child-1", "child-2")
    assert execution.metrics.complete is True
    assert execution.metrics.stages.vector_candidate_rows == 2
    assert execution.metrics.stages.vector_seed_rows == 1
    assert execution.metrics.stages.session_join_rows == 1
    assert execution.metrics.stages.node_join_rows == 2
    assert execution.metrics.stages.selected_node_rows == 1
    assert execution.metrics.stages.traversal_rows == 3
    assert execution.metrics.timings.vector_search_cpu_ms is not None
    assert execution.metrics.timings.relation_join_cpu_ms is not None
    assert execution.metrics.timings.graph_traversal_cpu_ms is not None
    assert execution.metrics.timings.total_cpu_ms is not None
    assert "stage=vector_search" in format_query_log(spec, execution)
    diagram = render_query_diagram(spec, execution)
    assert "VECTOR SEARCH" in diagram
    assert "RELATION JOIN" in diagram
    assert "GRAPH TRAVERSAL" in diagram
    assert "REC_CTE" in backend.queries.explain_vector_seeded_graph_traversal(spec)
    backend.close()


def test_result_limit_surfaces_incompleteness():
    backend = DuckDBUnifiedStore(embedding_dim=2)
    backend.units.upsert_units(
        [
            _unit("task", kind="task", embedding=[1.0, 0.0]),
            _unit("session", kind="session"),
            _unit("node", kind="node", fitness_score=1.0, is_buggy=False),
            _unit("child", kind="node"),
        ]
    )
    backend.graph.upsert_relationship("session", "task", "BELONGS_TO", {})
    backend.graph.upsert_relationship("node", "session", "IN_SESSION", {})
    backend.graph.upsert_relationship("node", "child", "HAS_CHILD", {})

    execution = backend.queries.vector_seeded_graph_traversal(
        VectorSeededTraversalSpec(
            query_vector=np.asarray([1.0, 0.0]),
            vector_k=1,
            seed_metadata_equals={"kind": "task"},
            node_metadata_equals={"is_buggy": False},
            max_hops=1,
            result_limit=1,
        )
    )

    assert len(execution.rows) == 1
    assert execution.metrics.complete is False
    assert execution.metrics.truncated_reason == "result_limit"
    backend.close()


def test_reopening_with_a_different_embedding_dimension_fails(tmp_path):
    database = tmp_path / "dimension.duckdb"
    DuckDBUnifiedStore(database, embedding_dim=2).close()

    with pytest.raises(ValueError, match="embedding_dim does not match"):
        DuckDBUnifiedStore(database, embedding_dim=3)


def test_empty_query_still_reports_zero_stage_cardinalities():
    backend = DuckDBUnifiedStore(embedding_dim=2)
    backend.units.upsert_units([_unit("not-a-task", kind="node", embedding=[1.0, 0.0])])

    execution = backend.queries.vector_seeded_graph_traversal(
        VectorSeededTraversalSpec(
            query_vector=np.asarray([1.0, 0.0]),
            seed_metadata_equals={"kind": "task"},
        )
    )

    assert execution.rows == ()
    assert execution.metrics.rows_returned == 0
    assert execution.metrics.stages.vector_candidate_rows == 0
    assert execution.metrics.stages.vector_seed_rows == 0
    assert execution.metrics.stages.traversal_rows == 0
    backend.close()
