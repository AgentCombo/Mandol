from __future__ import annotations

import numpy as np
import pytest

from mandol.domain.memory_unit import MemoryUnit
from mandol.domain.types import Uid
from mandol.infrastructure.in_memory_graph_store import InMemoryGraphStore
from mandol.infrastructure.in_memory_unified_query_store import InMemoryUnifiedQueryStore
from mandol.infrastructure.in_memory_unit_store import InMemoryUnitStore
from mandol.infrastructure.in_memory_vector_index import InMemoryCosineVectorIndex
from mandol.query import (
    VectorSeededTraversalSpec,
    explain_logical_plan,
    format_query_log,
    render_query_diagram,
)


def _unit(uid: str, *, kind: str, embedding=None, **metadata) -> MemoryUnit:
    return MemoryUnit(
        uid=Uid(uid),
        raw_data={"text_content": uid},
        metadata={"kind": kind, **metadata},
        embedding=(
            None
            if embedding is None
            else np.asarray(embedding, dtype=np.float32)
        ),
    )


def _executor() -> InMemoryUnifiedQueryStore:
    units = [
        _unit("task-a", kind="task", embedding=[1.0, 0.0]),
        _unit("task-b", kind="task", embedding=[0.0, 1.0]),
        _unit("session-a", kind="session"),
        _unit("session-b", kind="session"),
        _unit("node-best", kind="node", fitness_score=0.9, is_buggy=False),
        _unit("node-lower", kind="node", fitness_score=0.4, is_buggy=False),
        _unit("node-buggy", kind="node", fitness_score=1.0, is_buggy=True),
        _unit("child-1", kind="node", fitness_score=0.7, is_buggy=False),
        _unit("child-2", kind="node", fitness_score=0.6, is_buggy=False),
    ]
    store = InMemoryUnitStore()
    store.upsert_units(units)
    vector = InMemoryCosineVectorIndex(2)
    vector.upsert(
        [(unit.uid, unit.embedding) for unit in units if unit.embedding is not None]
    )
    graph = InMemoryGraphStore()
    for source, target, relation in [
        ("session-a", "task-a", "BELONGS_TO"),
        ("session-b", "task-b", "BELONGS_TO"),
        ("node-best", "session-a", "IN_SESSION"),
        ("node-lower", "session-a", "IN_SESSION"),
        ("node-buggy", "session-a", "IN_SESSION"),
        ("node-best", "child-1", "HAS_CHILD"),
        ("child-1", "child-2", "HAS_CHILD"),
        ("child-2", "node-best", "HAS_CHILD"),
    ]:
        graph.upsert_relationship(Uid(source), Uid(target), relation, {})
    return InMemoryUnifiedQueryStore(
        unit_store=store,
        vector_index=vector,
        graph_store=graph,
    )


def _spec(**overrides) -> VectorSeededTraversalSpec:
    values = {
        "query_vector": np.asarray([1.0, 0.0], dtype=np.float32),
        "vector_k": 1,
        "seed_metadata_equals": {"kind": "task"},
        "node_metadata_equals": {"kind": "node", "is_buggy": False},
        "nodes_per_seed": 1,
        "min_hops": 0,
        "max_hops": 3,
        "result_limit": 10,
    }
    values.update(overrides)
    return VectorSeededTraversalSpec(**values)


def test_fixed_plan_runs_vector_join_filter_and_path_traversal():
    executor = _executor()
    spec = _spec()

    execution = executor.vector_seeded_graph_traversal(spec, profile=True)

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
    assert len(execution.metrics.operators) == 9
    assert all(metric.elapsed_ms >= 0 for metric in execution.metrics.operators)
    assert execution.metrics.timings.vector_search_cpu_ms is not None
    assert execution.metrics.timings.relation_join_cpu_ms is not None
    assert execution.metrics.timings.graph_traversal_cpu_ms is not None

    log = format_query_log(spec, execution)
    assert "operator=op04_session_join" in log
    assert "input_rows=1 output_rows=1" in log
    diagram = render_query_diagram(spec, execution)
    assert "VECTOR SEARCH" in diagram
    assert "RELATION JOIN" in diagram
    assert "GRAPH TRAVERSAL" in diagram


def test_result_limit_reports_incomplete_execution():
    execution = _executor().vector_seeded_graph_traversal(_spec(result_limit=1))

    assert len(execution.rows) == 1
    assert execution.metrics.stages.traversal_rows == 3
    assert execution.metrics.complete is False
    assert execution.metrics.truncated_reason == "result_limit"


def test_min_hops_excludes_start_row_and_cycle_does_not_repeat_path_uid():
    execution = _executor().vector_seeded_graph_traversal(
        _spec(min_hops=1, max_hops=5)
    )

    assert [row.depth for row in execution.rows] == [1, 2]
    assert all(len(row.path) == len(set(row.path)) for row in execution.rows)


def test_empty_seed_filter_reports_zero_cardinalities():
    execution = _executor().vector_seeded_graph_traversal(
        _spec(seed_metadata_equals={"kind": "missing"})
    )

    assert execution.rows == ()
    assert execution.metrics.stages.vector_candidate_rows == 0
    assert execution.metrics.stages.vector_seed_rows == 0
    assert execution.metrics.stages.traversal_rows == 0


def test_explain_exposes_fixed_operator_tree():
    executor = _executor()
    plan = executor.explain_vector_seeded_graph_traversal(_spec())
    logical = explain_logical_plan(executor.logical_plan(_spec()))

    assert plan.startswith("IN_MEMORY_FIXED_PLAN")
    assert "VectorScan" in plan
    assert "EdgeJoin" in plan
    assert "GroupTopK" in plan
    assert "Traverse" in plan
    assert "stable_sort_limit" in plan
    assert "VectorScan" in logical
    assert "EdgeJoin" in logical
    assert "ProjectLimit" in logical


def test_filter_first_exact_candidate_matches_fixed_plan():
    baseline = _executor()
    candidate = InMemoryUnifiedQueryStore(
        unit_store=baseline._unit_store,
        vector_index=baseline._vector_index,
        graph_store=baseline._graph_store,
        vector_strategy="filter_first_exact",
    )

    expected = baseline.vector_seeded_graph_traversal(_spec())
    actual = candidate.vector_seeded_graph_traversal(_spec())

    assert actual.rows == expected.rows
    assert actual.metrics.stages == expected.metrics.stages
    algorithms = {metric.algorithm for metric in actual.metrics.operators}
    assert "numpy_filtered_exact_cosine" in algorithms
    assert candidate.explain_vector_seeded_graph_traversal(_spec()).startswith(
        "IN_MEMORY_CANDIDATE_PLAN[filter_first_exact]"
    )


def test_faiss_ann_adaptive_candidate_runs_end_to_end():
    pytest.importorskip("faiss")
    from mandol.infrastructure.faiss_hnsw_vector_index import FaissHNSWVectorIndex

    baseline = _executor()
    ann = FaissHNSWVectorIndex(2, ef_search=32)
    ann.upsert(
        [
            (unit.uid, unit.embedding)
            for unit in baseline._unit_store.list_units()
            if unit.embedding is not None
        ]
    )
    candidate = InMemoryUnifiedQueryStore(
        unit_store=baseline._unit_store,
        vector_index=ann,
        graph_store=baseline._graph_store,
        vector_strategy="ann_adaptive",
    )

    execution = candidate.vector_seeded_graph_traversal(_spec())

    assert execution.rows == baseline.vector_seeded_graph_traversal(_spec()).rows
    assert execution.metrics.stages.vector_candidate_rows == 2
    assert execution.metrics.stages.vector_seed_rows == 1
    assert execution.metrics.operators[0].algorithm == "ann_adaptive_widening"
    assert len(execution.metrics.operators) == 7
    assert "algorithm=ann_adaptive_widening" in format_query_log(_spec(), execution)
    assert "ann adaptive widening" in render_query_diagram(_spec(), execution)


def test_rejects_unknown_vector_strategy():
    baseline = _executor()

    with pytest.raises(ValueError, match="unsupported vector strategy"):
        InMemoryUnifiedQueryStore(
            unit_store=baseline._unit_store,
            vector_index=baseline._vector_index,
            graph_store=baseline._graph_store,
            vector_strategy="not-a-plan",  # type: ignore[arg-type]
        )


def test_in_memory_rows_and_stage_cardinalities_match_duckdb():
    pytest.importorskip("duckdb")
    from mandol.infrastructure.duckdb_unified_store import DuckDBUnifiedStore

    memory_executor = _executor()
    spec = _spec()
    expected = memory_executor.vector_seeded_graph_traversal(spec)

    with DuckDBUnifiedStore(embedding_dim=2) as backend:
        backend.units.upsert_units(memory_executor._unit_store.list_units())
        for source, target, rel_type, properties in (
            memory_executor._graph_store.get_all_edges()
        ):
            backend.graph.upsert_relationship(source, target, rel_type, properties)
        actual = backend.queries.vector_seeded_graph_traversal(spec)

    assert actual.rows == expected.rows
    assert actual.metrics.stages == expected.metrics.stages
