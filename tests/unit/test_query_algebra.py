from __future__ import annotations

import numpy as np

from mandol.domain.memory_space import MemorySpace
from mandol.domain.memory_unit import MemoryUnit
from mandol.domain.types import SpaceName, Uid
from mandol.infrastructure.in_memory_graph_store import InMemoryGraphStore
from mandol.infrastructure.in_memory_unit_store import InMemoryUnitStore
from mandol.infrastructure.in_memory_vector_index import InMemoryCosineVectorIndex
from mandol.query.algebra import (
    AndPredicate,
    BindingRow,
    ComparisonPredicate,
    EdgeExistsPredicate,
    FieldRef,
    InSpacePredicate,
)
from mandol.query.execution import (
    FilterOperator,
    InnerHashJoinOperator,
    QueryExecutionContext,
    UnitScanOperator,
)


def _unit(uid: str, *, group: str, count: int) -> MemoryUnit:
    return MemoryUnit(
        uid=Uid(uid),
        raw_data={"text_content": f"hello from {uid}"},
        metadata={"group": group, "count": count},
        embedding=np.asarray([1.0, 0.0], dtype=np.float32),
    )


def _context() -> QueryExecutionContext:
    store = InMemoryUnitStore()
    units = [
        _unit("u1", group="a", count=1),
        _unit("u2", group="a", count=2),
        _unit("u3", group="b", count=3),
    ]
    store.upsert_units(units)
    store.upsert_spaces(
        [
            MemorySpace(name=SpaceName("root"), child_spaces={SpaceName("child")}),
            MemorySpace(name=SpaceName("child"), unit_uids={Uid("u1"), Uid("u2")}),
        ]
    )
    vector = InMemoryCosineVectorIndex(2)
    vector.upsert([(unit.uid, unit.embedding) for unit in units])
    graph = InMemoryGraphStore()
    graph.upsert_relationship(Uid("u1"), Uid("u2"), "RELATED", {})
    return QueryExecutionContext(
        unit_store=store,
        vector_index=vector,
        graph_store=graph,
    )


def test_binding_row_is_copy_on_write_and_resolves_nested_fields():
    context = _context()
    original = BindingRow(bindings={"unit": Uid("u1")})
    updated = original.with_value("score", 0.75).bind("other", Uid("u2"))

    assert "score" not in original.values
    assert "other" not in original.bindings
    assert FieldRef("unit", "metadata.group").resolve(updated, context) == "a"
    assert FieldRef("unit", "raw_data.text_content").resolve(updated, context) == "hello from u1"
    assert FieldRef(None, "score").resolve(updated, context) == 0.75


def test_metadata_space_and_edge_predicates_compose():
    context = _context()
    row = BindingRow(bindings={"unit": Uid("u1"), "other": Uid("u2")})
    predicate = AndPredicate(
        (
            ComparisonPredicate(FieldRef("unit", "metadata.count"), "lte", 1),
            InSpacePredicate("unit", "root", recursive=True),
            EdgeExistsPredicate(
                "unit",
                "RELATED",
                direction="out",
                other_alias="other",
            ),
        )
    )

    assert predicate.evaluate(row, context) is True
    assert ComparisonPredicate(
        FieldRef("unit", "metadata.missing"),
        "neq",
        "anything",
    ).evaluate(row, context) is False


def test_filter_operator_combines_metadata_space_and_edge_filters():
    context = _context()
    operator = FilterOperator(
        "multimodal_filter",
        child=UnitScanOperator("all_units", alias="unit"),
        predicate=AndPredicate(
            (
                ComparisonPredicate(FieldRef("unit", "metadata.group"), "eq", "a"),
                InSpacePredicate("unit", "root", recursive=True),
                EdgeExistsPredicate("unit", "RELATED", direction="out"),
            )
        ),
        stage="relation_filter",
    )

    rows = operator.execute(context)

    assert [row.bindings["unit"] for row in rows] == [Uid("u1")]
    metric = next(
        item for item in context.operator_metrics if item.operator_id == "multimodal_filter"
    )
    assert metric.input_rows == 3
    assert metric.output_rows == 1


def test_filter_operator_reports_cardinality():
    context = _context()
    scan = UnitScanOperator("scan", alias="unit", space_name="root")
    operator = FilterOperator(
        "filter",
        child=scan,
        predicate=ComparisonPredicate(
            FieldRef("unit", "metadata.count"),
            "gt",
            1,
        ),
        stage="relation_filter",
    )

    rows = operator.execute(context)

    assert [row.bindings["unit"] for row in rows] == [Uid("u2")]
    metrics = {metric.operator_id: metric for metric in context.operator_metrics}
    assert metrics["scan"].input_rows == 3
    assert metrics["scan"].output_rows == 2
    assert metrics["filter"].input_rows == 2
    assert metrics["filter"].output_rows == 1


def test_inner_hash_join_supports_relational_equality_join():
    context = _context()
    join = InnerHashJoinOperator(
        "join",
        left=UnitScanOperator("left_scan", alias="left"),
        right=UnitScanOperator("right_scan", alias="right"),
        left_key=FieldRef("left", "metadata.group"),
        right_key=FieldRef("right", "metadata.group"),
    )

    rows = join.execute(context)

    pairs = {(row.bindings["left"], row.bindings["right"]) for row in rows}
    assert pairs == {
        (Uid("u1"), Uid("u1")),
        (Uid("u1"), Uid("u2")),
        (Uid("u2"), Uid("u1")),
        (Uid("u2"), Uid("u2")),
        (Uid("u3"), Uid("u3")),
    }
    metric = next(item for item in context.operator_metrics if item.operator_id == "join")
    assert metric.input_rows == 6
    assert metric.output_rows == 5


def test_in_memory_graph_store_preserves_parallel_typed_edges():
    graph = InMemoryGraphStore()
    graph.upsert_relationship(Uid("a"), Uid("b"), "TYPE_A", {"weight": 1})
    graph.upsert_relationship(Uid("a"), Uid("b"), "TYPE_B", {"weight": 2})

    assert graph.get_relationship(Uid("a"), Uid("b"), "TYPE_A") == {"weight": 1}
    assert graph.get_relationship(Uid("a"), Uid("b"), "TYPE_B") == {"weight": 2}
    assert len(graph.get_all_edges()) == 2
    assert graph.get_neighbors(Uid("a"), rel_type="TYPE_A") == [Uid("b")]
    assert graph.get_neighbors(Uid("b"), direction="both") == [Uid("a")]

    graph.delete_relationship(Uid("a"), Uid("b"), "TYPE_A")
    assert graph.get_relationship(Uid("a"), Uid("b"), "TYPE_A") is None
    assert graph.get_relationship(Uid("a"), Uid("b"), "TYPE_B") == {"weight": 2}

    graph.delete_relationship(Uid("a"), Uid("b"))
    assert graph.get_all_edges() == []
