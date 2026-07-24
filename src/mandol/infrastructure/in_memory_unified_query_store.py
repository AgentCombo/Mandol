"""Pure in-memory execution of Mandol's fixed multimodal query plan."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

from ..ports.graph_store import GraphStore
from ..ports.unified_query_store import UnifiedQueryStore
from ..ports.unit_store import UnitStore
from ..ports.vector_index import VectorIndex
from ..query.algebra import (
    EdgeJoin,
    FieldRef,
    Filter,
    GroupTopK,
    InnerJoin,
    LogicalOperator,
    ProjectLimit,
    TopK,
    Traverse,
    UnitScan,
    VectorScan,
    metadata_equals,
)
from ..query.execution import (
    AdaptiveAnnSeedOperator,
    EdgeJoinOperator,
    ExactVectorScoreOperator,
    FilterOperator,
    GroupTopKOperator,
    InnerHashJoinOperator,
    PhysicalOperator,
    ProjectLimitOperator,
    QueryExecutionContext,
    TopKOperator,
    TraverseOperator,
    UnitScanOperator,
    VectorScanOperator,
    explain_physical_plan,
    sum_stage_elapsed,
)
from ..query.vector_seeded import (
    QueryExecutionMetrics,
    QueryStageBreakdown,
    QueryStageTimings,
    VectorSeededTraversalExecution,
    VectorSeededTraversalRow,
    VectorSeededTraversalSpec,
)


@dataclass(frozen=True, slots=True)
class _CompiledPlan:
    logical: LogicalOperator
    physical: PhysicalOperator


VectorSeedStrategy = Literal[
    "exact_full_scan",
    "filter_first_exact",
    "ann_adaptive",
]


class InMemoryUnifiedQueryStore(UnifiedQueryStore):
    """Compile and execute a vector-seeded plan over Mandol's in-memory ports."""

    def __init__(
        self,
        *,
        unit_store: UnitStore,
        vector_index: VectorIndex,
        graph_store: GraphStore,
        vector_strategy: VectorSeedStrategy = "exact_full_scan",
        ann_initial_oversample: int = 4,
    ) -> None:
        if vector_strategy not in {
            "exact_full_scan",
            "filter_first_exact",
            "ann_adaptive",
        }:
            raise ValueError(f"unsupported vector strategy: {vector_strategy}")
        if int(ann_initial_oversample) <= 0:
            raise ValueError("ann_initial_oversample must be positive")
        self._unit_store = unit_store
        self._vector_index = vector_index
        self._graph_store = graph_store
        self._vector_strategy = vector_strategy
        self._ann_initial_oversample = int(ann_initial_oversample)

    def vector_seeded_graph_traversal(
        self,
        spec: VectorSeededTraversalSpec,
        *,
        profile: bool = False,
    ) -> VectorSeededTraversalExecution:
        """Run the fixed vector → typed joins → bounded traversal plan."""
        del profile  # In-memory instrumentation is always enabled and inexpensive.
        compiled = self._compile(spec)
        context = QueryExecutionContext(
            unit_store=self._unit_store,
            vector_index=self._vector_index,
            graph_store=self._graph_store,
        )

        started = time.perf_counter()
        result_rows = compiled.physical.execute(context)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        metrics_by_id = {metric.operator_id: metric for metric in context.operator_metrics}

        rows = tuple(
            VectorSeededTraversalRow(
                seed_uid=row.bindings["seed"],
                seed_score=float(row.values["seed_score"]),
                session_uid=row.bindings["session"],
                node_uid=row.bindings["node"],
                node_score=float(row.values["node_score"]),
                current_uid=row.bindings["current"],
                depth=int(row.values["depth"]),
                path=tuple(row.values["path"]),
            )
            for row in result_rows
        )

        traversal_rows = _metric_output(metrics_by_id, "op08_traverse")
        complete = traversal_rows <= spec.result_limit
        operator_metrics = tuple(context.operator_metrics)
        stage_timings = QueryStageTimings(
            vector_search_cpu_ms=sum_stage_elapsed(operator_metrics, "vector_search"),
            relation_join_cpu_ms=sum_stage_elapsed(operator_metrics, "relation_join"),
            graph_traversal_cpu_ms=sum_stage_elapsed(operator_metrics, "graph_traversal"),
            other_cpu_ms=sum_stage_elapsed(operator_metrics, "other"),
            total_cpu_ms=sum(metric.elapsed_ms for metric in operator_metrics),
        )
        stages = QueryStageBreakdown(
            vector_candidate_rows=self._vector_candidate_rows(metrics_by_id),
            vector_seed_rows=self._vector_seed_rows(metrics_by_id),
            session_join_rows=_metric_output(metrics_by_id, "op04_session_join"),
            node_join_rows=_metric_output(metrics_by_id, "op06_node_filter"),
            selected_node_rows=_metric_output(metrics_by_id, "op07_group_top_k"),
            traversal_rows=traversal_rows,
        )
        return VectorSeededTraversalExecution(
            rows=rows,
            metrics=QueryExecutionMetrics(
                elapsed_ms=elapsed_ms,
                rows_returned=len(rows),
                complete=complete,
                truncated_reason=None if complete else "result_limit",
                stages=stages,
                timings=stage_timings,
                operators=operator_metrics,
            ),
        )

    def explain_vector_seeded_graph_traversal(self, spec: VectorSeededTraversalSpec) -> str:
        compiled = self._compile(spec)
        label = (
            "IN_MEMORY_FIXED_PLAN"
            if self._vector_strategy == "exact_full_scan"
            else f"IN_MEMORY_CANDIDATE_PLAN[{self._vector_strategy}]"
        )
        return label + "\n" + explain_physical_plan(compiled.physical)

    def logical_plan(self, spec: VectorSeededTraversalSpec) -> LogicalOperator:
        """Return the backend-neutral tree, primarily for optimizer development."""
        return self._build_logical_plan(spec)

    def compile_logical_plan(self, plan: LogicalOperator) -> PhysicalOperator:
        """Compile an arbitrary supported logical tree to in-memory operators."""
        return self._compile_operator(plan)

    def _compile(self, spec: VectorSeededTraversalSpec) -> _CompiledPlan:
        logical = self._build_logical_plan(spec)
        physical = (
            self._compile_operator(logical)
            if self._vector_strategy == "exact_full_scan"
            else self._compile_candidate_plan(spec)
        )
        return _CompiledPlan(logical=logical, physical=physical)

    def _compile_candidate_plan(
        self,
        spec: VectorSeededTraversalSpec,
    ) -> PhysicalOperator:
        seed_predicate = metadata_equals("seed", spec.seed_metadata_equals)
        if self._vector_strategy == "filter_first_exact":
            scan = UnitScanOperator("op01_unit_scan", alias="seed")
            filtered = FilterOperator(
                "op02_seed_filter",
                child=scan,
                predicate=seed_predicate,
                stage="vector_search",
            )
            scored = ExactVectorScoreOperator(
                "op02b_exact_vector_score",
                child=filtered,
                query_vector=spec.query_vector,
                alias="seed",
                score_name="seed_score",
            )
            seed_root: PhysicalOperator = TopKOperator(
                "op03_vector_top_k",
                child=scored,
                count=spec.vector_k,
                score=FieldRef(None, "seed_score"),
                uid_alias="seed",
                descending=True,
                stage="vector_search",
            )
        else:
            seed_root = AdaptiveAnnSeedOperator(
                "op01_ann_seed",
                query_vector=spec.query_vector,
                alias="seed",
                score_name="seed_score",
                predicate=seed_predicate,
                count=spec.vector_k,
                initial_oversample=self._ann_initial_oversample,
            )

        session_join = EdgeJoinOperator(
            "op04_session_join",
            child=seed_root,
            bound_alias="seed",
            new_alias="session",
            rel_type=spec.session_relation,
            direction=spec.session_relation_direction,
            stage="relation_join",
        )
        node_join = EdgeJoinOperator(
            "op05_node_join",
            child=session_join,
            bound_alias="session",
            new_alias="node",
            rel_type=spec.node_relation,
            direction=spec.node_relation_direction,
            stage="relation_join",
        )
        node_filter = FilterOperator(
            "op06_node_filter",
            child=node_join,
            predicate=metadata_equals("node", spec.node_metadata_equals),
            stage="relation_join",
        )
        selected = GroupTopKOperator(
            "op07_group_top_k",
            child=node_filter,
            group_alias="seed",
            item_alias="node",
            score=FieldRef("node", f"metadata.{spec.node_score_field}"),
            count=spec.nodes_per_seed,
            score_output_name="node_score",
            descending=True,
            stage="relation_join",
        )
        traversed = TraverseOperator(
            "op08_traverse",
            child=selected,
            start_alias="node",
            current_alias="current",
            rel_type=spec.traversal_relation,
            direction=spec.traversal_direction,
            min_hops=spec.min_hops,
            max_hops=spec.max_hops,
            stage="graph_traversal",
        )
        return ProjectLimitOperator(
            "op09_project_limit",
            child=traversed,
            result_limit=spec.result_limit,
            stage="other",
        )

    def _vector_candidate_rows(self, metrics_by_id: dict[str, object]) -> int:
        if self._vector_strategy == "filter_first_exact":
            return _metric_output(metrics_by_id, "op02b_exact_vector_score")
        if self._vector_strategy == "ann_adaptive":
            return _metric_input(metrics_by_id, "op01_ann_seed")
        return _metric_output(metrics_by_id, "op02_seed_filter")

    def _vector_seed_rows(self, metrics_by_id: dict[str, object]) -> int:
        if self._vector_strategy == "ann_adaptive":
            return _metric_output(metrics_by_id, "op01_ann_seed")
        return _metric_output(metrics_by_id, "op03_vector_top_k")

    def _build_logical_plan(self, spec: VectorSeededTraversalSpec) -> LogicalOperator:
        scan = VectorScan(
            query_vector=spec.query_vector,
            alias="seed",
            score_name="seed_score",
        )
        seed_filter = Filter(
            input=scan,
            predicate=metadata_equals("seed", spec.seed_metadata_equals),
            stage="vector_search",
        )
        vector_top_k = TopK(
            input=seed_filter,
            count=spec.vector_k,
            score=FieldRef(None, "seed_score"),
            uid_alias="seed",
            stage="vector_search",
        )
        session_join = EdgeJoin(
            input=vector_top_k,
            bound_alias="seed",
            new_alias="session",
            rel_type=spec.session_relation,
            direction=spec.session_relation_direction,
            stage="relation_join",
        )
        node_join = EdgeJoin(
            input=session_join,
            bound_alias="session",
            new_alias="node",
            rel_type=spec.node_relation,
            direction=spec.node_relation_direction,
            stage="relation_join",
        )
        node_filter = Filter(
            input=node_join,
            predicate=metadata_equals("node", spec.node_metadata_equals),
            stage="relation_join",
        )
        group_top_k = GroupTopK(
            input=node_filter,
            group_alias="seed",
            item_alias="node",
            score=FieldRef("node", f"metadata.{spec.node_score_field}"),
            count=spec.nodes_per_seed,
            score_output_name="node_score",
            stage="relation_join",
        )
        traversal = Traverse(
            input=group_top_k,
            start_alias="node",
            current_alias="current",
            rel_type=spec.traversal_relation,
            direction=spec.traversal_direction,
            min_hops=spec.min_hops,
            max_hops=spec.max_hops,
        )
        return ProjectLimit(input=traversal, result_limit=spec.result_limit)

    def _compile_operator(self, node: LogicalOperator) -> PhysicalOperator:
        if isinstance(node, UnitScan):
            return UnitScanOperator(
                f"unit_scan_{node.alias}",
                alias=node.alias,
                space_name=node.space_name,
            )
        if isinstance(node, VectorScan):
            return VectorScanOperator(
                "op01_vector_scan",
                query_vector=node.query_vector,
                alias=node.alias,
                score_name=node.score_name,
            )
        if isinstance(node, Filter):
            operator_id = (
                "op02_seed_filter"
                if node.stage == "vector_search"
                else "op06_node_filter"
            )
            return FilterOperator(
                operator_id,
                child=self._compile_operator(node.input),
                predicate=node.predicate,
                stage=node.stage,
            )
        if isinstance(node, TopK):
            return TopKOperator(
                "op03_vector_top_k",
                child=self._compile_operator(node.input),
                count=node.count,
                score=node.score,
                uid_alias=node.uid_alias,
                descending=node.descending,
                stage=node.stage,
            )
        if isinstance(node, EdgeJoin):
            operator_id = (
                "op04_session_join"
                if node.new_alias == "session"
                else "op05_node_join"
            )
            return EdgeJoinOperator(
                operator_id,
                child=self._compile_operator(node.input),
                bound_alias=node.bound_alias,
                new_alias=node.new_alias,
                rel_type=node.rel_type,
                direction=node.direction,
                stage=node.stage,
            )
        if isinstance(node, InnerJoin):
            return InnerHashJoinOperator(
                "inner_hash_join",
                left=self._compile_operator(node.left),
                right=self._compile_operator(node.right),
                left_key=node.left_key,
                right_key=node.right_key,
                stage=node.stage,
            )
        if isinstance(node, GroupTopK):
            return GroupTopKOperator(
                "op07_group_top_k",
                child=self._compile_operator(node.input),
                group_alias=node.group_alias,
                item_alias=node.item_alias,
                score=node.score,
                count=node.count,
                score_output_name=node.score_output_name,
                descending=node.descending,
                stage=node.stage,
            )
        if isinstance(node, Traverse):
            return TraverseOperator(
                "op08_traverse",
                child=self._compile_operator(node.input),
                start_alias=node.start_alias,
                current_alias=node.current_alias,
                rel_type=node.rel_type,
                direction=node.direction,
                min_hops=node.min_hops,
                max_hops=node.max_hops,
                stage=node.stage,
            )
        if isinstance(node, ProjectLimit):
            return ProjectLimitOperator(
                "op09_project_limit",
                child=self._compile_operator(node.input),
                result_limit=node.result_limit,
                stage=node.stage,
            )
        raise TypeError(f"unsupported logical operator: {type(node).__name__}")


def _metric_output(metrics_by_id: dict[str, object], operator_id: str) -> int:
    metric = metrics_by_id.get(operator_id)
    return 0 if metric is None else int(metric.output_rows)


def _metric_input(metrics_by_id: dict[str, object], operator_id: str) -> int:
    metric = metrics_by_id.get(operator_id)
    return 0 if metric is None else int(metric.input_rows)
