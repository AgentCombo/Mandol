"""Contracts for vector-seeded relation joins and graph traversal."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from ..domain.types import Embedding, Uid

Direction = Literal["out", "in"]


@dataclass(frozen=True, slots=True)
class VectorSeededTraversalSpec:
    """Describe the fixed multimodal query used by the first DuckDB executor.

    The relation directions are relative to the row already bound.  For
    example, ``session_relation_direction="in"`` means that the session is
    the source of an edge whose target is the vector-seeded task:
    ``session -[BELONGS_TO]-> task``.
    """

    query_vector: Embedding
    vector_k: int = 10
    seed_metadata_equals: Mapping[str, Any] = field(default_factory=dict)

    session_relation: str = "BELONGS_TO"
    session_relation_direction: Direction = "in"
    node_relation: str = "IN_SESSION"
    node_relation_direction: Direction = "in"
    node_metadata_equals: Mapping[str, Any] = field(default_factory=dict)

    node_score_field: str = "fitness_score"
    nodes_per_seed: int = 1

    traversal_relation: str = "HAS_CHILD"
    traversal_direction: Direction = "out"
    min_hops: int = 0
    max_hops: int = 2
    result_limit: int = 100

    def __post_init__(self) -> None:
        vector = np.asarray(self.query_vector, dtype=np.float32).reshape(-1)
        object.__setattr__(self, "query_vector", vector)

        if self.vector_k <= 0:
            raise ValueError("vector_k must be positive")
        if self.nodes_per_seed <= 0:
            raise ValueError("nodes_per_seed must be positive")
        if self.min_hops < 0:
            raise ValueError("min_hops must be non-negative")
        if self.max_hops < self.min_hops:
            raise ValueError("max_hops must be greater than or equal to min_hops")
        if self.result_limit <= 0:
            raise ValueError("result_limit must be positive")

        for name, direction in (
            ("session_relation_direction", self.session_relation_direction),
            ("node_relation_direction", self.node_relation_direction),
            ("traversal_direction", self.traversal_direction),
        ):
            if direction not in {"out", "in"}:
                raise ValueError(f"{name} must be 'out' or 'in'")

        for name, value in (
            ("session_relation", self.session_relation),
            ("node_relation", self.node_relation),
            ("traversal_relation", self.traversal_relation),
            ("node_score_field", self.node_score_field),
        ):
            if not str(value).strip():
                raise ValueError(f"{name} must be non-empty")


@dataclass(frozen=True, slots=True)
class VectorSeededTraversalRow:
    """One path-preserving result row from the unified physical query."""

    seed_uid: Uid
    seed_score: float
    session_uid: Uid
    node_uid: Uid
    node_score: float
    current_uid: Uid
    depth: int
    path: tuple[Uid, ...]


@dataclass(frozen=True, slots=True)
class QueryStageBreakdown:
    """Actual CTE cardinalities observed inside one unified SQL statement."""

    vector_candidate_rows: int = 0
    vector_seed_rows: int = 0
    session_join_rows: int = 0
    node_join_rows: int = 0
    selected_node_rows: int = 0
    traversal_rows: int = 0


@dataclass(frozen=True, slots=True)
class QueryStageTimings:
    """Backend timing attributed to each logical stage.

    DuckDB reports profiler CPU time; the in-memory executor reports elapsed
    operator time.  Field names retain ``cpu_ms`` for API compatibility.
    """

    vector_search_cpu_ms: float | None = None
    relation_join_cpu_ms: float | None = None
    graph_traversal_cpu_ms: float | None = None
    other_cpu_ms: float | None = None
    total_cpu_ms: float | None = None


@dataclass(frozen=True, slots=True)
class OperatorExecutionMetrics:
    """Observed input/output cardinality and elapsed time for one operator."""

    operator_id: str
    operator_type: str
    stage: str
    algorithm: str
    input_rows: int
    output_rows: int
    elapsed_ms: float


@dataclass(frozen=True, slots=True)
class QueryExecutionMetrics:
    """Observed metrics from one physical execution."""

    elapsed_ms: float
    rows_returned: int
    complete: bool
    truncated_reason: str | None = None
    stages: QueryStageBreakdown = field(default_factory=QueryStageBreakdown)
    timings: QueryStageTimings = field(default_factory=QueryStageTimings)
    operators: tuple[OperatorExecutionMetrics, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class VectorSeededTraversalExecution:
    """Rows and observed metrics returned together for optimizer feedback."""

    rows: tuple[VectorSeededTraversalRow, ...]
    metrics: QueryExecutionMetrics


def format_query_log(
    spec: VectorSeededTraversalSpec,
    execution: VectorSeededTraversalExecution,
) -> str:
    """Format a compact stage-by-stage execution log."""
    metrics = execution.metrics
    stages = metrics.stages
    timings = metrics.timings
    vector_algorithm = _vector_algorithm(execution)
    lines = [
            (
                "query total_ms="
                f"{metrics.elapsed_ms:.2f} rows={metrics.rows_returned} "
                f"complete={metrics.complete}"
            ),
            (
                f"stage=vector_search algorithm={vector_algorithm} "
                f"candidate_rows={stages.vector_candidate_rows} "
                f"requested_k={spec.vector_k} output_rows={stages.vector_seed_rows} "
                f"cpu_ms={_format_ms(timings.vector_search_cpu_ms)}"
            ),
            (
                "stage=relation_join "
                f"session_edge={spec.session_relation} "
                f"session_rows={stages.session_join_rows} "
                f"node_edge={spec.node_relation} "
                f"node_rows={stages.node_join_rows} "
                f"selected_rows={stages.selected_node_rows} "
                f"cpu_ms={_format_ms(timings.relation_join_cpu_ms)}"
            ),
            (
                "stage=graph_traversal "
                f"edge={spec.traversal_relation} "
                f"hops={spec.min_hops}..{spec.max_hops} "
                f"input_seeds={stages.selected_node_rows} "
                f"output_rows={stages.traversal_rows} "
                f"cpu_ms={_format_ms(timings.graph_traversal_cpu_ms)}"
            ),
            (
                "stage=other "
                f"cpu_ms={_format_ms(timings.other_cpu_ms)} "
                f"profile_total_cpu_ms={_format_ms(timings.total_cpu_ms)}"
            ),
        ]
    lines.extend(
        (
            f"operator={operator.operator_id} type={operator.operator_type} "
            f"algorithm={operator.algorithm} stage={operator.stage} "
            f"input_rows={operator.input_rows} output_rows={operator.output_rows} "
            f"elapsed_ms={operator.elapsed_ms:.3f}"
        )
        for operator in metrics.operators
    )
    return "\n".join(lines)


def render_query_diagram(
    spec: VectorSeededTraversalSpec,
    execution: VectorSeededTraversalExecution,
) -> str:
    """Render the logical query annotated with actual row counts."""
    stages = execution.metrics.stages
    timings = execution.metrics.timings
    vector_algorithm = _vector_algorithm(execution).replace("_", " ")
    width = 62
    connector = f"{' ' * (width // 2 + 1)}│\n{' ' * (width // 2 + 1)}▼"
    boxes = [
        _diagram_box(
            "VECTOR SEARCH",
            [
                (
                    f"{vector_algorithm} · candidates={stages.vector_candidate_rows} "
                    f"· k={spec.vector_k} → seeds={stages.vector_seed_rows} "
                    f"· cpu={_format_ms(timings.vector_search_cpu_ms)}ms"
                )
            ],
            width=width,
        ),
        _diagram_box(
            "RELATION JOIN",
            [
                f"{spec.session_relation} → session rows={stages.session_join_rows}",
                f"{spec.node_relation} → node rows={stages.node_join_rows}",
                (
                    f"per-seed top-{spec.nodes_per_seed} "
                    f"→ selected={stages.selected_node_rows} "
                    f"· cpu={_format_ms(timings.relation_join_cpu_ms)}ms"
                ),
            ],
            width=width,
        ),
        _diagram_box(
            "GRAPH TRAVERSAL",
            [
                (
                    f"{spec.traversal_relation} "
                    f"· hops={spec.min_hops}..{spec.max_hops} "
                    f"→ rows={stages.traversal_rows} "
                    f"· cpu={_format_ms(timings.graph_traversal_cpu_ms)}ms"
                )
            ],
            width=width,
        ),
    ]
    result = (
        f"result rows={execution.metrics.rows_returned} ({execution.metrics.elapsed_ms:.2f} ms)"
    )
    return f"\n{connector}\n".join(boxes) + f"\n{connector}\n{result.center(width + 2)}"


def _vector_algorithm(execution: VectorSeededTraversalExecution) -> str:
    algorithms = {
        metric.algorithm
        for metric in execution.metrics.operators
        if metric.stage == "vector_search"
    }
    if "ann_adaptive_widening" in algorithms:
        return "ann_adaptive_widening"
    if "numpy_filtered_exact_cosine" in algorithms:
        return "filter_first_exact_cosine"
    return "exact_cosine_full_scan"


def _diagram_box(title: str, lines: list[str], *, width: int) -> str:
    content_width = width - 2
    rendered = [f"│ {title[:content_width]:<{content_width}} │"]
    rendered.extend(f"│ {line[:content_width]:<{content_width}} │" for line in lines)
    return "\n".join(
        [
            f"┌{'─' * width}┐",
            *rendered,
            f"└{'─' * width}┘",
        ]
    )


def _format_ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"
