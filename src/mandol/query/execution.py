"""Execution context and physical operators for in-memory query plans."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..domain.memory_unit import MemoryUnit
from ..domain.types import SpaceName, Uid
from ..ports.graph_store import GraphStore
from ..ports.unit_store import UnitStore
from ..ports.vector_index import VectorIndex
from .algebra import BindingRow, FieldRef, Predicate
from .vector_seeded import OperatorExecutionMetrics


@dataclass(slots=True)
class QueryExecutionContext:
    """Backend services, per-query caches, and observed operator metrics."""

    unit_store: UnitStore
    vector_index: VectorIndex
    graph_store: GraphStore
    operator_metrics: list[OperatorExecutionMetrics] = field(default_factory=list)
    _unit_cache: dict[Uid, MemoryUnit | None] = field(default_factory=dict)
    _space_cache: dict[tuple[SpaceName, bool], frozenset[Uid]] = field(default_factory=dict)

    def get_unit(self, uid: Uid | str) -> MemoryUnit | None:
        key = Uid(str(uid))
        if key not in self._unit_cache:
            self._unit_cache[key] = self.unit_store.get_unit(key)
        return self._unit_cache[key]

    def uids_in_space(self, name: SpaceName, *, recursive: bool) -> frozenset[Uid]:
        cache_key = (SpaceName(str(name)), bool(recursive))
        cached = self._space_cache.get(cache_key)
        if cached is not None:
            return cached
        space = self.unit_store.get_space(cache_key[0])
        if space is None:
            resolved: frozenset[Uid] = frozenset()
        elif recursive:
            resolved = frozenset(
                space.get_all_unit_uids(
                    recursive=True,
                    resolver=self.unit_store.get_space,
                )
            )
        else:
            resolved = frozenset(space.unit_uids)
        self._space_cache[cache_key] = resolved
        return resolved

    def record(
        self,
        operator: PhysicalOperator,
        *,
        input_rows: int,
        output_rows: int,
        elapsed_ms: float,
    ) -> None:
        self.operator_metrics.append(
            OperatorExecutionMetrics(
                operator_id=operator.operator_id,
                operator_type=type(operator).__name__,
                stage=operator.stage,
                algorithm=operator.algorithm,
                input_rows=int(input_rows),
                output_rows=int(output_rows),
                elapsed_ms=float(elapsed_ms),
            )
        )


class PhysicalOperator(ABC):
    """Materializing physical operator with exclusive timing instrumentation."""

    def __init__(
        self,
        operator_id: str,
        *,
        stage: str,
        algorithm: str,
        child: PhysicalOperator | None = None,
    ) -> None:
        self.operator_id = str(operator_id)
        self.stage = str(stage)
        self.algorithm = str(algorithm)
        self.child = child

    @abstractmethod
    def execute(self, context: QueryExecutionContext) -> list[BindingRow]:
        raise NotImplementedError

    @abstractmethod
    def describe(self) -> str:
        raise NotImplementedError

    def children(self) -> tuple[PhysicalOperator, ...]:
        return () if self.child is None else (self.child,)

    def _finish(
        self,
        context: QueryExecutionContext,
        *,
        started: float,
        input_rows: int,
        rows: list[BindingRow],
    ) -> list[BindingRow]:
        context.record(
            self,
            input_rows=input_rows,
            output_rows=len(rows),
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )
        return rows


class UnitScanOperator(PhysicalOperator):
    def __init__(self, operator_id: str, *, alias: str, space_name: str | None = None):
        super().__init__(operator_id, stage="relation_filter", algorithm="python_dict_scan")
        self.alias = str(alias)
        self.space_name = None if space_name is None else SpaceName(str(space_name))

    def execute(self, context: QueryExecutionContext) -> list[BindingRow]:
        started = time.perf_counter()
        units = context.unit_store.list_units()
        input_rows = len(units)
        allowed = (
            None
            if self.space_name is None
            else context.uids_in_space(self.space_name, recursive=True)
        )
        rows = [
            BindingRow(bindings={self.alias: Uid(str(unit.uid))})
            for unit in units
            if allowed is None or Uid(str(unit.uid)) in allowed
        ]
        return self._finish(
            context,
            started=started,
            input_rows=input_rows,
            rows=rows,
        )

    def describe(self) -> str:
        scope = "*" if self.space_name is None else str(self.space_name)
        return f"UnitScan(alias={self.alias}, space={scope})"


class VectorScanOperator(PhysicalOperator):
    """Score every indexed unit and bind the result UID and score."""

    def __init__(
        self,
        operator_id: str,
        *,
        query_vector: np.ndarray,
        alias: str,
        score_name: str,
    ) -> None:
        super().__init__(operator_id, stage="vector_search", algorithm="vector_index_full_scan")
        self.query_vector = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        self.alias = str(alias)
        self.score_name = str(score_name)

    def execute(self, context: QueryExecutionContext) -> list[BindingRow]:
        started = time.perf_counter()
        units = context.unit_store.list_units()
        input_rows = sum(unit.embedding is not None for unit in units)
        hits = context.vector_index.search(self.query_vector, top_k=max(1, len(units)))
        rows = [
            BindingRow(
                bindings={self.alias: Uid(str(uid))},
                values={self.score_name: float(score)},
            )
            for uid, score in hits
            if context.get_unit(uid) is not None
        ]
        return self._finish(
            context,
            started=started,
            input_rows=input_rows,
            rows=rows,
        )

    def describe(self) -> str:
        return f"VectorScan(alias={self.alias}, score={self.score_name})"


class ExactVectorScoreOperator(PhysicalOperator):
    """Batch-score only the units emitted by a selective child operator."""

    def __init__(
        self,
        operator_id: str,
        *,
        child: PhysicalOperator,
        query_vector: np.ndarray,
        alias: str,
        score_name: str,
    ) -> None:
        super().__init__(
            operator_id,
            stage="vector_search",
            algorithm="numpy_filtered_exact_cosine",
            child=child,
        )
        self.query_vector = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        self.alias = str(alias)
        self.score_name = str(score_name)

    def execute(self, context: QueryExecutionContext) -> list[BindingRow]:
        input_rows = self.child.execute(context)  # type: ignore[union-attr]
        started = time.perf_counter()
        score_rows: list[BindingRow] = []
        vectors: list[np.ndarray] = []
        for row in input_rows:
            uid = row.bindings.get(self.alias)
            unit = None if uid is None else context.get_unit(uid)
            if unit is None or unit.embedding is None:
                continue
            vector = np.asarray(unit.embedding, dtype=np.float32).reshape(-1)
            if vector.shape != self.query_vector.shape:
                continue
            score_rows.append(row)
            vectors.append(vector)

        rows: list[BindingRow] = []
        if vectors:
            matrix = np.stack(vectors)
            matrix_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            np.divide(matrix, matrix_norms, out=matrix, where=matrix_norms != 0)
            query = self.query_vector.copy()
            query_norm = float(np.linalg.norm(query))
            if query_norm != 0.0:
                query /= query_norm
            scores = matrix @ query
            rows = [
                row.with_value(self.score_name, float(score))
                for row, score in zip(score_rows, scores, strict=True)
            ]
        return self._finish(
            context,
            started=started,
            input_rows=len(input_rows),
            rows=rows,
        )

    def describe(self) -> str:
        return f"ExactVectorScore(alias={self.alias}, score={self.score_name})"


class AdaptiveAnnSeedOperator(PhysicalOperator):
    """Search ANN candidates, widening until enough rows pass the seed filter."""

    def __init__(
        self,
        operator_id: str,
        *,
        query_vector: np.ndarray,
        alias: str,
        score_name: str,
        predicate: Predicate,
        count: int,
        initial_oversample: int = 4,
    ) -> None:
        super().__init__(
            operator_id,
            stage="vector_search",
            algorithm="ann_adaptive_widening",
        )
        if int(count) <= 0:
            raise ValueError("count must be positive")
        if int(initial_oversample) <= 0:
            raise ValueError("initial_oversample must be positive")
        self.query_vector = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        self.alias = str(alias)
        self.score_name = str(score_name)
        self.predicate = predicate
        self.count = int(count)
        self.initial_oversample = int(initial_oversample)

    def execute(self, context: QueryExecutionContext) -> list[BindingRow]:
        started = time.perf_counter()
        total = sum(
            unit.embedding is not None for unit in context.unit_store.list_units()
        )
        recall = min(total, max(self.count, self.count * self.initial_oversample))
        rows: list[BindingRow] = []
        examined = 0
        while recall > 0:
            hits = context.vector_index.search(self.query_vector, top_k=recall)
            examined = len(hits)
            candidates = [
                BindingRow(
                    bindings={self.alias: Uid(str(uid))},
                    values={self.score_name: float(score)},
                )
                for uid, score in hits
                if context.get_unit(uid) is not None
            ]
            rows = [
                row for row in candidates if self.predicate.evaluate(row, context)
            ][: self.count]
            if len(rows) >= self.count or recall >= total or len(hits) < recall:
                break
            recall = min(total, recall * 2)
        return self._finish(
            context,
            started=started,
            input_rows=examined,
            rows=rows,
        )

    def describe(self) -> str:
        return (
            f"AdaptiveAnnSeed(alias={self.alias}, k={self.count}, "
            f"oversample={self.initial_oversample})"
        )


class FilterOperator(PhysicalOperator):
    def __init__(
        self,
        operator_id: str,
        *,
        child: PhysicalOperator,
        predicate: Predicate,
        stage: str,
    ) -> None:
        super().__init__(operator_id, stage=stage, algorithm="predicate_scan", child=child)
        self.predicate = predicate

    def execute(self, context: QueryExecutionContext) -> list[BindingRow]:
        input_rows = self.child.execute(context)  # type: ignore[union-attr]
        started = time.perf_counter()
        rows = [row for row in input_rows if self.predicate.evaluate(row, context)]
        return self._finish(
            context,
            started=started,
            input_rows=len(input_rows),
            rows=rows,
        )

    def describe(self) -> str:
        return f"Filter(predicate={type(self.predicate).__name__})"


class TopKOperator(PhysicalOperator):
    def __init__(
        self,
        operator_id: str,
        *,
        child: PhysicalOperator,
        count: int,
        score: FieldRef,
        uid_alias: str,
        descending: bool,
        stage: str,
    ) -> None:
        super().__init__(operator_id, stage=stage, algorithm="heap_top_k", child=child)
        self.count = int(count)
        self.score = score
        self.uid_alias = str(uid_alias)
        self.descending = bool(descending)

    def execute(self, context: QueryExecutionContext) -> list[BindingRow]:
        input_rows = self.child.execute(context)  # type: ignore[union-attr]
        started = time.perf_counter()

        def key(row: BindingRow) -> tuple[float, str]:
            score = _numeric(self.score.resolve(row, context))
            uid = str(row.bindings.get(self.uid_alias, ""))
            return ((-score if self.descending else score), uid)

        rows = sorted(input_rows, key=key)[: max(0, self.count)]
        return self._finish(
            context,
            started=started,
            input_rows=len(input_rows),
            rows=rows,
        )

    def describe(self) -> str:
        return f"TopK(k={self.count}, alias={self.uid_alias})"


class EdgeJoinOperator(PhysicalOperator):
    """Index-nested-loop join through typed graph adjacency."""

    def __init__(
        self,
        operator_id: str,
        *,
        child: PhysicalOperator,
        bound_alias: str,
        new_alias: str,
        rel_type: str,
        direction: str,
        stage: str,
    ) -> None:
        super().__init__(
            operator_id,
            stage=stage,
            algorithm="adjacency_index_nested_loop_join",
            child=child,
        )
        if direction not in {"out", "in"}:
            raise ValueError("direction must be 'out' or 'in'")
        self.bound_alias = str(bound_alias)
        self.new_alias = str(new_alias)
        self.rel_type = str(rel_type)
        self.direction = direction

    def execute(self, context: QueryExecutionContext) -> list[BindingRow]:
        input_rows = self.child.execute(context)  # type: ignore[union-attr]
        started = time.perf_counter()
        rows: list[BindingRow] = []
        for row in input_rows:
            bound_uid = row.bindings.get(self.bound_alias)
            if bound_uid is None:
                continue
            neighbors = context.graph_store.get_neighbors(
                bound_uid,
                rel_type=self.rel_type,
                direction=self.direction,
            )
            already_bound = row.bindings.get(self.new_alias)
            for neighbor in neighbors:
                if already_bound is not None and already_bound != neighbor:
                    continue
                if context.get_unit(neighbor) is None:
                    continue
                rows.append(row if already_bound is not None else row.bind(self.new_alias, neighbor))
        return self._finish(
            context,
            started=started,
            input_rows=len(input_rows),
            rows=rows,
        )

    def describe(self) -> str:
        return (
            f"EdgeJoin({self.bound_alias}, rel={self.rel_type}, "
            f"direction={self.direction}, bind={self.new_alias})"
        )


class InnerHashJoinOperator(PhysicalOperator):
    """Generic inner equality join over two materialized row streams."""

    def __init__(
        self,
        operator_id: str,
        *,
        left: PhysicalOperator,
        right: PhysicalOperator,
        left_key: FieldRef,
        right_key: FieldRef,
        stage: str = "relation_join",
    ) -> None:
        super().__init__(operator_id, stage=stage, algorithm="hash_join")
        self.left = left
        self.right = right
        self.left_key = left_key
        self.right_key = right_key

    def children(self) -> tuple[PhysicalOperator, ...]:
        return (self.left, self.right)

    def execute(self, context: QueryExecutionContext) -> list[BindingRow]:
        left_rows = self.left.execute(context)
        right_rows = self.right.execute(context)
        started = time.perf_counter()
        buckets: dict[Any, list[BindingRow]] = defaultdict(list)
        for row in right_rows:
            buckets[_hashable(self.right_key.resolve(row, context))].append(row)

        rows: list[BindingRow] = []
        for left_row in left_rows:
            key = _hashable(self.left_key.resolve(left_row, context))
            for right_row in buckets.get(key, ()):
                try:
                    rows.append(left_row.merge(right_row))
                except ValueError:
                    continue
        return self._finish(
            context,
            started=started,
            input_rows=len(left_rows) + len(right_rows),
            rows=rows,
        )

    def describe(self) -> str:
        return f"InnerHashJoin(left={self.left_key.path}, right={self.right_key.path})"


class GroupTopKOperator(PhysicalOperator):
    def __init__(
        self,
        operator_id: str,
        *,
        child: PhysicalOperator,
        group_alias: str,
        item_alias: str,
        score: FieldRef,
        count: int,
        score_output_name: str,
        descending: bool,
        stage: str,
    ) -> None:
        super().__init__(operator_id, stage=stage, algorithm="partitioned_top_k", child=child)
        self.group_alias = str(group_alias)
        self.item_alias = str(item_alias)
        self.score = score
        self.count = int(count)
        self.score_output_name = str(score_output_name)
        self.descending = bool(descending)

    def execute(self, context: QueryExecutionContext) -> list[BindingRow]:
        input_rows = self.child.execute(context)  # type: ignore[union-attr]
        started = time.perf_counter()
        groups: dict[Uid, list[BindingRow]] = defaultdict(list)
        for row in input_rows:
            group = row.bindings.get(self.group_alias)
            if group is not None:
                groups[group].append(row)

        rows: list[BindingRow] = []
        for group in sorted(groups, key=str):
            ranked: list[tuple[float, str, BindingRow]] = []
            for row in groups[group]:
                score = _numeric(self.score.resolve(row, context))
                item_uid = str(row.bindings.get(self.item_alias, ""))
                ranked.append((score, item_uid, row.with_value(self.score_output_name, score)))
            ranked.sort(key=lambda item: ((-item[0] if self.descending else item[0]), item[1]))
            rows.extend(item[2] for item in ranked[: max(0, self.count)])

        return self._finish(
            context,
            started=started,
            input_rows=len(input_rows),
            rows=rows,
        )

    def describe(self) -> str:
        return (
            f"GroupTopK(group={self.group_alias}, item={self.item_alias}, "
            f"k={self.count})"
        )


class TraverseOperator(PhysicalOperator):
    """Enumerate all bounded simple paths from each input row's start node."""

    def __init__(
        self,
        operator_id: str,
        *,
        child: PhysicalOperator,
        start_alias: str,
        current_alias: str,
        rel_type: str,
        direction: str,
        min_hops: int,
        max_hops: int,
        stage: str,
    ) -> None:
        super().__init__(operator_id, stage=stage, algorithm="bounded_path_bfs", child=child)
        if direction not in {"out", "in"}:
            raise ValueError("direction must be 'out' or 'in'")
        self.start_alias = str(start_alias)
        self.current_alias = str(current_alias)
        self.rel_type = str(rel_type)
        self.direction = direction
        self.min_hops = int(min_hops)
        self.max_hops = int(max_hops)

    def execute(self, context: QueryExecutionContext) -> list[BindingRow]:
        input_rows = self.child.execute(context)  # type: ignore[union-attr]
        started = time.perf_counter()
        rows: list[BindingRow] = []
        for input_row in input_rows:
            start_uid = input_row.bindings.get(self.start_alias)
            if start_uid is None:
                continue
            queue: deque[tuple[Uid, int, tuple[Uid, ...]]] = deque(
                [(start_uid, 0, (start_uid,))]
            )
            while queue:
                current_uid, depth, path = queue.popleft()
                current_row = (
                    input_row.bind(self.current_alias, current_uid)
                    .with_value("depth", depth)
                    .with_value("path", path)
                )
                if depth >= self.min_hops:
                    rows.append(current_row)
                if depth >= self.max_hops:
                    continue
                neighbors = context.graph_store.get_neighbors(
                    current_uid,
                    rel_type=self.rel_type,
                    direction=self.direction,
                )
                for neighbor in neighbors:
                    if neighbor in path or context.get_unit(neighbor) is None:
                        continue
                    queue.append((neighbor, depth + 1, (*path, neighbor)))

        return self._finish(
            context,
            started=started,
            input_rows=len(input_rows),
            rows=rows,
        )

    def describe(self) -> str:
        return (
            f"Traverse(start={self.start_alias}, rel={self.rel_type}, "
            f"hops={self.min_hops}..{self.max_hops})"
        )


class ProjectLimitOperator(PhysicalOperator):
    """Apply the public result ordering and limit while preserving full input count."""

    def __init__(
        self,
        operator_id: str,
        *,
        child: PhysicalOperator,
        result_limit: int,
        stage: str,
    ) -> None:
        super().__init__(operator_id, stage=stage, algorithm="stable_sort_limit", child=child)
        self.result_limit = int(result_limit)

    def execute(self, context: QueryExecutionContext) -> list[BindingRow]:
        input_rows = self.child.execute(context)  # type: ignore[union-attr]
        started = time.perf_counter()

        def key(row: BindingRow) -> tuple[Any, ...]:
            return (
                -_numeric(row.values.get("seed_score")),
                str(row.bindings.get("seed", "")),
                -_numeric(row.values.get("node_score")),
                str(row.bindings.get("node", "")),
                int(row.values.get("depth", 0)),
                str(row.bindings.get("current", "")),
                tuple(str(uid) for uid in row.values.get("path", ())),
            )

        rows = sorted(input_rows, key=key)[: max(0, self.result_limit)]
        return self._finish(
            context,
            started=started,
            input_rows=len(input_rows),
            rows=rows,
        )

    def describe(self) -> str:
        return f"ProjectLimit(limit={self.result_limit})"


def explain_physical_plan(root: PhysicalOperator) -> str:
    """Render a stable root-first physical operator tree."""
    lines: list[str] = []

    def visit(node: PhysicalOperator, prefix: str, is_last: bool) -> None:
        connector = "└─ " if is_last else "├─ "
        lines.append(
            f"{prefix}{connector}{node.operator_id}: {node.describe()} "
            f"[{node.algorithm}]"
        )
        children = node.children()
        child_prefix = prefix + ("   " if is_last else "│  ")
        for index, child in enumerate(children):
            visit(child, child_prefix, index == len(children) - 1)

    visit(root, "", True)
    return "\n".join(lines)


def sum_stage_elapsed(
    metrics: Sequence[OperatorExecutionMetrics],
    stage: str,
) -> float:
    return sum(metric.elapsed_ms for metric in metrics if metric.stage == stage)


def _numeric(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if np.isfinite(number) else 0.0


def _hashable(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_hashable(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((key, _hashable(item)) for key, item in value.items()))
    return value
