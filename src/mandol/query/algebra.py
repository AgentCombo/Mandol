"""Backend-neutral relational algebra for multimodal Mandol queries.

The algebra deliberately stays small: it models the operators needed by the
first vector-seeded graph traversal while keeping rows, predicates, and plan
nodes independent from a concrete storage backend.  A future optimizer can
therefore reorder or replace these logical nodes without changing query
results or the public query specification.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from ..domain.memory_unit import MemoryUnit
from ..domain.types import Embedding, SpaceName, Uid

if TYPE_CHECKING:
    from .execution import QueryExecutionContext

Direction = Literal["out", "in"]
ComparisonOperator = Literal["eq", "neq", "in", "contains", "gt", "gte", "lt", "lte"]

_MISSING = object()


def resolve_field_path(value: Any, path: str) -> Any:
    """Resolve a dot-separated mapping/attribute path or return a missing sentinel."""
    current = value
    for part in str(path).split("."):
        if isinstance(current, Mapping):
            if part not in current:
                return _MISSING
            current = current[part]
        elif hasattr(current, part):
            current = getattr(current, part)
        else:
            return _MISSING
    return current


def evaluate_comparison(actual: Any, operator: ComparisonOperator, target: Any) -> bool:
    """Evaluate one comparison with SQL-like missing-field semantics."""
    if actual is _MISSING:
        return False
    try:
        if operator == "eq":
            return actual == target
        if operator == "neq":
            return actual != target
        if operator == "in":
            return target is not None and actual in target
        if operator == "contains":
            return actual is not None and str(target).lower() in str(actual).lower()
        if operator == "gt":
            return actual is not None and actual > target
        if operator == "gte":
            return actual is not None and actual >= target
        if operator == "lt":
            return actual is not None and actual < target
        if operator == "lte":
            return actual is not None and actual <= target
    except (TypeError, ValueError):
        return False
    raise ValueError(f"unsupported comparison operator: {operator}")


@dataclass(frozen=True, slots=True)
class BindingRow:
    """One relational row containing named unit bindings and scalar values."""

    bindings: Mapping[str, Uid] = field(default_factory=dict)
    values: Mapping[str, Any] = field(default_factory=dict)

    def bind(self, alias: str, uid: Uid | str) -> BindingRow:
        bindings = dict(self.bindings)
        bindings[str(alias)] = Uid(str(uid))
        return BindingRow(bindings=bindings, values=dict(self.values))

    def with_value(self, name: str, value: Any) -> BindingRow:
        values = dict(self.values)
        values[str(name)] = value
        return BindingRow(bindings=dict(self.bindings), values=values)

    def merge(self, other: BindingRow) -> BindingRow:
        bindings = dict(self.bindings)
        for alias, uid in other.bindings.items():
            existing = bindings.get(alias)
            if existing is not None and existing != uid:
                raise ValueError(f"conflicting binding for alias {alias!r}")
            bindings[alias] = uid
        values = dict(self.values)
        for name, value in other.values.items():
            existing = values.get(name, _MISSING)
            if existing is not _MISSING and existing != value:
                raise ValueError(f"conflicting scalar value for {name!r}")
            values[name] = value
        return BindingRow(bindings=bindings, values=values)


@dataclass(frozen=True, slots=True)
class FieldRef:
    """Reference a scalar value or a field on a bound MemoryUnit.

    ``FieldRef(None, "seed_score")`` resolves a scalar row value.
    ``FieldRef("node", "metadata.kind")`` resolves a nested unit field.
    ``FieldRef("node", "uid")`` returns the bound UID.
    """

    alias: str | None
    path: str

    def resolve(self, row: BindingRow, context: QueryExecutionContext) -> Any:
        if self.alias is None:
            return row.values.get(self.path, _MISSING)

        uid = row.bindings.get(self.alias)
        if uid is None:
            return _MISSING
        if self.path in {"", "uid"}:
            return uid

        unit = context.get_unit(uid)
        if unit is None:
            return _MISSING
        return resolve_field_path(unit, self.path)


class Predicate(ABC):
    """Boolean expression evaluated against one binding row."""

    @abstractmethod
    def evaluate(self, row: BindingRow, context: QueryExecutionContext) -> bool:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class AlwaysTrue(Predicate):
    def evaluate(self, row: BindingRow, context: QueryExecutionContext) -> bool:
        return True


@dataclass(frozen=True, slots=True)
class ComparisonPredicate(Predicate):
    field: FieldRef
    operator: ComparisonOperator
    value: Any

    def __post_init__(self) -> None:
        if self.operator not in {"eq", "neq", "in", "contains", "gt", "gte", "lt", "lte"}:
            raise ValueError(f"unsupported comparison operator: {self.operator}")

    def evaluate(self, row: BindingRow, context: QueryExecutionContext) -> bool:
        actual = self.field.resolve(row, context)
        return evaluate_comparison(actual, self.operator, self.value)


@dataclass(frozen=True, slots=True)
class AndPredicate(Predicate):
    predicates: tuple[Predicate, ...]

    def evaluate(self, row: BindingRow, context: QueryExecutionContext) -> bool:
        return all(predicate.evaluate(row, context) for predicate in self.predicates)


@dataclass(frozen=True, slots=True)
class OrPredicate(Predicate):
    predicates: tuple[Predicate, ...]

    def evaluate(self, row: BindingRow, context: QueryExecutionContext) -> bool:
        return any(predicate.evaluate(row, context) for predicate in self.predicates)


@dataclass(frozen=True, slots=True)
class NotPredicate(Predicate):
    predicate: Predicate

    def evaluate(self, row: BindingRow, context: QueryExecutionContext) -> bool:
        return not self.predicate.evaluate(row, context)


@dataclass(frozen=True, slots=True)
class InSpacePredicate(Predicate):
    alias: str
    space_name: SpaceName | str
    recursive: bool = True

    def evaluate(self, row: BindingRow, context: QueryExecutionContext) -> bool:
        uid = row.bindings.get(self.alias)
        return uid is not None and uid in context.uids_in_space(
            SpaceName(str(self.space_name)),
            recursive=self.recursive,
        )


@dataclass(frozen=True, slots=True)
class EdgeExistsPredicate(Predicate):
    """Test whether a typed edge exists from a bound node.

    If ``other_alias`` is omitted, any neighbor satisfies the predicate.
    Direction is relative to ``alias``.
    """

    alias: str
    rel_type: str
    direction: Direction = "out"
    other_alias: str | None = None

    def evaluate(self, row: BindingRow, context: QueryExecutionContext) -> bool:
        uid = row.bindings.get(self.alias)
        if uid is None:
            return False
        neighbors = context.graph_store.get_neighbors(
            uid,
            rel_type=self.rel_type,
            direction=self.direction,
        )
        if self.other_alias is None:
            return bool(neighbors)
        other = row.bindings.get(self.other_alias)
        return other is not None and other in neighbors


def metadata_equals(alias: str, values: Mapping[str, Any]) -> Predicate:
    """Compile a metadata equality mapping into a conjunction."""
    predicates = tuple(
        ComparisonPredicate(
            field=FieldRef(alias, key if str(key).startswith("metadata.") else f"metadata.{key}"),
            operator="eq",
            value=value,
        )
        for key, value in values.items()
    )
    return AlwaysTrue() if not predicates else AndPredicate(predicates)


class LogicalOperator(ABC):
    """Base class for immutable backend-neutral logical plan nodes."""

    @abstractmethod
    def children(self) -> tuple[LogicalOperator, ...]:
        raise NotImplementedError

    @abstractmethod
    def describe(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class UnitScan(LogicalOperator):
    alias: str
    space_name: str | None = None

    def children(self) -> tuple[LogicalOperator, ...]:
        return ()

    def describe(self) -> str:
        scope = "*" if self.space_name is None else self.space_name
        return f"UnitScan(alias={self.alias}, space={scope})"


@dataclass(frozen=True, slots=True)
class VectorScan(LogicalOperator):
    query_vector: Embedding
    alias: str = "seed"
    score_name: str = "seed_score"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "query_vector",
            np.asarray(self.query_vector, dtype=np.float32).reshape(-1),
        )

    def children(self) -> tuple[LogicalOperator, ...]:
        return ()

    def describe(self) -> str:
        return f"VectorScan(alias={self.alias}, score={self.score_name})"


@dataclass(frozen=True, slots=True)
class Filter(LogicalOperator):
    input: LogicalOperator
    predicate: Predicate
    stage: str

    def children(self) -> tuple[LogicalOperator, ...]:
        return (self.input,)

    def describe(self) -> str:
        return f"Filter(stage={self.stage}, predicate={type(self.predicate).__name__})"


@dataclass(frozen=True, slots=True)
class TopK(LogicalOperator):
    input: LogicalOperator
    count: int
    score: FieldRef
    uid_alias: str
    descending: bool = True
    stage: str = "vector_search"

    def children(self) -> tuple[LogicalOperator, ...]:
        return (self.input,)

    def describe(self) -> str:
        order = "DESC" if self.descending else "ASC"
        return f"TopK(k={self.count}, score={self.score.path} {order})"


@dataclass(frozen=True, slots=True)
class EdgeJoin(LogicalOperator):
    input: LogicalOperator
    bound_alias: str
    new_alias: str
    rel_type: str
    direction: Direction
    stage: str

    def children(self) -> tuple[LogicalOperator, ...]:
        return (self.input,)

    def describe(self) -> str:
        return (
            f"EdgeJoin({self.bound_alias} -[{self.rel_type},{self.direction}]-> "
            f"{self.new_alias})"
        )


@dataclass(frozen=True, slots=True)
class InnerJoin(LogicalOperator):
    left: LogicalOperator
    right: LogicalOperator
    left_key: FieldRef
    right_key: FieldRef
    stage: str = "relation_join"

    def children(self) -> tuple[LogicalOperator, ...]:
        return (self.left, self.right)

    def describe(self) -> str:
        return (
            f"InnerJoin(left={self.left_key.alias}.{self.left_key.path}, "
            f"right={self.right_key.alias}.{self.right_key.path})"
        )


@dataclass(frozen=True, slots=True)
class GroupTopK(LogicalOperator):
    input: LogicalOperator
    group_alias: str
    item_alias: str
    score: FieldRef
    count: int
    score_output_name: str
    descending: bool = True
    stage: str = "relation_join"

    def children(self) -> tuple[LogicalOperator, ...]:
        return (self.input,)

    def describe(self) -> str:
        return (
            f"GroupTopK(group={self.group_alias}, item={self.item_alias}, "
            f"k={self.count}, score={self.score.path})"
        )


@dataclass(frozen=True, slots=True)
class Traverse(LogicalOperator):
    input: LogicalOperator
    start_alias: str
    current_alias: str
    rel_type: str
    direction: Direction
    min_hops: int
    max_hops: int
    stage: str = "graph_traversal"

    def children(self) -> tuple[LogicalOperator, ...]:
        return (self.input,)

    def describe(self) -> str:
        return (
            f"Traverse(start={self.start_alias}, edge={self.rel_type}, "
            f"direction={self.direction}, hops={self.min_hops}..{self.max_hops})"
        )


@dataclass(frozen=True, slots=True)
class ProjectLimit(LogicalOperator):
    input: LogicalOperator
    result_limit: int
    stage: str = "other"

    def children(self) -> tuple[LogicalOperator, ...]:
        return (self.input,)

    def describe(self) -> str:
        return f"ProjectLimit(limit={self.result_limit})"


def explain_logical_plan(root: LogicalOperator) -> str:
    """Render a stable tree with the root operator first."""
    lines: list[str] = []

    def visit(node: LogicalOperator, prefix: str, is_last: bool) -> None:
        connector = "└─ " if is_last else "├─ "
        lines.append(f"{prefix}{connector}{node.describe()}")
        children = node.children()
        child_prefix = prefix + ("   " if is_last else "│  ")
        for index, child in enumerate(children):
            visit(child, child_prefix, index == len(children) - 1)

    visit(root, "", True)
    return "\n".join(lines)


def unit_from_binding(
    row: BindingRow,
    alias: str,
    context: QueryExecutionContext,
) -> MemoryUnit | None:
    """Return the MemoryUnit bound to ``alias`` if it still exists."""
    uid = row.bindings.get(alias)
    return None if uid is None else context.get_unit(uid)
