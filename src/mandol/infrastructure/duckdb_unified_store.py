"""Unified DuckDB physical storage for units, vectors, spaces, and graph edges.

The compatibility facades implement Mandol's existing storage ports while
sharing one DuckDB connection and transaction manager.  The ``queries``
facade adds the first cross-modal physical operator: exact vector seeding,
two typed relation joins, per-seed top-k, and bounded graph traversal in one
SQL statement.
"""

from __future__ import annotations

import importlib
import json
import threading
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import TracebackType
from typing import Any, TypeVar

import numpy as np

from ..domain.memory_space import MemorySpace
from ..domain.memory_unit import MemoryUnit
from ..domain.types import Embedding, SpaceName, Uid
from ..ports.graph_store import GraphStore
from ..ports.unified_query_store import UnifiedQueryStore
from ..ports.unit_store import UnitStore
from ..ports.vector_index import VectorIndex
from ..query.vector_seeded import (
    QueryExecutionMetrics,
    QueryStageBreakdown,
    QueryStageTimings,
    VectorSeededTraversalExecution,
    VectorSeededTraversalRow,
    VectorSeededTraversalSpec,
)

_SCHEMA_VERSION = "1"
_StoreT = TypeVar("_StoreT", bound="DuckDBUnifiedStore")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


def _json_dumps(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        default=_json_default,
    )


def _json_loads(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    return json.loads(str(value))


def _as_vector(value: Embedding, dim: int, *, field_name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if vector.shape[0] != dim:
        raise ValueError(f"{field_name} dim mismatch: expected {dim}, got {vector.shape[0]}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{field_name} must contain only finite values")
    return vector


def _normalize_vector(value: Embedding, dim: int, *, field_name: str) -> np.ndarray:
    vector = _as_vector(value, dim, field_name=field_name)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return vector / norm


def _json_path(field_name: str) -> str:
    field = str(field_name).strip()
    if not field:
        raise ValueError("metadata field names must be non-empty")
    return field if field.startswith("$") else f"$.{field}"


def _metadata_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (str, int, float)):
        return str(value)
    raise TypeError("metadata equality filters support only scalar values")


class DuckDBUnifiedStore:
    """Own one DuckDB database and expose compatible and unified query facades.

    Args:
        database: ``":memory:"`` or a path to a DuckDB database file.
        embedding_dim: Fixed dimension used by dense vector columns.

    ``DuckDBUnifiedStore`` is safe for nested calls on one thread and
    serializes access to its single connection.  It is deliberately an
    embedded/single-process backend; concurrent serving can use one backend
    instance per process with a later multi-writer design.
    """

    def __init__(
        self,
        database: str | Path = ":memory:",
        *,
        embedding_dim: int,
    ) -> None:
        if int(embedding_dim) <= 0:
            raise ValueError("embedding_dim must be positive")
        try:
            duckdb = importlib.import_module("duckdb")
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise ImportError(
                "DuckDBUnifiedStore requires the 'duckdb' extra: pip install 'mandol[duckdb]'"
            ) from exc

        self.database = str(database)
        self.embedding_dim = int(embedding_dim)
        self._connection = duckdb.connect(self.database)
        self._lock = threading.RLock()
        self._transaction_depth = 0
        self._closed = False

        try:
            self._initialize_schema()
        except BaseException:
            self._connection.close()
            self._closed = True
            raise
        self.units: UnitStore = _DuckDBUnitStore(self)
        self.vectors: VectorIndex = _DuckDBVectorIndex(self)
        self.graph: GraphStore = _DuckDBGraphStore(self)
        self.queries = DuckDBUnifiedQueryStore(self)

    @property
    def connection(self) -> Any:
        """Return the underlying connection for diagnostics and EXPLAIN."""
        self._ensure_open()
        return self._connection

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Execute nested facade operations in one atomic transaction."""
        self._ensure_open()
        self._lock.acquire()
        outermost = self._transaction_depth == 0
        if outermost:
            self._connection.execute("BEGIN TRANSACTION")
        self._transaction_depth += 1
        try:
            yield
        except BaseException:
            self._transaction_depth -= 1
            if outermost:
                self._connection.execute("ROLLBACK")
            raise
        else:
            self._transaction_depth -= 1
            if outermost:
                self._connection.execute("COMMIT")
        finally:
            self._lock.release()

    def flush(self) -> None:
        """Checkpoint a file-backed database; in-memory writes are immediate."""
        self._ensure_open()
        if self.database == ":memory:" or self._transaction_depth:
            return
        with self._lock:
            self._connection.execute("CHECKPOINT")

    def close(self) -> None:
        """Close the shared database connection."""
        with self._lock:
            if self._closed:
                return
            if self._transaction_depth:
                raise RuntimeError("cannot close DuckDBUnifiedStore inside a transaction")
            self._connection.close()
            self._closed = True

    def __enter__(self: _StoreT) -> _StoreT:  # noqa: PYI019 - Python 3.10
        self._ensure_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("DuckDBUnifiedStore is closed")

    def _initialize_schema(self) -> None:
        dim = self.embedding_dim
        with self.transaction():
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS mandol_schema_meta (
                    key VARCHAR PRIMARY KEY,
                    value VARCHAR NOT NULL
                )
                """
            )
            self._connection.execute(
                """
                INSERT INTO mandol_schema_meta VALUES ('schema_version', ?)
                ON CONFLICT (key) DO NOTHING
                """,
                [_SCHEMA_VERSION],
            )
            self._connection.execute(
                """
                INSERT INTO mandol_schema_meta VALUES ('embedding_dim', ?)
                ON CONFLICT (key) DO NOTHING
                """,
                [str(dim)],
            )
            stored = dict(
                self._connection.execute("SELECT key, value FROM mandol_schema_meta").fetchall()
            )
            if stored.get("schema_version") != _SCHEMA_VERSION:
                raise ValueError(
                    f"unsupported Mandol DuckDB schema version: {stored.get('schema_version')!r}"
                )
            if stored.get("embedding_dim") != str(dim):
                raise ValueError(
                    "embedding_dim does not match existing database: "
                    f"expected {stored.get('embedding_dim')}, got {dim}"
                )

            self._connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS memory_units (
                    uid VARCHAR PRIMARY KEY,
                    raw_data JSON NOT NULL,
                    metadata JSON NOT NULL,
                    embedding FLOAT[{dim}],
                    sparse_embedding FLOAT[]
                )
                """
            )
            self._connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS memory_vectors (
                    uid VARCHAR PRIMARY KEY,
                    embedding FLOAT[{dim}] NOT NULL
                )
                """
            )
            self._connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS memory_spaces (
                    space_name VARCHAR PRIMARY KEY,
                    summary_text VARCHAR,
                    summary_embedding FLOAT[{dim}],
                    metadata JSON NOT NULL
                )
                """
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS space_memberships (
                    space_name VARCHAR NOT NULL,
                    uid VARCHAR NOT NULL,
                    PRIMARY KEY (space_name, uid)
                )
                """
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS space_children (
                    parent_space VARCHAR NOT NULL,
                    child_space VARCHAR NOT NULL,
                    PRIMARY KEY (parent_space, child_space)
                )
                """
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_edges (
                    source_uid VARCHAR NOT NULL,
                    target_uid VARCHAR NOT NULL,
                    rel_type VARCHAR NOT NULL,
                    properties JSON NOT NULL,
                    PRIMARY KEY (source_uid, target_uid, rel_type)
                )
                """
            )
            self._connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_space_memberships_uid
                ON space_memberships(uid)
                """
            )
            self._connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_edges_source_type
                ON memory_edges(source_uid, rel_type)
                """
            )
            self._connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_edges_target_type
                ON memory_edges(target_uid, rel_type)
                """
            )


class _DuckDBUnitStore(UnitStore):
    def __init__(self, backend: DuckDBUnifiedStore) -> None:
        self._backend = backend

    def upsert_units(self, units: Sequence[MemoryUnit]) -> None:
        if not units:
            return
        dim = self._backend.embedding_dim
        with self._backend.transaction():
            for unit in units:
                embedding = (
                    _as_vector(unit.embedding, dim, field_name="embedding").tolist()
                    if unit.embedding is not None
                    else None
                )
                sparse = (
                    np.asarray(unit.sparse_embedding, dtype=np.float32).reshape(-1).tolist()
                    if unit.sparse_embedding is not None
                    else None
                )
                self._backend.connection.execute(
                    """
                    INSERT INTO memory_units
                        (uid, raw_data, metadata, embedding, sparse_embedding)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT (uid) DO UPDATE SET
                        raw_data = excluded.raw_data,
                        metadata = excluded.metadata,
                        embedding = excluded.embedding,
                        sparse_embedding = excluded.sparse_embedding
                    """,
                    [
                        str(unit.uid),
                        _json_dumps(unit.raw_data),
                        _json_dumps(unit.metadata),
                        embedding,
                        sparse,
                    ],
                )
                if unit.embedding is not None:
                    normalized = _normalize_vector(unit.embedding, dim, field_name="embedding")
                    self._backend.connection.execute(
                        """
                        INSERT INTO memory_vectors (uid, embedding) VALUES (?, ?)
                        ON CONFLICT (uid) DO UPDATE SET
                            embedding = excluded.embedding
                        """,
                        [str(unit.uid), normalized.tolist()],
                    )

    def delete_units(self, uids: Iterable[Uid]) -> None:
        values = sorted({str(uid) for uid in uids})
        if not values:
            return
        placeholders = ", ".join("?" for _ in values)
        with self._backend.transaction():
            self._backend.connection.execute(
                f"DELETE FROM memory_units WHERE uid IN ({placeholders})", values
            )
            self._backend.connection.execute(
                f"DELETE FROM memory_vectors WHERE uid IN ({placeholders})", values
            )
            self._backend.connection.execute(
                f"DELETE FROM space_memberships WHERE uid IN ({placeholders})", values
            )

    def get_unit(self, uid: Uid) -> MemoryUnit | None:
        with self._backend._lock:
            row = self._backend.connection.execute(
                """
                SELECT uid, raw_data, metadata, embedding, sparse_embedding
                FROM memory_units
                WHERE uid = ?
                """,
                [str(uid)],
            ).fetchone()
        return _unit_from_row(row) if row is not None else None

    def get_units(self, uids: Sequence[Uid]) -> list[MemoryUnit]:
        requested = [str(uid) for uid in uids]
        if not requested:
            return []
        unique = list(dict.fromkeys(requested))
        placeholders = ", ".join("?" for _ in unique)
        with self._backend._lock:
            rows = self._backend.connection.execute(
                f"""
                SELECT uid, raw_data, metadata, embedding, sparse_embedding
                FROM memory_units
                WHERE uid IN ({placeholders})
                """,
                unique,
            ).fetchall()
        by_uid = {str(row[0]): _unit_from_row(row) for row in rows}
        return [by_uid[uid] for uid in requested if uid in by_uid]

    def list_units(self) -> list[MemoryUnit]:
        with self._backend._lock:
            rows = self._backend.connection.execute(
                """
                SELECT uid, raw_data, metadata, embedding, sparse_embedding
                FROM memory_units
                ORDER BY uid
                """
            ).fetchall()
        return [_unit_from_row(row) for row in rows]

    def upsert_spaces(self, spaces: Sequence[MemorySpace]) -> None:
        if not spaces:
            return
        dim = self._backend.embedding_dim
        with self._backend.transaction():
            for space in spaces:
                summary_embedding = (
                    _as_vector(
                        space.summary_embedding,
                        dim,
                        field_name="summary_embedding",
                    ).tolist()
                    if space.summary_embedding is not None
                    else None
                )
                name = str(space.name)
                self._backend.connection.execute(
                    """
                    INSERT INTO memory_spaces
                        (space_name, summary_text, summary_embedding, metadata)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT (space_name) DO UPDATE SET
                        summary_text = excluded.summary_text,
                        summary_embedding = excluded.summary_embedding,
                        metadata = excluded.metadata
                    """,
                    [
                        name,
                        space.summary_text,
                        summary_embedding,
                        _json_dumps(space.metadata),
                    ],
                )
                self._backend.connection.execute(
                    "DELETE FROM space_memberships WHERE space_name = ?", [name]
                )
                self._backend.connection.execute(
                    "DELETE FROM space_children WHERE parent_space = ?", [name]
                )
                if space.unit_uids:
                    self._backend.connection.executemany(
                        """
                        INSERT INTO space_memberships (space_name, uid)
                        VALUES (?, ?)
                        """,
                        [(name, str(uid)) for uid in sorted(space.unit_uids, key=str)],
                    )
                if space.child_spaces:
                    self._backend.connection.executemany(
                        """
                        INSERT INTO space_children (parent_space, child_space)
                        VALUES (?, ?)
                        """,
                        [(name, str(child)) for child in sorted(space.child_spaces, key=str)],
                    )

    def get_space(self, name: SpaceName) -> MemorySpace | None:
        key = str(name)
        with self._backend._lock:
            row = self._backend.connection.execute(
                """
                SELECT space_name, summary_text, summary_embedding, metadata
                FROM memory_spaces
                WHERE space_name = ?
                """,
                [key],
            ).fetchone()
            if row is None:
                return None
            unit_rows = self._backend.connection.execute(
                """
                SELECT uid FROM space_memberships
                WHERE space_name = ?
                ORDER BY uid
                """,
                [key],
            ).fetchall()
            child_rows = self._backend.connection.execute(
                """
                SELECT child_space FROM space_children
                WHERE parent_space = ?
                ORDER BY child_space
                """,
                [key],
            ).fetchall()
        return _space_from_rows(row, unit_rows, child_rows)

    def list_spaces(self) -> list[MemorySpace]:
        with self._backend._lock:
            names = [
                SpaceName(str(row[0]))
                for row in self._backend.connection.execute(
                    "SELECT space_name FROM memory_spaces ORDER BY space_name"
                ).fetchall()
            ]
        return [space for name in names if (space := self.get_space(name)) is not None]

    def clear(self) -> None:
        with self._backend.transaction():
            for table in (
                "space_memberships",
                "space_children",
                "memory_spaces",
                "memory_vectors",
                "memory_units",
            ):
                self._backend.connection.execute(f"DELETE FROM {table}")

    def flush(self) -> None:
        self._backend.flush()


class _DuckDBVectorIndex(VectorIndex):
    def __init__(self, backend: DuckDBUnifiedStore) -> None:
        self._backend = backend

    def dim(self) -> int:
        return self._backend.embedding_dim

    def upsert(self, items: Sequence[tuple[Uid, Embedding]]) -> None:
        if not items:
            return
        values = [
            (
                str(uid),
                _normalize_vector(
                    embedding,
                    self._backend.embedding_dim,
                    field_name="embedding",
                ).tolist(),
            )
            for uid, embedding in items
        ]
        with self._backend.transaction():
            self._backend.connection.executemany(
                """
                INSERT INTO memory_vectors (uid, embedding) VALUES (?, ?)
                ON CONFLICT (uid) DO UPDATE SET embedding = excluded.embedding
                """,
                values,
            )

    def delete(self, uids: Iterable[Uid]) -> None:
        values = sorted({str(uid) for uid in uids})
        if not values:
            return
        placeholders = ", ".join("?" for _ in values)
        with self._backend.transaction():
            self._backend.connection.execute(
                f"DELETE FROM memory_vectors WHERE uid IN ({placeholders})", values
            )

    def search(self, query: Embedding, top_k: int) -> list[tuple[Uid, float]]:
        limit = max(0, int(top_k))
        if limit == 0:
            return []
        vector = _normalize_vector(query, self._backend.embedding_dim, field_name="query").tolist()
        with self._backend._lock:
            rows = self._backend.connection.execute(
                f"""
                SELECT uid, array_inner_product(embedding, ?::FLOAT[{self.dim()}]) AS score
                FROM memory_vectors
                ORDER BY score DESC, uid
                LIMIT ?
                """,
                [vector, limit],
            ).fetchall()
        return [(Uid(str(uid)), float(score)) for uid, score in rows]

    def search_in_space(
        self,
        query: Embedding,
        space_name: str,
        candidates: set[Uid] | None,
        top_k: int,
    ) -> list[tuple[Uid, float]]:
        limit = max(0, int(top_k))
        if limit == 0 or candidates == set():
            return []
        vector = _normalize_vector(query, self._backend.embedding_dim, field_name="query").tolist()
        params: list[Any] = [vector, str(space_name)]
        candidate_sql = ""
        if candidates is not None:
            ordered = sorted({str(uid) for uid in candidates})
            placeholders = ", ".join("?" for _ in ordered)
            candidate_sql = f" AND v.uid IN ({placeholders})"
            params.extend(ordered)
        params.append(limit)
        with self._backend._lock:
            rows = self._backend.connection.execute(
                f"""
                SELECT
                    v.uid,
                    array_inner_product(
                        v.embedding,
                        ?::FLOAT[{self.dim()}]
                    ) AS score
                FROM memory_vectors v
                WHERE EXISTS (
                    SELECT 1
                    FROM space_memberships sm
                    WHERE sm.uid = v.uid
                      AND sm.space_name = ?
                )
                {candidate_sql}
                ORDER BY score DESC, v.uid
                LIMIT ?
                """,
                params,
            ).fetchall()
        return [(Uid(str(uid)), float(score)) for uid, score in rows]

    def rebuild(self, items: Sequence[tuple[Uid, Embedding]]) -> None:
        with self._backend.transaction():
            self._backend.connection.execute("DELETE FROM memory_vectors")
            self.upsert(items)


class _DuckDBGraphStore(GraphStore):
    def __init__(self, backend: DuckDBUnifiedStore) -> None:
        self._backend = backend

    def upsert_relationship(
        self, source: Uid, target: Uid, rel_type: str, properties: dict[str, Any]
    ) -> None:
        if not str(rel_type).strip():
            raise ValueError("rel_type must be non-empty")
        with self._backend.transaction():
            self._backend.connection.execute(
                """
                INSERT INTO memory_edges
                    (source_uid, target_uid, rel_type, properties)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (source_uid, target_uid, rel_type) DO UPDATE SET
                    properties = excluded.properties
                """,
                [
                    str(source),
                    str(target),
                    str(rel_type),
                    _json_dumps({k: v for k, v in properties.items() if v is not None}),
                ],
            )

    def delete_relationship(self, source: Uid, target: Uid, rel_type: str | None = None) -> None:
        with self._backend.transaction():
            if rel_type is None:
                self._backend.connection.execute(
                    """
                    DELETE FROM memory_edges
                    WHERE source_uid = ? AND target_uid = ?
                    """,
                    [str(source), str(target)],
                )
            else:
                self._backend.connection.execute(
                    """
                    DELETE FROM memory_edges
                    WHERE source_uid = ? AND target_uid = ? AND rel_type = ?
                    """,
                    [str(source), str(target), str(rel_type)],
                )

    def get_relationship(self, source: Uid, target: Uid, rel_type: str) -> dict[str, Any] | None:
        with self._backend._lock:
            row = self._backend.connection.execute(
                """
                SELECT properties
                FROM memory_edges
                WHERE source_uid = ? AND target_uid = ? AND rel_type = ?
                """,
                [str(source), str(target), str(rel_type)],
            ).fetchone()
        return dict(_json_loads(row[0])) if row is not None else None

    def get_neighbors(
        self, uid: Uid, *, rel_type: str | None = None, direction: str = "out"
    ) -> list[Uid]:
        if direction not in {"out", "in", "both"}:
            raise ValueError("direction must be 'out', 'in', or 'both'")
        params: list[Any] = [str(uid)]
        rel_sql = ""
        if rel_type is not None:
            rel_sql = " AND rel_type = ?"
            params.append(str(rel_type))

        if direction == "out":
            query = (
                "SELECT DISTINCT target_uid AS neighbor FROM memory_edges "
                f"WHERE source_uid = ?{rel_sql} ORDER BY neighbor"
            )
        elif direction == "in":
            query = (
                "SELECT DISTINCT source_uid AS neighbor FROM memory_edges "
                f"WHERE target_uid = ?{rel_sql} ORDER BY neighbor"
            )
        else:
            # Repeat parameters because the optional type predicate occurs in
            # both branches of the UNION.
            second_params = list(params)
            query = (
                "SELECT neighbor FROM ("
                "SELECT target_uid AS neighbor FROM memory_edges "
                f"WHERE source_uid = ?{rel_sql} "
                "UNION "
                "SELECT source_uid AS neighbor FROM memory_edges "
                f"WHERE target_uid = ?{rel_sql}"
                ") ORDER BY neighbor"
            )
            params.extend(second_params)

        with self._backend._lock:
            rows = self._backend.connection.execute(query, params).fetchall()
        return [Uid(str(row[0])) for row in rows]

    def get_all_edges(self) -> list[tuple[Uid, Uid, str, dict[str, Any]]]:
        with self._backend._lock:
            rows = self._backend.connection.execute(
                """
                SELECT source_uid, target_uid, rel_type, properties
                FROM memory_edges
                ORDER BY source_uid, target_uid, rel_type
                """
            ).fetchall()
        return [
            (
                Uid(str(source)),
                Uid(str(target)),
                str(rel_type),
                dict(_json_loads(properties)),
            )
            for source, target, rel_type, properties in rows
        ]

    def clear(self) -> None:
        with self._backend.transaction():
            self._backend.connection.execute("DELETE FROM memory_edges")

    def flush(self) -> None:
        self._backend.flush()


class DuckDBUnifiedQueryStore(UnifiedQueryStore):
    """Compile and execute one cross-modal physical plan in DuckDB."""

    def __init__(self, backend: DuckDBUnifiedStore) -> None:
        self._backend = backend

    def vector_seeded_graph_traversal(
        self,
        spec: VectorSeededTraversalSpec,
        *,
        profile: bool = False,
    ) -> VectorSeededTraversalExecution:
        """Execute exact vector seed -> two joins -> top-k -> path expansion."""
        sql, params = self._compile_vector_seeded_graph_traversal(spec)
        if profile:
            raw_rows, elapsed_ms, stage_timings = self._execute_profiled(sql, params)
        else:
            started = time.perf_counter()
            with self._backend._lock:
                raw_rows = self._backend.connection.execute(sql, params).fetchall()
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            stage_timings = QueryStageTimings()

        result_rows = [row for row in raw_rows if row[0] is not None]
        complete = len(result_rows) <= spec.result_limit
        visible_rows = result_rows[: spec.result_limit]
        stages = (
            QueryStageBreakdown(
                vector_candidate_rows=int(raw_rows[0][8]),
                vector_seed_rows=int(raw_rows[0][9]),
                session_join_rows=int(raw_rows[0][10]),
                node_join_rows=int(raw_rows[0][11]),
                selected_node_rows=int(raw_rows[0][12]),
                traversal_rows=int(raw_rows[0][13]),
            )
            if raw_rows
            else QueryStageBreakdown()
        )
        rows = tuple(
            VectorSeededTraversalRow(
                seed_uid=Uid(str(row[0])),
                seed_score=float(row[1]),
                session_uid=Uid(str(row[2])),
                node_uid=Uid(str(row[3])),
                node_score=float(row[4]),
                current_uid=Uid(str(row[5])),
                depth=int(row[6]),
                path=tuple(Uid(str(uid)) for uid in row[7]),
            )
            for row in visible_rows
        )
        metrics = QueryExecutionMetrics(
            elapsed_ms=elapsed_ms,
            rows_returned=len(rows),
            complete=complete,
            truncated_reason=None if complete else "result_limit",
            stages=stages,
            timings=stage_timings,
        )
        return VectorSeededTraversalExecution(rows=rows, metrics=metrics)

    def _execute_profiled(
        self,
        sql: str,
        params: Sequence[Any],
    ) -> tuple[list[tuple[Any, ...]], float, QueryStageTimings]:
        with TemporaryDirectory(prefix="mandol-duckdb-profile-") as directory:
            profile_path = Path(directory) / "query-profile.json"
            with self._backend._lock:
                connection = self._backend.connection
                connection.execute("PRAGMA enable_profiling='json'")
                connection.execute("SET profiling_output = ?", [str(profile_path)])
                try:
                    started = time.perf_counter()
                    rows = connection.execute(sql, params).fetchall()
                    elapsed_ms = (time.perf_counter() - started) * 1000.0
                finally:
                    connection.execute("PRAGMA disable_profiling")

            if not profile_path.exists():
                raise RuntimeError("DuckDB did not produce the requested JSON profile")
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
        return rows, elapsed_ms, _extract_stage_timings(profile)

    def explain_vector_seeded_graph_traversal(self, spec: VectorSeededTraversalSpec) -> str:
        """Return DuckDB's physical plan for the compiled unified query."""
        sql, params = self._compile_vector_seeded_graph_traversal(spec)
        with self._backend._lock:
            rows = self._backend.connection.execute(f"EXPLAIN {sql}", params).fetchall()
        return "\n".join(str(row[-1]) for row in rows)

    def _compile_vector_seeded_graph_traversal(
        self, spec: VectorSeededTraversalSpec
    ) -> tuple[str, list[Any]]:
        query_vector = _normalize_vector(
            spec.query_vector,
            self._backend.embedding_dim,
            field_name="query_vector",
        ).tolist()

        seed_filters, seed_params = _compile_metadata_equals(
            "seed_unit.metadata", spec.seed_metadata_equals
        )
        node_filters, node_params = _compile_metadata_equals(
            "node_unit.metadata", spec.node_metadata_equals
        )

        session_other, session_join = _edge_join(
            current_sql="vector_seeds.seed_uid",
            direction=spec.session_relation_direction,
        )
        node_other, node_join = _edge_join(
            current_sql="session_bindings.session_uid",
            direction=spec.node_relation_direction,
        )
        traversal_other, traversal_join = _edge_join(
            current_sql="traversal.current_uid",
            direction=spec.traversal_direction,
        )

        dim = self._backend.embedding_dim
        sql = f"""
            WITH RECURSIVE
            vector_candidates AS (
                SELECT
                    seed_unit.uid AS seed_uid,
                    array_inner_product(
                        seed_vector.embedding,
                        ?::FLOAT[{dim}]
                    ) AS seed_score
                FROM memory_units seed_unit
                JOIN memory_vectors seed_vector
                  ON seed_vector.uid = seed_unit.uid
                WHERE TRUE
                  {seed_filters}
            ),
            vector_seeds AS (
                SELECT seed_uid, seed_score
                FROM vector_candidates
                ORDER BY seed_score DESC, seed_uid
                LIMIT ?
            ),
            session_bindings AS (
                SELECT
                    vector_seeds.seed_uid,
                    vector_seeds.seed_score,
                    {session_other} AS session_uid
                FROM vector_seeds
                JOIN memory_edges session_edge
                  ON {session_join}
                 AND session_edge.rel_type = ?
                JOIN memory_units session_unit
                  ON session_unit.uid = {session_other}
            ),
            node_bindings AS (
                SELECT
                    session_bindings.seed_uid,
                    session_bindings.seed_score,
                    session_bindings.session_uid,
                    {node_other} AS node_uid,
                    COALESCE(
                        TRY_CAST(
                            json_extract_string(node_unit.metadata, ?)
                            AS DOUBLE
                        ),
                        0.0
                    ) AS node_score
                FROM session_bindings
                JOIN memory_edges node_edge
                  ON {node_join}
                 AND node_edge.rel_type = ?
                JOIN memory_units node_unit
                  ON node_unit.uid = {node_other}
                WHERE TRUE
                  {node_filters}
            ),
            ranked_nodes AS (
                SELECT
                    *,
                    row_number() OVER (
                        PARTITION BY seed_uid
                        ORDER BY node_score DESC, node_uid
                    ) AS node_rank
                FROM node_bindings
            ),
            selected_nodes AS (
                SELECT
                    seed_uid,
                    seed_score,
                    session_uid,
                    node_uid,
                    node_score
                FROM ranked_nodes
                WHERE node_rank <= ?
            ),
            traversal(
                seed_uid,
                seed_score,
                session_uid,
                node_uid,
                node_score,
                current_uid,
                depth,
                path
            ) AS (
                SELECT
                    seed_uid,
                    seed_score,
                    session_uid,
                    node_uid,
                    node_score,
                    node_uid AS current_uid,
                    0 AS depth,
                    [node_uid]::VARCHAR[] AS path
                FROM selected_nodes

                UNION ALL

                SELECT
                    traversal.seed_uid,
                    traversal.seed_score,
                    traversal.session_uid,
                    traversal.node_uid,
                    traversal.node_score,
                    {traversal_other} AS current_uid,
                    traversal.depth + 1 AS depth,
                    list_append(traversal.path, {traversal_other}) AS path
                FROM traversal
                JOIN memory_edges traversal_edge
                  ON {traversal_join}
                 AND traversal_edge.rel_type = ?
                JOIN memory_units traversed_unit
                  ON traversed_unit.uid = {traversal_other}
                WHERE traversal.depth < ?
                  AND NOT list_contains(traversal.path, {traversal_other})
            ),
            stage_counts AS (
                SELECT
                    (SELECT count(*) FROM vector_candidates)
                        AS vector_candidate_rows,
                    (SELECT count(*) FROM vector_seeds) AS vector_seed_rows,
                    (SELECT count(*) FROM session_bindings) AS session_join_rows,
                    (SELECT count(*) FROM node_bindings) AS node_join_rows,
                    (SELECT count(*) FROM selected_nodes) AS selected_node_rows,
                    (SELECT count(*) FROM traversal) AS traversal_rows
            ),
            result_rows AS (
                SELECT
                    seed_uid,
                    seed_score,
                    session_uid,
                    node_uid,
                    node_score,
                    current_uid,
                    depth,
                    path
                FROM traversal
                WHERE depth >= ?
                ORDER BY
                    seed_score DESC,
                    seed_uid,
                    node_score DESC,
                    node_uid,
                    depth,
                    current_uid
                LIMIT ?
            )
            SELECT
                result_rows.seed_uid,
                result_rows.seed_score,
                result_rows.session_uid,
                result_rows.node_uid,
                result_rows.node_score,
                result_rows.current_uid,
                result_rows.depth,
                result_rows.path,
                stage_counts.vector_candidate_rows,
                stage_counts.vector_seed_rows,
                stage_counts.session_join_rows,
                stage_counts.node_join_rows,
                stage_counts.selected_node_rows,
                stage_counts.traversal_rows
            FROM stage_counts
            LEFT JOIN result_rows ON TRUE
            ORDER BY
                result_rows.seed_score DESC NULLS LAST,
                result_rows.seed_uid,
                result_rows.node_score DESC NULLS LAST,
                result_rows.node_uid,
                result_rows.depth,
                result_rows.current_uid
        """
        params: list[Any] = [query_vector]
        params.extend(seed_params)
        params.extend(
            [
                spec.vector_k,
                spec.session_relation,
                _json_path(spec.node_score_field),
                spec.node_relation,
            ]
        )
        params.extend(node_params)
        params.extend(
            [
                spec.nodes_per_seed,
                spec.traversal_relation,
                spec.max_hops,
                spec.min_hops,
                spec.result_limit + 1,
            ]
        )
        return sql, params


def _compile_metadata_equals(
    metadata_sql: str, filters: Mapping[str, Any]
) -> tuple[str, list[Any]]:
    fragments: list[str] = []
    params: list[Any] = []
    for field_name in sorted(filters):
        value = filters[field_name]
        if value is None:
            fragments.append(f"AND json_extract({metadata_sql}, ?) IS NULL")
            params.append(_json_path(field_name))
            continue
        _metadata_scalar(value)
        fragments.append(f"AND json_extract({metadata_sql}, ?) = ?::JSON")
        params.extend([_json_path(field_name), _json_dumps(value)])
    return "\n".join(fragments), params


def _extract_stage_timings(profile: Mapping[str, Any]) -> QueryStageTimings:
    vector_seconds = _sum_present(
        _cte_build_cpu_seconds(profile, "vector_candidates"),
        _cte_build_cpu_seconds(profile, "vector_seeds"),
    )
    relation_seconds = _sum_present(
        _cte_build_cpu_seconds(profile, "session_bindings"),
        _cte_build_cpu_seconds(profile, "node_bindings"),
        _cte_build_cpu_seconds(profile, "selected_nodes"),
    )
    traversal_seconds = _cte_build_cpu_seconds(profile, "traversal")
    total_seconds = _optional_float(profile.get("cpu_time"))

    attributed = sum(
        value
        for value in (vector_seconds, relation_seconds, traversal_seconds)
        if value is not None
    )
    other_seconds = max(0.0, total_seconds - attributed) if total_seconds is not None else None
    return QueryStageTimings(
        vector_search_cpu_ms=_seconds_to_ms(vector_seconds),
        relation_join_cpu_ms=_seconds_to_ms(relation_seconds),
        graph_traversal_cpu_ms=_seconds_to_ms(traversal_seconds),
        other_cpu_ms=_seconds_to_ms(other_seconds),
        total_cpu_ms=_seconds_to_ms(total_seconds),
    )


def _cte_build_cpu_seconds(
    profile: Mapping[str, Any],
    cte_name: str,
) -> float | None:
    cte = _find_cte(profile, cte_name)
    if cte is None:
        return None
    children = cte.get("children") or []
    build_seconds = _subtree_cpu_seconds(children[0]) if children else 0.0
    return float(cte.get("operator_timing") or 0.0) + build_seconds


def _find_cte(
    node: Mapping[str, Any],
    cte_name: str,
) -> Mapping[str, Any] | None:
    extra_info = node.get("extra_info") or {}
    if node.get("operator_name") == "CTE" and extra_info.get("CTE Name") == cte_name:
        return node
    for child in node.get("children") or []:
        match = _find_cte(child, cte_name)
        if match is not None:
            return match
    return None


def _subtree_cpu_seconds(node: Mapping[str, Any]) -> float:
    return float(node.get("operator_timing") or 0.0) + sum(
        _subtree_cpu_seconds(child) for child in node.get("children") or []
    )


def _sum_present(*values: float | None) -> float | None:
    present = [value for value in values if value is not None]
    return sum(present) if present else None


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _seconds_to_ms(value: float | None) -> float | None:
    return None if value is None else value * 1000.0


def _edge_join(*, current_sql: str, direction: str) -> tuple[str, str]:
    if direction == "out":
        return (
            _edge_alias(current_sql, "target_uid"),
            f"{_edge_alias(current_sql, 'source_uid')} = {current_sql}",
        )
    if direction == "in":
        return (
            _edge_alias(current_sql, "source_uid"),
            f"{_edge_alias(current_sql, 'target_uid')} = {current_sql}",
        )
    raise ValueError("direction must be 'out' or 'in'")


def _edge_alias(current_sql: str, column: str) -> str:
    if current_sql.startswith("vector_seeds."):
        alias = "session_edge"
    elif current_sql.startswith("session_bindings."):
        alias = "node_edge"
    elif current_sql.startswith("traversal."):
        alias = "traversal_edge"
    else:  # pragma: no cover - internal compiler invariant
        raise ValueError(f"unknown edge join input: {current_sql}")
    return f"{alias}.{column}"


def _unit_from_row(row: Sequence[Any]) -> MemoryUnit:
    return MemoryUnit.from_dict(
        {
            "uid": str(row[0]),
            "raw_data": _json_loads(row[1]) or {},
            "metadata": _json_loads(row[2]) or {},
            "embedding": list(row[3]) if row[3] is not None else None,
            "sparse_embedding": list(row[4]) if row[4] is not None else None,
        }
    )


def _space_from_rows(
    row: Sequence[Any],
    unit_rows: Sequence[Sequence[Any]],
    child_rows: Sequence[Sequence[Any]],
) -> MemorySpace:
    return MemorySpace.from_dict(
        {
            "name": str(row[0]),
            "summary_text": row[1],
            "summary_embedding": list(row[2]) if row[2] is not None else None,
            "metadata": _json_loads(row[3]) or {},
            "unit_uids": [str(item[0]) for item in unit_rows],
            "child_spaces": [str(item[0]) for item in child_rows],
        }
    )


__all__ = ["DuckDBUnifiedQueryStore", "DuckDBUnifiedStore"]
