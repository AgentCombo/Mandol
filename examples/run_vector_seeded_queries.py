"""Open an existing synthetic DuckDB data set and run profiled queries."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import duckdb
import numpy as np
from vector_seeded_synthetic_data import positive_int

from mandol.infrastructure import DuckDBUnifiedStore
from mandol.query import (
    VectorSeededTraversalSpec,
    format_query_log,
    render_query_diagram,
)

logger = logging.getLogger("mandol.run_vector_seeded_queries")


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default="vector_seeded_synthetic.duckdb")
    parser.add_argument("--queries", type=positive_int, default=3)
    parser.add_argument("--vector-k", type=positive_int, default=5)
    parser.add_argument("--nodes-per-seed", type=positive_int, default=2)
    parser.add_argument("--result-limit", type=positive_int, default=200)
    parser.add_argument("--show-rows", type=non_negative_int, default=5)
    parser.add_argument(
        "--profile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="collect per-stage DuckDB operator CPU timings",
    )
    return parser.parse_args()


def read_database_metadata(database: Path) -> dict[str, int]:
    connection = duckdb.connect(str(database), read_only=True)
    try:
        schema_rows = connection.execute(
            """
            SELECT key, value
            FROM mandol_schema_meta
            WHERE key = 'embedding_dim'
            """
        ).fetchall()
        if not schema_rows:
            raise RuntimeError("database does not contain Mandol schema metadata")
        metadata = {"embedding_dim": int(schema_rows[0][1])}
        tables = {str(row[0]) for row in connection.execute("SHOW TABLES").fetchall()}
        if "synthetic_dataset_meta" in tables:
            metadata.update(
                {
                    str(key): int(value)
                    for key, value in connection.execute(
                        "SELECT key, value FROM synthetic_dataset_meta"
                    ).fetchall()
                }
            )
        return metadata
    finally:
        connection.close()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    database = Path(args.database).expanduser().resolve()
    if not database.exists():
        raise SystemExit(f"{database} does not exist; run generate_vector_seeded_dataset.py first")

    metadata = read_database_metadata(database)
    max_hops = metadata.get("trajectory_depth", 2)
    with DuckDBUnifiedStore(
        database,
        embedding_dim=metadata["embedding_dim"],
    ) as backend:
        task_rows = backend.connection.execute(
            """
            SELECT
                uid,
                embedding,
                CAST(json_extract_string(metadata, '$.bucket') AS INTEGER)
            FROM memory_units
            WHERE json_extract_string(metadata, '$.kind') = 'task'
            ORDER BY uid
            """
        ).fetchall()
        if not task_rows:
            raise SystemExit("database contains no synthetic task rows")

        logger.info(
            "database=%s tasks=%d profile=%s",
            database,
            len(task_rows),
            args.profile,
        )
        for query_index in range(args.queries):
            target_uid, query_vector, bucket = task_rows[(query_index * 7) % len(task_rows)]
            spec = VectorSeededTraversalSpec(
                query_vector=np.asarray(query_vector, dtype=np.float32),
                vector_k=args.vector_k,
                seed_metadata_equals={"kind": "task", "bucket": int(bucket)},
                node_metadata_equals={"kind": "node", "is_buggy": False},
                nodes_per_seed=args.nodes_per_seed,
                max_hops=max_hops,
                result_limit=args.result_limit,
            )
            execution = backend.queries.vector_seeded_graph_traversal(
                spec,
                profile=args.profile,
            )

            logger.info("\n=== QUERY %d target=%s ===", query_index + 1, target_uid)
            logger.info("%s", format_query_log(spec, execution))
            logger.info("\n%s", render_query_diagram(spec, execution))
            for row in execution.rows[: args.show_rows]:
                logger.info(
                    "result seed=%s node=%s current=%s depth=%d score=%.4f",
                    row.seed_uid,
                    row.node_uid,
                    row.current_uid,
                    row.depth,
                    row.node_score,
                )


if __name__ == "__main__":
    main()
