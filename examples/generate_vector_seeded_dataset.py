"""Generate a persistent synthetic Experience Graph DuckDB data set."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
from vector_seeded_synthetic_data import generate_dataset, positive_int

from mandol.infrastructure import DuckDBUnifiedStore

logger = logging.getLogger("mandol.generate_vector_seeded_dataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default="vector_seeded_synthetic.duckdb")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--tasks", type=positive_int, default=30)
    parser.add_argument("--sessions-per-task", type=positive_int, default=3)
    parser.add_argument("--nodes-per-session", type=positive_int, default=6)
    parser.add_argument("--branching-factor", type=positive_int, default=2)
    parser.add_argument("--trajectory-depth", type=positive_int, default=2)
    parser.add_argument("--embedding-dim", type=positive_int, default=8)
    parser.add_argument("--random-seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    database = Path(args.database).expanduser().resolve()
    if args.database == ":memory:":
        raise SystemExit("a persistent --database path is required")
    if database.exists() and not args.overwrite:
        raise SystemExit(f"{database} already exists; choose another path or pass --overwrite")
    if database.exists():
        database.unlink()
    database.parent.mkdir(parents=True, exist_ok=True)

    generation_started = time.perf_counter()
    units, edges = generate_dataset(
        rng=np.random.default_rng(args.random_seed),
        task_count=args.tasks,
        sessions_per_task=args.sessions_per_task,
        nodes_per_session=args.nodes_per_session,
        branching_factor=args.branching_factor,
        trajectory_depth=args.trajectory_depth,
        embedding_dim=args.embedding_dim,
    )
    generation_ms = (time.perf_counter() - generation_started) * 1000.0

    load_started = time.perf_counter()
    with DuckDBUnifiedStore(database, embedding_dim=args.embedding_dim) as backend:
        backend.units.upsert_units(units)
        with backend.transaction():
            for source, target, relation in edges:
                backend.graph.upsert_relationship(source, target, relation, {})
            backend.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS synthetic_dataset_meta (
                    key VARCHAR PRIMARY KEY,
                    value VARCHAR NOT NULL
                )
                """
            )
            metadata = {
                "tasks": args.tasks,
                "sessions_per_task": args.sessions_per_task,
                "nodes_per_session": args.nodes_per_session,
                "branching_factor": args.branching_factor,
                "trajectory_depth": args.trajectory_depth,
                "embedding_dim": args.embedding_dim,
                "random_seed": args.random_seed,
            }
            backend.connection.executemany(
                """
                INSERT INTO synthetic_dataset_meta VALUES (?, ?)
                ON CONFLICT (key) DO UPDATE SET value = excluded.value
                """,
                [(key, str(value)) for key, value in metadata.items()],
            )
    load_ms = (time.perf_counter() - load_started) * 1000.0

    logger.info("database=%s", database)
    logger.info(
        "dataset units=%d edges=%d generation_ms=%.2f load_ms=%.2f",
        len(units),
        len(edges),
        generation_ms,
        load_ms,
    )


if __name__ == "__main__":
    main()
