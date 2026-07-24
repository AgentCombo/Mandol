"""Generate a larger synthetic Experience Graph and run annotated queries."""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np

from mandol.domain import MemoryUnit
from mandol.infrastructure import DuckDBUnifiedStore
from mandol.query import (
    VectorSeededTraversalSpec,
    format_query_log,
    render_query_diagram,
)

logger = logging.getLogger("mandol.vector_seeded_large")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=positive_int, default=30)
    parser.add_argument("--sessions-per-task", type=positive_int, default=3)
    parser.add_argument("--nodes-per-session", type=positive_int, default=6)
    parser.add_argument("--branching-factor", type=positive_int, default=2)
    parser.add_argument("--trajectory-depth", type=positive_int, default=2)
    parser.add_argument("--queries", type=positive_int, default=3)
    parser.add_argument("--vector-k", type=positive_int, default=5)
    parser.add_argument("--nodes-per-seed", type=positive_int, default=2)
    parser.add_argument("--result-limit", type=positive_int, default=200)
    parser.add_argument("--show-rows", type=positive_int, default=5)
    parser.add_argument("--embedding-dim", type=positive_int, default=8)
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--database", default=":memory:")
    return parser.parse_args()


def unit(uid: str, kind: str, *, embedding=None, **metadata) -> MemoryUnit:
    return MemoryUnit(
        uid=uid,
        raw_data={"text_content": f"synthetic {kind} {uid}"},
        metadata={"kind": kind, **metadata},
        embedding=embedding,
    )


def generate_dataset(
    *,
    rng: np.random.Generator,
    task_count: int,
    sessions_per_task: int,
    nodes_per_session: int,
    branching_factor: int,
    trajectory_depth: int,
    embedding_dim: int,
) -> tuple[list[MemoryUnit], list[tuple[str, str, str]], list[np.ndarray]]:
    units: list[MemoryUnit] = []
    edges: list[tuple[str, str, str]] = []
    task_vectors: list[np.ndarray] = []

    for task_index in range(task_count):
        task_uid = f"task-{task_index:04d}"
        task_vector = rng.normal(size=embedding_dim).astype(np.float32)
        task_vectors.append(task_vector)
        units.append(
            unit(
                task_uid,
                "task",
                embedding=task_vector,
                bucket=task_index % 3,
                difficulty=task_index % 5,
            )
        )

        for session_index in range(sessions_per_task):
            session_uid = f"session-{task_index:04d}-{session_index:02d}"
            units.append(
                unit(
                    session_uid,
                    "session",
                    search_algorithm=("mcts" if session_index % 2 == 0 else "evolution"),
                )
            )
            edges.append((session_uid, task_uid, "BELONGS_TO"))

            for node_index in range(nodes_per_session):
                node_uid = f"node-{task_index:04d}-{session_index:02d}-{node_index:02d}"
                units.append(
                    unit(
                        node_uid,
                        "node",
                        fitness_score=float(rng.random()),
                        is_buggy=bool(rng.random() < 0.15),
                        generation=node_index,
                    )
                )
                edges.append((node_uid, session_uid, "IN_SESSION"))

                frontier = [node_uid]
                for depth in range(1, trajectory_depth + 1):
                    next_frontier: list[str] = []
                    for parent_index, parent_uid in enumerate(frontier):
                        for branch in range(branching_factor):
                            child_uid = (
                                f"trajectory-{task_index:04d}-{session_index:02d}-"
                                f"{node_index:02d}-{depth:02d}-{parent_index:03d}-"
                                f"{branch:02d}"
                            )
                            units.append(
                                unit(
                                    child_uid,
                                    "trajectory",
                                    fitness_score=float(rng.random()),
                                    depth=depth,
                                )
                            )
                            edges.append((parent_uid, child_uid, "HAS_CHILD"))
                            next_frontier.append(child_uid)
                    frontier = next_frontier

                # Add one back-edge per trajectory tree to exercise path-level
                # cycle detection without changing the expected maximum depth.
                if frontier:
                    edges.append((frontier[0], node_uid, "HAS_CHILD"))

    return units, edges, task_vectors


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rng = np.random.default_rng(args.random_seed)

    started = time.perf_counter()
    units, edges, task_vectors = generate_dataset(
        rng=rng,
        task_count=args.tasks,
        sessions_per_task=args.sessions_per_task,
        nodes_per_session=args.nodes_per_session,
        branching_factor=args.branching_factor,
        trajectory_depth=args.trajectory_depth,
        embedding_dim=args.embedding_dim,
    )

    with DuckDBUnifiedStore(
        args.database,
        embedding_dim=args.embedding_dim,
    ) as backend:
        backend.units.upsert_units(units)
        with backend.transaction():
            for source, target, relation in edges:
                backend.graph.upsert_relationship(source, target, relation, {})

        build_ms = (time.perf_counter() - started) * 1000.0
        logger.info(
            "dataset units=%d edges=%d tasks=%d build_ms=%.2f",
            len(units),
            len(edges),
            args.tasks,
            build_ms,
        )

        for query_index in range(args.queries):
            task_index = (query_index * 7) % args.tasks
            spec = VectorSeededTraversalSpec(
                query_vector=task_vectors[task_index],
                vector_k=args.vector_k,
                seed_metadata_equals={
                    "kind": "task",
                    "bucket": task_index % 3,
                },
                node_metadata_equals={"kind": "node", "is_buggy": False},
                nodes_per_seed=args.nodes_per_seed,
                max_hops=args.trajectory_depth,
                result_limit=args.result_limit,
            )
            execution = backend.queries.vector_seeded_graph_traversal(
                spec,
                profile=True,
            )

            logger.info("\n=== QUERY %d target=%s ===", query_index + 1, task_index)
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
