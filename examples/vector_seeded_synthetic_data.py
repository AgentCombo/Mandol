"""Shared synthetic Experience Graph data generation helpers."""

from __future__ import annotations

import numpy as np

from mandol.domain import MemoryUnit


def synthetic_unit(
    uid: str,
    kind: str,
    *,
    embedding=None,
    **metadata,
) -> MemoryUnit:
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
) -> tuple[list[MemoryUnit], list[tuple[str, str, str]]]:
    """Generate task/session/node rows and typed trajectory edges."""
    units: list[MemoryUnit] = []
    edges: list[tuple[str, str, str]] = []

    for task_index in range(task_count):
        task_uid = f"task-{task_index:04d}"
        task_vector = rng.normal(size=embedding_dim).astype(np.float32)
        units.append(
            synthetic_unit(
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
                synthetic_unit(
                    session_uid,
                    "session",
                    search_algorithm=("mcts" if session_index % 2 == 0 else "evolution"),
                )
            )
            edges.append((session_uid, task_uid, "BELONGS_TO"))

            for node_index in range(nodes_per_session):
                node_uid = f"node-{task_index:04d}-{session_index:02d}-{node_index:02d}"
                units.append(
                    synthetic_unit(
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
                                synthetic_unit(
                                    child_uid,
                                    "trajectory",
                                    fitness_score=float(rng.random()),
                                    depth=depth,
                                )
                            )
                            edges.append((parent_uid, child_uid, "HAS_CHILD"))
                            next_frontier.append(child_uid)
                    frontier = next_frontier

                if frontier:
                    edges.append((frontier[0], node_uid, "HAS_CHILD"))

    return units, edges


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("value must be positive")
    return parsed
