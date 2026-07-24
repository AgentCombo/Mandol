"""Run vector-seeded typed joins and graph traversal entirely in memory.

This example needs no model download, API key, or database.  Run it from the
repository root:

    python examples/vector_seeded_in_memory.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mandol.domain.memory_unit import MemoryUnit
from mandol.domain.types import Uid
from mandol.infrastructure import (
    InMemoryCosineVectorIndex,
    InMemoryGraphStore,
    InMemoryUnifiedQueryStore,
    InMemoryUnitStore,
)
from mandol.query import (
    VectorSeededTraversalSpec,
    format_query_log,
    render_query_diagram,
)

logger = logging.getLogger("mandol.vector_seeded_in_memory")


def memory(
    uid: str,
    kind: str,
    *,
    embedding: list[float] | None = None,
    **metadata,
) -> MemoryUnit:
    return MemoryUnit(
        uid=Uid(uid),
        raw_data={"text_content": uid},
        metadata={"kind": kind, **metadata},
        embedding=(
            None
            if embedding is None
            else np.asarray(embedding, dtype=np.float32)
        ),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    units = InMemoryUnitStore()
    vectors = InMemoryCosineVectorIndex(dim=2)
    graph = InMemoryGraphStore()

    records = [
        memory("task-vector", "task", embedding=[1.0, 0.0]),
        memory("task-unrelated", "task", embedding=[0.0, 1.0]),
        memory("session-1", "session"),
        memory("node-best", "node", fitness_score=0.95, is_buggy=False),
        memory("node-lower", "node", fitness_score=0.40, is_buggy=False),
        memory("node-buggy", "node", fitness_score=0.99, is_buggy=True),
        memory("trajectory-1", "node"),
        memory("trajectory-2", "node"),
    ]
    units.upsert_units(records)
    vectors.upsert(
        [
            (record.uid, record.embedding)
            for record in records
            if record.embedding is not None
        ]
    )
    for source, target, rel_type in [
        ("session-1", "task-vector", "BELONGS_TO"),
        ("node-best", "session-1", "IN_SESSION"),
        ("node-lower", "session-1", "IN_SESSION"),
        ("node-buggy", "session-1", "IN_SESSION"),
        ("node-best", "trajectory-1", "HAS_CHILD"),
        ("trajectory-1", "trajectory-2", "HAS_CHILD"),
    ]:
        graph.upsert_relationship(Uid(source), Uid(target), rel_type, {})

    executor = InMemoryUnifiedQueryStore(
        unit_store=units,
        vector_index=vectors,
        graph_store=graph,
    )
    spec = VectorSeededTraversalSpec(
        query_vector=np.asarray([1.0, 0.0], dtype=np.float32),
        vector_k=1,
        seed_metadata_equals={"kind": "task"},
        node_metadata_equals={"kind": "node", "is_buggy": False},
        nodes_per_seed=1,
        max_hops=2,
    )
    execution = executor.vector_seeded_graph_traversal(spec, profile=True)

    logger.info("PHYSICAL PLAN\n%s", executor.explain_vector_seeded_graph_traversal(spec))
    logger.info("\nRESULT ROWS")
    for row in execution.rows:
        logger.info(
            "seed=%s session=%s node=%s current=%s depth=%d path=%s",
            row.seed_uid,
            row.session_uid,
            row.node_uid,
            row.current_uid,
            row.depth,
            " -> ".join(row.path),
        )
    logger.info("\nEXECUTION BREAKDOWN\n%s", format_query_log(spec, execution))
    logger.info("\nQUERY DIAGRAM\n%s", render_query_diagram(spec, execution))


if __name__ == "__main__":
    main()
