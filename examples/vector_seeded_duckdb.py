"""Run an end-to-end vector-seeded graph traversal on unified DuckDB storage."""

from __future__ import annotations

import logging

import numpy as np

from mandol.application import SemanticGraphService, SemanticMapService
from mandol.domain import MemoryUnit
from mandol.infrastructure import DuckDBUnifiedStore
from mandol.ports import StaticEmbeddingProvider
from mandol.query import (
    VectorSeededTraversalSpec,
    format_query_log,
    render_query_diagram,
)

logger = logging.getLogger("mandol.vector_seeded_example")


def memory(
    uid: str,
    kind: str,
    *,
    embedding: list[float] | None = None,
    **metadata,
) -> MemoryUnit:
    return MemoryUnit(
        uid=uid,
        raw_data={"text_content": uid},
        metadata={"kind": kind, **metadata},
        embedding=np.asarray(embedding, dtype=np.float32) if embedding else None,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    with DuckDBUnifiedStore(embedding_dim=2) as backend:
        semantic_map = SemanticMapService(
            store=backend.units,
            index=backend.vectors,
            embedder=StaticEmbeddingProvider(dim=2),
        )
        semantic_graph = SemanticGraphService(
            semantic_map=semantic_map,
            graph_store=backend.graph,
        )

        # SemanticMap writes units and vectors into the same DuckDB transaction
        # managed by the compatibility facades.
        semantic_graph.add_unit(
            memory("task-vector-db", "task", embedding=[1.0, 0.0]),
            space_names=["experience"],
        )
        semantic_graph.add_unit(
            memory("task-unrelated", "task", embedding=[0.0, 1.0]),
            space_names=["experience"],
        )
        for unit in [
            memory("session-1", "session"),
            memory(
                "node-best",
                "node",
                fitness_score=0.95,
                is_buggy=False,
            ),
            memory(
                "node-lower",
                "node",
                fitness_score=0.40,
                is_buggy=False,
            ),
            memory("trajectory-1", "node"),
            memory("trajectory-2", "node"),
        ]:
            semantic_graph.add_unit(
                unit,
                space_names=["experience"],
                ensure_embedding=False,
            )

        # Relation joins are virtual graph patterns over typed edge rows.
        semantic_graph.add_relationship("session-1", "task-vector-db", "BELONGS_TO")
        semantic_graph.add_relationship("node-best", "session-1", "IN_SESSION")
        semantic_graph.add_relationship("node-lower", "session-1", "IN_SESSION")
        semantic_graph.add_relationship("node-best", "trajectory-1", "HAS_CHILD")
        semantic_graph.add_relationship("trajectory-1", "trajectory-2", "HAS_CHILD")

        spec = VectorSeededTraversalSpec(
            query_vector=np.asarray([1.0, 0.0], dtype=np.float32),
            vector_k=1,
            seed_metadata_equals={"kind": "task"},
            node_metadata_equals={"kind": "node", "is_buggy": False},
            nodes_per_seed=1,
            max_hops=2,
        )
        execution = backend.queries.vector_seeded_graph_traversal(
            spec,
            profile=True,
        )

        for row in execution.rows:
            logger.info(
                f"seed={row.seed_uid} session={row.session_uid} "
                f"node={row.node_uid} current={row.current_uid} "
                f"depth={row.depth} path={' -> '.join(row.path)}"
            )
        logger.info("\n%s", format_query_log(spec, execution))
        logger.info("\n%s", render_query_diagram(spec, execution))


if __name__ == "__main__":
    main()
