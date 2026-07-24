from __future__ import annotations

import numpy as np

from mandol.application.memory_system import MemorySystem, MemorySystemConfig
from mandol.domain.memory_unit import MemoryUnit
from mandol.domain.types import Uid
from mandol.infrastructure.in_memory_graph_store import InMemoryGraphStore
from mandol.infrastructure.in_memory_unit_store import InMemoryUnitStore
from mandol.infrastructure.in_memory_vector_index import InMemoryCosineVectorIndex
from mandol.infrastructure.stub_llm_provider import StubLLMProvider
from mandol.ports.embedding_provider import StaticEmbeddingProvider
from mandol.query import VectorSeededTraversalSpec


def test_memory_system_exposes_default_in_memory_multimodal_executor():
    units = InMemoryUnitStore()
    vectors = InMemoryCosineVectorIndex(2)
    edges = InMemoryGraphStore()
    system = MemorySystem(
        config=MemorySystemConfig(embedder_dim=2),
        embedder=StaticEmbeddingProvider(2),
        llm_provider=StubLLMProvider(),
        unit_store=units,
        vector_index=vectors,
        graph_store=edges,
    )
    records = [
        MemoryUnit(
            uid=Uid("task"),
            raw_data={"text_content": "task"},
            metadata={"kind": "task"},
            embedding=np.asarray([1.0, 0.0], dtype=np.float32),
        ),
        MemoryUnit(
            uid=Uid("session"),
            raw_data={"text_content": "session"},
            metadata={"kind": "session"},
        ),
        MemoryUnit(
            uid=Uid("node"),
            raw_data={"text_content": "node"},
            metadata={"kind": "node", "fitness_score": 1.0},
        ),
        MemoryUnit(
            uid=Uid("child"),
            raw_data={"text_content": "child"},
            metadata={"kind": "node"},
        ),
    ]
    units.upsert_units(records)
    vectors.upsert([(records[0].uid, records[0].embedding)])
    edges.upsert_relationship(Uid("session"), Uid("task"), "BELONGS_TO", {})
    edges.upsert_relationship(Uid("node"), Uid("session"), "IN_SESSION", {})
    edges.upsert_relationship(Uid("node"), Uid("child"), "HAS_CHILD", {})

    execution = system.vector_seeded_graph_traversal(
        VectorSeededTraversalSpec(
            query_vector=np.asarray([1.0, 0.0], dtype=np.float32),
            vector_k=1,
            seed_metadata_equals={"kind": "task"},
            node_metadata_equals={"kind": "node"},
            max_hops=1,
        ),
        profile=True,
    )

    assert [row.current_uid for row in execution.rows] == [Uid("node"), Uid("child")]
    assert execution.metrics.stages.session_join_rows == 1
    assert execution.metrics.stages.node_join_rows == 1
    assert system.queries.explain_vector_seeded_graph_traversal(
        VectorSeededTraversalSpec(
            query_vector=np.asarray([1.0, 0.0], dtype=np.float32),
        )
    ).startswith("IN_MEMORY_FIXED_PLAN")
