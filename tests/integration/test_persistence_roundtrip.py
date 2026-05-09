"""Integration tests for JSON persistence round-trip serialization.

Verifies that InMemoryUnitStore and related components can serialize
to JSON and restore their state without data loss.
"""

import numpy as np

from mandol.domain import MemorySpace, MemoryUnit
from mandol.infrastructure import InMemoryUnitStore


def test_in_memory_store_dump_load_json_roundtrip():
    store = InMemoryUnitStore()

    u1 = MemoryUnit(uid="u1", raw_data={"text_content": "hello"}, embedding=np.array([1, 2, 3], dtype=np.float32))
    u2 = MemoryUnit(uid="u2", raw_data={"text_content": "world"}, sparse_embedding=np.array([0.1, 0.0], dtype=np.float32))

    s1 = MemorySpace(name="docs")
    s1.add_unit("u1")
    s1.add_unit("u2")

    store.upsert_units([u1, u2])
    store.upsert_spaces([s1])

    dumped = store.dump_json()

    store2 = InMemoryUnitStore()
    store2.load_json(dumped)

    uuids = {u.uid for u in store2.list_units()}
    assert uuids == {"u1", "u2"}

    u1_loaded = store2.get_unit("u1")
    assert u1_loaded is not None
    assert u1_loaded.raw_data["text_content"] == "hello"
    assert np.allclose(u1_loaded.embedding, np.array([1, 2, 3], dtype=np.float32))

    s1_loaded = store2.get_space("docs")
    assert s1_loaded is not None
    assert {str(u) for u in s1_loaded.unit_uids} == {"u1", "u2"}
