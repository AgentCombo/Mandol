"""Unit tests for SemanticMapService add, search, and space operations."""


from mandol.application import SemanticMapService
from mandol.domain import MemoryUnit
from mandol.infrastructure import InMemoryCosineVectorIndex, InMemoryUnitStore
from mandol.ports import StaticEmbeddingProvider


def test_semantic_map_add_and_search_text():
    """Test that units added to a space are searchable by text."""
    store = InMemoryUnitStore()
    embedder = StaticEmbeddingProvider(dim=4, fill=1.0)
    index = InMemoryCosineVectorIndex(dim=4)

    sm = SemanticMapService(store=store, index=index, embedder=embedder)

    u1 = MemoryUnit(uid="u1", raw_data={"text_content": "hello"})
    u2 = MemoryUnit(uid="u2", raw_data={"text_content": "world"})
    sm.add_unit(u1, space_names=["docs"])
    sm.add_unit(u2, space_names=["docs"])

    results = sm.search_by_text("query", top_k=2, space_names=["docs"])
    assert len(results) == 2
    assert {r[0].uid for r in results} == {"u1", "u2"}


def test_get_units_in_spaces_union_intersection():
    """Test union and intersection modes for get_units_in_spaces."""
    store = InMemoryUnitStore()
    embedder = StaticEmbeddingProvider(dim=4, fill=1.0)
    index = InMemoryCosineVectorIndex(dim=4)

    sm = SemanticMapService(store=store, index=index, embedder=embedder)

    u1 = MemoryUnit(uid="u1", raw_data={"text_content": "a"})
    u2 = MemoryUnit(uid="u2", raw_data={"text_content": "b"})
    sm.add_unit(u1, space_names=["s1", "s2"])
    sm.add_unit(u2, space_names=["s2"])

    union = sm.get_units_in_spaces(["s1", "s2"], mode="union")
    assert {u.uid for u in union} == {"u1", "u2"}

    inter = sm.get_units_in_spaces(["s1", "s2"], mode="intersection")
    assert {u.uid for u in inter} == {"u1"}
