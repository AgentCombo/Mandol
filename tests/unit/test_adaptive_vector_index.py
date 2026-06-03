"""Unit tests for AdaptiveVectorIndex promotion, search, and rebuild.

Tests cover: upsert/delete, promotion at threshold, search in flat vs FAISS,
rebuild, and integration with SemanticMapService.
"""

from __future__ import annotations

import numpy as np

from mandol.application.semantic_map import SemanticMapService
from mandol.domain.memory_unit import MemoryUnit
from mandol.domain.types import SpaceName, Uid
from mandol.infrastructure.adaptive_vector_index import AdaptiveVectorIndex
from mandol.infrastructure.in_memory_unit_store import InMemoryUnitStore


class StubEmbeddingProvider:
    def __init__(self, dim: int):
        self._dim = dim

    def embedding_dim(self) -> int:
        return self._dim

    def embed_text(self, texts):
        return [np.ones(self._dim, dtype=np.float32) * (hash(str(t)) % 100) / 100.0 for t in texts]

    def embed_image_paths(self, image_paths):
        return [np.ones(self._dim, dtype=np.float32) for _ in image_paths]


def _orthogonal_emb(uid: str, dim: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    idx = int(uid.replace("u", "")) % dim
    vec[idx] = 1.0
    return vec


class TestAdaptiveVectorIndexInit:
    def test_dim_matches_constructor(self):
        abi = AdaptiveVectorIndex(dim=128)
        assert abi.dim() == 128

    def test_starts_with_flat_index(self):
        abi = AdaptiveVectorIndex(dim=4)
        assert abi._faiss_index is None

    def test_promote_threshold_default(self):
        abi = AdaptiveVectorIndex(dim=4)
        assert abi._promote_threshold == 100

    def test_promote_threshold_custom(self):
        abi = AdaptiveVectorIndex(dim=4, promote_threshold=5)
        assert abi._promote_threshold == 5


class TestAdaptiveVectorIndexUpsertDelete:
    def test_upsert_adds_to_active_index(self):
        abi = AdaptiveVectorIndex(dim=4)
        e = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        abi.upsert([(Uid("u1"), e)])
        hits = abi.search(e, top_k=5)
        assert len(hits) == 1
        assert hits[0][0] == Uid("u1")

    def test_upsert_multiple(self):
        abi = AdaptiveVectorIndex(dim=4)
        e1 = _orthogonal_emb("u1", 4)
        e2 = _orthogonal_emb("u2", 4)
        abi.upsert([(Uid("u1"), e1), (Uid("u2"), e2)])
        hits = abi.search(e1, top_k=5)
        assert len(hits) == 2

    def test_upsert_same_uid_overwrites(self):
        abi = AdaptiveVectorIndex(dim=4)
        e1 = _orthogonal_emb("u1", 4)
        e2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        abi.upsert([(Uid("u1"), e1)])
        abi.upsert([(Uid("u1"), e2)])
        hits = abi.search(e2, top_k=5)
        assert len(hits) == 1
        # Should match e2 more strongly (cosine ~1.0) than e1 (cosine ~0.0)
        assert hits[0][1] > 0.99

    def test_delete_removes_vector(self):
        abi = AdaptiveVectorIndex(dim=4)
        e = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        abi.upsert([(Uid("u1"), e)])
        abi.delete([Uid("u1")])
        hits = abi.search(e, top_k=5)
        assert len(hits) == 0

    def test_delete_nonexistent_is_noop(self):
        abi = AdaptiveVectorIndex(dim=4)
        abi.delete([Uid("nonexistent")])  # should not raise


class TestAdaptiveVectorIndexPromotion:
    def test_promotion_at_threshold(self):
        abi = AdaptiveVectorIndex(dim=4, promote_threshold=3)
        e = _orthogonal_emb("u0", 4)
        for i in range(3):
            abi.upsert([(Uid(f"u{i}"), e.copy())])
        assert abi._faiss_index is not None

    def test_no_promotion_below_threshold(self):
        abi = AdaptiveVectorIndex(dim=4, promote_threshold=10)
        e = _orthogonal_emb("u0", 4)
        for i in range(5):
            abi.upsert([(Uid(f"u{i}"), e.copy())])
        assert abi._faiss_index is None

    def test_search_works_after_promotion(self):
        abi = AdaptiveVectorIndex(dim=4, promote_threshold=3)
        for i in range(5):
            e = _orthogonal_emb(f"u{i}", 4)
            abi.upsert([(Uid(f"u{i}"), e)])
        assert abi._faiss_index is not None
        query = _orthogonal_emb("u0", 4)
        hits = abi.search(query, top_k=3)
        assert len(hits) == 3
        # u0 should be among the top results (FAISS is approximate so top-1 may vary)
        hit_uids = [u for u, _ in hits]
        assert Uid("u0") in hit_uids


class TestAdaptiveVectorIndexSearch:
    def test_search_returns_correct_top_k(self):
        abi = AdaptiveVectorIndex(dim=4)
        e1 = _orthogonal_emb("u1", 4)
        e2 = _orthogonal_emb("u2", 4)
        abi.upsert([(Uid("u1"), e1), (Uid("u2"), e2)])
        hits = abi.search(e1, top_k=1)
        assert len(hits) == 1

    def test_search_empty_index(self):
        abi = AdaptiveVectorIndex(dim=4)
        e = _orthogonal_emb("u1", 4)
        hits = abi.search(e, top_k=5)
        assert hits == []

    def test_search_in_space_respects_candidates(self):
        abi = AdaptiveVectorIndex(dim=4)
        e1 = _orthogonal_emb("u1", 4)
        e2 = _orthogonal_emb("u2", 4)
        abi.upsert([(Uid("u1"), e1), (Uid("u2"), e2)])
        hits = abi.search_in_space(e1, "sp", candidates={Uid("u1")}, top_k=5)
        assert [u for u, _ in hits] == [Uid("u1")]

    def test_search_in_space_no_candidates(self):
        abi = AdaptiveVectorIndex(dim=4)
        e1 = _orthogonal_emb("u1", 4)
        e2 = _orthogonal_emb("u2", 4)
        abi.upsert([(Uid("u1"), e1), (Uid("u2"), e2)])
        hits = abi.search_in_space(e1, "sp", candidates=None, top_k=5)
        assert len(hits) == 2

    def test_search_no_duplicates(self):
        abi = AdaptiveVectorIndex(dim=4, promote_threshold=3)
        e = _orthogonal_emb("u1", 4)
        for i in range(5):
            abi.upsert([(Uid(f"u{i}"), e.copy())])
        hits = abi.search(e, top_k=10)
        seen = set()
        for u, _ in hits:
            assert u not in seen
            seen.add(u)


class TestAdaptiveVectorIndexRebuild:
    def test_rebuild_clears_and_replaces(self):
        abi = AdaptiveVectorIndex(dim=4, promote_threshold=10)
        e = _orthogonal_emb("u1", 4)
        abi.upsert([(Uid("u1"), e)])
        assert len(abi.search(e, top_k=5)) == 1

        e2 = _orthogonal_emb("u2", 4)
        abi.rebuild([(Uid("u2"), e2)])
        # Old u1 should be gone
        hits = abi.search(e, top_k=5)
        assert len(hits) == 1
        assert hits[0][0] == Uid("u2")

    def test_rebuild_promotes_when_above_threshold(self):
        abi = AdaptiveVectorIndex(dim=4, promote_threshold=3)
        items = [(Uid(f"u{i}"), _orthogonal_emb(f"u{i}", 4)) for i in range(5)]
        abi.rebuild(items)
        assert abi._faiss_index is not None

    def test_rebuild_demotes_when_below_threshold(self):
        abi = AdaptiveVectorIndex(dim=4, promote_threshold=100)
        items = [(Uid(f"u{i}"), _orthogonal_emb(f"u{i}", 4)) for i in range(2)]
        abi.rebuild(items)
        assert abi._faiss_index is None


class TestSemanticMapServiceWithABI:
    def test_add_unit_indexes_in_abi(self):
        store = InMemoryUnitStore()
        abi = AdaptiveVectorIndex(dim=4)
        sm = SemanticMapService(store=store, index=abi, embedder=StubEmbeddingProvider(4))

        unit = MemoryUnit(uid=Uid("u1"), raw_data={"text_content": "hello"})
        sm.add_unit(unit, space_names=["sp"])

        assert sm._abi is abi
        hits = abi.search(np.ones(4, dtype=np.float32), top_k=5)
        assert len(hits) >= 1

    def test_search_in_space_respects_space_boundary(self):
        store = InMemoryUnitStore()
        abi = AdaptiveVectorIndex(dim=4)
        sm = SemanticMapService(store=store, index=abi, embedder=StubEmbeddingProvider(4))

        u1 = MemoryUnit(uid=Uid("u1"), raw_data={"text_content": "apple"})
        u2 = MemoryUnit(uid=Uid("u2"), raw_data={"text_content": "banana"})
        sm.add_unit(u1, space_names=["fruit"])
        sm.add_unit(u2, space_names=["baked"])

        hits = sm.search_in_space("apple", SpaceName("fruit"), top_k=5)
        assert len(hits) >= 1
        assert hits[0][0].uid == Uid("u1")

    def test_multiple_units_same_space(self):
        store = InMemoryUnitStore()
        abi = AdaptiveVectorIndex(dim=4)
        sm = SemanticMapService(store=store, index=abi, embedder=StubEmbeddingProvider(4))

        for i in range(5):
            sm.add_unit(
                MemoryUnit(uid=Uid(f"u{i}"), raw_data={"text_content": f"text{i}"}),
                space_names=["sp1"],
            )

        units = sm.get_units_in_spaces(["sp1"])
        assert len(units) == 5
