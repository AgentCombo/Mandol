"""Unit tests for AdaptiveVectorIndex promotion, search, and consistency checks.

Tests cover: factory support, upsert/delete, space promotion at threshold,
search in promoted/unpromoted spaces, consistency verification, rebuild,
and search reason tracking.
"""

from __future__ import annotations

import numpy as np
import pytest

from mandol.application.semantic_map import SemanticMapService
from mandol.domain.memory_unit import MemoryUnit
from mandol.domain.types import SpaceName, Uid
from mandol.infrastructure.adaptive_vector_index import (
    AdaptiveVectorIndex,
    IndexConsistencyError,
    VectorIndexFactory,
)
from mandol.infrastructure.faiss_vector_index import FaissVectorIndex
from mandol.infrastructure.in_memory_unit_store import InMemoryUnitStore


class StubEmbeddingProvider:
    def __init__(self, dim: int):
        self._dim = dim

    def embedding_dim(self) -> int:
        return self._dim

    def embed_text(self, texts):
        return [np.ones(self._dim, dtype=np.float32) * (hash(t) % 100) / 100.0 for t in texts]

    def embed_texts(self, texts):
        return self.embed_text(texts)


class StubReranker:
    def rerank(self, query, units, *, top_k=10):
        scored = [(u, float(len(u.raw_data.get("text_content", "")) % 10)) for u in units]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


def _orthogonal_emb(uid: str, dim: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    idx = int(uid.replace("u", "")) % dim
    vec[idx] = 1.0
    return vec


class TestVectorIndexFactory:
    def test_supports_flat(self):
        assert VectorIndexFactory.supports_type("Flat")
        assert VectorIndexFactory.supports_type("IDMap,Flat")

    def test_supports_ivf(self):
        assert VectorIndexFactory.supports_type("IVF")

    def test_supports_hnsw(self):
        assert VectorIndexFactory.supports_type("HNSW")

    def test_supports_pq(self):
        assert VectorIndexFactory.supports_type("PQ")

    def test_create_default(self):
        idx = VectorIndexFactory.create(128)
        assert isinstance(idx, FaissVectorIndex)
        assert idx.dim() == 128

    def test_create_with_type(self):
        idx = VectorIndexFactory.create(128, index_type="IDMap,Flat")
        assert isinstance(idx, FaissVectorIndex)


class TestAdaptiveVectorIndexStructure:
    def test_global_faiss_initialized(self):
        abi = AdaptiveVectorIndex(dim=4)
        assert isinstance(abi._global_faiss, FaissVectorIndex)

    def test_space_faiss_initialized_empty(self):
        abi = AdaptiveVectorIndex(dim=4)
        assert abi._space_faiss == {}

    def test_unpromoted_vectors_initialized_empty(self):
        abi = AdaptiveVectorIndex(dim=4)
        assert abi._unpromoted_vectors == {}


class TestAdaptiveVectorIndexUpsert:
    def test_upsert_to_global_faiss(self):
        abi = AdaptiveVectorIndex(dim=4)
        e = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        abi.upsert_to_space([(Uid("u1"), e)], SpaceName("sp"))

        assert Uid("u1") in abi._global_faiss._uid_to_internal

    def test_upsert_increments_space_count(self):
        abi = AdaptiveVectorIndex(dim=4, promote_threshold=3)
        e = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        abi.upsert_to_space([(Uid("u1"), e)], SpaceName("sp"))
        assert abi._space_counts[SpaceName("sp")] == 1

        abi.upsert_to_space([(Uid("u2"), e)], SpaceName("sp"))
        assert abi._space_counts[SpaceName("sp")] == 2

    def test_upsert_same_uid_same_space_no_double_count(self):
        abi = AdaptiveVectorIndex(dim=4)
        e = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        abi.upsert_to_space([(Uid("u1"), e)], SpaceName("sp"))
        abi.upsert_to_space([(Uid("u1"), e)], SpaceName("sp"))
        assert abi._space_counts[SpaceName("sp")] == 1

    def test_delete_removes_from_global_faiss(self):
        abi = AdaptiveVectorIndex(dim=4)
        e = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        abi.upsert_to_space([(Uid("u1"), e)], SpaceName("sp"))

        abi.delete([Uid("u1")])
        assert Uid("u1") not in abi._global_faiss._uid_to_internal

    def test_delete_from_space_removes_from_global_only(self):
        abi = AdaptiveVectorIndex(dim=4, promote_threshold=3)
        e = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        abi.upsert_to_space([(Uid("u1"), e), (Uid("u2"), e)], SpaceName("sp"))
        assert abi._space_counts[SpaceName("sp")] == 2

        abi.delete_from_space([Uid("u1")], SpaceName("sp"))
        assert Uid("u1") not in abi._global_faiss._uid_to_internal
        assert abi._space_counts[SpaceName("sp")] == 1


class TestAdaptiveVectorIndexPromotion:
    def test_promotion_at_threshold(self):
        abi = AdaptiveVectorIndex(dim=4, promote_threshold=3)
        e = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        for i in range(3):
            abi.upsert_to_space([(Uid(f"u{i}"), e.copy())], SpaceName("sp"))

        assert SpaceName("sp") in abi._space_faiss
        assert isinstance(abi._space_faiss[SpaceName("sp")], FaissVectorIndex)

    def test_promotion_clears_unpromoted_vectors(self):
        abi = AdaptiveVectorIndex(dim=4, promote_threshold=3)
        e = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        for i in range(3):
            abi.upsert_to_space([(Uid(f"u{i}"), e.copy())], SpaceName("sp"))

        for i in range(3):
            assert Uid(f"u{i}") not in abi._unpromoted_vectors

    def test_promotion_leaves_other_spaces_unaltered(self):
        abi = AdaptiveVectorIndex(dim=4, promote_threshold=3)
        e = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        for i in range(3):
            abi.upsert_to_space([(Uid(f"u{i}"), e.copy())], SpaceName("sp_a"))
        for i in range(2):
            abi.upsert_to_space([(Uid(f"b{i}"), e.copy())], SpaceName("sp_b"))

        assert SpaceName("sp_a") in abi._space_faiss
        assert SpaceName("sp_b") not in abi._space_faiss

    def test_incremental_add_to_promoted_space(self):
        abi = AdaptiveVectorIndex(dim=4, promote_threshold=3)
        e = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        for i in range(3):
            abi.upsert_to_space([(Uid(f"u{i}"), e.copy())], SpaceName("sp"))
        assert SpaceName("sp") in abi._space_faiss

        abi.upsert_to_space([(Uid("u3"), e)], SpaceName("sp"))
        assert Uid("u3") in abi._space_faiss[SpaceName("sp")]._uid_to_internal

    def test_force_promote(self):
        abi = AdaptiveVectorIndex(dim=4, promote_threshold=100)
        e = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        abi.upsert_to_space([(Uid("u1"), e)], SpaceName("sp"))
        assert SpaceName("sp") not in abi._space_faiss

        abi.force_promote(SpaceName("sp"))
        assert SpaceName("sp") in abi._space_faiss


class TestAdaptiveVectorIndexSearch:
    def test_search_uses_global_faiss(self):
        abi = AdaptiveVectorIndex(dim=4)
        e1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        e2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        abi.upsert([(Uid("u1"), e1), (Uid("u2"), e2)])

        hits = abi.search(e1, top_k=2)
        assert len(hits) == 2

    def test_search_global_after_promotion(self):
        abi = AdaptiveVectorIndex(dim=4, promote_threshold=3)
        e = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        for i in range(3):
            abi.upsert_to_space([(Uid(f"u{i}"), e.copy())], SpaceName("sp"))

        assert Uid("u0") in abi._global_faiss._uid_to_internal
        hits = abi.search(e, top_k=10)
        uids = [u for u, _ in hits]
        assert Uid("u0") in uids

    def test_search_in_space_promoted_uses_space_faiss(self):
        abi = AdaptiveVectorIndex(dim=4, promote_threshold=3)
        e1 = _orthogonal_emb("u1", 4)
        e2 = _orthogonal_emb("u2", 4)
        abi.upsert_to_space([(Uid("u1"), e1), (Uid("u2"), e2)], SpaceName("sp"))

        hits = abi.search_in_space(e1, SpaceName("sp"), candidates=None, top_k=2)
        assert len(hits) == 2

    def test_search_in_space_unpromoted_uses_brute_force(self):
        abi = AdaptiveVectorIndex(dim=4, promote_threshold=100)
        e1 = _orthogonal_emb("u1", 4)
        e2 = _orthogonal_emb("u2", 4)
        abi.upsert_to_space([(Uid("u1"), e1), (Uid("u2"), e2)], SpaceName("sp"))

        assert SpaceName("sp") not in abi._space_faiss
        hits = abi.search_in_space(e1, SpaceName("sp"), candidates=None, top_k=2)
        assert len(hits) == 2

    def test_search_in_space_filters_to_target_space(self):
        abi = AdaptiveVectorIndex(dim=4, promote_threshold=100)
        e = _orthogonal_emb("u1", 4)
        abi.upsert_to_space([(Uid("u1"), e)], SpaceName("sp_a"))
        abi.upsert_to_space([(Uid("u2"), e)], SpaceName("sp_b"))

        hits = abi.search_in_space(e, SpaceName("sp_a"), candidates=None, top_k=5)
        uids = [u for u, _ in hits]
        assert uids == [Uid("u1")]

    def test_search_in_space_with_candidates(self):
        abi = AdaptiveVectorIndex(dim=4, promote_threshold=100)
        e = _orthogonal_emb("u1", 4)
        abi.upsert_to_space([(Uid("u1"), e)], SpaceName("sp"))

        hits = abi.search_in_space(e, SpaceName("sp"), candidates={Uid("u1")}, top_k=5)
        assert [u for u, _ in hits] == [Uid("u1")]

    def test_search_in_space_empty_when_no_target_vectors(self):
        abi = AdaptiveVectorIndex(dim=4, promote_threshold=100)
        e = _orthogonal_emb("u99", 4)
        hits = abi.search_in_space(e, SpaceName("nonexistent"), candidates=None, top_k=5)
        assert hits == []

    def test_global_search_returns_all_vectors(self):
        abi = AdaptiveVectorIndex(dim=4, promote_threshold=2)
        e1 = _orthogonal_emb("u1", 4)
        e3 = _orthogonal_emb("u3", 4)

        abi.upsert_to_space([(Uid("u1"), e1), (Uid("u2"), e1)], SpaceName("sp"))
        abi.upsert([(Uid("u3"), e3)])

        hits = abi.search(e3, top_k=10)
        uids = [u for u, _ in hits]
        assert Uid("u1") in uids
        assert Uid("u3") in uids

    def test_global_search_no_duplicates(self):
        abi = AdaptiveVectorIndex(dim=4, promote_threshold=2)
        e1 = _orthogonal_emb("u1", 4)
        abi.upsert_to_space([(Uid("u1"), e1), (Uid("u2"), e1)], SpaceName("sp"))
        abi.upsert([(Uid("u3"), e1)])

        hits = abi.search(e1, top_k=10)
        seen = set()
        for u, _ in hits:
            assert u not in seen
            seen.add(u)


class TestAdaptiveVectorIndexConsistency:
    def test_verify_consistency_clean(self):
        abi = AdaptiveVectorIndex(dim=4)
        e = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        abi.upsert_to_space([(Uid("u1"), e)], SpaceName("sp"))
        errors = abi.verify_consistency(raises=False)
        assert errors == []

    def test_verify_consistency_uid_not_in_global(self):
        abi = AdaptiveVectorIndex(dim=4)
        abi._global_faiss._uid_to_internal[Uid("orphan")] = 0
        errors = abi.verify_consistency(raises=False)
        assert any("_global_faiss but not in _uid_to_spaces" in e for e in errors)

    def test_verify_consistency_untracked_unpromoted(self):
        abi = AdaptiveVectorIndex(dim=4)
        e = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        abi._unpromoted_vectors[Uid("orphan")] = e
        errors = abi.verify_consistency(raises=False)
        assert any("unpromoted_vectors" in e for e in errors)

    def test_verify_consistency_space_count_mismatch(self):
        abi = AdaptiveVectorIndex(dim=4)
        e = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        abi.upsert_to_space([(Uid("u1"), e)], SpaceName("sp"))
        abi._space_counts[SpaceName("sp")] = 999
        errors = abi.verify_consistency(raises=False)
        assert any("space_counts" in e for e in errors)

    def test_verify_consistency_promoted_space_missing_uids(self):
        abi = AdaptiveVectorIndex(dim=4, promote_threshold=3)
        e = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        abi.upsert_to_space([(Uid("u1"), e), (Uid("u2"), e), (Uid("u3"), e)], SpaceName("sp"))
        assert SpaceName("sp") in abi._space_faiss
        abi._space_faiss[SpaceName("sp")].delete([Uid("u1")])
        errors = abi.verify_consistency(raises=False)
        assert any("missing UIDs" in e for e in errors)

    def test_verify_consistency_raises(self):
        abi = AdaptiveVectorIndex(dim=4)
        abi._space_counts[SpaceName("sp")] = 999
        with pytest.raises(IndexConsistencyError):
            abi.verify_consistency(raises=True)

    def test_get_stats(self):
        abi = AdaptiveVectorIndex(dim=256, promote_threshold=10)
        stats = abi.get_stats()
        assert stats["dim"] == 256
        assert stats["threshold"] == 10
        assert stats["global_faiss_size"] == 0
        assert stats["space_faiss_count"] == 0
        assert stats["promoted_spaces"] == []


class TestAdaptiveVectorIndexRebuild:
    def test_rebuild_clears_everything(self):
        abi = AdaptiveVectorIndex(dim=4, promote_threshold=3)
        e = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        abi.upsert_to_space([(Uid("u1"), e)], SpaceName("sp"))

        abi.rebuild([(Uid("u2"), np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))])

        assert Uid("u1") not in abi._uid_to_spaces
        assert Uid("u2") in abi._uid_to_spaces
        assert len(abi._space_faiss) == 0
        assert len(abi._unpromoted_vectors) == 0

    def test_rebuild_space_tier2(self):
        abi = AdaptiveVectorIndex(dim=4, promote_threshold=5)
        e = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        items = [(Uid(f"u{i}"), e.copy()) for i in range(5)]
        abi.rebuild_space(SpaceName("sp"), items)
        assert SpaceName("sp") in abi._space_faiss

    def test_rebuild_space_below_threshold_not_promoted(self):
        abi = AdaptiveVectorIndex(dim=4, promote_threshold=100)
        e = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        abi.rebuild_space(SpaceName("sp"), [(Uid("u1"), e)])
        assert SpaceName("sp") not in abi._space_faiss
        assert Uid("u1") in abi._unpromoted_vectors


class TestAdaptiveVectorIndexSearchReason:
    def test_search_in_space_reason_tier2(self):
        abi = AdaptiveVectorIndex(dim=4, promote_threshold=2)
        e = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        abi.upsert_to_space([(Uid("u1"), e), (Uid("u2"), e)], SpaceName("sp"))
        _, reason = abi.search_in_space_with_reason(e, SpaceName("sp"), top_k=5)
        assert reason == "tier2_space_faiss"

    def test_search_in_space_reason_tier1(self):
        abi = AdaptiveVectorIndex(dim=4, promote_threshold=100)
        e = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        abi.upsert_to_space([(Uid("u1"), e)], SpaceName("sp"))
        _, reason = abi.search_in_space_with_reason(e, SpaceName("sp"), top_k=5)
        assert reason == "tier1_brute_force"


class TestSemanticMapServiceWithABI:
    def test_add_unit_to_promoted_space(self):
        store = InMemoryUnitStore()
        abi = AdaptiveVectorIndex(dim=4)
        sm = SemanticMapService(store=store, index=abi, embedder=StubEmbeddingProvider(4))

        unit = MemoryUnit(uid="u1", raw_data={"text_content": "hello"})
        sm.add_unit(unit, space_names=["sp"])

        assert sm._abi is abi
        assert Uid("u1") in abi._global_faiss._uid_to_internal

    def test_search_in_space_respects_space_boundary(self):
        store = InMemoryUnitStore()
        abi = AdaptiveVectorIndex(dim=4)
        sm = SemanticMapService(store=store, index=abi, embedder=StubEmbeddingProvider(4))

        u1 = MemoryUnit(uid="u1", raw_data={"text_content": "apple"})
        u2 = MemoryUnit(uid="u2", raw_data={"text_content": "banana"})
        sm.add_unit(u1, space_names=["fruit"])
        sm.add_unit(u2, space_names=["baked"])

        hits = sm.search_in_space("apple", SpaceName("fruit"), top_k=5)
        assert all(h[0].metadata.get("spaces", []) == ["fruit"] for h in hits)

    def test_verify_consistency_after_operations(self):
        store = InMemoryUnitStore()
        abi = AdaptiveVectorIndex(dim=4, promote_threshold=10)
        sm = SemanticMapService(store=store, index=abi, embedder=StubEmbeddingProvider(4))

        for i in range(5):
            sm.add_unit(
                MemoryUnit(uid=f"u{i}", raw_data={"text_content": f"text{i}"}),
                space_names=["sp1"],
            )

        errors = sm._abi.verify_consistency(raises=False)
        assert errors == []
