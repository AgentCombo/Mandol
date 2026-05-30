"""Unit tests for new SemanticMapService methods: filter_memory_units and search."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import numpy as np

from mandol.application.semantic_map import SemanticMapService
from mandol.domain.memory_unit import MemoryUnit
from mandol.domain.types import Uid


def _make_unit(uid: str, text: str = "content", metadata=None) -> MemoryUnit:
    """Create a test MemoryUnit."""
    _metadata = dict(metadata) if metadata else {}
    _metadata.setdefault("timestamp", "2025-01-01T00:00:00Z")
    return MemoryUnit(
        uid=Uid(uid),
        raw_data={"text_content": text},
        metadata=_metadata,
    )


class TestFilterMemoryUnits(unittest.TestCase):
    """Tests for filter_memory_units."""

    def setUp(self):
        self.store = MagicMock()
        self.index = MagicMock()
        self.index.dim.return_value = 768
        self.service = SemanticMapService(store=self.store, index=self.index)

    def test_returns_all_when_no_filter(self):
        u1 = _make_unit("u1")
        u2 = _make_unit("u2")
        self.store.get_space.return_value = None
        self.service.get_units_in_spaces = MagicMock(return_value=[u1, u2])

        result = self.service.filter_memory_units(
            candidate_units=[u1, u2], filter_condition=None
        )
        self.assertEqual(len(result), 2)

    def test_filters_by_eq_operator(self):
        u1 = _make_unit("u1", metadata={"entity_type": "Person"})
        u2 = _make_unit("u2", metadata={"entity_type": "Organization"})

        result = self.service.filter_memory_units(
            candidate_units=[u1, u2],
            filter_condition={"metadata.entity_type": {"eq": "Person"}},
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(str(result[0].uid), "u1")

    def test_filters_by_neq_operator(self):
        u1 = _make_unit("u1", metadata={"type": "A"})
        u2 = _make_unit("u2", metadata={"type": "B"})

        result = self.service.filter_memory_units(
            candidate_units=[u1, u2],
            filter_condition={"metadata.type": {"neq": "A"}},
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(str(result[0].uid), "u2")

    def test_filters_by_in_operator(self):
        u1 = _make_unit("u1", metadata={"tag": "alpha"})
        u2 = _make_unit("u2", metadata={"tag": "beta"})
        u3 = _make_unit("u3", metadata={"tag": "gamma"})

        result = self.service.filter_memory_units(
            candidate_units=[u1, u2, u3],
            filter_condition={"metadata.tag": {"in": ["alpha", "gamma"]}},
        )
        self.assertEqual(len(result), 2)
        uids = {str(u.uid) for u in result}
        self.assertIn("u1", uids)
        self.assertIn("u3", uids)

    def test_filters_by_contains_operator(self):
        u1 = _make_unit("u1", "hello world")
        u2 = _make_unit("u2", "goodbye")

        result = self.service.filter_memory_units(
            candidate_units=[u1, u2],
            filter_condition={"raw_data.text_content": {"contains": "hello"}},
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(str(result[0].uid), "u1")

    def test_filters_by_gt_operator(self):
        u1 = _make_unit("u1", metadata={"count": 5})
        u2 = _make_unit("u2", metadata={"count": 10})

        result = self.service.filter_memory_units(
            candidate_units=[u1, u2],
            filter_condition={"metadata.count": {"gt": 7}},
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(str(result[0].uid), "u2")

    def test_filters_by_lt_operator(self):
        u1 = _make_unit("u1", metadata={"count": 3})
        u2 = _make_unit("u2", metadata={"count": 10})

        result = self.service.filter_memory_units(
            candidate_units=[u1, u2],
            filter_condition={"metadata.count": {"lt": 5}},
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(str(result[0].uid), "u1")

    def test_filters_by_gte_operator(self):
        u1 = _make_unit("u1", metadata={"count": 5})
        u2 = _make_unit("u2", metadata={"count": 10})

        result = self.service.filter_memory_units(
            candidate_units=[u1, u2],
            filter_condition={"metadata.count": {"gte": 5}},
        )
        self.assertEqual(len(result), 2)

    def test_filters_by_lte_operator(self):
        u1 = _make_unit("u1", metadata={"count": 5})
        u2 = _make_unit("u2", metadata={"count": 10})

        result = self.service.filter_memory_units(
            candidate_units=[u1, u2],
            filter_condition={"metadata.count": {"lte": 5}},
        )
        self.assertEqual(len(result), 1)

    def test_multiple_conditions_anded(self):
        u1 = _make_unit("u1", metadata={"type": "Person", "active": True})
        u2 = _make_unit("u2", metadata={"type": "Person", "active": False})
        u3 = _make_unit("u3", metadata={"type": "Org", "active": True})

        result = self.service.filter_memory_units(
            candidate_units=[u1, u2, u3],
            filter_condition={
                "metadata.type": {"eq": "Person"},
                "metadata.active": {"eq": True},
            },
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(str(result[0].uid), "u1")

    def test_missing_field_skips_unit(self):
        u1 = _make_unit("u1", metadata={"has_field": True})
        u2 = _make_unit("u2", metadata={})

        result = self.service.filter_memory_units(
            candidate_units=[u1, u2],
            filter_condition={"metadata.nonexistent": {"eq": "whatever"}},
        )
        self.assertEqual(len(result), 0)

    def test_empty_filter_returns_all_candidates(self):
        u1 = _make_unit("u1")
        u2 = _make_unit("u2")
        result = self.service.filter_memory_units(
            candidate_units=[u1, u2], filter_condition={}
        )
        self.assertEqual(len(result), 2)


class TestUnifiedSearch(unittest.TestCase):
    """Tests for unified search() method."""

    def setUp(self):
        self.store = MagicMock()
        self.index = MagicMock()
        self.index.dim.return_value = 768
        self.embedder = MagicMock()
        self.embedder.embedding_dim.return_value = 768
        self.service = SemanticMapService(
            store=self.store, index=self.index, embedder=self.embedder,
        )

    def test_dense_search_with_text(self):
        u1 = _make_unit("u1", "hello world")
        u1.embedding = np.random.rand(768).astype(np.float32)
        self.store.get_unit.return_value = u1
        self.index.search.return_value = [(Uid("u1"), 0.9)]

        result = self.service.search("hello", k=5, retriever_type="dense")
        self.assertEqual(len(result), 1)
        self.assertEqual(str(result[0][0].uid), "u1")

    def test_dense_search_with_vector(self):
        u1 = _make_unit("u1")
        u1.embedding = np.random.rand(768).astype(np.float32)
        self.store.get_unit.return_value = u1
        self.index.search.return_value = [(Uid("u1"), 0.9)]

        query_vec = np.random.rand(768).astype(np.float32)
        result = self.service.search(query_vec, k=5, retriever_type="dense")
        self.assertEqual(len(result), 1)

    def test_defaults_to_dense(self):
        u1 = _make_unit("u1", "test")
        u1.embedding = np.random.rand(768).astype(np.float32)
        self.store.get_unit.return_value = u1
        self.index.search.return_value = [(Uid("u1"), 0.9)]

        result = self.service.search("test", k=5)
        self.assertEqual(len(result), 1)

    def test_bm25_search(self):
        u1 = _make_unit("u1", "hello world test document")
        u2 = _make_unit("u2", "unrelated content here")
        self.store.list_units.return_value = [u1, u2]

        result = self.service.search("hello world", k=2, retriever_type="bm25")
        self.assertGreater(len(result), 0)

    def test_sparse_search(self):
        u1 = _make_unit("u1", "machine learning is great")
        u2 = _make_unit("u2", "quantum physics")
        self.store.list_units.return_value = [u1, u2]

        result = self.service.search("machine learning", k=2, retriever_type="sparse")
        self.assertGreater(len(result), 0)

    def test_multi_retriever_fusion(self):
        u1 = _make_unit("u1", "hello world")
        u1.embedding = np.random.rand(768).astype(np.float32)
        self.store.get_unit.return_value = u1
        self.store.list_units.return_value = [u1]
        self.index.search.return_value = [(Uid("u1"), 0.9)]

        result = self.service.search(
            "hello", k=5, retrievers=["dense", "bm25"]
        )
        self.assertGreater(len(result), 0)

    def test_vector_query_with_bm25_raises(self):
        query_vec = np.random.rand(768).astype(np.float32)
        with self.assertRaises(ValueError):
            self.service.search(query_vec, k=5, retriever_type="bm25")

    def test_candidate_uids_filter(self):
        u1 = _make_unit("u1", "hello")
        u2 = _make_unit("u2", "hello too")
        u1.embedding = np.random.rand(768).astype(np.float32)
        u2.embedding = np.random.rand(768).astype(np.float32)
        self.store.get_unit.side_effect = lambda u: {
            Uid("u1"): u1, Uid("u2"): u2,
        }.get(Uid(str(u)))
        self.index.search.return_value = [(Uid("u1"), 0.9), (Uid("u2"), 0.8)]

        result = self.service.search(
            "hello", k=5, retriever_type="dense", candidate_uids=["u1"]
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(str(result[0][0].uid), "u1")


if __name__ == "__main__":
    unittest.main()
