"""Unit tests for MemoryRetrievalService group-based retrieval."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from mandol.application.services._retrieval import (
    MemoryRetrievalService,
    RETRIEVAL_GROUP_BASE,
    RETRIEVAL_GROUP_ENTITY,
    RETRIEVAL_GROUP_EVENT,
    RETRIEVAL_GROUP_SUMMARY,
)
from mandol.domain.memory_unit import MemoryUnit
from mandol.domain.types import SpaceName, Uid
from mandol.retrieval.types import SearchHit

_PIPELINE = "mandol.retrieval.pipeline"
_BASE_SPACE = "root_base"


def _patch_retriever(func):
    """Decorator that patches both HybridRetriever and HybridRetrieverConfig."""
    return patch(_PIPELINE + ".HybridRetrieverConfig", MagicMock())(
        patch(_PIPELINE + ".HybridRetriever")(func)
    )


class TestRetrievalServiceConstants(unittest.TestCase):
    def test_constants(self):
        self.assertEqual(RETRIEVAL_GROUP_BASE, "base")
        self.assertEqual(RETRIEVAL_GROUP_EVENT, "event")
        self.assertEqual(RETRIEVAL_GROUP_ENTITY, "entity")
        self.assertEqual(RETRIEVAL_GROUP_SUMMARY, "summary")


class TestMemoryRetrievalService(unittest.TestCase):
    def setUp(self):
        self.semantic_map = MagicMock()
        self.semantic_map.get_embedder.return_value = MagicMock()
        self.semantic_map.get_reranker.return_value = None
        self.semantic_map.get_store.return_value = MagicMock()

        self.graph = MagicMock()
        self.naming = MagicMock()
        self.naming.base_memory.return_value = SpaceName("root_base")
        self.naming.episodic_event.return_value = SpaceName("root_episodic_event")
        self.naming.knowledge_entity.return_value = SpaceName("root_knowledge_entity")
        self.naming.episodic_summary.return_value = SpaceName("root_episodic_summary")
        self.naming.knowledge_summary.return_value = SpaceName("root_knowledge_summary")
        self.naming.insights.return_value = SpaceName("root_insights")
        self.naming.emotional.return_value = SpaceName("root_emotional")
        self.naming.procedural.return_value = SpaceName("root_procedural")

        self.root = SpaceName("test_root")
        self.config = MagicMock()

        self.service = MemoryRetrievalService(
            semantic_map=self.semantic_map,
            graph=self.graph,
            naming=self.naming,
            root=self.root,
            config=self.config,
        )

    def _make_unit(self, uid: str, text: str) -> MemoryUnit:
        return MemoryUnit(
            uid=Uid(uid),
            raw_data={"text_content": text},
            metadata={"timestamp": "2025-01-01T00:00:00Z"},
        )

    def _make_hit(self, unit: MemoryUnit, score: float = 0.9) -> SearchHit:
        return SearchHit(unit=unit, final_score=score, scores={}, ranks={})

    # ── _get_retrieval_groups ──────────────────────────────────────────

    def test_get_retrieval_groups_returns_all_four(self):
        groups = self.service._get_retrieval_groups()
        self.assertIsInstance(groups, dict)
        self.assertIn(RETRIEVAL_GROUP_BASE, groups)
        self.assertIn(RETRIEVAL_GROUP_EVENT, groups)
        self.assertIn(RETRIEVAL_GROUP_ENTITY, groups)
        self.assertIn(RETRIEVAL_GROUP_SUMMARY, groups)

    def test_get_retrieval_groups_summary_includes_multiple_spaces(self):
        groups = self.service._get_retrieval_groups()
        label, space_names = groups[RETRIEVAL_GROUP_SUMMARY]
        self.assertEqual(len(space_names), 3)

    # ── _search_group ──────────────────────────────────────────────────

    @_patch_retriever
    def test_search_group_lazy_init(self, mock_hr_cls):
        mock_hr = MagicMock()
        mock_hr.search.return_value = [self._make_hit(self._make_unit("u1", "test"))]
        mock_hr_cls.return_value = mock_hr

        self.assertIsNone(self.service._hybrid_retriever)
        self.service._search_group("q", [SpaceName("s")], 5, False)
        self.assertIsNotNone(self.service._hybrid_retriever)
        mock_hr_cls.assert_called_once()

        self.service._search_group("q2", [SpaceName("s")], 5, False)
        self.assertEqual(mock_hr_cls.call_count, 1)

    @_patch_retriever
    def test_search_group_returns_unit_score_tuples(self, mock_hr_cls):
        unit = self._make_unit("u1", "hello")
        mock_hr = MagicMock()
        mock_hr.search.return_value = [self._make_hit(unit, 0.85)]
        mock_hr_cls.return_value = mock_hr

        result = self.service._search_group("q", [SpaceName("s")], 5, True)
        self.assertEqual(len(result), 1)
        self.assertIs(result[0][0], unit)
        self.assertAlmostEqual(result[0][1], 0.85)

    # ── holistic_retrieve ──────────────────────────────────────────────

    @_patch_retriever
    def test_holistic_retrieve_without_reranker(self, mock_hr_cls):
        u1 = self._make_unit("u1", "first")
        u2 = self._make_unit("u2", "second")
        mock_hr = MagicMock()
        mock_hr.search.return_value = [
            self._make_hit(u1, 0.9),
            self._make_hit(u2, 0.8),
        ]
        mock_hr_cls.return_value = mock_hr

        hits = self.service.holistic_retrieve("query", top_k=10, use_rerank=False)
        self.assertGreaterEqual(len(hits), 1)
        self.assertIsInstance(hits[0], SearchHit)

    @_patch_retriever
    def test_holistic_retrieve_with_reranker(self, mock_hr_cls):
        u1 = self._make_unit("u1", "first")
        mock_hr = MagicMock()
        mock_hr.search.return_value = [
            self._make_hit(u1, 0.9),
        ]
        mock_hr_cls.return_value = mock_hr

        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [(u1, 0.95)]
        self.semantic_map.get_reranker.return_value = mock_reranker

        hits = self.service.holistic_retrieve("query", top_k=10, use_rerank=True)
        self.assertEqual(len(hits), 1)
        self.assertAlmostEqual(hits[0].final_score, 0.95)
        self.assertIn("rerank", hits[0].scores)

    @_patch_retriever
    def test_holistic_retrieve_deduplicates_across_groups(self, mock_hr_cls):
        u1 = self._make_unit("u1", "shared")
        mock_hr = MagicMock()
        mock_hr.search.return_value = [
            self._make_hit(u1, 0.9),
            self._make_hit(u1, 0.7),
        ]
        mock_hr_cls.return_value = mock_hr

        hits = self.service.holistic_retrieve("query", top_k=10, use_rerank=False)
        uids = [str(h.unit.uid) for h in hits]
        self.assertEqual(len(uids), len(set(uids)))

    @_patch_retriever
    def test_holistic_retrieve_empty_result(self, mock_hr_cls):
        mock_hr = MagicMock()
        mock_hr.search.return_value = []
        mock_hr_cls.return_value = mock_hr

        hits = self.service.holistic_retrieve("query", top_k=10, use_rerank=False)
        self.assertEqual(hits, [])

    # ── holistic_retrieve auto_build_if_empty ───────────────────────────

    @_patch_retriever
    def test_holistic_retrieve_triggers_build_when_high_level_empty(self, mock_hr_cls):
        """Verify build_trigger is called when high-level spaces are empty."""
        base_unit = self._make_unit("b1", "base content")
        mock_hr = MagicMock()
        mock_hr.search.return_value = [self._make_hit(base_unit, 0.9)]
        mock_hr_cls.return_value = mock_hr

        build_called = []

        def side_effect_get_units(spaces):
            space_str = str(spaces[0]) if spaces else ""
            if _BASE_SPACE in space_str:
                return [base_unit]
            return []

        self.semantic_map.get_units_in_spaces = MagicMock(
            side_effect=side_effect_get_units
        )

        def build_trigger():
            build_called.append(True)

        hits = self.service.holistic_retrieve(
            "query",
            top_k=10,
            use_rerank=False,
            auto_build_if_empty=True,
            build_trigger=build_trigger,
        )
        self.assertEqual(len(build_called), 1)
        self.assertGreaterEqual(len(hits), 1)

    @_patch_retriever
    def test_holistic_retrieve_skips_build_when_high_level_populated(self, mock_hr_cls):
        """Verify build_trigger is not called when high-level spaces have data."""
        base_unit = self._make_unit("b1", "base content")
        entity_unit = self._make_unit("e1", "entity content")
        mock_hr = MagicMock()
        mock_hr.search.return_value = [self._make_hit(base_unit, 0.9)]
        mock_hr_cls.return_value = mock_hr

        build_called = []

        def side_effect_get_units(spaces):
            space_str = str(spaces[0]) if spaces else ""
            if "knowledge_entity" in space_str:
                return [entity_unit]
            if _BASE_SPACE in space_str:
                return [base_unit]
            return []

        self.semantic_map.get_units_in_spaces = MagicMock(
            side_effect=side_effect_get_units
        )

        def build_trigger():
            build_called.append(True)

        hits = self.service.holistic_retrieve(
            "query",
            top_k=10,
            use_rerank=False,
            auto_build_if_empty=True,
            build_trigger=build_trigger,
        )
        self.assertEqual(len(build_called), 0)
        self.assertGreaterEqual(len(hits), 1)

    @_patch_retriever
    def test_holistic_retrieve_skips_build_when_base_empty(self, mock_hr_cls):
        """Verify build_trigger is not called when no base memory exists."""
        mock_hr = MagicMock()
        mock_hr.search.return_value = []
        mock_hr_cls.return_value = mock_hr

        build_called = []

        def side_effect_get_units(spaces):
            return []

        self.semantic_map.get_units_in_spaces = MagicMock(
            side_effect=side_effect_get_units
        )

        def build_trigger():
            build_called.append(True)

        hits = self.service.holistic_retrieve(
            "query",
            top_k=10,
            use_rerank=False,
            auto_build_if_empty=True,
            build_trigger=build_trigger,
        )
        self.assertEqual(len(build_called), 0)
        self.assertEqual(hits, [])

    @_patch_retriever
    def test_holistic_retrieve_skips_build_when_disabled(self, mock_hr_cls):
        """Verify build_trigger is not called when auto_build_if_empty=False."""
        base_unit = self._make_unit("b1", "base content")
        mock_hr = MagicMock()
        mock_hr.search.return_value = [self._make_hit(base_unit, 0.9)]
        mock_hr_cls.return_value = mock_hr

        build_called = []

        def side_effect_get_units(spaces):
            space_str = str(spaces[0]) if spaces else ""
            if _BASE_SPACE in space_str:
                return [base_unit]
            return []

        self.semantic_map.get_units_in_spaces = MagicMock(
            side_effect=side_effect_get_units
        )

        def build_trigger():
            build_called.append(True)

        hits = self.service.holistic_retrieve(
            "query",
            top_k=10,
            use_rerank=False,
            auto_build_if_empty=False,
            build_trigger=build_trigger,
        )
        self.assertEqual(len(build_called), 0)
        self.assertGreaterEqual(len(hits), 1)

    @_patch_retriever
    def test_holistic_retrieve_no_trigger_when_none(self, mock_hr_cls):
        """Verify no error when build_trigger is None."""
        base_unit = self._make_unit("b1", "base content")
        mock_hr = MagicMock()
        mock_hr.search.return_value = [self._make_hit(base_unit, 0.9)]
        mock_hr_cls.return_value = mock_hr

        def side_effect_get_units(spaces):
            space_str = str(spaces[0]) if spaces else ""
            if _BASE_SPACE in space_str:
                return [base_unit]
            return []

        self.semantic_map.get_units_in_spaces = MagicMock(
            side_effect=side_effect_get_units
        )

        hits = self.service.holistic_retrieve(
            "query",
            top_k=10,
            use_rerank=False,
            auto_build_if_empty=True,
            build_trigger=None,
        )
        self.assertGreaterEqual(len(hits), 1)

    # ── _are_high_level_spaces_empty ───────────────────────────────────

    def test_are_high_level_spaces_empty_when_all_empty(self):
        """Verify _are_high_level_spaces_empty returns True when all empty."""
        self.semantic_map.get_units_in_spaces = MagicMock(return_value=[])
        self.assertTrue(self.service._are_high_level_spaces_empty())

    def test_are_high_level_spaces_empty_when_one_has_data(self):
        """Verify _are_high_level_spaces_empty returns False when any space has data."""
        entity_unit = self._make_unit("e1", "entity")

        def side_effect_get_units(spaces):
            space_str = str(spaces[0]) if spaces else ""
            if "knowledge_entity" in space_str:
                return [entity_unit]
            return []

        self.semantic_map.get_units_in_spaces = MagicMock(
            side_effect=side_effect_get_units
        )
        self.assertFalse(self.service._are_high_level_spaces_empty())

    # ── _is_base_memory_empty ──────────────────────────────────────────

    def test_is_base_memory_empty_when_empty(self):
        """Verify _is_base_memory_empty returns True when base is empty."""
        self.semantic_map.get_units_in_spaces = MagicMock(return_value=[])
        self.assertTrue(self.service._is_base_memory_empty())

    def test_is_base_memory_empty_when_populated(self):
        """Verify _is_base_memory_empty returns False when base has data."""
        base_unit = self._make_unit("b1", "base")
        self.semantic_map.get_units_in_spaces = MagicMock(return_value=[base_unit])
        self.assertFalse(self.service._is_base_memory_empty())

    # ── retrieve_in_space ──────────────────────────────────────────────

    @_patch_retriever
    def test_retrieve_in_space(self, mock_hr_cls):
        unit = self._make_unit("u1", "hello")
        mock_hr = MagicMock()
        mock_hr.search.return_value = [self._make_hit(unit, 0.9)]
        mock_hr_cls.return_value = mock_hr

        hits = self.service.retrieve_in_space("q", "root_knowledge_entity", top_k=5)
        self.assertEqual(len(hits), 1)
        self.assertIsInstance(hits[0], tuple)
        self.assertEqual(len(hits[0]), 2)

    # ── retrieve_by_view ───────────────────────────────────────────────

    @_patch_retriever
    def test_retrieve_by_view_base_memory(self, mock_hr_cls):
        mock_hr = MagicMock()
        mock_hr.search.return_value = [self._make_hit(self._make_unit("u1", "x"), 0.9)]
        mock_hr_cls.return_value = mock_hr

        hits = self.service.retrieve_by_view("q", "base_memory", top_k=5)
        self.assertEqual(len(hits), 1)

    @_patch_retriever
    def test_retrieve_by_view_all_valid_views(self, mock_hr_cls):
        mock_hr = MagicMock()
        mock_hr.search.return_value = [self._make_hit(self._make_unit("u1", "x"), 0.9)]
        mock_hr_cls.return_value = mock_hr

        valid_views = [
            "base_memory", "entity_relation", "event_causal", "emotional",
            "episodic", "knowledge", "procedural", "insights",
        ]
        for view in valid_views:
            hits = self.service.retrieve_by_view("q", view, top_k=5)
            self.assertIsInstance(hits, list)

    def test_retrieve_by_view_unknown_raises_valueerror(self):
        with self.assertRaises(ValueError):
            self.service.retrieve_by_view("q", "nonexistent_view", top_k=5)


if __name__ == "__main__":
    unittest.main()
