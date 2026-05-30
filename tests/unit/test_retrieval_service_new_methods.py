"""Unit tests for new MemoryRetrievalService methods.

Tests: retrieve_event_causal_chain, retrieve_with_reasoning_path,
retrieve_entity_timeline, retrieve_session_context.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from mandol.application.services._retrieval import MemoryRetrievalService
from mandol.domain.memory_unit import MemoryUnit
from mandol.domain.types import SpaceName, Uid
from mandol.retrieval.types import (
    CausalChainResult,
    ReasoningHit,
    SearchHit,
    SessionContextResult,
    TimelineResult,
)


_PIPELINE = "mandol.retrieval.pipeline"
_SUBGRAPH = "mandol.retrieval.subgraph_hop"


def _make_unit(uid: str, text: str = "content", metadata=None) -> MemoryUnit:
    """Create a MemoryUnit."""
    _metadata = dict(metadata) if metadata else {}
    _metadata.setdefault("timestamp", "2025-01-01T00:00:00Z")
    return MemoryUnit(
        uid=Uid(uid),
        raw_data={"text_content": text},
        metadata=_metadata,
    )


def _make_hit(unit: MemoryUnit, score: float = 0.9) -> SearchHit:
    return SearchHit(unit=unit, final_score=score, scores={}, ranks={})


def _patch_retriever(func):
    """Decorator that patches HybridRetriever and HybridRetrieverConfig."""
    return patch(_PIPELINE + ".HybridRetrieverConfig", MagicMock())(
        patch(_PIPELINE + ".HybridRetriever")(func)
    )


def _patch_subgraph(func):
    """Decorator that patches SubgraphHopRetriever."""
    return patch(_SUBGRAPH + ".SubgraphHopRetriever")(func)


class BaseRetrievalServiceTest(unittest.TestCase):
    """Base setup for MemoryRetrievalService tests."""

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


class TestRetrieveEventCausalChain(BaseRetrievalServiceTest):
    """Tests for retrieve_event_causal_chain."""

    @_patch_retriever
    def test_returns_empty_when_no_events_found(self, mock_hr_cls):
        mock_hr = MagicMock()
        mock_hr.search.return_value = []
        mock_hr_cls.return_value = mock_hr

        result = self.service.retrieve_event_causal_chain("query")
        self.assertIsInstance(result, CausalChainResult)
        self.assertIsNone(result.root_event)
        self.assertEqual(result.chain, [])

    @_patch_retriever
    def test_returns_chain_with_root_event(self, mock_hr_cls):
        event = _make_unit("ev1", metadata={"name": "Project Delay"})
        mock_hr = MagicMock()
        mock_hr.search.return_value = [_make_hit(event, 0.95)]
        mock_hr_cls.return_value = mock_hr

        self.graph.bfs_expand_units.return_value = []
        self.graph.get_relationship.return_value = None

        result = self.service.retrieve_event_causal_chain("project delay")
        self.assertIsNotNone(result.root_event)
        self.assertEqual(str(result.root_event.uid), "ev1")

    @_patch_retriever
    def test_traces_causal_edges(self, mock_hr_cls):
        event = _make_unit("ev1")
        ev2 = _make_unit("ev2")
        mock_hr = MagicMock()
        mock_hr.search.return_value = [_make_hit(event, 0.95)]
        mock_hr_cls.return_value = mock_hr

        self.graph.bfs_expand_units.return_value = [event, ev2]
        self.graph.get_relationship.return_value = {"confidence": 0.9}
        self.graph.get_explicit_neighbors.return_value = []

        result = self.service.retrieve_event_causal_chain(
            "delay", direction="both", max_hops=2
        )
        self.assertIsInstance(result, CausalChainResult)
        self.assertGreater(len(result.chain), 0)


class TestRetrieveWithReasoningPath(BaseRetrievalServiceTest):
    """Tests for retrieve_with_reasoning_path."""

    @_patch_retriever
    @_patch_subgraph
    def test_returns_reasoning_hits(self, mock_sg_hr_cls, mock_hr_cls):
        unit = _make_unit("u1", "explanation")
        mock_hr = MagicMock()
        mock_hr.search.return_value = [_make_hit(unit, 0.9)]
        mock_hr_cls.return_value = mock_hr

        from mandol.retrieval.subgraph_hop import SubgraphHopHit

        mock_sg = MagicMock()
        mock_sg.search.return_value = [
            SubgraphHopHit(
                unit=unit,
                final_score=0.85,
                scores={"rrf": 0.85},
                reasoning_path=[],
            )
        ]
        mock_sg_hr_cls.return_value = mock_sg

        result = self.service.retrieve_with_reasoning_path(
            "why cancelled?", top_k=5
        )
        self.assertGreater(len(result), 0)
        self.assertIsInstance(result[0], ReasoningHit)
        self.assertEqual(result[0].final_score, 0.85)

    @_patch_retriever
    @_patch_subgraph
    def test_filters_rel_types(self, mock_sg_hr_cls, mock_hr_cls):
        unit = _make_unit("u1")
        mock_hr = MagicMock()
        mock_hr.search.return_value = [_make_hit(unit, 0.9)]
        mock_hr_cls.return_value = mock_hr

        from mandol.retrieval.subgraph_hop import SubgraphHopHit

        mock_sg = MagicMock()
        mock_sg.search.return_value = [
            SubgraphHopHit(unit=unit, final_score=0.8, scores={}, reasoning_path=[])
        ]
        mock_sg_hr_cls.return_value = mock_sg

        # Should not crash with filtered rel_types
        result = self.service.retrieve_with_reasoning_path(
            "query", rel_types=["CAUSES", "COREF"]
        )
        self.assertEqual(len(result), 1)


class TestRetrieveEntityTimeline(BaseRetrievalServiceTest):
    """Tests for retrieve_entity_timeline."""

    @_patch_retriever
    def test_returns_empty_when_no_entity(self, mock_hr_cls):
        mock_hr = MagicMock()
        mock_hr.search.return_value = []
        mock_hr_cls.return_value = mock_hr

        result = self.service.retrieve_entity_timeline("missing")
        self.assertIsInstance(result, TimelineResult)
        self.assertIsNone(result.entity)
        self.assertEqual(result.events, [])

    @_patch_retriever
    def test_returns_timeline_for_entity(self, mock_hr_cls):
        entity = _make_unit("ent", "Zhang San", metadata={"name": "Zhang San"})
        related = _make_unit("rel1", metadata={
            "timestamp": "2025-02-01T00:00:00Z"
        })

        mock_hr = MagicMock()
        mock_hr.search.return_value = [
            _make_hit(entity, 0.95),
            _make_hit(related, 0.8),
        ]
        mock_hr_cls.return_value = mock_hr

        self.graph.get_explicit_neighbors.return_value = [related]

        result = self.service.retrieve_entity_timeline("Zhang San")
        self.assertIsNotNone(result.entity)
        self.assertEqual(str(result.entity.uid), "ent")

    @_patch_retriever
    def test_filters_by_time_range(self, mock_hr_cls):
        entity = _make_unit("ent", "Zhang San", metadata={"name": "Zhang San"})
        old_unit = _make_unit("old", metadata={"timestamp": "2024-01-01T00:00:00Z"})
        new_unit = _make_unit("new", metadata={"timestamp": "2025-06-01T00:00:00Z"})

        mock_hr = MagicMock()
        mock_hr.search.return_value = [
            _make_hit(entity, 0.95),
        ]
        mock_hr_cls.return_value = mock_hr

        self.graph.get_explicit_neighbors.return_value = [old_unit, new_unit]

        result = self.service.retrieve_entity_timeline(
            "Zhang San", time_range=("2025-01-01", "2025-12-31")
        )
        # Only new_unit should be within the time range
        timestamps = [ts for _, ts in result.events]
        self.assertTrue(all("2025" in ts for ts in timestamps if ts))


class TestRetrieveSessionContext(BaseRetrievalServiceTest):
    """Tests for retrieve_session_context."""

    @_patch_retriever
    def test_returns_empty_when_no_hits(self, mock_hr_cls):
        mock_hr = MagicMock()
        mock_hr.search.return_value = []
        mock_hr_cls.return_value = mock_hr

        result = self.service.retrieve_session_context("query")
        self.assertIsInstance(result, SessionContextResult)
        self.assertEqual(result.session_units, [])

    @_patch_retriever
    def test_groups_units_by_session(self, mock_hr_cls):
        u1 = _make_unit("u1", "content", metadata={"session_id": "sess1"})
        u2 = _make_unit("u2", "content", metadata={"session_id": "sess1"})
        u3 = _make_unit("u3", "content", metadata={"session_id": "sess2"})

        mock_hr = MagicMock()
        mock_hr.search.return_value = [
            _make_hit(u1, 0.9), _make_hit(u2, 0.8), _make_hit(u3, 0.7),
        ]
        mock_hr_cls.return_value = mock_hr

        result = self.service.retrieve_session_context("query")
        self.assertIn(result.session_id, {"sess1", "sess2"})
        self.assertGreater(len(result.session_units), 0)

    @_patch_retriever
    def test_explicit_session_id_filter(self, mock_hr_cls):
        u1 = _make_unit("u1", metadata={"session_id": "sess_a"})
        u2 = _make_unit("u2", metadata={"session_id": "sess_b"})

        mock_hr = MagicMock()
        mock_hr.search.return_value = [
            _make_hit(u1, 0.9), _make_hit(u2, 0.8),
        ]
        mock_hr_cls.return_value = mock_hr

        result = self.service.retrieve_session_context(
            "query", session_id="sess_b"
        )
        self.assertEqual(result.session_id, "sess_b")
        self.assertEqual(len(result.session_units), 1)

    @_patch_retriever
    def test_explicit_session_id_not_found(self, mock_hr_cls):
        u1 = _make_unit("u1", metadata={"session_id": "sess_a"})

        mock_hr = MagicMock()
        mock_hr.search.return_value = [_make_hit(u1, 0.9)]
        mock_hr_cls.return_value = mock_hr

        result = self.service.retrieve_session_context(
            "query", session_id="nonexistent"
        )
        self.assertEqual(result.session_id, "nonexistent")
        self.assertEqual(result.session_units, [])

    @_patch_retriever
    def test_falls_back_to_session_key(self, mock_hr_cls):
        u1 = _make_unit("u1", metadata={"session": "my_session"})

        mock_hr = MagicMock()
        mock_hr.search.return_value = [_make_hit(u1, 0.9)]
        mock_hr_cls.return_value = mock_hr

        result = self.service.retrieve_session_context("query")
        self.assertEqual(result.session_id, "my_session")
        self.assertEqual(len(result.session_units), 1)


if __name__ == "__main__":
    unittest.main()
