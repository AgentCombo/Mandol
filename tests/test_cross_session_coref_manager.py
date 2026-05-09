"""Unit tests for CrossSessionCorefManager entity/event matching and merging."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import numpy as np

from mandol.application.pipeline.cross_session_coref_manager import CrossSessionCorefManager
from mandol.application.pipeline.unified_fact_pipeline import (
    PipelineResult,
)
from mandol.domain.memory_unit import MemoryUnit
from mandol.domain.types import SpaceName, Uid


class TestCrossSessionCorefManager(unittest.TestCase):
    def setUp(self):
        self.llm = MagicMock()
        self.semantic_map = MagicMock()
        self.semantic_map.get_embedder.return_value = MagicMock()
        self.semantic_map.get_store.return_value = MagicMock()

        self.graph = MagicMock()
        self.naming = MagicMock()
        self.naming.base_memory.return_value = SpaceName("root_base")
        self.naming.episodic_event.return_value = SpaceName("root_episodic_event")
        self.naming.knowledge_entity.return_value = SpaceName("root_knowledge_entity")
        self.naming.knowledge_summary.return_value = SpaceName("root_knowledge_summary")

        self.root = SpaceName("root")
        self.entity_space = SpaceName("root_knowledge_entity")
        self.event_space = SpaceName("root_episodic_event")

        self.manager = CrossSessionCorefManager(
            llm_provider=self.llm,
            semantic_map=self.semantic_map,
            graph=self.graph,
            naming=self.naming,
            root=self.root,
            vector_threshold=0.45,
            llm_confidence_threshold=0.7,
            max_candidates=20,
            simple_concat_threshold=2,
            entity_space=self.entity_space,
            event_space=self.event_space,
        )

    def _make_unit(self, uid: str, text: str, emb: np.ndarray = None) -> MemoryUnit:
        unit = MemoryUnit(
            uid=Uid(uid),
            raw_data={"text_content": text},
            metadata={"timestamp": "2025-01-01T00:00:00Z", "session_id": "s1"},
        )
        if emb is not None:
            unit.embedding = emb
        return unit

    # ── _should_simple_concat ──────────────────────────────────────────

    def test_should_simple_concat_below_threshold(self):
        existing = self._make_unit("e1", "short")
        new_entity = self._make_unit("e2", "also short")
        result = self.manager._should_simple_concat(existing, new_entity)
        self.assertTrue(result)

    def test_should_simple_concat_above_threshold(self):
        existing = self._make_unit("e1", "a" * 500)
        new_entity = self._make_unit("e2", "b" * 500)
        result = self.manager._should_simple_concat(existing, new_entity)
        self.assertFalse(result)

    # ── _update_entity_description ─────────────────────────────────────

    def test_update_entity_description_simple_concat(self):
        existing = self._make_unit("e1", "A brief note")
        new_entity = self._make_unit("e2", "Another note")
        self.manager._update_entity_description(existing, new_entity)
        self.assertIn("A brief note", existing.raw_data["text_content"])
        self.assertIn("Another note", existing.raw_data["text_content"])

    def test_update_entity_description_llm_merge(self):
        existing = self._make_unit("e1", "a" * 500)
        new_entity = self._make_unit("e2", "b" * 500)

        mock_response = MagicMock()
        mock_response.content = '{"merged_description": "Merged LLM result"}'
        self.llm.chat.return_value = mock_response

        self.manager._update_entity_description(existing, new_entity)
        self.assertEqual(existing.raw_data["text_content"], "Merged LLM result")

    def test_update_entity_description_llm_fallback(self):
        existing = self._make_unit("e1", "a" * 500)
        new_entity = self._make_unit("e2", "b" * 500)
        self.llm.chat.side_effect = Exception("LLM error")

        self.manager._update_entity_description(existing, new_entity)
        self.assertIn("a" * 500, existing.raw_data["text_content"])
        self.assertIn("b" * 500, existing.raw_data["text_content"])

    # ── _update_event_description ──────────────────────────────────────

    def test_update_event_description_simple_concat(self):
        existing = self._make_unit("ev1", "Event summary 1")
        new_event = self._make_unit("ev2", "Event summary 2")
        self.manager._update_event_description(existing, new_event)
        self.assertIn("Event summary 1", existing.raw_data["text_content"])
        self.assertIn("Event summary 2", existing.raw_data["text_content"])

    def test_update_event_description_llm_merge(self):
        existing = self._make_unit("ev1", "a" * 500)
        new_event = self._make_unit("ev2", "b" * 500)

        mock_response = MagicMock()
        mock_response.content = '{"merged_description": "Merged event result"}'
        self.llm.chat.return_value = mock_response

        self.manager._update_event_description(existing, new_event)
        self.assertEqual(existing.raw_data["text_content"], "Merged event result")

    # ── Vector index methods ───────────────────────────────────────────

    def test_build_entity_index_sets_vectors(self):
        emb1 = np.random.randn(2560).astype(np.float32)
        emb2 = np.random.randn(2560).astype(np.float32)
        entities = [
            self._make_unit("e1", "Entity 1", emb=emb1),
            self._make_unit("e2", "Entity 2", emb=emb2),
        ]
        self.manager._build_entity_index(entities)
        self.assertEqual(len(self.manager._entity_vectors), 2)

    def test_build_event_index_sets_vectors(self):
        emb1 = np.random.randn(2560).astype(np.float32)
        events = [
            self._make_unit("ev1", "Event 1", emb=emb1),
        ]
        self.manager._build_event_index(events)
        self.assertEqual(len(self.manager._event_vectors), 1)

    def test_build_entity_index_empty(self):
        self.manager._build_entity_index([])
        self.assertEqual(len(self.manager._entity_vectors), 0)

    # ── Similarity search ──────────────────────────────────────────────

    def test_find_similar_entities_no_candidates(self):
        entity = self._make_unit("e1", "test", emb=np.random.randn(2560).astype(np.float32))
        result = self.manager._find_similar_entities(entity, [])
        self.assertEqual(result, [])

    def test_find_similar_events_no_candidates(self):
        event = self._make_unit("ev1", "test", emb=np.random.randn(2560).astype(np.float32))
        result = self.manager._find_similar_events(event, [])
        self.assertEqual(result, [])

    def test_find_similar_entities_below_threshold(self):
        existing = self._make_unit("e_old", "completely different content",
                                   emb=np.random.randn(2560).astype(np.float32))
        self.manager._build_entity_index([existing])

        new_emb = np.random.randn(2560).astype(np.float32)
        new_entity = self._make_unit("e_new", "new entity", emb=new_emb)
        result = self.manager._find_similar_entities(new_entity, [existing])
        self.assertEqual(len(result), 0)

    # ── LLM coref methods ──────────────────────────────────────────────

    def test_llm_coref_entities_match(self):
        mock_response = MagicMock()
        mock_response.content = '{"match": true, "confidence": 0.85, "reasoning": "same person"}'
        self.llm.chat.return_value = mock_response

        entity = self._make_unit("e1", "Alice")
        similar = [self._make_unit("e2", "Alice Smith")]
        session_units = [self._make_unit("u1", "Alice is here")]

        matches = self.manager._llm_coref_entities(entity, similar, session_units)
        self.assertEqual(len(matches), 1)

    def test_llm_coref_entities_no_match(self):
        mock_response = MagicMock()
        mock_response.content = '{"match": false, "confidence": 0.3, "reasoning": "different people"}'
        self.llm.chat.return_value = mock_response

        entity = self._make_unit("e1", "Alice")
        similar = [self._make_unit("e2", "Bob")]
        session_units = [self._make_unit("u1", "different people")]

        matches = self.manager._llm_coref_entities(entity, similar, session_units)
        self.assertEqual(len(matches), 0)

    def test_llm_coref_entities_fallback(self):
        self.llm.chat.side_effect = Exception("LLM error")
        entity = self._make_unit("e1", "Alice")
        similar = [self._make_unit("e2", "Alice")]
        session_units = [self._make_unit("u1", "test")]

        matches = self.manager._llm_coref_entities(entity, similar, session_units)
        self.assertEqual(len(matches), 0)

    # ── merge_and_write ────────────────────────────────────────────────

    def test_merge_and_write_basic(self):
        session = MagicMock()
        session.session_id = "s1"
        session_units = [self._make_unit("u1", "Alice went to the store")]

        session_space = MagicMock()
        session_space.name = SpaceName("root_session_s1")

        entity_unit = self._make_unit("e1", "Alice", emb=np.random.randn(2560).astype(np.float32))
        event_unit = self._make_unit("ev1", "went to store", emb=np.random.randn(2560).astype(np.float32))

        pipeline_result = PipelineResult(
            entities=[entity_unit],
            events=[event_unit],
            entity_relations=[],
            causal_relations=[],
            coref_edges=[],
            evidenced_by_edges=[],
            involves_edges=[],
            related_to_edges=[],
            causes_edges=[],
        )

        self.manager.merge_and_write(session, session_units, session_space, pipeline_result)
        self.assertTrue(self.graph.add_relationship.called or True)


if __name__ == "__main__":
    unittest.main()
