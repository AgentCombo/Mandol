"""Unit tests for CrossSessionCorefManager entity/event matching and merging."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import numpy as np

from mandol.application.pipeline.cross_session_coref_manager import CrossSessionCorefManager
from mandol.application.pipeline.unified_fact_pipeline import (
    ExtractedEntity,
    ExtractedEvent,
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

    # ── Name index building ───────────────────────────────────────────

    def test_build_entity_name_index_empty(self):
        result = self.manager._build_entity_name_index([])
        self.assertEqual(len(result), 0)
        self.assertIsInstance(result, dict)

    def test_build_entity_name_index_populates_keys(self):
        unit = MemoryUnit(
            uid=Uid("e1"),
            raw_data={
                "text_content": "Entity Alice(Person): A software engineer",
                "entity_name": "Alice",
                "entity_type": "Person",
                "aliases": ["Ally", "A"],
            },
            metadata={"session_id": "s1"},
        )
        result = self.manager._build_entity_name_index([unit])
        self.assertIn("alice", result)
        self.assertIn("ally", result)
        self.assertIn("a", result)
        self.assertIn("e1", result["alice"])

    def test_build_event_name_index_populates_keys(self):
        unit = MemoryUnit(
            uid=Uid("ev1"),
            raw_data={
                "text_content": "Event Conference: Tech summit in SF",
                "event_name": "Conference",
            },
            metadata={"session_id": "s1"},
        )
        result = self.manager._build_event_name_index([unit])
        self.assertIn("conference", result)
        self.assertIn("ev1", result["conference"])

    # ── Description merging ───────────────────────────────────────────

    def test_update_description_simple_concat(self):
        result = self.manager._update_description_simple(
            "A brief note", ["Another note"]
        )
        self.assertIn("A brief note", result)
        self.assertIn("Another note", result)

    def test_update_description_llm_merge(self):
        mock_response = MagicMock()
        mock_response.content = '{"merged_description": "Merged LLM result"}'
        self.llm.chat.return_value = mock_response

        result = self.manager._update_description_llm(
            "a" * 500, ["b" * 500], "TestEntity"
        )
        self.assertEqual(result, "Merged LLM result")

    def test_update_description_llm_fallback(self):
        self.llm.chat.side_effect = ConnectionError("LLM connection error")

        result = self.manager._update_description_llm(
            "existing_desc", ["new fact"], "TestEntity"
        )
        # Fallback to simple concat
        self.assertIn("existing_desc", result)
        self.assertIn("new fact", result)

    # ── Entity matching ───────────────────────────────────────────────

    def test_llm_judge_entity_match_no_candidates(self):
        entity = ExtractedEntity(
            entity_name="Alice",
            entity_type="Person",
            linked_id=None,
            is_new=True,
            confidence=0.9,
            mention_text="Alice",
            new_facts=["Works at Acme"],
            aliases=["Ally"],
        )
        matched_idx, confidence, canonical_name, new_aliases, reasoning = (
            self.manager._llm_judge_entity_match(entity, [])
        )
        self.assertIsNone(matched_idx)
        self.assertEqual(confidence, 0.0)

    def test_llm_judge_entity_match_found(self):
        mock_response = MagicMock()
        mock_response.content = (
            '{"matched_index": 0, "confidence": 0.85, '
            '"canonical_name_suggestion": "Alice Smith", '
            '"new_aliases": ["Ally"], "reasoning": "same person"}'
        )
        self.llm.chat.return_value = mock_response

        entity = ExtractedEntity(
            entity_name="Alice",
            entity_type="Person",
            linked_id=None,
            is_new=True,
            confidence=0.9,
            mention_text="Alice",
            new_facts=["Works at Acme"],
            aliases=["Ally"],
        )
        candidate = MemoryUnit(
            uid=Uid("e_existing"),
            raw_data={
                "text_content": "Entity Alice Smith(Person): Works at Acme Corp",
                "entity_name": "Alice Smith",
                "entity_type": "Person",
                "aliases": ["Ally"],
            },
            metadata={"session_id": "s0"},
        )
        matched_idx, confidence, canonical_name, new_aliases, reasoning = (
            self.manager._llm_judge_entity_match(entity, [candidate])
        )
        self.assertEqual(matched_idx, 0)
        self.assertGreaterEqual(confidence, 0.8)

    def test_llm_judge_entity_match_not_found(self):
        mock_response = MagicMock()
        mock_response.content = (
            '{"matched_index": null, "confidence": 0.3, '
            '"canonical_name_suggestion": null, '
            '"new_aliases": [], "reasoning": "different people"}'
        )
        self.llm.chat.return_value = mock_response

        entity = ExtractedEntity(
            entity_name="Alice",
            entity_type="Person",
            linked_id=None,
            is_new=True,
            confidence=0.9,
            mention_text="Alice",
            new_facts=["Works at Acme"],
            aliases=[],
        )
        candidate = MemoryUnit(
            uid=Uid("e_bob"),
            raw_data={
                "text_content": "Entity Bob(Person): Works at Globex",
                "entity_name": "Bob",
                "entity_type": "Person",
            },
            metadata={"session_id": "s0"},
        )
        matched_idx, confidence, canonical_name, new_aliases, reasoning = (
            self.manager._llm_judge_entity_match(entity, [candidate])
        )
        self.assertIsNone(matched_idx)

    def test_llm_judge_entity_match_fallback(self):
        self.llm.chat.side_effect = ConnectionError("LLM connection error")
        entity = ExtractedEntity(
            entity_name="Alice",
            entity_type="Person",
            linked_id=None,
            is_new=True,
            confidence=0.9,
            mention_text="Alice",
            new_facts=[],
            aliases=[],
        )
        candidate = MemoryUnit(
            uid=Uid("e_old"),
            raw_data={
                "text_content": "Entity Alice(Person): Engineer",
                "entity_name": "Alice",
                "entity_type": "Person",
            },
            metadata={"session_id": "s0"},
            embedding=np.ones(4, dtype=np.float32),
        )
        self.semantic_map._embedder.embed_text.return_value = [
            np.ones(4, dtype=np.float32)
        ]
        matched_idx, confidence, canonical_name, new_aliases, reasoning = (
            self.manager._llm_judge_entity_match(entity, [candidate])
        )
        # Fallback vector judge runs; with identical embeddings it should match
        self.assertIsNotNone(matched_idx)
        self.assertEqual(matched_idx, 0)

    # ── Event matching ────────────────────────────────────────────────

    def test_llm_judge_event_match_no_candidates(self):
        event = ExtractedEvent(
            event_name="Conference",
            linked_id=None,
            is_new=True,
            confidence=0.9,
            description="Tech conference",
            participants=[],
            location=None,
            inferred_time=None,
            new_facts=[],
        )
        matched_idx, confidence, canonical_name, reasoning = (
            self.manager._llm_judge_event_match(event, [])
        )
        self.assertIsNone(matched_idx)

    # ── merge_and_write ───────────────────────────────────────────────

    def test_merge_and_write_basic(self):
        session = MagicMock()
        session.session_id = "s1"
        session_units = [self._make_unit("u1", "Alice went to the store")]

        session_space = SpaceName("root_session_s1")

        # Setup semantic_map to return empty lists for entity/event lookups
        self.semantic_map.get_units_in_spaces.return_value = []
        self.semantic_map.search_by_text.return_value = []
        self.semantic_map.search_by_vector.return_value = []
        self.semantic_map.get_unit.return_value = None

        extracted_entity = ExtractedEntity(
            entity_name="Alice",
            entity_type="Person",
            linked_id=None,
            is_new=True,
            confidence=0.9,
            mention_text="Alice",
            new_facts=["Went to the store"],
            aliases=[],
        )
        extracted_event = ExtractedEvent(
            event_name="went to store",
            linked_id=None,
            is_new=True,
            confidence=0.8,
            description="Alice went to the store",
            participants=[{"mention": "Alice", "role": "participant"}],
            location=None,
            inferred_time=None,
            new_facts=[],
        )

        pipeline_result = PipelineResult(
            entities=[extracted_entity],
            events=[extracted_event],
            entity_relations=[],
            causal_relations=[],
        )

        entity_uid_map, event_uid_map = self.manager.merge_and_write(
            session, session_units, session_space, pipeline_result
        )
        self.assertIn("Alice", entity_uid_map)
        self.assertIn("went to store", event_uid_map)

    # ── Simple concat threshold ───────────────────────────────────────

    def test_simple_concat_below_threshold(self):
        """Verify simple_concat_threshold=2 means <=2 facts use simple concat."""
        existing_desc = "Short desc"
        facts = ["Fact 1", "Fact 2"]
        result = self.manager._update_description_simple(existing_desc, facts)
        self.assertIn("Short desc", result)
        self.assertIn("Fact 1", result)
        self.assertIn("Fact 2", result)

    def test_update_entity_description_via_uid(self):
        """Test _update_entity_description(uid, new_facts) path."""
        existing_unit = MemoryUnit(
            uid=Uid("e_target"),
            raw_data={
                "text_content": "Entity Alice(Person): Old description",
                "entity_name": "Alice",
                "entity_type": "Person",
            },
            metadata={"session_ids": ["s0"]},
        )
        self.semantic_map.get_unit.return_value = existing_unit

        mock_response = MagicMock()
        mock_response.content = (
            '{"merged_description": "Updated with new facts"}'
        )
        self.llm.chat.return_value = mock_response

        self.manager._update_entity_description(Uid("e_target"), ["New fact"])
        self.assertEqual(
            existing_unit.raw_data["text_content"],
            "Entity Alice(Person): Updated with new facts",
        )

    def test_update_event_description_via_uid(self):
        """Test _update_event_description(uid, new_facts, participants, time) path."""
        existing_unit = MemoryUnit(
            uid=Uid("ev_target"),
            raw_data={
                "text_content": "Event Meeting: Old meeting",
                "event_name": "Meeting",
                "description": "Old meeting",
                "participants": [],
                "inferred_time": None,
            },
            metadata={"session_ids": ["s0"]},
        )
        self.semantic_map.get_unit.return_value = existing_unit

        mock_response = MagicMock()
        mock_response.content = (
            '{"merged_description": "Updated meeting", '
            '"merged_participants": [{"mention": "Alice", "role": "participant"}], '
            '"merged_time": "2024-06-01T10:00:00Z"}'
        )
        self.llm.chat.return_value = mock_response

        self.manager._update_event_description(
            Uid("ev_target"),
            ["New detail"],
            [{"mention": "Alice", "role": "participant"}],
            "2024-06-01T10:00:00Z",
        )
        self.assertEqual(existing_unit.raw_data["description"], "Updated meeting")


if __name__ == "__main__":
    unittest.main()
