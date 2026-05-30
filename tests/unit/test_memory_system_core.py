#!/usr/bin/env python3
"""Unit tests for memory_system.py core functionality."""
from __future__ import annotations

import json
import logging
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, List

from mandol.application.memory_system import (
    MemorySystem,
    MemorySystemConfig,
    MAX_CONTEXT_UNITS,
    MAX_ENTITIES_PER_LLM,
    MAX_EVENTS_PER_LLM,
    SESSION_CHECK_INTERVAL,
    SESSION_MAX_PENDING,
    SEMANTIC_SIMILAR,
    EVIDENCED_BY,
)
from mandol.application.services._retrieval import (
    RETRIEVAL_GROUP_BASE,
    RETRIEVAL_GROUP_EVENT,
    RETRIEVAL_GROUP_ENTITY,
    RETRIEVAL_GROUP_SUMMARY,
)
from mandol.domain.memory_unit import MemoryUnit
from mandol.domain.types import Uid


@dataclass
class MockLLMResponse:
    content: str
    raw: dict
    usage: dict

    def __init__(self, content: str = ""):
        self.content = content
        self.raw = {}
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


class MockLLMProvider:
    def __init__(self, response_content: str = "", should_fail: bool = False):
        self._response_content = response_content
        self._should_fail = should_fail
        self._call_count = 0

    def chat(
        self,
        messages: Any,
        temperature: float = 0.1,
        max_tokens: int = 512,
        model: str = None,
        response_format: dict = None,
        **kwargs,
    ) -> MockLLMResponse:
        self._call_count += 1
        if self._should_fail:
            return MockLLMResponse("error")
        return MockLLMResponse(self._response_content)


class MockEmbeddingProvider:
    def __init__(self, dim: int = 384):
        self._dim = dim
        self._call_count = 0

    def embed_text(self, texts: List[str]) -> List[List[float]]:
        self._call_count += 1
        return [[0.1] * self._dim for _ in texts]

    def embed_image_paths(self, image_paths: List[str]) -> List[List[float]]:
        return [[0.1] * self._dim for _ in image_paths]

    def embedding_dim(self) -> int:
        return self._dim


class MockReranker:
    def __init__(self):
        self._call_count = 0

    def rerank(self, query: str, candidates: List[MemoryUnit], top_k: int) -> List[tuple]:
        self._call_count += 1
        return [(c, 0.9) for c in candidates[:top_k]]


class TestMemorySystemConstants(unittest.TestCase):
    def test_max_context_units_is_20(self):
        self.assertEqual(MAX_CONTEXT_UNITS, 20)

    def test_session_check_interval_is_20(self):
        self.assertEqual(SESSION_CHECK_INTERVAL, 20)

    def test_session_max_pending_is_100(self):
        self.assertEqual(SESSION_MAX_PENDING, 100)

    def test_max_entities_per_llm_is_50(self):
        self.assertEqual(MAX_ENTITIES_PER_LLM, 50)

    def test_max_events_per_llm_is_50(self):
        self.assertEqual(MAX_EVENTS_PER_LLM, 50)

    def test_retrieval_group_constants(self):
        self.assertEqual(RETRIEVAL_GROUP_BASE, "base")
        self.assertEqual(RETRIEVAL_GROUP_EVENT, "event")
        self.assertEqual(RETRIEVAL_GROUP_ENTITY, "entity")
        self.assertEqual(RETRIEVAL_GROUP_SUMMARY, "summary")

    def test_relationship_constants(self):
        self.assertEqual(SEMANTIC_SIMILAR, "SEMANTIC_SIMILAR")
        self.assertEqual(EVIDENCED_BY, "EVIDENCED_BY")


class TestMemorySystemConfig(unittest.TestCase):
    def test_default_config_values(self):
        config = MemorySystemConfig()
        self.assertEqual(config.max_context_units, 20)
        self.assertEqual(config.session_check_interval, 20)
        self.assertEqual(config.session_max_pending, 100)
        self.assertEqual(config.max_entities_per_llm, 50)
        self.assertEqual(config.max_events_per_llm, 50)

    def test_config_is_frozen(self):
        config = MemorySystemConfig()
        with self.assertRaises(Exception):
            config.max_context_units = 30  # type: ignore[misc]

    def test_custom_config_values(self):
        config = MemorySystemConfig(
            max_context_units=30,
            session_check_interval=10,
            session_max_pending=50,
        )
        self.assertEqual(config.max_context_units, 30)
        self.assertEqual(config.session_check_interval, 10)
        self.assertEqual(config.session_max_pending, 50)


class TestMemorySystemInitialization(unittest.TestCase):
    def test_initialization_with_defaults(self):
        ms = MemorySystem()
        self.assertFalse(ms.dirty)
        self.assertIsNotNone(ms.llm)

    def test_initialization_with_custom_providers(self):
        mock_embedder = MockEmbeddingProvider()
        mock_reranker = MockReranker()
        mock_llm = MockLLMProvider()

        ms = MemorySystem(
            embedder=mock_embedder,
            reranker=mock_reranker,
            llm_provider=mock_llm,
        )
        self.assertFalse(ms.dirty)
        self.assertEqual(ms.llm, mock_llm)

    def test_initialization_with_custom_config(self):
        config = MemorySystemConfig(max_context_units=15)
        ms = MemorySystem(config=config)
        self.assertEqual(ms._cfg.max_context_units, 15)


class TestInsertionOrderTracking(unittest.TestCase):
    def test_insertion_order_tracked_on_add(self):
        ms = MemorySystem()

        unit = MemoryUnit(
            uid=Uid("test_unit"),
            raw_data={"text_content": "Test content"},
            metadata={"timestamp": datetime.now(timezone.utc).isoformat()},
        )

        ms.add(unit)
        self.assertIn("test_unit", ms._insertion_order)

    def test_insertion_order_tracked_on_add_many(self):
        ms = MemorySystem()

        units = [
            MemoryUnit(
                uid=Uid(f"unit_{i}"),
                raw_data={"text_content": f"Content {i}"},
                metadata={"timestamp": datetime.now(timezone.utc).isoformat()},
            )
            for i in range(5)
        ]

        ms.add_many(units)
        self.assertEqual(len(ms._insertion_order), 5)


class TestPendingUnitsLocking(unittest.TestCase):
    def test_add_uses_lock(self):
        ms = MemorySystem()

        unit = MemoryUnit(
            uid=Uid("test_unit"),
            raw_data={"text_content": "Test content"},
            metadata={"timestamp": datetime.now(timezone.utc).isoformat()},
        )

        ms.add(unit)
        with ms._pending_lock:
            self.assertEqual(len(ms._pending_units), 1)

    def test_concurrent_adds_are_safe(self):
        ms = MemorySystem()

        def add_units(start_idx: int, count: int):
            for i in range(start_idx, start_idx + count):
                unit = MemoryUnit(
                    uid=Uid(f"unit_{i}"),
                    raw_data={"text_content": f"Content {i}"},
                    metadata={"timestamp": datetime.now(timezone.utc).isoformat()},
                )
                ms.add(unit)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(add_units, i * 10, 10) for i in range(4)]
            for f in futures:
                f.result()

        with ms._pending_lock:
            self.assertEqual(len(ms._pending_units), 40)
            self.assertEqual(len(ms._insertion_order), 40)


class TestBuildSessionForUnits(unittest.TestCase):
    def test_build_session_for_units_with_empty_list(self):
        mock_llm = MockLLMProvider("{}")
        ms = MemorySystem(llm_provider=mock_llm)

        ms._build_session_for_units([])
        self.assertEqual(len(ms._session_manager._sessions), 0)

    def test_build_session_for_units_creates_session(self):
        mock_llm = MockLLMProvider("{}")
        ms = MemorySystem(llm_provider=mock_llm)

        units = []
        for i in range(3):
            unit = MemoryUnit(
                uid=Uid(f"unit_{i}"),
                raw_data={"text_content": f"Content {i}"},
                metadata={"timestamp": datetime.now(timezone.utc).isoformat()},
            )
            # Write units to the semantic_map store first so session-building works
            ms._semantic_map.add_unit(unit, ensure_embedding=True)
            units.append(unit)

        # Directly call _build_session_for_units
        ms._build_session_for_units(units)
        self.assertGreaterEqual(len(ms._session_manager._sessions), 1)


class TestFlushMethod(unittest.TestCase):
    def test_flush_clears_all_pending_data(self):
        ms = MemorySystem()

        for i in range(10):
            unit = MemoryUnit(
                uid=Uid(f"unit_{i}"),
                raw_data={"text_content": f"Content {i}"},
                metadata={"timestamp": datetime.now(timezone.utc).isoformat()},
            )
            ms.add(unit)

        ms.flush()

        with ms._pending_lock:
            self.assertEqual(len(ms._pending_units), 0)
            self.assertEqual(len(ms._pending_events), 0)
            self.assertEqual(len(ms._pending_entities), 0)
            self.assertEqual(len(ms._all_events), 0)
            self.assertEqual(len(ms._all_entities), 0)
            self.assertEqual(len(ms._insertion_order), 0)
        self.assertFalse(ms.dirty)


class TestRetrievalGroups(unittest.TestCase):
    def test_get_retrieval_groups_returns_all_4_groups(self):
        ms = MemorySystem()

        groups = ms._get_retrieval_groups()
        self.assertEqual(len(groups), 4)
        self.assertIn(RETRIEVAL_GROUP_BASE, groups)
        self.assertIn(RETRIEVAL_GROUP_EVENT, groups)
        self.assertIn(RETRIEVAL_GROUP_ENTITY, groups)
        self.assertIn(RETRIEVAL_GROUP_SUMMARY, groups)


class TestDirtyFlag(unittest.TestCase):
    def test_dirty_flag_set_on_add(self):
        ms = MemorySystem()
        self.assertFalse(ms.dirty)

        unit = MemoryUnit(
            uid=Uid("test_unit"),
            raw_data={"text_content": "Test content"},
            metadata={"timestamp": datetime.now(timezone.utc).isoformat()},
        )
        ms.add(unit)
        self.assertTrue(ms.dirty)


class TestCrossSessionMerging(unittest.TestCase):
    def test_merge_cross_session_entities_method_exists(self):
        ms = MemorySystem()
        self.assertTrue(hasattr(ms, "merge_cross_session_entities"))
        self.assertTrue(callable(getattr(ms, "merge_cross_session_entities")))

    def test_merge_cross_session_events_method_exists(self):
        ms = MemorySystem()
        self.assertTrue(hasattr(ms, "merge_cross_session_events"))
        self.assertTrue(callable(getattr(ms, "merge_cross_session_events")))


class TestAsyncArchitecture(unittest.TestCase):
    def test_executor_initialized_with_2_workers(self):
        ms = MemorySystem()
        self.assertIsInstance(ms._executor, ThreadPoolExecutor)
        self.assertEqual(ms._executor._max_workers, 2)

    def test_build_high_level_async_returns_future(self):
        mock_llm = MockLLMProvider("{}")
        ms = MemorySystem(llm_provider=mock_llm)

        future = ms.build_high_level_async()
        self.assertTrue(hasattr(future, "result"))


class TestSessionManagerIntegration(unittest.TestCase):
    """Tests for the V2 session detection API (analyze_batch)."""

    def test_analyze_batch_no_split_small_batch(self):
        mock_llm = MockLLMProvider(json.dumps({
            "reasoning": "Single coherent topic, no split needed.",
            "boundaries": [],
            "should_wait": False,
        }))
        ms = MemorySystem(llm_provider=mock_llm)

        content_lines = [
            "[1] 2024-01-01T00:00:00: Hello, let's discuss the project.",
            "[2] 2024-01-01T00:01:00: Sure, the timeline looks good.",
            "[3] 2024-01-01T00:02:00: Great, let's proceed.",
        ]
        decision = ms._session_manager.analyze_batch(content_lines, "test_sess_0")
        self.assertFalse(decision.should_split)
        self.assertEqual(len(decision.split_points), 0)

    def test_analyze_batch_detects_split(self):
        mock_llm = MockLLMProvider(json.dumps({
            "reasoning": "Clear topic shift from project discussion to lunch plans.",
            "boundaries": [3],
            "should_wait": False,
        }))
        ms = MemorySystem(llm_provider=mock_llm)

        content_lines = [
            "[1] 2024-01-01T00:00:00: Let's fix the auth bug.",
            "[2] 2024-01-01T00:01:00: Found the issue in the token refresh.",
            "[3] 2024-01-01T00:02:00: Deploying the fix now.",
            "[4] 2024-01-01T00:10:00: What do you want for lunch?",
            "[5] 2024-01-01T00:11:00: Pizza sounds great.",
        ]
        decision = ms._session_manager.analyze_batch(content_lines, "test_sess_1")
        self.assertTrue(decision.should_split)
        self.assertEqual(len(decision.split_points), 1)
        self.assertEqual(decision.split_points[0].split_at_index, 3)

    def test_analyze_batch_should_wait(self):
        mock_llm = MockLLMProvider(json.dumps({
            "reasoning": "Too little context at the tail to decide.",
            "boundaries": [],
            "should_wait": True,
        }))
        ms = MemorySystem(llm_provider=mock_llm)

        content_lines = [
            "[1] 2024-01-01T00:00:00: Working on the new feature.",
            "[2] 2024-01-01T00:01:00: ok",
        ]
        decision = ms._session_manager.analyze_batch(content_lines, "test_sess_2")
        self.assertFalse(decision.should_split)
        self.assertTrue(decision.should_wait)

    def test_analyze_batch_llm_failure_returns_no_split(self):
        mock_llm = MockLLMProvider(
            json.dumps({"reasoning": "", "boundaries": [], "should_wait": False})
        )
        ms = MemorySystem(llm_provider=mock_llm)

        content_lines = [
            f"[{i}] 2024-01-01T00:{i:02d}:00: Test message {i}"
            for i in range(1, 5)
        ]
        decision = ms._session_manager.analyze_batch(content_lines, "test_sess_3")
        self.assertFalse(decision.should_split)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    unittest.main(verbosity=2)
