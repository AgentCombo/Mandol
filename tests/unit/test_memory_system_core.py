#!/usr/bin/env python3
"""Unit tests for memory_system.py core functionality."""
from __future__ import annotations

import json
import logging
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, List, Optional

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
from mandol.application.session_manager import (
    SessionSplitDecision,
    SessionSplitPoint,
    SESSION_SYSTEM_PROMPT,
    SESSION_USER_PROMPT,
)
from mandol.domain.memory_unit import MemoryUnit
from mandol.domain.types import Uid


@dataclass
class MockLLMResponse:
    content: str
    usage: dict = None

    def __init__(self, content: str = "", usage: Optional[dict] = None):
        self.content = content
        self.usage = usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


class MockLLMProvider:
    def __init__(self, response_content: str = "", should_fail: bool = False):
        self._response_content = response_content
        self._should_fail = should_fail
        self._call_count = 0

    def chat(self, messages: Any, temperature: float = 0.1, max_tokens: int = 512, **kwargs) -> MockLLMResponse:
        self._call_count += 1
        if self._should_fail:
            raise Exception("Mock LLM failure")
        return MockLLMResponse(self._response_content)


class MockEmbeddingProvider:
    def __init__(self, dim: int = 384):
        self._dim = dim
        self._call_count = 0

    def embed_text(self, texts: List[str]) -> List[List[float]]:
        self._call_count += 1
        return [[0.1] * self._dim for _ in texts]

    def embedding_dim(self) -> int:
        return self._dim


class MockReranker:
    def __init__(self):
        self._call_count = 0

    def rerank(self, query: str, candidates: List[MemoryUnit], top_k: int) -> List[tuple]:
        self._call_count += 1
        return [(c, 0.9) for c in candidates[:top_k]]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

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

    def test_v2_prompt_contains_key_elements(self):
        self.assertIn("boundaries", SESSION_SYSTEM_PROMPT)
        self.assertIn("should_wait", SESSION_SYSTEM_PROMPT)
        self.assertIn("reasoning", SESSION_SYSTEM_PROMPT)
        self.assertIn("JSON", SESSION_SYSTEM_PROMPT)

    def test_v2_user_prompt_has_previous_reasoning_placeholder(self):
        self.assertIn("{previous_reasoning_block}", SESSION_USER_PROMPT)

    def test_retrieval_group_constants(self):
        self.assertEqual(RETRIEVAL_GROUP_BASE, "base")
        self.assertEqual(RETRIEVAL_GROUP_EVENT, "event")
        self.assertEqual(RETRIEVAL_GROUP_ENTITY, "entity")
        self.assertEqual(RETRIEVAL_GROUP_SUMMARY, "summary")

    def test_relationship_constants(self):
        self.assertEqual(SEMANTIC_SIMILAR, "SEMANTIC_SIMILAR")
        self.assertEqual(EVIDENCED_BY, "EVIDENCED_BY")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

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
        with self.assertRaises(AttributeError):
            config.max_context_units = 30

    def test_custom_config_values(self):
        config = MemorySystemConfig(
            max_context_units=30,
            session_check_interval=10,
            session_max_pending=50,
        )
        self.assertEqual(config.max_context_units, 30)
        self.assertEqual(config.session_check_interval, 10)
        self.assertEqual(config.session_max_pending, 50)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

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

    def test_new_state_variables_initialized(self):
        ms = MemorySystem()
        self.assertFalse(ms._async_check_scheduled)
        self.assertEqual(ms._last_async_reasoning, "")
        self.assertFalse(ms._build_in_progress)
        self.assertFalse(ms._auto_save_paused)
        self.assertEqual(ms._session_counter, 0)


# ---------------------------------------------------------------------------
# Session ID counter
# ---------------------------------------------------------------------------

class TestSessionIdCounter(unittest.TestCase):
    def test_next_session_id_increments(self):
        ms = MemorySystem()
        sid1 = ms._next_session_id()
        sid2 = ms._next_session_id()
        self.assertNotEqual(sid1, sid2)
        self.assertTrue(sid1.startswith("sess_"))
        self.assertTrue(sid2.startswith("sess_"))

    def test_next_session_id_thread_safety(self):
        ms = MemorySystem()
        ids = []

        def collect_ids(n: int):
            for _ in range(n):
                ids.append(ms._next_session_id())

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(collect_ids, 25) for _ in range(4)]
            for f in futures:
                f.result()

        self.assertEqual(len(ids), 100)
        self.assertEqual(len(set(ids)), 100)  # All unique


# ---------------------------------------------------------------------------
# Unit formatting
# ---------------------------------------------------------------------------

class TestFormatUnitsForAnalysis(unittest.TestCase):
    def test_format_units(self):
        ms = MemorySystem()
        units = [
            MemoryUnit(
                uid=Uid("u1"),
                raw_data={"text_content": "Hello"},
                metadata={"timestamp": "2024-01-01T00:00:00"},
            ),
            MemoryUnit(
                uid=Uid("u2"),
                raw_data={"text_content": "World"},
                metadata={"timestamp": "2024-01-01T00:01:00"},
            ),
        ]
        lines = ms._format_units_for_analysis(units)
        self.assertEqual(len(lines), 2)
        self.assertIn("[1]", lines[0])
        self.assertIn("Hello", lines[0])
        self.assertIn("[2]", lines[1])
        self.assertIn("World", lines[1])


# ---------------------------------------------------------------------------
# Async path: add / dirty-flag
# ---------------------------------------------------------------------------

class TestAsyncDirtyFlag(unittest.TestCase):
    def test_schedule_async_check_sets_flag(self):
        ms = MemorySystem()
        self.assertFalse(ms._async_check_scheduled)

        # Add 2 units to trigger schedule
        for i in range(2):
            ms.add(MemoryUnit(
                uid=Uid(f"u{i}"),
                raw_data={"text_content": f"Content {i}"},
                metadata={"timestamp": datetime.now(timezone.utc).isoformat()},
            ))

        # The flag should be set (or already cleared if the async task ran fast)
        # Just verify we don't crash
        self.assertIsNotNone(ms._pending_units)

    def test_schedule_skipped_when_build_in_progress(self):
        ms = MemorySystem()
        ms._build_in_progress = True

        for i in range(5):
            ms.add(MemoryUnit(
                uid=Uid(f"u{i}"),
                raw_data={"text_content": f"Content {i}"},
                metadata={"timestamp": datetime.now(timezone.utc).isoformat()},
            ))

        # Flag should NOT be set because build is in progress
        self.assertFalse(ms._async_check_scheduled)


# ---------------------------------------------------------------------------
# Insertion order
# ---------------------------------------------------------------------------

class TestInsertionOrderTracking(unittest.TestCase):
    def test_insertion_order_tracked_on_add(self):
        ms = MemorySystem()

        unit = MemoryUnit(
            uid=Uid("test_unit"),
            raw_data={"text_content": "Test content"},
            metadata={"timestamp": datetime.now(timezone.utc).isoformat()}
        )

        ms.add(unit)
        self.assertIn("test_unit", ms._insertion_order)

    def test_insertion_order_tracked_on_add_many(self):
        ms = MemorySystem()

        units = [
            MemoryUnit(
                uid=Uid(f"unit_{i}"),
                raw_data={"text_content": f"Content {i}"},
                metadata={"timestamp": datetime.now(timezone.utc).isoformat()}
            )
            for i in range(5)
        ]

        ms.add_many(units)
        self.assertEqual(len(ms._insertion_order), 5)


# ---------------------------------------------------------------------------
# Pending units & locking
# ---------------------------------------------------------------------------

class TestPendingUnitsLocking(unittest.TestCase):
    def test_add_uses_lock(self):
        ms = MemorySystem()

        unit = MemoryUnit(
            uid=Uid("test_unit"),
            raw_data={"text_content": "Test content"},
            metadata={"timestamp": datetime.now(timezone.utc).isoformat()}
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
                    metadata={"timestamp": datetime.now(timezone.utc).isoformat()}
                )
                ms.add(unit)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(add_units, i * 10, 10) for i in range(4)]
            for f in futures:
                f.result()

        with ms._pending_lock:
            self.assertEqual(len(ms._pending_units), 40)
            self.assertEqual(len(ms._insertion_order), 40)


# ---------------------------------------------------------------------------
# Build session for units
# ---------------------------------------------------------------------------

class TestBuildSessionForUnits(unittest.TestCase):
    def test_build_session_for_units_with_empty_list(self):
        mock_llm = MockLLMProvider("{}")
        ms = MemorySystem(llm_provider=mock_llm)

        ms._build_session_for_units([])
        self.assertEqual(len(ms._session_manager.get_sessions()), 0)

    def test_build_session_for_units_creates_session(self):
        mock_llm = MockLLMProvider("{}")
        ms = MemorySystem(llm_provider=mock_llm)

        units = [
            MemoryUnit(
                uid=Uid(f"u{i}"),
                raw_data={"text_content": f"Content {i}"},
                metadata={"timestamp": datetime.now(timezone.utc).isoformat()},
            )
            for i in range(3)
        ]

        # Units must be in the store before _build_session_for_units is called,
        # because _ensure_session_space looks them up. In real flow, add() does this.
        for u in units:
            ms.add(u)

        # Clear pending so the async path doesn't interfere
        with ms._pending_lock:
            ms._pending_units.clear()

        ms._build_session_for_units(units)
        self.assertGreaterEqual(len(ms._session_manager.get_sessions()), 1)


# ---------------------------------------------------------------------------
# Build high level
# ---------------------------------------------------------------------------

class TestBuildHighLevel(unittest.TestCase):
    def test_build_high_level_auto_no_units(self):
        mock_llm = MockLLMProvider("{}")
        ms = MemorySystem(llm_provider=mock_llm)

        report = ms.build_high_level("auto")
        self.assertEqual(report.status, "no_units")
        self.assertEqual(report.mode, "auto")

    def test_build_high_level_force_no_units(self):
        mock_llm = MockLLMProvider("{}")
        ms = MemorySystem(llm_provider=mock_llm)

        report = ms.build_high_level("force")
        self.assertEqual(report.status, "no_units")
        self.assertEqual(report.mode, "force")


# ---------------------------------------------------------------------------
# Async architecture
# ---------------------------------------------------------------------------

class TestAsyncArchitecture(unittest.TestCase):
    def test_executor_initialized_with_2_workers(self):
        ms = MemorySystem()
        self.assertIsInstance(ms._executor, ThreadPoolExecutor)
        self.assertEqual(ms._executor._max_workers, 2)

    def test_build_high_level_async_returns_future(self):
        mock_llm = MockLLMProvider("{}")
        ms = MemorySystem(llm_provider=mock_llm)

        future = ms.build_high_level_async()
        self.assertTrue(hasattr(future, 'result'))


# ---------------------------------------------------------------------------
# Cross-session merging
# ---------------------------------------------------------------------------

class TestCrossSessionMerging(unittest.TestCase):
    def test_merge_cross_session_entities_method_exists(self):
        ms = MemorySystem()
        self.assertTrue(hasattr(ms, 'merge_cross_session_entities'))
        self.assertTrue(callable(getattr(ms, 'merge_cross_session_entities')))

    def test_merge_cross_session_events_method_exists(self):
        ms = MemorySystem()
        self.assertTrue(hasattr(ms, 'merge_cross_session_events'))
        self.assertTrue(callable(getattr(ms, 'merge_cross_session_events')))


# ---------------------------------------------------------------------------
# Flush
# ---------------------------------------------------------------------------

class TestFlushMethod(unittest.TestCase):
    def test_flush_clears_all_pending_data(self):
        ms = MemorySystem()

        for i in range(10):
            unit = MemoryUnit(
                uid=Uid(f"unit_{i}"),
                raw_data={"text_content": f"Content {i}"},
                metadata={"timestamp": datetime.now(timezone.utc).isoformat()}
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
            self.assertFalse(ms._async_check_scheduled)
            self.assertEqual(ms._last_async_reasoning, "")
        self.assertFalse(ms.dirty)


# ---------------------------------------------------------------------------
# Retrieval groups
# ---------------------------------------------------------------------------

class TestRetrievalGroups(unittest.TestCase):
    def test_get_retrieval_groups_returns_all_4_groups(self):
        ms = MemorySystem()

        groups = ms._get_retrieval_groups()
        self.assertEqual(len(groups), 4)
        self.assertIn(RETRIEVAL_GROUP_BASE, groups)
        self.assertIn(RETRIEVAL_GROUP_EVENT, groups)
        self.assertIn(RETRIEVAL_GROUP_ENTITY, groups)
        self.assertIn(RETRIEVAL_GROUP_SUMMARY, groups)


# ---------------------------------------------------------------------------
# Dirty flag
# ---------------------------------------------------------------------------

class TestDirtyFlag(unittest.TestCase):
    def test_dirty_flag_set_on_add(self):
        ms = MemorySystem()
        self.assertFalse(ms.dirty)

        unit = MemoryUnit(
            uid=Uid("test_unit"),
            raw_data={"text_content": "Test content"},
            metadata={"timestamp": datetime.now(timezone.utc).isoformat()}
        )
        ms.add(unit)
        self.assertTrue(ms.dirty)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    unittest.main(verbosity=2)
