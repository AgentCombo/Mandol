"""Unit tests for LLM retry, JSON fence stripping, and error fallback paths."""

from __future__ import annotations

import json
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from mandol.application._llm_retry import (
    retry_llm_json_call,
    strip_json_fences,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_DELAY,
)


# ── JSON fence stripping ────────────────────────────────────────────────

class TestStripJsonFences:
    def test_passthrough_clean_json(self):
        text = '{"key": "value"}'
        assert strip_json_fences(text) == text

    def test_strips_markdown_fence(self):
        text = '```json\n{"key": "value"}\n```'
        result = strip_json_fences(text)
        assert "key" in result
        assert "```" not in result

    def test_strips_preamble_text(self):
        text = 'Here is the response:\n{"key": "value"}'
        result = strip_json_fences(text)
        assert result == '{"key": "value"}'

    def test_strips_trailing_text(self):
        text = '{"key": "value"}\nSome extra text.'
        result = strip_json_fences(text)
        assert result == '{"key": "value"}'

    def test_strips_trailing_commas(self):
        text = '{"a": 1, "b": 2,}'
        result = strip_json_fences(text)
        assert "2," not in result

    def test_strips_single_line_comments(self):
        text = '{"a": 1 // comment\n, "b": 2}'
        result = strip_json_fences(text)
        parsed = json.loads(result)
        assert parsed["a"] == 1

    def test_handles_empty_string(self):
        assert strip_json_fences("") == ""

    def test_handles_whitespace_only(self):
        assert strip_json_fences("   ") == "   "

    def test_strips_markdown_without_lang(self):
        text = '```\n{"key": "value"}\n```'
        result = strip_json_fences(text)
        assert "key" in result

    def test_extracts_array(self):
        text = 'Prefix\n[1, 2, 3]\nSuffix'
        result = strip_json_fences(text)
        parsed = json.loads(result)
        assert parsed == [1, 2, 3]


# ── LLM retry with JSON parsing ─────────────────────────────────────────

@dataclass
class _MockResponse:
    content: str
    raw: dict
    usage: dict

    def __init__(self, content: str):
        self.content = content
        self.raw = {}
        self.usage = {}


class TestRetryLlmJsonCall:
    def _mock_llm(self, responses):
        """Create a mock LLM that returns the given responses in sequence."""
        llm = MagicMock()
        llm.chat.side_effect = [_MockResponse(r) for r in responses]
        return llm

    def test_success_first_attempt(self):
        llm = self._mock_llm(['{"result": "ok"}'])
        result = retry_llm_json_call(
            llm, [{"role": "user", "content": "test"}],
            lambda s: json.loads(s),
            context_label="test",
        )
        assert result == {"result": "ok"}
        assert llm.chat.call_count == 1

    def test_success_after_retry(self):
        llm = self._mock_llm([
            "not valid json at all",
            '{"result": "retry_ok"}',
        ])
        result = retry_llm_json_call(
            llm, [{"role": "user", "content": "test"}],
            lambda s: json.loads(s),
            context_label="test_retry",
        )
        assert result == {"result": "retry_ok"}
        assert llm.chat.call_count == 2

    def test_failure_all_retries_exhausted(self):
        llm = self._mock_llm([
            "garbage 1",
            "garbage 2",
            "garbage 3",
        ])
        with pytest.raises(json.JSONDecodeError):
            retry_llm_json_call(
                llm, [{"role": "user", "content": "test"}],
                lambda s: json.loads(s),
                context_label="test_fail",
            )
        # max_retries=2 → 1 initial + 2 retries = 3 calls
        assert llm.chat.call_count == 3

    def test_success_with_fenced_json(self):
        """LLM returns JSON wrapped in markdown fences — should parse after cleaning."""
        llm = self._mock_llm(['```json\n{"key": "value"}\n```'])
        result = retry_llm_json_call(
            llm, [{"role": "user", "content": "test"}],
            lambda s: json.loads(s),
            context_label="test_fence",
        )
        assert result == {"key": "value"}
        assert llm.chat.call_count == 1

    def test_success_with_trailing_comma(self):
        llm = self._mock_llm(['{"a": 1, "b": 2,}'])
        result = retry_llm_json_call(
            llm, [{"role": "user", "content": "test"}],
            lambda s: json.loads(s),
            context_label="test_comma",
        )
        assert result == {"a": 1, "b": 2}
        assert llm.chat.call_count == 1

    def test_custom_parse_fn(self):
        """Verify that parse_fn is called on the cleaned content."""
        llm = self._mock_llm(['42'])
        result = retry_llm_json_call(
            llm, [{"role": "user", "content": "test"}],
            lambda s: int(s.strip()),
            context_label="test_int",
        )
        assert result == 42


# ── Session split fallback path ──────────────────────────────────────────
# (When the LLM call fails entirely, analyze_batch returns a no-split decision)

class TestSessionSplitFallback:
    def test_analyze_batch_llm_failure_returns_no_split(self):
        from mandol.application.session_manager import SessionManager

        llm = MagicMock()
        llm.chat.side_effect = ConnectionError("Connection timeout")

        mgr = SessionManager(llm_provider=llm)
        lines = [f"[{i}] 2024-01-01T00:{i:02d}:00: Message {i}" for i in range(1, 6)]

        decision = mgr.analyze_batch(lines, "sess_fallback")
        assert decision.should_split is False
        assert decision.should_wait is False
        assert decision.split_points == []

    def test_analyze_batch_warns_on_failure(self):
        from mandol.application.session_manager import SessionManager

        llm = MagicMock()
        llm.chat.side_effect = ConnectionError("Connection timeout")

        warnings = []
        mgr = SessionManager(llm_provider=llm)
        lines = [f"[{i}] 2024-01-01T00:{i:02d}:00: Message {i}" for i in range(1, 4)]

        decision = mgr.analyze_batch(lines, "sess_warn", on_warning=lambda msg: warnings.append(msg))
        assert len(warnings) > 0
        assert "[FALLBACK]" in warnings[0]
        assert decision.should_split is False
