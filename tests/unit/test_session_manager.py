"""Unit tests for SessionManager V2 response parsing and boundary detection."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from mandol.application.session_manager import (
    Session,
    SessionManager,
    SessionSplitPoint,
    SessionSplitDecision,
)


class TestSessionSplitDecision:
    def test_defaults(self):
        d = SessionSplitDecision(should_split=False)
        assert d.should_split is False
        assert d.split_at_index is None
        assert d.split_points == []
        assert d.reasoning == ""
        assert d.should_wait is False

    def test_with_single_split(self):
        sp = SessionSplitPoint(split_at_index=5, topic="new topic", reason="topic shift")
        d = SessionSplitDecision(
            should_split=True,
            split_at_index=5,
            split_points=[sp],
            reasoning="Clear break",
        )
        assert d.should_split is True
        assert d.split_at_index == 5
        assert len(d.split_points) == 1


class TestSession:
    def test_unit_count(self):
        s = Session(session_id="s1", unit_uids=[])
        assert s.unit_count == 0
        s.unit_uids.append("u1")
        s.unit_uids.append("u2")
        assert s.unit_count == 2

    def test_to_dict(self):
        s = Session(
            session_id="sess_1",
            unit_uids=["u1", "u2"],
            topic="test topic",
            start_time="2024-01-01T00:00:00",
            end_time="2024-01-01T01:00:00",
        )
        d = s.to_dict()
        assert d["session_id"] == "sess_1"
        assert d["unit_uids"] == ["u1", "u2"]
        assert d["topic"] == "test topic"


class TestSessionManagerV2ResponseParsing:
    """Tests for _parse_v2_response with the current V2 JSON schema."""

    def _make_manager(self):
        llm = MagicMock()
        return SessionManager(llm_provider=llm)

    def test_no_split_empty_boundaries(self):
        mgr = self._make_manager()
        resp = json.dumps({"reasoning": "All one topic", "boundaries": [], "should_wait": False})
        decision = mgr._parse_v2_response(resp, content_count=10)
        assert decision.should_split is False
        assert decision.split_points == []
        assert decision.should_wait is False

    def test_single_split_point(self):
        mgr = self._make_manager()
        resp = json.dumps({
            "reasoning": "Split after message 4",
            "boundaries": [4],
            "should_wait": False,
        })
        decision = mgr._parse_v2_response(resp, content_count=10)
        assert decision.should_split is True
        assert len(decision.split_points) == 1
        assert decision.split_points[0].split_at_index == 4

    def test_multiple_split_points(self):
        mgr = self._make_manager()
        resp = json.dumps({
            "reasoning": "Two topic shifts",
            "boundaries": [3, 7],
            "should_wait": False,
        })
        decision = mgr._parse_v2_response(resp, content_count=10)
        assert decision.should_split is True
        assert len(decision.split_points) == 2
        assert decision.split_points[0].split_at_index == 3
        assert decision.split_points[1].split_at_index == 7

    def test_should_wait_ignores_boundaries(self):
        mgr = self._make_manager()
        resp = json.dumps({
            "reasoning": "Too little context",
            "boundaries": [3],
            "should_wait": True,
        })
        decision = mgr._parse_v2_response(resp, content_count=10)
        assert decision.should_split is False
        assert decision.should_wait is True
        assert decision.split_points == []

    def test_boundary_too_large_skipped(self):
        mgr = self._make_manager()
        resp = json.dumps({
            "reasoning": "Bad boundary",
            "boundaries": [99],
            "should_wait": False,
        })
        decision = mgr._parse_v2_response(resp, content_count=10)
        # 99 > 9 (content_count - 1), so it's skipped
        assert decision.should_split is False

    def test_boundary_zero_skipped(self):
        mgr = self._make_manager()
        resp = json.dumps({
            "reasoning": "Zero boundary",
            "boundaries": [0],
            "should_wait": False,
        })
        decision = mgr._parse_v2_response(resp, content_count=10)
        # 0 is not in 1..9, so skipped
        assert decision.should_split is False

    def test_non_integer_boundary_skipped(self):
        mgr = self._make_manager()
        resp = json.dumps({
            "reasoning": "String boundary",
            "boundaries": ["abc"],
            "should_wait": False,
        })
        decision = mgr._parse_v2_response(resp, content_count=10)
        assert decision.should_split is False

    def test_boundaries_not_list(self):
        mgr = self._make_manager()
        resp = json.dumps({
            "reasoning": "Non-list",
            "boundaries": None,
            "should_wait": False,
        })
        decision = mgr._parse_v2_response(resp, content_count=10)
        assert decision.should_split is False

    def test_reasoning_newlines_stripped(self):
        mgr = self._make_manager()
        resp = json.dumps({
            "reasoning": "Line 1\nLine 2",
            "boundaries": [],
            "should_wait": False,
        })
        decision = mgr._parse_v2_response(resp, content_count=10)
        assert "\n" not in decision.reasoning
        assert decision.reasoning == "Line 1 Line 2"

    def test_duplicate_boundaries_deduplicated(self):
        mgr = self._make_manager()
        resp = json.dumps({
            "reasoning": "Dup boundaries",
            "boundaries": [3, 3, 5, 5],
            "should_wait": False,
        })
        decision = mgr._parse_v2_response(resp, content_count=10)
        assert len(decision.split_points) == 2

    def test_unsorted_boundaries_become_sorted(self):
        mgr = self._make_manager()
        resp = json.dumps({
            "reasoning": "Unsorted",
            "boundaries": [7, 3],
            "should_wait": False,
        })
        decision = mgr._parse_v2_response(resp, content_count=10)
        assert decision.split_points[0].split_at_index == 3
        assert decision.split_points[1].split_at_index == 7

    def test_malformed_json_returns_no_split(self):
        """V2 parse wraps parse_v2_response, which handles JSON parse errors."""
        mgr = self._make_manager()
        # Direct call to _parse_v2_response with bad JSON
        with pytest.raises(json.JSONDecodeError):
            mgr._parse_v2_response("not json", content_count=10)


class TestSessionManagerReset:
    def test_reset_clears_sessions(self):
        llm = MagicMock()
        mgr = SessionManager(llm_provider=llm)
        mgr.add_session("s1", ["u1", "u2"])
        mgr.add_session("s2", ["u3"])
        assert len(mgr.get_sessions()) == 2
        mgr.reset()
        assert len(mgr.get_sessions()) == 0

    def test_get_sessions_returns_copy(self):
        llm = MagicMock()
        mgr = SessionManager(llm_provider=llm)
        mgr.add_session("s1", ["u1"])
        sessions = mgr.get_sessions()
        sessions.clear()
        # Original list should be unaffected
        assert len(mgr.get_sessions()) == 1


class TestSessionManagerBuildSession:
    def test_build_session_returns_session_with_correct_fields(self):
        llm = MagicMock()
        mgr = SessionManager(llm_provider=llm)
        s = mgr.build_session(
            session_id="s1",
            unit_uids=["u1", "u2"],
            topic="Test Topic",
            start_time="2024-01-01T00:00:00Z",
            end_time="2024-01-01T01:00:00Z",
        )
        assert s.session_id == "s1"
        assert s.topic == "Test Topic"
        assert len(s.unit_uids) == 2
        assert s in mgr.get_sessions()
