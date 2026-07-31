"""Deterministic auto session assignment for the default streaming path."""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Deque, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .l0_views import extract_original_text
from ..core.memory_unit import MemoryUnit
from ..utils.logging_config import create_module_logger


logger = create_module_logger("auto_builder.auto_session_assigner")


@dataclass
class AutoSessionConfig:
    enabled: bool = True
    enabled_styles: Tuple[str, ...] = ("default",)

    # Time boundaries
    hard_gap_minutes: int = 180
    soft_gap_minutes: int = 45
    day_boundary_split: bool = True

    # Topic shift
    topic_shift_threshold: float = 0.42
    boundary_score_threshold: float = 0.65

    # Session length guards
    max_units_per_session: int = 80
    max_chars_per_session: int = 12000
    min_units_per_session: int = 2
    min_chars_per_session: int = 200

    # Reserved for a future async/refine path. Never used synchronously here.
    enable_llm_refine: bool = False


@dataclass
class ActiveSessionState:
    session_id: str
    sample_id: str
    created_at: datetime
    start_time: Optional[datetime] = None
    last_time: Optional[datetime] = None
    unit_uids: List[str] = field(default_factory=list)
    unit_count: int = 0
    char_count: int = 0
    centroid_embedding: Optional[np.ndarray] = None
    recent_embeddings: Deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=8))
    participants: Set[str] = field(default_factory=set)
    is_finalized: bool = False
    finalized_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionAssignment:
    session_id: str
    is_new_session: bool
    confidence: float
    reason: str
    boundary_score: float = 0.0
    finalized_previous_session_id: Optional[str] = None


class AutoSessionAssigner:
    """Assign session metadata to units in default real-time/batch paths.

    The assigner is deliberately deterministic and lightweight: it uses explicit
    session fields, timestamps, length guards, optional existing embeddings, and
    a small set of topic-opening phrases. It never calls an LLM or loads a model.
    """

    # Only fields that can represent an event instant participate in gap logic.
    # ``date`` and ``session_date`` are calendar metadata, never event times.
    _EVENT_TIMESTAMP_FIELDS = ("timestamp", "datetime", "created_at")
    _SESSION_FIELDS = ("session_id",)
    _PARTICIPANT_FIELDS = ("participants", "speakers")
    _SPEAKER_FIELDS = ("speaker", "role", "user", "author")
    _NEW_TOPIC_PATTERNS = (
        "另外",
        "换个话题",
        "还有一件事",
        "接下来",
        "帮我",
        "请你",
        "new topic",
        "another question",
        "by the way",
    )

    def __init__(self, config: Optional[AutoSessionConfig] = None):
        self.config = config or AutoSessionConfig()
        self._active_sessions: Dict[str, ActiveSessionState] = {}
        self._counters: Dict[str, int] = {}

    def assign_batch(
        self,
        units: Sequence[MemoryUnit],
        sample_id: str,
        session_date: Optional[str] = None,
        participants: Optional[List[str]] = None,
        explicit_session_id: Optional[str] = None,
    ) -> List[MemoryUnit]:
        """Assign session_id to a batch, preserving explicit unit/session IDs."""
        assigned_units = list(units)
        for unit in assigned_units:
            self.assign_one(
                unit=unit,
                sample_id=sample_id,
                session_date=session_date,
                participants=participants,
                explicit_session_id=explicit_session_id,
            )
        return assigned_units

    def assign_one(
        self,
        unit: MemoryUnit,
        sample_id: str,
        session_date: Optional[str] = None,
        participants: Optional[List[str]] = None,
        explicit_session_id: Optional[str] = None,
    ) -> SessionAssignment:
        """Assign one unit in streaming order."""
        existing_session_id = self._extract_existing_session_id(unit)
        if existing_session_id:
            assignment = SessionAssignment(
                session_id=existing_session_id,
                is_new_session=existing_session_id != self._active_sessions.get(sample_id, ActiveSessionState("", sample_id, datetime.now())).session_id,
                confidence=1.0,
                reason="existing_session_id",
            )
            self._register_or_update_state(sample_id, existing_session_id, unit, participants, explicit=True)
            self._write_existing_session_metadata(unit, assignment, session_date, participants)
            return assignment

        if explicit_session_id:
            assignment = SessionAssignment(
                session_id=str(explicit_session_id),
                is_new_session=False,
                confidence=1.0,
                reason="explicit_session_id",
            )
            self._register_or_update_state(sample_id, assignment.session_id, unit, participants, explicit=True)
            self._write_session_metadata(unit, assignment, session_date, participants, auto_session=False)
            return assignment

        active = self._active_sessions.get(sample_id)
        split, reason, score, is_hard = self._should_split(unit, active)

        finalized_previous = None
        if split:
            if active is not None:
                active.is_finalized = True
                active.finalized_at = datetime.now()
                finalized_previous = active.session_id
            timestamp = self._extract_event_timestamp(unit)
            active = self._create_new_session(sample_id, timestamp)
            self._active_sessions[sample_id] = active
            is_new_session = True
        elif active is None:
            timestamp = self._extract_event_timestamp(unit)
            active = self._create_new_session(sample_id, timestamp)
            self._active_sessions[sample_id] = active
            reason = "first_unit"
            score = 1.0
            is_hard = True
            is_new_session = True
        else:
            is_new_session = False

        confidence = 1.0 if is_hard else max(0.5, min(1.0, score))
        assignment = SessionAssignment(
            session_id=active.session_id,
            is_new_session=is_new_session,
            confidence=confidence,
            reason=reason,
            boundary_score=score,
            finalized_previous_session_id=finalized_previous,
        )
        self._update_active_session(active, unit, participants)
        self._write_session_metadata(unit, assignment, session_date, participants, auto_session=True)
        return assignment

    def _extract_existing_session_id(self, unit: MemoryUnit) -> Optional[str]:
        for source in (unit.metadata or {}, unit.raw_data or {}):
            for field in self._SESSION_FIELDS:
                value = source.get(field)
                if value:
                    return str(value)
        return None

    def _write_existing_session_metadata(
        self,
        unit: MemoryUnit,
        assignment: SessionAssignment,
        session_date: Optional[str],
        participants: Optional[List[str]],
    ) -> None:
        metadata = unit.metadata if isinstance(unit.metadata, dict) else {}
        raw_data = unit.raw_data if isinstance(unit.raw_data, dict) else {}
        unit.metadata = metadata
        unit.raw_data = raw_data

        metadata.setdefault("session_id", assignment.session_id)
        raw_data.setdefault("session_id", assignment.session_id)
        metadata.setdefault("auto_session", False)
        metadata.setdefault("existing_session", True)
        self._write_common_session_fields(unit, assignment, session_date, participants, status="active")

    def _write_session_metadata(
        self,
        unit: MemoryUnit,
        assignment: SessionAssignment,
        session_date: Optional[str],
        participants: Optional[List[str]],
        auto_session: bool,
    ) -> None:
        metadata = unit.metadata if isinstance(unit.metadata, dict) else {}
        raw_data = unit.raw_data if isinstance(unit.raw_data, dict) else {}
        unit.metadata = metadata
        unit.raw_data = raw_data

        metadata["session_id"] = assignment.session_id
        raw_data["session_id"] = assignment.session_id
        metadata["auto_session"] = bool(auto_session)
        if auto_session:
            metadata["auto_session_confidence"] = assignment.confidence
            metadata["auto_session_reason"] = assignment.reason
            metadata["auto_session_boundary_score"] = assignment.boundary_score
        else:
            metadata["existing_session"] = True
        self._write_common_session_fields(unit, assignment, session_date, participants, status="active")

    def _write_common_session_fields(
        self,
        unit: MemoryUnit,
        assignment: SessionAssignment,
        session_date: Optional[str],
        participants: Optional[List[str]],
        status: str,
    ) -> None:
        inferred_date = self._infer_session_date(unit, session_date)
        unit.metadata.setdefault("session_date", inferred_date)
        unit.raw_data.setdefault("session_date", inferred_date)
        unit.metadata["session_status"] = status
        if participants:
            unit.metadata.setdefault("participants", list(participants))
            unit.raw_data.setdefault("participants", list(participants))
        if assignment.finalized_previous_session_id:
            unit.metadata["auto_session_finalized_previous_session_id"] = assignment.finalized_previous_session_id
        unit.refresh_text_cache()

    def _extract_event_timestamp(self, unit: MemoryUnit) -> Optional[datetime]:
        """Return a minute-level event timestamp for boundary calculations.

        Session/date-only metadata is intentionally excluded. A bare calendar
        date cannot establish elapsed minutes and must not become midnight in
        ``ActiveSessionState.last_time``.
        """
        for source in (unit.metadata or {}, unit.raw_data or {}):
            for field in self._EVENT_TIMESTAMP_FIELDS:
                parsed = self._parse_event_datetime(source.get(field))
                if parsed is not None:
                    return parsed
        created_time = getattr(unit, "created_time", None)
        return self._parse_event_datetime(created_time)

    def _parse_event_datetime(self, value: Any) -> Optional[datetime]:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, date):
            return None
        if isinstance(value, (int, float)):
            try:
                return datetime.fromtimestamp(value)
            except (OSError, OverflowError, ValueError):
                return None
        if not isinstance(value, str):
            return None

        text = value.strip()
        if not text:
            return None
        if re.fullmatch(r"\d{4}[-/]\d{2}[-/]\d{2}", text):
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        for candidate in (text, text.replace("/", "-")):
            try:
                return datetime.fromisoformat(candidate)
            except ValueError:
                pass
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
                try:
                    return datetime.strptime(candidate, fmt)
                except ValueError:
                    continue
        return None

    def _extract_text(self, unit: MemoryUnit) -> str:
        text = extract_original_text(unit)
        if text:
            return text
        return getattr(unit, "text_cached", "") or ""

    def _extract_participants(self, unit: MemoryUnit, participants: Optional[List[str]] = None) -> Set[str]:
        values: Set[str] = set()
        if participants:
            values.update(str(item) for item in participants if item)
        for source in (unit.metadata or {}, unit.raw_data or {}):
            for field in self._PARTICIPANT_FIELDS:
                raw_value = source.get(field)
                if isinstance(raw_value, (list, tuple, set)):
                    values.update(str(item) for item in raw_value if item)
                elif raw_value:
                    values.add(str(raw_value))
            for field in self._SPEAKER_FIELDS:
                raw_value = source.get(field)
                if raw_value:
                    values.add(str(raw_value))
        return values

    def _should_split(
        self,
        unit: MemoryUnit,
        active: Optional[ActiveSessionState],
    ) -> Tuple[bool, str, float, bool]:
        if active is None:
            return True, "first_unit", 1.0, True

        timestamp = self._extract_event_timestamp(unit)
        text = self._extract_text(unit)
        char_len = len(text)

        if active.unit_count >= self.config.max_units_per_session:
            return True, "max_units", 1.0, True
        if active.char_count + char_len > self.config.max_chars_per_session:
            return True, "max_chars", 1.0, True

        gap_minutes: Optional[float] = None
        day_changed = False
        if timestamp is not None and active.last_time is not None:
            gap_minutes = max(0.0, (timestamp - active.last_time).total_seconds() / 60.0)
            day_changed = timestamp.date() != active.last_time.date()
            if gap_minutes >= self.config.hard_gap_minutes:
                return True, f"hard_gap:{gap_minutes:.1f}m", 1.0, True

        min_reached = (
            active.unit_count >= self.config.min_units_per_session
            or active.char_count >= self.config.min_chars_per_session
        )
        if self.config.day_boundary_split and day_changed and min_reached:
            return True, "day_boundary", 1.0, True

        if active.unit_count < self.config.min_units_per_session and active.char_count < self.config.min_chars_per_session:
            return False, "debounce_short_session", 0.0, False

        score = 0.0
        reasons: List[str] = []
        if gap_minutes is not None and gap_minutes >= self.config.soft_gap_minutes:
            span = max(1.0, self.config.hard_gap_minutes - self.config.soft_gap_minutes)
            gap_score = 0.25 + min(0.10, 0.10 * (gap_minutes - self.config.soft_gap_minutes) / span)
            score += gap_score
            reasons.append(f"soft_gap:{gap_minutes:.1f}m")
        if day_changed:
            score += 0.35
            reasons.append("day_changed")

        topic_distance = self._topic_distance(unit.embedding, active)
        if topic_distance is not None and topic_distance >= self.config.topic_shift_threshold:
            score += 0.35
            reasons.append(f"topic_shift:{topic_distance:.3f}")

        unit_participants = self._extract_participants(unit)
        if unit_participants and active.participants and unit_participants.isdisjoint(active.participants):
            score += 0.12
            reasons.append("participants_changed")

        if self._looks_like_new_task(text):
            score += 0.10
            reasons.append("new_task_phrase")

        if score >= self.config.boundary_score_threshold:
            return True, "+".join(reasons) or "boundary_score", score, False
        return False, "+".join(reasons) or "same_session", score, False

    def _topic_distance(
        self,
        unit_embedding: Optional[np.ndarray],
        active: ActiveSessionState,
    ) -> Optional[float]:
        if unit_embedding is None or active.centroid_embedding is None:
            return None
        try:
            unit_vec = np.asarray(unit_embedding, dtype=np.float32).reshape(-1)
            centroid = np.asarray(active.centroid_embedding, dtype=np.float32).reshape(-1)
            if unit_vec.shape != centroid.shape:
                return None
            denom = float(np.linalg.norm(unit_vec) * np.linalg.norm(centroid))
            if denom <= 1e-12:
                return None
            similarity = float(np.dot(unit_vec, centroid) / denom)
            similarity = max(-1.0, min(1.0, similarity))
            return 1.0 - similarity
        except Exception as exc:
            logger.debug("topic distance failed for auto session: %s", exc)
            return None

    def _looks_like_new_task(self, text: str) -> bool:
        lowered = (text or "").lower()
        return any(pattern in lowered for pattern in self._NEW_TOPIC_PATTERNS)

    def _create_new_session(self, sample_id: str, timestamp: Optional[datetime]) -> ActiveSessionState:
        safe_sample_id = self._sanitize_sample_id(sample_id)
        self._counters[safe_sample_id] = self._counters.get(safe_sample_id, 0) + 1
        counter = self._counters[safe_sample_id]
        if timestamp is not None:
            session_suffix = f"{timestamp.strftime('%Y%m%d')}_{counter:04d}"
        else:
            session_suffix = f"{counter:04d}"
        session_id = f"{safe_sample_id}::auto_session::{session_suffix}"
        return ActiveSessionState(
            session_id=session_id,
            sample_id=sample_id,
            created_at=datetime.now(),
            start_time=timestamp,
            last_time=timestamp,
        )

    def _register_or_update_state(
        self,
        sample_id: str,
        session_id: str,
        unit: MemoryUnit,
        participants: Optional[List[str]],
        explicit: bool,
    ) -> None:
        active = self._active_sessions.get(sample_id)
        if active is None or active.session_id != session_id:
            timestamp = self._extract_event_timestamp(unit)
            active = ActiveSessionState(
                session_id=session_id,
                sample_id=sample_id,
                created_at=datetime.now(),
                start_time=timestamp,
                last_time=timestamp,
                metadata={"explicit": explicit},
            )
            self._active_sessions[sample_id] = active
        self._update_active_session(active, unit, participants)

    def _update_active_session(
        self,
        state: ActiveSessionState,
        unit: MemoryUnit,
        participants: Optional[List[str]],
    ) -> None:
        timestamp = self._extract_event_timestamp(unit)
        text = self._extract_text(unit)
        if state.start_time is None and timestamp is not None:
            state.start_time = timestamp
        if timestamp is not None:
            state.last_time = timestamp
        state.unit_uids.append(unit.uid)
        state.unit_count += 1
        state.char_count += len(text)
        state.participants.update(self._extract_participants(unit, participants))

        if unit.embedding is not None:
            vec = np.asarray(unit.embedding, dtype=np.float32).reshape(-1)
            if state.centroid_embedding is None:
                state.centroid_embedding = vec.copy()
            elif state.centroid_embedding.shape == vec.shape:
                previous_count = max(1, state.unit_count - 1)
                state.centroid_embedding = (
                    state.centroid_embedding * previous_count + vec
                ) / float(previous_count + 1)
            state.recent_embeddings.append(vec)

    def _infer_session_date(self, unit: MemoryUnit, session_date: Optional[str]) -> str:
        if session_date:
            return str(session_date)
        for source in (unit.metadata or {}, unit.raw_data or {}):
            value = source.get("session_date") or source.get("date")
            if value:
                return str(value)[:10]
        timestamp = self._extract_event_timestamp(unit)
        if timestamp is not None:
            return timestamp.date().isoformat()
        return "unknown"

    def _sanitize_sample_id(self, sample_id: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_.:-]+", "_", str(sample_id or "sample")).strip("_")
        return safe or "sample"
