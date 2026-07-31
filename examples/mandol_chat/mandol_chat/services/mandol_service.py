"""Mandol integration layer for the chat demo."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from mandol.auto_builder.auto_session_assigner import AutoSessionAssigner
from mandol.auto_builder.session_tracker import SessionTracker
from mandol.core.memory_unit import MemoryUnit

from ..config import ChatConfig
from ..schemas import MemoryHit


@dataclass
class AddMessageResult:
    unit: MemoryUnit
    session_id: str
    unit_uid: str
    auto_session_reason: Optional[str] = None
    auto_session_confidence: Optional[float] = None


class MandolService:
    """Single boundary for all Mandol operations used by the demo."""

    def __init__(self, config: ChatConfig):
        self.config = config
        self.config.data_dir.mkdir(parents=True, exist_ok=True)
        self.auto_session_assigner = AutoSessionAssigner()
        self.session_tracker = SessionTracker()
        self.units: Dict[str, MemoryUnit] = {}
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self._counter = 0
        self._last_session_id: Optional[str] = None
        self._semantic_graph = None
        self._real_mandol_ready = False
        self._init_semantic_graph_if_requested()

    @property
    def mandol_ready(self) -> bool:
        return True

    @property
    def real_embedding_enabled(self) -> bool:
        return self._real_mandol_ready

    @property
    def active_session_id(self) -> Optional[str]:
        return self._last_session_id

    def add_chat_message(
        self,
        role: str,
        content: str,
        user_id: str = "demo_user",
        speaker: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        force_session_id: Optional[str] = None,
    ) -> AddMessageResult:
        timestamp = timestamp or datetime.now()
        self._counter += 1
        uid = f"chat_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}_{self._counter:06d}"
        raw_data = {
            "content": content,
            "text_content": content,
            "role": role,
            "speaker": speaker or role,
            "user_id": user_id,
            "timestamp": timestamp.isoformat(),
            "source": "mandol_chat",
        }
        metadata = {
            "created": timestamp.isoformat(),
            "updated": timestamp.isoformat(),
            "source": "mandol_chat",
            "role": role,
            "user_id": user_id,
            "sample_id": user_id,
            "timestamp": timestamp.isoformat(),
        }
        unit = MemoryUnit(uid=uid, raw_data=raw_data, metadata=metadata)

        assignment = self.auto_session_assigner.assign_one(
            unit=unit,
            sample_id=user_id,
            participants=[user_id, "assistant"],
            explicit_session_id=force_session_id,
        )
        session_id = assignment.session_id
        self.units[uid] = unit
        self._last_session_id = session_id
        self._track_session(unit, assignment)
        self._write_to_semantic_graph(unit, user_id, session_id)

        return AddMessageResult(
            unit=unit,
            session_id=session_id,
            unit_uid=uid,
            auto_session_reason=unit.metadata.get("auto_session_reason"),
            auto_session_confidence=unit.metadata.get("auto_session_confidence"),
        )

    def search_memory(self, query: str, top_k: int = 5) -> List[MemoryHit]:
        if not query.strip():
            return []
        if self._real_mandol_ready:
            try:
                return self._search_real_mandol(query, top_k)
            except Exception:
                # Fall back to deterministic keyword search for demo resilience.
                pass
        return self._search_keyword(query, top_k)

    def list_sessions(self) -> Dict[str, Any]:
        sessions = sorted(self.sessions.values(), key=lambda item: item.get("created_at") or "")
        active = [session for session in sessions if not session.get("is_finalized")]
        return {
            "active_session_id": self._last_session_id,
            "active_sessions": active,
            "all_sessions": sessions,
            "stats": {
                "total_sessions": len(sessions),
                "active_sessions": len(active),
                "finalized_sessions": len(sessions) - len(active),
                "unit_count": len(self.units),
            },
        }

    def finalize_session(self, session_id: str) -> Dict[str, Any]:
        if session_id not in self.sessions:
            return {"success": False, "message": f"Session not found: {session_id}"}
        now = datetime.now().isoformat()
        session = self.sessions[session_id]
        session["is_finalized"] = True
        session["finalized_at"] = now
        session["pending_high_level_build"] = True
        self.session_tracker.finalize_session(session_id)
        for unit in self.units.values():
            if self._unit_session_id(unit) == session_id:
                unit.metadata["session_status"] = "finalized"
        return {"success": True, "session": session}

    def build_high_level_memory(
        self,
        session_id: Optional[str] = None,
        sample_id: str = "demo_user",
        build_hierarchical: bool = True,
        build_episodic: bool = True,
        build_entity_relation: bool = True,
    ) -> Dict[str, Any]:
        target_units = [
            unit
            for unit in self.units.values()
            if (session_id is None or self._unit_session_id(unit) == session_id)
            and unit.metadata.get("user_id") == sample_id
        ]
        if self.config.mock_llm:
            return {
                "success": True,
                "status": "mock_build",
                "message": "Mock mode: high-level memory build was not run. Configure a real LLM to build towers.",
                "result": {
                    "session_id": session_id,
                    "sample_id": sample_id,
                    "candidate_units": len(target_units),
                    "build_hierarchical": build_hierarchical,
                    "build_episodic": build_episodic,
                    "build_entity_relation": build_entity_relation,
                },
            }
        return {
            "success": False,
            "status": "requires_llm",
            "message": "High-level memory construction requires a configured real LLM in this demo.",
            "result": {"candidate_units": len(target_units), "session_id": session_id},
        }

    def reset(self) -> None:
        self.auto_session_assigner = AutoSessionAssigner()
        self.session_tracker = SessionTracker()
        self.units.clear()
        self.sessions.clear()
        self._counter = 0
        self._last_session_id = None
        for path in self.config.data_dir.glob("*.json"):
            if path.is_file():
                path.unlink()
        self._init_semantic_graph_if_requested()

    def _init_semantic_graph_if_requested(self) -> None:
        self._semantic_graph = None
        self._real_mandol_ready = False
        if not self.config.real_embedding:
            return
        try:
            from mandol.core.semantic_graph import SemanticGraph

            self._semantic_graph = SemanticGraph()
            self._real_mandol_ready = True
        except Exception:
            self._semantic_graph = None
            self._real_mandol_ready = False

    def _write_to_semantic_graph(self, unit: MemoryUnit, user_id: str, session_id: str) -> None:
        if not self._real_mandol_ready or self._semantic_graph is None:
            return
        try:
            self._semantic_graph.add_unit(
                unit,
                explicit_content_for_embedding=unit.raw_data.get("text_content"),
                content_type_for_embedding="text",
                space_names=[
                    "raw",
                    "chat",
                    self._space_name(f"user:{user_id}"),
                    self._space_name(f"session:{session_id}"),
                ],
                index_update_mode="incremental",
                generate_sparse_embedding=False,
            )
        except Exception:
            self._real_mandol_ready = False

    def _search_real_mandol(self, query: str, top_k: int) -> List[MemoryHit]:
        if self._semantic_graph is None:
            return []
        results = self._semantic_graph.search_similarity_in_graph(
            query_text=query,
            top_k=top_k,
            ms_names=["chat"],
            return_score=True,
        )
        hits = []
        for unit, score in results:
            hits.append(self._hit_from_unit(unit, float(score)))
        return hits

    def _search_keyword(self, query: str, top_k: int) -> List[MemoryHit]:
        query_tokens = self._tokens(query)
        scored = []
        for unit in self.units.values():
            content = self._unit_content(unit)
            content_tokens = self._tokens(content)
            overlap = query_tokens & content_tokens
            if not overlap:
                if query.strip().lower() not in content.lower():
                    continue
            score = self._keyword_score(query_tokens, content_tokens, query, content)
            scored.append((score, unit))
        scored.sort(key=lambda item: (item[0], item[1].metadata.get("created", "")), reverse=True)
        return [self._hit_from_unit(unit, score) for score, unit in scored[:top_k]]

    def _keyword_score(self, query_tokens: set[str], content_tokens: set[str], query: str, content: str) -> float:
        if not query_tokens:
            return 0.0
        overlap = len(query_tokens & content_tokens)
        score = overlap / math.sqrt(max(1, len(query_tokens)) * max(1, len(content_tokens)))
        if query.strip().lower() in content.lower():
            score += 0.5
        return round(float(score), 4)

    def _track_session(self, unit: MemoryUnit, assignment: Any) -> None:
        session_id = self._unit_session_id(unit) or assignment.session_id
        timestamp = unit.metadata.get("timestamp") or datetime.now().isoformat()
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "session_id": session_id,
                "created_at": timestamp,
                "start_time": timestamp,
                "last_time": timestamp,
                "unit_count": 0,
                "unit_uids": [],
                "is_finalized": False,
                "finalized_at": None,
                "pending_high_level_build": False,
                "auto_session": bool(unit.metadata.get("auto_session")),
                "auto_session_reason": unit.metadata.get("auto_session_reason"),
                "auto_session_confidence": unit.metadata.get("auto_session_confidence"),
            }
            self.session_tracker.create_session(
                session_id,
                session_type="chat",
                metadata={"source": "mandol_chat", "auto_session": bool(unit.metadata.get("auto_session"))},
            )
        session = self.sessions[session_id]
        session["last_time"] = timestamp
        session["unit_count"] += 1
        session["unit_uids"].append(unit.uid)
        session["auto_session_reason"] = unit.metadata.get("auto_session_reason") or session.get("auto_session_reason")
        session["auto_session_confidence"] = unit.metadata.get("auto_session_confidence") or session.get("auto_session_confidence")
        self.session_tracker.track_unit(session_id, unit.uid)

    def _hit_from_unit(self, unit: MemoryUnit, score: float) -> MemoryHit:
        return MemoryHit(
            uid=unit.uid,
            content=self._unit_content(unit),
            session_id=self._unit_session_id(unit),
            score=score,
            metadata=dict(unit.metadata or {}),
        )

    def _unit_content(self, unit: MemoryUnit) -> str:
        return str(
            (unit.raw_data or {}).get("text_content")
            or (unit.raw_data or {}).get("content")
            or getattr(unit, "text_cached", "")
        )

    def _unit_session_id(self, unit: MemoryUnit) -> Optional[str]:
        return (unit.metadata or {}).get("session_id") or (unit.raw_data or {}).get("session_id")

    def _tokens(self, text: str) -> set[str]:
        return {token.lower() for token in re.findall(r"[\w\u4e00-\u9fff]+", text or "") if token.strip()}

    def _space_name(self, value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.:-]+", "_", value).strip("_") or "space"

