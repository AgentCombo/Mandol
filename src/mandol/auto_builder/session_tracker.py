# mandol/auto_builder/session_tracker.py
"""Utilities for session tracker."""
import logging
from ..utils.logging_config import create_module_logger
from typing import Dict, List, Optional, Set
from datetime import datetime
from dataclasses import dataclass, field

logger = create_module_logger("auto_builder.session_tracker")


@dataclass
class SessionInfo:
    session_id: str
    session_type: str = "default"  # meeting, chat, task, document
    created_at: datetime = field(default_factory=datetime.now)
    unit_uids: Set[str] = field(default_factory=set)
    is_finalized: bool = False
    finalized_at: Optional[datetime] = None
    metadata: Dict = field(default_factory=dict)
    
    def add_unit(self, uid: str):
        """Add unit."""
        self.unit_uids.add(uid)
    
    def get_unit_count(self) -> int:
        """Return unit count."""
        return len(self.unit_uids)
    
    def finalize(self):
        """Run finalize."""
        self.is_finalized = True
        self.finalized_at = datetime.now()


class SessionTracker:
    
    def __init__(self):
        self._sessions: Dict[str, SessionInfo] = {}
        logger.info("SessionTracker initialized")
    
    def create_session(self, 
                      session_id: str,
                      session_type: str = "default",
                      metadata: Optional[Dict] = None) -> SessionInfo:
        """Build session."""
        if session_id in self._sessions:
            logger.warning(f"Session {session_id} already exists; returning the existing session")
            return self._sessions[session_id]
        
        session = SessionInfo(
            session_id=session_id,
            session_type=session_type,
            metadata=metadata or {}
        )
        self._sessions[session_id] = session
        
        logger.info(f"Creating session: {session_id} (type: {session_type})")
        return session
    
    def track_unit(self, session_id: str, unit_uid: str):
        """Args: session_id: Session ID."""
        if session_id not in self._sessions:
            self.create_session(session_id)
        
        self._sessions[session_id].add_unit(unit_uid)
        logger.debug(f"Tracking unit {unit_uid} in session {session_id}")
    
    def get_session(self, session_id: str) -> Optional[SessionInfo]:
        """Return session."""
        return self._sessions.get(session_id)
    
    def get_session_units(self, session_id: str) -> List[str]:
        """Return session units."""
        session = self.get_session(session_id)
        return list(session.unit_uids) if session else []
    
    def finalize_session(self, session_id: str):
        """Run finalize session."""
        session = self.get_session(session_id)
        if session:
            session.finalize()
            logger.info(f"Session {session_id} finalized ({session.get_unit_count()} units)")
    
    def get_active_sessions(self) -> List[SessionInfo]:
        """Return active sessions."""
        return [s for s in self._sessions.values() if not s.is_finalized]
    
    def get_all_sessions(self) -> List[SessionInfo]:
        """Return all sessions."""
        return list(self._sessions.values())
    
    def clear_finalized_sessions(self):
        """Remove finalized sessions."""
        finalized = [sid for sid, s in self._sessions.items() if s.is_finalized]
        for sid in finalized:
            del self._sessions[sid]
        logger.info(f"Cleaned up {len(finalized)} finalized sessions")
    
    def get_stats(self) -> Dict:
        """Return stats."""
        active = len(self.get_active_sessions())
        total = len(self._sessions)
        
        return {
            "total_sessions": total,
            "active_sessions": active,
            "finalized_sessions": total - active,
            "sessions": {
                sid: {
                    "type": s.session_type,
                    "unit_count": s.get_unit_count(),
                    "is_finalized": s.is_finalized,
                    "created_at": s.created_at.isoformat()
                }
                for sid, s in self._sessions.items()
            }
        }
