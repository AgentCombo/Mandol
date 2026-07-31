"""Session service facade."""

from __future__ import annotations

from typing import Any, Dict

from .mandol_service import MandolService


class SessionService:
    def __init__(self, mandol_service: MandolService):
        self.mandol_service = mandol_service

    def list_sessions(self) -> Dict[str, Any]:
        return self.mandol_service.list_sessions()

    def finalize(self, session_id: str) -> Dict[str, Any]:
        return self.mandol_service.finalize_session(session_id)

