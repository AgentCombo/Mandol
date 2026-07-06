"""Application dependency singletons."""

from __future__ import annotations

from .config import ChatConfig, get_config
from .services.chat_service import ChatService
from .services.llm_service import LLMService
from .services.mandol_service import MandolService
from .services.session_service import SessionService


_config: ChatConfig = get_config()
_mandol_service = MandolService(_config)
_llm_service = LLMService(_config)
_chat_service = ChatService(_mandol_service, _llm_service)
_session_service = SessionService(_mandol_service)


def get_mandol_service() -> MandolService:
    return _mandol_service


def get_llm_service() -> LLMService:
    return _llm_service


def get_chat_service() -> ChatService:
    return _chat_service


def get_session_service() -> SessionService:
    return _session_service


def reset_services() -> None:
    _mandol_service.reset()
    if _config.mock_llm:
        _llm_service.mode = "mock"

