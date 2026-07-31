"""Chat workflow service."""

from __future__ import annotations

from ..schemas import ChatRequest, ChatResponse
from .llm_service import LLMService
from .mandol_service import MandolService


class ChatService:
    def __init__(self, mandol_service: MandolService, llm_service: LLMService):
        self.mandol_service = mandol_service
        self.llm_service = llm_service

    def chat(self, request: ChatRequest) -> ChatResponse:
        user_result = self.mandol_service.add_chat_message(
            role="user",
            content=request.message,
            user_id=request.user_id,
            speaker=request.user_id,
        )
        memories = self.mandol_service.search_memory(request.message, request.top_k)
        reply = self.llm_service.generate_reply(
            message=request.message,
            memories=memories,
            session_id=user_result.session_id,
        )
        assistant_result = self.mandol_service.add_chat_message(
            role="assistant",
            content=reply,
            user_id=request.user_id,
            speaker="assistant",
            force_session_id=user_result.session_id,
        )
        return ChatResponse(
            assistant_message=reply,
            session_id=user_result.session_id,
            user_unit_uid=user_result.unit_uid,
            assistant_unit_uid=assistant_result.unit_uid,
            created_unit_uids=[user_result.unit_uid, assistant_result.unit_uid],
            retrieved_memories=memories,
            llm_mode=self.llm_service.mode,
        )

