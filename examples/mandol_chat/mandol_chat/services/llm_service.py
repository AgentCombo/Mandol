"""LLM wrapper with a safe mock fallback."""

from __future__ import annotations

from typing import List

from ..config import ChatConfig
from ..schemas import MemoryHit


class LLMService:
    def __init__(self, config: ChatConfig):
        self.config = config
        self.mode = "mock"
        self._client = None
        if not config.mock_llm:
            self._init_real_client()

    def generate_reply(self, message: str, memories: List[MemoryHit], session_id: str) -> str:
        if self.mode == "real" and self._client is not None:
            try:
                prompt = self._build_prompt(message, memories, session_id)
                return self._client.generate_answer(prompt=prompt, temperature=0.2, max_tokens=350)
            except Exception:
                self.mode = "mock"
        return self._mock_reply(message, memories)

    def _init_real_client(self) -> None:
        try:
            from mandol.llm.llm_client import LLMClient

            self._client = LLMClient(model_name=self.config.llm_model)
            self.mode = "real"
        except Exception:
            self._client = None
            self.mode = "mock"

    def _build_prompt(self, message: str, memories: List[MemoryHit], session_id: str) -> str:
        context = "\n".join(
            f"- [{hit.session_id}] {hit.content[:300]}" for hit in memories
        ) or "(no retrieved memories)"
        return (
            "You are a concise assistant in a Mandol memory demo.\n"
            f"Current session: {session_id}\n"
            f"Retrieved memories:\n{context}\n\n"
            f"User message: {message}\n"
            "Reply helpfully in 2-4 sentences."
        )

    def _mock_reply(self, message: str, memories: List[MemoryHit]) -> str:
        if memories:
            snippets = "；".join(hit.content[:60] for hit in memories[:2])
            return f"我已记录这条消息。根据记忆，我找到 {len(memories)} 条相关内容：{snippets}"
        if "?" in message or "？" in message:
            return "我已记录这个问题。当前还没有找到明显相关的历史记忆，你可以继续补充背景。"
        return "我已记录这条信息。你可以继续告诉我更多内容，或者试试右侧的记忆搜索。"

