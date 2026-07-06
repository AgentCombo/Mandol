"""Configuration for the Mandol chat demo."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ChatConfig:
    host: str = "127.0.0.1"
    port: int = 8008
    debug: bool = False
    mock_llm: bool = True
    real_embedding: bool = False
    llm_model: str = "gpt-4o-mini-closeai"
    data_dir: Path = Path("examples/mandol_chat/mandol_chat/data")
    auto_build_high_level_memory: bool = False
    default_user_id: str = "demo_user"
    default_top_k: int = 5


@lru_cache(maxsize=1)
def get_config() -> ChatConfig:
    data_dir = Path(os.getenv("MANDOL_CHAT_DATA_DIR", "examples/mandol_chat/mandol_chat/data"))
    return ChatConfig(
        host=os.getenv("MANDOL_CHAT_HOST", "127.0.0.1"),
        port=int(os.getenv("MANDOL_CHAT_PORT", "8008")),
        debug=_bool_env("MANDOL_CHAT_DEBUG", False),
        mock_llm=_bool_env("MANDOL_CHAT_MOCK_LLM", True),
        real_embedding=_bool_env("MANDOL_CHAT_REAL_EMBEDDING", False),
        llm_model=os.getenv("MANDOL_CHAT_LLM_MODEL", "gpt-4o-mini-closeai"),
        data_dir=data_dir,
        auto_build_high_level_memory=_bool_env("MANDOL_CHAT_AUTO_BUILD_HIGH_LEVEL", False),
        default_user_id=os.getenv("MANDOL_CHAT_DEFAULT_USER_ID", "demo_user"),
        default_top_k=int(os.getenv("MANDOL_CHAT_DEFAULT_TOP_K", "5")),
    )
