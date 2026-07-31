"""Pydantic schemas for the Mandol chat demo API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class MemoryHit(BaseModel):
    uid: str
    content: str
    session_id: Optional[str] = None
    score: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    user_id: str = "demo_user"
    top_k: int = Field(default=5, ge=0, le=20)


class ChatResponse(BaseModel):
    assistant_message: str
    session_id: str
    user_unit_uid: str
    assistant_unit_uid: str
    created_unit_uids: List[str]
    retrieved_memories: List[MemoryHit]
    llm_mode: str


class MemorySearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)


class MemorySearchResponse(BaseModel):
    query: str
    results: List[MemoryHit]


class HealthResponse(BaseModel):
    status: str
    mandol_ready: bool
    llm_mode: str
    active_session_id: Optional[str] = None
    real_embedding: bool


class SessionInfoResponse(BaseModel):
    active_session_id: Optional[str] = None
    active_sessions: List[Dict[str, Any]] = Field(default_factory=list)
    all_sessions: List[Dict[str, Any]] = Field(default_factory=list)
    stats: Dict[str, Any] = Field(default_factory=dict)


class BuildMemoryRequest(BaseModel):
    session_id: Optional[str] = None
    sample_id: str = "demo_user"
    build_hierarchical: bool = True
    build_episodic: bool = True
    build_entity_relation: bool = True


class BuildMemoryResponse(BaseModel):
    success: bool
    status: str
    message: str
    result: Dict[str, Any] = Field(default_factory=dict)


class ResetResponse(BaseModel):
    success: bool
    message: str
