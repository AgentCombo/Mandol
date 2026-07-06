"""FastAPI application for the Mandol chat demo."""

from __future__ import annotations

from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import get_config
from .dependencies import (
    get_chat_service,
    get_llm_service,
    get_mandol_service,
    get_session_service,
    reset_services,
)
from .schemas import (
    BuildMemoryRequest,
    BuildMemoryResponse,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    MemorySearchRequest,
    MemorySearchResponse,
    ResetResponse,
    SessionInfoResponse,
)
from .services.chat_service import ChatService
from .services.llm_service import LLMService
from .services.mandol_service import MandolService
from .services.session_service import SessionService


config = get_config()
APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"

app = FastAPI(
    title="Mandol Chat Demo",
    description="Default real-time streaming memory chat demo for Mandol.",
    version="0.1.0",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health", response_model=HealthResponse)
def health(
    mandol: MandolService = Depends(get_mandol_service),
    llm: LLMService = Depends(get_llm_service),
) -> HealthResponse:
    return HealthResponse(
        status="ok",
        mandol_ready=mandol.mandol_ready,
        llm_mode=llm.mode,
        active_session_id=mandol.active_session_id,
        real_embedding=mandol.real_embedding_enabled,
    )


@app.post("/api/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    return service.chat(request)


@app.post("/api/memory/search", response_model=MemorySearchResponse)
def search_memory(
    request: MemorySearchRequest,
    mandol: MandolService = Depends(get_mandol_service),
) -> MemorySearchResponse:
    return MemorySearchResponse(
        query=request.query,
        results=mandol.search_memory(request.query, request.top_k),
    )


@app.get("/api/sessions", response_model=SessionInfoResponse)
def sessions(service: SessionService = Depends(get_session_service)) -> SessionInfoResponse:
    return SessionInfoResponse(**service.list_sessions())


@app.post("/api/sessions/{session_id}/finalize")
def finalize_session(
    session_id: str,
    service: SessionService = Depends(get_session_service),
):
    result = service.finalize(session_id)
    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result.get("message", "Session not found"))
    return result


@app.post("/api/memory/build", response_model=BuildMemoryResponse)
def build_memory(
    request: BuildMemoryRequest,
    mandol: MandolService = Depends(get_mandol_service),
) -> BuildMemoryResponse:
    result = mandol.build_high_level_memory(
        session_id=request.session_id,
        sample_id=request.sample_id,
        build_hierarchical=request.build_hierarchical,
        build_episodic=request.build_episodic,
        build_entity_relation=request.build_entity_relation,
    )
    return BuildMemoryResponse(**result)


@app.post("/api/reset", response_model=ResetResponse)
def reset() -> ResetResponse:
    reset_services()
    return ResetResponse(success=True, message="Demo state reset.")

