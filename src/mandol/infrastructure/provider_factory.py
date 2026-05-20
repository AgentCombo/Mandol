"""Provider factory — resolves and instantiates infrastructure components.

Parses a MemorySystemYamlConfig and constructs the full provider suite
(embedder, reranker, LLM, unit store, graph store, vector index, BM25
index, sparse index). Returns a ProviderFactoryResult holding all
ready-to-use instances.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from ..ports.embedding_provider import EmbeddingProvider
from ..ports.reranker import Reranker
from .openai_compatible_embedding_provider import (
    OpenAICompatibleEmbeddingConfig,
    OpenAICompatibleEmbeddingProvider,
)
from .openai_compatible_reranker import OpenAICompatibleRerankConfig, OpenAICompatibleReranker


@dataclass(frozen=True, slots=True)
class ProviderFactoryResult:
    """Complete set of instantiated infrastructure providers.

    Holds embedder and reranker instances after parsing configuration.
    The LLM provider is constructed separately by MemorySystem itself
    (not through this factory).

    Attributes:
        embedder: Text/image embedding provider, or None if skipped.
        reranker: Cross-encoder reranker, or None if skipped.
    """

    embedder: Optional[EmbeddingProvider] = None
    reranker: Optional[Reranker] = None


def _get_nested(cfg: Mapping[str, Any], *keys: str) -> Any:
    """Walk nested dicts safely, returning None for missing or non-Mapping intermediate keys."""
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, Mapping):
            return None
        cur = cur.get(k)
    return cur


def _pick_device(device: str) -> str:
    """Select the torch device, falling back to CUDA if available and *device* is empty."""
    d = (device or "").strip().lower()
    if d and d != "auto":
        return d

    try:
        import torch

        cuda_ok = bool(
            getattr(torch, "cuda", None)
            and hasattr(torch.cuda, "is_available")
            and torch.cuda.is_available()
        )
        npu_ok = bool(
            getattr(torch, "npu", None)
            and hasattr(torch.npu, "is_available")
            and torch.npu.is_available()
        )
        if cuda_ok:
            return "cuda"
        if npu_ok:
            return "npu"
        return "cpu"
    except (ImportError, RuntimeError, AttributeError):
        return "cpu"


def build_providers_from_config(cfg: Mapping[str, Any]) -> ProviderFactoryResult:
    """Resolve and instantiate all configured infrastructure providers.

    Parses the top-level config dict (from MemorySystemYamlConfig)
    and builds embedder, reranker, LLM, unit store, graph store,
    vector index, BM25 index, and sparse index instances.

    Args:
        cfg: Configuration dict with 'embedding', 'reranker', etc. keys.

    Returns:
        ProviderFactoryResult holding all instantiated providers.
    """
    embedder = _build_embedder(cfg)
    reranker = _build_reranker(cfg)
    return ProviderFactoryResult(embedder=embedder, reranker=reranker)


def _build_embedder(cfg: Mapping[str, Any]) -> Optional[EmbeddingProvider]:
    """Construct the embedding provider from config, supporting local and remote modes.

    Supports 'sentence_transformers' (local) and 'remote' (OpenAI-compatible API)
    provider types, configurable via the 'embedding' section of cfg.
    """
    emb_cfg = cfg.get("embedding") or {}
    if not isinstance(emb_cfg, Mapping):
        return None

    mode = str(emb_cfg.get("mode") or "").strip().lower()
    if not mode:
        return None

    if mode == "remote":
        remote = emb_cfg.get("remote") or {}
        if not isinstance(remote, Mapping):
            remote = {}

        provider = str(remote.get("provider") or "openai_compatible").strip().lower()
        if provider not in {"uni_api", "openai_compatible", "openai"}:
            raise ValueError(f"unsupported embedding remote provider: {provider}")

        base_url = remote.get("base_url")
        api_path = remote.get("api_path")
        token_env = remote.get("token_env")
        timeout_s = remote.get("timeout_s")

        model = remote.get("model")
        dim = remote.get("dim")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("embedding.remote.model is required")
        if dim is None:
            raise ValueError("embedding.remote.dim is required")

        cfg_obj = OpenAICompatibleEmbeddingConfig(
            base_url=str(
                base_url
                or os.getenv("MANDOL_EMBEDDER_BASE_URL")
                or "https://api.openai.com/v1"
            ),
            api_path=str(
                api_path
                or os.getenv("MANDOL_EMBEDDER_API_PATH")
                or "/embeddings"
            ),
            token_env=str(
                token_env
                or os.getenv("MANDOL_EMBEDDER_API_KEY_ENV")
                or "MANDOL_EMBEDDER_API_KEY"
            ),
            timeout_s=int(
                timeout_s
                or int(os.getenv("MANDOL_EMBEDDER_TIMEOUT_S", "60"))
            ),
        )
        return OpenAICompatibleEmbeddingProvider(model=str(model), dim=int(dim), config=cfg_obj)

    if mode == "local":
        local = emb_cfg.get("local") or {}
        if not isinstance(local, Mapping):
            local = {}

        provider = str(local.get("provider") or "sentence_transformers").strip().lower()
        if provider not in {"sentence_transformers", "st"}:
            raise ValueError(f"unsupported embedding local provider: {provider}")

        model = local.get("model") or "Qwen/Qwen3-Embedding-4B"
        device = _pick_device(str(local.get("device") or "auto"))

        from .sentence_transformers_embedding_provider import SentenceTransformersEmbeddingProvider

        return SentenceTransformersEmbeddingProvider(model=str(model), device=device)

    raise ValueError(f"unsupported embedding mode: {mode}")


def _build_reranker(cfg: Mapping[str, Any]) -> Optional[Reranker]:
    """Construct the reranker from config, supporting local and remote modes.

    Supports 'sentence_transformers' (local CrossEncoder) and 'remote'
    (OpenAI-compatible rerank API) provider types.
    """
    rr_cfg = cfg.get("reranker") or {}
    if not isinstance(rr_cfg, Mapping):
        return None

    mode = str(rr_cfg.get("mode") or "").strip().lower()
    if not mode:
        return None

    text_key = str(rr_cfg.get("text_key") or "text_content")

    if mode == "remote":
        remote = rr_cfg.get("remote") or {}
        if not isinstance(remote, Mapping):
            remote = {}

        provider = str(remote.get("provider") or "openai_compatible").strip().lower()
        if provider not in {"uni_api", "openai_compatible", "openai"}:
            raise ValueError(f"unsupported reranker remote provider: {provider}")

        base_url = remote.get("base_url")
        api_path = remote.get("api_path")
        token_env = remote.get("token_env")
        timeout_s = remote.get("timeout_s")
        return_documents = remote.get("return_documents")

        model = remote.get("model")
        if not isinstance(model, str) or not model.strip():
            model = "bge-reranker-v2-m3"

        cfg_obj = OpenAICompatibleRerankConfig(
            base_url=str(
                base_url
                or os.getenv("MANDOL_RERANKER_BASE_URL")
                or ""
            ),
            api_path=str(
                api_path
                or os.getenv("MANDOL_RERANKER_API_PATH")
                or "/v1/rerank"
            ),
            token_env=str(
                token_env
                or os.getenv("MANDOL_RERANKER_API_KEY_ENV")
                or "MANDOL_RERANKER_API_KEY"
            ),
            timeout_s=int(
                timeout_s
                or int(os.getenv("MANDOL_RERANKER_TIMEOUT_S", "60"))
            ),
            return_documents=bool(return_documents) if return_documents is not None else (os.getenv("MANDOL_RERANKER_RETURN_DOCUMENTS", "0") in {"1", "true", "True"}),
        )
        return OpenAICompatibleReranker(model=str(model), config=cfg_obj, text_key=text_key)

    if mode == "local":
        local = rr_cfg.get("local") or {}
        if not isinstance(local, Mapping):
            local = {}

        provider = str(local.get("provider") or "cross_encoder").strip().lower()
        if provider not in {"cross_encoder", "sentence_transformers", "st"}:
            raise ValueError(f"unsupported reranker local provider: {provider}")

        model = local.get("model") or "Qwen/Qwen3-Reranker-4B"
        device = _pick_device(str(local.get("device") or "auto"))

        from .sentence_transformers_reranker import SentenceTransformersCrossEncoderReranker

        return SentenceTransformersCrossEncoderReranker(model=str(model), device=device, text_key=text_key)

    raise ValueError(f"unsupported reranker mode: {mode}")
