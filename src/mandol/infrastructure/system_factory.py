"""SystemFactory — dependency assembly for MemorySystem.

Creates and wires together concrete infrastructure implementations
so the application layer only depends on port abstractions.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from ..application.memory_system import MemorySystem, MemorySystemConfig
from ..ports.embedding_provider import EmbeddingProvider
from ..ports.llm_provider import LLMProvider
from ..ports.reranker import Reranker
from ..ports.unit_store import UnitStore
from ..ports.vector_index import VectorIndex
from ..ports.graph_store import GraphStore

from .adaptive_vector_index import AdaptiveVectorIndex
from .config import MemorySystemYamlConfig
from .in_memory_graph_store import InMemoryGraphStore
from .in_memory_unit_store import InMemoryUnitStore
from .openai_compatible_llm_provider import OpenAICompatibleLLMProvider
from .sentence_transformers_embedding_provider import SentenceTransformersEmbeddingProvider
from .sentence_transformers_reranker import SentenceTransformersCrossEncoderReranker

logger = logging.getLogger(__name__)


def _load_env_file(reference_path: str) -> None:
    """Load .env file from the directory hierarchy of *reference_path*."""
    for candidate in Path(reference_path).resolve().parents:
        env_path = candidate / ".env"
        if env_path.exists():
            try:
                from dotenv import load_dotenv
                load_dotenv(env_path, override=False)
            except ImportError:
                _parse_dotenv_manual(env_path)
            break


def _parse_dotenv_manual(env_path: Path) -> None:
    import re
    try:
        text = env_path.read_text(encoding="utf-8")
    except OSError:
        return
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$", line)
        if m:
            key, val = m.group(1), m.group(2).strip().strip("'\"")
            if key not in os.environ:
                os.environ[key] = val


class SystemFactory:
    """Assembles a MemorySystem with concrete infrastructure implementations.

    This factory lives in the infrastructure layer so that the application
    layer (MemorySystem) only imports port abstractions, preserving the
    hexagonal architecture constraint.

    Usage::

        system = SystemFactory.create_from_yaml("config.yaml")
        system = SystemFactory.create(config=MemorySystemConfig(...))
    """

    @staticmethod
    def create_from_yaml(
        yaml_path: str,
        *,
        embedder: Optional[EmbeddingProvider] = None,
        reranker: Optional[Reranker] = None,
        llm_provider: Optional[LLMProvider] = None,
        root: Optional[str] = None,
    ) -> MemorySystem:
        """Create a MemorySystem from a YAML configuration file.

        Args:
            yaml_path: Path to config.yaml.
            embedder: Optional custom embedder (overrides YAML config).
            reranker: Optional custom reranker (overrides YAML config).
            llm_provider: Optional custom LLM provider (overrides YAML config).
            root: Optional root space name.

        Returns:
            A fully assembled MemorySystem instance.
        """
        _load_env_file(yaml_path)
        yaml_config = MemorySystemYamlConfig.load_from_yaml(yaml_path)
        return SystemFactory._assemble(
            yaml_config=yaml_config,
            embedder=embedder,
            reranker=reranker,
            llm_provider=llm_provider,
            root=root,
        )

    @staticmethod
    def create(
        *,
        config: Optional[MemorySystemConfig] = None,
        embedder: Optional[EmbeddingProvider] = None,
        reranker: Optional[Reranker] = None,
        llm_provider: Optional[LLMProvider] = None,
        unit_store: Optional[UnitStore] = None,
        vector_index: Optional[VectorIndex] = None,
        graph_store: Optional[GraphStore] = None,
        storage_root: Optional[str] = None,
        enable_persistence: bool = False,
        auto_save_interval: int = 300,
        root: Optional[str] = None,
    ) -> MemorySystem:
        """Create a MemorySystem with programmatic configuration.

        Args:
            config: MemorySystemConfig dataclass (defaults if None).
            embedder: Optional custom embedding provider.
            reranker: Optional custom reranker.
            llm_provider: Optional custom LLM provider.
            unit_store: Optional UnitStore (defaults to InMemoryUnitStore).
            vector_index: Optional VectorIndex (defaults to AdaptiveVectorIndex).
            graph_store: Optional GraphStore (defaults to InMemoryGraphStore).
            storage_root: Directory for automatic persistence.
            enable_persistence: Enable automatic state persistence.
            auto_save_interval: Seconds between automatic save cycles.
            root: Optional root space name.

        Returns:
            A fully assembled MemorySystem instance.
        """
        return SystemFactory._assemble(
            config=config,
            embedder=embedder,
            reranker=reranker,
            llm_provider=llm_provider,
            unit_store=unit_store,
            vector_index=vector_index,
            graph_store=graph_store,
            storage_root=storage_root,
            enable_persistence=enable_persistence,
            auto_save_interval=auto_save_interval,
            root=root,
        )

    @staticmethod
    def _assemble(
        *,
        yaml_config: Optional[MemorySystemYamlConfig] = None,
        config: Optional[MemorySystemConfig] = None,
        embedder: Optional[EmbeddingProvider] = None,
        reranker: Optional[Reranker] = None,
        llm_provider: Optional[LLMProvider] = None,
        unit_store: Optional[UnitStore] = None,
        vector_index: Optional[VectorIndex] = None,
        graph_store: Optional[GraphStore] = None,
        storage_root: Optional[str] = None,
        enable_persistence: bool = False,
        auto_save_interval: int = 300,
        root: Optional[str] = None,
    ) -> MemorySystem:
        """Internal assembly: resolve YAML config, build defaults, wire system."""

        # --- Resolve config from YAML if provided ---
        if yaml_config is not None:
            yaml_cfg = yaml_config
            resolved_config = MemorySystemConfig(
                embedder_model=yaml_cfg.embedder.model,
                embedder_device=yaml_cfg.embedder.device,
                reranker_model=yaml_cfg.reranker.model,
                reranker_device=yaml_cfg.reranker.device,
                llm_model=yaml_cfg.llm.model,
                embedder_dim=yaml_cfg.embedder.dimension,
                promote_threshold=yaml_cfg.promote_threshold,
                chunk_max_tokens=yaml_cfg.chunk_max_tokens,
                session_time_gap_seconds=yaml_cfg.session_time_gap_seconds,
                session_check_interval=yaml_cfg.session_check_interval,
                session_max_pending=yaml_cfg.session_max_pending,
                similarity_top_k=yaml_cfg.similarity_top_k,
                similarity_threshold=yaml_cfg.similarity_threshold,
                similarity_recent_window=yaml_cfg.similarity_recent_window,
                bfs_expansion_per_seed=yaml_cfg.bfs_expansion_per_seed,
                bfs_expansion_hops=yaml_cfg.bfs_expansion_hops,
                max_context_units=yaml_cfg.max_context_units,
                max_entities_per_llm=yaml_cfg.max_entities_per_llm,
                max_events_per_llm=yaml_cfg.max_events_per_llm,
                use_remote_embedder=yaml_cfg.embedder.use_remote,
                use_remote_reranker=yaml_cfg.reranker.use_remote,
                embedder_remote_base_url=yaml_cfg.embedder.base_url,
                embedder_remote_api_path=yaml_cfg.embedder.api_path,
                embedder_remote_timeout=yaml_cfg.embedder.timeout,
                reranker_remote_base_url=yaml_cfg.reranker.base_url,
                reranker_remote_api_path=yaml_cfg.reranker.api_path,
                reranker_remote_timeout=yaml_cfg.reranker.timeout,
            )
            if storage_root is None:
                storage_root = yaml_cfg.storage_root
            if enable_persistence is False:
                enable_persistence = yaml_cfg.enable_persistence
            if auto_save_interval == 300:
                auto_save_interval = yaml_cfg.auto_save_interval

            if llm_provider is None and yaml_cfg.llm.api_key:
                llm_provider = OpenAICompatibleLLMProvider(
                    model=yaml_cfg.llm.model,
                    base_url=yaml_cfg.llm.base_url,
                    api_key=yaml_cfg.llm.api_key,
                )
        else:
            resolved_config = config or MemorySystemConfig()

        if resolved_config is None:
            resolved_config = MemorySystemConfig()

        # --- Build defaults for unset components ---
        if unit_store is None:
            unit_store = InMemoryUnitStore()

        if vector_index is None:
            vector_index = AdaptiveVectorIndex(
                resolved_config.embedder_dim,
                promote_threshold=resolved_config.promote_threshold,
            )

        if graph_store is None:
            graph_store = InMemoryGraphStore()

        if embedder is None:
            embedder = SystemFactory._build_default_embedder(resolved_config)

        if reranker is None:
            reranker = SystemFactory._build_default_reranker(resolved_config)

        if llm_provider is None:
            llm_base_url = os.getenv("MANDOL_LLM_BASE_URL", "")
            llm_api_key = os.getenv("MANDOL_LLM_API_KEY", "")
            llm_kwargs: dict = {"model": resolved_config.llm_model}
            if llm_base_url:
                llm_kwargs["base_url"] = llm_base_url
            if llm_api_key:
                llm_kwargs["api_key"] = llm_api_key
            llm_provider = OpenAICompatibleLLMProvider(**llm_kwargs)

        return MemorySystem(
            config=resolved_config,
            embedder=embedder,
            reranker=reranker,
            llm_provider=llm_provider,
            unit_store=unit_store,
            vector_index=vector_index,
            graph_store=graph_store,
            storage_root=storage_root,
            enable_persistence=enable_persistence,
            auto_save_interval=auto_save_interval,
            root=root,
        )

    @staticmethod
    def _build_default_embedder(config: MemorySystemConfig) -> EmbeddingProvider:
        """Create an embedding provider from config, with fallback."""
        if config.use_remote_embedder:
            try:
                from .openai_compatible_embedding_provider import (
                    OpenAICompatibleEmbeddingConfig,
                    OpenAICompatibleEmbeddingProvider,
                )
                api_config = OpenAICompatibleEmbeddingConfig(
                    base_url=config.embedder_remote_base_url,
                    api_path=config.embedder_remote_api_path,
                    timeout_s=config.embedder_remote_timeout,
                )
                return OpenAICompatibleEmbeddingProvider(
                    model=config.embedder_model,
                    dim=config.embedder_dim,
                    config=api_config,
                )
            except (ImportError, OSError) as exc:
                logger.warning("Failed to create remote embedder: %s", exc)
                from ..ports.embedding_provider import StaticEmbeddingProvider
                return StaticEmbeddingProvider(dim=config.embedder_dim)

        try:
            return SentenceTransformersEmbeddingProvider(
                model=config.embedder_model,
                device=config.embedder_device,
            )
        except (ImportError, OSError) as exc:
            logger.warning(
                "Failed to create embedding provider: %s. "
                "Falling back to StaticEmbeddingProvider (zero embeddings). "
                "Install sentence-transformers for real embeddings.",
                exc,
            )
            from ..ports.embedding_provider import StaticEmbeddingProvider
            return StaticEmbeddingProvider(dim=config.embedder_dim)

    @staticmethod
    def _build_default_reranker(config: MemorySystemConfig) -> Optional[Reranker]:
        """Create a reranker from config, with fallback. Returns None on failure."""
        if config.use_remote_reranker:
            try:
                from .openai_compatible_reranker import (
                    OpenAICompatibleRerankConfig,
                    OpenAICompatibleReranker,
                )
                api_config = OpenAICompatibleRerankConfig(
                    base_url=config.reranker_remote_base_url,
                    api_path=config.reranker_remote_api_path,
                    timeout_s=config.reranker_remote_timeout,
                )
                return OpenAICompatibleReranker(
                    model=config.reranker_model,
                    config=api_config,
                )
            except (ImportError, OSError) as exc:
                logger.warning("Failed to create remote reranker: %s", exc)
                return None

        try:
            return SentenceTransformersCrossEncoderReranker(
                model=config.reranker_model,
                device=config.reranker_device,
            )
        except (ImportError, OSError) as exc:
            logger.warning(
                "Failed to create reranker: %s. Reranking will be skipped.",
                exc,
            )
            return None
