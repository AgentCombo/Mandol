"""Infrastructure configuration for providers, storage, and system parameters.

Defines dataclass-based configuration objects for Milvus, Neo4j, LLM,
embedding, and reranker providers, along with the top-level YAML-driven
MemorySystemYamlConfig that aggregates all sub-configs.
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _env(key: str, default: str) -> str:
    """Read an environment variable, falling back to *default* if unset or empty.

    Args:
        key: Environment variable name.
        default: Value to return when the variable is missing or blank.

    Returns:
        The environment value or *default*.
    """
    v = os.getenv(key)
    return default if v is None or v == "" else v


@dataclass(frozen=True, slots=True)
class MilvusConfig:
    """Configuration for the Milvus vector database connection.

    Attributes:
        uri: Milvus server URI.
        user: Username for authentication.
        password: Password for authentication.
        db_name: Database name to use.
        collection: Collection name for storing memory units.
    """

    uri: str = _env("MANDOL_MILVUS_URI", "http://localhost:19530")
    user: str = _env("MANDOL_MILVUS_USER", "")
    password: str = _env("MANDOL_MILVUS_PASSWORD", "")
    db_name: str = _env("MANDOL_MILVUS_DB", "")
    collection: str = _env("MANDOL_MILVUS_COLLECTION", "mandol_memory_units")


@dataclass(frozen=True, slots=True)
class Neo4jConfig:
    """Configuration for the Neo4j graph database connection.

    Attributes:
        uri: Neo4j Bolt URI.
        user: Username for authentication.
        password: Password for authentication.
        database: Database name to use.
    """

    uri: str = _env("MANDOL_NEO4J_URI", "bolt://localhost:7687")
    user: str = _env("MANDOL_NEO4J_USER", "neo4j")
    password: str = _env("MANDOL_NEO4J_PASSWORD", "neo4j")
    database: str = _env("MANDOL_NEO4J_DATABASE", "neo4j")


@dataclass(frozen=True, slots=True)
class LLMConfig:
    """Configuration for LLM provider (OpenAI-compatible)."""
    api_key: str = _env("MANDOL_LLM_API_KEY", "")
    base_url: str = _env("MANDOL_LLM_BASE_URL", "https://api.openai.com/v1")
    model: str = _env("MANDOL_LLM_MODEL", "gpt-4o-mini")
    temperature: float = 0.1
    max_tokens: int = 4096


@dataclass(frozen=True, slots=True)
class EmbedderConfig:
    """Configuration for embedding provider."""
    model: str = _env("MANDOL_EMBEDDER_MODEL", "Qwen/Qwen3-Embedding-4B")
    device: str = _env("MANDOL_EMBEDDER_DEVICE", "cpu")
    dimension: int = 2560
    # Remote embedder settings
    use_remote: bool = _env("USE_REMOTE_EMBEDDER", "false").lower() == "true"
    base_url: str = _env("MANDOL_EMBEDDER_BASE_URL", "http://localhost:8000/v1")
    api_path: str = _env("MANDOL_EMBEDDER_API_PATH", "/embeddings")
    api_key: str = _env("MANDOL_EMBEDDER_API_KEY", "")
    timeout: int = 30


@dataclass(frozen=True, slots=True)
class RerankerConfig:
    """Configuration for reranker provider."""
    model: str = _env("MANDOL_RERANKER_MODEL", "Qwen/Qwen3-Reranker-4B")
    device: str = _env("MANDOL_RERANKER_DEVICE", "cpu")
    # Remote reranker settings
    use_remote: bool = _env("USE_REMOTE_RERANKER", "false").lower() == "true"
    base_url: str = _env("MANDOL_RERANKER_BASE_URL", "")
    api_path: str = _env("MANDOL_RERANKER_API_PATH", "/v1/rerank")
    api_key: str = _env("MANDOL_RERANKER_API_KEY", "")
    timeout: int = 30


@dataclass
class MemorySystemYamlConfig:
    """Configuration loaded from config.yaml, aggregating all sub-configs.

    Key fields:
        llm: LLM provider configuration (LLMConfig).
        embedder: Embedding provider configuration (EmbedderConfig).
        reranker: Reranker provider configuration (RerankerConfig).
        storage_root: Root directory for persistence files.
        enable_persistence: Whether to enable automatic state persistence.
        auto_save_interval: Seconds between automatic save cycles (default 300).
        chunk_max_tokens: Maximum tokens per text chunk for the DocumentChunker.
        session_time_gap_seconds: Idle gap (seconds) that triggers a new session.
        session_check_interval: Minimum pending units before session boundary check.
        session_max_pending: Hard limit: force-flush pending buffer when reached.
        similarity_threshold: Cosine similarity threshold (0.0–1.0) for SEMANTIC_SIMILAR edges.
        promote_threshold: Minimum vectors before promoting flat index to ANN.
    """

    llm: LLMConfig = field(default_factory=LLMConfig)
    embedder: EmbedderConfig = field(default_factory=EmbedderConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    storage_root: Optional[str] = None
    enable_persistence: bool = False
    auto_save_interval: int = 300        # Seconds
    chunk_max_tokens: int = 512          # Tokens per chunk
    session_time_gap_seconds: int = 1800 # 30 minutes
    session_check_interval: int = 20     # Minimum pending count
    session_max_pending: int = 100       # Hard flush limit
    similarity_top_k: int = 5            # Top-K candidates for similarity edge creation
    similarity_threshold: float = 0.7    # Cosine similarity threshold (0.0-1.0)
    similarity_recent_window: int = 20   # Recent window size for similarity checks
    bfs_expansion_per_seed: int = 3      # Max neighbors expanded per seed during BFS
    bfs_expansion_hops: int = 1          # Max BFS hop depth
    max_context_units: int = 20          # Max units in retrieval context window
    max_entities_per_llm: int = 50       # Max entities per LLM extraction call
    max_events_per_llm: int = 50         # Max events per LLM extraction call
    promote_threshold: int = 100         # Vectors needed before ANN promotion

    @classmethod
    def load_from_yaml(cls, yaml_path: str) -> "MemorySystemYamlConfig":
        """Load configuration from a YAML file."""
        try:
            import yaml
        except ImportError:
            logger.warning("PyYAML not installed, using default config")
            return cls()

        path = Path(yaml_path)
        if not path.exists():
            logger.warning(f"Config file {yaml_path} not found, using defaults")
            return cls()

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            if not isinstance(data, dict):
                logger.warning(f"Invalid config file format: {yaml_path}")
                return cls()

            # Parse LLM config
            llm_data = data.get("llm", {})
            # Priority: .env > config.yaml > defaults
            llm_api_key = _env("MANDOL_LLM_API_KEY", "") or llm_data.get("api_key", "")
            llm_base_url = _env("MANDOL_LLM_BASE_URL", "") or llm_data.get("base_url", "https://api.openai.com/v1")
            llm_model = _env("MANDOL_LLM_MODEL", "") or llm_data.get("model", "gpt-4o-mini")
            llm = LLMConfig(
                api_key=llm_api_key,
                base_url=llm_base_url,
                model=llm_model,
            )

            # Parse Embedder config
            embedder_data = data.get("embedder", {})
            # Priority: .env > config.yaml > defaults
            embedder_use_remote = _env("USE_REMOTE_EMBEDDER", "").lower() == "true" or embedder_data.get("use_remote", False)
            embedder_model = _env("MANDOL_EMBEDDER_MODEL", "") or embedder_data.get("model", "Qwen/Qwen3-Embedding-4B")
            embedder_device = _env("MANDOL_EMBEDDER_DEVICE", "") or embedder_data.get("device", "cpu")
            embedder_dimension = embedder_data.get("dimension", 2560)
            embedder_base_url = _env("MANDOL_EMBEDDER_BASE_URL", "") or embedder_data.get("base_url", "http://localhost:8000/v1")
            embedder_api_path = _env("MANDOL_EMBEDDER_API_PATH", "") or embedder_data.get("api_path", "/embeddings")
            embedder_api_key = _env("MANDOL_EMBEDDER_API_KEY", "") or embedder_data.get("api_key", "")
            embedder_timeout = embedder_data.get("timeout", 30)
            embedder = EmbedderConfig(
                model=embedder_model,
                device=embedder_device,
                dimension=embedder_dimension,
                use_remote=embedder_use_remote,
                base_url=embedder_base_url,
                api_path=embedder_api_path,
                api_key=embedder_api_key,
                timeout=embedder_timeout,
            )

            # Parse Reranker config
            reranker_data = data.get("reranker", {})
            # Priority: .env > config.yaml > defaults
            reranker_use_remote = _env("USE_REMOTE_RERANKER", "").lower() == "true" or reranker_data.get("use_remote", False)
            reranker_model = _env("MANDOL_RERANKER_MODEL", "") or reranker_data.get("model", "Qwen/Qwen3-Reranker-4B")
            reranker_device = _env("MANDOL_RERANKER_DEVICE", "") or reranker_data.get("device", "cpu")
            reranker_base_url = _env("MANDOL_RERANKER_BASE_URL", "") or reranker_data.get("base_url", "")
            reranker_api_path = _env("MANDOL_RERANKER_API_PATH", "") or reranker_data.get("api_path", "/v1/rerank")
            reranker_api_key = _env("MANDOL_RERANKER_API_KEY", "") or reranker_data.get("api_key", "")
            reranker_timeout = reranker_data.get("timeout", 30)
            reranker = RerankerConfig(
                model=reranker_model,
                device=reranker_device,
                use_remote=reranker_use_remote,
                base_url=reranker_base_url,
                api_path=reranker_api_path,
                api_key=reranker_api_key,
                timeout=reranker_timeout,
            )

            # Parse system config
            system_data = data.get("system", {})
            storage_data = data.get("storage", {})

            return cls(
                llm=llm,
                embedder=embedder,
                reranker=reranker,
                storage_root=storage_data.get("root", None),
                enable_persistence=storage_data.get("enable_persistence", False),
                auto_save_interval=storage_data.get("auto_save_interval", 300),
                chunk_max_tokens=system_data.get("chunk_max_tokens", 512),
                session_time_gap_seconds=system_data.get("session_time_gap_seconds", 1800),
                session_check_interval=system_data.get("session_check_interval", 20),
                session_max_pending=system_data.get("session_max_pending", 100),
                similarity_top_k=system_data.get("similarity_top_k", 5),
                similarity_threshold=system_data.get("similarity_threshold", 0.7),
                similarity_recent_window=system_data.get("similarity_recent_window", 20),
                bfs_expansion_per_seed=system_data.get("bfs_expansion_per_seed", 3),
                bfs_expansion_hops=system_data.get("bfs_expansion_hops", 1),
                max_context_units=system_data.get("max_context_units", 20),
                max_entities_per_llm=system_data.get("max_entities_per_llm", 50),
                max_events_per_llm=system_data.get("max_events_per_llm", 50),
                promote_threshold=system_data.get("promote_threshold", 100),
            )
        except Exception as e:
            logger.warning(f"Failed to load config from {yaml_path}: {e}")
            return cls()
