"""Configuration for the LoCoMo memory system example.

Defines dataclasses for memory system settings, remote provider endpoints,
and top-level LoCoMo configuration.  All values can be overridden via
environment variables; ``load_env_from_file`` reads a ``.env`` file on
import so that defaults are applied automatically.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import os


def _env(key: str, default: str) -> str:
    """Read an environment variable, falling back to *default* when unset or empty."""
    v = os.getenv(key)
    return default if v is None or v == "" else v


def _env_int(key: str, default: int) -> int:
    """Read an environment variable as ``int``, falling back to *default*."""
    v = os.getenv(key)
    if v is None or v == "":
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    """Read an environment variable as ``float``, falling back to *default*."""
    v = os.getenv(key)
    if v is None or v == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    """Read an environment variable as ``bool``, falling back to *default*.

    Values ``0``, ``false``, ``no``, ``off``, and empty string are treated
    as ``False``; any other non-empty value is ``True``.
    """
    v = os.getenv(key)
    if v is None or v == "":
        return default
    return v.lower() not in {"0", "false", "no", "off", ""}


@dataclass(frozen=True, slots=True)
class MemorySystemSettings:
    """Parameters forwarded to :class:`MemorySystemConfig`.

    Attributes:
        embedder_model: Model identifier for the embedding provider.
        embedder_device: Torch device for local embedding (``"cpu"`` or ``"cuda"``).
        embedder_dim: Output dimensionality of the embedding model.
        reranker_model: Model identifier for the reranker.
        reranker_device: Torch device for the local reranker.
        llm_model: Model identifier for the LLM provider.
        chunk_max_tokens: Maximum tokens per document chunk.
        session_time_gap_seconds: Seconds between dialogues to consider a new session.
        similarity_top_k: Number of candidates retrieved before reranking.
        similarity_threshold: Minimum cosine similarity for candidate filtering.
        similarity_recent_window: Size of the recency-biased window.
        bfs_expansion_per_seed: Neighbours expanded per seed in graph BFS.
        bfs_expansion_hops: Maximum BFS hop depth during retrieval.
    """
    embedder_model: str = "Qwen/Qwen3-Embedding-4B"
    embedder_device: str = "cpu"
    embedder_dim: int = 2560
    reranker_model: str = "Qwen/Qwen3-Reranker-4B"
    reranker_device: str = "cpu"
    llm_model: str = "gpt-4o-mini"
    chunk_max_tokens: int = 512
    session_time_gap_seconds: int = 1800
    similarity_top_k: int = 5
    similarity_threshold: float = 0.7
    similarity_recent_window: int = 20
    bfs_expansion_per_seed: int = 3
    bfs_expansion_hops: int = 1


@dataclass(frozen=True, slots=True)
class RemoteProviderSettings:
    """Endpoint configuration for remote OpenAI-compatible APIs.

    Attributes:
        embedding_base_url: Base URL for the embedding API.
        embedding_api_path: Path appended to *embedding_base_url*.
        embedding_token_env: Environment variable holding the embedding API token.
        embedding_timeout_s: Request timeout in seconds for embedding calls.
        embedding_require_token: Whether a non-empty token is mandatory.
        reranker_base_url: Base URL for the reranker API.
        reranker_api_path: Path appended to *reranker_base_url*.
        reranker_token_env: Environment variable holding the reranker API token.
        reranker_timeout_s: Request timeout in seconds for reranker calls.
        llm_base_url: Base URL for the LLM chat API.
        llm_api_key_env: Environment variable holding the LLM API key.
        llm_timeout_s: Request timeout in seconds for LLM calls.
    """
    embedding_base_url: str = "http://localhost:8000/v1"
    embedding_api_path: str = "/embeddings"
    embedding_token_env: str = "OPENAI_API_KEY"
    embedding_timeout_s: int = 30
    embedding_require_token: bool = False

    reranker_base_url: str = "https://your-reranker-api-endpoint.com"
    reranker_api_path: str = "/v1/rerank"
    reranker_token_env: str = "OPENAI_API_KEY"
    reranker_timeout_s: int = 30

    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key_env: str = "OPENAI_API_KEY"
    llm_timeout_s: int = 60


@dataclass
class LocomoMemoryConfig:
    """Top-level configuration for the LoCoMo example.

    Attributes:
        dataset_path: Path to the LoCoMo JSON dataset file.
        sample_count: Number of samples to process (1 = first sample only).
        memory_settings: Embedded memory-system parameters.
        remote_provider: Remote API endpoint settings.
        use_remote_embedder: Whether to use a remote embedding API.
        use_remote_reranker: Whether to use a remote reranker API.
        use_remote_llm: Whether to use a remote LLM API.
        log_level: Logging verbosity (``"DEBUG"``, ``"INFO"``, etc.).
        progress_report_interval: Log progress every N processed samples.
    """
    dataset_path: str = "datasets/locomo10.json"
    sample_count: int = 1
    memory_settings: MemorySystemSettings = field(default_factory=MemorySystemSettings)
    remote_provider: RemoteProviderSettings = field(default_factory=RemoteProviderSettings)
    use_remote_embedder: bool = True
    use_remote_reranker: bool = True
    use_remote_llm: bool = True
    log_level: str = "INFO"
    progress_report_interval: int = 1

    def __post_init__(self) -> None:
        """Override fields from environment variables when set."""
        self.memory_settings = MemorySystemSettings(
            embedder_model=_env("EMBEDDER_MODEL", self.memory_settings.embedder_model),
            embedder_device=_env("EMBEDDER_DEVICE", self.memory_settings.embedder_device),
            embedder_dim=_env_int("EMBEDDER_DIM", self.memory_settings.embedder_dim),
            reranker_model=_env("RERANKER_MODEL", self.memory_settings.reranker_model),
            reranker_device=_env("RERANKER_DEVICE", self.memory_settings.reranker_device),
            llm_model=_env("LLM_MODEL", self.memory_settings.llm_model),
            chunk_max_tokens=_env_int("CHUNK_MAX_TOKENS", self.memory_settings.chunk_max_tokens),
            session_time_gap_seconds=_env_int("SESSION_TIME_GAP_SECONDS", self.memory_settings.session_time_gap_seconds),
            similarity_top_k=_env_int("SIMILARITY_TOP_K", self.memory_settings.similarity_top_k),
            similarity_threshold=_env_float("SIMILARITY_THRESHOLD", self.memory_settings.similarity_threshold),
            similarity_recent_window=_env_int("SIMILARITY_RECENT_WINDOW", self.memory_settings.similarity_recent_window),
            bfs_expansion_per_seed=_env_int("BFS_EXPANSION_PER_SEED", self.memory_settings.bfs_expansion_per_seed),
            bfs_expansion_hops=_env_int("BFS_EXPANSION_HOPS", self.memory_settings.bfs_expansion_hops),
        )
        self.remote_provider = RemoteProviderSettings(
            embedding_base_url=_env("EMBEDDING_BASE_URL", self.remote_provider.embedding_base_url),
            embedding_api_path=_env("EMBEDDING_API_PATH", self.remote_provider.embedding_api_path),
            embedding_token_env=_env("EMBEDDING_TOKEN_ENV", self.remote_provider.embedding_token_env),
            embedding_timeout_s=_env_int("EMBEDDING_TIMEOUT_S", self.remote_provider.embedding_timeout_s),
            embedding_require_token=_env_bool("EMBEDDING_REQUIRE_TOKEN", self.remote_provider.embedding_require_token),
            reranker_base_url=_env("RERANKER_BASE_URL", self.remote_provider.reranker_base_url),
            reranker_api_path=_env("RERANKER_API_PATH", self.remote_provider.reranker_api_path),
            reranker_token_env=_env("RERANKER_TOKEN_ENV", self.remote_provider.reranker_token_env),
            reranker_timeout_s=_env_int("RERANKER_TIMEOUT_S", self.remote_provider.reranker_timeout_s),
            llm_base_url=_env("LLM_BASE_URL", self.remote_provider.llm_base_url),
            llm_api_key_env=_env("LLM_API_KEY_ENV", self.remote_provider.llm_api_key_env),
            llm_timeout_s=_env_int("LLM_TIMEOUT_S", self.remote_provider.llm_timeout_s),
        )
        self.dataset_path = _env("DATASET_PATH", self.dataset_path)
        self.sample_count = _env_int("SAMPLE_COUNT", self.sample_count)
        self.use_remote_embedder = _env_bool("USE_REMOTE_EMBEDDER", self.use_remote_embedder)
        self.use_remote_reranker = _env_bool("USE_REMOTE_RERANKER", self.use_remote_reranker)
        self.use_remote_llm = _env_bool("USE_REMOTE_LLM", self.use_remote_llm)
        self.log_level = _env("LOG_LEVEL", self.log_level)
        self.progress_report_interval = _env_int("PROGRESS_REPORT_INTERVAL", self.progress_report_interval)


def load_env_from_file(env_path: Optional[Path] = None) -> None:
    """Load key=value pairs from a ``.env`` file into ``os.environ``.

    Existing environment variables are **not** overwritten.  Lines starting
    with ``#`` and blank lines are ignored.

    Args:
        env_path: Path to the env file.  Defaults to ``.env`` beside this module.
    """
    if env_path is None:
        env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if key and value:
                    os.environ.setdefault(key, value)


load_env_from_file()

DEFAULT_CONFIG = LocomoMemoryConfig()
