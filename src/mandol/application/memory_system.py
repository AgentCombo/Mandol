"""MemorySystem — the main orchestration facade for the Mandol memory engine.

Coordinates all application-layer services including semantic mapping, graph
building, document chunking, session detection, entity/event extraction, insight
reduction, and multi-modal retrieval. Provides a unified interface for adding
memory units, building high-level memory, saving/loading state, and performing
holistic or per-group retrieval with reranking.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..infrastructure.persistence_manager import PersistenceManager
    from ..infrastructure.memory_monitor import MemoryMonitor

from ..domain.memory_space import MemorySpace
from ..domain.memory_unit import MemoryUnit
from ..domain.types import Embedding, SpaceName, Uid
from ..infrastructure.adaptive_vector_index import AdaptiveVectorIndex
from ..infrastructure.in_memory_graph_store import InMemoryGraphStore
from ..infrastructure.in_memory_unit_store import InMemoryUnitStore
from ..infrastructure.openai_compatible_llm_provider import OpenAICompatibleLLMProvider
from ..infrastructure.sentence_transformers_embedding_provider import SentenceTransformersEmbeddingProvider
from ..infrastructure.sentence_transformers_reranker import SentenceTransformersCrossEncoderReranker
from ..ports.embedding_provider import EmbeddingProvider, StaticEmbeddingProvider
from ..ports.llm_provider import LLMChatResponse, LLMProvider, ChatMessage
from ..ports.reranker import Reranker
from .chunker import DocumentChunker
from .extractors.entity_dedup import EntityDeduplicator
from .extractors.entity_relation_extract import EntityRelationExtractor
from .extractors.event_dedup import EventDeduplicator
from .extractors.event_causal_extract import EventCausalExtractor
from .reducers.global_insight_manager import GlobalInsightManager
from .reducers.insight_map_reducer import InsightMapReducer
from .legacy.multidim_semantic_graph import (
    MultiDimSemanticGraphBuilder,
    SpaceNamingPolicy,
)
from .semantic_graph import SemanticGraphService
from .semantic_map import SemanticMapService
from .session_manager import Session, SessionManager
from .reducers.summary_map_reducer import SummaryMapReducer
from .pipeline.unified_fact_pipeline import UnifiedFactPipeline, ExtractedEntity
from .pipeline.cross_session_coref_manager import CrossSessionCorefManager
from ..retrieval.types import SearchHit
from .services._retrieval import MemoryRetrievalService
from .services._persistence import MemoryPersistenceService, SaveResult, LoadResult

logger = logging.getLogger(__name__)


def _load_env_file(reference_path: str) -> None:
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


SEMANTIC_SIMILAR = "SEMANTIC_SIMILAR"
# Relationship type label connecting high-level units to their source data.
EVIDENCED_BY = "EVIDENCED_BY"

# Maximum number of context units in the async sliding window.
MAX_CONTEXT_UNITS = 20
# Upper bound on entities sent to the LLM in a single call.
MAX_ENTITIES_PER_LLM = 50
# Upper bound on events sent to the LLM in a single call.
MAX_EVENTS_PER_LLM = 50
# Number of accumulated units per batch for LLM session analysis.
SESSION_CHECK_INTERVAL = 20
# If pending units exceed this threshold, force a session build.
SESSION_MAX_PENDING = 100


@dataclass(frozen=True, slots=True)
class MemorySystemConfig:
    """Immutable configuration for the MemorySystem.

    Controls model selection, similarity thresholds, pipeline behavior,
    chunking parameters, and session detection settings.

    Attributes:
        embedder_model: HuggingFace model ID for the local embedder.
        embedder_device: Device for the embedder (cpu/cuda/cuda:0).
        reranker_model: HuggingFace cross-encoder model ID for reranking.
        reranker_device: Device for the reranker.
        llm_model: OpenAI-compatible model name for LLM calls.
        embedder_dim: Expected embedding dimension.
        promote_threshold: Minimum entries in an in-memory space before promotion.
        chunk_max_tokens: Token limit per document chunk.
        session_time_gap_seconds: Time gap hint injected into the session detection prompt for LLM reference only. Not used for hard splitting.
        session_check_interval: Unit count to trigger session boundary checks.
        session_max_pending: Max pending units before forced session build.
        similarity_top_k: Number of similarity edges to build.
        similarity_threshold: Minimum cosine score to create a similarity edge.
        similarity_recent_window: Recent units considered for similarity edges.
        bfs_expansion_per_seed: Units to collect per seed in BFS expansion.
        bfs_expansion_hops: BFS depth (hops) during graph expansion.
        max_context_units: Context window for session boundary LLM calls.
        max_entities_per_llm: Maximum entities in a single LLM dedup call.
        max_events_per_llm: Maximum events in a single LLM dedup call.
        use_unified_pipeline: If True, use UnifiedFactPipeline (recommended).
        incremental_cross_session_coref: Enable incremental coreference resolution.
        coref_vector_threshold: Cosine threshold for coreference candidate retrieval.
        coref_llm_confidence_threshold: LLM confidence floor for coreference merges.
        coref_max_candidates: Maximum coreference candidates per unit.
        coref_simple_concat_threshold: Below this count, use simple concatenation.
        unified_pipeline_top_k_entities: Top-K entities returned by unified pipeline.
        unified_pipeline_top_k_events: Top-K events returned by unified pipeline.
        use_remote_embedder: Use remote embedding API instead of local model.
        use_remote_reranker: Use remote rerank API instead of local model.
        embedder_remote_base_url: Base URL for remote embedding API.
        embedder_remote_api_path: API path for remote embeddings.
        embedder_remote_timeout: Timeout in seconds for remote embedding calls.
        reranker_remote_base_url: Base URL for remote rerank API.
        reranker_remote_api_path: API path for remote reranking.
        reranker_remote_timeout: Timeout in seconds for remote rerank calls.
    """
    embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedder_device: str = "cpu"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_device: str = "cpu"
    llm_model: str = "gpt-4o-mini"
    embedder_dim: int = 384
    promote_threshold: int = 100
    chunk_max_tokens: int = 512
    session_time_gap_seconds: int = 1800
    session_check_interval: int = SESSION_CHECK_INTERVAL
    session_max_pending: int = SESSION_MAX_PENDING
    similarity_top_k: int = 5
    similarity_threshold: float = 0.7
    similarity_recent_window: int = 20
    bfs_expansion_per_seed: int = 3
    bfs_expansion_hops: int = 1
    max_context_units: int = MAX_CONTEXT_UNITS
    max_entities_per_llm: int = MAX_ENTITIES_PER_LLM
    max_events_per_llm: int = MAX_EVENTS_PER_LLM
    use_unified_pipeline: bool = True
    incremental_cross_session_coref: bool = True
    coref_vector_threshold: float = 0.45
    coref_llm_confidence_threshold: float = 0.7
    coref_max_candidates: int = 20
    coref_simple_concat_threshold: int = 2
    unified_pipeline_top_k_entities: int = 10
    unified_pipeline_top_k_events: int = 5
    use_remote_embedder: bool = False
    use_remote_reranker: bool = False
    embedder_remote_base_url: str = ""
    embedder_remote_api_path: str = "/v1/embeddings"
    embedder_remote_timeout: int = 60
    reranker_remote_base_url: str = ""
    reranker_remote_api_path: str = "/v1/rerank"
    reranker_remote_timeout: int = 60


@dataclass
class BuildReport:
    """Outcome report from a build_high_level operation.

    Attributes:
        status: Result status (completed, no_units, error).
        mode: Build mode that was used (auto, force).
        sessions_processed: Number of sessions that were processed.
        units_processed: Total number of memory units processed.
        duration_seconds: Wall-clock duration of the build.
        error_message: Error description when status is 'error'.
        token_usage: Cumulative token consumption during the build.
        warnings: List of non-fatal warning messages collected during build.
    """
    status: str
    mode: str
    sessions_processed: int
    units_processed: int
    duration_seconds: float
    error_message: str = ""
    token_usage: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


class MemorySystem:
    """Main facade for the Mandol memory engine.

    Orchestrates all sub-services: SemanticMapService (storage + ANN search),
    SemanticGraphService (explicit/implicit graph), DocumentChunker,
    SessionManager (LLM-based session detection), UnifiedFactPipeline
    (entity/event extraction), SummaryMapReducer, InsightMapReducer,
    GlobalInsightManager, CrossSessionCorefManager, and MemoryMonitor.

    Supports incremental addition of units with automatic chunking and
    async session boundary detection. Provides holistic multi-group
    retrieval with Dense + BM25 + Sparse three-way recall, RRF fusion,
    BFS graph expansion, and Cross-Encoder reranking.

    Args:
        config: Immutable configuration dataclass (defaults used if None).
        embedder: Custom embedding provider (overrides model-based creation).
        reranker: Custom reranker (overrides model-based creation).
        llm_provider: LLM provider for extraction/reduction/dedup calls.
        storage_root: Directory for automatic persistence.
        enable_persistence: If True, register a PersistenceManager.
        auto_save_interval: Seconds between automatic state saves.

    Typical usage::

        system = MemorySystem(config=MemorySystemConfig(...))
        system.add(unit)
        system.build_high_level()
        results = system.holistic_retrieve("query text")
    """
    _DEFAULT_ROOT = SpaceName("default")
    _naming = SpaceNamingPolicy()

    def __init__(
        self,
        *,
        config: Optional[MemorySystemConfig] = None,
        embedder: Optional[EmbeddingProvider] = None,
        reranker: Optional[Reranker] = None,
        llm_provider: Optional[LLMProvider] = None,
        unit_store=None,
        vector_index=None,
        graph_store=None,
        storage_root: Optional[str] = None,
        enable_persistence: bool = False,
        auto_save_interval: int = 300,
        root: Optional[str] = None,
    ) -> None:
        self._cfg = config or MemorySystemConfig()
        self._root = SpaceName(root) if root else self._DEFAULT_ROOT
        self._dirty = False

        # Allow injection of infrastructure components (factory path).
        # Falls back to defaults for backward compatibility.
        if unit_store is not None:
            store = unit_store
        else:
            store = InMemoryUnitStore()

        if vector_index is not None:
            self._abi = vector_index
        else:
            self._abi = AdaptiveVectorIndex(
                self._cfg.embedder_dim,
                promote_threshold=self._cfg.promote_threshold,
            )

        if graph_store is not None:
            self._graph_store = graph_store
        else:
            self._graph_store = InMemoryGraphStore()

        _embedder = self._create_embedder(embedder)
        _reranker = self._create_reranker(reranker)

        self._semantic_map = SemanticMapService(
            store=store,
            index=self._abi,
            embedder=_embedder,
            reranker=_reranker,
        )
        self._graph = SemanticGraphService(
            semantic_map=self._semantic_map,
            graph_store=self._graph_store,
        )
        if llm_provider is None:
            llm_base_url = os.getenv("MANDOL_LLM_BASE_URL", "")
            llm_api_key = os.getenv("MANDOL_LLM_API_KEY", "")
            llm_kwargs: Dict[str, Any] = {"model": self._cfg.llm_model}
            if llm_base_url:
                llm_kwargs["base_url"] = llm_base_url
            if llm_api_key:
                llm_kwargs["api_key"] = llm_api_key
            llm_provider = OpenAICompatibleLLMProvider(**llm_kwargs)
        self._llm = llm_provider
        self._token_usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self._llm = self._wrap_llm_with_tracking(self._llm)
        self._builder = MultiDimSemanticGraphBuilder(graph=self._graph)
        self._layout_built = False

        self._chunker = DocumentChunker(
            max_tokens=self._cfg.chunk_max_tokens,
            overlap_tokens=0,
        )
        self._session_manager = SessionManager(
            llm_provider=self._llm,
            max_unit_count=20,
        )
        self._summary_reducer = SummaryMapReducer(llm_provider=self._llm)
        self._insight_reducer = InsightMapReducer(llm_provider=self._llm)
        self._global_insight_manager = GlobalInsightManager(llm_provider=self._llm)
        self._entity_dedup = EntityDeduplicator(
            llm_provider=self._llm,
            embedder=embedder,
            similarity_threshold=0.75,
            max_candidates_per_llm=self._cfg.max_entities_per_llm,
        )
        self._event_dedup = EventDeduplicator(
            llm_provider=self._llm,
            similarity_threshold=0.75,
        )
        self._entity_relation_extractor = EntityRelationExtractor(llm_provider=self._llm)
        self._event_causal_extractor = EventCausalExtractor(llm_provider=self._llm)
        self._unified_pipeline = UnifiedFactPipeline(
            llm_provider=self._llm,
            embedding_provider=_embedder,
            semantic_graph=self._graph,
            entity_space=self._naming.knowledge_entity(self._root),
            event_space=self._naming.episodic_event(self._root),
            top_k_entities=self._cfg.unified_pipeline_top_k_entities,
            top_k_events=self._cfg.unified_pipeline_top_k_events,
            on_warning=self._warn,
        )
        self._cross_session_coref_manager = CrossSessionCorefManager(
            llm_provider=self._llm,
            semantic_map=self._semantic_map,
            graph=self._graph,
            naming=self._naming,
            root=self._root,
            vector_threshold=self._cfg.coref_vector_threshold,
            llm_confidence_threshold=self._cfg.coref_llm_confidence_threshold,
            max_candidates=self._cfg.coref_max_candidates,
            simple_concat_threshold=self._cfg.coref_simple_concat_threshold,
            entity_space=self._naming.knowledge_entity(self._root),
            event_space=self._naming.episodic_event(self._root),
            on_warning=self._warn,
        )
        self._unified_pipeline._coref_manager = self._cross_session_coref_manager

        self._executor = ThreadPoolExecutor(max_workers=2)
        self._pending_lock = threading.Lock()
        self._pending_units: List[MemoryUnit] = []
        self._pending_events: List[MemoryUnit] = []
        self._pending_entities: List[MemoryUnit] = []
        self._all_events: List[MemoryUnit] = []
        self._all_entities: List[MemoryUnit] = []
        self._last_session_boundary_changed = False
        self._processed_session_ids: Set[str] = set()
        self._processed_similarity_pairs: Set[Tuple[str, str]] = set()
        self._insertion_order: List[str] = []

        # Async dirty-flag state
        self._async_check_scheduled = False
        self._last_async_reasoning = ""

        # Concurrency guards
        self._build_in_progress = False
        self._auto_save_paused = False

        # Global session ID counter (shared by sync and async paths)
        self._session_counter = 0
        self._session_counter_lock = threading.Lock()

        self._retrieval = MemoryRetrievalService(
            semantic_map=self._semantic_map,
            graph=self._graph,
            naming=self._naming,
            root=self._root,
            config=self._cfg,
        )
        self._p_svc = MemoryPersistenceService(
            semantic_map=self._semantic_map,
            graph_store=self._graph_store,
            naming=self._naming,
            root=self._root,
            config=self._cfg,
            session_manager=self._session_manager,
            abi=self._abi,
        )
        self._p_svc.attach_state(
            insertion_order=self._insertion_order,
            processed_session_ids=self._processed_session_ids,
            processed_similarity_pairs=self._processed_similarity_pairs,
            pending_lock=self._pending_lock,
            pending_units=self._pending_units,
            pending_events=self._pending_events,
            pending_entities=self._pending_entities,
            all_events=self._all_events,
            all_entities=self._all_entities,
            last_async_reasoning=self._last_async_reasoning,
            async_check_scheduled=self._async_check_scheduled,
        )

        self._persistence: Optional["PersistenceManager"] = None
        if enable_persistence and storage_root:
            try:
                from ..infrastructure.persistence_manager import PersistenceManager, MemorySystemStateLoader
                self._persistence = PersistenceManager(
                    storage_root=storage_root,
                    system=self,
                    auto_save_interval=auto_save_interval,
                )
                loader = MemorySystemStateLoader(self._persistence)
                pending_sessions = loader.load_into_system(self)
                if pending_sessions:
                    logger.warning(f"Loaded system with {len(pending_sessions)} pending sessions to rebuild")
                self._persistence.start_auto_save()
            except Exception as e:
                logger.warning(
                    "Failed to initialize persistence: %s. "
                    "Memory state will NOT survive process restart. "
                    "Check that the storage_root directory is writable.",
                    e,
                )
                self._persistence = None

        from ..infrastructure.memory_monitor import MemoryMonitor
        self._monitor = MemoryMonitor(system_ref=self)

        # Per-build warning collector (cleared on each build_high_level call)
        self._build_warnings: List[str] = []

    def _warn(self, message: str) -> None:
        """Record a non-fatal warning for the current build and log it."""
        logger.warning(message)
        self._build_warnings.append(message)

    def _create_embedder(
        self,
        custom_embedder: Optional[EmbeddingProvider],
    ) -> EmbeddingProvider:
        if custom_embedder is not None:
            return custom_embedder

        try:
            if self._cfg.use_remote_embedder:
                from ..infrastructure.openai_compatible_embedding_provider import (
                    OpenAICompatibleEmbeddingConfig,
                    OpenAICompatibleEmbeddingProvider,
                )
                config = OpenAICompatibleEmbeddingConfig(
                    base_url=self._cfg.embedder_remote_base_url,
                    api_path=self._cfg.embedder_remote_api_path,
                    timeout_s=self._cfg.embedder_remote_timeout,
                )
                return OpenAICompatibleEmbeddingProvider(
                    model=self._cfg.embedder_model,
                    dim=self._cfg.embedder_dim,
                    config=config,
                )
            else:
                return SentenceTransformersEmbeddingProvider(
                    model=self._cfg.embedder_model,
                    device=self._cfg.embedder_device,
                )
        except (ImportError, OSError) as e:
            logger.warning(
                "Failed to create embedding provider: %s. "
                "Falling back to StaticEmbeddingProvider (zero embeddings). "
                "Install sentence-transformers for real embeddings.",
                e,
            )
            return StaticEmbeddingProvider(dim=self._cfg.embedder_dim)

    def _create_reranker(
        self,
        custom_reranker: Optional[Reranker],
    ) -> Optional[Reranker]:
        if custom_reranker is not None:
            return custom_reranker

        try:
            if self._cfg.use_remote_reranker:
                from ..infrastructure.openai_compatible_reranker import (
                    OpenAICompatibleRerankConfig,
                    OpenAICompatibleReranker,
                )
                config = OpenAICompatibleRerankConfig(
                    base_url=self._cfg.reranker_remote_base_url,
                    api_path=self._cfg.reranker_remote_api_path,
                    timeout_s=self._cfg.reranker_remote_timeout,
                )
                return OpenAICompatibleReranker(
                    model=self._cfg.reranker_model,
                    config=config,
                )
            else:
                return SentenceTransformersCrossEncoderReranker(
                    model=self._cfg.reranker_model,
                    device=self._cfg.reranker_device,
                )
        except (ImportError, OSError) as e:
            logger.warning(
                "Failed to create reranker: %s. Reranking will be skipped.",
                e,
            )
            return None

    @classmethod
    def from_yaml_config(
        cls,
        yaml_path: str,
        *,
        embedder: Optional[EmbeddingProvider] = None,
        reranker: Optional[Reranker] = None,
        llm_provider: Optional[LLMProvider] = None,
        root: Optional[str] = None,
    ) -> "MemorySystem":
        """Create MemorySystem from YAML config file.

        Delegates to SystemFactory in the infrastructure layer to keep
        the application layer free of concrete infrastructure imports.

        Args:
            yaml_path: Path to config.yaml file.
            embedder: Optional custom embedder (overrides config).
            reranker: Optional custom reranker (overrides config).
            llm_provider: Optional custom LLM provider (overrides config).
            root: Optional root space name.
        """
        from ..infrastructure.system_factory import SystemFactory
        return SystemFactory.create_from_yaml(
            yaml_path,
            embedder=embedder,
            reranker=reranker,
            llm_provider=llm_provider,
            root=root,
        )

    @property
    def semantic_map(self) -> SemanticMapService:
        return self._semantic_map

    @property
    def graph(self) -> SemanticGraphService:
        return self._graph

    @property
    def llm(self) -> LLMProvider:
        return self._llm

    @property
    def dirty(self) -> bool:
        return self._dirty

    def _wrap_llm_with_tracking(self, provider: LLMProvider) -> LLMProvider:
        original_chat = provider.chat

        def tracked_chat(*args: Any, **kwargs: Any) -> LLMChatResponse:
            response = original_chat(*args, **kwargs)
            self._track_llm_usage(response)
            return response

        provider.chat = tracked_chat  # type: ignore[assignment]
        return provider

    def _track_llm_usage(self, response: LLMChatResponse) -> None:
        for key in self._token_usage:
            self._token_usage[key] += response.usage.get(key, 0)

    def get_token_usage(self) -> Dict[str, int]:
        return dict(self._token_usage)

    def reset_token_usage(self) -> None:
        self._token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def _ensure_layout(self) -> None:
        if self._layout_built:
            return
        self._builder.ensure_layout(self._root)
        self._layout_built = True

    def _get_last_unit(self) -> Optional[MemoryUnit]:
        base_space_name = self._naming.base_memory(self._root)
        units = self._semantic_map.get_units_in_spaces([base_space_name])
        if not units:
            return None
        if self._insertion_order:
            for uid_str in reversed(self._insertion_order):
                for u in units:
                    if str(u.uid) == uid_str:
                        return u
        sorted_units = sorted(
            units,
            key=lambda u: u.metadata.get("timestamp", ""),
        )
        return sorted_units[-1] if sorted_units else None

    def _ensure_session_space(self, session: Session) -> MemorySpace:
        """Create or get a session-namespaced memory space.

        Registers the session space as a child of the base memory space and
        assigns all units in the session to this space.

        Args:
            session: The Session object containing the units.

        Returns:
            The created or existing session MemorySpace.
        """
        session_space_name = SpaceName(f"{self._root}_session_{session.session_id}")
        
        # Check if space already exists
        existing = self._semantic_map.get_space(session_space_name)
        if existing is not None:
            return existing
        
        # Create session space
        session_space = self._semantic_map.create_space(session_space_name)
        
        # Register as child of base_memory
        base_space = self._naming.base_memory(self._root)
        self._semantic_map.attach_child_space(base_space, session_space_name, ensure_exists=True)
        
        # Register all session units to this space
        for unit_uid in session.unit_uids:
            self._semantic_map.add_unit_to_space(unit_uid, session_space_name)
        
        logger.info(f"Created session space {session_space_name} with {len(session.unit_uids)} units")
        return session_space

    def _process_session_high_level_memory(
        self,
        session: Session,
        session_units: List[MemoryUnit],
        session_space: MemorySpace,
    ) -> None:
        """Run all high-level memory builders for a completed session.

        Processes summaries (episodic/knowledge/emotional/procedural),
        insights (merged into global insight store), and entity/event
        extraction (unified pipeline or legacy path).

        Args:
            session: The Session object.
            session_units: The MemoryUnits belonging to this session.
            session_space: The session's MemorySpace.
        """
        logger.info(
            "Building high-level memory for session %s (%d units)...",
            session.session_id, len(session_units),
        )

        # Process summaries
        t_summary_start = time.time()
        summaries = self._summary_reducer.process_session(session, session_units)
        logger.info(
            "Session %s: summaries generated in %.1fs (%d categories)",
            session.session_id, time.time() - t_summary_start, len(summaries),
        )

        summary_space_map = {
            "episodic": self._naming.episodic_summary(self._root),
            "knowledge": self._naming.knowledge_summary(self._root),
            "emotional": self._naming.emotional(self._root),
            "procedural": self._naming.procedural(self._root),
        }
        for cat, summary_units_list in summaries.items():
            target_space = summary_space_map.get(cat, self._naming.episodic_summary(self._root))
            for summary_unit in summary_units_list:
                self._semantic_map.add_unit(
                    summary_unit,
                    space_names=[target_space, session_space.name],
                    ensure_embedding=True,
                )
                for src_unit in session_units:
                    self._graph.add_relationship(
                        source_uid=str(summary_unit.uid),
                        target_uid=str(src_unit.uid),
                        relationship_name=EVIDENCED_BY,
                        score=1.0,
                    )

        # Process insights (as intermediate data, merge immediately to global)
        if summaries:
            insights = self._insight_reducer.process_session(session, summaries)
            if insights:
                self._global_insight_manager.merge_and_update(
                    session=session,
                    session_insights=insights,
                    semantic_map=self._semantic_map,
                    graph=self._graph,
                    naming=self._naming,
                    root_space=self._root,
                )

        if self._cfg.use_unified_pipeline:
            t_pipeline_start = time.time()
            self._process_session_with_unified_pipeline(session, session_units, session_space)
            logger.info(
                "Session %s: unified pipeline completed in %.1fs",
                session.session_id, time.time() - t_pipeline_start,
            )
        else:
            self._process_entities_for_session(session, session_units, session_space)
            self._process_events_for_session(session, session_units, session_space)

        # Post-condition: warn if summary produced no results for a non-trivial session
        if not summaries and len(session_units) >= 3:
            self._warn(
                f"[FALLBACK] Summary generation produced NO summaries for session "
                f"{session.session_id} with {len(session_units)} units. "
                f"LLM may be failing."
            )

    def _process_session_with_unified_pipeline(
        self,
        session: Session,
        session_units: List[MemoryUnit],
        session_space: MemorySpace,
    ) -> None:
        """Extract entities and events using the unified fact pipeline.

        Args:
            session: The Session object.
            session_units: The MemoryUnits in this session.
            session_space: The session's MemorySpace.
        """
        logger.info(
            "Session %s: running unified entity/event extraction on %d dialogue units...",
            session.session_id, len(session_units),
        )
        result = self._unified_pipeline.process_session(
            dialogue_units=session_units,
            session_id=session.session_id,
        )
        logger.info(
            "Session %s: extracted %d entities, %d events, %d relations, %d causal",
            session.session_id,
            len(result.entities), len(result.events),
            len(result.entity_relations), len(result.causal_relations),
        )

        if self._cfg.incremental_cross_session_coref:
            self._cross_session_coref_manager.merge_and_write(
                session=session,
                session_units=session_units,
                session_space=session_space,
                pipeline_result=result,
            )
        else:
            self._write_pipeline_result_directly(
                result, session_space, session_units, session.session_id
            )

    def _write_pipeline_result_directly(
        self,
        result,
        session_space: MemorySpace,
        session_units: List[MemoryUnit],
        session_id: str,
    ) -> None:
        """Write pipeline results directly to semantic_map and graph (legacy mode).

        Args:
            result: The pipeline result object.
            session_space: The session's MemorySpace.
            session_units: The session's units for evidenced-by edges.
            session_id: Session identifier for UID generation.
        """
        entity_space = self._naming.knowledge_entity(self._root)
        event_space = self._naming.episodic_event(self._root)

        if result.entities and isinstance(result.entities[0], ExtractedEntity):
            entity_units, coref_edges_e, evidenced_by_edges_e = self._unified_pipeline._create_entity_units(
                result.entities, session_units, session_id
            )
            event_units, coref_edges_ev, evidenced_by_edges_ev, involves_edges = self._unified_pipeline._create_event_units(
                result.events, session_units, session_id, entity_units
            )

            related_to_edges = self._unified_pipeline._create_related_to_edges(
                result.entity_relations
            )
            causes_edges = self._unified_pipeline._create_causes_edges(
                result.causal_relations
            )

            result.coref_edges = coref_edges_e + coref_edges_ev
            result.evidenced_by_edges = evidenced_by_edges_e + evidenced_by_edges_ev
            result.involves_edges = involves_edges
            result.related_to_edges = related_to_edges
            result.causes_edges = causes_edges
        else:
            entity_units = result.entities
            event_units = result.events

        for entity_unit in entity_units:
            self._semantic_map.add_unit(
                entity_unit,
                space_names=[entity_space, session_space.name],
                ensure_embedding=True,
            )

        for event_unit in event_units:
            self._semantic_map.add_unit(
                event_unit,
                space_names=[event_space, session_space.name],
                ensure_embedding=True,
            )

        self._unified_pipeline.write_edges_to_graph(result)

        with self._pending_lock:
            self._pending_entities.extend(entity_units)
            self._pending_events.extend(event_units)
            self._all_entities.extend(entity_units)
            self._all_events.extend(event_units)

    def _process_entities_for_session(
        self,
        session: Session,
        session_units: List[MemoryUnit],
        session_space: MemorySpace,
    ) -> None:
        """Extract, deduplicate, and store entities for a single session.

        Extracts entities from unit raw_data, deduplicates them, stores in
        the entity space, links them to source units via EVIDENCED_BY edges,
        and extracts entity-entity relationships.

        Args:
            session: The Session object.
            session_units: The MemoryUnits in this session.
            session_space: The session's MemorySpace.
        """
        entity_units = []
        for unit in session_units:
            extracted = unit.raw_data.get("entities", [])
            if isinstance(extracted, list):
                for i, ent_text in enumerate(extracted):
                    if isinstance(ent_text, str) and ent_text.strip():
                        entity_units.append(MemoryUnit(
                            uid=Uid(f"{session.session_id}:entity:{i}"),
                            raw_data={"text_content": ent_text.strip()},
                            metadata={
                                "type": "entity",
                                "session_id": session.session_id,
                            },
                        ))

        if not entity_units:
            return

        deduplicated = self._entity_dedup.deduplicate(entity_units)
        entity_space = self._naming.knowledge_entity(self._root)
        for entity_unit in deduplicated:
            self._semantic_map.add_unit(
                entity_unit,
                space_names=[entity_space, session_space.name],
                ensure_embedding=True,
            )
            for src_unit in session_units:
                self._graph.add_relationship(
                    source_uid=str(entity_unit.uid),
                    target_uid=str(src_unit.uid),
                    relationship_name=EVIDENCED_BY,
                    score=1.0,
                )

        relations = self._entity_relation_extractor.extract_relations(
            deduplicated, session.session_id
        )
        entity_text_to_uid = {e.raw_data.get("text_content", ""): str(e.uid) for e in deduplicated}
        for rel in relations:
            head_uid = entity_text_to_uid.get(rel.head_entity)
            tail_uid = entity_text_to_uid.get(rel.tail_entity)
            if head_uid and tail_uid:
                self._graph.add_relationship(
                    source_uid=head_uid,
                    target_uid=tail_uid,
                    relationship_name=f"REL_{rel.relation_type.upper()}",
                    score=rel.confidence,
                )

        with self._pending_lock:
            self._pending_entities.extend(deduplicated)
            self._all_entities.extend(deduplicated)

    def _process_events_for_session(
        self,
        session: Session,
        session_units: List[MemoryUnit],
        session_space: MemorySpace,
    ) -> None:
        """Extract, deduplicate, and store events for a single session.

        Extracts events from unit raw_data, deduplicates them, stores in the
        event space, links them to source units, and extracts causal event
        relationships (CAUSES / CAUSED_BY).

        Args:
            session: The Session object.
            session_units: The MemoryUnits in this session.
            session_space: The session's MemorySpace.
        """
        event_units = []
        for unit in session_units:
            extracted = unit.raw_data.get("events", [])
            if isinstance(extracted, list):
                for i, evt_text in enumerate(extracted):
                    if isinstance(evt_text, str) and evt_text.strip():
                        event_units.append(MemoryUnit(
                            uid=Uid(f"{session.session_id}:event:{i}"),
                            raw_data={"text_content": evt_text.strip()},
                            metadata={
                                "type": "event",
                                "session_id": session.session_id,
                                "timestamp": unit.metadata.get("timestamp", ""),
                            },
                        ))

        if not event_units:
            return

        deduplicated = self._event_dedup.deduplicate(event_units)
        event_space = self._naming.episodic_event(self._root)
        for event_unit in deduplicated:
            self._semantic_map.add_unit(
                event_unit,
                space_names=[event_space, session_space.name],
                ensure_embedding=True,
            )
            for src_unit in session_units:
                self._graph.add_relationship(
                    source_uid=str(event_unit.uid),
                    target_uid=str(src_unit.uid),
                    relationship_name=EVIDENCED_BY,
                    score=1.0,
                )

        causal_relations = self._event_causal_extractor.extract_causal_relations(
            deduplicated, session.session_id
        )
        event_text_to_uid = {e.raw_data.get("text_content", ""): str(e.uid) for e in deduplicated}
        for causal in causal_relations:
            cause_uid = event_text_to_uid.get(causal.cause_event)
            effect_uid = event_text_to_uid.get(causal.effect_event)
            if cause_uid and effect_uid:
                self._graph.add_relationship(
                    source_uid=cause_uid,
                    target_uid=effect_uid,
                    relationship_name="CAUSES",
                    score=causal.confidence,
                )
                self._graph.add_relationship(
                    source_uid=effect_uid,
                    target_uid=cause_uid,
                    relationship_name="CAUSED_BY",
                    score=causal.confidence,
                )

        with self._pending_lock:
            self._pending_events.extend(deduplicated)
            self._all_events.extend(deduplicated)

    # ------------------------------------------------------------------
    # Async real-time path: add / add_many (dirty-flag pattern)
    # ------------------------------------------------------------------

    def add(self, unit: MemoryUnit) -> None:
        """Add a single memory unit to the system.

        Writes the unit to base space, optionally chunks it, builds immediate
        similarity edges, then enqueues it for async session boundary checking.
        The LLM call happens in a background executor worker — add() returns
        immediately without blocking on LLM.

        Args:
            unit: The MemoryUnit to add.
        """
        self._ensure_layout()
        base_space = self._naming.base_memory(self._root)
        unit.metadata.setdefault("timestamp", unit.metadata.get("_system_created_at"))
        unit.metadata.setdefault("spaces", [base_space])

        self._insertion_order.append(str(unit.uid))

        added_units: List[MemoryUnit] = []
        if self._chunker.should_chunk(unit):
            chunk_result = self._chunker.chunk_unit(unit)

            for chunk_unit in chunk_result.chunks:
                self._semantic_map.add_unit(
                    chunk_unit,
                    space_names=[base_space],
                    ensure_embedding=True,
                )
                added_units.append(chunk_unit)
                self._insertion_order.append(str(chunk_unit.uid))

            self._dirty = True
            with self._pending_lock:
                self._pending_units.extend(chunk_result.chunks)
            logger.info(f"Chunked unit {unit.uid} into {len(chunk_result.chunks)} chunks")
        else:
            self._semantic_map.add_unit(
                unit,
                space_names=[base_space],
                ensure_embedding=True,
            )
            self._dirty = True
            with self._pending_lock:
                self._pending_units.append(unit)
            added_units.append(unit)

        self._build_immediate_similarity_edges(added_units)
        self._schedule_async_check()

    def add_many(self, units: Sequence[MemoryUnit]) -> None:
        """Add multiple memory units in one call.

        Each unit is individually chunked and enqueued. The dirty-flag
        mechanism naturally aggregates LLM calls: if all units arrive within
        one LLM response window, only one check fires; if they arrive slowly,
        each triggers independently.

        Args:
            units: Sequence of MemoryUnit objects to add.
        """
        self._ensure_layout()
        base_space = self._naming.base_memory(self._root)
        for unit in units:
            unit.metadata.setdefault("timestamp", unit.metadata.get("_system_created_at"))
            unit.metadata.setdefault("spaces", [base_space])

        added_units: List[MemoryUnit] = []
        for unit in units:
            if self._chunker.should_chunk(unit):
                chunk_result = self._chunker.chunk_unit(unit)

                for chunk_unit in chunk_result.chunks:
                    self._semantic_map.add_unit(
                        chunk_unit,
                        space_names=[base_space],
                        ensure_embedding=True,
                    )
                    added_units.append(chunk_unit)
                    self._insertion_order.append(str(chunk_unit.uid))
                with self._pending_lock:
                    self._pending_units.extend(chunk_result.chunks)
            else:
                self._semantic_map.add_unit(
                    unit,
                    space_names=[base_space],
                    ensure_embedding=True,
                )
                with self._pending_lock:
                    self._pending_units.append(unit)
                self._insertion_order.append(str(unit.uid))
                added_units.append(unit)

        self._dirty = True

        if added_units:
            self._build_immediate_similarity_edges(added_units)

        if units:
            self._schedule_async_check()

    def _schedule_async_check(self) -> None:
        """Dirty-flag: ensure at most one _do_async_check runs at a time.

        If _build_in_progress is True (sync build is running), units are
        accumulated in _pending_units but no async task is scheduled —
        build_high_level will process them when it finishes.
        """
        if self._build_in_progress:
            return

        with self._pending_lock:
            if self._async_check_scheduled:
                return
            if len(self._pending_units) < 2:
                return
            self._async_check_scheduled = True
            self._executor.submit(self._do_async_check)

    def _do_async_check(self) -> None:
        """Execute one round of async session boundary detection.

        Runs in an executor worker. Takes a sliding window of up to 20 units
        from the tail of _pending_units, calls analyze_batch (V2 unified
        prompt), processes split points, and self-checks whether more units
        arrived during the LLM call.
        """
        with self._pending_lock:
            pending_count_before = len(self._pending_units)
            window = list(self._pending_units[-min(MAX_CONTEXT_UNITS, pending_count_before):])

        # Format content lines for analyze_batch
        content_lines = self._format_units_for_analysis(window)
        current_session_id = self._next_session_id()

        decision = self._session_manager.analyze_batch(
            content_lines,
            current_session_id,
            previous_reasoning=self._last_async_reasoning,
            on_warning=self._warn,
        )

        self._last_async_reasoning = decision.reasoning

        with self._pending_lock:
            if decision.should_split and decision.split_points:
                # Process splits from rightmost to leftmost so earlier
                # split indices remain valid after each mutation.
                for split_point in sorted(decision.split_points, key=lambda sp: sp.split_at_index, reverse=True):
                    # Map window-relative split_at_index back to _pending_units
                    actual_split = len(self._pending_units) - len(window) + split_point.split_at_index
                    if actual_split <= 0 or actual_split >= len(self._pending_units):
                        continue

                    front = self._pending_units[:actual_split]
                    back = self._pending_units[actual_split:]
                    self._pending_units = back
                    self._executor.submit(self._build_session_for_units, list(front))

            # 100-unit upper bound protection
            if len(self._pending_units) >= SESSION_MAX_PENDING:
                all_units = list(self._pending_units)
                self._pending_units.clear()
                self._executor.submit(self._build_session_for_units, all_units)
                logger.warning(
                    "Force flushing %d pending units due to max pending limit",
                    len(all_units),
                )

            pending_count_after = len(self._pending_units)

            # Only re-submit if new units arrived during the LLM call
            # (i.e. the pending count grew). This prevents infinite loops
            # when the LLM consistently returns no-split decisions.
            if pending_count_after >= 2 and pending_count_after > pending_count_before:
                self._executor.submit(self._do_async_check)
            else:
                self._async_check_scheduled = False

    def _format_units_for_analysis(self, units: List[MemoryUnit]) -> List[str]:
        """Format a list of MemoryUnits into index-prefixed content lines.

        Args:
            units: The units to format.

        Returns:
            List of strings like "[1] 2024-01-01T00:00:00: hello world".
        """
        lines: List[str] = []
        for i, u in enumerate(units):
            text = u.raw_data.get("text_content", "") if isinstance(u.raw_data, dict) else ""
            timestamp = u.metadata.get("timestamp", "")
            lines.append(f"[{i + 1}] {timestamp}: {text}")
        return lines

    def _next_session_id(self) -> str:
        """Generate a unique session ID (thread-safe).

        Shared by both sync and async paths.
        """
        with self._session_counter_lock:
            sid = f"sess_{datetime.now(timezone.utc).strftime('%Y%m%d')}_{self._session_counter:03d}"
            self._session_counter += 1
            return sid

    def _build_session_for_units(self, units: List[MemoryUnit]) -> None:
        """Build a session and high-level memory for the given units.

        Used by both the async path (_do_async_check) and the sync path
        (build_high_level). Thread-safe via the global session ID counter.

        Args:
            units: The MemoryUnits that comprise the new session.
        """
        if not units:
            return

        sorted_units = sorted(
            units,
            key=lambda u: u.metadata.get("timestamp", ""),
        )

        session_id = self._next_session_id()
        topic = "Auto-merged Session"

        if sorted_units:
            first_text = sorted_units[0].raw_data.get("text_content", "") if isinstance(sorted_units[0].raw_data, dict) else ""
            if first_text:
                topic = first_text[:50] + "..." if len(first_text) > 50 else first_text

        start_time = sorted_units[0].metadata.get("timestamp", "") if sorted_units else ""
        end_time = sorted_units[-1].metadata.get("timestamp", "") if sorted_units else ""

        unit_uids = [str(u.uid) for u in sorted_units]
        session = self._session_manager.build_session(
            session_id=session_id,
            unit_uids=unit_uids,
            topic=topic,
            start_time=start_time,
            end_time=end_time,
        )

        # Create session space and register all units
        session_space = self._ensure_session_space(session)

        # Process high-level memory (summary, insight, entity, event)
        self._process_session_high_level_memory(session, sorted_units, session_space)

        self._processed_session_ids.add(session.session_id)
        logger.info(f"Built high-level memory for session {session.session_id}")

        self._build_similarity_edges_for_units(sorted_units)

    def build_high_level(self, mode: str = "auto") -> BuildReport:
        """Build high-level memory structures from accumulated base units.

        Runs session detection (LLM-based V2 unified prompt), summary/insight
        generation, entity/event extraction, and cross-session coreference
        resolution.

        Args:
            mode: Build mode — \"auto\" processes new (unassigned) units only,
                \"force\" rebuilds everything from scratch via export-rebuild.

        Returns:
            A BuildReport summarizing the operation outcome.
        """
        start_time = datetime.now(timezone.utc)
        self._auto_save_paused = True
        self._build_warnings.clear()

        try:
            self._ensure_layout()
            base_space_name = self._naming.base_memory(self._root)
            all_base_units = self._semantic_map.get_units_in_spaces([base_space_name])

            if not all_base_units:
                return BuildReport(
                    status="no_units",
                    mode=mode,
                    sessions_processed=0,
                    units_processed=0,
                    duration_seconds=0.0,
                    token_usage=self.get_token_usage(),
                    warnings=list(self._build_warnings),
                )

            if mode == "force":
                return self._build_high_level_force(all_base_units, start_time)
            else:
                return self._build_high_level_auto(all_base_units, start_time)

        except Exception as e:
            logger.error(
                "build_high_level failed: %s", e,
                exc_info=True,
            )
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            return BuildReport(
                status="error",
                mode=mode,
                sessions_processed=0,
                units_processed=0,
                duration_seconds=duration,
                error_message=str(e),
                token_usage=self.get_token_usage(),
                warnings=list(self._build_warnings),
            )
        finally:
            self._auto_save_paused = False

    # ------------------------------------------------------------------
    # Sync auto mode
    # ------------------------------------------------------------------

    def _build_high_level_auto(
        self,
        all_base_units: List[MemoryUnit],
        start_time: datetime,
    ) -> BuildReport:
        """Auto mode: process only units not yet assigned to any session."""
        # Step 1: Wait for all in-flight async tasks to complete
        self._executor.submit(lambda: None).result()

        # Step 2: Drain remaining pending units, reset async state
        with self._pending_lock:
            remaining = list(self._pending_units)
            self._pending_units.clear()
            self._async_check_scheduled = False
            self._last_async_reasoning = ""

        # Step 3: Activate build guard
        self._build_in_progress = True
        try:
            # Step 4: Build assigned_uids set from existing sessions
            assigned_uids: Set[str] = set()
            for session in self._session_manager.get_sessions():
                for uid in session.unit_uids:
                    assigned_uids.add(str(uid))

            # Step 5: Filter to unassigned ("bare") units
            bare_units = [u for u in all_base_units if str(u.uid) not in assigned_uids]

            # Step 6: Merge remaining pending (dedup)
            bare_uids = {str(u.uid) for u in bare_units}
            for u in remaining:
                if str(u.uid) not in bare_uids:
                    bare_units.append(u)
                    bare_uids.add(str(u.uid))

            if not bare_units:
                return BuildReport(
                    status="no_units",
                    mode="auto",
                    sessions_processed=0,
                    units_processed=0,
                    duration_seconds=(datetime.now(timezone.utc) - start_time).total_seconds(),
                    token_usage=self.get_token_usage(),
                    warnings=list(self._build_warnings),
                )

            # Step 7: Sort and batch
            sorted_units = sorted(bare_units, key=lambda u: u.metadata.get("timestamp", ""))

            return self._run_sync_session_detection(
                sorted_units, all_base_units, start_time, "auto"
            )
        finally:
            self._build_in_progress = False

    # ------------------------------------------------------------------
    # Sync force mode: export-rebuild strategy
    # ------------------------------------------------------------------

    def _build_high_level_force(
        self,
        all_base_units: List[MemoryUnit],
        start_time: datetime,
    ) -> BuildReport:
        """Force mode: cancel async tasks, rebuild store/index/graph from scratch."""
        # Step 1: Cancel all async tasks
        try:
            self._executor.shutdown(wait=True, cancel_futures=False)
        except (RuntimeError, OSError):
            pass
        self._executor = ThreadPoolExecutor(max_workers=2)

        with self._pending_lock:
            self._pending_units.clear()
            self._pending_events.clear()
            self._pending_entities.clear()
            self._async_check_scheduled = False
            self._last_async_reasoning = ""

        # Step 2: Export base units and temporal edges
        base_units = [u for u in all_base_units]

        temporal_edges: List[Tuple[str, str, str, Dict[str, Any]]] = []
        try:
            for source, target, rel_type, props in self._graph_store.get_all_edges():
                if rel_type in ("IS_BEFORE", "IS_AFTER"):
                    temporal_edges.append((str(source), str(target), str(rel_type), dict(props)))
        except (KeyError, AttributeError, RuntimeError) as e:
            # Graph store iteration may fail if edges are malformed or the
            # underlying store type doesn't support iteration.
            logger.warning("Failed to extract temporal edges: %s", e)

        # Step 3: Rebuild store, index, graph
        new_store = InMemoryUnitStore()
        self._abi = AdaptiveVectorIndex(
            self._cfg.embedder_dim,
            promote_threshold=self._cfg.promote_threshold,
        )
        self._graph_store = InMemoryGraphStore()

        self._semantic_map = SemanticMapService(
            store=new_store,
            index=self._abi,
            embedder=self._semantic_map.get_embedder(),
            reranker=self._semantic_map.get_reranker(),
        )
        self._graph = SemanticGraphService(
            semantic_map=self._semantic_map,
            graph_store=self._graph_store,
        )

        # Re-import base units
        for unit in base_units:
            new_store.upsert_units([unit])

        # Re-import temporal edges
        for source, target, rel_type, props in temporal_edges:
            try:
                self._graph_store.upsert_relationship(
                    Uid(str(source)), Uid(str(target)), str(rel_type), props
                )
            except (KeyError, ValueError, TypeError) as e:
                logger.warning("Failed to restore temporal edge %s->%s: %s", source, target, e)

        # Rebuild vector index
        items: List[Tuple[Uid, np.ndarray]] = []
        for u in base_units:
            if u.embedding is not None:
                uid = Uid(str(u.uid))
                emb = np.asarray(u.embedding, dtype=np.float32).reshape(-1)
                if emb.shape[0] == self._abi.dim():
                    items.append((uid, emb))
        self._abi.rebuild(items)

        # Step 4: Reset all derived state
        self._session_manager.reset()
        self._processed_session_ids.clear()
        self._processed_similarity_pairs.clear()
        self._global_insight_manager = GlobalInsightManager(llm_provider=self._llm)
        self._cross_session_coref_manager = CrossSessionCorefManager(
            llm_provider=self._llm,
            semantic_map=self._semantic_map,
            graph=self._graph,
            naming=self._naming,
            root=self._root,
            vector_threshold=self._cfg.coref_vector_threshold,
            llm_confidence_threshold=self._cfg.coref_llm_confidence_threshold,
            max_candidates=self._cfg.coref_max_candidates,
            simple_concat_threshold=self._cfg.coref_simple_concat_threshold,
            entity_space=self._naming.knowledge_entity(self._root),
            event_space=self._naming.episodic_event(self._root),
            on_warning=self._warn,
        )

        # Update service references
        self._retrieval = MemoryRetrievalService(
            semantic_map=self._semantic_map,
            graph=self._graph,
            naming=self._naming,
            root=self._root,
            config=self._cfg,
        )
        self._p_svc = MemoryPersistenceService(
            semantic_map=self._semantic_map,
            graph_store=self._graph_store,
            naming=self._naming,
            root=self._root,
            config=self._cfg,
            session_manager=self._session_manager,
            abi=self._abi,
        )
        self._p_svc.attach_state(
            insertion_order=self._insertion_order,
            processed_session_ids=self._processed_session_ids,
            processed_similarity_pairs=self._processed_similarity_pairs,
            pending_lock=self._pending_lock,
            pending_units=self._pending_units,
            pending_events=self._pending_events,
            pending_entities=self._pending_entities,
            all_events=self._all_events,
            all_entities=self._all_entities,
            last_async_reasoning=self._last_async_reasoning,
            async_check_scheduled=self._async_check_scheduled,
        )

        self._build_in_progress = True
        try:
            sorted_units = sorted(base_units, key=lambda u: u.metadata.get("timestamp", ""))
            return self._run_sync_session_detection(
                sorted_units, base_units, start_time, "force"
            )
        finally:
            self._build_in_progress = False

    # ------------------------------------------------------------------
    # Shared sync session detection loop
    # ------------------------------------------------------------------

    def _run_sync_session_detection(
        self,
        sorted_units: List[MemoryUnit],
        all_units_lookup: List[MemoryUnit],
        start_time: datetime,
        mode: str,
    ) -> BuildReport:
        """Run the LLM-based session detection loop over sorted units.

        Uses a sliding context window (last 15 units + current batch of 20)
        to prevent unbounded prompt growth. When the LLM detects a topic
        boundary, the session is split at that point. When no boundary is
        detected across many batches, units accumulate in the background but
        only the sliding window is sent to the LLM.

        On the last batch, any remaining units are force-closed as a session.
        """
        # Batch into groups of 20
        batches: List[List[MemoryUnit]] = []
        for i in range(0, len(sorted_units), SESSION_CHECK_INTERVAL):
            batches.append(sorted_units[i:i + SESSION_CHECK_INTERVAL])

        logger.info(
            "Starting sync session detection: %d units in %d batches (mode=%s)",
            len(sorted_units), len(batches), mode,
        )

        # Max context units carried over from previous batch for LLM context.
        # The LLM sees at most CONTEXT_CARRYOVER + SESSION_CHECK_INTERVAL ≈ 35 lines.
        CONTEXT_CARRYOVER = 15

        current_session_units: List[MemoryUnit] = []
        previous_reasoning = ""
        sessions_processed = 0
        total_units_processed = 0

        for batch_idx, batch in enumerate(batches):
            is_last = (batch_idx == len(batches) - 1)

            # Build sliding window: context carryover from previous batch + new batch.
            # This caps the LLM prompt at ~35 lines regardless of how many units
            # have accumulated in current_session_units.
            if len(current_session_units) > len(batch):
                context_carryover = current_session_units[-min(CONTEXT_CARRYOVER, len(current_session_units) - len(batch)):]
            else:
                context_carryover = []
            window_for_llm = context_carryover + list(batch)
            # Offset to map LLM-relative split indices back to current_session_units
            llm_offset = len(current_session_units) - len(window_for_llm)

            current_session_units.extend(batch)

            content_lines = self._format_units_for_analysis(window_for_llm)
            current_session_id = self._next_session_id()

            if batch_idx == 0:
                previous_reasoning = ""  # No prior context for the first batch

            decision = self._session_manager.analyze_batch(
                content_lines,
                current_session_id,
                previous_reasoning=previous_reasoning,
                on_warning=self._warn,
            )

            previous_reasoning = decision.reasoning

            # Process split points rightmost-first so earlier indices stay valid
            if decision.should_split and decision.split_points:
                remaining = list(current_session_units)
                for split_point in sorted(decision.split_points, key=lambda sp: sp.split_at_index, reverse=True):
                    # LLM reports index relative to window_for_llm — map to current_session_units
                    raw_split_idx = split_point.split_at_index
                    split_idx = llm_offset + raw_split_idx
                    if split_idx <= 0 or split_idx >= len(remaining):
                        logger.debug(
                            "Skipping split at mapped_idx=%d (raw=%d, offset=%d, remaining=%d)",
                            split_idx, raw_split_idx, llm_offset, len(remaining),
                        )
                        continue

                    session_units = remaining[:split_idx]
                    remaining = remaining[split_idx:]

                    if session_units:
                        logger.info(
                            "Batch %d/%d: split detected at LLM-idx=%d → building session with %d units "
                            "(window=%d units, offset=%d)",
                            batch_idx + 1, len(batches), raw_split_idx,
                            len(session_units), len(window_for_llm), llm_offset,
                        )
                        self._build_session_for_units(session_units)
                        sessions_processed += 1
                        total_units_processed += len(session_units)

                current_session_units = remaining
                # Clear previous_reasoning after a split since we start a fresh session
                previous_reasoning = ""

            # On last batch, ignore should_wait and close the session
            if is_last and current_session_units:
                if decision.should_wait:
                    logger.info(
                        "Last batch should_wait=True ignored, closing session with %d units",
                        len(current_session_units),
                    )
                self._build_session_for_units(current_session_units)
                sessions_processed += 1
                total_units_processed += len(current_session_units)
                current_session_units = []

        # Any leftover (shouldn't happen, but be safe)
        if current_session_units:
            self._build_session_for_units(current_session_units)
            sessions_processed += 1
            total_units_processed += len(current_session_units)

        self._dirty = False
        duration = (datetime.now(timezone.utc) - start_time).total_seconds()

        return BuildReport(
            status="completed",
            mode=mode,
            sessions_processed=sessions_processed,
            units_processed=total_units_processed,
            duration_seconds=duration,
            token_usage=self.get_token_usage(),
            warnings=list(self._build_warnings),
        )

    def build_high_level_async(self, mode: str = "auto") -> Future[Dict[str, Any]]:
        """Submit an async build_high_level to the thread pool.

        Args:
            mode: Build mode (\"auto\" or \"force\").

        Returns:
            A concurrent.futures.Future resolving to a dict with build status.
        """
        return self._executor.submit(self._do_build_high_level_async, mode)

    def _do_build_high_level_async(self, mode: str) -> Dict[str, Any]:
        return self.build_high_level(mode=mode)

    def merge_cross_session_entities(self) -> None:
        """Deduplicate entities across all sessions.

        If the unified pipeline is active, delegates to the pipeline's
        internal cross-session merge; otherwise uses the standalone
        EntityDeduplicator.
        """
        if self._cfg.use_unified_pipeline:
            merged = self._unified_pipeline.merge_cross_session_entities()
            logger.info(f"Cross-session entity merge (unified pipeline): {len(merged)} unique entities")
            return

        if len(self._all_entities) < 2:
            return

        deduplicated = self._entity_dedup.deduplicate(self._all_entities)
        if len(deduplicated) < len(self._all_entities):
            entity_space = self._naming.knowledge_entity(self._root)
            existing_uids = set()
            for u in self._semantic_map.get_units_in_spaces([entity_space]):
                existing_uids.add(str(u.uid))

            for entity_unit in deduplicated:
                uid_str = str(entity_unit.uid)
                if uid_str not in existing_uids:
                    self._semantic_map.add_unit(
                        entity_unit,
                        space_names=[entity_space],
                        ensure_embedding=True,
                    )

            self._all_entities = deduplicated
            logger.info(f"Cross-session entity merge: {len(self._all_entities)} unique entities")

    def merge_cross_session_events(self) -> None:
        """Deduplicate events across all sessions.

        If the unified pipeline is active, delegates to the pipeline's
        internal cross-session merge; otherwise uses the standalone
        EventDeduplicator.
        """
        if self._cfg.use_unified_pipeline:
            merged = self._unified_pipeline.merge_cross_session_events()
            logger.info(f"Cross-session event merge (unified pipeline): {len(merged)} unique events")
            return

        if len(self._all_events) < 2:
            return

        deduplicated = self._event_dedup.deduplicate(self._all_events)
        if len(deduplicated) < len(self._all_events):
            event_space = self._naming.episodic_event(self._root)
            existing_uids = set()
            for u in self._semantic_map.get_units_in_spaces([event_space]):
                existing_uids.add(str(u.uid))

            for event_unit in deduplicated:
                uid_str = str(event_unit.uid)
                if uid_str not in existing_uids:
                    self._semantic_map.add_unit(
                        event_unit,
                        space_names=[event_space],
                        ensure_embedding=True,
                    )

            self._all_events = deduplicated
            logger.info(f"Cross-session event merge: {len(self._all_events)} unique events")

    def _get_retrieval_groups(self):
        return self._retrieval._get_retrieval_groups()

    def holistic_retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        use_rerank: bool = True,
        auto_build_if_empty: bool = True,
        skip_views: Optional[List[str]] = None,
    ) -> List[SearchHit]:
        """Unified multi-group memory retrieval across all memory spaces.

        Internal pipeline:
        1. If auto_build_if_empty is True and high-level memory is empty,
           triggers build_high_level("auto").
        2. Builds 4 retrieval groups: BASE / ENTITY / EVENT / SUMMARY.
        3. Each group performs independent three-way recall (Dense + BM25 +
           Sparse), RRF fusion, and BFS graph expansion.
        4. All candidates are merged.
        5. Cross-Encoder Reranker performs global reranking.

        Args:
            query: The search query text.
            top_k: Number of results to return.
            use_rerank: Whether to apply reranking (default True).
            auto_build_if_empty: Whether to auto-build high-level memory when
                empty (default True).
            skip_views: Optional list of retrieval group keys to skip.
                Valid keys: "base", "entity", "event", "summary".

        Returns:
            List of SearchHit results ordered by relevance.
        """
        return self._retrieval.holistic_retrieve(
            query,
            top_k=top_k,
            use_rerank=use_rerank,
            auto_build_if_empty=auto_build_if_empty,
            build_trigger=lambda: self.build_high_level("auto"),
            skip_views=skip_views,
        )

    def retrieve_in_space(
        self,
        query: str,
        space_name: str,
        *,
        top_k: int = 10,
        use_rerank: bool = True,
    ) -> List[SearchHit]:
        """Run the full retrieval pipeline within a specific memory space.

        Args:
            query: The search query text.
            space_name: Target space name (e.g. \"root_knowledge_entity\").
            top_k: Number of results to return.
            use_rerank: Whether to apply reranking.

        Returns:
            List of SearchHit results ordered by relevance.
        """
        return self._retrieval.retrieve_in_space(
            query, space_name, top_k=top_k, use_rerank=use_rerank
        )

    def retrieve_by_view(
        self,
        query: str,
        view: str,
        *,
        top_k: int = 10,
        use_rerank: bool = True,
    ) -> List[SearchHit]:
        """Retrieve by a named memory category (view).

        Args:
            query: The search query text.
            view: Category name. Valid values:
                - \"base_memory\": Raw conversational memory.
                - \"entity_relation\": Entity relationship graph.
                - \"event_causal\": Event causal chain.
                - \"emotional\": Emotional summaries.
                - \"episodic\": Episodic summaries.
                - \"knowledge\": Knowledge summaries.
                - \"procedural\": Procedural summaries.
                - \"insights\": Global insights.
            top_k: Number of results to return.
            use_rerank: Whether to apply reranking.

        Returns:
            List of SearchHit results ordered by relevance.
        """
        return self._retrieval.retrieve_by_view(
            query, view, top_k=top_k, use_rerank=use_rerank
        )

    search = holistic_retrieve

    _DEFAULT_ASK_SYSTEM_PROMPT = (
        "You are an intelligent assistant powered by a memory system. "
        "Answer the user's question based on the retrieved memory content below. "
        "If the retrieved results contain no relevant information, say so honestly. "
        "Do not fabricate information.\n\n"
        "Retrieved results:\n{context}"
    )

    def ask(
        self,
        query: str,
        *,
        top_k: int = 5,
        use_rerank: bool = True,
        auto_build_if_empty: bool = True,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
    ) -> str:
        """End-to-end RAG: retrieve with holistic_retrieve, then generate an answer via LLM.

        Args:
            query: The user's natural language question.
            top_k: Number of retrieval results to feed into the LLM.
            use_rerank: Whether to apply reranking during retrieval.
            auto_build_if_empty: Whether to auto-build high-level memory when empty.
            system_prompt: Custom system prompt template. Use ``{context}`` as a
                placeholder for the formatted retrieval results. If *None*, a
                built-in default prompt is used.
            temperature: LLM sampling temperature.
            max_tokens: Maximum tokens in the LLM response.

        Returns:
            The LLM-generated natural language answer as a string.
        """
        hits = self.holistic_retrieve(
            query,
            top_k=top_k,
            use_rerank=use_rerank,
            auto_build_if_empty=auto_build_if_empty,
        )
        return self.ask_with_hits(
            query,
            hits,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def ask_with_hits(
        self,
        query: str,
        hits: List[SearchHit],
        *,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate a natural language answer from pre-retrieved SearchHit results.

        Use this when you want to control the retrieval step yourself (e.g. calling
        ``retrieve_in_space`` or ``retrieve_by_view`` instead of
        ``holistic_retrieve``) and only need the LLM generation step.

        Args:
            query: The user's natural language question.
            hits: Pre-retrieved SearchHit list to use as context.
            system_prompt: Custom system prompt template. Use ``{context}`` as a
                placeholder for the formatted retrieval results. If *None*, a
                built-in default prompt is used.
            temperature: LLM sampling temperature.
            max_tokens: Maximum tokens in the LLM response.

        Returns:
            The LLM-generated natural language answer as a string.
        """
        context_parts: list[str] = []
        for i, hit in enumerate(hits, 1):
            text = hit.unit.raw_data.get("text_content", "")
            context_parts.append(f"[{i}] (score: {hit.final_score:.3f}) {text}")
        context = "\n".join(context_parts)

        prompt_template = system_prompt or self._DEFAULT_ASK_SYSTEM_PROMPT
        rendered_system = prompt_template.format(context=context)

        combined = rendered_system + "\n\nUser question: " + query
        messages: list[ChatMessage] = [
            {"role": "user", "content": combined},
        ]

        response = self._llm.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.content

    def _search_group(self, query, space_names, top_k, use_rerank):
        return self._retrieval._search_group(query, space_names, top_k, use_rerank)

    def _build_immediate_similarity_edges(self, new_units: List[MemoryUnit]) -> None:
        if not new_units:
            return

        embedder = self._semantic_map.get_embedder()
        if embedder is None:
            return

        new_units_with_emb = [u for u in new_units if u.embedding is not None]
        if not new_units_with_emb:
            return

        base_space = self._naming.base_memory(self._root)
        all_base_units = self._semantic_map.get_units_in_spaces([base_space])

        recent_units = sorted(
            all_base_units,
            key=lambda u: u.metadata.get("timestamp", ""),
        )[-self._cfg.similarity_recent_window:]

        existing_units = [u for u in recent_units if u not in new_units and u.embedding is not None]

        for new_unit in new_units_with_emb:
            if new_unit.embedding is None:
                continue

            for existing in existing_units:
                if existing.embedding is None:
                    continue

                pair_key = tuple(sorted([str(new_unit.uid), str(existing.uid)]))
                if pair_key in self._processed_similarity_pairs:
                    continue

                score = self._compute_cosine_similarity(new_unit.embedding, existing.embedding)

                if score >= self._cfg.similarity_threshold:
                    try:
                        self._graph.add_relationship(
                            source_uid=str(new_unit.uid),
                            target_uid=str(existing.uid),
                            relationship_name=SEMANTIC_SIMILAR,
                            score=score,
                        )
                        self._processed_similarity_pairs.add(pair_key)
                    except KeyError:
                        # Source or target unit no longer in the semantic map.
                        logger.debug(
                            "Immediate similarity edge skipped — UID not found: %s -> %s",
                            new_unit.uid, existing.uid,
                        )

    def _build_similarity_edges_for_units(self, units: List[MemoryUnit]) -> None:
        if not units:
            return

        embedder = self._semantic_map.get_embedder()
        if embedder is None:
            return

        units_with_emb = [u for u in units if u.embedding is not None]
        if not units_with_emb:
            return

        base_space = self._naming.base_memory(self._root)
        all_base_units = self._semantic_map.get_units_in_spaces([base_space])

        existing_units = [u for u in all_base_units if u not in units and u.embedding is not None]

        for new_unit in units_with_emb:
            if new_unit.embedding is None:
                continue

            for existing in existing_units:
                if existing.embedding is None:
                    continue

                pair_key = tuple(sorted([str(new_unit.uid), str(existing.uid)]))
                if pair_key in self._processed_similarity_pairs:
                    continue

                score = self._compute_cosine_similarity(new_unit.embedding, existing.embedding)

                if score >= self._cfg.similarity_threshold:
                    try:
                        self._graph.add_relationship(
                            source_uid=str(new_unit.uid),
                            target_uid=str(existing.uid),
                            relationship_name=SEMANTIC_SIMILAR,
                            score=score,
                        )
                        self._processed_similarity_pairs.add(pair_key)
                    except KeyError:
                        logger.debug(
                            "Cross-session similarity edge skipped — UID not found: %s -> %s",
                            new_unit.uid, existing.uid,
                        )

    def flush(self) -> None:
        """Persist all stores and clear pending state."""
        self._semantic_map.flush()
        self._graph_store.flush()
        with self._pending_lock:
            self._pending_units.clear()
            self._pending_events.clear()
            self._pending_entities.clear()
            self._all_events.clear()
            self._all_entities.clear()
            self._async_check_scheduled = False
            self._last_async_reasoning = ""
        self._processed_session_ids.clear()
        self._processed_similarity_pairs.clear()
        self._insertion_order.clear()
        self._dirty = False

    def save(self, storage_path: Optional[str] = None) -> "SaveResult":
        """Save system state to disk.

        Delegates to PersistenceManager if it was enabled, otherwise requires
        an explicit storage_path.

        Args:
            storage_path: Directory path for JSON-based persistence.

        Returns:
            A SaveResult describing the save outcome.

        Raises:
            ValueError: If persistence is not enabled and no storage_path given.
        """
        if storage_path is not None:
            return self._p_svc._save_to_path(storage_path)

        if self._persistence is not None:
            return self._persistence.save_full()

        raise ValueError(
            "storage_path is required when persistence is not enabled. "
            "Call save('/path/to/dir') to save to a specific directory."
        )

    def _save_to_path(self, storage_path: str) -> "SaveResult":
        return self._p_svc._save_to_path(storage_path)

    def _load(self, storage_path: str) -> LoadResult:
        result = self._p_svc._load(storage_path)
        self._layout_built = True
        return result

    @classmethod
    def load(
        cls,
        storage_path: str,
        *,
        embedder: Optional[EmbeddingProvider] = None,
        reranker: Optional[Reranker] = None,
        llm_provider: Optional[LLMProvider] = None,
    ) -> "MemorySystem":
        """Reconstruct a MemorySystem from a previously saved state.

        Loads the config snapshot from disk, instantiates a new
        MemorySystem with the saved (or overridden) providers, then
        restores all units, spaces, edges, sessions, and processed
        state.

        Args:
            storage_path: Directory path containing the saved JSON state.
            embedder: Optional EmbeddingProvider override; defaults to the
                provider specified in the saved config.
            reranker: Optional Reranker override.
            llm_provider: Optional LLMProvider override.

        Returns:
            A fully restored MemorySystem instance.
        """
        from ..infrastructure.json_persistence import JsonPersistenceEngine

        _load_env_file(storage_path)

        engine = JsonPersistenceEngine(storage_path)
        config_data = engine.load_config()

        config = MemorySystemConfig()
        root = cls._DEFAULT_ROOT

        if config_data is not None:
            cfg_dict = config_data.get("memory_system_config", {})
            if cfg_dict:
                try:
                    config = MemorySystemConfig(**cfg_dict)
                except TypeError:
                    logger.warning("Saved config is incompatible, using defaults")
            root = SpaceName(config_data.get("root", cls._DEFAULT_ROOT))

        system = cls(
            config=config,
            embedder=embedder,
            reranker=reranker,
            llm_provider=llm_provider,
            root=str(root),
        )
        system._load(storage_path)
        return system

    def _extract_graph_edges(self) -> List[Tuple[str, str, str, Dict[str, Any]]]:
        return self._p_svc._extract_graph_edges()

    def _get_pending_session_ids(self) -> Set[str]:
        return self._p_svc._get_pending_session_ids()

    def _reset_state(self) -> None:
        self._p_svc._reset_state()

    @staticmethod
    def _compute_cosine_similarity(
        emb_a: Embedding,
        emb_b: Embedding,
    ) -> float:
        a = np.asarray(emb_a, dtype=np.float32).reshape(-1)
        b = np.asarray(emb_b, dtype=np.float32).reshape(-1)
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        return float(np.dot(a / norm_a, b / norm_b))

    def _rebuild_vector_index(self, units: List[MemoryUnit]) -> None:
        self._p_svc._rebuild_vector_index(units)

    @property
    def persistence(self) -> Optional["PersistenceManager"]:
        return self._persistence

    @property
    def monitor(self) -> "MemoryMonitor":
        """Access the system memory monitor.

        Returns compact single-line status::

            print(system.monitor.status_line())

        Get dictionary-format stats::

            stats = system.monitor.to_dict()
        """
        return self._monitor
