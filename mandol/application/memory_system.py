"""MemorySystem — the main orchestration facade for the Mandol memory engine.

Coordinates all application-layer services including semantic mapping, graph
building, document chunking, session detection, entity/event extraction, insight
reduction, and multi-modal retrieval. Provides a unified interface for adding
memory units, building high-level memory, saving/loading state, and performing
holistic or per-group retrieval with reranking.
"""

from __future__ import annotations

import json
import logging
import threading
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
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
from ..infrastructure.config import MemorySystemYamlConfig
from ..ports.embedding_provider import EmbeddingProvider, StaticEmbeddingProvider
from ..ports.llm_provider import LLMProvider, ChatMessage
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

# Relationship type label for semantically similar units (cosine similarity).
SEMANTIC_SIMILAR = "SEMANTIC_SIMILAR"
# Relationship type label connecting high-level units to their source data.
EVIDENCED_BY = "EVIDENCED_BY"

# Maximum number of context units to consider when checking session boundaries.
MAX_CONTEXT_UNITS = 20
# Upper bound on entities sent to the LLM in a single call.
MAX_ENTITIES_PER_LLM = 50
# Upper bound on events sent to the LLM in a single call.
MAX_EVENTS_PER_LLM = 50
# Number of accumulated units that triggers a session boundary check.
SESSION_CHECK_INTERVAL = 20
# If pending units exceed this threshold, force a session build.
SESSION_MAX_PENDING = 100


# Prompt template used by the LLM to check whether a batch of units
# contains a conversational session boundary.
SESSION_BOUNDARY_WITH_INDEX_PROMPT = """You are a conversation session segmentation expert. Your task is to identify if there's a clear topic boundary where a NEW session should start.

**CORE PRINCIPLE: When uncertain, MERGE (set split_at_index to -1). Only split when you see an EXPLICIT topic boundary.**

## Conversation Context
Memory Fragments with timestamps:
{units_text}

## What to Look For

**EXPLICIT TOPIC BOUNDARIES (warrant split):**
- Complete topic change: Conversation shifts to something completely unrelated
- Cross-day boundaries: Date is clearly different from previous messages
- Major event transitions: New chapter starts (e.g., after "We finished X", "Now let's move to Y")

**DO NOT SPLIT (merge these into current session):**
- Short acknowledgments: "ok", "yes", "lol", "sure", "thanks", "got it"
- Task completion flows: "I'll start" -> "Done!" -> "Great!" should be ONE session
- Continuous discussions on the same topic, even with pauses
- Greetings ("hi", "hello") and farewells ("bye", "goodbye") that are part of the conversation
- System-like messages that don't change the topic

## Time Information
Timestamps are shown above. Use them as REFERENCE, not as mandatory split points.
- A time gap alone is NOT sufficient reason to split
- If two topics are discussed within a time gap, they might still belong to the same session
- Only use time as supporting evidence for topic changes

## Output Format (JSON only):
{{
  "should_split": boolean,
  "split_at_index": integer (0-{max_index}, split BEFORE units[split_at_index]; only valid when should_split=true),
  "reason": "brief explanation of why this is/isn't a topic boundary"
}}

**IMPORTANT**:
- If should_split=false, set split_at_index to -1
- split_at_index=0 means the NEW session starts at the very first unit
- Prefer FALSE (merge) when uncertain"""


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
        session_time_gap_seconds: Time gap hint for session detection.
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
    """
    status: str
    mode: str
    sessions_processed: int
    units_processed: int
    duration_seconds: float
    error_message: str = ""


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
        storage_root: Optional[str] = None,
        enable_persistence: bool = False,
        auto_save_interval: int = 300,
    ) -> None:
        self._cfg = config or MemorySystemConfig()
        self._root = self._DEFAULT_ROOT
        self._dirty = False

        store = InMemoryUnitStore()
        self._abi = AdaptiveVectorIndex(
            self._cfg.embedder_dim,
            promote_threshold=self._cfg.promote_threshold,
        )
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
        self._llm = llm_provider or OpenAICompatibleLLMProvider(model=self._cfg.llm_model)
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
                logger.warning(f"Failed to initialize persistence: {e}")
                self._persistence = None

        from ..infrastructure.memory_monitor import MemoryMonitor
        self._monitor = MemoryMonitor(system_ref=self)

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
    ) -> "MemorySystem":
        """Create MemorySystem from YAML config file.
        
        Args:
            yaml_path: Path to config.yaml file
            embedder: Optional custom embedder (overrides config)
            reranker: Optional custom reranker (overrides config)
            llm_provider: Optional custom LLM provider (overrides config)
        """
        yaml_config = MemorySystemYamlConfig.load_from_yaml(yaml_path)

        # Build internal config
        config = MemorySystemConfig(
            embedder_model=yaml_config.embedder.model,
            embedder_device=yaml_config.embedder.device,
            reranker_model=yaml_config.reranker.model,
            reranker_device=yaml_config.reranker.device,
            llm_model=yaml_config.llm.model,
            embedder_dim=yaml_config.embedder.dimension,
            promote_threshold=yaml_config.promote_threshold,
            chunk_max_tokens=yaml_config.chunk_max_tokens,
            session_time_gap_seconds=yaml_config.session_time_gap_seconds,
            session_check_interval=yaml_config.session_check_interval,
            session_max_pending=yaml_config.session_max_pending,
            similarity_top_k=yaml_config.similarity_top_k,
            similarity_threshold=yaml_config.similarity_threshold,
            similarity_recent_window=yaml_config.similarity_recent_window,
            bfs_expansion_per_seed=yaml_config.bfs_expansion_per_seed,
            bfs_expansion_hops=yaml_config.bfs_expansion_hops,
            max_context_units=yaml_config.max_context_units,
            max_entities_per_llm=yaml_config.max_entities_per_llm,
            max_events_per_llm=yaml_config.max_events_per_llm,
            use_remote_embedder=yaml_config.embedder.use_remote,
            use_remote_reranker=yaml_config.reranker.use_remote,
            embedder_remote_base_url=yaml_config.embedder.base_url,
            embedder_remote_api_path=yaml_config.embedder.api_path,
            embedder_remote_timeout=yaml_config.embedder.timeout,
            reranker_remote_base_url=yaml_config.reranker.base_url,
            reranker_remote_api_path=yaml_config.reranker.api_path,
            reranker_remote_timeout=yaml_config.reranker.timeout,
        )

        # Create LLM provider if not provided
        if llm_provider is None and yaml_config.llm.api_key:
            llm_provider = OpenAICompatibleLLMProvider(
                model=yaml_config.llm.model,
                base_url=yaml_config.llm.base_url,
                api_key=yaml_config.llm.api_key,
            )

        return cls(
            config=config,
            embedder=embedder,
            reranker=reranker,
            llm_provider=llm_provider,
            storage_root=yaml_config.storage_root,
            enable_persistence=yaml_config.enable_persistence,
            auto_save_interval=yaml_config.auto_save_interval,
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
        # Process summaries
        summaries = self._summary_reducer.process_session(session, session_units)

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
            self._process_session_with_unified_pipeline(session, session_units, session_space)
        else:
            self._process_entities_for_session(session, session_units, session_space)
            self._process_events_for_session(session, session_units, session_space)

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
        result = self._unified_pipeline.process_session(
            dialogue_units=session_units,
            session_id=session.session_id,
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

    def add(self, unit: MemoryUnit) -> None:
        """Add a single memory unit to the system.

        The unit is assigned to the base memory space, optionally chunked if
        it exceeds the token limit, and indexed for vector search. Triggers
        async session boundary check if enough units have accumulated.

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
        self._check_session_boundary_async()

    def add_many(self, units: Sequence[MemoryUnit]) -> None:
        """Add multiple memory units in one call.

        Each unit is individually chunked if necessary and indexed. A single
        async session boundary check is triggered at the end.

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
            self._check_session_boundary_async()

    def _check_session_boundary_async(self) -> None:
        with self._pending_lock:
            pending_count = len(self._pending_units)
            if pending_count < self._cfg.session_check_interval:
                return
            if pending_count >= self._cfg.session_max_pending:
                context_units = self._pending_units
            else:
                context_units = self._pending_units[-MAX_CONTEXT_UNITS:] if pending_count > MAX_CONTEXT_UNITS else self._pending_units[:]

        should_split, split_at_index = self._should_start_new_session_for_batch(context_units)

        with self._pending_lock:
            if should_split and split_at_index >= 0:
                actual_split_point = len(self._pending_units) - len(context_units) + split_at_index
                if actual_split_point > 0:
                    session_units = self._pending_units[:actual_split_point]
                    remaining_units = self._pending_units[actual_split_point:]
                    self._pending_units = remaining_units
                    self._executor.submit(self._build_session_for_units, session_units)
                    logger.info(f"Session boundary detected: splitting at index {actual_split_point}, submitting {len(session_units)} units for async build")
            else:
                if len(self._pending_units) >= self._cfg.session_max_pending:
                    session_units = self._pending_units[:]
                    self._pending_units = []
                    self._executor.submit(self._build_session_for_units, session_units)
                    logger.warning(f"Force flushing {len(session_units)} pending units due to max pending limit")

    def _should_start_new_session_for_batch(self, context_units: List[MemoryUnit]) -> Tuple[bool, int]:
        if len(context_units) < 2:
            return False, -1

        units_text = []
        for i, u in enumerate(context_units):
            raw = u.raw_data if isinstance(u.raw_data, dict) else {}
            text = raw.get("text_content", "")
            timestamp = raw.get("session_datetime", "")
            speaker = raw.get("speaker", "")
            dia_id = raw.get("dia_id", "")

            if text:
                if timestamp:
                    header = f"[{i}] [{timestamp}] {speaker} ({dia_id})"
                else:
                    header = f"[{i}] {speaker} ({dia_id})"
                text_preview = text[:300] + "..." if len(text) > 300 else text
                units_text.append(f"{header}: {text_preview}")

        if len(units_text) < 2:
            return False, -1

        max_index = len(units_text) - 1
        prompt_content = SESSION_BOUNDARY_WITH_INDEX_PROMPT.format(
            units_text="\n---\n".join(units_text),
            max_index=max_index,
            max_index_plus_one=len(units_text),
        )

        messages: List[ChatMessage] = [
            {"role": "system", "content": "You are a session segmentation expert. Output JSON only."},
            {"role": "user", "content": prompt_content},
        ]

        try:
            response = self._llm.chat(messages, temperature=0.1, max_tokens=512)
            data = json.loads(response.content)
            should_split = bool(data.get("should_split", False))
            split_at_index = data.get("split_at_index", -1)
            if split_at_index is None:
                split_at_index = -1
            try:
                split_at_index = int(split_at_index)
            except (ValueError, TypeError):
                split_at_index = -1
            logger.info(f"LLM session boundary check: should_split={should_split}, split_at_index={split_at_index}, reason={data.get('reason', '')}")
            return should_split, split_at_index
        except Exception as e:
            logger.warning(f"LLM session boundary check failed: {e}")
            return False, -1

    def _build_session_for_units(self, units: List[MemoryUnit]) -> None:
        """Async session builder for units split at a detected boundary.

        Creates a session describing these units, links them together,
        and builds high-level memory for the session.

        Args:
            units: The MemoryUnits that comprise the new session.
        """
        if not units:
            return

        sorted_units = sorted(
            units,
            key=lambda u: u.metadata.get("timestamp", ""),
        )

        session_id = f"sess_{datetime.now(timezone.utc).strftime('%Y%m%d')}_{len(self._session_manager._sessions):03d}"
        topic = "Auto-merged Session"

        if sorted_units:
            first_text = sorted_units[0].raw_data.get("text_content", "") if isinstance(sorted_units[0].raw_data, dict) else ""
            if first_text:
                topic = first_text[:50] + "..." if len(first_text) > 50 else first_text

        session = Session(
            session_id=session_id,
            unit_uids=[u.uid for u in sorted_units],
            topic=topic,
            start_time=datetime.now(timezone.utc).isoformat(),
            end_time=datetime.now(timezone.utc).isoformat(),
        )

        # Create session space and register all units
        session_space = self._ensure_session_space(session)

        # Register session in session manager
        self._session_manager._sessions.append(session)

        # Process high-level memory (summary, insight, entity, event)
        self._process_session_high_level_memory(session, sorted_units, session_space)

        self._processed_session_ids.add(session.session_id)
        logger.info(f"Built high-level memory for session {session.session_id}")

        self._build_similarity_edges_for_units(sorted_units)

    def build_high_level(self, mode: str = "auto") -> BuildReport:
        """Build high-level memory structures from accumulated base units.

        Runs session detection, summary/insight generation, entity/event
        extraction, and cross-session coreference resolution.

        Args:
            mode: Build mode — \"auto\" processes new units only, \"force\"
                reprocesses everything from scratch.

        Returns:
            A BuildReport summarizing the operation outcome.
        """
        start_time = datetime.now(timezone.utc)

        try:
            self._ensure_layout()
            base_space_name = self._naming.base_memory(self._root)
            all_units = self._semantic_map.get_units_in_spaces([base_space_name])

            if not all_units:
                return BuildReport(
                    status="no_units",
                    mode=mode,
                    sessions_processed=0,
                    units_processed=0,
                    duration_seconds=0.0,
                )

            sorted_units = sorted(
                all_units,
                key=lambda u: u.metadata.get("timestamp", ""),
            )

            if mode == "force":
                try:
                    self._executor.shutdown(wait=True, cancel_futures=False)
                except Exception:
                    pass
                self._executor = ThreadPoolExecutor(max_workers=2)

                self._processed_session_ids.clear()
                self._session_manager.reset()
                self._dirty = False
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
                )

                all_units_for_splitting = sorted_units
            else:
                pending_copy = list(self._pending_units)
                self._pending_units.clear()
                all_units_for_splitting = sorted_units + pending_copy

            sessions = self._session_manager.split_sessions(all_units_for_splitting)

            sessions_processed = 0
            total_units_processed = 0
            for session in sessions:
                if session.session_id in self._processed_session_ids:
                    continue

                unit_map = {str(u.uid): u for u in sorted_units}
                session_units = [
                    unit_map[str(uid)] for uid in session.unit_uids
                    if str(uid) in unit_map
                ]

                if not session_units:
                    continue

                # Create session space and register all units
                session_space = self._ensure_session_space(session)

                self._process_session_high_level_memory(session, session_units, session_space)

                self._processed_session_ids.add(session.session_id)
                sessions_processed += 1
                total_units_processed += len(session_units)

                self._build_similarity_edges_for_units(session_units)

            self._dirty = False
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()

            return BuildReport(
                status="completed",
                mode=mode,
                sessions_processed=sessions_processed,
                units_processed=total_units_processed,
                duration_seconds=duration,
            )
        except Exception as e:
            logger.error(f"build_high_level failed: {e}\n{traceback.format_exc()}")
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            return BuildReport(
                status="error",
                mode=mode,
                sessions_processed=0,
                units_processed=0,
                duration_seconds=duration,
                error_message=str(e),
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
    ) -> List[SearchHit]:
        """Unified multi-group memory retrieval across all memory spaces.

        Internal pipeline:
        1. If auto_build_if_empty is True and high-level memory is empty,
           triggers build_high_level(\"auto\").
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

        Returns:
            List of SearchHit results ordered by relevance.
        """
        return self._retrieval.holistic_retrieve(
            query,
            top_k=top_k,
            use_rerank=use_rerank,
            auto_build_if_empty=auto_build_if_empty,
            build_trigger=lambda: self.build_high_level("auto"),
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
        "你是一个基于记忆系统的智能助手。根据以下检索到的记忆内容回答用户的问题。"
        "如果检索结果中没有相关信息，请如实说明。不要编造信息。\n\n"
        "检索结果：\n{context}"
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

        messages: list[ChatMessage] = [
            {"role": "system", "content": rendered_system},
            {"role": "user", "content": query},
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
                    except Exception as e:
                        logger.debug(f"Could not add immediate similarity edge: {e}")

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
                    except Exception as e:
                        logger.debug(f"Could not add cross-session similarity edge: {e}")

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
        )

        system._root = root
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
