"""Infrastructure layer — concrete implementations of all port interfaces.

Provides in-memory, FAISS, Milvus, Neo4j, Sentence-Transformers, and
OpenAI-compatible provider implementations for embedders, rerankers,
LLMs, unit stores, graph stores, vector indexes, BM25/sparse indexes,
persistence engines, and the provider factory wiring.

Exports:
    AdaptiveVectorIndex, InMemoryCosineVectorIndex, InMemoryGraphStore,
    InMemoryUnitStore, DuckDBUnifiedStore (optional), FAISSVectorIndex,
    MilvusUnitStore (optional),
    Neo4jGraphStore (optional), OpenAICompatibleLLMProvider,
    ProviderFactoryResult, build_providers_from_config,
    RankBM25Index, TfidfSparseIndex, SentenceTransformersEmbeddingProvider,
    SentenceTransformersCrossEncoderReranker, StubLLMProvider,
    OpenAICompatibleEmbeddingProvider, OpenAICompatibleReranker,
    JsonPersistenceEngine, PersistenceError, SaveResult, VerificationResult,
    IndexRebuilder, PersistenceManager, MemorySystemStateLoader, MemoryMonitor.
"""

from .adaptive_vector_index import AdaptiveVectorIndex
from .duckdb_unified_store import DuckDBUnifiedQueryStore, DuckDBUnifiedStore
from .faiss_hnsw_vector_index import FaissHNSWVectorIndex
from .faiss_vector_index import FaissVectorIndex
from .in_memory_graph_store import InMemoryGraphStore
from .in_memory_unified_query_store import InMemoryUnifiedQueryStore
from .in_memory_unit_store import InMemoryUnitStore
from .in_memory_vector_index import InMemoryCosineVectorIndex
from .json_persistence import (
    IndexRebuilder,
    JsonPersistenceEngine,
    PersistenceError,
    SaveResult,
    VerificationResult,
)
from .memory_monitor import MemoryMonitor
from .openai_compatible_embedding_provider import (
    OpenAICompatibleEmbeddingProvider,
    UniApiEmbeddingProvider,
)
from .openai_compatible_llm_provider import OpenAICompatibleLLMProvider
from .openai_compatible_reranker import OpenAICompatibleReranker, UniApiReranker
from .persistence_manager import MemorySystemStateLoader, PersistenceManager
from .provider_factory import ProviderFactoryResult, build_providers_from_config
from .rank_bm25_index import RankBM25Index
from .sentence_transformers_embedding_provider import SentenceTransformersEmbeddingProvider
from .sentence_transformers_reranker import SentenceTransformersCrossEncoderReranker
from .stub_llm_provider import StubLLMProvider
from .tfidf_sparse_index import TfidfSparseIndex

try:
    from .milvus_unit_store import MilvusUnitStore
except ImportError:
    MilvusUnitStore = None  # type: ignore

try:
    from .neo4j_graph_store import Neo4jGraphStore
except ImportError:
    Neo4jGraphStore = None  # type: ignore

__all__ = [
    "AdaptiveVectorIndex",
    "DuckDBUnifiedQueryStore",
    "DuckDBUnifiedStore",
    "FaissHNSWVectorIndex",
    "FaissVectorIndex",
    "InMemoryCosineVectorIndex",
    "InMemoryGraphStore",
    "InMemoryUnifiedQueryStore",
    "InMemoryUnitStore",
    "IndexRebuilder",
    "JsonPersistenceEngine",
    "MemoryMonitor",
    "MemorySystemStateLoader",
    "MilvusUnitStore",
    "Neo4jGraphStore",
    "OpenAICompatibleEmbeddingProvider",
    "OpenAICompatibleLLMProvider",
    "OpenAICompatibleReranker",
    "PersistenceError",
    "PersistenceManager",
    "ProviderFactoryResult",
    "RankBM25Index",
    "SaveResult",
    "SentenceTransformersCrossEncoderReranker",
    "SentenceTransformersEmbeddingProvider",
    "StubLLMProvider",
    "TfidfSparseIndex",
    "UniApiEmbeddingProvider",
    "UniApiReranker",
    "VerificationResult",
    "build_providers_from_config",
]
