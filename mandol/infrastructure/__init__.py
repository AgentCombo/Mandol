"""Infrastructure layer — concrete implementations of all port interfaces.

Provides in-memory, FAISS, Milvus, Neo4j, Sentence-Transformers, and
OpenAI-compatible provider implementations for embedders, rerankers,
LLMs, unit stores, graph stores, vector indexes, BM25/sparse indexes,
persistence engines, and the provider factory wiring.

Exports:
    AdaptiveVectorIndex, InMemoryCosineVectorIndex, InMemoryGraphStore,
    InMemoryUnitStore, FAISSVectorIndex, MilvusUnitStore (optional),
    Neo4jGraphStore (optional), OpenAICompatibleLLMProvider,
    ProviderFactoryResult, build_providers_from_config,
    RankBM25Index, TfidfSparseIndex, SentenceTransformersEmbeddingProvider,
    SentenceTransformersCrossEncoderReranker, StubLLMProvider,
    OpenAICompatibleEmbeddingProvider, OpenAICompatibleReranker,
    JsonPersistenceEngine, PersistenceError, SaveResult, VerificationResult,
    IndexRebuilder, PersistenceManager, MemorySystemStateLoader, MemoryMonitor.
"""

from .in_memory_graph_store import InMemoryGraphStore
from .in_memory_unit_store import InMemoryUnitStore
from .in_memory_vector_index import InMemoryCosineVectorIndex
from .faiss_vector_index import FaissVectorIndex
from .openai_compatible_llm_provider import OpenAICompatibleLLMProvider
from .provider_factory import ProviderFactoryResult, build_providers_from_config
from .stub_llm_provider import StubLLMProvider
from .sentence_transformers_embedding_provider import SentenceTransformersEmbeddingProvider
from .sentence_transformers_reranker import SentenceTransformersCrossEncoderReranker
from .openai_compatible_embedding_provider import OpenAICompatibleEmbeddingProvider, UniApiEmbeddingProvider
from .openai_compatible_reranker import OpenAICompatibleReranker, UniApiReranker
from .rank_bm25_index import RankBM25Index
from .tfidf_sparse_index import TfidfSparseIndex
from .adaptive_vector_index import AdaptiveVectorIndex
from .json_persistence import JsonPersistenceEngine, PersistenceError, SaveResult, VerificationResult, IndexRebuilder
from .persistence_manager import PersistenceManager, MemorySystemStateLoader
from .memory_monitor import MemoryMonitor

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
    "InMemoryCosineVectorIndex",
    "InMemoryGraphStore",
    "InMemoryUnitStore",
    "FaissVectorIndex",
    "MilvusUnitStore",
    "Neo4jGraphStore",
    "OpenAICompatibleLLMProvider",
    "ProviderFactoryResult",
    "RankBM25Index",
    "SentenceTransformersCrossEncoderReranker",
    "SentenceTransformersEmbeddingProvider",
    "StubLLMProvider",
    "OpenAICompatibleEmbeddingProvider",
    "OpenAICompatibleReranker",
    "TfidfSparseIndex",
    "UniApiEmbeddingProvider",
    "UniApiReranker",
    "build_providers_from_config",
    "JsonPersistenceEngine",
    "PersistenceError",
    "SaveResult",
    "VerificationResult",
    "IndexRebuilder",
    "PersistenceManager",
    "MemorySystemStateLoader",
    "MemoryMonitor",
]
