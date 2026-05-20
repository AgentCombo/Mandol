"""Port (abstraction layer) interfaces for the Mandol memory system.

Defines the abstract contracts (ABCs) that all infrastructure
implementations must satisfy:
  - VectorIndex: dense vector similarity search.
  - GraphStore: explicit relationship storage and traversal.
  - EmbeddingProvider: text/image → dense vector embedding.
  - LLMProvider: chat completion for LLM-backed operations.
  - BM25Index: lexical/keyword search.
  - SparseIndex: sparse vector similarity search.
  - Reranker: cross-encoder reranking.
  - UnitStore: unit and space CRUD persistence.

Exports:
    VectorIndex, GraphStore, EmbeddingProvider, StaticEmbeddingProvider,
    LLMProvider, LLMChatResponse, BM25Index, SparseIndex, Reranker, UnitStore.
"""

from .bm25_index import BM25Index
from .embedding_provider import EmbeddingProvider, StaticEmbeddingProvider
from .graph_store import GraphStore
from .llm_provider import LLMChatResponse, LLMProvider
from .reranker import Reranker
from .sparse_index import SparseIndex
from .unit_store import UnitStore
from .vector_index import VectorIndex

__all__ = [
    "BM25Index",
    "EmbeddingProvider",
    "GraphStore",
    "LLMChatResponse",
    "LLMProvider",
    "Reranker",
    "SparseIndex",
    "StaticEmbeddingProvider",
    "UnitStore",
    "VectorIndex",
]
