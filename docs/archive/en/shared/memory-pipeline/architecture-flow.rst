Architecture Depth: Pipeline Extension Points
=============================================

This chapter is aimed at developers and contributors who need to understand the internal architecture. For the complete pipeline description (chunking, session segmentation, summary map-reduce, unified fact pipeline, cross-session merging, etc.), read :doc:`detailed-flow` first. This chapter focuses on architecture layers and extension points.

.. _diagram-placeholder-arch-en-1:

.. admonition:: Placeholder: Hexagonal Architecture Layer Diagram
   :class: hint

   **What this diagram should show:**

   Four-layer architecture: **Domain (MemoryUnit / MemorySpace / Uid) → Ports (EmbeddingProvider / LLMProvider / Reranker / VectorIndex / GraphStore / UnitStore) → Application (MemorySystem / SemanticMapService / SemanticGraphService / SessionManager / UnifiedFactPipeline / SummaryMapReducer / GlobalInsightManager) → Infrastructure (FAISS / BM25 / SentenceTransformers / OpenAI / InMemoryStore)**.

   Arrows between layers should indicate dependency direction (application depends on ports, infrastructure implements ports).

Developer Extension Points
--------------------------

Each stage of Mandol's pipeline has extension points where developers can influence behavior by implementing port interfaces or adjusting configuration.

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Pipeline Stage
     - Customizable
     - Extension Method
   * - **Chunking**
     - Chunk size, overlap tokens, token estimation algorithm
     - Configure ``chunk_max_tokens``, ``overlap_tokens``; replace token estimator in ``DocumentChunker``
   * - **Vectorization (Embedding)**
     - Embedding model, dimension
     - Implement ``EmbeddingProvider`` port
   * - **Session Segmentation**
     - Segmentation prompt, check interval, context window size
     - Configure ``session_check_interval``, ``session_time_gap_seconds``; override prompt templates in ``SessionManager``
   * - **Summary Generation (Map-Reduce)**
     - Four summary type prompts, chunk token budget, map/reduce strategy
     - Override prompt templates per category in ``SummaryMapReducer``; adjust ``map_chunk_max_tokens``
   * - **Entity/Event Extraction (Unified Pipeline)**
     - Extraction prompts, multi-signal retrieval params, candidate counts
     - Override four prompt types in ``UnifiedFactPipeline``; adjust ``top_k_entities``, ``top_k_events``; replace retrieval strategy
   * - **Cross-Session Coref**
     - Merge judgment prompt, vector similarity threshold, LLM confidence threshold, max candidates
     - Configure ``coref_vector_threshold``, ``coref_llm_confidence_threshold``, ``coref_max_candidates``; override LLM judge prompt
   * - **Insight Distillation**
     - Insight extraction prompt, global insight merge prompt
     - Override prompt templates in ``InsightMapReducer`` and ``GlobalInsightManager``
   * - **Retrieval**
     - Reranker model, RRF parameters, BFS expansion parameters
     - Implement ``Reranker`` port; configure ``bfs_expansion_per_seed``, ``bfs_expansion_hops``

Key Configuration Reference
----------------------------

The following parameters directly affect high-level memory construction behavior:

.. list-table::
   :header-rows: 1
   :widths: 35 20 45

   * - Parameter
     - Default
     - Purpose
   * - ``chunk_max_tokens``
     - 512
     - Token threshold triggering chunking
   * - ``overlap_tokens``
     - 0
     - Context overlap tokens between adjacent chunks
   * - ``session_check_interval``
     - 20
     - Number of accumulated memories before triggering an LLM session check
   * - ``session_max_pending``
     - 100
     - Pending queue upper bound; force flush when exceeded
   * - ``similarity_threshold``
     - 0.7
     - Minimum cosine similarity for SEMANTIC_SIMILAR edge creation
   * - ``similarity_recent_window``
     - 20
     - Number of recent memories considered in immediate similarity calculation
   * - ``use_unified_pipeline``
     - true
     - Whether to use the unified fact pipeline (recommended); false falls back to legacy dimension builders
   * - ``incremental_cross_session_coref``
     - true
     - Whether to enable incremental cross-session coreference resolution
   * - ``coref_vector_threshold``
     - 0.75
     - Vector similarity threshold for coref candidate recall
   * - ``coref_llm_confidence_threshold``
     - 0.6
     - Minimum LLM confidence for coreference judgment
   * - ``coref_max_candidates``
     - 10
     - Maximum coref candidates
   * - ``bfs_expansion_per_seed``
     - 5
     - Neighbors fetched per seed node during BFS expansion
   * - ``bfs_expansion_hops``
     - 1
     - Number of BFS expansion hops
   * - ``auto_build_if_empty``
     - true
     - Whether to auto-trigger build when high-level memory is empty on retrieval

Component Relationships
-----------------------

.. code-block::

   MemorySystem                         # Main entry facade
   ├── SemanticMapService               # Memory unit CRUD, vector indexing, space management
   │   ├── UnitStore                    #   Unit persistence
   │   ├── AdaptiveVectorIndex          #   Adaptive vector index
   │   │   ├── BruteForceVectorIndex    #     Used when < promote_threshold
   │   │   └── FAISSVectorIndex         #     Auto-switch when >= promote_threshold
   │   ├── EmbeddingProvider            #   Vectorization abstraction
   │   └── Reranker                     #   Reranking abstraction
   ├── SemanticGraphService             # Graph relationship service
   │   └── GraphStore                   #   Graph storage abstraction
   ├── DocumentChunker                  # Document chunking
   ├── SessionManager                   # Session detection & management
   ├── SummaryMapReducer                # Four-category summary map-reduce
   ├── InsightMapReducer                # Session-level insight extraction
   ├── GlobalInsightManager             # Global insight accumulation & merging
   ├── UnifiedFactPipeline              # Entity/event/relation unified extraction
   │   └── CrossSessionCorefManager     #   Cross-session coreference resolution
   └── HybridRetriever                  # Retrieval orchestration
       ├── DenseRetriever               #   Dense vector retrieval
       ├── Bm25Retriever                #   BM25 keyword retrieval
       ├── SparseRetriever              #   Sparse vector retrieval
       ├── SubgraphHopRetriever         #   Graph expansion retrieval
       └── RRFusion                     #   Rank fusion

Port Interfaces
---------------

Each port defines an abstraction replaceable in ``infrastructure/``. Core method signatures:

**EmbeddingProvider**
   - ``embed_text(texts: list[str]) -> list[Embedding]``
   - ``embed_image_paths(paths: list[str]) -> list[Embedding]``
   - ``embedding_dim() -> int``

**LLMProvider**
   - ``generate(prompt: str, **kwargs) -> str``
   - ``generate_structured(prompt: str, schema: dict, **kwargs) -> dict``

**Reranker**
   - ``rerank(query: str, units: list[MemoryUnit], top_k: int) -> list[tuple[MemoryUnit, float]]``

**VectorIndex**
   - ``upsert(items: list[tuple[Uid, Embedding]]) -> None``
   - ``search(query: Embedding, top_k: int) -> list[tuple[Uid, float]]``
   - ``delete(uids: list[Uid]) -> None``
   - ``rebuild(items: list[tuple[Uid, Embedding]]) -> None``
   - ``dim() -> int``

**UnitStore**
   - ``upsert_units(units: list[MemoryUnit]) -> None``
   - ``delete_units(uids: list[Uid]) -> None``
   - ``get_unit(uid: Uid) -> MemoryUnit | None``
   - ``list_units() -> list[MemoryUnit]``
   - ``get_units(uids: list[Uid]) -> list[MemoryUnit]``
   - ``upsert_spaces(spaces: list[MemorySpace]) -> None``
   - ``get_space(name: SpaceName) -> MemorySpace | None``
   - ``list_spaces() -> list[MemorySpace]``
   - ``flush() -> None``

**GraphStore**
   - ``upsert_relationship(source: Uid, target: Uid, rel_type: str, properties: dict) -> None``
   - ``delete_relationship(source: Uid, target: Uid, rel_type: str | None) -> None``
   - ``get_relationship(source: Uid, target: Uid, rel_type: str) -> dict | None``
   - ``get_neighbors(uid: Uid, rel_type: str | None, direction: str) -> list[Uid]``
   - ``flush() -> None``
