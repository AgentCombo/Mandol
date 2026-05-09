Architecture Depth: Memory Pipeline Code-Level Implementation
================================================================

This chapter provides code-level implementation details for developers who need to extend or debug the system.

Core Class Relationships
--------------------------

.. code-block::

   MemorySystem
   ├── SemanticMapService (system.semantic_map)
   │   ├── UnitStore (unit persistence)
   │   ├── AdaptiveVectorIndex (adaptive vector index)
   │   │   ├── BruteForceVectorIndex (< promote_threshold)
   │   │   └── FAISSVectorIndex (>= promote_threshold)
   │   ├── EmbeddingProvider (vectorization)
   │   └── Reranker (reranking)
   ├── SemanticGraphService (system.semantic_graph)
   │   └── GraphStore (graph storage)
   │       └── InMemoryGraphStore (default)
   ├── SessionManager (session management)
   │   └── Session (session object)
   ├── MultiDimSemanticGraph (multi-dimensional construction)
   │   ├── HighLevelSummary (summary builder)
   │   ├── EntityRelation (entity builder)
   │   ├── EventCausal (event builder)
   │   └── SemanticSimilarity (similarity builder)
   └── HybridRetriever (retrieval orchestration)
       ├── DenseRetriever (dense vector retrieval)
       ├── Bm25Retriever (keyword retrieval)
       ├── SparseRetriever (sparse vector retrieval)
       ├── SubgraphHopRetriever (graph retrieval)
       └── RRFusion (rank fusion)

Key Implementation Details
----------------------------

Adaptive Vector Index
~~~~~~~~~~~~~~~~~~~~~~~

The system uses ``AdaptiveVectorIndex`` to automatically switch between brute-force search and FAISS indexing based on data volume:

- When unit count < ``promote_threshold`` (default 100): Uses numpy brute-force search
- When unit count >= ``promote_threshold``: Automatically promotes to FAISS HNSW index
- Promotion is transparent, no manual intervention needed

Session Segmentation Strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The system uses a dual strategy for session segmentation:

1. **Time interval**: When the timestamp difference between adjacent memories exceeds ``session_time_gap_seconds``, force a new session
2. **LLM detection**: When accumulated memories reach ``session_check_interval``, call LLM to determine session boundaries

Cross-Session Merging
~~~~~~~~~~~~~~~~~~~~~~~

``build_high_level()`` internally triggers cross-session entity and event merging:

1. **Entity merging**: LLM determines whether two entity names refer to the same concept
2. **Event merging**: High event similarity + close timing → merged as the same event
3. **Coreference resolution**: Creates COREF edges from dialogue units to global entities

Retrieval Pipeline Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The retrieval pipeline is orchestrated by ``HybridRetriever``:

1. **Query routing**: Distributes the query to four retrieval groups
2. **Three-way recall**: Each group independently executes Dense + BM25 + Sparse retrieval
3. **RRF fusion**: Merges results using Reciprocal Rank Fusion with formula ``RRF(d) = Σ 1/(k + rank_r(d))``
4. **BFS expansion**: Expands candidate set using SubgraphHopRetriever
5. **Global reranking**: Cross-Encoder Reranker produces final scores
