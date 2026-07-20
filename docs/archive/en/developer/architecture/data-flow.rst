Data Flow Details
===================

Complete data flow from raw text to retrieval results.

Phase 1: Write Path
---------------------

.. mermaid::

   sequenceDiagram
       participant U as Caller
       participant MSYS as MemorySystem
       participant SMAP as SemanticMapService
       participant EMB as EmbeddingProvider
       participant US as UnitStore
       participant VI as VectorIndex

       U->>MSYS: add(unit)
       MSYS->>MSYS: UnitPipeline.preprocess(unit)
       Note over MSYS: Chunking + name normalization
       MSYS->>SMAP: add_unit(unit)
       SMAP->>EMB: embed_text(text)
       EMB-->>SMAP: embedding
       SMAP->>US: upsert_units([unit])
       SMAP->>VI: upsert([(uid, embedding)])
       SMAP-->>MSYS: ok
       Note over MSYS: Similarity edge creation + enqueue to SessionManager

Phase 2: Build Path
---------------------

.. mermaid::

   sequenceDiagram
       participant U as Caller
       participant MSYS as MemorySystem
       participant SMGR as SessionManager
       participant LLM as LLMProvider
       participant MDG as MultiDimSemanticGraph

       U->>MSYS: build_high_level(mode)
       MSYS->>SMGR: process_pending_sessions()
       SMGR->>LLM: Session boundary detection
       LLM-->>SMGR: Segmentation results
       SMGR-->>MSYS: sessions

       loop For each session
           MSYS->>MDG: build_session(session)
           MDG->>LLM: Extract summaries/entities/events/relationships
           LLM-->>MDG: Structured results
           MDG-->>MSYS: Build results
       end

       MSYS->>MSYS: Cross-session entity/event merging
       MSYS-->>U: BuildReport

Phase 3: Read Path
--------------------

.. mermaid::

   sequenceDiagram
       participant U as Caller
       participant MSYS as MemorySystem
       participant HR as HybridRetriever
       participant RER as Reranker

       U->>MSYS: holistic_retrieve(query)
       MSYS->>HR: retrieve(query)

       par Four groups in parallel
           HR->>HR: BASE group: Dense+BM25+Sparse→RRF
           HR->>HR: ENTITY group: Dense+BM25+Sparse→RRF
           HR->>HR: EVENT group: Dense+BM25+Sparse→RRF
           HR->>HR: SUMMARY group: Dense+BM25+Sparse→RRF
       end

       HR->>HR: BFS expansion per group
       HR->>HR: Global candidate merging
       HR->>RER: rerank(query, candidates)
       RER-->>HR: Reranked results
       HR-->>MSYS: Final results
       MSYS-->>U: SearchHit[]
