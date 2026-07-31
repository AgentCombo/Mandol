Component Diagram
===================

.. mermaid::

   graph TB
       subgraph MemorySystem
           MSYS[MemorySystem Main Entry]
       end

       subgraph Services
           SMAP[SemanticMapService]
           SGPH[SemanticGraphService]
           SMGR[SessionManager]
       end

       subgraph Builders
           MDG[MultiDimSemanticGraph]
           HL[HighLevelSummary]
           ER[EntityRelation]
           EC[EventCausal]
           SS[SemanticSimilarity]
       end

       subgraph Retrievers
           HR[HybridRetriever]
           dense[DenseRetriever]
           bm25[Bm25Retriever]
           sparse[SparseRetriever]
           hop_ret[SubgraphHopRetriever]
           fusion[RRFusion]
       end

       subgraph StorePorts[Stores - ports]
           US[UnitStore]
           GS[GraphStore]
           VI[VectorIndex]
       end

       MSYS --> SMAP
       MSYS --> SGPH
       MSYS --> SMGR
       SMGR --> MDG
       MDG --> HL
       MDG --> ER
       MDG --> EC
       MDG --> SS
       MSYS --> HR
       HR --> dense
       HR --> bm25
       HR --> sparse
       HR --> hop_ret
       HR --> fusion
       SMAP --> VI
       SMAP --> US
       SGPH --> GS
       SMAP --> SMGR

Component Details
------------------

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Component
     - Layer
     - Responsibility
   * - MemorySystem
     - application
     - Unified entry point, orchestrates services
   * - SemanticMapService
     - application
     - Unit CRUD + indexing + spaces
   * - SemanticGraphService
     - application
     - Graph relationships + graph traversal
   * - SessionManager
     - application
     - Session lifecycle
   * - MultiDimSemanticGraph
     - application
     - Multi-dimensional build orchestration
   * - HybridRetriever
     - application
     - Multi-path retrieval + RRF + BFS
   * - DimensionBuilders
     - application
     - Extraction logic for each dimension
