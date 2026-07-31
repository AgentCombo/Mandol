组件关系图
============

.. mermaid::

   graph TB
       subgraph MemorySystem
           MSYS[MemorySystem 主入口]
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
           graph[SubgraphHopRetriever]
           fusion[RRFusion]
       end

       subgraph Stores["Stores (ports)"]
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
       HR --> graph
       HR --> fusion
       SMAP --> VI
       SMAP --> US
       SGPH --> GS
       SMAP --> SMGR

组件详解
--------

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - 组件
     - 所属层
     - 职责
   * - MemorySystem
     - application
     - 统一入口，编排各服务
   * - SemanticMapService
     - application
     - 单元CRUD + 索引 + 空间
   * - SemanticGraphService
     - application
     - 图关系 + 图遍历
   * - SessionManager
     - application
     - 会话生命周期
   * - MultiDimSemanticGraph
     - application
     - 多维度构建编排
   * - HybridRetriever
     - application
     - 多路检索 + RRF + BFS
   * - DimensionBuilders
     - application
     - 各维度的提取逻辑
