六边形架构详解
================

Mandol 采用六边形（端口-适配器）架构，确保业务逻辑不依赖外部基础设施。

分层结构
--------

.. mermaid::

   graph TB
       subgraph Domain["domain/ 领域层"]
           MU[MemoryUnit]
           MS[MemorySpace]
           TY[Uid / SpaceName / Embedding]
       end

       subgraph Ports["ports/ 端口层"]
           EMB[EmbeddingProvider]
           LLM[LLMProvider]
           RR[Reranker]
           VI[VectorIndex]
           GS[GraphStore]
           US[UnitStore]
       end

       subgraph App["application/ 应用层"]
           MSYS[MemorySystem]
           SMAP[SemanticMapService]
           SGPH[SemanticGraphService]
           SMGR[SessionManager]
           MDG[MultiDimSemanticGraph]
       end

       subgraph Infra["infrastructure/ 基础设施层"]
           FA[FAISS VectorIndex]
           BM[BM25 Index]
           ST[SentenceTransformers]
           OA[OpenAI LLM]
           IMS[InMemory Store/Graph]
       end

       SMAP --> EMB
       SMAP --> VI
       SMAP --> US
       SGPH --> GS
       MDG --> LLM
       MDG --> EMB
       ST -.-> EMB
       OA -.-> EMB
       OA -.-> LLM
       OA -.-> RR
       FA -.-> VI
       BM -.-> VI
       IMS -.-> US
       IMS -.-> GS

各层职责
--------

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - 层
     - 职责
     - 示例
   * - domain
     - 领域模型，纯数据对象
     - MemoryUnit, MemorySpace, Uid
   * - ports
     - 抽象接口定义
     - EmbeddingProvider, VectorIndex
   * - application
     - 业务逻辑编排
     - MemorySystem, SessionManager
   * - infrastructure
     - 具体实现，可替换
     - FAISS, OpenAI API, SentenceTransformers

依赖方向
--------

::

   application → ports ← infrastructure
                    ↑
                  domain

- 所有层都可以依赖 domain
- application 只依赖 ports，不依赖 infrastructure
- infrastructure 实现 ports 中的接口
- 从未有 infrastructure → application 的依赖
