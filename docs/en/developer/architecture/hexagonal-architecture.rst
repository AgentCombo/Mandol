Hexagonal Architecture Details
=================================

Mandol adopts a hexagonal (ports-and-adapters) architecture, ensuring business logic doesn't depend on external infrastructure.

Layered Structure
------------------

.. mermaid::

   graph TB
       subgraph Domain["domain/ Domain Layer"]
           MU[MemoryUnit]
           MS[MemorySpace]
           TY[Uid / SpaceName / Embedding]
       end

       subgraph Ports["ports/ Port Layer"]
           EMB[EmbeddingProvider]
           LLM[LLMProvider]
           RR[Reranker]
           VI[VectorIndex]
           GS[GraphStore]
           US[UnitStore]
       end

       subgraph App["application/ Application Layer"]
           MSYS[MemorySystem]
           SMAP[SemanticMapService]
           SGPH[SemanticGraphService]
           SMGR[SessionManager]
           MDG[MultiDimSemanticGraph]
       end

       subgraph Infra["infrastructure/ Infrastructure Layer"]
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

Layer Responsibilities
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Layer
     - Responsibility
     - Examples
   * - domain
     - Domain models, pure data objects
     - MemoryUnit, MemorySpace, Uid
   * - ports
     - Abstract interface definitions
     - EmbeddingProvider, VectorIndex
   * - application
     - Business logic orchestration
     - MemorySystem, SessionManager
   * - infrastructure
     - Concrete implementations, replaceable
     - FAISS, OpenAI API, SentenceTransformers

Dependency Direction
---------------------

::

   application → ports ← infrastructure
                    ↑
                  domain

- All layers can depend on domain
- application only depends on ports, not infrastructure
- infrastructure implements interfaces defined in ports
- There is never a dependency from infrastructure → application
