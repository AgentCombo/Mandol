Concept Relationship Map
==========================

How do Mandol's core concepts relate to each other? The diagram below provides a global view.

.. mermaid::

   graph TB
       U[User] -->|creates| MU[MemoryUnit]
       MU -->|stored in| SMAP[SemanticMap]
       SMAP -->|organized into| MS[MemorySpace]
       MS -->|hierarchical nesting| MS2[MemorySpace Child]

       U -->|calls| ADD[add]
       ADD -->|writes to| SMAP
       ADD -->|vectorizes| EMB[EmbeddingProvider]
       EMB -->|writes to| VI[VectorIndex]

       U -->|calls| BHL[build_high_level]
       BHL -->|triggers| SM[SessionManager]
       SM -->|segments| SESS[Session]
       SESS -->|drives| MDG[MultiDimSemanticGraph]
       MDG -->|extracts| ENT[Entity]
       MDG -->|extracts| EVT[Event]
       MDG -->|generates| SUMM[Summary]
       MDG -->|establishes| REL[Relationship]

       ENT -->|writes to| SGPH[SemanticGraph]
       EVT -->|writes to| SGPH
       SUMM -->|writes to| SMAP
       REL -->|writes to| SGPH

       U -->|calls| HR[holistic_retrieve]
       HR -->|recalls| DENSE[Dense Retrieval]
       HR -->|recalls| BM25[BM25 Keyword Retrieval]
       HR -->|recalls| SPARSE[Sparse Retrieval]
       DENSE -->|fuses| RRF[RRF Fusion]
       BM25 -->|fuses| RRF
       SPARSE -->|fuses| RRF
       RRF -->|expands| BFS[BFS Graph Expansion]
       SGPH --> BFS
       BFS -->|reranks| RR[Reranker]
       RR -->|returns| HIT[SearchHit]

   style MU fill:#e1f5fe
   style SMAP fill:#e1f5fe
   style MS fill:#e1f5fe
   style SGPH fill:#e8f5e9
   style HR fill:#fff3e0
   style HIT fill:#fce4ec

Core Concepts in One Sentence
------------------------------

- **MemoryUnit**: The smallest thing to remember — a message, a knowledge point, or an event
- **MemorySpace**: A category box, grouping memories by topic or hierarchy
- **SemanticMap**: The archive manager, handling storage and lookup
- **SemanticGraph**: The relationship network, recording "who has what relationship with whom"
- **SessionManager**: The clerk, determining when a new topic starts
- **build_high_level**: The digestion command, turning raw notes into structured cards
- **holistic_retrieve**: The query command, the system automatically finds the most relevant content

Data Flow Diagram
------------------

::

   Raw conversation → add() → Vectorization + Indexing
                              ↓
              build_high_level()
                              ↓
        Session segmentation → Entity / Event / Summary / Relationship
                              ↓
              holistic_retrieve()
                              ↓
      Three-way recall → RRF fusion → BFS expansion → Rerank
                              ↓
                        SearchHit[]
