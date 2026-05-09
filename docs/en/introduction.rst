Project Introduction and Design Philosophy
============================================

.. note::

   For basic users, please refer to :doc:`/basic-user/what-is-mandol`. This document will be restructured in a future version, please update your bookmarks.

Project Background
-------------------

AI Agents need to remember, organize, and retrieve conversation history like humans during interactions with users. Traditional memory systems typically use simple vector databases or key-value stores, lacking modeling of semantic relationships between memories and unable to support multi-dimensional memory organization and retrieval.

Mandol was created to provide a complete memory management solution, including:

- **Automatic session segmentation**: LLM-based session boundary detection, splitting continuous conversations into meaningful session units
- **Multi-dimensional memory construction**: Extracting summaries, entities, events, insights, and other memory types from raw conversations
- **Semantic graph relationship modeling**: Establishing various associations including entity relationships, event causality, and semantic similarity
- **Hybrid retrieval strategy**: Combining dense vectors, sparse vectors, and keyword retrieval, improving recall through RRF fusion and BFS expansion

Design Philosophy
------------------

Ports and Adapters Architecture (Hexagonal Architecture)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The system adopts a strict ports-and-adapters layered pattern:

- **domain/**: Core domain models (MemoryUnit, MemorySpace, type definitions), independent of any external frameworks
- **ports/**: Abstract interface definitions (EmbeddingProvider, LLMProvider, VectorIndex, GraphStore, etc.)
- **application/**: Application service layer (MemorySystem, SemanticMapService, SemanticGraphService, etc.), orchestrating business workflows
- **infrastructure/**: Infrastructure implementations (in-memory storage, FAISS, OpenAI-compatible LLM/Embedding, etc.)

This architecture allows flexible replacement of underlying implementations, for example:
- Replacing in-memory vector index with FAISS index
- Replacing in-memory graph store with Neo4j
- Replacing local SentenceTransformers with OpenAI API

Multi-Perspective Memory Organization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Memory is not a flat list, but a hierarchical spatial structure organized by semantic dimensions. Each session generates the following space hierarchy:

- **base_memory_{suffix}**: Raw memory units
- **episodic_{suffix}**: Episodic memory (summary + events)
- **knowledge_{suffix}**: Knowledge memory (summary + entities)
- **emotional_{suffix}**: Emotional memory
- **procedural_{suffix}**: Procedural memory
- **insights_{suffix}**: Insights memory

Where ``{suffix}`` is a unique identifier generated based on the session's starting message index (e.g., ``msg_0_25``).

This organization supports fine-grained spatial retrieval, such as "retrieve only knowledge entities" or "retrieve only events."

Unified Retrieval Interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Users only need to call ``system.holistic_retrieve(query, top_k)`` to perform full-memory retrieval. The system internally executes the following flow:

1. **Group recall**: Distributes the retrieval request to BASE, ENTITY, EVENT, SUMMARY retrieval groups
2. **Three-way recall**: Each group independently executes Dense + BM25 + Sparse retrieval
3. **RRF fusion**: Merges multi-path results using Reciprocal Rank Fusion
4. **BFS expansion**: Expands candidate set based on graph relationships
5. **Global reranking**: After all candidates are merged, reranks via Cross-Encoder Reranker

Core Capabilities
------------------

- **Session management**: Automatic session boundary detection, supporting both manual and automatic session segmentation modes
- **Multi-type memory extraction**: Supports five memory types — summary, entity, event, insight, and global insight
- **Relationship modeling**: Entity relationships, event causality, semantic similarity, evidence association
- **Cross-session merging**: Automatically merges identical entity/event references across different sessions
- **Persistence**: Supports complete state export and loading in JSON format
- **Hybrid retrieval**: Unified retrieval strategy with three-way recall + RRF + BFS + Reranker
