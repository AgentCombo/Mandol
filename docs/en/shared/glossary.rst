Unified Glossary
==================

This document establishes the Chinese-English mapping and code references for core terms in Mandol, ensuring terminology consistency across documentation and conversations.

Core Data Structures
--------------------

.. list-table::
   :header-rows: 1
   :widths: 22 28 50

   * - Chinese
     - English / Code Identifier
     - Description
   * - 记忆单元
     - MemoryUnit
     - The smallest memory carrier in the system, encapsulating a conversation, an entity, or an event
   * - 记忆空间
     - MemorySpace
     - Logical container for memory units, supporting hierarchical nesting
   * - 语义索引
     - SemanticMap / SemanticMapService
     - Responsible for memory unit CRUD, vector indexing, and space management
   * - 语义图
     - SemanticGraph / SemanticGraphService
     - Responsible for relationship modeling and graph traversal between memory units
   * - 唯一标识
     - Uid
     - Unique ID for each memory unit

Retrieval
---------

.. list-table::
   :header-rows: 1
   :widths: 22 28 50

   * - Chinese
     - English / Code Identifier
     - Description
   * - 全记忆检索
     - holistic_retrieve
     - Unified retrieval entry point, automatically coordinating multi-group multi-path retrieval
   * - 按视图检索
     - retrieve_by_view
     - Filter retrieval results by predefined semantic views
   * - 空间内检索
     - retrieve_in_space
     - Retrieve within a specified memory space
   * - 检索命中
     - SearchHit
     - Single retrieval result unit, carrying final_score / scores / ranks
   * - 倒数排名融合
     - RRF (Reciprocal Rank Fusion)
     - Parameter-free algorithm for merging results from multiple retrievers
   * - BFS 图扩展
     - bfs_expand_units
     - Expand candidate set along graph relationships using retrieval results as seeds
   * - 重排序
     - Rerank (Cross-Encoder)
     - Fine-tune merged candidates using Cross-Encoder model

High-Level Memory Types
------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 28 50

   * - Chinese
     - English / Code Identifier
     - Description
   * - 基础记忆
     - Base Memory
     - Raw conversation data, without high-level processing
   * - 情景记忆
     - Episodic Memory
     - Events and episodic summaries, recording "what happened"
   * - 知识记忆
     - Knowledge Memory
     - Entities and knowledge summaries, recording "what is known"
   * - 情感记忆
     - Emotional Memory
     - User emotional states and preference summaries
   * - 程序记忆
     - Procedural Memory
     - Operation steps and process summaries
   * - 洞察记忆
     - Insights
     - Deep pattern recognition distilled from multi-perspective summaries

Graph Relationship Types
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 28 50

   * - Chinese
     - English / Code Identifier
     - Description
   * - 时序前驱
     - PRECEDES
     - Temporal ordering of conversations/events (before → after)
   * - 时序后继
     - FOLLOWS
     - Temporal ordering of conversations/events (after → before)
   * - 语义相似
     - SEMANTIC_SIMILAR
     - Semantic association based on vector cosine similarity
   * - 相关
     - RELATED_TO
     - Generic entity relationship (includes subtypes like located_in / works_at)
   * - 共指
     - COREF
     - Coreference from conversation unit to global entity
   * - 导致
     - CAUSES
     - Event causal relationship (A causes B)
   * - 被导致
     - CAUSED_BY
     - Event causal relationship (B is caused by A)
   * - 涉及
     - INVOLVES
     - Relationship between event and its participants/locations
   * - 证据支撑
     - EVIDENCED_BY
     - High-level memory pointing to the raw data that supports it

Architecture
------------

.. list-table::
   :header-rows: 1
   :widths: 22 28 50

   * - Chinese
     - English / Code Identifier
     - Description
   * - 端口
     - Port
     - Abstract interface definition, independent of concrete implementation
   * - 适配器 / 基础设施
     - Adapter / Infrastructure
     - Concrete implementation of a port (e.g., FAISS, OpenAI API)
   * - 会话管理器
     - SessionManager
     - Responsible for session segmentation, merging, and lifecycle management
   * - 维度构建器
     - DimensionBuilder
     - Responsible for extracting and modeling relationships for one type of memory
   * - 嵌入提供者
     - EmbeddingProvider
     - Abstract interface for text/image vectorization
   * - 重排器
     - Reranker
     - Abstract interface for retrieval result reranking
