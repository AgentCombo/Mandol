Detailed Depth: Memory Pipeline Internal Mechanism
=====================================================

This chapter provides a detailed description of each stage's internal mechanism for users who need to understand how the system works.

Stage 1: Write Path — From Raw Text to Vector Index
-------------------------------------------------------

When you call ``system.add(unit)``, the following happens:

1. **Preprocessing**: UnitPipeline.preprocess(unit) — chunking + name normalization
2. **Vectorization**: EmbeddingProvider generates dense vectors for text_content
3. **Storage**: UnitStore persists the unit
4. **Indexing**: AdaptiveVectorIndex inserts the vector into the index
5. **Similarity edge construction**: Computes cosine similarity with the most recent N memories, creates SEMANTIC_SIMILAR edges above threshold
6. **Session queue**: Unit is added to SessionManager's pending queue

Stage 2: Build Path — From Raw Memories to High-Level Structures
-------------------------------------------------------------------

When you call ``system.build_high_level(mode)``, the following happens:

1. **Session segmentation**: SessionManager detects topic boundaries through LLM semantic analysis
2. **Multi-dimensional construction**: MultiDimSemanticGraph orchestrates the following builders for each session:

   - **HighLevelSummary**: Generates episodic, knowledge, emotional, and procedural summaries
   - **EntityRelation**: Extracts entities and relationships
   - **EventCausal**: Extracts events and causal relationships
   - **SemanticSimilarity**: Builds similarity edges

3. **Cross-session merging**: Merges identical entity/event references across sessions
4. **Insight extraction**: Distills global insights from summaries

Multi-Perspective Memory Representation
-----------------------------------------

Each session generates the following space hierarchy:

.. code-block::

   root
   ├── root_base_memory_{suffix}
   │   └── [Raw dialogue units]
   └── root_high_level_memory_{suffix}
       ├── root_episodic_{suffix}
       │   ├── root_episodic_summary_{suffix}
       │   │   └── [Episodic summary units]
       │   └── root_episodic_event_{suffix}
       │       └── [Event units]
       ├── root_knowledge_{suffix}
       │   ├── root_knowledge_summary_{suffix}
       │   │   └── [Knowledge summary units]
       │   └── root_knowledge_entity_{suffix}
       │       └── [Entity units]
       ├── root_emotional_{suffix}
       │   └── [Emotional summary units]
       ├── root_procedural_{suffix}
       │   └── [Procedural summary units]
       └── root_insights_{suffix}
           └── [Insight units]

Where ``{suffix}`` is a unique identifier generated based on the session's starting message index (e.g., ``msg_0_25``).

Stage 3: Read Path — From Query to SearchHit
-----------------------------------------------

When you call ``system.holistic_retrieve(query)``, the following happens:

1. **Query vectorization**: EmbeddingProvider generates dense vector for the query
2. **Group recall**: Distributes the retrieval request to four groups:

   - **BASE**: Raw conversation memories
   - **ENTITY**: Knowledge entities
   - **EVENT**: Episodic events
   - **SUMMARY**: Summaries and insights

3. **Three-way recall**: Each group independently executes Dense + BM25 + Sparse retrieval
4. **RRF fusion**: Merges three-way results using Reciprocal Rank Fusion
5. **BFS expansion**: Expands candidate set based on graph relationships
6. **Global reranking**: After all groups are merged, reranks via Cross-Encoder Reranker
7. **Returns**: List of SearchHit objects
