Multi-Perspective Construction and Representation
=================================================

.. note::

   This document has been migrated to :doc:`/shared/memory-pipeline/detailed-flow` and :doc:`/shared/memory-pipeline/architecture-flow`. This page will be removed in a future version, please update your bookmarks.

This section describes the construction pipeline of the Mandol multi-perspective semantic memory system and the specific representation of each memory type within the system.

System Architecture Overview
----------------------------

.. mermaid::

   graph TB
       A[Raw Dialogue Data] --> B[MemorySystem.add]
       B --> C[SemanticMap]
       C --> D[Chunking & Embedding]
       D --> E[MemoryUnit]
       E --> F[VectorIndex]
       E --> G[BM25Index]
       E --> H[SparseIndex]
       
       I[build_high_level] --> J[SessionManager]
       J --> K[Session Segmentation]
       K --> L[MultiDimSemanticGraph]
       
       L --> M[LayoutNormalization]
       L --> N[SemanticSimilarity]
       L --> O[HighLevelSummary]
       L --> P[EventCausal]
       L --> Q[EntityRelation]
       
       M --> R[Space Hierarchy]
       N --> S[Semantic Relation Edges]
       O --> T[Summary Units]
       P --> U[Event Causal Edges]
       Q --> V[Entity Relation Edges]

Construction Pipeline
---------------------
--------------------

The entire construction pipeline consists of the following stages:

1. **Data Input**: Add raw dialogue data via ``add()`` method
2. **Chunking and Vectorization**: Automatic chunking and generation of dense/sparse vectors
3. **Session Segmentation**: LLM-based session boundary detection
4. **Space Layout**: Creating multi-dimensional space hierarchy
5. **Dimension Construction**: Extracting summaries, entities, events, etc. and establishing relationships

Space Naming Strategy
---------------------
--------------------

``SpaceNamingPolicy`` is responsible for generating unique namespaces for each session. The space hierarchy is as follows:

.. code-block::

   root
   ├── base_memory_{suffix}          # Base memory (raw units)
   └── high_level_memory_{suffix}    # High-level memory
       ├── episodic_{suffix}         # Episodic memory
       │   ├── episodic_summary      # Episodic summary
       │   └── episodic_event        # Episodic events
       ├── knowledge_{suffix}        # Knowledge memory
       │   ├── knowledge_summary     # Knowledge summary
       │   └── knowledge_entity      # Knowledge entities
       ├── emotional_{suffix}        # Emotional memory
       ├── procedural_{suffix}       # Procedural memory
       └── insights_{suffix}         # Insight memory

Where ``{suffix}`` is a unique identifier generated based on the session's starting message index (e.g., ``msg_0_25``).

Multi-Perspective Memory Representation
---------------------------------------

.. note::

   All multi-perspective memory units below are automatically constructed by the system in ``build_high_level()``.
   Users should not manually create them. The node structures shown here are only for understanding the system's internal memory representation patterns.

.. _en-representation-base-memory:

Base Memory
^^^^^^^^^^^

Base memory stores raw dialogue units as the system's underlying data source.

**Edge Types**:

- ``PRECEDES``: Temporal edge (previous dialogue pointing to next)
- ``FOLLOWS``: Temporal edge (next dialogue pointing to previous)
- ``SEMANTIC_SIMILAR``: Semantic similarity edge (based on vector similarity threshold)

**Graph Structure Example**:

.. mermaid::

   graph LR
       D1["dialogue_001<br>Went to Beijing"] -->|PRECEDES| D2["dialogue_002<br>Visited the Palace Museum"]
       D2 -->|PRECEDES| D3["dialogue_003<br>Also went to the Great Wall"]
       D2 -->|FOLLOWS| D1
       D3 -->|FOLLOWS| D2
       D1 -.->|SEMANTIC_SIMILAR| D2

.. _en-representation-entity-relation:

Entity Relation
^^^^^^^^^^^^^^^

The entity relation perspective extracts named entities from dialogue and establishes semantic relationships between entities.

**Edge Types**:

- ``RELATED_TO``: Generic relation edge (with subtypes: hometown, lives_in, works_at, located_in, part_of)
- ``COREF``: Coreference edge (between base dialogue units and entities, indicating mention/reference)
- ``ALIAS_OF``: Alias edge (entity alias relationships)
- ``EVIDENCED_BY``: Provenance edge (entity pointing to original dialogues that mention it)

.. note::

   **Coreference Resolution Mechanism**:

   In the current coreference resolution flow, ``COREF`` edges are established directly between **base dialogue memory units → global entities**,
   indicating that the dialogue unit "mentions" the global entity. When multiple dialogue units mention the same global entity,
   the following two edges form a complete coreference chain:

   1. EVIDENCED_BY edge (entity → dialogue): indicates which original dialogues support/mention the entity
   2. COREF edge (dialogue → entity): indicates the specific reference relationship from the dialogue unit to the entity

   This design enables cross-session same-reference entities to be quickly located through graph traversal.

**Graph Structure Example**:

.. mermaid::

   graph LR
       E1["Beijing<br>Place"] -->|RELATED_TO<br>located_in| E2["Palace Museum<br>Place"]
       E2 -->|RELATED_TO<br>part_of| E1
       E1 -->|RELATED_TO<br>located_in| E3["Great Wall<br>Place"]

       D1["dialogue_001"] -->|COREF<br>mentions| E1
       D2["dialogue_002"] -->|COREF<br>mentions| E2
       D3["dialogue_003"] -->|COREF<br>mentions| E1

       D1 -.->|EVIDENCED_BY| E1
       D2 -.->|EVIDENCED_BY| E2

.. _en-representation-event-causal:

Event Causal
^^^^^^^^^^^^

The event causal perspective extracts events from dialogue and establishes causal relationship chains between events.

**Edge Types**:

- ``CAUSES``: Causal relationship (event A causes event B)
- ``CAUSED_BY``: Caused-by relationship (event B is caused by event A)
- ``INVOLVES``: Event-entity edge (with subtypes: participant, location, organizer, victim)
- ``PRECEDES`` / ``FOLLOWS``: Temporal edges (temporal ordering of events)
- ``EVIDENCED_BY``: Provenance edge (event pointing to original dialogues that support it)

**Graph Structure Example**:

.. mermaid::

   graph LR
       EV1["Business trip to Beijing"] -->|CAUSES| EV2["Visited the Palace Museum"]
       EV2 -->|CAUSES| EV3["Learned about history"]
       EV1 -->|PRECEDES| EV2
       EV2 -->|PRECEDES| EV3
       EV1 -.->|INVOLVES<br>location| E1["Beijing"]
       EV2 -.->|INVOLVES<br>location| E2["Palace Museum"]
       D1["dialogue_001"] -.->|EVIDENCED_BY| EV1
       D2["dialogue_002"] -.->|EVIDENCED_BY| EV2

.. _en-representation-summaries:

Summaries (Episodic, Knowledge, Emotional, Procedural)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each session generates four types of summaries:

- **Episodic Summary**: High-level overview of events in a session
- **Knowledge Summary**: Key concepts and facts extracted from dialogue
- **Emotional Summary**: User's emotional states and attitudes
- **Procedural Summary**: Operational steps, methods, and techniques

All summary types use ``EVIDENCED_BY`` edges to trace back to their supporting original dialogue units.

.. _en-representation-insights:

Insights
^^^^^^^^

The insight perspective extracts deep-level observations and pattern recognition from dialogue.

**Edge Types**:

- ``EVIDENCED_BY``: Provenance edge (insight pointing to supporting dialogues/summaries)
- ``SEMANTIC_SIMILAR``: Semantic similarity edge with other insights

Global Graph Structure Overview
-------------------------------

.. note::

   **Evidence Provenance**:

   - High-level summaries (episodic, knowledge, procedural, emotional) have ``EVIDENCED_BY`` edges pointing directly to base dialogue memory units,
     indicating these summaries are distilled from raw dialogue data
   - Insight memory has ``EVIDENCED_BY`` edges pointing to all four types of high-level summaries,
     indicating insights are deep-level observations synthesized from multi-perspective information
   - Entity relations and event causality also receive evidence support from base memory through ``EVIDENCED_BY`` edges

.. mermaid::

   graph TB
       subgraph Base Memory["Base Memory"]
           D1["dialogue_001<br>Went to Beijing"]
           D2["dialogue_002<br>Visited the Palace Museum"]
           D3["dialogue_003<br>Went to the Great Wall"]
           D1 -->|PRECEDES| D2
           D2 -->|PRECEDES| D3
       end

       subgraph Entity Relation["Entity Relation"]
           E1["Beijing<br>Place"]
           E2["Palace Museum<br>Place"]
           E3["Great Wall<br>Place"]
           E1 -->|RELATED_TO| E2
           E1 -->|RELATED_TO| E3
       end

       subgraph Event Causal["Event Causal"]
           EV1["Business trip to Beijing"]
           EV2["Visited the Palace Museum"]
           EV3["Visited the Great Wall"]
           EV1 -->|CAUSES| EV2
           EV2 -->|CAUSES| EV3
           EV1 -.->|INVOLVES| E1
           EV2 -.->|INVOLVES| E2
       end

       subgraph Summaries["High-Level Summaries"]
           EM["Emotional Summary"]
           ES["Episodic Summary"]
           KS["Knowledge Summary"]
           PS["Procedural Summary"]
       end

       I1["Insights<br>Synthesized from all summaries"]

       D1 -.->|EVIDENCED_BY| EM
       D2 -.->|EVIDENCED_BY| EM
       D3 -.->|EVIDENCED_BY| EM

       D1 -.->|EVIDENCED_BY| ES
       D2 -.->|EVIDENCED_BY| ES
       D3 -.->|EVIDENCED_BY| ES

       D1 -.->|EVIDENCED_BY| KS
       D2 -.->|EVIDENCED_BY| KS
       D3 -.->|EVIDENCED_BY| KS

       D2 -.->|EVIDENCED_BY| PS

       D1 -.->|EVIDENCED_BY| E1
       D2 -.->|EVIDENCED_BY| E2
       D1 -.->|EVIDENCED_BY| EV1
       D2 -.->|EVIDENCED_BY| EV2

       D1 -->|COREF<br>mentions| E1
       D2 -->|COREF<br>mentions| E2
       D3 -->|COREF<br>mentions| E1

       EM -.->|EVIDENCED_BY| I1
       ES -.->|EVIDENCED_BY| I1
       KS -.->|EVIDENCED_BY| I1
       PS -.->|EVIDENCED_BY| I1

Dimension Builders
------------------

The system orchestrates five dimension builders through ``MultiDimSemanticGraph``:

LayoutNormalizationDimension
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Responsible for creating all space hierarchies and establishing parent-child relationships:

1. Create ``base_memory`` space (containing raw units)
2. Create ``high_level_memory`` space
3. Create ``episodic``, ``knowledge``, ``emotional``, ``procedural`` subspaces
4. Create summary/entity/event spaces for each subspace
5. Establish ``insights`` space

SemanticSimilarityDimension
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Computes semantic similarity between units and adds relationship edges:

1. Get all units within a space
2. Compute cosine similarity between unit pairs
3. Add ``SEMANTIC_SIMILAR`` relationships for pairs exceeding the threshold

HighLevelSummaryApplicatorDimension
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Applies summary units to corresponding spaces:

1. Get session summaries from ``SummaryMapReducer``
2. Create summary units and add to corresponding spaces
3. Establish ``EVIDENCED_BY`` relationships connecting original units to summaries

EventCausalApplicatorDimension
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Applies event units and establishes causal chains:

1. Get deduplicated events from ``EventDeduper``
2. Create event units and add to corresponding spaces
3. Add ``CAUSES`` / ``CAUSED_BY`` relationships based on LLM-extracted causality

EntityRelationApplicatorDimension
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Applies entity units and establishes entity relationships:

1. Get deduplicated entities from ``EntityDeduper``
2. Create entity units and add to corresponding spaces
3. Add entity relation edges based on LLM-extracted relationships
4. Establish ``EVIDENCED_BY`` relationships connecting original units to entities

Session Management
------------------

``SessionManager`` is responsible for session segmentation and management:

- **LLM-driven session segmentation**: Uses LLM to detect session boundaries between messages
- **Time boundary detection**: Supports time-interval-based session segmentation
- **Topic continuity assessment**: Detects boundaries based on dialogue topic changes
- **Asynchronous session building**: Supports ``build_session_async()`` for background processing

Cross-Session Merging
---------------------

After construction is complete, the system automatically performs cross-session merging:

1. **Entity Merging**: Using ``EntityDeduper`` to merge entities with the same reference across different sessions
2. **Event Merging**: Using ``EventDeduper`` to merge identical events across different sessions

These operations are automatically triggered within ``build_high_level()`` and are transparent to the user.

Construction Flow Diagram
-------------------------

.. mermaid::

   sequenceDiagram
       participant U as User
       participant MS as MemorySystem
       participant SM as SessionManager
       participant MDG as MultiDimSemanticGraph
       participant LLM as LLM/Embedder

       U->>MS: add(unit)
       MS->>MS: Chunking & Vectorization
       
       U->>MS: build_high_level(mode="auto")
       MS->>SM: Get unprocessed sessions
       SM->>LLM: Session segmentation
       LLM-->>SM: Session boundaries
       
       SM->>MDG: build_session(session)
       MDG->>MDG: LayoutNormalization
       MDG->>LLM: Extract summaries/entities/events
       LLM-->>MDG: Extraction results
       MDG->>MDG: SemanticSimilarity
       MDG->>MDG: HighLevelSummary
       MDG->>MDG: EventCausal
       MDG->>MDG: EntityRelation
       MDG-->>MS: Build complete
       
       MS->>MS: merge_cross_session_entities
       MS->>MS: merge_cross_session_events
       MS-->>U: BuildReport
