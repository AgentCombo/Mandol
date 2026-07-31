Detailed Depth: Complete Memory Building and Retrieval Pipeline
===========================================================

This chapter provides a natural-language walkthrough of Mandol's complete technical pipeline — from receiving raw dialogue to returning retrieval results. Complex flowcharts are replaced with clear narrative descriptions, with placeholder sections where diagrams will be added in a future documentation pass.

.. note::

   For a quick overview, read :doc:`basic-flow` first. It summarizes the core process in three sentences.

Overall Pipeline
----------------

A memory unit goes through three major stages from ``add()`` to being retrievable via ``holistic_retrieve()``:

1. **Write stage (add)**：Raw data ingestion — chunking, vectorization, similarity edge construction
2. **Build stage (build_high_level)**：Structured processing of ingested memories — session segmentation, multi-type summary generation, entity/event extraction, insight distillation, cross-session merging
3. **Read stage (holistic_retrieve)**：Multi-group multi-path recall, fusion, graph expansion, reranking

Each stage is detailed below.

.. _pipeline-stage1-en:

Stage 1: add() — Raw Data Ingestion
------------------------------------

When you call ``system.add(unit)``, the following steps execute in order:

**1. Auto-Chunking**

The system checks the memory unit's text length. If it exceeds ``chunk_max_tokens`` (default 512), the text is split into multiple sub-units along sentence boundaries. Each sub-unit retains a reference to the parent unit (``parent_uid``) and carries a ``chunk_index`` marker.

Chunking strategy:
  - Token estimation via ``tiktoken`` (cl100k_base encoding) or a heuristic fallback
  - Sentence splitting on end-of-sentence punctuation (``. ! ? 。！？``), accumulating sentences until hitting the token limit
  - Configurable ``overlap_tokens`` to preserve context across adjacent chunks

Short texts bypass chunking and proceed as a single unit.

**2. Vectorization & Storage**

For each unit (or chunked sub-unit), the EmbeddingProvider generates a dense embedding. BM25 and TF-IDF sparse indices are also built. The unit is persisted to the UnitStore and the embedding is written to the VectorIndex.

**3. Immediate Similarity Edges**

Newly stored units are compared against the most recent ``similarity_recent_window`` (default 20) existing units via cosine similarity. Pairs exceeding ``similarity_threshold`` (default 0.7) receive a ``SEMANTIC_SIMILAR`` graph edge. Already-processed pairs are skipped.

**4. Pending Queue**

The unit is appended to ``_pending_units``. The system supports two session detection modes:

- **Synchronous mode**: When you explicitly call ``build_high_level()``, all queued units are processed in batch
- **Asynchronous mode** (default): The system monitors the queue in the background and triggers session detection automatically when the accumulated count reaches a threshold — no explicit call needed

In async mode, session detection runs in a separate thread pool and does not block ``add()``. If pending units exceed ``SESSION_MAX_PENDING`` (default 100), a forced flush protection triggers.

.. _pipeline-stage2-en:

Stage 2: build_high_level() — High-Level Semantic Construction
---------------------------------------------------------------

This is Mandol's most critical stage. ``mode="auto"`` processes only incremental (newly pending) units. ``mode="force"`` performs an export-rebuild strategy: exporting all base units and temporal edges, clearing store/index/graph, and re-running the full build pipeline from scratch.

.. _diagram-placeholder-en-1:

.. admonition:: Placeholder: High-Level Memory Construction Pipeline Overview
   :class: hint

   **What this diagram should show:**

   Starting from raw memory units written by ``add()``, the flow proceeds through **Chunking → Session Segmentation → Space Layout → Summary Generation (4 categories in parallel) → Entity/Event Extraction → Insight Distillation → Cross-Session Merging → Global Insight Update**. Use horizontal flow arrows connecting each stage, with key configuration parameters and outputs annotated below each stage.

   A horizontal or layered vertical layout is recommended, emphasizing the narrative of "how a raw dialogue becomes structured memory."

The following subsections explain each stage in detail.

.. _pipeline-sessioning-en:

2.1 Session Segmentation
^^^^^^^^^^^^^^^^^^^^^^^^

The system feeds adjacent memory unit content to the LLM for semantic topic boundary detection.

**How it works**:

- SessionManager takes the most recent ``MAX_CONTEXT_UNITS`` (default 20) units from the pending queue, sorts them by timestamp, and formats them as numbered, timestamped text lines
- These lines, along with a system prompt, are sent to the LLM, which determines split points based on semantic/episodic boundaries
- The LLM returns a list of split point indices (1-based line numbers) and a ``should_wait`` flag indicating whether more context is needed
- Split points are processed right-to-left (keeping earlier indices valid), and each segment becomes an independent Session

**Async self-scheduling**:

In async mode, after each LLM call completes, the system checks whether new units have entered the queue. If so, another detection round is scheduled automatically, creating an adaptive "detect when there's data, idle when there isn't" rhythm. This enables continuous session boundary discovery in real-time dialogue scenarios without manual polling.

**Safety nets**:

- When pending units exceed ``SESSION_MAX_PENDING`` (default 100), all are force-flushed into a single session
- After LLM retries are exhausted, a "no split" fallback prevents pipeline stalls

.. _pipeline-space-layout-en:

2.2 Space Layout
^^^^^^^^^^^^^^^^

Each newly detected Session receives a unique session ID (format: ``sess_YYYYMMDD_NNN``) and a corresponding memory space is created:

.. code-block::

   {root}_session_{session_id}    # e.g., default_session_sess_20260519_001

This space is a child of ``base_memory``. All units belonging to this session (raw dialogues and subsequently extracted high-level units) are assigned to this space for per-session retrieval and provenance tracing.

The system also ensures the following global space hierarchy exists (lazily created, idempotent):

.. code-block::

   {root}_base_memory              # Base memory
   {root}_high_level_memory        # High-level memory root
   ├── {root}_episodic             # Episodic memory
   │   ├── {root}_episodic_summary #   Episodic summaries
   │   └── {root}_episodic_event   #   Episodic events (canonical)
   ├── {root}_knowledge            # Knowledge memory
   │   ├── {root}_knowledge_summary#   Knowledge summaries
   │   └── {root}_knowledge_entity #   Knowledge entities (canonical)
   ├── {root}_emotional            # Emotional memory
   ├── {root}_procedural           # Procedural memory
   └── {root}_insights             # Insight memory (global, continuously updated)

.. _pipeline-summary-en:

2.3 Multi-Type Summary Generation (Map-Reduce)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For each Session's dialogue units, the system uses a **Map-Reduce pattern** to generate four categories of summaries, avoiding LLM context window overflow from a single call.

**Map phase**:
  - Session units are split into token-budgeted chunks (default max 2560 tokens per chunk, 30 units)
  - For each chunk, **parallel** LLM calls extract four categories (independent prompts, 4 parallel calls):

    * **Episodic Summary**: Timeline, key people, main events, locations
    * **Knowledge Summary**: Core concepts, key facts, techniques, prerequisite knowledge
    * **Emotional Summary**: User preferences, emotional reactions, behavioral patterns
    * **Procedural Summary**: Process steps, decision points, preconditions

**Reduce phase**:
  - For each category, chunk-level summaries are pairwise-merged via LLM
  - Multiple reduction rounds converge to a single session-level summary per category
  - Each category's reduction is independent — all four run in parallel

**Result persistence**:
  - Each final summary is wrapped as a MemoryUnit and stored in the corresponding space
  - ``EVIDENCED_BY`` graph edges are created from summaries to source dialogue units for full provenance

.. _pipeline-entity-event-en:

2.4 Entity & Event Extraction (Unified Fact Pipeline)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The system uses the Unified Fact Pipeline (``UnifiedFactPipeline``) to extract entities and events from session dialogues. This pipeline replaces the legacy five-dimension builder architecture.

**Entity extraction**:
  - Concatenates all session dialogue text
  - Retrieves existing entities as context via multi-signal search (name/alias matching + BM25 keyword + vector similarity)
  - Calls the LLM to identify new entities from dialogue, automatically linking them to existing entities (``linked_id``)
  - Extracted entities include: name, type (Person/Place/Organization etc.), description, aliases

**Event extraction**:
  - Based on extracted entities and existing event context, calls the LLM to identify events
  - Extracted events include: event type, participants, time, location, description

**Relation & causal extraction** (runs in parallel with event extraction):
  - **Entity relations**: LLM identifies semantic relationships between entities (``located_in``, ``works_at``, ``part_of``, ``hometown``, etc.)
  - **Event causality**: LLM identifies causal (``CAUSES`` / ``CAUSED_BY``) and temporal relationships between events

.. _pipeline-insight-en:

2.5 Insight Extraction
^^^^^^^^^^^^^^^^^^^^^^

After all four summary categories are generated, the InsightMapReducer feeds the summary texts to the LLM to distill deeper insights:

- **Pattern recognition**: Cross-category behavioral patterns and preferences
- **Causal relationships**: Deep causal chains discovered from surface events
- **Predictive insights**: Inferences and suggestions about future behavior
- **Behavioral characteristics**: Deep behavioral and cognitive traits
- **Optimization recommendations**: Specific actionable suggestions
- **Risk warnings**: Potential issues and concerns

Insight units are stored in the insights space, with ``EVIDENCED_BY`` edges pointing to all four supporting summary categories.

.. _pipeline-cross-session-en:

2.6 Cross-Session Merging & Global Insight
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After a single Session is processed, the system automatically performs cross-session merging and accumulation.

**Entity/Event Cross-Session Merging (CrossSessionCorefManager)**:
  - For newly extracted entities, candidate matches across sessions are found via multi-signal search
  - An LLM judge determines whether two entities are coreferential (same real-world entity)
  - Merged canonical entities are shared across sessions, with ``COREF`` edges from each mentioning dialogue unit
  - Events follow the same pattern: high similarity + close timing → merged as the same event

**Global Insight Accumulation (GlobalInsightManager)**:
  - The first session's insights become the initial global insight value
  - Subsequent session insights are incrementally merged with the existing global insight via LLM
  - LLM merge strategy: fuse overlapping content, append unique content, maintain diversity
  - The global insight is persisted as a single MemoryUnit (``global_insight_v1``), re-embedded after each merge
  - On LLM merge failure, falls back to simple union concatenation

.. _pipeline-stage3-en:

Stage 3: holistic_retrieve() — Unified Retrieval
-------------------------------------------------

The retrieval pipeline after construction is complete:

**1. Query vectorization**: Generate a dense embedding for the query text

**2. Group recall**: The retrieval request is distributed to four retrieval groups, with each group operating independently

   - **BASE**: Raw dialogues + procedural summaries
   - **ENTITY**: Knowledge entities
   - **EVENT**: Episodic events
   - **SUMMARY**: Episodic/knowledge/emotional/insight summaries

**3. Three-path recall**: Each group independently executes dense vector, BM25 keyword, and sparse vector retrieval

**4. RRF fusion**: Merges three-path results using Reciprocal Rank Fusion

**5. BFS graph expansion**: Expands the candidate set along graph relations starting from top-K fusion results (parameters: ``bfs_expansion_per_seed`` / ``bfs_expansion_hops``)

**6. Global reranking**: Cross-Encoder reranks all candidates and returns the final ``SearchHit`` list

.. note::

   If ``holistic_retrieve()`` detects that high-level memory is empty (``build_high_level()`` has not been run),
   and the parameter ``auto_build_if_empty=True`` (default), the system automatically triggers
   ``build_high_level("auto")`` to ensure retrieval does not return empty results.

.. _diagram-placeholder-en-2:

.. admonition:: Placeholder: Three-Stage Retrieval Pipeline Overview
   :class: hint

   **What this diagram should show:**

   The Stage 3 retrieval pipeline: **Query Input → Vectorization → Four Parallel Group Recalls (BASE/ENTITY/EVENT/SUMMARY) → Three-Path Retrieval per Group (Dense/BM25/Sparse) → RRF Fusion → BFS Graph Expansion → Cross-Encoder Reranking → SearchHit Output**.

   A left-to-right pipeline layout is recommended, with different colors distinguishing each recall group and fusion step.

Build Report (BuildReport)
---------------------------

``build_high_level()`` returns a ``BuildReport`` object with the following fields:

- ``status``: Build status (success / partial / failed)
- ``mode``: Build mode used (auto / force)
- ``sessions_processed``: Number of sessions processed
- ``units_processed``: Number of units processed
- ``duration_seconds``: Total elapsed time
- ``token_usage``: LLM token consumption stats (prompt_tokens / completion_tokens / total_tokens)
- ``warnings``: List of warnings during the build
- ``error_message``: Error message on failure

You can also query cumulative token usage at any time via ``system.get_token_usage()``.

Multi-Perspective Memory Representation
-----------------------------------------

This section shows how each memory type is represented in the system — node structure, edge types, and graph examples.

.. note::

   All multi-perspective memory units below are automatically constructed by the system during
   ``build_high_level()``. Users should never create them manually. The node structures shown here
   are for understanding the system's internal memory representation patterns only.

.. _representation-base-memory-en:

Base Memory
^^^^^^^^^^^

Base memory stores raw dialogue units — the system's foundational data source.

**Node representation**: Base dialogue node

.. code-block:: python

   MemoryUnit(
       uid="dialogue_msg_001",
       raw_data={
           "text_content": "I went to Beijing yesterday and visited the Forbidden City and the Great Wall.",
           "speaker": "user",
           "role": "user",
       },
       metadata={
           "timestamp": "2024-01-15T10:00:00",
           "space_name": "root_base_memory_msg_0_25",
           "session_id": "session_001",
           "chunk_id": "chunk_0",
       },
       embedding=[0.1, 0.2, ..., 0.768],
       sparse_embedding={12: 0.5, 45: 0.3, ...},
   )

**Edge types**:

- ``PRECEDES``: Temporal edge (preceding dialogue points to following)
- ``FOLLOWS``: Temporal edge (following dialogue points to preceding)
- ``SEMANTIC_SIMILAR``: Semantic similarity edge (based on vector similarity threshold)

.. _diagram-placeholder-en-3:

.. admonition:: Placeholder: Base Memory Graph Example
   :class: hint

   **What this diagram should show:**

   Three dialogue units (dialogue_001/002/003) with PRECEDES/FOLLOWS temporal edges and SEMANTIC_SIMILAR edges. A simple directed graph with text summaries on nodes.

.. _representation-entity-relation-en:

Entity Relation
^^^^^^^^^^^^^^^

The entity relation perspective extracts named entities from dialogue and establishes semantic relationships between them.

**Node representation**: Entity node

.. code-block:: python

   MemoryUnit(
       uid="entity_beijing_001",
       raw_data={
           "text_content": "Beijing",
           "entity_name": "Beijing",
           "entity_type": "Place",
           "description": "Capital of China, political and cultural center",
       },
       metadata={
           "space_name": "root_knowledge_entity_msg_0_25",
           "entity_type": "Place",
           "entity_id": "beijing_001",
           "session_id": "session_001",
           "mentions": ["dialogue_msg_001", "dialogue_msg_002"],
       },
       embedding=[...],
   )

**Edge types**:

- ``RELATED_TO``: General relationship edge (subtypes: hometown, lives_in, works_at, located_in, part_of)
- ``COREF``: Coreference edge (dialogue unit → entity), indicating the dialogue mentions this entity
- ``ALIAS_OF``: Alias edge (entity alias relationship)
- ``EVIDENCED_BY``: Provenance edge (entity points to the original dialogue that mentions it)

.. note::

   **Coreference Resolution Mechanism**:

   In the current coreference pipeline, ``COREF`` edges are directly established between
   **base dialogue units → canonical entities**, indicating that the dialogue unit "mentions" that entity.
   When multiple dialogue units refer to the same canonical entity, a complete coreference chain is formed through:

   1. EVIDENCED_BY edges (entity → dialogue): which dialogues support/mention the entity
   2. COREF edges (dialogue → entity): which entity the dialogue specifically refers to

   This design enables cross-session coreference traversal via graph queries to locate all related dialogue units.

.. _diagram-placeholder-en-4:

.. admonition:: Placeholder: Entity Relation Graph Example
   :class: hint

   **What this diagram should show:**

   Entity nodes (Beijing/Forbidden City/Great Wall) with RELATED_TO edges between them, and COREF/EVIDENCED_BY edges from dialogue units to entities. Highlight spatial relationships (located_in/part_of) and the coreference resolution chain.

.. _representation-event-causal-en:

Event Causal
^^^^^^^^^^^^

The event causal perspective extracts events from dialogue and builds causal chains between them.

**Node representation**: Event node

.. code-block:: python

   MemoryUnit(
       uid="event_visit_beijing_001",
       raw_data={
           "text_content": "User went to Beijing on a business trip",
           "event_type": "action_event",
           "participants": ["user", "Beijing"],
           "time": "2024-01-14",
       },
       metadata={
           "space_name": "root_episodic_event_msg_0_25",
           "event_type": "action_event",
           "event_id": "visit_beijing_001",
           "session_id": "session_001",
           "evidence_uids": ["dialogue_msg_001"],
       },
       embedding=[...],
   )

**Edge types**:

- ``CAUSES``: Causal edge (event A causes event B)
- ``CAUSED_BY``: Reverse causal edge (event B is caused by event A)
- ``INVOLVES``: Event-entity edge (subtypes: participant, location, organizer, victim)
- ``PRECEDES`` / ``FOLLOWS``: Temporal edges (event ordering)
- ``EVIDENCED_BY``: Provenance edge (event points to supporting original dialogue)

.. _diagram-placeholder-en-5:

.. admonition:: Placeholder: Event Causal Graph Example
   :class: hint

   **What this diagram should show:**

   Event nodes (Business Trip to Beijing/Visit Forbidden City/Learn About History) with CAUSES causal chain and PRECEDES temporal edges, plus INVOLVES edges from events to entities (Beijing/Forbidden City). Show the complete event-entity-dialogue three-layer provenance chain.

.. _representation-emotional-summary-en:

Emotional Summary
^^^^^^^^^^^^^^^^^

The emotional summary perspective captures the user's emotional states and attitudes expressed in dialogue.

**Node representation**: Emotional summary node

.. code-block:: python

   MemoryUnit(
       uid="emotional_summary_msg_0_25",
       raw_data={
           "text_content": '{"user_preferences": ["Enjoys historical and cultural sites", "Prefers in-depth travel"], "emotional_reactions": ["Excited", "Proud", "Deeply impressed"], "behavioral_patterns": ["Actively learns about site backgrounds", "Records visit experiences in detail"]}',
           "summary_type": "emotional",
       },
       metadata={
           "space_name": "root_emotional_msg_0_25",
           "summary_type": "emotional",
           "session_id": "session_001",
           "evidence_uids": ["dialogue_msg_001", "dialogue_msg_002"],
       },
       embedding=[...],
   )

**Edge types**:

- ``EVIDENCED_BY``: Provenance edge (emotional summary points to supporting raw dialogues)
- ``SEMANTIC_SIMILAR``: Semantic similarity edge with other emotional summaries

.. _representation-episodic-summary-en:

Episodic Summary
^^^^^^^^^^^^^^^^

The episodic summary perspective creates high-level overviews of events within a session for fast retrieval.

**Node representation**: Episodic summary node

.. code-block:: python

   MemoryUnit(
       uid="episodic_summary_msg_0_25",
       raw_data={
           "text_content": '{"timeline": ["2024-01-14 Arrived in Beijing", "2024-01-15 Visited the Forbidden City", "2024-01-16 Toured the Great Wall"], "key_people": ["User"], "main_events": ["Beijing business trip", "Visited Forbidden City", "Toured Great Wall"], "location_info": ["Beijing", "Forbidden City", "Great Wall"], "event_relationships": ["Business trip led to visiting the Forbidden City", "After Forbidden City, toured the Great Wall"]}',
           "summary_type": "episodic",
           "time_range": "2024-01-14 ~ 2024-01-16",
           "key_events": ["Beijing business trip", "Visited Forbidden City", "Toured Great Wall"],
           "key_entities": ["Beijing", "Forbidden City", "Great Wall"],
       },
       metadata={
           "space_name": "root_episodic_summary_msg_0_25",
           "summary_type": "episodic",
           "session_id": "session_001",
           "evidence_uids": ["event_visit_beijing_001", "event_visit_gugong_001"],
       },
       embedding=[...],
   )

**Edge types**:

- ``EVIDENCED_BY``: Provenance edge (episodic summary points to supporting events/dialogues)

.. _representation-knowledge-summary-en:

Knowledge Summary
^^^^^^^^^^^^^^^^^

The knowledge summary perspective distills facts and knowledge points from dialogue.

**Node representation**: Knowledge summary node

.. code-block:: python

   MemoryUnit(
       uid="knowledge_summary_msg_0_25",
       raw_data={
           "text_content": '{"core_concepts": ["Beijing - Capital of China", "Forbidden City - Imperial Palace", "Great Wall - Military Defense Structure"], "key_facts": ["Beijing is China\'s political and cultural center", "The Forbidden City is located on Beijing\'s central axis", "The Great Wall is an ancient Chinese defense structure"], "techniques_methods": [], "prerequisites_knowledge": ["Basic Chinese history"], "related_concepts": ["Ming and Qing Dynasty History", "Ancient Architecture", "World Cultural Heritage"]}',
           "summary_type": "knowledge",
           "facts": [
               {"subject": "Beijing", "predicate": "is", "object": "capital of China"},
               {"subject": "Forbidden City", "predicate": "is", "object": "Ming and Qing imperial palace"},
               {"subject": "Great Wall", "predicate": "is", "object": "ancient military defense structure"},
           ],
       },
       metadata={
           "space_name": "root_knowledge_summary_msg_0_25",
           "summary_type": "knowledge",
           "session_id": "session_001",
           "evidence_uids": ["dialogue_msg_001", "dialogue_msg_002"],
       },
       embedding=[...],
   )

**Edge types**:

- ``EVIDENCED_BY``: Provenance edge (knowledge summary points to supporting raw dialogues/entities)

.. _representation-procedural-summary-en:

Procedural Summary
^^^^^^^^^^^^^^^^^^

The procedural summary perspective extracts steps, methods, and techniques from dialogue.

**Node representation**: Procedural summary node

.. code-block:: python

   MemoryUnit(
       uid="procedural_summary_msg_0_25",
       raw_data={
           "text_content": '{"process_name": ["Forbidden City Visit Flow"], "key_steps": ["Book tickets online in advance", "Enter through the Meridian Gate and follow the central axis", "Focus on the Hall of Supreme Harmony, Palace of Heavenly Purity, and Imperial Garden", "Approximately 2-3 hours total"], "decision_points": ["Whether to hire a guide", "Choose route (highlights / full tour)"], "preconditions": ["Reserve tickets in advance", "Bring ID"], "expected_outcomes": ["Complete main attraction visit", "Learn about Forbidden City history and culture"], "optimization_opportunities": ["Avoid peak holiday periods", "Use audio guide app"]}',
           "summary_type": "procedural",
           "steps": [
               "Book tickets online in advance",
               "Enter through the Meridian Gate, follow the central axis",
               "Focus on Hall of Supreme Harmony, Palace of Heavenly Purity, Imperial Garden",
               "Approximately 2-3 hours total",
           ],
       },
       metadata={
           "space_name": "root_procedural_msg_0_25",
           "summary_type": "procedural",
           "session_id": "session_001",
           "evidence_uids": ["dialogue_msg_002"],
       },
       embedding=[...],
   )

**Edge types**:

- ``EVIDENCED_BY``: Provenance edge (procedural summary points to supporting raw dialogues)

.. _representation-insights-en:

Insights
^^^^^^^^

The insights perspective distills deep patterns and discoveries from dialogue.

**Node representation**: Insight node

.. code-block:: python

   MemoryUnit(
       uid="insight_cultural_interest_001",
       raw_data={
           "text_content": '{"pattern_recognition": ["User has sustained interest in Chinese historical and cultural sites", "Prefers deep cultural experiences over superficial tourism"], "causal_relationships": ["Historical and cultural interest drives proactive background research"], "predictive_insights": ["May be interested in similar sites like the Summer Palace and Temple of Heaven", "May ask for more historical details in the future"], "behavioral_characteristics": ["Records visit experiences in detail", "Actively learns about historical and cultural context"], "optimization_recommendations": ["Recommend other Beijing historical sites (Summer Palace, Temple of Heaven, Old Summer Palace)", "Provide in-depth guided tour services"], "risk_warnings": []}',
           "insight_type": "preference",
           "confidence": 0.78,
           "actionable_suggestion": "Recommend Beijing historical and cultural sites such as the Summer Palace, Temple of Heaven, and Old Summer Palace",
       },
       metadata={
           "space_name": "root_insights_msg_0_25",
           "insight_type": "preference",
           "session_id": "session_001",
           "evidence_uids": [
               "dialogue_msg_001",
               "emotional_summary_msg_0_25",
               "episodic_summary_msg_0_25",
           ],
       },
       embedding=[...],
   )

**Edge types**:

- ``EVIDENCED_BY``: Provenance edge (insight points to supporting raw dialogues/summaries)
- ``SEMANTIC_SIMILAR``: Semantic similarity edge with other insights

Provenance System
^^^^^^^^^^^^^^^^^

.. note::

   **Provenance Notes**:

   - High-level summaries (episodic, knowledge, procedural, emotional) have ``EVIDENCED_BY`` edges
     pointing directly to base dialogue units, showing that these summaries are distilled from raw dialogue data
   - Insight memory units have ``EVIDENCED_BY`` edges pointing to all four summary categories,
     showing that insights synthesize multi-perspective information into deeper discoveries
   - Entity relations and event causal structures also obtain evidence support from base memory via ``EVIDENCED_BY`` edges

.. _diagram-placeholder-en-6:

.. admonition:: Placeholder: Global Multi-Perspective Memory Graph Overview
   :class: hint

   **What this diagram should show:**

   A comprehensive view combining Base Memory, Entity Relations, Event Causality, four Summary types (Emotional/Episodic/Knowledge/Procedural), and Insights into one diagram. Use color or regions to distinguish each perspective subgraph, clearly showing the EVIDENCED_BY provenance edges and COREF coreference edges between layers.

   Core information layers (bottom to top):

   1. **Base Memory Layer**: Raw dialogue units, PRECEDES/FOLLOWS temporal edges
   2. **Entity/Event Layer**: Extracted from dialogue, RELATED_TO/CAUSES/INVOLVES edges
   3. **Summary Layer**: Four summary types, EVIDENCED_BY pointing to base memory
   4. **Insight Layer**: Deep insights, EVIDENCED_BY pointing to summary layer

   Cross-layer edges clearly illustrate the progressive abstraction: "raw data → structured facts → distilled summaries → deep insights."

Space Hierarchy
---------------

Each session's high-level memory is organized in the following space hierarchy:

.. code-block::

   root
   ├── base_memory_{suffix}          # Base memory (raw units)
   │   └── session_{session_id}      # Per-session spaces (dynamically created)
   └── high_level_memory_{suffix}    # High-level memory
       ├── episodic_{suffix}         # Episodic memory
       │   ├── episodic_summary      # Episodic summaries
       │   └── episodic_event        # Episodic events (canonical)
       ├── knowledge_{suffix}        # Knowledge memory
       │   ├── knowledge_summary     # Knowledge summaries
       │   └── knowledge_entity      # Knowledge entities (canonical)
       ├── emotional_{suffix}        # Emotional memory
       ├── procedural_{suffix}       # Procedural memory
       └── insights_{suffix}         # Insight memory (global, continuously updated)

Where ``{suffix}`` is a unique identifier generated from the session's starting message index. For retrieval views and methods corresponding to each space, see :doc:`/shared/retrieval-reference`.
