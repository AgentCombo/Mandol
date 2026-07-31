Retrieval Interface Reference
===============================

This document lists all Mandol retrieval interfaces, categorized by layer. Interface statuses are:

- **Public**: Implemented, ready to use
- **Planned**: Designed but not yet implemented; similar effects can currently be achieved by combining existing interfaces
- **Experimental**: Implemented but API may change

Unified Retrieval Interfaces (MemorySystem Layer)
---------------------------------------------------

Called directly through the ``system`` object, these are the most commonly used retrieval entry points.

.. list-table::
   :header-rows: 1
   :widths: 22 8 10 14 14 32

   * - Interface
     - Status
     - For Users
     - Memory Level
     - Data Structure
     - Description
   * - ``holistic_retrieve`` / ``search``
     - Public
     - Basic/Advanced
     - BASE+ENTITY+EVENT+SUMMARY
     - Dense+BM25+Sparse+Graph+Reranker
     - One-line full-memory retrieval
   * - ``retrieve_by_view``
     - Public
     - Advanced
     - Determined by view
     - Same as above
     - Filter by semantic perspective (entity/event/summary)
   * - ``retrieve_in_space``
     - Public
     - Advanced
     - Determined by space_name
     - Same as above
     - Precise retrieval by space name
   * - ``retrieve_event_causal_chain``
     - Planned
     - Advanced/Developer
     - EVENT
     - GraphStore (CAUSES/CAUSED_BY)
     - Causal chain tracing, answering "why"
   * - ``smart_quantized_query``
     - Planned
     - Advanced/Developer
     - All
     - All + LLM compression
     - Maximize information density under token budget constraints
   * - ``retrieve_with_reasoning_path``
     - Planned
     - Advanced
     - All
     - HybridRetriever + weighted multi-hop graph expansion
     - Explainable reasoning path
   * - ``retrieve_entity_timeline``
     - Planned
     - Advanced
     - BASE+EVENT
     - UnitStore (timestamp sorting)
     - Timeline perspective
   * - ``retrieve_session_context``
     - Planned
     - Basic/Advanced
     - BASE
     - SessionManager + UnitStore
     - Session-level context restoration

holistic_retrieve / search
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   hits = system.holistic_retrieve("Where did Zhang San go?", top_k=10, use_rerank=True)
   # Or shorthand
   hits = system.search("Where did Zhang San go?")

Automatically searches all four memory levels (BASE / ENTITY / EVENT / SUMMARY), executing three-way recall → RRF fusion → BFS graph expansion → Reranker reranking.

retrieve_by_view
~~~~~~~~~~~~~~~~~

.. code-block:: python

   hits = system.retrieve_by_view("complaint content", view="entity_relation", top_k=10)

Filter by predefined semantic views. View parameter mapping:

.. list-table::
   :header-rows: 1
   :widths: 22 30 20 28

   * - view Value
     - Corresponding Space
     - Memory Level
     - Description
   * - ``base_memory``
     - root_base_memory
     - BASE
     - Raw conversations/documents
   * - ``entity_relation``
     - root_knowledge_entity
     - ENTITY
     - Entities and relationships
   * - ``event_causal``
     - root_episodic_event
     - EVENT
     - Events and causality
   * - ``emotional``
     - root_emotional
     - SUMMARY
     - Emotional summaries
   * - ``episodic``
     - root_episodic_summary
     - SUMMARY
     - Episodic summaries
   * - ``knowledge``
     - root_knowledge_summary
     - SUMMARY
     - Knowledge summaries
   * - ``procedural``
     - root_procedural
     - SUMMARY
     - Procedural summaries
   * - ``insights``
     - root_insights
     - SUMMARY
     - Insights

retrieve_in_space
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   hits = system.retrieve_in_space("order status", space_name="Support-UserA", top_k=10)

Retrieve within a specified memory space, suitable for scenarios where spaces are organized by business logic.

Planned Interfaces
~~~~~~~~~~~~~~~~~~~

**retrieve_event_causal_chain** — Event causal chain retrieval

.. warning::

   This interface is planned and not yet implemented. The signature below is subject to change.

Traces causal chains along CAUSES / CAUSED_BY edges, returning complete causes and effects.

.. code-block:: python

   # Planned — not yet available
   result = system.retrieve_event_causal_chain(
       "project delay",
       max_hops=3,
       direction="both",    # "forward" / "backward" / "both"
       top_k=5,
   )

**smart_quantized_query** — Smart quantized query

.. warning::

   This interface is planned and not yet implemented. The signature below is subject to change.

Produces optimally compact context under token budget constraints through three stages: query routing → smart denoising → cascade bin packing.

.. code-block:: python

   # Planned — not yet available
   result = system.smart_quantized_query(
       "Zhang San's recent business trip plans",
       max_context_tokens=2000,
       routing_strategy="auto",
       denoise_threshold=0.5,
       compression_ratio=0.3,
       top_k=20,
   )

**retrieve_with_reasoning_path** — Retrieval with reasoning path

.. warning::

   This interface is planned and not yet implemented. The signature below is subject to change.

Weighted multi-hop graph expansion based on SubgraphHopRetriever, returning results with complete reasoning paths.

**retrieve_entity_timeline** — Entity timeline retrieval

.. warning::

   This interface is planned and not yet implemented. The signature below is subject to change.

Returns all events and conversations related to a specified entity, sorted by time.

**retrieve_session_context** — Session context retrieval

.. warning::

   This interface is planned and not yet implemented. The signature below is subject to change.

Retrieves the complete context of a specified session, supporting expansion to adjacent sessions.

Service Layer Retrieval Interfaces (SemanticMapService)
---------------------------------------------------------

Called through ``system.semantic_map``, providing more fine-grained retrieval control.

.. list-table::
   :header-rows: 1
   :widths: 22 8 10 14 14 32

   * - Interface
     - Status
     - For Users
     - Memory Level
     - Data Structure
     - Description
   * - ``search_by_text``
     - Public
     - Developer
     - Determined by space_names
     - Dense vector
     - Text query, returns units + scores
   * - ``search_by_text_with_rerank``
     - Public
     - Developer
     - Determined by space_names
     - Dense + Reranker
     - Text query + reranking
   * - ``search_by_vector``
     - Public
     - Developer
     - Determined by space_names
     - Dense vector
     - Provide your own embedding vector
   * - ``search_in_space``
     - Public
     - Developer
     - Determined by space_name
     - Adaptive index
     - In-space retrieval with candidate set filtering
   * - ``get_units_in_spaces``
     - Public
     - Developer
     - Determined by space_names
     - UnitStore
     - Exact space query, no similarity ranking
   * - ``get_unit``
     - Public
     - Developer
     - All
     - UnitStore
     - Exact lookup of single unit by UID
   * - ``list_units``
     - Public
     - Developer
     - All
     - UnitStore
     - Return all memory units

Graph Traversal Interfaces (SemanticGraphService)
---------------------------------------------------

Called through ``system.semantic_graph`` or ``system.graph``, providing graph structure traversal, neighbor discovery, and hierarchical traceability.

.. list-table::
   :header-rows: 1
   :widths: 22 8 10 14 14 32

   * - Interface
     - Status
     - For Users
     - Memory Level
     - Data Structure
     - Description
   * - ``get_explicit_neighbors``
     - Public
     - Developer
     - All
     - GraphStore
     - Get explicit neighbors (nodes with direct edges)
   * - ``get_implicit_neighbors``
     - Public
     - Developer
     - All
     - Vector index
     - Get implicit neighbors (embedding-similar but no edge)
   * - ``bfs_expand_units``
     - Public
     - Developer
     - All
     - GraphStore (BFS)
     - BFS graph expansion
   * - ``get_relationship``
     - Public
     - Developer
     - All
     - GraphStore
     - Query specific relationship edge
   * - ``SubgraphHopRetriever.search``
     - Experimental
     - Developer
     - All
     - HybridRetriever + weighted multi-hop graph expansion
     - Multi-hop reasoning retrieval with reasoning_path
   * - ``retrieve_entity_subgraph``
     - Planned
     - Advanced
     - ENTITY
     - GraphStore (RELATED_TO/ALIAS_OF)
     - Entity relationship panorama
   * - ``trace_evidence``
     - Planned
     - Advanced/Developer
     - All→BASE
     - GraphStore (EVIDENCED_BY)
     - Top-down tracing
   * - ``trace_coref``
     - Planned
     - Advanced/Developer
     - BASE→ENTITY/EVENT
     - GraphStore (COREF)
     - Bottom-up coreference resolution
   * - ``retrieve_summary_evidence_chain``
     - Planned
     - Advanced
     - SUMMARY→BASE
     - GraphStore (EVIDENCED_BY + associations)
     - Summary evidence chain
   * - ``retrieve_entity_involvement``
     - Planned
     - Advanced
     - ENTITY+EVENT
     - GraphStore (INVOLVES)
     - All events an entity participates in

Recommendations by User Level
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 15 30 55

   * - User Level
     - Recommended Interface
     - Description
   * - Basic
     - ``system.search(query)``
     - Single-call retrieval across all memory levels
   * - Advanced
     - ``system.retrieve_by_view(query, view="...")``
     - Retrieve specific category of memories by semantic view name
   * - Advanced
     - ``system.retrieve_in_space(query, space_name="...")``
     - Precise retrieval by space name
   * - Developer
     - ``system.semantic_map.search_by_text_with_rerank(...)``
     - Direct vector search + reranking on semantic map
   * - Developer
     - ``system.semantic_map.search_in_space(...)``
     - Space-level search with candidate set filtering
   * - Developer
     - ``system.graph.bfs_expand_units(...)``
     - Graph traversal expansion
   * - Developer
     - ``SubgraphHopRetriever.search(...)``
     - Experimental multi-hop reasoning retrieval
