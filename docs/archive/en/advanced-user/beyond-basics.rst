Beyond Basics: What Advanced Users Can Do
============================================

As an advanced user, you need more than just the "add → build → retrieve" three-step workflow. You need a complete memory management toolbox.

Public Interface Overview
--------------------------

Beyond ``add/build_high_level/holistic_retrieve``, MemorySystem provides the following management interfaces:

.. code-block:: python

   system = MemorySystem.from_yaml_config("config.yaml")

   # === Space Management ===
   system.semantic_map.create_space("Support-UserA")
   system.semantic_map.attach_child_space("Support-UserA", "Session-20240301")
   spaces = system.semantic_map.list_spaces()

   # === Graph Management ===
   system.semantic_graph.add_relationship(uid_a, uid_b, "RELATED_TO")
   neighbors = system.semantic_graph.get_explicit_neighbors(uid_a)
   system.semantic_graph.delete_relationship(uid_a, uid_b, "RELATED_TO")

   # === Fine-grained Retrieval ===
   hits = system.retrieve_by_view("complaint content", view="knowledge")
   hits = system.retrieve_in_space("order status", space_name="Support-UserA")

   # === Status Monitoring ===
   print(system.monitor)                       # Compact one-line status
   stats = system.monitor.to_dict()            # Programmatic access

   # === State Maintenance ===
   system.flush()
   stats = system.semantic_map.count_units()

Retrieval Interface Overview
----------------------------

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
     - Filter by semantic perspective
   * - ``retrieve_in_space``
     - Public
     - Advanced
     - Determined by space_name
     - Same as above
     - Filter by space scope
   * - ``search_by_text``
     - Public
     - Developer
     - Determined by space_names
     - Dense vector
     - Direct vector retrieval
   * - ``search_by_text_with_rerank``
     - Public
     - Developer
     - Determined by space_names
     - Dense + Reranker
     - Vector retrieval + reranking
   * - ``bfs_expand_units``
     - Public
     - Developer
     - All
     - GraphStore (BFS)
     - Graph traversal expansion

For the complete retrieval interface reference (including signatures, parameters, view mapping table), see :doc:`/shared/retrieval-reference`.

retrieve_by_view View Mapping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 20 28

   * - view Value
     - Memory Level
     - Description
   * - ``base_memory``
     - BASE
     - Raw conversations/documents
   * - ``entity_relation``
     - ENTITY
     - Entities and relationships
   * - ``event_causal``
     - EVENT
     - Events and causality
   * - ``emotional``
     - SUMMARY
     - Emotional summaries
   * - ``episodic``
     - SUMMARY
     - Episodic summaries
   * - ``knowledge``
     - SUMMARY
     - Knowledge summaries
   * - ``procedural``
     - SUMMARY
     - Procedural summaries
   * - ``insights``
     - SUMMARY
     - Insights

Management Interfaces
---------------------

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Interface Type
     - Layer
     - Purpose
   * - ``create_space`` etc.
     - SemanticMapService (management)
     - Space CRUD management
   * - ``add_relationship`` etc.
     - SemanticGraphService (management)
     - Graph relationship CRUD management
   * - ``flush``
     - MemorySystem (management)
     - Persist cache to storage

Planned Retrieval Interfaces
-----------------------------

The following interfaces are designed but not yet implemented. Similar effects can currently be achieved by combining existing interfaces. For the complete design, see :doc:`/shared/retrieval-reference`.

.. list-table::
   :header-rows: 1
   :widths: 25 10 18 47

   * - Interface
     - Status
     - Memory Level
     - Core Value
   * - ``retrieve_event_causal_chain``
     - Planned
     - EVENT
     - Causal chain tracing, answering "why"
   * - ``retrieve_entity_subgraph``
     - Planned
     - ENTITY
     - Entity relationship panorama
   * - ``smart_quantized_query``
     - Planned
     - All
     - Maximize information density under token budget constraints
   * - ``retrieve_with_reasoning_path``
     - Planned
     - All
     - Explainable reasoning path
   * - ``retrieve_entity_timeline``
     - Planned
     - BASE+EVENT
     - Timeline perspective
   * - ``retrieve_session_context``
     - Planned
     - BASE
     - Session-level context restoration
   * - ``trace_evidence``
     - Planned
     - All→BASE
     - Top-down tracing (EVIDENCED_BY)
   * - ``trace_coref``
     - Planned
     - BASE→ENTITY/EVENT
     - Bottom-up coreference resolution (COREF)
   * - ``retrieve_summary_evidence_chain``
     - Planned
     - SUMMARY→BASE
     - Summary evidence chain
   * - ``retrieve_entity_involvement``
     - Planned
     - ENTITY+EVENT
     - All events an entity participates in (INVOLVES)

Next, choose the corresponding chapter to dive deeper based on your needs.
