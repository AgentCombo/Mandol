MemorySystem Complete Reference
=================================

MemorySystem is Mandol's unified entry point class.

Initialization
---------------

``MemorySystem(**kwargs)`` — Create with default configuration

.. code-block:: python

   system = MemorySystem(
       embedder=None,       # Optional[EmbeddingProvider]
       reranker=None,       # Optional[Reranker]
       llm_provider=None,   # Optional[LLMProvider]
   )

``MemorySystem.from_yaml_config(yaml_path, **override)`` — Create from YAML

.. code-block:: python

   system = MemorySystem.from_yaml_config("config.yaml")

``MemorySystem.load(directory, **override)`` — Load from directory (class method)

.. code-block:: python

   system = MemorySystem.load("./snapshot")

Data Management
-----------------

``add(unit: MemoryUnit) -> None``

``add_many(units: Sequence[MemoryUnit]) -> None``

``save(directory: str) -> None``

Memory Building
-----------------

``build_high_level(mode: str = "auto") -> BuildReport``

- ``mode="auto"``: Incremental (recommended)
- ``mode="force"``: Full rebuild

``BuildReport`` contains:

- ``sessions_processed: int``
- ``units_processed: int``

Retrieval Interfaces
---------------------

``holistic_retrieve(query, top_k=10, use_rerank=True) -> list[SearchHit]``

``retrieve_by_view(query, view, top_k=10, use_rerank=True) -> list[SearchHit]``

``retrieve_in_space(query, space_name, top_k=10, use_rerank=True) -> list[SearchHit]``

``search(query, top_k=10, use_rerank=True, use_graph_expansion=False, space_names=None, retriever_types=None) -> list[SearchHit]`` *(Planned)*

Maintenance Interfaces
-----------------------

``flush() -> None`` — Persist cache

Properties
-----------

- ``system.semantic_map`` — ``SemanticMapService``
- ``system.semantic_graph`` — ``SemanticGraphService``

Configuration Parameters (MemorySystemConfig 30+ fields)
---------------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Field
     - Default
     - Description
   * - ``chunk_max_tokens``
     - 512
     - Maximum tokens per chunk
   * - ``session_time_gap_seconds``
     - 1800
     - Time interval segmentation threshold
   * - ``session_check_interval``
     - 20
     - LLM detection interval
   * - ``session_max_pending``
     - 100
     - Maximum pending count
   * - ``similarity_threshold``
     - 0.7
     - Similarity edge threshold
   * - ``similarity_top_k``
     - 5
     - Vector retrieval recall count
   * - ``similarity_recent_window``
     - 20
     - Similarity computation window
   * - ``bfs_expansion_per_seed``
     - 3
     - BFS neighbors per seed
   * - ``bfs_expansion_hops``
     - 1
     - BFS hop count
   * - ``max_entities_per_llm``
     - 50
     - Entity deduplication candidate count
   * - ``max_events_per_llm``
     - 50
     - Event deduplication candidate count
   * - ``promote_threshold``
     - 100
     - Index upgrade threshold
