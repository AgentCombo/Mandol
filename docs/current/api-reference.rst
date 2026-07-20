Current API Reference
=====================

Top-level package
-----------------

``mandol.__all__`` currently exports:

* ``MemoryUnit``
* ``MemorySpace``
* ``MemorySpaceRegistry``
* ``TowerSpace``
* ``SemanticMap``
* ``SemanticGraph``
* ``cluster_nodes``
* ``ClusterMethod``
* ``__version__``

Core helpers
------------

``mandol.core`` also provides:

* ``create_semantic_map(text_model=None, image_model=None, embedding_dim=None,
  preset=None, **kwargs)``
* ``create_semantic_graph(semantic_map_instance=None, preset=None, **kwargs)``
* ``create_memory_unit(uid, content, content_type="text", metadata=None,
  **kwargs)``
* ``create_memory_space(name, faiss_index_type=None)``
* ``get_default_core_config()``
* ``get_model_preset(preset_name)``
* ``get_core_component_status()``
* ``validate_embedding_dim(embedding_dim)``
* ``get_recommended_faiss_index(num_vectors, embedding_dim)``

Retrieval package
-----------------

``mandol.retrieval`` exports:

* ``BaseRetriever``
* ``RetrievalInterface``
* ``MultiRetrievalInterface``
* ``RetrievalMethod``
* ``RetrievalResult``
* ``parse_retrieval_methods``
* ``parse_weights``
* ``MultiRetriever``
* ``BM25Retriever``
* ``SPLADERetriever``
* ``CosineRetrieverAdapter``
* ``GraphContext``
* ``PathInfo``
* ``GraphContextExpander``
* ``RerankerManager``
* ``QueryBundle``
* ``ScoreFusion``

Three-tower retrieval
---------------------

``mandol.triple_retrieval`` exports:

* ``TripleTowerRetriever``
* ``TripleTowerConfig``
* ``TripleTowerSearchMode``
* ``RerankStrategy``
* ``TripleTowerResult``
* ``HierarchicalTowerResult``
* ``GraphTowerResult``
* ``EpisodicTowerResult``
* ``SecondStageRerankResult``
* ``TowerRetrievalStats``
* ``create_triple_tower_retriever``

Other retrieval-facing packages
-------------------------------

``mandol.hierarchical`` exposes hierarchical retrieval types and clustering
helpers for already-built hierarchical memory spaces.

``mandol.entity_relation`` exposes ``EntityRelationGraphRetriever``,
``GraphRetrievalConfig`` and ``create_graph_retriever``.

``mandol.episodic`` exposes ``EpisodicMemoryRetriever``,
``EpisodicRetrievalConfig`` and ``create_episodic_retriever``.

High-level memory construction
------------------------------

``mandol.auto_builder`` exposes the maintained high-level construction path:

* ``MemoryOrchestrator`` and ``OrchestratorConfig``
* ``HighLevelMemoryBuilder`` and ``HighLevelMemoryBuildConfig``
* ``build_high_level_memory``
* ``HierarchicalAutoBuilder``, ``EntityRelationAutoBuilder`` and
  ``EpisodicAutoBuilder`` with their configuration and result types
* ``PipelineStrategy`` and ``STYLE_STRATEGIES``

Routing and quantification
--------------------------

``mandol.memory_router`` exposes ``LocomoTowerRouter``,
``LongMemEvalTowerRouter`` and their routing configuration and category types.
These policies are used by the paper's router + quantification workflows.

``mandol.quantification`` exposes ``SemanticQuantifier``, ``QueryExpander``,
confidence-aware pruning, cascade pruning and the associated factory functions.

Providers, storage and clustering
---------------------------------

``mandol.llm`` provides ``LLMClient``, ``LocalLLMClient``, provider/model
registries and client factories. ``mandol.storage`` provides ``DuckDBOperator``
and ``TieredStorageManager``. ``mandol.cluster`` provides the Leiden and DBSCAN
clusterer interfaces used by the builders.
