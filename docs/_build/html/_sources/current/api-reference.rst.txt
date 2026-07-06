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
