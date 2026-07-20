Retrieval Interface Classification
===================================

.. note::

   This document has been migrated to :doc:`/shared/retrieval-reference`. This page will be removed in a future version, please update your bookmarks.

This section describes the retrieval interface classification of the Mandol memory system, progressing from low-level data structures to multi-perspective memory, and then to unified full-memory retrieval.

Interface Naming Convention
---------------------------

- **Public interfaces**: No prefix, for direct user invocation (e.g., ``get_unit``, ``holistic_retrieve``)
- **Internal interfaces**: Underscore prefix ``_``, for internal system use, not exposed to users (e.g., ``_bfs_expand_units``)

.. _en-retrieval-memory-unit:

Interfaces for MemoryUnit
-------------------------

The following interfaces are provided by ``SemanticMap`` for accessing and managing individual memory units.

get_unit
^^^^^^^^

Get a specific memory unit by UID.

.. code-block:: python

   unit = system.semantic_map.get_unit("dialogue_001")

Parameters:

- ``uid``: Memory unit unique identifier

Returns:

- ``MemoryUnit``: The corresponding memory unit, or ``None`` if not found

get_all_units
^^^^^^^^^^^^^

Get all memory units.

.. code-block:: python

   units = system.semantic_map.get_all_units()

Returns:

- ``List[MemoryUnit]``: List of all memory units

filter_memory_units
^^^^^^^^^^^^^^^^^^^

Filter memory units by conditions.

.. code-block:: python

   filtered = system.semantic_map.filter_memory_units(
       candidate_units=None,
       filter_condition=lambda u: u.raw_data.get("speaker") == "user",
       ms_names=["root_base_memory"],
       recursive=True,
   )

Parameters:

- ``candidate_units``: Candidate unit set (None means all units)
- ``filter_condition``: Filter condition function
- ``ms_names``: Space name list to search
- ``recursive``: Whether to recursively search subspaces

Returns:

- ``List[MemoryUnit]``: List of units matching the filter condition

.. _en-retrieval-semantic-map:

Interfaces for SemanticMap
--------------------------

search
^^^^^^

**Unified semantic retrieval interface** supporting multiple retriever types and custom retriever combinations.

.. code-block:: python

   results = system.semantic_map.search(
       "query",
       k=10,
       retriever_type="dense"
   )

   results = system.semantic_map.search(
       "query",
       k=10,
       retrievers=["dense", "bm25", "sparse"]
   )

Parameters:

- ``query``: Query text
- ``k``: Number of results to return
- ``retriever_type``: Single retriever type (``"dense"``, ``"bm25"``, ``"sparse"``)
- ``retrievers``: List of retriever types for multi-path retrieval

Returns:

- ``List[SearchHit]``: Search results, each containing ``unit``, ``score``, ``retriever_name``

search_hybrid
^^^^^^^^^^^^^

Comprehensive hybrid retrieval (Dense + BM25 + Sparse multi-way recall + RRF fusion + optional graph expansion).

.. code-block:: python

   results = system.semantic_map.search_hybrid(
       "query",
       top_k=10,
       use_graph_expansion=True,
       bfs_depth=2,
       rerank=True
   )

Parameters:

- ``query``: Query text
- ``top_k``: Number of results to return
- ``use_graph_expansion``: Whether to use graph expansion
- ``bfs_depth``: BFS expansion depth
- ``rerank``: Whether to use reranking

Returns:

- ``List[SearchHit]``: Search results after fusion and optional reranking

.. _en-retrieval-semantic-graph:

Interfaces for SemanticGraph
----------------------------

get_explicit_neighbors
^^^^^^^^^^^^^^^^^^^^^^

Get explicit relationship neighbors.

.. code-block:: python

   neighbors = system.graph.get_explicit_neighbors(
       uids=["entity_001"],
       rel_type="RELATED_TO",
       direction="both"
   )

Parameters:

- ``uids``: List of seed node UIDs
- ``rel_type``: Relationship type filter (None means all types)
- ``direction``: Direction (``"both"``, ``"outgoing"``, ``"incoming"``)

Returns:

- ``Dict[Uid, List[Uid]]``: Mapping from seed node to neighbor list

get_implicit_neighbors
^^^^^^^^^^^^^^^^^^^^^^

Get implicit semantic neighbors.

.. code-block:: python

   neighbors = system.graph.get_implicit_neighbors(
       uids=["entity_001"],
       top_k=5,
       ms_names=["root_knowledge_entity"]
   )

Parameters:

- ``uids``: List of seed node UIDs
- ``top_k``: Number of neighbors to return
- ``ms_names``: Space name list to search

Returns:

- ``Dict[Uid, List[Tuple[Uid, float]]]``: Mapping from seed node to (neighbor, similarity) list

get_edges_of_unit
^^^^^^^^^^^^^^^^^

Get all relationship edges of a unit.

.. code-block:: python

   edges = system.graph.get_edges_of_unit(
       uid="entity_001",
       rel_type="RELATED_TO",
       direction="outgoing"
   )

Parameters:

- ``uid``: Memory unit UID
- ``rel_type``: Relationship type filter
- ``direction``: Direction filter

Returns:

- ``List[Dict]``: List of edge information dictionaries

search_graph_relations
^^^^^^^^^^^^^^^^^^^^^^

Search graph relationship edges.

.. code-block:: python

   relations = system.graph.search_graph_relations(
       seed_nodes=["entity_001"],
       relation_types=["RELATED_TO", "CAUSES"],
       max_depth=2,
       limit=50
   )

Parameters:

- ``seed_nodes``: List of seed node UIDs
- ``relation_types``: Relationship types to search
- ``max_depth``: Maximum search depth
- ``limit``: Maximum number of results

Returns:

- ``List[Dict]``: List of relationship information

.. _en-retrieval-holistic:

Unified Full-Memory Retrieval Interface
---------------------------------------

holistic_retrieve
^^^^^^^^^^^^^^^^^

Full-memory retrieval interface, the most comprehensive retrieval method in the system.

.. code-block:: python

   hits = system.holistic_retrieve("query", top_k=5)

**Internal Pipeline**:

1. Get 4 retrieval groups: BASE / ENTITY / EVENT / SUMMARY
2. Each group independently executes: Dense + BM25 + Sparse three-way recall → RRF fusion → BFS expansion
3. All candidates merged
4. Cross-Encoder Reranker global reranking

.. mermaid::

   graph TB
       Q[Query] --> G1[BASE Group]
       Q --> G2[ENTITY Group]
       Q --> G3[EVENT Group]
       Q --> G4[SUMMARY Group]
       
       G1 --> R1[Dense + BM25 + Sparse]
       G2 --> R2[Dense + BM25 + Sparse]
       G3 --> R3[Dense + BM25 + Sparse]
       G4 --> R4[Dense + BM25 + Sparse]
       
       R1 --> F1[RRF Fusion]
       R2 --> F2[RRF Fusion]
       R3 --> F3[RRF Fusion]
       R4 --> F4[RRF Fusion]
       
       F1 --> B1[BFS Expansion]
       F2 --> B2[BFS Expansion]
       F3 --> B3[BFS Expansion]
       F4 --> B4[BFS Expansion]
       
       B1 --> M[Merge All Candidates]
       B2 --> M
       B3 --> M
       B4 --> M
       
       M --> RK[Cross-Encoder Reranker]
       RK --> O[Top-K Results]

Parameters:

- ``query``: Query text
- ``top_k``: Number of results to return (default 5)

Returns:

- ``List[SearchHit]``: Search results, each containing ``unit``, ``final_score``, ``retriever_name``

retrieve_in_space
^^^^^^^^^^^^^^^^^

Execute full-memory retrieval pipeline within a specified space.

.. code-block:: python

   hits = system.retrieve_in_space(
       "query",
       space_name="root_knowledge_entity",
       top_k=5
   )

Parameters:

- ``query``: Query text
- ``space_name``: Space name to search within
- ``top_k``: Number of results to return

Returns:

- ``List[SearchHit]``: Search results

retrieve_by_view
^^^^^^^^^^^^^^^^

Retrieve by multi-perspective category.

Available views: ``base_memory``, ``entity_relation``, ``event_causal``, ``emotional``, ``episodic``, ``knowledge``, ``procedural``, ``insights``

.. code-block:: python

   events = system.retrieve_by_view("What happened?", view="event_causal", top_k=5)
   entities = system.retrieve_by_view("Who is involved?", view="entity_relation", top_k=5)

Parameters:

- ``query``: Query text
- ``view``: Perspective category name
- ``top_k``: Number of results to return

Returns:

- ``List[SearchHit]``: Search results

Retrieval Module Public Interfaces
----------------------------------

HybridRetriever
^^^^^^^^^^^^^^^

Dense + BM25 + Sparse three-way recall → RRF fusion → BFS expansion → reranking.

.. code-block:: python

   from mandol.retrieval.pipeline import HybridRetriever

   retriever = HybridRetriever(
       semantic_map=system.semantic_map,
       graph=system.graph,
       reranker=system.reranker,
   )
   results = retriever.retrieve("query", top_k=10)

BM25Retriever
^^^^^^^^^^^^^

BM25 keyword retrieval for precise keyword matching scenarios.

.. code-block:: python

   from mandol.retrieval.bm25 import BM25Retriever

   retriever = BM25Retriever(semantic_map=system.semantic_map)
   results = retriever.retrieve("query", top_k=10)

SparseRetriever
^^^^^^^^^^^^^^^

Sparse vector retrieval using SPLADE and similar sparse vector representations.

.. code-block:: python

   from mandol.retrieval.sparse import SparseRetriever

   retriever = SparseRetriever(semantic_map=system.semantic_map)
   results = retriever.retrieve("query", top_k=10)

SubgraphHopRetriever
^^^^^^^^^^^^^^^^^^^^

Subgraph hop retrieval for cross-session/multi-hop QA scenarios.

.. code-block:: python

   from mandol.retrieval.subgraph_hop import SubgraphHopRetriever

   retriever = SubgraphHopRetriever(
       semantic_map=system.semantic_map,
       graph=system.graph,
   )
   results = retriever.retrieve("query", top_k=10)

Retrieval Strategy Comparison
-----------------------------

| Strategy | Applicable Scenarios | Advantages | Disadvantages |
|----------|---------------------|------------|---------------|
| Dense | Semantic similarity queries | Understands semantics, handles synonyms | May miss precise keyword matches |
| BM25 | Precise keyword matching | Exact keyword recall | Cannot understand semantics |
| Sparse | Expanded term matching | Balances semantics and precision | Higher computational cost |
| Hybrid | General scenarios | Multi-path recall, high coverage | Higher computational cost |
| SubgraphHop | Cross-session/multi-hop QA | Graph structure support | Depends on graph quality |
