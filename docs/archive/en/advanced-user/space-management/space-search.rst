In-Space Retrieval
====================

Retrieve within a specified space — faster and more focused than global retrieval.

Basic Usage
------------

.. code-block:: python

   hits = system.retrieve_in_space(
       "return policy",
       space_name="Support-UserA",
       top_k=5
   )

Low-Level Search Interfaces
-----------------------------

If ``retrieve_in_space`` doesn't meet your needs, you can directly use SemanticMapService's low-level search interfaces:

.. code-block:: python

   results = system.semantic_map.search_by_vector(query_embedding, top_k=20)
   results = system.semantic_map.search_by_text_with_rerank(
       "return process", top_k=10
   )
   results = system.semantic_map.search_in_space(
       query_embedding, space_name="Support-UserA", top_k=10
   )

In-Space vs Holistic Retrieval
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * - Feature
     - ``retrieve_in_space``
     - ``holistic_retrieve``
   * - Search scope
     - Specified space
     - All spaces
   * - Multi-path recall
     - Yes (Dense+BM25+Sparse)
     - Yes
   * - BFS expansion
     - No
     - Yes
   * - Latency
     - Lower
     - Higher
   * - Use case
     - Precise retrieval with known space scope
     - Cross-space comprehensive retrieval
