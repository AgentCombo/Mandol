BFS Graph Expansion
======================

BFS expansion is a key step in ``holistic_retrieve`` and can also be used independently for graph traversal exploration.

Direct Usage
-------------

.. code-block:: python

   seeds = [Uid("entity_zhangsan"), Uid("event_meeting")]
   expanded = system.semantic_graph.bfs_expand_units(
         seeds=seeds,
         per_seed=5,
         hops=2,
   )
   print(f"Discovered {len(expanded)} related nodes from {len(seeds)} seeds")

Parameter Description
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Parameter
     - Default
     - Recommended Range
     - Description
   * - ``per_seed``
     - 3
     - 1-10
     - Number of neighbors to pull per seed
   * - ``hops``
     - 1
     - 0-2
     - Expansion hop count; hops=0 disables expansion

Position in the Retrieval Pipeline
-------------------------------------

.. code-block::

   holistic_retrieve(query)
   ├── 1. Group recall (four groups)
   ├── 2. Three-way retrieval per group (Dense/BM25/Sparse)
   ├── 3. RRF fusion
   ├── 4. BFS expansion ← Used at this step
   └── 5. Global Rerank
