Retrieval Parameters
=======================

similarity_threshold
----------------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Default
     - Description
   * - ``similarity_threshold``
     - 0.7
     - SEMANTIC_SIMILAR edge creation threshold, affects graph density

- Increase (0.8-0.9): More precise, sparser graph
- Decrease (0.5-0.6): More edges, stronger BFS effect

similarity_top_k
------------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Default
     - Description
   * - ``similarity_top_k``
     - 5
     - Initial recall count per group for vector retrieval

bfs_expansion_per_seed / bfs_expansion_hops
---------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Default
     - Description
   * - ``bfs_expansion_per_seed``
     - 3
     - Number of neighbors expanded per seed
   * - ``bfs_expansion_hops``
     - 1
     - BFS hop count; hops=0 disables expansion

.. list-table::
   :header-rows: 1
   :widths: 25 50 25

   * - Scenario
     - Configuration
     - Latency
   * - Precise retrieval
     - per_seed=0, hops=0
     - Lowest
   * - General
     - per_seed=3, hops=1
     - Medium
   * - Multi-hop reasoning
     - per_seed=5, hops=2
     - ~2-3x
