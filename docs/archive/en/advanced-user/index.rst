Advanced User Guide
====================

You've already mastered the basic :doc:`/basic-user/index` workflow. This section takes you deeper into controlling every aspect of the memory system.

Capability Map
--------------

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - I want to...
     - Description
     - Go to
   * - Manage memory spaces
     - Create, hierarchize, delete spaces
     - :doc:`space-management/index`
   * - Manipulate graph relationships
     - Manually add/delete relationships, query neighbors, BFS expansion
     - :doc:`graph-management/index`
   * - Control session segmentation
     - Debug segmentation strategies, cross-session merging
     - :doc:`session-control/index`
   * - Choose retrieval strategies
     - Holistic vs by-view vs in-space vs generic search()
     - :doc:`retrieval-strategies/index`
   * - Fine-tune parameters
     - Chunk/session/retrieval/LLM/index — five parameter categories
     - :doc:`parameter-tuning/index`
   * - Reproduce complete scenarios
     - LoCoMo full 19 sessions / LongMemEval complete evaluation
     - :doc:`scenarios/locomo-full`
   * - Performance optimization
     - Reduce latency / control memory / save LLM costs
     - :doc:`performance/index`
   * - Troubleshoot issues
     - Retrieval quality / build errors / performance bottlenecks
     - :doc:`troubleshooting-advanced`

.. toctree::
   :maxdepth: 2
   :hidden:

   beyond-basics
   space-management/index
   graph-management/index
   session-control/index
   retrieval-strategies/index
   parameter-tuning/index
   scenarios/locomo-full
   scenarios/longmemeval-eval
   performance/index
   troubleshooting-advanced
