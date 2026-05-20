Basic Depth: Three-Step Memory Pipeline
========================================

Step 1: Add Memories
--------------------

Add conversation records to the system. The system automatically vectorizes and stores them.

.. code-block:: python

   unit = MemoryUnit(
       uid=Uid("msg_001"),
       raw_data={"text_content": "Zhang San went to Beijing on a business trip today"},
   )
   system.add(unit)

Step 2: Build High-Level Memories
----------------------------------

After adding a batch of data, call ``build_high_level()``. The system internally performs the following:

- **Chunking**: Overly long memory units are split into smaller sub-units, ensuring each unit fits within LLM processing limits
- **Session segmentation**: LLM-based semantic analysis identifies topic boundaries, grouping adjacent memories into sessions by theme
- **High-level extraction**: For each session, the system extracts entity relationships, event causality, and multi-type summaries (episodic/knowledge/emotional/procedural), then further distills deep insights; cross-session entities and events are automatically merged and deduplicated, and global insights accumulate continuously

All of this is transparent to you — just one method call:

.. code-block:: python

   system.build_high_level(mode="auto")

.. important::

   If you skip this step, the raw data is stored but not yet organized into entities, events, and summaries — retrieval of high-level memories will return empty results. If you only need to retrieve raw conversations (BASE group), this step is not needed.

Step 3: Retrieve Memories
--------------------------

Once construction is complete, use natural language queries to retrieve relevant memories.

.. code-block:: python

   hits = system.holistic_retrieve("Where did Zhang San go?", top_k=5)

   for hit in hits:
       print(f"Relevance {hit.final_score:.2f}: {hit.unit.raw_data['text_content']}")

The entire process is just these three steps: **add memories → build high-level memories → retrieve**.

.. note::

   For complete details on each step inside ``build_high_level()`` (chunking strategy,
   session detection mechanism, extraction methods for each high-level memory type,
   cross-session merging logic, etc.), see :doc:`detailed-flow`.
