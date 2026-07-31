Quick Start
===========

.. note::

   This document has been migrated to :doc:`/basic-user/five-minute-start`. This page will be removed in a future version, please update your bookmarks.

This guide will help you run the Mandol memory system in a few minutes. Choose one of the two modes below based on your use case.

Mode 1: Remote API (Recommended for Beginners)
-----------------------------------------------

This mode calls Embedding, Reranker, and LLM services through remote APIs, requiring no local GPU. Suitable for quick experimentation and development debugging.

**Prerequisite**: ``OPENAI_API_KEY`` environment variable is configured, and ``config.yaml`` is ready.

.. code-block:: python

   from mandol import MemorySystem, MemoryUnit, Uid

   system = MemorySystem.from_yaml_config("config.yaml")

   unit = MemoryUnit(
       uid=Uid("dialogue_001"),
       raw_data={"text_content": "Zhang San went to Beijing on a business trip today"},
       metadata={"timestamp": "2024-01-15T10:00:00"},
   )
   system.add(unit)

   unit2 = MemoryUnit(
       uid=Uid("dialogue_002"),
       raw_data={"text_content": "Li Si said they're going to Shanghai for a meeting next week"},
       metadata={"timestamp": "2024-01-15T10:05:00"},
   )
   system.add(unit2)

   report = system.build_high_level(mode="auto")
   print(f"Processed {report.sessions_processed} sessions, {report.units_processed} memories")

   hits = system.holistic_retrieve("Where did Zhang San go?", top_k=5)
   for hit in hits:
       print(f"[{hit.final_score:.3f}] {hit.unit.raw_data['text_content']}")

   system.save("./memory_snapshot")
   system2 = MemorySystem.load("./memory_snapshot")

Mode 2: Local Models (No API Key Needed)
-----------------------------------------

This mode uses local Sentence-Transformers models for Embedding and Reranker, requiring no API Key. You need to install the ``sentence-transformers`` optional dependency.

.. code-block:: bash

   pip install mandol[sentence-transformers]

.. code-block:: python

   from mandol import MemorySystem, MemoryUnit, Uid

   system = MemorySystem()

   unit = MemoryUnit(
       uid=Uid("dialogue_001"),
       raw_data={"text_content": "Zhang San went to Beijing on a business trip today"},
       metadata={"timestamp": "2024-01-15T10:00:00"},
   )
   system.add(unit)

   unit2 = MemoryUnit(
       uid=Uid("dialogue_002"),
       raw_data={"text_content": "Li Si said they're going to Shanghai for a meeting next week"},
       metadata={"timestamp": "2024-01-15T10:05:00"},
   )
   system.add(unit2)

   report = system.build_high_level(mode="auto")
   print(f"Processed {report.sessions_processed} sessions, {report.units_processed} memories")

   hits = system.holistic_retrieve("Where did Zhang San go?", top_k=5)
   for hit in hits:
       print(f"[{hit.final_score:.3f}] {hit.unit.raw_data['text_content']}")

   system.save("./memory_snapshot")
   system2 = MemorySystem.load("./memory_snapshot")

.. note::

   In local mode, model files are automatically downloaded on first run (~2-4 GB). Ensure a stable internet connection and sufficient disk space. Subsequent runs will use cached models without repeated downloads.

What Happens After add()?
--------------------------

When you call ``system.add(unit)``, Mandol automatically performs the following:

1. **Auto-chunking**: If the memory unit's text exceeds ``chunk_max_tokens`` (default 512), the system automatically splits it into smaller chunks
2. **Auto-vectorization**: Generates Embedding vectors for each memory unit (or chunked sub-unit)
3. **Session detection**: The system asynchronously detects session boundaries, triggering detection when accumulated memory units reach ``session_check_interval`` (default 20)
4. **Similarity edge creation**: Computes cosine similarity between new memories and the most recent ``similarity_recent_window`` (default 20) memories, creating ``SEMANTIC_SIMILAR`` edges

build_high_level() Details
---------------------------

``build_high_level()`` is Mandol's core construction method, responsible for extracting high-level semantic structures from raw conversation memories. After calling, the system executes:

1. **Session segmentation**: Sorts all raw memories chronologically, detects session boundaries via LLM, and splits continuous conversations into independent sessions
2. **Summary extraction**: Generates episodic, knowledge, emotional, and procedural summaries for each session
3. **Insight extraction**: Further distills global insights from summaries
4. **Entity extraction and deduplication**: Identifies entities (people, places, concepts, etc.) from conversations and merges identical references across sessions
5. **Event extraction and deduplication**: Identifies events from conversations and establishes causal relationships
6. **Relationship graph construction**: Builds entity relationship edges (e.g., ``REL_WORKS_AT``) and event causal edges (e.g., ``CAUSES``, ``CAUSED_BY``)

**When to call**:

- Call after adding a batch of memory units, e.g., after a conversation round ends
- ``mode="auto"``: Only process sessions that haven't been built yet (incremental mode, recommended)
- ``mode="force"``: Clear all high-level memories and rebuild (full rebuild mode)

holistic_retrieve() Retrieval Pipeline
---------------------------------------

``holistic_retrieve()`` is Mandol's unified retrieval interface, internally executing the following flow:

1. **Group recall**: Distributes the retrieval request to four retrieval groups:
   - **BASE**: Raw conversation memories
   - **ENTITY**: Knowledge entities
   - **EVENT**: Episodic events
   - **SUMMARY**: Summaries and insights
2. **Three-way recall**: Each group independently executes Dense (dense vector), BM25 (keyword), and Sparse (sparse vector) retrieval
3. **RRF fusion**: Merges three-way results using Reciprocal Rank Fusion
4. **BFS expansion**: Expands candidate set based on semantic graph relationships (controlled by ``bfs_expansion_per_seed`` and ``bfs_expansion_hops``)
5. **Global reranking**: After all groups are merged, reranks via Cross-Encoder Reranker and returns final results

You can also use more fine-grained retrieval interfaces:

- ``retrieve_by_view(query, view="entity_relation")``: Retrieve by perspective category
- ``retrieve_in_space(query, space_name="root_knowledge_entity")``: Retrieve within a specific space

Next Steps
----------

- Read :doc:`configuration` for detailed configuration options
- Read :doc:`../data_structures` for core data structures
- Read :doc:`../retrieval/index` for retrieval interface details
