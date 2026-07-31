LoCoMo Long Conversation Memory (Quick Start)
=================================================

This example is based on the LoCoMo benchmark dataset, demonstrating Mandol's memory capabilities for **multi-turn long conversations**.

Data Overview
--------------

From the LoCoMo dataset's conv-26 sample, a real long conversation between Caroline and Melanie:

.. list-table::
   :header-rows: 0
   :widths: 25 75

   * - Sample
     - conv-26: Caroline & Melanie
   * - Scale
     - 19 sessions, 419 turns, 199 QA pairs
   * - Quick mode
     - Only processes the first 3 sessions (Demo)

How to Run
-----------

.. code-block:: bash

   cd examples/locomo
   cp .env.template .env
   # Edit .env and fill in API Key

   # Demo mode (quick start, first 3 sessions)
   python run_example.py

   # Custom query
   python run_example.py --query "What is Caroline's identity?"

Core Code Flow
---------------

.. code-block:: python

   from locomo.locomo_memory_system import LocomoMemorySystem
   from locomo.config import LocomoMemoryConfig

   config = LocomoMemoryConfig()
   system = LocomoMemorySystem(config=config)

   result = system.load_and_process_samples()       # Load data
   build = system.build_high_level_memories("auto")  # Build high-level memories
   stats = system.get_memory_stats()                 # View statistics

   hits = system.search("What is Caroline's identity?", top_k=5)
   for h in hits:
       print(f"[{h.final_score:.3f}] {h.unit.raw_data['text_content'][:100]}")

Preset Query Examples
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Query
     - Category
     - Description
   * - What is Caroline's identity?
     - Single-hop
     - Find a single fact
   * - When did Caroline go to the LGBTQ support group?
     - Temporal
     - Temporal reasoning
   * - What fields would Caroline likely pursue in her education?
     - Multi-hop
     - Requires synthesizing multiple pieces of information

Expected Output Reference
---------------------------

::

   MEMORY STATISTICS
     Total units: 127
     Total spaces: 38
     Dialogues processed: 63
     Sessions processed: 3

   QUERY RESULTS
   ────────────────────────────────
   Query: What is Caroline's identity?
   [1] 0.934 | Dialogue D1:0 Caroline said: I have been questioning my identity...
   [2] 0.891 | Entity: Caroline - a person questioning gender identity

For the full reproduction (all 19 sessions), see :doc:`/advanced-user/scenarios/locomo-full`.

Example source: `examples/locomo/README.md <../../examples/locomo/README.md>`_
