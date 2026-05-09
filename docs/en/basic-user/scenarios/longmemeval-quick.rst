LongMemEval Long Document Memory (Quick Start)
==================================================

This example is based on the LongMemEval benchmark, demonstrating Mandol's ability to handle **long document information retention and precise retrieval**.

Data Overview
--------------

A complete article about "Machine Learning in Healthcare: History & Future":

.. list-table::
   :header-rows: 0
   :widths: 25 75

   * - Sample
     - ML in Healthcare: History & Future
   * - Scale
     - 468 words / ~3200 characters
   * - QA
     - 12 questions covering all 6 retrieval categories

How to Run
-----------

.. code-block:: bash

   cd examples/longmemeval
   cp .env.template .env
   # Edit .env and fill in API Key

   # Use built-in synthetic data
   python run_example.py

   # Custom query
   python run_example.py --query "What was the first FDA-approved AI diagnostic tool?"

Core Code Flow
---------------

.. code-block:: python

   from mandol import MemorySystem, MemoryUnit, Uid
   import json

   system = MemorySystem.from_yaml_config("config.yaml")

   with open("data/longmemeval_example.json") as f:
       data = json.load(f)

   # Add after chunking
   for i, chunk in enumerate(data["passage"].split(". ")):
       if chunk.strip():
           system.add(MemoryUnit(
               uid=Uid(f"lme-001_chunk_{i}"),
               raw_data={"text_content": chunk.strip()},
           ))

   system.build_high_level(mode="auto")

   hits = system.holistic_retrieve(
       "What was the first FDA-approved AI diagnostic tool?", top_k=5
   )

Six Retrieval Categories
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 35 45

   * - Category
     - Full Name
     - Capability Tested
   * - SS-Pref
     - Single-Session Preference
     - Fact retrieval based on user preferences
   * - SS-Asst
     - Single-Session Assistant
     - Precise information extraction
   * - Temporal
     - Temporal Reasoning
     - Temporal relationship reasoning
   * - Multi-S
     - Multi-Session
     - Cross-session information synthesis
   * - Know.Upd.
     - Knowledge Update
     - Knowledge change tracking
   * - SS-User
     - Single-Session User
     - User-specific detail retention

Expected Output Reference
---------------------------

::

   MEMORY SYSTEM STATISTICS
     Total memory spaces: 8
     Total memory units:  15

   EVALUATION RESULTS
   [+ ] Q01 [SS-Pref   ] What was the first FDA-approved AI diagnostic tool?
   [+ ] Q02 [SS-Asst   ] Which organization launched an AI Lab in January?
   ...
   ────────────────────────────────
   Category Breakdown:
     SS-Pref:      2/2 correct (100.0%)
     SS-Asst:      2/2 correct (100.0%)
     Temporal:     2/2 correct (100.0%)
   ────────────────────────────────
     Total:       12/12 correct (100.0%)

For the full evaluation process (HuggingFace full dataset), see :doc:`/advanced-user/scenarios/longmemeval-eval`.

Example source: `examples/longmemeval/README.md <../../examples/longmemeval/README.md>`_
