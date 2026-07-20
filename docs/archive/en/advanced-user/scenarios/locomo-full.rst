LoCoMo Full Reproduction
============================

Full processing and retrieval of all 19 sessions and 419 conversation turns.

Data Overview
--------------

.. list-table::
   :header-rows: 0
   :widths: 25 75

   * - Dataset
     - LoCoMo conv-26: Caroline & Melanie
   * - Scale
     - 19 sessions, 419 turns, 199 QA pairs
   * - Features
     - Highly personal long conversations, including identity exploration and emotional expression

Running the Full Pipeline
---------------------------

.. code-block:: bash

   cd examples/locomo
   python run_example.py --full

Graph Relationship Construction
---------------------------------

The LoCoMo example manually builds three types of temporal relationships:

.. code-block:: python

   # Temporal predecessor
   system.semantic_graph.add_relationship(
       prev_uid, curr_uid, "PRECEDES"
   )
   # Temporal successor
   system.semantic_graph.add_relationship(
       curr_uid, prev_uid, "FOLLOWS"
   )

Incremental Update Strategy
-----------------------------

.. code-block:: python

   # After processing the first 3 sessions
   system.build_high_level(mode="auto")
   system.flush()

   # Add sessions 4-6
   load_sessions(4, 6)
   system.build_high_level(mode="auto")  # Incremental only
   system.flush()

Flush strategy: ``flush()`` persists in-memory indexes and storage. Recommend flushing after each batch of sessions.

Space Organization
-------------------

.. code-block::

   locomo_conv_26
   ├── locomo_conv_26_session_1
   ├── locomo_conv_26_session_2
   └── ...

Each session gets an independent space, named ``{base_root}_session_{n}``.

Retrieval Statistics
---------------------

.. code-block::

   MEMORY STATISTICS
     Total units: 835
     Total spaces: 57
     Dialogues processed: 419
     Sessions processed: 19
