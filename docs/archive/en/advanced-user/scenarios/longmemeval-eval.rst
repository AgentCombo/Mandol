LongMemEval Full Evaluation
===============================

Full 6-category evaluation using the HuggingFace dataset.

Getting the Full Dataset
--------------------------

.. code-block:: bash

   cd examples/longmemeval
   python download_data.py

6 Category Evaluation Details
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 28 50

   * - Category
     - Full Name
     - Tests
   * - SS-Pref
     - Single-Session Preference
     - User preferences/opinions
   * - SS-Asst
     - Single-Session Assistant
     - Precise fact extraction
   * - Temporal
     - Temporal
     - Temporal relationship reasoning
   * - Multi-S
     - Multi-Session
     - Cross-session synthesis
   * - Know.Upd.
     - Knowledge Update
     - Knowledge change tracking
   * - SS-User
     - Single-Session User
     - User-specific details

Running the Evaluation
------------------------

.. code-block:: bash

   python run_example.py --eval

Per-Category Analysis
-----------------------

.. code-block::

   Category Breakdown:
     SS-Pref:      180/200 (90.0%)
     SS-Asst:      195/200 (97.5%)
     Temporal:     170/200 (85.0%)
     Multi-S:       88/100 (88.0%)
     Know.Upd.:     75/100 (75.0%)
     SS-User:      185/200 (92.5%)
   Total:         893/1000 (89.3%)

Custom Evaluation
-------------------

.. code-block:: python

   # Evaluate specific categories only
   python run_example.py --eval --category temporal,knowledge-update

   # Custom query
   python run_example.py --query "What happened in 2018?"
