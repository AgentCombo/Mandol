Scenario Session Configuration
================================

Customer Service Conversations
-------------------------------

.. code-block:: yaml

   system:
     session_time_gap_seconds: 300      # 5 minutes
     session_check_interval: 10         # Check every 10 messages

Sparse Conversations (Personal Assistant)
------------------------------------------

.. code-block:: yaml

   system:
     session_time_gap_seconds: 1800     # 30 minutes
     session_check_interval: 20         # Default

Dense Without Boundaries (Knowledge Base)
-------------------------------------------

.. code-block:: yaml

   system:
     session_time_gap_seconds: 86400    # 24 hours, almost no segmentation
     session_check_interval: 200
     session_max_pending: 1000

Multi-User Mixed
-----------------

.. code-block:: python

   for user_id in unique_users:
       user_units = get_user_history(user_id)
       system.add_many(user_units)
       system.build_high_level(mode="auto")

   # Each user gets an independent space for isolation
   system.semantic_map.create_space(f"User-{user_id}")

Configuration Impact
--------------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Adjustment
     - Effect
     - Cost
   * - Decrease gap_seconds
     - More sessions, more focused summaries
     - Summary fragmentation
   * - Increase gap_seconds
     - Fewer sessions, more complete summaries
     - Cross-topic confusion
   * - Decrease check_interval
     - Faster boundary detection
     - More LLM calls
   * - Increase check_interval
     - Fewer LLM calls
     - Delayed boundary detection
