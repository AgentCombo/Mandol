Session Segmentation Mechanism Details
========================================

Mandol uses a dual strategy for session segmentation.

Strategy 1: Time Interval Segmentation
----------------------------------------

When the ``timestamp`` difference between two adjacent memories exceeds ``session_time_gap_seconds`` → force a new session.

.. code-block:: python

   # Configuration
   system = MemorySystem.from_yaml_config("config.yaml")
   # system:
   #   session_time_gap_seconds: 1800  # 30 minutes

Strategy 2: LLM Smart Segmentation
------------------------------------

When ``session_check_interval`` memories are accumulated → call LLM to determine session boundaries.

.. code-block:: python

   # Configuration
   # session_check_interval: 20
   # session_max_pending: 100

Debugging Segmentation Results
-------------------------------

.. code-block:: python

   report = system.build_high_level(mode="auto")
   print(f"Sessions processed: {report.sessions_processed}")

   # View session information in SessionManager
   stats = system.get_memory_stats()
   print(stats)

Manual Forced Segmentation
---------------------------

If automatic segmentation doesn't meet your needs, you can manually mark session_id in metadata:

.. code-block:: python

   # Memories with the same session_id won't be segmented
   unit.metadata["session_id"] = "manual-session-20240301"
