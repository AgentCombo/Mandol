SessionManager Reference
===========================

Responsible for session segmentation and lifecycle management.

Access
-------

Created internally by MemorySystem, controlled by ``session_time_gap_seconds`` and other configuration.

Main Methods
-------------

- ``add_unit(unit: MemoryUnit) -> None`` — Add unit to pending queue
- ``process_pending_sessions() -> list[Session]`` — Process session segmentation in queue
- ``get_sessions() -> list[Session]`` — Get all segmented sessions

Session object contains:

- ``id`` — Session ID
- ``units`` — MemoryUnit list
- ``start_time`` / ``end_time`` — Time range
- ``summary`` — Session summary

Usage Example
--------------

.. code-block:: python

   # SessionManager is an internal component of MemorySystem
   # Used indirectly through build_high_level()
   report = system.build_high_level(mode="auto")
   print(f"Processed {report.sessions_processed} sessions")
