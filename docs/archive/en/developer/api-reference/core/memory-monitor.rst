MemoryMonitor Reference
==========================

Lightweight runtime monitor providing real-time data on system status and memory usage.

Access
-------

.. code-block:: python

   monitor = system.monitor

Compact Status Line
---------------------

``status_line() -> str``

Returns a compact single-line status string, suitable for embedding in logs and monitoring scripts:

.. code-block:: python

   print(system.monitor.status_line())
   # => [MemSys] units=12450 | spaces=8 | graph:15300n/48200e | idx:11200↑/1250↓ | pend:18u/350e/420et | sess:86(avg145) | mem:156.6MB | DIRTY

   # Shorthand (__str__ delegates to status_line)
   print(system.monitor)

Status Line Field Descriptions
--------------------------------

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Field
     - Meaning
   * - ``units``
     - Total MemoryUnit count
   * - ``spaces``
     - MemorySpace count
   * - ``graph:Nn/Ee``
     - Graph node count / edge count
   * - ``idx:P↑/U↓``
     - Vector index promoted / unpromoted count
   * - ``pend:Uu/Ee/Et``
     - Pending queue: units / events / entities
   * - ``sess:N(avgS)``
     - Total sessions (average units per session)
   * - ``mem:XX.XMB``
     - Process RSS physical memory (MB)
   * - ``DIRTY/CLEAN``
     - Whether there are unpersisted changes

Programmatic Access
---------------------

``to_dict() -> dict``

Returns a structured dictionary with 17 monitoring metrics:

.. code-block:: python

   stats = system.monitor.to_dict()
   # {
   #   "total_units": 12450,
   #   "total_spaces": 8,
   #   "graph_nodes": 15300,
   #   "graph_edges": 48200,
   #   "vector_index_global": 12450,
   #   "vector_index_promoted": 11200,
   #   "vector_index_unpromoted": 1250,
   #   "pending_units": 18,
   #   "pending_events": 350,
   #   "pending_entities": 420,
   #   "total_sessions": 86,
   #   "avg_session_size": 145.0,
   #   "rss_memory_mb": 156.6,
   #   "memory_source": "psutil",
   #   "dirty": True,
   #   "persistence_enabled": False,
   #   "llm_model": "gpt-4o-mini",
   #   "embedder_model": "Qwen/Qwen3-Embedding-4B",
   #   "embedder_dim": 2560,
   #   "use_unified_pipeline": True,
   # }

Memory Measurement Schemes
----------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Scheme
     - Measures
     - Description
   * - psutil (preferred)
     - Process physical RSS
     - Real memory from OS perspective, includes C extensions (numpy/FAISS)
   * - tracemalloc (fallback)
     - Python heap allocations
     - Only Python objects, excludes C extensions, values are lower

Install psutil: ``pip install mandol[monitoring]``

When psutil is not installed, the status line is annotated with ``(tracemalloc)``.

Performance Overhead
---------------------

``status_line()`` single call < 1ms (measured < 0.5ms with 10000 units), suitable for polling scenarios.
