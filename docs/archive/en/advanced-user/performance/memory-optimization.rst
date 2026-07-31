Memory Optimization
======================

Real-Time Memory Monitoring
-----------------------------

Use ``system.monitor`` to check system memory usage and running status at any time:

.. code-block:: python

   # Compact one-line status
   print(system.monitor)

   # Programmatic access
   stats = system.monitor.to_dict()
   print(f"RSS Memory: {stats['rss_memory_mb']:.1f} MB")
   print(f"Data Source: {stats['memory_source']}")
   print(f"Memory Units: {stats['total_units']}")
   print(f"Graph Nodes/Edges: {stats['graph_nodes']}n/{stats['graph_edges']}e")

Status line format:

::

   [MemSys] units=<total> | spaces=<N> | graph:<nodes>n/<edges>e | idx:<promoted>↑/<unpromoted>↓ | pend:<pending> | sess:<sessions>(avg<size>) | mem:<RSS>MB | <DIRTY/CLEAN>

.. note::

   Memory measurement prioritizes ``psutil`` for real process RSS (physical memory), falling back to ``tracemalloc`` when not installed (only tracks Python heap, values are lower). Install with: ``pip install mandol[monitoring]``

Memory Usage Analysis
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Component
     - Per 1000 Units
     - Description
   * - Vector Index
     - ~3-12 MB
     - dim=768: ~3MB, dim=4096: ~12MB
   * - Graph Store
     - ~2-5 MB
     - Depends on edge density
   * - Raw Data
     - ~1-5 MB
     - Depends on text length
   * - Models
     - ~2-8 GB
     - Embedding + Rerank models

Optimization Strategies
-------------------------

1. **Regular trimming**: Delete expired/unused units
2. **Lower-dimension Embedding**: Use ``dim=512`` or ``dim=384`` models
3. **Smaller chunks**: ``chunk_max_tokens=256`` reduces per-unit size
4. **Alternative indexes**: Switch to FAISS for large-scale data for more predictable memory
5. **Use remote models**: Remote API mode doesn't consume local GPU/CPU memory
