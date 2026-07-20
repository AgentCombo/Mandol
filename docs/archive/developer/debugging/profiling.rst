性能 Profiling
==================

使用 Python 内置工具。

cProfile
--------

.. code-block:: bash

   python -m cProfile -o profile.out run_example.py
   python -m pstats profile.out
   stats.sort_stats('cumulative').print_stats(20)

memory_profiler
---------------

.. code-block:: bash

   pip install memory_profiler
   python -m memory_profiler run_example.py

时间分解
--------

.. code-block:: python

   def timed_retrieve(system, query):
       import time
       t = {}
       t['embed'] = time.perf_counter()
       emb = system._get_embedding(query)
       t['embed'] = time.perf_counter() - t['embed']

       t['search'] = time.perf_counter()
       hits = system.holistic_retrieve(query, top_k=10)
       t['search'] = time.perf_counter() - t['search']

       return hits, t

   hits, timing = timed_retrieve(system, "query")
   for step, duration in timing.items():
       print(f"{step}: {duration*1000:.1f}ms")
