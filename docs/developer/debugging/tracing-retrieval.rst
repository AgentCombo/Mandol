追踪检索链路
==============

启用日志
--------

.. code-block:: python

   import logging
   logging.basicConfig(level=logging.DEBUG)

在 pipeline 的关键节点插桩：

.. code-block:: python

   import time
   t0 = time.perf_counter()
   hits = system.holistic_retrieve("query", top_k=10)
   t1 = time.perf_counter()
   print(f"检索耗时: {(t1-t0)*1000:.0f}ms")

   for hit in hits[:3]:
       print(f"  [{hit.final_score:.3f}] {hit.unit.uid}")
       print(f"    scores: {hit.scores}")
       print(f"    ranks: {hit.ranks}")

逐路调试
--------

.. code-block:: python

   # 单独测试各路检索器
   ed = system._get_embedding("query")
   d_hits = system.semantic_map.search_by_vector(ed, top_k=20)
   print(f"Dense: {len(d_hits)}")

   # 检查图扩展
   expanded = system.semantic_graph.bfs_expand_units(
       seeds=[h[0] for h in d_hits[:3]],
       per_seed=3, hops=1,
   )
   print(f"BFS 扩展: {len(expanded)}")
