MemoryMonitor 参考
========================

轻量级运行时监控器，提供系统状态和内存占用的实时数据。

访问方式
--------

.. code-block:: python

   monitor = system.monitor

紧凑状态行
----------

``status_line() -> str``

返回单行紧凑状态字符串，适合嵌入日志和监控脚本：

.. code-block:: python

   print(system.monitor.status_line())
   # => [MemSys] units=12450 | spaces=8 | graph:15300n/48200e | idx:11200↑/1250↓ | pend:18u/350e/420et | sess:86(avg145) | mem:156.6MB | DIRTY

   # 简写（__str__ 委托给 status_line）
   print(system.monitor)

状态行字段说明
--------------

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - 字段
     - 含义
   * - ``units``
     - MemoryUnit 总数
   * - ``spaces``
     - MemorySpace 数量
   * - ``graph:Nn/Ee``
     - 图节点数 / 边数
   * - ``idx:P↑/U↓``
     - 向量索引中已提升 (promoted) / 未提升 (unpromoted) 数量
   * - ``pend:Uu/Ee/Et``
     - 待处理队列：units / events / entities
   * - ``sess:N(avgS)``
     - 会话总数 (平均每个会话的 unit 数)
   * - ``mem:XX.XMB``
     - 进程 RSS 物理内存 (MB)
   * - ``DIRTY/CLEAN``
     - 是否有未持久化的变更

程序化访问
----------

``to_dict() -> dict``

返回结构化字典，包含 17 个监控指标：

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

内存测量方案
------------

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - 方案
     - 测量内容
     - 说明
   * - psutil (优先)
     - 进程物理 RSS
     - OS 视角的真实内存，包含 C 扩展 (numpy/FAISS)
   * - tracemalloc (回退)
     - Python 堆分配
     - 仅 Python 对象，不包含 C 扩展，数值偏低

安装 psutil：``pip install mandol[monitoring]``

未安装 psutil 时状态行末尾标注 ``(tracemalloc)``。

性能开销
--------

``status_line()`` 单次调用 < 1ms（10000 单元实测 < 0.5ms），可在轮询场景下使用。
