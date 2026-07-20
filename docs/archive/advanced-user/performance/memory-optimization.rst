内存优化
========

实时监控内存
------------

使用 ``system.monitor`` 可以随时查看系统内存占用和运行状态：

.. code-block:: python

   # 紧凑单行状态
   print(system.monitor)

   # 程序化访问
   stats = system.monitor.to_dict()
   print(f"RSS 内存: {stats['rss_memory_mb']:.1f} MB")
   print(f"数据来源: {stats['memory_source']}")
   print(f"记忆单元数: {stats['total_units']}")
   print(f"图节点/边: {stats['graph_nodes']}n/{stats['graph_edges']}e")

状态行格式：

::

   [MemSys] units=<总数> | spaces=<N> | graph:<节点>n/<边>e | idx:<已提升>↑/<未提升>↓ | pend:<待处理> | sess:<会话数>(avg<大小>) | mem:<RSS>MB | <DIRTY/CLEAN>

.. note::

   内存测量优先使用 ``psutil`` 获取进程真实 RSS（物理内存），未安装时自动回退到 ``tracemalloc``（仅追踪 Python 堆，数值偏低）。安装方式：``pip install mandol[monitoring]``

内存占用分析
------------

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - 组件
     - 每 1000 单元
     - 说明
   * - 向量索引
     - ~3-12 MB
     - dim=768: ~3MB, dim=4096: ~12MB
   * - 图存储
     - ~2-5 MB
     - 取决于边密度
   * - 原始数据
     - ~1-5 MB
     - 取决于文本长度
   * - 模型
     - ~2-8 GB
     - Embedding + Rerank 模型

优化策略
--------

1. **定期 trim**：删除过期/不用的单元
2. **降维 Embedding**：使用 ``dim=512`` 或 ``dim=384`` 模型
3. **分块缩小**：``chunk_max_tokens=256`` 减少每个单元的大小
4. **替代索引**：大规模数据切换 FAISS → 内存更可预测
5. **使用远程模型**：远程 API 模式不占用本地 GPU/CPU 内存
