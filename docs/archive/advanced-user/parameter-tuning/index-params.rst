索引参数
========

promote_threshold
------------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - 参数
     - 默认值
     - 说明
   * - ``promote_threshold``
     - 100
     - 达到此数量后从暴力搜索升级为索引

- < 100 单元：暴力搜索（精确但慢）
- >= 100 单元：FAISS/BM25/TF-IDF 索引

LLM 调用参数
------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - 参数
     - 默认值
     - 说明
   * - ``max_entities_per_llm``
     - 50
     - 实体去重时每次 LLM 调用最大候选数
   * - ``max_events_per_llm``
     - 50
     - 事件去重时每次 LLM 调用最大候选数

- 增大（100-200）：更全面的去重，但成本更高
- 减小（20-30）：省钱，但可能漏掉重复

flush 与 rebuild
----------------

.. code-block:: python

   system.flush()                               # 刷缓存到磁盘
   system.semantic_map.rebuild_index_from_store()  # 从存储重建索引
