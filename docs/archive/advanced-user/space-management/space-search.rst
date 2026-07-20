空间内检索
============

在指定空间内检索，比全局检索更快、更聚焦。

基础用法
--------

.. code-block:: python

   hits = system.retrieve_in_space(
       "退货政策",
       space_name="客服-用户A",
       top_k=5
   )

搜索底层接口
------------

如果 ``retrieve_in_space`` 不满足需求，可以直接使用 SemanticMapService 的底层检索接口：

.. code-block:: python

   results = system.semantic_map.search_by_vector(query_embedding, top_k=20)
   results = system.semantic_map.search_by_text_with_rerank(
       "退货流程", top_k=10
   )
   <!-- TODO: 验证 API 签名 -->
   results = system.semantic_map.search_in_space(
       query_embedding, space_name="客服-用户A", top_k=10
   )

空间检索 vs 全记忆检索
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * - 特性
     - ``retrieve_in_space``
     - ``holistic_retrieve``
   * - 检索范围
     - 指定空间
     - 全部空间
   * - 多路召回
     - 是（Dense+BM25+Sparse）
     - 是
   * - BFS 扩展
     - 否
     - 是
   * - 延迟
     - 较低
     - 较高
   * - 适用场景
     - 已知空间范围的精确检索
     - 跨空间综合检索
