全记忆 vs 按视图 vs 空间内
===================================

三种检索方式对比。

holistic_retrieve — 全记忆检索
---------------------------

.. code-block:: python

   hits = system.holistic_retrieve("退款政策", top_k=10)

自动处理全部四组、三路检索、RRF 融合、BFS 扩展、Rerank。适合不确定数据在哪里的场景。

retrieve_by_view — 按语义视角
------------------------------

.. code-block:: python

   hits = system.retrieve_by_view("退款政策", view="knowledge", top_k=10)

只检索知识类记忆，跳过对话/事件/情感等。适合明确想知道「系统知道什么」。

可选视图：

- ``knowledge`` / ``entity_relation`` / ``event_causal`` / ``emotional`` / ``episodic`` / ``procedural`` / ``insights`` / ``base_memory``

retrieve_in_space — 按空间范围
-------------------------------

.. code-block:: python

   hits = system.retrieve_in_space("退款政策", space_name="客服-用户A")

只检索指定空间。适合已知数据在哪里的精确查询。

search()（预想）— 可定制管线
-------------------------

.. code-block:: python

   hits = system.search("退款政策", retriever_types=["dense"], use_graph_expansion=True)

.. warning:: ⚠️ Planned — 此接口尚未实现，签名可能变更

可定制检索器组合和图扩展。适合需要控制检索管线的场景。

选择流程图
----------

.. code-block::

   确定数据在哪个空间？
   ├── 是 → retrieve_in_space(query, space_name=X)
   └── 否 → 需要按类型过滤？
             ├── 是 → retrieve_by_view(query, view=X)
             └── 否 → holistic_retrieve(query)
