跨会话合并
============

``build_high_level()`` 内部会自动触发跨会话实体和事件的合并。

实体合并
--------

.. code-block:: python

   # 自动调用（在 build_high_level 中）
   # 手动调用（调试场景）：
   merged = system.semantic_graph._cross_session_entity_merge()

合并检测逻辑：LLM 匹配两个实体名是否指同一概念。例如「张三」「张总」「老张」→ 合并为「张三」。

<!-- TODO: 验证 API 签名 -->

事件合并
--------

.. code-block:: python

   merged = system.semantic_graph._cross_session_event_merge()

合并检测逻辑：事件相似度高 + 时间接近 → 合并为同一事件。

合并效果验证
-------------

.. code-block:: python

   # 合并前：查询返回多个同名实体的结果
   # 合并后：同名实体归一化
   hits = system.retrieve_in_space("张三", space_name="root_knowledge_entity")
   for hit in hits:
       print(hit.unit.raw_data.get("text_content", ""))

调优合并参数
-------------

.. code-block:: yaml

   system:
     max_entities_per_llm: 50     # 每次 LLM 去重的候选数
     max_events_per_llm: 50      # 同上（事件）

增大这些值 → 去重更全面但 LLM 成本增加。
