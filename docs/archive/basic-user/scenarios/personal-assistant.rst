个人助手长期记忆
==================

跨越多天记住用户的习惯、日程、人际关系。

完整代码（可直接运行）
-----------------------

可运行示例：``examples/personal_assistant/run_personal_assistant.py``

.. code-block:: python

   from mandol import MemorySystem, MemoryUnit, Uid

   system = MemorySystem.from_yaml_config("config.yaml")

   session1 = [
       ("pa_1", "我下周二要和客户做项目汇报", "2024-03-11T09:00:00"),
       ("pa_2", "汇报内容是Q1的销售数据分析", "2024-03-11T09:01:00"),
   ]
   session2 = [
       ("pa_3", "周末想去爬山，推荐一下附近的路线", "2024-03-16T10:00:00"),
       ("pa_4", "香山和百望山都不错，香山风景更好但人比较多", "2024-03-16T10:01:00"),
   ]
   for uid, text, ts in session1 + session2:
       system.add(MemoryUnit(
           uid=Uid(uid),
           raw_data={"text_content": text},
           metadata={"timestamp": ts},
       ))

   system.build_high_level(mode="auto")

   hits = system.holistic_retrieve("我最近有什么安排？", top_k=5)
   entities = system.retrieve_in_space("客户", space_name="root_knowledge_entity", top_k=5)

   system.save("./pa_memory")

预期输出对照
-----------

**holistic_retrieve("我最近有什么安排？")**：

.. code-block::

   [0.923] 我下周二要和客户做项目汇报           ← 工作安排（来自会话1）
   [0.876] 汇报内容是Q1的销售数据分析
   [0.845] 周末想去爬山，推荐一下附近的路线      ← 生活安排（来自会话2）
   [0.812] 香山和百望山都不错

两个会话间隔 5 天，系统自动识别为不同的会话，但检索时仍能跨会话综合返回结果。

**retrieve_in_space("客户", space_name="root_knowledge_entity")**：

.. code-block::

   [0.901] 实体: 客户 - 汇报对象，涉及Q1销售数据分析

跨会话记忆的底层机制
---------------------

1. **话题识别**：系统通过 LLM 语义分析自动检测话题变化，将不同主题的对话分组
2. **实体合并**：不同话题中提到的同一实体自动合并
3. **统一检索**：``holistic_retrieve`` 在四组中并行检索，天然跨话题
