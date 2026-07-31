场景二：个人助手长期记忆
========================

.. note::

   本文档已迁移至 :doc:`/basic-user/scenarios/personal-assistant`。本页面将在后续版本中移除，请更新你的书签。

本场景演示如何使用 Mandol 构建个人助手的长期记忆系统，记住用户的习惯、日程、人际关系，提供主动提醒和建议。

场景说明
--------

个人助手需要跨越多个会话记住用户的信息。用户可能在不同时间谈论工作和生活，Mandol 能够：

- 自动检测会话边界，将不同主题的对话分割为独立会话
- 跨会话检索，综合多个会话中的信息
- 提取实体和事件，构建用户的个人知识图谱

完整代码示例
------------

可运行示例：``examples/personal_assistant/run_personal_assistant.py``

.. code-block:: python

   from mandol import MemorySystem, MemoryUnit, Uid

   system = MemorySystem.from_yaml_config("config.yaml")

   session1 = [
       MemoryUnit(
           uid=Uid("pa_001"),
           raw_data={"text_content": "我下周二要和客户做项目汇报"},
           metadata={"timestamp": "2024-03-11T09:00:00", "speaker": "user", "session_id": "s1"},
       ),
       MemoryUnit(
           uid=Uid("pa_002"),
           raw_data={"text_content": "汇报内容是Q1的销售数据分析"},
           metadata={"timestamp": "2024-03-11T09:01:00", "speaker": "user", "session_id": "s1"},
       ),
   ]

   session2 = [
       MemoryUnit(
           uid=Uid("pa_003"),
           raw_data={"text_content": "周末想去爬山，推荐一下附近的路线"},
           metadata={"timestamp": "2024-03-16T10:00:00", "speaker": "user", "session_id": "s2"},
       ),
       MemoryUnit(
           uid=Uid("pa_004"),
           raw_data={"text_content": "香山和百望山都不错，香山风景更好但人比较多"},
           metadata={"timestamp": "2024-03-16T10:01:00", "speaker": "assistant", "session_id": "s2"},
       ),
   ]

   for unit in session1 + session2:
       system.add(unit)

   report = system.build_high_level(mode="auto")

   hits = system.holistic_retrieve("我最近有什么安排？", top_k=5)

   entities = system.retrieve_in_space("客户", space_name="root_knowledge_entity", top_k=5)

   system.save("./pa_memory")

检索结果说明
^^^^^^^^^^^^

**holistic_retrieve("我最近有什么安排？")**

全记忆检索会跨会话召回结果。预期返回：

- 来自会话1：项目汇报（工作安排）
- 来自会话2：爬山计划（生活安排）

两个会话的时间间隔超过 ``session_time_gap_seconds`` （默认 1800 秒 = 30 分钟），系统会自动将它们识别为不同会话，但检索时仍能跨会话综合返回结果。

**retrieve_in_space("客户", space_name="root_knowledge_entity")**

在知识实体空间中检索"客户"相关实体。预期返回从会话1中提取的"客户"实体，以及与项目汇报相关的知识摘要。

跨会话记忆能力
--------------

Mandol 的跨会话记忆能力基于以下机制：

1. **会话分割**：通过时间间隔（``session_time_gap_seconds``）和 LLM 智能检测，自动将连续对话分割为独立会话
2. **跨会话实体合并**：不同会话中提到的同一实体（如"客户"）会被自动合并，通过 ``coref_vector_threshold`` 和 LLM 共指消解实现
3. **统一检索**：``holistic_retrieve`` 在四组（BASE/ENTITY/EVENT/SUMMARY）中并行检索，天然支持跨会话信息综合
4. **图扩展**：BFS 扩展沿语义图边遍历，能发现跨会话的隐式关联

会话分割参数建议
^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - 场景
     - 推荐值
     - 说明
   * - 客服对话
     - 300 秒（5 分钟）
     - 客服对话会话通常较短，5分钟无交互即视为新会话
   * - 个人助手
     - 1800 秒（30 分钟）
     - 默认值，适合日常对话节奏
   * - 知识库
     - 86400 秒（24 小时）
     - 知识库导入通常无会话边界，设为极大值
