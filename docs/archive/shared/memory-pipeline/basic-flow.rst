基础深度：三步记忆流程
===============================

第一步：添加记忆
----------------

将对话记录添加到系统，系统自动向量化并存储。

.. code-block:: python

   unit = MemoryUnit(
       uid=Uid("msg_001"),
       raw_data={"text_content": "张三今天去北京出差了"},
   )
   system.add(unit)

第二步：构建高阶记忆
--------------------

添加完一批数据后，调用 ``build_high_level()``。系统自动识别话题边界、提取关键人物、事件和知识点。

.. code-block:: python

   system.build_high_level(mode="auto")

.. important::

   如果不执行这一步，系统尚未对记忆进行结构化处理，检索实体/事件/摘要时会返回空结果。数据已存储但未构建高阶索引，无法通过语义视图检索。仅检索原始对话（BASE 组）时无需等待此步骤。

第三步：检索记忆
----------------

构建完成后，使用自然语言查询检索相关记忆。

.. code-block:: python

   hits = system.holistic_retrieve("张三去了哪里？", top_k=5)

   for hit in hits:
       print(f"相关性 {hit.final_score:.2f}: {hit.unit.raw_data['text_content']}")

整个过程就这三步：**添加记忆 → 构建高阶记忆 → 检索**。
