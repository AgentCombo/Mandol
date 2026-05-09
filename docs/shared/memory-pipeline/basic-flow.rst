基础深度：三句话理解记忆流程
===============================

Mandol 的工作方式就像一个会主动记笔记的助手。

第一步：喂给它对话
------------------

你把对话记录告诉系统，系统会自动理解并记住。

.. code-block:: python

   unit = MemoryUnit(
       uid=Uid("msg_001"),
       raw_data={"text_content": "张三今天去北京出差了"},
   )
   system.add(unit)

第二步：让它消化整理
--------------------

添加完一批数据后，调用一次「消化」指令。系统会自动识别话题边界、提取关键人物、事件和知识点。

.. code-block:: python

   system.build_high_level(mode="auto")

.. important::

   如果不执行这一步，系统还没有对记忆进行整理，检索实体/事件/摘要时会返回空结果。就像一个记了笔记但还没复习的人——信息在脑子里，但无法快速提取。仅检索原始对话（BASE 组）时无需等待此步骤。

第三步：向它提问
----------------

系统消化好后，你就可以像问人一样用自然语言提问。

.. code-block:: python

   hits = system.holistic_retrieve("张三去了哪里？", top_k=5)

   for hit in hits:
       print(f"相关性 {hit.final_score:.2f}: {hit.unit.raw_data['text_content']}")

整个过程就这三步：**喂数据 → 让它消化 → 问它问题**。
