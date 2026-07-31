LoCoMo 长对话记忆（快速体验）
================================

本示例基于 LoCoMo benchmark 数据集，演示 Mandol 处理**多轮长对话**的记忆能力。

数据概述
--------

来自 LoCoMo 数据集的 conv-26 样本，一段关于 Caroline 和 Melanie 的真实长对话：

.. list-table::
   :header-rows: 0
   :widths: 25 75

   * - 样本
     - conv-26: Caroline & Melanie
   * - 规模
     - 19 个会话, 419 轮对话, 199 个问答对
   * - 快速模式
     - 仅处理前 3 个会话（Demo）

运行方式
--------

.. code-block:: bash

   cd examples/locomo
   cp .env.template .env
   # 编辑 .env 填入 API Key

   # Demo 模式（快速体验，前 3 个会话）
   python run_example.py

   # 自定义查询
   python run_example.py --query "What is Caroline's identity?"

代码核心流程
------------

.. code-block:: python

   from locomo.locomo_memory_system import LocomoMemorySystem
   from locomo.config import LocomoMemoryConfig

   config = LocomoMemoryConfig()
   system = LocomoMemorySystem(config=config)

   result = system.load_and_process_samples()       # 加载数据
   build = system.build_high_level_memories("auto")  # 构建高阶记忆
   stats = system.get_memory_stats()                 # 查看统计

   hits = system.search("What is Caroline's identity?", top_k=5)
   for h in hits:
       print(f"[{h.final_score:.3f}] {h.unit.raw_data['text_content'][:100]}")

预设查询示例
------------

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - 查询
     - 类别
     - 说明
   * - What is Caroline's identity?
     - Single-hop
     - 查找单条事实
   * - When did Caroline go to the LGBTQ support group?
     - Temporal
     - 时间推理
   * - What fields would Caroline likely pursue in her education?
     - Multi-hop
     - 需要综合多条信息

预期输出参考
-----------

::

   MEMORY STATISTICS
     Total units: 127
     Total spaces: 38
     Dialogues processed: 63
     Sessions processed: 3

   QUERY RESULTS
   ────────────────────────────────
   Query: What is Caroline's identity?
   [1] 0.934 | Dialogue D1:0 Caroline said: I have been questioning my identity...
   [2] 0.891 | Entity: Caroline - a person questioning gender identity

完整复现（19 个会话全量处理）请参阅 :doc:`/advanced-user/scenarios/locomo-full`。

示例源码：`examples/locomo/README.md <../../examples/locomo/README.md>`_
