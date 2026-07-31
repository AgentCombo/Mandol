LongMemEval 长文本记忆（快速体验）
=====================================

本示例基于 LongMemEval benchmark，演示 Mandol 处理**长文档信息保留与精准检索**的能力。

数据概述
--------

一篇关于「机器学习在医疗健康领域的历史与未来」的完整文章：

.. list-table::
   :header-rows: 0
   :widths: 25 75

   * - 样本
     - ML in Healthcare: History & Future
   * - 规模
     - 468 词 / 约 3200 字符
   * - QA
     - 12 个问题，覆盖全部 6 个检索类别

运行方式
--------

.. code-block:: bash

   cd examples/longmemeval
   cp .env.template .env
   # 编辑 .env 填入 API Key

   # 使用内置合成数据
   python run_example.py

   # 自定义查询
   python run_example.py --query "What was the first FDA-approved AI diagnostic tool?"

代码核心流程
------------

.. code-block:: python

   from mandol import MemorySystem, MemoryUnit, Uid
   import json

   system = MemorySystem.from_yaml_config("config.yaml")

   with open("data/longmemeval_example.json") as f:
       data = json.load(f)

   # 分块后添加
   for i, chunk in enumerate(data["passage"].split(". ")):
       if chunk.strip():
           system.add(MemoryUnit(
               uid=Uid(f"lme-001_chunk_{i}"),
               raw_data={"text_content": chunk.strip()},
           ))

   system.build_high_level(mode="auto")

   hits = system.holistic_retrieve(
       "What was the first FDA-approved AI diagnostic tool?", top_k=5
   )

六种检索类别
------------

.. list-table::
   :header-rows: 1
   :widths: 20 35 45

   * - 类别
     - 全称
     - 考察能力
   * - SS-Pref
     - Single-Session Preference
     - 基于用户偏好的事实检索
   * - SS-Asst
     - Single-Session Assistant
     - 精确信息提取
   * - Temporal
     - Temporal Reasoning
     - 时间关系推理
   * - Multi-S
     - Multi-Session
     - 跨会话信息综合
   * - Know.Upd.
     - Knowledge Update
     - 知识变化追踪
   * - SS-User
     - Single-Session User
     - 用户特定细节保留

预期输出参考
-----------

::

   MEMORY SYSTEM STATISTICS
     Total memory spaces: 8
     Total memory units:  15

   EVALUATION RESULTS
   [+ ] Q01 [SS-Pref   ] What was the first FDA-approved AI diagnostic tool?
   [+ ] Q02 [SS-Asst   ] Which organization launched an AI Lab in January?
   ...
   ────────────────────────────────
   Category Breakdown:
     SS-Pref:      2/2 correct (100.0%)
     SS-Asst:      2/2 correct (100.0%)
     Temporal:     2/2 correct (100.0%)
   ────────────────────────────────
     Total:       12/12 correct (100.0%)

完整评估流程（HuggingFace 全量数据）请参阅 :doc:`/advanced-user/scenarios/longmemeval-eval`。

示例源码：`examples/longmemeval/README.md <../../examples/longmemeval/README.md>`_
