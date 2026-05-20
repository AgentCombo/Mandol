你的第一条完整记忆
=====================

本章从零开始，走完一个完整的**创建 → 添加 → 构建 → 检索 → 保存 → 加载**闭环。

完整代码（可直接运行）
-----------------------

.. code-block:: python

   from mandol import MemorySystem, MemoryUnit, Uid

   # 1. 创建系统
   system = MemorySystem.from_yaml_config("config.yaml")

   # 2. 添加几条对话
   conversations = [
       ("msg_1", "我下周二要和李总开会讨论Q2计划", "2024-03-11T09:00:00"),
       ("msg_2", "李总希望重点看海外市场的增长数据", "2024-03-11T09:01:00"),
       ("msg_3", "周末想带孩子去科技馆，有什么推荐吗", "2024-03-16T10:00:00"),
       ("msg_4", "科技馆最近有航天主题展，很适合小朋友", "2024-03-16T10:01:00"),
   ]
   for uid, text, ts in conversations:
       system.add(MemoryUnit(
           uid=Uid(uid),
           raw_data={"text_content": text},
           metadata={"timestamp": ts},
       ))

   print("已添加 4 条记忆")

   # 3. 构建高阶记忆
   report = system.build_high_level(mode="auto")
   print(f"已构建 {report.sessions_processed} 个会话")

   # 4. 检索
   queries = [
       "我下周有什么安排？",
       "李总想了解什么数据？",
       "周末去哪里玩？",
   ]
   for q in queries:
       hits = system.holistic_retrieve(q, top_k=3)
       print(f"\n查询: {q}")
       if hits:
           best = hits[0]
           print(f"  最相关: {best.unit.raw_data['text_content'][:80]}")
           print(f"  置信度: {best.final_score:.3f}")

   # 5. 查看系统状态（输出单元数、空间数、图谱节点/边、索引状态等运行指标）
   print(system.monitor)

   # 6. 保存
   system.save("./my_first_memory")

   # 7. 加载验证
   system2 = MemorySystem.load("./my_first_memory")
   hits = system2.holistic_retrieve("李总", top_k=2)
   print(f"\n从文件加载后检索 '李总':")
   for hit in hits:
       print(f"  {hit.final_score:.3f} | {hit.unit.raw_data['text_content'][:80]}")

预期输出参考
-------------

.. code-block::

   已添加 4 条记忆
   已构建 2 个会话
   [MemSys] units=8 | spaces=5 | graph:12n/8e | idx:8↑/0↓ | pend:0u/0e/0et | sess:2(avg2) | mem:156.6MB | DIRTY

   查询: 我下周有什么安排？
     最相关: 我下周二要和李总开会讨论Q2计划
     置信度: 0.923

   查询: 李总想了解什么数据？
     最相关: 李总希望重点看海外市场的增长数据
     置信度: 0.951

   查询: 周末去哪里玩？
     最相关: 周末想带孩子去科技馆，有什么推荐吗
     置信度: 0.912

   从文件加载后检索 '李总':
     0.968 | 我下周二要和李总开会讨论Q2计划
     0.923 | 李总希望重点看海外市场的增长数据

其中 ``print(system.monitor)`` 输出的状态行含义：

.. list-table::
   :widths: 18 82

   * - 字段
     - 含义
   * - ``units``
     - MemoryUnit 总数
   * - ``spaces``
     - MemorySpace 数量
   * - ``graph:Nn/Ee``
     - 图节点数 / 边数
   * - ``idx:P↑/U↓``
     - 向量索引中已提升 / 未提升数量
   * - ``pend:Uu/Ee/Et``
     - 待处理队列：units / events / entities
   * - ``sess:N(avgS)``
     - 会话总数（平均每个会话的 unit 数）
   * - ``mem:XX.XMB``
     - 进程物理内存 (MB)
   * - ``DIRTY/CLEAN``
     - 是否有未持久化的变更

完整的监控 API 见 :doc:`/developer/api-reference/core/memory-monitor`。

处理流程说明
-------------

1. 系统通过 LLM 语义分析检测到工作讨论和周末计划是不同话题，分割为 2 个会话
2. ``build_high_level`` 提取了实体（李总、科技馆）、事件（开会、参观）和摘要
3. 检索「周末去哪里」时，系统通过语义相似度匹配「科技馆」和「航天展」的关联
4. 保存加载后记忆完全恢复，无需重新构建

.. warning::

   **关于 ``text_content`` 字段**：

   - 字段名必须是 ``text_content``，不能写成 ``text``、``content``、``body`` 等
   - 内容必须是**纯文本**，不支持 PDF、Markdown、Word 等格式，需先用外部工具提取为纯文本
   - 系统仅对 ``text_content`` 和 ``image_path`` 两个键自动向量化，其他字段仅存储不检索

   详见 :doc:`/shared/data-format-guide`。

下一步
------

- :doc:`understanding-results` — 理解检索结果中每个字段的含义
- :doc:`scenarios/index` — 查看具体业务场景的示例
