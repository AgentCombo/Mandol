场景三：知识库问答系统
========================

.. note::

   本文档已迁移至 :doc:`/basic-user/scenarios/knowledge-base`。本页面将在后续版本中移除，请更新你的书签。

本场景演示如何使用 Mandol 构建企业知识库问答系统，将文档/FAQ 导入记忆系统，支持精准语义检索。

场景说明
--------

企业知识库通常包含大量政策文档、操作流程、FAQ 等。传统关键词搜索难以处理用户用词与文档用词不一致的情况。Mandol 能够：

- 自动将文档导入并构建语义索引
- 支持语义匹配，即使用词不同也能找到相关内容
- 自动提取知识实体，构建知识关系图

完整代码示例
------------

可运行示例：``examples/knowledge_base/run_knowledge_base.py``

.. code-block:: python

   from mandol import MemorySystem, MemoryUnit, Uid

   system = MemorySystem.from_yaml_config("config.yaml")

   knowledge_units = [
       MemoryUnit(
           uid=Uid("kb_001"),
           raw_data={"text_content": "公司年假政策：入职满1年可享10天年假，满5年15天，满10年20天"},
           metadata={"timestamp": "2024-01-01T00:00:00", "source": "hr_policy"},
       ),
       MemoryUnit(
           uid=Uid("kb_002"),
           raw_data={"text_content": "报销流程：填写报销单→部门经理审批→财务审核→打款，周期约5个工作日"},
           metadata={"timestamp": "2024-01-01T00:00:00", "source": "finance_policy"},
       ),
       MemoryUnit(
           uid=Uid("kb_003"),
           raw_data={"text_content": "远程办公规定：每周最多2天远程，需提前一天在OA系统申请"},
           metadata={"timestamp": "2024-01-01T00:00:00", "source": "hr_policy"},
       ),
       MemoryUnit(
           uid=Uid("kb_004"),
           raw_data={"text_content": "加班补偿：工作日加班按1.5倍工资计算，周末加班按2倍，法定节假日按3倍"},
           metadata={"timestamp": "2024-01-01T00:00:00", "source": "hr_policy"},
       ),
       MemoryUnit(
           uid=Uid("kb_005"),
           raw_data={"text_content": "新员工入职培训为期3天，包括公司文化、制度学习和部门介绍"},
           metadata={"timestamp": "2024-01-01T00:00:00", "source": "hr_policy"},
       ),
   ]
   system.add_many(knowledge_units)

   report = system.build_high_level(mode="auto")

   hits = system.holistic_retrieve("我可以在家办公吗？", top_k=3)

   knowledge = system.retrieve_by_view("休假有多少天？", view="knowledge", top_k=3)

   system.save("./kb_memory")

检索结果说明
^^^^^^^^^^^^

**holistic_retrieve("我可以在家办公吗？")**

用户查询"在家办公"，而文档中使用的是"远程办公"。Mandol 的语义检索能理解两者含义相同，预期返回 ``kb_003`` （远程办公规定）。

这体现了 Mandol 的核心优势——**语义匹配而非关键词匹配**：

- 关键词搜索："在家办公" ≠ "远程办公" → 无结果
- 语义检索："在家办公" ≈ "远程办公" → 精准命中

**retrieve_by_view("休假有多少天？", view="knowledge")**

知识视角检索从知识实体空间中查找。预期返回年假政策的知识摘要，包含"入职1年→10天"、"入职5年→15天"、"入职10年→20天"等结构化知识。

语义匹配 vs 关键词匹配
----------------------

.. list-table::
   :header-rows: 1
   :widths: 25 37 38

   * - 特性
     - 关键词搜索
     - Mandol 语义检索
   * - 匹配方式
     - 精确词匹配
     - 语义向量相似度
   * - 同义词处理
     - 无法处理
     - 自动理解同义词
   * - 查询"在家办公"
     - 无法匹配"远程办公"
     - 精准匹配"远程办公"
   * - 查询"加班费"
     - 需精确包含"加班补偿"
     - 自动关联"加班补偿"条款
   * - 多路召回
     - 单路
     - Dense + BM25 + Sparse 三路
   * - 图扩展
     - 无
     - BFS 扩展发现关联知识

批量导入建议
------------

当知识库文档较多时，建议：

1. **使用 ``add_many`` 批量导入**：比逐条 ``add`` 更高效，减少索引重建次数
2. **合理设置 ``chunk_max_tokens``**：长文档建议设为 1024，短 FAQ 建议设为 256
3. **关闭会话分割**：知识库导入通常无会话概念，设置 ``session_time_gap_seconds: 86400``
4. **分批构建**：大量文档可分批导入并调用 ``build_high_level(mode="auto")``，增量模式仅处理新增内容
