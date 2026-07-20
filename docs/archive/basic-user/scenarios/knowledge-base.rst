知识库问答
============

将文档/FAQ 导入记忆系统，支持语义检索。

完整代码（可直接运行）
-----------------------

可运行示例：``examples/knowledge_base/run_knowledge_base.py``

.. code-block:: python

   from mandol import MemorySystem, MemoryUnit, Uid

   system = MemorySystem.from_yaml_config("config.yaml")

   docs = [
       ("kb_1", "公司年假政策：入职满1年可享10天年假，满5年15天，满10年20天"),
       ("kb_2", "报销流程：填写报销单→部门经理审批→财务审核→打款，周期约5个工作日"),
       ("kb_3", "远程办公规定：每周最多2天远程，需提前一天在OA系统申请"),
       ("kb_4", "加班补偿：工作日加班按1.5倍工资计算，周末加班按2倍，法定节假日按3倍"),
   ]
   system.add_many([
       MemoryUnit(
           uid=Uid(uid),
           raw_data={"text_content": text},
           metadata={"timestamp": "2024-01-01T00:00:00", "source": "hr_policy"},
       )
       for uid, text in docs
   ])

   system.build_high_level(mode="auto")

   hits = system.holistic_retrieve("我可以在家办公吗？", top_k=3)
   knowledge = system.retrieve_by_view("休假有多少天？", view="knowledge", top_k=3)

   system.save("./kb_memory")

预期输出对照
-----------

**holistic_retrieve("我可以在家办公吗？")**：

.. code-block::

   最相关: 远程办公规定：每周最多2天远程，需提前一天在OA系统申请
   置信度: 0.934

虽然你问的是「在家办公」，但系统理解「在家办公」=「远程办公」，给出了精准匹配。

**retrieve_by_view("休假有多少天？", view="knowledge")**：

.. code-block::

   最相关: 知识摘要: 年假政策 - 入职1年10天, 入职5年15天, 入职10年20天
   置信度: 0.918

语义检索 vs 关键词检索
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 25 37 38

   * - 特性
     - 关键词搜索
     - Mandol 语义检索
   * - 「在家办公」匹配「远程办公」
     - ❌ 无结果
     - ✅ 精准命中
   * - 「加班费」匹配「加班补偿」
     - ❌ 需精确用词
     - ✅ 自动关联
   * - 多路召回
     - 单路
     - Dense + BM25 + Sparse

批量导入建议
-----------

- 大量文档用 ``add_many`` 批量导入
- 长文档保持默认 ``chunk_max_tokens: 512`` 即可（实测效果最佳）
- 知识库场景下，系统仍会按语义话题变化进行会话分割——「会话」不限于对话，而是语义连贯的主题单元

关于「会话」的理解
-------------------

Mandol 中的「会话」(Session) 不只是对话场景的概念。它的本质是**语义话题的边界**：

- 在对话场景中，一次会话 = 一段连贯的对话
- 在知识库场景中，一次会话 = 一组主题相关的文档
- 在日志分析场景中，一次会话 = 一段相关的事件序列

系统通过 LLM 语义分析自动检测话题边界，而非仅靠时间间隔。因此知识库等非对话场景同样适用。
