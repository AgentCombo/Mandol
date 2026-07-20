LLM 成本优化
==============

成本来源
--------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 操作
     - 每次调用内容
   * - 会话边界检测
     - 每次 session_check_interval 条记忆调用一次
   * - 实体提取 + 去重
     - 每个会话一次提取 + 分批去重
   * - 事件提取 + 去重
     - 每个会话一次提取 + 分批去重
   * - 摘要生成
     - 每个会话生成 4 类摘要
   * - 跨会话合并
     - 每次 build_high_level 最多 1 次

降低策略
--------

1. **增大 check_interval**：减少 LLM 检测频率
   .. code-block:: yaml

      session_check_interval: 50

2. **使用轻量模型**：
   .. code-block:: yaml

      llm:
        model: "gpt-4o-mini"

3. **减少实体/事件去重**：
   .. code-block:: yaml

      max_entities_per_llm: 100
      max_events_per_llm: 100

4. **批量处理**：积累足够多数据后一次性 build，减少 build 频率

成本预估
--------

以 gpt-4o-mini 为例，1000 条对话（约 50 个会话）的 build 成本约 $0.50-$2.00。
