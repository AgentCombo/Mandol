高级用户入口
============

你已经掌握了基础的 :doc:`/basic-user/index` 流程。本节介绍记忆系统各环节的精细控制方法。

能力地图
--------

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - 我要...
     - 说明
     - 前往
   * - 管理记忆空间
     - 创建、层级化、删除空间
     - :doc:`space-management/index`
   * - 操作图关系
     - 手动增删关系、查询邻居、BFS 扩展
     - :doc:`graph-management/index`
   * - 控制会话分割
     - 调试分割策略、跨会话合并
     - :doc:`session-control/index`
   * - 选择检索策略
     - 全记忆 vs 按视图 vs 空间内 vs 通用 search()
     - :doc:`retrieval-strategies/index`
   * - 精细调参
     - 分块/会话/检索/LLM/索引 五大类参数
     - :doc:`parameter-tuning/index`
   * - 完整场景复现
     - LoCoMo 全量 19 sessions / LongMemEval 完整评估
     - :doc:`scenarios/locomo-full`
   * - 性能优化
     - 降低延迟 / 控制内存 / 节省 LLM 成本
     - :doc:`performance/index`
   * - 排查疑难问题
     - 检索质量 / 构建错误 / 性能瓶颈
     - :doc:`troubleshooting-advanced`

.. toctree::
   :maxdepth: 2
   :hidden:

   beyond-basics
   space-management/index
   graph-management/index
   session-control/index
   retrieval-strategies/index
   parameter-tuning/index
   scenarios/locomo-full
   scenarios/longmemeval-eval
   performance/index
   troubleshooting-advanced
