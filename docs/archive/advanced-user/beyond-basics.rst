超越基础：高级用户可以做什么
===============================

高级用户在「添加→构建→检索」三步之外，还需要更精细的记忆管理能力。

公开接口总览
------------

MemorySystem 除了 ``add/build_high_level/holistic_retrieve`` 外，还提供了以下管理接口：

.. code-block:: python

   system = MemorySystem.from_yaml_config("config.yaml")

   # === 空间管理 ===
   system.semantic_map.create_space("客服-用户A")
   system.semantic_map.attach_child_space("客服-用户A", "会话-20240301")
   spaces = system.semantic_map.list_spaces()

   # === 图管理 ===
   system.semantic_graph.add_relationship(uid_a, uid_b, "RELATED_TO")
   neighbors = system.semantic_graph.get_explicit_neighbors(uid_a)
   system.semantic_graph.delete_relationship(uid_a, uid_b, "RELATED_TO")

   # === 精细化检索 ===
   hits = system.retrieve_by_view("投诉内容", view="knowledge")
   hits = system.retrieve_in_space("订单状态", space_name="客服-用户A")

   # === 状态监控 ===
   print(system.monitor)                       # 紧凑单行状态
   stats = system.monitor.to_dict()            # 程序化访问

   # === 状态维护 ===
   system.flush()
   stats = system.semantic_map.count_units()

检索接口总览
------------

.. list-table::
   :header-rows: 1
   :widths: 22 8 10 14 14 32

   * - 接口
     - 状态
     - 适合用户
     - 记忆层级
     - 数据结构
     - 说明
   * - ``holistic_retrieve`` / ``search``
     - 公开
     - 基础/高级
     - BASE+ENTITY+EVENT+SUMMARY
     - Dense+BM25+Sparse+Graph+Reranker
     - 单次调用检索全部记忆层级
   * - ``retrieve_by_view``
     - 公开
     - 高级
     - 由 view 决定
     - 同上
     - 按语义视角过滤
   * - ``retrieve_in_space``
     - 公开
     - 高级
     - 由 space_name 决定
     - 同上
     - 按空间范围过滤
   * - ``search_by_text``
     - 公开
     - 开发者
     - 由 space_names 决定
     - Dense 向量
     - 直接向量检索
   * - ``search_by_text_with_rerank``
     - 公开
     - 开发者
     - 由 space_names 决定
     - Dense + Reranker
     - 向量检索+重排序
   * - ``bfs_expand_units``
     - 公开
     - 开发者
     - 全部
     - GraphStore (BFS)
     - 图遍历扩展

完整的检索接口参考（含签名、参数、view 映射表）见 :doc:`/shared/retrieval-reference`。

retrieve_by_view 视图映射
~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 20 28

   * - view 值
     - 记忆层级
     - 说明
   * - ``base_memory``
     - BASE
     - 原始对话/文档
   * - ``entity_relation``
     - ENTITY
     - 实体与关系
   * - ``event_causal``
     - EVENT
     - 事件与因果
   * - ``emotional``
     - SUMMARY
     - 情感摘要
   * - ``episodic``
     - SUMMARY
     - 情节摘要
   * - ``knowledge``
     - SUMMARY
     - 知识摘要
   * - ``procedural``
     - SUMMARY
     - 程序性摘要
   * - ``insights``
     - SUMMARY
     - 洞察

管理接口
--------

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - 接口类型
     - 所属层
     - 用途
   * - ``create_space`` 等
     - SemanticMapService（管理）
     - 空间的 CRUD 管理
   * - ``add_relationship`` 等
     - SemanticGraphService（管理）
     - 图关系的 CRUD 管理
   * - ``flush``
     - MemorySystem（管理）
     - 持久化缓存到存储

预想检索接口
------------

.. warning:: ⚠️ Planned — 以下接口已设计但尚未实现，签名可能变更。当前可通过组合现有接口达到类似效果。完整设计见 :doc:`/shared/retrieval-reference` 。

.. list-table::
   :header-rows: 1
   :widths: 25 10 18 47

   * - 接口
     - 状态
     - 记忆层级
     - 核心价值
   * - ``retrieve_event_causal_chain``
     - 预想
     - EVENT
     - 因果链追溯，回答"为什么"
   * - ``retrieve_entity_subgraph``
     - 预想
     - ENTITY
     - 实体关系全景
   * - ``smart_quantized_query``
     - 预想
     - 全部
     - Token 预算约束下最大化信息密度
   * - ``retrieve_with_reasoning_path``
     - 预想
     - 全部
     - 推理路径可解释
   * - ``retrieve_entity_timeline``
     - 预想
     - BASE+EVENT
     - 时间线视角
   * - ``retrieve_session_context``
     - 预想
     - BASE
     - 会话级上下文恢复
   * - ``trace_evidence``
     - 预想
     - 全部→BASE
     - 自顶向下溯源（EVIDENCED_BY）
   * - ``trace_coref``
     - 预想
     - BASE→ENTITY/EVENT
     - 自底向上共指消解（COREF）
   * - ``retrieve_summary_evidence_chain``
     - 预想
     - SUMMARY→BASE
     - 摘要溯源链
   * - ``retrieve_entity_involvement``
     - 预想
     - ENTITY+EVENT
     - 实体参与的所有事件（INVOLVES）

接下来，请根据需要选择对应的章节深入学习。
