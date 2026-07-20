检索接口参考
============

本文档列出 Mandol 所有检索接口，按所属层级分类。接口状态分为：

- **公开**：已实现，可直接使用
- **预想**：已设计但尚未实现，当前可通过组合现有接口达到类似效果
- **实验性**：已实现但 API 可能变动

统一检索接口（MemorySystem 层）
-------------------------------

通过 ``system`` 对象直接调用，是最常用的检索入口。

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
     - 按语义视角过滤（实体/事件/摘要）
   * - ``retrieve_in_space``
     - 公开
     - 高级
     - 由 space_name 决定
     - 同上
     - 按空间名精确检索
   * - ``retrieve_event_causal_chain``
     - 预想
     - 高级/开发者
     - EVENT
     - GraphStore (CAUSES/CAUSED_BY)
     - 因果链追溯，回答"为什么"
   * - ``smart_quantized_query``
     - 预想
     - 高级/开发者
     - 全部
     - 全部 + LLM 压缩
     - Token 预算约束下最大化信息密度
   * - ``retrieve_with_reasoning_path``
     - 预想
     - 高级
     - 全部
     - HybridRetriever + 加权多跳图扩展
     - 推理路径可解释
   * - ``retrieve_entity_timeline``
     - 预想
     - 高级
     - BASE+EVENT
     - UnitStore (timestamp 排序)
     - 时间线视角
   * - ``retrieve_session_context``
     - 预想
     - 基础/高级
     - BASE
     - SessionManager + UnitStore
     - 会话级上下文恢复

holistic_retrieve / search
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   hits = system.holistic_retrieve("张三去了哪里？", top_k=10, use_rerank=True)
   # 或简写
   hits = system.search("张三去了哪里？")

自动搜索全部四组记忆层级（BASE / ENTITY / EVENT / SUMMARY），执行三路召回 → RRF 融合 → BFS 图扩展 → Reranker 重排序。

retrieve_by_view
~~~~~~~~~~~~~~~~

.. code-block:: python

   hits = system.retrieve_by_view("投诉内容", view="entity_relation", top_k=10)

按预定义语义视图过滤。view 参数映射：

.. list-table::
   :header-rows: 1
   :widths: 22 30 20 28

   * - view 值
     - 对应空间
     - 记忆层级
     - 说明
   * - ``base_memory``
     - root_base_memory
     - BASE
     - 原始对话/文档
   * - ``entity_relation``
     - root_knowledge_entity
     - ENTITY
     - 实体与关系
   * - ``event_causal``
     - root_episodic_event
     - EVENT
     - 事件与因果
   * - ``emotional``
     - root_emotional
     - SUMMARY
     - 情感摘要
   * - ``episodic``
     - root_episodic_summary
     - SUMMARY
     - 情节摘要
   * - ``knowledge``
     - root_knowledge_summary
     - SUMMARY
     - 知识摘要
   * - ``procedural``
     - root_procedural
     - SUMMARY
     - 程序性摘要
   * - ``insights``
     - root_insights
     - SUMMARY
     - 洞察

retrieve_in_space
~~~~~~~~~~~~~~~~~

.. code-block:: python

   hits = system.retrieve_in_space("订单状态", space_name="客服-用户A", top_k=10)

在指定记忆空间内检索，适合已按业务逻辑组织空间的场景。

retrieve_event_causal_chain — 事件因果链检索
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning:: ⚠️ Planned — 此接口尚未实现，签名可能变更

沿 CAUSES / CAUSED_BY 边追溯事件的因果链，返回完整的前因后果。

.. code-block:: python

   result = system.retrieve_event_causal_chain(
       "项目延期",
       max_hops=3,
       direction="both",    # "forward" / "backward" / "both"
       top_k=5,
   )
   # result: CausalChainResult
   #   .chain: [CausalStep(source, target, causal_type, confidence, direction), ...]
   #   .root_event: 起始事件
   #   .leaf_events: 终端事件

**适用范围**：EVENT 层级，沿因果边遍历。典型场景："项目为什么延期？" → 供应商问题 → 物流延迟 → 订单取消。

**当前替代方案**：

.. code-block:: python

   events = system.retrieve_by_view("项目延期", view="event_causal", top_k=3)
   chain = system.graph.bfs_expand_units(
       seeds=[h.unit for h in events], per_seed=3, hops=3, rel_type="CAUSES"
   )

smart_quantized_query — 智能量化查询
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning:: ⚠️ Planned — 此接口尚未实现，签名可能变更

在 token 预算约束下，通过查询路由 → 智能去噪 → 级联装箱三阶段，产出信息密度最优的紧凑上下文。

.. code-block:: python

   result = system.smart_quantized_query(
       "张三最近的出差安排",
       max_context_tokens=2000,
       routing_strategy="auto",       # "auto" / "all" / "base_only" / "knowledge_only"
       denoise_threshold=0.5,
       compression_ratio=0.3,
       top_k=20,
   )
   # result: QuantizedQueryResult
   #   .context: "张三于1月15日前往北京出差，原因是..."  (紧凑上下文，可直接注入 LLM prompt)
   #   .source_uids: [uid1, uid2, ...]
   #   .space_distribution: {"base": 0.3, "entity": 0.4, "event": 0.2, "summary": 0.1}
   #   .routing_decision: {"selected_spaces": [...], "confidence": 0.85}
   #   .denoise_stats: {"removed": 5, "kept": 15}
   #   .total_tokens: 1847

**适用范围**：跨全部记忆层级，核心价值是 **在 token 预算约束下最大化信息密度** 。适用于：

- LLM Agent 上下文注入（对话历史 + 知识库 + 事件记忆）
- RAG 管线中的检索后处理
- 需要控制 API 成本的批量查询场景

**三阶段流程**：

::

   阶段1：智能路由 (Smart Routing)
     查询特征分析 → 空间匹配 → 交叉验证信息熵筛选
     输出: 需要检索的空间列表 + 路由置信度

   阶段2：智能去噪 (Smart Denoising)
     多维度质量评估 → 低质量过滤 → 重复/冗余消除
     输出: 去噪后的高质量候选集

   阶段3：智能上下文生成 (Smart Context Generation)
     级联装箱 → 重要性排序 → LLM 压缩摘要 → token 计数截断
     输出: 紧凑上下文字符串

retrieve_with_reasoning_path — 带推理路径的检索
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning:: ⚠️ Planned — 此接口尚未实现，签名可能变更

基于 SubgraphHopRetriever 的加权多跳图扩展，返回结果包含完整的推理路径。

.. code-block:: python

   hits = system.retrieve_with_reasoning_path(
       "为什么订单被取消了？",
       max_hops=2,
       hop_decay=0.85,
       top_k=5,
       rel_types=None,    # None = 所有关系类型
   )
   # hits: List[ReasoningHit]
   #   .unit, .final_score, .reasoning_path: [ReasoningStep, ...]
   #   ReasoningStep: source_uid, target_uid, rel_type, rel_weight, direction

**适用范围**：跨全部记忆层级，图扩展检索。典型场景："为什么订单取消？" → 订单取消 ─CAUSES→ 物流延迟 ─CAUSED_BY→ 供应商问题。

retrieve_entity_timeline — 实体时间线检索
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning:: ⚠️ Planned — 此接口尚未实现，签名可能变更

按时间排序返回与指定实体相关的所有事件和对话，形成时间线。

.. code-block:: python

   result = system.retrieve_entity_timeline(
       "张三",
       time_range=("2024-01-01", "2024-03-31"),   # 可选
       top_k=20,
   )
   # result: TimelineResult
   #   .events: [(MemoryUnit, timestamp), ...]  按时间排序
   #   .entity: 中心实体

**适用范围**：BASE + EVENT 层级。典型场景："张三最近一个月发生了什么？"

retrieve_session_context — 会话上下文检索
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning:: ⚠️ Planned — 此接口尚未实现，签名可能变更

检索指定会话的完整上下文，支持跨相邻会话扩展。

.. code-block:: python

   result = system.retrieve_session_context(
       "项目预算讨论",
       session_id=None,           # None = 自动匹配最相关会话
       include_adjacent=True,     # 包含相邻会话
       top_k=10,
   )
   # result: SessionContextResult
   #   .session_units: [MemoryUnit, ...]
   #   .session_id: 会话 ID
   #   .adjacent_sessions: [SessionContextResult, ...]  相邻会话

**适用范围**：BASE 层级。典型场景："继续上次关于项目预算的讨论"。

服务层检索接口（SemanticMapService）
-------------------------------------

通过 ``system.semantic_map`` 调用，提供更精细的检索控制。

.. list-table::
   :header-rows: 1
   :widths: 22 8 10 14 14 32

   * - 接口
     - 状态
     - 适合用户
     - 记忆层级
     - 数据结构
     - 说明
   * - ``search_by_text``
     - 公开
     - 开发者
     - 由 space_names 决定
     - Dense 向量
     - 文本查询，返回单元+分数
   * - ``search_by_text_with_rerank``
     - 公开
     - 开发者
     - 由 space_names 决定
     - Dense + Reranker
     - 文本查询+重排序
   * - ``search_by_vector``
     - 公开
     - 开发者
     - 由 space_names 决定
     - Dense 向量
     - 自行提供嵌入向量
   * - ``search_in_space``
     - 公开
     - 开发者
     - 由 space_name 决定
     - 自适应索引
     - 空间内检索，支持候选集过滤
   * - ``get_units_in_spaces``
     - 公开
     - 开发者
     - 由 space_names 决定
     - UnitStore
     - 精确空间查询，无相似度排序
   * - ``get_unit``
     - 公开
     - 开发者
     - 全部
     - UnitStore
     - 按 UID 精确查找单个单元
   * - ``list_units``
     - 公开
     - 开发者
     - 全部
     - UnitStore
     - 返回所有记忆单元

.. code-block:: python

   results = system.semantic_map.search_by_text("张三", top_k=5)

   results = system.semantic_map.search_by_text_with_rerank(
       "张三", top_k=5, recall_k=20, use_rerank=True
   )

   results = system.semantic_map.search_by_vector(query_embedding, top_k=5)

   results = system.semantic_map.search_in_space(
       "张三", space_name="客服-用户A", candidates=uid_set, top_k=5
   )

图遍历接口（SemanticGraphService）
-----------------------------------

通过 ``system.semantic_graph`` 或 ``system.graph`` 调用，提供图结构的遍历、邻居发现和层级溯源。

.. list-table::
   :header-rows: 1
   :widths: 22 8 10 14 14 32

   * - 接口
     - 状态
     - 适合用户
     - 记忆层级
     - 数据结构
     - 说明
   * - ``get_explicit_neighbors``
     - 公开
     - 开发者
     - 全部
     - GraphStore
     - 获取显式邻居（有直接边的节点）
   * - ``get_implicit_neighbors``
     - 公开
     - 开发者
     - 全部
     - 向量索引
     - 获取隐式邻居（嵌入相似但无边）
   * - ``bfs_expand_units``
     - 公开
     - 开发者
     - 全部
     - GraphStore (BFS)
     - BFS 图扩展
   * - ``get_relationship``
     - 公开
     - 开发者
     - 全部
     - GraphStore
     - 查询特定关系边
   * - ``SubgraphHopRetriever.search``
     - 实验性
     - 开发者
     - 全部
     - HybridRetriever + 加权多跳图扩展
     - 多跳推理检索，含 reasoning_path
   * - ``retrieve_entity_subgraph``
     - 预想
     - 高级
     - ENTITY
     - GraphStore (RELATED_TO/ALIAS_OF)
     - 实体关系全景
   * - ``trace_evidence``
     - 预想
     - 高级/开发者
     - 全部→BASE
     - GraphStore (EVIDENCED_BY)
     - 自顶向下溯源
   * - ``trace_coref``
     - 预想
     - 高级/开发者
     - BASE→ENTITY/EVENT
     - GraphStore (COREF)
     - 自底向上共指消解
   * - ``retrieve_summary_evidence_chain``
     - 预想
     - 高级
     - SUMMARY→BASE
     - GraphStore (EVIDENCED_BY + 关联)
     - 摘要溯源链
   * - ``retrieve_entity_involvement``
     - 预想
     - 高级
     - ENTITY+EVENT
     - GraphStore (INVOLVES)
     - 实体参与的所有事件

.. code-block:: python

   neighbors = system.graph.get_explicit_neighbors(
       [uid_a], rel_type="RELATED_TO", direction="out"
   )

   similar = system.graph.get_implicit_neighbors([uid_a], top_k=10)

   expanded = system.graph.bfs_expand_units(
       seeds=seed_units, per_seed=3, hops=2, rel_type="CAUSES"
   )

   rel = system.graph.get_relationship(uid_a, uid_b, "CAUSES")

SubgraphHopRetriever — 多跳推理检索
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mandol.retrieval import SubgraphHopRetriever

   retriever = SubgraphHopRetriever(config, semantic_map, graph, reranker)
   hits = retriever.search("为什么项目延期？", top_k=5)
   for hit in hits:
       print(f"[{hit.final_score:.3f}] {hit.unit.raw_data.get('text_content', '')}")
       for step in hit.reasoning_path:
           print(f"  → {step.rel_type} ({step.rel_weight:.2f}) → {step.target_uid}")

retrieve_entity_subgraph — 实体子图检索
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning:: ⚠️ Planned — 此接口尚未实现，签名可能变更

以指定实体为中心，沿 RELATED_TO / ALIAS_OF / LOCATED_IN 等边扩展，返回实体关系全景。

.. code-block:: python

   result = system.retrieve_entity_subgraph(
       "张三",
       max_depth=2,
       rel_types=["RELATED_TO", "ALIAS_OF"],   # None = 所有关系类型
       top_k=10,
   )
   # result: EntitySubgraphResult
   #   .center_entity: 中心实体
   #   .neighbors: [MemoryUnit, ...]
   #   .relationships: [RelationshipInfo, ...]
   #   .depth_map: {uid: 距中心跳数}

**适用范围**：ENTITY 层级，横向关系遍历。典型场景："张三的所有关系" → 同事李四、公司XX科技、家乡北京。

trace_evidence — 自顶向下溯源
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning:: ⚠️ Planned — 此接口尚未实现，签名可能变更

从高阶记忆（实体/事件/摘要/洞察）出发，沿 EVIDENCED_BY 边向下追溯到原始对话证据。

.. code-block:: python

   result = system.trace_evidence(
       uid,                       # 高阶记忆的 UID
       max_depth=2,
       top_k=10,
   )
   # result: EvidenceChainResult
   #   .source: MemoryUnit           起始高阶记忆
   #   .evidence: [MemoryUnit, ...]  原始对话证据
   #   .depth_map: {uid: 距源深度}

**适用范围**：ENTITY/EVENT/SUMMARY → BASE，沿 EVIDENCED_BY 边自顶向下。典型场景：

- "这个实体信息的原始来源是什么？"
- "这条摘要是基于哪些对话生成的？"
- "这个洞察的支撑证据有哪些？"

**当前替代方案**：

.. code-block:: python

   evidence = system.graph.get_explicit_neighbors(
       [uid], rel_type="EVIDENCED_BY", direction="out"
   )

trace_coref — 自底向上共指消解
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning:: ⚠️ Planned — 此接口尚未实现，签名可能变更

从基础对话单元出发，沿 COREF 边向上追溯到规范实体/事件，再沿 EVIDENCED_BY 边回到其他引用同一实体的对话。

.. code-block:: python

   result = system.trace_coref(
       uid,                       # 基础对话单元的 UID
       max_depth=2,
       top_k=10,
   )
   # result: CorefTraceResult
   #   .source: MemoryUnit              起始对话单元
   #   .canonical_entities: [MemoryUnit, ...]  指向的规范实体
   #   .canonical_events: [MemoryUnit, ...]    指向的规范事件
   #   .coref_chain: [MemoryUnit, ...]   引用同一实体的其他对话单元

**适用范围**：BASE → ENTITY/EVENT，沿 COREF 边自底向上。典型场景：

- "这句话中的'他'指的是谁？" → 追溯到规范实体"张三"
- "还有哪些对话提到了同一个实体？" → 通过 COREF → EVIDENCED_BY 找到所有引用

**当前替代方案**：

.. code-block:: python

   entities = system.graph.get_explicit_neighbors(
       [uid], rel_type="COREF", direction="out"
   )
   for entity in entities:
       refs = system.graph.get_explicit_neighbors(
           [entity.uid], rel_type="EVIDENCED_BY", direction="out"
       )

retrieve_summary_evidence_chain — 摘要溯源链
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning:: ⚠️ Planned — 此接口尚未实现，签名可能变更

从摘要出发，沿 EVIDENCED_BY 边追溯到原始对话，再沿 COREF 边向上到实体/事件，形成完整的溯源链。

.. code-block:: python

   result = system.retrieve_summary_evidence_chain(
       uid,                       # 摘要单元的 UID
       include_entities=True,
       include_events=True,
       top_k=10,
   )
   # result: SummaryEvidenceChainResult
   #   .summary: MemoryUnit               摘要
   #   .evidence_units: [MemoryUnit, ...]  原始对话证据
   #   .related_entities: [MemoryUnit, ...] 关联实体
   #   .related_events: [MemoryUnit, ...]   关联事件

**适用范围**：SUMMARY → BASE（EVIDENCED_BY）+ ENTITY/EVENT（横向关联）。典型场景：

- "这段摘要的完整上下文是什么？" → 原始对话 + 关联实体 + 关联事件
- 验证摘要准确性：对比摘要与原始证据

retrieve_entity_involvement — 实体参与事件检索
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning:: ⚠️ Planned — 此接口尚未实现，签名可能变更

从实体出发，沿 INVOLVES 边（反向）找到涉及该实体的所有事件，支持按角色过滤。

.. code-block:: python

   result = system.retrieve_entity_involvement(
       "张三",
       role=None,                 # "participant" / "location" / "organizer" / "victim" / None
       top_k=20,
   )
   # result: EntityInvolvementResult
   #   .entity: MemoryUnit
   #   .events: [(MemoryUnit, role), ...]   事件 + 角色
   #   .causal_chains: [[MemoryUnit, ...], ...]  从事件延伸的因果链

**适用范围**：ENTITY → EVENT，沿 INVOLVES 边反向遍历。典型场景：

- "张三参与了哪些事件？" → 返回事件列表 + 角色（参与者/组织者/受害者）
- "这个地点发生了什么？" → 返回在该地点发生的事件

层级遍历模式
~~~~~~~~~~~~

::

   ┌─────────────────────────────────────────────────────────┐
   │                    INSIGHT (洞察)                        │
   │                                                         │
   │  trace_evidence ──EVIDENCED_BY──→ SUMMARY/BASE          │
   └─────────────────────────────────────────────────────────┘
                            │
                            ▼
   ┌─────────────────────────────────────────────────────────┐
   │              SUMMARY (四类摘要)                          │
   │                                                         │
   │  retrieve_summary_evidence_chain:                       │
   │    SUMMARY ──EVIDENCED_BY──→ BASE (溯源)                │
   │    SUMMARY ──关联──→ ENTITY / EVENT (横向)              │
   └─────────────────────────────────────────────────────────┘
                            │
                            ▼
   ┌──────────────────┐  INVOLVES   ┌──────────────────┐
   │    ENTITY        │ ◄────────── │     EVENT        │
   │                  │ ──────────► │                  │
   │ retrieve_entity_ │ RELATED_TO  │ retrieve_event_  │
   │ subgraph         │             │ causal_chain     │
   │                  │             │                  │
   │ retrieve_entity_ │ ◄─INVOLVES─ │                  │
   │ involvement      │   (反向)    │                  │
   └──────────────────┘             └──────────────────┘
            ▲                              ▲
            │ COREF (自底向上)              │ COREF (自底向上)
            │                              │
   ┌────────┴──────────────────────────────┴──────────┐
   │                   BASE (对话单元)                  │
   │                                                   │
   │ trace_coref: BASE ──COREF──→ ENTITY/EVENT         │
   │ trace_evidence: ENTITY/EVENT ──EVIDENCED_BY──→ BASE│
   │ retrieve_entity_timeline: 按 timestamp 排序        │
   │ retrieve_session_context: 按 Session 分组          │
   └───────────────────────────────────────────────────┘

检索流程全景
------------

::

   用户调用
     |
     v
   MemorySystem.holistic_retrieve() / retrieve_by_view() / retrieve_in_space()
     |
     v
   MemoryRetrievalService (服务层)
     |
     +--->_search_group()
           |
           v
         HybridRetriever.search() (三路混合检索引擎)
           |
           +---> Dense:  SemanticMapService.search_by_text() → VectorIndex.search()
           +---> BM25:   BM25Index.search() / _fallback_bm25_search()
           +---> Sparse: SparseIndex.search() / _fallback_sparse_search()
           |
           v
         rrf_fusion() (RRF融合)
           |
           v
         SemanticGraphService.bfs_expand_units() (BFS图扩展)
           |
           v
         Reranker.rerank() (Cross-Encoder重排序)
           |
           v
         List[SearchHit] 返回

按用户层级的推荐
----------------

.. list-table::
   :header-rows: 1
   :widths: 15 30 55

   * - 用户层级
     - 推荐接口
     - 说明
   * - 基础
     - ``system.search(query)``
     - 一行调用，自动搜索全部记忆层级
   * - 高级
     - ``system.retrieve_by_view(query, view="...")``
     - 按语义视图名检索特定类别记忆
   * - 高级
     - ``system.retrieve_in_space(query, space_name="...")``
     - 按空间名精确检索
   * - 开发者
     - ``system.semantic_map.search_by_text_with_rerank(...)``
     - 直接操作语义图进行向量搜索+重排序
   * - 开发者
     - ``system.semantic_map.search_in_space(...)``
     - 带候选集过滤的空间级搜索
   * - 开发者
     - ``system.graph.bfs_expand_units(...)``
     - 图遍历扩展
   * - 开发者
     - ``SubgraphHopRetriever.search(...)``
     - 实验性多跳推理检索
