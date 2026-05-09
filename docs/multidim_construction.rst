多视角构建与表示
==================

.. note::

   本文档内容已迁移至 :doc:`/shared/memory-pipeline/detailed-flow` 中的「多视角记忆表示」章节。本页面将在后续版本中移除，请更新你的书签。

本节介绍 Mandol 多视角语义记忆系统的构建流程，以及每类子记忆在系统中的具体表示方式。

系统架构概览
------------

.. mermaid::

   graph TB
       A[原始对话数据] --> B[MemorySystem.add]
       B --> C[SemanticMap]
       C --> D[分块与 Embedding]
       D --> E[MemoryUnit]
       E --> F[VectorIndex 索引]
       E --> G[BM25Index 索引]
       E --> H[SparseIndex 索引]
       
       I[build_high_level] --> J[SessionManager]
       J --> K[会话分割]
       K --> L[MultiDimSemanticGraph]
       
       L --> M[LayoutNormalization]
       L --> N[SemanticSimilarity]
       L --> O[HighLevelSummary]
       L --> P[EventCausal]
       L --> Q[EntityRelation]
       
       M --> R[空间层级]
       N --> S[语义关系边]
       O --> T[摘要单元]
       P --> U[事件因果边]
       Q --> V[实体关系边]

构建流程
--------

整个构建流程分为以下几个阶段：

1. **数据输入**：通过 ``add()`` 方法添加原始对话数据
2. **分块与向量化**：自动分块并生成稠密/稀疏向量
3. **会话分割**：基于 LLM 检测会话边界
4. **空间布局**：创建多维度空间层级
5. **维度构建**：提取摘要、实体、事件等并建立关系

空间命名策略
------------

``SpaceNamingPolicy`` 负责为每个会话生成唯一的命名空间。空间层级结构如下：

.. code-block::

   root
   ├── base_memory_{suffix}          # 基础记忆（原始单元）
   └── high_level_memory_{suffix}    # 高阶记忆
       ├── episodic_{suffix}         # 情景记忆
       │   ├── episodic_summary      # 情景摘要
       │   └── episodic_event        # 情景事件
       ├── knowledge_{suffix}        # 知识记忆
       │   ├── knowledge_summary     # 知识摘要
       │   └── knowledge_entity      # 知识实体
       ├── emotional_{suffix}        # 情感记忆
       ├── procedural_{suffix}       # 程序记忆
       └── insights_{suffix}         # 洞察记忆

其中 ``{suffix}`` 是基于会话起始消息索引生成的唯一标识（如 ``msg_0_25``）。

多视角记忆表示
--------------

本节展示每类子记忆在系统中的具体表示方法，包括节点结构、边类型和图结构示例。

.. note::

   以下所有多视角记忆单元均由系统在 ``build_high_level()`` 中自动构建，
   用户无需也不应手动创建。此处展示的节点结构仅用于帮助理解系统内部的记忆表示模式。

.. _representation-base-memory:

基础记忆 (Base Memory)
^^^^^^^^^^^^^^^^^^^^^^

基础记忆存储原始对话单元，是系统的底层数据源。

**节点表示**：基础对话节点

.. code-block:: python

   MemoryUnit(
       uid="dialogue_msg_001",
       raw_data={
           "text_content": "我昨天去了北京，参观了故宫和长城。",
           "speaker": "user",
           "role": "user",
       },
       metadata={
           "timestamp": "2024-01-15T10:00:00",
           "space_name": "root_base_memory_msg_0_25",
           "session_id": "session_001",
           "chunk_id": "chunk_0",
       },
       embedding=[0.1, 0.2, ..., 0.768],
       sparse_embedding={12: 0.5, 45: 0.3, ...},
   )

**边类型**：

- ``PRECEDES``：时序边（前一条对话指向后一条）
- ``FOLLOWS``：时序边（后一条对话指向前一条）
- ``SEMANTIC_SIMILAR``：语义相似边（基于向量相似度阈值）

**图结构示例**：

.. mermaid::

   graph LR
       D1["dialogue_001<br>我去北京出差"] -->|PRECEDES| D2["dialogue_002<br>参观了故宫"]
       D2 -->|PRECEDES| D3["dialogue_003<br>还去了长城"]
       D2 -->|FOLLOWS| D1
       D3 -->|FOLLOWS| D2
       D1 -.->|SEMANTIC_SIMILAR| D2

.. _representation-entity-relation:

实体关系 (Entity Relation)
^^^^^^^^^^^^^^^^^^^^^^^^^^

实体关系视角从对话中提取命名实体，并建立实体间的语义关系。

**节点表示**：实体节点

.. code-block:: python

   MemoryUnit(
       uid="entity_beijing_001",
       raw_data={
           "text_content": "北京",
           "entity_name": "北京",
           "entity_type": "Place",
           "description": "中国的首都，政治文化中心",
       },
       metadata={
           "space_name": "root_knowledge_entity_msg_0_25",
           "entity_type": "Place",
           "entity_id": "beijing_001",
           "session_id": "session_001",
           "mentions": ["dialogue_msg_001", "dialogue_msg_002"],
       },
       embedding=[...],
   )

   MemoryUnit(
       uid="entity_gugong_001",
       raw_data={
           "text_content": "故宫",
           "entity_name": "故宫",
           "entity_type": "Place",
           "description": "明清两代皇家宫殿，位于北京中轴线",
       },
       metadata={
           "space_name": "root_knowledge_entity_msg_0_25",
           "entity_type": "Place",
           "entity_id": "gugong_001",
           "mentions": ["dialogue_msg_002"],
       },
       embedding=[...],
   )

**边类型**：

- ``RELATED_TO``：通用关系边（含子类型：hometown、lives_in、works_at、located_in、part_of）
- ``COREF``：**共指边（建立在基础对话记忆单元→实体之间）**，表示对话单元对实体的提及指代关系
- ``ALIAS_OF``：别名边（实体别名关系）
- ``EVIDENCED_BY``：溯源边（实体指向提及它的原始对话）

.. note::

   **指代消解机制说明**：

   在新的指代消解流程中，``COREF`` 边直接建立在 **基础对话记忆单元 → 全局实体** 之间，
   表示该对话单元"提及"了该全局实体。当多个对话单元提及同一全局实体时，
   通过以下两条边形成完整的指代链路：

   1. EVIDENCED_BY 边（实体 → 对话）：表示实体被哪些原始对话所支撑/提及
   2. COREF 边（对话 → 实体）：表示对话单元对实体的具体指代关系

   这种设计使得跨会话的同一指代可以通过图遍历快速定位所有相关对话单元。

**图结构示例**：

.. mermaid::

   graph LR
       E1["北京<br>Place"] -->|RELATED_TO<br>located_in| E2["故宫<br>Place"]
       E2 -->|RELATED_TO<br>part_of| E1
       E1 -->|RELATED_TO<br>located_in| E3["长城<br>Place"]

       D1["dialogue_001"] -->|COREF<br>提及| E1
       D2["dialogue_002"] -->|COREF<br>提及| E2
       D3["dialogue_003"] -->|COREF<br>提及| E1

       D1 -.->|EVIDENCED_BY| E1
       D2 -.->|EVIDENCED_BY| E2

.. _representation-event-causal:

事件因果 (Event Causal)
^^^^^^^^^^^^^^^^^^^^^^^

事件因果视角从对话中提取事件，并建立事件间的因果关系链。

**节点表示**：事件节点

.. code-block:: python

   MemoryUnit(
       uid="event_visit_beijing_001",
       raw_data={
           "text_content": "用户去了北京出差",
           "event_type": "action_event",
           "participants": ["user", "北京"],
           "time": "2024-01-14",
       },
       metadata={
           "space_name": "root_episodic_event_msg_0_25",
           "event_type": "action_event",
           "event_id": "visit_beijing_001",
           "session_id": "session_001",
           "evidence_uids": ["dialogue_msg_001"],
       },
       embedding=[...],
   )

   MemoryUnit(
       uid="event_visit_gugong_001",
       raw_data={
           "text_content": "用户参观了故宫",
           "event_type": "action_event",
           "participants": ["user", "故宫"],
       },
       metadata={
           "space_name": "root_episodic_event_msg_0_25",
           "event_type": "action_event",
           "event_id": "visit_gugong_001",
           "evidence_uids": ["dialogue_msg_002"],
       },
       embedding=[...],
   )

**边类型**：

- ``CAUSES``：因果关系（事件A导致事件B）
- ``CAUSED_BY``：被因果关系（事件B被事件A导致）
- ``INVOLVES``：事件-实体边（含子类型：participant、location、organizer、victim）
- ``PRECEDES`` / ``FOLLOWS``：时序边（事件发生的先后顺序）
- ``EVIDENCED_BY``：溯源边（事件指向支撑它的原始对话）

**图结构示例**：

.. mermaid::

   graph LR
       EV1["出差北京"] -->|CAUSES| EV2["参观故宫"]
       EV2 -->|CAUSES| EV3["了解历史文化"]
       EV1 -->|PRECEDES| EV2
       EV2 -->|PRECEDES| EV3
       EV1 -.->|INVOLVES<br>location| E1["北京"]
       EV2 -.->|INVOLVES<br>location| E2["故宫"]
       D1["dialogue_001"] -.->|EVIDENCED_BY| EV1
       D2["dialogue_002"] -.->|EVIDENCED_BY| EV2

.. _representation-emotional-summary:

情感总结 (Emotional Summary)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

情感总结视角捕捉用户在对话中表达的情感状态和态度。

**节点表示**：情感总结节点

.. code-block:: python

   MemoryUnit(
       uid="emotional_summary_msg_0_25",
       raw_data={
           "text_content": '{"user_preferences": ["喜欢历史文化景点", "偏好深度游"], "emotional_reactions": ["兴奋", "自豪", "印象深刻"], "behavioral_patterns": ["主动了解景点背景", "详细记录参观体验"]}',
           "summary_type": "emotional",
       },
       metadata={
           "space_name": "root_emotional_msg_0_25",
           "summary_type": "emotional",
           "session_id": "session_001",
           "evidence_uids": ["dialogue_msg_001", "dialogue_msg_002"],
       },
       embedding=[...],
   )

**边类型**：

- ``EVIDENCED_BY``：溯源边（情感总结指向支撑它的原始对话）
- ``SEMANTIC_SIMILAR``：与其他情感总结的语义相似边

**图结构示例**：

.. mermaid::

   graph LR
       EM["情感总结<br>兴奋、自豪"] -.->|EVIDENCED_BY| D1["dialogue_001"]
       EM -.->|EVIDENCED_BY| D2["dialogue_002"]
       EM -.->|EVIDENCED_BY| D3["dialogue_003"]

.. _representation-episodic-summary:

情景总结 (Episodic Summary)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

情景总结视角对会话中的事件进行高层概括，形成可快速检索的情景摘要。

**节点表示**：情景总结节点

.. code-block:: python

   MemoryUnit(
       uid="episodic_summary_msg_0_25",
       raw_data={
           "text_content": '{"timeline": ["2024-01-14 抵达北京", "2024-01-15 参观故宫", "2024-01-16 游览长城"], "key_people": ["用户"], "main_events": ["北京出差", "参观故宫", "游览长城"], "location_info": ["北京", "故宫", "长城"], "event_relationships": ["出差导致参观故宫", "故宫参观后游览长城"]}',
           "summary_type": "episodic",
           "time_range": "2024-01-14 ~ 2024-01-16",
           "key_events": ["出差北京", "参观故宫", "游览长城"],
           "key_entities": ["北京", "故宫", "长城"],
       },
       metadata={
           "space_name": "root_episodic_summary_msg_0_25",
           "summary_type": "episodic",
           "session_id": "session_001",
           "evidence_uids": ["event_visit_beijing_001", "event_visit_gugong_001"],
       },
       embedding=[...],
   )

**边类型**：

- ``EVIDENCED_BY``：溯源边（情景总结指向支撑它的事件/对话）

**图结构示例**：

.. mermaid::

   graph LR
       ES["情景总结<br>北京出差之行"] -.->|EVIDENCED_BY| EV1["出差北京"]
       ES -.->|EVIDENCED_BY| EV2["参观故宫"]
       ES -.->|EVIDENCED_BY| EV3["游览长城"]

.. _representation-knowledge-summary:

知识总结 (Knowledge Summary)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

知识总结视角提炼对话中涉及的知识点和事实信息。

**节点表示**：知识总结节点

.. code-block:: python

   MemoryUnit(
       uid="knowledge_summary_msg_0_25",
       raw_data={
           "text_content": '{"core_concepts": ["北京-中国首都", "故宫-皇家宫殿", "长城-军事防御工程"], "key_facts": ["北京是中国的政治文化中心", "故宫位于北京中轴线", "长城是中国古代防御工程"], "techniques_methods": [], "prerequisites_knowledge": ["中国历史基础知识"], "related_concepts": ["明清历史", "古代建筑", "世界文化遗产"]}',
           "summary_type": "knowledge",
           "facts": [
               {"subject": "北京", "predicate": "是", "object": "中国首都"},
               {"subject": "故宫", "predicate": "是", "object": "明清皇家宫殿"},
               {"subject": "长城", "predicate": "是", "object": "古代军事防御工程"},
           ],
       },
       metadata={
           "space_name": "root_knowledge_summary_msg_0_25",
           "summary_type": "knowledge",
           "session_id": "session_001",
           "evidence_uids": ["dialogue_msg_001", "dialogue_msg_002"],
       },
       embedding=[...],
   )

**边类型**：

- ``EVIDENCED_BY``：溯源边（知识总结指向支撑它的原始对话/实体）

**图结构示例**：

.. mermaid::

   graph LR
       KS["知识总结<br>北京故宫长城知识"] -.->|EVIDENCED_BY| D1["dialogue_001"]
       KS -.->|EVIDENCED_BY| E1["北京实体"]
       KS -.->|EVIDENCED_BY| E2["故宫实体"]

.. _representation-procedural-summary:

程序总结 (Procedural Summary)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

程序总结视角提取对话中涉及的操作步骤、方法和技巧。

**节点表示**：程序总结节点

.. code-block:: python

   MemoryUnit(
       uid="procedural_summary_msg_0_25",
       raw_data={
           "text_content": '{"process_name": ["故宫参观流程"], "key_steps": ["提前网上购票", "从午门进入沿中轴线参观", "重点看太和殿乾清宫御花园", "全程约2-3小时"], "decision_points": ["是否需要导游讲解", "选择游览路线（精华线/全景线）"], "preconditions": ["提前预约门票", "携带身份证件"], "expected_outcomes": ["完成主要景点参观", "了解故宫历史文化"], "optimization_opportunities": ["避开节假日高峰", "使用语音导览APP"]}',
           "summary_type": "procedural",
           "steps": [
               "提前网上购票",
               "从午门进入，沿中轴线参观",
               "重点看太和殿、乾清宫、御花园",
               "全程约2-3小时",
           ],
       },
       metadata={
           "space_name": "root_procedural_msg_0_25",
           "summary_type": "procedural",
           "session_id": "session_001",
           "evidence_uids": ["dialogue_msg_002"],
       },
       embedding=[...],
   )

**边类型**：

- ``EVIDENCED_BY``：溯源边（程序总结指向支撑它的原始对话）

.. _representation-insights:

洞见 (Insights)
^^^^^^^^^^^^^^^

洞见视角从对话中提炼深层次的洞察和模式识别。

**节点表示**：洞见节点

.. code-block:: python

   MemoryUnit(
       uid="insight_cultural_interest_001",
       raw_data={
           "text_content": '{"pattern_recognition": ["用户对中国历史文化景点有持续兴趣", "倾向于深度文化体验而非浅层游览"], "causal_relationships": ["历史文化兴趣导致主动了解背景信息"], "predictive_insights": ["可能对颐和园、天坛等类似景点感兴趣", "未来可能询问更多历史细节"], "behavioral_characteristics": ["详细记录参观体验", "主动了解历史文化背景"], "optimization_recommendations": ["推荐北京其他历史文化景点（颐和园、天坛、圆明园）", "提供深度讲解服务"], "risk_warnings": []}',
           "insight_type": "preference",
           "confidence": 0.78,
           "actionable_suggestion": "推荐颐和园、天坛、圆明园等北京历史文化景点",
       },
       metadata={
           "space_name": "root_insights_msg_0_25",
           "insight_type": "preference",
           "session_id": "session_001",
           "evidence_uids": [
               "dialogue_msg_001",
               "emotional_summary_msg_0_25",
               "episodic_summary_msg_0_25",
           ],
       },
       embedding=[...],
   )

**边类型**：

- ``EVIDENCED_BY``：溯源边（洞见指向支撑它的原始对话/摘要）
- ``SEMANTIC_SIMILAR``：与其他洞见的语义相似边

全局图结构总览
--------------

以下展示了完整的多视角记忆图结构：

.. note::

   **证据溯源说明**：

   - 高阶总结（情景、知识、程序、情感）的 ``EVIDENCED_BY`` 边直接指向基础对话记忆单元，
     表示这些总结是从原始对话数据中提炼得到的
   - 洞见记忆的 ``EVIDENCED_BY`` 边指向所有四类高阶总结（情景、知识、程序、情感），
     表示洞见是综合多视角信息后提炼出的深层次洞察
   - 实体关系和事件因果也通过 ``EVIDENCED_BY`` 边从基础记忆中获得证据支撑

.. mermaid::

   graph TB
       subgraph Base Memory["基础记忆 (Base Memory)"]
           D1["dialogue_001<br>我去北京出差"]
           D2["dialogue_002<br>参观了故宫"]
           D3["dialogue_003<br>还去了长城"]
           D1 -->|PRECEDES| D2
           D2 -->|PRECEDES| D3
       end

       subgraph Entity Relation["实体关系 (Entity Relation)"]
           E1["北京<br>Place"]
           E2["故宫<br>Place"]
           E3["长城<br>Place"]
           E1 -->|RELATED_TO| E2
           E1 -->|RELATED_TO| E3
       end

       subgraph Event Causal["事件因果 (Event Causal)"]
           EV1["出差北京"]
           EV2["参观故宫"]
           EV3["游览长城"]
           EV1 -->|CAUSES| EV2
           EV2 -->|CAUSES| EV3
           EV1 -.->|INVOLVES| E1
           EV2 -.->|INVOLVES| E2
       end

       subgraph Summaries["高阶总结 (Summaries)"]
           EM["情感总结<br>(Emotional)"]
           ES["情景总结<br>(Episodic)"]
           KS["知识总结<br>(Knowledge)"]
           PS["程序总结<br>(Procedural)"]
       end

       I1["洞见记忆<br>(Insights)<br>综合四类总结"]

       %% 基础记忆 -> 高阶总结 的证据溯源边
       D1 -.->|EVIDENCED_BY| EM
       D2 -.->|EVIDENCED_BY| EM
       D3 -.->|EVIDENCED_BY| EM

       D1 -.->|EVIDENCED_BY| ES
       D2 -.->|EVIDENCED_BY| ES
       D3 -.->|EVIDENCED_BY| ES

       D1 -.->|EVIDENCED_BY| KS
       D2 -.->|EVIDENCED_BY| KS
       D3 -.->|EVIDENCED_BY| KS

       D2 -.->|EVIDENCED_BY| PS

       %% 基础记忆 -> 实体关系/事件因果 的证据溯源边
       D1 -.->|EVIDENCED_BY| E1
       D2 -.->|EVIDENCED_BY| E2
       D1 -.->|EVIDENCED_BY| EV1
       D2 -.->|EVIDENCED_BY| EV2

       %% 基础记忆 -> 实体的 COREF 指代边
       D1 -->|COREF<br>提及| E1
       D2 -->|COREF<br>提及| E2
       D3 -->|COREF<br>提及| E1

       %% 高阶总结 -> 洞见的证据溯源边（洞见由四类总结共同支撑）
       EM -.->|EVIDENCED_BY| I1
       ES -.->|EVIDENCED_BY| I1
       KS -.->|EVIDENCED_BY| I1
       PS -.->|EVIDENCED_BY| I1

维度构建器
----------

系统通过 ``MultiDimSemanticGraph`` 编排五个维度构建器：

LayoutNormalizationDimension
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

负责创建所有空间层级并建立父子关系：

1. 创建 ``base_memory`` 空间（包含原始单元）
2. 创建 ``high_level_memory`` 空间
3. 创建 ``episodic``、``knowledge``、``emotional``、``procedural`` 子空间
4. 创建各子空间的摘要/实体/事件空间
5. 建立 ``insights`` 空间

SemanticSimilarityDimension
^^^^^^^^^^^^^^^^^^^^^^^^^^^

计算单元间语义相似度并添加关系边：

1. 获取空间内所有单元
2. 计算单元对的余弦相似度
3. 对相似度超过阈值的单元对添加 ``SEMANTIC_SIMILAR`` 关系

HighLevelSummaryApplicatorDimension
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

应用摘要单元到对应空间：

1. 从 ``SummaryMapReducer`` 获取会话摘要
2. 创建摘要单元并添加到对应空间
3. 建立 ``EVIDENCED_BY`` 关系连接原始单元与摘要

EventCausalApplicatorDimension
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

应用事件单元并建立因果链：

1. 从 ``EventDeduper`` 获取去重后的事件
2. 创建事件单元并添加到对应空间
3. 基于 LLM 提取的因果关系添加 ``CAUSES`` / ``CAUSED_BY`` 关系

EntityRelationApplicatorDimension
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

应用实体单元并建立实体关系：

1. 从 ``EntityDeduper`` 获取去重后的实体
2. 创建实体单元并添加到对应空间
3. 基于 LLM 提取的关系添加实体关系边
4. 建立 ``EVIDENCED_BY`` 关系连接原始单元与实体

会话管理
--------

``SessionManager`` 负责会话分割与管理：

- **LLM 驱动的会话分割**：使用 LLM 判断消息之间的会话边界
- **时间边界检测**：支持基于时间间隔的会话分割
- **主题连续性判断**：基于对话主题变化检测边界
- **异步会话构建**：支持 ``build_session_async()`` 后台处理

跨会话合并
----------

在构建完成后，系统自动执行跨会话合并：

1. **实体合并**：使用 ``EntityDeduper`` 合并不同会话中相同指代的实体
2. **事件合并**：使用 ``EventDeduper`` 合并不同会话中相同的事件

这些操作在 ``build_high_level()`` 内部自动触发，对用户透明。

构建流程图
----------

.. mermaid::

   sequenceDiagram
       participant U as 用户
       participant MS as MemorySystem
       participant SM as SessionManager
       participant MDG as MultiDimSemanticGraph
       participant LLM as LLM/Embedder

       U->>MS: add(unit)
       MS->>MS: 分块与向量化
       
       U->>MS: build_high_level(mode="auto")
       MS->>SM: 获取未处理会话
       SM->>LLM: 会话分割
       LLM-->>SM: 会话边界
       
       SM->>MDG: build_session(session)
       MDG->>MDG: LayoutNormalization
       MDG->>LLM: 提取摘要/实体/事件
       LLM-->>MDG: 提取结果
       MDG->>MDG: SemanticSimilarity
       MDG->>MDG: HighLevelSummary
       MDG->>MDG: EventCausal
       MDG->>MDG: EntityRelation
       MDG-->>MS: 构建完成
       
       MS->>MS: merge_cross_session_entities
       MS->>MS: merge_cross_session_events
       MS-->>U: BuildReport
