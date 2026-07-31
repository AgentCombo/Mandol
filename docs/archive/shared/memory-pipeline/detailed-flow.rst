高级深度：完整记忆构建与检索流程
=====================================

本节以自然语言详细描述 Mandol 从接收原始对话到返回检索结果的完整技术流程。文中不再使用过于复杂的流程图，而是在关键节点提供流程描述，方便你理解每个环节的内部机制和可调参数。

.. note::

   如果你想快速了解流程概览，请先阅读 :doc:`basic-flow`，那里用三句话总结了核心过程。

整体流程
--------

一条记忆从调用 ``add()`` 到能够通过 ``holistic_retrieve()`` 被检索到，经历了三个大阶段：

1. **写入阶段（add）**：原始数据入库，包括自动分块、向量化、相似度建边
2. **构建阶段（build_high_level）**：对已入库的记忆进行结构化加工，包括会话分割、多类型摘要生成、实体/事件提取、洞察提炼、跨会话合并
3. **检索阶段（holistic_retrieve）**：多组多路召回、融合、图扩展、重排序

下面逐一展开每个阶段的内部细节。

.. _pipeline-stage1:

阶段一：add() — 原始数据入库
-----------------------------

当你调用 ``system.add(unit)`` 时，系统依次执行以下步骤：

**1. 自动分块（Chunking）**

系统首先检查记忆单元的文本长度。如果超过 ``chunk_max_tokens``（默认 512 token），则按句子边界将文本切分为多个子单元。每个子单元保留对父单元的引用（``parent_uid``），并携带 ``chunk_index`` 标记分块序号。

分块策略：
  - 使用 ``tiktoken``（cl100k_base 编码）或启发式算法估算 token 数
  - 按句末标点（``. ! ? 。！？``）切分句子，逐句累积直到接近上限
  - 可配置 ``overlap_tokens`` 在相邻分块间保留上下文重叠

如果文本较短，不满足分块条件，则直接作为单个单元进入下一步。

**2. 向量化与存储**

对每个单元（或分块后的子单元）的文本内容，调用 EmbeddingProvider 生成稠密向量（Dense Embedding），同时为 BM25 和 TF-IDF 构建稀疏索引。随后将单元持久化到 UnitStore，并将向量写入 VectorIndex。

**3. 即时相似度建边**

新单元入库后，系统立即计算它与最近 ``similarity_recent_window``（默认 20）条已有记忆的余弦相似度。相似度超过 ``similarity_threshold``（默认 0.7）的单元对之间建立 ``SEMANTIC_SIMILAR`` 图边。已处理过的单元对会被跳过，避免重复计算。

**4. 进入待处理队列**

单元被追加到待处理队列 ``_pending_units`` 中。系统支持两种会话检测模式：

- **同步模式**：你显式调用 ``build_high_level()`` 时，批量处理队列中的所有待处理单元
- **异步模式**（默认开启）：系统在后台自动检测队列长度，当累积单元数达到阈值时触发会话检测，无需等待显式调用

异步模式下，会话检测在独立线程池中运行，不会阻塞 ``add()`` 的返回。当待处理单元超过 ``SESSION_MAX_PENDING``（默认 100）时，会触发强制刷新保护。

.. _pipeline-stage2:

阶段二：build_high_level() — 高阶语义构建
------------------------------------------

这是 Mandol 最核心的阶段。``mode="auto"`` 仅处理增量（新的待处理单元），``mode="force"`` 则执行导出-重建策略：导出所有基础单元和时序边，清空存储/索引/图，从头重新执行全部构建步骤。

.. _diagram-placeholder-1:

.. image:: /_static/images/pipeline-build-overview.png
    :alt: 高阶记忆构建流程全景图
    :align: center

下面按流程顺序逐一说明每个子阶段。

.. _pipeline-sessioning:

2.1 会话分割（Sessioning）
^^^^^^^^^^^^^^^^^^^^^^^^^^

系统将相邻的记忆单元内容送入 LLM 进行语义主题划分。

**工作原理**：

- SessionManager 取待处理队列中最近 ``MAX_CONTEXT_UNITS``（默认 20）条单元，按时间戳排序后格式化为带编号和时间的文本行
- 将这些文本连同系统提示词一起发送给 LLM，LLM 根据语义/情景边界判断哪些位置应该分割
- LLM 返回分割点列表（1-based 行号索引），以及一个 ``should_wait`` 标志，表示是否需要等待更多上下文再做判断
- 从右向左处理分割点（保持左侧索引有效），每个分割段独立成为一个 Session

**异步自调度机制**：

在异步模式下，单次 LLM 调用完成后会检查是否有新单元进入队列。如果有，自动再次调度检测，形成"有数据就检测、没数据就停"的自适应节奏。这确保了实时对话场景下，会话边界能在后台被持续发现，无需人工周期性调用。

**兜底保护**：

- 当待处理单元超过 ``SESSION_MAX_PENDING``（默认 100）时，直接强制全部 flush 为一个会话
- LLM 调用重试耗尽后，返回"不分割"的兜底结果，避免流程卡死

.. _pipeline-space-layout:

2.2 空间布局（Space Layout）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

每个新检测到的 Session 会获得唯一的 session ID（格式：``sess_YYYYMMDD_NNN``），并创建对应的记忆空间：

.. code-block::

   {root}_session_{session_id}    # 例如 default_session_sess_20260519_001

该空间作为 ``base_memory`` 的子空间，本 Session 内所有单元（包括原始对话和后续提取的高阶单元）都会被归属到这个空间下，方便按 Session 粒度检索和追溯。

同时，系统确保如下全局空间层级存在（按需创建，幂等）：

.. code-block::

   {root}_base_memory              # 基础记忆
   {root}_high_level_memory        # 高阶记忆根
   ├── {root}_episodic             # 情景记忆
   │   ├── {root}_episodic_summary #   情景摘要
   │   └── {root}_episodic_event   #   情景事件
   ├── {root}_knowledge            # 知识记忆
   │   ├── {root}_knowledge_summary#   知识摘要
   │   └── {root}_knowledge_entity #   知识实体
   ├── {root}_emotional            # 情感记忆
   ├── {root}_procedural           # 程序记忆
   └── {root}_insights             # 洞察记忆

.. _pipeline-summary:

2.3 多类型摘要生成（Summary Map-Reduce）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

对每个 Session 内的原始对话单元，系统采用 **Map-Reduce 模式** 生成四类摘要，避免单次 LLM 调用超出上下文窗口。

**Map 阶段**：
  - 将 Session 内单元按 token 预算分块（每块默认最多 2560 token，30 条单元）
  - 对每个分块，**并行**调用 LLM 进行四类提取（各类型独立 prompt，共 4 次并行调用）：

    * **情景摘要（Episodic）**：时间线、关键人物、主要事件、地点信息
    * **知识摘要（Knowledge）**：核心概念、关键事实、技术方法、前置知识
    * **情感摘要（Emotional）**：用户偏好、情感反应、行为模式
    * **程序摘要（Procedural）**：操作流程、关键步骤、决策点、前置条件

**Reduce 阶段**：
  - 对每类摘要的分块结果，两两合并送入 LLM 进行归约
  - 多轮归约后，每类最终收敛为一个 Session 级摘要
  - 每类归约相互独立，四类并行执行

**结果落盘**：
  - 每类摘要被封装为 MemoryUnit，存储到对应空间
  - 与原始对话单元之间建立 ``EVIDENCED_BY`` 图边，保证可追溯

.. _pipeline-entity-event:

2.4 实体与事件提取（Unified Fact Pipeline）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

系统使用统一事实管线（``UnifiedFactPipeline``）从 Session 对话中提取实体和事件。这一管线取代了旧版的五维度构建器架构。

**实体提取**：
  - 拼接 Session 内所有对话文本
  - 通过多信号检索（名称别名匹配 + BM25 关键词 + 向量相似度）召回已有实体作为上下文
  - 调用 LLM 从对话中识别新实体，并自动与已有实体关联（``linked_id``）
  - 提取的实体包含：名称、类型（Person/Place/Organization 等）、描述、别名

**事件提取**：
  - 基于已提取的实体和现有事件上下文，调用 LLM 从对话中识别事件
  - 提取的事件包含：事件类型、参与者、时间、地点、描述

**关系与因果提取**（与事件提取并行）：
  - **实体关系**：调用 LLM 判断实体间语义关系，类型包括 ``located_in``、``works_at``、``part_of``、``hometown`` 等
  - **事件因果**：调用 LLM 判断事件间的因果关系（``CAUSES`` / ``CAUSED_BY``）和时序关系

.. _pipeline-insight:

2.5 洞察提炼（Insight Extraction）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

四类摘要生成完毕后，系统调用 InsightMapReducer 将摘要文本送入 LLM，提炼更深层的洞察：

- **模式识别**：跨类别的行为模式和偏好
- **因果关系**：从表层事件中发现的深层因果链
- **预测性洞察**：对未来行为的推断和建议
- **行为特征**：深层的行为和思维特征
- **优化建议**：具体的可操作建议
- **风险提示**：潜在问题和警告

洞察单元同样被存储到对应空间，并与支撑它的四类摘要之间建立 ``EVIDENCED_BY`` 边。

.. _pipeline-cross-session:

2.6 跨会话合并与全局洞察
^^^^^^^^^^^^^^^^^^^^^^^^^^

单个 Session 处理完成后，系统自动执行跨会话级别的合并与累积。

**实体/事件跨会话合并（CrossSessionCorefManager）**：
  - 对于新提取的实体，通过多信号检索找到跨 Session 的候选匹配
  - 调用 LLM 裁判判断两个实体是否为同一指代（共指消解）
  - 合并后的规范实体跨多个 Session 共享，从各对话单元建立 ``COREF`` 边指向它
  - 事件合并遵循相同模式：高相似度 + 时间接近 → 合并为同一事件

**全局洞察累积（GlobalInsightManager）**：
  - 首个 Session 的洞察直接成为全局洞察的初始值
  - 后续 Session 的洞察通过 LLM 与现有全局洞察进行增量合并
  - LLM 合并策略：重叠内容融合、独有内容追加、保持多样性
  - 全局洞察作为单一 MemoryUnit（``global_insight_v1``）持久化，每次合并后重新生成向量
  - LLM 合并失败时，退化为简单的并集拼接

.. _pipeline-stage3:

阶段三：holistic_retrieve() — 统一检索
----------------------------------------

构建完成后的检索管线：

**1. 查询向量化**：对查询文本生成 Dense Embedding

**2. 分组召回**：检索请求分发到四个检索组，各组独立执行

   - **BASE**：原始对话 + 程序总结
   - **ENTITY**：知识实体
   - **EVENT**：情景事件
   - **SUMMARY**：情景/知识/情感/洞察总结

**3. 三路召回**：每组内部独立执行稠密向量、BM25 关键词、稀疏向量三路检索

**4. RRF 融合**：使用倒数排名融合（Reciprocal Rank Fusion）合并三路结果

**5. BFS 图扩展**：以融合结果 Top-K 为种子，沿图关系扩展候选集（参数：``bfs_expansion_per_seed`` / ``bfs_expansion_hops``）

**6. 全局 Rerank**：通过 Cross-Encoder 对所有候选重排序，返回最终 ``SearchHit`` 列表

.. note::

   如果 ``holistic_retrieve()`` 发现高阶记忆为空（尚未执行过 ``build_high_level()``），
   并且参数 ``auto_build_if_empty=True``（默认），系统会自动触发一次 ``build_high_level("auto")``，
   确保检索不会返回空结果。

.. _diagram-placeholder-2:

.. image:: /_static/images/pipeline-retrieval-overview.png
    :alt: 三阶段检索管线全景图
    :align: center

构建报告（BuildReport）
------------------------

``build_high_level()`` 返回一个 ``BuildReport`` 对象，包含以下字段供你了解构建结果：

- ``status``：构建状态（success / partial / failed）
- ``mode``：本次构建模式（auto / force）
- ``sessions_processed``：处理的会话数
- ``units_processed``：处理的单元数
- ``duration_seconds``：总耗时
- ``token_usage``：LLM token 消耗统计（prompt_tokens / completion_tokens / total_tokens）
- ``warnings``：构建过程中的警告信息列表
- ``error_message``：失败时的错误信息

你也可以随时通过 ``system.get_token_usage()`` 查询累计 token 消耗。

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

.. _diagram-placeholder-3:

.. image:: /_static/images/base-memory-graph.png
    :alt: 基础记忆图结构示例
    :align: center

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

**边类型**：

- ``RELATED_TO``：通用关系边（含子类型：hometown、lives_in、works_at、located_in、part_of）
- ``COREF``：共指边（建立在基础对话记忆单元 → 实体之间），表示对话单元对实体的提及指代关系
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

.. _diagram-placeholder-4:

.. image:: /_static/images/entity-relation-graph.png
    :alt: 实体关系图结构示例
    :align: center

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

**边类型**：

- ``CAUSES``：因果关系（事件A导致事件B）
- ``CAUSED_BY``：被因果关系（事件B被事件A导致）
- ``INVOLVES``：事件-实体边（含子类型：participant、location、organizer、victim）
- ``PRECEDES`` / ``FOLLOWS``：时序边（事件发生的先后顺序）
- ``EVIDENCED_BY``：溯源边（事件指向支撑它的原始对话）

.. _diagram-placeholder-5:

.. image:: /_static/images/event-causal-graph.png
    :alt: 事件因果图结构示例
    :align: center

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

证据溯源体系
^^^^^^^^^^^^

.. note::

   **证据溯源说明**：

   - 高阶总结（情景、知识、程序、情感）的 ``EVIDENCED_BY`` 边直接指向基础对话记忆单元，
     表示这些总结是从原始对话数据中提炼得到的
   - 洞见记忆的 ``EVIDENCED_BY`` 边指向所有四类高阶总结（情景、知识、程序、情感），
     表示洞见是综合多视角信息后提炼出的深层次洞察
   - 实体关系和事件因果也通过 ``EVIDENCED_BY`` 边从基础记忆中获得证据支撑

.. _diagram-placeholder-6:

.. image:: /_static/images/global-memory-graph-overview.png
    :alt: 全局多视角记忆图结构总览
    :align: center

空间层级结构
------------

每个会话的高阶记忆按以下空间层级组织：

.. code-block::

   root
   ├── base_memory_{suffix}          # 基础记忆（原始单元）
   │   └── session_{session_id}      # 各会话空间（动态创建）
   └── high_level_memory_{suffix}    # 高阶记忆
       ├── episodic_{suffix}         # 情景记忆
       │   ├── episodic_summary      # 情景摘要
       │   └── episodic_event        # 情景事件（规范事件）
       ├── knowledge_{suffix}        # 知识记忆
       │   ├── knowledge_summary     # 知识摘要
       │   └── knowledge_entity      # 知识实体（规范实体）
       ├── emotional_{suffix}        # 情感记忆
       ├── procedural_{suffix}       # 程序记忆
       └── insights_{suffix}         # 洞察记忆（全局洞察持续更新）

其中 ``{suffix}`` 是基于会话起始消息索引生成的唯一标识。关于各空间对应的检索视图和检索方式，请参见 :doc:`/shared/retrieval-reference`。
