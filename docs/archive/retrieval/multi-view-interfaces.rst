多视角记忆接口
=====================

以下接口面向对话数据集的多维度检索，支持从不同视角访问记忆信息。

面向对话数据集的多视角记忆独特接口
------------------------------------

以下接口针对**对话数据集**的多视角记忆结构设计，提供针对特定语义维度的检索能力。

.. note::

   **数据集适配说明**：

   多视角记忆的构建方法和结果表示**高度依赖于数据集类型**。
   当前系统主要支持**对话数据集**（如用户对话、客服记录等），
   后续将扩展支持**代码数据集**（如代码仓库、Issue 跟踪等）。

   不同数据集类型将拥有不同的：

   - **多视角维度定义**：对话数据集包含情景/知识/程序/情感/洞见；
     代码数据集可能包含架构/依赖/API 等维度
   - **构建流程**：对话使用 LLM 摘要提取；代码可能使用 AST 分析
   - **检索接口**：本节接口专用于对话数据集，代码数据集将有独立接口集

   未来计划通过**父类派生**的方式，为不同数据集类型提供定制化的
   多视角记忆检索接口（参见 :ref:`retrieval-dataset-base-class`）。

基础记忆 (Base Memory)
^^^^^^^^^^^^^^^^^^^^^^

base_memory 空间存储原始对话单元。

get_base_units
""""""""""""""

获取基础记忆空间中的所有单元。

**签名**：

.. code-block:: python

   def get_base_units(recursive: bool = True) -> List[MemoryUnit]

**使用示例**：

.. code-block:: python

   base_units = system.graph.get_units_in_memory_space(["root_base_memory"])

get_implicit_neighbors_base
"""""""""""""""""""""""""""""

获取基础记忆的隐式邻居（基于语义相似边）。

**签名**：

.. code-block:: python

   def get_implicit_neighbors_base(
       uid: str,
       top_k: int = 5
   ) -> List[MemoryUnit]

**使用示例**：

.. code-block:: python

   # 获取与某对话语义相似的邻居
   similar = system.graph.get_implicit_neighbors(
       [dialogue_uid], top_k=5
   )

实体关系 (Entity Relation)
^^^^^^^^^^^^^^^^^^^^^^^^^^

knowledge_entity 空间存储实体单元，由实体关系边连接。

get_entities_by_type
""""""""""""""""""""

按实体类别过滤实体。

**签名**：

.. code-block:: python

   def get_entities_by_type(
       entity_type: str,
       recursive: bool = True
   ) -> List[MemoryUnit]

**使用示例**：

.. code-block:: python

   # 获取所有人物实体
   persons = system.graph.filter_memory_units(
       ms_names=["root_knowledge_entity"],
       filter_condition={"metadata.entity_type": {"eq": "Person"}}
   )

get_entity_relations
""""""""""""""""""""

获取某实体的关系边及关联实体。

**签名**：

.. code-block:: python

   def get_entity_relations(
       entity_uid: str,
       rel_type: Optional[str] = None
   ) -> List[Dict[str, Any]]

**使用示例**：

.. code-block:: python

   # 获取某实体的所有关系
   relations = system.graph.get_edges_of_unit("entity_001")

trace_provenance（暂未实现）
"""""""""""""""""""""""""""""

.. warning:: 📋 预想接口 — 此接口尚未实现，以下文档描述目标设计，API 可能变更。

**高阶记忆溯源分析**：追溯任意记忆单元的完整证据链路。

从指定单元出发，沿 ``EVIDENCED_BY`` 边递归回溯，
构建完整的证据溯源树，展示该记忆单元的数据来源和推导过程。

**签名**：

.. code-block:: python

   def trace_provenance(
       uid: str,
       max_depth: int = 5,
       include_coref: bool = True
   ) -> ProvenanceTree

**返回值类型**：

.. code-block:: python

   @dataclass
   class ProvenanceNode:
       uid: str
       unit_type: str              # "dialogue", "entity", "event", "summary", "insight"
       depth: int                  # 距离根节点的深度
       evidence_type: str          # "EVIDENCED_BY", "COREF", "CAUSES", etc.
       children: List['ProvenanceNode']

   @dataclass
   class ProvenanceTree:
       root: ProvenanceNode
       total_nodes: int
       max_depth_reached: int
       source_dialogues: List[str]  # 最终溯源到的原始对话 UID

**使用示例**：

.. code-block:: python

   # 追溯某个洞见的完整来源
   tree = system.graph.trace_provenance("insight_cultural_interest_001")

   # 获取所有支撑该洞见的原始对话
   source_dialogues = tree.source_dialogues

   # 追溯某个实体关系的来源
   entity_tree = system.graph.trace_provenance("entity_beijing_001")

**应用场景**：

- Agent 需要解释其回答或决策的依据时
- 验证某个结论是否有足够的证据支撑
- 发现记忆中的潜在错误或矛盾来源

analyze_entity_lifecycle（暂未实现）
"""""""""""""""""""""""""""""""""""""

.. warning:: 📋 预想接口 — 此接口尚未实现，以下文档描述目标设计，API 可能变更。

**实体生命周期分析**：追踪实体在对话历史中的出现、关系演变和行为模式。

分析指定实体的完整生命周期，包括：
- 首次提及时间和上下文
- 关系演变过程（新增/断裂的关系）
- 参与的事件序列
- 与其他实体的交互模式

**签名**：

.. code-block:: python

   def analyze_entity_lifecycle(
       entity_uid: str,
       include_events: bool = True,
       include_relations: bool = True,
       time_range: Optional[Tuple[str, str]] = None
   ) -> EntityLifecycle

**返回值类型**：

.. code-block:: python

   @dataclass
   class EntityMention:
       dialogue_uid: str
       timestamp: str
       context: str                   # 提及时的上下文摘要
       role: str                      # "subject", "object", "location", etc.

   @dataclass
   class RelationChange:
       relation_type: str
       target_entity: str
       change_type: str               # "added", "removed", "strengthened"
       timestamp: str
       evidence_uid: str

   @dataclass
   class EntityLifecycle:
       entity_uid: str
       entity_name: str
       first_mention: EntityMention
       last_mention: EntityMention
       total_mentions: int
       mention_timeline: List[EntityMention]
       relation_history: List[RelationChange]
       participating_events: List[str]
       interaction_patterns: Dict[str, Any]

**使用示例**：

.. code-block:: python

   # 分析"北京"实体的完整生命周期
   lifecycle = system.graph.analyze_entity_lifecycle("entity_beijing_001")

   # 查看首次提及
   print(f"首次提及: {lifecycle.first_mention.timestamp}")
   print(f"上下文: {lifecycle.first_mention.context}")

   # 查看关系演变
   for change in lifecycle.relation_history:
       print(f"{change.timestamp}: {change.change_type} {change.relation_type} -> {change.target_entity}")

**应用场景**：

- Agent 理解实体在长期对话中的角色变化
- 发现用户的兴趣转移或关注点演变
- 构建个性化的实体知识图谱

事件因果 (Event Causal)
^^^^^^^^^^^^^^^^^^^^^^^

episodic_event 空间存储事件单元，由因果边连接。

get_event_causal_chain
""""""""""""""""""""""

获取事件因果链（从某事件向前/向后追溯）。

**签名**：

.. code-block:: python

   def get_event_causal_chain(
       event_uid: str,
       direction: str = "both",
       max_hops: int = 3
   ) -> List[MemoryUnit]

**使用示例**：

.. code-block:: python

   # 获取某事件的前因后果
   chain = system.graph.search_graph_relations(
       seed_nodes=[event_uid],
       relation_types=["CAUSES", "CAUSED_BY"],
       max_depth=3
   )

extract_event_narrative_chain（暂未实现）
""""""""""""""""""""""""""""""""""""""""""

.. warning:: 📋 预想接口 — 此接口尚未实现，以下文档描述目标设计，API 可能变更。

**事件叙事链提取**：将因果链转化为自然语言叙事。

基于事件因果关系图，提取完整的叙事链条，
生成结构化的事件发展故事线，支持正向（因→果）和反向（果→因）叙事。

**签名**：

.. code-block:: python

   def extract_event_narrative_chain(
       event_uid: str,
       direction: str = "forward",
       max_length: int = 10,
       include_entities: bool = True
   ) -> NarrativeChain

**返回值类型**：

.. code-block:: python

   @dataclass
   class NarrativeStep:
       event_uid: str
       event_description: str
       causal_relation: str           # "causes", "caused_by", "precedes"
       participants: List[str]
       timestamp: Optional[str]
       evidence_strength: float

   @dataclass
   class NarrativeChain:
       root_event: str
       direction: str
       steps: List[NarrativeStep]
       full_narrative: str            # 生成的完整叙事文本
       key_turning_points: List[int]  # 关键转折点的步骤索引

**使用示例**：

.. code-block:: python

   # 提取从"出差北京"开始的正向叙事链
   narrative = system.graph.extract_event_narrative_chain(
       "event_visit_beijing_001",
       direction="forward"
   )

   # 获取生成的叙事文本
   print(narrative.full_narrative)
   # 输出示例: "用户首先出差到北京，这导致他参观了故宫，随后又游览了长城..."

   # 查看关键转折点
   for idx in narrative.key_turning_points:
       step = narrative.steps[idx]
       print(f"转折点: {step.event_description}")

**应用场景**：

- Agent 回答"发生了什么？"类问题时提供结构化叙述
- 自动生成会议纪要或事件报告
- 帮助用户回顾和理解复杂的事件序列

情感总结 (Emotional Summary)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

emotional 空间存储情感总结单元。

get_emotional_summary
""""""""""""""""""""""

获取某会话的情感总结。

**签名**：

.. code-block:: python

   def get_emotional_summary(
       session_suffix: str
   ) -> Optional[MemoryUnit]

**使用示例**：

.. code-block:: python

   summary = system.graph.get_units_in_memory_space(
       ["root_emotional_msg_0_25"]
   )

情景总结 (Episodic Summary)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

episodic_summary 空间存储情景总结单元。

get_episodic_summary
"""""""""""""""""""""

获取某会话的情景总结。

**签名**：

.. code-block:: python

   def get_episodic_summary(
       session_suffix: str
   ) -> Optional[MemoryUnit]

**使用示例**：

.. code-block:: python

   summary = system.graph.get_units_in_memory_space(
       ["root_episodic_summary_msg_0_25"]
   )

知识总结 (Knowledge Summary)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

knowledge_summary 空间存储知识总结单元。

get_knowledge_summary
""""""""""""""""""""""

获取某会话的知识总结。

**签名**：

.. code-block:: python

   def get_knowledge_summary(
       session_suffix: str
   ) -> Optional[MemoryUnit]

程序总结 (Procedural Summary)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

procedural 空间存储程序总结单元。

get_procedural_summary
"""""""""""""""""""""""

获取某会话的程序总结。

**签名**：

.. code-block:: python

   def get_procedural_summary(
       session_suffix: str
   ) -> Optional[MemoryUnit]

洞见 (Insights)
^^^^^^^^^^^^^^^

insights 空间存储洞见单元。

get_insights
""""""""""""

获取某会话或全局的洞见。

**签名**：

.. code-block:: python

   def get_insights(
       session_suffix: Optional[str] = None,
       recursive: bool = True
   ) -> List[MemoryUnit]

compare_multi_view_consistency（暂未实现）
""""""""""""""""""""""""""""""""""""""""""

.. warning:: 📋 预想接口 — 此接口尚未实现，以下文档描述目标设计，API 可能变更。

**多视角一致性对比**：对比不同视角记忆之间的一致性和互补性。

检查同一主题在不同视角（情景、知识、程序、情感）中的描述是否一致，
发现潜在的矛盾或互补信息。

**签名**：

.. code-block:: python

   def compare_multi_view_consistency(
       query: str,
       views: Optional[List[str]] = None,
       consistency_threshold: float = 0.7
   ) -> ConsistencyReport

**参数**：

- ``query``：要分析的主题或查询
- ``views``：要对比的视角列表（默认使用全部视角）
- ``consistency_threshold``：一致性阈值（0-1）

**返回值类型**：

.. code-block:: python

   @dataclass
   class ViewComparison:
       view_name: str
       relevant_units: List[MemoryUnit]
       key_points: List[str]           # 该视角的关键观点
       confidence: float               # 该视角的相关性置信度

   @dataclass
   class ConsistencyIssue:
       type: str                       # "contradiction", "gap", "reinforcement"
       involved_views: List[str]
       description: str
       severity: str                   # "low", "medium", "high"

   @dataclass
   class ConsistencyReport:
       query: str
       comparisons: Dict[str, ViewComparison]
       overall_consistency_score: float
       issues: List[ConsistencyIssue]
       synthesis: str                  # 综合各视角的一致性总结

**使用示例**：

.. code-block:: python

   # 对比用户对"北京之行"的多视角描述
   report = system.graph.compare_multi_view_consistency(
       "北京之行",
       views=["episodic", "emotional", "knowledge"]
   )

   # 检查整体一致性分数
   print(f"一致性评分: {report.overall_consistency_score}")

   # 查看发现的矛盾或互补点
   for issue in report.issues:
       if issue.type == "contradiction":
           print(f"发现矛盾: {issue.description}")

**应用场景**：

- Agent 在整合多源信息前进行一致性检查
- 发现用户表达中的潜在矛盾或变化
- 为复杂问题提供多维度的综合分析

