核心数据结构参考
================

本文档是 Mandol 记忆系统核心数据结构的完整参考，涵盖所有属性、方法和接口签名。基础概念速览见 :doc:`/basic-user/basic-concepts`，术语对照见 :doc:`/shared/glossary`。

.. _dsref-memory-unit:

记忆单元 (MemoryUnit)
----------------------

``MemoryUnit`` 是记忆系统的最小存储单元。每个单元包含原始数据、元数据和可选的向量表示。

核心属性
^^^^^^^^

.. code-block:: python

   from mandol import MemoryUnit, Uid

   unit = MemoryUnit(
       uid=Uid("dialogue_001"),
       raw_data={
           "text_content": "张三去北京出差",
           "speaker": "user",
       },
       metadata={
           "timestamp": "2024-01-15T10:00:00",
           "spaces": ["root_base_memory"],
           "session_id": "session_001",
       },
       embedding=None,
       sparse_embedding=None,
   )

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - 属性
     - 类型
     - 说明
   * - ``uid``
     - ``Uid``（定义于 ``mandol/domain/types.py``）
     - 唯一标识符
   * - ``raw_data``
     - ``Dict[str, Any]``
     - 原始数据字典，系统自动对 ``text_content`` 和 ``image_path`` 字段向量化
   * - ``metadata``
     - ``Dict[str, Any]``
     - 系统元数据，包含时间戳、空间信息等
   * - ``embedding``
     - ``Optional[numpy.ndarray]``
     - 稠密向量表示（由 EmbeddingProvider 生成）
   * - ``sparse_embedding``
     - ``Optional[numpy.ndarray]``
     - 稀疏向量表示（由 SparseIndex 生成）

关键方法
^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - 方法
     - 类型
     - 说明
   * - ``to_dict()``
     - 实例方法
     - 序列化为字典，用于持久化
   * - ``from_dict(data)``
     - 类方法
     - 从字典反序列化为 MemoryUnit
   * - ``get_user_metadata()``
     - 实例方法
     - 获取用户自定义元数据（排除 ``_system_`` 前缀的系统字段）
   * - ``touch()``
     - 实例方法
     - 更新 ``_system_updated_at`` 时间戳

使用示例
^^^^^^^^

.. note::

   以下示例展示的是**基础对话单元**的创建方式，用户通过 ``system.add(unit)`` 添加。
   多视角记忆单元（实体、事件、摘要、洞见等）由系统在 ``build_high_level()`` 中自动构建，
   **不需要也不应该**手动创建和插入，否则可能与系统自动构建的单元产生 UID 冲突或关系不一致。

.. code-block:: python

   from mandol import MemorySystem, MemoryUnit, Uid

   system = MemorySystem()

   unit = MemoryUnit(
       uid=Uid("dialogue_001"),
       raw_data={"text_content": "张三去北京出差", "speaker": "user"},
       metadata={"timestamp": "2024-01-15T10:00:00"},
   )
   system.add(unit)

   data = unit.to_dict()
   restored_unit = MemoryUnit.from_dict(data)

完整的 ``raw_data`` 字段参考见 :doc:`/shared/data-format-guide`。

.. _dsref-memory-space:

记忆空间 (MemorySpace)
----------------------

``MemorySpace`` 是组织 MemoryUnit 的逻辑空间，支持树形层级关系。每个空间可以包含多个记忆单元和多个子空间。

核心属性
^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 22 30 48

   * - 属性
     - 类型
     - 说明
   * - ``name``
     - ``SpaceName``
     - 空间名称
   * - ``unit_uids``
     - ``Set[Uid]``
     - 包含的记忆单元 UID 集合
   * - ``child_spaces``
     - ``Set[SpaceName]``
     - 子空间名称集合
   * - ``summary_text``
     - ``Optional[str]``
     - 空间摘要文本
   * - ``summary_embedding``
     - ``Optional[numpy.ndarray]``
     - 摘要向量
   * - ``metadata``
     - ``Dict[str, Any]``
     - 元数据字典

关键方法
^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 35 20 45

   * - 方法
     - 类型
     - 说明
   * - ``add_unit(uid)``
     - 实例方法
     - 添加记忆单元到空间
   * - ``remove_unit(uid)``
     - 实例方法
     - 从空间移除记忆单元
   * - ``add_child_space(name)``
     - 实例方法
     - 添加子空间
   * - ``remove_child_space(name)``
     - 实例方法
     - 移除子空间
   * - ``set_summary(text, embedding)``
     - 实例方法
     - 设置空间摘要及向量
   * - ``get_all_unit_uids(recursive=True, resolver=...)``
     - 实例方法
     - 递归获取所有单元 UID（需传入 resolver 以支持递归）
   * - ``get_all_child_space_names(recursive=True, resolver=...)``
     - 实例方法
     - 递归获取所有子空间名称
   * - ``to_dict()`` / ``from_dict(data)``
     - 实例/类方法
     - 序列化与反序列化
   * - ``touch()``
     - 实例方法
     - 更新时间戳

层级关系
^^^^^^^^

空间支持树形层级结构，由 ``SemanticMapService`` 通过 ``SpaceNamingPolicy`` 自动管理：

.. code-block::

   root
   ├── root_base_memory
   │   └── [unit_001, unit_002, ...]
   └── root_high_level_memory
       ├── root_episodic
       │   ├── root_episodic_summary
       │   └── root_episodic_event
       ├── root_knowledge
       │   ├── root_knowledge_summary
       │   └── root_knowledge_entity
       ├── root_emotional
       ├── root_procedural
       └── root_insights

使用示例
^^^^^^^^

.. code-block:: python

   from mandol.domain.memory_space import MemorySpace
   from mandol.domain.types import SpaceName, Uid

   root = MemorySpace(name=SpaceName("root"))
   root.add_child_space(SpaceName("base_memory"))
   root.add_child_space(SpaceName("knowledge_entity"))
   root.add_unit(Uid("unit_001"))

.. _dsref-semantic-map:

语义索引 (SemanticMap)
----------------------

``SemanticMapService``（位于 ``mandol.application.semantic_map``）是管理记忆存储与向量索引的核心服务。
它整合了 ``UnitStore``（单元存储）、``AdaptiveVectorIndex``（自适应向量索引）、``EmbeddingProvider``（向量化）和
``Reranker``（重排序），为语义检索提供完整基础设施。

.. note::

   **名称说明**：代码中类名为 ``SemanticMapService``，通过 ``system.semantic_map`` 属性访问。
   文档中统一简称为 SemanticMap。

构造函数
^^^^^^^^

.. code-block:: python

   SemanticMapService(
       *,
       store: UnitStore,
       index: VectorIndex,
       embedder: Optional[EmbeddingProvider] = None,
       reranker: Optional[Reranker] = None,
       default_text_key: str = "text_content",
       default_image_path_key: str = "image_path",
       max_units_in_memory: int = 10000,
   )

核心接口
^^^^^^^^

**空间管理**：

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - 方法
     - 说明
   * - ``create_space(name) -> MemorySpace``
     - 创建新空间（已存在则返回已有）
   * - ``get_space(name) -> Optional[MemorySpace]``
     - 获取空间
   * - ``list_spaces() -> List[MemorySpace]``
     - 列出所有空间
   * - ``add_unit_to_space(uid, space_name)``
     - 将单元加入指定空间
   * - ``attach_child_space(parent, child, *, ensure_exists=True)``
     - 建立空间父子关系
   * - ``ensure_child_space(parent, child) -> MemorySpace``
     - 确保子空间存在并关联

**单元管理**：

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - 方法
     - 说明
   * - ``add_unit(unit, *, space_names, ensure_embedding=True, rebuild_index_immediately=False, embedding_text=None, embedding_image_path=None)``
     - 添加记忆单元
   * - ``upsert_unit(unit, *, rebuild_index_immediately=False)``
     - 插入或更新单元
   * - ``delete_unit(uid)``
     - 删除单元（同时从所有空间移除）
   * - ``get_unit(uid) -> Optional[MemoryUnit]``
     - 获取单个单元
   * - ``list_units() -> List[MemoryUnit]``
     - 列出所有单元

**空间检索**：

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - 方法
     - 说明
   * - ``get_units_in_spaces(space_names, *, mode="union", recursive=True) -> List[MemoryUnit]``
     - 获取指定空间内的所有单元

**语义检索**：

.. list-table::
   :header-rows: 1
   :widths: 60 40

   * - 方法
     - 说明
   * - ``search_by_text(query_text, *, top_k=10, space_names=None, recursive=True) -> List[Tuple[MemoryUnit, float]]``
     - 文本语义检索
   * - ``search_by_vector(query, *, top_k=10, space_names=None, recursive=True) -> List[Tuple[MemoryUnit, float]]``
     - 向量语义检索
   * - ``search_by_text_with_rerank(query_text, *, top_k=10, recall_k=None, space_names=None, recursive=True, use_rerank=True) -> List[Tuple[MemoryUnit, float]]``
     - 带重排序的文本语义检索
   * - ``search_in_space(query_text, space_name, candidates=None, *, top_k=10, recall_k=None) -> List[Tuple[MemoryUnit, float]]``
     - 在指定空间内检索

**索引维护**：

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - 方法
     - 说明
   * - ``rebuild_index_from_store()``
     - 从存储中重建向量索引
   * - ``set_embedder(embedder)``
     - 更换向量化器
   * - ``set_reranker(reranker)``
     - 更换重排序器
   * - ``flush()``
     - 清空所有数据

使用示例
^^^^^^^^

.. code-block:: python

   custom_space = system.semantic_map.create_space("my_custom_space")
   system.semantic_map.add_unit_to_space(Uid("dialogue_001"), "my_custom_space")

   results = system.semantic_map.search_by_text("北京", top_k=10)

   entities = system.semantic_map.get_units_in_spaces(["root_knowledge_entity"])

   results = system.semantic_map.search_by_text_with_rerank("张三去了哪里？", top_k=5)

.. _dsref-semantic-graph:

语义关系图 (SemanticGraph)
--------------------------

``SemanticGraphService``（位于 ``mandol.application.semantic_graph``）是管理记忆单元之间图关系结构的核心服务。
它由显式关系（实体关系、事件因果、溯源、共指等）和隐式语义关系（向量相似度）两部分组成，
图存储委托给 ``GraphStore`` 实现（默认为 ``InMemoryGraphStore``）。

.. note::

   **名称说明**：代码中类名为 ``SemanticGraphService``，通过 ``system.graph`` 属性访问。
   文档中统一简称为 SemanticGraph。

构造函数
^^^^^^^^

.. code-block:: python

   SemanticGraphService(
       *,
       semantic_map: SemanticMapService,
       graph_store: GraphStore,
   )

核心接口
^^^^^^^^

**单元管理**：

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - 方法
     - 说明
   * - ``add_unit(unit, *, space_names, ensure_embedding=True, rebuild_index_immediately=False)``
     - 添加单元（委托给 SemanticMap）
   * - ``delete_unit(uid)``
     - 删除单元及其所有关联边

**关系管理**：

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - 方法
     - 说明
   * - ``add_relationship(source_uid, target_uid, relationship_name, **properties)``
     - 添加关系边
   * - ``get_relationship(source_uid, target_uid, relationship_name) -> Optional[Dict]``
     - 查询关系边属性
   * - ``delete_relationship(source_uid, target_uid, relationship_name=None)``
     - 删除关系边

**图检索**：

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - 方法
     - 说明
   * - ``get_explicit_neighbors(uids, *, rel_type=None, direction="out") -> List[MemoryUnit]``
     - 获取显式关系邻居
   * - ``get_implicit_neighbors(uids, *, top_k=10) -> List[Tuple[MemoryUnit, float]]``
     - 获取隐式语义邻居
   * - ``get_units_in_spaces(space_names, *, mode="union", recursive=True) -> List[MemoryUnit]``
     - 获取指定空间内单元
   * - ``bfs_expand_units(seeds, *, per_seed=3, hops=1, rel_type=None) -> List[MemoryUnit]``
     - BFS 图扩展

**维护**：

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - 方法
     - 说明
   * - ``flush()``
     - 清空图数据
   * - ``get_graph_store() -> GraphStore``
     - 获取底层图存储

图结构组成
^^^^^^^^^^

SemanticGraph 由两部分组成：

**显式关系**：

.. list-table::
   :header-rows: 1
   :widths: 18 40 42

   * - 边类型
     - 说明
     - 子类型
   * - ``RELATED_TO``
     - 实体关系边
     - hometown、lives_in、works_at、located_in、part_of
   * - ``CAUSES``
     - 事件因果关系（A 导致 B）
     -
   * - ``CAUSED_BY``
     - 事件因果关系（B 被 A 导致）
     -
   * - ``INVOLVES``
     - 事件-实体边
     - participant、location、organizer、victim
   * - ``EVIDENCED_BY``
     - 溯源边（指向原始对话单元）
     -
   * - ``COREF``
     - 共指边（跨会话合并相同指代实体/事件）
     -
   * - ``ALIAS_OF``
     - 别名边（实体别名关系）
     -
   * - ``PRECEDES``
     - 时序边（对话/事件的时间先后顺序）
     -
   * - ``FOLLOWS``
     - 时序边（对话/事件的时间先后顺序）
     -

**隐式语义关系**：

- 基于向量相似度构建的 ``SEMANTIC_SIMILAR`` 边
- 在 ``add()`` 时通过 ``_build_immediate_similarity_edges()`` 增量构建
- 边的权重为向量余弦相似度

使用示例
^^^^^^^^

.. code-block:: python

   from mandol import MemorySystem
   from mandol.domain.types import Uid

   system = MemorySystem()

   system.graph.add_relationship(
       source_uid=Uid("entity_001"),
       target_uid=Uid("entity_002"),
       relationship_name="RELATED_TO",
       subtype="works_at",
   )

   neighbors = system.graph.get_explicit_neighbors(
       [Uid("entity_001")],
       rel_type="RELATED_TO",
       direction="out",
   )

   similar = system.graph.get_implicit_neighbors(
       [Uid("dialogue_001")],
       top_k=5,
   )

   expanded = system.graph.bfs_expand_units(
       seeds=[unit],
       per_seed=3,
       hops=1,
   )

.. _dsref-types:

类型系统
--------

系统使用以下类型别名来增强类型安全，所有类型定义于 ``mandol/domain/types.py``：

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - 类型
     - 定义
     - 说明
   * - ``Uid``
     - NewType
     - 记忆单元唯一标识符
   * - ``SpaceName``
     - NewType
     - 空间名称
   * - ``Embedding``
     - 类型别名
     - 向量表示（通常是 float32 numpy 数组）

``SearchHit`` 定义于 ``mandol/retrieval/types.py``：

.. code-block:: python

   @dataclass
   class SearchHit:
       unit: MemoryUnit
       final_score: float
       scores: Dict[str, float]
       ranks: Dict[str, int]
