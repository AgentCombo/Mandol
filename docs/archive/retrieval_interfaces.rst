检索接口分类
============

.. note::

   本文档内容已迁移至 :doc:`/shared/retrieval-reference`。本页面将在后续版本中移除，请更新你的书签。

本节介绍 Mandol 记忆系统的检索接口分类，从底层数据结构到多视角记忆，再到全记忆统一检索，按层级递进说明。

接口命名约定
------------

- **公开接口**：无前缀，面向用户直接调用（如 ``get_unit``、``holistic_retrieve``）
- **内部接口**：系统内部调用的检索管线组件（如 ``rrf_fusion``、``HybridRetriever``），不暴露给基础用户

.. _retrieval-memory-unit:

适用于 MemoryUnit 的操作
-------------------------

.. note::
   **✅ 已实现** 以下接口均由 ``SemanticMapService`` 提供，通过 ``system.semantic_map`` 访问。


get_unit
^^^^^^^^

获取单个记忆单元。

**签名**：

.. code-block:: python

   def get_unit(uid: Uid) -> Optional[MemoryUnit]

**使用示例**：

.. code-block:: python

   from mandol.domain.types import Uid
   unit = system.semantic_map.get_unit(Uid("dialogue_001"))

list_units
^^^^^^^^^^

列出所有记忆单元。

**签名**：

.. code-block:: python

   def list_units() -> List[MemoryUnit]

**使用示例**：

.. code-block:: python

   all_units = system.semantic_map.list_units()

filter_memory_units
^^^^^^^^^^^^^^^^^^^

.. warning::
   **📋 预想接口** — 此方法尚未实现，接口签名仅供设计参考。

按条件过滤记忆单元，支持嵌套字段查询。当前可用的替代方案是 ``get_units_in_spaces()`` 结合 Python 列表推导进行筛选。

**预想签名**：

.. code-block:: python

   def filter_memory_units(
       candidate_units: Optional[List[MemoryUnit]] = None,
       filter_condition: Optional[dict] = None,
       ms_names: Optional[List[str]] = None,
       recursive: bool = True
   ) -> List[MemoryUnit]

**当前替代方案**：

.. code-block:: python

   # 获取特定空间的单元
   units = system.semantic_map.get_units_in_spaces(
       ["root_knowledge_entity"]
   )

   # 按元数据条件过滤（使用 Python 列表推导）
   entities = system.semantic_map.get_units_in_spaces(
       ["root_knowledge_entity"]
   )
   persons = [
       u for u in entities
       if u.metadata.get("entity_type") == "Person"
   ]

delete_unit
^^^^^^^^^^^

删除记忆单元及其所有关联。

**签名**：

.. code-block:: python

   def delete_unit(uid: Uid) -> None

**使用示例**：

.. code-block:: python

   system.semantic_map.delete_unit(Uid("dialogue_001"))

.. _retrieval-semantic-map:

适用于 SemanticMap 的接口
-------------------------

以下接口由 ``SemanticMapService``（通过 ``system.semantic_map`` 访问）提供，面向向量空间的语义检索。

.. note::
   **✅ 已实现** 以下为核心语义检索方法。

search_by_text
^^^^^^^^^^^^^^

基于文本的语义检索。

**签名**：

.. code-block:: python

   def search_by_text(
       query_text: str,
       *,
       top_k: int = 10,
       space_names: Optional[List[str]] = None,
       recursive: bool = True,
   ) -> List[Tuple[MemoryUnit, float]]

**参数**：

- ``query_text``：查询文本
- ``top_k``：返回结果数量（默认 10）
- ``space_names``：限定检索的空间名称列表（None 表示全部空间）
- ``recursive``：是否递归搜索子空间（默认 True）

**使用示例**：

.. code-block:: python

   results = system.semantic_map.search_by_text("张三去了哪里？", top_k=10)

search_by_vector
^^^^^^^^^^^^^^^^

基于向量的语义检索。

**签名**：

.. code-block:: python

   def search_by_vector(
       query: np.ndarray,
       *,
       top_k: int = 10,
       space_names: Optional[List[str]] = None,
       recursive: bool = True,
   ) -> List[Tuple[MemoryUnit, float]]

**使用示例**：

.. code-block:: python

   query_embedding = embedder.encode(["query"])[0]
   results = system.semantic_map.search_by_vector(query_embedding, top_k=10)

search_by_text_with_rerank
^^^^^^^^^^^^^^^^^^^^^^^^^^

带 Cross-Encoder 重排序的文本语义检索。先通过向量检索召回更多候选（``recall_k``），
再用重排序模型精排，提升检索精度。

**签名**：

.. code-block:: python

   def search_by_text_with_rerank(
       query_text: str,
       *,
       top_k: int = 10,
       recall_k: Optional[int] = None,
       space_names: Optional[List[str]] = None,
       recursive: bool = True,
       use_rerank: bool = True,
   ) -> List[Tuple[MemoryUnit, float]]

**使用示例**：

.. code-block:: python

   # 带重排序的语义检索
   results = system.semantic_map.search_by_text_with_rerank(
       "张三在北京做了什么？", top_k=5
   )

   # 关闭重排序（等同于 search_by_text）
   results = system.semantic_map.search_by_text_with_rerank(
       "query", top_k=10, use_rerank=False
   )

search_in_space
^^^^^^^^^^^^^^^

在指定空间内检索，支持传入候选单元列表缩小搜索范围。

**签名**：

.. code-block:: python

   def search_in_space(
       query_text: str,
       space_name: str,
       candidates: Optional[List[MemoryUnit]] = None,
       *,
       top_k: int = 10,
       recall_k: Optional[int] = None,
   ) -> List[Tuple[MemoryUnit, float]]

**使用示例**：

.. code-block:: python

   # 在实体空间内检索
   results = system.semantic_map.search_in_space(
       "张三", "root_knowledge_entity", top_k=5
   )

unified_search
^^^^^^^^^^^^^^

.. warning::
   **📋 预想接口** — 统一的 ``search()`` 方法尚未实现。

计划中的统一检索接口，支持通过 ``retriever_type`` 参数选择不同检索器后端
（dense / bm25 / sparse），或通过 ``retrievers`` 参数组合多路召回并自动 RRF 融合。

当前等效功能可通过 ``HybridRetriever``（参见 :ref:`retrieval-pipeline`）实现：

.. code-block:: python

   from mandol.retrieval.pipeline import HybridRetriever
   retriever = HybridRetriever(graph=system.graph, ...)
   results = retriever.search(query_text, top_k=10)

.. _retrieval-semantic-graph:

适用于 SemanticGraph 的接口
---------------------------

以下接口由 ``SemanticGraphService``（通过 ``system.graph`` 访问）提供，用于基于图关系的检索。

.. note::
   **✅ 已实现** 以下为核心图检索与操作方法。

get_explicit_neighbors
^^^^^^^^^^^^^^^^^^^^^^

获取指定单元的显式关系邻居（如实体关系、事件因果等）。

**签名**：

.. code-block:: python

   def get_explicit_neighbors(
       uids: List[Uid],
       *,
       rel_type: Optional[str] = None,
       direction: str = "out",
   ) -> List[MemoryUnit]

**参数**：

- ``uids``：源单元 UID 列表
- ``rel_type``：关系类型过滤（如 ``"CAUSES"``、``"RELATED_TO"``）
- ``direction``：检索方向，可选 ``"out"``（出边）、``"in"``（入边）、``"both"``（双向）

**使用示例**：

.. code-block:: python

   from mandol.domain.types import Uid

   # 获取因果关系的出边邻居
   causes = system.graph.get_explicit_neighbors(
       [Uid(event_uid)],
       rel_type="CAUSES",
       direction="out",
   )

   # 获取所有入边邻居
   inbound = system.graph.get_explicit_neighbors(
       [Uid(entity_uid)],
       direction="in",
   )

get_implicit_neighbors
^^^^^^^^^^^^^^^^^^^^^^

获取指定单元的隐式语义邻居（基于向量相似度边的 BFS 扩展）。

**签名**：

.. code-block:: python

   def get_implicit_neighbors(
       uids: List[Uid],
       *,
       top_k: int = 10,
   ) -> List[Tuple[MemoryUnit, float]]

**使用示例**：

.. code-block:: python

   neighbors = system.graph.get_implicit_neighbors(
       [Uid(dialogue_uid)], top_k=5
   )
   for unit, score in neighbors:
       print(f"{unit.uid}: similarity={score:.3f}")

bfs_expand_units
^^^^^^^^^^^^^^^^

从种子单元出发进行 BFS 图扩展，沿关系边跳转收集关联单元。

**签名**：

.. code-block:: python

   def bfs_expand_units(
       seeds: List[MemoryUnit],
       *,
       per_seed: int = 3,
       hops: int = 1,
       rel_type: Optional[str] = None,
   ) -> List[MemoryUnit]

**使用示例**：

.. code-block:: python

   expanded = system.graph.bfs_expand_units(
       seeds=top_results,
       per_seed=3,
       hops=1,
   )

add_relationship
^^^^^^^^^^^^^^^^

手动添加显式关系边。

**签名**：

.. code-block:: python

   def add_relationship(
       source_uid: Uid,
       target_uid: Uid,
       relationship_name: str,
       **properties,
   ) -> None

**使用示例**：

.. code-block:: python

   system.graph.add_relationship(
       source_uid=Uid("entity_001"),
       target_uid=Uid("entity_002"),
       relationship_name="RELATED_TO",
       subtype="works_at",
   )

delete_relationship
^^^^^^^^^^^^^^^^^^^

删除关系边。

**签名**：

.. code-block:: python

   def delete_relationship(
       source_uid: Uid,
       target_uid: Uid,
       relationship_name: Optional[str] = None,
   ) -> None

get_edges_of_unit
^^^^^^^^^^^^^^^^^

.. warning::
   **📋 预想接口** — 获取指定节点所有关系边的便捷方法尚未实现。

当前替代方案：

.. code-block:: python

   # 获取所有显式邻居（不限定类型和方向）
   neighbors = system.graph.get_explicit_neighbors(
       [Uid("entity_001")], direction="both"
   )

   # 查询特定关系
   rel = system.graph.get_relationship(
       Uid("entity_001"), Uid("entity_002"), "RELATED_TO"
   )

search_graph_relations
^^^^^^^^^^^^^^^^^^^^^^

.. warning::
   **📋 预想接口** — 图关系搜索方法尚未实现。

当前可通过 ``bfs_expand_units()`` + ``get_explicit_neighbors()`` 组合实现等效功能：

.. code-block:: python

   # 从种子节点 BFS 展开获取关联单元
   expanded = system.graph.bfs_expand_units(
       seeds=seed_units, per_seed=5, hops=2
   )

get_node_neighbors
^^^^^^^^^^^^^^^^^^

.. warning::
   **📋 预想接口** — 综合邻居查询方法尚未实现。

当前可通过以下组合实现结构邻居 + 语义邻居：

.. code-block:: python

   # 结构邻居
   structural = system.graph.get_explicit_neighbors(
       [Uid(node_uid)], direction="both"
   )

   # 语义邻居
   semantic = system.graph.get_implicit_neighbors(
       [Uid(node_uid)], top_k=5
   )

delete_unit
^^^^^^^^^^^

从图中删除单元及其所有关联边。

**签名**：

.. code-block:: python

   def delete_unit(uid: Uid) -> None

.. _retrieval-multi-view:

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

.. warning::
   **📋 预想接口** — ``get_base_units()`` / ``get_implicit_neighbors_base()`` 等快捷方法尚未实现。

当前可通过 ``SemanticMapService`` 和 ``SemanticGraphService`` 的通用接口访问：

.. code-block:: python

   from mandol.domain.types import Uid

   # 获取基础记忆空间的所有单元
   base_units = system.semantic_map.get_units_in_spaces(["root_base_memory"])

   # 获取语义相似的邻居
   similar = system.graph.get_implicit_neighbors(
       [Uid(dialogue_uid)], top_k=5
   )

实体关系 (Entity Relation)
^^^^^^^^^^^^^^^^^^^^^^^^^^

knowledge_entity 空间存储实体单元，由实体关系边连接。

.. warning::
   **📋 预想接口** — ``get_entities_by_type()`` / ``get_entity_relations()`` 等快捷方法尚未实现。

当前访问方式：

.. code-block:: python

   from mandol.domain.types import Uid

   # 获取所有实体
   entities = system.semantic_map.get_units_in_spaces(
       ["root_knowledge_entity"]
   )

   # 按类型过滤（Python 列表推导）
   persons = [u for u in entities if u.metadata.get("entity_type") == "Person"]

   # 获取实体的显式关系邻居
   relations = system.graph.get_explicit_neighbors(
       [Uid("entity_001")],
       rel_type="RELATED_TO",
       direction="out",
   )

事件因果 (Event Causal)
^^^^^^^^^^^^^^^^^^^^^^^

episodic_event 空间存储事件单元，由因果边连接。

.. warning::
   **📋 预想接口** — ``get_event_causal_chain()`` 快捷方法尚未实现。

当前通过 BFS 图扩展获取因果链：

.. code-block:: python

   # 获取所有事件单元
   events = system.semantic_map.get_units_in_spaces(["root_episodic_event"])

   # 沿因果边追溯
   causal_chain = system.graph.bfs_expand_units(
       seeds=events[:3],
       per_seed=3,
       hops=2,
       rel_type="CAUSES",
   )

情感 / 情景 / 知识 / 程序 / 洞见总结
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

各维度总结（emotional、episodic_summary、knowledge_summary、procedural、insights）存储在对应的 MemorySpace 中。

.. warning::
   **📋 预想接口** — ``get_emotional_summary()`` / ``get_episodic_summary()`` / ``get_knowledge_summary()`` / ``get_procedural_summary()`` / ``get_insights()`` 等按维度或会话过滤的快捷方法尚未实现。

**推荐方式**：使用 ``retrieve_by_view()`` 进行按视角检索，或使用 ``holistic_retrieve()`` 自动覆盖所有维度：

.. code-block:: python

   # 按视角检索
   episodic_hits = system.retrieve_by_view("北京之行", view="episodic", top_k=5)
   emotional_hits = system.retrieve_by_view("用户心情", view="emotional", top_k=5)
   knowledge_hits = system.retrieve_by_view("专业知识", view="knowledge", top_k=5)

   # 全记忆检索（自动覆盖全部维度）
   all_hits = system.holistic_retrieve("张三去了哪里？", top_k=10)

**直接访问空间**（高级用户）：

.. code-block:: python

   summaries = system.semantic_map.get_units_in_spaces(
       ["root_episodic_summary"], recursive=True
   )

高阶记忆高级接口（预想接口）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

以下接口针对事件因果和实体关系等高阶记忆结构设计，
提供能够充分利用记忆图结构模式的高级分析能力。

.. note::
   **设计目标**：这些高级接口不仅仅是简单的检索，而是基于记忆系统的**图结构特性**
   提供深层次的分析能力，帮助 Agent 更好地理解和利用记忆。

.. warning::
   以下全部为 **📋 预想接口**，尚未实现。

**trace_provenance**（高阶记忆溯源分析）
  追溯任意记忆单元的完整证据链路，沿 ``EVIDENCED_BY`` 边递归回溯。

**compare_multi_view_consistency**（多视角一致性对比）
  对比不同视角记忆之间的一致性和互补性。

**analyze_entity_lifecycle**（实体生命周期分析）
  追踪实体在对话历史中的出现、关系演变和行为模式。

**extract_event_narrative_chain**（事件叙事链提取）
  将因果链转化为自然语言叙事，生成结构化的事件发展故事线。

.. _retrieval-internal:

检索管线内部接口
------------------

以下接口是系统内部调用的检索管线组件，供开发者了解检索流程的实现细节。

.. note::
   以下接口仅供**开发者参考**，基础用户和高级用户无需关心。

rrf_fusion
^^^^^^^^^^

倒数排名融合（Reciprocal Rank Fusion），合并多路检索结果。
位于 ``mandol.retrieval.fusion`` 模块中。

**签名**：

.. code-block:: python

   def rrf_fusion(
       result_lists: List[List[Tuple[MemoryUnit, float]]],
       k: int = 60,
       top_k: Optional[int] = None,
   ) -> List[Tuple[MemoryUnit, float]]

HybridRetriever
^^^^^^^^^^^^^^^

混合检索器，整合 Dense + BM25 + Sparse 三路召回 → RRF 融合 → BFS 扩展 → 重排序。
位于 ``mandol.retrieval.pipeline`` 模块中。

**签名**：

.. code-block:: python

   class HybridRetriever:
       def __init__(
           self,
           *,
           graph: SemanticGraphService,
           bm25_index: Optional[Bm25Retriever] = None,
           sparse_index: Optional[TfidfSparseRetriever] = None,
           reranker: Optional[Reranker] = None,
           config: Optional[RetrievalConfig] = None,
           text_extractor: Optional[Callable] = None,
       )
       def search(self, query: str, top_k: int = 10) -> List[Tuple[MemoryUnit, float]]

Bm25Retriever
^^^^^^^^^^^^^

BM25 关键词检索器。位于 ``mandol.retrieval.bm25`` 模块中。

**签名**：

.. code-block:: python

   class Bm25Retriever:
       def search(self, query: str, top_k: int) -> List[Tuple[MemoryUnit, float]]

TfidfSparseRetriever
^^^^^^^^^^^^^^^^^^^^

TF-IDF 稀疏向量检索器。位于 ``mandol.retrieval.sparse`` 模块中。

**签名**：

.. code-block:: python

   class TfidfSparseRetriever:
       def search(self, query: str, top_k: int) -> List[Tuple[MemoryUnit, float]]

SubgraphHopRetriever
^^^^^^^^^^^^^^^^^^^^

子图跳转检索器，基于图结构进行子图扩展检索。位于 ``mandol.retrieval.subgraph_hop`` 模块中。

**签名**：

.. code-block:: python

   class SubgraphHopRetriever:
       def search(self, seeds: List[MemoryUnit], top_k: int) -> List[Tuple[MemoryUnit, float]]

.. _retrieval-holistic:

全记忆统一检索接口
------------------

以下接口由 ``MemorySystem`` 提供，是系统层级的检索功能。

holistic_retrieve
^^^^^^^^^^^^^^^^^

全记忆检索接口，是系统最通用、最强大的检索方法。

**签名**：

.. code-block:: python

   def holistic_retrieve(
       query: str,
       top_k: int = 10,
       use_rerank: bool = True
   ) -> List[SearchHit]

**内部流程**：

1. 获取 4 个检索组：BASE / ENTITY / EVENT / SUMMARY
2. 每组独立执行：
   - Dense + BM25 + Sparse 三路召回
   - RRF 融合
   - BFS 扩展
3. 所有候选合并
4. Cross-Encoder Reranker 全局重排序

**使用示例**：

.. code-block:: python

   # 一站式全记忆检索
   hits = system.holistic_retrieve("张三做了什么？", top_k=10)

   # 仅检索实体
   entity_hits = system.holistic_retrieve("北京", top_k=5)

   # 关闭重排序（更快）
   hits = system.holistic_retrieve("query", top_k=10, use_rerank=False)

检索流程图
^^^^^^^^^^

.. mermaid::

   graph LR
       A[用户查询] --> B[分组召回]
       B --> C[BASE组]
       B --> D[ENTITY组]
       B --> E[EVENT组]
       B --> F[SUMMARY组]
       
       C --> G[Dense检索]
       C --> H[BM25检索]
       C --> I[Sparse检索]
       
       G --> J[RRF融合]
       H --> J
       I --> J
       
       J --> K[BFS扩展]
       K --> L[候选合并]
       D --> L
       E --> L
       F --> L
       
       L --> M[Reranker重排]
       M --> N[最终结果]

retrieve_in_space
^^^^^^^^^^^^^^^^^

在指定空间内执行全记忆检索管线。

**签名**：

.. code-block:: python

   def retrieve_in_space(
       query: str,
       space_name: str,
       top_k: int = 10,
       use_rerank: bool = True
   ) -> List[SearchHit]

**使用示例**：

.. code-block:: python

   # 仅在知识实体空间检索
   hits = system.retrieve_in_space(
       "北京", 
       space_name="root_knowledge_entity",
       top_k=5
   )

retrieve_by_view
^^^^^^^^^^^^^^^^

按多视角类别检索（如 base_memory、entity_relation、event_causal 等）。

**签名**：

.. code-block:: python

   def retrieve_by_view(
       query: str,
       view: str,
       top_k: int = 10,
       use_rerank: bool = True
   ) -> List[SearchHit]

**视角列表**：

- ``base_memory``：基础对话记忆
- ``entity_relation``：实体关系
- ``event_causal``：事件因果
- ``emotional``：情感总结
- ``episodic``：情景总结
- ``knowledge``：知识总结
- ``procedural``：程序总结
- ``insights``：洞见

**使用示例**：

.. code-block:: python

   # 仅检索事件因果视角
   events = system.retrieve_by_view(
       "发生了什么？",
       view="event_causal",
       top_k=5
   )

   # 仅检索情感视角
   emotions = system.retrieve_by_view(
       "用户感受如何？",
       view="emotional",
       top_k=5
   )

smart_quantized_query
^^^^^^^^^^^^^^^^^^^^^

智能量化查询接口（预想接口，后续实现）。

采用三阶段级联量化（智能路由、智能去噪、智能上下文生成），
通过量化的方式逐级筛选和压缩检索结果，
在无 LLM 参与的情况下完成从多源检索到紧凑上下文的全流程，
最终送入 LLM 生成答案。

.. note::

   **设计理念**：

   传统检索方法通常需要 LLM 参与重排序、摘要生成等步骤，成本较高。
   智能量化查询通过纯量化方法实现高效筛选，将 LLM 调用推迟到最终答案生成阶段，
   大幅降低成本和延迟。

**三阶段流程**：

.. mermaid::

   graph LR
       A[用户查询] --> B[阶段1: 智能路由]
       B --> C[阶段2: 智能去噪]
       C --> D[阶段3: 智能上下文生成]
       D --> E[紧凑上下文]
       E --> F[LLM生成答案]

**阶段详情**：

1. **智能路由 (Smart Routing)**
   - 基于查询特征自动判断应该检索哪些记忆空间
   - 使用向量相似度、关键词匹配等量化信号
   - 输出：候选空间列表及初始权重

2. **智能去噪 (Smart Denoising)**
   - 对初步检索结果进行多维度质量评估
   - 基于相关性分数、证据强度、时间新鲜度等指标
   - 过滤低质量和重复结果
   - 输出去噪后的高质量候选集

3. **智能上下文生成 (Smart Context Generation)**
   - 将多个检索结果压缩为紧凑的上下文表示
   - 基于重要性排序、冗余消除、信息密度优化
   - 控制最终上下文的 token 数量
   - 输出：结构化的紧凑上下文，可直接送入 LLM

**签名**：

.. code-block:: python

   def smart_quantized_query(
       query: str,
       max_context_tokens: int = 2000,
       routing_strategy: str = "auto",
       denoise_threshold: float = 0.5,
       compression_ratio: float = 0.3,
       **kwargs
   ) -> QuantizedQueryResult

**参数**：

- ``query``：用户查询文本
- ``max_context_tokens``：最大上下文 token 数（默认 2000）
- ``routing_strategy``：路由策略，可选 ``"auto"``、``"balanced"``、``"comprehensive"``
- ``denoise_threshold``：去噪阈值（0-1，越高越严格）
- ``compression_ratio``：压缩比率（0-1，越小越紧凑）

**返回值类型**：

.. code-block:: python

   @dataclass
   class QuantizedQueryResult:
       context: str                    # 紧凑上下文字符串
       source_uids: List[str]          # 来源单元 UID 列表
       space_distribution: Dict[str, float]  # 各空间占比
       routing_decision: Dict[str, Any]      # 路由决策详情
       denoise_stats: Dict[str, Any]         # 去噪统计
       total_tokens: int                # 实际 token 数

**使用示例**：

.. code-block:: python

   # 标准智能量化查询
   result = system.smart_quantized_query(
       "张三最近在做什么项目？",
       max_context_tokens=2000
   )

   # 获取紧凑上下文送入 LLM
   context = result.context
   answer = llm.generate(f"基于以下上下文回答问题：\n{context}")

   # 高压缩模式（适用于长对话历史）
   result = system.smart_quantized_query(
       query,
       max_context_tokens=1000,
       compression_ratio=0.2
   )

检索模块公开接口
----------------

以下接口位于 ``mandol.retrieval/`` 模块，面向高级用户，可用于构建自定义检索策略。

HybridRetriever
^^^^^^^^^^^^^^^

混合检索器，实现 Dense + BM25 + Sparse 三路召回 → RRF 融合 → BFS 扩展 → 重排序。

**使用示例**：

.. code-block:: python

   from mandol.retrieval.pipeline import HybridRetriever

   hybrid = HybridRetriever(
       graph=system.graph,
       bm25_index=bm25_retriever,
       sparse_index=sparse_retriever,
       reranker=reranker,
   )

   hits = hybrid.search("query", top_k=10)

Bm25Retriever
^^^^^^^^^^^^^

BM25 关键词检索器，适用于精确关键词匹配场景。

**使用示例**：

.. code-block:: python

   from mandol.retrieval.bm25 import Bm25Retriever

   bm25 = Bm25Retriever()
   bm25.index_units(units)
   results = bm25.search("query", 10)

TfidfSparseRetriever
^^^^^^^^^^^^^^^^^^^^

TF-IDF 稀疏向量检索器。

**使用示例**：

.. code-block:: python

   from mandol.retrieval.sparse import TfidfSparseRetriever

   sparse = TfidfSparseRetriever()
   sparse.index_units(units)
   results = sparse.search("query", 10)

SubgraphHopRetriever
^^^^^^^^^^^^^^^^^^^^

子图跳转检索器，适用于跨会话/多跳问答场景。

**使用示例**：

.. code-block:: python

   from mandol.retrieval.subgraph_hop import SubgraphHopRetriever

   subgraph = SubgraphHopRetriever(graph=system.graph)
   results = subgraph.search(seeds, 10)

后续扩展
^^^^^^^^

检索模块后续将添加更多接口：

- 统一检索器路由接口
- 自定义检索器组合
- 其他高级检索策略

.. _retrieval-dataset-base-class:

数据集适配基类（预想设计，后续实现）
------------------------------------

为支持不同数据集类型（对话、代码等）的多视角记忆检索，
系统计划采用**面向对象继承**的设计模式，通过**父类派生**
的方式为不同数据集提供定制化的接口。

设计架构
^^^^^^^^

.. mermaid::

   graph TB
       Base["BaseMultiViewRetriever<br>（抽象基类）"]

       Dialog["DialogueMultiViewRetriever<br>（对话数据集）"]
       Code["CodeMultiViewRetriever<br>（代码数据集）"]
       Future["FutureDatasetRetriever<br>（未来数据集）"]

       Base --> Dialog
       Base --> Code
       Base --> Future

       subgraph DialogInterfaces["对话数据集专属接口"]
           D1[get_episodic_summary]
           D2[get_emotional_summary]
           D3[trace_provenance]
           D4[compare_multi_view_consistency]
       end

       subgraph CodeInterfaces["代码数据集专属接口（预想）"]
           C1[get_architecture_view]
           C2[get_dependency_graph]
           C3[trace_api_usage]
           C4[analyze_code_evolution]
       end

       Dialog --> DialogInterfaces
       Code --> CodeInterfaces

基类定义
^^^^^^^^

.. code-block:: python

   from abc import ABC, abstractmethod
   from typing import List, Optional, Any, Dict

   class BaseMultiViewRetriever(ABC):
       """
       多视角记忆检索器的抽象基类。

       定义所有数据集类型共有的通用接口和属性，
       子类根据具体数据集类型实现特定的多视角维度和检索逻辑。
       """

       @abstractmethod
       def get_dataset_type(self) -> str:
           """返回数据集类型标识"""
           pass

       @abstractmethod
       def get_available_views(self) -> List[str]:
           """返回该数据集支持的多视角列表"""
           pass

       @abstractmethod
       def retrieve_by_view(
           self,
           query: str,
           view: str,
           top_k: int = 10,
           **kwargs
       ) -> List[Any]:
           """按视角检索的通用接口"""
           pass

       # 通用高级接口（所有数据集类型共享）
       def trace_provenance(self, uid: str, **kwargs) -> Any:
           """溯源分析（默认实现，子类可覆盖）"""
           pass

       def get_unit(self, uid: str) -> Optional[Any]:
           """获取单个单元"""
           pass

       def filter_units(
           self,
           filter_condition: Optional[dict] = None,
           **kwargs
       ) -> List[Any]:
           """过滤单元"""
           pass

对话数据集实现示例
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   class DialogueMultiViewRetriever(BaseMultiViewRetriever):
       """
       面向对话数据集的多视角记忆检索器。

       支持的多视角维度：
       - episodic：情景记忆
       - knowledge：知识记忆
       - procedural：程序记忆
       - emotional：情感记忆
       - insights：洞见记忆
       """

       def get_dataset_type(self) -> str:
           return "dialogue"

       def get_available_views(self) -> List[str]:
           return [
               "base_memory",
               "entity_relation",
               "event_causal",
               "episodic",
               "knowledge",
               "procedural",
               "emotional",
               "insights",
           ]

       def get_episodic_summary(
           self,
           session_suffix: str,
           **kwargs
       ) -> Optional[Any]:
           """获取情景总结（对话数据集特有）"""
           pass

       def get_emotional_summary(
           self,
           session_suffix: str,
           **kwargs
       ) -> Optional[Any]:
           """获取情感总结（对话数据集特有）"""
           pass

       def compare_multi_view_consistency(
           self,
           query: str,
           **kwargs
       ) -> Any:
           """多视角一致性对比（对话数据集特有）"""
           pass

代码数据集预想设计
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   class CodeMultiViewRetriever(BaseMultiViewRetriever):
       """
       面向代码数据集的多视角记忆检索器（预想设计）。

       支持的多视角维度（预想）：
       - architecture：代码架构视图
       - dependency：依赖关系视图
       - api_usage：API 使用模式
       - evolution：代码演变历史
       """

       def get_dataset_type(self) -> str:
           return "code"

       def get_available_views(self) -> List[str]:
           return [
               "base_memory",         # 原始代码单元
               "architecture",        # 架构层次结构
               "dependency",          # 模块依赖关系
               "api_usage",           # API 调用模式
               "evolution",           # 代码变更历史
               "insights",            # 代码洞察
           ]

       def get_architecture_view(
           self,
           module_path: Optional[str] = None,
           **kwargs
       ) -> Any:
           """获取代码架构视图（代码数据集特有）"""
           pass

       def get_dependency_graph(
           self,
           module_uid: str,
           depth: int = 2,
           **kwargs
       ) -> Any:
           """获取依赖关系图（代码数据集特有）"""
           pass

       def trace_api_usage(
           self,
           api_name: str,
           **kwargs
       ) -> Any:
           """追踪 API 使用情况（代码数据集特有）"""
           pass

       def analyze_code_evolution(
           self,
           file_path: str,
           time_range: Optional[tuple] = None,
           **kwargs
       ) -> Any:
           """分析代码演变历史（代码数据集特有）"""
           pass

使用示例
^^^^^^^^

.. code-block:: python

   from mandol.retrieval.dataset_base import DialogueMultiViewRetriever

   # 创建对话数据集检索器实例
   dialogue_retriever = DialogueMultiViewRetriever(
       graph=system.graph,
       semantic_map=system.semantic_map
   )

   # 使用对话数据集特有的接口
   summary = dialogue_retriever.get_episodic_summary("msg_0_25")
   report = dialogue_retriever.compare_multi_view_consistency("北京之行")

   # 未来扩展到代码数据集
   # from mandol.retrieval.dataset_base import CodeMultiViewRetriever
   # code_retriever = CodeMultiViewRetriever(graph, semantic_map)
   # deps = code_retriever.get_dependency_graph("module_001")

.. note::

   **接口命名约定补充说明**：

   - **公开用户接口**：无前缀（如 ``get_unit``、``holistic_retrieve``、``search``）
   - **系统内部接口**：位于 ``mandol.retrieval`` 模块中（如 ``rrf_fusion``、``HybridRetriever``、``Bm25Retriever``）

   预想接口通过 📋 标记区分，内部管线组件在 :ref:`retrieval-internal` 中说明。