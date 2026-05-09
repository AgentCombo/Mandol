检索模块公开接口
========================

以下为可扩展的检索器框架接口。

检索模块公开接口
----------------

以下接口位于 ``retrieval/`` 模块，面向高级用户，可用于构建自定义检索策略。

HybridRetriever
^^^^^^^^^^^^^^^

混合检索器，实现 Dense + BM25 + Sparse 三路召回 → RRF 融合 → BFS 扩展 → 重排序。

**使用示例**：

.. code-block:: python

   from mandol.retrieval.pipeline import HybridRetriever, HybridRetrieverConfig

   hybrid = HybridRetriever(
       graph=system.graph,
       config=HybridRetrieverConfig(
           per_method_k=60,
           bfs_per_seed=3,
           bfs_hops=1,
           parallel_search=True,
       ),
   )

   hits = hybrid.search_hybrid("query", top_k=10, ms_names=["root_base_memory"])

BM25Retriever
^^^^^^^^^^^^^

BM25 关键词检索器，适用于精确关键词匹配场景。

**使用示例**：

.. code-block:: python

   from mandol.retrieval.bm25 import BM25Retriever

   bm25 = BM25Retriever()
   bm25.index_units(units)
   results = bm25.search("query", top_k=10)

SparseRetriever
^^^^^^^^^^^^^^^

稀疏向量检索器，使用 SPLADE 等稀疏向量表示进行检索。

SubgraphHopRetriever
^^^^^^^^^^^^^^^^^^^^

子图跳检索器，适用于跨会话/多跳问答场景，支持基于图关系的跳转检索。

后续扩展
^^^^^^^^

检索模块后续将添加更多接口：

- 指定检索器检索
- 自定义检索器组合
- 其他高级检索策略

.. _retrieval-dataset-base-class:

数据集适配基类（预想设计，后续实现）
------------------------------------

.. warning:: 📋 预想接口 — 此接口尚未实现，以下文档描述目标设计，API 可能变更。


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
   - **系统内部接口**：下划线前缀 ``_`` （如 ``_bfs_expand_units``、``_rrf_fusion``、``_triple_retrieval_rrf``）
   - **预想接口**：无下划线前缀，但标注"（预想接口，后续实现）"

   所有不暴露给最终用户的系统内部方法都应使用下划线前缀，
