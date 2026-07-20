扩展指南
========

.. note::

   本文档内容已迁移至 :doc:`/developer/extending/index`。本页面将在后续版本中移除，请更新你的书签。

本节介绍如何扩展记忆系统，包括自定义组件和添加新功能。

自定义 Embedding Provider
-------------------------

如果需要使用自定义的 Embedding 模型，可以实现 ``EmbeddingProvider`` 接口。

接口定义
^^^^^^^^

.. code-block:: python

   from mandol.ports.embedding_provider import EmbeddingProvider
   import numpy as np
   from typing import List

   class MyEmbedder(EmbeddingProvider):
       def embedding_dim(self) -> int:
           """返回向量维度"""
           return 768

       def embed_text(self, texts: List[str]) -> List[np.ndarray]:
           """将文本列表转换为向量列表"""
           # 实现你的 embedding 逻辑
           pass

       def embed_image_paths(self, paths: List[str]) -> List[np.ndarray]:
           """将图片路径列表转换为向量列表（可选）"""
           # 如果不支持图片，可以抛出 NotImplementedError
           raise NotImplementedError("Image embedding not supported")

使用自定义 Provider
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from mandol import MemorySystem

   # 方式一：构造时直接传入自定义 embedder
   system = MemorySystem(embedder=MyEmbedder())

   # 方式二：通过 YAML 配置 + 自定义 embedder 覆盖
   system = MemorySystem.from_yaml_config("config.yaml", embedder=MyEmbedder())

   # 方式三：运行时动态替换
   system.semantic_map.set_embedder(MyEmbedder())

自定义 Graph Store
------------------

如果需要将图存储从内存切换到 Neo4j 或其他图数据库，可以实现 ``GraphStore`` 接口。

接口定义
^^^^^^^^

.. code-block:: python

   from mandol.ports.graph_store import GraphStore
   from mandol.domain.types import Uid
   from typing import List, Dict, Any, Optional

   class MyGraphStore(GraphStore):
       def get_neighbors(
           self,
           uid: Uid,
           rel_type: Optional[str] = None,
           direction: str = "both"
       ) -> List[Uid]:
           """获取邻居节点"""
           pass

       def upsert_relationship(
           self,
           source: Uid,
           target: Uid,
           rel_type: str,
           props: Optional[Dict[str, Any]] = None
       ) -> None:
           """添加或更新关系"""
           pass

       def delete_relationship(
           self,
           source: Uid,
           target: Uid,
           rel_type: str
       ) -> None:
           """删除关系"""
           pass

       def get_relationships(
           self,
           uid: Uid,
           rel_type: Optional[str] = None
       ) -> List[Dict[str, Any]]:
           """获取关系列表"""
           pass

       def get_all_relationships(self) -> List[Dict[str, Any]]:
           """获取所有关系"""
           pass

.. note::

   当前默认使用 ``InMemoryGraphStore``。切换到外部图数据库（如 Neo4j）时，
   需将自定义 ``GraphStore`` 传入 ``SemanticGraphService`` 构造函数。

添加新的维度构建器
------------------

如果需要添加新的记忆维度，可以实现 ``DimensionBuilder`` 接口并注册到构建流程。

接口定义
^^^^^^^^

.. code-block:: python

   from mandol.application.multidim_semantic_graph import (
       DimensionBuilder,
       MultiDimBuildContext,
   )

   class MyCustomDimension(DimensionBuilder):
       name = "my_custom_dim"

       def build(self, ctx: MultiDimBuildContext) -> None:
           """维度构建逻辑"""
           # 访问当前会话的上下文信息
           session_idx = ctx.config.get("session_idx")

           # 获取命名策略
           naming = ctx.naming

           # 获取图服务
           graph = ctx.graph

           # 实现你的维度构建逻辑
           pass

注册维度构建器
^^^^^^^^^^^^^^

.. code-block:: python

   from mandol.application.multidim_semantic_graph import MultiDimSemanticGraphBuilder

   # 创建构建器并注册自定义维度
   builder = MultiDimSemanticGraphBuilder(
       graph=system.graph,
   )
   # 自定义维度的注册需修改 MemorySystem 内部的 builder 初始化逻辑，
   # 或在 build_high_level 之前注入自定义逻辑

.. note::

   当前 ``MultiDimSemanticGraphBuilder`` 的维度构建器列表在 ``MemorySystem.__init__`` 中初始化。
   如需添加自定义维度，建议通过继承或工厂模式扩展 ``MemorySystem``。

自定义 Reranker
---------------

如果需要自定义重排序模型，可以实现 ``Reranker`` 接口。

接口定义
^^^^^^^^

.. code-block:: python

   from mandol.ports.reranker import Reranker
   from mandol.domain.memory_unit import MemoryUnit
   from typing import List, Tuple

   class MyReranker(Reranker):
       def rerank(
           self,
           query: str,
           units: List[MemoryUnit],
           top_k: int,
       ) -> List[Tuple[MemoryUnit, float]]:
           """对检索结果进行重排序

           Args:
               query: 查询文本
               units: 候选记忆单元列表
               top_k: 返回结果数量

           Returns:
               (MemoryUnit, score) 的排序列表
           """
           # 实现你的重排序逻辑
           pass

使用自定义 Reranker
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from mandol import MemorySystem

   # 构造时传入
   system = MemorySystem(reranker=MyReranker())

   # 运行时动态替换
   system.semantic_map.set_reranker(MyReranker())

自定义 LLM Provider
-------------------

如果需要使用自定义的 LLM 服务，可以实现 ``LLMProvider`` 接口。

接口定义
^^^^^^^^

.. code-block:: python

   from mandol.ports.llm_provider import LLMProvider, ChatMessage, ChatResponse
   from typing import List, Optional

   class MyLLMProvider(LLMProvider):
       def chat(
           self,
           messages: List[ChatMessage],
           temperature: float = 0.1,
           max_tokens: int = 1024,
           **kwargs
       ) -> ChatResponse:
           """对话生成

           Args:
               messages: 对话消息列表，每项为 {"role": str, "content": str}
               temperature: 温度参数（0-2）
               max_tokens: 最大生成 token 数

           Returns:
               ChatResponse: 包含 content 字段的响应对象
           """
           pass

.. note::

   ``ChatResponse`` 是一个包含 ``content`` 字段的响应对象。
   当前默认使用 ``OpenAICompatibleLLMProvider``，兼容 OpenAI API 格式。
   自定义实现中，``chat`` 方法返回的 ``ChatResponse`` 需包含 ``content: str`` 字段。

扩展提示
--------

- **测试优先**：所有自定义组件都应该有对应的单元测试
- **接口兼容**：确保实现所有接口方法，避免运行时错误
- **性能考量**：对于大规模数据，考虑使用批量操作和异步处理
- **日志记录**：在关键操作中添加日志，便于调试和监控
