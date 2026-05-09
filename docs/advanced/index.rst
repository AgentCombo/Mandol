高级指南
========

本节介绍 Mandol 的高级用法，包括自定义组件、系统扩展、多用户系统、性能优化和生产环境部署。

自定义 Embedding Provider
-------------------------

如果需要使用自定义的 Embedding 模型，可以实现 ``EmbeddingProvider`` 接口：

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

   system = MemorySystem.from_yaml_config("config.yaml", embedder=MyEmbedder())

自定义 Reranker
---------------

如果需要自定义重排序模型，可以实现 ``Reranker`` 接口：

.. code-block:: python

   from mandol.ports.reranker import Reranker
   from typing import List, Tuple

   class MyReranker(Reranker):
       def rerank(self, query: str, documents: List[str], top_k: int = 10) -> List[Tuple[int, float]]:
           scored = [(i, 1.0 / (i + 1)) for i in range(len(documents))]
           return scored[:top_k]

   system = MemorySystem.from_yaml_config("config.yaml", reranker=MyReranker())

自定义 LLM Provider
-------------------

如果需要使用自定义的 LLM 服务，可以实现 ``LLMProvider`` 接口：

.. code-block:: python

   from mandol.ports.llm_provider import LLMProvider, ChatMessage
   from typing import List, Optional

   class MyLLMProvider(LLMProvider):
       def chat_completion(
           self,
           messages: List[ChatMessage],
           max_tokens: Optional[int] = None,
           temperature: Optional[float] = None,
           **kwargs
       ) -> str:
           """完成对话生成"""
           pass

   system = MemorySystem.from_yaml_config("config.yaml", llm_provider=MyLLMProvider())

自定义 Graph Store
------------------

如果需要将图存储从内存切换到其他图数据库，可以实现 ``GraphStore`` 接口。

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
           # 访问当前会话的会话索引
           session_idx = ctx.session_idx

           # 获取命名策略
           naming_policy = ctx.naming_policy

           # 获取空间索引
           space_index = ctx.space_index

           # 获取单元存储
           unit_store = ctx.unit_store

           # 获取图存储
           graph_store = ctx.graph_store

           # 获取嵌入提供者
           embedder = ctx.embedder

           # 获取 LLM 提供者
           llm = ctx.llm

           # 实现你的维度构建逻辑
           pass

注册维度构建器
^^^^^^^^^^^^^^

.. code-block:: python

   from mandol.application.multidim_semantic_graph import MultiDimSemanticGraph

   # 创建构建器并注册自定义维度
   builder = MultiDimSemanticGraph(
       graph_service=system.graph,
       space_naming_policy=space_naming_policy,
       dimension_builders=[
           LayoutNormalizationDimension(),
           SemanticSimilarityDimension(),
           HighLevelSummaryApplicatorDimension(),
           EventCausalApplicatorDimension(),
           EntityRelationApplicatorDimension(),
           MyCustomDimension(),  # 添加自定义维度
       ],
   )

扩展提示
--------

- **测试优先**：所有自定义组件都应该有对应的单元测试
- **接口兼容**：确保实现所有接口方法，避免运行时错误
- **性能考量**：对于大规模数据，考虑使用批量操作和异步处理
- **日志记录**：在关键操作中添加日志，便于调试和监控

多用户系统
----------

在多用户场景中，需要为不同用户维护独立的记忆空间：

.. code-block:: python

   from mandol import MemorySystem, MemoryUnit, Uid

   system = MemorySystem()

   for user_id, conversations in user_data.items():
       for conv in conversations:
           unit = MemoryUnit(
               uid=Uid(f"{user_id}_{conv['id']}"),
               raw_data={"text_content": conv["text"]},
               metadata={
                   "timestamp": conv["timestamp"],
                   "user_id": user_id,
               },
           )
           system.add(unit)

   system.build_high_level(mode="auto")

   hits = system.holistic_retrieve("查询内容", top_k=10)
   user_hits = [h for h in hits if h.unit.metadata.get("user_id") == "user_001"]

**关键点**：

- 在 ``metadata`` 中标记 ``user_id``
- 检索后通过 ``metadata`` 过滤结果
- 如需完全隔离，可为每个用户创建独立的 ``MemorySystem`` 实例

性能优化
--------

检索延迟优化
^^^^^^^^^^^^

1. **使用 FAISS 索引**：``pip install mandol[faiss]``，当记忆数量超过 ``promote_threshold`` 时自动启用
2. **减小 BFS 扩展**：设置 ``bfs_expansion_hops: 0`` 关闭图扩展
3. **关闭重排序**：``holistic_retrieve(query, use_rerank=False)``
4. **使用 GPU**：``MANDOL_EMBEDDER_DEVICE=cuda`` 加速向量化

内存占用优化
^^^^^^^^^^^^

1. **使用远程 API**：``USE_REMOTE_EMBEDDER=true`` 避免本地模型加载
2. **启用持久化**：定期 ``save``/``load`` 释放内存
3. **减小窗口**：降低 ``similarity_recent_window``

LLM 成本优化
^^^^^^^^^^^^

1. **增大检测间隔**：``session_check_interval: 50`` 减少 LLM 调用
2. **使用便宜模型**：``MANDOL_LLM_MODEL=gpt-4o-mini``
3. **增量构建**：始终使用 ``build_high_level(mode="auto")`` 避免重复处理

生产环境部署
------------

持久化策略
^^^^^^^^^^

Mandol 是基于内存的轻量级记忆系统，通过 ``save``/``load`` 实现数据持久化：

.. code-block:: python

   system = MemorySystem.from_yaml_config("config.yaml")

   system.add(unit)
   system.build_high_level(mode="auto")

   system.save("./data/memory")

   system2 = MemorySystem.load("./data/memory")

监控与日志
^^^^^^^^^^

Mandol 使用 Python 标准 logging 模块，可通过配置日志级别查看运行信息：

.. code-block:: python

   import logging
   logging.basicConfig(level=logging.INFO)
