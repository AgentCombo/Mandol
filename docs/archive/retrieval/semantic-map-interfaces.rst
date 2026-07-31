SemanticMap 检索接口
========================

以下接口由 ``SemanticMap`` 提供，面向向量空间的语义检索。

适用于 SemanticMap 的接口
-------------------------

以下接口由 ``SemanticMap`` 提供，面向向量空间的语义检索。

.. note::

   **统一检索接口设计**：

   SemanticMap 提供统一的 ``search`` 接口，支持多种检索器后端。
   用户可通过 ``retriever_type`` 参数指定使用哪种检索策略，
   也可通过 ``retrievers`` 参数组合多个检索器进行联合检索。

search
^^^^^^

**统一的语义检索接口**，支持多种检索器类型和自定义检索器组合。

**签名**：

.. code-block:: python

   def search(
       query: Union[str, np.ndarray],
       k: int = 5,
       retriever_type: Optional[str] = None,
       retrievers: Optional[List[str]] = None,
       ms_names: Optional[List[str]] = None,
       candidate_uids: Optional[List[str]] = None,
       **kwargs
   ) -> List[Tuple[MemoryUnit, float]]

**参数**：

- ``query``：查询文本或预生成的向量
- ``k``：返回结果数量
- ``retriever_type``：单一检索器类型，可选值：
  - ``"dense"``：稠密向量检索（默认）
  - ``"bm25"``：BM25 关键词检索
  - ``"sparse"``：稀疏向量检索（SPLADE 等）
- ``retrievers``：多检索器组合列表，如 ``["dense", "bm25", "sparse"]``
  - 当指定此参数时，系统将自动执行多路召回 + RRF 融合
- ``ms_names``：限定检索的空间名称列表
- ``candidate_uids``：限定候选单元 UID 列表

**使用示例**：

.. code-block:: python

   # 使用单一检索器（Dense 向量检索）
   results = system.semantic_map.search(
       "query",
       k=10,
       retriever_type="dense"
   )

   # 使用 BM25 关键词检索
   results = system.semantic_map.search(
       "北京故宫",
       k=10,
       retriever_type="bm25"
   )

   # 多检索器组合（自动 RRF 融合）
   results = system.semantic_map.search(
       "query",
       k=10,
       retrievers=["dense", "bm25", "sparse"]
   )

   # 限定空间检索
   results = system.semantic_map.search(
       "query",
       k=10,
       retriever_type="dense",
       ms_names=["root_knowledge_entity"]
   )

search_similarity_by_text
^^^^^^^^^^^^^^^^^^^^^^^^^

基于文本的语义检索便捷方法（内部调用 ``search(retriever_type="dense")``）。

**签名**：

.. code-block:: python

   def search_similarity_by_text(
       query_text: str,
       k: int = 5,
       ms_names: Optional[List[str]] = None,
       candidate_uids: Optional[List[str]] = None
   ) -> List[Tuple[MemoryUnit, float]]

**使用示例**：

.. code-block:: python

   results = system.semantic_map.search_similarity_by_text("query", k=10)

search_similarity_by_vector
^^^^^^^^^^^^^^^^^^^^^^^^^^^

基于向量的语义检索便捷方法（内部调用 ``search`` 并传入向量）。

**签名**：

.. code-block:: python

   def search_similarity_by_vector(
       query_embedding: np.ndarray,
       k: int = 5,
       ms_names: Optional[List[str]] = None,
       candidate_uids: Optional[List[str]] = None
   ) -> List[Tuple[MemoryUnit, float]]

**使用示例**：

.. code-block:: python

   query_embedding = embedder.encode(["query"])[0]
   results = system.semantic_map.search_similarity_by_vector(
       query_embedding, k=10
   )

search_hybrid
^^^^^^^^^^^^^

综合混合检索（Dense + BM25 + Sparse 多路召回 + RRF 融合 + 可选图扩展）。

这是一个高级检索方法，不仅执行 Dense + BM25 + Sparse 的多路召回和融合，
还会结合图结构信息进行子图扩展和关系推理，是系统中最强大的检索能力。

**核心特点**：

1. **多路召回**：同时使用 Dense、BM25、Sparse 三种检索器
2. **RRF 融合**：倒数排名融合合并多路结果
3. **BFS 图扩展**：基于种子节点在 SemanticGraph 中进行子图跳转扩展
4. **关系感知重排序**：考虑图结构关系对结果进行重排序

**签名**：

.. code-block:: python

   def search_hybrid(
       query: str,
       top_k: int = 10,
       ms_names: Optional[List[str]] = None,
       use_graph_expansion: bool = True,
       bfs_depth: int = 2,
       rerank: bool = True,
       **kwargs
   ) -> List[Tuple[MemoryUnit, float]]

**参数**：

- ``query``：查询文本
- ``top_k``：最终返回结果数量
- ``ms_names``：限定检索空间
- ``use_graph_expansion``：是否启用图扩展（默认 True）
- ``bfs_depth``：BFS 扩展深度（默认 2）
- ``rerank``：是否使用 Cross-Encoder 重排序（默认 True）

**内部流程**：

.. mermaid::

   graph LR
       A[用户查询] --> B[多路召回<br>Dense+BM25+Sparse]
       B --> C[RRF融合]
       C --> D[BFS图扩展<br>SemanticGraph]
       D --> E[候选合并去重]
       E --> F{use_rerank?}
       F -->|Yes| G[Cross-Encoder<br>重排序]
       F -->|No| H[直接返回]
       G --> I[最终结果]

**使用示例**：

.. code-block:: python

   # 完整混合检索（含图扩展和重排序）
   results = system.semantic_map.search_hybrid(
       "张三在北京做了什么？",
       top_k=10,
       use_graph_expansion=True,
       bfs_depth=2,
       rerank=True
   )

   # 仅多路召回（不含图扩展）
   results = system.semantic_map.search_hybrid(
       "query",
       top_k=10,
       use_graph_expansion=False
   )

.. _retrieval-semantic-graph:
