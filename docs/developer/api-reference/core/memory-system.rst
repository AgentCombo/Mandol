MemorySystem 完整参考
=======================

MemorySystem 是 Mandol 的统一入口类。

初始化
------

``MemorySystem(**kwargs)`` — 使用默认配置创建

.. code-block:: python

   system = MemorySystem(
       embedder=None,       # Optional[EmbeddingProvider]
       reranker=None,       # Optional[Reranker]
       llm_provider=None,   # Optional[LLMProvider]
   )

``MemorySystem.from_yaml_config(yaml_path, **override)`` — 从 YAML 创建

.. code-block:: python

   system = MemorySystem.from_yaml_config("config.yaml")

``MemorySystem.load(directory, **override)`` — 从目录加载（类方法）

.. code-block:: python

   system = MemorySystem.load("./snapshot")

数据管理
--------

``add(unit: MemoryUnit) -> None``

``add_many(units: Sequence[MemoryUnit]) -> None``

``save(directory: str) -> None``

记忆构建
--------

``build_high_level(mode: str = "auto") -> BuildReport``

- ``mode="auto"``：增量（推荐）
- ``mode="force"``：全量重建

``BuildReport`` 包含：

- ``sessions_processed: int``
- ``units_processed: int``

检索接口
--------

``holistic_retrieve(query, top_k=10, use_rerank=True) -> list[SearchHit]``

``retrieve_by_view(query, view, top_k=10, use_rerank=True) -> list[SearchHit]``

``retrieve_in_space(query, space_name, top_k=10, use_rerank=True) -> list[SearchHit]``

``search(query, top_k=10, use_rerank=True, use_graph_expansion=False, space_names=None, retriever_types=None) -> list[SearchHit]`` *(预想)*

维护接口
--------

``flush() -> None`` — 持久化缓存

属性
----

- ``system.semantic_map`` — ``SemanticMapService``
- ``system.semantic_graph`` — ``SemanticGraphService``

配置参数（MemorySystemConfig 30+ 字段）
----------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - 字段
     - 默认值
     - 说明
   * - ``chunk_max_tokens``
     - 512
     - 分块最大 token 数
   * - ``session_time_gap_seconds``
     - 1800
     - 时间间隔分割阈值
   * - ``session_check_interval``
     - 20
     - LLM 检测间隔
   * - ``session_max_pending``
     - 100
     - 最大待处理数
   * - ``similarity_threshold``
     - 0.7
     - 相似边阈值
   * - ``similarity_top_k``
     - 5
     - 向量检索召回数
   * - ``similarity_recent_window``
     - 20
     - 相似度计算窗口
   * - ``bfs_expansion_per_seed``
     - 3
     - BFS 每个种子的邻居数
   * - ``bfs_expansion_hops``
     - 1
     - BFS 跳数
   * - ``max_entities_per_llm``
     - 50
     - 实体去重候选数
   * - ``max_events_per_llm``
     - 50
     - 事件去重候选数
   * - ``promote_threshold``
     - 100
     - 索引升级阈值
