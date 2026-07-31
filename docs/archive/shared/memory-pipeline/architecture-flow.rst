开发者深度：架构级流程与组件关系
=======================================

本节面向需要理解内部架构的开发者和贡献者。关于完整的流程描述（包含分块、会话分割、摘要 Map-Reduce、统一事实管线、跨会话合并等细节），请先阅读 :doc:`detailed-flow`。本节聚焦于架构分层和可扩展点。

.. _diagram-placeholder-arch-1:

.. image:: /_static/images/hexagonal-architecture.png
    :alt: 六边形架构分层图
    :align: center

开发者可介入的环节
------------------

Mandol 的管道在每个阶段都留有扩展点，开发者可以通过实现端口接口或调整配置来影响流程。

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - 管道阶段
     - 可定制内容
     - 扩展方式
   * - **分块（Chunking）**
     - 分块大小、重叠 token 数、token 估计算法
     - 配置 ``chunk_max_tokens``、``overlap_tokens``；替换 ``DocumentChunker`` 中的 token 估算器
   * - **向量化（Embedding）**
     - Embedding 模型、维度
     - 实现 ``EmbeddingProvider`` 端口
   * - **会话分割（Sessioning）**
     - 分割 prompt、检测间隔、上下文窗口大小
     - 配置 ``session_check_interval``、``session_time_gap_seconds``；覆盖 ``SessionManager`` 的 prompt 模板
   * - **摘要生成（Summary Map-Reduce）**
     - 四种摘要类型的 prompt、分块 token 预算、Map/Reduce 策略
     - 覆盖 ``SummaryMapReducer`` 中各类型的 prompt 模板；调整 ``map_chunk_max_tokens``
   * - **实体/事件提取（Unified Fact Pipeline）**
     - 提取 prompt、多信号检索参数、候选数量
     - 覆盖 ``UnifiedFactPipeline`` 中的四类 prompt；调整 ``top_k_entities``、``top_k_events``；替换检索策略
   * - **跨会话合并（Cross-Session Coref）**
     - 合并判定 prompt、向量相似度阈值、LLM 置信度阈值、最大候选数
     - 配置 ``coref_vector_threshold``、``coref_llm_confidence_threshold``、``coref_max_candidates``；覆盖 LLM 裁判 prompt
   * - **洞察提炼（Insight）**
     - 洞察提取 prompt、全局洞察合并 prompt
     - 覆盖 ``InsightMapReducer`` 和 ``GlobalInsightManager`` 中的 prompt 模板
   * - **检索（Retrieval）**
     - Reranker 模型、RRF 参数、BFS 扩展参数
     - 实现 ``Reranker`` 端口；配置 ``bfs_expansion_per_seed``、``bfs_expansion_hops``

关键配置参数速查
----------------

以下参数直接影响高阶记忆构建流程的行为：

.. list-table::
   :header-rows: 1
   :widths: 35 20 45

   * - 参数
     - 默认值
     - 作用
   * - ``chunk_max_tokens``
     - 512
     - 触发分块的 token 上限
   * - ``overlap_tokens``
     - 0
     - 相邻分块间的上下文重叠 token 数
   * - ``session_check_interval``
     - 20
     - 累积多少条记忆后触发一次 LLM 会话检测
   * - ``session_max_pending``
     - 100
     - 待处理队列上限，超过后强制 flush
   * - ``similarity_threshold``
     - 0.7
     - 建立 SEMANTIC_SIMILAR 边的最小余弦相似度
   * - ``similarity_recent_window``
     - 20
     - 即时相似度计算时考虑最近多少条记忆
   * - ``use_unified_pipeline``
     - true
     - 是否使用统一事实管线（推荐）；false 回退到旧版维度构建器
   * - ``incremental_cross_session_coref``
     - true
     - 是否启用增量跨会话共指消解
   * - ``coref_vector_threshold``
     - 0.75
     - 共指候选召回的向量相似度阈值
   * - ``coref_llm_confidence_threshold``
     - 0.6
     - LLM 判定共指的最低置信度
   * - ``coref_max_candidates``
     - 10
     - 共指候选的最大数量
   * - ``bfs_expansion_per_seed``
     - 5
     - BFS 扩展时每个种子节点获取的邻居数
   * - ``bfs_expansion_hops``
     - 1
     - BFS 扩展的跳数
   * - ``auto_build_if_empty``
     - true
     - 检索时若高阶记忆为空，是否自动触发 build

组件关系
--------

.. code-block::

   MemorySystem                         # 主入口门面
   ├── SemanticMapService               # 记忆单元的 CRUD、向量索引、空间管理
   │   ├── UnitStore                    #   单元持久化
   │   ├── AdaptiveVectorIndex          #   自适应向量索引
   │   │   ├── BruteForceVectorIndex    #     < promote_threshold 时使用
   │   │   └── FAISSVectorIndex         #     >= promote_threshold 时自动切换
   │   ├── EmbeddingProvider            #   向量化抽象
   │   └── Reranker                     #   重排序抽象
   ├── SemanticGraphService             # 图关系服务
   │   └── GraphStore                   #   图存储抽象
   ├── DocumentChunker                  # 文档分块
   ├── SessionManager                   # 会话检测与管理
   ├── SummaryMapReducer                # 四类摘要 Map-Reduce
   ├── InsightMapReducer                # Session 级洞察提取
   ├── GlobalInsightManager             # 全局洞察累积合并
   ├── UnifiedFactPipeline              # 实体/事件/关系统一提取
   │   └── CrossSessionCorefManager     #   跨会话共指消解
   └── HybridRetriever                  # 检索编排
       ├── DenseRetriever               #   稠密向量检索
       ├── Bm25Retriever                #   BM25 关键词检索
       ├── SparseRetriever              #   稀疏向量检索
       ├── SubgraphHopRetriever         #   图扩展检索
       └── RRFusion                     #   排名融合

端口接口一览
------------

每个端口定义了一个可在 ``infrastructure/`` 中替换的抽象。以下为各端口的核心方法签名：

**EmbeddingProvider**
   - ``embed_text(texts: list[str]) -> list[Embedding]``
   - ``embed_image_paths(paths: list[str]) -> list[Embedding]``
   - ``embedding_dim() -> int``

**LLMProvider**
   - ``generate(prompt: str, **kwargs) -> str``
   - ``generate_structured(prompt: str, schema: dict, **kwargs) -> dict``

**Reranker**
   - ``rerank(query: str, units: list[MemoryUnit], top_k: int) -> list[tuple[MemoryUnit, float]]``

**VectorIndex**
   - ``upsert(items: list[tuple[Uid, Embedding]]) -> None``
   - ``search(query: Embedding, top_k: int) -> list[tuple[Uid, float]]``
   - ``delete(uids: list[Uid]) -> None``
   - ``rebuild(items: list[tuple[Uid, Embedding]]) -> None``
   - ``dim() -> int``

**UnitStore**
   - ``upsert_units(units: list[MemoryUnit]) -> None``
   - ``delete_units(uids: list[Uid]) -> None``
   - ``get_unit(uid: Uid) -> MemoryUnit | None``
   - ``list_units() -> list[MemoryUnit]``
   - ``get_units(uids: list[Uid]) -> list[MemoryUnit]``
   - ``upsert_spaces(spaces: list[MemorySpace]) -> None``
   - ``get_space(name: SpaceName) -> MemorySpace | None``
   - ``list_spaces() -> list[MemorySpace]``
   - ``flush() -> None``

**GraphStore**
   - ``upsert_relationship(source: Uid, target: Uid, rel_type: str, properties: dict) -> None``
   - ``delete_relationship(source: Uid, target: Uid, rel_type: str | None) -> None``
   - ``get_relationship(source: Uid, target: Uid, rel_type: str) -> dict | None``
   - ``get_neighbors(uid: Uid, rel_type: str | None, direction: str) -> list[Uid]``
   - ``flush() -> None``
