全记忆统一检索接口
===========================

以下接口提供跨空间、跨视角的综合检索能力。

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
       top_k: int = 5,
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

.. warning:: 📋 预想接口 — 此接口尚未实现，以下文档描述目标设计，API 可能变更。

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
