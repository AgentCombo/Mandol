参数调优指南
============

Mandol 提供了丰富的配置参数，允许你根据不同场景优化系统行为。本指南介绍各参数的作用、推荐值及调优策略。

分块参数
--------

chunk_max_tokens
^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - 参数
     - 默认值
     - 说明
   * - ``chunk_max_tokens``
     - 512
     - 记忆分块最大 token 数

**调优建议**：

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - 场景
     - 推荐值
     - 说明
   * - 短对话（客服/聊天）
     - 256
     - 对话片段通常较短，小分块粒度更精确
   * - 中等对话（个人助手）
     - 512
     - 默认值，适合大多数场景
   * - 长文档（知识库/文档）
     - 1024
     - 长文档需要更大的上下文窗口

会话分割参数
------------

session_time_gap_seconds
^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - 参数
     - 默认值
     - 说明
   * - ``session_time_gap_seconds``
     - 1800
     - 基于时间间隔的会话分割阈值（秒）

**调优建议**：

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - 场景
     - 推荐值
     - 说明
   * - 客服对话
     - 300（5 分钟）
     - 客服会话通常较短，5 分钟无交互即视为新会话
   * - 个人助手
     - 1800（30 分钟）
     - 默认值，适合日常对话节奏
   * - 知识库导入
     - 86400（24 小时）
     - 知识库无会话概念，设为极大值避免分割

session_check_interval
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - 参数
     - 默认值
     - 说明
   * - ``session_check_interval``
     - 20
     - 每累积多少条记忆触发一次会话边界检测

**调优建议**：增大此值可减少 LLM 调用次数（降低成本），但会延迟会话分割。实时场景建议保持默认值 20。

检索参数
--------

similarity_threshold
^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - 参数
     - 默认值
     - 说明
   * - ``similarity_threshold``
     - 0.7
     - 语义相似边建立阈值

**调优建议**：

- **提高阈值** （0.8-0.9）：更精确但召回少，适合精确匹配场景
- **降低阈值** （0.5-0.6）：召回多但噪声多，适合探索性检索场景
- 此参数影响 ``SEMANTIC_SIMILAR`` 边的密度，间接影响 BFS 扩展效果

bfs_expansion_per_seed / bfs_expansion_hops
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - 参数
     - 默认值
     - 说明
   * - ``bfs_expansion_per_seed``
     - 3
     - BFS 扩展时每个种子节点扩展的邻居数
   * - ``bfs_expansion_hops``
     - 1
     - BFS 扩展的跳数

**调优建议**：

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - 场景
     - 推荐配置
     - 说明
   * - 精确检索
     - per_seed=0, hops=0
     - 关闭图扩展，仅依赖向量/BM25 召回
   * - 一般场景
     - per_seed=3, hops=1
     - 默认配置，平衡召回率和延迟
   * - 多跳推理
     - per_seed=5, hops=2
     - 适合需要跨多个实体推理的复杂查询

.. note::

   BFS 扩展会显著增加检索延迟。``hops=2`` 时延迟约为 ``hops=1`` 的 2-3 倍。

similarity_top_k
^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - 参数
     - 默认值
     - 说明
   * - ``similarity_top_k``
     - 5
     - 向量检索返回的候选数量

**调优建议**：此参数影响每组检索的初始召回数量。增大可提高召回率但增加后续处理开销。一般场景 5-10 即可，复杂查询可设为 15-20。

LLM 调用参数
-------------

max_entities_per_llm / max_events_per_llm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - 参数
     - 默认值
     - 说明
   * - ``max_entities_per_llm``
     - 50
     - 实体去重时每次 LLM 调用的最大候选数
   * - ``max_events_per_llm``
     - 50
     - 事件去重时每次 LLM 调用的最大候选数

**调优建议**：

- **增大** （100-200）：可处理更多候选，去重更全面，但 LLM 成本增加
- **减小** （20-30）：降低 LLM 成本，但可能遗漏重复实体/事件

索引参数
--------

promote_threshold
^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - 参数
     - 默认值
     - 说明
   * - ``promote_threshold``
     - 100
     - FAISS/BM25/TF-IDF 索引升级阈值

**调优建议**：

- 当记忆单元数量 < 100 时，系统使用暴力搜索（精确但慢）
- 当记忆单元数量 >= 100 时，系统自动升级为 FAISS/BM25/TF-IDF 索引
- 如果你的数据集较小（< 100 条），可以降低此值以提前启用索引

场景推荐配置
------------
-----------

客服对话场景
^^^^^^^^^^^^

.. code-block:: yaml

   system:
     chunk_max_tokens: 256
     session_time_gap_seconds: 300
     similarity_top_k: 5
     similarity_threshold: 0.7
     bfs_expansion_per_seed: 3
     bfs_expansion_hops: 1

个人助手场景
^^^^^^^^^^^^

.. code-block:: yaml

   system:
     chunk_max_tokens: 512
     session_time_gap_seconds: 1800
     similarity_top_k: 10
     similarity_threshold: 0.65
     bfs_expansion_per_seed: 5
     bfs_expansion_hops: 2

知识库场景
^^^^^^^^^^

.. code-block:: yaml

   system:
     chunk_max_tokens: 1024
     session_time_gap_seconds: 86400
     similarity_top_k: 10
     similarity_threshold: 0.7
     bfs_expansion_per_seed: 3
     bfs_expansion_hops: 1
