配置详解
========

.. note::

   基础配置已迁移至 :doc:`/basic-user/configuration-simple`。完整参数调优已迁移至 :doc:`/advanced-user/parameter-tuning/index`。本页面将在后续版本中移除，请更新你的书签。

Mandol 的配置遵循三级优先机制，灵活支持不同部署场景。

配置优先级
----------

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - 优先级
     - 来源
     - 说明
   * - 1（最高）
     - ``.env`` 环境变量
     - 敏感信息（API Key、密码等），始终覆盖其他来源
   * - 2
     - ``config.yaml``
     - 非敏感配置（模型名称、设备、阈值等）
   * - 3（最低）
     - 代码默认值
     - 内置默认值，无需任何配置即可运行

即：``.env 环境变量 > config.yaml > 代码默认值``。

环境变量
--------

以下为 ``.env`` 文件中所有可配置的环境变量：

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - 变量名
     - 默认值
     - 说明
   * - ``OPENAI_API_KEY``
     - （空）
     - OpenAI API 密钥，用于 LLM 调用
   * - ``OPENAI_API_BASE``
     - ``https://api.openai.com/v1``
     - LLM API 基础 URL，可替换为兼容接口
   * - ``MANDOL_LLM_MODEL``
     - ``gpt-4o-mini``
     - LLM 模型名称
   * - ``MANDOL_LLM_TIMEOUT_S``
     - ``60``
     - LLM 请求超时时间（秒）
   * - ``MANDOL_EMBEDDER_MODEL``
     - ``Qwen/Qwen3-Embedding-4B``
     - Embedding 模型名称
   * - ``MANDOL_EMBEDDER_DEVICE``
     - ``cpu``
     - Embedding 设备，可选 ``cpu`` 或 ``cuda``
   * - ``USE_REMOTE_EMBEDDER``
     - ``false``
     - 是否使用远程 Embedder 服务
   * - ``MANDOL_EMBEDDER_BASE_URL``
     - ``http://localhost:8000/v1``
     - 远程 Embedder 服务 URL
   * - ``MANDOL_EMBEDDER_API_PATH``
     - ``/embeddings``
     - 远程 Embedder API 路径
   * - ``MANDOL_EMBEDDER_API_KEY``
     - （空）
     - 远程 Embedder API 密钥
   * - ``MANDOL_RERANKER_MODEL``
     - ``Qwen/Qwen3-Reranker-4B``
     - Reranker 模型名称
   * - ``MANDOL_RERANKER_DEVICE``
     - ``cpu``
     - Reranker 设备，可选 ``cpu`` 或 ``cuda``
   * - ``USE_REMOTE_RERANKER``
     - ``false``
     - 是否使用远程 Reranker 服务
   * - ``MANDOL_RERANKER_BASE_URL``
     - （空）
     - 远程 Reranker 服务 URL
   * - ``MANDOL_RERANKER_API_PATH``
     - ``/v1/rerank``
     - 远程 Reranker API 路径
   * - ``MANDOL_RERANKER_API_KEY``
     - （空）
     - 远程 Reranker API 密钥

YAML 配置
---------

``config.yaml`` 文件分为五个配置段：llm、embedder、reranker、system、storage。

LLM 配置
^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 15 25 40

   * - 参数
     - 类型
     - 默认值
     - 说明
   * - ``base_url``
     - str
     - ``https://api.openai.com/v1``
     - LLM API 基础 URL，可替换为任意 OpenAI 兼容接口
   * - ``model``
     - str
     - ``gpt-4o-mini``
     - LLM 模型名称

Embedder 配置
^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 15 25 40

   * - 参数
     - 类型
     - 默认值
     - 说明
   * - ``model``
     - str
     - ``Qwen/Qwen3-Embedding-4B``
     - Embedding 模型名称（本地模式使用 sentence-transformers）
   * - ``device``
     - str
     - ``cpu``
     - 推理设备，可选 ``cpu`` 或 ``cuda``
   * - ``dimension``
     - int
     - ``2560``
     - Embedding 向量维度
   * - ``use_remote``
     - bool
     - ``false``
     - 是否使用远程 Embedder 服务
   * - ``base_url``
     - str
     - ``http://localhost:8000/v1``
     - 远程 Embedder 服务 URL（仅远程模式）
   * - ``api_path``
     - str
     - ``/embeddings``
     - 远程 Embedder API 路径（仅远程模式）
   * - ``api_key``
     - str
     - （空）
     - 远程 Embedder API 密钥（仅远程模式，建议通过 .env 设置）
   * - ``timeout``
     - int
     - ``30``
     - 远程 Embedder 请求超时时间（秒，仅远程模式）

Reranker 配置
^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 15 25 40

   * - 参数
     - 类型
     - 默认值
     - 说明
   * - ``model``
     - str
     - ``Qwen/Qwen3-Reranker-4B``
     - Reranker 模型名称（本地模式使用 CrossEncoder）
   * - ``device``
     - str
     - ``cpu``
     - 推理设备，可选 ``cpu`` 或 ``cuda``
   * - ``use_remote``
     - bool
     - ``false``
     - 是否使用远程 Reranker 服务
   * - ``base_url``
     - str
     - （空）
     - 远程 Reranker 服务 URL（仅远程模式）
   * - ``api_path``
     - str
     - ``/v1/rerank``
     - 远程 Reranker API 路径（仅远程模式）
   * - ``api_key``
     - str
     - （空）
     - 远程 Reranker API 密钥（仅远程模式，建议通过 .env 设置）
   * - ``timeout``
     - int
     - ``30``
     - 远程 Reranker 请求超时时间（秒，仅远程模式）

System 配置
^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 15 20 35

   * - 参数
     - 类型
     - 默认值
     - 说明
   * - ``chunk_max_tokens``
     - int
     - ``512``
     - 记忆分块最大 token 数
   * - ``session_time_gap_seconds``
     - int
     - ``1800``
     - 基于时间间隔的会话分割阈值（秒），即 30 分钟
   * - ``session_check_interval``
     - int
     - ``20``
     - 每累积多少条记忆触发一次会话边界检测
   * - ``session_max_pending``
     - int
     - ``100``
     - 待处理记忆最大数量，超过此值强制分割
   * - ``similarity_top_k``
     - int
     - ``5``
     - 向量检索返回的候选数量
   * - ``similarity_threshold``
     - float
     - ``0.7``
     - 语义相似边建立阈值
   * - ``similarity_recent_window``
     - int
     - ``20``
     - 计算相似度时的最近记忆窗口大小
   * - ``bfs_expansion_per_seed``
     - int
     - ``3``
     - BFS 扩展时每个种子节点扩展的邻居数
   * - ``bfs_expansion_hops``
     - int
     - ``1``
     - BFS 扩展的跳数
   * - ``max_context_units``
     - int
     - ``20``
     - LLM 单次调用的最大上下文记忆数
   * - ``max_entities_per_llm``
     - int
     - ``50``
     - 实体去重时每次 LLM 调用的最大候选数
   * - ``max_events_per_llm``
     - int
     - ``50``
     - 事件去重时每次 LLM 调用的最大候选数
   * - ``promote_threshold``
     - int
     - ``100``
     - FAISS/BM25/TF-IDF 索引升级阈值

Storage 配置
^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 15 25 35

   * - 参数
     - 类型
     - 默认值
     - 说明
   * - ``root``
     - str
     - ``null``
     - 持久化存储根路径，如 ``./data/memory``
   * - ``enable_persistence``
     - bool
     - ``false``
     - 是否启用自动持久化
   * - ``auto_save_interval``
     - int
     - ``300``
     - 自动保存间隔（秒）

远程模式 vs 本地模式
--------------------

.. list-table::
   :header-rows: 1
   :widths: 25 37 38

   * - 特性
     - 远程模式
     - 本地模式
   * - Embedding
     - 通过 OpenAI 兼容 API 调用远程服务
     - 本地 Sentence-Transformers 推理
   * - Reranker
     - 通过 OpenAI 兼容 API 调用远程服务
     - 本地 CrossEncoder 推理
   * - API Key
     - 需要配置 ``OPENAI_API_KEY``
     - 无需任何 API Key
   * - GPU 需求
     - 无（服务端 GPU）
     - 推荐有 GPU，CPU 也可运行但较慢
   * - 首次启动
     - 无需下载模型
     - 需下载模型文件（约 2-4 GB）
   * - 适用场景
     - 生产环境、快速体验
     - 离线环境、隐私敏感场景
   * - 配置方式
     - ``USE_REMOTE_EMBEDDER=true`` + ``USE_REMOTE_RERANKER=true``
     - ``USE_REMOTE_EMBEDDER=false`` + ``USE_REMOTE_RERANKER=false``

常见配置模式
------------
-----------

最小化 CPU 配置
^^^^^^^^^^^^^^^

适合快速体验，所有模型在 CPU 上运行：

.. code-block:: yaml

   llm:
     base_url: "https://api.openai.com/v1"
     model: "gpt-4o-mini"

   embedder:
     model: "Qwen/Qwen3-Embedding-4B"
     device: "cpu"
     dimension: 2560
     use_remote: false

   reranker:
     model: "Qwen/Qwen3-Reranker-4B"
     device: "cpu"
     use_remote: false

   storage:
     root: null
     enable_persistence: false

   system:
     chunk_max_tokens: 512
     session_time_gap_seconds: 1800
     similarity_top_k: 5
     similarity_threshold: 0.7

GPU 加速配置
^^^^^^^^^^^^

适合有 GPU 的开发环境，本地模型使用 CUDA 加速：

.. code-block:: yaml

   llm:
     base_url: "https://api.openai.com/v1"
     model: "gpt-4o-mini"

   embedder:
     model: "Qwen/Qwen3-Embedding-4B"
     device: "cuda"
     dimension: 2560
     use_remote: false

   reranker:
     model: "Qwen/Qwen3-Reranker-4B"
     device: "cuda"
     use_remote: false

   storage:
     root: "./data/memory"
     enable_persistence: true
     auto_save_interval: 300

   system:
     chunk_max_tokens: 512
     session_time_gap_seconds: 1800
     similarity_top_k: 5
     similarity_threshold: 0.7

远程 API 配置
^^^^^^^^^^^^^^

适合使用远程 API 服务的场景：

.. code-block:: yaml

   llm:
     base_url: "https://api.openai.com/v1"
     model: "gpt-4o-mini"

   embedder:
     model: "Qwen/Qwen3-Embedding-4B"
     device: "cpu"
     dimension: 2560
     use_remote: true
     base_url: "http://localhost:8000/v1"
     api_path: "/embeddings"
     timeout: 30

   reranker:
     model: "Qwen/Qwen3-Reranker-4B"
     device: "cpu"
     use_remote: true
     base_url: "https://your-reranker-api-endpoint.com"
     api_path: "/v1/rerank"
     timeout: 30

   storage:
     root: "./data/memory"
     enable_persistence: true
     auto_save_interval: 300

   system:
     chunk_max_tokens: 512
     session_time_gap_seconds: 1800
     session_check_interval: 20
     session_max_pending: 100
     similarity_top_k: 5
     similarity_threshold: 0.7
     similarity_recent_window: 20
     bfs_expansion_per_seed: 3
     bfs_expansion_hops: 1
     max_context_units: 20
     max_entities_per_llm: 50
     max_events_per_llm: 50
     promote_threshold: 100

对应的 ``.env`` 文件：

.. code-block:: bash

   OPENAI_API_KEY=sk-your-api-key-here
   USE_REMOTE_EMBEDDER=true
   MANDOL_EMBEDDER_BASE_URL=http://localhost:8000/v1
   MANDOL_EMBEDDER_API_KEY=your-embedder-api-key
   USE_REMOTE_RERANKER=true
   MANDOL_RERANKER_BASE_URL=https://your-reranker-api-endpoint.com
   MANDOL_RERANKER_API_KEY=your-reranker-api-key
