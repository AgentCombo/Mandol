最简配置
============

对于大多数基础用户，只需关注以下 4 个配置项。

配置 1：API Key
---------------

写在 ``.env`` 文件中：

.. code-block:: bash

   OPENAI_API_KEY=sk-your-key-here

配置 2：模型选择
----------------

写在 ``config.yaml`` 中：

.. code-block:: yaml

   llm:
     model: "gpt-4o-mini"          # LLM 模型（用于会话分割、实体/事件提取）
     base_url: "https://api.openai.com/v1"

   embedder:
     model: "Qwen/Qwen3-Embedding-4B"   # 向量化模型
     device: "cpu"                       # 或 "cuda"

   reranker:
     model: "Qwen/Qwen3-Reranker-4B"    # 重排序模型（提升检索精度）
     device: "cpu"                       # 或 "cuda"

.. tip::

   Embedder 负责将文本转为向量用于检索，Reranker 负责对检索结果精排。两者都支持本地模型或远程 API 模式。如果 GPU 显存有限，建议 Embedder 用本地模型、Reranker 用远程 API。

配置 3：分块大小
----------------

控制每段文本的最大长度：

.. code-block:: yaml

   system:
     chunk_max_tokens: 512    # 默认值

配置 4：远程 API 开关
----------------------

如不想下载本地模型，可使用远程 API：

.. code-block:: yaml

   embedder:
     use_remote: true
     base_url: "http://your-api-endpoint/v1"
     api_path: "/embeddings"
     api_key: "your-key"

   reranker:
     use_remote: true
     base_url: "http://your-api-endpoint"
     api_path: "/v1/rerank"
     api_key: "your-key"

不需要改的其他配置
------------------

以下配置使用默认值即可满足大多数场景：

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - 配置项
     - 默认值
     - 什么时候需要改
   * - ``similarity_threshold``
     - 0.7
     - 想要更精确的相似匹配可提高到 0.8
   * - ``bfs_expansion_hops``
     - 1
     - 需要更多图扩展结果可设为 2
   * - ``session_check_interval``
     - 20
     - 控制多久触发一次 LLM 话题边界检测

完整配置参考请查阅 :doc:`/advanced-user/parameter-tuning/index`。

使用预设配置
------------

如果你不想手写 YAML，可以直接用预设：

.. code-block:: python

   from mandol import MemorySystem

   # 什么都不传 = 全默认（本地模型模式）
   system = MemorySystem()

   # 从 YAML 文件加载
   system = MemorySystem.from_yaml_config("config.yaml")
