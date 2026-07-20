MemorySystem API 参考
=====================

``MemorySystem`` 是 Mandol 记忆系统的主入口类，提供数据管理、记忆构建和检索的全部公开接口。

初始化
------

MemorySystem()
^^^^^^^^^^^^^^

.. code-block:: python

   from mandol import MemorySystem

   system = MemorySystem()

使用默认配置创建实例。默认使用本地 Qwen3-Embedding-4B 和 Qwen3-Reranker-4B 模型。

from_yaml_config()
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   system = MemorySystem.from_yaml_config("config.yaml")

从 YAML 配置文件创建实例。支持自定义 LLM、Embedder、Reranker 等全部配置。

**参数**：

- ``yaml_path`` (str)：YAML 配置文件路径
- ``embedder`` (Optional[EmbeddingProvider])：自定义 Embedder 实例
- ``reranker`` (Optional[Reranker])：自定义 Reranker 实例
- ``llm_provider`` (Optional[LLMProvider])：自定义 LLM Provider 实例

load()
^^^^^^

.. code-block:: python

   system = MemorySystem.load("./memory_snapshot")

从目录加载已保存的记忆系统状态。类方法。

**参数**：

- ``directory`` (str)：保存目录路径
- ``embedder`` (Optional[EmbeddingProvider])：自定义 Embedder 实例
- ``reranker`` (Optional[Reranker])：自定义 Reranker 实例
- ``llm_provider`` (Optional[LLMProvider])：自定义 LLM Provider 实例

数据管理
--------

add()
^^^^^

.. code-block:: python

   system.add(unit)

添加单条记忆单元。系统自动执行分块、向量化、会话检测和相似度建边。

**参数**：

- ``unit`` (MemoryUnit)：记忆单元实例

add_many()
^^^^^^^^^^

.. code-block:: python

   system.add_many(units)

批量添加记忆单元，比逐条 ``add`` 更高效。

**参数**：

- ``units`` (Sequence[MemoryUnit])：记忆单元序列

save()
^^^^^^

.. code-block:: python

   system.save("./memory_snapshot")

将记忆系统状态导出为目录（包含多个 JSON 文件）。

**参数**：

- ``directory`` (str)：导出目录路径

记忆构建
--------

build_high_level()
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   report = system.build_high_level(mode="auto")

从原始对话记忆中提取高层语义结构。

**参数**：

- ``mode`` (str)：构建模式
  - ``"auto"``：增量模式，仅处理未构建过的会话（推荐）
  - ``"force"``：全量重建模式，清除所有高层记忆并重新构建

**返回值**：``BuildReport`` 对象，包含：

- ``sessions_processed`` (int)：处理的会话数
- ``units_processed`` (int)：处理的记忆单元数

**自动构建流程**：

1. 会话分割（LLM 驱动）
2. 情景 / 知识 / 情感 / 程序摘要生成
3. 洞察提取与全局合并
4. 实体提取与去重
5. 事件提取与去重
6. 实体关系构建
7. 事件因果链构建
8. 跨会话实体/事件合并

build_high_level_async()
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   system.build_high_level_async()

异步执行记忆构建，不阻塞主线程。内部逻辑与 ``build_high_level`` 相同。

检索接口
--------

holistic_retrieve()
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   hits = system.holistic_retrieve("张三去了哪里？", top_k=10)

全记忆统一检索，跨空间、跨视角的综合检索。

**参数**：

- ``query`` (str)：查询文本
- ``top_k`` (int)：返回结果数量，默认 10
- ``use_rerank`` (bool)：是否使用 Cross-Encoder 重排序，默认 True

**返回值**：``List[SearchHit]``，每个 SearchHit 包含：

- ``unit`` (MemoryUnit)：命中的记忆单元
- ``final_score`` (float)：最终得分
- ``scores`` (Dict[str, float])：各阶段得分
- ``ranks`` (Dict[str, int])：各阶段排名

**检索流程**：

1. 分组召回：BASE / ENTITY / EVENT / SUMMARY 四组独立检索
2. 每组内部：Dense + BM25 + Sparse 三路召回 → RRF 融合 → BFS 扩展
3. 全局重排：所有候选合并后通过 Cross-Encoder Reranker 重排序

retrieve_by_view()
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   hits = system.retrieve_by_view("客户喜欢什么？", view="knowledge", top_k=5)

按视角类别检索。

**参数**：

- ``query`` (str)：查询文本
- ``view`` (str)：视角名称，可选值：
  - ``"base_memory"``：原始对话记忆
  - ``"entity_relation"``：知识实体
  - ``"event_causal"``：情景事件
  - ``"emotional"``：情感摘要
  - ``"episodic"``：情景摘要
  - ``"knowledge"``：知识摘要
  - ``"procedural"``：程序摘要
  - ``"insights"``：洞察
- ``top_k`` (int)：返回结果数量，默认 10
- ``use_rerank`` (bool)：是否使用重排序，默认 True

**返回值**：``List[SearchHit]``

retrieve_in_space()
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   hits = system.retrieve_in_space("客户", space_name="root_knowledge_entity", top_k=5)

在指定空间内检索。

**参数**：

- ``query`` (str)：查询文本
- ``space_name`` (str)：空间名称，如 ``"root_knowledge_entity"``、``"root_base_memory"``
- ``top_k`` (int)：返回结果数量，默认 10
- ``use_rerank`` (bool)：是否使用重排序，默认 True

**返回值**：``List[SearchHit]``

属性访问
--------

semantic_map
^^^^^^^^^^^^

.. code-block:: python

   smap = system.semantic_map

获取底层 SemanticMap 实例，用于直接调用 SemanticMap 的检索接口（如 ``search``、``search_hybrid``）。

semantic_graph
^^^^^^^^^^^^^^

.. code-block:: python

   sgraph = system.semantic_graph

获取底层 SemanticGraph 实例，用于直接调用图检索接口（如 ``get_explicit_neighbors``、``search_graph_relations``）。
