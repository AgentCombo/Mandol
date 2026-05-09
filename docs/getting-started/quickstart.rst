快速开始
========

.. note::

   本文档已迁移至 :doc:`/basic-user/five-minute-start`。本页面将在后续版本中移除，请更新你的书签。

本指南将帮助你在几分钟内运行 Mandol 记忆系统。根据你的使用场景，选择以下两种模式之一。

模式一：远程 API（推荐新手使用）
--------------------------------

此模式通过远程 API 调用 Embedding、Reranker 和 LLM 服务，无需本地 GPU，适合快速体验和开发调试。

**前提**：已配置 ``OPENAI_API_KEY`` 环境变量，并准备好 ``config.yaml`` 配置文件。

.. code-block:: python

   from mandol import MemorySystem, MemoryUnit, Uid

   system = MemorySystem.from_yaml_config("config.yaml")

   unit = MemoryUnit(
       uid=Uid("dialogue_001"),
       raw_data={"text_content": "张三今天去北京出差了"},
       metadata={"timestamp": "2024-01-15T10:00:00"},
   )
   system.add(unit)

   unit2 = MemoryUnit(
       uid=Uid("dialogue_002"),
       raw_data={"text_content": "李四说下周要去上海开会"},
       metadata={"timestamp": "2024-01-15T10:05:00"},
   )
   system.add(unit2)

   report = system.build_high_level(mode="auto")
   print(f"已处理 {report.sessions_processed} 个会话，{report.units_processed} 条记忆")

   hits = system.holistic_retrieve("张三去了哪里？", top_k=5)
   for hit in hits:
       print(f"[{hit.final_score:.3f}] {hit.unit.raw_data['text_content']}")

   system.save("./memory_snapshot")
   system2 = MemorySystem.load("./memory_snapshot")

模式二：本地模型（无需 API Key）
--------------------------------

此模式使用本地 Sentence-Transformers 模型进行 Embedding 和 Reranker，无需任何 API Key。需要安装 ``sentence-transformers`` 可选依赖。

.. code-block:: bash

   pip install mandol[sentence-transformers]

.. code-block:: python

   from mandol import MemorySystem, MemoryUnit, Uid

   system = MemorySystem()

   unit = MemoryUnit(
       uid=Uid("dialogue_001"),
       raw_data={"text_content": "张三今天去北京出差了"},
       metadata={"timestamp": "2024-01-15T10:00:00"},
   )
   system.add(unit)

   unit2 = MemoryUnit(
       uid=Uid("dialogue_002"),
       raw_data={"text_content": "李四说下周要去上海开会"},
       metadata={"timestamp": "2024-01-15T10:05:00"},
   )
   system.add(unit2)

   report = system.build_high_level(mode="auto")
   print(f"已处理 {report.sessions_processed} 个会话，{report.units_processed} 条记忆")

   hits = system.holistic_retrieve("张三去了哪里？", top_k=5)
   for hit in hits:
       print(f"[{hit.final_score:.3f}] {hit.unit.raw_data['text_content']}")

   system.save("./memory_snapshot")
   system2 = MemorySystem.load("./memory_snapshot")

.. note::

   本地模式首次运行时会自动下载模型文件（约 2-4 GB），请确保网络畅通和足够的磁盘空间。后续运行将使用缓存，无需重复下载。

add() 之后发生了什么？
----------------------

当你调用 ``system.add(unit)`` 时，Mandol 会自动执行以下操作：

1. **自动分块**：如果记忆单元的文本超过 ``chunk_max_tokens`` （默认 512），系统会自动将其拆分为多个小块
2. **自动向量化**：对每个记忆单元（或分块后的子单元）生成 Embedding 向量
3. **会话检测**：系统会异步检测会话边界，当累积的记忆单元达到 ``session_check_interval`` （默认 20）时触发检测
4. **相似度建边**：在新记忆与最近 ``similarity_recent_window`` （默认 20）条记忆之间，计算余弦相似度并建立 ``SEMANTIC_SIMILAR`` 边

build_high_level() 详解
-----------------------

``build_high_level()`` 是 Mandol 的核心构建方法，负责从原始对话记忆中提取高层语义结构。调用后系统会执行：

1. **会话分割**：将所有原始记忆按时间排序，通过 LLM 检测会话边界，将连续对话分割为独立会话
2. **摘要提取**：为每个会话生成情景摘要、知识摘要、情感摘要和程序摘要
3. **洞察提取**：从摘要中进一步提炼全局洞察
4. **实体提取与去重**：从对话中识别实体（人物、地点、概念等），并在跨会话间合并相同指代
5. **事件提取与去重**：从对话中识别事件，建立事件因果关系
6. **关系建图**：构建实体关系边（如 ``REL_WORKS_AT``）和事件因果边（如 ``CAUSES``、``CAUSED_BY``）

**何时调用**：

- 在添加完一批记忆单元后调用，例如一轮对话结束后
- ``mode="auto"``：仅处理未构建过的会话（增量模式，推荐）
- ``mode="force"``：清除所有高层记忆并重新构建（全量重建模式）

holistic_retrieve() 检索管线
----------------------------

``holistic_retrieve()`` 是 Mandol 的统一检索接口，内部执行以下流程：

1. **分组召回**：将检索请求分发到四个检索组：
   - **BASE**：原始对话记忆
   - **ENTITY**：知识实体
   - **EVENT**：情景事件
   - **SUMMARY**：摘要与洞察
2. **三路召回**：每组内部独立执行 Dense（稠密向量）、BM25（关键词）、Sparse（稀疏向量）三路检索
3. **RRF 融合**：使用倒数排名融合（Reciprocal Rank Fusion）合并三路结果
4. **BFS 扩展**：基于语义图关系扩展候选集（通过 ``bfs_expansion_per_seed`` 和 ``bfs_expansion_hops`` 控制）
5. **全局重排**：所有组合并后，通过 Cross-Encoder Reranker 重排序，返回最终结果

你也可以使用更细粒度的检索接口：

- ``retrieve_by_view(query, view="entity_relation")``：按视角类别检索
- ``retrieve_in_space(query, space_name="root_knowledge_entity")``：在指定空间内检索

下一步
------

- 阅读 :doc:`configuration` 了解详细配置选项
- 阅读 :doc:`../data_structures` 了解核心数据结构
- 阅读 :doc:`../retrieval/index` 了解检索接口细节
