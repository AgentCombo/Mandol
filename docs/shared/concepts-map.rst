概念关系图
============

Mandol 的核心概念之间如何关联？下图给出全局视图。

.. _diagram-placeholder-concepts-1:

.. image:: /_static/images/core-concepts-map.png
    :alt: 核心概念关系全景图
    :align: center

核心概念一句话解释
------------------

- **MemoryUnit**：最小记忆载体，一条消息 / 一个知识点 / 一个事件
- **MemorySpace**：逻辑容器，按主题或层次组织记忆单元
- **SemanticMap**：存储与检索服务，管理单元的存放和查找
- **SemanticGraph**：关系图，记录记忆单元间的关系
- **SessionManager**：会话管理器，检测话题边界并分割会话
- **DocumentChunker**：文档分块器，将过长记忆按句子边界切分
- **SummaryMapReducer**：摘要 Map-Reduce 处理器，分块提取四种摘要后归约合并
- **UnifiedFactPipeline**：统一事实管线，从对话中提取实体、事件及其关系
- **CrossSessionCorefManager**：跨会话共指消解，合并不同 Session 中相同指代的实体/事件
- **GlobalInsightManager**：全局洞察管理器，跨 Session 累积合并深层洞察
- **build_high_level**：处理指令，将原始数据构建为结构化记忆
- **holistic_retrieve**：检索指令，从全记忆中返回最相关的内容

高阶记忆构建流程简图
--------------------

::

   原始对话 → add()
       │
       ├── 1. 分块（过长文本按句子边界切分）
       ├── 2. 向量化 + 存储 + 相似度建边
       └── 3. 进入待处理队列
              ↓
       build_high_level()
       │
       ├── 4. 会话分割（LLM 语义分析，检测话题边界）
       ├── 5. 空间创建（每 Session 独立空间，全局空间层级幂等创建）
       ├── 6. 四类摘要 Map-Reduce（情景/知识/情感/程序）
       ├── 7. 实体/事件/关系统一提取 + 跨会话共指消解
       ├── 8. Session 洞察提炼
       └── 9. 全局洞察增量合并
              ↓
       holistic_retrieve()
       │
       ├── 10. 四组三路召回
       ├── 11. RRF 融合 + BFS 图扩展
       └── 12. Cross-Encoder 重排 → SearchHit[]
