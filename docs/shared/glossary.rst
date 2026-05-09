统一术语表
============

本文档建立 Mandol 中核心术语的中英文对照及代码映射，确保文档和对话中的术语一致性。

核心数据结构
------------

.. list-table::
   :header-rows: 1
   :widths: 22 28 50

   * - 中文
     - 英文 / 代码标识
     - 说明
   * - 记忆单元
     - MemoryUnit
     - 系统中最小的记忆载体，封装一段对话、一个实体或一个事件
   * - 记忆空间
     - MemorySpace
     - 记忆单元的逻辑容器，支持层级嵌套
   * - 语义索引
     - SemanticMap / SemanticMapService
     - 负责记忆单元的 CRUD、向量索引和空间管理
   * - 语义图
     - SemanticGraph / SemanticGraphService
     - 负责记忆单元间的关系建模和图遍历
   * - 唯一标识
     - Uid
     - 每个记忆单元的唯一 ID

检索相关
--------

.. list-table::
   :header-rows: 1
   :widths: 22 28 50

   * - 中文
     - 英文 / 代码标识
     - 说明
   * - 全记忆检索
     - holistic_retrieve
     - 系统统一检索入口，自动协调多组多路检索
   * - 按视图检索
     - retrieve_by_view
     - 按预定义语义视图过滤检索结果
   * - 空间内检索
     - retrieve_in_space
     - 在指定记忆空间内检索
   * - 检索命中
     - SearchHit
     - 单次检索的结果单元，携带 final_score / scores / ranks
   * - 倒数排名融合
     - RRF (Reciprocal Rank Fusion)
     - 融合多个检索器结果的无参数算法
   * - BFS 图扩展
     - bfs_expand_units
     - 以检索结果为种子沿图关系扩展候选集
   * - 重排序
     - Rerank (Cross-Encoder)
     - 对融合后的候选用 Cross-Encoder 模型精排

高阶记忆类型
------------

.. list-table::
   :header-rows: 1
   :widths: 22 28 50

   * - 中文
     - 英文 / 代码标识
     - 说明
   * - 基础记忆
     - Base Memory
     - 原始对话数据，未经高层加工
   * - 情景记忆
     - Episodic Memory
     - 事件及情景摘要，记录"发生了什么"
   * - 知识记忆
     - Knowledge Memory
     - 实体及知识摘要，记录"知道什么"
   * - 情感记忆
     - Emotional Memory
     - 用户情感状态和偏好总结
   * - 程序记忆
     - Procedural Memory
     - 操作步骤和流程总结
   * - 洞察记忆
     - Insights
     - 从多视角总结中提炼的深层模式识别

图关系类型
----------

.. list-table::
   :header-rows: 1
   :widths: 22 28 50

   * - 中文
     - 英文 / 代码标识
     - 说明
   * - 时序前驱
     - PRECEDES
     - 对话/事件的时间先后关系（前 → 后）
   * - 时序后继
     - FOLLOWS
     - 对话/事件的时间先后关系（后 → 前）
   * - 语义相似
     - SEMANTIC_SIMILAR
     - 基于向量余弦相似度的语义关联
   * - 相关
     - RELATED_TO
     - 实体间通用关系（含 located_in / works_at 等子类型）
   * - 共指
     - COREF
     - 对话单元 → 全局实体的指代关系
   * - 导致
     - CAUSES
     - 事件因果关系（A 导致 B）
   * - 被导致
     - CAUSED_BY
     - 事件因果关系（B 被 A 导致）
   * - 涉及
     - INVOLVES
     - 事件与其参与者/位置的关系
   * - 证据支撑
     - EVIDENCED_BY
     - 高阶记忆指向支撑它的原始数据

架构相关
--------

.. list-table::
   :header-rows: 1
   :widths: 22 28 50

   * - 中文
     - 英文 / 代码标识
     - 说明
   * - 端口
     - Port
     - 抽象接口定义，不依赖具体实现
   * - 适配器 / 基础设施
     - Adapter / Infrastructure
     - 端口的具体实现（如 FAISS、OpenAI API）
   * - 会话管理器
     - SessionManager
     - 负责会话分割、合并和生命周期管理
   * - 维度构建器
     - DimensionBuilder
     - 负责一类记忆的提取和关系建模
   * - 嵌入提供者
     - EmbeddingProvider
     - 文本/图像向量化的抽象接口
   * - 重排器
     - Reranker
     - 检索结果精排的抽象接口
