记忆建立与检索全流程
======================

Mandol 的核心骨干流程只有三步：**添加记忆 → 构建高阶语义 → 检索**。无论你是哪个层次的用户，都需要理解这个流程。本章节提供三个深度的流程说明，你可以根据需要选择阅读。

.. toctree::
   :maxdepth: 1

   basic-flow
   detailed-flow
   architecture-flow

流程全景图
----------

.. mermaid::

   graph LR
       A[MemoryUnit] -->|add| B[向量化 + 索引]
       B -->|build_high_level| C[会话分割]
       C --> D[多维度构建]
       D --> E[实体 / 事件 / 摘要 / 关系]
       E -->|holistic_retrieve| F[多路召回]
       F --> G[RRF 融合 + BFS 扩展 + Rerank]
       G --> H[SearchHit 结果]
