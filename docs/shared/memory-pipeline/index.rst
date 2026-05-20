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

::

   add()                       build_high_level()              holistic_retrieve()
   ─────                       ──────────────────              ───────────────────
   原始对话                    会话分割                          四组召回
     ↓                           ↓                               (BASE/ENTITY
   分块（超长文本切分）         空间创建                           /EVENT/SUMMARY)
     ↓                           ↓                                  ↓
   向量化 + 存储                四类摘要 Map-Reduce              三路检索
     ↓                           (情景/知识/情感/程序)            (Dense/BM25/Sparse)
   相似度建边                      ↓                               ↓
     ↓                         实体/事件/关系提取                RRF 融合
   待处理队列                     ↓                               ↓
                               洞察提炼                          BFS 图扩展
                                  ↓                               ↓
                               跨会话合并                        Rerank 重排
                                  ↓                               ↓
                               全局洞察累积                      SearchHit[]

.. note::

   - **基础用户**：阅读 :doc:`basic-flow`，了解三步流程和 ``build_high_level()`` 内部做了什么
   - **高级用户**：阅读 :doc:`detailed-flow`，理解每个子阶段的完整机制和可调参数
   - **开发者**：阅读 :doc:`architecture-flow`，了解架构分层和每个阶段的扩展/定制方式
