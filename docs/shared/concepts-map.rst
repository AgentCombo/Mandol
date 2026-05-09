概念关系图
============

Mandol 的核心概念之间如何关联？下图给出全局视图。

.. mermaid::

   graph TB
       U[用户] -->|创建| MU[MemoryUnit 记忆单元]
       MU -->|存储在| SMAP[SemanticMap 语义索引]
       SMAP -->|组织到| MS[MemorySpace 记忆空间]
       MS -->|层级嵌套| MS2[MemorySpace 子空间]

       U -->|调用| ADD[add 添加]
       ADD -->|写入| SMAP
       ADD -->|向量化| EMB[EmbeddingProvider]
       EMB -->|写入| VI[VectorIndex 向量索引]

       U -->|调用| BHL[build_high_level 构建高阶记忆]
       BHL -->|触发| SM[SessionManager 会话管理]
       SM -->|分割| SESS[Session 会话]
       SESS -->|驱动| MDG[MultiDimSemanticGraph 多维度构建]
       MDG -->|提取| ENT[Entity 实体]
       MDG -->|提取| EVT[Event 事件]
       MDG -->|生成| SUMM[Summary 摘要]
       MDG -->|建立| REL[Relationship 关系]

       ENT -->|写入| SGPH[SemanticGraph 语义图]
       EVT -->|写入| SGPH
       SUMM -->|写入| SMAP
       REL -->|写入| SGPH

       U -->|调用| HR[holistic_retrieve 全记忆检索]
       HR -->|召回| DENSE[Dense 稠密检索]
       HR -->|召回| BM25[BM25 关键词检索]
       HR -->|召回| SPARSE[Sparse 稀疏检索]
       DENSE -->|融合| RRF[RRF 融合]
       BM25 -->|融合| RRF
       SPARSE -->|融合| RRF
       RRF -->|扩展| BFS[BFS 图扩展]
       SGPH --> BFS
       BFS -->|精排| RR[Reranker 重排]
       RR -->|返回| HIT[SearchHit 检索命中]

   style MU fill:#e1f5fe
   style SMAP fill:#e1f5fe
   style MS fill:#e1f5fe
   style SGPH fill:#e8f5e9
   style HR fill:#fff3e0
   style HIT fill:#fce4ec

核心概念一句话解释
------------------

- **MemoryUnit**：你要记住的最小事，一条消息 / 一个知识点 / 一个事件
- **MemorySpace**：分类盒，把记忆按主题或层次分组
- **SemanticMap**：档案室管理员，管存放和查找
- **SemanticGraph**：关系网络，记录「谁和谁有什么关系」
- **SessionManager**：书记员，判断什么时候开始新话题
- **build_high_level**：消化指令，把原始笔记变成结构化卡片
- **holistic_retrieve**：提问指令，系统自动找到最相关的内容

数据流简图
----------

::

   原始对话 → add() → 向量化 + 索引
                              ↓
              build_high_level()
                              ↓
        会话分割 → 实体 / 事件 / 摘要 / 关系
                              ↓
              holistic_retrieve()
                              ↓
      三路召回 → RRF 融合 → BFS 扩展 → Rerank
                              ↓
                        SearchHit[]
