数据流详解
==========

从原始文本到检索结果的完整数据流向。

阶段一：写入 (Write Path)
--------------------------

.. mermaid::

   sequenceDiagram
       participant U as 调用方
       participant MSYS as MemorySystem
       participant SMAP as SemanticMapService
       participant EMB as EmbeddingProvider
       participant US as UnitStore
       participant VI as VectorIndex

       U->>MSYS: add(unit)
       MSYS->>MSYS: UnitPipeline.preprocess(unit)
       Note over MSYS: 分块 + 命名规范化
       MSYS->>SMAP: add_unit(unit)
       SMAP->>EMB: embed_text(text)
       EMB-->>SMAP: embedding
       SMAP->>US: upsert_units([unit])
       SMAP->>VI: upsert([(uid, embedding)])
       SMAP-->>MSYS: ok
       Note over MSYS: 相似度建边 + 送入 SessionManager 队列

阶段二：构建 (Build Path)
--------------------------

.. mermaid::

   sequenceDiagram
       participant U as 调用方
       participant MSYS as MemorySystem
       participant SMGR as SessionManager
       participant LLM as LLMProvider
       participant MDG as MultiDimSemanticGraph

       U->>MSYS: build_high_level(mode)
       MSYS->>SMGR: process_pending_sessions()
       SMGR->>LLM: 会话边界检测
       LLM-->>SMGR: 分割结果
       SMGR-->>MSYS: sessions

       loop 每个 session
           MSYS->>MDG: build_session(session)
           MDG->>LLM: 提取摘要/实体/事件/关系
           LLM-->>MDG: 结构化结果
           MDG-->>MSYS: 构建结果
       end

       MSYS->>MSYS: 跨会话实体/事件合并
       MSYS-->>U: BuildReport

阶段三：检索 (Read Path)
-------------------------

.. mermaid::

   sequenceDiagram
       participant U as 调用方
       participant MSYS as MemorySystem
       participant HR as HybridRetriever
       participant RER as Reranker

       U->>MSYS: holistic_retrieve(query)
       MSYS->>HR: retrieve(query)

       par 四组并行
           HR->>HR: BASE组: Dense+BM25+Sparse→RRF
           HR->>HR: ENTITY组: Dense+BM25+Sparse→RRF
           HR->>HR: EVENT组: Dense+BM25+Sparse→RRF
           HR->>HR: SUMMARY组: Dense+BM25+Sparse→RRF
       end

       HR->>HR: 各组 BFS 扩展
       HR->>HR: 全局候选合并
       HR->>RER: rerank(query, candidates)
       RER-->>HR: 重排结果
       HR-->>MSYS: 最终结果
       MSYS-->>U: SearchHit[]
