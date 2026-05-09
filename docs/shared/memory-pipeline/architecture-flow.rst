开发者深度：架构级流程与组件关系
=======================================

本节面向需要理解内部架构的开发者和贡献者，展示六边形架构各层在记忆处理流程中的职责和交互。

六边形架构分层
--------------

.. mermaid::

   graph TB
       subgraph Domain["domain/ 领域层"]
           MU[MemoryUnit]
           MS[MemorySpace]
           TY[Uid / SpaceName / Embedding]
       end

       subgraph Ports["ports/ 端口层"]
           EMB[EmbeddingProvider]
           LLM[LLMProvider]
           RR[Reranker]
           VI[VectorIndex]
           GS[GraphStore]
           US[UnitStore]
       end

       subgraph App["application/ 应用层"]
           MSYS[MemorySystem]
           SMAP[SemanticMapService]
           SGPH[SemanticGraphService]
           SMGR[SessionManager]
           MDG[MultiDimSemanticGraph]
       end

       subgraph Infra["infrastructure/ 基础设施层"]
           FA[FAISS]
           BM[BM25]
           TF[TF-IDF]
           ST[SentenceTransformers]
           OA[OpenAI]
           IMS[InMemoryStore]
       end

       MSYS --> SMAP
       MSYS --> SGPH
       MSYS --> SMGR
       SMAP --> EMB
       SMAP --> VI
       SMAP --> US
       SGPH --> GS
       SGPH --> SMAP
       MDG --> LLM
       MDG --> EMB

       ST -.-> EMB
       OA -.-> EMB
       OA -.-> LLM
       OA -.-> RR
       FA -.-> VI
       BM -.-> VI
       TF -.-> VI
       IMS -.-> US
       IMS -.-> GS

组件关系与数据流
----------------

.. mermaid::

   sequenceDiagram
       participant U as 调用方
       participant MSYS as MemorySystem
       participant SMAP as SemanticMapService
       participant SGPH as SemanticGraphService
       participant SMGR as SessionManager
       participant MDG as MultiDimSemanticGraph
       participant EMB as EmbeddingProvider
       participant LLM as LLMProvider
       participant RER as Reranker
       participant VI as VectorIndex
       participant GS as GraphStore
       participant US as UnitStore
       participant INFRA as Infrastructure

       U->>MSYS: add(unit)

       MSYS->>SMAP: add_unit(unit)
       SMAP->>EMB: embed_text(text)
       EMB->>INFRA: 调用本地/远程模型
       INFRA-->>EMB: Embedding 向量
       EMB-->>SMAP: embedding
       SMAP->>US: upsert_units([unit])
       SMAP->>VI: upsert([(uid, embedding)])

       U->>MSYS: build_high_level(mode)

       MSYS->>SMGR: 获取未处理会话
       SMGR->>LLM: 会话边界检测
       LLM->>INFRA: 调用 LLM API
       INFRA-->>LLM: 检测结果
       LLM-->>SMGR: 会话边界

       SMGR->>MDG: build_session(session)
       MDG->>LLM: 提取摘要 / 实体 / 事件 / 关系
       LLM->>INFRA: 调用 LLM API
       INFRA-->>LLM: 提取结果
       LLM-->>MDG: 结构化结果

       MDG->>SMAP: create_space / add_unit
       SMAP->>US: 写入存储
       SMAP->>VI: 写入索引

       MDG->>SGPH: add_relationship
       SGPH->>GS: upsert_relationship

       U->>MSYS: holistic_retrieve(query)

       MSYS->>SMAP: search_by_text_with_rerank
       SMAP->>EMB: embed_text(query)
       SMAP->>VI: search(embedding, top_k)

       MSYS->>SGPH: bfs_expand_units(seeds)
       SGPH->>GS: get_neighbors(uid)

       MSYS->>MSYS: RRF 融合所有结果
       MSYS->>RER: rerank(query, candidates)
       RER->>INFRA: 调用 Cross-Encoder
       INFRA-->>RER: 重排分数
       RER-->>MSYS: 最终排序结果

       MSYS-->>U: SearchHit 列表

端口接口一览
------------

每个端口定义了一个可在 ``infrastructure/`` 中替换的抽象。以下为各端口的核心方法签名：

**EmbeddingProvider**
   - ``embed_text(texts: list[str]) -> list[Embedding]``
   - ``embed_image_paths(paths: list[str]) -> list[Embedding]``
   - ``embedding_dim() -> int``

**LLMProvider**
   - ``generate(prompt: str, **kwargs) -> str``
   - ``generate_structured(prompt: str, schema: dict, **kwargs) -> dict``

**Reranker**
   - ``rerank(query: str, units: list[MemoryUnit], top_k: int) -> list[tuple[MemoryUnit, float]]``

**VectorIndex**
   - ``upsert(items: list[tuple[Uid, Embedding]]) -> None``
   - ``search(query: Embedding, top_k: int) -> list[tuple[Uid, float]]``
   - ``delete(uids: list[Uid]) -> None``
   - ``rebuild(items: list[tuple[Uid, Embedding]]) -> None``
   - ``dim() -> int``

**UnitStore**
   - ``upsert_units(units: list[MemoryUnit]) -> None``
   - ``delete_units(uids: list[Uid]) -> None``
   - ``get_unit(uid: Uid) -> MemoryUnit | None``
   - ``list_units() -> list[MemoryUnit]``
   - ``get_units(uids: list[Uid]) -> list[MemoryUnit]``
   - ``upsert_spaces(spaces: list[MemorySpace]) -> None``
   - ``get_space(name: SpaceName) -> MemorySpace | None``
   - ``list_spaces() -> list[MemorySpace]``
   - ``flush() -> None``

**GraphStore**
   - ``upsert_relationship(source: Uid, target: Uid, rel_type: str, properties: dict) -> None``
   - ``delete_relationship(source: Uid, target: Uid, rel_type: str | None) -> None``
   - ``get_relationship(source: Uid, target: Uid, rel_type: str) -> dict | None``
   - ``get_neighbors(uid: Uid, rel_type: str | None, direction: str) -> list[Uid]``
   - ``flush() -> None``
