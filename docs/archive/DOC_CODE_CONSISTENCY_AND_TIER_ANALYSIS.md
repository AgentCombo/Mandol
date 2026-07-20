# Mandol 文档-代码一致性分析与三层次用户友好度评估

> 生成日期：2026-05-06
> 目的：为开源前的文档完善提供依据

---

## 第一部分：文档与代码一致性分析

### 1.1 README_CN.md 一致性审计

README_CN.md 已经在 DOC_OPTIMIZATION_PLAN.md 的指导下进行了大幅改进，整体与代码一致性较好。以下是逐项核对结果：

| README 内容 | 代码实际情况 | 一致性 | 备注 |
|------------|-------------|--------|------|
| `pip install mandol` | ✅ setup.py/pyproject.toml 存在 | ✅ 一致 | |
| `MemorySystem.from_yaml_config("config.yaml")` | ✅ `memory_system.py:L356` 存在 | ✅ 一致 | |
| `MemoryUnit(uid=Uid("msg_001"), raw_data={...})` | ✅ `memory_unit.py:L17` 构造函数匹配 | ✅ 一致 | |
| `system.add(unit)` | ✅ `memory_system.py:L743` 存在 | ✅ 一致 | |
| `system.build_high_level(mode="auto")` | ✅ `memory_system.py:L943` 存在 | ✅ 一致 | |
| `system.holistic_retrieve("query", top_k=5)` | ✅ `memory_system.py:L1111` 存在 | ✅ 一致 | |
| `system.save("./memory_snapshot")` | ✅ `memory_system.py:L1302` 存在 | ✅ 一致 | |
| `MemorySystem.load("./memory_snapshot")` | ✅ `memory_system.py:L1322` 类方法存在 | ✅ 一致 | |
| config.yaml 中 `embedder.use_remote` | ✅ `config.py` 中 `EmbedderConfig.use_remote` 存在 | ✅ 一致 | |
| config.yaml 中 `reranker.use_remote` | ✅ `config.py` 中 `RerankerConfig.use_remote` 存在 | ✅ 一致 | |
| 环境变量 `MANDOL_EMBEDDER_MODEL` | ⚠️ 代码中通过 `MemorySystemConfig.embedder_model` 使用，非直接读环境变量 | ⚠️ 部分一致 | 环境变量需通过 config 或代码显式读取，并非 zero-config |
| 环境变量 `USE_REMOTE_EMBEDDER` | ⚠️ 代码中为 `MemorySystemConfig.use_remote_embedder`，非直接读环境变量 | ⚠️ 部分一致 | 同上 |
| 检索流程「三路召回 → RRF 融合 → BFS 扩展 → 重排序」 | ✅ `_retrieval.py` + `pipeline.py` 实现一致 | ✅ 一致 | |
| 自动构建流程描述 | ✅ `memory_system.py:_build_session_for_units` 实现一致 | ✅ 一致 | |
| FAQ 中 `use_rerank=False` 参数 | ✅ `holistic_retrieve` 签名中 `use_rerank: bool = True` | ✅ 一致 | |

**README 中发现的潜在问题**：

1. **环境变量表**：表格中列举的 `MANDOL_EMBEDDER_MODEL`、`USE_REMOTE_EMBEDDER` 等环境变量，代码中并非直接通过 `os.environ` 读取，而是通过 `MemorySystemConfig` dataclass 或 YAML 配置传入。基础用户若直接设置环境变量期望生效，可能会困惑。建议在表格中明确说明"需配合 YAML config 或 `MemorySystemConfig` 使用"。

2. **`MemorySpace` 的类比**：README 将 MemorySpace 类比为"文件柜的抽屉"，将 SemanticMap 类比为"图书馆的检索卡片"。从代码实现来看，`MemorySpace` 实际上是纯粹的树形容器（记录 unit_uids + child_spaces），而 `SemanticMapService` 才负责索引与检索。这个类比基本准确，但 SemanticMap 的"索引系统"不仅是检索卡片，还包含了向量索引（AdaptiveVectorIndex）、存储（UnitStore）等更复杂的能力。

3. **快速开始示例缺少 `build_high_level` 调用**：README 远程 API 模式示例中直接 `system.add(unit)` 后调用 `holistic_retrieve`，但实际检索可能需要在 `build_high_level()` 之后才能检索到高阶记忆内容。代码中 `add()` 方法只做基础存储和异步会话检测，BASE 组的检索可以直接工作，但 ENTITY/EVENT/SUMMARY 组的检索需要 `build_high_level` 完成后才有数据。当前示例只插入一条数据，BASE 组检索能工作，但容易让用户误以为不需要调用 `build_high_level`。

---

### 1.2 docs/ Sphinx 文档一致性审计

#### 1.2.1 `data_structures.rst` — 多处严重不一致

| 文档描述 | 实际代码 | 问题严重度 |
|---------|---------|-----------|
| `SemanticMap` 有属性 `ms_space`、`faiss_index`、`uid_to_index` | 实际类名是 `SemanticMapService`，无 `ms_space` 属性，使用 `AdaptiveVectorIndex`（非直接 `faiss_index`） | 🔴 严重 |
| `SemanticMap` 有方法 `add_to_faiss(uids, embeddings)` | 实际不存在此方法，向量索引用 `_index.upsert()` 内部管理 | 🔴 严重 |
| `SemanticMap` 有方法 `get_faiss_index()` | 实际无此方法 | 🔴 严重 |
| `SemanticGraph` 有属性 `explicit_graph`、`implicit_graph` | 实际是 `SemanticGraphService`，图存储委托给 `InMemoryGraphStore`，无独立的 explicit/implicit 分拆 | 🔴 严重 |
| `SemanticGraph` 有方法 `build_all_implicit_edges(similarity_threshold)` | 实际隐式边通过 `_build_immediate_similarity_edges` 和 `_build_similarity_edges_for_units` 增量构建 | 🔴 严重 |
| 构造示例 `SemanticMap(space=space, config=SemanticMapConfig(...))` | 实际构造函数为 `SemanticMapService(*, store=, index=, embedder=, reranker=)` | 🔴 严重 |
| 类型路径 `src/memory/domain/types.py` | 实际路径为 `mandol/domain/types.py` | 🟡 中等 |
| MemoryUnit 构造示例 `uid="dialogue_001"` (字符串) | 实际要求 `uid=Uid(...)` 类型 | 🟡 中等 |

**原因分析**：`data_structures.rst` 描述的是早期的设计版本或重构前的 API，与当前 `SemanticMapService` / `SemanticGraphService` 的实现严重脱节。这会导致开发者参考文档后写出无法运行的代码。

#### 1.2.2 `retrieval_interfaces.rst` — 接口与实现部分对齐

| 文档描述的接口 | 代码中是否存在 | 一致度 |
|--------------|-------------|--------|
| `get_unit(uid)` | ✅ `SemanticMapService.get_unit` (L164) | ✅ |
| `get_all_units()` | ❌ 实际方法名为 `list_units()` (L167) | 🔴 方法名不一致 |
| `filter_memory_units(candidate_units, filter_condition, ms_names, recursive)` | ❌ 不存在此方法 | 🔴 预想接口未标注 |
| `search(query, k, retriever_type, retrievers, ms_names, candidate_uids)` | ❌ 不存在此签名的统一方法，实际分散在 `search_by_text` / `search_by_vector` / `search_by_text_with_rerank` | 🔴 严重 |
| `search_similarity_by_text(query_text, k, ms_names)` | ❌ 实际为 `search_by_text` (L287) | 🔴 方法名不一致 |
| `search_similarity_by_vector(query_embedding, k, ms_names)` | ❌ 实际为 `search_by_vector` (L261) | 🔴 方法名不一致 |
| `search_hybrid(query, top_k, ms_names, use_graph_expansion, bfs_depth, rerank)` | ⚠️ 类似功能在 `HybridRetriever.search()` 中实现 | 🟡 部分一致 |
| `get_explicit_neighbors`, `get_implicit_neighbors` | ✅ `SemanticGraphService` (L81, L101) | ✅ |
| `bfs_expand_units` | ✅ 以 `_bfs_expand_units` 在内部使用 | 🟡 方法可见性不同 |
| 接口命名约定「公开接口无前缀，内部接口下划线前缀」 | ✅ 代码遵循此约定 | ✅ |

#### 1.2.3 `multidim_construction.rst` — 架构描述与实现对齐较好

该文档描述空间命名策略 (`SpaceNamingPolicy`) 和多维度构建器 (`DimensionBuilder`) 的接口设计，与代码中 `multidim_semantic_graph.py` 的实现一致。文档中包含的构建流程图准确反映了系统内部的空间组织方式。

不过，该文档面向的是开发者/高级用户，对基础用户过于深入。

#### 1.2.4 `extending.rst` — 扩展示例可运行性待验证

文档中列出的扩展方式（自定义 Embedding Provider、Graph Store、Dimension Builder 等）基于 ports 接口设计，架构描述正确。但示例代码使用的是早期 API（如 `get_all_units()`），需要更新以匹配当前实现。

#### 1.2.5 `docs/en/` 英文文档 — 严重不完整

英文文档目录 `docs/en/` 下仅有 `index.rst` 有实际内容，其余 5 个文件（`introduction.rst`、`data_structures.rst`、`multidim_construction.rst`、`retrieval_interfaces.rst`、`extending.rst`）内容量与中文版相比缺失严重。

---

### 1.3 一致性总结

| 文档 | 与代码一致度 | 主要问题 |
|------|------------|---------|
| README_CN.md | 🟢 良好（~85%） | 环境变量生效路径不够明确；快速开始示例可能误导可不用 build_high_level |
| README.md (EN) | 🟡 待验证 | 需确认与中文版同步 |
| data_structures.rst | 🔴 严重脱节（~30%） | 类名、方法名、属性、构造参数全部过时 |
| retrieval_interfaces.rst | 🟡 部分不一致（~50%） | 方法名对不上，存在未标注的预想接口 |
| multidim_construction.rst | 🟢 良好（~80%） | 架构层描述准确 |
| extending.rst | 🟡 部分不一致（~55%） | 示例代码 API 过时 |
| docs/en/ (全部) | 🔴 严重不完整 | 英文内容大幅落后于中文版 |
| api-reference/*.md | 🟡 中等（~60%） | 覆盖4个核心类但 API 细节有遗漏 |

---

## 第二部分：关键术语翻译讨论

### 2.1 SemanticMap 和 SemanticGraph 的翻译

结合代码实际逻辑，我对这两个核心术语的翻译做了深入分析：

#### SemanticMap

**代码本质**：`SemanticMapService`（[semantic_map.py](../../mandol/application/semantic_map.py)）的核心职责是：
- 管理 `MemoryUnit` 的存储（通过 `UnitStore`）
- 管理向量索引（通过 `AdaptiveVectorIndex`）
- 管理 `MemorySpace` 树形组织
- 提供语义检索（`search_by_text`、`search_by_vector`、`search_by_text_with_rerank`）

它本质上是一个 **"带向量索引的记忆存储与检索引擎"**。"Map" 在这里的含义更接近计算机科学中的 "映射"（key → value 的索引结构），而不是地理上的"地图"。

| 候选翻译 | 优点 | 缺点 |
|---------|------|------|
| 语义地图 | 直译，简短 | 暗示二维空间布局，与向量索引本质不匹配 |
| **语义索引** | 准确反映"索引+检索"核心功能 | 丢失了 "Map" 的空间组织暗示 |
| 语义映射表 | 体现 key→value 的映射关系 | 太长，偏数据库术语 |
| 语义库 | 体现"存储库"概念 | 丢失了"Map"含义 |
| **保留 SemanticMap** | 零歧义，开发者友好 | 基础用户需要额外学习 |

**我的建议**：
- **面向基础用户**：使用「**语义索引 (SemanticMap)**」— 首次出现时括号标注英文，后续统一用 SemanticMap。这样既能让基础用户直观理解"这是个索引/检索系统"，又不丢失技术精确性。
- **面向开发者**：直接使用 **SemanticMap**，因为开发者需要与代码中类名对应。

#### SemanticGraph

**代码本质**：`SemanticGraphService`（[semantic_graph.py](../../mandol/application/semantic_graph.py)）的核心职责是：
- 管理节点（MemoryUnit）之间的显式关系（RELATED_TO、CAUSES、EVIDENCED_BY、COREF 等）
- 管理隐式语义相似度边（SEMANTIC_SIMILAR）
- BFS 图扩展检索
- 图的持久化

它本质上是一个 **"记忆关系网络/知识图谱"**，中文"图"（Graph Theory 中的 Graph）的翻译是准确的。

| 候选翻译 | 优点 | 缺点 |
|---------|------|------|
| 语义图 | 简洁，直译 | "图" 在中文中有多重含义（图片/图表/图谱） |
| **语义关系图** | 强调节点间的关系网络 | 稍长 |
| 语义图谱 | 类似"知识图谱"的表述 | 暗示大规模知识库，不完全匹配 |
| 语义网络 | 体现网络结构 | 与 Semantic Web 术语冲突 |
| **保留 SemanticGraph** | 零歧义 | 基础用户需要学习 |

**我的建议**：
- **面向基础用户**：使用「**语义关系图 (SemanticGraph)**」— "图"指图论中的 Graph，但加上"关系"二字帮助普通用户理解。
- **面向开发者**：直接使用 **SemanticGraph**。

### 2.2 其他关键术语翻译建议

| 英文术语 | 当前翻译 | 建议翻译 | 理由 |
|---------|---------|---------|------|
| MemoryUnit | 记忆单元 | **记忆单元 (MemoryUnit)** ✅ | 翻译准确，保持 |
| MemorySpace | 记忆空间 | **记忆空间 (MemorySpace)** ✅ | 翻译准确，保持 |
| Holistic Retrieve | 全记忆检索 | **全记忆检索 (Holistic Retrieve)** ✅ | 翻译准确，保持 |
| Session | 会话 | **会话 (Session)** ✅ | 翻译准确，保持 |
| Entity | 实体 | **实体 (Entity)** ✅ | 翻译准确，保持 |
| Event | 事件 | **事件 (Event)** ✅ | 翻译准确，保持 |
| High-Level Memory | 高阶记忆 | **高阶记忆 (High-Level Memory)** ✅ | 翻译准确，保持 |
| RRF (Reciprocal Rank Fusion) | RRF 融合 | **RRF 倒数排名融合** | 首次出现时展开全称 |
| BFS Expansion | BFS 扩展 | **BFS 图扩展** | 加"图"字更精确 |
| Cross-Encoder Reranker | 重排序器 | **Cross-Encoder 重排序器** | 保留技术名词 |
| Dense / BM25 / Sparse | 稠密/BM25/稀疏 | 一致，无需改变 | |
| Embedding | 向量化 | **向量化 (Embedding)** | 保持双语标注 |
| Chunk | 分块 | **分块 (Chunk)** | 翻译准确，保持 |

### 2.3 统一的术语策略

**推荐策略**：「首次中文全称 + 括号英文 + 后续统一用英文术语」

```
例：语义索引 (SemanticMap) 负责管理所有记忆的向量表示和高效检索。
SemanticMap 内部使用 AdaptiveVectorIndex 实现自适应索引切换。
```

这样做的优势：
1. 基础用户通过中文描述快速理解概念
2. 高级用户和开发者可以直接将术语映射到代码中的类名
3. 避免"翻译导致的歧义"（如 SemanticMap 译为"语义地图"让人以为是地理空间概念）
4. 与国际开源项目惯例一致（如 LangChain、LlamaIndex 等项目的文档策略）

---

## 第三部分：三层次用户友好度评估

### 3.1 基础用户（只会 `MemoryUnit` → `add()` → `build_high_level()` → `holistic_retrieve()`）

#### 当前状态评分：🟢 3.5/5（相比旧版有大幅提升）

| 评估维度 | 评分 | 说明 |
|---------|------|------|
| **安装上手** | 4/5 | 双模式示例（远程API/本地模型）+ 环境配置表，比旧版改进显著 |
| **核心概念** | 4/5 | 新增通俗解释 + 类比 + 术语表，已有大幅改善 |
| **快速开始** | 3/5 | 代码示例可运行，但缺少 `build_high_level()` 的明确说明；只有 1 条数据的示例太简单 |
| **常见问题** | 4/5 | FAQ 新增，覆盖安装/运行/性能三大类 |
| **结果理解** | 3/5 | `SearchHit` 结果对象的用法只在代码中出现，缺少文字说明 |

#### 已解决的问题（相比 DOC_OPTIMIZATION_PLAN.md v1.0）
- ✅ 新增了环境准备章节（Python版本/系统资源/模型下载说明）
- ✅ 新增了双模式快速开始（远程API / 本地模型）
- ✅ 新增了核心概念通俗解释
- ✅ 新增了 FAQ 章节

#### 仍存在的问题

1. **`MemoryUnit` 的字段约束不够明确**：基础用户需要知道 `raw_data` 中目前只支持纯文本（`text_content` 字段）和图片路径（`image_path` 字段），其他任意字段不会被自动 embedding。README 和快速开始文档中应明确这一点。

2. **缺少 3 个完整的 end-to-end 示例**：当前快速开始中只有 1 条记忆的插入和检索，不足以让基础用户理解为真正的对话场景服务。建议按 DOC_OPTIMIZATION_PLAN 的规划，放置 3 个场景示例（客服/个人助手/知识库）在基础用户章节。

3. **`build_high_level()` 是否必须调用？**：代码中 `add()` 会触发异步自动构建（`_check_session_boundary_async`），但基础用户可能在插入后立即调用 `holistic_retrieve` 时发现 ENTITY/EVENT/SUMMARY 组为空。文档应明确说明：BASE 组检索不需要 `build_high_level()`；而要检索高阶记忆（实体/事件/摘要），需要等待自动构建完成或手动调用 `build_high_level()`。

4. **`SearchHit` 用法不直观**：`for hit in hits: print(hit.unit.raw_data['text_content'])` 需要访问嵌套属性，基础用户可能不知道 `hit.final_score`、`hit.unit` 的用法。

---

### 3.2 高级用户（在基础用户之上，使用公开的管理/控制接口）

#### 当前状态评分：🟡 2.5/5

#### 高级用户应该能使用的接口（代码中已实现）

根据代码审计，以下接口已经是公开的，高级用户应该能使用：

**检索接口**（MemorySystem 公开方法）：

| 方法 | 代码位置 | 文档覆盖 | 用途 |
|------|---------|---------|------|
| `holistic_retrieve(query, top_k, use_rerank)` | `memory_system.py:L1111` | ✅ 有 | 全记忆统一检索 |
| `retrieve_in_space(query, space_name, top_k, use_rerank)` | `memory_system.py:L1138` | ✅ 有 | 按空间检索 |
| `retrieve_by_view(query, view, top_k, use_rerank)` | `memory_system.py:L1161` | ✅ 有 | 按多视角检索 |

**管理接口**（MemorySystem 公开方法）：

| 方法 | 代码位置 | 文档覆盖 | 用途 |
|------|---------|---------|------|
| `add(unit)` | `memory_system.py:L743` | ✅ 有 | 添加单条记忆 |
| `add_many(units)` | `memory_system.py:L782` | ✅ 有 | 批量添加 |
| `build_high_level(mode)` | `memory_system.py:L943` | ✅ 有 | 手动触发高阶构建 |
| `build_high_level_async(mode)` | `memory_system.py:L1046` | ⚠️ 提到但不详细 | 异步构建 |
| `merge_cross_session_entities()` | `memory_system.py:L1052` | ❌ 无 | 跨会话实体合并 |
| `merge_cross_session_events()` | `memory_system.py:L1080` | ❌ 无 | 跨会话事件合并 |
| `save(path)` / `load(path)` | `memory_system.py:L1302/L1322` | ✅ 有 | 持久化 |
| `flush()` | `memory_system.py:L1288` | ❌ 无 | 清空所有数据 |

**SemanticMap 公开接口**（通过 `system.semantic_map` 访问）：

| 方法 | 代码位置 | 文档覆盖 | 用途 |
|------|---------|---------|------|
| `create_space(name)` | `semantic_map.py:L71` | ❌ 无 | 创建自定义空间 |
| `get_space(name)` | `semantic_map.py:L81` | ❌ 无 | 获取空间 |
| `list_spaces()` | `semantic_map.py:L84` | ❌ 无 | 列出所有空间 |
| `add_unit_to_space(uid, space_name)` | `semantic_map.py:L170` | ❌ 无 | 将单元加入空间 |
| `attach_child_space(parent, child)` | `semantic_map.py:L188` | ❌ 无 | 建立空间层级 |
| `delete_unit(uid)` | `semantic_map.py:L149` | ❌ 无 | 删除记忆单元 |
| `search_by_text_with_rerank(...)` | `semantic_map.py:L302` | ❌ 无 | 带重排的语义搜索 |
| `rebuild_index_from_store()` | `semantic_map.py:L375` | ❌ 无 | 重建向量索引 |

**SemanticGraph 公开接口**（通过 `system.graph` 访问）：

| 方法 | 代码位置 | 文档覆盖 | 用途 |
|------|---------|---------|------|
| `add_relationship(src, tgt, rel, props)` | `semantic_graph.py:L49` | ✅ 有 | 手动添加关系 |
| `get_relationship(src, tgt, rel)` | `semantic_graph.py:L64` | ❌ 无 | 查询关系 |
| `delete_relationship(src, tgt, rel)` | `semantic_graph.py:L71` | ❌ 无 | 删除关系 |
| `get_explicit_neighbors(uids, rel_type, dir)` | `semantic_graph.py:L81` | ✅ 有 | 显式邻居查询 |
| `get_implicit_neighbors(uids, top_k)` | `semantic_graph.py:L101` | ✅ 有 | 隐式邻居查询 |
| `bfs_expand_units(seeds, per_seed, hops)` | `semantic_graph.py:L129` | ✅ 有 | BFS 图扩展 |
| `delete_unit(uid)` | `semantic_graph.py:L36` | ❌ 无 | 删除节点及相关边 |

#### 高级用户文档的不足

1. **空间管理完全没有文档**：`system.semantic_map.create_space()`、`attach_child_space()` 等空间管理接口在代码中存在但文档中完全没有提及。高级用户无法知道如何主动划分 MemorySpace。

2. **会话管理仅停留在参数层面**：文档中只提到 `session_time_gap_seconds` 参数，但 `SessionManager` 提供了 LLM 驱动的智能会话分割，高级用户可以通过 `system._session_manager` 了解和管理会话状态（虽然 `_session_manager` 名义上是内部属性，但高级用户场景下可以暴露为只读属性）。

3. **图结构操作文档过于简略**：`add_relationship` / `delete_relationship` 的文档只列出签名，没有说明关系类型的语义（什么时候应该添加什么类型的关系？），缺少"图修剪"、"图合并"等高级图管理操作。

4. **参数调优缺少场景关联**：`user-guide/parameter-tuning.rst` 列出了参数和调优建议，但没有将参数组合按场景打包推荐（如"客服场景推荐配置"、"个人助手推荐配置"）。

5. **Cross-Encoder Reranker 的工作原理未解释**：高级用户需要知道 use_rerank 的性能/质量权衡。

---

### 3.3 开发者（所有管理接口 + 源码修改 + 贡献流程）

#### 当前状态评分：🔴 1.5/5

#### 开发者的核心需求与现状差距

| 开发者需求 | 当前满足度 | 差距描述 |
|-----------|----------|---------|
| **完整 API Reference** | 🔴 严重不足 | 仅覆盖 4 个核心类（MemoryUnit/MemorySpace/SemanticMap/SemanticGraph），MemorySystem/SessionManager/检索管线/Config 等均无文档 |
| **架构设计决策 (ADR)** | 🔴 不存在 | 为什么选择六边形架构？为什么 `SemanticMapService` 不叫 `SemanticMap`？没有解释 |
| **贡献指南** | 🟡 存在但不完整 | 缺少代码规范(ruff配置已存在)、测试本地运行方法、PR 提交格式 |
| **类型系统文档** | 🔴 路径引用错误 | `data_structures.rst` 写的是 `src/memory/domain/types.py`，实际为 `mandol/domain/types.py` |
| **内部接口与公开接口区分** | 🟡 需改进 | 部分 `_` 前缀内部接口在文档中出现但未标注不可用 |
| **预想接口标注** | 🔴 完全缺失 | `retrieval_interfaces.rst` 中 `filter_memory_units`、`search(query, k, retriever_type, ...)` 等方法实际不存在，但未标注「📋 预想接口」 |
| **ports 层抽象接口文档** | 🔴 不存在 | 6+ 个端口接口（EmbeddingProvider/LLMProvider/Reranker/UnitStore/GraphStore/VectorIndex）无文档 |

#### 开发者最痛的几个点

1. **不知道哪些接口能用、哪些还没实现**：例如 `data_structures.rst` 中描述的 `SemanticMap.add_to_faiss()` 实际不存在，开发者按文档调用会直接报错。

2. **MemorySystem 的 30+ 配置参数无完整文档**：`MemorySystemConfig` dataclass 有 30+ 个字段，开发者需要了解每个字段的类型、默认值、作用、与哪些模块关联。

3. **检索管线的内部流程不可见**：`HybridRetriever.search()` → `rrf_fusion()` → BFS expand → rerank 这一整条链路，开发者需要理解才能做性能调优或二次开发。

4. **无法快速搭建开发环境并贡献代码**：虽然 `pip install mandol[dev]` 存在，但没有说明开发环境的具体步骤（虚拟环境选择、pre-commit hooks 配置等）。

---

## 第四部分：与优秀开源项目文档对比

| 维度 | LangChain | LlamaIndex | Mem0 | Mandol 现状 | 差距 |
|------|-----------|-----------|------|------------|------|
| 三层次分层 | Concepts → How-to → API | 同样三层 | 基础→高级→API | 混合在一起 | 🔴 |
| 快速开始可运行性 | 零配置 5 行 | 零配置 5 行 | 零配置 5 行 | 需 API Key 或模型下载 | 🟡 |
| 完整 API 文档 | autodoc+ 手写 | autodoc | 手写 (Mintlify) | 手写 4 个 .md | 🔴 |
| 场景示例 | 10+ | 15+ | 5+ | 3 个 (已规划) | 🟡 |
| 术语一致性 | 核心术语双语首次标注 | 统一英文 | 英文为主 | 中文混用，不一致 | 🟡 |
| 贡献指南 | 详细 + CI/CD 说明 | 详细 | 有 | 基本存在 | 🟡 |
| FAQ/故障排除 | 社区论坛 | Discord+GitHub | GitHub Discussions | 已新增 FAQ | 🟢 |
| 多语言文档 | 仅英文 | 仅英文 | 英/中/日/韩 | 英/中 | 🟢 |
| 文档站点 | 自建+Docusaurus | 自建 | Mintlify | Sphinx(未部署) | 🟡 |

---

## 第五部分：改进建议与优先级

### P0（开源前必须完成）

1. **修正 `data_structures.rst` 中所有过时的 API**
   - 将 `SemanticMap` → `SemanticMapService`
   - 将 `SemanticGraph` → `SemanticGraphService`
   - 更新所有属性、方法列表以匹配实际代码
   - 修正类型路径引用

2. **标注 `retrieval_interfaces.rst` 中的所有预想接口**
   - 使用 `📋 预想接口` 标签标注尚未实现的方法
   - 修正现有方法名（`get_all_units` → `list_units`，`search_similarity_by_text` → `search_by_text` 等）

3. **按三层次重组文档结构**
   - 基础用户（README + getting-started + 3个场景示例）
   - 高级用户（user-guide + 空间管理 + 图管理 + 参数调优）
   - 开发者（api-reference + 架构深入 + 贡献指南）

4. **补充 MemoryUnit 插入模式说明**
   - 明确只支持 `text_content` 和 `image_path` 两种字段
   - 提供标准的实例化模板

### P1（开源后首个迭代）

5. **补全 API Reference**（MemorySystem、SessionManager、检索管线、Config、Ports）
6. **实现英文文档完整镜像**
7. **统一术语策略**：「首次中文全称(英文) + 后续统一英文」
8. **添加 ADR 架构决策记录**

### P2（持续改进）

9. **部署文档站点**（Sphinx → GitHub Pages）
10. **添加交互式示例**（Jupyter Notebook）
11. **定期审计**文档与代码的一致性

---

## 附录：代码公开接口速查表

以下是从代码中提取的所有 **已实现的公开接口**，供文档编写参考：

### MemorySystem 公开属性
- `semantic_map` → `SemanticMapService`
- `graph` → `SemanticGraphService`
- `llm` → `LLMProvider`
- `dirty` → `bool`

### MemorySystem 公开方法
- `add(unit: MemoryUnit) -> None`
- `add_many(units: Sequence[MemoryUnit]) -> None`
- `build_high_level(mode: str = "auto") -> BuildReport`
- `build_high_level_async(mode: str = "auto") -> Future`
- `merge_cross_session_entities() -> None`
- `merge_cross_session_events() -> None`
- `holistic_retrieve(query, *, top_k=10, use_rerank=True) -> List[SearchHit]`
- `retrieve_in_space(query, space_name, *, top_k=10, use_rerank=True) -> List[SearchHit]`
- `retrieve_by_view(query, view, *, top_k=10, use_rerank=True) -> List[SearchHit]`
- `save(storage_path=None) -> SaveResult`
- `load(storage_path) -> MemorySystem` (类方法)
- `from_yaml_config(yaml_path, ...) -> MemorySystem` (类方法)
- `flush() -> None` (清空)

### SemanticMapService 公开方法
- `create_space(name) -> MemorySpace`
- `get_space(name) -> Optional[MemorySpace]`
- `list_spaces() -> List[MemorySpace]`
- `add_unit(unit, *, space_names, ensure_embedding, ...) -> None`
- `upsert_unit(unit) -> None`
- `delete_unit(uid) -> None`
- `get_unit(uid) -> Optional[MemoryUnit]`
- `list_units() -> List[MemoryUnit]`
- `add_unit_to_space(uid, space_name) -> None`
- `attach_child_space(parent, child) -> None`
- `ensure_child_space(parent, child) -> MemorySpace`
- `get_units_in_spaces(space_names, *, mode, recursive) -> List[MemoryUnit]`
- `search_by_vector(query, *, top_k, space_names, recursive) -> List[Tuple[MemoryUnit, float]]`
- `search_by_text(query_text, *, top_k, space_names, recursive) -> List[Tuple[MemoryUnit, float]]`
- `search_by_text_with_rerank(query_text, *, top_k, recall_k, space_names, recursive, use_rerank) -> List[Tuple[MemoryUnit, float]]`
- `search_in_space(query_text, space_name, candidates, *, top_k, recall_k) -> List[Tuple[MemoryUnit, float]]`
- `rebuild_index_from_store() -> None`
- `set_embedder(embedder) -> None`
- `set_reranker(reranker) -> None`

### SemanticGraphService 公开方法
- `add_unit(unit, *, space_names, ensure_embedding) -> None`
- `delete_unit(uid) -> None`
- `add_relationship(source_uid, target_uid, relationship_name, **properties) -> None`
- `get_relationship(source_uid, target_uid, relationship_name) -> Optional[Dict]`
- `delete_relationship(source_uid, target_uid, relationship_name=None) -> None`
- `get_explicit_neighbors(uids, *, rel_type, direction) -> List[MemoryUnit]`
- `get_implicit_neighbors(uids, *, top_k) -> List[Tuple[MemoryUnit, float]]`
- `get_units_in_spaces(space_names, *, mode, recursive) -> List[MemoryUnit]`
- `bfs_expand_units(seeds, *, per_seed, hops, rel_type) -> List[MemoryUnit]`
