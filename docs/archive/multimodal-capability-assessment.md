# Mandol 记忆系统多模态支持能力全面评估

## 1. 多模态数据支持现状

### 1.1 总体评估

当前 Mandol 记忆系统**以文本为核心模态**，对图像模态有**接口层的预留设计**，但对音频、视频等模态**完全未涉及**。系统整体仍处于"文本为主、图像为辅（接口预备）"的阶段。

### 1.2 核心数据模型：`MemoryUnit`

所有模态数据统一通过 `MemoryUnit` 承载：

```python
@dataclass(slots=True)
class MemoryUnit:
    uid: Uid
    raw_data: Dict[str, Any]          # 核心：存储任意键值对的柔性容器
    metadata: Dict[str, Any]
    embedding: Optional[Embedding]        # 稠密向量
    sparse_embedding: Optional[Embedding]  # 稀疏向量
```

- `raw_data` 是一个 `Dict[str, Any]`，理论上可以存放任意类型的数据（包括图像路径、base64 编码等）
- `embedding` 存储统一向量空间中的表示，无论是文本还是图像都映射为同一个维度的向量

### 1.3 文本模态处理机制（完整支持）

文本处理是整个系统的核心流程，覆盖完整：

| 处理环节 | 关键组件 | 机制说明 |
|---------|---------|---------|
| 文本嵌入 | `SentenceTransformersEmbeddingProvider` | `SentenceTransformer.encode()` 生成稠密向量 |
| 文本切块 | `DocumentChunker` | 基于 token 数的句子级切分 |
| BM25/稀疏检索 | `HybridRetriever._fallback_bm25_search()` | 基于 `text_content` 字段提取文本构建倒排索引 |
| 重排序 | `SentenceTransformersCrossEncoderReranker` / `OpenAICompatibleReranker` | 基于 `text_content` 提取文本进行 Cross-Encoder 打分 |
| 高层记忆 | `UnifiedFactPipeline` / `SemanticMap` | 实体提取、事件提取、摘要生成等均基于文本 |

### 1.4 图像模态处理机制（接口预备，实质缺失）

#### 接口定义

`EmbeddingProvider` 接口定义了 `embed_image_paths` 方法：

```python
class EmbeddingProvider(ABC):
    def embed_text(self, texts: Sequence[str], **kwargs: Any) -> List[Embedding]: ...
    def embed_image_paths(self, image_paths: Sequence[str], **kwargs: Any) -> List[Embedding]: ...
```

#### 两个实际实现的处理方式

| Provider | 实际行为 |
|----------|---------|
| `SentenceTransformersEmbeddingProvider` | **将图像路径当作文本处理**——`self.embed_text([str(p) for p in image_paths])`，即把文件路径字符串编码为向量 |
| `OpenAICompatibleEmbeddingProvider` | **将路径字符串直接传给 API**——`self._embed(list(image_paths))` |

**结论**：两个实现都**没有真正加载和编码图像像素内容**。`SentenceTransformersEmbeddingProvider` 明确注释 "Not supported in ST text models; treat path as text"。

#### SemanticMapService 中的图像路径预留

`SemanticMapService._embed_for_unit()` 包含唯一的图像嵌入调度逻辑：

- **文本优先原则**：先尝试 `text_content` 嵌入，仅在文本为空时才使用 `image_path`
- 这意味着一个 MemoryUnit **要么是文本单元，要么是图像单元**，不支持图文混合嵌入

### 1.5 其他模态支持情况

| 模态 | 支持状态 | 说明 |
|------|---------|------|
| **文本** | 完整支持 | 全链路覆盖 |
| **图像** | 接口预留 | 嵌入实现为退化的路径字符串编码 |
| **音频** | 无支持 | 无任何接口定义、字段预留或处理逻辑 |
| **视频** | 无支持 | 同上 |
| **3D/点云** | 无支持 | 同上 |
| **结构化表格** | 间接支持 | 可通过 `raw_data` 存放序列化数据，但无专用处理 |

---

## 2. MemoryUnit 插入规范

### 2.1 文字内容的标准插入方式

```python
unit = MemoryUnit(
    uid=Uid("msg_001"),
    raw_data={
        "text_content": "张三今天去北京出差了。",  # 核心字段
    },
    metadata={
        "timestamp": "2024-01-15T10:00:00",
        "speaker": "user",
    },
)
system.add(unit)
```

#### 字段规范

| 字段路径 | 类型 | 必需 | 说明 |
|---------|------|------|------|
| `raw_data["text_content"]` | `str` | 是（对文本单元） | 默认文本键，整个系统的文本提取、嵌入、BM25/稀疏索引都依赖此键 |
| `metadata["timestamp"]` | `str` (ISO 8601) | 推荐 | 时间排序和会话分割的核心依据 |
| `metadata["speaker"]` | `str` | 可选 | 说话人标识 |
| `metadata["dia_id"]` | `str` | 可选 | 对话 ID |

### 2.2 图片内容的插入方式

```python
image_unit = MemoryUnit(
    uid=Uid("img_001"),
    raw_data={
        "image_path": "/path/to/image.jpg",  # 默认图像键
    },
    metadata={
        "timestamp": "2024-01-15T11:00:00",
        "type": "image",
    },
)
system.add(image_unit)
```

#### 限制与注意事项

1. **文本优先**：同时提供 `text_content` 和 `image_path` 时，仅使用文本生成向量
2. **不支持图文混合**：一个 MemoryUnit 只能取文本或图像其一
3. **不支持 base64 / bytes**：仅接受文件路径字符串

### 2.3 其他模态数据

**当前无任何官方支持**。若需临时存放其他模态，可：
- 利用 `raw_data` 字典存放自定义字段
- 通过 `metadata` 添加模态类型标记
- **但系统不会自动为其生成有意义的向量表征**

---

## 3. 插入后处理流程

### 3.1 `MemorySystem.add()` 的完整处理链路

```
add(unit)
  ├── _ensure_layout()             # 确保空间拓扑已构建
  ├── 设置默认 metadata
  ├── should_chunk() ?             # 判断是否需要分块
  │   ├── 是: chunk_unit()         # 句子级切分 → 多个子 MemoryUnit
  │   │   └── semantic_map.add_unit() × N
  │   └── 否:
  │       └── semantic_map.add_unit()
  ├── _build_immediate_similarity_edges()   # 与最近窗口内的单元建立语义相似边
  └── _check_session_boundary_async()       # 异步检测会话边界
```

### 3.2 高层记忆构建——必须手动触发

`add()` / `add_many()` 只完成基础存储和嵌入，**高层记忆（实体、事件、摘要、洞察）不会自动生成**。

必须调用以下方法触发：

| 方法 | 触发方式 | 说明 |
|------|---------|------|
| `build_high_level(mode="auto")` | 手动 | 同步执行 |
| `build_high_level_async(mode="auto")` | 手动 | 异步版本 |
| 自动触发（部分） | 异步 | pending units 达到阈值时自动检测会话边界 |

### 3.3 对多模态单元的影响

- 图像单元的嵌入依赖于 Provider 的实现质量
- 分块器对图像单元跳过切分
- 会话分割 Prompt 构建仅提取 `text_content`，图像单元静默跳过
- `UnifiedFactPipeline` 仅处理 `text_content`

---

## 4. 对检索功能的影响

### 4.1 检索架构

```
HybridRetriever.search(query)
  ├── Dense Search     → semantic_map.search_by_text()    # 文本查询的稠密向量检索
  ├── BM25 Search      → bm25_index.search()              # 词袋匹配
  ├── Sparse Search    → sparse_index.search()            # TF-IDF 稀疏向量
  ├── RRF Fusion       → 倒数排名融合三路结果
  ├── BFS Expansion    → 图上的广度优先邻居扩展
  └── Rerank           → Cross-Encoder 重排序
```

### 4.2 多模态对检索准确性的影响

#### Dense 检索

- **查询侧**：仅支持文本查询（无 `search_by_image`）
- **文档侧**：图像单元向量由 `embed_image_paths()` 生成，ST provider 退化为路径字符串编码，语义检索准确性为零

#### BM25 / Sparse 检索

- `_extract_text()` 按优先级查找文本字段
- 图像单元仅 `image_path` 时，可能匹配到路径字符串中的关键词（如 `cat.jpg`），但仅限巧合

#### Rerank 路径

- 两个 Reranker 实现均仅提取文本字段，图像单元可能被丢弃

### 4.3 缺失能力

1. `search_by_image` 接口（以图搜图）
2. 跨模态检索接口（以文搜图、以图搜文）
3. 多模态查询编码器

---

## 5. 对生成逻辑的影响

### 5.1 现有生成组件

| 组件 | 功能 | 多模态支持 |
|------|------|-----------|
| `SummaryMapReducer` | 会话级摘要生成 | 仅处理文本 |
| `InsightMapReducer` | 洞察提取 | 仅处理文本 |
| `UnifiedFactPipeline` | 实体/事件/关系提取 | 仅处理文本 |
| `GlobalInsightManager` | 全局洞察合并 | 仅处理文本 |
| `CrossSessionCorefManager` | 跨会话共指消解 | 仅处理文本 |

所有 LLM 调用均通过 `ChatMessage` 传递纯文本 `content`，不支持多模态输入。

### 5.2 验证发现的缺陷

**缺陷 #1**：`UnifiedFactPipeline` 调用不存在的 `embed()` 方法

```python
# unified_fact_pipeline.py:L302, L331
query_emb = self._embedder.embed(dialogue_text)   # ❌ EmbeddingProvider 无此方法
```

正确调用应为 `self._embedder.embed_text([dialogue_text])[0]`。

**缺陷 #2**：图像单元在高层记忆中被完全跳过（`_build_dialogue_text()` 仅提取 `text_content`）

**缺陷 #3**：会话分割 Prompt 构建仅提取 `text_content`，图像单元被静默跳过

---

## 6. 潜在问题与改进建议

### 6.1 缺陷汇总

| 编号 | 严重程度 | 问题描述 |
|------|---------|---------|
| P1 | 严重 | `UnifiedFactPipeline` 调用不存在的 `embedder.embed()` 方法 |
| P2 | 高 | `SentenceTransformersEmbeddingProvider` 对图像路径仅作文本编码 |
| P3 | 高 | 图像单元在会话分割、BM25、重排序中被跳过或降级 |
| P4 | 中 | 不支持图文混合嵌入（一个 unit 只能取文本或图像其一） |
| P5 | 中 | 缺失 `search_by_image` / 跨模态检索接口 |
| P6 | 中 | 缺失真正的图像编码器（如 CLIP、SigLIP）集成 |
| P7 | 低 | 缺失音频、视频等模态的任何接口或字段预留 |
| P8 | 低 | LLM 调用不支持多模态输入 |

### 6.2 改进路线图

```
Phase 1 (Bug Fix):     修复 embed() 调用错误
Phase 2 (Image MVP):   引入多模态 Embedder → 真正的图像嵌入 → 图文混合嵌入
Phase 3 (Retrieval):   跨模态检索接口 → search_by_image → 多模态 Reranker
Phase 4 (Generation):  LLM 多模态输入 → 图像感知摘要 → 视觉实体识别
Phase 5 (Full Modal):  音频/视频支持 → 多模态融合索引 → 端到端多模态记忆
```

---

## 总结

当前 Mandol 记忆系统的多模态支持处于**"接口预留、实质缺失"**状态：

- **文本**：完整支持，覆盖全链路
- **图像**：接口层有预留，实际实现退化为路径字符串编码
- **其他模态**：无任何支持
- **Bug**：`UnifiedFactPipeline` 中存在调用不存在方法的严重缺陷

系统架构（`MemoryUnit.raw_data` 柔性容器 + `EmbeddingProvider` 接口分离）为多模态扩展奠定了良好基础，但距离真正的多模态记忆系统仍有大量工作要做。
