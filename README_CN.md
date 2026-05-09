[English](README.md) | [中文](README_CN.md)

# Mandol

> 面向长对话智能体的内存原生分层记忆系统

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Docs](https://img.shields.io/badge/docs-Sphinx-green.svg)](docs/)

## Mandol 是什么？

Mandol 是一个基于内存数据结构的轻量级记忆系统。它通过语义索引 (SemanticMap) 和语义关系图 (SemanticGraph) 提供统一存储与混合检索，在内存层面融合键值、向量与图，支持零 IPC 的原生混合检索。目前，Mandol 暂时主要面向对话数据集进行了实现与验证，在 LoCoMo、LongMemEval 等长对话记忆基准测试中表现出色。

## 环境准备

### Python 版本

- 最低要求：Python 3.9+
- 推荐版本：Python 3.10 或 3.11

### 包管理工具

使用 pip：

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
pip install mandol
```

使用 conda：

```bash
conda create -n mandol python=3.10
conda activate mandol
pip install mandol
```

### 系统资源配置

| 配置项 | 最低要求（远程 API） | 最低要求（本地 Embedding） | 最低要求（本地 Embedding+Reranker） | 推荐配置 |
|--------|---------------------|--------------------------|-------------------------------------|----------|
| CPU | 4 核 | 4 核 | 8 核 | 8 核+ |
| 内存 | 8 GB | 16 GB | 32 GB | 16-64 GB |
| GPU | 无（CPU 可运行） | 无（CPU 可运行，GPU 更快） | 建议 NVIDIA 8GB+ 显存 | NVIDIA 16GB+ 显存 |
| 磁盘 | 2 GB | 6 GB | 10 GB | 10 GB+ |

### 模型下载

| 模型 | 用途 | 大小 | 下载方式 |
|------|------|------|---------|
| `Qwen/Qwen3-Embedding-4B` | 文本向量化 | ~4 GB | 首次运行自动下载至 `~/.cache/huggingface/` |
| `Qwen/Qwen3-Reranker-4B` | 检索重排序 | ~4 GB | 首次运行自动下载至 `~/.cache/huggingface/` |

> **提示**：若使用远程 API 模式，无需下载本地模型，仅需配置 API 端点即可。

## 安装

### 基础安装

```bash
pip install mandol
```

或从源码安装：

```bash
git clone https://github.com/your-org/mandol.git
cd mandol
pip install -e .
```

### 可选依赖

```bash
pip install mandol[faiss]                    # FAISS 向量索引加速
pip install mandol[sentence-transformers]    # 本地 Embedding/Reranker 模型
pip install mandol[openai]                   # OpenAI API 支持
pip install mandol[milvus]                   # Milvus 向量数据库
pip install mandol[neo4j]                    # Neo4j 图数据库
pip install mandol[all]                      # 安装所有可选依赖
pip install mandol[dev]                      # 开发工具（pytest、ruff 等）
```

### 环境变量配置

```bash
cp .env.example .env
# 编辑 .env 文件，填入 API Key：
# OPENAI_API_KEY=sk-your-key-here
```

### 验证安装

```bash
python -c "from mandol import MemorySystem, MemoryUnit, Uid; print('Mandol 安装成功！')"
```

## 快速开始

### 模式一：远程 API（推荐新手，零本地模型）

```python
from mandol import MemorySystem, MemoryUnit, Uid

system = MemorySystem.from_yaml_config("config.yaml")
# 在 config.yaml 中设置 embedder.use_remote: true 和 reranker.use_remote: true，
# 并配置 API 端点和密钥。

unit = MemoryUnit(
    uid=Uid("msg_001"),
    raw_data={"text_content": "张三今天去北京出差了"},
    metadata={"timestamp": "2024-01-15T10:00:00"},
)
system.add(unit)

system.build_high_level(mode="auto")

hits = system.holistic_retrieve("张三去了哪里？", top_k=5)
for hit in hits:
    print(f"[{hit.final_score:.3f}] {hit.unit.raw_data['text_content']}")

system.save("./memory_snapshot")
system2 = MemorySystem.load("./memory_snapshot")
```

> **关于 ``build_high_level()``**：系统在 ``add()`` 时会异步检测会话边界并自动触发高阶记忆构建。
> - 仅检索原始对话（BASE 组）：无需等待，``add()`` 后即可检索
> - 检索实体/事件/摘要（ENTITY / EVENT / SUMMARY 组）：需等待自动构建完成或手动调用 ``build_high_level()``
> - 插入少量数据后立即检索：建议手动调用 ``build_high_level()`` 确保高阶记忆可用

### 模式二：本地模型（无需 API Key，需下载模型）

```python
from mandol import MemorySystem, MemoryUnit, Uid

system = MemorySystem()
# 默认使用本地 Qwen3-Embedding-4B 和 Qwen3-Reranker-4B
# 首次运行自动下载模型（共约 8 GB）

unit = MemoryUnit(
    uid=Uid("msg_001"),
    raw_data={"text_content": "张三今天去北京出差了"},
    metadata={"timestamp": "2024-01-15T10:00:00"},
)
system.add(unit)

system.build_high_level(mode="auto")

hits = system.holistic_retrieve("张三去了哪里？", top_k=5)
for hit in hits:
    print(f"[{hit.final_score:.3f}] {hit.unit.raw_data['text_content']}")

system.save("./memory_snapshot")
```

## 核心概念

### 记忆系统是什么？

想象你有一个智能助手，它能记住你说过的一切，并在需要时精准回忆。Mandol 就是这样的"记忆大脑"——它不仅存储对话，还能：

- 🧠 **自动提取关键信息** — 人名、地点、事件
- 🔗 **建立信息之间的关联** — 谁在哪做了什么，因果关系
- 🔍 **精准检索** — 不只是关键词匹配，而是语义理解

### 关键术语

| 术语 | 通俗解释 | 类比 |
|------|---------|------|
| MemoryUnit | 一条记忆记录 | 一张便签 |
| MemorySpace | 记忆的分类文件夹 | 文件柜的抽屉 |
| SemanticMap | 语义索引：记忆的向量索引与检索引擎 | 图书馆的检索卡片 |
| SemanticGraph | 语义关系图：记忆之间的关联网络 | 思维导图 |
| 会话 (Session) | 一次连贯的对话 | 一次会议 |
| 实体 (Entity) | 对话中提到的人/地/物 | 名片 |
| 事件 (Event) | 对话中发生的事情 | 日记条目 |

> **关于 MemoryUnit 的插入模式**：目前 ``raw_data`` 中系统会自动向量化的字段为：
> - ``text_content``：纯文本内容 → 稠密向量
> - ``image_path``：图片文件路径 → 图片向量
>
> 其他字段（如 ``speaker``、``source`` 等）会作为元数据存储，但不会自动生成向量。

### 工作流程

```
[用户输入] → [分块 + 向量化] → [会话分割]
→ [提取实体/事件/摘要] → [构建关系图]
→ [检索：三路召回 → RRF 融合 → BFS 扩展 → 重排序]
→ [返回结果]
```

## 核心功能

### 1. 数据管理

| 操作 | 方法 | 说明 |
|------|------|------|
| 添加单条记忆 | `add(unit)` | 自动分块、自动 embedding |
| 批量添加 | `add_many(units)` | 批量处理更高效 |
| 保存状态 | `save(directory)` | 导出为目录（包含多个 JSON 文件） |
| 加载状态 | `MemorySystem.load(directory)` | 从目录恢复状态（类方法） |

### 2. 记忆构建

添加记忆后，系统自动异步完成高阶记忆构建，无需手动调用。如需手动干预，可使用以下接口：

| 操作 | 方法 | 说明 |
|------|------|------|
| 强制重建 | `build_high_level(mode="force")` | 清空状态，重新处理所有会话 |
| 异步重建 | `build_high_level_async()` | 后台执行构建 |

**自动构建流程**：
- 会话分割（LLM 驱动）
- 情景摘要 / 知识摘要 / 情感摘要 / 过程摘要生成
- 洞察提取与全局合并
- 实体提取与去重
- 事件提取与去重
- 实体关系构建
- 事件因果链构建
- 跨会话实体/事件合并

### 3. 检索功能

#### 全记忆检索（推荐）

```python
hits = system.holistic_retrieve("query", top_k=10)
```

**检索流程**：
1. 分组召回：BASE / ENTITY / EVENT / SUMMARY 四组独立检索
2. 每组内部：Dense + BM25 + Sparse 三路召回 → RRF 融合 → BFS 扩展
3. 全局重排：所有候选合并后通过 Cross-Encoder Reranker 重排序

系统同时提供语义检索、图关系检索等底层接口，详见[开发者文档](docs/index.rst)。

## 配置选项

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `MANDOL_EMBEDDER_MODEL` | Embedding 模型 | `Qwen/Qwen3-Embedding-4B` |
| `MANDOL_EMBEDDER_DEVICE` | Embedding 设备 | `cpu` |
| `MANDOL_RERANKER_MODEL` | Reranker 模型 | `Qwen/Qwen3-Reranker-4B` |
| `MANDOL_RERANKER_DEVICE` | Reranker 设备 | `cpu` |
| `MANDOL_LLM_MODEL` | LLM 模型 | `gpt-4o-mini` |
| `OPENAI_API_KEY` | OpenAI API Key | `""` |
| `MANDOL_LLM_BASE_URL` | OpenAI API Base URL | `https://api.openai.com/v1` |
| `USE_REMOTE_EMBEDDER` | 是否使用远程 Embedder | `false` |
| `USE_REMOTE_RERANKER` | 是否使用远程 Reranker | `false` |

> **注意**：以上环境变量需通过 YAML 配置文件（``config.yaml``）或 ``MemorySystemConfig`` dataclass 传入系统。直接设置 ``os.environ`` 不会自动生效，请参考下方 YAML 配置示例进行配置。

### YAML 配置

```yaml
llm:
  model: "gpt-4o-mini"
  base_url: "https://api.openai.com/v1"
  api_key: "sk-..."

embedder:
  model: "Qwen/Qwen3-Embedding-4B"
  device: "cuda"
  dimension: 2560
  use_remote: false
  base_url: "http://localhost:8000/v1"
  api_path: "/embeddings"
  api_key: ""
  timeout: 30

reranker:
  model: "Qwen/Qwen3-Reranker-4B"
  device: "cuda"
  use_remote: false
  base_url: ""
  api_path: "/v1/rerank"
  api_key: ""
  timeout: 30

system:
  chunk_max_tokens: 512
  session_time_gap_seconds: 1800
  session_check_interval: 20
  session_max_pending: 100
  similarity_top_k: 5
  similarity_threshold: 0.7
  similarity_recent_window: 20
  bfs_expansion_per_seed: 3
  bfs_expansion_hops: 1
  max_context_units: 20
  max_entities_per_llm: 50
  max_events_per_llm: 50
  promote_threshold: 100

storage:
  root: null
  enable_persistence: false
  auto_save_interval: 300
```

## 架构概览

```
┌──────────────────────────────────────────────────────────────────────┐
│                          检索 API 层                                 │
│  holistic_retrieve() → 四组并行检索 → 全局 Cross-Encoder Rerank → Top-K │
├──────────────────────────────────────────────────────────────────────┤
│                           记忆层次层                                  │
│  ┌──────────────────┐  ┌──────────────────────────────────────────┐ │
│  │  基础记忆 (Base)  │  │           高阶记忆 (High-Level)          │ │
│  │  原始对话片段     │  │  情景 │ 知识 │ 情感 │ 过程              │ │
│  │                   │  │  摘要 │ 实体 │ 事件 │ 洞察              │ │
│  └──────────────────┘  └──────────────────────────────────────────┘ │
├──────────────────────────────────────────────────────────────────────┤
│                        核心数据结构层                                 │
│  MemoryUnit ←→ MemorySpace ←→ SemanticMap ←→ SemanticGraph          │
│  (记忆单元)    (逻辑空间)     (语义索引)      (语义关系图)            │
└──────────────────────────────────────────────────────────────────────┘
```

详见 [开发者文档](docs/index.rst)

## 性能测试

Mandol 在 LoCoMo（Long Conversational Memory）和 LongMemEval 基准数据集上进行了全面评估。评估采用 LLM 作为评判者（LLM-as-judge），判断生成答案与正确答案是否一致。

### 关键指标

| 指标 | 说明 |
|------|------|
| F1 Score | LLM-as-judge 评估生成答案与正确答案的一致性 |
| 响应时间 | 从发起查询到返回结果的端到端延迟 |
| 内存占用 | 系统运行时的峰值 RSS |
| 索引构建时间 | `build_high_level()` 完成的总耗时 |

### LoCoMo 结果（GPT-4.1-mini 骨干网络）

| 系统 | 平均Token | 单跳 | 多跳 | 时序 | 开放域 | 总体 |
|------|----------|------|------|------|--------|------|
| Mem0 | 1.0k | 68.97 | 61.70 | 58.26 | 50.00 | 64.20 |
| MemU | 4.0k | 74.91 | 72.34 | 43.61 | 54.17 | 66.67 |
| MemOS | 2.5k | 85.37 | 79.43 | 75.08 | 64.58 | 80.76 |
| Zep | 1.4k | 90.84 | 81.91 | 77.26 | 75.00 | 85.22 |
| EverMemOS† | 2.3k | 95.32 | 89.01 | 90.13 | 77.43 | 91.97 |
| **Mandol (Ours)** | **1.9k** | **95.36** | **92.20** | 87.85 | **79.17** | **92.21** |

### LongMemEval 结果（GPT-4.1-mini 骨干网络）

| 系统 | 平均Token | SS-Pref | SS-Asst | 时序 | 多会话 | 知识更新 | SS-User | 总体 |
|------|----------|---------|---------|------|--------|----------|---------|------|
| EverMemOS | 2.8k | 93.33 | 85.71 | 77.44 | 73.68 | 89.74 | 97.14 | 83.00 |
| **Mandol (Ours)** | 2.3k | **96.67** | **98.21** | **87.22** | **77.44** | **89.74** | **98.57** | **88.40** |

> **注意**：总体指标不包含对抗性查询。† 表示使用 EverMemOS 官方实现复现的结果。最优结果以**粗体**标注。

### 快速复现

```bash
# LoCoMo
cd benchmarks/locomo && bash scripts/env.sh
python build_graph.py --config configs/base.yaml --output output/
python retrieve.py --config configs/base.yaml --input output/ --output output/
python generate.py --config configs/base.yaml --input output/ --output output/
python evaluate.py --input output/ --output output/

# LongMemEval
cd benchmarks/longmemeval && bash scripts/env.sh
python build_graph.py --config configs/base.yaml --output output/
python retrieve.py --config configs/base.yaml --input output/ --output output/
python generate.py --config configs/base.yaml --input output/ --output output/
python evaluate.py --input output/ --output output/
```

📖 完整的测试环境配置、数据集说明、消融实验及性能对比表格，请参阅 [LoCoMo Benchmark 文档](benchmarks/locomo/README.md) 和 [LongMemEval Benchmark 文档](benchmarks/longmemeval/README.md)。

## 常见问题 (FAQ)

### 安装问题

**Q：`pip install mandol` 报错 "No matching distribution found"**
A：请确认 Python 版本 >= 3.9，并尝试 `pip install --upgrade pip`。

**Q：安装 `faiss-cpu` 失败**
A：尝试 `conda install -c conda-forge faiss-cpu` 或 `pip install faiss-cpu --no-deps`。

### 运行错误

**Q：`MemorySystem()` 初始化时报 CUDA out of memory**
A：设置环境变量 `MANDOL_EMBEDDER_DEVICE=cpu` 和 `MANDOL_RERANKER_DEVICE=cpu`，或使用远程 API 模式（`USE_REMOTE_EMBEDDER=true`）。

**Q：`holistic_retrieve` 返回空结果**
A：请确认已调用 `build_high_level()` 或等待自动构建完成，检查是否有足够的记忆数据（建议至少 5 条以上）。

**Q：LLM API 调用超时**
A：检查 `OPENAI_API_KEY` 是否正确，网络是否可达 API 端点，可设置 `MANDOL_LLM_TIMEOUT_S=120` 增加超时时间。

### 性能优化

**Q：检索速度慢怎么优化？**
A：
1. 使用 FAISS 索引加速：`pip install mandol[faiss]`
2. 减小 `bfs_expansion_hops`（默认 1 → 0）
3. 关闭重排序：`holistic_retrieve(query, use_rerank=False)`
4. 使用 GPU 加速 Embedding：`MANDOL_EMBEDDER_DEVICE=cuda`

**Q：内存占用过高怎么优化？**
A：
1. 使用远程 Embedding/Reranker 替代本地模型
2. 减小 `similarity_recent_window`
3. 启用持久化并定期 `save`/`load`

## 文档

- [开发者文档](docs/index.rst) - 架构设计、数据结构、检索接口、扩展指南

## 许可

Apache License 2.0 - 详见 [LICENSE](LICENSE)
