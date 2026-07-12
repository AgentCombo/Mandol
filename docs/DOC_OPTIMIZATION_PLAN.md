# Mandol 文档体系现状评估报告与优化实施方案

> 生成日期：2026-04-28
> 版本：v1.0（初版）→ v1.1（根据反馈修订）

---

## 第一部分：文档现状评估

### 一、现有文档资产清单

| 文档类型 | 文件 | 状态 |
|---------|------|------|
| README（英文） | README.md | ✅ 存在，约284行 |
| README（中文） | README_CN.md | ✅ 存在，约210行 |
| Sphinx 文档（中文） | docs/ | ✅ 存在，5个 .rst 文件 |
| Sphinx 文档（英文） | docs/en/ | ✅ 存在，但仅 index.rst 有内容 |
| API Reference | docs/api-reference/ | ⚠️ 仅4个 .md 文件，覆盖不完整 |
| 快速开始示例 | examples/quick_start.py | ✅ 存在，22行 |
| 对话演示 | examples/dialogue_demo/ | ✅ 存在 |
| LoCoMo 开发评测 | experimental/self_host_benchmarks/locomo/README.md | ⚠️ 持续开发中 |
| 配置模板 | .env.example / config.yaml | ✅ 存在 |

---

### 二、三视角现状评估

#### 2.1 基础用户视角评估

**评估维度：简单使用和功能复现方面的清晰度、完整性和易用性**

| 评估项 | 现状 | 评分 | 问题详述 |
|--------|------|------|---------|
| **环境准备** | ❌ 缺失 | 1/5 | 无 Python 版本要求说明（仅 badge 写 3.9+），无包管理工具指南，无系统资源最低配置，无模型下载链接及存放路径规范 |
| **安装步骤** | ⚠️ 不完整 | 2/5 | 仅 `pip install mandol` 和 `pip install -e .`，缺少依赖安装（可选依赖 faiss/milvus/neo4j 等）、环境变量配置步骤、验证安装成功的方法 |
| **核心概念** | ⚠️ 过于技术化 | 2/5 | TL;DR 使用"多库异构架构""零 IPC 的原生混合检索"等术语，基础用户难以理解；缺少通俗的类比和图解 |
| **快速开始** | ⚠️ 不可直接运行 | 2/5 | 代码示例依赖 LLM API Key 和 Embedding 模型，但未说明如何获取和配置；`MemorySystem()` 默认配置需要下载 Qwen3-Embedding-4B（约 4GB），未提及 |
| **常见 QA** | ❌ 缺失 | 0/5 | 完全没有 FAQ/故障排除章节 |
| **性能测试** | ⚠️ 不完整 | 2/5 | experimental/self_host_benchmarks/locomo/README.md 有开发流程说明，但关键指标定义仍需完善 |

**基础用户核心痛点**：
1. **"装不上"**：不知道需要 LLM API Key，不知道 Embedding 模型需要下载，不知道 GPU 要求
2. **"跑不起来"**：快速开始示例直接 `MemorySystem()` 会尝试加载本地模型，CPU 环境下可能 OOM
3. **"看不懂"**：术语过于专业，缺少从零开始的引导
4. **"出错了无处查"**：没有 FAQ，没有错误码说明，没有 troubleshooting

---

#### 2.2 高级用户视角评估

**评估维度：功能自定义和个性化配置方面的指导充分性**

| 评估项 | 现状 | 评分 | 问题详述 |
|--------|------|------|---------|
| **实用场景示例** | ⚠️ 不足 | 2/5 | 仅有 1 个 dialogue_demo，缺少多场景（客服对话、个人助手、知识库等）完整示例 |
| **系统架构深入** | ✅ 较好 | 3/5 | introduction.rst 有架构说明，但数据流向图不够清晰，核心算法原理（RRF 融合权重、BFS 扩展策略）未深入解释 |
| **自定义空间划分** | ⚠️ 不充分 | 2/5 | extending.rst 提到了 DimensionBuilder 接口，但缺少完整的自定义空间划分策略指南和实战示例 |
| **图结构管理** | ⚠️ 不充分 | 2/5 | SemanticGraph API 有列出，但缺少高级图管理技巧（如批量关系操作、图修剪、图合并等） |
| **会话划分最佳实践** | ❌ 缺失 | 1/5 | 仅提到 `session_time_gap_seconds` 参数，缺少不同场景下的会话划分策略建议 |
| **参数调优** | ❌ 缺失 | 1/5 | 配置表列出了参数，但缺少每个参数的调优建议、参数间的关联关系、不同场景的推荐配置 |
| **持久化与部署** | ⚠️ 不充分 | 2/5 | 仅提到 save/load，缺少 Milvus/Neo4j 生产部署指南、数据迁移方案、备份恢复策略 |

**高级用户核心痛点**：
1. **"不知道怎么调"**：30+ 配置参数没有调优指南
2. **"不知道最佳实践"**：会话划分、空间命名、图构建等缺少最佳实践
3. **"不知道怎么扩展"**：自定义维度构建器、自定义 Provider 的文档过于简略
4. **"不知道怎么部署"**：生产环境部署（Milvus + Neo4j）缺少完整指南

---

#### 2.3 开发者视角评估

**评估维度：系统掌控和二次开发方面的技术深度和接口完整性**

| 评估项 | 现状 | 评分 | 问题详述 |
|--------|------|------|---------|
| **API 完整性** | ⚠️ 不完整 | 2/5 | API Reference 仅覆盖 4 个核心类，缺少 MemorySystem、SessionManager、各 DimensionBuilder、检索模块等的 API 文档 |
| **参数说明** | ⚠️ 不精确 | 2/5 | 部分方法签名缺少参数类型和返回值类型；异常处理说明完全缺失 |
| **接口分类** | ⚠️ 混乱 | 2/5 | 公开接口与内部接口（下划线前缀）混在一起，预想接口（未实现）与已实现接口未明确区分 |
| **代码示例** | ⚠️ 不足 | 2/5 | 多数 API 仅有 1-2 行示例，缺少完整的增删改查操作示例和运行结果说明 |
| **架构决策记录** | ❌ 缺失 | 0/5 | 无 ADR（Architecture Decision Records），缺少关键设计决策的背景和理由 |
| **贡献指南** | ⚠️ 存在但不完整 | 2/5 | 有 CONTRIBUTING.md 但未与文档体系整合，缺少代码规范、测试要求、PR 流程说明 |
| **类型系统文档** | ❌ 缺失 | 1/5 | types.py 中的类型别名（Uid, SpaceName, Embedding, SearchHit）文档中引用路径错误（写成了 `src/memory/domain/types.py`） |

**开发者核心痛点**：
1. **"API 文档不全"**：MemorySystem 的 30+ 配置参数、SemanticMapService 的 15+ 方法、检索模块的各类 Retriever 均无完整文档
2. **"不知道哪些已实现"**：预想接口（trace_provenance、compare_multi_view_consistency 等）与已实现接口混在一起
3. **"类型引用错误"**：文档中多处路径引用与实际代码不符
4. **"缺少异常说明"**：所有 API 均无异常处理文档

---

### 三、与优秀开源项目的差距分析

| 维度 | Mem0 | Zep | LangChain | Mandol 现状 | 差距 |
|------|------|-----|-----------|------------|------|
| README 多语言 | 英/中/日/韩 | 仅英文 | 仅英文 | 英/中 | 与 Mem0 差 2 种语言 |
| 快速开始可运行性 | 5行代码零配置 | Docker+SDK | 3-5行代码 | 需配置 API Key + 下载模型 | 差距大 |
| 配置文档覆盖 | 14向量库/13LLM/9Embedder | 部署配置为主 | Memory类型+后端 | 仅环境变量表+YAML示例 | 差距极大 |
| API 文档完整性 | Python SDK+REST API | 3语言SDK+REST API | 自动生成 | 手写4个.md | 差距极大 |
| 实用场景示例 | 10+集成示例 | 3SDK+集成 | How-to Guides | 1个demo | 差距大 |
| FAQ/故障排除 | 有 | 有 | GitHub Discussions | 无 | 完全缺失 |
| 性能基准 | LoCoMo对比表格 | 较少 | 无 | 占位符 | 需补充 |
| 文档站点 | Mintlify | Mintlify | Sphinx/自建 | Sphinx（未部署） | 需部署 |

---

## 第二部分：文档优化实施方案

### 一、总体设计原则

1. **分层递进**：README（5分钟上手）→ 快速开始（30分钟理解）→ 高级指南（深度定制）→ API Reference（开发参考）
2. **双路径导航**：基础用户路径（安装→使用→常见问题）与高级用户路径（架构→配置→扩展）
3. **示例驱动**：每个概念至少配一个可运行示例，每个 API 至少配一个使用场景
4. **手工维护 API 文档**：不使用 autodoc 自动生成，手工编写 API 文档以保留预想接口（未实现但需文档化的接口）
5. **中英同步**：中文为主文档，英文为镜像文档，保持结构一致

---

### 二、README 文档优化方案（面向基础用户）

#### 2.1 新 README 结构设计

```
README.md / README_CN.md
├── 顶部徽章 + 多语言切换
├── 一句话定位 + 核心特性卡片
├── 环境准备 ★新增
│   ├── Python 版本要求
│   ├── 包管理工具指南
│   ├── 系统资源最低配置
│   └── 模型下载与存放
├── 安装步骤 ★增强
│   ├── 基础安装
│   ├── 可选依赖安装
│   ├── 环境变量配置
│   └── 验证安装
├── 快速开始 ★增强
│   ├── 最小可运行示例（远程API模式）
│   ├── 本地模型模式示例
│   └── 结果验证
├── 核心概念 ★新增
│   ├── 记忆系统是什么（通俗解释）
│   ├── 关键术语表
│   └── 工作流程图解
├── 核心功能速览
│   ├── 数据管理
│   ├── 记忆构建
│   └── 检索功能
├── 配置选项
│   ├── 环境变量
│   └── YAML 配置
├── 架构概览
├── 性能测试 ★调整
│   ├── 简要说明 + 关键指标
│   └── 链接 → experimental/self_host_benchmarks/locomo/README.md（开发评测说明）
├── 常见问题 (FAQ) ★新增
├── 文档链接
└── 许可证
```

#### 2.2 各章节详细内容规划

**（1）环境准备（全新章节）**

```markdown
## 环境准备

### Python 版本
- 最低要求：Python 3.9+
- 推荐版本：Python 3.10 或 3.11

### 包管理工具
# 使用 pip
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
pip install mandol

# 使用 conda
conda create -n mandol python=3.10
conda activate mandol
pip install mandol

### 系统资源最低配置
| 配置项 | 最低要求 | 推荐配置 |
|--------|---------|---------|
| CPU | 4 核 | 8 核+ |
| RAM | 8 GB（远程模型）/ 16 GB（本地 Embedding）/ 32 GB（本地 Embedding+Reranker） | 16 GB+ / 32 GB+ / 64 GB+ |
| GPU | 无（CPU 可运行） | NVIDIA GPU 8GB+ VRAM（本地推理加速） |
| 磁盘 | 2 GB | 10 GB+（含模型文件） |

### 模型下载与存放
| 模型 | 用途 | 大小 | 下载方式 |
|------|------|------|---------|
| Qwen/Qwen3-Embedding-4B | 文本向量化 | ~4 GB | 首次运行自动下载至 `~/.cache/huggingface/` |
| Qwen/Qwen3-Reranker-4B | 检索重排序 | ~4 GB | 首次运行自动下载至 `~/.cache/huggingface/` |

💡 提示：若使用远程 API 模式，无需下载本地模型，仅需配置 API 端点。
```

**（2）安装步骤（增强）**

```markdown
## 安装

### 基础安装
pip install mandol

### 可选依赖
pip install mandol[faiss]              # FAISS 向量索引加速
pip install mandol[sentence-transformers]  # 本地 Embedding/Reranker 模型
pip install mandol[openai]             # OpenAI API 支持
pip install mandol[milvus]             # Milvus 向量数据库
pip install mandol[neo4j]              # Neo4j 图数据库
pip install mandol[all]                # 安装所有可选依赖

### 环境变量配置
cp .env.example .env
# 编辑 .env 文件，填入 API Key：
# OPENAI_API_KEY=sk-your-key-here

### 验证安装
python -c "from mandol import MemorySystem, MemoryUnit, Uid; print('Mandol 安装成功！')"
```

**（3）快速开始（增强——双模式示例）**

```markdown
## 快速开始

### 模式一：远程 API（推荐新手，零本地模型）

from mandol import MemorySystem, MemoryUnit, Uid

system = MemorySystem.from_yaml_config("config.yaml")
# config.yaml 中配置 use_remote: true 和 API 端点

unit = MemoryUnit(
    uid=Uid("msg_001"),
    raw_data={"text_content": "张三今天去北京出差了"},
    metadata={"timestamp": "2024-01-15T10:00:00"},
)
system.add(unit)

hits = system.holistic_retrieve("张三去了哪里？", top_k=5)
for hit in hits:
    print(f"[{hit.final_score:.3f}] {hit.unit.raw_data['text_content']}")

### 模式二：本地模型（无需 API Key，需下载模型）

from mandol import MemorySystem, MemoryUnit, Uid

system = MemorySystem()  # 默认使用本地 Qwen 模型

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

**（4）核心概念（全新章节）**

```markdown
## 核心概念

### 记忆系统是什么？
想象你有一个智能助手，它能记住你说过的一切，并在需要时精准回忆。
Mandol 就是这样的"记忆大脑"——它不仅存储对话，还能：
- 🧠 自动提取关键信息（人名、地点、事件）
- 🔗 建立信息之间的关联（谁在哪做了什么，因果关系）
- 🔍 精准检索（不只是关键词匹配，而是语义理解）

### 关键术语
| 术语 | 通俗解释 | 类比 |
|------|---------|------|
| MemoryUnit | 一条记忆记录 | 一张便签 |
| MemorySpace | 记忆的分类文件夹 | 文件柜的抽屉 |
| SemanticMap | 记忆的索引系统 | 图书馆的检索卡片 |
| SemanticGraph | 记忆之间的关联网络 | 思维导图 |
| 会话 (Session) | 一次连贯的对话 | 一次会议 |
| 实体 (Entity) | 对话中提到的人/地/物 | 名片 |
| 事件 (Event) | 对话中发生的事情 | 日记条目 |

### 工作流程
[用户输入] → [分块+向量化] → [会话分割] → [提取实体/事件/摘要]
→ [构建关系图] → [检索时：三路召回→融合→扩展→重排] → [返回结果]
```

**（5）性能测试（调整——README 中简要提及，详细内容链接到 benchmarks 目录）**

```markdown
## 性能测试

Mandol 在 LoCoMo（Long Conversational Memory）基准数据集上进行了全面评估，
覆盖单跳、多跳、时序、开放域和对抗性五类查询。

### 关键指标
| 指标 | 说明 |
|------|------|
| F1 Score | 检索准确率与召回率的调和平均 |
| 响应时间 | 从发起查询到返回结果的端到端延迟 |
| 内存占用 | 系统运行时的峰值 RSS |

### 快速复现
cd experimental/self_host_benchmarks/locomo && source scripts/env.sh
python build_graph.py --config configs/base.yaml --output output/
python retrieve.py --config configs/base.yaml --input output/ --output output/

📖 完整的测试环境配置、数据集说明、消融实验及性能对比表格，
请参阅 [LoCoMo 开发评测文档](../experimental/self_host_benchmarks/locomo/README.md)。
```

**（6）常见问题 FAQ（全新章节）**

```markdown
## 常见问题 (FAQ)

### 安装问题
Q: pip install mandol 报错 "No matching distribution found"
A: 请确认 Python 版本 >= 3.9，并尝试 pip install --upgrade pip

Q: 安装 faiss-cpu 失败
A: 尝试 conda install -c conda-forge faiss-cpu 或 pip install faiss-cpu --no-deps

### 运行错误
Q: MemorySystem() 初始化时报 CUDA out of memory
A: 设置环境变量 MANDOL_EMBEDDER_DEVICE=cpu 和 MANDOL_RERANKER_DEVICE=cpu，
   或使用远程 API 模式（USE_REMOTE_EMBEDDER=true）

Q: holistic_retrieve 返回空结果
A: 请确认已调用 build_high_level() 或等待自动构建完成，
   检查是否有足够的记忆数据（建议至少 5 条以上）

Q: LLM API 调用超时
A: 检查 OPENAI_API_KEY 是否正确，网络是否可达 API 端点，
   可设置 MANDOL_LLM_TIMEOUT_S=120 增加超时时间

### 性能优化
Q: 检索速度慢怎么优化？
A: 1) 使用 FAISS 索引加速：pip install mandol[faiss]
   2) 减小 bfs_expansion_hops（默认1→0）
   3) 关闭重排序：holistic_retrieve(query, use_rerank=False)
   4) 使用 GPU 加速 Embedding：MANDOL_EMBEDDER_DEVICE=cuda

Q: 内存占用过高怎么优化？
A: 1) 使用远程 Embedding/Reranker 替代本地模型
   2) 减小 similarity_recent_window
   3) 启用持久化并定期 save/load
```

---

### 三、Sphinx 文档优化方案（面向高级用户/开发者）

#### 3.1 新 Sphinx 文档结构设计

```
docs/
├── conf.py
├── index.rst                          # 文档首页 + 导航
├── getting-started/                   ★新增
│   ├── installation.rst               # 详细安装指南
│   ├── quickstart.rst                 # 快速开始（含双模式）
│   └── configuration.rst              # 配置详解
├── core-concepts/                     ★重组
│   ├── architecture.rst               # 系统架构深入
│   ├── data-flow.rst                  # 数据流向详解
│   ├── memory-model.rst               # 记忆模型理论
│   ├── retrieval-algorithm.rst        # 检索算法原理
│   └── glossary.rst                   # 术语表
├── data-structures/                   # 数据结构（保留+增强）
│   ├── memory_unit.rst
│   ├── memory_space.rst
│   ├── semantic_map.rst
│   ├── semantic_graph.rst
│   └── types.rst
├── user-guide/                        ★新增
│   ├── basic-usage.rst                # 基本使用指南
│   ├── session-management.rst         # 会话管理最佳实践
│   ├── space-strategy.rst             # 空间划分策略
│   ├── graph-management.rst           # 图结构管理高级技巧
│   ├── parameter-tuning.rst           # 参数调优指南
│   ├── persistence-deployment.rst     # 持久化与部署
│   └── scenarios/                     # 场景示例
│       ├── customer-support.rst       # 客服对话场景
│       ├── personal-assistant.rst     # 个人助手场景
│       └── knowledge-base.rst         # 知识库场景
├── retrieval/                         # 检索接口（保留+增强）
│   ├── overview.rst                   # 检索体系概览
│   ├── holistic-retrieve.rst          # 全记忆检索
│   ├── space-retrieve.rst             # 空间检索
│   ├── view-retrieve.rst              # 视角检索
│   ├── hybrid-retriever.rst           # 混合检索器
│   └── custom-retriever.rst           # 自定义检索策略
├── extending/                         # 扩展指南（保留+增强）
│   ├── custom-embedding.rst
│   ├── custom-llm.rst
│   ├── custom-reranker.rst
│   ├── custom-graph-store.rst
│   ├── custom-dimension-builder.rst
│   └── custom-unit-store.rst
├── api-reference/                     ★手工编写（不使用autodoc）
│   ├── memory_system.rst              # MemorySystem 完整 API
│   ├── semantic_map_service.rst       # SemanticMapService 完整 API
│   ├── semantic_graph_service.rst     # SemanticGraphService 完整 API（含预想接口 trace_provenance 等）
│   ├── session_manager.rst            # SessionManager API
│   ├── retrieval_pipeline.rst         # 检索管线 API（含预想接口 smart_quantized_query）
│   ├── config.rst                     # 配置类 API
│   ├── ports.rst                      # 端口接口定义
│   └── types.rst                      # 类型系统
├── advanced/                          ★新增
│   ├── production-deployment.rst      # 生产环境部署
│   ├── milvus-integration.rst         # Milvus 集成指南
│   ├── neo4j-integration.rst          # Neo4j 集成指南
│   ├── performance-optimization.rst   # 性能优化指南
│   └── migration-guide.rst            # 版本迁移指南
├── contributing/                      ★新增
│   ├── development-setup.rst          # 开发环境搭建
│   ├── code-style.rst                 # 代码规范
│   ├── testing.rst                    # 测试指南
│   └── documentation.rst              # 文档贡献指南
└── en/                                # 英文镜像
    └── (同上结构)
```

#### 3.2 场景示例详细规划（至少3个）

**场景一：客服对话记忆系统**

```python
"""
场景：电商客服对话记忆系统
目标：记住用户的历史订单、偏好、投诉记录，提供个性化服务
"""
from mandol import MemorySystem, MemoryUnit, Uid

system = MemorySystem.from_yaml_config("config.yaml")

# 添加多轮客服对话
conversations = [
    MemoryUnit(uid=Uid("cs_001"), raw_data={"text_content": "我想退昨天买的蓝色运动鞋，尺码不合适", "speaker": "customer"}, metadata={"timestamp": "2024-03-10T14:00:00"}),
    MemoryUnit(uid=Uid("cs_002"), raw_data={"text_content": "好的，已为您提交退货申请，预计3-5个工作日退款", "speaker": "agent"}, metadata={"timestamp": "2024-03-10T14:01:00"}),
    MemoryUnit(uid=Uid("cs_003"), raw_data={"text_content": "下次我想买42码的同一款，有货吗？", "speaker": "customer"}, metadata={"timestamp": "2024-03-10T14:02:00"}),
]
for unit in conversations:
    system.add(unit)

system.build_high_level(mode="auto")

# 检索：用户之前买过什么？
hits = system.holistic_retrieve("这个客户之前买了什么鞋？", top_k=3)
# 预期结果：返回关于蓝色运动鞋购买的对话记录

# 按视角检索：用户偏好
preferences = system.retrieve_by_view("客户喜欢什么？", view="knowledge", top_k=5)
# 预期结果：返回知识摘要中的用户偏好实体

# 按视角检索：事件因果
events = system.retrieve_by_view("退货的原因是什么？", view="event_causal", top_k=5)
# 预期结果：返回"尺码不合适→退货"的因果链
```

**场景二：个人助手长期记忆**

```python
"""
场景：个人助手长期记忆
目标：记住用户的习惯、日程、人际关系，提供主动提醒和建议
"""
from mandol import MemorySystem, MemoryUnit, Uid

system = MemorySystem.from_yaml_config("config.yaml")

# 跨会话添加记忆
# 会话1：用户谈论工作
session1 = [
    MemoryUnit(uid=Uid("pa_001"), raw_data={"text_content": "我下周二要和客户做项目汇报", "speaker": "user"}, metadata={"timestamp": "2024-03-11T09:00:00", "session_id": "s1"}),
    MemoryUnit(uid=Uid("pa_002"), raw_data={"text_content": "汇报内容是Q1的销售数据分析", "speaker": "user"}, metadata={"timestamp": "2024-03-11T09:01:00", "session_id": "s1"}),
]
# 会话2：用户谈论生活（时间间隔 > session_time_gap_seconds）
session2 = [
    MemoryUnit(uid=Uid("pa_003"), raw_data={"text_content": "周末想去爬山，推荐一下附近的路线", "speaker": "user"}, metadata={"timestamp": "2024-03-16T10:00:00", "session_id": "s2"}),
]
for unit in session1 + session2:
    system.add(unit)

system.build_high_level(mode="auto")

# 跨会话检索：综合工作和生活信息
hits = system.holistic_retrieve("我最近有什么安排？", top_k=5)
# 预期结果：返回项目汇报（工作）+ 爬山计划（生活）

# 检索特定空间：仅查实体
entities = system.retrieve_in_space("客户", space_name="root_knowledge_entity", top_k=5)
```

**场景三：知识库问答系统**

```python
"""
场景：企业知识库问答
目标：将文档/FAQ导入记忆系统，支持精准知识检索
"""
from mandol import MemorySystem, MemoryUnit, Uid

system = MemorySystem.from_yaml_config("config.yaml")

# 批量导入知识文档
knowledge_units = [
    MemoryUnit(uid=Uid("kb_001"), raw_data={"text_content": "公司年假政策：入职满1年可享10天年假，满5年15天，满10年20天"}, metadata={"timestamp": "2024-01-01T00:00:00", "source": "hr_policy"}),
    MemoryUnit(uid=Uid("kb_002"), raw_data={"text_content": "报销流程：填写报销单→部门经理审批→财务审核→打款，周期约5个工作日"}, metadata={"timestamp": "2024-01-01T00:00:00", "source": "finance_policy"}),
    MemoryUnit(uid=Uid("kb_003"), raw_data={"text_content": "远程办公规定：每周最多2天远程，需提前一天在OA系统申请"}, metadata={"timestamp": "2024-01-01T00:00:00", "source": "hr_policy"}),
]
system.add_many(knowledge_units)
system.build_high_level(mode="auto")

# 语义检索：即使用词不同也能找到
hits = system.holistic_retrieve("我可以在家办公吗？", top_k=3)
# 预期结果：返回远程办公规定（kb_003）

# 知识视角检索
knowledge = system.retrieve_by_view("休假有多少天？", view="knowledge", top_k=3)
# 预期结果：返回年假政策的知识摘要
```

#### 3.3 核心概念深入解释规划

**系统架构深入**（architecture.rst）：
- 六边形架构的分层设计与依赖方向图
- 各层职责与交互关系
- 与传统多库异构架构的对比（附图）

**数据流向详解**（data-flow.rst）：
- 从用户输入到检索输出的完整数据流图（Mermaid 序列图）
- 各阶段的数据变换：原始文本 → 分块 → 向量化 → 索引 → 检索 → 排序
- 数据在各组件间的流转路径

**检索算法原理**（retrieval-algorithm.rst）：
- Dense 检索：向量相似度计算原理（余弦相似度 / 内积）
- BM25 检索：TF-IDF 变体与关键词匹配
- Sparse 检索：SPLADE 稀疏向量表示
- RRF 融合：倒数排名融合公式与权重计算
- BFS 扩展：图遍历策略与扩展深度选择
- Cross-Encoder 重排序：交叉编码器原理与延迟权衡

#### 3.4 高级用户指南详细规划

**自定义空间划分策略**（space-strategy.rst）：
- SpaceNamingPolicy 的工作机制
- 自定义命名策略的实现方法
- 不同场景的空间划分建议（对话 vs 文档 vs 代码）

**图结构管理高级技巧**（graph-management.rst）：
- 批量关系操作（add_relation / delete_relationship）
- 图修剪策略（移除低权重边、孤立节点清理）
- 图合并（跨系统合并语义图）
- 图可视化方法

**会话划分最佳实践**（session-management.rst）：
- 基于时间的会话分割（session_time_gap_seconds 调优）
- 基于 LLM 的智能会话分割
- 不同场景的推荐配置：
  - 客服对话：短会话（gap=300s）
  - 个人助手：中等会话（gap=1800s）
  - 知识库：无会话分割（gap=∞）

**参数调优方法**（parameter-tuning.rst）：

| 参数 | 默认值 | 调优建议 | 影响范围 |
|------|--------|---------|---------|
| `chunk_max_tokens` | 512 | 短对话可降至256；长文档可升至1024 | 分块粒度 |
| `session_time_gap_seconds` | 1800 | 客服场景300；个人助手1800；知识库86400 | 会话分割灵敏度 |
| `similarity_threshold` | 0.7 | 提高则更精确但召回少；降低则召回多但噪声多 | 隐式边密度 |
| `bfs_expansion_per_seed` | 3 | 增大可发现更多关联但延迟增加 | 图扩展广度 |
| `bfs_expansion_hops` | 1 | 设为0关闭图扩展；2+适合多跳问答 | 图扩展深度 |
| `max_entities_per_llm` | 50 | 增大可提取更多实体但LLM成本增加 | 实体提取数量 |
| `coref_vector_threshold` | 0.45 | 提高则跨会话合并更严格 | 共指消解精度 |

#### 3.5 开发者接口文档完整规划

**API 文档策略：手工编写 + 预想接口就近放置**

由于项目中存在预想接口（planned interfaces）需要保留文档，不使用 Sphinx autodoc 自动生成。
所有 API 文档采用手工编写 .rst 文件的方式，确保：

1. **已实现接口**：完整文档（签名/参数/返回值/异常/示例）
2. **预想接口**：标注状态标签，保留设计文档（签名/返回值类型/应用场景/设计理念）
3. **就近放置**：预想接口放在相关已实现接口旁边，保持上下文连贯，用醒目的状态标签区分

**接口状态标签体系**：

.. code-block:: rst

   ✅ 已实现    — 接口已实现并经过测试，可直接使用
   🔧 实验性    — 接口已实现但 API 可能变更
   📋 预想接口  — 接口尚未实现，文档描述目标设计

**就近放置示例**：

在 `semantic_graph_service.rst` 中：

.. code-block:: rst

   get_explicit_neighbors
   ^^^^^^^^^^^^^^^^^^^^^^^

   .. status-badge:: ✅ 已实现

   获取指定单元的显式关系邻居。

   ...（已实现接口的完整文档）

   trace_provenance
   ^^^^^^^^^^^^^^^^

   .. status-badge:: 📋 预想接口（后续实现）

   高阶记忆溯源分析：追溯任意记忆单元的完整证据链路。

   ...（预想接口的设计文档）

在 `retrieval_pipeline.rst` 中：

.. code-block:: rst

   holistic_retrieve
   ^^^^^^^^^^^^^^^^^

   .. status-badge:: ✅ 已实现

   全记忆检索接口。

   ...（已实现接口的完整文档）

   smart_quantized_query
   ^^^^^^^^^^^^^^^^^^^^^^

   .. status-badge:: 📋 预想接口（后续实现）

   智能量化查询接口。

   ...（预想接口的设计文档）

**API 文档覆盖清单**：

| 模块 | 类/接口 | 当前状态 | 需补充内容 |
|------|---------|---------|-----------|
| application | MemorySystem | ❌ 无文档 | 完整 API：构造函数、所有公开方法、配置类、返回值类型、异常 |
| application | SemanticMapService | ⚠️ 仅 .md 概览 | 完整方法签名、参数说明、返回值、异常、使用示例 |
| application | SemanticGraphService | ⚠️ 仅 .md 概览 | 完整方法签名、参数说明、返回值、异常、使用示例 |
| application | SessionManager | ❌ 无文档 | 完整 API |
| application | MultiDimSemanticGraph | ❌ 无文档 | 维度构建器接口与注册机制 |
| domain | MemoryUnit | ⚠️ .rst 有 | 补充异常说明、边界情况 |
| domain | MemorySpace | ⚠️ .rst 有 | 补充异常说明 |
| domain | types | ❌ 无文档 | Uid/SpaceName/Embedding/SearchHit 类型说明 |
| infrastructure | config | ❌ 无文档 | 所有配置类完整字段说明 |
| ports | 全部接口 | ❌ 无文档 | 6个端口接口的完整定义 |
| retrieval | HybridRetriever | ⚠️ .rst 提及 | 完整 API |
| retrieval | BM25Retriever | ⚠️ .rst 提及 | 完整 API |
| retrieval | SparseRetriever | ⚠️ .rst 提及 | 完整 API |
| retrieval | SubgraphHopRetriever | ⚠️ .rst 提及 | 完整 API |
| planned | trace_provenance | 📋 预想 | 保留设计文档，标注状态 |
| planned | compare_multi_view_consistency | 📋 预想 | 保留设计文档，标注状态 |
| planned | analyze_entity_lifecycle | 📋 预想 | 保留设计文档，标注状态 |
| planned | extract_event_narrative_chain | 📋 预想 | 保留设计文档，标注状态 |
| planned | smart_quantized_query | 📋 预想 | 保留设计文档，标注状态 |
| planned | BaseMultiViewRetriever | 📋 预想 | 保留设计文档，标注状态 |

**API 文档格式规范**（每个方法必须包含）：

.. code-block:: rst

   method_name
   ^^^^^^^^^^^^

   .. status-badge:: ✅ 已实现

   一句话描述。

   **签名**：

   .. code-block:: python

      def method_name(self, param1: Type1, param2: Type2 = default) -> ReturnType

   **参数**：

   - ``param1`` (``Type1``)：参数1说明
   - ``param2`` (``Type2``，默认 ``default``)：参数2说明

   **返回值**：

   ``ReturnType``：返回值说明

   **异常**：

   - ``ValueError``：当 xxx 时抛出
   - ``RuntimeError``：当 xxx 时抛出

   **使用示例**：

   .. code-block:: python

      result = system.method_name(param1="value")
      print(result)  # 输出: ...

   **参见**：

   - :ref:`related-section`

**预想接口文档格式**：

.. code-block:: rst

   trace_provenance
   ^^^^^^^^^^^^^^^^

   .. status-badge:: 📋 预想接口（后续实现）

   高阶记忆溯源分析：追溯任意记忆单元的完整证据链路。

   **设计目标**：

   从指定单元出发，沿 EVIDENCED_BY 边递归回溯，
   构建完整的证据溯源树，展示该记忆单元的数据来源和推导过程。

   **目标签名**：

   .. code-block:: python

      def trace_provenance(
          uid: str,
          max_depth: int = 5,
          include_coref: bool = True
      ) -> ProvenanceTree

   **目标返回值类型**：

   ...（保留现有设计文档内容）

   **应用场景**：

   - Agent 需要解释其回答或决策的依据时
   - 验证某个结论是否有足够的证据支撑

   **实现依赖**：

   - 需要 EVIDENCED_BY 边的完整构建
   - 需要 COREF 边的跨会话合并

---

### 四、文档体系全面优化措施

#### 4.1 文档结构调整

| 措施 | 说明 | 优先级 |
|------|------|--------|
| 新增 getting-started 目录 | 将安装、快速开始、配置从 README 扩展为独立文档 | P0 |
| 新增 user-guide 目录 | 高级使用指南，含场景示例和最佳实践 | P0 |
| 新增 advanced 目录 | 生产部署、性能优化、集成指南 | P1 |
| 新增 contributing 目录 | 开发贡献指南 | P1 |
| 重组 api-reference | 手工编写 .rst 文件，区分已实现/预想接口 | P0 |
| 拆分 retrieval_interfaces.rst | 当前 1590 行过长，按检索层级拆分为 5-6 个文件 | P1 |
| 增强 experimental/self_host_benchmarks/locomo/README.md | 开发评测说明移至此处，README 仅简要提及+链接 | P0 |
| 英文文档同步 | docs/en/ 下镜像中文文档结构 | P2 |

#### 4.2 内容补充清单

| 补充项 | 目标文档 | 优先级 |
|--------|---------|--------|
| 环境准备章节（Python版本/包管理/系统配置/模型下载） | README | P0 |
| 双模式快速开始（远程API/本地模型） | README + getting-started | P0 |
| 核心概念通俗解释 + 术语表 | README + core-concepts | P0 |
| FAQ（安装/运行/性能） | README | P0 |
| 性能测试简要说明 + 链接到 benchmarks | README | P0 |
| experimental/self_host_benchmarks/locomo/README.md 开发评测说明 | experimental | P0 |
| 3个完整场景示例 | user-guide/scenarios | P0 |
| 参数调优指南 | user-guide/parameter-tuning | P0 |
| 会话划分最佳实践 | user-guide/session-management | P0 |
| MemorySystem 完整 API 文档 | api-reference | P0 |
| 配置类完整文档 | api-reference/config | P0 |
| 端口接口完整文档 | api-reference/ports | P1 |
| 预想接口文档（就近放置，标注状态） | api-reference 各相关文件 | P1 |
| 检索算法原理深入 | core-concepts/retrieval-algorithm | P1 |
| 数据流向详解 | core-concepts/data-flow | P1 |
| Milvus/Neo4j 集成指南 | advanced/ | P1 |
| 生产部署指南 | advanced/production-deployment | P1 |
| 性能优化指南 | advanced/performance-optimization | P1 |
| 开发环境搭建 | contributing/development-setup | P2 |
| ADR 架构决策记录 | contributing/ | P2 |

#### 4.3 示例完善

| 示例 | 类型 | 优先级 | 说明 |
|------|------|--------|------|
| 客服对话记忆系统 | 场景示例 | P0 | 含完整代码+注释+预期输出 |
| 个人助手长期记忆 | 场景示例 | P0 | 含跨会话操作+预期输出 |
| 知识库问答系统 | 场景示例 | P0 | 含批量导入+语义检索+预期输出 |
| 自定义 Embedding Provider | 扩展示例 | P1 | 含完整实现+注册+验证 |
| 自定义 DimensionBuilder | 扩展示例 | P1 | 含完整实现+注册+效果 |
| Milvus 生产部署 | 部署示例 | P1 | 含 docker-compose + 配置 |
| Neo4j 图存储集成 | 集成示例 | P1 | 含配置+数据迁移 |

#### 4.4 术语统一

| 当前问题 | 统一方案 |
|---------|---------|
| "语义地图"/"SemanticMap"/"SemanticMapService" 混用 | 面向用户统一称"语义地图 (SemanticMap)"，API 文档用类名 |
| "语义图"/"SemanticGraph"/"SemanticGraphService" 混用 | 面向用户统一称"语义图 (SemanticGraph)"，API 文档用类名 |
| "高阶记忆"/"高层记忆"/"High-Level Memory" 混用 | 统一为"高阶记忆 (High-Level Memory)" |
| "全记忆检索"/"统一检索"/"Holistic Retrieve" 混用 | 统一为"全记忆检索 (Holistic Retrieve)" |
| types.py 路径引用错误（`src/memory/domain/types.py`） | 修正为 `mandol/domain/types.py` |
| "记忆单元"/"MemoryUnit" 混用 | 首次出现用"记忆单元 (MemoryUnit)"，后续统一用 MemoryUnit |

#### 4.5 Sphinx 配置优化

| 措施 | 说明 |
|------|------|
| 不使用 autodoc | 手工编写 API 文档，保留预想接口 |
| 保持 furo 主题 | 继续使用当前配置的 furo 主题，现代简洁风格 |
| 添加 intersphinx | 链接到 Python 标准库、numpy 等外部文档 |
| 添加 mermaid 支持 | 已有 sphinxcontrib-mermaid，确保所有 Mermaid 图正常渲染 |
| 添加 copybutton | 代码块一键复制 |
| 添加 page-toc | 页面内目录导航 |
| 接口状态标签 | 使用 Sphinx admonition 指令实现，无需自定义扩展 |

**接口状态标签的 Sphinx admonition 实现**：

已实现接口使用 `.. note::`：

.. code-block:: rst

   get_explicit_neighbors
   ^^^^^^^^^^^^^^^^^^^^^^^

   .. note:: ✅ 已实现 — 此接口已实现并经过测试，可直接使用。

   获取指定单元的显式关系邻居。

预想接口使用 `.. warning::`（更醒目）：

.. code-block:: rst

   trace_provenance
   ^^^^^^^^^^^^^^^^

   .. warning:: 📋 预想接口 — 此接口尚未实现，以下文档描述目标设计，API 可能变更。

   高阶记忆溯源分析：追溯任意记忆单元的完整证据链路。

实验性接口使用 `.. caution::`：

.. code-block:: rst

   some_experimental_api
   ^^^^^^^^^^^^^^^^^^^^^^

   .. caution:: 🔧 实验性 — 此接口已实现但 API 可能变更，不建议生产环境使用。

---

### 五、实施路线图

| 阶段 | 时间 | 任务 | 交付物 |
|------|------|------|--------|
| **Phase 1：README 优先** | 第1周 | README 环境准备+安装+双模式快速开始+FAQ+核心概念+性能测试链接；experimental/self_host_benchmarks/locomo/README.md 开发评测说明 | 更新的 README.md / README_CN.md + experimental/self_host_benchmarks/locomo/README.md |
| **Phase 2：Sphinx 拆分重组** | 第2周 | 拆分现有 5 个 .rst 为新目录结构；retrieval_interfaces.rst 拆分为 retrieval/ 下 5-6 个文件；添加状态标签 | 重组的 docs/ 目录结构 |
| **Phase 3：场景示例+高级指南** | 第3周 | 3个完整场景示例 + 参数调优指南 + 会话管理最佳实践 | user-guide/scenarios/ + parameter-tuning.rst + session-management.rst |
| **Phase 4：API 文档补全** | 第4周 | MemorySystem + SemanticMapService + SemanticGraphService 完整 API（手工编写，预想接口就近放置） | api-reference/ 手工 .rst 文档 |
| **Phase 5：高级+贡献+英文** | 第5-6周 | 生产部署+Milvus/Neo4j集成+性能优化+贡献指南+英文完整镜像 | advanced/ + contributing/ + docs/en/ |

---

### 六、质量保障措施

1. **文档审查清单**：每个文档发布前检查——可运行性、术语一致性、链接有效性、示例完整性
2. **CI 集成**：在 GitHub Actions 中添加文档构建检查（`docs.yml` 已有，需确保手工 .rst 构建无错误）
3. **用户反馈循环**：在文档中添加"本文档有帮助吗？"反馈按钮，收集用户痛点
4. **版本同步**：每次发版时同步更新 API 文档和 CHANGELOG
5. **定期审计**：每季度审计一次文档与代码的一致性

---

## 第三部分：修订记录

| 版本 | 日期 | 修订内容 |
|------|------|---------|
| v1.0 | 2026-04-28 | 初版：完整评估报告与优化方案 |
| v1.1 | 2026-04-28 | 根据反馈修订：(1) 性能测试复现说明移至 benchmarks 目录，README 仅简要提及+链接；(2) 不使用 autodoc，改为手工编写 API 文档以保留预想接口；(3) 新增接口状态标签体系（✅已实现/🔧实验性/📋预想接口） |
| v1.2 | 2026-04-28 | 根据反馈修订：预想接口采用"就近放置"策略，放在相关已实现接口旁边，用醒目的状态标签区分；取消 planned-interfaces.rst 集中管理方式 |
| v1.3 | 2026-04-28 | 根据反馈确认：(1) 英文文档采用完整镜像策略；(2) Sphinx 主题保持 furo；(3) 接口状态标签使用 Sphinx admonition 指令实现 |
