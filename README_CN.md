[English](README.md) | [中文](README_CN.md)

# Mandol

> 以内存为核心的高效精确智能体分层记忆系统

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Docs](https://img.shields.io/badge/docs-Sphinx-green.svg)](docs/)

![Mandol 分层架构](README.assets/Mandol%20framework.png)

---

## 📖 Mandol 是什么？

Mandol 是一套以内存为核心、具备高效精确检索能力的智能体分层记忆系统，实现复杂记忆信息的统一表示、高效存储与高效精确检索，为下一代智能体认知架构提供理论支撑与技术方案。

系统基于纯 Python 内存数据结构，融合键值、向量与图三种索引范式，提供统一的存储与混合检索接口，无需外部依赖即可运行。向下可按需桥接 Milvus、Neo4j 等外部存储引擎，向上提供 `add()` → `holistic_retrieve()` 的极简操作模型。其核心创新在于将传统「被动召回-排序」检索范式转变为「智能路由 → 量化去噪 → 高质量上下文生成」主动检索新范式。

**在主流对话记忆基准上，Mandol 以较低的 Token 消耗实现了 SOTA 级别的综合表现：**

| 维度 | Mandol | EverMemOS | Zep | Mem0 |
|------|--------|-----------|-----|------|
| **LoCoMo 评分** | **92.21** (1.9k Token) | 91.97 (2.7k Token) | 85.22 (1.4k Token) | 64.20 (1.0k Token) |
| **LongMemEval 评分** | **88.40** (2.3k Token) | 83.00 (2.8k Token) | 63.80 (1.6k Token) | 66.40 (1.1k Token) |
| **检索方法** | 智能路由 + 三路混合召回 + BFS 图扩展 | 语义 + BM25 | 语义 + BM25 + 时间图遍历 | 语义 + BM25 + 实体匹配 |
| **图关系建模** | 显式实体关系图 + 事件因果链 | 扁平情景记忆 | 三层时间知识图谱 | 知识图谱 |
| **部署模式** | 纯内存 / FAISS / Milvus / Neo4j | 云平台 | 云托管 / 自托管 | 库 / 自托管 / 云平台 |
| **开源协议** | Apache 2.0 | — | Apache 2.0 | Apache 2.0 |

> Mandol 以 1.9k Token 达到 LoCoMo 92.21 分——Token 效率是同等精度系统 EverMemOS（2.7k）的 1.4 倍，是 Mem0 v2.0（7.0k）的 3.7 倍。LongMemEval 上以 2.3k Token 达到 88.40 分，较 EverMemOS（2.8k / 83.00）在 Token 减少 18% 的同时评分提升 5.4 个百分点。

---

## ✨ 核心特性

### 轻量级架构

纯 Python 实现，核心逻辑采用六边形架构（端口-适配器模式），`MemorySystem()` 无参构造即可启动完整记忆系统，零外部依赖。向下可按需桥接 FAISS（向量加速）、Milvus（向量数据库）、Neo4j（图数据库）等外部引擎，通过 YAML 配置即可切换，无需修改业务代码。

### 简单易用

三步操作模型覆盖核心流程：`add()` 写入记忆 → `build_high_level()` 构建高阶结构 → `holistic_retrieve()` 混合检索。`save()` / `load()` 一键持久化与恢复，所有状态导出为 JSON 目录。远程 API 模式下无需下载本地模型，仅需配置 API 端点即可快速体验。

```python
from mandol import MemorySystem, MemoryUnit, Uid

system = MemorySystem.from_yaml_config("config.yaml")

system.add(MemoryUnit(
    uid=Uid("msg_001"),
    raw_data={"text_content": "张三今天去北京出差了"},
    metadata={"timestamp": "2024-01-15T10:00:00"},
))

system.build_high_level(mode="auto")

hits = system.holistic_retrieve("张三去了哪里？", top_k=5)
for hit in hits:
    print(f"[{hit.final_score:.3f}] {hit.unit.raw_data['text_content']}")

system.save("./memory_snapshot")
```

### 统一记忆表示

单一 `MemoryUnit` 抽象统一承载文本内容（`text_content`）与图像（`image_path`）等异构信息，自动完成向量化。`MemorySpace` 树形层级支持按 BASE / ENTITY / EVENT / SUMMARY 等维度灵活组织记忆。`SemanticGraph` 以有向图显式建模实体间关系（`RELATED_TO`、`PART_OF` 等）与事件因果链（`CAUSED_BY`、`HAPPENED_AFTER` 等），支持多跳图遍历检索。

### 层级化记忆结构

- **基础记忆层（Base）**：原始数据片段，`add()` 后立即可检索
- **高阶记忆层（High-Level）**：系统自动完成会话分割（LLM 驱动）、实体提取与去重、事件提取与去重、实体关系构建、事件因果链构建、多类型摘要生成（情景 / 知识 / 情感 / 过程）及全局洞察提取
- **跨会话共指消解**：自动合并跨会话的同一实体和事件，维护一致的知识表示

---

## 🎯 核心能力

### 记忆信息的统一表示

将文本、图像等异构信息统一建模为 `MemoryUnit`，按语义维度组织到可扩展的 `MemorySpace` 树形层级中，同时构建显式的实体关系图和事件因果链。

`MemoryUnit` 封装原始数据、向量表示和元数据——`text_content` 自动生成 Dense 向量，`image_path` 自动提取图像向量，其他自定义字段作为元数据存储。`MemorySpace` 支持父子空间嵌套，可按应用需求灵活扩展记忆维度。`SemanticGraph` 在上述空间之上维护有向图结构，将实体间语义关联和事件间时序因果关系显式化，为图检索扩展提供基础。

### 低延迟统一存储与混合查询

向量索引、BM25 关键词索引、TF-IDF 稀疏索引和图索引均在内存中统一维护，单次查询同时覆盖多路召回通道，实现毫秒级混合检索。

检索流程：三路并行召回（Dense ANN + BM25 + TF-IDF 稀疏）→ RRF（Reciprocal Rank Fusion）融合三路独立排序 → BFS 图扩展（以融合后的 Top 种子节点为起点沿关系边多跳遍历）→ Cross-Encoder Reranker 对全局合并去重后的候选集精排。四组记忆空间（BASE / ENTITY / EVENT / SUMMARY）各自独立执行上述流水线，最终合并去重后统一重排。

```python
# 全记忆混合检索
hits = system.holistic_retrieve("query", top_k=10)

# 按语义视图精确检索
hits = system.retrieve_by_view("entity_relation", "query")   # 实体关系图遍历
hits = system.retrieve_by_view("event_causal", "query")      # 事件因果链追溯
hits = system.retrieve_by_view("episodic", "query")          # 情景摘要检索
```

### 高效精确检索与 Token 开销优化

通过智能路由、多级量化和自适应索引，在保证检索精度的同时显著降低送入 LLM 的上下文 Token 消耗。

这一流程分为三个紧密衔接的阶段：

**第一阶段 — 智能路由与多源并行检索**：为不同记忆源（基础对话、事件链、实体关系）构建独立并行的检索管道，设计轻量级查询意图分类器，依据查询语义特征动态决定各记忆源的参与程度，以最小计算代价实现最大化信息覆盖。

**第二阶段 — 量化去噪与冲突消解**：在各记忆源内部，基于语义相关性统计分布建立自适应阈值，过滤低相关性候选，去除噪声。进而构建跨源冲突消解模块，对不同记忆源返回的候选结果进行重叠检测和语义矛盾分析，利用融合相关性、可信度和时间因素的综合评估分数进行消歧去重，完成检索结果的精确筛选。

**第三阶段 — Token 约束下的高质量上下文生成**：采用相关性与多样性联合优化的上下文选择算法，在最大化语义相关性的同时，引入信息源多样性约束，确保上下文覆盖不同来源的证据。配合 Token 弹性适配器，根据查询复杂度动态调整相关性与多样性的平衡权重，生成紧凑的高信息密度上下文。

在 LoCoMo 基准上，Mandol 以平均 1.9k Token 实现 92.21 的综合评分，而 Mem0 需 7.0k Token 才达到 91.6——Mandol 以 73% 更少的 Token 消耗实现了更高的检索精度。

---

## 🚀 相比竞品优势

（竞品对比数据详见上方「📖 Mandol 是什么？」章节。）

### 创新检索范式

传统记忆系统将检索视为单向流水线（Embedding 召回 → Rerank 排序 → Top-K），检索过程被动、单一且缺乏对噪声的控制。Mandol 将这一范式重构为「智能路由 → 量化去噪 → 高质量上下文生成」主动检索新范式：首先依据查询意图动态路由到最相关的记忆源，然后在各源内部及跨源之间进行多级量化过滤与冲突消解，最后在 Token 约束下生成高信息密度上下文。这一范式转变使检索从被动的「匹配-返回」升级为主动的「理解-筛选-归纳」。

### 智能检索流水线：路由 → 去噪 → 上下文生成

Mandol 的检索并非传统的「统一召回-统一排序」，而是一条三阶段的智能流水线，每个阶段解决一个核心问题：

**第一阶段 · 意图路由 — 解决「去哪找」**：Mandol 构建了 BASE / ENTITY / EVENT / SUMMARY 四组并行检索管道，通过轻量级意图分类器动态决定各管道的参与程度和资源分配。系统同时提供 `retrieve_by_view()` 接口，支持 8 种语义视图的精确检索调度——`base_memory`（事实匹配）、`entity_relation`（实体关系图遍历）、`event_causal`（事件因果链追溯）、`episodic`（情景摘要）、`knowledge`（知识摘要）、`procedural`（过程摘要）、`emotional`（情感摘要）、`insights`（全局洞察）——每种视图映射到最优的检索路径，避免无关空间的无效计算。

**第二阶段 · 量化去噪 — 解决「哪些真正有用」**：路由完成后，三层过滤依次生效。第一层，各检索管道内部基于语义相关性统计分布建立自适应阈值，过滤低相关性候选；第二层，RRF 融合多路独立排序信号，Cross-Encoder Reranker 在全局候选集上进行成对精细打分，实现多路信号的精确校准；第三层，跨源重叠检测与语义矛盾分析模块，基于相关性、可信度和时间因素的综合评分消歧去重。同时，LLM 驱动的实体/事件去重模块在语义层面消除冗余，确保同一实体/事件在系统中的唯一表示。

**第三阶段 · Token 约束下的上下文生成 — 解决「如何用最少 Token 传递最多信息」**：筛选后的候选进入上下文构造阶段。Mandol 采用相关性与多样性联合优化策略——在最大化语义相关性的同时引入信息源多样性约束，确保上下文覆盖不同来源的证据而非集中在单一高相关区域。Token 弹性适配器根据查询复杂度动态调整两者平衡权重：简单事实查询侧重相关性，复杂推理查询增强多样性。最终效果：LoCoMo 上 1.9k Token 达到 92.21 分，而同等精度的 Mem0 v2.0 消耗 7.0k Token（Mandol 节省 73%），EverMemOS 消耗 2.3k Token（Mandol 节省 17% 且评分更高）。

### 多底层数据库支持

六边形架构（端口-适配器模式）实现核心逻辑与存储后端的完全解耦。同一套 API 可切换不同的底层基础设施：向量索引（内存精确搜索 → FAISS ANN 自适应切换）、图存储（内存 → Neo4j）、单元存储（内存 → Milvus）、Embedding / Reranker（本地 SentenceTransformers 模型 → 远程 OpenAI 兼容 API）。所有后端切换仅需修改 YAML 配置文件，业务代码零改动。

```yaml
# 切换后端示例：从本地模型切换至远程 API
embedder:
  use_remote: true
  base_url: "https://api.example.com/v1"

# 切换图存储至 Neo4j
graph_store:
  backend: neo4j
  uri: "bolt://localhost:7687"
```

---

## 🏆 经典应用案例

### 长对话记忆基准 LoCoMo —— 全面领先的复杂记忆推理

在 LoCoMo 基准（10 段长对话 × 200+ 轮交互，覆盖单跳/多跳/时序/开放域查询）中，Mandol 以 **92.21 的综合评分**超越 Zep（85.22）、Mem0（64.20）、MemOS（80.76），与 EverMemOS（91.97）持平。核心优势体现在**多跳推理**（92.20 分，所有系统中最高）——这得益于 `SemanticGraph` 的显式实体关系图与 BFS 图扩展机制，能够沿关系边多跳遍历发现非直接关联的证据。同时 **Token 消耗仅 1.9k**，比同等精度的 EverMemOS（2.3k）低 17%，比 Mem0 v2.0（7.0k）低 73%。

> 当查询「张经理去年的决策对今年 Q2 的项目延期有何影响」时，Mandol 沿事件因果链 `决策A → 团队调整 → 资源转移 → 项目B延期 → Q2交付推迟` 完成 4 跳追溯，而纯向量检索仅能返回包含「张经理」「Q2」等关键词的孤立片段。

### 长记忆评估基准 LongMemEval —— 跨会话记忆保持的新标杆

LongMemEval 侧重多会话场景下的记忆保持与知识更新能力。Mandol 以 **88.40 综合评分**较 EverMemOS（83.00）**提升 5.4 个百分点**。在助手侧记忆（SS-Asst 98.21）和用户侧记忆（SS-User 98.57）两个子项上接近满分，证明系统在多轮交互中几乎无信息遗漏。知识更新评分 89.74——当同一事实存在新旧两个版本时，系统准确采纳新信息并消解冲突，验证了跨会话共指消解与「优先采纳新信息」策略的有效性。

> 在 LongMemEval 的知识更新场景中，用户先告知「我住在北京」，三天后告知「我搬到上海了」。Mandol 正确返回上海地址，而向量相似度检索有概率因「北京」的历史出现频次更高而错误返回旧地址——这正是显式时序因果图相较纯向量检索的结构性优势。

### 智能客服 —— 多维关联查询，一次解决

多轮客服对话中，用户问题跨越多轮甚至多天。当用户询问「昨天买的蓝色衬衫降价了怎么办」，系统需同时关联**时序事件**（降价发生时间）、**商品属性**（蓝色衬衫 SKU）、**用户信息**（购买记录、会员等级）三个维度的记忆。

Mandol 的方案：`retrieve_by_view("event_causal")` 沿价格变动事件链追溯（调价时间 → 原因 → 幅度 → 适用规则），`retrieve_by_view("entity_relation")` 检索用户-商品-订单的实体关联网络，`ask()` 综合生成包含「您的订单符合价保规则，可退差价 ¥35」的准确回复，无需人工查询订单号和商品 SKU。对比纯关键词匹配系统仅能返回含「降价」「衬衫」的 FAQ 条目，Mandol 的多维关联查询直接锁定具体订单和适用策略，提升一次解决率。

### 软件开发 —— 跨源混合检索，分钟级根因定位

当开发者请求「分析支付模块异常与近一周上线功能的关联」，信息分散在 PR 讨论、Issue 评论、变更日志和设计文档中。单一检索源无法覆盖如此分散的关联信息。

Mandol 的方案：`holistic_retrieve()` 跨 BASE/ENTITY/EVENT/SUMMARY 四组空间并行检索，`SemanticGraph` 自动构建模块-函数-开发者-版本关联图，`retrieve_by_view("event_causal")` 将上线事件时间线与异常事件时间线自动对齐。检索结果涵盖「支付模块新增的汇率转换函数」（代码变更）、「PR #342 中关于异常处理的讨论」（讨论上下文）、「v2.3.1 上线时间与首次异常报错间隔 12 分钟」（时序关联）——跨源混合检索将根因分析从天级缩短至分钟级。

### 医疗 —— 即时多维融合，辅助紧急决策

医生请求「对服用阿司匹林后发热的患者提供紧急检查支持」时，关键信息分散在跨科室病历、用药记录和检查报告中，任何遗漏都可能影响诊断质量。

Mandol 的方案：`retrieve_by_view("entity_relation")` 检索患者-药物-症状关联网络，`retrieve_by_view("event_causal")` 追溯用药→症状→检查→诊断完整因果链，`retrieve_by_view("knowledge")` 获取阿司匹林禁忌症和发热诊疗规程摘要，`ask()` 综合生成「患者 3 日前开始服用阿司匹林，今日体温 39.2°C，建议优先排查药物热及血小板计数」的结构化建议。系统在毫秒级内将跨科室、跨时间维度的分散信息汇聚为决策支持上下文，降低跨科室信息遗漏风险。

---

## ⚡ 快速开始

### 安装

```bash
pip install mandol
```

支持可选依赖以启用额外后端：

```bash
pip install mandol[faiss]                 # FAISS 向量索引加速
pip install mandol[sentence-transformers] # 本地 Embedding/Reranker 模型
pip install mandol[openai]                # OpenAI API 支持
pip install mandol[milvus]                # Milvus 向量数据库
pip install mandol[neo4j]                 # Neo4j 图数据库
pip install mandol[all]                   # 安装所有可选依赖
```

### 配置

复制环境变量模板并填入 API Key：

```bash
cp .env.example .env
```

或通过 YAML 配置文件进行完整配置：

```yaml
llm:
  model: "gpt-4o-mini"
  base_url: "https://api.openai.com/v1"
  api_key: "sk-..."

embedder:
  model: "Qwen/Qwen3-Embedding-4B"
  device: "cpu"
  use_remote: false

reranker:
  model: "Qwen/Qwen3-Reranker-4B"
  device: "cpu"
  use_remote: false

system:
  chunk_max_tokens: 512
  bfs_expansion_hops: 1
  max_context_units: 20
```

远程 API 模式下无需下载本地模型（约 8 GB），仅需将 `use_remote` 设置为 `true` 并配置 API 端点即可快速体验。

### 三步使用

```python
from mandol import MemorySystem, MemoryUnit, Uid

system = MemorySystem.from_yaml_config("config.yaml")

# 1. 写入记忆
system.add(MemoryUnit(
    uid=Uid("msg_001"),
    raw_data={"text_content": "张三今天去北京出差了"},
    metadata={"timestamp": "2024-01-15T10:00:00"},
))

# 2. 构建高阶记忆结构
system.build_high_level(mode="auto")

# 3. 混合检索
hits = system.holistic_retrieve("张三去了哪里？", top_k=5)

system.save("./memory_snapshot")          # 持久化
system2 = MemorySystem.load("./memory_snapshot")  # 恢复
```

> **提示**：系统在 `add()` 时会异步检测会话边界并自动触发高阶记忆构建。仅检索原始对话（BASE 组）无需等待；检索实体/事件/摘要（ENTITY / EVENT / SUMMARY 组）需等待自动构建完成或手动调用 `build_high_level()`。插入少量数据后建议手动调用以确保高阶记忆可用。

---

## 📚 文档与社区

### 文档

完整的 API 参考、架构设计和最佳实践指南已通过 Sphinx 构建，涵盖基础用户、高级用户和开发者三个入口：

> 🔗 在线文档：[https://mandol.readthedocs.io](https://mandol.readthedocs.io)（即将上线）

本地构建文档：

```bash
cd docs && make html
```

### 参与贡献

我们欢迎社区贡献！提交 PR 前请阅读 [CONTRIBUTING.md](CONTRIBUTING.md)，了解开发环境搭建、代码规范（Ruff，行长 100 字符）、测试要求和 PR 流程。

### 反馈与讨论

- **Issue**：[GitHub Issues](https://github.com/your-org/mandol/issues) — 报告 Bug 或请求新功能
- **讨论**：[GitHub Discussions](https://github.com/your-org/mandol/discussions) — 使用问题、最佳实践交流
- **社区**：[Discord](https://discord.gg/mandol) — 实时交流与社区支持

---

## 📄 许可

Apache License 2.0 - 详见 [LICENSE](LICENSE)
