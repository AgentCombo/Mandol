# Mandol

> Mandol：一种面向长对话的智能体内存记忆系统

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Homepage](https://img.shields.io/badge/Homepage-agentcombo.github.io%2FMandol-blue)](https://agentcombo.github.io/Mandol)
[![Docs](https://img.shields.io/badge/Docs-agentcombo.github.io%2FMandol%2Fdocs-green)](https://agentcombo.github.io/Mandol/docs)
[![Paper](https://img.shields.io/badge/Paper-arXiv:2606.29778-red.svg)](https://arxiv.org/abs/2606.29778)

[English](README.md) | [中文](README_CN.md)

> [!IMPORTANT]
> `main` 分支目前处于持续重构状态。若需要精确复现论文实验结果，请使用
> [`paper-repro`](https://github.com/AgentCombo/Mandol/tree/paper-repro) 分支。
> 当前公开的 PyPI 包 `mandol==0.1.0a1` 与 GitHub release 对应论文复现版本。

![Mandol Overview](README.assets/Mandol-overview.png)

---

## 📑 目录

<details>
<summary><b>展开/收起</b></summary>

- [📖 Mandol 是什么？](#mandol-是什么)
- [💡 核心创新](#核心创新)
- [✨ 关键特性](#关键特性)
- [📊 与主流记忆系统对比](#与主流记忆系统对比)
- [🏆 应用案例](#应用案例)
- [🔬 论文复现](#-论文复现)
- [⚡ 快速开始](#快速开始)
- [📚 文档与社区](#文档与社区)
- [📄 引用](#引用)
- [📄 许可](#许可)

</details>

---

## 📖 Mandol 是什么？

Mandol 是一套以内存为核心、具备高效精确检索能力的智能体分层记忆系统，实现复杂记忆信息的统一表示、高效存储与高效精确检索，为下一代智能体认知架构提供理论支撑与技术方案。

系统基于纯 Python 内存数据结构，融合键值、向量与图三种索引范式，提供统一的存储与混合检索接口，无需外部依赖即可运行。向下可按需桥接 Milvus、Neo4j 等外部存储引擎，向上提供 `add()` → `holistic_retrieve()` 的极简操作模型。其核心创新在于将传统「被动召回-排序」检索范式转变为「智能路由 → 量化去噪 → 高质量上下文生成」主动检索新范式。

**在主流对话记忆基准上，Mandol 以较低的 Token 消耗实现了 SOTA 级别的综合表现：**

| **维度** | **Mem0** | **Zep** | **MemOS** | **EverMemOS** | **Mandol** |
|---|---|---|---|---|---|
| **记忆组织与表示** | 文本向量 + 元数据 | 文本向量 + 时序知识图谱 | 文本向量 + 图/树摘要 | 文本向量 + 高层摘要 | **结构化语义图 + 抽象高阶记忆 + 层级化记忆** |
| **存储架构** | 单一关系数据库（含向量扩展） | 关系数据库 + 自定义图引擎 + 图数据库 | 图数据库 + 向量数据库组合 | 混合多组件数据库（文档、检索、向量、缓存） | **内存语义数据结构 + 进程内数据库 DuckDB/DuckPGQ** |
| **检索与查询机制** | 向量语义检索 + 关键词过滤 | 多步图遍历拓扑搜索 + 重排序 | 向量检索 + 动态图节点召回 | 多路径路由 + LLM 多轮查询改写 | **内存多路并行召回 + 智能路由 + 量化去噪 + 上下文优化** |
| **I/O 开销与资源** | 中等：受限于传统数据库的行级更新与单路径索引 | 高：频繁的事实提取与跨服务通信导致系统延迟高 | 高：多数据库导致沉重的 I/O 开销 | 极高：极度碎片化的组件栈导致严重的跨存储网络与序列化开销 | **极低：核心算子均在进程内原生执行，完全消除跨存储网络与通信瓶颈** |
| | | | | | |
| **LoCoMo 评分** | 64.20 (1.0k Tokens) | 85.22 (1.4k Tokens) | 80.76 (2.5k Tokens) | 91.97 (2.7k Tokens) | **92.21 (1.9k Tokens)** |
| **LongMemEval 评分** | 66.40 (1.1k Tokens) | 63.80 (1.6k Tokens) | 77.80 (1.4k Tokens) | 83.00 (2.8k Tokens) | **88.40 (2.3k Tokens)** |

> Mandol 以 1.9k Token 达到 LoCoMo 92.21 分——Token 效率是同等精度系统 EverMemOS（2.7k）的 1.4 倍，是 Mem0 v2.0（7.0k）的 3.7 倍。LongMemEval 上以 2.3k Token 达到 88.40 分，较 EverMemOS（2.8k / 83.00）在 Token 减少 18% 的同时评分提升 5.4 个百分点。

---

## 💡 核心创新

### （一）理论模型创新：分层式理论记忆模型

提出分层式理论记忆模型，将记忆系统划分为基础记忆层、高阶记忆层和智能查询层。通过结构化语义图统一表征多模态、关联复杂的记忆信息，引入隐式语义边按需生成策略兼顾结构化精确性与语义灵活性，并建立基础与高阶记忆的双向可追溯机制。该模型实现了复杂记忆信息的统一表示，并为后续存储和智能量化检索提供了理论基础。相比现有向量表示难以刻画结构关系、知识图谱对多模态和语义相似支持不足的局限，该模型构建了从原始信息存储、抽象知识提炼到查询调度的统一理论框架。

![分层架构示意](README.assets/分层式理论记忆模型.png)

### （二）存储架构创新：基于内存语义数据结构的统一存储架构

提出基于内存语义数据结构的统一存储架构，设计 SemanticMap 与 SemanticGraph 协同的内存语义数据结构，在物理层面实现键值存储、向量索引与图结构的原生融合，消除多库碎片化问题。该架构通过原子化混合检索算子将向量匹配、图遍历等操作统一封装为内存原子操作，有效降低查询延迟，为上层智能量化查询提供了标准化、可组合的执行单元；同时，采用「内存活跃态-数据库持久态」协同架构，实现性能与存储容量的有效平衡。

![统一存储框架](README.assets/基于内存语义数据结构的统一存储架构.png)

### （三）检索机制创新：智能路由与量化检索方法

提出一种智能路由与量化检索方法，将检索过程从被动「召回-排序」模式，转变为「智能路由-量化去噪-高质量上下文生成」新范式。通过查询意图驱动的智能路由、量化去噪和冲突消解、以及 Token 约束下的高质量上下文生成等创新设计，在有限的计算与 Token 预算下，实现对复杂多源记忆的高效精确检索。

![量化检索管线](README.assets/智能路由与量化检索.png)

---

## ✨ 关键特性

### 轻量级架构

纯 Python 实现，核心逻辑采用六边形架构（端口-适配器模式），`MemorySystem()` 无参构造即可启动完整记忆系统，零外部依赖。通过 YAML 配置即可切换 FAISS、Milvus、Neo4j 等外部引擎，无需修改业务代码。

### 简单易用

三步操作模型覆盖核心流程：`add()` 写入记忆 → `build_high_level()` 构建高阶结构 → `holistic_retrieve()` 混合检索。`save()` / `load()` 一键持久化与恢复。远程 API 模式下无需下载本地模型，仅需配置 API 端点即可快速体验。

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

单一 `MemoryUnit` 抽象统一承载文本（`text_content`）与图像（`image_path`）等异构信息，自动完成向量化。`MemorySpace` 树形层级支持按 BASE / ENTITY / EVENT / SUMMARY 等维度灵活组织记忆。`SemanticGraph` 以有向图显式建模实体间关系与事件因果链，支持多跳图遍历检索。

### 层级化记忆结构

- **基础记忆层（Base）**：原始数据片段，`add()` 后立即可检索
- **高阶记忆层（High-Level）**：系统自动完成会话分割（LLM 驱动）、实体提取与去重、事件提取与去重、实体关系构建、事件因果链构建、多类型摘要生成（情景 / 知识 / 情感 / 过程）及全局洞察提取
- **跨会话共指消解**：自动合并跨会话的同一实体和事件，维护一致的知识表示

### 多底层数据库支持

六边形架构实现核心逻辑与存储后端的完全解耦。同一套 API 可切换不同的底层基础设施：向量索引（内存精确搜索 → FAISS ANN 自适应切换）、图存储（内存 → Neo4j）、单元存储（内存 → Milvus）、Embedding / Reranker（本地模型 → 远程 OpenAI 兼容 API）。所有后端切换仅需修改 YAML 配置文件，业务代码零改动。

```yaml
# 示例：从本地模型切换至远程 API
embedder:
  use_remote: true
  base_url: "https://api.example.com/v1"

# 切换图存储至 Neo4j
graph_store:
  backend: neo4j
  uri: "bolt://localhost:7687"
```

---

## 📊 与主流记忆系统对比

Mandol 与现有记忆系统的本质区别在于检索范式：传统系统将检索视为单向流水线（Embedding 召回 → Rerank 排序 → Top-K），检索过程被动且缺乏对噪声的控制。Mandol 将这一范式重构为三阶段主动检索流水线——首先依据查询意图动态路由到最相关的记忆源，然后在各源内部及跨源之间进行多级量化过滤与冲突消解，最后在 Token 约束下生成高信息密度上下文。这一范式转变使检索从被动的「匹配-返回」升级为主动的「理解-筛选-归纳」。

在架构层面，Mandol 采用六边形架构（端口-适配器模式），核心检索逻辑与底层存储引擎完全解耦，支持从纯内存模式到 FAISS、Milvus、Neo4j 等外部引擎的灵活切换（详见上文「多底层数据库支持」）。

> 详细的基准对比数据见上方「[Mandol 是什么？](#mandol-是什么)」章节中的性能表格。

---

## 🏆 应用案例

### 长对话记忆基准 LoCoMo

在 LoCoMo 基准（10 段长对话 × 200+ 轮交互，覆盖单跳/多跳/时序/开放域查询）中，Mandol 在所有系统中取得最高的**多跳推理**评分（92.20 分）。这得益于 `SemanticGraph` 的显式实体关系图与 BFS 图扩展机制，能够沿关系边多跳遍历发现非直接关联的证据。

> 当查询「张经理去年的决策对今年 Q2 的项目延期有何影响」时，Mandol 沿事件因果链 `决策A → 团队调整 → 资源转移 → 项目B延期 → Q2交付推迟` 完成 4 跳追溯，而纯向量检索仅能返回包含「张经理」「Q2」等关键词的孤立片段。

### 长记忆评估基准 LongMemEval

LongMemEval 侧重多会话场景下的记忆保持与知识更新能力。Mandol 在助手侧记忆（SS-Asst 98.21）和用户侧记忆（SS-User 98.57）两个子项上接近满分，知识更新评分 89.74——当同一事实存在新旧两个版本时，系统准确采纳新信息并消解冲突，验证了跨会话共指消解与「优先采纳新信息」策略的有效性。

### 智能客服

多轮客服对话中，当用户询问「昨天买的蓝色衬衫降价了怎么办」，系统需同时关联**时序事件**（降价发生时间）、**商品属性**（蓝色衬衫 SKU）、**用户信息**（购买记录、会员等级）三个维度的记忆。Mandol 通过多维关联查询直接锁定具体订单和适用价保策略，生成包含「您的订单符合价保规则，可退差价 ¥35」的准确回复，提升一次解决率。

### 软件开发

当开发者请求「分析支付模块异常与近一周上线功能的关联」，信息分散在 PR 讨论、Issue 评论、变更日志和设计文档中。Mandol 跨 BASE/ENTITY/EVENT/SUMMARY 四组空间并行检索，`SemanticGraph` 自动构建模块-函数-开发者-版本关联图，检索结果涵盖代码变更、讨论上下文和时序关联，将根因分析从天级缩短至分钟级。

### 医疗

医生请求「对服用阿司匹林后发热的患者提供紧急检查支持」时，关键信息分散在跨科室病历、用药记录和检查报告中。Mandol 通过实体关系图检索、事件因果链追溯和知识摘要获取，在毫秒级内将跨科室、跨时间维度的分散信息汇聚为结构化决策支持上下文，降低跨科室信息遗漏风险。

---

## 🔬 论文复现

论文中报告的 LoCoMo 和 LongMemEval 实验结果基于冻结的
[`paper-repro`](https://github.com/AgentCombo/Mandol/tree/paper-repro)
版本生成。若需要忠实复现论文实验，请直接克隆该分支：

```bash
git clone --branch paper-repro --single-branch https://github.com/AgentCombo/Mandol.git
cd Mandol
```

请按照 `paper-repro` 分支中的基准专项说明运行：

- [LoCoMo 论文复现](https://github.com/AgentCombo/Mandol/blob/paper-repro/benchmark_locomo/REPRODUCE.md)
- [LongMemEval 论文复现](https://github.com/AgentCombo/Mandol/blob/paper-repro/benchmark_longmemeval/REPRODUCE.md)

`main` 分支及其中的 `experimental/self_host_benchmarks/` 工作流目前仍在持续重构，不是生成论文报告结果时使用的精确复现入口。数据集、实验配置和中间产物的获取方式，请以 `paper-repro` 分支中的对应文档为准。

## ⚡ 快速开始

### 安装

#### 已发布的 paper-repro 包

当前已发布的 PyPI 包对应 `paper-repro` 分支：

```bash
pip install mandol==0.1.0a1
```

如果需要完整论文复现环境，请使用 `paper-repro` 源码分支，并安装 artifact stack：

```bash
uv sync --extra dev --extra cuda --group spacy-model
```

如果本机不支持 CUDA 或 flash-attention，可以去掉 `--extra cuda`：

```bash
uv sync --extra dev --group spacy-model
```

#### main 分支可选后端

以下可选依赖组属于 `main` 分支开发版本，可能与已发布的 `paper-repro` 包不完全一致：

```bash
pip install mandol[faiss]                 # FAISS 向量索引加速
pip install mandol[sentence-transformers] # 本地 Embedding/Reranker 模型
pip install mandol[openai]                # OpenAI API 支持
pip install mandol[milvus]                # Milvus 向量数据库
pip install mandol[neo4j]                 # Neo4j 图数据库
pip install mandol[all]                   # 安装所有可选依赖
```

> 如需严格复现论文实验结果，请使用 [`paper-repro`](https://github.com/AgentCombo/Mandol/tree/paper-repro) 分支。
> 完整的安装指南、配置说明和进阶用法请参阅 [在线文档](https://agentcombo.github.io/Mandol/docs)。

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

> **提示**：系统在 `add()` 时会自动检测会话边界并触发高阶记忆构建。插入少量数据后建议手动调用 `build_high_level()` 以确保高阶记忆可用。更多配置选项和进阶用法请参阅 [在线文档](https://agentcombo.github.io/Mandol/docs)。

---

## 📚 文档与社区

### 文档

完整的 API 参考、架构设计和最佳实践指南已通过 Sphinx 构建，涵盖基础用户、高级用户和开发者三个入口：

> 🔗 在线文档：[https://agentcombo.github.io/Mandol/docs](https://agentcombo.github.io/Mandol/docs)（即将上线）

本地构建文档：

```bash
cd docs && make html
```

### 参与贡献

我们欢迎社区贡献！提交 PR 前请阅读 [CONTRIBUTING.md](CONTRIBUTING.md)，了解开发环境搭建、代码规范（Ruff，行长 100 字符）、测试要求和 PR 流程。

### 反馈与讨论

- **Issue**：[GitHub Issues](https://github.com/AgentCombo/Mandol/issues) — 报告 Bug 或请求新功能
- **讨论**：[GitHub Discussions](https://github.com/AgentCombo/Mandol/discussions) — 使用问题、最佳实践交流
- **社区**：扫描下方二维码加入 Mandol 微信用户群

<img src="README.assets/mandol_wechat_user_group_qr_20260713.jpg" alt="Mandol 微信用户群" width="300">

---

## 📄 引用

如果本工作对您的研究有帮助，请引用我们的论文：

```bibtex
@misc{zhang2026mandol,
  title={Mandol: An Agglomerative Agent Memory System for Long-Term Conversations}, 
  author={Yuhan Zhang and Zhiyuan Guo and Ziheng Zeng and Wei Wang and Wentao Wu and Lijie Xu},
  year={2026},
  eprint={2606.29778},
  archivePrefix={arXiv},
  primaryClass={cs.DB},
  url={https://arxiv.org/abs/2606.29778}, 
}
```

---

## 📄 许可

Apache License 2.0 - 详见 [LICENSE](LICENSE)
