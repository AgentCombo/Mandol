# Mandol

> Mandol：面向智能体系统的内存语义记忆运行时。

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Homepage](https://img.shields.io/badge/Homepage-agentcombo.github.io%2FMandol-blue)](https://agentcombo.github.io/Mandol)
[![Docs](https://img.shields.io/badge/Docs-agentcombo.github.io%2FMandol%2Fdocs-green)](https://agentcombo.github.io/Mandol/docs)
[![Paper](https://img.shields.io/badge/arXiv-2606.29778-b31b1b.svg)](https://arxiv.org/abs/2606.29778)

[English](README.md) | [中文](README_CN.md)

![Mandol Overview](README.assets/Mandol-overview-v2.png)

## 当前代码范围

本仓库当前提供 `src/mandol` 下的 `mandol` Python 包，以及
`benchmark_locomo`、`benchmark_longmemeval`、`benchmark_self_host` 下的论文复现流程。

公开 Python 入口以以下组件为核心：

- `MemoryUnit`：基础记忆记录。
- `MemorySpace`：树形逻辑命名空间，用于组织记忆单元归属。
- `SemanticMap`：内存单元存储、embedding 生成、FAISS 索引、稀疏检索、持久化和空间过滤相似度搜索。
- `SemanticGraph`：记忆单元和记忆空间之上的图层，提供关系 API、图遍历、检索辅助、L2 存储支持和沙盒化持久化。
- `MultiRetriever`：BM25、SPLADE、余弦检索、图扩展、分数融合与 reranker 编排。
- `TripleTowerRetriever`：面向已构建记忆空间的分层、实体关系、情景记忆三塔检索编排。
- `memory_router`：论文 router + quantification 工作流使用的 LoCoMo 和 LongMemEval 路由策略。

仓库中的历史笔记可能仍提到 `MemorySystem`、`Uid`、`mandol.ports` 或
`mandol.retrieval.pipeline.HybridRetriever`。这些名称不属于当前包的公开导出。维护中的 README、docs 和 website 使用 `MemoryUnit`、`SemanticMap`、`SemanticGraph`、`MultiRetriever` 以及当前子包接口。

## 环境要求

- Python `>=3.12,<3.13`
- 完整研究/运行栈主要面向 Linux
- 推荐使用 `uv` 管理可复现环境
- 模型驱动的复现实验需要配置相应 provider key

`pyproject.toml` 中的默认依赖有意保持完整：Torch、transformers、sentence-transformers、FAISS CPU、DuckDB、图算法库、LLM 客户端、检索/重排序工具、benchmark 依赖和可选集成客户端都会随基础环境安装。

## 环境配置

如果本机尚未安装 `uv`：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

在仓库根目录创建基础运行环境：

```bash
uv sync
```

日常开发和文档构建推荐：

```bash
uv sync --extra dev --extra docs --group spacy-model
```

论文复现和性能对比推荐安装完整 artifact 栈。论文中的性能结果是在安装相关 extra 后测试得到的；如果要对齐吞吐性能，请使用全量安装路径：

```bash
uv sync --extra dev --extra cuda --group spacy-model
```

如果机器不具备兼容的 CUDA / flash-attention 环境，可以去掉 `--extra cuda`。准确率复现实验仍可运行，但检索和重排序吞吐可能与论文性能环境不同。
`cuda` extra 针对论文 artifact 固定到 Linux x86_64 / Python 3.12 / Torch 2.8 / CUDA 12 的 flash-attention wheel。如果该 wheel 与本机环境不匹配，请去掉 `--extra cuda`，或手动安装兼容版本的 flash-attn。

环境创建后验证本地 editable 包：

```bash
uv run python -c "import mandol; print(mandol.__version__)"
```

## 安装 Mandol 包

从本仓库开发时，`uv sync` 会把本地 `src/mandol` 包安装进环境。若要构建与 PyPI 发布一致的包产物：

```bash
uv build
```

本地测试构建出的 wheel：

```bash
uv pip install --force-reinstall dist/mandol-*.whl
uv run python -c "from mandol import MemoryUnit, SemanticGraph, SemanticMap; print('ok')"
```

正式发布到 PyPI 后，用户可以通过以下方式安装运行时包：

```bash
python -m pip install mandol
```

benchmark 目录是论文 artifact 的一部分，不属于运行时包本体。复现论文结果时请使用源码 checkout。

## 可选加速项

Mandol 不依赖加速项也可以运行，但论文 artifact 使用以下可选路径提升吞吐：

- `--extra cuda`：安装 `pyproject.toml` 中声明的 flash-attention extra。代码只会在依赖可用时传入 flash-attention 配置。
- `--group spacy-model`：安装部分抽取与检索工具使用的大型英文 spaCy 模型。部分路径有 fallback，但完整复现环境建议安装。
- `RERANKER_BACKEND=vllm`：在可用时将兼容的 reranker scoring 交给 vLLM HTTP 服务。
- 本地模型缓存：在共享机器上建议提前缓存 Hugging Face 和 sentence-transformers 模型，避免把首次下载时间计入 benchmark。

vLLM reranker 示例配置：

```bash
export RERANKER_BACKEND=vllm
export VLLM_API_URL=http://127.0.0.1:8000/score
export VLLM_API_KEY=EMPTY
```

## Provider Key

运行配置通过 `mandol.utils.config_manager.settings` 读取，支持项目根目录 `.env` 和系统环境变量。常用键包括：

```bash
export DASHSCOPE_API_KEY=...
export CLOSEAI_API_KEY=...
export OPENAI_API_KEY=...
export OPENROUTER_API_KEY=...
export SILICONFLOW_API_KEY=...
export CSTCLOUD_API_KEY=...
export HF_TOKEN=...
```

当前 provider 配置中，`CLOSEAI_API_KEY` 可以回退到 `OPENAI_API_KEY`。`CLOSEAI_*` 是论文 artifact 中使用的 OpenAI-compatible provider alias。如果不使用该网关，可以配置 `OPENAI_API_KEY`，或在 provider 配置中将模型 alias 映射到自己的服务。本地配置建议参考 `env.template`；不要提交 `.env` 文件。

## 数据集准备

大型公开数据集和生成图产物不会提交到 Git。每个 dataset 目录都提供了 README，记录官方来源和本地放置路径。

LoCoMo10：

```bash
mkdir -p benchmark_locomo/dataset/locomo
curl -fL https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json \
  -o benchmark_locomo/dataset/locomo/locomo10.json

mkdir -p benchmark_self_host/locomo10/dataset
cp benchmark_locomo/dataset/locomo/locomo10.json \
  benchmark_self_host/locomo10/dataset/locomo10.json
```

LongMemEval small split：

```bash
mkdir -p benchmark_longmemeval/dataset/LongMemEval
curl -fL https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json \
  -o benchmark_longmemeval/dataset/LongMemEval/longmemeval_s_cleaned.json

mkdir -p benchmark_self_host/longmemeval/dataset
cp benchmark_longmemeval/dataset/LongMemEval/longmemeval_s_cleaned.json \
  benchmark_self_host/longmemeval/dataset/longmemeval_s_cleaned.json
```

LongMemEval medium split 仅在运行 `--dataset-size m` 时需要：

```bash
curl -fL https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_m_cleaned.json \
  -o benchmark_longmemeval/dataset/LongMemEval/longmemeval_m_cleaned.json
```

官方数据来源：

- LoCoMo: https://github.com/snap-research/locomo
- LongMemEval: https://github.com/xiaowu0162/LongMemEval
- LongMemEval cleaned files:
  https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned

## 快速开始

下面示例使用轻量 MiniLM 预设，并关闭实时 SPLADE 向量生成，便于第一次运行。创建 `SemanticMap` 会加载 embedding 模型；如果本地没有缓存，sentence-transformers 可能会从 Hugging Face 下载模型。

```python
from mandol import MemoryUnit, SemanticGraph, SemanticMap

semantic_map = SemanticMap(
    embedding_model_name="all-MiniLM-L6-v2",
    use_flash_attention=False,
)
graph = SemanticGraph(semantic_map_instance=semantic_map)

graph.add_unit(
    MemoryUnit(
        uid="msg_001",
        raw_data={"text_content": "张三今天去了北京出差。"},
        metadata={"timestamp": "2026-06-21T09:00:00"},
    ),
    space_names=["demo"],
    generate_sparse_embedding=False,
)
graph.add_unit(
    MemoryUnit(
        uid="msg_002",
        raw_data={"text_content": "他将讨论 Q2 交付计划。"},
        metadata={"timestamp": "2026-06-21T09:05:00"},
    ),
    space_names=["demo"],
    generate_sparse_embedding=False,
)

graph.add_relationship("msg_001", "msg_002", "NEXT")

hits = graph.search_similarity_in_graph(
    query_text="张三去了哪里？",
    top_k=3,
    ms_names=["demo"],
    return_score=True,
)

for unit, score in hits:
    print(f"{score:.3f} {unit.uid}: {unit.text_cached}")
```

多路检索入口：

```python
from mandol.retrieval import MultiRetriever

retriever = MultiRetriever(graph)
results = retriever.smart_search(
    "张三去了哪里？",
    methods=["bm25", "cosine"],
    top_k=5,
    rerank_method=None,
    space_names=["demo"],
)
```

## 持久化

完整状态快照请使用 `SemanticGraph.save_graph()` 和 `SemanticGraph.load_graph()`。它们会保留图拓扑、SemanticMap 数据、已构建的检索索引以及沙盒化 DuckDB L2 存储副本。

```python
graph.save_graph("./memory_snapshot", build_sparse_vectors=False)

restored = SemanticGraph.load_graph(
    "./memory_snapshot",
    embedding_model_name="all-MiniLM-L6-v2",
    use_flash_attention=False,
)
```

`SemanticMap.save_map()` 与 `SemanticMap.load_map()` 也存在，但它们只保存/加载 map 层，不保存 `SemanticGraph` 拓扑。

## 模型配置

`SemanticMap` 的模型注册表位于 `src/mandol/core/semantic_map.py`。当前常用预设包括：

| 模型名 | 类型 | 维度 | 说明 |
| --- | --- | ---: | --- |
| `Qwen/Qwen3-Embedding-0.6B` | 本地 | 1024 | 默认文本 embedding 模型 |
| `Qwen/Qwen3-Embedding-4B` | 本地 | 2560 | 更大的本地文本模型 |
| `Qwen/Qwen3-Embedding-8B` | 本地 | 4096 | 更大的本地文本模型 |
| `Qwen/Qwen3-Embedding-0.6B-remote` | 云端 | 1024 | SiliconFlow 适配器 |
| `BAAI/bge-m3` / `bge-m3` | 本地 | 1024 | 文本 embedding 模型 |
| `all-MiniLM-L6-v2` | 本地 | 384 | 轻量 CPU 友好选项 |
| `jinaai/jina-clip-v2` | 本地 | 1024 | 文本与图像模态 |
| `jinaai/jina-embeddings-v4` | 本地 | 2048 | 文本与图像模态 |

## 复现流程

论文准确率结果使用已经构建好的三塔记忆空间，并运行 router + quantification 工作流：

- LoCoMo: [benchmark_locomo/REPRODUCE.md](benchmark_locomo/REPRODUCE.md)
- LongMemEval:
  [benchmark_longmemeval/REPRODUCE.md](benchmark_longmemeval/REPRODUCE.md)

self-host 工作流使用 Mandol 自身的高阶记忆生成路径，不使用 router + quantification：

- LoCoMo10 self-host:
  [benchmark_self_host/locomo10/REPRODUCE.md](benchmark_self_host/locomo10/REPRODUCE.md)
- LongMemEval self-host:
  [benchmark_self_host/longmemeval/REPRODUCE.md](benchmark_self_host/longmemeval/REPRODUCE.md)

长时间运行前建议先执行入口检查：

```bash
uv run python -m benchmark_locomo.task_eval.locomo_triple_router_quantification --help
uv run python -m benchmark_longmemeval.task_eval.benchmark_triple_router_quantification --help
uv run python -m benchmark_self_host.locomo10.build_graph --help
uv run python -m benchmark_self_host.longmemeval.build_graph --help
```

必要图产物生成完成后，建议先跑一个带真实 LLM 调用的低成本 task-eval smoke，再启动完整 benchmark：

```bash
uv run python -m benchmark_locomo.task_eval.locomo_triple_router_quantification \
  --sample-ids conv-30 \
  --max-questions 1 \
  --llm-model gpt-4.1-mini-closeai \
  --llm-evaluate-model gpt-4o-mini-closeai \
  --output-dir benchmark_locomo/task_eval/results/smoke/gpt41_mini

uv run python -m benchmark_longmemeval.task_eval.benchmark_triple_router_quantification \
  --dataset-size s \
  --start-qa 0 \
  --end-qa 0 \
  --max-tests 1 \
  --llm-model gpt-4.1-mini-closeai \
  --llm-evaluate-model gpt-4o-mini-closeai \
  --output-dir benchmark_longmemeval/task_eval/results/smoke/gpt41_mini
```

论文中的模型角色如下：

下列名称是仓库配置中解析的 Mandol provider alias。若使用不同的模型网关，请保持模型角色不变，并在本地配置中将这些 alias 映射到等价的模型端点。

- LoCoMo 记忆/抽取生成：`qwen-3.5-plus-thinking`
- LongMemEval 记忆/抽取生成：`qwen-3-plus`
- 去重：`deepseek-v3.2-dashscope`
- task-eval 被评测模型：`gpt-4.1-mini-closeai` 和 `gpt-4o-mini-closeai`
- task-eval judge 模型：`gpt-4o-mini-closeai`

复现论文表格时应保持上述模型角色不变。Qwen/DeepSeek 用于记忆生成和去重；GPT 模型用于任务评测和 judge。

## 说明与限制

本仓库作为 Mandol 论文的研究 artifact 和 Python 参考实现发布，不是生产级在线服务。

- 完整复现需要外部模型 provider 和本地模型下载。
- 不同硬件、依赖版本、模型服务版本和随机性可能导致结果存在小幅差异。
- `cuda` extra 具有平台相关性；如果本机不支持 flash-attention，可以去掉该 extra。
- 大型数据集、生成图、模型缓存和 benchmark 输出不会提交到 Git。

## 性能测试口径

LoCoMo 检索性能测试需要先构建每个样本的统一图。请在三塔离线图都生成完成后、运行固定 QPS 检索 benchmark 之前执行：

```bash
bash benchmark_locomo/dataset_maker/build_unified.sh
```

该脚本会调用 `benchmark_locomo/dataset_maker/build_unified_graph.py`，并将统一图写入：

```text
benchmark_locomo/dataset/locomo/unified_per_sample_graphs
```

LoCoMo 的两个性能入口衡量的是不同 API 边界：

- 插入延迟：
  `benchmark_locomo/task_eval/locomo_triple_input_speed.py` 按目标 QPS 调度请求，并只计时每次
  `SemanticGraph.add_unit(...)` 调用本体，其中 `index_update_mode="incremental"` 且
  `generate_sparse_embedding=True`。这段计时包含 add 路径内部的 dense embedding 生成、实时 SPLADE sparse embedding 生成，以及增量索引更新。输出的 `latency_ms` 不包含调度 sleep、memory pool 构造、图初始化、warmup 和结果文件写入。
- 检索延迟：
  `benchmark_locomo/task_eval/locomo_triple_smart_search_qps.py` 会先加载统一图并执行 warmup，然后计时每个已调度的
  `MultiRetriever.smart_search(...)` 或 `smart_search_async(...)` 请求。输出的 `latency_ms` 包含一次请求内部的 BM25、余弦检索、SPLADE、分数融合、设置 `--rerank-method` 时触发的 rerank、结果解析以及 Python async/thread 包装开销。目前提供的 speed 脚本使用 `--rerank-method baai`，因此当前 smart-search QPS 结果包含重排序时间；不包含图加载、warmup、固定 QPS 调度 sleep 和报告写入。报告中还会单独记录 base retrieval 阶段的 `retrieval_time_ms` 和 rerank 阶段的 `rerank_time_ms`。

## 包结构

```text
src/mandol/
  core/                MemoryUnit、MemorySpace、SemanticMap、SemanticGraph
  retrieval/           MultiRetriever、BM25、SPLADE、余弦检索、融合、reranker
  triple_retrieval/    三塔检索编排
  hierarchical/        分层记忆检索组件
  entity_relation/     实体关系图检索组件
  episodic/            情景记忆检索器
  quantification/      查询扩展、剪枝、语义量化
  memory_router/       LoCoMo 与 LongMemEval 塔路由器
  llm/                 LLM 客户端与 provider 封装
  storage/             DuckDB 与分层存储辅助
  cluster/             Leiden 与 DBSCAN 聚类辅助
  utils/               配置、日志、模型管理
```

## 文档

当前维护的文档入口是 `docs/index.rst`。构建方式：

```bash
uv sync --extra docs
uv run sphinx-build -b html docs docs/_build/html
```

Docusaurus 静态首页位于 `website/`：

```bash
cd website
npm install
npm run build
```

## 引用

如果你在研究中使用 Mandol，请引用 arXiv 论文：

```bibtex
@misc{zhang2026mandol,
  title         = {Mandol: An Agglomerative Agent Memory System for Long-Term Conversations},
  author        = {Yuhan Zhang and Zhiyuan Guo and Ziheng Zeng and Wei Wang and Wentao Wu and Lijie Xu},
  year          = {2026},
  eprint        = {2606.29778},
  archivePrefix = {arXiv},
  primaryClass  = {cs.DB},
  doi           = {10.48550/arXiv.2606.29778},
  url           = {https://arxiv.org/abs/2606.29778}
}
```

## 许可证

Mandol 使用 Apache License 2.0 发布。详见 [LICENSE](LICENSE)。
