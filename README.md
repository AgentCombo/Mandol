# Mandol

> Mandol: an in-memory semantic memory runtime for agent systems.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Homepage](https://img.shields.io/badge/Homepage-agentcombo.github.io%2FMandol-blue)](https://agentcombo.github.io/Mandol)
[![Current Docs](https://img.shields.io/badge/Current_Docs-agentcombo.github.io%2FMandol%2Fdocs-green)](https://agentcombo.github.io/Mandol/docs)
[![Paper](https://img.shields.io/badge/arXiv-2606.29778-b31b1b.svg)](https://arxiv.org/abs/2606.29778)
[![PyPI](https://img.shields.io/badge/PyPI-0.1.0a2-blue)](https://pypi.org/project/mandol/0.1.0a2/)

[English](README.md) | [Chinese](README_CN.md)

> [!IMPORTANT]
> This branch is frozen for exact reproduction of the experiments reported in
> the Mandol paper. Use the commands, model roles, datasets, and benchmark
> settings documented in this checkout for paper comparisons. Current runtime
> development and the online documentation continue on
> [`main`](https://github.com/AgentCombo/Mandol/tree/main).

![Mandol Overview](README.assets/Mandol-overview-v2.png)

## Current Scope

This frozen branch exposes the `mandol` Python package under `src/mandol` and the
paper reproduction workflows under `benchmark_locomo`,
`benchmark_longmemeval`, and `benchmark_self_host`.

The public Python surface is centered on:

- `MemoryUnit`: the basic memory record.
- `MemorySpace`: a tree-like logical namespace for unit membership.
- `SemanticMap`: in-memory semantic indexing with RocksDB-backed automatic
  payload paging, embedding generation, sparse retrieval support, persistence,
  and space-filtered similarity search.
- `SemanticGraph`: a graph layer over memory units and spaces, with relationship
  APIs, graph traversal, retrieval helpers, RocksDB payload paging, and
  sandboxed persistence.
- `MultiRetriever`: BM25, SPLADE, cosine, graph expansion, score fusion, and
  reranker orchestration.
- `triple_retrieval`: `TripleTowerRetriever` and related result/config classes
  for routing, dispatching, pruning, reranking, and packaging hierarchical,
  entity-relation, and episodic tower results over already-built memory spaces.
- `auto_builder`: high-level memory construction pipelines that transform raw
  or L0 units into hierarchical summaries, episodic facts, and entity-relation
  memories with LLM extraction, deduplication, and batched graph writes.
- `memory_router`: LoCoMo and LongMemEval routing policies used by the paper
  router + quantification workflows.

Pre-refactor design material is preserved under `docs/archive/` and is not part
of the current API contract. Maintained docs and examples use `MemoryUnit`,
`SemanticMap`, `SemanticGraph`, `MultiRetriever`, `triple_retrieval`,
`auto_builder`, and the current subpackages directly.

## Requirements

- Python `>=3.12,<3.13`
- Linux for the full research/runtime stack
- `uv` for reproducible local environments
- Provider keys for model-backed reproduction runs

The default dependency set in `pyproject.toml` is intentionally broad. It
contains Torch, transformers, sentence-transformers, FAISS CPU, RocksDB, graph
libraries, LLM clients, retrieval/rerank tools, benchmark dependencies, and
optional integration clients needed by the artifact scripts.

## Environment Setup

Install `uv` if it is not already available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create the base runtime environment from the repository root:

```bash
uv sync
```

For day-to-day development and documentation work:

```bash
uv sync --extra dev --extra docs --group spacy-model
```

For paper reproduction and performance runs, install the full artifact stack.
The performance numbers reported for the paper were measured with the relevant
extras installed; use this full path when comparing throughput:

```bash
uv sync --extra dev --extra cuda --group spacy-model
```

If your machine does not have a CUDA/flash-attention-compatible setup, omit
`--extra cuda`. The workflows still run, but retrieval and reranking throughput
may differ from the paper performance setting.
The `cuda` extra is pinned to a Linux x86_64 / Python 3.12 / Torch 2.8 /
CUDA 12 flash-attention wheel for the paper artifact. If this wheel does not
match your platform, omit `--extra cuda` or install a compatible flash-attn
build manually.

After syncing, verify the local editable package:

```bash
uv run python -c "import mandol; print(mandol.__version__)"
```

## Installing Mandol As A Package

For development from this checkout, `uv sync` installs the local `src/mandol`
package into the environment. To build the same artifacts that would be uploaded
to PyPI:

```bash
uv build
```

To test the built wheel locally:

```bash
uv pip install --force-reinstall dist/mandol-*.whl
uv run python -c "from mandol import MemoryUnit, SemanticGraph, SemanticMap; print('ok')"
```

Mandol is published on PyPI. The current public alpha release is
[`mandol==0.1.0a2`](https://pypi.org/project/mandol/0.1.0a2/):

```bash
python -m pip install mandol==0.1.0a2
```

Pinning the version is recommended for this alpha artifact release.

The benchmark directories are repository artifacts, not part of the runtime
package. Use the source checkout when reproducing paper results.

## Optional Acceleration

Mandol runs without acceleration extras, but the paper artifact uses the
following optional paths for higher throughput:

- `--extra cuda`: installs the flash-attention extra declared in
  `pyproject.toml`. The code only passes flash-attention options when the
  dependency is available.
- `--group spacy-model`: installs the large English spaCy model used by some
  extraction and retrieval utilities. Tokenization falls back where supported,
  but the full artifact environment should include it.
- `RERANKER_BACKEND=vllm`: routes compatible reranker scoring through a vLLM
  HTTP endpoint when available.
- Local model caches: pre-download Hugging Face and sentence-transformers
  models on shared machines to avoid counting first-run downloads in benchmark
  timing.

Example vLLM reranker configuration:

```bash
export RERANKER_BACKEND=vllm
export VLLM_API_URL=http://127.0.0.1:8000/score
export VLLM_API_KEY=EMPTY
```

## Provider Keys

Runtime configuration is read through `mandol.utils.config_manager.settings`.
The project root `.env` and system environment variables are supported. Common
keys include:

```bash
export DASHSCOPE_API_KEY=...
export CLOSEAI_API_KEY=...
export OPENAI_API_KEY=...
export OPENROUTER_API_KEY=...
export SILICONFLOW_API_KEY=...
export CSTCLOUD_API_KEY=...
export HF_TOKEN=...
```

`CLOSEAI_API_KEY` falls back to `OPENAI_API_KEY` in the current provider
configuration. `CLOSEAI_*` is an OpenAI-compatible provider alias used by the
paper artifact configuration. If you do not use this gateway, set
`OPENAI_API_KEY` or map the model alias to your own provider. Use
`env.template` as the local environment template; never commit `.env` files.

## Dataset Preparation

Large public datasets and generated graph artifacts are intentionally ignored by
Git. Each dataset directory contains a README with source links and placement
instructions.

LoCoMo10:

```bash
mkdir -p benchmark_locomo/dataset/locomo
curl -fL https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json \
  -o benchmark_locomo/dataset/locomo/locomo10.json

mkdir -p benchmark_self_host/locomo10/dataset
cp benchmark_locomo/dataset/locomo/locomo10.json \
  benchmark_self_host/locomo10/dataset/locomo10.json
```

LongMemEval small split:

```bash
mkdir -p benchmark_longmemeval/dataset/LongMemEval
curl -fL https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json \
  -o benchmark_longmemeval/dataset/LongMemEval/longmemeval_s_cleaned.json

mkdir -p benchmark_self_host/longmemeval/dataset
cp benchmark_longmemeval/dataset/LongMemEval/longmemeval_s_cleaned.json \
  benchmark_self_host/longmemeval/dataset/longmemeval_s_cleaned.json
```

LongMemEval medium split is only needed for `--dataset-size m`:

```bash
curl -fL https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_m_cleaned.json \
  -o benchmark_longmemeval/dataset/LongMemEval/longmemeval_m_cleaned.json
```

Official dataset sources:

- LoCoMo: https://github.com/snap-research/locomo
- LongMemEval: https://github.com/xiaowu0162/LongMemEval
- LongMemEval cleaned files:
  https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned

## Quick Start

This example uses the light MiniLM preset and disables realtime SPLADE vector
generation so the first run stays small. Creating a `SemanticMap` still loads an
embedding model; if the model is not already cached, sentence-transformers may
download it from Hugging Face.

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
        raw_data={"text_content": "Zhang San travelled to Beijing today."},
        metadata={"timestamp": "2026-06-21T09:00:00"},
    ),
    space_names=["demo"],
    generate_sparse_embedding=False,
)
graph.add_unit(
    MemoryUnit(
        uid="msg_002",
        raw_data={"text_content": "He will discuss the Q2 delivery plan."},
        metadata={"timestamp": "2026-06-21T09:05:00"},
    ),
    space_names=["demo"],
    generate_sparse_embedding=False,
)

graph.add_relationship("msg_001", "msg_002", "NEXT")

hits = graph.search_similarity_in_graph(
    query_text="Where did Zhang San go?",
    top_k=3,
    ms_names=["demo"],
    return_score=True,
)

for unit, score in hits:
    print(f"{score:.3f} {unit.uid}: {unit.text_cached}")
```

For multi-method retrieval:

```python
from mandol.retrieval import MultiRetriever

retriever = MultiRetriever(graph)
results = retriever.smart_search(
    "Where did Zhang San go?",
    methods=["bm25", "cosine"],
    top_k=5,
    rerank_method=None,
    space_names=["demo"],
)
```

## Persistence

Use `SemanticGraph.save_graph()` and `SemanticGraph.load_graph()` for complete
state snapshots. They preserve graph topology, semantic map data, retrieval
indices when built, and the sandboxed RocksDB payload store when persistent
storage is enabled.

```python
graph.save_graph("./memory_snapshot", build_sparse_vectors=False)

restored = SemanticGraph.load_graph(
    "./memory_snapshot",
    embedding_model_name="all-MiniLM-L6-v2",
    use_flash_attention=False,
)
```

`SemanticMap.save_map()` and `SemanticMap.load_map()` also exist for resident
map-only state and do not preserve `SemanticGraph` topology. Direct
`SemanticMap.save_map()` calls fail closed while tiered paging is enabled;
use `SemanticGraph.save_graph()` so resident and cold payload state are saved
together.

RocksDB is the only supported persistent payload backend in this paper
artifact. Calling `SemanticGraph.connect_to_l2()` enables automatic tiered
payload paging: retrieval indexes, UID mappings, MemorySpace membership, and
graph topology remain resident, while cold `MemoryUnit` payloads are written
to RocksDB and paged back into the resident cache when needed. High and low
watermarks control eviction automatically.

```python
graph.connect_to_l2(
    "./l2_database",
    max_capacity=100_000,
    high_watermark=0.85,
    low_watermark=0.70,
)
```

If `connect_to_l2()` is not called, Mandol runs normally with payloads held in
memory. 

Tiered eviction is triggered from the existing add path. Candidate selection
and eviction scheduling occur within the add call, while RocksDB persistence
and resident-cache removal may complete asynchronously in the tiered-storage
executor. Cold-result materialization remains inside the search call that
requires the payload. `save_graph()` waits for previously submitted eviction
work before taking a snapshot, but concurrent graph mutation by another user
thread is not supported during the save. 

## Model Configuration

`SemanticMap` has a built-in model registry in
`src/mandol/core/semantic_map.py`. Current presets include:

| Model name | Type | Dim | Notes |
| --- | --- | ---: | --- |
| `Qwen/Qwen3-Embedding-0.6B` | local | 1024 | Default text embedding model |
| `Qwen/Qwen3-Embedding-4B` | local | 2560 | Larger local text model |
| `Qwen/Qwen3-Embedding-8B` | local | 4096 | Larger local text model |
| `Qwen/Qwen3-Embedding-0.6B-remote` | cloud | 1024 | SiliconFlow adapter |
| `BAAI/bge-m3` / `bge-m3` | local | 1024 | Text embedding model |
| `all-MiniLM-L6-v2` | local | 384 | Lightweight CPU-friendly option |
| `jinaai/jina-clip-v2` | local | 1024 | Text and image modalities |
| `jinaai/jina-embeddings-v4` | local | 2048 | Text and image modalities |

## Reproduction Workflows

The paper accuracy numbers use router + quantification workflows over generated
three-tower memory spaces:

- LoCoMo: [benchmark_locomo/REPRODUCE.md](benchmark_locomo/REPRODUCE.md)
- LongMemEval:
  [benchmark_longmemeval/REPRODUCE.md](benchmark_longmemeval/REPRODUCE.md)

The self-host workflows use Mandol's own high-level memory-generation path
without router + quantification:

- LoCoMo10 self-host:
  [benchmark_self_host/locomo10/REPRODUCE.md](benchmark_self_host/locomo10/REPRODUCE.md)
- LongMemEval self-host:
  [benchmark_self_host/longmemeval/REPRODUCE.md](benchmark_self_host/longmemeval/REPRODUCE.md)

Recommended smoke checks before long runs:

```bash
uv run python -m benchmark_locomo.task_eval.locomo_triple_router_quantification --help
uv run python -m benchmark_longmemeval.task_eval.benchmark_triple_router_quantification --help
uv run python -m benchmark_self_host.locomo10.build_graph --help
uv run python -m benchmark_self_host.longmemeval.build_graph --help
```

After the required graph artifacts exist, run a bounded real-LLM task-eval smoke
before launching full benchmark jobs:

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

Paper model roles:

The names below are Mandol provider aliases resolved by the repository
configuration. When using a different provider gateway, keep the roles fixed
but map each alias to an equivalent model endpoint in your local configuration.

- LoCoMo memory/extraction generation: `qwen-3.5-plus-thinking`
- LongMemEval memory/extraction generation: `qwen-3-plus`
- Deduplication: `deepseek-v3.2-dashscope`
- Task-eval evaluated models: `gpt-4.1-mini-closeai` and
  `gpt-4o-mini-closeai`
- Task-eval judge model: `gpt-4o-mini-closeai`

`qwen-3-plus` denotes the non-thinking variant. The thinking-enabled alias is
`qwen-3-plus-thinking`. The LongMemEval memory/extraction results reported in
the paper were generated with `qwen-3-plus`, not the thinking-enabled variant.

The model names above should be kept fixed when reproducing the paper tables.
The Qwen/DeepSeek models are used for memory generation and deduplication; the
GPT models are used for task evaluation and judging.

## Notes and Limitations

This repository is released as a research artifact and Python reference
implementation for the Mandol paper. It is not intended to be a
production-ready service.

- Full reproduction requires external model providers and local model downloads.
- Reported numbers may vary slightly across hardware, dependency versions,
  model provider versions, and random seeds.
- The `cuda` extra is platform-specific and can be omitted when
  flash-attention is unavailable.
- Large datasets, generated graphs, model caches, and benchmark outputs are
  intentionally excluded from Git.

## Performance Measurement Scope

LoCoMo retrieval-performance tests require unified per-sample graphs. Build them
after the three offline towers have been generated and before running the
fixed-QPS search benchmark:

```bash
bash benchmark_locomo/dataset_maker/build_unified.sh
```

The wrapper calls `benchmark_locomo/dataset_maker/build_unified_graph.py` and
writes unified graph folders to:

```text
benchmark_locomo/dataset/locomo/unified_per_sample_graphs
```

The two reported LoCoMo performance entrypoints measure different API scopes:

- Insertion latency:
  `benchmark_locomo/task_eval/locomo_triple_input_speed.py` schedules requests
  at the target QPS and measures only the body of each
  `SemanticGraph.add_unit(...)` call with `index_update_mode="incremental"` and
  `generate_sparse_embedding=True`. This timed call includes dense embedding
  generation, realtime SPLADE sparse embedding generation, and incremental
  index updates performed by the add path. The reported `latency_ms` excludes
  request scheduling sleep, memory-pool construction, graph initialization,
  warmup, and result-file writing.
- Search latency:
  `benchmark_locomo/task_eval/locomo_triple_smart_search_qps.py` loads a
  unified graph, runs warmup requests, then measures each scheduled
  `MultiRetriever.smart_search(...)` or `smart_search_async(...)` call. The
  reported `latency_ms` covers query dispatch through BM25, cosine, SPLADE,
  score fusion, reranking when `--rerank-method` is set, response parsing, and
  Python async/thread wrapper overhead inside one request. The provided speed
  scripts pass `--rerank-method baai`, so the current smart-search QPS numbers
  include reranking. The metric excludes graph loading, warmup, fixed QPS
  scheduling sleep, and report writing. The report also records
  `retrieval_time_ms` for the base retrieval phase and `rerank_time_ms` for the
  reranking phase.

## Package Layout

```text
src/mandol/
  core/                MemoryUnit, MemorySpace, SemanticMap, SemanticGraph
  retrieval/           MultiRetriever, BM25, SPLADE, cosine, fusion, rerankers
  triple_retrieval/    Three-tower routing, dispatch, pruning, rerank packaging
  auto_builder/        High-level builders, strategy presets, batched graph writes
  hierarchical/        Retrieval-facing hierarchical memory components
  entity_relation/     Retrieval-facing entity/relation graph components
  episodic/            Episodic memory retriever
  quantification/      Query expansion, pruning, semantic quantification
  memory_router/       LoCoMo and LongMemEval tower routers
  llm/                 LLM clients and provider wrappers
  storage/             RocksDB payload persistence and tiered-cache helpers
  cluster/             Leiden and DBSCAN clustering helpers
  utils/               Configuration, logging, model management
```

## Documentation

The maintained documentation entry point is `docs/index.rst`. Build it with:

```bash
uv sync --extra docs
uv run sphinx-build -b html docs docs/_build/html
```

The Docusaurus website lives in `website/` and is a separate static front page:

```bash
cd website
npm install
npm run build
```

## Citation

If you use Mandol in your research, please cite the arXiv paper:

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

## License

Mandol is released under the Apache License 2.0. See [LICENSE](LICENSE).

## Community

For discussion and user support, you can join the Mandol WeChat user group.

<img src="picture/mandol_wechat_user_group_qr_20260713.jpg" alt="Mandol WeChat user group QR code" width="260" />
