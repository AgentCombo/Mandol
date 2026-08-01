# LoCoMo Reproduction Guide

> This guide belongs to the frozen `paper-repro` branch. Current runtime
> development continues on `main`; use this checkout for paper comparisons.

This guide documents the LoCoMo reproduction path used for the paper numbers in
this repository. Run commands from the repository root unless noted otherwise.

## Scope

The paper accuracy numbers for LoCoMo come from the router + quantification
workflow:

```text
benchmark_locomo/task_eval/locomo_triple_router_quantification.py
```

The cited retrieval-time numbers come from:

```text
benchmark_locomo/task_eval/locomo_triple_input_speed.py
benchmark_locomo/task_eval/locomo_triple_smart_search_qps.py
```

The intended paper model setup is:

- Memory/extraction generation backbone: `qwen-3.5-plus-thinking`
- Memory deduplication backbone: `deepseek-v3.2-dashscope`
- Task-eval evaluated models (`--llm-model`) reported in the paper:
  `gpt-4.1-mini-closeai` and `gpt-4o-mini-closeai`
- Task-eval judge model (`--llm-evaluate-model`) used by the historical
  scripts: `gpt-4o-mini-closeai`
- Reranker: `baai`
- Embedding model used by graph builders: `Qwen/Qwen3-Embedding-0.6B`
- Sparse model used by unified graph builders: `naver/splade-v3`

In the paper text, `deepseek-V3.2-chat` refers to the non-thinking DeepSeek V3.2
chat mode. In the current Mandol model registry this is represented as
`deepseek-v3.2-dashscope` with `actual_model=deepseek-v3.2` and
`enable_thinking=False`.

## Environment

Use Python 3.12 and the repository `pyproject.toml` environment.

```bash
uv sync --extra dev --extra cuda --group spacy-model
export PYTHONPATH="$PWD/src:$PWD"
```

The paper performance numbers were measured with the relevant extras installed.
If your machine cannot install the CUDA/flash-attention extra, omit
`--extra cuda`; accuracy reproduction still works, but throughput may differ.

Set the provider keys needed by your run:

```bash
export DASHSCOPE_API_KEY=...
```

`DASHSCOPE_API_KEY` is required for the Qwen/DeepSeek memory generation and
deduplication steps, and for DashScope rerankers. The default paper task-eval
commands below use local BAAI reranking and CloseAI-compatible GPT models, so
also set:

```bash
export CLOSEAI_API_KEY=...
```

`CLOSEAI_API_KEY` falls back to `OPENAI_API_KEY` in the current provider
configuration. If you switch to OpenRouter, set `OPENROUTER_API_KEY`.

## Frozen Artifact Check

Before a long run, verify the entrypoints:

```bash
uv run python -m benchmark_locomo.task_eval.locomo_triple_input_speed --help
uv run python -m benchmark_locomo.task_eval.locomo_triple_smart_search_qps --help
uv run python -m benchmark_locomo.task_eval.locomo_triple_router_quantification --help
```

In this frozen checkout, the first two speed entrypoints import successfully.
The router-quantification command should also print its CLI options without
starting a benchmark run.

## Data And Graph Preparation

Download the public LoCoMo10 source file from the official LoCoMo repository:

```bash
mkdir -p benchmark_locomo/dataset/locomo
curl -fL https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json \
  -o benchmark_locomo/dataset/locomo/locomo10.json
```

Dataset source:

- https://github.com/snap-research/locomo
- https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json

The task-eval script reads these default graph locations from
`src/mandol/core/paths.py`:

```text
benchmark_locomo/dataset/locomo/locomo10.json
benchmark_locomo/dataset/locomo/hierarchical_content/step4_semantic_graphs
benchmark_locomo/dataset/locomo/entity_relation/step3_semantic_graph
benchmark_locomo/dataset/locomo/episodic_memory/step3_loaded
```

If these outputs already exist and match the paper snapshot, you can skip graph
generation. To regenerate them:

```bash
bash benchmark_locomo/dataset_maker/run_all_locomo_dataset_maker_workflows.sh
bash benchmark_locomo/dataset_maker/build_unified.sh
```

The first script runs the three offline makers:

- `locomo_episodic_memory/scripts/pipeline.sh`
- `locomo_graph_maker/scripts/pipeline.sh`
- `locomo_hierarchical_content_maker/scripts/pipeline.sh`

The second script builds unified per-sample graphs under:

```text
benchmark_locomo/dataset/locomo/unified_per_sample_graphs
```

Those unified graphs are used by the smart-search QPS benchmark.

## Paper Accuracy Run

The paper main run is router + cascade quantification. The shell script
`benchmark_locomo/scripts/run_router_quantification_cascade_closeai.sh`
contains the parameter template and already uses the GPT task-eval model family. Keep
Qwen/DeepSeek in the memory-generation and deduplication stage, not in the
paper task-eval `--llm-model`.

Recommended dynamic-adaptive paper commands:

```bash
uv run python -m benchmark_locomo.task_eval.locomo_triple_router_quantification \
  --qa-dataset benchmark_locomo/dataset/locomo/locomo10.json \
  --llm-model gpt-4.1-mini-closeai \
  --llm-evaluate-model gpt-4o-mini-closeai \
  --topk-hierarchical 15 \
  --topk-similarity 30 \
  --topk-graph 0 \
  --topk-episodic 30 \
  --final-top-k 20 \
  --weight-hierarchical 0.34 \
  --weight-graph 0.33 \
  --weight-episodic 0.33 \
  --enable-second-stage-rerank \
  --rerank-strategy tower_separate \
  --reranker-type baai \
  --enable-router \
  --router-strategy aggressive \
  --enable-cascade-pruner \
  --cascade-prune-mode DYNAMIC_ADAPTIVE \
  --cascade-adaptive-dataset locomo \
  --cascade-mad-multiplier 3.0 \
  --cascade-lambda-mmr 0.7 \
  --cascade-cliff-tolerance 2.5 \
  --cascade-max-context-tokens 1500 \
  --output-dir benchmark_locomo/task_eval/results/paper_router_quantification/gpt41_mini_dynamic \
  --log-level INFO

uv run python -m benchmark_locomo.task_eval.locomo_triple_router_quantification \
  --qa-dataset benchmark_locomo/dataset/locomo/locomo10.json \
  --llm-model gpt-4o-mini-closeai \
  --llm-evaluate-model gpt-4o-mini-closeai \
  --topk-hierarchical 15 \
  --topk-similarity 30 \
  --topk-graph 0 \
  --topk-episodic 30 \
  --final-top-k 20 \
  --weight-hierarchical 0.34 \
  --weight-graph 0.33 \
  --weight-episodic 0.33 \
  --enable-second-stage-rerank \
  --rerank-strategy tower_separate \
  --reranker-type baai \
  --enable-router \
  --router-strategy aggressive \
  --enable-cascade-pruner \
  --cascade-prune-mode DYNAMIC_ADAPTIVE \
  --cascade-adaptive-dataset locomo \
  --cascade-mad-multiplier 3.0 \
  --cascade-lambda-mmr 0.7 \
  --cascade-cliff-tolerance 3.0 \
  --cascade-max-context-tokens 1800 \
  --output-dir benchmark_locomo/task_eval/results/paper_router_quantification/gpt4o_mini_dynamic \
  --log-level INFO
```

When router and cascade are enabled, the script appends suffixes to the output
directory. The actual output directory will end with:

```text
_routed_aggressive_cascade
```

The current router implementation has explicit routing tables for GPT-4.1-mini
and GPT-4o-mini. The commands above use those calibrated tables. Qwen/DeepSeek
belong to memory generation and deduplication, not the task-eval `--llm-model`
or `--llm-evaluate-model` for the paper accuracy rows.

For a cheaper smoke run, restrict to one sample and one formal question:

```bash
uv run python -m benchmark_locomo.task_eval.locomo_triple_router_quantification \
  --sample-ids conv-30 \
  --max-questions 1 \
  --llm-model gpt-4.1-mini-closeai \
  --llm-evaluate-model gpt-4o-mini-closeai \
  --topk-hierarchical 15 \
  --topk-similarity 30 \
  --topk-graph 0 \
  --topk-episodic 30 \
  --final-top-k 20 \
  --enable-router \
  --router-strategy aggressive \
  --enable-cascade-pruner \
  --cascade-prune-mode DYNAMIC_ADAPTIVE \
  --cascade-adaptive-dataset locomo \
  --output-dir benchmark_locomo/task_eval/results/smoke/gpt41_mini
```

## Accuracy Outputs

The main script writes per-sample and final reports into the output directory:

```text
sample_conv-*.json
sample_*_readable_*.txt
final_summary_*.json
final_summary_readable_*.txt
```

Use `final_summary_*.json` as the machine-readable aggregate. The readable text
file is useful for quickly checking LLM accuracy, category breakdown, routing
distribution, token usage, and cascade statistics.

## Speed Runs

The combined speed script is:

```bash
bash benchmark_locomo/scripts/run_speed_benchmarks.sh
```

For paper retrieval-time reporting, the two core commands are the insertion
speed benchmark and the smart-search fixed-QPS benchmark.

The smart-search QPS benchmark reads unified per-sample graphs. Build them after
the episodic, entity-relation, and hierarchical graph makers have finished:

```bash
bash benchmark_locomo/dataset_maker/build_unified.sh
```

This wrapper calls `benchmark_locomo/dataset_maker/build_unified_graph.py` and
writes:

```text
benchmark_locomo/dataset/locomo/unified_per_sample_graphs
```

Measurement scope:

- `locomo_triple_input_speed.py` measures only each
  `SemanticGraph.add_unit(...)` call body with incremental index updates and
  realtime dense and SPLADE sparse embedding generation. It uses
  `index_update_mode="incremental"` and excludes scheduling sleep, graph
  initialization, warmup, and output writing.
- `locomo_triple_smart_search_qps.py` measures each scheduled
  `MultiRetriever.smart_search(...)` or `smart_search_async(...)` request after
  graph load and warmup. `latency_ms` covers base retrieval, score fusion,
  reranking when `--rerank-method` is set, response parsing, and Python
  async/thread wrapper overhead. The commands below pass `--rerank-method baai`,
  so the reported smart-search QPS numbers include reranking. It excludes graph
  loading, warmup, fixed-QPS scheduling sleep, and report writing. The report
  also separates `retrieval_time_ms` and `rerank_time_ms`.

Insertion speed:

```bash
uv run python -m benchmark_locomo.task_eval.locomo_triple_input_speed \
  --data-dir benchmark_locomo/dataset/locomo \
  --total-requests 2000 \
  --qps 10
```

The default output is:

```text
benchmark_locomo/task_eval/results/locomo_tri_tower_input_speed_results/benchmark_triple_input_speed_<timestamp>.json
```

Smart-search QPS with vLLM HTTP reranker:

```bash
RERANKER_BACKEND=vllm \
VLLM_API_URL="${VLLM_API_URL:-http://127.0.0.1:8000/score}" \
uv run python -m benchmark_locomo.task_eval.locomo_triple_smart_search_qps \
  --qps 10 \
  --top-k 35 \
  --rerank-method baai
```

Smart-search QPS with native local reranker:

```bash
RERANKER_BACKEND=native \
uv run python -m benchmark_locomo.task_eval.locomo_triple_smart_search_qps \
  --qps 10 \
  --top-k 35 \
  --rerank-method baai
```

The QPS benchmark writes timestamped reports under:

```text
benchmark_locomo/task_eval/results/smart_search_qps_results/
```

Run both `--qps 10` and `--qps 5` if you need the same comparison matrix as
`benchmark_locomo/scripts/run_speed_benchmarks.sh`.

## Related Scripts

- `benchmark_locomo/scripts/run_router_quantification_cascade_closeai.sh`:
  parameter-rich router + quantification + cascade template for the paper-style
  CloseAI runs.
- `benchmark_locomo/scripts/run_reproduction_suite.sh`: expanded reproduction
  launcher for historical baselines, router runs, cascade runs, and ablations.
- `benchmark_locomo/scripts/run_router_only_closeai.sh`: router-only baseline,
  not the paper router + quantification main entry.
- `benchmark_locomo/scripts/run_speed_benchmarks.sh`: serial speed-test
  pipeline used for insertion speed and smart-search QPS measurements.
