# LongMemEval Reproduction Guide

> This guide belongs to the frozen `paper-repro` artifact, whose package
> version is `0.1.0`. Current runtime development continues on `main`; use the
> `paper-repro` branch or `v0.1.0-paper-repro` tag for paper comparisons.

This guide documents the LongMemEval reproduction path used for the paper
numbers in this repository. Run commands from the repository root unless noted
otherwise.

## Scope

The paper accuracy numbers for LongMemEval come from the router +
quantification workflow:

```text
benchmark_longmemeval/task_eval/benchmark_triple_router_quantification.py
```

The intended paper model setup is:

- Memory/extraction generation backbone: `qwen-3-plus`
- Memory deduplication backbone: `deepseek-v3.2-dashscope`
- Task-eval evaluated models (`--llm-model`) reported in the paper:
  `gpt-4.1-mini-closeai` and `gpt-4o-mini-closeai`
- Task-eval judge model (`--llm-evaluate-model`) used by the historical
  scripts: `gpt-4o-mini-closeai`
- Dataset size: `s` by default unless a table explicitly says otherwise
- Fusion method: `concatenation`
- Reranker: `baai`
- Tower budgets: sentence `60`, episodic `40`, entity `40`, final `25`

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

Before a long run, verify the entrypoint:

```bash
uv run python -m benchmark_longmemeval.task_eval.benchmark_triple_router_quantification --help
```

This command should print the CLI options for dataset size, tower graph
directories, router settings, and cascade quantification settings. It does not
run the benchmark.

## Data And Graph Preparation

Download the cleaned LongMemEval split from the official LongMemEval dataset
release:

```bash
mkdir -p benchmark_longmemeval/dataset/LongMemEval
curl -fL https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json \
  -o benchmark_longmemeval/dataset/LongMemEval/longmemeval_s_cleaned.json
```

If you plan to run `--dataset-size m`, also download the medium split:

```bash
curl -fL https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_m_cleaned.json \
  -o benchmark_longmemeval/dataset/LongMemEval/longmemeval_m_cleaned.json
```

Dataset source:

- https://github.com/xiaowu0162/LongMemEval
- https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned

The task-eval script reads these default graph locations from
`src/mandol/core/paths.py`:

```text
benchmark_longmemeval/dataset/LongMemEval/longmemeval_s_cleaned.json
benchmark_longmemeval/dataset/LongMemEval/longmemeval_m_cleaned.json
benchmark_longmemeval/dataset/LongMemEval/longmemeval_hierarchical/step3_semantic_graphs
benchmark_longmemeval/dataset/LongMemEval/episodic_memory_graphs_new
benchmark_longmemeval/dataset/LongMemEval/entity_relation_graphs_new
```

If these outputs already exist and match the paper snapshot, you can skip graph
generation. To regenerate the three towers, run the maker scripts in this
order.

Hierarchical tower:

```bash
bash benchmark_longmemeval/dataset_maker/longmemeval_hierarchical/scripts/pipeline.sh
```

Episodic tower:

```bash
bash benchmark_longmemeval/dataset_maker/longmemeval_episodic_memory_new/qwen3-plus-sh/step1.sh
bash benchmark_longmemeval/dataset_maker/longmemeval_episodic_memory_new/qwen3-plus-sh/step2.sh
bash benchmark_longmemeval/dataset_maker/longmemeval_episodic_memory_new/scripts/step3.sh
bash benchmark_longmemeval/dataset_maker/longmemeval_episodic_memory_new/scripts/step4.sh
```

Entity-relation tower:

```bash
bash benchmark_longmemeval/dataset_maker/longmemeval_entity_relation_new/qwen3-plus-sh/step1.sh
bash benchmark_longmemeval/dataset_maker/longmemeval_entity_relation_new/qwen3-plus-sh/step1.5.sh
bash benchmark_longmemeval/dataset_maker/longmemeval_entity_relation_new/scripts/step2.sh
bash benchmark_longmemeval/dataset_maker/longmemeval_entity_relation_new/scripts/step2.5.sh
bash benchmark_longmemeval/dataset_maker/longmemeval_entity_relation_new/scripts/step3.sh
```

The `qwen3-plus-sh` scripts generate DashScope batch request files in 50-QA
ranges. Follow the printed upload/download instructions for the DashScope batch
API, place successful result JSONL files back under the corresponding
`batch_results` directory, and then continue with deduplication and graph-save
steps.

The deduplication steps default to `deepseek-v3.2-dashscope`. If you invoke the
Python modules manually, keep that model fixed:

```bash
uv run python -m benchmark_longmemeval.dataset_maker.longmemeval_episodic_memory_new.step3_deduplication \
  --dedup-model deepseek-v3.2-dashscope

uv run python -m benchmark_longmemeval.dataset_maker.longmemeval_entity_relation_new.step2_entity_deduplication \
  --auto-load \
  --dedup-model deepseek-v3.2-dashscope
```

## Paper Accuracy Run

The paper main run is router + cascade quantification. The shell script
`benchmark_longmemeval/scripts/run_reproduction_suite.sh` contains the expanded
parameter template and already uses the GPT task-eval model family. Keep
Qwen/DeepSeek in the memory-generation and deduplication stage, not in the
paper task-eval `--llm-model`.

Recommended dynamic-adaptive paper commands:

```bash
uv run python -m benchmark_longmemeval.task_eval.benchmark_triple_router_quantification \
  --dataset-size s \
  --llm-model gpt-4.1-mini-closeai \
  --llm-evaluate-model gpt-4o-mini-closeai \
  --sentence-top-k 60 \
  --episodic-top-k 40 \
  --entity-top-k 40 \
  --final-top-k 25 \
  --rerank-method baai \
  --fusion-method concatenation \
  --enable-router \
  --router-strategy aggressive \
  --enable-cascade-pruner \
  --cascade-prune-mode DYNAMIC_ADAPTIVE \
  --cascade-adaptive-dataset longmemeval \
  --cascade-mad-multiplier 3.0 \
  --cascade-lambda-mmr 0.6 \
  --cascade-cliff-tolerance 2.5 \
  --cascade-max-context-tokens 2000 \
  --output-dir benchmark_longmemeval/task_eval/results/paper_router_quantification/gpt41_mini_dynamic

uv run python -m benchmark_longmemeval.task_eval.benchmark_triple_router_quantification \
  --dataset-size s \
  --llm-model gpt-4o-mini-closeai \
  --llm-evaluate-model gpt-4o-mini-closeai \
  --sentence-top-k 60 \
  --episodic-top-k 40 \
  --entity-top-k 40 \
  --final-top-k 25 \
  --rerank-method baai \
  --fusion-method concatenation \
  --enable-router \
  --router-strategy aggressive \
  --enable-cascade-pruner \
  --cascade-prune-mode DYNAMIC_ADAPTIVE \
  --cascade-adaptive-dataset longmemeval \
  --cascade-mad-multiplier 3.0 \
  --cascade-lambda-mmr 0.6 \
  --cascade-cliff-tolerance 3.0 \
  --cascade-max-context-tokens 2200 \
  --output-dir benchmark_longmemeval/task_eval/results/paper_router_quantification/gpt4o_mini_dynamic
```

When router and cascade are enabled, the script appends suffixes to the output
directory. The actual output directory will end with:

```text
_routed_aggressive_cascade
```

The current router implementation has explicit routing tables for GPT-5,
GPT-4.1-mini, and GPT-4o-mini. The commands above use the GPT-4.1-mini and
GPT-4o-mini calibrated tables. Qwen/DeepSeek belong to memory generation and
deduplication, not the task-eval `--llm-model` or `--llm-evaluate-model` for
the paper accuracy rows.

For a cheap smoke run:

```bash
uv run python -m benchmark_longmemeval.task_eval.benchmark_triple_router_quantification \
  --dataset-size s \
  --start-qa 0 \
  --end-qa 0 \
  --max-tests 1 \
  --llm-model gpt-4.1-mini-closeai \
  --llm-evaluate-model gpt-4o-mini-closeai \
  --sentence-top-k 60 \
  --episodic-top-k 40 \
  --entity-top-k 40 \
  --final-top-k 25 \
  --rerank-method baai \
  --fusion-method concatenation \
  --enable-router \
  --router-strategy aggressive \
  --enable-cascade-pruner \
  --cascade-prune-mode DYNAMIC_ADAPTIVE \
  --cascade-adaptive-dataset longmemeval \
  --output-dir benchmark_longmemeval/task_eval/results/smoke/gpt41_mini
```

## Accuracy Outputs

The main script writes:

```text
summary_<timestamp>.json
results_<timestamp>.json
report_<timestamp>.txt
individual_reports/qa_<index>_report.json
```

Use `summary_<timestamp>.json` as the machine-readable aggregate. It includes
test counts, score averages, token statistics, retrieval statistics, router
configuration, and cascade quantification statistics.

## Related Scripts

- `benchmark_longmemeval/scripts/run_reproduction_suite.sh`:
  expanded historical launcher for router + cascade, router-only baselines, and
  ablations.
- `benchmark_longmemeval/scripts/run_router_only_closeai.sh`:
  router-only baseline, not the paper router + quantification main entry.
- `benchmark_longmemeval/scripts/run_triple_tower_cli.sh`:
  configurable tri-tower baseline without router + quantification.
- `benchmark_longmemeval/scripts/run_speed_benchmarks.sh`:
  documents that LongMemEval speed helpers are private in the public artifact.
  Public latency/QPS runs use `benchmark_locomo/scripts/run_speed_benchmarks.sh`.
