# LoCoMo10 Self-Host Reproduction Guide

> This self-host guide is preserved in the frozen `paper-repro` branch, but it
> is separate from the router + quantification workflow used for paper tables.

This directory reproduces LoCoMo10 with Mandol's own high-level memory
generation path. It is separate from the paper router + quantification workflow
under `benchmark_locomo`.

## Scope

The self-host workflow does not use router + quantification. It builds Mandol
memories directly, then retrieves, generates answers, and scores them:

```text
build_graph.py -> retrieve.py -> generate.py -> score.py
```

The one-click scripts live under:

```text
benchmark_self_host/locomo10/scripts/
```

## Current Code Check

Before a long run, verify the build entrypoint:

```bash
uv run python -m benchmark_self_host.locomo10.build_graph --help
```

This command should print the `mandol.auto_builder` build options without
starting model calls.

## Environment

Use Python 3.12 and the repository `pyproject.toml` environment.

```bash
uv sync --extra dev --extra cuda --group spacy-model
export PYTHONPATH="$PWD/src:$PWD"
```

The paper performance environment uses the relevant extras. If your machine
cannot install the CUDA/flash-attention extra, omit `--extra cuda`; the self-host
workflow still runs, but throughput may differ.

Set model provider keys according to the generation and judge models you choose.
The existing scripts default to CloseAI-compatible models:

```bash
export CLOSEAI_API_KEY=...
```

If you override the build models to DashScope Qwen/DeepSeek models, also set:

```bash
export DASHSCOPE_API_KEY=...
```

The `locomo10` auto-builder strategy already defaults to
`qwen-3.5-plus-thinking` for extraction and `deepseek-v3.2-dashscope` for
deduplication. `GENERATION_MODEL` and `JUDGE_MODEL` control the downstream
answer-generation and scoring stages.

## Dataset

Download the public LoCoMo10 source file from the official LoCoMo repository:

```bash
mkdir -p benchmark_self_host/locomo10/dataset
curl -fL https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json \
  -o benchmark_self_host/locomo10/dataset/locomo10.json
```

If you already downloaded the paper benchmark copy, reuse it:

```bash
cp benchmark_locomo/dataset/locomo/locomo10.json \
  benchmark_self_host/locomo10/dataset/locomo10.json
```

Dataset source:

- https://github.com/snap-research/locomo
- https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json

## One-Click Run

```bash
GENERATION_MODEL=gpt-4o-mini-closeai \
JUDGE_MODEL=gpt-4o-mini-closeai \
bash benchmark_self_host/locomo10/scripts/run_all.sh
```

By default, outputs go to:

```text
benchmark_self_host/locomo10/test_runs/<timestamp>__gen-<generation-model>__judge-<judge-model>/
```

The run root contains:

```text
memory/
retrieve/
generate/
score/
logs/
```

## Recommended Smoke Run

Use a single sample and a small question cap before running all LoCoMo10
samples:

```bash
SAMPLE_IDS="conv-30" \
MAX_QUESTIONS=3 \
GENERATION_MODEL=gpt-4o-mini-closeai \
JUDGE_MODEL=gpt-4o-mini-closeai \
bash benchmark_self_host/locomo10/scripts/run_all.sh
```

If you only want to validate graph construction without LLM answer generation:

```bash
SAMPLE_IDS="conv-30" \
BUILD_EXTRA_ARGS="--dry-run" \
bash benchmark_self_host/locomo10/scripts/run_build.sh
```

## Stage-By-Stage Resume

Set the same `RUN_ROOT` and run the stage you need:

```bash
RUN_ROOT=benchmark_self_host/locomo10/test_runs/<run-id> \
bash benchmark_self_host/locomo10/scripts/run_build.sh

RUN_ROOT=benchmark_self_host/locomo10/test_runs/<run-id> \
bash benchmark_self_host/locomo10/scripts/run_retrieve.sh

RUN_ROOT=benchmark_self_host/locomo10/test_runs/<run-id> \
GENERATION_MODEL=gpt-4o-mini-closeai \
bash benchmark_self_host/locomo10/scripts/run_generate.sh

RUN_ROOT=benchmark_self_host/locomo10/test_runs/<run-id> \
JUDGE_MODEL=gpt-4o-mini-closeai \
bash benchmark_self_host/locomo10/scripts/run_score.sh
```

Build-stage controls:

```bash
FORCE=1          # rebuild existing sample directories
SKIP_EXISTING=1  # skip samples that already have output
NO_RESUME=1      # ignore build checkpoints
LIMIT=10         # build at most N samples after filtering
```

## Build Configuration

`build_graph.py` builds one SemanticGraph per sample under
`$RUN_ROOT/memory/<sample_id>/`. The current design includes:

- L0 per-session contextual retrieval units
- Hierarchical L1 and L2 units
- Episodic facts with optional deduplication
- Entity/relation memory units
- Resumable checkpoints and stage artifacts

Useful overrides:

```bash
BUILD_EXTRA_ARGS="--extraction-model qwen-3.5-plus-thinking --dedup-model deepseek-v3.2-dashscope"
BUILD_EXTRA_ARGS="--no-episodic-dedup --no-entity-dedup"
BUILD_EXTRA_ARGS="--embedding-model Qwen/Qwen3-Embedding-0.6B"
```

## Retrieval Configuration

`retrieve.py` reads `$RUN_ROOT/memory` and writes per-sample retrieval records to
`$RUN_ROOT/retrieve`.

Default LoCoMo10 retrieval parameters:

```text
topk-hierarchical = 15
topk-similarity   = 30
topk-episodic     = 30
final-top-k       = 20
reranker-type     = baai
rerank-strategy   = tower_separate
retrieval-mode    = both
```

Useful overrides:

```bash
RETRIEVE_EXTRA_ARGS="--sample-ids conv-30 --max-questions 5"
RETRIEVE_EXTRA_ARGS="--reranker-type qwen --final-top-k 20"
RETRIEVE_EXTRA_ARGS="--no-second-stage-rerank"
```

## Generation And Scoring

`generate.py` reads retrieval outputs and writes:

```text
$RUN_ROOT/generate/generate_summary.json
$RUN_ROOT/generate/<sample_id>/generation_results.jsonl
```

`score.py` reads generation outputs and writes:

```text
$RUN_ROOT/score/score_summary.json
$RUN_ROOT/score/score_results.jsonl
$RUN_ROOT/score/<sample_id>/score_summary.json
$RUN_ROOT/score/<sample_id>/score_results.jsonl
```

Use `score_summary.json` for the final self-host aggregate. It contains
evaluated counts, correct counts, LLM-judge score, timing metrics, and category
breakdowns.

## Useful Variables

```bash
SAMPLE_IDS="conv-26 conv-30"
LIMIT=10
MAX_QUESTIONS=20
FORCE=1
SKIP_EXISTING=1
NO_RESUME=1
BUILD_EXTRA_ARGS="..."
RETRIEVE_EXTRA_ARGS="..."
GENERATE_EXTRA_ARGS="..."
SCORE_EXTRA_ARGS="..."
DRY_RUN=1
JUDGE_WORKERS=4
```

The stage scripts log each command to:

```text
$RUN_ROOT/logs/build.log
$RUN_ROOT/logs/retrieve.log
$RUN_ROOT/logs/generate.log
$RUN_ROOT/logs/score.log
```
