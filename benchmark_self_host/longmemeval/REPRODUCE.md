# LongMemEval Self-Host Reproduction Guide

> This self-host guide is preserved in the frozen `paper-repro` branch, but it
> is separate from the router + quantification workflow used for paper tables.

This directory reproduces LongMemEval with Mandol's own high-level memory
generation path. It is separate from the paper router + quantification workflow
under `benchmark_longmemeval`.

## Scope

The self-host workflow does not use router + quantification. It builds Mandol
memories directly, then retrieves, generates answers, and scores them:

```text
build_graph.py -> retrieve.py -> generate.py -> score.py
```

The one-click scripts live under:

```text
benchmark_self_host/longmemeval/scripts/
```

## Current Code Check

Before a long run, verify the build entrypoint:

```bash
uv run python -m benchmark_self_host.longmemeval.build_graph --help
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

The `longmemeval` auto-builder strategy currently defaults to
`qwen-3-plus-latest` for extraction and `deepseek-v3.2-dashscope` for
deduplication. For exact paper-aligned memory construction, override extraction
to `qwen-3-plus`. `GENERATION_MODEL` and `JUDGE_MODEL` control the downstream
answer-generation and scoring stages.

## Dataset

Download the cleaned LongMemEval small split from the official dataset release:

```bash
mkdir -p benchmark_self_host/longmemeval/dataset
curl -fL https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json \
  -o benchmark_self_host/longmemeval/dataset/longmemeval_s_cleaned.json
```

If you already downloaded the paper benchmark copy, reuse it:

```bash
cp benchmark_longmemeval/dataset/LongMemEval/longmemeval_s_cleaned.json \
  benchmark_self_host/longmemeval/dataset/longmemeval_s_cleaned.json
```

Dataset source:

- https://github.com/xiaowu0162/LongMemEval
- https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned

## One-Click Run

```bash
GENERATION_MODEL=gpt-4o-mini-closeai \
JUDGE_MODEL=gpt-4o-mini-closeai \
bash benchmark_self_host/longmemeval/scripts/run_all.sh
```

By default, outputs go to:

```text
benchmark_self_host/longmemeval/test_runs/<timestamp>__gen-<generation-model>__judge-<judge-model>/
```

The run root contains:

```text
memory/
retrieve/
generate/
score/
logs/
```

For LongMemEval, `generate.py` defaults to placing generation files beside the
retrieval files. The provided `run_generate.sh` script explicitly writes to
`$RUN_ROOT/generate`, and `run_score.sh` scores from that directory.

## Recommended Smoke Run

Use one QA and a small question cap before running all LongMemEval samples:

```bash
SAMPLE_IDS="qa_0" \
MAX_QUESTIONS=1 \
GENERATION_MODEL=gpt-4o-mini-closeai \
JUDGE_MODEL=gpt-4o-mini-closeai \
bash benchmark_self_host/longmemeval/scripts/run_all.sh
```

Alternatively, use range controls:

```bash
START_INDEX=0 \
END_INDEX=0 \
MAX_QUESTIONS=1 \
bash benchmark_self_host/longmemeval/scripts/run_all.sh
```

If you only want to validate graph construction without LLM answer generation:

```bash
SAMPLE_IDS="qa_0" \
BUILD_EXTRA_ARGS="--dry-run" \
bash benchmark_self_host/longmemeval/scripts/run_build.sh
```

## Stage-By-Stage Resume

Set the same `RUN_ROOT` and run the stage you need:

```bash
RUN_ROOT=benchmark_self_host/longmemeval/test_runs/<run-id> \
bash benchmark_self_host/longmemeval/scripts/run_build.sh

RUN_ROOT=benchmark_self_host/longmemeval/test_runs/<run-id> \
bash benchmark_self_host/longmemeval/scripts/run_retrieve.sh

RUN_ROOT=benchmark_self_host/longmemeval/test_runs/<run-id> \
GENERATION_MODEL=gpt-4o-mini-closeai \
bash benchmark_self_host/longmemeval/scripts/run_generate.sh

RUN_ROOT=benchmark_self_host/longmemeval/test_runs/<run-id> \
JUDGE_MODEL=gpt-4o-mini-closeai \
bash benchmark_self_host/longmemeval/scripts/run_score.sh
```

Build-stage controls:

```bash
START_INDEX=0
END_INDEX=499
LIMIT=50
FORCE=1
SKIP_EXISTING=1
NO_RESUME=1
ALLOW_PARTIAL_EXTRACTION=0
```

`ALLOW_PARTIAL_EXTRACTION=1` is the script default and passes
`--allow-partial-extraction` to `build_graph.py`. Set it to `0` when you want
the build to fail on any extraction group error.

## Build Configuration

`build_graph.py` builds one SemanticGraph per QA under
`$RUN_ROOT/memory/<sample_id>/`. The current design includes:

- L0 user-message and assistant-chunk memory units
- Episodic facts with LongMemEval-specific prompts
- Entity and relation memory units
- Optional relation extraction
- Comparison metadata against offline LongMemEval graph outputs

The default dataset and offline reference root are:

```text
benchmark_self_host/longmemeval/dataset/longmemeval_s_cleaned.json
benchmark_longmemeval/dataset/LongMemEval
```

Useful overrides:

```bash
BUILD_EXTRA_ARGS="--extraction-model qwen-3-plus --dedup-model deepseek-v3.2-dashscope"
BUILD_EXTRA_ARGS="--skip-hierarchical --skip-episodic"
BUILD_EXTRA_ARGS="--embedding-model Qwen/Qwen3-Embedding-0.6B --build-splade"
```

## Retrieval Configuration

`retrieve.py` reads `$RUN_ROOT/memory` and writes per-QA retrieval records to
`$RUN_ROOT/retrieve`.

Default LongMemEval retrieval parameters:

```text
sentence-top-k = 60
episodic-top-k = 40
entity-top-k   = 40
final-top-k    = 25
rerank-method  = baai
fusion-method  = rrf
```

Useful overrides:

```bash
RETRIEVE_EXTRA_ARGS="--sample-ids qa_0"
RETRIEVE_EXTRA_ARGS="--start-index 0 --end-index 9"
RETRIEVE_EXTRA_ARGS="--disable-second-stage-rerank"
RETRIEVE_EXTRA_ARGS="--rerank-method qwen --final-top-k 25"
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
evaluated counts, correct counts, LLM-judge score, timing metrics, and
question-type breakdowns.

## Useful Variables

```bash
SAMPLE_IDS="qa_0 qa_371"
START_INDEX=0
END_INDEX=499
LIMIT=50
MAX_QUESTIONS=20
FORCE=1
SKIP_EXISTING=1
NO_RESUME=1
ALLOW_PARTIAL_EXTRACTION=0
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
