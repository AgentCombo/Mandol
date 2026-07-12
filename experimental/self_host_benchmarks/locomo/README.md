# LoCoMo Experimental Self-host Workflow

> [!NOTE]
> This is an experimental self-host workflow for the refactored `main` branch.
> It is not the exact pipeline used to produce the paper results. For faithful
> paper reproduction, use the
> [LoCoMo workflow in `paper-repro`](https://github.com/AgentCombo/Mandol/blob/paper-repro/benchmark_locomo/REPRODUCE.md).

Development guide for evaluating the refactored Mandol runtime on the LoCoMo
(Long Conversational Memory) dataset.

## Overview

LoCoMo is a benchmark designed to evaluate long-term conversational memory systems. It tests a system's ability to recall, reason over, and synthesize information from multi-session dialogues. The dataset contains five query categories; the main evaluation covers four (Single-hop, Multi-hop, Temporal, Open-domain), while Adversarial queries (category 5) are excluded by default.

## Pipeline Overview

This development workflow follows a four-stage pipeline:

```
build_graph → retrieve → generate → evaluate
```

| Step | Script | Description |
|------|--------|-------------|
| 1 | `build_graph.py` | Load LoCoMo dataset and construct multi-dimensional semantic graph |
| 2 | `retrieve.py` | Execute retrieval queries against the built graph |
| 3 | `generate.py` | Generate answers using LLM based on retrieved context |
| 4 | `evaluate.py` | Score generated answers against ground truth |

Each step communicates through JSON files on disk, enabling independent execution, incremental resume, and easy debugging. The pipeline can be run end-to-end via `run.py` or step-by-step via individual scripts.

## Test Environment

| Component | Specification |
|-----------|--------------|
| CPU | Intel Xeon Platinum 8458P |
| RAM | 120 GB |
| GPU | NVIDIA H800 80GB |
| Python | 3.10.12 |
| OS | Ubuntu 22.04 LTS |

> **Note**: Results may vary across hardware and provider versions. The above
> environment is currently used to validate this development workflow.

## Dataset Description

- **LoCoMo (Long Conversational Memory)**: 10 long multi-session dialogues, each with multiple QA pairs
- **5 query categories**:
  - **Single-hop** (category 1): Direct fact retrieval from a single dialogue turn
  - **Multi-hop** (category 2): Reasoning across multiple dialogue turns or sessions
  - **Temporal** (category 3): Questions involving time-based ordering or recency
  - **Open-domain** (category 4): Broad questions requiring comprehensive memory synthesis
  - **Adversarial** (category 5): Questions designed to confuse or mislead the retrieval system
- **Data file**: `locomo10.json` in `data/` (a bundled development/evaluation dataset copy)

## Key Metrics

| Metric | Description |
|--------|-------------|
| LLM Judge Accuracy | Correctness judged by an LLM grader (primary metric) |
| Per-Category Accuracy | Breakdown by query category (single-hop, multi-hop, temporal, open-domain) |
| Token Usage | Total LLM tokens consumed during build and generation phases |
| Retrieval Latency | Per-query time for the retrieval pipeline |

## Environment Setup

```bash
cp ../../../.env.example ../../../.env
# Edit ../../../.env with the provider endpoints and credentials to use.
```

The optional `scripts/env.sh` file is retained for standalone development
utilities. Source it only when those legacy environment variable names are
required; the documented four-stage workflow uses the repository-level `.env`.

## Configuration

The benchmark uses two configuration layers:

### Layer 1: Environment Variables (`.env`)

Controls **provider connectivity**: API keys, base URLs, models, timeouts, and retry settings. Loaded from the `.env` file in the project root. See [`.env.example`](../../../.env.example) for all available variables.

Key variables for the benchmark:

| Variable | Purpose | Default |
|----------|---------|---------|
| `MANDOL_LLM_API_KEY` | API key for the LLM provider | — |
| `MANDOL_LLM_BASE_URL` | Base URL for the LLM API | `https://api.openai.com/v1` |
| `MANDOL_LLM_MODEL` | Model name for generation and evaluation | `gpt-4o-mini` |
| `MANDOL_LLM_TIMEOUT_S` | Request timeout for LLM calls | `60` |
| `MANDOL_EMBEDDER_API_KEY` | API key for the embedding provider | — |
| `MANDOL_EMBEDDER_BASE_URL` | Base URL for the embedding API | — |
| `MANDOL_EMBEDDER_TIMEOUT_S` | Request timeout for embedding calls | `60` |
| `MANDOL_RERANKER_API_KEY` | API key for the reranker | — |
| `MANDOL_RERANKER_BASE_URL` | Base URL for the reranker API | — |
| `MANDOL_RERANKER_TIMEOUT_S` | Request timeout for rerank calls | `60` |

> **Priority**: Environment variables take the highest precedence. If a variable is set in both `.env` and a YAML config, the environment variable wins.

### Layer 2: YAML Config Files (`configs/*.yaml`)

Controls **experiment parameters**: sample selection, retrieval settings, generation parameters, and system configuration. Located under `configs/`.

Default `configs/base.yaml`:

```yaml
embedder:
  dimension: 4096

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

experiment:
  sample_ids: ["conv-41"]
  skip_categories: [5]
  output_dir: "output"
  config_name: null

retrieval:
  top_k: 10
  skip_views: []

generation:
  max_tokens: 256
  temperature: 0.3

evaluation:
  llm_judge_runs: 1
```

| Section | Controls |
|---------|----------|
| `embedder` | Embedding dimension |
| `system` | BFS expansion, similarity thresholds, chunk size, session detection |
| `storage` | Persistence root and auto-save settings |
| `experiment` | Sample IDs, skipped categories, output directory |
| `retrieval` | Top-K, which views to skip |
| `generation` | Max tokens, temperature |
| `evaluation` | Number of LLM judge runs |

To test only samples `conv-1` and `conv-2`, edit the config:

```yaml
experiment:
  sample_ids: ["conv-1", "conv-2"]
```

Set `sample_ids: []` to process all samples.

> **Note**: The `adapter/config.py` dataclass (`LocomoAdapterConfig`) is for the older adapter path and is separate from the YAML pipeline configs. When running the 4-step pipeline, use the YAML configs.

`locomo_benchmark.py` and `evaluation.py` are standalone development utilities
retained for ongoing migration. They are not imported by the documented
`run.py` workflow and may require additional experimental modules.

## Data Preparation

The repository includes `data/locomo10.json` as a development/evaluation copy.
It must not be treated as the authoritative input artifact for the frozen paper
pipeline; follow the `paper-repro` guide when comparing against paper tables.

## Run Workflow

### End-to-end Pipeline (recommended)

```bash
python run.py --config configs/base.yaml --output output/
```

With forced rebuild:

```bash
python run.py --config configs/base.yaml --output output/ --force
```

Run specific stages only:

```bash
python run.py --config configs/base.yaml --stages build,retrieve
```

### Smoke Test (fast validation)

```bash
python run.py --smoke --config configs/base.yaml
```

Runs a self-contained pipeline on a small data subset (sample `conv-26`, 3 sessions, 5 QA) without spawning subprocesses. Includes a persistence round-trip test. Use `--keep-output` to retain the temporary output directory.

### Step-by-Step Execution

```bash
# Step 1: Build graph
python build_graph.py --config configs/base.yaml --data data/locomo10.json --output output/

# Step 2: Retrieve
python retrieve.py --config configs/base.yaml --data data/locomo10.json --output output/

# Step 3: Generate
python generate.py --config configs/base.yaml --output output/

# Step 4: Evaluate
python evaluate.py --config configs/base.yaml --output output/
```

Add `--force` to any step to re-run it even if results already exist.

## Published Paper Reference Results

The tables below are published-paper reference results only. They are not the
expected output of this refactored development workflow. Use the frozen
[`paper-repro` LoCoMo instructions](https://github.com/AgentCombo/Mandol/blob/paper-repro/benchmark_locomo/REPRODUCE.md)
for comparisons against these values.

### GPT-4o-mini Backbone

| System | Avg. Tok. | Single-hop | Multi-hop | Temporal | Open-domain | Overall |
|--------|-----------|------------|-----------|----------|-------------|---------|
| Mem0 | 1.0k | 66.71 | 58.16 | 55.45 | 40.62 | 61.00 |
| MemU | 4.0k | 72.77 | 62.41 | 33.96 | 46.88 | 61.15 |
| MemOS | 2.5k | 81.45 | 69.15 | 72.27 | 60.42 | 75.87 |
| Zep | 1.4k | 88.11 | 71.99 | 74.45 | 66.67 | 81.06 |
| EverMemOS† | 2.5k | 91.68 | 82.74 | 79.34 | 70.14 | 86.13 |
| **Mandol (Ours)** | **1.9k** | **93.82** | **85.11** | **89.10** | 65.63 | **89.48** |

### GPT-4.1-mini Backbone

| System | Avg. Tok. | Single-hop | Multi-hop | Temporal | Open-domain | Overall |
|--------|-----------|------------|-----------|----------|-------------|---------|
| Mem0 | 1.0k | 68.97 | 61.70 | 58.26 | 50.00 | 64.20 |
| MemU | 4.0k | 74.91 | 72.34 | 43.61 | 54.17 | 66.67 |
| MemOS | 2.5k | 85.37 | 79.43 | 75.08 | 64.58 | 80.76 |
| Zep | 1.4k | 90.84 | 81.91 | 77.26 | 75.00 | 85.22 |
| EverMemOS† | 2.3k | 95.32 | 89.01 | 90.13 | 77.43 | 91.97 |
| **Mandol (Ours)** | **1.9k** | **95.36** | **92.20** | 87.85 | **79.17** | **92.21** |

> **Note**: † denotes results reproduced using the official EverMemOS implementation, with concurrency patches applied to ensure evaluation stability. The overall metric excludes adversarial queries (category 5). Best results per backbone are in **bold**.

## Output Files

After a full pipeline run, the output directory contains:

```
output/<config_name>/
├── build_stats.json          # Build summary (samples, sessions, units, duration, token usage)
├── evaluation_summary.json   # Evaluation results (overall accuracy, per-category breakdown)
├── evaluation_report.txt     # Human-readable evaluation report
└── <sample_id>/
    ├── build.json            # Per-sample build metrics
    ├── retrieval.json        # Retrieved hits per query (with scores and ranks)
    ├── generation.json       # Generated answers per query (raw and extracted)
    ├── evaluation.json       # LLM judge decisions per query
    └── graph/                # Persisted MemorySystem state
        └── data/
            ├── units.json    # All memory units
            ├── spaces.json   # Memory space hierarchy
            ├── graph.json    # Relationship edges
            └── sessions.json # Detected sessions
```

## Directory Structure

```
locomo/
├── README.md              # This file
├── run.py                 # End-to-end pipeline orchestrator (+ smoke test)
├── build_graph.py         # Step 1: Build graph
├── retrieve.py            # Step 2: Retrieve
├── generate.py            # Step 3: Generate
├── evaluate.py            # Step 4: Evaluate
├── evaluation.py          # Standalone evaluation utilities
├── pipeline_utils.py      # Shared utilities and prompt templates
├── locomo_benchmark.py    # Standalone development evaluation utility
├── adapter/               # LoCoMo adapter for Mandol
│   ├── __init__.py
│   ├── locomo_adapter.py
│   └── config.py
├── data/
│   └── locomo10.json      # LoCoMo dataset
├── scripts/
│   └── env.sh             # Environment setup
├── configs/               # Experiment configurations
│   └── base.yaml          # Full pipeline (baseline)
├── baselines/             # Baseline implementations
│   ├── README.md
│   ├── mem0/
│   └── letta/
├── results/               # Run logs
└── output/                # Pipeline output
```

## Relationship to the Paper Artifact

The workflow in this directory has been refactored for self-hosted and
general-purpose use. Its prompts, graph-construction pipeline, retrieval
stages, model assignments, configuration defaults, and evaluation process may
differ from the frozen paper artifact.

For comparisons against the published paper tables, use the
[LoCoMo workflow in the `paper-repro` branch](https://github.com/AgentCombo/Mandol/blob/paper-repro/benchmark_locomo/REPRODUCE.md).
