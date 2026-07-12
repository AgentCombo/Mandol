# LongMemEval Experimental Self-host Workflow

> [!NOTE]
> This is an experimental self-host workflow for the refactored `main` branch.
> It is not the exact pipeline used to produce the paper results. For faithful
> paper reproduction, use the
> [LongMemEval workflow in `paper-repro`](https://github.com/AgentCombo/Mandol/blob/paper-repro/benchmark_longmemeval/REPRODUCE.md).

Development guide for evaluating the refactored Mandol runtime on the
LongMemEval dataset.

## Overview

LongMemEval is a benchmark for evaluating long-term memory in conversational AI systems. Each sample consists of a large haystack of dialogue sessions (up to 100+) with a single question requiring the system to locate and recall specific personal information buried in the conversation history.

**Question categories** (7 types):

| Type | Description |
|------|-------------|
| SS-User | Single-session, answer in a user turn |
| SS-Asst | Single-session, answer in an assistant turn |
| SS-Pref | Single-session, user preference/opinion |
| Temporal | Time-related reasoning |
| Multi-S | Multi-session, answer spans multiple sessions |
| Know.Upd. | Knowledge update — new info overrides old |
| — | — |

- **Data file**: `longmemeval_s_cleaned.json` in `data/` directory
- **Dataset source**: [HuggingFace](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)
- **Samples**: 500 (each with one question)

## Pipeline Overview

This development workflow follows a four-stage pipeline:

```
build_graph → retrieve → generate → evaluate
```

| Step | Script | Description |
|------|--------|-------------|
| 1 | `build_graph.py` | Load LongMemEval dataset and construct multi-dimensional semantic graph |
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

## Data Preparation

Download the dataset from HuggingFace:

```bash
# Option 1: Using huggingface-cli
huggingface-cli download xiaowu0162/longmemeval-cleaned --repo-type dataset --local-dir ./data

# Option 2: Using git lfs
git lfs install
git clone https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned ./data
```

The expected file is `data/longmemeval_s_cleaned.json`.

## Configuration

The benchmark uses two configuration layers:

### Layer 1: Environment Variables (`.env`)

Controls **provider connectivity**: API keys, base URLs, models, timeouts, and retry settings. Loaded from the `.env` file in the project root. See [`.env.example`](../../../.env.example) for all available variables.

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

### Layer 2: YAML Config Files (`configs/base.yaml`)

Controls **experiment parameters**: sample selection, retrieval settings, generation parameters, and system configuration.

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
  question_ids: []
  skip_types: []
  output_dir: "output"
  config_name: null
  dataset_path: "data/longmemeval_s_cleaned.json"

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
| `experiment` | Question IDs, skipped types, output directory, dataset path |
| `retrieval` | Top-K, which views to skip |
| `generation` | Max tokens, temperature |
| `evaluation` | Number of LLM judge runs |

To test only specific samples, edit the `experiment.question_ids` field:

```yaml
experiment:
  question_ids: ["e47becba", "118b2229"]  # Process only these samples
  # question_ids: []                       # Process all samples
```

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

Runs a self-contained pipeline on a small data subset (sample `e47becba`, 3 sessions, 1 query) without spawning subprocesses. Includes a persistence round-trip test. Use `--keep-output` to retain the temporary output directory.

### Step-by-Step Execution

```bash
# Step 1: Build graph
python build_graph.py --config configs/base.yaml --data data/longmemeval_s_cleaned.json --output output/

# Step 2: Retrieve
python retrieve.py --config configs/base.yaml --data data/longmemeval_s_cleaned.json --output output/

# Step 3: Generate
python generate.py --config configs/base.yaml --output output/

# Step 4: Evaluate
python evaluate.py --config configs/base.yaml --output output/
```

Add `--force` to any step to re-run it even if results already exist.

## Published Paper Reference Results

The tables below are published-paper reference results only. They are not the
expected output of this refactored development workflow. Use the frozen
[`paper-repro` LongMemEval instructions](https://github.com/AgentCombo/Mandol/blob/paper-repro/benchmark_longmemeval/REPRODUCE.md)
for comparisons against these values.

### GPT-4o-mini Backbone

| System | Avg.Tok. | SS-Pref | SS-Asst | Temporal | Multi-S | Know.Upd. | SS-User | Overall |
|--------|----------|---------|---------|----------|---------|-----------|---------|---------|
| Mem0 | 1.1k | 90.00 | 26.78 | 72.18 | 63.15 | 66.67 | 82.86 | 66.40 |
| Zep | 1.6k | 53.30 | 75.00 | 54.10 | 47.40 | 74.40 | 92.90 | 63.80 |
| MEMOS | 1.4k | 96.67 | 67.86 | 77.44 | 70.67 | 74.26 | 95.71 | 77.80 |
| **Mandol (Ours)** | 2.1k | **96.67** | **98.21** | **78.95** | **74.44** | **88.46** | **97.14** | **85.00** |

### GPT-4.1-mini Backbone

| System | Avg.Tok. | SS-Pref | SS-Asst | Temporal | Multi-S | Know.Upd. | SS-User | Overall |
|--------|----------|---------|---------|----------|---------|-----------|---------|---------|
| EverMemOS | 2.8k | 93.33 | 85.71 | 77.44 | 73.68 | 89.74 | 97.14 | 83.00 |
| **Mandol (Ours)** | 2.3k | **96.67** | **98.21** | **87.22** | **77.44** | **89.74** | **98.57** | **88.40** |

> **Note**: SS denotes Single-Session. Best overall results are in **bold**.

## Output Files

After a full pipeline run, the output directory contains:

```
output/<config_name>/
├── build_stats.json          # Build summary (samples, sessions, units, duration, token usage)
├── evaluation_summary.json   # Evaluation results (overall accuracy, per-type breakdown)
├── evaluation_report.txt     # Human-readable evaluation report
└── <question_id>/
    ├── build.json            # Per-sample build metrics
    ├── retrieval.json        # Retrieved hits (with scores and ranks)
    ├── generation.json       # Generated answer (raw and extracted)
    ├── evaluation.json       # LLM judge decision
    └── graph/                # Persisted MemorySystem state
        └── data/
            ├── units.json    # All memory units
            ├── spaces.json   # Memory space hierarchy
            ├── graph.json    # Relationship edges
            └── sessions.json # Detected sessions
```

## Directory Structure

```
longmemeval/
├── README.md              # This file
├── run.py                 # End-to-end pipeline orchestrator (+ smoke test)
├── build_graph.py         # Step 1: Build graph
├── retrieve.py            # Step 2: Retrieve
├── generate.py            # Step 3: Generate
├── evaluate.py            # Step 4: Evaluate
├── pipeline_utils.py      # Shared utilities and prompt templates
├── adapter/               # LongMemEval adapter for Mandol
│   ├── __init__.py
│   └── longmemeval_adapter.py
├── data/                  # Dataset directory (download required)
│   └── longmemeval_s_cleaned.json
├── configs/               # Experiment configurations
│   └── base.yaml
└── output/                # Pipeline output
```

## Relationship to the Paper Artifact

The workflow in this directory has been refactored for self-hosted and
general-purpose use. Its prompts, graph-construction pipeline, retrieval
stages, model assignments, configuration defaults, and evaluation process may
differ from the frozen paper artifact.

For comparisons against the published paper tables, use the
[LongMemEval workflow in the `paper-repro` branch](https://github.com/AgentCombo/Mandol/blob/paper-repro/benchmark_longmemeval/REPRODUCE.md).
