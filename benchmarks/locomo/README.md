# LoCoMo Benchmark

Reproduction guide for Mandol's evaluation on the LoCoMo (Long Conversational Memory) benchmark dataset.

## Overview

LoCoMo is a benchmark designed to evaluate long-term conversational memory systems. It tests a system's ability to recall, reason over, and synthesize information from multi-session dialogues. The dataset contains five query categories; the main evaluation covers four (Single-hop, Multi-hop, Temporal, Open-domain), while Adversarial queries are tested separately in the ablation study.

## Pipeline Overview

The benchmark reproduction follows a 4-step pipeline:

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
| CPU | Intel Xeon E5-2680 v4 @ 2.40GHz |
| RAM | 64 GB DDR4 |
| GPU | NVIDIA Tesla V100 32GB |
| Python | 3.10.12 |
| OS | Ubuntu 22.04 LTS |

> **Note**: Results may vary on different hardware configurations. The above environment is used for the reference results reported in the paper.

## Dataset Description

- **LoCoMo (Long Conversational Memory)**: 10 long multi-session dialogues
- **5 query categories**:
  - **Single-hop**: Direct fact retrieval from a single dialogue turn
  - **Multi-hop**: Reasoning across multiple dialogue turns or sessions
  - **Temporal**: Questions involving time-based ordering or recency
  - **Open-domain**: Broad questions requiring comprehensive memory synthesis
  - **Adversarial**: Questions designed to confuse or mislead the retrieval system
- **Data file**: `locomo10.json` in `data/` directory

## Key Metrics

| Metric | Description |
|--------|-------------|
| LLM Judge Accuracy | Correctness judged by an LLM grader (primary metric) |
| Per-Category Accuracy | Breakdown by query category (single-hop, multi-hop, temporal, open-domain) |
| Token Usage | Total LLM tokens consumed during build and generation phases |
| Build Time | Total wall-clock time for graph construction (`build_high_level()`) |
| Retrieval Latency | Average time per query for the retrieval pipeline |

## Environment Setup

```bash
bash scripts/env.sh
```

## Configuration

The benchmark uses two configuration layers:

### Layer 1: Environment Variables (`.env`)

Controls **provider connectivity**: API keys, base URLs, models, timeouts, and retry settings. Loaded from the `.env` file in the project root or the adapter directory. See [`.env.example`](../../.env.example) for all available variables.

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

Controls **experiment parameters**: sample selection, retrieval settings, generation parameters, and ablation configuration. Located under `configs/`.

Key sections in a YAML config:

| Section | Controls |
|---------|----------|
| `embedder` | Embedding dimension |
| `system` | BFS expansion, similarity thresholds, chunk size, session detection |
| `storage` | Persistence root and auto-save settings |
| `experiment` | Sample IDs, skipped categories, output directory |
| `retrieval` | Top-K, which views to skip (ablation) |
| `generation` | Max tokens, temperature |
| `evaluation` | Number of LLM judge runs |

Example: to test only sample `conv-1` and `conv-2`, edit the config:

```yaml
experiment:
  sample_ids: ["conv-1", "conv-2"]
```

> **Note**: The `adapter/config.py` dataclass (`LocomoMemoryConfig`) is for the older adapter path and is separate from the YAML pipeline configs. It also reads environment variables. When running the 4-step pipeline, use the YAML configs — environment variables still override YAML values where applicable.

## Data Preparation

Place `locomo10.json` in `data/` directory (already included in the repository).

## Run Benchmark

### End-to-end Pipeline (recommended)

```bash
python run.py --config configs/base.yaml --output output/
```

With forced rebuild:

```bash
python run.py --config configs/base.yaml --output output/ --force
```

Run specific steps only:

```bash
python run.py --config configs/base.yaml --steps build,retrieve
```

### Step-by-Step Execution

```bash
# Step 1: Build graph
python build_graph.py --config configs/base.yaml --output output/

# Step 2: Retrieve
python retrieve.py --config configs/base.yaml --output output/

# Step 3: Generate
python generate.py --config configs/base.yaml --output output/

# Step 4: Evaluate
python evaluate.py --config configs/base.yaml --output output/
```

### Selecting Specific Samples

Edit the `experiment.sample_ids` field in the config YAML:

```yaml
experiment:
  sample_ids: ["conv-1", "conv-2"]  # Process only these samples
  # sample_ids: []                   # Process all samples
```

## Ablation Experiments

Ablation configs disable specific memory system views to measure each component's contribution. All configs share the same base system parameters — only the retrieval views differ.

| Config | Description | What is removed |
|--------|-------------|----------------|
| `base.yaml` | Full pipeline | — (baseline, all four retrieval groups active) |
| `ablation_no_base.yaml` | Without base memory | Raw dialogue units (base_memory view) excluded |
| `ablation_no_entity.yaml` | Without entity-relation | Entity-relation graph (T₂ tower) excluded |
| `ablation_no_event.yaml` | Without event-causality | Event-causality graph (T₁ tower) excluded |
| `ablation_no_summary.yaml` | Without summaries | Hierarchical summary view (T₀ tower) excluded |
| `ablation_no_graph.yaml` | Without graph expansion | BFS graph expansion disabled; all high-level views skipped |

### Ablation Details

**Full pipeline (`base.yaml`)**: Four retrieval groups (base + entity + event + summary), each performing Dense + BM25 + Sparse three-way recall, RRF fusion, BFS graph expansion, and Cross-Encoder reranking.

**No base memory (`ablation_no_base.yaml`)**: Removes the base memory view from retrieval. Only entity, event, and summary views participate.

**No entity-relation (`ablation_no_entity.yaml`)**: Removes the entity-relation graph view (T₂). Base memory, event-causality, and summary views are retained.

**No event-causality (`ablation_no_event.yaml`)**: Removes the event-causality graph view (T₁). Base memory, entity-relation, and summary views are retained.

**No summaries (`ablation_no_summary.yaml`)**: Removes the hierarchical summary view (T₀). Base memory, entity-relation, and event-causality views are retained.

**No graph expansion (`ablation_no_graph.yaml`)**: Disables BFS expansion (hops=0, per_seed=0) and skips all high-level memory views. Only base memory retrieval with pure vector/keyword/sparse recall.

### Tri-Tower Architectural Elasticity

Mandol's retrieval is powered by three complementary towers (T₀, T₁, T₂). The following ablation measures how reallocating a fixed retrieval budget across towers changes recall and robustness. Removing T₀ weakens standard reasoning but improves adversarial robustness, revealing it as a high-recall but high-noise operator.

| Tower | Name | Description | Ablation Config |
|-------|------|-------------|-----------------|
| T₀ | Hierarchical | High-level summaries (episodic, emotional, procedural, knowledge summaries + insights) | `ablation_no_summary.yaml` |
| T₁ | Episodic | Event-causality graph | `ablation_no_event.yaml` |
| T₂ | Entity-Rel. | Entity-relation graph | `ablation_no_entity.yaml` |

#### GPT-4.1-mini Backbone

| Configuration | Single | Multi | Temp. | Open | Adv. | Overall (w/o Adv.) | Overall (w/ Adv.) |
|---------------|--------|-------|-------|------|------|--------------------|--------------------|
| Mandol (Full Tri-Tower) | 95.01 | **92.55** | 87.23 | **78.13** | 95.29 | 91.88 | 92.65 |
| w/o T₀ (Hierarchical) | 91.44 | 86.17 | 85.67 | 72.92 | **95.96** | 88.12 | 89.88 |
| w/o T₁ (Episodic) | 95.24 | 90.07 | **89.10** | 72.92 | 92.83 | 91.62 | 91.89 |
| w/o T₂ (Entity-Rel.) | **95.96** | 91.13 | 88.79 | 75.00 | 95.29 | **92.27** | **92.95** |

#### GPT-4o-mini Backbone

| Configuration | Single | Multi | Temp. | Open | Adv. | Overall (w/o Adv.) | Overall (w/ Adv.) |
|---------------|--------|-------|-------|------|------|--------------------|--------------------|
| Mandol (Full Tri-Tower) | **93.70** | **86.88** | 86.60 | **73.96** | 68.39 | **89.74** | 84.94 |
| w/o T₀ (Hierarchical) | 89.18 | 83.33 | 82.55 | 63.54 | **94.39** | 85.13 | **87.21** |
| w/o T₁ (Episodic) | 93.34 | 84.40 | **88.79** | 66.67 | 78.03 | 89.09 | 86.61 |
| w/o T₂ (Entity-Rel.) | 93.22 | 86.52 | 86.60 | 65.63 | 78.25 | 88.90 | 86.51 |

> **Note**: Best results per column within each backbone are in **bold**.

## Expected Results

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

> **Note**: † denotes results reproduced using the official EverMemOS implementation, with concurrency patches applied to ensure evaluation stability. The overall metric excludes adversarial queries. Best results per backbone are in **bold**.

## Output Files

After a full pipeline run, the output directory contains:

```
output/<config_name>/
├── build_stats.json          # Build summary (sessions, units, duration, token usage)
├── retrieval_stats.json       # Retrieval summary (queries, duration)
├── generation_stats.json      # Generation summary (queries, duration, token usage)
├── evaluation_summary.json    # Evaluation results (accuracy, per-category breakdown)
├── evaluation_report.txt      # Human-readable evaluation report
└── <sample_id>/
    ├── build.json             # Per-sample build metrics
    ├── retrieval.json         # Retrieved hits per query (with scores and ranks)
    ├── generation.json        # Generated answers (raw and extracted)
    ├── evaluation.json        # LLM judge decisions per query
    └── graph/                 # Persisted MemorySystem state
        └── data/
            ├── units.json     # All memory units
            ├── spaces.json    # Memory space hierarchy
            ├── graph.json     # Relationship edges
            └── sessions.json  # Detected sessions
```

## Directory Structure

```
locomo/
├── README.md              # This file
├── run.py                 # End-to-end pipeline orchestrator
├── build_graph.py         # Step 1: Build graph
├── retrieve.py            # Step 2: Retrieve
├── generate.py            # Step 3: Generate
├── evaluate.py            # Step 4: Evaluate
├── pipeline_utils.py      # Shared utilities and prompt templates
├── adapter/               # LoCoMo adapter for Mandol
│   ├── __init__.py
│   ├── locomo_adapter.py
│   └── config.py
├── data/
│   └── locomo10.json      # LoCoMo dataset
├── scripts/
│   └── env.sh             # Environment setup
├── configs/               # Experiment configurations
│   ├── base.yaml                  # Full pipeline (baseline)
│   ├── ablation_no_base.yaml      # Without base memory view
│   ├── ablation_no_entity.yaml    # Without entity-relation view
│   ├── ablation_no_event.yaml     # Without event-causality view
│   ├── ablation_no_graph.yaml     # Without graph expansion
│   └── ablation_no_summary.yaml   # Without summary view
├── baselines/             # Baseline implementations (planned)
│   ├── README.md
│   ├── mem0/
│   └── letta/
├── results/               # Run logs
└── output/                 # Pipeline output
```
