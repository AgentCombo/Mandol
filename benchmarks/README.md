# Benchmarks

This directory contains benchmark reproduction scripts for evaluating Mandol on standard long-term memory datasets.

## Available Benchmarks

| Benchmark | Description | Dataset | Source |
|-----------|-------------|---------|--------|
| [LoCoMo](locomo/) | Long-Conversation Memory benchmark with single-hop, multi-hop, temporal, and open-domain questions | `locomo10.json` (included) | [Github](https://github.com/snap-research/locomo) |
| [LongMemEval](longmemeval/) | Long-term memory evaluation across single-session user, assistant, preference, temporal, multi-session, and knowledge update questions | `longmemeval_s_cleaned.json` | [HuggingFace](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) |

## Pipeline

Both benchmarks follow the same 4-step pipeline:

```
build_graph → retrieve → generate → evaluate
```

| Step | Description |
|------|-------------|
| `build_graph.py` | Load dataset, construct multi-dimensional semantic graph, build high-level memories |
| `retrieve.py` | Execute holistic retrieval queries against the built graph |
| `generate.py` | Generate answers via LLM based on retrieved context |
| `evaluate.py` | Score generated answers against ground truth with an LLM judge |

Each step communicates through JSON files on disk, enabling independent execution, incremental resume, and easy debugging. The pipeline can be run end-to-end via `run.py` or step-by-step.

## Quick Start

### Smoke Test (fast validation)

```bash
# LoCoMo
cd locomo && python run.py --smoke --config configs/base.yaml

# LongMemEval
cd longmemeval && python run.py --smoke --config configs/base.yaml
```

The smoke test runs a self-contained pipeline on a small data subset (a few sessions, one query) without spawning subprocesses.

### Full Benchmark

```bash
cd <benchmark>/
python run.py --config configs/base.yaml --output output/
```

With forced rebuild of all stages:

```bash
python run.py --config configs/base.yaml --output output/ --force
```

Run specific stages only:

```bash
python run.py --config configs/base.yaml --stages build,retrieve
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

## Configuration

Each benchmark uses two layers of configuration:

### Layer 1: Environment Variables (`.env`)

Controls **provider connectivity**: API keys, base URLs, models, timeouts, and retry settings. Loaded from the `.env` file in the project root. See [`.env.example`](../../.env.example) for all available variables.

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
  sample_ids: []       # LoCoMo: filter by sample_id
  question_ids: []      # LongMemEval: filter by question_id
  skip_categories: []   # LoCoMo: categories to exclude
  output_dir: "output"
  config_name: null
  dataset_path: "data/dataset.json"

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
| `experiment` | Sample/question IDs, output directory, dataset path |
| `retrieval` | Top-K, which views to skip |
| `generation` | Max tokens, temperature |
| `evaluation` | Number of LLM judge runs |

To run only specific samples, edit the `experiment.sample_ids` (LoCoMo) or `experiment.question_ids` (LongMemEval) field in the config.

## Test Environment

| Component | Specification |
|-----------|--------------|
| CPU | Intel Xeon Platinum 8458P |
| RAM | 120 GB |
| GPU | NVIDIA H800 80GB |
| Python | 3.10.12 |
| OS | Ubuntu 22.04 LTS |

> **Note**: Results may vary on different hardware configurations. The above environment is used for the reference results reported in the paper.

## Output Files

After a full pipeline run, the output directory contains:

```
output/<config_name>/
├── build_stats.json          # Build summary (samples, sessions, units, duration, tokens)
├── evaluation_summary.json   # Evaluation results (overall accuracy, per-type breakdown)
├── evaluation_report.txt     # Human-readable evaluation report
└── <sample_id>/
    ├── build.json            # Per-sample build metrics
    ├── retrieval.json        # Retrieved hits with scores and ranks
    ├── generation.json       # Generated answers (raw and extracted)
    ├── evaluation.json       # LLM judge decision
    └── graph/                # Persisted MemorySystem state
        └── data/
            ├── units.json
            ├── spaces.json
            ├── graph.json
            └── sessions.json
```

## Reproducing Paper Results

For detailed reproduction steps, expected results, and dataset-specific configurations, see the README in each benchmark subdirectory:

- [LoCoMo](locomo/README.md)
- [LongMemEval](longmemeval/README.md)
