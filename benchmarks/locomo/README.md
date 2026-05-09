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
| F1 Score | Harmonic mean of precision and recall for retrieval accuracy |
| Response Time | End-to-end latency from query submission to result return |
| Memory Usage | Peak RSS (Resident Set Size) during system operation |
| Index Build Time | Total wall-clock time for `build_high_level()` completion |

## Environment Setup

```bash
bash scripts/env.sh
```

## Data Preparation

Place `locomo10.json` in `data/` directory (already included in the repository).

## Run Benchmark

### Full Pipeline

```bash
# Step 1: Build graph
python build_graph.py --config configs/base.yaml --output output/

# Step 2: Retrieve
python retrieve.py --config configs/base.yaml --input output/ --output output/

# Step 3: Generate
python generate.py --config configs/base.yaml --input output/ --output output/

# Step 4: Evaluate
python evaluate.py --input output/ --output output/
```

### Specific Samples

```bash
python build_graph.py --config configs/base.yaml --sample-ids conv-1 conv-2 --output output/
```

## Ablation Experiments

| Config | Description | What is removed |
|--------|-------------|----------------|
| `base.yaml` | Full pipeline | — (baseline) |
| `ablation_no_graph.yaml` | Without graph expansion | BFS expansion from SemanticGraph is disabled |
| `ablation_no_rerank.yaml` | Without cross-encoder reranking | Cross-Encoder Reranker step is skipped |
| `ablation_no_sparse.yaml` | Without sparse retrieval | SPLADE sparse vector retrieval is removed from the three-way recall |

### Ablation Details

**Full pipeline (`base.yaml`)**: Dense + BM25 + Sparse → RRF fusion → BFS graph expansion → Cross-Encoder reranking. This is the complete Mandol retrieval pipeline.

**No graph expansion (`ablation_no_graph.yaml`)**: Removes the BFS expansion step that traverses SemanticGraph edges to discover related memory units beyond the initial recall set. This ablation tests the contribution of graph-structured relationships to retrieval quality.

**No reranking (`ablation_no_rerank.yaml`)**: Removes the Cross-Encoder reranking step that re-scores and re-orders candidates after fusion. This ablation tests the contribution of the expensive but precise reranking model to final result quality.

**No sparse retrieval (`ablation_no_sparse.yaml`)**: Removes the SPLADE sparse vector retrieval from the three-way recall, leaving only Dense + BM25. This ablation tests the contribution of learned sparse representations to the multi-way recall strategy.

### Tri-Tower Architectural Elasticity

Mandol's retrieval is powered by three complementary towers (T₀, T₁, T₂). The following ablation measures how reallocating a fixed retrieval budget across towers changes recall and robustness. Removing T₀ weakens standard reasoning but improves adversarial robustness, revealing it as a high-recall but high-noise operator.

| Tower | Name | Description |
|-------|------|-------------|
| T₀ | Hierarchical | High-level summaries (episodic, emotional, procedural, knowledge summaries + insights) |
| T₁ | Episodic | Event-causality graph |
| T₂ | Entity-Rel. | Entity-relation graph |

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

| System | Avg.\,Tok. | Single-hop | Multi-hop | Temporal | Open-domain | Overall |
|--------|------------|------------|-----------|----------|-------------|---------|
| Mem0 | 1.0k | 66.71 | 58.16 | 55.45 | 40.62 | 61.00 |
| MemU | 4.0k | 72.77 | 62.41 | 33.96 | 46.88 | 61.15 |
| MemOS | 2.5k | 81.45 | 69.15 | 72.27 | 60.42 | 75.87 |
| Zep | 1.4k | 88.11 | 71.99 | 74.45 | 66.67 | 81.06 |
| EverMemOS† | 2.5k | 91.68 | 82.74 | 79.34 | 70.14 | 86.13 |
| **Mandol (Ours)** | **1.9k** | **93.82** | **85.11** | **89.10** | 65.63 | **89.48** |

### GPT-4.1-mini Backbone

| System | Avg.\,Tok. | Single-hop | Multi-hop | Temporal | Open-domain | Overall |
|--------|------------|------------|-----------|----------|-------------|---------|
| Mem0 | 1.0k | 68.97 | 61.70 | 58.26 | 50.00 | 64.20 |
| MemU | 4.0k | 74.91 | 72.34 | 43.61 | 54.17 | 66.67 |
| MemOS | 2.5k | 85.37 | 79.43 | 75.08 | 64.58 | 80.76 |
| Zep | 1.4k | 90.84 | 81.91 | 77.26 | 75.00 | 85.22 |
| EverMemOS† | 2.3k | 95.32 | 89.01 | 90.13 | 77.43 | 91.97 |
| **Mandol (Ours)** | **1.9k** | **95.36** | **92.20** | 87.85 | **79.17** | **92.21** |

> **Note**: † denotes results reproduced using the official EverMemOS implementation, with concurrency patches applied to ensure evaluation stability. The overall metric excludes adversarial queries. Best results per backbone are in **bold**.

## Directory Structure

```
locomo/
├── README.md              # This file
├── build_graph.py         # Step 1: Build graph
├── retrieve.py            # Step 2: Retrieve
├── generate.py            # Step 3: Generate
├── evaluate.py            # Step 4: Evaluate
├── run.py                 # Legacy entry point (deprecated)
├── adapter/               # LoCoMo adapter for Mandol
│   ├── __init__.py
│   ├── locomo_adapter.py
│   └── config.py
├── data/
│   └── locomo10.json      # LoCoMo dataset
├── scripts/
│   └── env.sh             # Environment setup
├── configs/               # Experiment configurations
│   ├── base.yaml
│   ├── ablation_no_graph.yaml
│   ├── ablation_no_rerank.yaml
│   └── ablation_no_sparse.yaml
├── baselines/             # Baseline implementations
│   ├── README.md
│   ├── mem0/
│   └── letta/
└── results/               # Results output
```
