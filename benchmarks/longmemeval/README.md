# LongMemEval Benchmark

Reproduction guide for Mandol's evaluation on the LongMemEval dataset.

## Pipeline Overview

The benchmark reproduction follows a 4-step pipeline:

```
build_graph → retrieve → generate → evaluate
```

| Step | Script | Description |
|------|--------|-------------|
| 1 | `build_graph.py` | Load LongMemEval dataset and construct multi-dimensional semantic graph |
| 2 | `retrieve.py` | Execute retrieval queries against the built graph |
| 3 | `generate.py` | Generate answers using LLM based on retrieved context |
| 4 | `evaluate.py` | Score generated answers against ground truth |

## Data Preparation

Download the dataset from HuggingFace:

```bash
# Option 1: Using huggingface-cli
huggingface-cli download xiaowu0162/longmemeval-cleaned --repo-type dataset --local-dir ./data

# Option 2: Using git lfs
git lfs install
git clone https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned ./data
```

Dataset URL: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned

## Environment Setup

```bash
bash scripts/env.sh
```

## Run Benchmark

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

### View Paper Results

To view the paper results directly:

```bash
# GPT-4o-mini backbone results
python evaluate.py --use-paper-results --backbone gpt-4o-mini --output output/

# GPT-4.1-mini backbone results
python evaluate.py --use-paper-results --backbone gpt-4.1-mini --output output/
```

## Results (Paper Table 2)

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

**Note:** SS denotes Single-Session. Best overall results are in **bold**, and the second-best are underlined.

## Tri-Tower Architectural Elasticity

Mandol's retrieval is powered by three complementary towers (T₀, T₁, T₂). The following ablation shows how different tower allocations alter the retrieval plan under a fixed context budget.

| Tower | Name | Description |
|-------|------|-------------|
| T₀ | Hierarchical | High-level summaries (episodic, emotional, procedural, knowledge summaries + insights) |
| T₁ | Episodic | Event-causality graph |
| T₂ | Entity-Rel. | Entity-relation graph |

### GPT-4.1-mini Backbone

| Configuration | Know.\,Upd. | Multi-S | SS-Asst | SS-Pref | SS-User | Temporal | Overall |
|---------------|-------------|---------|---------|---------|---------|----------|---------|
| Mandol (Full Tri-Tower) | **92.31** | 76.69 | **100.00** | **96.67** | 97.14 | **86.47** | 88.40 |
| w/o T₀ (Hierarchical) | 88.46 | 76.69 | 91.07 | **96.67** | **98.57** | 84.21 | 86.40 |
| w/o T₁ (Episodic) | **92.31** | 72.93 | **100.00** | 93.33 | **98.57** | 83.46 | 86.60 |
| w/o T₂ (Entity-Rel.) | 89.74 | **82.71** | **100.00** | **96.67** | 97.14 | **86.47** | **89.60** |

### GPT-4o-mini Backbone

| Configuration | Know.\,Upd. | Multi-S | SS-Asst | SS-Pref | SS-User | Temporal | Overall |
|---------------|-------------|---------|---------|---------|---------|----------|---------|
| Mandol (Full Tri-Tower) | **89.74** | 71.43 | **96.43** | **96.67** | 95.71 | **80.45** | **84.40** |
| w/o T₀ (Hierarchical) | 82.05 | **74.44** | 87.50 | 90.00 | **97.14** | 79.70 | 82.60 |
| w/o T₁ (Episodic) | 85.90 | 69.92 | 94.64 | **96.67** | **97.14** | 79.70 | 83.20 |
| w/o T₂ (Entity-Rel.) | 84.62 | **74.44** | 94.64 | 93.33 | 95.71 | 78.20 | 83.40 |

> **Note**: Best results per column within each backbone are in **bold**. SS denotes Single-Session.

## Directory Structure

```
longmemeval/
├── README.md              # This file
├── build_graph.py         # Step 1: Build graph
├── retrieve.py            # Step 2: Retrieve
├── generate.py            # Step 3: Generate
├── evaluate.py            # Step 4: Evaluate
├── run.py                 # Legacy entry point
├── configs/               # Experiment configurations
│   └── base.yaml
├── data/                  # Dataset directory (download required)
│   └── ...
└── output/                # Results output
```
