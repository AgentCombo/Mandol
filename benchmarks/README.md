# Benchmarks

This directory contains benchmark reproduction scripts for evaluating Mandol on standard long-term memory datasets.

## Available Benchmarks

| Benchmark | Description | Dataset |
|-----------|-------------|---------|
| [LoCoMo](locomo/) | Long-Conversation Memory benchmark with multi-hop, temporal, open-domain, and adversarial questions | `locomo10.json` (included) |
| [LongMemEval](longmemeval/) | Long-term memory evaluation benchmark | [HuggingFace](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) |

## Quick Start

1. Set up environment:
   ```bash
   # For LoCoMo
   cd locomo && bash scripts/env.sh

   # For LongMemEval
   cd longmemeval && bash scripts/env.sh
   ```

2. Prepare data (see individual benchmark READMEs for details)

3. Run benchmark:
   ```bash
   python run.py --config configs/base.yaml --output results/
   ```

## Reproducing Paper Results

For detailed reproduction steps, ablation experiments, and expected results, see the README in each benchmark subdirectory.
