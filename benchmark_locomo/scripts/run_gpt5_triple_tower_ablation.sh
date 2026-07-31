#!/usr/bin/env bash
# Purpose: launch LoCoMo GPT-5 triple-tower generation ablations evaluated with
# GPT-4o-mini. This script does not enable router or quantification.
# Runs: benchmark_locomo.task_eval.locomo_triple.
set -euo pipefail

# LoCoMo10 GPT-5 generation experiments.
# Generation model: gpt-5-closeai
# Evaluation model: gpt-4o-mini-closeai
# Entry: benchmark_locomo.task_eval.locomo_triple
# No router / no quantification in this script.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
while [[ "$REPO_ROOT" != "/" && ! -d "$REPO_ROOT/src/mandol" ]]; do
  REPO_ROOT="$(dirname "$REPO_ROOT")"
done
if [[ ! -d "$REPO_ROOT/src/mandol" ]]; then
  echo "Could not locate Mandol repo root from $SCRIPT_DIR" >&2
  exit 1
fi
cd "$REPO_ROOT"

mkdir -p \
  benchmark_locomo/task_eval/results/locomo_tri_tower_ablation_gpt5/default_retrieval_eval_gpt4o \
  benchmark_locomo/task_eval/results/locomo_tri_tower_ablation_gpt5/top50_h15_ge35_eval_gpt4o \
  benchmark_locomo/task_eval/results/locomo_tri_tower_ablation_gpt5/top50_h25_ge25_eval_gpt4o \
  nohup_output

# 1. Default retrieval config.
# Effective final context budget follows current LoCoMo default:
#   H=15 direct + reranked(G/E)=20 => about 35 final memory units.
nohup uv run python -m benchmark_locomo.task_eval.locomo_triple \
  --qa-dataset benchmark_locomo/dataset/locomo/locomo10.json \
  --llm-model gpt-5-closeai \
  --llm-evaluate-model gpt-4o-mini-closeai \
  --generation-max-tokens 8192 \
  --enable-second-stage-rerank \
  --rerank-strategy tower_separate \
  --reranker-type baai \
  --fusion-strategy context_aware \
  --topk-hierarchical 15 \
  --topk-similarity 30 \
  --topk-graph 0 \
  --topk-episodic 30 \
  --final-top-k 20 \
  --output-dir benchmark_locomo/task_eval/results/locomo_tri_tower_ablation_gpt5/default_retrieval_eval_gpt4o \
  --log-level INFO \
  > nohup_output/locomo10_gpt5_default_retrieval_eval_gpt4o.log 2>&1 &

# 2. Heterogeneous total top-k ~= 50.
# Keep H direct path at 15 to avoid unreranked noise; expand G/E candidate pools
# and let the reranker choose 35 from Graph/Episodic:
#   H=15 direct + reranked(G/E)=35 => about 50 final memory units.
nohup uv run python -m benchmark_locomo.task_eval.locomo_triple \
  --qa-dataset benchmark_locomo/dataset/locomo/locomo10.json \
  --llm-model gpt-5-closeai \
  --llm-evaluate-model gpt-4o-mini-closeai \
  --generation-max-tokens 8192 \
  --enable-second-stage-rerank \
  --rerank-strategy tower_separate \
  --reranker-type baai \
  --fusion-strategy context_aware \
  --topk-hierarchical 15 \
  --topk-similarity 45 \
  --topk-graph 0 \
  --topk-episodic 45 \
  --final-top-k 35 \
  --output-dir benchmark_locomo/task_eval/results/locomo_tri_tower_ablation_gpt5/top50_h15_ge35_eval_gpt4o \
  --log-level INFO \
  > nohup_output/locomo10_gpt5_top50_heterogeneous_eval_gpt4o.log 2>&1 &

# 3. Heterogeneous total top-k ~= 50, H-heavy split.
# Increase the H direct path to 25 and let Graph/Episodic contribute 25 after rerank:
#   H=25 direct + reranked(G/E)=25 => about 50 final memory units.
nohup uv run python -m benchmark_locomo.task_eval.locomo_triple \
  --qa-dataset benchmark_locomo/dataset/locomo/locomo10.json \
  --llm-model gpt-5-closeai \
  --llm-evaluate-model gpt-4o-mini-closeai \
  --generation-max-tokens 8192 \
  --enable-second-stage-rerank \
  --rerank-strategy tower_separate \
  --reranker-type baai \
  --fusion-strategy context_aware \
  --topk-hierarchical 25 \
  --topk-similarity 35 \
  --topk-graph 0 \
  --topk-episodic 35 \
  --final-top-k 25 \
  --output-dir benchmark_locomo/task_eval/results/locomo_tri_tower_ablation_gpt5/top50_h25_ge25_eval_gpt4o \
  --log-level INFO \
  > nohup_output/locomo10_gpt5_top50_h25_ge25_eval_gpt4o.log 2>&1 &
