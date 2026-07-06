#!/usr/bin/env bash
# Purpose: launch the full LoCoMo quantification ablation matrix with every
# command expanded so individual runs can be copied and rerun directly.
# Runs: benchmark_locomo.task_eval.locomo_quantification_ablation.
set -euo pipefail

# LoCoMo quantification ablation expanded commands.
# Each command contains all parameters so it can be copied and rerun alone.
# Default behavior: save individual reports.
# To disable individual reports for a copied command, append:
#   --no-save-individual-reports

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
while [[ "$REPO_ROOT" != "/" && ! -d "$REPO_ROOT/src/mandol" ]]; do
  REPO_ROOT="$(dirname "$REPO_ROOT")"
done
if [[ ! -d "$REPO_ROOT/src/mandol" ]]; then
  echo "Could not locate AgentMemorySystem repo root from $SCRIPT_DIR" >&2
  exit 1
fi
cd "$REPO_ROOT"
mkdir -p nohup_output

# 1. GPT-4o-mini, Full Mandol
nohup uv run python -m benchmark_locomo.task_eval.locomo_quantification_ablation \
  --ablation-mode full \
  --llm-model gpt-4o-mini-closeai \
  --llm-evaluate-model gpt-4o-mini-closeai \
  --reranker-type baai \
  --topk-hierarchical 15 \
  --topk-similarity 30 \
  --topk-graph 0 \
  --topk-episodic 30 \
  --final-top-k 20 \
  --weight-hierarchical 0.34 \
  --weight-graph 0.33 \
  --weight-episodic 0.33 \
  --rerank-strategy tower_separate \
  --fusion-strategy context_aware \
  --router-strategy aggressive \
  --cascade-mad-multiplier 3.0 \
  --cascade-lambda-mmr 0.7 \
  --cascade-cliff-tolerance 3.0 \
  --cascade-max-context-tokens 1800 \
  --output-dir benchmark_locomo/task_eval/results/locomo_quantification_ablation/gpt4o_mini \
  --log-level INFO \
  > nohup_output/locomo_quant_ablation_gpt4o_mini_full.log 2>&1 &

# 2. GPT-4o-mini, w/o Context Generation
nohup uv run python -m benchmark_locomo.task_eval.locomo_quantification_ablation \
  --ablation-mode no_context_generation \
  --llm-model gpt-4o-mini-closeai \
  --llm-evaluate-model gpt-4o-mini-closeai \
  --reranker-type baai \
  --topk-hierarchical 15 \
  --topk-similarity 30 \
  --topk-graph 0 \
  --topk-episodic 30 \
  --final-top-k 20 \
  --weight-hierarchical 0.34 \
  --weight-graph 0.33 \
  --weight-episodic 0.33 \
  --rerank-strategy tower_separate \
  --fusion-strategy context_aware \
  --router-strategy aggressive \
  --cascade-mad-multiplier 3.0 \
  --cascade-lambda-mmr 0.7 \
  --cascade-cliff-tolerance 3.0 \
  --cascade-max-context-tokens 1800 \
  --output-dir benchmark_locomo/task_eval/results/locomo_quantification_ablation/gpt4o_mini \
  --log-level INFO \
  > nohup_output/locomo_quant_ablation_gpt4o_mini_no_context_generation.log 2>&1 &

# 3. GPT-4o-mini, w/o Denoising
nohup uv run python -m benchmark_locomo.task_eval.locomo_quantification_ablation \
  --ablation-mode no_denoising \
  --llm-model gpt-4o-mini-closeai \
  --llm-evaluate-model gpt-4o-mini-closeai \
  --reranker-type baai \
  --topk-hierarchical 15 \
  --topk-similarity 30 \
  --topk-graph 0 \
  --topk-episodic 30 \
  --final-top-k 20 \
  --weight-hierarchical 0.34 \
  --weight-graph 0.33 \
  --weight-episodic 0.33 \
  --rerank-strategy tower_separate \
  --fusion-strategy context_aware \
  --router-strategy aggressive \
  --cascade-mad-multiplier 3.0 \
  --cascade-lambda-mmr 0.7 \
  --cascade-cliff-tolerance 3.0 \
  --cascade-max-context-tokens 1800 \
  --output-dir benchmark_locomo/task_eval/results/locomo_quantification_ablation/gpt4o_mini \
  --log-level INFO \
  > nohup_output/locomo_quant_ablation_gpt4o_mini_no_denoising.log 2>&1 &

# 4. GPT-4o-mini, w/o Routing
nohup uv run python -m benchmark_locomo.task_eval.locomo_quantification_ablation \
  --ablation-mode no_routing \
  --llm-model gpt-4o-mini-closeai \
  --llm-evaluate-model gpt-4o-mini-closeai \
  --reranker-type baai \
  --topk-hierarchical 15 \
  --topk-similarity 30 \
  --topk-graph 0 \
  --topk-episodic 30 \
  --final-top-k 20 \
  --weight-hierarchical 0.34 \
  --weight-graph 0.33 \
  --weight-episodic 0.33 \
  --rerank-strategy tower_separate \
  --fusion-strategy context_aware \
  --router-strategy aggressive \
  --cascade-mad-multiplier 3.0 \
  --cascade-lambda-mmr 0.7 \
  --cascade-cliff-tolerance 3.0 \
  --cascade-max-context-tokens 1800 \
  --output-dir benchmark_locomo/task_eval/results/locomo_quantification_ablation/gpt4o_mini \
  --log-level INFO \
  > nohup_output/locomo_quant_ablation_gpt4o_mini_no_routing.log 2>&1 &

# 5. GPT-4.1-mini, Full Mandol
nohup uv run python -m benchmark_locomo.task_eval.locomo_quantification_ablation \
  --ablation-mode full \
  --llm-model gpt-4.1-mini-closeai \
  --llm-evaluate-model gpt-4o-mini-closeai \
  --reranker-type baai \
  --topk-hierarchical 15 \
  --topk-similarity 30 \
  --topk-graph 0 \
  --topk-episodic 30 \
  --final-top-k 20 \
  --weight-hierarchical 0.34 \
  --weight-graph 0.33 \
  --weight-episodic 0.33 \
  --rerank-strategy tower_separate \
  --fusion-strategy context_aware \
  --router-strategy aggressive \
  --cascade-mad-multiplier 3.0 \
  --cascade-lambda-mmr 0.7 \
  --cascade-cliff-tolerance 2.5 \
  --cascade-max-context-tokens 1500 \
  --output-dir benchmark_locomo/task_eval/results/locomo_quantification_ablation/gpt41_mini \
  --log-level INFO \
  > nohup_output/locomo_quant_ablation_gpt41_mini_full.log 2>&1 &

# 6. GPT-4.1-mini, w/o Context Generation
nohup uv run python -m benchmark_locomo.task_eval.locomo_quantification_ablation \
  --ablation-mode no_context_generation \
  --llm-model gpt-4.1-mini-closeai \
  --llm-evaluate-model gpt-4o-mini-closeai \
  --reranker-type baai \
  --topk-hierarchical 15 \
  --topk-similarity 30 \
  --topk-graph 0 \
  --topk-episodic 30 \
  --final-top-k 20 \
  --weight-hierarchical 0.34 \
  --weight-graph 0.33 \
  --weight-episodic 0.33 \
  --rerank-strategy tower_separate \
  --fusion-strategy context_aware \
  --router-strategy aggressive \
  --cascade-mad-multiplier 3.0 \
  --cascade-lambda-mmr 0.7 \
  --cascade-cliff-tolerance 2.5 \
  --cascade-max-context-tokens 1500 \
  --output-dir benchmark_locomo/task_eval/results/locomo_quantification_ablation/gpt41_mini \
  --log-level INFO \
  > nohup_output/locomo_quant_ablation_gpt41_mini_no_context_generation.log 2>&1 &

# 7. GPT-4.1-mini, w/o Denoising
nohup uv run python -m benchmark_locomo.task_eval.locomo_quantification_ablation \
  --ablation-mode no_denoising \
  --llm-model gpt-4.1-mini-closeai \
  --llm-evaluate-model gpt-4o-mini-closeai \
  --reranker-type baai \
  --topk-hierarchical 15 \
  --topk-similarity 30 \
  --topk-graph 0 \
  --topk-episodic 30 \
  --final-top-k 20 \
  --weight-hierarchical 0.34 \
  --weight-graph 0.33 \
  --weight-episodic 0.33 \
  --rerank-strategy tower_separate \
  --fusion-strategy context_aware \
  --router-strategy aggressive \
  --cascade-mad-multiplier 3.0 \
  --cascade-lambda-mmr 0.7 \
  --cascade-cliff-tolerance 2.5 \
  --cascade-max-context-tokens 1500 \
  --output-dir benchmark_locomo/task_eval/results/locomo_quantification_ablation/gpt41_mini \
  --log-level INFO \
  > nohup_output/locomo_quant_ablation_gpt41_mini_no_denoising.log 2>&1 &

# 8. GPT-4.1-mini, w/o Routing
nohup uv run python -m benchmark_locomo.task_eval.locomo_quantification_ablation \
  --ablation-mode no_routing \
  --llm-model gpt-4.1-mini-closeai \
  --llm-evaluate-model gpt-4o-mini-closeai \
  --reranker-type baai \
  --topk-hierarchical 15 \
  --topk-similarity 30 \
  --topk-graph 0 \
  --topk-episodic 30 \
  --final-top-k 20 \
  --weight-hierarchical 0.34 \
  --weight-graph 0.33 \
  --weight-episodic 0.33 \
  --rerank-strategy tower_separate \
  --fusion-strategy context_aware \
  --router-strategy aggressive \
  --cascade-mad-multiplier 3.0 \
  --cascade-lambda-mmr 0.7 \
  --cascade-cliff-tolerance 2.5 \
  --cascade-max-context-tokens 1500 \
  --output-dir benchmark_locomo/task_eval/results/locomo_quantification_ablation/gpt41_mini \
  --log-level INFO \
  > nohup_output/locomo_quant_ablation_gpt41_mini_no_routing.log 2>&1 &
