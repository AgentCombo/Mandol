#!/usr/bin/env bash
# Purpose: launch the four LoCoMo quantification ablation modes for
# GPT-4.1-mini generation and GPT-4o-mini judging.
# Runs: benchmark_locomo.task_eval.locomo_quantification_ablation.
set -euo pipefail

# LoCoMo quantification ablation, separated commands.
# Generation model: gpt-4.1-mini-closeai
# Evaluation model: gpt-4o-mini-closeai
# Default behavior: save individual reports.
# To disable individual reports, manually append:
#   --no-save-individual-reports
# To run a small subset manually, append for local debugging:
#   --sample-ids <sample_id> ...

OUTPUT_ROOT="benchmark_locomo/task_eval/results/locomo_quantification_ablation/gpt41_mini"
LLM_MODEL="gpt-4.1-mini-closeai"
EVAL_MODEL="gpt-4o-mini-closeai"
RERANKER="baai"
CASCADE_MAX_CONTEXT_TOKENS=1500
CASCADE_CLIFF_TOLERANCE=2.5

mkdir -p nohup_output

COMMON_ARGS="
  --llm-model ${LLM_MODEL}
  --llm-evaluate-model ${EVAL_MODEL}
  --reranker-type ${RERANKER}
  --topk-hierarchical 15
  --topk-similarity 30
  --topk-graph 0
  --topk-episodic 30
  --final-top-k 20
  --weight-hierarchical 0.34
  --weight-graph 0.33
  --weight-episodic 0.33
  --rerank-strategy tower_separate
  --fusion-strategy context_aware
  --router-strategy aggressive
  --cascade-mad-multiplier 3.0
  --cascade-lambda-mmr 0.7
  --cascade-cliff-tolerance ${CASCADE_CLIFF_TOLERANCE}
  --cascade-max-context-tokens ${CASCADE_MAX_CONTEXT_TOKENS}
  --output-dir ${OUTPUT_ROOT}
"

# 1. Full Mandol
nohup uv run python -m benchmark_locomo.task_eval.locomo_quantification_ablation \
  --ablation-mode full \
  ${COMMON_ARGS} \
  > nohup_output/locomo_quant_ablation_gpt41_mini_full.log 2>&1 &

# 2. w/o Context Generation
nohup uv run python -m benchmark_locomo.task_eval.locomo_quantification_ablation \
  --ablation-mode no_context_generation \
  ${COMMON_ARGS} \
  > nohup_output/locomo_quant_ablation_gpt41_mini_no_context_generation.log 2>&1 &

# 3. w/o Denoising
nohup uv run python -m benchmark_locomo.task_eval.locomo_quantification_ablation \
  --ablation-mode no_denoising \
  ${COMMON_ARGS} \
  > nohup_output/locomo_quant_ablation_gpt41_mini_no_denoising.log 2>&1 &

# 4. w/o Routing
nohup uv run python -m benchmark_locomo.task_eval.locomo_quantification_ablation \
  --ablation-mode no_routing \
  ${COMMON_ARGS} \
  > nohup_output/locomo_quant_ablation_gpt41_mini_no_routing.log 2>&1 &
