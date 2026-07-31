#!/usr/bin/env bash
# Purpose: launch the four LongMemEval quantification ablation modes for
# GPT-4.1-mini generation and GPT-4o-mini judging.
# Runs: benchmark_longmemeval.task_eval.benchmark_quantification_ablation.
set -euo pipefail

# LongMemEval quantification ablation, separated commands.
# Generation model: gpt-4.1-mini-closeai
# Evaluation model: gpt-4o-mini-closeai
#
# The four ablation modes are intentionally separate commands so any single
# mode can be copied, executed, or rerun independently.

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
mkdir -p nohup_output

OUTPUT_ROOT="benchmark_longmemeval/task_eval/results/longmemeval_quantification_ablation/gpt41_mini"
LLM_MODEL="gpt-4.1-mini-closeai"
EVAL_MODEL="gpt-4o-mini-closeai"
RERANKER="baai"

SENTENCE_TOP_K=60
EPISODIC_TOP_K=40
ENTITY_TOP_K=40
FINAL_TOP_K=25
CASCADE_MAX_CONTEXT_TOKENS=2000
CASCADE_CLIFF_TOLERANCE=2.5

COMMON_ARGS="
  --llm-model ${LLM_MODEL}
  --llm-evaluate-model ${EVAL_MODEL}
  --rerank-method ${RERANKER}
  --sentence-top-k ${SENTENCE_TOP_K}
  --episodic-top-k ${EPISODIC_TOP_K}
  --entity-top-k ${ENTITY_TOP_K}
  --final-top-k ${FINAL_TOP_K}
  --fusion-method concatenation
  --router-strategy aggressive
  --cascade-mad-multiplier 3.0
  --cascade-cliff-tolerance ${CASCADE_CLIFF_TOLERANCE}
  --cascade-max-context-tokens ${CASCADE_MAX_CONTEXT_TOKENS}
  --cascade-lambda-mmr 0.6
  --output-dir ${OUTPUT_ROOT}
"

# 1. Full Mandol
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_quantification_ablation \
  --ablation-mode full \
  ${COMMON_ARGS} \
  > nohup_output/longmemeval_quant_ablation_gpt41_mini_full.log 2>&1 &

# 2. w/o Context Generation
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_quantification_ablation \
  --ablation-mode no_context_generation \
  ${COMMON_ARGS} \
  > nohup_output/longmemeval_quant_ablation_gpt41_mini_no_context_generation.log 2>&1 &

# 3. w/o Denoising
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_quantification_ablation \
  --ablation-mode no_denoising \
  ${COMMON_ARGS} \
  > nohup_output/longmemeval_quant_ablation_gpt41_mini_no_denoising.log 2>&1 &

# 4. w/o Routing
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_quantification_ablation \
  --ablation-mode no_routing \
  ${COMMON_ARGS} \
  > nohup_output/longmemeval_quant_ablation_gpt41_mini_no_routing.log 2>&1 &
