#!/usr/bin/env bash
# Purpose: aggregate LongMemEval reproduction launcher with expanded commands
# for router+quantification+cascade, router-only, ablation, and GPT-5 sections.
# Runs: background jobs matching the paper reproduction command groups.

# Expanded LongMemEval reproduction script.
# Commands are intentionally listed one by one instead of calling child scripts.
# All nohup commands below run in the background, preserving the original
# benchmark-launch semantics.

AMS_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AMS_REPO_ROOT="$AMS_SCRIPT_DIR"
while [[ "$AMS_REPO_ROOT" != "/" && ! -d "$AMS_REPO_ROOT/src/mandol" ]]; do
    AMS_REPO_ROOT="$(dirname "$AMS_REPO_ROOT")"
done
if [[ ! -d "$AMS_REPO_ROOT/src/mandol" ]]; then
    echo "Could not locate AgentMemorySystem repo root from $AMS_SCRIPT_DIR" >&2
    exit 1
fi
cd "$AMS_REPO_ROOT"
if [[ -d "$AMS_REPO_ROOT/.venv/bin" ]]; then
    export PATH="$AMS_REPO_ROOT/.venv/bin:$PATH"
fi
export PYTHONPATH="$AMS_REPO_ROOT/src:$AMS_REPO_ROOT:${PYTHONPATH:-}"
if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required to run this reproduction script" >&2
    exit 1
fi

mkdir -p nohup_output \
    benchmark_longmemeval/task_eval/results/triple_fusion_router_cascade_mad_v3 \
    benchmark_longmemeval/task_eval/results/tower_router/gpt4.1_mini \
    benchmark_longmemeval/task_eval/results/tower_router/gpt4o_mini \
    benchmark_longmemeval/task_eval/results/ablations/gpt4o_mini/full_tri_tower \
    benchmark_longmemeval/task_eval/results/ablations/gpt4o_mini/ablation_wo_sentence \
    benchmark_longmemeval/task_eval/results/ablations/gpt4o_mini/ablation_wo_episodic \
    benchmark_longmemeval/task_eval/results/ablations/gpt4o_mini/ablation_wo_entity \
    benchmark_longmemeval/task_eval/results/ablations/gpt4.1_mini/full_tri_tower \
    benchmark_longmemeval/task_eval/results/ablations/gpt4.1_mini/ablation_wo_sentence \
    benchmark_longmemeval/task_eval/results/ablations/gpt4.1_mini/ablation_wo_episodic \
    benchmark_longmemeval/task_eval/results/ablations/gpt4.1_mini/ablation_wo_entity

echo "=== LongMemEval expanded reproduction commands ==="

# ============================================================================
# LongMemEval router + cascade main experiments (CloseAI, background parallel)
# Source: run_router_quantification_cascade_closeai.sh
# ============================================================================

# LongMemEval router + cascade main experiment: GPT-4o-mini STRICT_THRESHOLD.
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple_router_quantification \
    --llm-model gpt-4o-mini-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
    --sentence-top-k 60 \
    --episodic-top-k 40 \
    --entity-top-k 40 \
    --final-top-k 25 \
    --rerank-method baai \
    --fusion-method concatenation \
    --enable-router \
    --router-strategy aggressive \
    --enable-cascade-pruner \
    --cascade-mad-multiplier 3.0 \
    --cascade-lambda-mmr 0.6 \
    --cascade-prune-mode STRICT_THRESHOLD \
    --cascade-absolute-min-score -9.0 \
    --cascade-max-context-tokens 2200 \
    --output-dir benchmark_longmemeval/task_eval/results/triple_fusion_router_cascade_mad_v3/gpt4o_mini_strict \
    > nohup_output/longmem_mad3_gpt4o_strict.log 2>&1 &

# LongMemEval router + cascade main experiment: GPT-4o-mini CLIFF_EARLY_STOP.
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple_router_quantification \
    --llm-model gpt-4o-mini-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
    --sentence-top-k 60 \
    --episodic-top-k 40 \
    --entity-top-k 40 \
    --final-top-k 25 \
    --rerank-method baai \
    --fusion-method concatenation \
    --enable-router \
    --router-strategy aggressive \
    --enable-cascade-pruner \
    --cascade-mad-multiplier 3.0 \
    --cascade-lambda-mmr 0.6 \
    --cascade-prune-mode CLIFF_EARLY_STOP \
    --cascade-cliff-tolerance 3.0 \
    --cascade-max-context-tokens 2200 \
    --output-dir benchmark_longmemeval/task_eval/results/triple_fusion_router_cascade_mad_v3/gpt4o_mini_cliff \
    > nohup_output/longmem_mad3_gpt4o_cliff.log 2>&1 &

# LongMemEval router + cascade main experiment: GPT-4o-mini BUDGET_MAX.
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple_router_quantification \
    --llm-model gpt-4o-mini-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
    --sentence-top-k 60 \
    --episodic-top-k 40 \
    --entity-top-k 40 \
    --final-top-k 25 \
    --rerank-method baai \
    --fusion-method concatenation \
    --enable-router \
    --router-strategy aggressive \
    --enable-cascade-pruner \
    --cascade-mad-multiplier 3.0 \
    --cascade-lambda-mmr 0.6 \
    --cascade-prune-mode BUDGET_MAX \
    --cascade-max-context-tokens 2000 \
    --output-dir benchmark_longmemeval/task_eval/results/triple_fusion_router_cascade_mad_v3/gpt4o_mini_budget \
    > nohup_output/longmem_mad3_gpt4o_budget.log 2>&1 &

# LongMemEval router + cascade main experiment: GPT-4.1-mini STRICT_THRESHOLD.
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple_router_quantification \
    --llm-model gpt-4.1-mini-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
    --sentence-top-k 60 \
    --episodic-top-k 40 \
    --entity-top-k 40 \
    --final-top-k 25 \
    --rerank-method baai \
    --fusion-method concatenation \
    --enable-router \
    --router-strategy aggressive \
    --enable-cascade-pruner \
    --cascade-mad-multiplier 3.0 \
    --cascade-lambda-mmr 0.6 \
    --cascade-prune-mode STRICT_THRESHOLD \
    --cascade-absolute-min-score -8.0 \
    --cascade-max-context-tokens 2000 \
    --output-dir benchmark_longmemeval/task_eval/results/triple_fusion_router_cascade_mad_v3/gpt41_mini_strict \
    > nohup_output/longmem_mad3_gpt41_strict.log 2>&1 &

# LongMemEval router + cascade main experiment: GPT-4.1-mini CLIFF_EARLY_STOP.
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple_router_quantification \
    --llm-model gpt-4.1-mini-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
    --sentence-top-k 60 \
    --episodic-top-k 40 \
    --entity-top-k 40 \
    --final-top-k 25 \
    --rerank-method baai \
    --fusion-method concatenation \
    --enable-router \
    --router-strategy aggressive \
    --enable-cascade-pruner \
    --cascade-mad-multiplier 3.0 \
    --cascade-lambda-mmr 0.6 \
    --cascade-prune-mode CLIFF_EARLY_STOP \
    --cascade-cliff-tolerance 2.5 \
    --cascade-max-context-tokens 2000 \
    --output-dir benchmark_longmemeval/task_eval/results/triple_fusion_router_cascade_mad_v3/gpt41_mini_cliff \
    > nohup_output/longmem_mad3_gpt41_cliff.log 2>&1 &

# LongMemEval router + cascade main experiment: GPT-4.1-mini BUDGET_MAX.
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple_router_quantification \
    --llm-model gpt-4.1-mini-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
    --sentence-top-k 60 \
    --episodic-top-k 40 \
    --entity-top-k 40 \
    --final-top-k 25 \
    --rerank-method baai \
    --fusion-method concatenation \
    --enable-router \
    --router-strategy aggressive \
    --enable-cascade-pruner \
    --cascade-mad-multiplier 3.0 \
    --cascade-lambda-mmr 0.6 \
    --cascade-prune-mode BUDGET_MAX \
    --cascade-max-context-tokens 1800 \
    --output-dir benchmark_longmemeval/task_eval/results/triple_fusion_router_cascade_mad_v3/gpt41_mini_budget \
    > nohup_output/longmem_mad3_gpt41_budget.log 2>&1 &

# LongMemEval router + cascade main experiment: GPT-4.1-mini DYNAMIC_ADAPTIVE.
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple_router_quantification \
    --llm-model gpt-4.1-mini-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
    --sentence-top-k 60 \
    --episodic-top-k 40 \
    --entity-top-k 40 \
    --final-top-k 25 \
    --rerank-method baai \
    --fusion-method concatenation \
    --enable-router \
    --router-strategy aggressive \
    --enable-cascade-pruner \
    --cascade-mad-multiplier 3.0 \
    --cascade-lambda-mmr 0.6 \
    --cascade-prune-mode DYNAMIC_ADAPTIVE \
    --cascade-adaptive-dataset longmemeval \
    --cascade-cliff-tolerance 2.5 \
    --cascade-max-context-tokens 2000 \
    --output-dir benchmark_longmemeval/task_eval/results/triple_fusion_router_cascade_mad_v3/gpt41_mini_dynamic_adaptive \
    > nohup_output/longmem_mad3_gpt41_dynamic_adaptive.log 2>&1 &

# LongMemEval router + cascade main experiment: GPT-4o-mini DYNAMIC_ADAPTIVE.
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple_router_quantification \
    --llm-model gpt-4o-mini-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
    --sentence-top-k 60 \
    --episodic-top-k 40 \
    --entity-top-k 40 \
    --final-top-k 25 \
    --rerank-method baai \
    --fusion-method concatenation \
    --enable-router \
    --router-strategy aggressive \
    --enable-cascade-pruner \
    --cascade-mad-multiplier 3.0 \
    --cascade-lambda-mmr 0.6 \
    --cascade-prune-mode DYNAMIC_ADAPTIVE \
    --cascade-adaptive-dataset longmemeval \
    --cascade-cliff-tolerance 3.0 \
    --cascade-max-context-tokens 2200 \
    --output-dir benchmark_longmemeval/task_eval/results/triple_fusion_router_cascade_mad_v3/gpt4o_mini_dynamic_adaptive \
    > nohup_output/longmem_mad3_gpt4o_dynamic_adaptive.log 2>&1 &

# ============================================================================
# LongMemEval router + cascade ablation experiments (CloseAI, background parallel)
# Source: run_router_quantification_cascade_closeai.sh
# ============================================================================

# Ablation: GPT-4.1-mini No Router with DYNAMIC_ADAPTIVE cascade.
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple_router_quantification \
    --llm-model gpt-4.1-mini-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
    --sentence-top-k 60 \
    --episodic-top-k 40 \
    --entity-top-k 40 \
    --final-top-k 25 \
    --rerank-method baai \
    --fusion-method concatenation \
    --enable-cascade-pruner \
    --cascade-mad-multiplier 3.0 \
    --cascade-lambda-mmr 0.6 \
    --cascade-prune-mode DYNAMIC_ADAPTIVE \
    --cascade-adaptive-dataset longmemeval \
    --cascade-cliff-tolerance 2.5 \
    --cascade-max-context-tokens 2000 \
    --output-dir benchmark_longmemeval/task_eval/results/triple_fusion_router_cascade_mad_v3/gpt41_mini_ablation_no_router \
    > nohup_output/longmem_mad3_gpt41_ablation_no_router.log 2>&1 &

# Ablation: GPT-4o-mini No Router with DYNAMIC_ADAPTIVE cascade.
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple_router_quantification \
    --llm-model gpt-4o-mini-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
    --sentence-top-k 60 \
    --episodic-top-k 40 \
    --entity-top-k 40 \
    --final-top-k 25 \
    --rerank-method baai \
    --fusion-method concatenation \
    --enable-cascade-pruner \
    --cascade-mad-multiplier 3.0 \
    --cascade-lambda-mmr 0.6 \
    --cascade-prune-mode DYNAMIC_ADAPTIVE \
    --cascade-adaptive-dataset longmemeval \
    --cascade-cliff-tolerance 3.0 \
    --cascade-max-context-tokens 2200 \
    --output-dir benchmark_longmemeval/task_eval/results/triple_fusion_router_cascade_mad_v3/gpt4o_mini_ablation_no_router \
    > nohup_output/longmem_mad3_gpt4o_ablation_no_router.log 2>&1 &

# Ablation: GPT-4.1-mini No Stage1/2 with router + DYNAMIC_ADAPTIVE cascade.
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple_router_quantification \
    --llm-model gpt-4.1-mini-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
    --sentence-top-k 60 \
    --episodic-top-k 40 \
    --entity-top-k 40 \
    --final-top-k 25 \
    --rerank-method baai \
    --fusion-method concatenation \
    --enable-router \
    --router-strategy aggressive \
    --enable-cascade-pruner \
    --cascade-mad-multiplier 3.0 \
    --cascade-lambda-mmr 0.6 \
    --cascade-prune-mode DYNAMIC_ADAPTIVE \
    --cascade-adaptive-dataset longmemeval \
    --cascade-cliff-tolerance 2.5 \
    --cascade-max-context-tokens 2000 \
    --no-cascade-stage1 \
    --no-cascade-stage2 \
    --output-dir benchmark_longmemeval/task_eval/results/triple_fusion_router_cascade_mad_v3/gpt41_mini_ablation_no_stage12 \
    > nohup_output/longmem_mad3_gpt41_ablation_no_stage12.log 2>&1 &

# Ablation: GPT-4o-mini No Stage1/2 with router + DYNAMIC_ADAPTIVE cascade.
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple_router_quantification \
    --llm-model gpt-4o-mini-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
    --sentence-top-k 60 \
    --episodic-top-k 40 \
    --entity-top-k 40 \
    --final-top-k 25 \
    --rerank-method baai \
    --fusion-method concatenation \
    --enable-router \
    --router-strategy aggressive \
    --enable-cascade-pruner \
    --cascade-mad-multiplier 3.0 \
    --cascade-lambda-mmr 0.6 \
    --cascade-prune-mode DYNAMIC_ADAPTIVE \
    --cascade-adaptive-dataset longmemeval \
    --cascade-cliff-tolerance 3.0 \
    --cascade-max-context-tokens 2200 \
    --no-cascade-stage1 \
    --no-cascade-stage2 \
    --output-dir benchmark_longmemeval/task_eval/results/triple_fusion_router_cascade_mad_v3/gpt4o_mini_ablation_no_stage12 \
    > nohup_output/longmem_mad3_gpt4o_ablation_no_stage12.log 2>&1 &

# Ablation: GPT-4.1-mini No Packing with router + BUDGET_MAX cascade.
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple_router_quantification \
    --llm-model gpt-4.1-mini-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
    --sentence-top-k 60 \
    --episodic-top-k 40 \
    --entity-top-k 40 \
    --final-top-k 25 \
    --rerank-method baai \
    --fusion-method concatenation \
    --enable-router \
    --router-strategy aggressive \
    --enable-cascade-pruner \
    --cascade-mad-multiplier 3.0 \
    --cascade-lambda-mmr 0.6 \
    --cascade-prune-mode BUDGET_MAX \
    --cascade-cliff-tolerance 2.5 \
    --cascade-max-context-tokens 2000 \
    --no-cascade-stage3-mmr \
    --output-dir benchmark_longmemeval/task_eval/results/triple_fusion_router_cascade_mad_v3/gpt41_mini_ablation_no_packing \
    > nohup_output/longmem_mad3_gpt41_ablation_no_packing.log 2>&1 &

# Ablation: GPT-4o-mini No Packing with router + BUDGET_MAX cascade.
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple_router_quantification \
    --llm-model gpt-4o-mini-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
    --sentence-top-k 60 \
    --episodic-top-k 40 \
    --entity-top-k 40 \
    --final-top-k 25 \
    --rerank-method baai \
    --fusion-method concatenation \
    --enable-router \
    --router-strategy aggressive \
    --enable-cascade-pruner \
    --cascade-mad-multiplier 3.0 \
    --cascade-lambda-mmr 0.6 \
    --cascade-prune-mode BUDGET_MAX \
    --cascade-cliff-tolerance 3.0 \
    --cascade-max-context-tokens 2200 \
    --no-cascade-stage3-mmr \
    --output-dir benchmark_longmemeval/task_eval/results/triple_fusion_router_cascade_mad_v3/gpt4o_mini_ablation_no_packing \
    > nohup_output/longmem_mad3_gpt4o_ablation_no_packing.log 2>&1 &

# ============================================================================
# Router-only baseline (CloseAI, background parallel)
# Source: run_router_only_closeai.sh
# ============================================================================

# Router-only baseline: GPT-4.1-mini CloseAI aggressive.
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple_router \
    --dataset-size s \
    --llm-model gpt-4.1-mini-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
    --sentence-top-k 60 \
    --episodic-top-k 40 \
    --entity-top-k 40 \
    --final-top-k 25 \
    --enable-router \
    --router-strategy aggressive \
    --output-dir "benchmark_longmemeval/task_eval/results/tower_router/gpt4.1_mini" \
    > log_lme_routed_4.1_aggressive_closeai.log 2>&1 &

# Router-only baseline: GPT-4.1-mini CloseAI conservative.
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple_router \
    --dataset-size s \
    --llm-model gpt-4.1-mini-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
    --sentence-top-k 60 \
    --episodic-top-k 40 \
    --entity-top-k 40 \
    --final-top-k 25 \
    --enable-router \
    --router-strategy conservative \
    --output-dir "benchmark_longmemeval/task_eval/results/tower_router/gpt4.1_mini" \
    > log_lme_routed_4.1_conservative_closeai.log 2>&1 &

# Router-only baseline: GPT-4o-mini CloseAI aggressive.
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple_router \
    --dataset-size s \
    --llm-model gpt-4o-mini-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
    --sentence-top-k 60 \
    --episodic-top-k 40 \
    --entity-top-k 40 \
    --final-top-k 25 \
    --enable-router \
    --router-strategy aggressive \
    --output-dir "benchmark_longmemeval/task_eval/results/tower_router/gpt4o_mini" \
    > log_lme_routed_4o_aggressive_closeai.log 2>&1 &

# Router-only baseline: GPT-4o-mini CloseAI conservative.
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple_router \
    --dataset-size s \
    --llm-model gpt-4o-mini-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
    --sentence-top-k 60 \
    --episodic-top-k 40 \
    --entity-top-k 40 \
    --final-top-k 25 \
    --enable-router \
    --router-strategy conservative \
    --output-dir "benchmark_longmemeval/task_eval/results/tower_router/gpt4o_mini" \
    > log_lme_routed_4o_conservative_closeai.log 2>&1 &

# ============================================================================
# Triple-fusion baseline and ablation (CloseAI, background parallel)
# Source: run_triple_tower_ablation_closeai.sh
# ============================================================================

# Triple-fusion baseline: GPT-4o-mini CloseAI full tri-tower.
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple \
    --dataset-size s \
    --llm-model gpt-4o-mini-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
    --final-top-k 25 \
    --output-dir "benchmark_longmemeval/task_eval/results/ablations/gpt4o_mini/full_tri_tower" \
    > log_baseline_gpt4o.log 2>&1 &

# Ablation: GPT-4o-mini CloseAI without Sentence tower.
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple \
    --dataset-size s \
    --llm-model gpt-4o-mini-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
    --final-top-k 25 \
    --disable-sentence \
    --output-dir "benchmark_longmemeval/task_eval/results/ablations/gpt4o_mini/ablation_wo_sentence" \
    > log_wo_sentence_gpt4o.log 2>&1 &

# Ablation: GPT-4o-mini CloseAI without Episodic tower.
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple \
    --dataset-size s \
    --llm-model gpt-4o-mini-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
    --final-top-k 25 \
    --disable-episodic \
    --output-dir "benchmark_longmemeval/task_eval/results/ablations/gpt4o_mini/ablation_wo_episodic" \
    > log_wo_episodic_gpt4o.log 2>&1 &

# Ablation: GPT-4o-mini CloseAI without Entity tower.
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple \
    --dataset-size s \
    --llm-model gpt-4o-mini-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
    --final-top-k 25 \
    --disable-entity \
    --output-dir "benchmark_longmemeval/task_eval/results/ablations/gpt4o_mini/ablation_wo_entity" \
    > log_wo_entity_gpt4o.log 2>&1 &

# Triple-fusion baseline: GPT-4.1-mini CloseAI full tri-tower.
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple \
    --dataset-size s \
    --llm-model gpt-4.1-mini-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
    --final-top-k 25 \
    --output-dir "benchmark_longmemeval/task_eval/results/ablations/gpt4.1_mini/full_tri_tower" \
    > log_baseline_gpt4.1.log 2>&1 &

# Ablation: GPT-4.1-mini CloseAI without Sentence tower.
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple \
    --dataset-size s \
    --llm-model gpt-4.1-mini-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
    --final-top-k 25 \
    --disable-sentence \
    --output-dir "benchmark_longmemeval/task_eval/results/ablations/gpt4.1_mini/ablation_wo_sentence" \
    > log_wo_sentence_gpt4.1.log 2>&1 &

# Ablation: GPT-4.1-mini CloseAI without Episodic tower.
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple \
    --dataset-size s \
    --llm-model gpt-4.1-mini-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
    --final-top-k 25 \
    --disable-episodic \
    --output-dir "benchmark_longmemeval/task_eval/results/ablations/gpt4.1_mini/ablation_wo_episodic" \
    > log_wo_episodic_gpt4.1.log 2>&1 &

# Ablation: GPT-4.1-mini CloseAI without Entity tower.
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple \
    --dataset-size s \
    --llm-model gpt-4.1-mini-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
    --final-top-k 25 \
    --disable-entity \
    --output-dir "benchmark_longmemeval/task_eval/results/ablations/gpt4.1_mini/ablation_wo_entity" \
    > log_wo_entity_gpt4.1.log 2>&1 &

# ============================================================================
# OpenRouter contrast: triple-fusion baseline and ablation (background parallel)
# Source: run_triple_tower_ablation_openrouter.sh
# ============================================================================

# OpenRouter contrast: GPT-4o-mini full tri-tower.
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple \
    --dataset-size s \
    --llm-model gpt-4o-mini-openrouter \
    --final-top-k 25 \
    --output-dir "benchmark_longmemeval/task_eval/results/ablations/gpt4o_mini/full_tri_tower" \
    > log_baseline_gpt4o.log 2>&1 &

# OpenRouter contrast ablation: GPT-4o-mini without Sentence tower.
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple \
    --dataset-size s \
    --llm-model gpt-4o-mini-openrouter \
    --final-top-k 25 \
    --disable-sentence \
    --output-dir "benchmark_longmemeval/task_eval/results/ablations/gpt4o_mini/ablation_wo_sentence" \
    > log_wo_sentence_gpt4o.log 2>&1 &

# OpenRouter contrast ablation: GPT-4o-mini without Episodic tower.
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple \
    --dataset-size s \
    --llm-model gpt-4o-mini-openrouter \
    --final-top-k 25 \
    --disable-episodic \
    --output-dir "benchmark_longmemeval/task_eval/results/ablations/gpt4o_mini/ablation_wo_episodic" \
    > log_wo_episodic_gpt4o.log 2>&1 &

# OpenRouter contrast ablation: GPT-4o-mini without Entity tower.
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple \
    --dataset-size s \
    --llm-model gpt-4o-mini-openrouter \
    --final-top-k 25 \
    --disable-entity \
    --output-dir "benchmark_longmemeval/task_eval/results/ablations/gpt4o_mini/ablation_wo_entity" \
    > log_wo_entity_gpt4o.log 2>&1 &

# OpenRouter contrast: GPT-4.1-mini full tri-tower.
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple \
    --dataset-size s \
    --llm-model gpt-4.1-mini-openrouter \
    --final-top-k 25 \
    --output-dir "benchmark_longmemeval/task_eval/results/ablations/gpt4.1_mini/full_tri_tower" \
    > log_baseline_gpt4.1.log 2>&1 &

# OpenRouter contrast ablation: GPT-4.1-mini without Sentence tower.
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple \
    --dataset-size s \
    --llm-model gpt-4.1-mini-openrouter \
    --final-top-k 25 \
    --disable-sentence \
    --output-dir "benchmark_longmemeval/task_eval/results/ablations/gpt4.1_mini/ablation_wo_sentence" \
    > log_wo_sentence_gpt4.1.log 2>&1 &

# OpenRouter contrast ablation: GPT-4.1-mini without Episodic tower.
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple \
    --dataset-size s \
    --llm-model gpt-4.1-mini-openrouter \
    --final-top-k 25 \
    --disable-episodic \
    --output-dir "benchmark_longmemeval/task_eval/results/ablations/gpt4.1_mini/ablation_wo_episodic" \
    > log_wo_episodic_gpt4.1.log 2>&1 &

# OpenRouter contrast ablation: GPT-4.1-mini without Entity tower.
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple \
    --dataset-size s \
    --llm-model gpt-4.1-mini-openrouter \
    --final-top-k 25 \
    --disable-entity \
    --output-dir "benchmark_longmemeval/task_eval/results/ablations/gpt4.1_mini/ablation_wo_entity" \
    > log_wo_entity_gpt4.1.log 2>&1 &

# ============================================================================
# OpenRouter contrast: router-only baseline (background parallel)
# Source: run_router_only_openrouter.sh
# ============================================================================

# OpenRouter router-only contrast: GPT-4.1-mini aggressive.
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple_router \
    --dataset-size s \
    --llm-model gpt-4.1-mini-openrouter \
    --llm-evaluate-model gpt-4o-mini-openrouter \
    --sentence-top-k 60 \
    --episodic-top-k 40 \
    --entity-top-k 40 \
    --final-top-k 25 \
    --enable-router \
    --router-strategy aggressive \
    --output-dir "benchmark_longmemeval/task_eval/results/tower_router/gpt4.1_mini" \
    > log_lme_routed_4.1_aggressive_openrouter.log 2>&1 &

# OpenRouter router-only contrast: GPT-4.1-mini conservative.
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple_router \
    --dataset-size s \
    --llm-model gpt-4.1-mini-openrouter \
    --llm-evaluate-model gpt-4o-mini-openrouter \
    --sentence-top-k 60 \
    --episodic-top-k 40 \
    --entity-top-k 40 \
    --final-top-k 25 \
    --enable-router \
    --router-strategy conservative \
    --output-dir "benchmark_longmemeval/task_eval/results/tower_router/gpt4.1_mini" \
    > log_lme_routed_4.1_conservative_openrouter.log 2>&1 &

# OpenRouter router-only contrast: GPT-4o-mini aggressive.
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple_router \
    --dataset-size s \
    --llm-model gpt-4o-mini-openrouter \
    --llm-evaluate-model gpt-4o-mini-openrouter \
    --sentence-top-k 60 \
    --episodic-top-k 40 \
    --entity-top-k 40 \
    --final-top-k 25 \
    --enable-router \
    --router-strategy aggressive \
    --output-dir "benchmark_longmemeval/task_eval/results/tower_router/gpt4o_mini" \
    > log_lme_routed_4o_aggressive_openrouter.log 2>&1 &

# OpenRouter router-only contrast: GPT-4o-mini conservative.
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple_router \
    --dataset-size s \
    --llm-model gpt-4o-mini-openrouter \
    --llm-evaluate-model gpt-4o-mini-openrouter \
    --sentence-top-k 60 \
    --episodic-top-k 40 \
    --entity-top-k 40 \
    --final-top-k 25 \
    --enable-router \
    --router-strategy conservative \
    --output-dir "benchmark_longmemeval/task_eval/results/tower_router/gpt4o_mini" \
    > log_lme_routed_4o_conservative_openrouter.log 2>&1 &

# ============================================================================
# Speed test note
# Source: run_speed_benchmarks.sh
# ============================================================================

echo "[speed] LongMemEval speed helpers are private and not launched by the public reproduction suite."
echo "[speed] Use benchmark_locomo/scripts/run_speed_benchmarks.sh for the public LoCoMo latency/QPS path."

echo "All LongMemEval reproduction commands have been submitted."
