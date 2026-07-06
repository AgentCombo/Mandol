#!/usr/bin/env bash
# Purpose: aggregate LoCoMo reproduction launcher with expanded commands for
# ablation, router-only, router+quantification+cascade, and speed sections.
# Runs: background jobs controlled by RUN_* environment toggles below.

# Expanded total reproduction script for benchmark_locomo.
# It intentionally expands commands instead of calling the smaller scripts, so
# each command can be copied and run from the repository root.

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
    echo "uv is required to run this benchmark script" >&2
    exit 1
fi

RUN_FAIR_ABLATION="${RUN_FAIR_ABLATION:-1}"
RUN_ROUTER_CLOSEAI="${RUN_ROUTER_CLOSEAI:-1}"
RUN_ROUTER_OPENROUTER="${RUN_ROUTER_OPENROUTER:-0}"
RUN_ROUTER_CASCADE_CLOSEAI="${RUN_ROUTER_CASCADE_CLOSEAI:-1}"
RUN_DEBUG_SPEED="${RUN_DEBUG_SPEED:-0}"

mkdir -p nohup_output

echo "Expanded reproduction commands loaded."
echo "Sections: fair_ablation=$RUN_FAIR_ABLATION router_closeai=$RUN_ROUTER_CLOSEAI router_openrouter=$RUN_ROUTER_OPENROUTER router_cascade=$RUN_ROUTER_CASCADE_CLOSEAI debug_speed=$RUN_DEBUG_SPEED"

# ==============================================================================
# 1. Fair ablation: tower_separate LoCoMo10 ablation with baselines
# Source: run_triple_tower_ablation_closeai.sh
# ==============================================================================
if [[ "$RUN_FAIR_ABLATION" == "1" ]]; then
    echo "[fair_ablation] launching 8 background jobs..."

    mkdir -p benchmark_locomo/task_eval/results/locomo10_ablation_separate/gpt_4o_mini/full_tri_tower
    nohup uv run python -m benchmark_locomo.task_eval.locomo_triple \
        --qa-dataset "benchmark_locomo/dataset/locomo/locomo10.json" \
        --llm-model gpt-4o-mini-closeai \
        --llm-evaluate-model gpt-4o-mini-closeai \
        --enable-second-stage-rerank \
        --rerank-strategy tower_separate \
        --output-dir "benchmark_locomo/task_eval/results/locomo10_ablation_separate/gpt_4o_mini/full_tri_tower" \
        --topk-hierarchical 15 \
        --topk-similarity 30 \
        --topk-episodic 30 \
        --final-top-k 20 \
        > benchmark_locomo/task_eval/results/locomo10_ablation_separate/gpt_4o_mini/full_tri_tower/run.log 2>&1 &

    mkdir -p benchmark_locomo/task_eval/results/locomo10_ablation_separate/gpt_4o_mini/wo_episodic
    nohup uv run python -m benchmark_locomo.task_eval.locomo_triple \
        --qa-dataset "benchmark_locomo/dataset/locomo/locomo10.json" \
        --llm-model gpt-4o-mini-closeai \
        --llm-evaluate-model gpt-4o-mini-closeai \
        --enable-second-stage-rerank \
        --rerank-strategy tower_separate \
        --output-dir "benchmark_locomo/task_eval/results/locomo10_ablation_separate/gpt_4o_mini/wo_episodic" \
        --topk-hierarchical 15 \
        --topk-similarity 30 \
        --topk-episodic 0 \
        --final-top-k 20 \
        > benchmark_locomo/task_eval/results/locomo10_ablation_separate/gpt_4o_mini/wo_episodic/run.log 2>&1 &

    mkdir -p benchmark_locomo/task_eval/results/locomo10_ablation_separate/gpt_4o_mini/wo_graph
    nohup uv run python -m benchmark_locomo.task_eval.locomo_triple \
        --qa-dataset "benchmark_locomo/dataset/locomo/locomo10.json" \
        --llm-model gpt-4o-mini-closeai \
        --llm-evaluate-model gpt-4o-mini-closeai \
        --enable-second-stage-rerank \
        --rerank-strategy tower_separate \
        --output-dir "benchmark_locomo/task_eval/results/locomo10_ablation_separate/gpt_4o_mini/wo_graph" \
        --topk-hierarchical 15 \
        --topk-similarity 0 \
        --topk-graph 0 \
        --no-entity-relation \
        --topk-episodic 30 \
        --final-top-k 20 \
        > benchmark_locomo/task_eval/results/locomo10_ablation_separate/gpt_4o_mini/wo_graph/run.log 2>&1 &

    mkdir -p benchmark_locomo/task_eval/results/locomo10_ablation_separate/gpt_4o_mini/wo_hierarchical
    nohup uv run python -m benchmark_locomo.task_eval.locomo_triple \
        --qa-dataset "benchmark_locomo/dataset/locomo/locomo10.json" \
        --llm-model gpt-4o-mini-closeai \
        --llm-evaluate-model gpt-4o-mini-closeai \
        --enable-second-stage-rerank \
        --rerank-strategy tower_separate \
        --output-dir "benchmark_locomo/task_eval/results/locomo10_ablation_separate/gpt_4o_mini/wo_hierarchical" \
        --topk-hierarchical 0 \
        --topk-similarity 30 \
        --topk-episodic 30 \
        --final-top-k 35 \
        > benchmark_locomo/task_eval/results/locomo10_ablation_separate/gpt_4o_mini/wo_hierarchical/run.log 2>&1 &

    mkdir -p benchmark_locomo/task_eval/results/locomo10_ablation_separate/gpt_4.1_mini/full_tri_tower
    nohup uv run python -m benchmark_locomo.task_eval.locomo_triple \
        --qa-dataset "benchmark_locomo/dataset/locomo/locomo10.json" \
        --llm-model gpt-4.1-mini-closeai \
        --llm-evaluate-model gpt-4o-mini-closeai \
        --enable-second-stage-rerank \
        --rerank-strategy tower_separate \
        --output-dir "benchmark_locomo/task_eval/results/locomo10_ablation_separate/gpt_4.1_mini/full_tri_tower" \
        --topk-hierarchical 15 \
        --topk-similarity 30 \
        --topk-episodic 30 \
        --final-top-k 20 \
        > benchmark_locomo/task_eval/results/locomo10_ablation_separate/gpt_4.1_mini/full_tri_tower/run.log 2>&1 &

    mkdir -p benchmark_locomo/task_eval/results/locomo10_ablation_separate/gpt_4.1_mini/wo_episodic
    nohup uv run python -m benchmark_locomo.task_eval.locomo_triple \
        --qa-dataset "benchmark_locomo/dataset/locomo/locomo10.json" \
        --llm-model gpt-4.1-mini-closeai \
        --llm-evaluate-model gpt-4o-mini-closeai \
        --enable-second-stage-rerank \
        --rerank-strategy tower_separate \
        --output-dir "benchmark_locomo/task_eval/results/locomo10_ablation_separate/gpt_4.1_mini/wo_episodic" \
        --topk-hierarchical 15 \
        --topk-similarity 30 \
        --topk-episodic 0 \
        --final-top-k 20 \
        > benchmark_locomo/task_eval/results/locomo10_ablation_separate/gpt_4.1_mini/wo_episodic/run.log 2>&1 &

    mkdir -p benchmark_locomo/task_eval/results/locomo10_ablation_separate/gpt_4.1_mini/wo_graph
    nohup uv run python -m benchmark_locomo.task_eval.locomo_triple \
        --qa-dataset "benchmark_locomo/dataset/locomo/locomo10.json" \
        --llm-model gpt-4.1-mini-closeai \
        --llm-evaluate-model gpt-4o-mini-closeai \
        --enable-second-stage-rerank \
        --rerank-strategy tower_separate \
        --output-dir "benchmark_locomo/task_eval/results/locomo10_ablation_separate/gpt_4.1_mini/wo_graph" \
        --topk-hierarchical 15 \
        --topk-similarity 0 \
        --topk-graph 0 \
        --no-entity-relation \
        --topk-episodic 30 \
        --final-top-k 20 \
        > benchmark_locomo/task_eval/results/locomo10_ablation_separate/gpt_4.1_mini/wo_graph/run.log 2>&1 &

    mkdir -p benchmark_locomo/task_eval/results/locomo10_ablation_separate/gpt_4.1_mini/wo_hierarchical
    nohup uv run python -m benchmark_locomo.task_eval.locomo_triple \
        --qa-dataset "benchmark_locomo/dataset/locomo/locomo10.json" \
        --llm-model gpt-4.1-mini-closeai \
        --llm-evaluate-model gpt-4o-mini-closeai \
        --enable-second-stage-rerank \
        --rerank-strategy tower_separate \
        --output-dir "benchmark_locomo/task_eval/results/locomo10_ablation_separate/gpt_4.1_mini/wo_hierarchical" \
        --topk-hierarchical 0 \
        --topk-similarity 30 \
        --topk-episodic 30 \
        --final-top-k 35 \
        > benchmark_locomo/task_eval/results/locomo10_ablation_separate/gpt_4.1_mini/wo_hierarchical/run.log 2>&1 &
fi

# ==============================================================================
# 2. Router-only benchmark: CloseAI
# Source: run_router_only_closeai.sh
# ==============================================================================
if [[ "$RUN_ROUTER_CLOSEAI" == "1" ]]; then
    echo "[router_closeai] launching 4 background jobs..."
    mkdir -p benchmark_locomo/task_eval/results/tower_router/gpt4.1_mini \
        benchmark_locomo/task_eval/results/tower_router/gpt4o_mini

    nohup uv run python -m benchmark_locomo.task_eval.locomo_triple_router \
        --llm-model gpt-4.1-mini-closeai \
        --llm-evaluate-model gpt-4o-mini-closeai \
        --topk-hierarchical 15 \
        --topk-similarity 30 \
        --topk-graph 0 \
        --topk-episodic 30 \
        --final-top-k 20 \
        --weight-hierarchical 0.34 \
        --weight-graph 0.33 \
        --weight-episodic 0.33 \
        --enable-second-stage-rerank \
        --rerank-strategy tower_separate \
        --enable-router \
        --router-strategy aggressive \
        --output-dir "benchmark_locomo/task_eval/results/tower_router/gpt4.1_mini" \
        > log_locomo_routed_4.1_aggressive_closeai.log 2>&1 &

    nohup uv run python -m benchmark_locomo.task_eval.locomo_triple_router \
        --llm-model gpt-4.1-mini-closeai \
        --llm-evaluate-model gpt-4o-mini-closeai \
        --topk-hierarchical 15 \
        --topk-similarity 30 \
        --topk-graph 0 \
        --topk-episodic 30 \
        --final-top-k 20 \
        --weight-hierarchical 0.34 \
        --weight-graph 0.33 \
        --weight-episodic 0.33 \
        --enable-second-stage-rerank \
        --rerank-strategy tower_separate \
        --enable-router \
        --router-strategy conservative \
        --output-dir "benchmark_locomo/task_eval/results/tower_router/gpt4.1_mini" \
        > log_locomo_routed_4.1_conservative_closeai.log 2>&1 &

    nohup uv run python -m benchmark_locomo.task_eval.locomo_triple_router \
        --llm-model gpt-4o-mini-closeai \
        --llm-evaluate-model gpt-4o-mini-closeai \
        --topk-hierarchical 15 \
        --topk-similarity 30 \
        --topk-graph 0 \
        --topk-episodic 30 \
        --final-top-k 20 \
        --weight-hierarchical 0.34 \
        --weight-graph 0.33 \
        --weight-episodic 0.33 \
        --enable-second-stage-rerank \
        --rerank-strategy tower_separate \
        --enable-router \
        --router-strategy aggressive \
        --output-dir "benchmark_locomo/task_eval/results/tower_router/gpt4o_mini" \
        > log_locomo_routed_4o_aggressive_closeai.log 2>&1 &

    nohup uv run python -m benchmark_locomo.task_eval.locomo_triple_router \
        --llm-model gpt-4o-mini-closeai \
        --llm-evaluate-model gpt-4o-mini-closeai \
        --topk-hierarchical 15 \
        --topk-similarity 30 \
        --topk-graph 0 \
        --topk-episodic 30 \
        --final-top-k 20 \
        --weight-hierarchical 0.34 \
        --weight-graph 0.33 \
        --weight-episodic 0.33 \
        --enable-second-stage-rerank \
        --rerank-strategy tower_separate \
        --enable-router \
        --router-strategy conservative \
        --output-dir "benchmark_locomo/task_eval/results/tower_router/gpt4o_mini" \
        > log_locomo_routed_4o_conservative_closeai.log 2>&1 &
fi

# ==============================================================================
# 3. Router-only benchmark: OpenRouter
# Source: run_router_only_openrouter.sh
# Disabled by default because it preserves the same output directories as CloseAI.
# ==============================================================================
if [[ "$RUN_ROUTER_OPENROUTER" == "1" ]]; then
    echo "[router_openrouter] launching 4 background jobs..."
    mkdir -p benchmark_locomo/task_eval/results/tower_router/gpt4.1_mini \
        benchmark_locomo/task_eval/results/tower_router/gpt4o_mini

    nohup uv run python -m benchmark_locomo.task_eval.locomo_triple_router \
        --llm-model gpt-4.1-mini-openrouter \
        --llm-evaluate-model gpt-4o-mini-openrouter \
        --topk-hierarchical 15 \
        --topk-similarity 30 \
        --topk-graph 0 \
        --topk-episodic 30 \
        --final-top-k 20 \
        --weight-hierarchical 0.34 \
        --weight-graph 0.33 \
        --weight-episodic 0.33 \
        --enable-second-stage-rerank \
        --rerank-strategy tower_separate \
        --enable-router \
        --router-strategy aggressive \
        --output-dir "benchmark_locomo/task_eval/results/tower_router/gpt4.1_mini" \
        > log_locomo_routed_4.1_aggressive_openrouter.log 2>&1 &

    nohup uv run python -m benchmark_locomo.task_eval.locomo_triple_router \
        --llm-model gpt-4.1-mini-openrouter \
        --llm-evaluate-model gpt-4o-mini-openrouter \
        --topk-hierarchical 15 \
        --topk-similarity 30 \
        --topk-graph 0 \
        --topk-episodic 30 \
        --final-top-k 20 \
        --weight-hierarchical 0.34 \
        --weight-graph 0.33 \
        --weight-episodic 0.33 \
        --enable-second-stage-rerank \
        --rerank-strategy tower_separate \
        --enable-router \
        --router-strategy conservative \
        --output-dir "benchmark_locomo/task_eval/results/tower_router/gpt4.1_mini" \
        > log_locomo_routed_4.1_conservative_openrouter.log 2>&1 &

    nohup uv run python -m benchmark_locomo.task_eval.locomo_triple_router \
        --llm-model gpt-4o-mini-openrouter \
        --llm-evaluate-model gpt-4o-mini-openrouter \
        --topk-hierarchical 15 \
        --topk-similarity 30 \
        --topk-graph 0 \
        --topk-episodic 30 \
        --final-top-k 20 \
        --weight-hierarchical 0.34 \
        --weight-graph 0.33 \
        --weight-episodic 0.33 \
        --enable-second-stage-rerank \
        --rerank-strategy tower_separate \
        --enable-router \
        --router-strategy aggressive \
        --output-dir "benchmark_locomo/task_eval/results/tower_router/gpt4o_mini" \
        > log_locomo_routed_4o_aggressive_openrouter.log 2>&1 &

    nohup uv run python -m benchmark_locomo.task_eval.locomo_triple_router \
        --llm-model gpt-4o-mini-openrouter \
        --llm-evaluate-model gpt-4o-mini-openrouter \
        --topk-hierarchical 15 \
        --topk-similarity 30 \
        --topk-graph 0 \
        --topk-episodic 30 \
        --final-top-k 20 \
        --weight-hierarchical 0.34 \
        --weight-graph 0.33 \
        --weight-episodic 0.33 \
        --enable-second-stage-rerank \
        --rerank-strategy tower_separate \
        --enable-router \
        --router-strategy conservative \
        --output-dir "benchmark_locomo/task_eval/results/tower_router/gpt4o_mini" \
        > log_locomo_routed_4o_conservative_openrouter.log 2>&1 &
fi

# ==============================================================================
# 4. Router + cascade / quantification benchmark: CloseAI
# Source: run_router_quantification_cascade_expanded_closeai.sh
# ==============================================================================
if [[ "$RUN_ROUTER_CASCADE_CLOSEAI" == "1" ]]; then
    echo "[router_cascade_closeai] launching 14 background jobs..."
    mkdir -p benchmark_locomo/task_eval/results/locomo_tri_tower_router_cascade_results_v2

    nohup uv run python -m benchmark_locomo.task_eval.locomo_triple_router_quantification \
        --llm-model gpt-4.1-mini-closeai \
        --llm-evaluate-model gpt-4o-mini-closeai \
        --topk-hierarchical 15 \
        --topk-similarity 30 \
        --topk-graph 0 \
        --topk-episodic 30 \
        --final-top-k 20 \
        --weight-hierarchical 0.34 \
        --weight-graph 0.33 \
        --weight-episodic 0.33 \
        --enable-second-stage-rerank \
        --rerank-strategy tower_separate \
        --reranker-type baai \
        --enable-router \
        --router-strategy aggressive \
        --enable-cascade-pruner \
        --cascade-mad-multiplier 3.0 \
        --cascade-lambda-mmr 0.7 \
        --cascade-prune-mode STRICT_THRESHOLD \
        --cascade-absolute-min-score -5.0 \
        --cascade-max-context-tokens 1500 \
        --output-dir benchmark_locomo/task_eval/results/locomo_tri_tower_router_cascade_results_v2/gpt41_mini_strict \
        --log-level INFO \
        > nohup_output/router_cascade_v2_gpt41_strict.log 2>&1 &

    nohup uv run python -m benchmark_locomo.task_eval.locomo_triple_router_quantification \
        --llm-model gpt-4.1-mini-closeai \
        --llm-evaluate-model gpt-4o-mini-closeai \
        --topk-hierarchical 15 \
        --topk-similarity 30 \
        --topk-graph 0 \
        --topk-episodic 30 \
        --final-top-k 20 \
        --weight-hierarchical 0.34 \
        --weight-graph 0.33 \
        --weight-episodic 0.33 \
        --enable-second-stage-rerank \
        --rerank-strategy tower_separate \
        --reranker-type baai \
        --enable-router \
        --router-strategy aggressive \
        --enable-cascade-pruner \
        --cascade-mad-multiplier 3.0 \
        --cascade-lambda-mmr 0.7 \
        --cascade-prune-mode CLIFF_EARLY_STOP \
        --cascade-cliff-tolerance 2.5 \
        --cascade-max-context-tokens 1500 \
        --output-dir benchmark_locomo/task_eval/results/locomo_tri_tower_router_cascade_results_v2/gpt41_mini_cliff \
        --log-level INFO \
        > nohup_output/router_cascade_v2_gpt41_cliff.log 2>&1 &

    nohup uv run python -m benchmark_locomo.task_eval.locomo_triple_router_quantification \
        --llm-model gpt-4.1-mini-closeai \
        --llm-evaluate-model gpt-4o-mini-closeai \
        --topk-hierarchical 15 \
        --topk-similarity 30 \
        --topk-graph 0 \
        --topk-episodic 30 \
        --final-top-k 20 \
        --weight-hierarchical 0.34 \
        --weight-graph 0.33 \
        --weight-episodic 0.33 \
        --enable-second-stage-rerank \
        --rerank-strategy tower_separate \
        --reranker-type baai \
        --enable-router \
        --router-strategy aggressive \
        --enable-cascade-pruner \
        --cascade-mad-multiplier 3.0 \
        --cascade-lambda-mmr 0.7 \
        --cascade-prune-mode BUDGET_MAX \
        --cascade-max-context-tokens 1500 \
        --output-dir benchmark_locomo/task_eval/results/locomo_tri_tower_router_cascade_results_v2/gpt41_mini_budget \
        --log-level INFO \
        > nohup_output/router_cascade_v2_gpt41_budget.log 2>&1 &

    nohup uv run python -m benchmark_locomo.task_eval.locomo_triple_router_quantification \
        --llm-model gpt-4o-mini-closeai \
        --llm-evaluate-model gpt-4o-mini-closeai \
        --topk-hierarchical 15 \
        --topk-similarity 30 \
        --topk-graph 0 \
        --topk-episodic 30 \
        --final-top-k 20 \
        --weight-hierarchical 0.34 \
        --weight-graph 0.33 \
        --weight-episodic 0.33 \
        --enable-second-stage-rerank \
        --rerank-strategy tower_separate \
        --reranker-type baai \
        --enable-router \
        --router-strategy aggressive \
        --enable-cascade-pruner \
        --cascade-mad-multiplier 3.0 \
        --cascade-lambda-mmr 0.7 \
        --cascade-prune-mode STRICT_THRESHOLD \
        --cascade-absolute-min-score -6.0 \
        --cascade-max-context-tokens 2000 \
        --output-dir benchmark_locomo/task_eval/results/locomo_tri_tower_router_cascade_results_v2/gpt4o_mini_strict \
        --log-level INFO \
        > nohup_output/router_cascade_v2_gpt4o_strict.log 2>&1 &

    nohup uv run python -m benchmark_locomo.task_eval.locomo_triple_router_quantification \
        --llm-model gpt-4o-mini-closeai \
        --llm-evaluate-model gpt-4o-mini-closeai \
        --topk-hierarchical 15 \
        --topk-similarity 30 \
        --topk-graph 0 \
        --topk-episodic 30 \
        --final-top-k 20 \
        --weight-hierarchical 0.34 \
        --weight-graph 0.33 \
        --weight-episodic 0.33 \
        --enable-second-stage-rerank \
        --rerank-strategy tower_separate \
        --reranker-type baai \
        --enable-router \
        --router-strategy aggressive \
        --enable-cascade-pruner \
        --cascade-mad-multiplier 3.0 \
        --cascade-lambda-mmr 0.7 \
        --cascade-prune-mode CLIFF_EARLY_STOP \
        --cascade-cliff-tolerance 3.0 \
        --cascade-max-context-tokens 2000 \
        --output-dir benchmark_locomo/task_eval/results/locomo_tri_tower_router_cascade_results_v2/gpt4o_mini_cliff \
        --log-level INFO \
        > nohup_output/router_cascade_v2_gpt4o_cliff.log 2>&1 &

    nohup uv run python -m benchmark_locomo.task_eval.locomo_triple_router_quantification \
        --llm-model gpt-4o-mini-closeai \
        --llm-evaluate-model gpt-4o-mini-closeai \
        --topk-hierarchical 15 \
        --topk-similarity 30 \
        --topk-graph 0 \
        --topk-episodic 30 \
        --final-top-k 20 \
        --weight-hierarchical 0.34 \
        --weight-graph 0.33 \
        --weight-episodic 0.33 \
        --enable-second-stage-rerank \
        --rerank-strategy tower_separate \
        --reranker-type baai \
        --enable-router \
        --router-strategy aggressive \
        --enable-cascade-pruner \
        --cascade-mad-multiplier 3.0 \
        --cascade-lambda-mmr 0.7 \
        --cascade-prune-mode BUDGET_MAX \
        --cascade-max-context-tokens 1800 \
        --output-dir benchmark_locomo/task_eval/results/locomo_tri_tower_router_cascade_results_v2/gpt4o_mini_budget \
        --log-level INFO \
        > nohup_output/router_cascade_v2_gpt4o_budget.log 2>&1 &

    nohup uv run python -m benchmark_locomo.task_eval.locomo_triple_router_quantification \
        --llm-model gpt-4.1-mini-closeai \
        --llm-evaluate-model gpt-4o-mini-closeai \
        --topk-hierarchical 15 \
        --topk-similarity 30 \
        --topk-graph 0 \
        --topk-episodic 30 \
        --final-top-k 20 \
        --weight-hierarchical 0.34 \
        --weight-graph 0.33 \
        --weight-episodic 0.33 \
        --enable-second-stage-rerank \
        --rerank-strategy tower_separate \
        --reranker-type baai \
        --enable-router \
        --router-strategy aggressive \
        --enable-cascade-pruner \
        --cascade-mad-multiplier 3.0 \
        --cascade-lambda-mmr 0.7 \
        --cascade-prune-mode DYNAMIC_ADAPTIVE \
        --cascade-adaptive-dataset locomo \
        --cascade-cliff-tolerance 2.5 \
        --cascade-max-context-tokens 1500 \
        --output-dir benchmark_locomo/task_eval/results/locomo_tri_tower_router_cascade_results_v2/gpt41_mini_dynamic_adaptive \
        --log-level INFO \
        > nohup_output/router_cascade_v2_gpt41_dynamic_adaptive.log 2>&1 &

    nohup uv run python -m benchmark_locomo.task_eval.locomo_triple_router_quantification \
        --llm-model gpt-4o-mini-closeai \
        --llm-evaluate-model gpt-4o-mini-closeai \
        --topk-hierarchical 15 \
        --topk-similarity 30 \
        --topk-graph 0 \
        --topk-episodic 30 \
        --final-top-k 20 \
        --weight-hierarchical 0.34 \
        --weight-graph 0.33 \
        --weight-episodic 0.33 \
        --enable-second-stage-rerank \
        --rerank-strategy tower_separate \
        --reranker-type baai \
        --enable-router \
        --router-strategy aggressive \
        --enable-cascade-pruner \
        --cascade-mad-multiplier 3.0 \
        --cascade-lambda-mmr 0.7 \
        --cascade-prune-mode DYNAMIC_ADAPTIVE \
        --cascade-adaptive-dataset locomo \
        --cascade-cliff-tolerance 3.0 \
        --cascade-max-context-tokens 1800 \
        --output-dir benchmark_locomo/task_eval/results/locomo_tri_tower_router_cascade_results_v2/gpt4o_mini_dynamic_adaptive \
        --log-level INFO \
        > nohup_output/router_cascade_v2_gpt4o_dynamic_adaptive.log 2>&1 &

    nohup uv run python -m benchmark_locomo.task_eval.locomo_triple_router_quantification \
        --llm-model gpt-4.1-mini-closeai \
        --llm-evaluate-model gpt-4o-mini-closeai \
        --topk-hierarchical 15 \
        --topk-similarity 30 \
        --topk-graph 0 \
        --topk-episodic 30 \
        --final-top-k 20 \
        --weight-hierarchical 0.34 \
        --weight-graph 0.33 \
        --weight-episodic 0.33 \
        --enable-second-stage-rerank \
        --rerank-strategy tower_separate \
        --reranker-type baai \
        --enable-cascade-pruner \
        --cascade-mad-multiplier 3.0 \
        --cascade-lambda-mmr 0.7 \
        --cascade-prune-mode DYNAMIC_ADAPTIVE \
        --cascade-adaptive-dataset locomo \
        --cascade-cliff-tolerance 2.5 \
        --cascade-max-context-tokens 1500 \
        --output-dir benchmark_locomo/task_eval/results/locomo_tri_tower_router_cascade_results_v2/gpt41_mini_ablation_no_router \
        --log-level INFO \
        > nohup_output/router_cascade_v2_gpt41_ablation_no_router.log 2>&1 &

    nohup uv run python -m benchmark_locomo.task_eval.locomo_triple_router_quantification \
        --llm-model gpt-4o-mini-closeai \
        --llm-evaluate-model gpt-4o-mini-closeai \
        --topk-hierarchical 15 \
        --topk-similarity 30 \
        --topk-graph 0 \
        --topk-episodic 30 \
        --final-top-k 20 \
        --weight-hierarchical 0.34 \
        --weight-graph 0.33 \
        --weight-episodic 0.33 \
        --enable-second-stage-rerank \
        --rerank-strategy tower_separate \
        --reranker-type baai \
        --enable-cascade-pruner \
        --cascade-mad-multiplier 3.0 \
        --cascade-lambda-mmr 0.7 \
        --cascade-prune-mode DYNAMIC_ADAPTIVE \
        --cascade-adaptive-dataset locomo \
        --cascade-cliff-tolerance 3.0 \
        --cascade-max-context-tokens 1800 \
        --output-dir benchmark_locomo/task_eval/results/locomo_tri_tower_router_cascade_results_v2/gpt4o_mini_ablation_no_router \
        --log-level INFO \
        > nohup_output/router_cascade_v2_gpt4o_ablation_no_router.log 2>&1 &

    nohup uv run python -m benchmark_locomo.task_eval.locomo_triple_router_quantification \
        --llm-model gpt-4.1-mini-closeai \
        --llm-evaluate-model gpt-4o-mini-closeai \
        --topk-hierarchical 15 \
        --topk-similarity 30 \
        --topk-graph 0 \
        --topk-episodic 30 \
        --final-top-k 20 \
        --weight-hierarchical 0.34 \
        --weight-graph 0.33 \
        --weight-episodic 0.33 \
        --enable-second-stage-rerank \
        --rerank-strategy tower_separate \
        --reranker-type baai \
        --enable-router \
        --router-strategy aggressive \
        --enable-cascade-pruner \
        --cascade-mad-multiplier 3.0 \
        --cascade-lambda-mmr 0.7 \
        --cascade-prune-mode DYNAMIC_ADAPTIVE \
        --cascade-adaptive-dataset locomo \
        --cascade-cliff-tolerance 2.5 \
        --cascade-max-context-tokens 1500 \
        --no-cascade-stage1 \
        --no-cascade-stage2 \
        --output-dir benchmark_locomo/task_eval/results/locomo_tri_tower_router_cascade_results_v2/gpt41_mini_ablation_no_stage12 \
        --log-level INFO \
        > nohup_output/router_cascade_v2_gpt41_ablation_no_stage12.log 2>&1 &

    nohup uv run python -m benchmark_locomo.task_eval.locomo_triple_router_quantification \
        --llm-model gpt-4o-mini-closeai \
        --llm-evaluate-model gpt-4o-mini-closeai \
        --topk-hierarchical 15 \
        --topk-similarity 30 \
        --topk-graph 0 \
        --topk-episodic 30 \
        --final-top-k 20 \
        --weight-hierarchical 0.34 \
        --weight-graph 0.33 \
        --weight-episodic 0.33 \
        --enable-second-stage-rerank \
        --rerank-strategy tower_separate \
        --reranker-type baai \
        --enable-router \
        --router-strategy aggressive \
        --enable-cascade-pruner \
        --cascade-mad-multiplier 3.0 \
        --cascade-lambda-mmr 0.7 \
        --cascade-prune-mode DYNAMIC_ADAPTIVE \
        --cascade-adaptive-dataset locomo \
        --cascade-cliff-tolerance 3.0 \
        --cascade-max-context-tokens 1800 \
        --no-cascade-stage1 \
        --no-cascade-stage2 \
        --output-dir benchmark_locomo/task_eval/results/locomo_tri_tower_router_cascade_results_v2/gpt4o_mini_ablation_no_stage12 \
        --log-level INFO \
        > nohup_output/router_cascade_v2_gpt4o_ablation_no_stage12.log 2>&1 &

    nohup uv run python -m benchmark_locomo.task_eval.locomo_triple_router_quantification \
        --llm-model gpt-4.1-mini-closeai \
        --llm-evaluate-model gpt-4o-mini-closeai \
        --topk-hierarchical 15 \
        --topk-similarity 30 \
        --topk-graph 0 \
        --topk-episodic 30 \
        --final-top-k 20 \
        --weight-hierarchical 0.34 \
        --weight-graph 0.33 \
        --weight-episodic 0.33 \
        --enable-second-stage-rerank \
        --rerank-strategy tower_separate \
        --reranker-type baai \
        --enable-router \
        --router-strategy aggressive \
        --enable-cascade-pruner \
        --cascade-mad-multiplier 3.0 \
        --cascade-lambda-mmr 0.7 \
        --cascade-prune-mode BUDGET_MAX \
        --cascade-cliff-tolerance 2.5 \
        --cascade-max-context-tokens 1500 \
        --no-cascade-stage3-mmr \
        --output-dir benchmark_locomo/task_eval/results/locomo_tri_tower_router_cascade_results_v2/gpt41_mini_ablation_no_packing \
        --log-level INFO \
        > nohup_output/router_cascade_v2_gpt41_ablation_no_packing.log 2>&1 &

    nohup uv run python -m benchmark_locomo.task_eval.locomo_triple_router_quantification \
        --llm-model gpt-4o-mini-closeai \
        --llm-evaluate-model gpt-4o-mini-closeai \
        --topk-hierarchical 15 \
        --topk-similarity 30 \
        --topk-graph 0 \
        --topk-episodic 30 \
        --final-top-k 20 \
        --weight-hierarchical 0.34 \
        --weight-graph 0.33 \
        --weight-episodic 0.33 \
        --enable-second-stage-rerank \
        --rerank-strategy tower_separate \
        --reranker-type baai \
        --enable-router \
        --router-strategy aggressive \
        --enable-cascade-pruner \
        --cascade-mad-multiplier 3.0 \
        --cascade-lambda-mmr 0.7 \
        --cascade-prune-mode BUDGET_MAX \
        --cascade-cliff-tolerance 3.0 \
        --cascade-max-context-tokens 1800 \
        --no-cascade-stage3-mmr \
        --output-dir benchmark_locomo/task_eval/results/locomo_tri_tower_router_cascade_results_v2/gpt4o_mini_ablation_no_packing \
        --log-level INFO \
        > nohup_output/router_cascade_v2_gpt4o_ablation_no_packing.log 2>&1 &
fi

# ==============================================================================
# 5. Optional speed/debug commands
# Source: run_speed_benchmarks.sh
# ==============================================================================
if [[ "$RUN_DEBUG_SPEED" == "1" ]]; then
    echo "[debug_speed] running LoCoMo insertion and reranked smart-search QPS pipeline..."
    export ENVIRONMENT=speed

    uv run python -m benchmark_locomo.task_eval.locomo_triple_input_speed \
        --data-dir benchmark_locomo/dataset/locomo \
        --total-requests 2000 \
        --qps 10 \
        >> benchmark_triple_input_speed_output.log 2>&1

    RERANKER_BACKEND=vllm VLLM_API_URL="${VLLM_API_URL:-http://127.0.0.1:8000/score}" \
        uv run python -m benchmark_locomo.task_eval.locomo_triple_smart_search_qps \
            --qps 10 \
            --top-k 35 \
            --rerank-method baai \
            >> benchmark_smart_search_qps_10_vllm_output.log 2>&1

    RERANKER_BACKEND=native \
        uv run python -m benchmark_locomo.task_eval.locomo_triple_smart_search_qps \
            --qps 10 \
            --top-k 35 \
            --rerank-method baai \
            >> benchmark_smart_search_qps_10_native_output.log 2>&1

    RERANKER_BACKEND=vllm VLLM_API_URL="${VLLM_API_URL:-http://127.0.0.1:8000/score}" \
        uv run python -m benchmark_locomo.task_eval.locomo_triple_smart_search_qps \
            --qps 5 \
            --top-k 35 \
            --rerank-method baai \
            >> benchmark_smart_search_qps_5_vllm_output.log 2>&1

    RERANKER_BACKEND=native \
        uv run python -m benchmark_locomo.task_eval.locomo_triple_smart_search_qps \
            --qps 5 \
            --top-k 35 \
            --rerank-method baai \
            >> benchmark_smart_search_qps_5_native_output.log 2>&1
fi

echo "All selected reproduction sections have been submitted."
