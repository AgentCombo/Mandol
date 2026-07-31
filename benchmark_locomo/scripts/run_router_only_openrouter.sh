#!/bin/bash
# Purpose: launch LoCoMo router-only benchmark runs through OpenRouter models.
# Runs: benchmark_locomo.task_eval.locomo_triple_router with aggressive and
# conservative routing for GPT-4.1-mini and GPT-4o-mini.

# Resolve repository root for the src/ package layout.
AMS_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AMS_REPO_ROOT="$AMS_SCRIPT_DIR"
while [[ "$AMS_REPO_ROOT" != "/" && ! -d "$AMS_REPO_ROOT/src/mandol" ]]; do
    AMS_REPO_ROOT="$(dirname "$AMS_REPO_ROOT")"
done
if [[ ! -d "$AMS_REPO_ROOT/src/mandol" ]]; then
    echo "Could not locate Mandol repo root from $AMS_SCRIPT_DIR" >&2
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

mkdir -p benchmark_locomo/task_eval/results/tower_router/gpt4.1_mini \
    benchmark_locomo/task_eval/results/tower_router/gpt4o_mini


# ==============================================================================
# LoCoMo 路由 Benchmark (Tower Router) - OpenRouter API
#
# 基于消融实验的最优塔组合，按 category 动态路由三塔/双塔。
# 参数对齐消融实验基线 (full_tri_tower):
#   topk_hierarchical=15, topk_similarity=30, topk_episodic=30, final_top_k=20
#   weight_hierarchical=0.34, weight_graph=0.33, weight_episodic=0.33
#   rerank=tower_separate
#
# 脚本名: locomo_triple_router.py
# 输出根目录: benchmark_locomo/task_eval/results/tower_router/
# ==============================================================================

# ---------- GPT-4.1-mini + aggressive ----------
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

# ---------- GPT-4.1-mini + conservative ----------
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

# ---------- GPT-4o-mini + aggressive ----------
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

# ---------- GPT-4o-mini + conservative ----------
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

echo "4 个 LoCoMo 路由 Benchmark 任务已启动 (OpenRouter)。"
echo "输出目录: benchmark_locomo/task_eval/results/tower_router/"
echo "  gpt4.1_mini_routed_aggressive"
echo "  gpt4.1_mini_routed_conservative"
echo "  gpt4o_mini_routed_aggressive"
echo "  gpt4o_mini_routed_conservative"
echo "可使用 'tail -f log_locomo_routed_4.1_aggressive_openrouter.log' 实时查看进度。"
