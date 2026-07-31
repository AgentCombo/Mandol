#!/bin/bash
# Purpose: launch LongMemEval router-only benchmark runs through CloseAI models.
# Runs: benchmark_longmemeval.task_eval.benchmark_triple_router with aggressive
# and conservative routing for GPT-4.1-mini and GPT-4o-mini.

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

mkdir -p benchmark_longmemeval/task_eval/results/tower_router/gpt4.1_mini \
    benchmark_longmemeval/task_eval/results/tower_router/gpt4o_mini


# ==============================================================================
# LongMemEval 路由 Benchmark (Tower Router) - CloseAI API
#
# 基于消融实验的最优塔组合，按 question_type 动态路由三塔/双塔。
# 参数对齐消融实验基线: sentence_top_k=60, episodic_top_k=40, entity_top_k=40,
#                       rerank_method=baai, final_top_k=25
#
# 脚本名: benchmark_triple_router.py
# 输出根目录: benchmark_longmemeval/task_eval/results/tower_router/
# ==============================================================================

# ---------- GPT-4.1-mini + aggressive ----------
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

# ---------- GPT-4.1-mini + conservative ----------
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

# ---------- GPT-4o-mini + aggressive ----------
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

# ---------- GPT-4o-mini + conservative ----------
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

echo "4 个 LongMemEval 路由 Benchmark 任务已启动 (CloseAI)。"
echo "输出目录: benchmark_longmemeval/task_eval/results/tower_router/"
echo "  gpt4.1_mini_routed_aggressive"
echo "  gpt4.1_mini_routed_conservative"
echo "  gpt4o_mini_routed_aggressive"
echo "  gpt4o_mini_routed_conservative"
echo "可使用 'tail -f log_lme_routed_4.1_aggressive_closeai.log' 实时查看进度。"
