#!/usr/bin/env bash
# Purpose: run LoCoMo insertion and reranked smart-search QPS benchmarks,
# including vLLM and native reranker comparisons.
# Requires: unified per-sample graphs built under benchmark_locomo/dataset/locomo.

# Resolve repository root for the src/ package layout.
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

# 定义统一的环境变量
export ENVIRONMENT=speed

echo "=== 开始运行 AgentMemorySystem 性能压测流水线 ==="

echo "[1/5] 运行三塔 10QPS 插入无漂移压测..."
uv run python -m benchmark_locomo.task_eval.locomo_triple_input_speed \
    --data-dir benchmark_locomo/dataset/locomo \
    --total-requests 2000 \
    --qps 10 \
    >> benchmark_triple_input_speed_output.log 2>&1


# ==========================================
# 2. 智搜 10QPS 压测对比 (vLLM vs Native, with reranking)
# ==========================================
echo "[2/5] 运行智搜 10QPS 压测 - vLLM HTTP reranker..."
RERANKER_BACKEND=vllm VLLM_API_URL="${VLLM_API_URL:-http://127.0.0.1:8000/score}" \
    uv run python -m benchmark_locomo.task_eval.locomo_triple_smart_search_qps \
        --qps 10 \
        --top-k 35 \
        --rerank-method baai \
        >> benchmark_smart_search_qps_10_vllm_output.log 2>&1

echo "[3/5] 运行智搜 10QPS 压测 - native local reranker..."
RERANKER_BACKEND=native \
    uv run python -m benchmark_locomo.task_eval.locomo_triple_smart_search_qps \
        --qps 10 \
        --top-k 35 \
        --rerank-method baai \
        >> benchmark_smart_search_qps_10_native_output.log 2>&1


# ==========================================
# 3. 智搜 5QPS 压测对比 (vLLM vs Native, with reranking)
# ==========================================
echo "[4/5] 运行智搜 5QPS 压测 - vLLM HTTP reranker..."
RERANKER_BACKEND=vllm VLLM_API_URL="${VLLM_API_URL:-http://127.0.0.1:8000/score}" \
    uv run python -m benchmark_locomo.task_eval.locomo_triple_smart_search_qps \
        --qps 5 \
        --top-k 35 \
        --rerank-method baai \
        >> benchmark_smart_search_qps_5_vllm_output.log 2>&1

echo "[5/5] 运行智搜 5QPS 压测 - native local reranker..."
RERANKER_BACKEND=native \
    uv run python -m benchmark_locomo.task_eval.locomo_triple_smart_search_qps \
        --qps 5 \
        --top-k 35 \
        --rerank-method baai \
        >> benchmark_smart_search_qps_5_native_output.log 2>&1

echo "=== 所有压测任务串行执行完毕！ ==="
