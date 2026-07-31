#!/bin/bash
# Purpose: launch LoCoMo tower-removal ablations without the full baseline,
# staged across GPT-4o-mini and GPT-4.1-mini CloseAI runs.
# Runs: benchmark_locomo.task_eval.locomo_triple.

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


echo "🚀 开始提交【纯消融实验】任务 (不含 Baseline)..."
echo "🔒 核心逻辑: 严格维持 35 条 Context Budget 预算公平对齐"
echo "⏸️  执行策略: 第一组(gpt-4o-mini) 结束后再启动 第二组(gpt-4.1-mini)"
echo "=============================================================================="

# ==============================================================================
# 🌟 第一组：基于 GPT-4o-mini 的消融实验
# ==============================================================================
echo "▶️  提交 GPT-4o-mini 任务组 (共 3 个实验)..."

# 1. 无情景记忆 w/o Episodic (直通车15 + 图谱独享20)
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
echo "  [1/6] GPT-4o-mini w/o Episodic 启动"

# 2. 无知识图谱 w/o Graph (直通车15 + 情景独享20)
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
echo "  [2/6] GPT-4o-mini w/o Graph 启动"

# 3. 无分层记忆 w/o Hierarchical (直通车断掉 + 图谱/情景争夺35)
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
echo "  [3/6] GPT-4o-mini w/o Hierarchical 启动"

# ------------------------------------------------------------------------------
# 🚧 核心控制：等待第一组全部完成 🚧
# ------------------------------------------------------------------------------
echo "⏳ 正在等待 GPT-4o-mini 组的 3 个消融实验执行完毕，以释放显存..."
wait
echo "✅ GPT-4o-mini 组任务已全部完成！准备启动下一组..."
echo "------------------------------------------------------------------------------"


# ==============================================================================
# 🌟 第二组：基于 GPT-4.1-mini 的消融实验
# ==============================================================================
echo "▶️  提交 GPT-4.1-mini 任务组 (共 3 个实验)..."

# 4. 无情景记忆 w/o Episodic
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
echo "  [4/6] GPT-4.1-mini w/o Episodic 启动"

# 5. 无知识图谱 w/o Graph
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
echo "  [5/6] GPT-4.1-mini w/o Graph 启动"

# 6. 无分层记忆 w/o Hierarchical
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
echo "  [6/6] GPT-4.1-mini w/o Hierarchical 启动"

# ------------------------------------------------------------------------------
# 🚧 等待最终完成 🚧
# ------------------------------------------------------------------------------
wait

echo "=============================================================================="
echo "🎉 所有消融实验批次（两组，共6个任务）均已执行完毕！"
echo "👉 输出目录: locomo10_ablation_separate/"
echo "=============================================================================="
