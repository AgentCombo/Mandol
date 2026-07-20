#!/bin/bash
# Purpose: launch the LoCoMo triple-tower ablation suite through CloseAI models,
# including the full baseline and tower-removal variants.
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


echo "🚀 开始提交基于【非对称双轨架构 (tower_separate)】的 LoCoMo10 消融实验任务..."
echo "🔒 核心逻辑: 直通车(15) + 严控重排序出口(20)"
echo "=============================================================================="

# ==============================================================================
# 🌟 第一组：基于 GPT-4o-mini 的消融实验
# ==============================================================================
echo "▶️  提交 GPT-4o-mini 任务组..."

# 1. Baseline: 完整三塔 (直通车15 + 图谱/情景争夺20)
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
echo "  [1/8] GPT-4o-mini Baseline 启动"

# 2. 无情景记忆 w/o Episodic (直通车15 + 图谱独享20)
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
echo "  [2/8] GPT-4o-mini w/o Episodic 启动"

# 3. 无知识图谱 w/o Graph (直通车15 + 情景独享20)
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
echo "  [3/8] GPT-4o-mini w/o Graph 启动"

# 4. 无分层记忆 w/o Hierarchical (直通车断掉 + 图谱/情景争夺20)
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
echo "  [4/8] GPT-4o-mini w/o Hierarchical 启动"


# ==============================================================================
# 🌟 第二组：基于 GPT-4.1-mini 的消融实验
# ==============================================================================
echo "▶️  提交 GPT-4.1-mini 任务组..."

# 1. Baseline: 完整三塔
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
echo "  [5/8] GPT-4.1-mini Baseline 启动"

# 2. 无情景记忆 w/o Episodic
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
echo "  [6/8] GPT-4.1-mini w/o Episodic 启动"

# 3. 无知识图谱 w/o Graph
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
echo "  [7/8] GPT-4.1-mini w/o Graph 启动"

# 4. 无分层记忆 w/o Hierarchical
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
echo "  [8/8] GPT-4.1-mini w/o Hierarchical 启动"

echo "=============================================================================="
echo "🎉 所有的 8 个任务已基于 tower_separate 架构成功放置在后台运行！"
echo "👉 输出目录已更新为: locomo10_ablation_separate/"
echo "你可以使用 'top' 或 'htop' 查看进程，或者使用 'tail -f' 查看对应目录下的 run.log"
echo "=============================================================================="

# #!/bin/bash

# echo "🚀 开始提交所有的 LoCoMo10 消融实验任务到后台..."

# # ==============================================================================
# # 🌟 第一组：基于 GPT-4o-mini 的消融实验
# # ==============================================================================
# echo "▶️  提交 GPT-4o-mini 任务组..."

# # 1. Baseline: 完整三塔
# mkdir -p benchmark_locomo/task_eval/results/locomo10_ablation/gpt_4o_mini/full_tri_tower
# nohup uv run python -m benchmark_locomo.task_eval.locomo_triple \
#     --qa-dataset "benchmark_locomo/dataset/locomo/locomo10.json" \
#     --llm-model gpt-4o-mini-closeai \
#     --llm-evaluate-model gpt-4o-mini-closeai \
#     --enable-second-stage-rerank \
#     --rerank-strategy unified_rerank \
#     --output-dir "benchmark_locomo/task_eval/results/locomo10_ablation/gpt_4o_mini/full_tri_tower" \
#     --topk-hierarchical 15 \
#     --topk-similarity 30 \
#     --topk-graph 0 \
#     --no-entity-relation \
#     --topk-episodic 30 \
#     --final-top-k 20 \
#     > benchmark_locomo/task_eval/results/locomo10_ablation/gpt_4o_mini/full_tri_tower/run.log 2>&1 &
# echo "  [1/8] GPT-4o-mini Baseline 启动"

# # 2. 无情景记忆 (w/o Episodic)
# mkdir -p benchmark_locomo/task_eval/results/locomo10_ablation/gpt_4o_mini/wo_episodic
# nohup uv run python -m benchmark_locomo.task_eval.locomo_triple \
#     --qa-dataset "benchmark_locomo/dataset/locomo/locomo10.json" \
#     --llm-model gpt-4o-mini-closeai \
#     --llm-evaluate-model gpt-4o-mini-closeai \
#     --enable-second-stage-rerank \
#     --rerank-strategy unified_rerank \
#     --output-dir "benchmark_locomo/task_eval/results/locomo10_ablation/gpt_4o_mini/wo_episodic" \
#     --topk-hierarchical 15 \
#     --topk-similarity 30 \
#     --topk-graph 0 \
#     --no-entity-relation \
#     --topk-episodic 0 \
#     --final-top-k 20 \
#     > benchmark_locomo/task_eval/results/locomo10_ablation/gpt_4o_mini/wo_episodic/run.log 2>&1 &
# echo "  [2/8] GPT-4o-mini w/o Episodic 启动"

# # 3. 无知识图谱节点 (w/o Graph)
# mkdir -p benchmark_locomo/task_eval/results/locomo10_ablation/gpt_4o_mini/wo_graph
# nohup uv run python -m benchmark_locomo.task_eval.locomo_triple \
#     --qa-dataset "benchmark_locomo/dataset/locomo/locomo10.json" \
#     --llm-model gpt-4o-mini-closeai \
#     --llm-evaluate-model gpt-4o-mini-closeai \
#     --enable-second-stage-rerank \
#     --rerank-strategy unified_rerank \
#     --output-dir "benchmark_locomo/task_eval/results/locomo10_ablation/gpt_4o_mini/wo_graph" \
#     --topk-hierarchical 15 \
#     --topk-similarity 0 \
#     --topk-graph 0 \
#     --no-entity-relation \
#     --topk-episodic 30 \
#     --final-top-k 20 \
#     > benchmark_locomo/task_eval/results/locomo10_ablation/gpt_4o_mini/wo_graph/run.log 2>&1 &
# echo "  [3/8] GPT-4o-mini w/o Graph 启动"

# # 4. 无分层记忆 (w/o Hierarchical)
# mkdir -p benchmark_locomo/task_eval/results/locomo10_ablation/gpt_4o_mini/wo_hierarchical
# nohup uv run python -m benchmark_locomo.task_eval.locomo_triple \
#     --qa-dataset "benchmark_locomo/dataset/locomo/locomo10.json" \
#     --llm-model gpt-4o-mini-closeai \
#     --llm-evaluate-model gpt-4o-mini-closeai \
#     --enable-second-stage-rerank \
#     --rerank-strategy unified_rerank \
#     --output-dir "benchmark_locomo/task_eval/results/locomo10_ablation/gpt_4o_mini/wo_hierarchical" \
#     --topk-hierarchical 0 \
#     --topk-similarity 30 \
#     --topk-graph 0 \
#     --no-entity-relation \
#     --topk-episodic 30 \
#     --final-top-k 20 \
#     > benchmark_locomo/task_eval/results/locomo10_ablation/gpt_4o_mini/wo_hierarchical/run.log 2>&1 &
# echo "  [4/8] GPT-4o-mini w/o Hierarchical 启动"


# # ==============================================================================
# # 🌟 第二组：基于 GPT-4.1-mini 的消融实验
# # ==============================================================================
# echo "▶️  提交 GPT-4.1-mini 任务组..."

# # 1. Baseline: 完整三塔
# mkdir -p benchmark_locomo/task_eval/results/locomo10_ablation/gpt_4.1_mini/full_tri_tower
# nohup uv run python -m benchmark_locomo.task_eval.locomo_triple \
#     --qa-dataset "benchmark_locomo/dataset/locomo/locomo10.json" \
#     --llm-model gpt-4.1-mini-closeai \
#     --llm-evaluate-model gpt-4o-mini-closeai \
#     --enable-second-stage-rerank \
#     --rerank-strategy unified_rerank \
#     --output-dir "benchmark_locomo/task_eval/results/locomo10_ablation/gpt_4.1_mini/full_tri_tower" \
#     --topk-hierarchical 15 \
#     --topk-similarity 30 \
#     --topk-graph 0 \
#     --no-entity-relation \
#     --topk-episodic 30 \
#     --final-top-k 20 \
#     > benchmark_locomo/task_eval/results/locomo10_ablation/gpt_4.1_mini/full_tri_tower/run.log 2>&1 &
# echo "  [5/8] GPT-4.1-mini Baseline 启动"

# # 2. 无情景记忆 (w/o Episodic)
# mkdir -p benchmark_locomo/task_eval/results/locomo10_ablation/gpt_4.1_mini/wo_episodic
# nohup uv run python -m benchmark_locomo.task_eval.locomo_triple \
#     --qa-dataset "benchmark_locomo/dataset/locomo/locomo10.json" \
#     --llm-model gpt-4.1-mini-closeai \
#     --llm-evaluate-model gpt-4o-mini-closeai \
#     --enable-second-stage-rerank \
#     --rerank-strategy unified_rerank \
#     --output-dir "benchmark_locomo/task_eval/results/locomo10_ablation/gpt_4.1_mini/wo_episodic" \
#     --topk-hierarchical 15 \
#     --topk-similarity 30 \
#     --topk-graph 0 \
#     --no-entity-relation \
#     --topk-episodic 0 \
#     --final-top-k 20 \
#     > benchmark_locomo/task_eval/results/locomo10_ablation/gpt_4.1_mini/wo_episodic/run.log 2>&1 &
# echo "  [6/8] GPT-4.1-mini w/o Episodic 启动"

# # 3. 无知识图谱节点 (w/o Graph)
# mkdir -p benchmark_locomo/task_eval/results/locomo10_ablation/gpt_4.1_mini/wo_graph
# nohup uv run python -m benchmark_locomo.task_eval.locomo_triple \
#     --qa-dataset "benchmark_locomo/dataset/locomo/locomo10.json" \
#     --llm-model gpt-4.1-mini-closeai \
#     --llm-evaluate-model gpt-4o-mini-closeai \
#     --enable-second-stage-rerank \
#     --rerank-strategy unified_rerank \
#     --output-dir "benchmark_locomo/task_eval/results/locomo10_ablation/gpt_4.1_mini/wo_graph" \
#     --topk-hierarchical 15 \
#     --topk-similarity 0 \
#     --topk-graph 0 \
#     --no-entity-relation \
#     --topk-episodic 30 \
#     --final-top-k 20 \
#     > benchmark_locomo/task_eval/results/locomo10_ablation/gpt_4.1_mini/wo_graph/run.log 2>&1 &
# echo "  [7/8] GPT-4.1-mini w/o Graph 启动"

# # 4. 无分层记忆 (w/o Hierarchical)
# mkdir -p benchmark_locomo/task_eval/results/locomo10_ablation/gpt_4.1_mini/wo_hierarchical
# nohup uv run python -m benchmark_locomo.task_eval.locomo_triple \
#     --qa-dataset "benchmark_locomo/dataset/locomo/locomo10.json" \
#     --llm-model gpt-4.1-mini-closeai \
#     --llm-evaluate-model gpt-4o-mini-closeai \
#     --enable-second-stage-rerank \
#     --rerank-strategy unified_rerank \
#     --output-dir "benchmark_locomo/task_eval/results/locomo10_ablation/gpt_4.1_mini/wo_hierarchical" \
#     --topk-hierarchical 0 \
#     --topk-similarity 30 \
#     --topk-graph 0 \
#     --no-entity-relation \
#     --topk-episodic 30 \
#     --final-top-k 20 \
#     > benchmark_locomo/task_eval/results/locomo10_ablation/gpt_4.1_mini/wo_hierarchical/run.log 2>&1 &
# echo "  [8/8] GPT-4.1-mini w/o Hierarchical 启动"

# echo "=============================================================================="
# echo "🎉 所有的 8 个任务已成功放置在后台运行！"
# echo "你可以使用 'top' 或 'htop' 查看进程，或者使用 'tail -f' 查看对应目录下的 run.log"
# echo "=============================================================================="
