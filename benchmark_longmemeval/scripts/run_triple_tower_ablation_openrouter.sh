#!/bin/bash
# Purpose: launch LongMemEval triple-tower ablations through OpenRouter models,
# including full, w/o sentence, w/o episodic, and w/o entity variants.
# Runs: benchmark_longmemeval.task_eval.benchmark_triple.

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

mkdir -p \
    benchmark_longmemeval/task_eval/results/ablations/gpt4o_mini/full_tri_tower \
    benchmark_longmemeval/task_eval/results/ablations/gpt4o_mini/ablation_wo_sentence \
    benchmark_longmemeval/task_eval/results/ablations/gpt4o_mini/ablation_wo_episodic \
    benchmark_longmemeval/task_eval/results/ablations/gpt4o_mini/ablation_wo_entity \
    benchmark_longmemeval/task_eval/results/ablations/gpt4.1_mini/full_tri_tower \
    benchmark_longmemeval/task_eval/results/ablations/gpt4.1_mini/ablation_wo_sentence \
    benchmark_longmemeval/task_eval/results/ablations/gpt4.1_mini/ablation_wo_episodic \
    benchmark_longmemeval/task_eval/results/ablations/gpt4.1_mini/ablation_wo_entity


# ==============================================================================
# 实验组 1: 基于 gpt-4o-mini-openrouter 
# 输出根目录: benchmark_longmemeval/task_eval/results/ablations/gpt4o_mini
# ==============================================================================

# Baseline: 完整三塔 (Sentence + Episodic + Entity)
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple \
    --dataset-size s \
    --llm-model gpt-4o-mini-openrouter \
    --final-top-k 25 \
    --output-dir "benchmark_longmemeval/task_eval/results/ablations/gpt4o_mini/full_tri_tower" \
    > log_baseline_gpt4o.log 2>&1 &

# 无原始对话 w/o Sentence (仅 Episodic + Entity)
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple \
    --dataset-size s \
    --llm-model gpt-4o-mini-openrouter \
    --final-top-k 25 \
    --disable-sentence \
    --output-dir "benchmark_longmemeval/task_eval/results/ablations/gpt4o_mini/ablation_wo_sentence" \
    > log_wo_sentence_gpt4o.log 2>&1 &

# 无情景记忆 w/o Episodic (仅 Sentence + Entity)
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple \
    --dataset-size s \
    --llm-model gpt-4o-mini-openrouter \
    --final-top-k 25 \
    --disable-episodic \
    --output-dir "benchmark_longmemeval/task_eval/results/ablations/gpt4o_mini/ablation_wo_episodic" \
    > log_wo_episodic_gpt4o.log 2>&1 &

# 无实体关系 w/o Entity (仅 Sentence + Episodic)
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple \
    --dataset-size s \
    --llm-model gpt-4o-mini-openrouter \
    --final-top-k 25 \
    --disable-entity \
    --output-dir "benchmark_longmemeval/task_eval/results/ablations/gpt4o_mini/ablation_wo_entity" \
    > log_wo_entity_gpt4o.log 2>&1 &


# ==============================================================================
# 实验组 2: 基于 gpt-4.1-mini-openrouter 
# 输出根目录: benchmark_longmemeval/task_eval/results/ablations/gpt4.1_mini
# ==============================================================================

# Baseline: 完整三塔 (Sentence + Episodic + Entity)
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple \
    --dataset-size s \
    --llm-model gpt-4.1-mini-openrouter \
    --final-top-k 25 \
    --output-dir "benchmark_longmemeval/task_eval/results/ablations/gpt4.1_mini/full_tri_tower" \
    > log_baseline_gpt4.1.log 2>&1 &

# 无原始对话 w/o Sentence (仅 Episodic + Entity)
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple \
    --dataset-size s \
    --llm-model gpt-4.1-mini-openrouter \
    --final-top-k 25 \
    --disable-sentence \
    --output-dir "benchmark_longmemeval/task_eval/results/ablations/gpt4.1_mini/ablation_wo_sentence" \
    > log_wo_sentence_gpt4.1.log 2>&1 &

# 无情景记忆 w/o Episodic (仅 Sentence + Entity)
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple \
    --dataset-size s \
    --llm-model gpt-4.1-mini-openrouter \
    --final-top-k 25 \
    --disable-episodic \
    --output-dir "benchmark_longmemeval/task_eval/results/ablations/gpt4.1_mini/ablation_wo_episodic" \
    > log_wo_episodic_gpt4.1.log 2>&1 &

# 无实体关系 w/o Entity (仅 Sentence + Episodic)
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple \
    --dataset-size s \
    --llm-model gpt-4.1-mini-openrouter \
    --final-top-k 25 \
    --disable-entity \
    --output-dir "benchmark_longmemeval/task_eval/results/ablations/gpt4.1_mini/ablation_wo_entity" \
    > log_wo_entity_gpt4.1.log 2>&1 &

# nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple \
#     --dataset-size s \
#     --llm-model gpt-5-openrouter \
#     --llm-evaluate-model gpt-5-openrouter \
#     --final-top-k 25 \
#     --generation-max-tokens 8192 \
#     --output-dir "benchmark_longmemeval/task_eval/results/ablations/gpt5/full_tri_tower" \
#     > log_baseline_gpt5.log 2>&1 &

echo "8个后台消融实验任务已启动。可使用 'tail -f log_baseline_gpt4o.log' 实时查看进度。"
