#!/usr/bin/env bash
# Purpose: launch LongMemEval GPT-5 triple-tower and tower-removal ablations,
# including GPT-5 self-judged and GPT-4o-mini judged variants.
# Runs: benchmark_longmemeval.task_eval.benchmark_triple and router variants.
set -euo pipefail

mkdir -p \
    benchmark_longmemeval/task_eval/results/ablations/gpt5/top-50_25_6_9/full_tri_tower \
    benchmark_longmemeval/task_eval/results/ablations/gpt5/top-50_25_6_9/ablation_wo_sentence \
    benchmark_longmemeval/task_eval/results/ablations/gpt5/top-50_25_6_9/ablation_wo_episodic \
    benchmark_longmemeval/task_eval/results/ablations/gpt5/top-50_25_6_9/ablation_wo_entity \
    benchmark_longmemeval/task_eval/results/ablations/gpt5/eval_gpt4o_mini/full_tri_tower \
    benchmark_longmemeval/task_eval/results/ablations/gpt5/eval_gpt4o_mini/top-50_25_6_9/full_tri_tower \
    benchmark_longmemeval/task_eval/results/ablations/gpt5/eval_gpt4o_mini/top-50_25_6_9/ablation_wo_sentence \
    benchmark_longmemeval/task_eval/results/ablations/gpt5/eval_gpt4o_mini/top-50_25_6_9/ablation_wo_episodic \
    benchmark_longmemeval/task_eval/results/ablations/gpt5/eval_gpt4o_mini/top-50_25_6_9/ablation_wo_entity \
    benchmark_longmemeval/task_eval/results/ablations/gpt5/eval_gpt4o_mini/router_top50_aggressive \
    benchmark_longmemeval/task_eval/results/ablations/gpt5/eval_gpt4o_mini/router_quantification_top50_aggressive \
    benchmark_longmemeval/task_eval/results/ablations/gpt5/eval_gpt4o_mini/router_quantification_top50_dynamic_adaptive_acc_first

nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple \
    --dataset-size s \
    --llm-model gpt-5-closeai \
    --llm-evaluate-model gpt-5-closeai \
    --final-top-k 25 \
    --generation-max-tokens 8192 \
    --output-dir "benchmark_longmemeval/task_eval/results/ablations/gpt5/full_tri_tower" \
    > log_baseline_gpt5.log 2>&1 &

# Baseline top-50: 完整三塔 (Sentence + Episodic + Entity)
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple \
    --dataset-size s \
    --llm-model gpt-5-closeai \
    --llm-evaluate-model gpt-5-closeai \
    --final-top-k 50 \
    --generation-max-tokens 8192 \
    --output-dir "benchmark_longmemeval/task_eval/results/ablations/gpt5/top-50_25_6_9/full_tri_tower" \
    > log_baseline_gpt5_top50.log 2>&1 &

# 无原始对话 w/o Sentence (仅 Episodic + Entity), top-k=50
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple \
    --dataset-size s \
    --llm-model gpt-5-closeai \
    --llm-evaluate-model gpt-5-closeai \
    --final-top-k 50 \
    --generation-max-tokens 8192 \
    --disable-sentence \
    --output-dir "benchmark_longmemeval/task_eval/results/ablations/gpt5/ablation_wo_sentence" \
    > log_wo_sentence_gpt5_top50.log 2>&1 &

# 无情景记忆 w/o Episodic (仅 Sentence + Entity), top-k=50
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple \
    --dataset-size s \
    --llm-model gpt-5-closeai \
    --llm-evaluate-model gpt-5-closeai \
    --final-top-k 50 \
    --generation-max-tokens 8192 \
    --disable-episodic \
    --output-dir "benchmark_longmemeval/task_eval/results/ablations/gpt5/ablation_wo_episodic" \
    > log_wo_episodic_gpt5_top50.log 2>&1 &

# 无实体关系 w/o Entity (仅 Sentence + Episodic), top-k=50
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple \
    --dataset-size s \
    --llm-model gpt-5-closeai \
    --llm-evaluate-model gpt-5-closeai \
    --final-top-k 50 \
    --generation-max-tokens 8192 \
    --disable-entity \
    --output-dir "benchmark_longmemeval/task_eval/results/ablations/gpt5/ablation_wo_entity" \
    > log_wo_entity_gpt5_top50.log 2>&1 &

nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple \
    --dataset-size s \
    --llm-model gpt-5-openrouter \
    --llm-evaluate-model gpt-5-openrouter \
    --final-top-k 25 \
    --generation-max-tokens 8192 \
    --output-dir "benchmark_longmemeval/task_eval/results/ablations/gpt5/full_tri_tower" \
    > log_baseline_gpt5.log 2>&1 &

# Baseline top-25: GPT-5 生成 + gpt-4o-mini 评估
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple \
    --dataset-size s \
    --llm-model gpt-5-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
    --final-top-k 25 \
    --generation-max-tokens 8192 \
    --output-dir "benchmark_longmemeval/task_eval/results/ablations/gpt5/eval_gpt4o_mini/full_tri_tower" \
    > log_baseline_gpt5_eval_gpt4o.log 2>&1 &

# Baseline top-50: GPT-5 生成 + gpt-4o-mini 评估
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple \
    --dataset-size s \
    --llm-model gpt-5-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
    --final-top-k 50 \
    --generation-max-tokens 8192 \
    --output-dir "benchmark_longmemeval/task_eval/results/ablations/gpt5/eval_gpt4o_mini/full_tri_tower" \
    > log_baseline_gpt5_top50_eval_gpt4o.log 2>&1 &

# 无原始对话 w/o Sentence (仅 Episodic + Entity), top-k=50, gpt-4o-mini 评估
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple \
    --dataset-size s \
    --llm-model gpt-5-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
    --final-top-k 50 \
    --generation-max-tokens 8192 \
    --disable-sentence \
    --output-dir "benchmark_longmemeval/task_eval/results/ablations/gpt5/eval_gpt4o_mini/ablation_wo_sentence" \
    > log_wo_sentence_gpt5_top50_eval_gpt4o.log 2>&1 &

# 无情景记忆 w/o Episodic (仅 Sentence + Entity), top-k=50, gpt-4o-mini 评估
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple \
    --dataset-size s \
    --llm-model gpt-5-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
    --final-top-k 50 \
    --generation-max-tokens 8192 \
    --disable-episodic \
    --output-dir "benchmark_longmemeval/task_eval/results/ablations/gpt5/eval_gpt4o_mini/ablation_wo_episodic" \
    > log_wo_episodic_gpt5_top50_eval_gpt4o.log 2>&1 &

# 无实体关系 w/o Entity (仅 Sentence + Episodic), top-k=50, gpt-4o-mini 评估
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple \
    --dataset-size s \
    --llm-model gpt-5-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
    --final-top-k 50 \
    --generation-max-tokens 8192 \
    --disable-entity \
    --output-dir "benchmark_longmemeval/task_eval/results/ablations/gpt5/eval_gpt4o_mini/ablation_wo_entity" \
    > log_wo_entity_gpt5_top50_eval_gpt4o.log 2>&1 &

# GPT-5 LongMemEval tower router, top-k=50, gpt-4o-mini 评估
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple_router \
    --dataset-size s \
    --llm-model gpt-5-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
    --final-top-k 50 \
    --generation-max-tokens 8192 \
    --enable-router \
    --router-strategy aggressive \
    --output-dir "benchmark_longmemeval/task_eval/results/ablations/gpt5/eval_gpt4o_mini/router_top50_aggressive" \
    > log_router_gpt5_top50_eval_gpt4o.log 2>&1 &

# GPT-5 LongMemEval tower router quantification entry, top-k=50, gpt-4o-mini 评估
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple_router_quantification \
    --dataset-size s \
    --llm-model gpt-5-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
    --final-top-k 50 \
    --generation-max-tokens 8192 \
    --enable-router \
    --router-strategy aggressive \
    --output-dir "benchmark_longmemeval/task_eval/results/ablations/gpt5/eval_gpt4o_mini/router_quantification_top50_aggressive" \
    > log_router_quantification_gpt5_top50_eval_gpt4o.log 2>&1 &

# GPT-5 LongMemEval router + cascade quantification, accuracy-first denoising.
# Rationale from GPT-5 ablations: keep top-50 router, avoid hard 2500-token compression,
# preserve evidence-chain categories, and only trim long-tail/redundant context.
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple_router_quantification \
    --dataset-size s \
    --llm-model gpt-5-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
    --final-top-k 50 \
    --generation-max-tokens 8192 \
    --enable-router \
    --router-strategy aggressive \
    --enable-cascade-pruner \
    --cascade-prune-mode DYNAMIC_ADAPTIVE \
    --cascade-adaptive-dataset longmemeval \
    --cascade-max-context-tokens 5000 \
    --cascade-mad-multiplier 3.5 \
    --cascade-lambda-mmr 0.85 \
    --cascade-tower-min-ratio "H:0.50,E:0.20,KG:0.15" \
    --output-dir "benchmark_longmemeval/task_eval/results/ablations/gpt5/eval_gpt4o_mini/router_quantification_top50_dynamic_adaptive_acc_first" \
    > log_router_quantification_gpt5_top50_dynamic_adaptive_acc_first_eval_gpt4o.log 2>&1 &
