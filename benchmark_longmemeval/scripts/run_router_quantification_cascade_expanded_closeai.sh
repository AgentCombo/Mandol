#!/bin/bash
# Purpose: expanded LongMemEval router + quantification + cascade launcher where
# each experiment is written as a standalone command for copying or debugging.
# Runs: benchmark_longmemeval.task_eval.benchmark_triple_router_quantification.

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


# ============================================================================
# LongMemEval 路由 + 级联量化 — MAD + Raw Logits (v6)
# Expanded batch launcher aligned with run_router_quantification_cascade_closeai.sh.
# ============================================================================
#
# v5.1→v6 关键改动:
#   ✓ 移除 min-max 归一化, 直接在 raw logits 上做 MAD
#   ✓ Stage1 ON (k=3.0): ~99% survival (SOTA), 仅移除极端噪声
#   ✓ 新增 STRICT_THRESHOLD + CLIFF_EARLY_STOP 两种模式
#   ✓ 废弃 tau_high/tau_med/target_confidence 旧参数
#   ✓ 三种模式参数分别优化 (精度优先, 兼顾 token)

#!/bin/bash

# 确保日志输出目录存在
mkdir -p nohup_output
mkdir -p benchmark_longmemeval/task_eval/results/triple_fusion_router_cascade_mad_v3

echo "🚀 开始启动所有 6 个 MAD v6 benchmark 任务..."

# ============================================================================
# GPT-4o-mini 配置 (保守策略)
# ============================================================================

echo "启动 1/6: GPT-4o-mini + STRICT_THRESHOLD"
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
    --cascade-prune-mode STRICT_THRESHOLD \
    --cascade-mad-multiplier 3.0 \
    --cascade-absolute-min-score -9.0 \
    --cascade-max-context-tokens 2200 \
    --cascade-lambda-mmr 0.6 \
    --output-dir benchmark_longmemeval/task_eval/results/triple_fusion_router_cascade_mad_v3/gpt4o_mini_strict \
    > nohup_output/longmem_mad3_gpt4o_strict.log 2>&1 &

echo "启动 2/6: GPT-4o-mini + CLIFF_EARLY_STOP"
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
    --cascade-prune-mode CLIFF_EARLY_STOP \
    --cascade-mad-multiplier 3.0 \
    --cascade-cliff-tolerance 3.0 \
    --cascade-max-context-tokens 2200 \
    --cascade-lambda-mmr 0.6 \
    --output-dir benchmark_longmemeval/task_eval/results/triple_fusion_router_cascade_mad_v3/gpt4o_mini_cliff \
    > nohup_output/longmem_mad3_gpt4o_cliff.log 2>&1 &

echo "启动 3/6: GPT-4o-mini + BUDGET_MAX"
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
    --cascade-prune-mode BUDGET_MAX \
    --cascade-mad-multiplier 3.0 \
    --cascade-max-context-tokens 2000 \
    --cascade-lambda-mmr 0.6 \
    --output-dir benchmark_longmemeval/task_eval/results/triple_fusion_router_cascade_mad_v3/gpt4o_mini_budget \
    > nohup_output/longmem_mad3_gpt4o_budget.log 2>&1 &

# ============================================================================
# GPT-4.1-mini 配置 (适度激进策略)
# ============================================================================

echo "启动 4/6: GPT-4.1-mini + STRICT_THRESHOLD"
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
    --cascade-prune-mode STRICT_THRESHOLD \
    --cascade-mad-multiplier 3.0 \
    --cascade-absolute-min-score -8.0 \
    --cascade-max-context-tokens 2000 \
    --cascade-lambda-mmr 0.6 \
    --output-dir benchmark_longmemeval/task_eval/results/triple_fusion_router_cascade_mad_v3/gpt41_mini_strict \
    > nohup_output/longmem_mad3_gpt41_strict.log 2>&1 &

echo "启动 5/6: GPT-4.1-mini + CLIFF_EARLY_STOP"
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
    --cascade-prune-mode CLIFF_EARLY_STOP \
    --cascade-mad-multiplier 3.0 \
    --cascade-cliff-tolerance 2.5 \
    --cascade-max-context-tokens 2000 \
    --cascade-lambda-mmr 0.6 \
    --output-dir benchmark_longmemeval/task_eval/results/triple_fusion_router_cascade_mad_v3/gpt41_mini_cliff \
    > nohup_output/longmem_mad3_gpt41_cliff.log 2>&1 &

echo "启动 6/6: GPT-4.1-mini + BUDGET_MAX"
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
    --cascade-prune-mode BUDGET_MAX \
    --cascade-mad-multiplier 3.0 \
    --cascade-max-context-tokens 1800 \
    --cascade-lambda-mmr 0.6 \
    --output-dir benchmark_longmemeval/task_eval/results/triple_fusion_router_cascade_mad_v3/gpt41_mini_budget \
    > nohup_output/longmem_mad3_gpt41_budget.log 2>&1 &

echo "============================================================================"
echo "✅ 6 个任务已全部在后台启动！"
echo "可以使用 'jobs' 查看后台进程，或 'tail -f nohup_output/longmem_mad3_*.log' 查看日志"
echo "============================================================================"

# ============================================================================
# 【第三组】 DYNAMIC_ADAPTIVE v4 模式 (预算基准修复 + Stage 2 精细化)
#
# v3→v4 修复:
#   Bug A: budget_fraction 基准从 effective_budget(封顶后) 改为 max_tokens(绝对)
#     qa_69 user: 564*0.80=451 双重压缩 → 2000*0.80=1600 正确
#   Bug B: EVIDENCE_CHAIN max_s2_drops=2 移除(4.1 net-5, 4o net-2)
#     保留 MODERATE_CUT max_s2=5 (qa_403 有效, 0 hurt)
#
# Stage 3 (budget_fraction 基于 max_tokens):
#   multi/temporal      → EVIDENCE_CHAIN: frac=1.0, s2=无限
#   knowledge/assistant → MODERATE_CUT:   frac=0.90, s2 cap=5
#   user/preference     → AGGRESSIVE_CUT: frac=0.80, s2=无限
#
# v3 实验结果 (改善但 4.1 未达标):
#   v3 4.1-mini: acc=0.880(440/500) (-1.4pp vs Router), tok=1068
#   v3 4o-mini:  acc=0.850(425/500) (≥SOTA 0.844 ✓), tok=957
#
# BUDGET 对标:
#   4.1-mini BUDGET: 0.882/1752tok | 4o-mini BUDGET: 0.852/1598tok
# ============================================================================

echo "启动 7/8: GPT-4.1-mini + DYNAMIC_ADAPTIVE v4"
# 设计依据 (v4 budget_frac 基于 max_tokens + Stage 2 精细化):
#   T_max=2000, cliff_tol=2.5 (保留但不参与 DA 早停)
#   EVIDENCE_CHAIN (frac=1.0, s2=无限): ≈BUDGET, 恼复 v3 s2-cap 损失的6q
#   MODERATE_CUT (frac=0.90, s2 cap=5): knowledge/assistant (qa_403 有效)
#   AGGRESSIVE_CUT (frac=0.80, s2=无限):
#     budget_frac_limit=2000*0.80=1600 (v3 bug: effective*0.80双重压缩)
#   v3 per-cat: multi 0.759(-7), temporal 0.865(0), know 0.923(+2),
#               asst 0.982(-1), user 0.971(-1), pref 0.967(0)
#   预期: acc≈0.886-0.892, tok≈1100-1400
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
    --cascade-prune-mode DYNAMIC_ADAPTIVE \
    --cascade-adaptive-dataset longmemeval \
    --cascade-mad-multiplier 3.0 \
    --cascade-cliff-tolerance 2.5 \
    --cascade-max-context-tokens 2000 \
    --cascade-lambda-mmr 0.6 \
    --output-dir benchmark_longmemeval/task_eval/results/triple_fusion_router_cascade_mad_v3/gpt41_mini_dynamic_adaptive \
    > nohup_output/longmem_mad3_gpt41_dynamic_adaptive.log 2>&1 &

echo "启动 8/8: GPT-4o-mini + DYNAMIC_ADAPTIVE v4"
# 设计依据 (v4 budget_frac 基于 max_tokens + Stage 2 精细化):
#   T_max=2200, cliff_tol=3.0 (保留但不参与 DA 早停)
#   EVIDENCE_CHAIN (frac=1.0, s2=无限): 恼复 v3 s2-cap 损失的3q(4o)
#   MODERATE_CUT (frac=0.90, s2 cap=5): 保留
#   AGGRESSIVE_CUT (frac=0.80, s2=无限): budget_frac_limit=2200*0.80=1760
#   v3 per-cat: multi 0.767(+3), temporal 0.827(-5), know 0.936(-2),
#               asst 0.982(+1), user 0.986(0), pref 0.967(+1)
#   预期: acc≈0.852-0.858, tok≈960-1200
#     → v2 知识更新: 0.872(-4q vs Router), qa_375/383 frac=0.90 过低→保持
#   single-session-user/pref (AGGRESSIVE_CUT, frac=0.80, no s2 cap):
#     → v2: 持平Router; frac↑0.55→0.80 减少偶发丢失
#   预期: acc≈0.848-0.858 (≥SOTA 0.844), tok≈1000-1300
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
    --cascade-prune-mode DYNAMIC_ADAPTIVE \
    --cascade-adaptive-dataset longmemeval \
    --cascade-mad-multiplier 3.0 \
    --cascade-cliff-tolerance 3.0 \
    --cascade-max-context-tokens 2200 \
    --cascade-lambda-mmr 0.6 \
    --output-dir benchmark_longmemeval/task_eval/results/triple_fusion_router_cascade_mad_v3/gpt4o_mini_dynamic_adaptive \
    > nohup_output/longmem_mad3_gpt4o_dynamic_adaptive.log 2>&1 &

echo "============================================================================"
echo "✅ 8 个任务已全部在后台启动 (6 原有 + 2 DYNAMIC_ADAPTIVE)！"
echo "可以使用 'jobs' 查看后台进程，或 'tail -f nohup_output/longmem_mad3_*.log' 查看日志"
echo "============================================================================"

# # 确保日志输出目录存在
# mkdir -p nohup_output

# BASE_OUTPUT="benchmark_longmemeval/task_eval/results/triple_fusion_router_cascade_mad_v3"
# SCRIPT="benchmark_longmemeval/task_eval/benchmark_triple_router_quantification.py"

# echo "🚀 开始启动所有 6 个 MAD v6 benchmark 任务..."

# # ============================================================================
# # GPT-4o-mini 配置 (Router 基线: 0.850 | 1768 tok)
# # 策略: 保守, 最大限度保护精度
# # ============================================================================

# echo "启动 1/6: GPT-4o-mini + STRICT_THRESHOLD (min_score=-9.0)"
# nohup python3 $SCRIPT \
#     --llm-model gpt-4o-mini-closeai \
#     --llm-evaluate-model gpt-4o-mini-closeai \
#     --sentence-top-k 60 \
#     --episodic-top-k 40 \
#     --entity-top-k 40 \
#     --final-top-k 25 \
#     --rerank-method baai \
#     --fusion-method concatenation \
#     --enable-router \
#     --router-strategy aggressive \
#     --enable-cascade-pruner \
#     --cascade-prune-mode STRICT_THRESHOLD \
#     --cascade-mad-multiplier 3.0 \
#     --cascade-absolute-min-score -9.0 \
#     --cascade-max-context-tokens 2200 \
#     --cascade-lambda-mmr 0.6 \
#     --output-dir ${BASE_OUTPUT}/gpt4o_mini_strict \
#     > nohup_output/longmem_mad3_gpt4o_strict.log 2>&1 &

# echo "启动 2/6: GPT-4o-mini + CLIFF_EARLY_STOP (cliff_tol=3.0)"
# nohup python3 $SCRIPT \
#     --llm-model gpt-4o-mini-closeai \
#     --llm-evaluate-model gpt-4o-mini-closeai \
#     --sentence-top-k 60 \
#     --episodic-top-k 40 \
#     --entity-top-k 40 \
#     --final-top-k 25 \
#     --rerank-method baai \
#     --fusion-method concatenation \
#     --enable-router \
#     --router-strategy aggressive \
#     --enable-cascade-pruner \
#     --cascade-prune-mode CLIFF_EARLY_STOP \
#     --cascade-mad-multiplier 3.0 \
#     --cascade-cliff-tolerance 3.0 \
#     --cascade-max-context-tokens 2200 \
#     --cascade-lambda-mmr 0.6 \
#     --output-dir ${BASE_OUTPUT}/gpt4o_mini_cliff \
#     > nohup_output/longmem_mad3_gpt4o_cliff.log 2>&1 &

# echo "启动 3/6: GPT-4o-mini + BUDGET_MAX (T=2000)"
# nohup python3 $SCRIPT \
#     --llm-model gpt-4o-mini-closeai \
#     --llm-evaluate-model gpt-4o-mini-closeai \
#     --sentence-top-k 60 \
#     --episodic-top-k 40 \
#     --entity-top-k 40 \
#     --final-top-k 25 \
#     --rerank-method baai \
#     --fusion-method concatenation \
#     --enable-router \
#     --router-strategy aggressive \
#     --enable-cascade-pruner \
#     --cascade-prune-mode BUDGET_MAX \
#     --cascade-mad-multiplier 3.0 \
#     --cascade-max-context-tokens 2000 \
#     --cascade-lambda-mmr 0.6 \
#     --output-dir ${BASE_OUTPUT}/gpt4o_mini_budget \
#     > nohup_output/longmem_mad3_gpt4o_budget.log 2>&1 &

# # ============================================================================
# # GPT-4.1-mini 配置 (Router 基线: 0.894 | 1923 tok)
# # 策略: 适度激进, 更大精度余量
# # ============================================================================

# echo "启动 4/6: GPT-4.1-mini + STRICT_THRESHOLD (min_score=-8.0)"
# nohup python3 $SCRIPT \
#     --llm-model gpt-4.1-mini-closeai \
#     --llm-evaluate-model gpt-4o-mini-closeai \
#     --sentence-top-k 60 \
#     --episodic-top-k 40 \
#     --entity-top-k 40 \
#     --final-top-k 25 \
#     --rerank-method baai \
#     --fusion-method concatenation \
#     --enable-router \
#     --router-strategy aggressive \
#     --enable-cascade-pruner \
#     --cascade-prune-mode STRICT_THRESHOLD \
#     --cascade-mad-multiplier 3.0 \
#     --cascade-absolute-min-score -8.0 \
#     --cascade-max-context-tokens 2000 \
#     --cascade-lambda-mmr 0.6 \
#     --output-dir ${BASE_OUTPUT}/gpt41_mini_strict \
#     > nohup_output/longmem_mad3_gpt41_strict.log 2>&1 &

# echo "启动 5/6: GPT-4.1-mini + CLIFF_EARLY_STOP (cliff_tol=2.5)"
# nohup python3 $SCRIPT \
#     --llm-model gpt-4.1-mini-closeai \
#     --llm-evaluate-model gpt-4o-mini-closeai \
#     --sentence-top-k 60 \
#     --episodic-top-k 40 \
#     --entity-top-k 40 \
#     --final-top-k 25 \
#     --rerank-method baai \
#     --fusion-method concatenation \
#     --enable-router \
#     --router-strategy aggressive \
#     --enable-cascade-pruner \
#     --cascade-prune-mode CLIFF_EARLY_STOP \
#     --cascade-mad-multiplier 3.0 \
#     --cascade-cliff-tolerance 2.5 \
#     --cascade-max-context-tokens 2000 \
#     --cascade-lambda-mmr 0.6 \
#     --output-dir ${BASE_OUTPUT}/gpt41_mini_cliff \
#     > nohup_output/longmem_mad3_gpt41_cliff.log 2>&1 &

# echo "启动 6/6: GPT-4.1-mini + BUDGET_MAX (T=1800)"
# nohup python3 $SCRIPT \
#     --llm-model gpt-4.1-mini-closeai \
#     --llm-evaluate-model gpt-4o-mini-closeai \
#     --sentence-top-k 60 \
#     --episodic-top-k 40 \
#     --entity-top-k 40 \
#     --final-top-k 25 \
#     --rerank-method baai \
#     --fusion-method concatenation \
#     --enable-router \
#     --router-strategy aggressive \
#     --enable-cascade-pruner \
#     --cascade-prune-mode BUDGET_MAX \
#     --cascade-mad-multiplier 3.0 \
#     --cascade-max-context-tokens 1800 \
#     --cascade-lambda-mmr 0.6 \
#     --output-dir ${BASE_OUTPUT}/gpt41_mini_budget \
#     > nohup_output/longmem_mad3_gpt41_budget.log 2>&1 &

# echo "============================================================================"
# echo "✅ 6 个 MAD v6 任务已全部在后台启动！"
# echo "============================================================================"
# echo ""
# echo "📋 v5.1→v6 关键改动:"
# echo "  ✓ 移除 min-max 归一化, 直接在 raw logits 上做 MAD"
# echo "  ✓ Stage1 ON (k=3.0): ~99% SOTA survival, 仅移除极端噪声"
# echo "  ✓ 新增 STRICT_THRESHOLD + CLIFF_EARLY_STOP 两种模式"
# echo "  ✓ 废弃 tau_high/tau_med/target_confidence 旧参数"
# echo ""
# echo "📋 任务列表:"
# echo "  1. GPT-4o-mini  + STRICT (-9.0)   → ${BASE_OUTPUT}/gpt4o_mini_strict"
# echo "  2. GPT-4o-mini  + CLIFF  (3.0)    → ${BASE_OUTPUT}/gpt4o_mini_cliff"
# echo "  3. GPT-4o-mini  + BUDGET (T=2000)  → ${BASE_OUTPUT}/gpt4o_mini_budget"
# echo "  4. GPT-4.1-mini + STRICT (-8.0)   → ${BASE_OUTPUT}/gpt41_mini_strict"
# echo "  5. GPT-4.1-mini + CLIFF  (2.5)    → ${BASE_OUTPUT}/gpt41_mini_cliff"
# echo "  6. GPT-4.1-mini + BUDGET (T=1800)  → ${BASE_OUTPUT}/gpt41_mini_budget"
# echo ""
# echo "📊 基线:"
# echo "  Router 4o-mini:  LLM_Acc=0.850, 1768 tok"
# echo "  Router 4.1-mini: LLM_Acc=0.894, 1923 tok"
# echo ""
# echo "可以使用 'jobs' 查看后台进程，或 'tail -f nohup_output/longmem_mad3_*.log' 查看日志"

# ============================================================================
# 🔬 消融实验 (Ablation Study for Figure 6)
# ============================================================================
# 基线: DYNAMIC_ADAPTIVE (上方 7/8 号实验)
# 消融 A: 关闭路由 (No Router) — 保留级联 DYNAMIC_ADAPTIVE
# 消融 B: 关闭 Stage1/2 去重 (No Dedup) — 保留路由 + DYNAMIC_ADAPTIVE 装箱
# 消融 C: 关闭 Stage3 MMR 装箱 (No Packing) — 保留路由 + Stage1/2, BUDGET_MAX 贪心
# ============================================================================

# ── 消融 A: 关闭路由 ──────────────────────────────────────────────────────────

# A1. GPT-4.1-mini: 无路由, DYNAMIC_ADAPTIVE (T_max=2000, cliff=2.5)
echo "Starting: GPT-4.1-mini Ablation No-Router..."
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
    --cascade-prune-mode DYNAMIC_ADAPTIVE \
    --cascade-adaptive-dataset longmemeval \
    --cascade-mad-multiplier 3.0 \
    --cascade-cliff-tolerance 2.5 \
    --cascade-max-context-tokens 2000 \
    --cascade-lambda-mmr 0.6 \
    --output-dir benchmark_longmemeval/task_eval/results/triple_fusion_router_cascade_mad_v3/gpt41_mini_ablation_no_router \
    > nohup_output/longmem_mad3_gpt41_ablation_no_router.log 2>&1 &

# A2. GPT-4o-mini: 无路由, DYNAMIC_ADAPTIVE (T_max=2200, cliff=3.0)
echo "Starting: GPT-4o-mini Ablation No-Router..."
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
    --cascade-prune-mode DYNAMIC_ADAPTIVE \
    --cascade-adaptive-dataset longmemeval \
    --cascade-mad-multiplier 3.0 \
    --cascade-cliff-tolerance 3.0 \
    --cascade-max-context-tokens 2200 \
    --cascade-lambda-mmr 0.6 \
    --output-dir benchmark_longmemeval/task_eval/results/triple_fusion_router_cascade_mad_v3/gpt4o_mini_ablation_no_router \
    > nohup_output/longmem_mad3_gpt4o_ablation_no_router.log 2>&1 &

# ── 消融 B: 关闭 Stage1/2 去重 ──────────────────────────────────────────────

# B1. GPT-4.1-mini: 路由+级联, 关闭 Stage1+2, DYNAMIC_ADAPTIVE 装箱 (T_max=2000, cliff=2.5)
echo "Starting: GPT-4.1-mini Ablation No-Stage12..."
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
    --cascade-prune-mode DYNAMIC_ADAPTIVE \
    --cascade-adaptive-dataset longmemeval \
    --cascade-mad-multiplier 3.0 \
    --cascade-cliff-tolerance 2.5 \
    --cascade-max-context-tokens 2000 \
    --cascade-lambda-mmr 0.6 \
    --no-cascade-stage1 \
    --no-cascade-stage2 \
    --output-dir benchmark_longmemeval/task_eval/results/triple_fusion_router_cascade_mad_v3/gpt41_mini_ablation_no_stage12 \
    > nohup_output/longmem_mad3_gpt41_ablation_no_stage12.log 2>&1 &

# B2. GPT-4o-mini: 路由+级联, 关闭 Stage1+2, DYNAMIC_ADAPTIVE 装箱 (T_max=2200, cliff=3.0)
echo "Starting: GPT-4o-mini Ablation No-Stage12..."
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
    --cascade-prune-mode DYNAMIC_ADAPTIVE \
    --cascade-adaptive-dataset longmemeval \
    --cascade-mad-multiplier 3.0 \
    --cascade-cliff-tolerance 3.0 \
    --cascade-max-context-tokens 2200 \
    --cascade-lambda-mmr 0.6 \
    --no-cascade-stage1 \
    --no-cascade-stage2 \
    --output-dir benchmark_longmemeval/task_eval/results/triple_fusion_router_cascade_mad_v3/gpt4o_mini_ablation_no_stage12 \
    > nohup_output/longmem_mad3_gpt4o_ablation_no_stage12.log 2>&1 &

# ── 消融 C: 关闭 Stage3 MMR 装箱 ────────────────────────────────────────────

# C1. GPT-4.1-mini: 路由+Stage1/2, 关闭 MMR 装箱 (退化为贪心), BUDGET_MAX (T_max=2000)
echo "Starting: GPT-4.1-mini Ablation No-Packing..."
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
    --cascade-prune-mode BUDGET_MAX \
    --cascade-mad-multiplier 3.0 \
    --cascade-cliff-tolerance 2.5 \
    --cascade-max-context-tokens 2000 \
    --cascade-lambda-mmr 0.6 \
    --no-cascade-stage3-mmr \
    --output-dir benchmark_longmemeval/task_eval/results/triple_fusion_router_cascade_mad_v3/gpt41_mini_ablation_no_packing \
    > nohup_output/longmem_mad3_gpt41_ablation_no_packing.log 2>&1 &

# C2. GPT-4o-mini: 路由+Stage1/2, 关闭 MMR 装箱 (退化为贪心), BUDGET_MAX (T_max=2200)
echo "Starting: GPT-4o-mini Ablation No-Packing..."
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
    --cascade-prune-mode BUDGET_MAX \
    --cascade-mad-multiplier 3.0 \
    --cascade-cliff-tolerance 3.0 \
    --cascade-max-context-tokens 2200 \
    --cascade-lambda-mmr 0.6 \
    --no-cascade-stage3-mmr \
    --output-dir benchmark_longmemeval/task_eval/results/triple_fusion_router_cascade_mad_v3/gpt4o_mini_ablation_no_packing \
    > nohup_output/longmem_mad3_gpt4o_ablation_no_packing.log 2>&1 &

echo ""
echo "🔬 消融实验已启动 (6 个任务):"
echo "  A1. GPT-4.1-mini + No-Router    → .../gpt41_mini_ablation_no_router"
echo "  A2. GPT-4o-mini  + No-Router    → .../gpt4o_mini_ablation_no_router"
echo "  B1. GPT-4.1-mini + No-Stage12   → .../gpt41_mini_ablation_no_stage12"
echo "  B2. GPT-4o-mini  + No-Stage12   → .../gpt4o_mini_ablation_no_stage12"
echo "  C1. GPT-4.1-mini + No-Packing   → .../gpt41_mini_ablation_no_packing"
echo "  C2. GPT-4o-mini  + No-Packing   → .../gpt4o_mini_ablation_no_packing"
# echo "============================================================================"
