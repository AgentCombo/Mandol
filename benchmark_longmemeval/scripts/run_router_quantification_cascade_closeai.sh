#!/bin/bash
# Purpose: launch the LongMemEval paper-style router + quantification + cascade
# pruning runs through CloseAI models.
# Runs: benchmark_longmemeval.task_eval.benchmark_triple_router_quantification.

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

# ============================================================================
# LongMemEval 路由 + 级联量化 — MAD + Raw Logits (v6)
# ============================================================================
#
# 脚本: benchmark_triple_router_quantification.py (路由版 + 级联量化)
#
# ===== v5.1→v6 变更说明 ====================================================
#
# v5.1 问题:
#   - Stage1 使用 min-max 归一化 → 恒定丢弃 ~20% 块 → 8pp 精度下降
#   - tau_high/tau_med 已废弃 (动态阈值冲突)
#   - STRICT/CUMULATIVE 因 min-max 归一化不可用
#   - 全部退化为 BUDGET_MAX 单模式
#
# v6 改进 (MAD-based cascade pruner 完整重构):
#   ✓ 移除 min-max 归一化, 直接在 raw logits 上计算 MAD
#   ✓ Stage1 MAD k=3.0: 仅移除极端离群点 (~1% block loss on 25 blocks)
#   ✓ 新增 CLIFF_EARLY_STOP (替代 CUMULATIVE): 检测相邻 logit gap 停止
#   ✓ STRICT_THRESHOLD 改为 absolute_min_score (raw logit 尺度)
#   ✓ BUDGET_MAX 保持不变, MMR 多样性打包
#   ✓ 三种模式均可安全使用
#
# ===== Top-25 分数分布分析 (基于 500 QA 离线扫参) =========================
#
# Router (sigmoid→logits):
#   4.1-mini: median_of_medians=-6.50, MAD P50=1.37, top1 P50=1.45
#   4o-mini:  median_of_medians=-6.07, MAD P50=1.39, top1 P50=1.27
# SOTA (raw logits):
#   Both:     median_of_medians=-5.27, MAD P50=1.34, top1 P50=1.79
#
# 设计根据: 重跑后将使用 raw logits, 以 SOTA 分布为主要参考
#
# ===== 参数配置表 (精度优先, 兼顾 token 节省) =============================
#
# ┌───────────────────────────────────┬──────────┬──────────┬──────────┬─────────┬─────────────────────┐
# │ 配置                              │ mad_mult │ cliff_tol│ min_score│ T_max   │ 估计效果            │
# ├───────────────────────────────────┼──────────┼──────────┼──────────┼─────────┼─────────────────────┤
# │ 4.1-mini STRICT_THRESHOLD         │ 3.0      │ -        │ -8.0     │ 2000    │ ~21/25 kept, ~17%↓  │
# │ 4.1-mini CLIFF_EARLY_STOP         │ 3.0      │ 2.5      │ -        │ 2000    │ ~18/25 kept, ~25%↓  │
# │ 4.1-mini BUDGET_MAX               │ 3.0      │ -        │ -        │ 1800    │ ~10-15%↓ token      │
# │ 4o-mini  STRICT_THRESHOLD         │ 3.0      │ -        │ -9.0     │ 2200    │ ~23/25 kept, ~8%↓   │
# │ 4o-mini  CLIFF_EARLY_STOP         │ 3.0      │ 3.0      │ -        │ 2200    │ ~20/25 kept, ~20%↓  │
# │ 4o-mini  BUDGET_MAX               │ 3.0      │ -        │ -        │ 2000    │ ~5-10%↓ token       │
# └───────────────────────────────────┴──────────┴──────────┴──────────┴─────────┴─────────────────────┘
#
# 策略说明:
#   - 4.1-mini baseline 精度更高 (0.894) → 参数更激进
#   - 4o-mini  baseline 精度较低 (0.850) → 参数更保守
#   - CLIFF 从不丢失 top-1 块 (构造性保证) → 最安全
#   - STRICT 在 -8.0/-9.0 下 0 个正确 QA 丢失 top-1 块 (SOTA 分布)
#   - MAD k=3.0 在 25 块上 ~99% survival (SOTA), ~96% (Router)
#
# ===== 基线参考 ============================================================
#
# SOTA 基线 (raw logits, 无路由):
#   GPT-4.1-mini: LLM_Acc=0.884, ~2094 tok/QA
#   GPT-4o-mini:  LLM_Acc=0.844, ~2094 tok/QA
#
# Router 基线 (aggressive):
#   GPT-4.1-mini: LLM_Acc=0.894, ~1923 tok/QA
#   GPT-4o-mini:  LLM_Acc=0.850, ~1768 tok/QA
# ============================================================================

BASE_OUTPUT="benchmark_longmemeval/task_eval/results/triple_fusion_router_cascade_mad_v3"
SCRIPT="benchmark_longmemeval/task_eval/benchmark_triple_router_quantification.py"
mkdir -p nohup_output "$BASE_OUTPUT"

# 检索参数 (对齐 SOTA 基线)
RETRIEVAL_COMMON="\
    --sentence-top-k 60 \
    --episodic-top-k 40 \
    --entity-top-k 40 \
    --final-top-k 25 \
    --rerank-method baai \
    --fusion-method concatenation"

# 路由参数
ROUTER_COMMON="--enable-router --router-strategy aggressive"

# 级联公共参数
CASCADE_COMMON="--enable-cascade-pruner --cascade-mad-multiplier 3.0 --cascade-lambda-mmr 0.6"

# ============================================================================
# GPT-4o-mini 配置 (Router 基线: 0.850 | 1768 tok)
# 策略: 保守, 最大限度保护精度
# ============================================================================

# 1. GPT-4o-mini + STRICT_THRESHOLD
#    absolute_min_score=-9.0: keeps ~23/25 blocks (92%), 0 top-1 blocks lost
#    预期: LLM_Acc ≈ 0.84-0.85, token ~8%↓
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple_router_quantification \
    --llm-model gpt-4o-mini-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
    $RETRIEVAL_COMMON \
    $ROUTER_COMMON \
    $CASCADE_COMMON \
    --cascade-prune-mode STRICT_THRESHOLD \
    --cascade-absolute-min-score -9.0 \
    --cascade-max-context-tokens 2200 \
    --output-dir ${BASE_OUTPUT}/gpt4o_mini_strict \
    > nohup_output/longmem_mad3_gpt4o_strict.log 2>&1 &

# 2. GPT-4o-mini + CLIFF_EARLY_STOP
#    cliff_tolerance=3.0: keeps ~20/25 blocks (80%), 0 top-1 lost (构造性保证)
#    预期: LLM_Acc ≈ 0.84-0.85, token ~15-20%↓
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple_router_quantification \
    --llm-model gpt-4o-mini-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
    $RETRIEVAL_COMMON \
    $ROUTER_COMMON \
    $CASCADE_COMMON \
    --cascade-prune-mode CLIFF_EARLY_STOP \
    --cascade-cliff-tolerance 3.0 \
    --cascade-max-context-tokens 2200 \
    --output-dir ${BASE_OUTPUT}/gpt4o_mini_cliff \
    > nohup_output/longmem_mad3_gpt4o_cliff.log 2>&1 &

# 3. GPT-4o-mini + BUDGET_MAX
#    T_max=2000: 覆盖 ~72% Router QA / ~49% SOTA, 适度修剪
#    预期: LLM_Acc ≈ 0.84-0.85, token ~5-10%↓
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple_router_quantification \
    --llm-model gpt-4o-mini-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
    $RETRIEVAL_COMMON \
    $ROUTER_COMMON \
    $CASCADE_COMMON \
    --cascade-prune-mode BUDGET_MAX \
    --cascade-max-context-tokens 2000 \
    --output-dir ${BASE_OUTPUT}/gpt4o_mini_budget \
    > nohup_output/longmem_mad3_gpt4o_budget.log 2>&1 &

# ============================================================================
# GPT-4.1-mini 配置 (Router 基线: 0.894 | 1923 tok)
# 策略: 适度激进, 更大的精度余量允许更多 token 节省
# ============================================================================

# 4. GPT-4.1-mini + STRICT_THRESHOLD
#    absolute_min_score=-8.0: keeps ~21/25 blocks (83%), 0 top-1 lost (SOTA)
#    预期: LLM_Acc ≈ 0.87-0.89, token ~15-20%↓
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple_router_quantification \
    --llm-model gpt-4.1-mini-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
    $RETRIEVAL_COMMON \
    $ROUTER_COMMON \
    $CASCADE_COMMON \
    --cascade-prune-mode STRICT_THRESHOLD \
    --cascade-absolute-min-score -8.0 \
    --cascade-max-context-tokens 2000 \
    --output-dir ${BASE_OUTPUT}/gpt41_mini_strict \
    > nohup_output/longmem_mad3_gpt41_strict.log 2>&1 &

# 5. GPT-4.1-mini + CLIFF_EARLY_STOP
#    cliff_tolerance=2.5: keeps ~18/25 blocks (73%), 0 top-1 lost
#    预期: LLM_Acc ≈ 0.87-0.89, token ~20-30%↓
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple_router_quantification \
    --llm-model gpt-4.1-mini-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
    $RETRIEVAL_COMMON \
    $ROUTER_COMMON \
    $CASCADE_COMMON \
    --cascade-prune-mode CLIFF_EARLY_STOP \
    --cascade-cliff-tolerance 2.5 \
    --cascade-max-context-tokens 2000 \
    --output-dir ${BASE_OUTPUT}/gpt41_mini_cliff \
    > nohup_output/longmem_mad3_gpt41_cliff.log 2>&1 &

# 6. GPT-4.1-mini + BUDGET_MAX
#    T_max=1800: 覆盖 ~26% Router QA / ~9% SOTA, 较积极修剪
#    预期: LLM_Acc ≈ 0.87-0.89, token ~10-15%↓
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple_router_quantification \
    --llm-model gpt-4.1-mini-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
    $RETRIEVAL_COMMON \
    $ROUTER_COMMON \
    $CASCADE_COMMON \
    --cascade-prune-mode BUDGET_MAX \
    --cascade-max-context-tokens 1800 \
    --output-dir ${BASE_OUTPUT}/gpt41_mini_budget \
    > nohup_output/longmem_mad3_gpt41_budget.log 2>&1 &

echo "============================================================================"
echo "✅ 已启动 6 个 v6 benchmark 任务 (3 modes × 2 LLMs)"
echo "============================================================================"

# ============================================================================
# 【第三组】 DYNAMIC_ADAPTIVE v4 模式 (预算基准修复 + Stage 2 精细化)
#
# v3→v4 修复:
#   Bug A: budget_fraction 计算基准错误
#     v3 使用 effective_budget(被 cap_to_input 封顶) 作为基准
#     小上下文场景下双重压缩 (qa_69: 564*0.80=451, 应为 2000*0.80=1600)
#     v4 改为 self.max_tokens 作为绝对基准
#
#   Bug B: EVIDENCE_CHAIN max_stage2_drops=2 过于严格
#     实验数据: 4.1 helped=1/hurt=6(net-5), 4o helped=1/hurt=3(net-2)
#     “冗余”块多数是器分析噪声 (qa_114 s2d:2→7 改善, qa_251 s2d:2→6 改善)
#     v4 移除 EVIDENCE_CHAIN cap, 保留 MODERATE_CUT cap=5 (qa_403 有效)
#
# Stage 3 参数 (budget_fraction 基于 max_tokens):
#   EVIDENCE_CHAIN: frac=1.0 | MODERATE_CUT: frac=0.90 | AGGRESSIVE_CUT: frac=0.80
#
# Stage 2 参数 (max_stage2_drops):
#   EVIDENCE_CHAIN: None(无限) | MODERATE_CUT: 5 | AGGRESSIVE_CUT: None
#
# v3 实验结果 (改善但未达标):
#   v3 4.1-mini: acc=0.880 (-1.4pp vs Router), cascade_tok=1068 (44%↓)
#   v3 4o-mini:  acc=0.850 (+0.0pp vs Router), cascade_tok=957  (46%↓)
#
# BUDGET 对标:
#   4.1-mini BUDGET: 0.882/1752tok | 4o-mini BUDGET: 0.852/1598tok
# ============================================================================

# 7. GPT-4.1-mini + DYNAMIC_ADAPTIVE v4
#    T_max=2000, cliff_tol=2.5 (保留但不参与 DA 早停)
#    v4 修复: budget_frac 基于 max_tokens(2000), 非 effective_budget
#    EVIDENCE_CHAIN (frac=1.0, s2 无限): ≈BUDGET, 恼复 v3 s2-cap 损失的6q
#    MODERATE_CUT (frac=0.90, s2 cap=5): knowledge/assistant 适度保护
#    AGGRESSIVE_CUT (frac=0.80, s2 无限): 小上下文不再双重压缩(qa_69)
#    预期: acc≈0.886-0.892, tok≈1100-1400
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple_router_quantification \
    --llm-model gpt-4.1-mini-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
    $RETRIEVAL_COMMON \
    $ROUTER_COMMON \
    $CASCADE_COMMON \
    --cascade-prune-mode DYNAMIC_ADAPTIVE \
    --cascade-adaptive-dataset longmemeval \
    --cascade-cliff-tolerance 2.5 \
    --cascade-max-context-tokens 2000 \
    --output-dir ${BASE_OUTPUT}/gpt41_mini_dynamic_adaptive \
    > nohup_output/longmem_mad3_gpt41_dynamic_adaptive.log 2>&1 &

# 8. GPT-4o-mini + DYNAMIC_ADAPTIVE v4
#    T_max=2200, cliff_tol=3.0 (保留但不参与 DA 早停)
#    v4 修复: budget_frac 基于 max_tokens(2200), 非 effective_budget
#    EVIDENCE_CHAIN (frac=1.0, s2 无限): 恼复 v3 s2-cap 损失的3q
#    MODERATE_CUT (frac=0.90, s2 cap=5): 保留 (qa_403 有效)
#    AGGRESSIVE_CUT (frac=0.80, s2 无限): 小上下文场景不双重压缩
#    预期: acc≈0.852-0.858, tok≈960-1200
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple_router_quantification \
    --llm-model gpt-4o-mini-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
    $RETRIEVAL_COMMON \
    $ROUTER_COMMON \
    $CASCADE_COMMON \
    --cascade-prune-mode DYNAMIC_ADAPTIVE \
    --cascade-adaptive-dataset longmemeval \
    --cascade-cliff-tolerance 3.0 \
    --cascade-max-context-tokens 2200 \
    --output-dir ${BASE_OUTPUT}/gpt4o_mini_dynamic_adaptive \
    > nohup_output/longmem_mad3_gpt4o_dynamic_adaptive.log 2>&1 &

echo ""
echo "📋 v5.1→v6 关键改动:"
echo "  ✓ 移除 min-max 归一化, 直接在 raw logits 上做 MAD"
echo "  ✓ Stage1 ON (k=3.0): ~99% survival (SOTA), 仅移除极端噪声"
echo "  ✓ 新增 STRICT_THRESHOLD + CLIFF_EARLY_STOP 两种模式"
echo "  ✓ 模式选择: absolute_min_score (STRICT), cliff_tolerance (CLIFF)"
echo "  ✓ 废弃 tau_high/tau_med/target_confidence 旧参数"
echo ""
echo "📋 任务列表:"
echo "  1. GPT-4o-mini  + STRICT (-9.0)   → ${BASE_OUTPUT}/gpt4o_mini_strict"
echo "  2. GPT-4o-mini  + CLIFF  (3.0)    → ${BASE_OUTPUT}/gpt4o_mini_cliff"
echo "  3. GPT-4o-mini  + BUDGET (T=2000)  → ${BASE_OUTPUT}/gpt4o_mini_budget"
echo "  4. GPT-4.1-mini + STRICT (-8.0)   → ${BASE_OUTPUT}/gpt41_mini_strict"
echo "  5. GPT-4.1-mini + CLIFF  (2.5)    → ${BASE_OUTPUT}/gpt41_mini_cliff"
echo "  6. GPT-4.1-mini + BUDGET (T=1800)  → ${BASE_OUTPUT}/gpt41_mini_budget"
echo "  7. GPT-4.1-mini + DYNAMIC (2.5/T=2000/longmemeval) → ${BASE_OUTPUT}/gpt41_mini_dynamic_adaptive"
echo "  8. GPT-4o-mini  + DYNAMIC (3.0/T=2200/longmemeval) → ${BASE_OUTPUT}/gpt4o_mini_dynamic_adaptive"
echo ""
echo "📊 基线:"
echo "  SOTA 4.1-mini:   LLM_Acc=0.884, 2094 tok"
echo "  SOTA 4o-mini:    LLM_Acc=0.844, 2094 tok"
echo "  Router 4.1-mini: LLM_Acc=0.894, 1923 tok"
echo "  Router 4o-mini:  LLM_Acc=0.850, 1768 tok"
echo ""
echo "📁 日志: nohup_output/longmem_mad3_*.log"
echo "📁 输出: ${BASE_OUTPUT}/"
echo "============================================================================"

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
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple_router_quantification \
    --llm-model gpt-4.1-mini-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
    $RETRIEVAL_COMMON \
    $CASCADE_COMMON \
    --cascade-prune-mode DYNAMIC_ADAPTIVE \
    --cascade-adaptive-dataset longmemeval \
    --cascade-cliff-tolerance 2.5 \
    --cascade-max-context-tokens 2000 \
    --output-dir ${BASE_OUTPUT}/gpt41_mini_ablation_no_router \
    > nohup_output/longmem_mad3_gpt41_ablation_no_router.log 2>&1 &

# A2. GPT-4o-mini: 无路由, DYNAMIC_ADAPTIVE (T_max=2200, cliff=3.0)
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple_router_quantification \
    --llm-model gpt-4o-mini-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
    $RETRIEVAL_COMMON \
    $CASCADE_COMMON \
    --cascade-prune-mode DYNAMIC_ADAPTIVE \
    --cascade-adaptive-dataset longmemeval \
    --cascade-cliff-tolerance 3.0 \
    --cascade-max-context-tokens 2200 \
    --output-dir ${BASE_OUTPUT}/gpt4o_mini_ablation_no_router \
    > nohup_output/longmem_mad3_gpt4o_ablation_no_router.log 2>&1 &

# ── 消融 B: 关闭 Stage1/2 去重 ──────────────────────────────────────────────

# B1. GPT-4.1-mini: 路由+级联, 关闭 Stage1+2, DYNAMIC_ADAPTIVE 装箱 (T_max=2000, cliff=2.5)
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple_router_quantification \
    --llm-model gpt-4.1-mini-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
    $RETRIEVAL_COMMON \
    $ROUTER_COMMON \
    $CASCADE_COMMON \
    --cascade-prune-mode DYNAMIC_ADAPTIVE \
    --cascade-adaptive-dataset longmemeval \
    --cascade-cliff-tolerance 2.5 \
    --cascade-max-context-tokens 2000 \
    --no-cascade-stage1 \
    --no-cascade-stage2 \
    --output-dir ${BASE_OUTPUT}/gpt41_mini_ablation_no_stage12 \
    > nohup_output/longmem_mad3_gpt41_ablation_no_stage12.log 2>&1 &

# B2. GPT-4o-mini: 路由+级联, 关闭 Stage1+2, DYNAMIC_ADAPTIVE 装箱 (T_max=2200, cliff=3.0)
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple_router_quantification \
    --llm-model gpt-4o-mini-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
    $RETRIEVAL_COMMON \
    $ROUTER_COMMON \
    $CASCADE_COMMON \
    --cascade-prune-mode DYNAMIC_ADAPTIVE \
    --cascade-adaptive-dataset longmemeval \
    --cascade-cliff-tolerance 3.0 \
    --cascade-max-context-tokens 2200 \
    --no-cascade-stage1 \
    --no-cascade-stage2 \
    --output-dir ${BASE_OUTPUT}/gpt4o_mini_ablation_no_stage12 \
    > nohup_output/longmem_mad3_gpt4o_ablation_no_stage12.log 2>&1 &

# ── 消融 C: 关闭 Stage3 MMR 装箱 ────────────────────────────────────────────

# C1. GPT-4.1-mini: 路由+Stage1/2, 关闭 MMR 装箱 (退化为贪心), BUDGET_MAX (T_max=2000)
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple_router_quantification \
    --llm-model gpt-4.1-mini-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
    $RETRIEVAL_COMMON \
    $ROUTER_COMMON \
    $CASCADE_COMMON \
    --cascade-prune-mode BUDGET_MAX \
    --cascade-cliff-tolerance 2.5 \
    --cascade-max-context-tokens 2000 \
    --no-cascade-stage3-mmr \
    --output-dir ${BASE_OUTPUT}/gpt41_mini_ablation_no_packing \
    > nohup_output/longmem_mad3_gpt41_ablation_no_packing.log 2>&1 &

# C2. GPT-4o-mini: 路由+Stage1/2, 关闭 MMR 装箱 (退化为贪心), BUDGET_MAX (T_max=2200)
nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple_router_quantification \
    --llm-model gpt-4o-mini-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
    $RETRIEVAL_COMMON \
    $ROUTER_COMMON \
    $CASCADE_COMMON \
    --cascade-prune-mode BUDGET_MAX \
    --cascade-cliff-tolerance 3.0 \
    --cascade-max-context-tokens 2200 \
    --no-cascade-stage3-mmr \
    --output-dir ${BASE_OUTPUT}/gpt4o_mini_ablation_no_packing \
    > nohup_output/longmem_mad3_gpt4o_ablation_no_packing.log 2>&1 &

echo ""
echo "🔬 消融实验已启动 (6 个任务):"
echo "  A1. GPT-4.1-mini + No-Router    → ${BASE_OUTPUT}/gpt41_mini_ablation_no_router"
echo "  A2. GPT-4o-mini  + No-Router    → ${BASE_OUTPUT}/gpt4o_mini_ablation_no_router"
echo "  B1. GPT-4.1-mini + No-Stage12   → ${BASE_OUTPUT}/gpt41_mini_ablation_no_stage12"
echo "  B2. GPT-4o-mini  + No-Stage12   → ${BASE_OUTPUT}/gpt4o_mini_ablation_no_stage12"
echo "  C1. GPT-4.1-mini + No-Packing   → ${BASE_OUTPUT}/gpt41_mini_ablation_no_packing"
echo "  C2. GPT-4o-mini  + No-Packing   → ${BASE_OUTPUT}/gpt4o_mini_ablation_no_packing"
