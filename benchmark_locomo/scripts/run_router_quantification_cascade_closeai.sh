#!/bin/bash
# Purpose: launch the LoCoMo paper-style router + quantification + cascade
# pruning runs through CloseAI models.
# Runs: benchmark_locomo.task_eval.locomo_triple_router_quantification.

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
# LoCoMo 路由 + 级联量化 Benchmark — CloseAI 模型 (v2: MAD-based)
# ============================================================================
#
# 脚本: locomo_triple_router_quantification.py (路由版 + 级联量化)
#
# 架构流程:
#   Query → Router(category→tower组合) → 三塔/双塔并行检索
#       → CE Rerank → CascadeConfidencePruner(S1+S2+S3) → LLM生成 → 评估
#
# v2 更新: 从旧 tau_high/tau_med 切换到 MAD-based Stage1 + CLIFF/STRICT/BUDGET Stage3
#
# 与 prune_cascade_closeai.sh 的关键区别:
#   1. 引入前置路由: LocomoTowerRouter 按问题 category 动态选择塔组合
#      - GPT-4.1-mini aggressive: 路由无效 (all → H+G+E, acc=0.931)
#      - GPT-4o-mini aggressive: Cat1→H+E, Cat2→H+G, Cat5→G+E (acc=0.884, +3.7pp)
#   2. 路由后候选池因类别而异，4o-mini 尤其异构
#   3. 级联参数需要适配路由后的异构候选池
#
# ===== 数据特征 (基于 1986 QA 纯检索 + 路由模拟分析) ==========================
#
# --- GPT-4.1-mini (all → H+G+E, 等同非路由版) ---
#   候选池: mean=33.2 chunks/QA, tokens mean=2365
#   CE Score: mean=-0.883, P50=-1.000, range=[-10.312, 10.500]
#   Per-QA top-1: P50=4.031, P10=1.078, min=-4.281
#   Per-QA MAD: P50=1.031
#   Per-QA max cliff: P50=1.406, P90=2.812
#
# --- GPT-4o-mini (aggressive routing, 异构池) ---
#   候选池: mean=27.2 chunks/QA (Cat5 G+E: 17.9, Cat1 H+E: 24.2)
#   Token: mean=1884, P50=2140, P10=632 (Cat5 仅 648 tok!)
#   CE Score: mean=-0.827, P50=-0.949
#   Per-QA top-1: P50=3.688, P10=-0.453 (Cat5 top-1 很低!)
#   Per-QA MAD: P50=0.914
#
#   Per-category breakdown (4o-mini routed):
#     Cat1 (H+E,  n=282): 24.2 chunks, 1844 tok, top1 P50=3.625
#     Cat2 (H+G,  n=321): 24.5 chunks, 2256 tok, top1 P50=3.625
#     Cat3 (H+G+E, n=96): 32.7 chunks, 2291 tok, top1 P50=1.727
#     Cat4 (H+G+E,n=841): 33.6 chunks, 2365 tok, top1 P50=5.125
#     Cat5 (G+E,  n=446): 17.9 chunks,  648 tok, top1 P50=0.463 ← 极低!
#
# ===== 帕累托最优参数设计依据 (MAD-based) ====================================
#
# Stage1 (MAD 离群点过滤):
#   k=3.0: 99.2-99.5% survived, 0 top-1 lost → 保守安全
#   作用于原始 BAAI logits, threshold = median - k * MAD
#
# Stage3 (模式决定):
#   STRICT_THRESHOLD: absolute_min_score 绝对下限 → 过滤低质量噪音
#   CLIFF_EARLY_STOP: cliff_tolerance 悬崖检测 → 自适应截断
#   BUDGET_MAX: 纯预算驱动贪心装箱
#
# ┌────────────────────────┬──────┬──────────────────┬───────┬────────┬─────────────────────────────────────────────┐
# │ Config                 │ MAD  │ Mode Param       │ T_max │ λ_mmr  │ 设计依据                                    │
# ├────────────────────────┼──────┼──────────────────┼───────┼────────┼─────────────────────────────────────────────┤
# │ 4.1 STRICT             │ 3.0  │ min_score=-5.0   │ 1500  │ 0.7    │ 路由无效,同非路由;0 top-1 lost;激进压缩     │
# │ 4.1 CLIFF              │ 3.0  │ cliff_tol=2.5    │ 1500  │ 0.7    │ P10=2 可接受;高acc留够裕度                   │
# │ 4.1 BUDGET             │ 3.0  │ —                │ 1500  │ 0.7    │ 纯预算控制;高acc留够裕度                     │
# │ 4o  STRICT             │ 3.0  │ min_score=-6.0   │ 2000  │ 0.7    │ Cat5 top1 min=-5.281,-5.0会空1个QA;-6.0安全 │
# │ 4o  CLIFF              │ 3.0  │ cliff_tol=3.0    │ 2000  │ 0.7    │ 异构池 P10=14;Cat5不触发cliff→T_max兜底     │
# │ 4o  BUDGET             │ 3.0  │ —                │ 1800  │ 0.7    │ 预算驱动;Cat5(648)安全;Cat2/4适度压缩        │
# └────────────────────────┴──────┴──────────────────┴───────┴────────┴─────────────────────────────────────────────┘
#
# 4.1-mini 设计说明:
#   - 路由 all→H+G+E，候选池与非路由版相同 → 参数与 prune_cascade_closeai.sh v2 对齐
#   - T_max=1500: 仅 0.1% QA 自然在预算内 → 几乎所有 QA 都被积极裁剪
#   - SOTA_router=0.931, acc margin 充足 → 可激进压缩
#
# 4o-mini 设计说明:
#   - 路由创造了异构候选池: Cat5 仅 648 tok (自然已紧凑)
#   - T_max 必须足够高以保护 Cat5 低分但关键的内容
#   - STRICT -6.0 (非 -5.0): Cat5 G+E 分数极低, -5.0 会清空 1 个 QA
#   - CLIFF tol=3.0 (非 2.5): 异构池 P10=14 (安全), tol=2.5 → P10=2 (危险)
#   - BUDGET T_max=1800: Cat5(648) 安全通过, Cat1(1844→1800) 轻微裁剪,
#     Cat2(2256→1800) 显著裁剪, Cat4(2365→1800) 显著裁剪
#   - lambda_mmr=0.7: 路由已提供多样性, 偏重相关性
#
# 基线对比:
#   无量化 full-tower:    4.1-mini acc=0.9366, 4o-mini acc=0.8474, ~2365 tok/QA
#   无量化 router(agg):   4.1-mini acc=0.931,  4o-mini acc=0.884,  ~1884 tok/QA(4o)
#   level-cascade v2:     4.1-mini T≈1500,     4o-mini T≈1800
# ============================================================================

BASE_OUTPUT="benchmark_locomo/task_eval/results/locomo_tri_tower_router_cascade_results_v2"
SCRIPT="benchmark_locomo.task_eval.locomo_triple_router_quantification"
mkdir -p nohup_output "$BASE_OUTPUT"

# 检索参数 (对齐消融实验基线)
RETRIEVAL_COMMON="\
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
    --reranker-type baai"

# 路由参数
ROUTER_COMMON="--enable-router --router-strategy aggressive"

# 级联公共参数 (MAD-based, 所有模式共用)
#   mad_multiplier=3.0: Stage1 99.2-99.5% survived, 0 top-1 lost
#   lambda_mmr=0.7: 路由已提供多样性，偏重相关性
#   cap_to_input_tokens=True (默认): 级联只做减法
#   tower_min_ratio=None (默认): 不强制重分配塔预算
CASCADE_COMMON="\
    --enable-cascade-pruner \
    --cascade-mad-multiplier 3.0 \
    --cascade-lambda-mmr 0.7"

# ============================================================================
# GPT-4.1-mini (routing=no-op, all→H+G+E; SOTA_router=0.931, ~2365 tok/QA)
# ============================================================================

# 1. GPT-4.1-mini + STRICT_THRESHOLD
#    min_score=-5.0: 0 top-1 lost, 94.5% kept → 去除 5.5% 低质噪音
#    T_max=1500: 激进预算, 高 acc 余量允许
nohup uv run python -m $SCRIPT \
    --llm-model gpt-4.1-mini-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
    $RETRIEVAL_COMMON \
    $ROUTER_COMMON \
    $CASCADE_COMMON \
    --cascade-prune-mode STRICT_THRESHOLD \
    --cascade-absolute-min-score -5.0 \
    --cascade-max-context-tokens 1500 \
    --output-dir ${BASE_OUTPUT}/gpt41_mini_strict \
    --log-level INFO \
    > nohup_output/router_cascade_v2_gpt41_strict.log 2>&1 &

# 2. GPT-4.1-mini + CLIFF_EARLY_STOP
#    cliff_tol=2.5: 悬崖检测, P10=2 可接受
#    T_max=1500: cliff 自适应截断 + 预算兜底
nohup uv run python -m $SCRIPT \
    --llm-model gpt-4.1-mini-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
    $RETRIEVAL_COMMON \
    $ROUTER_COMMON \
    $CASCADE_COMMON \
    --cascade-prune-mode CLIFF_EARLY_STOP \
    --cascade-cliff-tolerance 2.5 \
    --cascade-max-context-tokens 1500 \
    --output-dir ${BASE_OUTPUT}/gpt41_mini_cliff \
    --log-level INFO \
    > nohup_output/router_cascade_v2_gpt41_cliff.log 2>&1 &

# 3. GPT-4.1-mini + BUDGET_MAX
#    T_max=1500: 纯预算驱动，贪心装箱
nohup uv run python -m $SCRIPT \
    --llm-model gpt-4.1-mini-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
    $RETRIEVAL_COMMON \
    $ROUTER_COMMON \
    $CASCADE_COMMON \
    --cascade-prune-mode BUDGET_MAX \
    --cascade-max-context-tokens 1500 \
    --output-dir ${BASE_OUTPUT}/gpt41_mini_budget \
    --log-level INFO \
    > nohup_output/router_cascade_v2_gpt41_budget.log 2>&1 &

# ============================================================================
# GPT-4o-mini (aggressive routing; SOTA_router=0.884, ~1884 tok/QA)
#
# 关键约束:
#   Cat5→G+E: 仅 648 tok, top-1 min=-5.281 → 需宽松阈值
#   Cat2→H+G: 2256 tok → 主要压缩目标
#   Cat4→H+G+E: 2365 tok → 主要压缩目标
# ============================================================================

# 4. GPT-4o-mini + STRICT_THRESHOLD
#    min_score=-6.0: 保护 Cat5 (min top1=-5.281), 0 top-1 lost, 98% kept
#    T_max=2000: Cat5(648) 安全, Cat2/4 适度裁剪
nohup uv run python -m $SCRIPT \
    --llm-model gpt-4o-mini-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
    $RETRIEVAL_COMMON \
    $ROUTER_COMMON \
    $CASCADE_COMMON \
    --cascade-prune-mode STRICT_THRESHOLD \
    --cascade-absolute-min-score -6.0 \
    --cascade-max-context-tokens 2000 \
    --output-dir ${BASE_OUTPUT}/gpt4o_mini_strict \
    --log-level INFO \
    > nohup_output/router_cascade_v2_gpt4o_strict.log 2>&1 &

# 5. GPT-4o-mini + CLIFF_EARLY_STOP
#    cliff_tol=3.0: 异构池安全 (P10=14), Cat5 不触发 cliff → T_max 兜底
#    T_max=2000: 保护路由准确率增益
nohup uv run python -m $SCRIPT \
    --llm-model gpt-4o-mini-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
    $RETRIEVAL_COMMON \
    $ROUTER_COMMON \
    $CASCADE_COMMON \
    --cascade-prune-mode CLIFF_EARLY_STOP \
    --cascade-cliff-tolerance 3.0 \
    --cascade-max-context-tokens 2000 \
    --output-dir ${BASE_OUTPUT}/gpt4o_mini_cliff \
    --log-level INFO \
    > nohup_output/router_cascade_v2_gpt4o_cliff.log 2>&1 &

# 6. GPT-4o-mini + BUDGET_MAX
#    T_max=1800: Cat5(648) 安全, Cat1(1844→1800) 轻裁, Cat2/4 显著裁剪
nohup uv run python -m $SCRIPT \
    --llm-model gpt-4o-mini-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
    $RETRIEVAL_COMMON \
    $ROUTER_COMMON \
    $CASCADE_COMMON \
    --cascade-prune-mode BUDGET_MAX \
    --cascade-max-context-tokens 1800 \
    --output-dir ${BASE_OUTPUT}/gpt4o_mini_budget \
    --log-level INFO \
    > nohup_output/router_cascade_v2_gpt4o_budget.log 2>&1 &

echo "============================================================================"
echo "✅ 已启动 6 个路由+级联量化 v2 benchmark 任务 (3 modes × 2 LLMs)"
echo "============================================================================"

# ============================================================================
# 【第三组】 DYNAMIC_ADAPTIVE v4 模式 (LoCoMo 调优版)
#
# v4 算法: budget_fraction 基准 = max_tokens, EVIDENCE_CHAIN s2 cap 移除
# Stage 3 参数 (budget_fraction 基于 max_tokens):
#   Cat1(聚合)+Cat5(对抗) → EVIDENCE_CHAIN: frac=1.0, s2=无限
#   Cat2(时序)             → MODERATE_CUT:   frac=0.90, s2 cap=5
#   Cat3(开放)+Cat4(简单)  → AGGRESSIVE_CUT: frac=0.80, s2=无限
#
# LoCoMo 调优依据 (基于 1986 题消融实验数据):
#   候选池: 全类别均 ~2400 tok, ~34 chunks (4.1 routing=no-op)
#   4o-mini routing 异构池: Cat5(G+E)~648tok, Cat1(H+E)~1844tok, Cat2-4~2300tok
#   STRICT @T=1500: 4.1 acc=0.931 tok=1456 (BEST cascade for 4.1)
#   STRICT @T=2000: 4o acc=0.886 tok=1725 (BEST cascade for 4o)
#   Stage1 MAD k=3.0: threshold≈-4.09 (比 STRICT min_score=-5.0 更激进去噪)
#   Stage2 跨塔消歧: 额外去重, STRICT 无此能力
#   ⇒ DA 去噪质量 ≥ STRICT, 可在更低 T_max 下匹配 STRICT 准确率
#
# 设计目标: 匹配 STRICT/Router 准确率基线, 最大化 token 节约
# ============================================================================

# 7. GPT-4.1-mini + DYNAMIC_ADAPTIVE v4 (LoCoMo 调优)
#    T_max=1500 (↓从 2000; STRICT@1500 已达 0.931, DA 去噪 ≥ STRICT)
#    cliff_tol=2.5 (保留但不参与 DA 早停)
#    Cat1/5 (EVIDENCE_CHAIN, frac=1.0): limit=1500, ≈STRICT 预算水平
#    Cat3/4 (AGGRESSIVE_CUT, frac=0.80): limit=1200, Cat3 去噪有益, Cat4 鲁棒
#    Cat2   (MODERATE_CUT, frac=0.90):   limit=1350, 适度压缩
#    预期: acc≈0.928-0.932 (≈STRICT 0.931), tok≈1320 (vs STRICT 1456, -9%)
nohup uv run python -m $SCRIPT \
    --llm-model gpt-4.1-mini-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
    $RETRIEVAL_COMMON \
    $ROUTER_COMMON \
    $CASCADE_COMMON \
    --cascade-prune-mode DYNAMIC_ADAPTIVE \
    --cascade-adaptive-dataset locomo \
    --cascade-cliff-tolerance 2.5 \
    --cascade-max-context-tokens 1500 \
    --output-dir ${BASE_OUTPUT}/gpt41_mini_dynamic_adaptive \
    --log-level INFO \
    > nohup_output/router_cascade_v2_gpt41_dynamic_adaptive.log 2>&1 &

# 8. GPT-4o-mini + DYNAMIC_ADAPTIVE v4 (LoCoMo 调优)
#    T_max=1800 (↓从 2000; 兼顾 Cat1 保护与 token 节约)
#    cliff_tol=3.0 (保留但不参与 DA 早停)
#    Cat5 (EVIDENCE_CHAIN, frac=1.0): limit=1800, 输入~648tok 全量通过
#    Cat1 (EVIDENCE_CHAIN, frac=1.0): limit=1800, 输入~1844tok 仅微量裁剪(~44tok≈2chunk)
#    Cat3/4 (AGGRESSIVE_CUT, frac=0.80): limit=1440, Cat3 去噪有益(CLIFF+4pp), Cat4 鲁棒
#    Cat2   (MODERATE_CUT, frac=0.90):   limit=1620, CLIFF@839tok 仍达 0.888 → 1620 安全
#    预期: acc≈0.883-0.888 (≥Router 0.884), tok≈1310 (vs STRICT 1725, -24%)
nohup uv run python -m $SCRIPT \
    --llm-model gpt-4o-mini-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
    $RETRIEVAL_COMMON \
    $ROUTER_COMMON \
    $CASCADE_COMMON \
    --cascade-prune-mode DYNAMIC_ADAPTIVE \
    --cascade-adaptive-dataset locomo \
    --cascade-cliff-tolerance 3.0 \
    --cascade-max-context-tokens 1800 \
    --output-dir ${BASE_OUTPUT}/gpt4o_mini_dynamic_adaptive \
    --log-level INFO \
    > nohup_output/router_cascade_v2_gpt4o_dynamic_adaptive.log 2>&1 &

echo ""
echo "📋 v2 关键更新 (MAD-based):"
echo "  ✓ Stage1: MAD 离群点过滤 (k=3.0, 99.2%+ survived)"
echo "  ✓ Stage3: STRICT/CLIFF/BUDGET 三模式 (替代旧 tau_high/tau_med)"
echo "  ✓ cap_to_input_tokens=True: 级联只做减法"
echo "  ✓ tower_min_ratio=None: 不强制重分配塔预算"
echo ""
echo "📋 参数设计:"
echo "  4.1-mini (routing=no-op, 同非路由):"
echo "    STRICT:  mad=3.0, min_score=-5.0, T=1500"
echo "    CLIFF:   mad=3.0, cliff_tol=2.5,  T=1500"
echo "    BUDGET:  mad=3.0,                  T=1500"
echo "    DYNAMIC: mad=3.0, cliff_tol=2.5,  T=1500, adaptive=locomo"
echo "  4o-mini (aggressive routing, 异构池):"
echo "    STRICT:  mad=3.0, min_score=-6.0, T=2000 (保护 Cat5 低分内容)"
echo "    CLIFF:   mad=3.0, cliff_tol=3.0,  T=2000 (异构安全)"
echo "    BUDGET:  mad=3.0,                  T=1800 (Cat5 安全+适度压缩)"
echo "    DYNAMIC: mad=3.0, cliff_tol=3.0,  T=1800, adaptive=locomo"
echo ""
echo "📊 基线对比:"
echo "  无量化 full-tower:  4.1=0.9366, 4o=0.8474, ~2365 tok/QA"
echo "  无量化 router(agg): 4.1=0.931,  4o=0.884,  ~1884 tok/QA(4o)"
echo ""
echo "📁 日志: nohup_output/router_cascade_v2_*.log"
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

# A1. GPT-4.1-mini: 无路由, DYNAMIC_ADAPTIVE (T_max=1500, cliff=2.5)
nohup uv run python -m $SCRIPT \
    --llm-model gpt-4.1-mini-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
    $RETRIEVAL_COMMON \
    $CASCADE_COMMON \
    --cascade-prune-mode DYNAMIC_ADAPTIVE \
    --cascade-adaptive-dataset locomo \
    --cascade-cliff-tolerance 2.5 \
    --cascade-max-context-tokens 1500 \
    --output-dir ${BASE_OUTPUT}/gpt41_mini_ablation_no_router \
    --log-level INFO \
    > nohup_output/router_cascade_v2_gpt41_ablation_no_router.log 2>&1 &

# A2. GPT-4o-mini: 无路由, DYNAMIC_ADAPTIVE (T_max=1800, cliff=3.0)
nohup uv run python -m $SCRIPT \
    --llm-model gpt-4o-mini-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
    $RETRIEVAL_COMMON \
    $CASCADE_COMMON \
    --cascade-prune-mode DYNAMIC_ADAPTIVE \
    --cascade-adaptive-dataset locomo \
    --cascade-cliff-tolerance 3.0 \
    --cascade-max-context-tokens 1800 \
    --output-dir ${BASE_OUTPUT}/gpt4o_mini_ablation_no_router \
    --log-level INFO \
    > nohup_output/router_cascade_v2_gpt4o_ablation_no_router.log 2>&1 &

# ── 消融 B: 关闭 Stage1/2 去重 ──────────────────────────────────────────────

# B1. GPT-4.1-mini: 路由+级联, 关闭 Stage1+2, DYNAMIC_ADAPTIVE 装箱 (T_max=1500, cliff=2.5)
nohup uv run python -m $SCRIPT \
    --llm-model gpt-4.1-mini-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
    $RETRIEVAL_COMMON \
    $ROUTER_COMMON \
    $CASCADE_COMMON \
    --cascade-prune-mode DYNAMIC_ADAPTIVE \
    --cascade-adaptive-dataset locomo \
    --cascade-cliff-tolerance 2.5 \
    --cascade-max-context-tokens 1500 \
    --no-cascade-stage1 \
    --no-cascade-stage2 \
    --output-dir ${BASE_OUTPUT}/gpt41_mini_ablation_no_stage12 \
    --log-level INFO \
    > nohup_output/router_cascade_v2_gpt41_ablation_no_stage12.log 2>&1 &

# B2. GPT-4o-mini: 路由+级联, 关闭 Stage1+2, DYNAMIC_ADAPTIVE 装箱 (T_max=1800, cliff=3.0)
nohup uv run python -m $SCRIPT \
    --llm-model gpt-4o-mini-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
    $RETRIEVAL_COMMON \
    $ROUTER_COMMON \
    $CASCADE_COMMON \
    --cascade-prune-mode DYNAMIC_ADAPTIVE \
    --cascade-adaptive-dataset locomo \
    --cascade-cliff-tolerance 3.0 \
    --cascade-max-context-tokens 1800 \
    --no-cascade-stage1 \
    --no-cascade-stage2 \
    --output-dir ${BASE_OUTPUT}/gpt4o_mini_ablation_no_stage12 \
    --log-level INFO \
    > nohup_output/router_cascade_v2_gpt4o_ablation_no_stage12.log 2>&1 &

# ── 消融 C: 关闭 Stage3 MMR 装箱 ────────────────────────────────────────────

# C1. GPT-4.1-mini: 路由+Stage1/2, 关闭 MMR 装箱 (退化为贪心), BUDGET_MAX (T_max=1500)
nohup uv run python -m $SCRIPT \
    --llm-model gpt-4.1-mini-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
    $RETRIEVAL_COMMON \
    $ROUTER_COMMON \
    $CASCADE_COMMON \
    --cascade-prune-mode BUDGET_MAX \
    --cascade-cliff-tolerance 2.5 \
    --cascade-max-context-tokens 1500 \
    --no-cascade-stage3-mmr \
    --output-dir ${BASE_OUTPUT}/gpt41_mini_ablation_no_packing \
    --log-level INFO \
    > nohup_output/router_cascade_v2_gpt41_ablation_no_packing.log 2>&1 &

# C2. GPT-4o-mini: 路由+Stage1/2, 关闭 MMR 装箱 (退化为贪心), BUDGET_MAX (T_max=1800)
nohup uv run python -m $SCRIPT \
    --llm-model gpt-4o-mini-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
    $RETRIEVAL_COMMON \
    $ROUTER_COMMON \
    $CASCADE_COMMON \
    --cascade-prune-mode BUDGET_MAX \
    --cascade-cliff-tolerance 3.0 \
    --cascade-max-context-tokens 1800 \
    --no-cascade-stage3-mmr \
    --output-dir ${BASE_OUTPUT}/gpt4o_mini_ablation_no_packing \
    --log-level INFO \
    > nohup_output/router_cascade_v2_gpt4o_ablation_no_packing.log 2>&1 &

echo ""
echo "🔬 消融实验已启动 (6 个任务):"
echo "  A1. GPT-4.1-mini + No-Router    → ${BASE_OUTPUT}/gpt41_mini_ablation_no_router"
echo "  A2. GPT-4o-mini  + No-Router    → ${BASE_OUTPUT}/gpt4o_mini_ablation_no_router"
echo "  B1. GPT-4.1-mini + No-Stage12   → ${BASE_OUTPUT}/gpt41_mini_ablation_no_stage12"
echo "  B2. GPT-4o-mini  + No-Stage12   → ${BASE_OUTPUT}/gpt4o_mini_ablation_no_stage12"
echo "  C1. GPT-4.1-mini + No-Packing   → ${BASE_OUTPUT}/gpt41_mini_ablation_no_packing"
echo "  C2. GPT-4o-mini  + No-Packing   → ${BASE_OUTPUT}/gpt4o_mini_ablation_no_packing"
