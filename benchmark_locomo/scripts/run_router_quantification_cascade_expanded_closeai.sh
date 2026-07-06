#!/bin/bash
# Purpose: expanded LoCoMo router + quantification + cascade launcher where
# each experiment is written as a standalone command for copying or debugging.
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
# LoCoMo 路由 + 级联量化 Benchmark 独立运行脚本 (v2: MAD-based)
# 该脚本已将所有公共变量展开，方便单独复制粘贴运行或调试某一个具体配置
# ============================================================================

# 创建日志目录
mkdir -p nohup_output
mkdir -p benchmark_locomo/task_eval/results/locomo_tri_tower_router_cascade_results_v2

# ============================================================================
# 【第一组】 GPT-4.1-mini (路由 all→H+G+E，候选池同非路由版)
# SOTA_router=0.931, acc margin 充足 → 允许使用激进的压缩策略 (T=1500)
# ============================================================================

# 1. GPT-4.1-mini + STRICT_THRESHOLD 模式
# 设计依据: min_score=-5.0 可去除 5.5% 低质噪音且不丢失 top-1; T_max=1500 激进预算
echo "Starting: GPT-4.1-mini STRICT_THRESHOLD..."
nohup uv run python -m benchmark_locomo.task_eval.locomo_triple_router_quantification \
    --llm-model gpt-4.1-mini-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
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
    --reranker-type baai \
    --enable-router \
    --router-strategy aggressive \
    --enable-cascade-pruner \
    --cascade-mad-multiplier 3.0 \
    --cascade-lambda-mmr 0.7 \
    --cascade-prune-mode STRICT_THRESHOLD \
    --cascade-absolute-min-score -5.0 \
    --cascade-max-context-tokens 1500 \
    --output-dir benchmark_locomo/task_eval/results/locomo_tri_tower_router_cascade_results_v2/gpt41_mini_strict \
    --log-level INFO \
    > nohup_output/router_cascade_v2_gpt41_strict.log 2>&1 &

# 2. GPT-4.1-mini + CLIFF_EARLY_STOP 模式
# 设计依据: cliff_tol=2.5 悬崖检测，结合 T_max=1500 预算兜底
echo "Starting: GPT-4.1-mini CLIFF_EARLY_STOP..."
nohup uv run python -m benchmark_locomo.task_eval.locomo_triple_router_quantification \
    --llm-model gpt-4.1-mini-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
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
    --reranker-type baai \
    --enable-router \
    --router-strategy aggressive \
    --enable-cascade-pruner \
    --cascade-mad-multiplier 3.0 \
    --cascade-lambda-mmr 0.7 \
    --cascade-prune-mode CLIFF_EARLY_STOP \
    --cascade-cliff-tolerance 2.5 \
    --cascade-max-context-tokens 1500 \
    --output-dir benchmark_locomo/task_eval/results/locomo_tri_tower_router_cascade_results_v2/gpt41_mini_cliff \
    --log-level INFO \
    > nohup_output/router_cascade_v2_gpt41_cliff.log 2>&1 &

# 3. GPT-4.1-mini + BUDGET_MAX 模式
# 设计依据: 纯预算驱动，贪心装箱至 T_max=1500
echo "Starting: GPT-4.1-mini BUDGET_MAX..."
nohup uv run python -m benchmark_locomo.task_eval.locomo_triple_router_quantification \
    --llm-model gpt-4.1-mini-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
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
    --reranker-type baai \
    --enable-router \
    --router-strategy aggressive \
    --enable-cascade-pruner \
    --cascade-mad-multiplier 3.0 \
    --cascade-lambda-mmr 0.7 \
    --cascade-prune-mode BUDGET_MAX \
    --cascade-max-context-tokens 1500 \
    --output-dir benchmark_locomo/task_eval/results/locomo_tri_tower_router_cascade_results_v2/gpt41_mini_budget \
    --log-level INFO \
    > nohup_output/router_cascade_v2_gpt41_budget.log 2>&1 &


# ============================================================================
# 【第二组】 GPT-4o-mini (Aggressive 路由，产生了异构候选池)
# 关键约束: Cat5→G+E(仅648tok，分数极低); Cat2/4 为主要压缩目标
# ============================================================================

# 4. GPT-4o-mini + STRICT_THRESHOLD 模式
# 设计依据: min_score=-6.0 以保护 Cat5(min top1=-5.281); T_max=2000 保证 Cat5 安全同时裁剪 Cat2/4
echo "Starting: GPT-4o-mini STRICT_THRESHOLD..."
nohup uv run python -m benchmark_locomo.task_eval.locomo_triple_router_quantification \
    --llm-model gpt-4o-mini-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
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
    --reranker-type baai \
    --enable-router \
    --router-strategy aggressive \
    --enable-cascade-pruner \
    --cascade-mad-multiplier 3.0 \
    --cascade-lambda-mmr 0.7 \
    --cascade-prune-mode STRICT_THRESHOLD \
    --cascade-absolute-min-score -6.0 \
    --cascade-max-context-tokens 2000 \
    --output-dir benchmark_locomo/task_eval/results/locomo_tri_tower_router_cascade_results_v2/gpt4o_mini_strict \
    --log-level INFO \
    > nohup_output/router_cascade_v2_gpt4o_strict.log 2>&1 &

# 5. GPT-4o-mini + CLIFF_EARLY_STOP 模式
# 设计依据: cliff_tol=3.0 保障异构池安全 (Tol=2.5太危险); Cat5 不触发 cliff 靠 T=2000 兜底
echo "Starting: GPT-4o-mini CLIFF_EARLY_STOP..."
nohup uv run python -m benchmark_locomo.task_eval.locomo_triple_router_quantification \
    --llm-model gpt-4o-mini-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
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
    --reranker-type baai \
    --enable-router \
    --router-strategy aggressive \
    --enable-cascade-pruner \
    --cascade-mad-multiplier 3.0 \
    --cascade-lambda-mmr 0.7 \
    --cascade-prune-mode CLIFF_EARLY_STOP \
    --cascade-cliff-tolerance 3.0 \
    --cascade-max-context-tokens 2000 \
    --output-dir benchmark_locomo/task_eval/results/locomo_tri_tower_router_cascade_results_v2/gpt4o_mini_cliff \
    --log-level INFO \
    > nohup_output/router_cascade_v2_gpt4o_cliff.log 2>&1 &

# 6. GPT-4o-mini + BUDGET_MAX 模式
# 设计依据: T_max=1800，Cat5(648) 安全通过，Cat2/Cat4 显著裁剪
echo "Starting: GPT-4o-mini BUDGET_MAX..."
nohup uv run python -m benchmark_locomo.task_eval.locomo_triple_router_quantification \
    --llm-model gpt-4o-mini-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
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
    --reranker-type baai \
    --enable-router \
    --router-strategy aggressive \
    --enable-cascade-pruner \
    --cascade-mad-multiplier 3.0 \
    --cascade-lambda-mmr 0.7 \
    --cascade-prune-mode BUDGET_MAX \
    --cascade-max-context-tokens 1800 \
    --output-dir benchmark_locomo/task_eval/results/locomo_tri_tower_router_cascade_results_v2/gpt4o_mini_budget \
    --log-level INFO \
    > nohup_output/router_cascade_v2_gpt4o_budget.log 2>&1 &

# ============================================================================
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
# 设计依据:
#   T_max=1500 (↓从 2000; STRICT@1500 已达 0.931, DA 去噪 ≥ STRICT)
#   cliff_tol=2.5 (保留但不参与 DA 早停)
#   Cat1/5 (EVIDENCE_CHAIN, frac=1.0): limit=1500, ≈STRICT 预算水平
#   Cat3/4 (AGGRESSIVE_CUT, frac=0.80): limit=1200, Cat3 去噪有益, Cat4 鲁棒
#   Cat2   (MODERATE_CUT, frac=0.90):   limit=1350, 适度压缩
#   预期: acc≈0.928-0.932 (≈STRICT 0.931), tok≈1320 (vs STRICT 1456, -9%)
echo "Starting: GPT-4.1-mini DYNAMIC_ADAPTIVE..."
nohup uv run python -m benchmark_locomo.task_eval.locomo_triple_router_quantification \
    --llm-model gpt-4.1-mini-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
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
    --reranker-type baai \
    --enable-router \
    --router-strategy aggressive \
    --enable-cascade-pruner \
    --cascade-mad-multiplier 3.0 \
    --cascade-lambda-mmr 0.7 \
    --cascade-prune-mode DYNAMIC_ADAPTIVE \
    --cascade-adaptive-dataset locomo \
    --cascade-cliff-tolerance 2.5 \
    --cascade-max-context-tokens 1500 \
    --output-dir benchmark_locomo/task_eval/results/locomo_tri_tower_router_cascade_results_v2/gpt41_mini_dynamic_adaptive \
    --log-level INFO \
    > nohup_output/router_cascade_v2_gpt41_dynamic_adaptive.log 2>&1 &

# 8. GPT-4o-mini + DYNAMIC_ADAPTIVE v4 (LoCoMo 调优)
# 设计依据:
#   T_max=1800 (↓从 2000; 兼顾 Cat1 保护与 token 节约)
#   cliff_tol=3.0 (保留但不参与 DA 早停)
#   Cat5 (EVIDENCE_CHAIN, frac=1.0): limit=1800, 输入~648tok 全量通过
#   Cat1 (EVIDENCE_CHAIN, frac=1.0): limit=1800, 输入~1844tok 仅微量裁剪(~44tok≈2chunk)
#   Cat3/4 (AGGRESSIVE_CUT, frac=0.80): limit=1440, Cat3 去噪有益(CLIFF+4pp), Cat4 鲁棒
#   Cat2   (MODERATE_CUT, frac=0.90):   limit=1620, CLIFF@839tok 仍达 0.888 → 1620 安全
#   预期: acc≈0.883-0.888 (≥Router 0.884), tok≈1310 (vs STRICT 1725, -24%)
echo "Starting: GPT-4o-mini DYNAMIC_ADAPTIVE..."
nohup uv run python -m benchmark_locomo.task_eval.locomo_triple_router_quantification \
    --llm-model gpt-4o-mini-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
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
    --reranker-type baai \
    --enable-router \
    --router-strategy aggressive \
    --enable-cascade-pruner \
    --cascade-mad-multiplier 3.0 \
    --cascade-lambda-mmr 0.7 \
    --cascade-prune-mode DYNAMIC_ADAPTIVE \
    --cascade-adaptive-dataset locomo \
    --cascade-cliff-tolerance 3.0 \
    --cascade-max-context-tokens 1800 \
    --output-dir benchmark_locomo/task_eval/results/locomo_tri_tower_router_cascade_results_v2/gpt4o_mini_dynamic_adaptive \
    --log-level INFO \
    > nohup_output/router_cascade_v2_gpt4o_dynamic_adaptive.log 2>&1 &

echo "============================================================================"
echo "✅ 已成功下发 8 个展开式的后台任务 (6 原有 + 2 DYNAMIC_ADAPTIVE)。"
echo "可以使用 'tail -f nohup_output/router_cascade_v2_gpt4o_budget.log' 等命令查看运行日志。"

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
echo "Starting: GPT-4.1-mini Ablation No-Router..."
nohup uv run python -m benchmark_locomo.task_eval.locomo_triple_router_quantification \
    --llm-model gpt-4.1-mini-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
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
    --reranker-type baai \
    --enable-cascade-pruner \
    --cascade-mad-multiplier 3.0 \
    --cascade-lambda-mmr 0.7 \
    --cascade-prune-mode DYNAMIC_ADAPTIVE \
    --cascade-adaptive-dataset locomo \
    --cascade-cliff-tolerance 2.5 \
    --cascade-max-context-tokens 1500 \
    --output-dir benchmark_locomo/task_eval/results/locomo_tri_tower_router_cascade_results_v2/gpt41_mini_ablation_no_router \
    --log-level INFO \
    > nohup_output/router_cascade_v2_gpt41_ablation_no_router.log 2>&1 &

# A2. GPT-4o-mini: 无路由, DYNAMIC_ADAPTIVE (T_max=1800, cliff=3.0)
echo "Starting: GPT-4o-mini Ablation No-Router..."
nohup uv run python -m benchmark_locomo.task_eval.locomo_triple_router_quantification \
    --llm-model gpt-4o-mini-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
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
    --reranker-type baai \
    --enable-cascade-pruner \
    --cascade-mad-multiplier 3.0 \
    --cascade-lambda-mmr 0.7 \
    --cascade-prune-mode DYNAMIC_ADAPTIVE \
    --cascade-adaptive-dataset locomo \
    --cascade-cliff-tolerance 3.0 \
    --cascade-max-context-tokens 1800 \
    --output-dir benchmark_locomo/task_eval/results/locomo_tri_tower_router_cascade_results_v2/gpt4o_mini_ablation_no_router \
    --log-level INFO \
    > nohup_output/router_cascade_v2_gpt4o_ablation_no_router.log 2>&1 &

# ── 消融 B: 关闭 Stage1/2 去重 ──────────────────────────────────────────────

# B1. GPT-4.1-mini: 路由+级联, 关闭 Stage1+2, DYNAMIC_ADAPTIVE 装箱 (T_max=1500, cliff=2.5)
echo "Starting: GPT-4.1-mini Ablation No-Stage12..."
nohup uv run python -m benchmark_locomo.task_eval.locomo_triple_router_quantification \
    --llm-model gpt-4.1-mini-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
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
    --reranker-type baai \
    --enable-router \
    --router-strategy aggressive \
    --enable-cascade-pruner \
    --cascade-mad-multiplier 3.0 \
    --cascade-lambda-mmr 0.7 \
    --cascade-prune-mode DYNAMIC_ADAPTIVE \
    --cascade-adaptive-dataset locomo \
    --cascade-cliff-tolerance 2.5 \
    --cascade-max-context-tokens 1500 \
    --no-cascade-stage1 \
    --no-cascade-stage2 \
    --output-dir benchmark_locomo/task_eval/results/locomo_tri_tower_router_cascade_results_v2/gpt41_mini_ablation_no_stage12 \
    --log-level INFO \
    > nohup_output/router_cascade_v2_gpt41_ablation_no_stage12.log 2>&1 &

# B2. GPT-4o-mini: 路由+级联, 关闭 Stage1+2, DYNAMIC_ADAPTIVE 装箱 (T_max=1800, cliff=3.0)
echo "Starting: GPT-4o-mini Ablation No-Stage12..."
nohup uv run python -m benchmark_locomo.task_eval.locomo_triple_router_quantification \
    --llm-model gpt-4o-mini-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
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
    --reranker-type baai \
    --enable-router \
    --router-strategy aggressive \
    --enable-cascade-pruner \
    --cascade-mad-multiplier 3.0 \
    --cascade-lambda-mmr 0.7 \
    --cascade-prune-mode DYNAMIC_ADAPTIVE \
    --cascade-adaptive-dataset locomo \
    --cascade-cliff-tolerance 3.0 \
    --cascade-max-context-tokens 1800 \
    --no-cascade-stage1 \
    --no-cascade-stage2 \
    --output-dir benchmark_locomo/task_eval/results/locomo_tri_tower_router_cascade_results_v2/gpt4o_mini_ablation_no_stage12 \
    --log-level INFO \
    > nohup_output/router_cascade_v2_gpt4o_ablation_no_stage12.log 2>&1 &

# ── 消融 C: 关闭 Stage3 MMR 装箱 ────────────────────────────────────────────

# C1. GPT-4.1-mini: 路由+Stage1/2, 关闭 MMR 装箱 (退化为贪心), BUDGET_MAX (T_max=1500)
echo "Starting: GPT-4.1-mini Ablation No-Packing..."
nohup uv run python -m benchmark_locomo.task_eval.locomo_triple_router_quantification \
    --llm-model gpt-4.1-mini-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
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
    --reranker-type baai \
    --enable-router \
    --router-strategy aggressive \
    --enable-cascade-pruner \
    --cascade-mad-multiplier 3.0 \
    --cascade-lambda-mmr 0.7 \
    --cascade-prune-mode BUDGET_MAX \
    --cascade-cliff-tolerance 2.5 \
    --cascade-max-context-tokens 1500 \
    --no-cascade-stage3-mmr \
    --output-dir benchmark_locomo/task_eval/results/locomo_tri_tower_router_cascade_results_v2/gpt41_mini_ablation_no_packing \
    --log-level INFO \
    > nohup_output/router_cascade_v2_gpt41_ablation_no_packing.log 2>&1 &

# C2. GPT-4o-mini: 路由+Stage1/2, 关闭 MMR 装箱 (退化为贪心), BUDGET_MAX (T_max=1800)
echo "Starting: GPT-4o-mini Ablation No-Packing..."
nohup uv run python -m benchmark_locomo.task_eval.locomo_triple_router_quantification \
    --llm-model gpt-4o-mini-closeai \
    --llm-evaluate-model gpt-4o-mini-closeai \
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
    --reranker-type baai \
    --enable-router \
    --router-strategy aggressive \
    --enable-cascade-pruner \
    --cascade-mad-multiplier 3.0 \
    --cascade-lambda-mmr 0.7 \
    --cascade-prune-mode BUDGET_MAX \
    --cascade-cliff-tolerance 3.0 \
    --cascade-max-context-tokens 1800 \
    --no-cascade-stage3-mmr \
    --output-dir benchmark_locomo/task_eval/results/locomo_tri_tower_router_cascade_results_v2/gpt4o_mini_ablation_no_packing \
    --log-level INFO \
    > nohup_output/router_cascade_v2_gpt4o_ablation_no_packing.log 2>&1 &

echo ""
echo "🔬 消融实验已启动 (6 个任务):"
echo "  A1. GPT-4.1-mini + No-Router    → .../gpt41_mini_ablation_no_router"
echo "  A2. GPT-4o-mini  + No-Router    → .../gpt4o_mini_ablation_no_router"
echo "  B1. GPT-4.1-mini + No-Stage12   → .../gpt41_mini_ablation_no_stage12"
echo "  B2. GPT-4o-mini  + No-Stage12   → .../gpt4o_mini_ablation_no_stage12"
echo "  C1. GPT-4.1-mini + No-Packing   → .../gpt41_mini_ablation_no_packing"
echo "  C2. GPT-4o-mini  + No-Packing   → .../gpt4o_mini_ablation_no_packing"
echo "============================================================================"
