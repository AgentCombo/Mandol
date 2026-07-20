#!/usr/bin/env bash
set -euo pipefail

# Purpose: Deduplicate LongMemEval episodic-memory extraction outputs before indexing.

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
    echo "uv is required to run this dataset maker script" >&2
    exit 1
fi

# LongMemEval Episodic Memory V2 - Step 3: 情景记忆事实去重
# 支持自动DBSCAN参数优化和LLM精细去重

# 1. 使用默认参数运行（启用自动优化 + LLM去重）
uv run python -m benchmark_longmemeval.dataset_maker.longmemeval_episodic_memory_new.step3_deduplication

# 2. 禁用LLM精细去重（仅使用规则合并，速度更快）
# uv run python -m benchmark_longmemeval.dataset_maker.longmemeval_episodic_memory_new.step3_deduplication --no-llm-dedup

# 3. 禁用自动参数优化（使用手动参数）
# uv run python -m benchmark_longmemeval.dataset_maker.longmemeval_episodic_memory_new.step3_deduplication --no-auto-optimize --eps 0.15 --min_samples 1

# 4. 禁用按类别优化（使用全局优化参数）
# uv run python -m benchmark_longmemeval.dataset_maker.longmemeval_episodic_memory_new.step3_deduplication --no-optimize-per-category

# 5. 使用自定义LLM服务
# uv run python -m benchmark_longmemeval.dataset_maker.longmemeval_episodic_memory_new.step3_deduplication --dedup-model deepseek-reasoner --llm-base-url https://api.deepseek.com/v1

# 6. 调整LLM去重阈值
# uv run python -m benchmark_longmemeval.dataset_maker.longmemeval_episodic_memory_new.step3_deduplication --llm-cluster-threshold 3 --large-cluster-threshold 20

# 7. 处理指定范围的QA（分批处理大型任务）
# uv run python -m benchmark_longmemeval.dataset_maker.longmemeval_episodic_memory_new.step3_deduplication --start-qa 0 --end-qa 99
# uv run python -m benchmark_longmemeval.dataset_maker.longmemeval_episodic_memory_new.step3_deduplication --start-qa 100 --end-qa 199
# uv run python -m benchmark_longmemeval.dataset_maker.longmemeval_episodic_memory_new.step3_deduplication --start-qa 200 --end-qa 299
# uv run python -m benchmark_longmemeval.dataset_maker.longmemeval_episodic_memory_new.step3_deduplication --start-qa 300 --end-qa 399
# uv run python -m benchmark_longmemeval.dataset_maker.longmemeval_episodic_memory_new.step3_deduplication --start-qa 400 --end-qa 499
