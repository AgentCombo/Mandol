#!/usr/bin/env bash
set -euo pipefail

# Purpose: Inspect and retry failed LongMemEval entity deduplication jobs.

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

# 1. 先扫描查看缺失情况
uv run python -m benchmark_longmemeval.dataset_maker.longmemeval_entity_relation_new.step2_5_retry_failed_deduplication --scan-only

# 2. 同时检查空文件和无效文件
uv run python -m benchmark_longmemeval.dataset_maker.longmemeval_entity_relation_new.step2_5_retry_failed_deduplication --scan-only --include-invalid

# 3. 重试所有缺失的 QA
uv run python -m benchmark_longmemeval.dataset_maker.longmemeval_entity_relation_new.step2_5_retry_failed_deduplication --retry

# 4. 只重试指定范围
uv run python -m benchmark_longmemeval.dataset_maker.longmemeval_entity_relation_new.step2_5_retry_failed_deduplication --retry --start-index 300 --end-index 400

# 5. 手动指定要重试的索引
uv run python -m benchmark_longmemeval.dataset_maker.longmemeval_entity_relation_new.step2_5_retry_failed_deduplication --retry --specific-indices 322 329 333 341 345 346 348 349 350

# 6. 使用自定义 LLM 配置
uv run python -m benchmark_longmemeval.dataset_maker.longmemeval_entity_relation_new.step2_5_retry_failed_deduplication --retry --llm-model deepseek-reasoner --parallel-workers 5
