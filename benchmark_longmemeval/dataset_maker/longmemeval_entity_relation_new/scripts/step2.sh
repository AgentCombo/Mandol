#!/usr/bin/env bash
set -euo pipefail

# Purpose: Deduplicate LongMemEval entity-relation extraction outputs before graph construction.

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

# LongMemEval Entity Relation - Step 2: 实体去重
# 支持DBSCAN聚类 + LLM精细去重

# 1. 使用默认参数运行（自动加载 + LLM去重）
uv run python -m benchmark_longmemeval.dataset_maker.longmemeval_entity_relation_new.step2_entity_deduplication --auto-load

# 2. 禁用LLM去重（仅使用规则合并，速度更快）
# uv run python -m benchmark_longmemeval.dataset_maker.longmemeval_entity_relation_new.step2_entity_deduplication --auto-load --no-llm-dedup

# 3. 指定LLM模型
# uv run python -m benchmark_longmemeval.dataset_maker.longmemeval_entity_relation_new.step2_entity_deduplication --auto-load --dedup-model deepseek-reasoner

# 4. 使用自定义LLM服务
# uv run python -m benchmark_longmemeval.dataset_maker.longmemeval_entity_relation_new.step2_entity_deduplication --auto-load \
#     --dedup-model deepseek-ai/DeepSeek-V3.2-Exp \
#     --llm-base-url https://api.siliconflow.cn/v1

# 5. 禁用逐QA优化（加快速度）
# uv run python -m benchmark_longmemeval.dataset_maker.longmemeval_entity_relation_new.step2_entity_deduplication --auto-load --no-per-qa-optimization

# 6. 单线程处理（最稳定）
# uv run python -m benchmark_longmemeval.dataset_maker.longmemeval_entity_relation_new.step2_entity_deduplication --auto-load --parallel-workers 1

# 7. 处理单个文件
# uv run python -m benchmark_longmemeval.dataset_maker.longmemeval_entity_relation_new.step2_entity_deduplication \
#     --result-file benchmark_longmemeval/dataset_maker/longmemeval_entity_relation_new/batch_results/0-49_success.jsonl

# 8. 处理指定范围的QA
# uv run python -m benchmark_longmemeval.dataset_maker.longmemeval_entity_relation_new.step2_entity_deduplication --auto-load --start-index 0 --end-index 99
# uv run python -m benchmark_longmemeval.dataset_maker.longmemeval_entity_relation_new.step2_entity_deduplication --auto-load --start-index 100 --end-index 199
# uv run python -m benchmark_longmemeval.dataset_maker.longmemeval_entity_relation_new.step2_entity_deduplication --auto-load --start-index 200 --end-index 299
# uv run python -m benchmark_longmemeval.dataset_maker.longmemeval_entity_relation_new.step2_entity_deduplication --auto-load --start-index 300 --end-index 399
# uv run python -m benchmark_longmemeval.dataset_maker.longmemeval_entity_relation_new.step2_entity_deduplication --auto-load --start-index 400 --end-index 499

# 9. 使用不带LLM的去重版本（更快但精度略低）
# uv run python -m benchmark_longmemeval.dataset_maker.longmemeval_entity_relation_new.step2_entity_deduplication_without_llm --auto-load
