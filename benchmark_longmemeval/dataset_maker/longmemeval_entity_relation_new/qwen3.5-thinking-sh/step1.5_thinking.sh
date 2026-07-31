#!/bin/bash

# Purpose: Build thinking-enabled retry requests for failed LongMemEval entity-relation extraction batches.

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

# LongMemEval Entity Relation - Step 1.5: 重试失败的批量请求
# 读取阿里云百炼的错误结果文件，生成新的补救 batch 请求文件

# 1. 重试 error 目录下的所有失败请求 (开启 thinking 模式)
python benchmark_longmemeval/dataset_maker/longmemeval_entity_relation_new/step1.5_retry_failed_requests.py \
    --error-files benchmark_longmemeval/dataset_maker/longmemeval_entity_relation_new/batch_results/error \
    --output-file benchmark_longmemeval/dataset_maker/longmemeval_entity_relation_new/batch_requests/retry_batch_requests.jsonl \
    --enable-thinking \
    --thinking-budget 2048

# 2. 重试指定的错误文件 (开启 thinking 模式)
# python benchmark_longmemeval/dataset_maker/longmemeval_entity_relation_new/step1.5_retry_failed_requests.py \
#     --error-files benchmark_longmemeval/dataset_maker/longmemeval_entity_relation_new/batch_results/error/0-49_error.jsonl \
#     --output-file benchmark_longmemeval/dataset_maker/longmemeval_entity_relation_new/batch_requests/retry_0_49.jsonl \
#     --enable-thinking \
#     --thinking-budget 2048

# 3. 使用不同模型重试 (开启 thinking 模式)
# python benchmark_longmemeval/dataset_maker/longmemeval_entity_relation_new/step1.5_retry_failed_requests.py \
#     --error-files benchmark_longmemeval/dataset_maker/longmemeval_entity_relation_new/batch_results/error \
#     --output-file benchmark_longmemeval/dataset_maker/longmemeval_entity_relation_new/batch_requests/retry_qwen_max.jsonl \
#     --model qwen-max \
#     --enable-thinking \
#     --thinking-budget 2048
