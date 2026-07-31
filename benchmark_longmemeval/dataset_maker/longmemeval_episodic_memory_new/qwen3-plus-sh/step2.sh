#!/bin/bash

# Purpose: Build retry requests for failed LongMemEval episodic-memory extraction batches.

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

# LongMemEval Episodic Memory V2 - Step 2: 失败请求重新生成
# 用于处理因内容审查等原因失败的请求

# 1. 重新生成所有失败的请求（默认每组1个session）
python benchmark_longmemeval/dataset_maker/longmemeval_episodic_memory_new/step2_build_failed_requests.py

# 2. 指定每组2个session重新生成
# python benchmark_longmemeval/dataset_maker/longmemeval_episodic_memory_new/step2_build_failed_requests.py --sessions-per-group 2

# 3. 只重新生成特定QA的失败请求
# python benchmark_longmemeval/dataset_maker/longmemeval_episodic_memory_new/step2_build_failed_requests.py --qa-indices 19 27 44

# 4. 使用不同模型
# python benchmark_longmemeval/dataset_maker/longmemeval_episodic_memory_new/step2_build_failed_requests.py --model qwen-max
