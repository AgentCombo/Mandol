#!/usr/bin/env bash
set -euo pipefail

# Purpose: Build retry requests for failed LongMemEval episodic-memory extraction batches using uv.

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
    echo "uv is required to run this dataset maker script" >&2
    exit 1
fi

# LongMemEval Episodic Memory V2 - Step 2: 失败请求重新生成
# 用于处理因内容审查等原因失败的请求

# 1. 重新生成所有失败的请求（默认每组1个session）
uv run python -m benchmark_longmemeval.dataset_maker.longmemeval_episodic_memory_new.step2_build_failed_requests

# 2. 指定每组2个session重新生成
# uv run python -m benchmark_longmemeval.dataset_maker.longmemeval_episodic_memory_new.step2_build_failed_requests --sessions-per-group 2

# 3. 只重新生成特定QA的失败请求
# uv run python -m benchmark_longmemeval.dataset_maker.longmemeval_episodic_memory_new.step2_build_failed_requests --qa-indices 19 27 44

# 4. 使用不同模型
# uv run python -m benchmark_longmemeval.dataset_maker.longmemeval_episodic_memory_new.step2_build_failed_requests --model qwen-max
