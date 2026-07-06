#!/bin/bash

# Purpose: Deduplicate and enhance extracted LoCoMo episodic facts before retrieval indexing.

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

# Step 2: 事实去重与增强 (DBSCAN + LLM)
# 使用DBSCAN聚类 + LLM精细去重，生成累积事实和时间线

cd "$AMS_REPO_ROOT"

echo "========================================"
echo "Step 2: 事实去重与增强 (DBSCAN + LLM)"
echo "========================================"

uv run python -m benchmark_locomo.dataset_maker.locomo_episodic_memory.step2_deduplicate_and_enhance \
    --input-dir "benchmark_locomo/dataset/locomo/episodic_memory/step1_facts" \
    --output-dir "benchmark_locomo/dataset/locomo/episodic_memory/step2_enhanced" \
    --dedup-model "deepseek-v3.2-dashscope" \
    --auto-optimize \
    --workers 10

echo "Step 2 完成!"
