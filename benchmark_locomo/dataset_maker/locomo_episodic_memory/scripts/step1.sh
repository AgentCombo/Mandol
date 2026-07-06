#!/bin/bash

# Purpose: Extract episodic facts from LoCoMo conversations with the configured LLM backend.

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

# Step 1: 情景事实抽取
# 从locomo10对话中抽取可回答问题的事实

cd "$AMS_REPO_ROOT"

echo "========================================"
echo "Step 1: 情景事实抽取"
echo "========================================"

uv run python -m benchmark_locomo.dataset_maker.locomo_episodic_memory.step1_extract_episodic_facts \
    --input-file "benchmark_locomo/dataset/locomo/locomo10.json" \
    --output-dir "benchmark_locomo/dataset/locomo/episodic_memory/step1_facts" \
    --extract-model "deepseek-reasoner" \
    --max-workers 6

echo "Step 1 完成!"
