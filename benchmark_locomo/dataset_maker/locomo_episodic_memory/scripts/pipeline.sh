#!/bin/bash

# Purpose: Run the full LoCoMo episodic-memory pipeline from fact extraction to retrieval index loading.

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

# Pipeline: 完整的情景记忆生成流程
# 执行所有步骤

cd "$AMS_REPO_ROOT"

echo "========================================"
echo "情景记忆生成 Pipeline"
echo "========================================"

# Step 1: 事实抽取
echo ""
echo "[1/3] 事实抽取..."
bash benchmark_locomo/dataset_maker/locomo_episodic_memory/scripts/step1.sh

# Step 2: 去重增强
echo ""
echo "[2/3] 去重增强..."
bash benchmark_locomo/dataset_maker/locomo_episodic_memory/scripts/step2.sh

# Step 3: 加载索引
echo ""
echo "[3/3] 加载索引..."
bash benchmark_locomo/dataset_maker/locomo_episodic_memory/scripts/step3.sh

echo ""
echo "========================================"
echo "Pipeline 完成!"
echo "========================================"
echo "输出目录: benchmark_locomo/dataset/locomo/episodic_memory/"
