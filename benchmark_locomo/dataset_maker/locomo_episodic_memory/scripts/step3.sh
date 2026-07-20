#!/bin/bash

# Purpose: Load enhanced LoCoMo episodic memories into Mandol semantic retrieval artifacts.

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

# Step 3: 加载到检索系统 (SemanticGraph)
# 构建索引，准备检索，分样本保存

cd "$AMS_REPO_ROOT"

echo "========================================"
echo "Step 3: 加载情景记忆到 SemanticGraph"
echo "========================================"

# 默认配置
INPUT_DIR="benchmark_locomo/dataset/locomo/episodic_memory/step2_enhanced"
OUTPUT_DIR="benchmark_locomo/dataset/locomo/episodic_memory/step3_loaded"
EMBEDDING_MODEL="Qwen/Qwen3-Embedding-0.6B"

# 可选参数
# --no-splade: 禁用SPLADE稀疏向量构建
# --no-index: 禁用索引构建
# --embedding-model MODEL: 指定嵌入模型
# --debug: 启用调试模式

uv run python -m benchmark_locomo.dataset_maker.locomo_episodic_memory.step3_load_to_retrieval_batch \
    --input-dir "${INPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --embedding-model "${EMBEDDING_MODEL}" \
    --no-splade

echo ""
echo "Step 3 完成!"
echo "输出目录: ${OUTPUT_DIR}"
echo "每个样本保存在独立的子目录中，包含:"
echo "  - semantic_map_data/: SemanticMap向量数据"
echo "  - semantic_graph.json: NetworkX图结构"
echo "  - sample_metadata.json: 样本元数据"
echo "  - sample_index.json: 样本索引"
echo "  - episodic_facts.json: 原始增强数据"
