#!/usr/bin/env bash

# Purpose: Run example LoCoMo entity-relation semantic graph builds with embedding and SPLADE options.

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

# 批量处理模式（SPLADE 已默认启用）
uv run python -m benchmark_locomo.dataset_maker.locomo_graph_maker.step3_locomo_entity_relation_semantic_graph_batch \
    --batch-mode

# 指定嵌入模型 + 批量处理
uv run python -m benchmark_locomo.dataset_maker.locomo_graph_maker.step3_locomo_entity_relation_semantic_graph_batch \
    --batch-mode \
    --text-embedding-model "Qwen/Qwen3-Embedding-0.6B"

# 完整参数示例（含并行处理）
uv run python -m benchmark_locomo.dataset_maker.locomo_graph_maker.step3_locomo_entity_relation_semantic_graph_batch \
    --batch-mode \
    --text-embedding-model "Qwen/Qwen3-Embedding-0.6B" \
    --splade-model "naver/splade-v3" \
    --splade-batch-size 32 \
    --enable-parallel \
    --max-workers 10

# 处理指定样本
uv run python -m benchmark_locomo.dataset_maker.locomo_graph_maker.step3_locomo_entity_relation_semantic_graph_batch \
    --batch-mode \
    --sample-ids conv-26 conv-30

# 如果需要禁用 SPLADE
uv run python -m benchmark_locomo.dataset_maker.locomo_graph_maker.step3_locomo_entity_relation_semantic_graph_batch \
    --batch-mode --no-splade
