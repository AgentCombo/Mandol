#!/usr/bin/env bash
set -euo pipefail

# Purpose: Save deduplicated LongMemEval episodic memories into Mandol semantic map artifacts.

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

# LongMemEval Episodic Memory V2 - Step 4: 保存到 SemanticGraph
# 加载情景记忆到语义图谱

# 1. 处理所有 QA（默认配置）
uv run python -m benchmark_longmemeval.dataset_maker.longmemeval_episodic_memory_new.step4_saved_in_semantic_map_batch

# 2. 处理 QA 0-99
# uv run python -m benchmark_longmemeval.dataset_maker.longmemeval_episodic_memory_new.step4_saved_in_semantic_map_batch --start-qa 0 --end-qa 99

# 3. 分批处理（避免内存不足）
# uv run python -m benchmark_longmemeval.dataset_maker.longmemeval_episodic_memory_new.step4_saved_in_semantic_map_batch --start-qa 0 --end-qa 99
# uv run python -m benchmark_longmemeval.dataset_maker.longmemeval_episodic_memory_new.step4_saved_in_semantic_map_batch --start-qa 100 --end-qa 199
# uv run python -m benchmark_longmemeval.dataset_maker.longmemeval_episodic_memory_new.step4_saved_in_semantic_map_batch --start-qa 200 --end-qa 299
# uv run python -m benchmark_longmemeval.dataset_maker.longmemeval_episodic_memory_new.step4_saved_in_semantic_map_batch --start-qa 300 --end-qa 399
# uv run python -m benchmark_longmemeval.dataset_maker.longmemeval_episodic_memory_new.step4_saved_in_semantic_map_batch --start-qa 400 --end-qa 499

# 4. 指定嵌入模型
# uv run python -m benchmark_longmemeval.dataset_maker.longmemeval_episodic_memory_new.step4_saved_in_semantic_map_batch --embedding-model "Qwen/Qwen3-Embedding-0.6B"

# 5. 禁用 SPLADE（减少内存占用）
# uv run python -m benchmark_longmemeval.dataset_maker.longmemeval_episodic_memory_new.step4_saved_in_semantic_map_batch --no-splade

# 6. 单线程调试模式
# uv run python -m benchmark_longmemeval.dataset_maker.longmemeval_episodic_memory_new.step4_saved_in_semantic_map_batch --no-parallel --debug

# 7. 调整并行工作线程数
# uv run python -m benchmark_longmemeval.dataset_maker.longmemeval_episodic_memory_new.step4_saved_in_semantic_map_batch --max-workers 8
