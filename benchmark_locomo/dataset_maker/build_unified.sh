#!/usr/bin/env bash

# Purpose: Merge LoCoMo episodic, entity-relation, and hierarchical artifacts into unified per-sample graphs.

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

uv run benchmark_locomo/dataset_maker/build_unified_graph.py \
  --episodic-dir benchmark_locomo/dataset/locomo/episodic_memory/step3_loaded \
  --entity-dir benchmark_locomo/dataset/locomo/entity_relation/step3_semantic_graph \
  --hierarchical-dir benchmark_locomo/dataset/locomo/hierarchical_content/step4_semantic_graphs \
  --output-dir benchmark_locomo/dataset/locomo/unified_per_sample_graphs \
  --sample-ids all \
  --embedding-model Qwen/Qwen3-Embedding-0.6B \
  --splade-model naver/splade-v3
