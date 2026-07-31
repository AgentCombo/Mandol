#!/usr/bin/env bash
set -euo pipefail

# Purpose: Run all LoCoMo dataset-maker workflows used to build benchmark retrieval artifacts.

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

echo "[dataset_maker] LoCoMo episodic memory workflow"
bash benchmark_locomo/dataset_maker/locomo_episodic_memory/scripts/pipeline.sh

echo "[dataset_maker] LoCoMo entity-relation graph workflow"
bash benchmark_locomo/dataset_maker/locomo_graph_maker/scripts/pipeline.sh

echo "[dataset_maker] LoCoMo hierarchical content workflow"
bash benchmark_locomo/dataset_maker/locomo_hierarchical_content_maker/scripts/pipeline.sh

echo "[dataset_maker] All LoCoMo dataset maker workflows completed"
echo "[dataset_maker] Outputs:"
echo "  - benchmark_locomo/dataset/locomo/episodic_memory"
echo "  - benchmark_locomo/dataset/locomo/entity_relation"
echo "  - benchmark_locomo/dataset/locomo/hierarchical_content"
