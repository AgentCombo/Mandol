#!/usr/bin/env bash
set -euo pipefail

# Purpose: Run the full LoCoMo entity-relation graph pipeline from entity extraction to semantic graph loading.

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

echo "[locomo_graph_maker] Step 1: entity extraction"
bash benchmark_locomo/dataset_maker/locomo_graph_maker/scripts/step1.sh

echo "[locomo_graph_maker] Step 2: relation generation"
bash benchmark_locomo/dataset_maker/locomo_graph_maker/scripts/step2.sh

echo "[locomo_graph_maker] Step 3: semantic graph build"
bash benchmark_locomo/dataset_maker/locomo_graph_maker/scripts/step3.sh

echo "[locomo_graph_maker] Pipeline completed"
