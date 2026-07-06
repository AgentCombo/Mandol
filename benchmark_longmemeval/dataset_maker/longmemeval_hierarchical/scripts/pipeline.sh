#!/usr/bin/env bash
set -euo pipefail

# Purpose: Run the full LongMemEval hierarchical workflow from raw message conversion to graph loading.

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

echo "[longmemeval_hierarchical] Raw message-level dataset generation"
bash benchmark_longmemeval/dataset_maker/longmemeval_hierarchical/scripts/raw_longmemeval_generator.sh

echo "[longmemeval_hierarchical] Step 1: L0 graph generation"
bash benchmark_longmemeval/dataset_maker/longmemeval_hierarchical/scripts/step1.sh

echo "[longmemeval_hierarchical] Step 2: L1/L2 summary request"
bash benchmark_longmemeval/dataset_maker/longmemeval_hierarchical/scripts/step2.sh

echo "[longmemeval_hierarchical] Step 3: semantic graph load"
bash benchmark_longmemeval/dataset_maker/longmemeval_hierarchical/scripts/step3.sh

echo "[longmemeval_hierarchical] Pipeline completed"
