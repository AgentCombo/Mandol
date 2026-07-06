#!/usr/bin/env bash
set -euo pipefail

# Purpose: Create LongMemEval L1/L2 summary requests for the hierarchical workflow.

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

STEP2_MODULE_FILE="benchmark_longmemeval/dataset_maker/longmemeval_hierarchical/step2_L1_L2_summary_request.py"
if [[ ! -s "$STEP2_MODULE_FILE" ]]; then
    echo "[longmemeval_hierarchical] $STEP2_MODULE_FILE is empty; skipping Step 2."
    exit 0
fi

uv run python -m benchmark_longmemeval.dataset_maker.longmemeval_hierarchical.step2_L1_L2_summary_request
