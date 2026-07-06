#!/usr/bin/env bash
# Purpose: document that LongMemEval speed helpers are private in the public artifact.

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
    echo "uv is required to run this benchmark script" >&2
    exit 1
fi

cat <<'EOF'
LongMemEval speed-test helpers were moved to task_eval/private and are not part
of the public artifact workflow. The paper-facing public latency smoke path is
the LoCoMo QPS pipeline:

  bash benchmark_locomo/scripts/run_speed_benchmarks.sh

For LongMemEval accuracy reproduction, use router + quantification scripts such
as benchmark_longmemeval/scripts/run_router_quantification_cascade_closeai.sh.
EOF
