#!/usr/bin/env bash
# Purpose: verify that the main LoCoMo task-eval entrypoints import and expose
# --help before launching expensive reproduction jobs.
# Runs: lightweight Python module checks only; no benchmark data is evaluated.
set -uo pipefail

AMS_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AMS_REPO_ROOT="$AMS_SCRIPT_DIR"
while [[ "$AMS_REPO_ROOT" != "/" && ! -d "$AMS_REPO_ROOT/src/mandol" ]]; do
    AMS_REPO_ROOT="$(dirname "$AMS_REPO_ROOT")"
done
if [[ ! -d "$AMS_REPO_ROOT/src/mandol" ]]; then
    echo "FAIL repo-root: could not locate AgentMemorySystem repo root from $AMS_SCRIPT_DIR" >&2
    exit 1
fi

cd "$AMS_REPO_ROOT"
if [[ -d "$AMS_REPO_ROOT/.venv/bin" ]]; then
    export PATH="$AMS_REPO_ROOT/.venv/bin:$PATH"
fi
export PYTHONPATH="$AMS_REPO_ROOT/src:$AMS_REPO_ROOT:${PYTHONPATH:-}"

if command -v uv >/dev/null 2>&1; then
    PYTHON_CMD=(uv run python)
elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD=(python)
else
    echo "FAIL python: neither uv nor python is available" >&2
    exit 1
fi

MODULES=(
    benchmark_locomo.task_eval.locomo_triple
    benchmark_locomo.task_eval.locomo_triple_router
    benchmark_locomo.task_eval.locomo_triple_router_quantification
    benchmark_locomo.task_eval.locomo_triple_input_speed
    benchmark_locomo.task_eval.locomo_triple_smart_search_qps
)

status=0
for module in "${MODULES[@]}"; do
    printf 'CHECK %-70s ' "$module"
    output_file="$(mktemp)"
    if "${PYTHON_CMD[@]}" -m "$module" --help >"$output_file" 2>&1; then
        echo "PASS"
    else
        echo "FAIL"
        sed -n '1,120p' "$output_file" | sed 's/^/    /'
        status=1
    fi
    rm -f "$output_file"
done

exit "$status"
