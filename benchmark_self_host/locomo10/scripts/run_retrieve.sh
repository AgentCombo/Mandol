#!/usr/bin/env bash
set -euo pipefail

# Purpose: Retrieve LoCoMo10 contexts from the self-host Mandol memory artifacts.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

print_repro_config
build_sample_args

CMD=(
    "$PYTHON" "$SELF_HOST_DIR/retrieve.py"
    --memory-dir "$RUN_ROOT/memory"
    --output-dir "$RUN_ROOT/retrieve"
)
if [[ -n "${MAX_QUESTIONS:-}" ]]; then
    CMD+=(--max-questions "$MAX_QUESTIONS")
fi
CMD+=("${SAMPLE_ARGS[@]}")
append_extra_args RETRIEVE_EXTRA_ARGS
CMD+=("${EXTRA_ARGS[@]}")

run_logged "$LOG_DIR/retrieve.log" "${CMD[@]}"
