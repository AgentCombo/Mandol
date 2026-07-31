#!/usr/bin/env bash
set -euo pipefail

# Purpose: Retrieve LongMemEval contexts from the self-host Mandol memory artifacts.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

print_repro_config
build_sample_args
append_range_args

CMD=(
    "$PYTHON" "$SELF_HOST_DIR/retrieve.py"
    --memory-dir "$RUN_ROOT/memory"
    --output-dir "$RUN_ROOT/retrieve"
)
CMD+=("${SAMPLE_ARGS[@]}" "${RANGE_ARGS[@]}")
append_extra_args RETRIEVE_EXTRA_ARGS
CMD+=("${EXTRA_ARGS[@]}")

run_logged "$LOG_DIR/retrieve.log" "${CMD[@]}"
