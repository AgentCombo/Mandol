#!/usr/bin/env bash
set -euo pipefail

# Purpose: Build Mandol self-host memory artifacts for the LongMemEval reproduction.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

print_repro_config
build_sample_args
append_range_args

CMD=("$PYTHON" "$SELF_HOST_DIR/build_graph.py" --output-dir "$RUN_ROOT/memory")
if [[ "${FORCE:-0}" == "1" ]]; then
    CMD+=(--force)
fi
if [[ "${SKIP_EXISTING:-0}" == "1" ]]; then
    CMD+=(--skip-existing)
fi
if [[ "${NO_RESUME:-0}" == "1" ]]; then
    CMD+=(--no-resume)
fi
if [[ "${ALLOW_PARTIAL_EXTRACTION:-1}" == "1" ]]; then
    CMD+=(--allow-partial-extraction)
fi
CMD+=("${SAMPLE_ARGS[@]}" "${RANGE_ARGS[@]}")
append_extra_args BUILD_EXTRA_ARGS
CMD+=("${EXTRA_ARGS[@]}")

run_logged "$LOG_DIR/build.log" "${CMD[@]}"
