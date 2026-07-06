#!/usr/bin/env bash
set -euo pipefail

# Purpose: Generate LongMemEval answers from retrieved self-host Mandol contexts.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

print_repro_config
build_sample_args

CMD=(
    "$PYTHON" "$SELF_HOST_DIR/generate.py"
    --retrieval-dir "$RUN_ROOT/retrieve"
    --output-dir "$RUN_ROOT/generate"
    --model "$GENERATION_MODEL"
    --save-prompts
)
if [[ -n "${MAX_QUESTIONS:-}" ]]; then
    CMD+=(--max-questions "$MAX_QUESTIONS")
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    CMD+=(--dry-run)
fi
CMD+=("${SAMPLE_ARGS[@]}")
append_extra_args GENERATE_EXTRA_ARGS
CMD+=("${EXTRA_ARGS[@]}")

run_logged "$LOG_DIR/generate.log" "${CMD[@]}"
