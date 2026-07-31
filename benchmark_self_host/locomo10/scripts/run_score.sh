#!/usr/bin/env bash
set -euo pipefail

# Purpose: Score generated LoCoMo10 answers with the configured judge model.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

print_repro_config
build_sample_args

CMD=(
    "$PYTHON" "$SELF_HOST_DIR/score.py"
    --generation-dir "$RUN_ROOT/generate"
    --output-dir "$RUN_ROOT/score"
    --model "$JUDGE_MODEL"
    --workers "$JUDGE_WORKERS"
)
if [[ -n "${MAX_QUESTIONS:-}" ]]; then
    CMD+=(--max-questions "$MAX_QUESTIONS")
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    CMD+=(--dry-run)
fi
CMD+=("${SAMPLE_ARGS[@]}")
append_extra_args SCORE_EXTRA_ARGS
CMD+=("${EXTRA_ARGS[@]}")

run_logged "$LOG_DIR/score.log" "${CMD[@]}"
