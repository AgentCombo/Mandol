#!/usr/bin/env bash
set -euo pipefail

# Purpose: Run the complete LongMemEval self-host reproduction pipeline: build, retrieve, generate, and score.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
export RUN_ROOT GENERATION_MODEL JUDGE_MODEL PYTHONPATH PYTHON LOG_DIR JUDGE_WORKERS

print_repro_config
echo "Starting LongMemEval self-host reproduction."

bash "$SCRIPT_DIR/run_build.sh"
bash "$SCRIPT_DIR/run_retrieve.sh"
bash "$SCRIPT_DIR/run_generate.sh"
bash "$SCRIPT_DIR/run_score.sh"

echo
echo "LongMemEval reproduction finished."
echo "Run root: $RUN_ROOT"
