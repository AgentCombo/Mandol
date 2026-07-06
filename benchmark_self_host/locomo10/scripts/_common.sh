#!/usr/bin/env bash

# Purpose: Define shared environment, paths, defaults, and helper functions for LoCoMo10 self-host scripts.

AMS_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF_HOST_DIR="$(cd "$AMS_SCRIPT_DIR/.." && pwd)"
AMS_REPO_ROOT="$SELF_HOST_DIR"
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

if [[ -z "${PYTHON:-}" ]]; then
    if [[ -x "$AMS_REPO_ROOT/.venv/bin/python" ]]; then
        PYTHON="$AMS_REPO_ROOT/.venv/bin/python"
    else
        PYTHON="python"
    fi
fi

GENERATION_MODEL="${GENERATION_MODEL:-gpt-4o-mini-closeai}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o-mini-closeai}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ID="${RUN_ID:-${RUN_STAMP}__gen-${GENERATION_MODEL}__judge-${JUDGE_MODEL}}"
RUN_ROOT="${RUN_ROOT:-$SELF_HOST_DIR/test_runs/$RUN_ID}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
JUDGE_WORKERS="${JUDGE_WORKERS:-1}"

mkdir -p "$RUN_ROOT" "$LOG_DIR"

build_sample_args() {
    SAMPLE_ARGS=()
    if [[ -n "${SAMPLE_IDS:-}" ]]; then
        read -r -a _sample_ids <<< "$SAMPLE_IDS"
        SAMPLE_ARGS+=(--sample-ids "${_sample_ids[@]}")
    fi
}

append_extra_args() {
    local var_name="$1"
    local value="${!var_name:-}"
    EXTRA_ARGS=()
    if [[ -n "$value" ]]; then
        # shellcheck disable=SC2206
        EXTRA_ARGS=( $value )
    fi
}

print_repro_config() {
    echo "Repo root       : $AMS_REPO_ROOT"
    echo "Run root        : $RUN_ROOT"
    echo "Generation model: $GENERATION_MODEL"
    echo "Judge model     : $JUDGE_MODEL"
    echo "Python          : $PYTHON"
    if [[ -n "${SAMPLE_IDS:-}" ]]; then
        echo "Sample IDs      : $SAMPLE_IDS"
    fi
}

run_logged() {
    local log_file="$1"
    shift
    echo
    printf '+ '
    printf '%q ' "$@"
    echo
    "$@" 2>&1 | tee "$log_file"
}
