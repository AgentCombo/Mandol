#!/usr/bin/env bash
set -euo pipefail

# Purpose: Generate LongMemEval entity-relation batch requests for all QA index ranges using uv.

# Resolve repository root for the src/ package layout.
AMS_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AMS_REPO_ROOT="$AMS_SCRIPT_DIR"
while [[ "$AMS_REPO_ROOT" != "/" && ! -d "$AMS_REPO_ROOT/src/mandol" ]]; do
    AMS_REPO_ROOT="$(dirname "$AMS_REPO_ROOT")"
done
if [[ ! -d "$AMS_REPO_ROOT/src/mandol" ]]; then
    echo "Could not locate Mandol repo root from $AMS_SCRIPT_DIR" >&2
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

# 任务 1: QA 0-49
uv run python -m benchmark_longmemeval.dataset_maker.longmemeval_entity_relation_new.step1_entity_batch_requests --start-index 0 --end-index 49 --sessions-per-group 1

# 任务 2: QA 50-99
uv run python -m benchmark_longmemeval.dataset_maker.longmemeval_entity_relation_new.step1_entity_batch_requests --start-index 50 --end-index 99 --sessions-per-group 1

# 任务 3: QA 100-149
uv run python -m benchmark_longmemeval.dataset_maker.longmemeval_entity_relation_new.step1_entity_batch_requests --start-index 100 --end-index 149 --sessions-per-group 1

# 任务 4: QA 150-199
uv run python -m benchmark_longmemeval.dataset_maker.longmemeval_entity_relation_new.step1_entity_batch_requests --start-index 150 --end-index 199 --sessions-per-group 1

# 任务 5: QA 200-249
uv run python -m benchmark_longmemeval.dataset_maker.longmemeval_entity_relation_new.step1_entity_batch_requests --start-index 200 --end-index 249 --sessions-per-group 1

# 任务 6: QA 250-299
uv run python -m benchmark_longmemeval.dataset_maker.longmemeval_entity_relation_new.step1_entity_batch_requests --start-index 250 --end-index 299 --sessions-per-group 1

# 任务 7: QA 300-349
uv run python -m benchmark_longmemeval.dataset_maker.longmemeval_entity_relation_new.step1_entity_batch_requests --start-index 300 --end-index 349 --sessions-per-group 1

# 任务 8: QA 350-399
uv run python -m benchmark_longmemeval.dataset_maker.longmemeval_entity_relation_new.step1_entity_batch_requests --start-index 350 --end-index 399 --sessions-per-group 1

# 任务 9: QA 400-449
uv run python -m benchmark_longmemeval.dataset_maker.longmemeval_entity_relation_new.step1_entity_batch_requests --start-index 400 --end-index 449 --sessions-per-group 1

# 任务 10: QA 450-499
uv run python -m benchmark_longmemeval.dataset_maker.longmemeval_entity_relation_new.step1_entity_batch_requests --start-index 450 --end-index 499 --sessions-per-group 1
