#!/usr/bin/env bash
# Purpose: rerun only LongMemEval GPT-5 cases whose prior individual reports
# contain an empty generated_answer, writing outputs to a separate directory.
# Runs: benchmark_longmemeval.task_eval.benchmark_triple.
set -euo pipefail

# Rerun only GPT-5 samples whose existing individual report has an empty generated_answer.
# Results are written to a separate directory first; this script does not overwrite the original run.

BASE_DIR="benchmark_longmemeval/task_eval/results/ablations/gpt5/full_tri_tower"
OUTPUT_DIR="benchmark_longmemeval/task_eval/results/ablations/gpt5/full_tri_tower_empty_rerun_max8192"
LOG_DIR="logs/longmemeval_gpt5_empty_rerun"
LOG_FILE="${LOG_DIR}/rerun_empty_gpt5_max8192.log"

LLM_MODEL="gpt-5-closeai"
EVAL_MODEL="gpt-4o-mini-closeai"
GENERATION_MAX_TOKENS=8192

mkdir -p "${LOG_DIR}"

EMPTY_IDS="$(
  UV_NO_SYNC=1 uv run python - "${BASE_DIR}" <<'PY' | tail -n 1
import json
import re
import sys
from pathlib import Path

base_dir = Path(sys.argv[1])
report_dir = base_dir / "individual_reports"
ids = []

for path in sorted(report_dir.glob("qa_*_report.json")):
    match = re.search(r"qa_(\d+)_report\.json$", path.name)
    if not match:
        continue
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        continue
    if not str(report.get("generated_answer", "")).strip():
        ids.append(int(match.group(1)))

print(",".join(str(i) for i in sorted(ids)))
PY
)"

if [[ -z "${EMPTY_IDS}" ]]; then
  echo "No empty generated_answer reports found under ${BASE_DIR}."
  exit 0
fi

COUNT="$(awk -F',' '{print NF}' <<< "${EMPTY_IDS}")"
echo "Found ${COUNT} empty generations."
echo "Output dir: ${OUTPUT_DIR}"
echo "Log file: ${LOG_FILE}"

nohup uv run python -m benchmark_longmemeval.task_eval.benchmark_triple \
  --dataset-size s \
  --llm-model "${LLM_MODEL}" \
  --llm-evaluate-model "${EVAL_MODEL}" \
  --final-top-k 25 \
  --sample-ids "${EMPTY_IDS}" \
  --generation-max-tokens "${GENERATION_MAX_TOKENS}" \
  --output-dir "${OUTPUT_DIR}" \
  > "${LOG_FILE}" 2>&1 &

echo "Started rerun in background. PID: $!"
echo "Follow with: tail -f ${LOG_FILE}"
