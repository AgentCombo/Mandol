#!/usr/bin/env python3
"""Step 4: Evaluate - LLM judge evaluation of generated answers.

Loads generation results, runs LLM judge for each answer, and computes
per-category accuracy.  Supports multiple judge runs per query to detect
misjudgments.  Per-query resume is supported.

Usage:
    python evaluate.py --config configs/base.yaml [--output output/] [--force]
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from pipeline_utils import (
    EVALUATION_PROMPT_TEMPLATE,
    aggregate_llm_judge_accuracy,
    build_llm_provider_from_config,
    generate_report,
    load_config,
    load_dataset,
    load_json,
    load_or_init_results,
    parse_judge_label,
    save_json,
    update_results_file,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def evaluate_single_sample(
    sample_id: str,
    output_dir: Path,
    llm,
    llm_judge_runs: int,
    force: bool = False,
) -> dict:
    result_path = output_dir / sample_id / "evaluation.json"

    if force and result_path.exists():
        result_path.unlink()
        logger.info("Force mode: deleted existing results for %s", sample_id)

    if not force and result_path.exists():
        import json
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("status") == "completed":
            logger.info("Skipping %s: evaluation already completed", sample_id)
            return {"sample_id": sample_id, "status": "skipped", "queries_processed": 0, "token_usage": {}}

    generation_path = output_dir / sample_id / "generation.json"
    generation = load_json(generation_path)
    if generation is None:
        logger.error("Generation results not found for %s, run generate.py first", sample_id)
        return {"sample_id": sample_id, "status": "error", "queries_processed": 0, "token_usage": {}}

    generation_results = generation.get("results", [])
    total_queries = len(generation_results)
    if total_queries == 0:
        logger.info("No queries to evaluate for %s", sample_id)
        return {"sample_id": sample_id, "status": "skipped", "queries_processed": 0, "token_usage": {}}

    results, completed = load_or_init_results(result_path, total_queries)
    logger.info("Evaluating %s: %d queries (%d already done)", sample_id, total_queries, completed)

    t0 = time.time()
    accumulated_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for i, item in enumerate(generation_results):
        if i < completed:
            continue

        question = item["question"]
        gold_answer = str(item.get("answer", ""))
        generated_answer = item.get("generated_answer_extracted") or item.get("generated_answer", "")

        prompt = (EVALUATION_PROMPT_TEMPLATE
            .replace("{question}", question)
            .replace("{gold_answer}", gold_answer)
            .replace("{generated_answer}", generated_answer))

        # Run the judge multiple times to detect misjudgments
        run_labels: list[str] = []
        run_raws: list[str] = []
        run_tokens: list[dict] = []
        run_times: list[float] = []

        for _ in range(max(1, llm_judge_runs)):
            q_t0 = time.time()
            response = llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=100,
            )
            q_elapsed = time.time() - q_t0

            label = parse_judge_label(response.content)
            run_labels.append(label)
            run_raws.append(response.content)
            run_times.append(round(q_elapsed, 4))
            usage = response.usage or {}
            run_tokens.append(usage)
            for k in accumulated_tokens:
                accumulated_tokens[k] += usage.get(k, 0)

        # Majority vote across runs
        label_counts = Counter(run_labels)
        majority_label = label_counts.most_common(1)[0][0]
        agreement = label_counts[majority_label] / len(run_labels) if run_labels else 1.0
        accuracy = 1.0 if majority_label == "CORRECT" else 0.0

        results.append({
            "question": question,
            "category": item.get("category", 0),
            "gold_answer": gold_answer,
            "generated_answer": generated_answer,
            "llm_judge_label": majority_label,
            "llm_judge_accuracy": accuracy,
            "llm_judge_agreement": round(agreement, 4),
            "llm_judge_runs": run_labels,
            "llm_judge_raw": run_raws[0] if len(run_raws) == 1 else run_raws,
            "evaluation_time_seconds": round(sum(run_times), 4),
            "token_usage": run_tokens[0] if len(run_tokens) == 1 else run_tokens,
        })

        update_results_file(result_path, sample_id, results, total_queries)

        if (i + 1) % 10 == 0 or i + 1 == total_queries:
            logger.info("  %s: %d/%d queries evaluated", sample_id, i + 1, total_queries)

    elapsed = time.time() - t0
    queries_done = len(results) - completed
    logger.info("  %s evaluation: %d queries in %.1fs, token usage: %s", sample_id, queries_done, elapsed, accumulated_tokens)
    return {
        "sample_id": sample_id,
        "status": "completed",
        "queries_processed": queries_done,
        "duration_seconds": round(elapsed, 3),
        "token_usage": accumulated_tokens,
    }


def main():
    parser = argparse.ArgumentParser(description="LoCoMo Benchmark - Step 4: Evaluate")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to config YAML")
    parser.add_argument("--output", type=str, default=None, help="Output directory (overrides config)")
    parser.add_argument("--force", action="store_true", help="Force re-evaluation even if results exist")
    args = parser.parse_args()

    cfg = load_config(args.config)
    experiment = cfg.get("experiment", {})
    evaluation_cfg = cfg.get("evaluation", {})

    output_dir = Path(args.output or experiment.get("output_dir", "output"))
    config_name = experiment.get("config_name", "default")
    output_dir = output_dir / config_name

    llm_judge_runs = int(evaluation_cfg.get("llm_judge_runs", 1))

    sample_ids_override = experiment.get("sample_ids", [])
    dataset_path = experiment.get("dataset_path", "data/locomo10.json")
    samples = load_dataset(dataset_path, sample_ids_override or None)
    sample_ids = [s["sample_id"] for s in samples]

    llm = build_llm_provider_from_config(args.config)

    logger.info("Output directory: %s", output_dir)
    logger.info("llm_judge_runs=%d", llm_judge_runs)

    all_results = []
    for idx, sample in enumerate(samples, 1):
        sid = sample["sample_id"]
        logger.info("[%d/%d] Processing sample: %s", idx, len(samples), sid)
        result = evaluate_single_sample(
            sample_id=sid,
            output_dir=output_dir,
            llm=llm,
            llm_judge_runs=llm_judge_runs,
            force=args.force,
        )
        all_results.append(result)

    total_queries = sum(r.get("queries_processed", 0) for r in all_results)
    total_duration = sum(r.get("duration_seconds", 0) for r in all_results)
    total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for r in all_results:
        for k in total_tokens:
            total_tokens[k] += r.get("token_usage", {}).get(k, 0)

    accuracy_summary = aggregate_llm_judge_accuracy(output_dir, sample_ids)
    accuracy_summary["total_samples"] = len(samples)
    accuracy_summary["completed_samples"] = sum(1 for r in all_results if r.get("status") == "completed")
    accuracy_summary["skipped_samples"] = sum(1 for r in all_results if r.get("status") == "skipped")
    accuracy_summary["error_samples"] = sum(1 for r in all_results if r.get("status") == "error")
    accuracy_summary["total_queries_evaluated"] = total_queries
    accuracy_summary["total_duration_seconds"] = round(total_duration, 3)
    accuracy_summary["llm_judge_runs"] = llm_judge_runs
    accuracy_summary["total_token_usage"] = total_tokens
    accuracy_summary["timestamp"] = datetime.now(timezone.utc).isoformat()

    save_json(output_dir / "evaluation_summary.json", accuracy_summary)
    generate_report(accuracy_summary, output_dir / "evaluation_report.txt")

    logger.info(
        "Evaluation complete. Accuracy: %.4f (%d questions), %d samples, duration: %.1fs, tokens: %s",
        accuracy_summary.get("accuracy", 0.0),
        accuracy_summary.get("total", 0),
        accuracy_summary.get("completed_samples", 0),
        total_duration,
        total_tokens,
    )


if __name__ == "__main__":
    main()

# conda activate mandol && cd benchmarks/locomo/ && nohup python evaluate.py --config configs/base.yaml --force > results/evaluate.log 2>&1 &
