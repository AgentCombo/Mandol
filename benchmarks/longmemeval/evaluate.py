#!/usr/bin/env python3
"""Step 4: Evaluate - LLM judge evaluation of generated answers.

Each LongMemEval sample has one question.  Loads the generation result,
runs LLM judge, and computes per-type accuracy.

Usage:
    python evaluate.py --config configs/base.yaml [--output output/] [--force]
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from mandol import MemorySystem

from pipeline_utils import (
    EVALUATION_PROMPT_TEMPLATE,
    aggregate_llm_judge_accuracy,
    generate_report,
    load_config,
    load_dataset,
    load_json,
    parse_judge_label,
    save_json,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def evaluate_single_sample(
    question_id: str,
    output_dir: Path,
    force: bool = False,
) -> dict:
    """Run LLM judge evaluation for a single LongMemEval sample (1 question).

    Args:
        question_id: The sample's question_id.
        output_dir: Root output directory.
        force: If True, re-evaluate even if results exist.

    Returns:
        A dict with evaluation stats.
    """
    result_path = output_dir / question_id / "evaluation.json"

    if force and result_path.exists():
        result_path.unlink()
        logger.info("Force mode: deleted existing results for %s", question_id)

    if not force and result_path.exists():
        logger.info("Skipping %s: evaluation already completed", question_id)
        return {"question_id": question_id, "status": "skipped"}

    generation_path = output_dir / question_id / "generation.json"
    generation = load_json(generation_path)
    if generation is None:
        logger.error("Generation results not found for %s, run generate.py first", question_id)
        return {"question_id": question_id, "status": "error"}

    system = MemorySystem.load(str(output_dir / question_id / "graph"))
    system.reset_token_usage()

    question = generation["question"]
    gold_answer = generation.get("answer", "")
    generated_answer = generation.get("generated_answer_extracted") or generation.get("generated_answer", "")

    prompt = EVALUATION_PROMPT_TEMPLATE.format(
        question=question,
        gold_answer=gold_answer,
        generated_answer=generated_answer,
    )

    t0 = time.time()
    response = system.llm.chat(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=100,
    )
    elapsed = time.time() - t0

    label = parse_judge_label(response.content)
    accuracy = 1.0 if label == "CORRECT" else 0.0

    result = {
        "question_id": question_id,
        "question": question,
        "question_type": generation.get("question_type", ""),
        "gold_answer": gold_answer,
        "generated_answer": generated_answer,
        "llm_judge_label": label,
        "llm_judge_accuracy": accuracy,
        "llm_judge_raw": response.content,
        "evaluation_time_seconds": round(elapsed, 4),
        "token_usage": response.usage,
    }
    save_json(result_path, result)
    logger.info("  %s: judge=%s (%.1fs)", question_id, label, elapsed)

    total_usage = system.get_token_usage()
    return {
        "question_id": question_id,
        "status": "completed",
        "queries_processed": 1,
        "duration_seconds": round(elapsed, 3),
        "token_usage": total_usage,
    }


def main():
    parser = argparse.ArgumentParser(description="LongMemEval Benchmark - Step 4: Evaluate")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to config YAML")
    parser.add_argument("--data", type=str, default=None, help="Path to LongMemEval dataset (overrides config)")
    parser.add_argument("--output", type=str, default=None, help="Output directory (overrides config)")
    parser.add_argument("--force", action="store_true", help="Force re-evaluation even if results exist")
    args = parser.parse_args()

    cfg = load_config(args.config)
    experiment = cfg.get("experiment", {})
    evaluation_cfg = cfg.get("evaluation", {})

    dataset_path = args.data or experiment.get("dataset_path", "data/longmemeval_s_cleaned.json")
    if not Path(dataset_path).is_absolute():
        dataset_path = str((Path(__file__).resolve().parent / dataset_path).resolve())
    output_dir = Path(args.output or experiment.get("output_dir", "output"))
    config_name = experiment.get("config_name", "default")
    output_dir = output_dir / config_name

    question_ids_override = experiment.get("question_ids", [])
    samples = load_dataset(dataset_path, question_ids_override or None)
    question_ids = [s["question_id"] for s in samples]

    logger.info("Output directory: %s", output_dir)

    all_results = []
    for idx, sample in enumerate(samples, 1):
        qid = sample["question_id"]
        logger.info("[%d/%d] Processing sample: %s", idx, len(samples), qid)
        result = evaluate_single_sample(
            question_id=qid,
            output_dir=output_dir,
            force=args.force,
        )
        all_results.append(result)

    total_queries = sum(r.get("queries_processed", 0) for r in all_results)
    total_duration = sum(r.get("duration_seconds", 0) for r in all_results)
    total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for r in all_results:
        for k in total_tokens:
            total_tokens[k] += r.get("token_usage", {}).get(k, 0)

    accuracy_summary = aggregate_llm_judge_accuracy(output_dir, question_ids)
    accuracy_summary["total_samples"] = len(samples)
    accuracy_summary["completed_samples"] = sum(1 for r in all_results if r.get("status") == "completed")
    accuracy_summary["skipped_samples"] = sum(1 for r in all_results if r.get("status") == "skipped")
    accuracy_summary["error_samples"] = sum(1 for r in all_results if r.get("status") == "error")
    accuracy_summary["total_queries_evaluated"] = total_queries
    accuracy_summary["total_duration_seconds"] = round(total_duration, 3)
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

# conda activate mandol && cd benchmarks/longmemeval/ && nohup python evaluate.py --config configs/base.yaml --force > results/evaluate.log 2>&1 &
