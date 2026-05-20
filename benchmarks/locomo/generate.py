#!/usr/bin/env python3
"""Step 3: Generate - Produce answers using retrieved context and LLM.

Loads retrieval results, constructs prompts from top-k context, calls
the LLM, and records generated answers with token usage.  Per-query
resume is supported.

Usage:
    python generate.py --config configs/base.yaml [--output output/] [--force]
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from pipeline_utils import (
    GENERATION_PROMPT_TEMPLATE,
    build_llm_provider_from_config,
    extract_final_answer,
    load_config,
    load_dataset,
    load_json,
    load_or_init_results,
    save_json,
    update_results_file,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def generate_single_sample(
    sample_id: str,
    output_dir: Path,
    llm,
    max_tokens: int,
    temperature: float,
    force: bool = False,
) -> dict:
    result_path = output_dir / sample_id / "generation.json"

    if force and result_path.exists():
        result_path.unlink()
        logger.info("Force mode: deleted existing results for %s", sample_id)

    if not force and result_path.exists():
        import json
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("status") == "completed":
            logger.info("Skipping %s: generation already completed", sample_id)
            return {"sample_id": sample_id, "status": "skipped", "queries_processed": 0, "token_usage": {}}

    retrieval_path = output_dir / sample_id / "retrieval.json"
    retrieval = load_json(retrieval_path)
    if retrieval is None:
        logger.error("Retrieval results not found for %s, run retrieve.py first", sample_id)
        return {"sample_id": sample_id, "status": "error", "queries_processed": 0, "token_usage": {}}

    retrieval_results = retrieval.get("results", [])
    total_queries = len(retrieval_results)
    if total_queries == 0:
        logger.info("No queries to generate for %s", sample_id)
        return {"sample_id": sample_id, "status": "skipped", "queries_processed": 0, "token_usage": {}}

    results, completed = load_or_init_results(result_path, total_queries)
    logger.info("Generating %s: %d queries (%d already done)", sample_id, total_queries, completed)

    t0 = time.time()
    accumulated_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for i, item in enumerate(retrieval_results):
        if i < completed:
            continue

        context_parts = []
        for j, hit in enumerate(item.get("top_k_hits", [])):
            text = hit.get("text_content", "")
            if text:
                context_parts.append(f"[{j + 1}] {text}")
        context = "\n".join(context_parts)

        prompt = GENERATION_PROMPT_TEMPLATE.replace("{question}", item["question"]).replace("{context}", context)

        q_t0 = time.time()
        response = llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        q_elapsed = time.time() - q_t0

        usage = response.usage or {}
        for k in accumulated_tokens:
            accumulated_tokens[k] += usage.get(k, 0)

        results.append({
            "question": item["question"],
            "answer": item.get("answer", ""),
            "category": item.get("category", 0),
            "evidence": item.get("evidence", ""),
            "generated_answer": response.content,
            "generated_answer_extracted": extract_final_answer(response.content),
            "generation_time_seconds": round(q_elapsed, 4),
            "token_usage": usage,
        })

        update_results_file(result_path, sample_id, results, total_queries)

        if (i + 1) % 10 == 0 or i + 1 == total_queries:
            logger.info("  %s: %d/%d queries generated", sample_id, i + 1, total_queries)

    elapsed = time.time() - t0
    queries_done = len(results) - completed
    logger.info("  %s generation: %d queries in %.1fs, token usage: %s", sample_id, queries_done, elapsed, accumulated_tokens)
    return {
        "sample_id": sample_id,
        "status": "completed",
        "queries_processed": queries_done,
        "duration_seconds": round(elapsed, 3),
        "token_usage": accumulated_tokens,
    }


def main():
    parser = argparse.ArgumentParser(description="LoCoMo Benchmark - Step 3: Generate")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to config YAML")
    parser.add_argument("--output", type=str, default=None, help="Output directory (overrides config)")
    parser.add_argument("--force", action="store_true", help="Force regeneration even if results exist")
    args = parser.parse_args()

    cfg = load_config(args.config)
    experiment = cfg.get("experiment", {})
    generation_cfg = cfg.get("generation", {})

    output_dir = Path(args.output or experiment.get("output_dir", "output"))
    config_name = experiment.get("config_name", "default")
    output_dir = output_dir / config_name

    max_tokens = generation_cfg.get("max_tokens", 256)
    temperature = generation_cfg.get("temperature", 0.3)

    sample_ids_override = experiment.get("sample_ids", [])
    dataset_path = experiment.get("dataset_path", "data/locomo10.json")
    samples = load_dataset(dataset_path, sample_ids_override or None)

    llm = build_llm_provider_from_config(args.config)

    logger.info("Output directory: %s", output_dir)
    logger.info("max_tokens=%d, temperature=%.2f", max_tokens, temperature)

    all_results = []
    for idx, sample in enumerate(samples, 1):
        sid = sample["sample_id"]
        logger.info("[%d/%d] Processing sample: %s", idx, len(samples), sid)
        result = generate_single_sample(
            sample_id=sid,
            output_dir=output_dir,
            llm=llm,
            max_tokens=max_tokens,
            temperature=temperature,
            force=args.force,
        )
        all_results.append(result)

    total_queries = sum(r.get("queries_processed", 0) for r in all_results)
    total_duration = sum(r.get("duration_seconds", 0) for r in all_results)
    total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for r in all_results:
        for k in total_tokens:
            total_tokens[k] += r.get("token_usage", {}).get(k, 0)

    stats = {
        "config_name": config_name,
        "total_samples": len(samples),
        "completed_samples": sum(1 for r in all_results if r.get("status") == "completed"),
        "skipped_samples": sum(1 for r in all_results if r.get("status") == "skipped"),
        "error_samples": sum(1 for r in all_results if r.get("status") == "error"),
        "total_queries_generated": total_queries,
        "total_duration_seconds": round(total_duration, 3),
        "avg_query_seconds": round(total_duration / max(total_queries, 1), 3),
        "total_token_usage": total_tokens,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    save_json(output_dir / "generation_stats.json", stats)
    logger.info("Generation complete: %s", stats)


if __name__ == "__main__":
    main()

# conda activate mandol && cd benchmarks/locomo/ && nohup python generate.py --config configs/base.yaml --force > results/generate.log 2>&1 &
