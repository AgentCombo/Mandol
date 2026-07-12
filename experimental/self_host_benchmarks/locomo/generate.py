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

from mandol import MemorySystem

from pipeline_utils import (
    GENERATION_PROMPT_TEMPLATE,
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
    max_tokens: int,
    temperature: float,
    force: bool = False,
) -> None:
    result_path = output_dir / sample_id / "generation.json"

    if not force and result_path.exists():
        import json
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("status") == "completed":
            logger.info("Skipping %s: generation already completed", sample_id)
            return

    retrieval_path = output_dir / sample_id / "retrieval.json"
    retrieval = load_json(retrieval_path)
    if retrieval is None:
        logger.error("Retrieval results not found for %s, run retrieve.py first", sample_id)
        return

    retrieval_results = retrieval.get("results", [])
    total_queries = len(retrieval_results)
    if total_queries == 0:
        logger.info("No queries to generate for %s", sample_id)
        return

    results, completed = load_or_init_results(result_path, total_queries)
    logger.info("Generating %s: %d queries (%d already done)", sample_id, total_queries, completed)

    system = MemorySystem.load(str(output_dir / sample_id / "graph"))
    system.reset_token_usage()

    for i, item in enumerate(retrieval_results):
        if i < completed:
            continue

        context_parts = []
        for j, hit in enumerate(item.get("top_k_hits", [])):
            text = hit.get("text_content", "")
            if text:
                context_parts.append(f"[{j + 1}] {text}")
        context = "\n".join(context_parts)

        prompt = GENERATION_PROMPT_TEMPLATE.format(
            question=item["question"],
            context=context,
        )

        t0 = time.time()
        response = system.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        elapsed = time.time() - t0

        results.append({
            "question": item["question"],
            "answer": item.get("answer", ""),
            "category": item.get("category", 0),
            "evidence": item.get("evidence", ""),
            "generated_answer": response.content,
            "generated_answer_extracted": extract_final_answer(response.content),
            "generation_time_seconds": round(elapsed, 4),
            "token_usage": response.usage,
        })

        update_results_file(result_path, sample_id, results, total_queries)

        if (i + 1) % 10 == 0 or i + 1 == total_queries:
            logger.info("  %s: %d/%d queries generated", sample_id, i + 1, total_queries)

    total_usage = system.get_token_usage()
    logger.info("  %s generation token usage: %s", sample_id, total_usage)


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

    logger.info("Output directory: %s", output_dir)
    logger.info("max_tokens=%d, temperature=%.2f", max_tokens, temperature)

    for sample in samples:
        sid = sample["sample_id"]
        generate_single_sample(
            sample_id=sid,
            output_dir=output_dir,
            max_tokens=max_tokens,
            temperature=temperature,
            force=args.force,
        )

    stats = {
        "config_name": config_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    save_json(output_dir / "generation_stats.json", stats)
    logger.info("Generation complete for all samples")


if __name__ == "__main__":
    main()
