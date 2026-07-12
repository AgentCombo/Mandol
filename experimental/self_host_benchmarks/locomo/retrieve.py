#!/usr/bin/env python3
"""Step 2: Retrieve - Load built graphs and retrieve context for each QA query.

For each sample, loads the persisted MemorySystem and runs holistic_retrieve
on every question.  Results are written per-query for fine-grained resume.

Usage:
    python retrieve.py --config configs/base.yaml [--output output/] [--force]
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from mandol import MemorySystem

from adapter.locomo_adapter import load_locomo_sample
from pipeline_utils import (
    filter_qa,
    is_graph_built,
    load_config,
    load_dataset,
    load_or_init_results,
    update_results_file,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def retrieve_single_sample(
    sample_id: str,
    dataset_path: str,
    output_dir: Path,
    top_k: int,
    skip_views: list,
    skip_categories: list,
    force: bool = False,
) -> None:
    result_path = output_dir / sample_id / "retrieval.json"

    if force and result_path.exists():
        result_path.unlink()
        logger.info("Force mode: deleted existing results for %s", sample_id)

    if not force and result_path.exists():
        import json
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("status") == "completed":
            logger.info("Skipping %s: retrieval already completed", sample_id)
            return

    if not is_graph_built(output_dir, sample_id):
        logger.error("Graph not built for %s, run build_graph.py first", sample_id)
        return

    sample = load_locomo_sample(dataset_path=dataset_path, sample_id=sample_id)
    qa_list = filter_qa(sample.get("qa", []), skip_categories)
    total_queries = len(qa_list)

    if total_queries == 0:
        logger.info("No queries to retrieve for %s (all filtered)", sample_id)
        return

    results, completed = load_or_init_results(result_path, total_queries)
    logger.info("Retrieving %s: %d queries (%d already done)", sample_id, total_queries, completed)

    system = MemorySystem.load(str(output_dir / sample_id / "graph"))

    for i, qa in enumerate(qa_list):
        if i < completed:
            continue

        t0 = time.time()
        hits = system.holistic_retrieve(
            query=qa["question"],
            top_k=top_k,
            use_rerank=True,
            auto_build_if_empty=False,
            skip_views=skip_views or None,
        )
        elapsed = time.time() - t0

        results.append({
            "question": qa["question"],
            "answer": qa.get("answer", ""),
            "category": qa.get("category", 0),
            "evidence": qa.get("evidence", ""),
            "retrieval_time_seconds": round(elapsed, 4),
            "top_k_hits": [
                {
                    "uid": str(hit.unit.uid),
                    "text_content": hit.unit.raw_data.get("text_content", ""),
                    "final_score": hit.final_score,
                    "scores": {k: round(v, 6) for k, v in hit.scores.items()},
                    "ranks": dict(hit.ranks),
                }
                for hit in hits
            ],
        })

        update_results_file(result_path, sample_id, results, total_queries)

        if (i + 1) % 10 == 0 or i + 1 == total_queries:
            logger.info("  %s: %d/%d queries retrieved", sample_id, i + 1, total_queries)


def main():
    parser = argparse.ArgumentParser(description="LoCoMo Benchmark - Step 2: Retrieve")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to config YAML")
    parser.add_argument("--data", type=str, default=None, help="Path to LoCoMo dataset (overrides config)")
    parser.add_argument("--output", type=str, default=None, help="Output directory (overrides config)")
    parser.add_argument("--force", action="store_true", help="Force re-retrieve even if results exist")
    args = parser.parse_args()

    cfg = load_config(args.config)
    experiment = cfg.get("experiment", {})
    retrieval_cfg = cfg.get("retrieval", {})

    dataset_path = args.data or experiment.get("dataset_path", "data/locomo10.json")
    output_dir = Path(args.output or experiment.get("output_dir", "output"))
    config_name = experiment.get("config_name", "default")
    output_dir = output_dir / config_name

    top_k = retrieval_cfg.get("top_k", 10)
    skip_views = retrieval_cfg.get("skip_views", [])
    skip_categories = experiment.get("skip_categories", [5])

    sample_ids_override = experiment.get("sample_ids", [])
    samples = load_dataset(dataset_path, sample_ids_override or None)

    logger.info("Output directory: %s", output_dir)
    logger.info("top_k=%d, skip_views=%s, skip_categories=%s", top_k, skip_views, skip_categories)

    for sample in samples:
        sid = sample["sample_id"]
        retrieve_single_sample(
            sample_id=sid,
            dataset_path=dataset_path,
            output_dir=output_dir,
            top_k=top_k,
            skip_views=skip_views,
            skip_categories=skip_categories,
            force=args.force,
        )

    logger.info("Retrieval complete for all samples")


if __name__ == "__main__":
    main()
