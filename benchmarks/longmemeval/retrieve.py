#!/usr/bin/env python3
"""Step 2: Retrieve - Load built graphs and retrieve context for each question.

Each LongMemEval sample contains exactly one question.  Loads the persisted
MemorySystem, runs holistic_retrieve, and writes the result.

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

from pipeline_utils import (
    is_graph_built,
    load_config,
    load_dataset,
    save_json,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def retrieve_single_sample(
    question_id: str,
    question: str,
    answer: str,
    question_type: str,
    output_dir: Path,
    top_k: int,
    skip_views: list,
    force: bool = False,
) -> None:
    """Run holistic_retrieve for a single LongMemEval sample (1 question).

    Args:
        question_id: The sample's question_id.
        question: The question text.
        answer: The ground-truth answer.
        question_type: The question type (e.g. "single-session-user").
        output_dir: Root output directory.
        top_k: Number of hits to retrieve.
        skip_views: Optional views to skip in retrieval.
        force: If True, re-retrieve even if results exist.
    """
    result_path = output_dir / question_id / "retrieval.json"

    if force and result_path.exists():
        result_path.unlink()
        logger.info("Force mode: deleted existing results for %s", question_id)

    if not force and result_path.exists():
        logger.info("Skipping %s: retrieval already completed", question_id)
        return

    if not is_graph_built(output_dir, question_id):
        logger.error("Graph not built for %s, run build_graph.py first", question_id)
        return

    system = MemorySystem.load(str(output_dir / question_id / "graph"))

    t0 = time.time()
    hits = system.holistic_retrieve(
        query=question,
        top_k=top_k,
        use_rerank=True,
        auto_build_if_empty=False,
        skip_views=skip_views or None,
    )
    elapsed = time.time() - t0

    result = {
        "question_id": question_id,
        "question": question,
        "answer": answer,
        "question_type": question_type,
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
    }
    save_json(result_path, result)
    logger.info("  %s: retrieved %d hits (%.1fs)", question_id, len(hits), elapsed)


def main():
    parser = argparse.ArgumentParser(description="LongMemEval Benchmark - Step 2: Retrieve")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to config YAML")
    parser.add_argument("--data", type=str, default=None, help="Path to LongMemEval dataset (overrides config)")
    parser.add_argument("--output", type=str, default=None, help="Output directory (overrides config)")
    parser.add_argument("--force", action="store_true", help="Force re-retrieve even if results exist")
    args = parser.parse_args()

    cfg = load_config(args.config)
    experiment = cfg.get("experiment", {})
    retrieval_cfg = cfg.get("retrieval", {})

    dataset_path = args.data or experiment.get("dataset_path", "data/longmemeval_s_cleaned.json")
    if not Path(dataset_path).is_absolute():
        dataset_path = str((Path(__file__).resolve().parent / dataset_path).resolve())
    output_dir = Path(args.output or experiment.get("output_dir", "output"))
    config_name = experiment.get("config_name", "default")
    output_dir = output_dir / config_name

    top_k = retrieval_cfg.get("top_k", 10)
    skip_views = retrieval_cfg.get("skip_views", [])

    question_ids_override = experiment.get("question_ids", [])
    samples = load_dataset(dataset_path, question_ids_override or None)

    logger.info("Output directory: %s", output_dir)
    logger.info("top_k=%d, skip_views=%s", top_k, skip_views)
    logger.info("Samples to process: %d", len(samples))

    for sample in samples:
        qid = sample["question_id"]
        retrieve_single_sample(
            question_id=qid,
            question=sample["question"],
            answer=sample.get("answer", ""),
            question_type=sample.get("question_type", ""),
            output_dir=output_dir,
            top_k=top_k,
            skip_views=skip_views,
            force=args.force,
        )

    logger.info("Retrieval complete for all samples")


if __name__ == "__main__":
    main()

# conda activate mandol && cd benchmarks/longmemeval/ && nohup python retrieve.py --config configs/base.yaml --force > results/retrieve.log 2>&1 &
