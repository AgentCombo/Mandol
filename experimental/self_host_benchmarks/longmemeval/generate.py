#!/usr/bin/env python3
"""Step 3: Generate - Produce answers using retrieved context and LLM.

Each LongMemEval sample has one question.  Loads the retrieval result,
constructs prompt from top-k context, calls the LLM, and writes the answer.

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
    save_json,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def generate_single_sample(
    question_id: str,
    output_dir: Path,
    max_tokens: int,
    temperature: float,
    force: bool = False,
) -> None:
    """Generate an answer for a single LongMemEval sample (1 question).

    Args:
        question_id: The sample's question_id.
        output_dir: Root output directory.
        max_tokens: Max tokens for LLM generation.
        temperature: LLM sampling temperature.
        force: If True, regenerate even if results exist.
    """
    result_path = output_dir / question_id / "generation.json"

    if force and result_path.exists():
        result_path.unlink()

    if not force and result_path.exists():
        logger.info("Skipping %s: generation already completed", question_id)
        return

    retrieval_path = output_dir / question_id / "retrieval.json"
    retrieval = load_json(retrieval_path)
    if retrieval is None:
        logger.error("Retrieval results not found for %s, run retrieve.py first", question_id)
        return

    system = MemorySystem.load(str(output_dir / question_id / "graph"))
    system.reset_token_usage()

    question = retrieval["question"]
    answer = retrieval.get("answer", "")
    question_type = retrieval.get("question_type", "")

    context_parts = []
    for j, hit in enumerate(retrieval.get("top_k_hits", [])):
        text = hit.get("text_content", "")
        if text:
            context_parts.append(f"[{j + 1}] {text}")
    context = "\n".join(context_parts)

    prompt = GENERATION_PROMPT_TEMPLATE.format(
        question=question,
        context=context,
    )

    t0 = time.time()
    response = system.llm.chat(
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    elapsed = time.time() - t0

    result = {
        "question_id": question_id,
        "question": question,
        "answer": answer,
        "question_type": question_type,
        "generated_answer": response.content,
        "generated_answer_extracted": extract_final_answer(response.content),
        "generation_time_seconds": round(elapsed, 4),
        "token_usage": response.usage,
    }
    save_json(result_path, result)

    total_usage = system.get_token_usage()
    logger.info("  %s: generated in %.1fs, tokens=%s", question_id, elapsed, total_usage)


def main():
    parser = argparse.ArgumentParser(description="LongMemEval Benchmark - Step 3: Generate")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to config YAML")
    parser.add_argument("--data", type=str, default=None, help="Path to LongMemEval dataset (overrides config)")
    parser.add_argument("--output", type=str, default=None, help="Output directory (overrides config)")
    parser.add_argument("--force", action="store_true", help="Force regeneration even if results exist")
    args = parser.parse_args()

    cfg = load_config(args.config)
    experiment = cfg.get("experiment", {})
    generation_cfg = cfg.get("generation", {})

    dataset_path = args.data or experiment.get("dataset_path", "data/longmemeval_s_cleaned.json")
    if not Path(dataset_path).is_absolute():
        dataset_path = str((Path(__file__).resolve().parent / dataset_path).resolve())
    output_dir = Path(args.output or experiment.get("output_dir", "output"))
    config_name = experiment.get("config_name", "default")
    output_dir = output_dir / config_name

    max_tokens = generation_cfg.get("max_tokens", 256)
    temperature = generation_cfg.get("temperature", 0.3)

    question_ids_override = experiment.get("question_ids", [])
    samples = load_dataset(dataset_path, question_ids_override or None)

    logger.info("Output directory: %s", output_dir)
    logger.info("max_tokens=%d, temperature=%.2f", max_tokens, temperature)

    for sample in samples:
        qid = sample["question_id"]
        generate_single_sample(
            question_id=qid,
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
