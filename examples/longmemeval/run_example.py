#!/usr/bin/env python3
"""
LongMemEval Memory Evaluation Example — Long-Text Information Retention & Retrieval

This script demonstrates how to use the Mandol memory system to process
long articles and perform precise information retrieval. Supports both
built-in synthetic data and the full HuggingFace dataset.

Usage:
    python run_example.py                                # Use built-in synthetic data
    python run_example.py --data-dir data/               # Use HuggingFace data
    python run_example.py --query "What is OsteoDetect?" # Custom query
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from mandol import MemorySystem, MemoryUnit, Uid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_env_file() -> None:
    """Read a ``.env`` file next to this script and set unset env vars."""
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip()
            if key and value and key not in os.environ:
                os.environ[key] = value


def load_example_data(data_path: Path) -> dict:
    """Load a LongMemEval JSON data file.

    Args:
        data_path: Path to the JSON file.

    Returns:
        The parsed data dict containing ``sample_id``, ``passage``, and ``qa``.

    Exits with code 1 if the file does not exist.
    """
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        logger.error("Run 'python download_data.py' first or use the built-in synthetic data.")
        sys.exit(1)
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Loaded: {data.get('title', 'Unknown')} ({data['metadata']['word_count']} words, {len(data['qa'])} QAs)")
    return data


def process_passage(system: MemorySystem, data: dict) -> int:
    """Chunk a passage and add the chunks as memory units.

    Args:
        system: An initialized :class:`MemorySystem`.
        data: A dict with ``sample_id``, ``passage``, and optional ``title``.

    Returns:
        The number of chunks created.
    """
    sample_id = data["sample_id"]
    passage = data["passage"]
    title = data.get("title", "")

    chunks = _simple_chunk(passage, chunk_size=512)
    logger.info(f"Split passage into {len(chunks)} chunks")

    for i, chunk in enumerate(chunks):
        uid = f"{sample_id}_chunk_{i}"
        unit = MemoryUnit(
            uid=Uid(uid),
            raw_data={
                "text_content": f"[{title}] {chunk}",
                "chunk_index": i,
                "title": title,
            },
            metadata={
                "unit_type": "document_chunk",
                "sample_id": sample_id,
                "source": "longmemeval",
            },
        )
        system.add(unit)

    return len(chunks)


def _simple_chunk(text: str, chunk_size: int = 512) -> list[str]:
    """Split *text* into sentence-based chunks of up to *chunk_size* words.

    Args:
        text: The full passage text.
        chunk_size: Maximum number of words per chunk.

    Returns:
        A list of chunk strings.
    """
    sentences = text.replace("\n", " ").split(". ")
    chunks = []
    current = ""
    for s in sentences:
        if not s.strip():
            continue
        candidate = s
        if current:
            candidate = current + ". " + s
        if len(candidate.split()) > chunk_size:
            if current:
                chunks.append(current.strip() + ".")
            current = s
        else:
            current = candidate
    if current:
        chunks.append(current.strip())
    return chunks


def run_evaluation(system: MemorySystem, data: dict, top_k: int, use_rerank: bool) -> dict:
    """Evaluate QA pairs against the memory system and print a category breakdown.

    An answer is considered found when ≥ 50 % of its words appear in the
    top retrieved chunk.

    Args:
        system: An initialized :class:`MemorySystem`.
        data: A dict containing a ``qa`` list of question/answer pairs.
        top_k: Number of retrieval results per query.
        use_rerank: Whether to apply the reranker.

    Returns:
        A dict with ``total``, ``correct``, ``rate``, and ``details``.
    """
    qa_pairs = data.get("qa", [])
    results = []
    correct = 0

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    for i, qa in enumerate(qa_pairs):
        question = qa["question"]
        expected = qa["answer"]
        category = qa.get("category", "unknown")

        hits = system.holistic_retrieve(question, top_k=top_k, use_rerank=use_rerank)

        found = False
        best_score = 0.0
        best_text = ""
        for hit in hits:
            text_content = hit.unit.raw_data.get("text_content", "")
            if _contains_answer(text_content.lower(), expected.lower()):
                found = True
                best_score = max(best_score, hit.final_score)
                best_text = text_content[:100]
                break

        if found:
            correct += 1

        status = "PASS" if found else "FAIL"
        indicator = "+" if found else "-"
        ev_str = (
            f"[{indicator}] Q{i+1:02d} [{category:10s}] {question[:65]:65s}"
        )

        results.append({
            "question": question,
            "expected": expected,
            "found": found,
            "category": category,
            "best_score": best_score,
        })

        print(f"{ev_str}")

    category_stats = {}
    for r in results:
        cat = r["category"]
        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "correct": 0}
        category_stats[cat]["total"] += 1
        if r["found"]:
            category_stats[cat]["correct"] += 1

    print("\n" + "-" * 70)
    print("Category Breakdown:")
    print("-" * 70)
    for cat in sorted(category_stats):
        stats = category_stats[cat]
        rate = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        bar = "█" * int(rate / 10) + "░" * (10 - int(rate / 10))
        print(f"  {cat:12s} [{bar}] {rate:5.1f}%  ({stats['correct']}/{stats['total']})")

    total_rate = correct / len(qa_pairs) * 100 if qa_pairs else 0
    total_bar = "█" * int(total_rate / 10) + "░" * (10 - int(total_rate / 10))
    print(f"  {'─' * 50}")
    print(f"  {'Total':12s} [{total_bar}] {total_rate:5.1f}%  ({correct}/{len(qa_pairs)})")
    print("─" * 70)

    return {"total": len(qa_pairs), "correct": correct, "rate": total_rate, "details": results}


def _contains_answer(context: str, answer: str) -> bool:
    """Return ``True`` when ≥ 50 % of *answer* words appear in *context*."""
    answer_words = set(answer.lower().split())
    context_words = set(context.lower().split())
    overlap = answer_words & context_words
    if len(overlap) == 0:
        return False
    return len(overlap) / len(answer_words) >= 0.5


def display_system_stats(system: MemorySystem) -> None:
    """Print a summary of memory spaces and units to stdout."""
    spaces = system.semantic_map.list_spaces()
    units = system.semantic_map.list_units()

    print("\n" + "─" * 70)
    print("MEMORY SYSTEM STATISTICS")
    print("─" * 70)
    print(f"  Total memory spaces: {len(spaces)}")
    print(f"  Total memory units:  {len(units)}")

    for sp in spaces[:5]:
        print(f"    Space: {sp.name} ({len(sp.unit_uids)} units)")
    if len(spaces) > 5:
        print(f"    ... and {len(spaces) - 5} more spaces")


def main() -> None:
    """Entry point: load data, build memories, and run evaluation or custom query."""
    load_env_file()

    parser = argparse.ArgumentParser(description="LongMemEval Example — Long-Text Information Memory & Retrieval")
    parser.add_argument("--data", type=str, default="data/longmemeval_example.json", help="Data file path")
    parser.add_argument("--query", type=str, default=None, help="Custom query text")
    parser.add_argument("--top-k", type=int, default=5, help="Number of retrieval results to return")
    parser.add_argument("--no-rerank", action="store_true", help="Disable reranking")
    parser.add_argument("--no-eval", action="store_true", help="Skip evaluation, run custom query only")
    args = parser.parse_args()

    data_path = (Path(__file__).parent / args.data).resolve()

    print("=" * 70)
    print("  Mandol Memory System — LongMemEval Example")
    print(f"  Data: {data_path.name}")
    print("=" * 70)

    logger.info("Step 1/5: Loading data...")
    data = load_example_data(data_path)

    logger.info("Step 2/5: Initializing memory system...")
    config_path = Path(__file__).parent / "config.yaml"
    if config_path.exists():
        system = MemorySystem.from_yaml_config(str(config_path))
    else:
        system = MemorySystem()

    logger.info("Step 3/5: Processing passage into memory units...")
    chunk_count = process_passage(system, data)
    logger.info(f"Added {chunk_count} document chunks")

    logger.info("Step 4/5: Building high-level memories...")
    result = system.build_high_level(mode="auto")

    high_level_count = 0
    if isinstance(result, dict):
        details = result.get("details", {})
        for group_name in ("entity", "event", "summary"):
            group_data = details.get(group_name, {})
            if isinstance(group_data, dict):
                high_level_count += len(group_data.get("units", []))
    logger.info(f"High-level memories built (approx {high_level_count} derived units)")

    display_system_stats(system)

    logger.info("Step 5/5: Running queries...")

    if args.query:
        hits = system.holistic_retrieve(args.query, top_k=args.top_k, use_rerank=not args.no_rerank)
        print(f"\n{'─' * 70}")
        print(f"Query: {args.query}")
        print(f"{'─' * 70}")
        for rank, hit in enumerate(hits, 1):
            text = hit.unit.raw_data.get("text_content", str(hit.unit.uid))[:120]
            print(f"  [{rank}] score={hit.final_score:.4f} | {text}")
    else:
        eval_result = run_evaluation(system, data, top_k=args.top_k, use_rerank=not args.no_rerank)
        if eval_result["rate"] == 100.0:
            logger.info("All queries passed!")
        else:
            logger.warning(f"Pass rate: {eval_result['rate']:.1f}%")

    print("\n" + "=" * 70)
    print("  LongMemEval example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
