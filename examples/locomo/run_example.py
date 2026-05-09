#!/usr/bin/env python3
"""
LoCoMo Memory System Example — conv-26 Multi-Session Long Dialogue Demo

This script demonstrates how to use the Mandol memory system to process
a long conversation example from the LoCoMo dataset. Supports two modes:
  - demo: Quick demonstration, only processes the first 3 sessions
  - full: Complete processing of conv-26's 19 sessions

Usage:
    python run_example.py                    # Demo mode (default)
    python run_example.py --mode full        # Full mode
    python run_example.py --query "What is Caroline's identity?"  # Custom query
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from locomo.config import LocomoMemoryConfig, load_env_from_file
from locomo.locomo_memory_system import LocomoMemorySystem

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

load_env_from_file()

PRESET_QUERIES = [
    {
        "question": "What is Caroline's identity?",
        "category": "Single-hop",
    },
    {
        "question": "When did Caroline go to the LGBTQ support group?",
        "category": "Temporal",
    },
    {
        "question": "What fields would Caroline likely pursue in her education?",
        "category": "Multi-hop",
    },
    {
        "question": "What did Melanie enjoy about camping with her kids?",
        "category": "Open-domain",
    },
    {
        "question": "How did Caroline and Melanie describe the power of music?",
        "category": "Open-domain",
    },
]


def load_example_data() -> list[dict]:
    """Load the conv-26 example data from the bundled JSON file.

    Returns:
        A one-element list containing the conv-26 sample dict.

    Exits with code 1 if the data file is missing.
    """
    data_path = Path(__file__).parent / "data" / "locomo_example_conv26.json"
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        logger.error("Ensure the example data has been extracted from locomo10.json")
        sys.exit(1)
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Loaded example data: {data_path} ({data_path.stat().st_size / 1024:.1f} KB)")
    return [data]


def truncate_sessions(data: list[dict], max_sessions: int) -> list[dict]:
    """Keep only the first *max_sessions* sessions per sample.

    Preserves speaker metadata (``speaker_a``, ``speaker_b``) and
    per-session date-time / summary keys.

    Args:
        data: List of LoCoMo sample dicts.
        max_sessions: Maximum number of sessions to retain.

    Returns:
        A new list of samples with truncated conversations.
    """
    result = []
    for sample in data:
        conv = sample.get("conversation", {})
        if not isinstance(conv, dict):
            result.append(sample)
            continue

        keys_to_keep = {"speaker_a", "speaker_b"}
        truncated = {}
        for k in keys_to_keep:
            if k in conv:
                truncated[k] = conv[k]

        for n in range(1, max_sessions + 1):
            skey = f"session_{n}"
            dtkey = f"session_{n}_date_time"
            sumkey = f"session_{n}_summary"
            if skey in conv:
                truncated[skey] = conv[skey]
            if dtkey in conv:
                truncated[dtkey] = conv[dtkey]
            if sumkey in conv:
                truncated[sumkey] = conv[sumkey]

        sample_copy = dict(sample)
        sample_copy["conversation"] = truncated
        result.append(sample_copy)

    dialogue_count = sum(
        len(v)
        for s in result
        for k, v in s.get("conversation", {}).items()
        if k.startswith("session_") and isinstance(v, list)
    )
    logger.info(f"Truncated to {max_sessions} sessions, {dialogue_count} dialogues")
    return result


def run_queries(system: LocomoMemorySystem, queries: list[dict], top_k: int, use_rerank: bool) -> None:
    """Execute a list of queries against the memory system and print results.

    Args:
        system: An initialized :class:`LocomoMemorySystem`.
        queries: Each dict must contain ``"question"`` and optionally ``"category"``.
        top_k: Number of results per query.
        use_rerank: Whether to apply the reranker.
    """
    print("\n" + "=" * 70)
    print("QUERY RESULTS")
    print("=" * 70)

    for i, q in enumerate(queries):
        question = q["question"]
        category = q.get("category", "")

        print(f"\n{'─' * 70}")
        print(f"Query {i + 1}: {question}")
        print(f"Category: {category}")
        print(f"{'─' * 70}")

        hits = system.search(question, top_k=top_k, use_rerank=use_rerank)

        if not hits:
            print("  (no results found)")
            continue

        for rank, hit in enumerate(hits, 1):
            score = hit.final_score
            raw = hit.unit.raw_data
            text_preview = raw.get("text_content", raw.get("text", str(raw)[:120]))[:120]
            print(f"  [{rank}] score={score:.4f} | {text_preview}")

    print(f"\n{'─' * 70}")
    print(f"Total: {len(queries)} queries executed")


def main() -> None:
    """Entry point: load data, build memories, and run preset or custom queries."""
    parser = argparse.ArgumentParser(
        description="LoCoMo Memory System Example — conv-26 Multi-Session Long Dialogue Demo"
    )
    parser.add_argument(
        "--mode",
        choices=["demo", "full"],
        default="demo",
        help="Run mode: demo (3 sessions) / full (all 19 sessions)",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Custom query text (overrides preset queries)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of retrieval results to return",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable the reranking step",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  Mandol Memory System — LoCoMo Example")
    print(f"  Sample: conv-26 (Caroline & Melanie)")
    print(f"  Mode: {args.mode}")
    print("=" * 70)

    max_sessions = 3 if args.mode == "demo" else 999
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        logger.info(f"Loading environment from: {env_file}")

    logger.info("Step 1/5: Loading data...")
    raw_data = load_example_data()

    logger.info("Step 2/5: Preparing sessions...")
    if args.mode == "demo":
        data = truncate_sessions(raw_data, max_sessions=max_sessions)
    else:
        data = raw_data
        dialogue_count = sum(
            len(v)
            for s in data
            for k, v in s.get("conversation", {}).items()
            if k.startswith("session_") and isinstance(v, list)
        )
        logger.info(f"Full mode: processing all {dialogue_count} dialogues")

    logger.info("Step 3/5: Initializing memory system...")
    config = LocomoMemoryConfig()
    config.sample_count = len(data)
    system = LocomoMemorySystem(config=config)

    logger.info("Step 4/5: Processing dialogues and building high-level memories...")
    result = system.load_and_process_samples()
    logger.info(f"Load result: {json.dumps(result, indent=2)}")

    build_result = system.build_high_level_memories(mode="auto")
    logger.info(f"Build result: {json.dumps({k: v for k, v in build_result.items() if k != 'details'}, indent=2, default=str)}")

    stats = system.get_memory_stats()
    print("\n" + "─" * 70)
    print("MEMORY STATISTICS")
    print("─" * 70)
    print(f"  Total units: {stats.get('total_units', 0)}")
    print(f"  Total spaces: {stats.get('total_spaces', 0)}")
    print(f"  Processed samples: {stats.get('processed_sample_ids', [])}")
    proc = stats.get("processing_stats", {})
    print(f"  Dialogues processed: {proc.get('dialogues_processed', 0)}")
    print(f"  Sessions processed: {proc.get('sessions_processed', 0)}")
    print(f"  Units added: {proc.get('units_added', 0)}")

    logger.info("Step 5/5: Running queries...")
    if args.query:
        queries = [{"question": args.query, "category": "custom"}]
    else:
        queries = PRESET_QUERIES

    run_queries(system, queries, top_k=args.top_k, use_rerank=not args.no_rerank)

    print("\n" + "=" * 70)
    print("  Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
