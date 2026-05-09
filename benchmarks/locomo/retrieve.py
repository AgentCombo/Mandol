#!/usr/bin/env python3
"""
Step 2: Retrieve - Execute retrieval queries against the built graph.

This script loads the graph snapshot and runs retrieval queries from the LoCoMo
question set, producing ranked candidate lists for each question.

Usage:
    python retrieve.py [--config configs/base.yaml] [--input output/] [--output output/]

Input:
    - Graph snapshot from build_graph step: {input}/graph_snapshot/

Output:
    - Retrieval results saved to {output}/retrieval_results.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    """Entry point for the LoCoMo retrieval step (placeholder)."""
    parser = argparse.ArgumentParser(description="LoCoMo Benchmark - Step 2: Retrieve")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to config YAML")
    parser.add_argument("--data", type=str, default="data/locomo10.json", help="Path to LoCoMo dataset")
    parser.add_argument("--input", type=str, default="output/", help="Input directory (from build_graph)")
    parser.add_argument("--output", type=str, default="output/", help="Output directory")
    parser.add_argument("--top-k", type=int, default=10, help="Number of candidates per query")
    args = parser.parse_args()

    raise NotImplementedError(
        "Benchmark retrieve script is under development. "
        "See README.md for manual reproduction steps."
    )


if __name__ == "__main__":
    main()
