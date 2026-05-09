#!/usr/bin/env python3
"""
Step 2: Retrieve - Execute retrieval queries against the built graph.

Usage:
    python retrieve.py [--config configs/base.yaml] [--input output/] [--output output/]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))


def main():
    """Entry point for the LongMemEval retrieval step (placeholder)."""
    parser = argparse.ArgumentParser(description="LongMemEval Benchmark - Step 2: Retrieve")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--input", type=str, default="output/")
    parser.add_argument("--output", type=str, default="output/")
    args = parser.parse_args()

    raise NotImplementedError(
        "Benchmark retrieve script is under development. "
        "See README.md for manual reproduction steps."
    )


if __name__ == "__main__":
    main()
