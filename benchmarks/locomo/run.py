#!/usr/bin/env python3
"""LoCoMo Benchmark — End-to-end pipeline runner.

Orchestrates the full 4-step pipeline: build_graph → retrieve → generate → evaluate.
Each step is executed as a subprocess so failures are isolated and logs are preserved.

Usage:
    python run.py --config configs/base.yaml [--output output/] [--force] [--steps build,retrieve,generate,evaluate]
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_SCRIPT_DIR = Path(__file__).resolve().parent

STEPS = {
    "build": {
        "script": "build_graph.py",
        "description": "Step 1: Build Graph",
    },
    "retrieve": {
        "script": "retrieve.py",
        "description": "Step 2: Retrieve",
    },
    "generate": {
        "script": "generate.py",
        "description": "Step 3: Generate",
    },
    "evaluate": {
        "script": "evaluate.py",
        "description": "Step 4: Evaluate",
    },
}


def run_step(step_name: str, config: str, output: str, force: bool) -> dict:
    info = STEPS[step_name]
    script = _SCRIPT_DIR / info["script"]
    logger.info("=== %s ===", info["description"])
    cmd = [sys.executable, str(script), "--config", config, "--output", output]
    if force:
        cmd.append("--force")

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - t0

    success = result.returncode == 0
    if not success:
        logger.error("%s FAILED with exit code %d in %.1fs", info["description"], result.returncode, elapsed)
    else:
        logger.info("%s completed in %.1fs", info["description"], elapsed)

    return {
        "step": step_name,
        "status": "completed" if success else "error",
        "duration_seconds": round(elapsed, 3),
        "exit_code": result.returncode,
    }


def main():
    parser = argparse.ArgumentParser(description="LoCoMo Benchmark — End-to-end pipeline runner")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to config YAML")
    parser.add_argument("--output", type=str, default=None, help="Output directory (overrides config)")
    parser.add_argument("--force", action="store_true", help="Force re-execute all steps")
    parser.add_argument(
        "--steps",
        type=str,
        default="build,retrieve,generate,evaluate",
        help="Comma-separated list of steps to run (default: all four)",
    )
    args = parser.parse_args()

    # Resolve output directory from config if not provided.
    # Each sub-script appends the config_name, so only pass the base dir.
    if args.output is None:
        from pipeline_utils import load_config
        cfg = load_config(args.config)
        experiment = cfg.get("experiment", {})
        output = experiment.get("output_dir", "output")
    else:
        output = args.output

    step_names = [s.strip() for s in args.steps.split(",") if s.strip() in STEPS]
    if not step_names:
        logger.error("No valid steps specified. Available: %s", list(STEPS.keys()))
        sys.exit(1)

    logger.info("Pipeline: %s", " → ".join(step_names))
    logger.info("Config: %s, Output: %s, Force: %s", args.config, output, args.force)

    pipeline_t0 = time.time()
    results = []

    for step_name in step_names:
        result = run_step(step_name, args.config, output, args.force)
        results.append(result)
        if result["status"] == "error":
            logger.error("Pipeline halted at %s due to error", step_name)
            break

    total_elapsed = time.time() - pipeline_t0
    completed = sum(1 for r in results if r["status"] == "completed")
    errors = sum(1 for r in results if r["status"] == "error")

    logger.info("=" * 50)
    logger.info("Pipeline complete: %d/%d steps succeeded, %d errors, total %.1fs",
                completed, len(results), errors, total_elapsed)
    for r in results:
        logger.info("  %s: %s (%.1fs)", r["step"], r["status"], r["duration_seconds"])


if __name__ == "__main__":
    main()
