#!/usr/bin/env python3
"""Step 1: Build Graph - Load LoCoMo dataset and construct semantic graph.

For each sample, creates a MemorySystem, writes dialogues, builds
high-level memories, and persists the graph.  Supports per-example
resume via manifest detection.

Usage:
    python build_graph.py --config configs/base.yaml [--data data/locomo10.json] [--output output/] [--force]
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

from adapter.locomo_adapter import load_locomo_sample, write_sample_to_graph
from pipeline_utils import (
    is_graph_built,
    load_config,
    load_dataset,
    save_json,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def build_single_sample(
    sample_id: str,
    dataset_path: str,
    config_path: str,
    output_dir: Path,
    force: bool = False,
) -> dict:
    graph_dir = output_dir / sample_id / "graph"

    if not force and is_graph_built(output_dir, sample_id):
        logger.info("Skipping %s: graph already built", sample_id)
        existing = output_dir / sample_id / "build.json"
        if existing.exists():
            import json
            return json.loads(existing.read_text(encoding="utf-8"))
        return {"sample_id": sample_id, "status": "skipped"}

    logger.info("Building graph for sample: %s", sample_id)

    system = MemorySystem.from_yaml_config(config_path, root=sample_id)

    sample = load_locomo_sample(dataset_path=dataset_path, sample_id=sample_id)
    write_sample_to_graph(graph=system.graph, sample=sample)

    t0 = time.time()
    report = system.build_high_level(mode="auto")
    elapsed = time.time() - t0

    system.save(str(graph_dir))

    build_result = {
        "sample_id": sample_id,
        "status": report.status,
        "sessions_processed": report.sessions_processed,
        "units_processed": report.units_processed,
        "duration_seconds": round(elapsed, 3),
        "token_usage": report.token_usage,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if report.error_message:
        build_result["error_message"] = report.error_message
    save_json(output_dir / sample_id / "build.json", build_result)

    if report.status == "error":
        logger.error(
            "Build FAILED for %s: %s (processed %d sessions before error)",
            sample_id,
            report.error_message,
            report.sessions_processed,
        )
    else:
        logger.info(
            "Built %s: %d sessions, %d units, %.1fs, tokens=%s",
            sample_id,
            report.sessions_processed,
            report.units_processed,
            elapsed,
            report.token_usage,
        )
    return build_result


def main():
    parser = argparse.ArgumentParser(description="LoCoMo Benchmark - Step 1: Build Graph")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to config YAML")
    parser.add_argument("--data", type=str, default=None, help="Path to LoCoMo dataset (overrides config)")
    parser.add_argument("--output", type=str, default=None, help="Output directory (overrides config)")
    parser.add_argument("--force", action="store_true", help="Force rebuild even if graph exists")
    args = parser.parse_args()

    cfg = load_config(args.config)
    experiment = cfg.get("experiment", {})

    dataset_path = args.data or experiment.get("dataset_path", "data/locomo10.json")
    output_dir = Path(args.output or experiment.get("output_dir", "output"))
    config_name = experiment.get("config_name", "default")
    output_dir = output_dir / config_name

    sample_ids_override = experiment.get("sample_ids", [])
    samples = load_dataset(dataset_path, sample_ids_override or None)

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", output_dir)
    logger.info("Samples to process: %d", len(samples))

    all_results = []
    for sample in samples:
        sid = sample["sample_id"]
        result = build_single_sample(
            sample_id=sid,
            dataset_path=dataset_path,
            config_path=args.config,
            output_dir=output_dir,
            force=args.force,
        )
        all_results.append(result)

    total_sessions = sum(r.get("sessions_processed", 0) for r in all_results)
    total_units = sum(r.get("units_processed", 0) for r in all_results)
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
        "total_sessions": total_sessions,
        "total_units": total_units,
        "total_duration_seconds": round(total_duration, 3),
        "total_token_usage": total_tokens,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    save_json(output_dir / "build_stats.json", stats)
    logger.info("Build complete: %s", stats)


if __name__ == "__main__":
    main()
