#!/usr/bin/env python3
"""LoCoMo experimental self-host evaluation runner.

Two modes:
  --smoke     Fast validation on a small subset (1 sample, 3 sessions, 5 QA,
              includes persistence round-trip). Self-contained — does NOT call
              subprocess scripts.
  (default)   Full pipeline: calls build_graph → retrieve → generate → evaluate
              as subprocess steps. Each step supports per-query resume.

Usage:
    # Smoke test — quick validation
    python run.py --smoke --config configs/base.yaml

    # Full development evaluation
    python run.py --config configs/base.yaml --data data/locomo10.json --output output/
    python run.py --config configs/base.yaml --force
    python run.py --config configs/base.yaml --stages build,retrieve
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mandol import MemorySystem
from adapter.locomo_adapter import load_locomo_sample, write_sample_to_graph
from pipeline_utils import (
    EVALUATION_PROMPT_TEMPLATE,
    GENERATION_PROMPT_TEMPLATE,
    extract_final_answer,
    load_config,
    load_dataset,
    parse_judge_label,
    save_json,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run")

_SCRIPT_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_and_load_env() -> None:
    for candidate in _SCRIPT_DIR.resolve().parents:
        env_path = candidate / ".env"
        if env_path.exists():
            try:
                from dotenv import load_dotenv  # type: ignore
                load_dotenv(env_path, override=False)
                logger.info("Loaded .env from %s", env_path)
            except ImportError:
                _parse_dotenv_manual(env_path)
            return
    logger.warning("No .env file found in ancestor directories")


def _parse_dotenv_manual(env_path: Path) -> None:
    import os
    try:
        text = env_path.read_text(encoding="utf-8")
    except OSError:
        return
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$", line)
        if m:
            key, val = m.group(1), m.group(2).strip().strip("'\"")
            if key not in os.environ:
                os.environ[key] = val


def _qa_in_sessions(q: dict, allowed: set) -> bool:
    evidence = q.get("evidence", [])
    if not evidence:
        return False
    for e in evidence:
        m = re.match(r"D(\d+):", str(e))
        if m and int(m.group(1)) not in allowed:
            return False
    return True


def _run_script(script_name: str, args: List[str], stage: str) -> bool:
    script_path = _SCRIPT_DIR / script_name
    cmd = [sys.executable, str(script_path)] + args
    logger.info("Stage '%s': %s", stage, " ".join(cmd))
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(_SCRIPT_DIR))
    elapsed = time.time() - t0
    if result.returncode == 0:
        logger.info("Stage '%s' completed in %.1fs", stage, elapsed)
        return True
    logger.error("Stage '%s' FAILED (exit=%d, %.1fs)", stage, result.returncode, elapsed)
    return False


# ---------------------------------------------------------------------------
# Smoke test (self-contained, no subprocess)
# ---------------------------------------------------------------------------

def _smoke_build(sample: dict, config_path: str, output_dir: Path) -> Optional[MemorySystem]:
    """Phase 1: build graph from a small subset of one sample."""
    conv = sample["conversation"]
    sample_id = sample["sample_id"]

    sessions_to_keep = {1, 2, 3}
    trimmed_conv = {}
    for k, v in conv.items():
        if k.startswith("session_") and not k.endswith("_date_time"):
            sn = int(re.match(r"session_(\d+)", k).group(1))
            if sn not in sessions_to_keep:
                continue
        trimmed_conv[k] = v
    trimmed_sample = {**sample, "conversation": trimmed_conv}

    system = MemorySystem.from_yaml_config(config_path, root=sample_id)

    t0 = time.time()
    write_sample_to_graph(graph=system.graph, sample=trimmed_sample, batch_embed=True)
    logger.info("Dialogues written in %.1fs", time.time() - t0)

    t0 = time.time()
    report = system.build_high_level(mode="auto")
    elapsed = time.time() - t0
    logger.info(
        "build_high_level: status=%s, sessions=%d, units=%d, time=%.1fs",
        report.status, report.sessions_processed, report.units_processed, elapsed,
    )

    if report.status == "error":
        logger.error("Smoke build FAILED: %s", report.error_message)
        return None

    return system


def _smoke_retrieve(system: MemorySystem, queries: list) -> list:
    """Phase 2: holistic_retrieve on each query."""
    results = []
    for q in queries:
        t0 = time.time()
        hits = system.holistic_retrieve(q["question"], top_k=10, use_rerank=True)
        elapsed = time.time() - t0
        results.append({
            "question": q["question"],
            "answer": q["answer"],
            "category": q["category"],
            "hits": [
                {
                    "uid": str(h.unit.uid),
                    "text": h.unit.raw_data.get("text_content", "")[:100],
                    "score": round(h.final_score, 4),
                }
                for h in hits[:5]
            ],
            "duration_s": round(elapsed, 3),
        })
        logger.info("Retrieved %d hits for: %.60s (%.1fs)", len(hits), q["question"], elapsed)
    return results


def _build_smoke_llm_provider() -> "OpenAICompatibleLLMProvider":
    """Construct an LLM provider from environment variables for smoke tests."""
    import os as _os
    from mandol.infrastructure.openai_compatible_llm_provider import OpenAICompatibleLLMProvider

    model = _os.getenv("MANDOL_LLM_MODEL", "gpt-4o-mini")
    api_key = _os.getenv("MANDOL_LLM_API_KEY", "")
    base_url = _os.getenv("MANDOL_LLM_BASE_URL", "")
    llm_kwargs = {"model": model}
    if base_url:
        llm_kwargs["base_url"] = base_url
    if api_key:
        llm_kwargs["api_key"] = api_key
    return OpenAICompatibleLLMProvider(**llm_kwargs)


def _smoke_generate(retrieval_results: list) -> list:
    """Phase 3: generate answers via LLM using retrieved context."""
    config_path = str(_SCRIPT_DIR / "configs" / "base.yaml")
    cfg = load_config(config_path)
    generation_cfg = cfg.get("generation", {})

    llm = _build_smoke_llm_provider()

    results = []
    for r in retrieval_results:
        context = "\n\n---\n\n".join(
            f"[Memory {i + 1}] {h['text']}" for i, h in enumerate(r["hits"])
        )
        prompt = GENERATION_PROMPT_TEMPLATE.format(question=r["question"], context=context)

        t0 = time.time()
        response = llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=generation_cfg.get("temperature", 0.3),
            max_tokens=generation_cfg.get("max_tokens", 256),
        )
        elapsed = time.time() - t0

        results.append({
            "question": r["question"],
            "gold_answer": r["answer"],
            "category": r["category"],
            "generated_answer": response.content,
            "generated_answer_extracted": extract_final_answer(response.content),
            "duration_s": round(elapsed, 3),
        })
        logger.info("Generated answer for: %.60s (%.1fs)", r["question"], elapsed)
    return results


def _smoke_evaluate(generation_results: list) -> list:
    """Phase 4: LLM judge evaluation."""
    llm = _build_smoke_llm_provider()

    results = []
    for g in generation_results:
        prompt = EVALUATION_PROMPT_TEMPLATE.format(
            question=g["question"],
            gold_answer=str(g["gold_answer"]),
            generated_answer=g["generated_answer_extracted"],
        )
        t0 = time.time()
        response = llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=100,
        )
        elapsed = time.time() - t0

        label = parse_judge_label(response.content)
        results.append({
            "question": g["question"],
            "gold_answer": g["gold_answer"],
            "generated_answer": g["generated_answer_extracted"][:200],
            "judge_label": label,
            "accuracy": 1 if label and "CORRECT" in label.upper() else 0,
            "duration_s": round(elapsed, 3),
        })
        logger.info("Judge: %.60s → %s (%.1fs)", g["question"], label, elapsed)
    return results


def _smoke_persistence(system: MemorySystem, output_dir: Path) -> dict:
    """Phase 5: save → load → retrieve round-trip."""
    save_path = str(output_dir / "smoke_test_graph")
    t0 = time.time()
    save_result = system.save(save_path)
    save_time = time.time() - t0

    logger.info(
        "Saved: units=%d, spaces=%d, edges=%d, sessions=%d (%.1fs)",
        save_result.stats.get("unit_count", 0),
        save_result.stats.get("space_count", 0),
        save_result.stats.get("edge_count", 0),
        save_result.stats.get("session_count", 0),
        save_time,
    )

    t0 = time.time()
    loaded = MemorySystem.load(save_path)
    load_time = time.time() - t0

    hits = loaded.holistic_retrieve("What did Caroline research?", top_k=5)
    logger.info("Loaded system retrieval: %d hits (%.1fs)", len(hits), load_time)

    return {
        "save_units": save_result.stats.get("unit_count", 0),
        "save_spaces": save_result.stats.get("space_count", 0),
        "save_edges": save_result.stats.get("edge_count", 0),
        "save_sessions": save_result.stats.get("session_count", 0),
        "save_time_s": round(save_time, 1),
        "load_time_s": round(load_time, 1),
        "loaded_retrieval_hits": len(hits),
    }


def _smoke_main(config_path: str, keep_output: bool = False) -> bool:
    """Run a fast smoke test on a small data subset (conv-26, sessions 1-3, 5 QA)."""
    _find_and_load_env()

    dataset_path = str(_SCRIPT_DIR / "data" / "locomo10.json")
    sample = load_locomo_sample(dataset_path=dataset_path, sample_id="conv-26")
    logger.info("Loaded sample: %s", sample["sample_id"])

    all_qa = sample["qa"]
    test_queries = [
        {"question": q["question"], "answer": q["answer"], "category": q["category"]}
        for q in all_qa
        if _qa_in_sessions(q, {1, 2, 3}) and q.get("category") != 5
    ][:5]
    logger.info("Selected %d test queries: %s", len(test_queries), [q["category"] for q in test_queries])

    output_dir = Path(tempfile.mkdtemp(prefix="mandol_smoke_"))
    logger.info("Output directory: %s", output_dir)

    success = True
    results = {"test_queries": len(test_queries)}

    try:
        # Phase 1: Build
        logger.info("=" * 50)
        logger.info("PHASE 1: Build graph (smoke)")
        logger.info("=" * 50)
        system = _smoke_build(sample, config_path, output_dir)
        if system is None:
            logger.error("BUILD FAILED — aborting smoke test")
            return False

        # Phase 2: Retrieve
        logger.info("=" * 50)
        logger.info("PHASE 2: Retrieval (%d queries)", len(test_queries))
        logger.info("=" * 50)
        retrieval_results = _smoke_retrieve(system, test_queries)
        avg_hits = sum(len(r["hits"]) for r in retrieval_results) / max(len(retrieval_results), 1)
        results["retrieval"] = {"queries": len(retrieval_results), "avg_hits": avg_hits}

        # Phase 3: Generate
        logger.info("=" * 50)
        logger.info("PHASE 3: Generation (%d queries)", len(test_queries))
        logger.info("=" * 50)
        generation_results = _smoke_generate(retrieval_results)
        results["generation"] = {"queries": len(generation_results)}

        # Phase 4: Evaluate
        logger.info("=" * 50)
        logger.info("PHASE 4: Evaluation (%d queries)", len(test_queries))
        logger.info("=" * 50)
        evaluation_results = _smoke_evaluate(generation_results)
        acc = sum(e["accuracy"] for e in evaluation_results)
        total = len(evaluation_results)
        results["evaluation"] = {
            "queries": total,
            "correct": acc,
            "accuracy": round(acc / total, 3) if total else 0,
        }

        # Phase 5: Persistence round-trip
        logger.info("=" * 50)
        logger.info("PHASE 5: Persistence round-trip")
        logger.info("=" * 50)
        persistence_stats = _smoke_persistence(system, output_dir)
        results["persistence"] = persistence_stats

    except Exception:
        logger.exception("Smoke test failed with exception")
        success = False
    finally:
        if not keep_output:
            shutil.rmtree(output_dir, ignore_errors=True)
            logger.info("Cleaned up %s", output_dir)
        else:
            logger.info("Output kept at %s", output_dir)

    # Report
    logger.info("=" * 60)
    logger.info("SMOKE TEST REPORT")
    logger.info("=" * 60)
    logger.info("Test queries: %d", results.get("test_queries", 0))

    retrieval = results.get("retrieval", {})
    logger.info("Retrieval: %d queries, avg %.1f hits", retrieval.get("queries", 0), retrieval.get("avg_hits", 0))

    eval_stats = results.get("evaluation", {})
    logger.info(
        "Evaluation: %d/%d correct (%.1f%%)",
        eval_stats.get("correct", 0), eval_stats.get("queries", 0),
        eval_stats.get("accuracy", 0) * 100,
    )

    pers = results.get("persistence", {})
    logger.info(
        "Persistence: save=%d units, load retrieval=%d hits",
        pers.get("save_units", 0), pers.get("loaded_retrieval_hits", 0),
    )

    if success:
        logger.info("SMOKE TEST PASSED")
    else:
        logger.error("SMOKE TEST FAILED")
    return success


# ---------------------------------------------------------------------------
# Full pipeline (subprocess orchestration)
# ---------------------------------------------------------------------------

def _full_main(args: argparse.Namespace) -> None:
    config_path = args.config
    if not Path(config_path).is_absolute():
        config_path = str((_SCRIPT_DIR / config_path).resolve())

    cfg = load_config(config_path)

    data_path = args.data
    if not data_path:
        experiment = cfg.get("experiment", {})
        ds_path = experiment.get("dataset_path", "data/locomo10.json")
        if not Path(ds_path).is_absolute():
            ds_path = str((_SCRIPT_DIR / ds_path).resolve())
        data_path = ds_path

    output_dir = args.output
    if not output_dir:
        experiment = cfg.get("experiment", {})
        out = experiment.get("output_dir", "output")
        if not Path(out).is_absolute():
            out = str((_SCRIPT_DIR / out).resolve())
        config_name = experiment.get("config_name", "default")
        output_dir = str(Path(out) / config_name)

    if not Path(data_path).exists():
        logger.error("Dataset not found: %s", data_path)
        sys.exit(1)

    force_flag = ["--force"] if args.force else []
    shared = [f"--config={config_path}", f"--data={data_path}", f"--output={output_dir}"] + force_flag

    stage_set = {s.strip() for s in args.stages.split(",")}
    all_stages = ["build", "retrieve", "generate", "evaluate"]
    stages_to_run = [s for s in all_stages if s in stage_set]

    if not stages_to_run:
        logger.error("No valid stages selected. Choose from: %s", ", ".join(all_stages))
        sys.exit(1)

    stage_scripts = {
        "build": "build_graph.py",
        "retrieve": "retrieve.py",
        "generate": "generate.py",
        "evaluate": "evaluate.py",
    }

    logger.info("=" * 60)
    logger.info("LoCoMo Experimental Self-host Pipeline")
    logger.info("Config: %s", config_path)
    logger.info("Dataset: %s", data_path)
    logger.info("Output: %s", output_dir)
    logger.info("Stages: %s", " → ".join(stages_to_run))
    logger.info("Force: %s", args.force)
    logger.info("=" * 60)

    for stage in stages_to_run:
        success = _run_script(stage_scripts[stage], shared, stage)
        if not success:
            logger.error("Pipeline stopped due to failure in stage '%s'", stage)
            sys.exit(1)

    logger.info("=" * 60)
    logger.info("Pipeline completed successfully. Output: %s", output_dir)
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LoCoMo Experimental Self-host Evaluation Runner"
    )
    parser.add_argument(
        "--config", type=str, default="configs/base.yaml",
        help="Path to config YAML (default: configs/base.yaml)",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Run a fast smoke test on a small data subset (conv-26, 3 sessions, 5 QA)",
    )
    parser.add_argument(
        "--keep-output", action="store_true",
        help="Keep temporary output directory (smoke mode only)",
    )
    # Full pipeline args
    parser.add_argument("--data", type=str, default=None, help="Path to dataset (overrides config)")
    parser.add_argument("--output", type=str, default=None, help="Output directory (overrides config)")
    parser.add_argument("--force", action="store_true", help="Force rebuild all stages")
    parser.add_argument(
        "--stages", type=str, default="build,retrieve,generate,evaluate",
        help="Comma-separated stages to run (default: build,retrieve,generate,evaluate)",
    )
    args = parser.parse_args()

    logger.warning(
        "Mandol experimental self-host evaluation workflow\n"
        "This is not the frozen paper-reproduction pipeline.\n"
        "For exact paper reproduction, use the paper-repro branch."
    )

    if args.smoke:
        success = _smoke_main(args.config, keep_output=args.keep_output)
        if not success:
            sys.exit(1)
    else:
        _full_main(args)


if __name__ == "__main__":
    main()
