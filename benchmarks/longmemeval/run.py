#!/usr/bin/env python3
"""LongMemEval Benchmark — Full Pipeline Runner.

Two modes:
  --smoke     Fast validation on a small subset (1 sample, 3 sessions,
              includes persistence round-trip). Self-contained — does NOT call
              subprocess scripts.
  (default)   Full pipeline: calls build_graph → retrieve → generate → evaluate
              as subprocess steps. Each step supports per-query resume.

Usage:
    # Smoke test — quick validation
    python run.py --smoke --config configs/base.yaml

    # Full benchmark
    python run.py --config configs/base.yaml --data data/longmemeval_s_cleaned.json --output output/
    python run.py --config configs/base.yaml --force
    python run.py --config configs/base.yaml --stages build,retrieve
"""
from __future__ import annotations

import argparse
import json
import logging
import os
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
from adapter.longmemeval_adapter import load_longmemeval_sample, write_sample_to_graph
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
    """Load .env file from ancestor directories."""
    for candidate in _SCRIPT_DIR.resolve().parents:
        env_path = candidate / ".env"
        if env_path.exists():
            try:
                from dotenv import load_dotenv
                load_dotenv(env_path, override=False)
                logger.info("Loaded .env from %s", env_path)
            except ImportError:
                _parse_dotenv_manual(env_path)
            return
    logger.warning("No .env file found in ancestor directories")


def _parse_dotenv_manual(env_path: Path) -> None:
    """Minimal .env parser without python-dotenv."""
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


def _run_script(script_name: str, script_args: List[str], stage: str) -> bool:
    """Run a pipeline subprocess script.

    Args:
        script_name: Name of the .py script in _SCRIPT_DIR.
        script_args: Arguments to pass.
        stage: Human-readable stage name for logging.

    Returns:
        True if the subprocess succeeded.
    """
    script_path = _SCRIPT_DIR / script_name
    cmd = [sys.executable, str(script_path)] + script_args
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


def _smoke_build(sample: dict, config_path: str, output_dir: Path) -> Optional[MemorySystem]:
    """Phase 1: build graph from a small subset of one sample."""
    qid = sample["question_id"]
    sessions = sample.get("haystack_sessions", [])[:3]  # Only first 3 sessions
    dates = sample.get("haystack_dates", [])[:3]
    session_ids = sample.get("haystack_session_ids", [])[:3]

    trimmed_sample = {
        **sample,
        "haystack_sessions": sessions,
        "haystack_dates": dates,
        "haystack_session_ids": session_ids,
    }

    system = MemorySystem.from_yaml_config(config_path, root=qid)

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


def _smoke_retrieve(system: MemorySystem, sample: dict) -> dict:
    """Phase 2: holistic_retrieve on the sample's question."""
    question = sample["question"]
    t0 = time.time()
    hits = system.holistic_retrieve(question, top_k=10, use_rerank=True)
    elapsed = time.time() - t0
    result = {
        "question": question,
        "answer": sample.get("answer", ""),
        "question_type": sample.get("question_type", ""),
        "hits": [
            {
                "uid": str(h.unit.uid),
                "text": h.unit.raw_data.get("text_content", "")[:100],
                "score": round(h.final_score, 4),
            }
            for h in hits[:5]
        ],
        "duration_s": round(elapsed, 3),
    }
    logger.info("Retrieved %d hits for: %.60s (%.1fs)", len(hits), question, elapsed)
    return result


def _smoke_generate(retrieval_result: dict) -> dict:
    """Phase 3: generate answer via LLM using retrieved context."""
    config_path = str(_SCRIPT_DIR / "configs" / "base.yaml")
    cfg = load_config(config_path)
    generation_cfg = cfg.get("generation", {})

    llm = _build_smoke_llm_provider()

    context = "\n\n---\n\n".join(
        f"[Memory {i + 1}] {h['text']}" for i, h in enumerate(retrieval_result["hits"])
    )
    prompt = GENERATION_PROMPT_TEMPLATE.format(
        question=retrieval_result["question"],
        context=context,
    )

    t0 = time.time()
    response = llm.chat(
        messages=[{"role": "user", "content": prompt}],
        temperature=generation_cfg.get("temperature", 0.3),
        max_tokens=generation_cfg.get("max_tokens", 256),
    )
    elapsed = time.time() - t0

    return {
        "question": retrieval_result["question"],
        "gold_answer": retrieval_result["answer"],
        "question_type": retrieval_result["question_type"],
        "generated_answer": response.content,
        "generated_answer_extracted": extract_final_answer(response.content),
        "duration_s": round(elapsed, 3),
    }


def _smoke_evaluate(generation_result: dict) -> dict:
    """Phase 4: LLM judge evaluation."""
    llm = _build_smoke_llm_provider()

    prompt = EVALUATION_PROMPT_TEMPLATE.format(
        question=generation_result["question"],
        gold_answer=str(generation_result["gold_answer"]),
        generated_answer=generation_result["generated_answer_extracted"],
    )
    t0 = time.time()
    response = llm.chat(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=100,
    )
    elapsed = time.time() - t0

    label = parse_judge_label(response.content)
    return {
        "question": generation_result["question"],
        "gold_answer": generation_result["gold_answer"],
        "generated_answer": generation_result["generated_answer_extracted"][:200],
        "judge_label": label,
        "accuracy": 1 if label and "CORRECT" in label.upper() else 0,
        "duration_s": round(elapsed, 3),
    }


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

    hits = loaded.holistic_retrieve("What did the user say about their fitness goals?", top_k=5)
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
    """Run a fast smoke test on a small data subset (first sample, 3 sessions).

    The smoke test exercises the entire pipeline:
    1. Build graph from first 3 haystack sessions
    2. Retrieve on the sample's question
    3. Generate answer via LLM
    4. LLM judge evaluation
    5. Persistence round-trip (save → load → retrieve)
    """
    _find_and_load_env()

    dataset_path = str(_SCRIPT_DIR / "data" / "longmemeval_s_cleaned.json")
    sample = load_longmemeval_sample(dataset_path=dataset_path, question_id="e47becba")
    logger.info("Loaded sample: %s (type=%s)", sample["question_id"], sample.get("question_type", ""))

    output_dir = Path(tempfile.mkdtemp(prefix="mandol_smoke_longmemeval_"))
    logger.info("Output directory: %s", output_dir)

    success = True
    results = {"question_id": sample["question_id"], "question_type": sample.get("question_type", "")}

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
        logger.info("PHASE 2: Retrieval")
        logger.info("=" * 50)
        retrieval_result = _smoke_retrieve(system, sample)
        results["retrieval"] = {
            "query": sample["question"][:80],
            "hits_count": len(retrieval_result["hits"]),
        }

        # Phase 3: Generate
        logger.info("=" * 50)
        logger.info("PHASE 3: Generation")
        logger.info("=" * 50)
        generation_result = _smoke_generate(retrieval_result)
        results["generation"] = {"generated_answer": generation_result["generated_answer_extracted"][:120]}

        # Phase 4: Evaluate
        logger.info("=" * 50)
        logger.info("PHASE 4: Evaluation")
        logger.info("=" * 50)
        evaluation_result = _smoke_evaluate(generation_result)
        results["evaluation"] = {
            "judge_label": evaluation_result["judge_label"],
            "accuracy": evaluation_result["accuracy"],
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
    logger.info("Sample: %s (%s)", results.get("question_id", ""), results.get("question_type", ""))

    retrieval = results.get("retrieval", {})
    logger.info("Retrieval: %s hits for '%.60s'", retrieval.get("hits_count", 0), retrieval.get("query", ""))

    eval_stats = results.get("evaluation", {})
    logger.info("Evaluation: label=%s, accuracy=%s", eval_stats.get("judge_label", ""), eval_stats.get("accuracy", 0))

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
    """Run the full pipeline via subprocess scripts."""
    config_path = args.config
    if not Path(config_path).is_absolute():
        config_path = str((_SCRIPT_DIR / config_path).resolve())

    cfg = load_config(config_path)

    data_path = args.data
    if not data_path:
        experiment = cfg.get("experiment", {})
        ds_path = experiment.get("dataset_path", "data/longmemeval_s_cleaned.json")
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
    logger.info("LongMemEval Benchmark Pipeline")
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
        description="LongMemEval Benchmark — Full Pipeline Runner"
    )
    parser.add_argument(
        "--config", type=str, default="configs/base.yaml",
        help="Path to config YAML (default: configs/base.yaml)",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Run a fast smoke test on a small data subset (sample e47becba, 3 sessions)",
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

    if args.smoke:
        success = _smoke_main(args.config, keep_output=args.keep_output)
        if not success:
            sys.exit(1)
    else:
        _full_main(args)


if __name__ == "__main__":
    main()
