#!/usr/bin/env python3
"""Smoke test for the full LoCoMo end-to-end pipeline.

Uses a small subset of conv-26 (sessions 1-3, 5 QA queries) to verify
that session splitting, graph building, retrieval, generation, and
evaluation all work correctly with the current package layout.

Usage:
    python smoke_test.py [--keep-output]
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("smoke_test")

# ---------------------------------------------------------------------------
# Resolve paths relative to this script
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
_BENCHMARK_DIR = _SCRIPT_DIR


def _find_and_load_env() -> None:
    """Load .env from the nearest parent directory of this script."""
    for candidate in _SCRIPT_DIR.resolve().parents:
        env_path = candidate / ".env"
        if env_path.exists():
            try:
                from dotenv import load_dotenv

                load_dotenv(env_path, override=False)
                logger.info("Loaded .env from %s", env_path)
            except ImportError:
                logger.warning("python-dotenv not available, relying on OS env")
            return
    logger.warning("No .env file found in ancestor directories")


# ---------------------------------------------------------------------------
# Phase 1: session splitting & graph building
# ---------------------------------------------------------------------------
def _build_graph(sample: dict, output_dir: Path) -> tuple:
    """Create MemorySystem, write sessions 1-3, and build high-level memory."""
    from mandol import MemorySystem

    conv = sample["conversation"]
    sample_id = sample["sample_id"]

    # Filter to sessions 1-3
    sessions_to_keep = {1, 2, 3}
    trimmed_conv = {}
    for k, v in conv.items():
        if k.startswith("session_") and not k.endswith("_date_time"):
            sn = int(re.match(r"session_(\d+)", k).group(1))
            if sn not in sessions_to_keep:
                continue
        trimmed_conv[k] = v
    trimmed_sample = {**sample, "conversation": trimmed_conv}

    config_path = str(_BENCHMARK_DIR / "configs" / "base.yaml")
    system = MemorySystem.from_yaml_config(config_path, root=sample_id)
    logger.info("MemorySystem created via from_yaml_config")

    # Write dialogues
    from adapter.locomo_adapter import write_sample_to_graph

    t0 = time.time()
    write_sample_to_graph(
        graph=system.graph,
        sample=trimmed_sample,
        batch_embed=True,
    )
    logger.info("Dialogues written in %.1fs", time.time() - t0)

    # Build high-level
    t0 = time.time()
    report = system.build_high_level(mode="auto")
    elapsed = time.time() - t0
    logger.info(
        "build_high_level: status=%s, sessions=%d, units=%d, tokens=%s, time=%.1fs",
        report.status,
        report.sessions_processed,
        report.units_processed,
        report.token_usage,
        elapsed,
    )

    # Collect stats
    stats = {
        "sample_id": sample_id,
        "sessions_processed": report.sessions_processed,
        "units_processed": report.units_processed,
        "build_duration_s": round(elapsed, 1),
        "build_token_usage": report.token_usage,
    }

    if report.status == "error":
        logger.error("Build failed: %s", report.error_message)
        return None, stats

    return system, stats


# ---------------------------------------------------------------------------
# Phase 2: retrieval
# ---------------------------------------------------------------------------
def _run_retrieval(system, queries: list) -> list:
    """Run holistic_retrieve on each query."""
    results = []
    for q in queries:
        t0 = time.time()
        hits = system.holistic_retrieve(q["question"], top_k=10, use_rerank=True)
        elapsed = time.time() - t0
        results.append(
            {
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
            }
        )
        logger.info(
            "Retrieved %d hits for: %s (%.1fs)",
            len(hits),
            q["question"][:60],
            elapsed,
        )
    return results


# ---------------------------------------------------------------------------
# Phase 3: generation
# ---------------------------------------------------------------------------
def _run_generation(retrieval_results: list) -> list:
    """Generate answers using the pipeline prompt template."""
    from pipeline_utils import (
        GENERATION_PROMPT_TEMPLATE,
        extract_final_answer,
        build_llm_provider_from_config,
    )

    llm = build_llm_provider_from_config(str(_BENCHMARK_DIR / "configs" / "base.yaml"))

    results = []
    for r in retrieval_results:
        context = "\n\n---\n\n".join(
            f"[Memory {i + 1}] {h['text']}"
            for i, h in enumerate(r["hits"])
        )
        prompt = (
            GENERATION_PROMPT_TEMPLATE.replace("{question}", r["question"])
            .replace("{context}", context)
        )

        t0 = time.time()
        response = llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2048,
        )
        elapsed = time.time() - t0

        final_answer = extract_final_answer(response.content)
        results.append(
            {
                "question": r["question"],
                "gold_answer": r["answer"],
                "category": r["category"],
                "generated_answer": response.content,
                "generated_answer_extracted": final_answer,
                "token_usage": (
                    response.usage.model_dump()
                    if hasattr(response.usage, "model_dump")
                    else response.usage
                ),
                "duration_s": round(elapsed, 3),
            }
        )
        logger.info(
            "Generated answer for: %s (%.1fs)",
            r["question"][:60],
            elapsed,
        )
    return results


# ---------------------------------------------------------------------------
# Phase 4: evaluation
# ---------------------------------------------------------------------------
def _run_evaluation(generation_results: list) -> list:
    """Evaluate generated answers using the LLM judge."""
    from pipeline_utils import (
        EVALUATION_PROMPT_TEMPLATE,
        parse_judge_label,
        build_llm_provider_from_config,
    )

    llm = build_llm_provider_from_config(str(_BENCHMARK_DIR / "configs" / "base.yaml"))

    results = []
    for g in generation_results:
        prompt = (
            EVALUATION_PROMPT_TEMPLATE.replace("{question}", g["question"])
            .replace("{gold_answer}", str(g["gold_answer"]))
            .replace("{generated_answer}", g["generated_answer_extracted"])
        )

        t0 = time.time()
        response = llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=256,
        )
        elapsed = time.time() - t0

        label = parse_judge_label(response.content)
        results.append(
            {
                "question": g["question"],
                "gold_answer": g["gold_answer"],
                "generated_answer": g["generated_answer_extracted"][:200],
                "judge_label": label,
                "accuracy": 1 if label and "CORRECT" in label.upper() else 0,
                "duration_s": round(elapsed, 3),
            }
        )
        logger.info(
            "Judge: %s → %s (%.1fs)",
            g["question"][:60],
            label,
            elapsed,
        )
    return results


# ---------------------------------------------------------------------------
# Phase 5: persistence round-trip
# ---------------------------------------------------------------------------
def _test_persistence(system, output_dir: Path) -> dict:
    """Save, create fresh system, and load back."""
    from mandol import MemorySystem

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

    # Quick retrieval on loaded system
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="LoCoMo smoke test")
    parser.add_argument(
        "--keep-output",
        action="store_true",
        help="Keep temporary output directory",
    )
    args = parser.parse_args()

    _find_and_load_env()

    # Load sample
    from adapter.locomo_adapter import load_locomo_sample

    dataset_path = str(_BENCHMARK_DIR / "data" / "locomo10.json")
    sample = load_locomo_sample(dataset_path=dataset_path, sample_id="conv-26")
    logger.info("Loaded sample: %s", sample["sample_id"])

    # Select 5 test queries covering all categories
    all_qa = sample["qa"]
    test_queries = [
        {"question": q["question"], "answer": q["answer"], "category": q["category"]}
        for q in all_qa
        if _qa_in_sessions(q, {1, 2, 3}) and q.get("category") != 5
    ][:5]
    logger.info(
        "Selected %d test queries: %s",
        len(test_queries),
        [q["category"] for q in test_queries],
    )

    output_dir = Path(tempfile.mkdtemp(prefix="mandol_smoke_"))
    logger.info("Output directory: %s", output_dir)

    success = True
    results = {"test_queries": len(test_queries)}

    try:
        # Phase 1
        logger.info("=" * 50)
        logger.info("PHASE 1: Session splitting & graph building")
        logger.info("=" * 50)
        system, build_stats = _build_graph(sample, output_dir)
        results["build"] = build_stats
        if system is None:
            logger.error("BUILD FAILED — aborting smoke test")
            success = False
            return

        # Phase 2
        logger.info("=" * 50)
        logger.info("PHASE 2: Retrieval (%d queries)", len(test_queries))
        logger.info("=" * 50)
        retrieval_results = _run_retrieval(system, test_queries)
        results["retrieval"] = {
            "queries": len(retrieval_results),
            "avg_hits": (
                sum(len(r["hits"]) for r in retrieval_results) / len(retrieval_results)
                if retrieval_results
                else 0
            ),
        }

        # Phase 3
        logger.info("=" * 50)
        logger.info("PHASE 3: Generation (%d queries)", len(test_queries))
        logger.info("=" * 50)
        generation_results = _run_generation(retrieval_results)
        results["generation"] = {"queries": len(generation_results)}

        # Phase 4
        logger.info("=" * 50)
        logger.info("PHASE 4: Evaluation (%d queries)", len(test_queries))
        logger.info("=" * 50)
        evaluation_results = _run_evaluation(generation_results)
        acc = sum(e["accuracy"] for e in evaluation_results)
        total = len(evaluation_results)
        results["evaluation"] = {
            "queries": total,
            "correct": acc,
            "accuracy": round(acc / total, 3) if total else 0,
        }

        # Phase 5
        logger.info("=" * 50)
        logger.info("PHASE 5: Persistence round-trip")
        logger.info("=" * 50)
        persistence_stats = _test_persistence(system, output_dir)
        results["persistence"] = persistence_stats

    except Exception:
        logger.exception("Smoke test failed with exception")
        success = False
    finally:
        if not args.keep_output:
            shutil.rmtree(output_dir, ignore_errors=True)
            logger.info("Cleaned up %s", output_dir)
        else:
            logger.info("Output kept at %s", output_dir)

    # Final report
    logger.info("=" * 60)
    logger.info("SMOKE TEST REPORT")
    logger.info("=" * 60)
    logger.info("Test queries: %d", results.get("test_queries", 0))

    build = results.get("build", {})
    logger.info(
        "Build: %d sessions, %d units, status=%s",
        build.get("sessions_processed", 0),
        build.get("units_processed", 0),
        "OK" if build else "FAILED",
    )

    retrieval = results.get("retrieval", {})
    logger.info("Retrieval: %d queries, avg %.1f hits", retrieval.get("queries", 0), retrieval.get("avg_hits", 0))

    gen = results.get("generation", {})
    logger.info("Generation: %d queries", gen.get("queries", 0))

    eval_stats = results.get("evaluation", {})
    logger.info(
        "Evaluation: %d/%d correct (%.1f%%)",
        eval_stats.get("correct", 0),
        eval_stats.get("queries", 0),
        eval_stats.get("accuracy", 0) * 100,
    )

    pers = results.get("persistence", {})
    logger.info(
        "Persistence: save=%d units, load retrieval=%d hits",
        pers.get("save_units", 0),
        pers.get("loaded_retrieval_hits", 0),
    )

    if success:
        logger.info("SMOKE TEST PASSED")
    else:
        logger.error("SMOKE TEST FAILED")
        sys.exit(1)


def _qa_in_sessions(q: dict, allowed: set) -> bool:
    """Check if a QA pair only references sessions in *allowed*."""
    evidence = q.get("evidence", [])
    if not evidence:
        return False
    for e in evidence:
        m = re.match(r"D(\d+):", str(e))
        if m and int(m.group(1)) not in allowed:
            return False
    return True


if __name__ == "__main__":
    main()
