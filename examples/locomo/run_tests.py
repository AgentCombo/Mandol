#!/usr/bin/env python3
"""Memory system test runner for the LoCoMo dataset.

Runs a configurable set of tests against a :class:`LocomoMemorySystem`,
collects pass/fail results, and writes a JSON + Markdown report to the
``reports/`` directory.

Usage:
    python run_tests.py                       # Default sample count from config
    python run_tests.py --sample-count 3      # Override sample count
    python run_tests.py --skip-tests retrieval  # Skip specific test groups
"""
from __future__ import annotations

import json
import logging
import sys
import argparse
from datetime import datetime, timezone
from pathlib import Path

from locomo.config import LocomoMemoryConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def generate_report(report_data: dict, output_dir: Path) -> Path:
    """Write a JSON and Markdown test report to *output_dir*.

    Args:
        report_data: Dict with ``summary``, ``tests``, and timing fields.
        output_dir: Directory where report files are written.

    Returns:
        Path to the generated JSON report file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"test_report_{timestamp}.json"

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    md_path = output_dir / f"test_report_{timestamp}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Memory System Test Report\n\n")
        f.write(f"**Test Run Date**: {report_data.get('test_start_time', 'N/A')}\n\n")
        f.write(f"**Total Duration**: {report_data.get('total_duration_seconds', 0):.2f} seconds\n\n")

        summary = report_data.get("summary", {})
        f.write("## Summary\n\n")
        f.write(f"- **Total Tests**: {summary.get('total_tests', 0)}\n")
        f.write(f"- **Passed**: {summary.get('passed', 0)}\n")
        f.write(f"- **Failed**: {summary.get('failed', 0)}\n")
        f.write(f"- **Pass Rate**: {summary.get('pass_rate', '0%')}\n\n")

        f.write("## Test Results\n\n")
        f.write("| Test Name | Status | Duration (s) | Details |\n")
        f.write("|-----------|--------|-------------|--------|\n")
        for test in report_data.get("tests", []):
            status_icon = "PASS" if test.get("status") == "PASS" else "FAIL"
            details_str = json.dumps(test.get("details", {}))[:100]
            f.write(f"| {test.get('test_name', '')} | {status_icon} | {test.get('duration_seconds', 0):.2f} | {details_str} |\n")

        f.write("\n## Detailed Test Data\n\n")
        for test in report_data.get("tests", []):
            f.write(f"### {test.get('test_name', '')}\n\n")
            f.write(f"**Status**: {test.get('status', 'UNKNOWN')}\n\n")
            f.write(f"**Duration**: {test.get('duration_seconds', 0):.2f}s\n\n")
            f.write(f"**Details**:\n```json\n{json.dumps(test.get('details', {}), indent=2, ensure_ascii=False)}\n```\n\n")

        f.write("\n## Issues and Recommendations\n\n")
        failed_tests = [t for t in report_data.get("tests", []) if t.get("status") == "FAIL"]
        if failed_tests:
            f.write("### Identified Issues\n\n")
            for test in failed_tests:
                f.write(f"- **{test.get('test_name', '')}**: {test.get('details', {}).get('error', 'Unknown error')}\n")
            f.write("\n### Recommendations\n\n")
            f.write("1. Review failed test cases and fix underlying issues\n")
            f.write("2. Ensure all required services (LLM, Embedding, Reranker) are accessible\n")
            f.write("3. Verify dataset path and permissions\n")
        else:
            f.write("All tests passed successfully!\n")

    logger.info(f"Report saved to: {report_path}")
    logger.info(f"Markdown report saved to: {md_path}")
    return report_path


def main():
    """Entry point: configure the tester, run all tests, and save the report."""
    parser = argparse.ArgumentParser(description="Memory System Test Runner for LoCoMo Dataset")
    parser.add_argument("--sample-count", type=int, default=None, help="Number of samples to process (default: from config)")
    parser.add_argument("--skip-tests", nargs="+", default=[], help="Tests to skip")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Memory System Test Runner for LoCoMo Dataset")
    logger.info("=" * 60)

    config = LocomoMemoryConfig()

    if args.sample_count is not None:
        config.sample_count = args.sample_count
        logger.info(f"Overriding sample_count to: {args.sample_count}")

    tester = MemorySystemTester(config=config, skip_tests=args.skip_tests)
    report_data = tester.run_all_tests()

    output_dir = Path(__file__).parent / "reports"
    report_path = generate_report(report_data, output_dir)

    summary = report_data.get("summary", {})
    if summary.get("failed", 0) > 0:
        logger.warning(f"Tests completed with {summary.get('failed', 0)} failures")
        sys.exit(1)
    else:
        logger.info("All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
