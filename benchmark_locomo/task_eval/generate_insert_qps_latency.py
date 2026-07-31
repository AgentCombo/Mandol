#!/usr/bin/env python3
"""Generate a latency README from a LoCoMo insert QPS benchmark JSON report.

The insert benchmark stores formal test records in the top-level ``records``
array. Warmup latency in ``summary.config.warmup_latency_ms`` is metadata only
and is intentionally excluded from all latency statistics.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional


NumberGetter = Callable[[Dict[str, Any], Dict[str, Any]], Optional[float]]


@dataclass(frozen=True)
class MetricSpec:
	key: str
	label: str
	description: str
	getter: NumberGetter


def _as_float(value: Any) -> Optional[float]:
	if value is None:
		return None
	try:
		number = float(value)
	except (TypeError, ValueError):
		return None
	if math.isnan(number) or math.isinf(number):
		return None
	return number


def _first_number(record: Dict[str, Any], *keys: str) -> Optional[float]:
	for key in keys:
		value = _as_float(record.get(key))
		if value is not None:
			return value
	return None


def _processing_latency(record: Dict[str, Any], _: Dict[str, Any]) -> Optional[float]:
	return _first_number(record, "latency_ms", "processing_latency_ms", "process_latency_ms")


def _schedule_latency(record: Dict[str, Any], _: Dict[str, Any]) -> Optional[float]:
	schedule_latency = _first_number(record, "schedule_drift_ms", "schedule_latency_ms", "drift_ms")
	if schedule_latency is not None:
		return schedule_latency

	expected_offset = _as_float(record.get("expected_start_offset_s"))
	actual_offset = _as_float(record.get("actual_start_offset_s"))
	if expected_offset is None or actual_offset is None:
		return None
	return (actual_offset - expected_offset) * 1000.0


def _e2e_latency(record: Dict[str, Any], summary: Dict[str, Any]) -> Optional[float]:
	processing_latency = _processing_latency(record, summary)
	schedule_latency = _schedule_latency(record, summary)
	if processing_latency is None:
		return None
	return processing_latency + (schedule_latency or 0.0)


def _budget_overrun_latency(record: Dict[str, Any], summary: Dict[str, Any]) -> Optional[float]:
	processing_latency = _processing_latency(record, summary)
	target_qps = _as_float(summary.get("target_qps")) or _as_float(
		summary.get("config", {}).get("qps") if isinstance(summary.get("config"), dict) else None
	)
	if processing_latency is None or target_qps is None or target_qps <= 0:
		return None
	budget_ms = 1000.0 / target_qps
	return max(processing_latency - budget_ms, 0.0)


METRICS: List[MetricSpec] = [
	MetricSpec(
		key="processing_latency_ms",
		label="处理延迟",
		description="单次 add_unit 插入处理耗时，优先读取 latency_ms。",
		getter=_processing_latency,
	),
	MetricSpec(
		key="schedule_latency_ms",
		label="调度延迟",
		description="实际启动时间相对计划启动时间的漂移，优先读取 schedule_drift_ms。",
		getter=_schedule_latency,
	),
	MetricSpec(
		key="e2e_latency_ms",
		label="E2E 延迟",
		description="处理延迟 + 调度延迟。",
		getter=_e2e_latency,
	),
	MetricSpec(
		key="budget_overrun_latency_ms",
		label="超预算延迟",
		description="处理延迟超过单请求 QPS 预算的部分，未超过时记为 0。",
		getter=_budget_overrun_latency,
	),
]


def percentile(values: List[float], percent: float) -> Optional[float]:
	if not values:
		return None
	ordered_values = sorted(values)
	if len(ordered_values) == 1:
		return ordered_values[0]

	rank = (len(ordered_values) - 1) * (percent / 100.0)
	lower_index = math.floor(rank)
	upper_index = math.ceil(rank)
	if lower_index == upper_index:
		return ordered_values[int(rank)]

	weight = rank - lower_index
	return ordered_values[lower_index] * (1.0 - weight) + ordered_values[upper_index] * weight


def summarize_values(values: List[float]) -> Dict[str, Any]:
	if not values:
		return {"count": 0, "mean": None, "p90": None, "p99": None, "min": None, "max": None}
	return {
		"count": len(values),
		"mean": sum(values) / len(values),
		"p90": percentile(values, 90),
		"p99": percentile(values, 99),
		"min": min(values),
		"max": max(values),
	}


def load_report(json_path: Path) -> Dict[str, Any]:
	with json_path.open("r", encoding="utf-8") as file:
		report = json.load(file)
	if not isinstance(report, dict):
		raise ValueError(f"Benchmark report must be a JSON object: {json_path}")
	return report


def get_summary(report: Dict[str, Any]) -> Dict[str, Any]:
	summary = report.get("summary", {})
	return summary if isinstance(summary, dict) else {}


def iter_formal_records(report: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
	records = report.get("records", [])
	if not isinstance(records, list):
		return

	for record in records:
		if not isinstance(record, dict):
			continue
		if record.get("is_warmup") is True:
			continue
		phase = str(record.get("phase") or record.get("record_type") or "").lower()
		if phase == "warmup":
			continue
		yield record


def collect_metric_values(records: List[Dict[str, Any]], summary: Dict[str, Any]) -> Dict[str, List[float]]:
	values_by_metric: Dict[str, List[float]] = {metric.key: [] for metric in METRICS}
	for record in records:
		for metric in METRICS:
			value = metric.getter(record, summary)
			if value is not None:
				values_by_metric[metric.key].append(value)
	return values_by_metric


def summarize_by_metric(records: List[Dict[str, Any]], summary: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
	values_by_metric = collect_metric_values(records, summary)
	return {metric.key: summarize_values(values_by_metric[metric.key]) for metric in METRICS}


def group_records_by_memory_level(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
	grouped_records: Dict[str, List[Dict[str, Any]]] = {}
	for record in records:
		memory_level = str(record.get("memory_level") or "unknown")
		grouped_records.setdefault(memory_level, []).append(record)
	return grouped_records


def _format_ms(value: Optional[float]) -> str:
	if value is None:
		return "N/A"
	return f"{value:.3f}"


def _format_number(value: Any) -> str:
	if value is None:
		return "N/A"
	if isinstance(value, float):
		return f"{value:.3f}"
	return str(value)


def _format_percent(value: Any) -> str:
	number = _as_float(value)
	if number is None:
		return "N/A"
	return f"{number:.2f}%"


def _metric_table(metric_summary: Dict[str, Dict[str, Any]]) -> List[str]:
	lines = [
		"| 指标 | 样本数 | 平均值(ms) | P90(ms) | P99(ms) | 最小值(ms) | 最大值(ms) | 说明 |",
		"| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
	]
	for metric in METRICS:
		stats = metric_summary[metric.key]
		lines.append(
			"| {label} | {count} | {mean} | {p90} | {p99} | {min_value} | {max_value} | {desc} |".format(
				label=metric.label,
				count=stats["count"],
				mean=_format_ms(stats["mean"]),
				p90=_format_ms(stats["p90"]),
				p99=_format_ms(stats["p99"]),
				min_value=_format_ms(stats["min"]),
				max_value=_format_ms(stats["max"]),
				desc=metric.description,
			)
		)
	return lines


def _memory_level_table(
	grouped_records: Dict[str, List[Dict[str, Any]]],
	summary: Dict[str, Any],
) -> List[str]:
	lines = [
		"| Memory Level | 请求数 | 成功数 | 失败数 | 处理均值(ms) | 处理 P90(ms) | 处理 P99(ms) | 调度均值(ms) | E2E P99(ms) |",
		"| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
	]
	for memory_level in sorted(grouped_records):
		records = grouped_records[memory_level]
		metric_summary = summarize_by_metric(records, summary)
		successes = sum(1 for record in records if record.get("success") is True)
		failures = sum(1 for record in records if record.get("success") is False)
		lines.append(
			"| {level} | {total} | {successes} | {failures} | {processing_mean} | {processing_p90} | {processing_p99} | {schedule_mean} | {e2e_p99} |".format(
				level=memory_level,
				total=len(records),
				successes=successes,
				failures=failures,
				processing_mean=_format_ms(metric_summary["processing_latency_ms"]["mean"]),
				processing_p90=_format_ms(metric_summary["processing_latency_ms"]["p90"]),
				processing_p99=_format_ms(metric_summary["processing_latency_ms"]["p99"]),
				schedule_mean=_format_ms(metric_summary["schedule_latency_ms"]["mean"]),
				e2e_p99=_format_ms(metric_summary["e2e_latency_ms"]["p99"]),
			)
		)
	return lines


def build_markdown_report(json_path: Path, report: Dict[str, Any]) -> str:
	summary = get_summary(report)
	config = summary.get("config", {}) if isinstance(summary.get("config"), dict) else {}
	records = list(iter_formal_records(report))
	metric_summary = summarize_by_metric(records, summary)
	grouped_records = group_records_by_memory_level(records)

	successful = sum(1 for record in records if record.get("success") is True)
	failed = sum(1 for record in records if record.get("success") is False)
	target_qps = _as_float(summary.get("target_qps")) or _as_float(config.get("qps"))
	budget_ms = 1000.0 / target_qps if target_qps and target_qps > 0 else None
	generated_at = datetime.now(timezone.utc).isoformat()

	lines: List[str] = [
		"# Insert QPS Latency Analysis",
		"",
		f"- Source JSON: `{json_path.name}`",
		f"- Generated at: `{generated_at}`",
		"- Included records: formal insert records from top-level `records` only.",
		"- Excluded warmup: `summary.config.warmup_latency_ms` is reported as metadata only and is not included in any latency statistics.",
		"",
		"## Run Summary",
		"",
		f"- Target QPS: `{_format_number(target_qps)}`",
		f"- Per-request QPS budget: `{_format_ms(budget_ms)} ms`",
		f"- Total formal requests: `{len(records)}`",
		f"- Successful requests: `{successful}`",
		f"- Failed requests: `{failed}`",
		f"- Success rate: `{_format_percent(summary.get('success_rate'))}`",
		f"- Over-budget requests: `{_format_number(summary.get('over_budget_count'))}`",
		f"- Warmup latency excluded: `{_format_ms(_as_float(config.get('warmup_latency_ms')))} ms`",
		f"- Init time: `{_format_number(config.get('init_time_s'))} s`",
		f"- Embedding model: `{_format_number(config.get('embedding_model'))}`",
		"",
		"## Latency Summary",
		"",
		*_metric_table(metric_summary),
		"",
		"## Memory-Level Summary",
		"",
		*_memory_level_table(grouped_records, summary),
		"",
		"## Field Mapping",
		"",
		"- 处理延迟: `latency_ms`",
		"- 调度延迟: `schedule_drift_ms`; if missing, fallback to `(actual_start_offset_s - expected_start_offset_s) * 1000`",
		"- E2E 延迟: `latency_ms + schedule_drift_ms`",
		"- 超预算延迟: `max(latency_ms - 1000 / target_qps, 0)`",
		"",
	]
	return "\n".join(lines)


def default_output_path(json_path: Path) -> Path:
	return json_path.parent / "README_insert_qps_latency.md"


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Analyze formal insert latency metrics from a LoCoMo QPS benchmark JSON report.",
	)
	parser.add_argument(
		"json_path",
		type=Path,
		help="Path to benchmark_triple_input_speed_*.json.",
	)
	parser.add_argument(
		"-o",
		"--output",
		type=Path,
		default=None,
		help="Output markdown path. Defaults to README_insert_qps_latency.md beside the JSON file.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	json_path = args.json_path.expanduser().resolve()
	if not json_path.exists():
		raise FileNotFoundError(f"JSON file not found: {json_path}")
	if not json_path.is_file():
		raise ValueError(f"JSON path is not a file: {json_path}")

	output_path = args.output.expanduser().resolve() if args.output else default_output_path(json_path)
	report = load_report(json_path)
	markdown = build_markdown_report(json_path, report)

	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text(markdown, encoding="utf-8")
	print(f"Wrote insert latency analysis to {output_path}")


if __name__ == "__main__":
	main()
