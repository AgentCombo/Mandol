#!/usr/bin/env python3
"""Generate a latency README from a smart_search QPS benchmark JSON report.

Only formal test records under ``samples[*].records`` are included. Global
warmup and per-sample warmup records are intentionally ignored.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional


NumberGetter = Callable[[Dict[str, Any]], Optional[float]]


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


def _processing_latency(record: Dict[str, Any]) -> Optional[float]:
	return _first_number(record, "latency_ms", "processing_latency_ms", "process_latency_ms")


def _schedule_latency(record: Dict[str, Any]) -> Optional[float]:
	return _first_number(record, "schedule_drift_ms", "schedule_latency_ms", "drift_ms")


def _e2e_latency(record: Dict[str, Any]) -> Optional[float]:
	processing = _processing_latency(record)
	schedule = _schedule_latency(record)
	if processing is None:
		return None
	return processing + (schedule or 0.0)


def _retrieval_latency(record: Dict[str, Any]) -> Optional[float]:
	return _first_number(record, "retrieval_time_ms", "retrieval_latency_ms", "retrieval_time")


def _rerank_latency(record: Dict[str, Any]) -> Optional[float]:
	return _first_number(record, "rerank_time_ms", "rerank_latency_ms", "rerank_time")


METRICS: List[MetricSpec] = [
	MetricSpec(
		key="processing_latency_ms",
		label="处理延迟",
		description="单请求实际处理耗时，优先读取 latency_ms。",
		getter=_processing_latency,
	),
	MetricSpec(
		key="schedule_latency_ms",
		label="调度延迟",
		description="请求实际启动时间相对计划启动时间的漂移，优先读取 schedule_drift_ms。",
		getter=_schedule_latency,
	),
	MetricSpec(
		key="e2e_latency_ms",
		label="E2E 延迟",
		description="处理延迟 + 调度延迟。",
		getter=_e2e_latency,
	),
	MetricSpec(
		key="retrieval_latency_ms",
		label="检索延迟",
		description="检索阶段耗时，优先读取 retrieval_time_ms。",
		getter=_retrieval_latency,
	),
	MetricSpec(
		key="rerank_latency_ms",
		label="重排序延迟",
		description="重排序阶段耗时，优先读取 rerank_time_ms。",
		getter=_rerank_latency,
	),
]


def percentile(values: List[float], percent: float) -> Optional[float]:
	"""Return percentile using linear interpolation, similar to numpy.percentile."""
	if not values:
		return None
	ordered = sorted(values)
	if len(ordered) == 1:
		return ordered[0]

	rank = (len(ordered) - 1) * (percent / 100.0)
	lower = math.floor(rank)
	upper = math.ceil(rank)
	if lower == upper:
		return ordered[int(rank)]

	weight = rank - lower
	return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_values(values: List[float]) -> Dict[str, Optional[float]]:
	if not values:
		return {"count": 0, "mean": None, "p90": None, "p99": None}
	return {
		"count": len(values),
		"mean": sum(values) / len(values),
		"p90": percentile(values, 90),
		"p99": percentile(values, 99),
	}


def load_report(json_path: Path) -> Dict[str, Any]:
	with json_path.open("r", encoding="utf-8") as file:
		data = json.load(file)
	if not isinstance(data, dict):
		raise ValueError(f"Benchmark report must be a JSON object: {json_path}")
	return data


def iter_formal_records(report: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
	"""Yield only formal sample records, excluding global and per-sample warmups."""
	samples = report.get("samples", {})
	if not isinstance(samples, dict):
		return

	for sample_id, sample_data in samples.items():
		if not isinstance(sample_data, dict) or sample_data.get("skipped"):
			continue
		records = sample_data.get("records", [])
		if not isinstance(records, list):
			continue
		for record in records:
			if not isinstance(record, dict):
				continue
			normalized = dict(record)
			normalized.setdefault("sample_id", sample_id)
			yield normalized


def collect_metric_values(records: List[Dict[str, Any]]) -> Dict[str, List[float]]:
	values_by_metric: Dict[str, List[float]] = {metric.key: [] for metric in METRICS}
	for record in records:
		for metric in METRICS:
			value = metric.getter(record)
			if value is not None:
				values_by_metric[metric.key].append(value)
	return values_by_metric


def summarize_by_metric(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Optional[float]]]:
	values_by_metric = collect_metric_values(records)
	return {
		metric.key: summarize_values(values_by_metric[metric.key])
		for metric in METRICS
	}


def group_records_by_sample(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
	grouped: Dict[str, List[Dict[str, Any]]] = {}
	for record in records:
		sample_id = str(record.get("sample_id") or "unknown")
		grouped.setdefault(sample_id, []).append(record)
	return grouped


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


def _metric_table(summary: Dict[str, Dict[str, Optional[float]]]) -> List[str]:
	lines = [
		"| 指标 | 样本数 | 平均值(ms) | P90(ms) | P99(ms) | 说明 |",
		"| --- | ---: | ---: | ---: | ---: | --- |",
	]
	for metric in METRICS:
		stats = summary[metric.key]
		lines.append(
			"| {label} | {count} | {mean} | {p90} | {p99} | {desc} |".format(
				label=metric.label,
				count=stats["count"],
				mean=_format_ms(stats["mean"]),
				p90=_format_ms(stats["p90"]),
				p99=_format_ms(stats["p99"]),
				desc=metric.description,
			)
		)
	return lines


def _sample_table(grouped_records: Dict[str, List[Dict[str, Any]]]) -> List[str]:
	lines = [
		"| Sample | 请求数 | 成功数 | 失败数 | 处理均值(ms) | 调度均值(ms) | E2E P99(ms) | 检索均值(ms) | 重排序均值(ms) |",
		"| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
	]
	for sample_id in sorted(grouped_records):
		records = grouped_records[sample_id]
		summary = summarize_by_metric(records)
		successes = sum(1 for record in records if record.get("success") is True)
		failures = sum(1 for record in records if record.get("success") is False)
		lines.append(
			"| {sample} | {total} | {successes} | {failures} | {processing} | {schedule} | {e2e_p99} | {retrieval} | {rerank} |".format(
				sample=sample_id,
				total=len(records),
				successes=successes,
				failures=failures,
				processing=_format_ms(summary["processing_latency_ms"]["mean"]),
				schedule=_format_ms(summary["schedule_latency_ms"]["mean"]),
				e2e_p99=_format_ms(summary["e2e_latency_ms"]["p99"]),
				retrieval=_format_ms(summary["retrieval_latency_ms"]["mean"]),
				rerank=_format_ms(summary["rerank_latency_ms"]["mean"]),
			)
		)
	return lines


def build_markdown_report(json_path: Path, report: Dict[str, Any]) -> str:
	records = list(iter_formal_records(report))
	grouped_records = group_records_by_sample(records)
	summary = summarize_by_metric(records)

	config = report.get("config", {}) if isinstance(report.get("config"), dict) else {}
	aggregate_summary = report.get("aggregate_summary", {})
	if not isinstance(aggregate_summary, dict):
		aggregate_summary = {}

	successful = sum(1 for record in records if record.get("success") is True)
	failed = sum(1 for record in records if record.get("success") is False)
	generated_at = datetime.now(timezone.utc).isoformat()

	lines: List[str] = [
		"# QPS Latency Analysis",
		"",
		f"- Source JSON: `{json_path.name}`",
		f"- Generated at: `{generated_at}`",
		"- Included records: formal test records from `samples[*].records` only.",
		"- Excluded records: top-level `warmup` and per-sample `sample_warmup` records.",
		"",
		"## Run Summary",
		"",
		f"- Target QPS: `{_format_number(config.get('qps'))}`",
		f"- Top K: `{_format_number(config.get('top_k'))}`",
		f"- Rerank method: `{_format_number(config.get('rerank_method'))}`",
		f"- Search mode: `{_format_number(config.get('search_mode'))}`",
		f"- Formal samples: `{len(grouped_records)}`",
		f"- Formal requests: `{len(records)}`",
		f"- Successful requests: `{successful}`",
		f"- Failed requests: `{failed}`",
		f"- Aggregate actual throughput QPS: `{_format_number(aggregate_summary.get('actual_throughput_qps'))}`",
		"",
		"## Latency Summary",
		"",
		*_metric_table(summary),
		"",
		"## Per-Sample Summary",
		"",
		*_sample_table(grouped_records),
		"",
		"## Field Mapping",
		"",
		"- 处理延迟: `latency_ms`",
		"- 调度延迟: `schedule_drift_ms`",
		"- E2E 延迟: `latency_ms + schedule_drift_ms`",
		"- 检索延迟: `retrieval_time_ms`",
		"- 重排序延迟: `rerank_time_ms`",
		"",
	]
	return "\n".join(lines)


def default_output_path(json_path: Path) -> Path:
	return json_path.parent / "README_qps_latency.md"


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Analyze formal-test latency metrics from a QPS benchmark JSON report.",
	)
	parser.add_argument(
		"json_path",
		type=Path,
		help="Path to qps_benchmark_report_*.json.",
	)
	parser.add_argument(
		"-o",
		"--output",
		type=Path,
		default=None,
		help="Output markdown path. Defaults to README_qps_latency.md beside the JSON file.",
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
	print(f"Wrote latency analysis to {output_path}")


if __name__ == "__main__":
	main()
