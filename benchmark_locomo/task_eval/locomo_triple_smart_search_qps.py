#!/usr/bin/env python3
"""Fixed-QPS benchmark for MultiRetriever.smart_search on unified LoCoMo graphs."""

from __future__ import annotations

import argparse
import asyncio
import contextvars
import gc
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from mandol.core.semantic_graph import SemanticGraph
from mandol.retrieval.advance_retriever import cleanup_retrieval_resources
from mandol.retrieval.retrieval_interface import RetrievalMethod
from mandol.utils.config_manager import settings


LOGGER = logging.getLogger("benchmark_smart_search_qps")
WARMUP_SAMPLE_ID = "conv-26"
DEFAULT_UNIFIED_DIR = PROJECT_ROOT / "benchmark_locomo" / "dataset" / "locomo" / "unified_per_sample_graphs"
DEFAULT_DATA_PATH = PROJECT_ROOT / "benchmark_locomo" / "dataset" / "locomo" / "locomo10.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "benchmark_locomo" / "task_eval" / "results" / "smart_search_qps_results"
DEFAULT_METHODS = [
	RetrievalMethod.BM25,
	RetrievalMethod.COSINE_SIMILARITY,
	RetrievalMethod.SPLADE,
]


@dataclass(frozen=True)
class BenchmarkConfig:
	unified_dir: Path
	data_dir: Path
	qps: float
	sample_ids: List[str]
	top_k: int = 35
	rerank_method: Optional[str] = None
	search_mode: str = "auto"
	sample_warmup_requests: int = 12
	output_json: Optional[Path] = None
	warmup_sample_id: str = WARMUP_SAMPLE_ID
	log_level: str = "INFO"


@dataclass(frozen=True)
class QueryItem:
	sample_id: str
	question_id: str
	question: str
	answer: str = ""
	category: Optional[Any] = None
	evidence: List[Any] = field(default_factory=list)


@dataclass
class PhaseTiming:
	retrieval_time_ms: float = 0.0
	rerank_time_ms: float = 0.0


@dataclass
class RequestRecord:
	sample_id: str
	question_id: str
	question: str
	index: int
	category: Optional[Any]
	success: bool
	error: Optional[str]
	result_count: int
	latency_ms: float
	retrieval_time_ms: float
	rerank_time_ms: float
	schedule_drift_ms: float
	scheduled_offset_s: float
	started_at: str
	completed_at: str


@dataclass
class RunSummary:
	total_requests: int
	successful_requests: int
	failed_requests: int
	success_rate: float
	actual_throughput_qps: float
	duration_seconds: float
	latency_mean_ms: Optional[float]
	latency_p50_ms: Optional[float]
	latency_p90_ms: Optional[float]
	latency_p95_ms: Optional[float]
	latency_p99_ms: Optional[float]
	retrieval_mean_ms: Optional[float]
	rerank_mean_ms: Optional[float]
	drift_mean_ms: Optional[float]
	drift_p95_ms: Optional[float]


_CURRENT_TIMING: contextvars.ContextVar[Optional[PhaseTiming]] = contextvars.ContextVar(
	"smart_search_qps_phase_timing",
	default=None,
)


class SmartSearchProfiler:
	"""Attach per-request phase timers to one MultiRetriever instance."""

	BASE_MARKER = "_smart_search_qps_profiler_installed"

	@classmethod
	def instrument(cls, retriever: Any) -> None:
		if getattr(retriever, cls.BASE_MARKER, False):
			return

		original_base_retrieval = retriever._execute_base_retrieval
		original_reranking = retriever._execute_reranking
		original_base_retrieval_async = getattr(retriever, "_execute_base_retrieval_async", None)
		original_reranking_async = getattr(retriever, "_execute_reranking_async", None)

		def timed_base_retrieval(*args: Any, **kwargs: Any) -> Any:
			start = time.perf_counter()
			try:
				return original_base_retrieval(*args, **kwargs)
			finally:
				timing = _CURRENT_TIMING.get()
				if timing is not None:
					timing.retrieval_time_ms += (time.perf_counter() - start) * 1000.0

		def timed_reranking(*args: Any, **kwargs: Any) -> Any:
			start = time.perf_counter()
			try:
				return original_reranking(*args, **kwargs)
			finally:
				timing = _CURRENT_TIMING.get()
				if timing is not None:
					timing.rerank_time_ms += (time.perf_counter() - start) * 1000.0

		async def timed_base_retrieval_async(*args: Any, **kwargs: Any) -> Any:
			start = time.perf_counter()
			try:
				return await original_base_retrieval_async(*args, **kwargs)
			finally:
				timing = _CURRENT_TIMING.get()
				if timing is not None:
					timing.retrieval_time_ms += (time.perf_counter() - start) * 1000.0

		async def timed_reranking_async(*args: Any, **kwargs: Any) -> Any:
			start = time.perf_counter()
			try:
				return await original_reranking_async(*args, **kwargs)
			finally:
				timing = _CURRENT_TIMING.get()
				if timing is not None:
					timing.rerank_time_ms += (time.perf_counter() - start) * 1000.0

		retriever._execute_base_retrieval = timed_base_retrieval
		retriever._execute_reranking = timed_reranking
		if original_base_retrieval_async is not None:
			retriever._execute_base_retrieval_async = timed_base_retrieval_async
		if original_reranking_async is not None:
			retriever._execute_reranking_async = timed_reranking_async
		setattr(retriever, cls.BASE_MARKER, True)


class LoCoMoQueryLoader:
	"""Load LoCoMo QA queries from extracted or raw dataset JSON files."""

	FILE_NAMES = (
		"locomo_qa_by_sample.json",
		"locomo_qa_extracted.json",
		"locomo10.json",
	)

	def __init__(self, data_dir: Path):
		self.data_dir = data_dir

	def load(self) -> Dict[str, List[QueryItem]]:
		payload, source_path = self._load_payload()
		queries = self._parse_payload(payload)
		if not queries:
			raise ValueError(f"No queries found in {source_path}")

		total = sum(len(items) for items in queries.values())
		LOGGER.info("Loaded %d queries from %s (%d samples)", total, source_path, len(queries))
		return queries

	def _load_payload(self) -> Tuple[Any, Path]:
		candidates = self._candidate_files()
		for path in candidates:
			if path.exists() and path.is_file():
				with path.open("r", encoding="utf-8") as handle:
					return json.load(handle), path
		raise FileNotFoundError(
			"Could not find LoCoMo query JSON. Checked: "
			+ ", ".join(str(path) for path in candidates)
		)

	def _candidate_files(self) -> List[Path]:
		if self.data_dir.is_file():
			return [self.data_dir]

		candidates: List[Path] = []
		roots = [self.data_dir]
		if self.data_dir.parent != self.data_dir:
			roots.append(self.data_dir.parent)
		for root in roots:
			candidates.extend(root / name for name in self.FILE_NAMES)
			candidates.extend(root / "locomo" / name for name in self.FILE_NAMES)

		if self.data_dir.exists():
			for name in self.FILE_NAMES:
				candidates.extend(sorted(self.data_dir.rglob(name))[:3])

		deduped: Dict[Path, None] = {}
		for candidate in candidates:
			deduped[candidate] = None
		return list(deduped.keys())

	def _parse_payload(self, payload: Any) -> Dict[str, List[QueryItem]]:
		queries: Dict[str, List[QueryItem]] = {}

		if isinstance(payload, dict) and isinstance(payload.get("samples"), dict):
			for sample_id, sample_payload in payload["samples"].items():
				self._add_sample_queries(queries, str(sample_id), sample_payload)
			return queries

		if isinstance(payload, dict) and isinstance(payload.get("samples"), list):
			for item in payload["samples"]:
				sample_id = self._sample_id_from_item(item)
				if sample_id:
					self._add_sample_queries(queries, sample_id, item)
			return queries

		if isinstance(payload, dict) and isinstance(payload.get("qa_pairs"), list):
			for index, qa_item in enumerate(payload["qa_pairs"]):
				sample_id = self._sample_id_from_item(qa_item)
				if sample_id:
					query = self._query_from_qa(sample_id, qa_item, index)
					if query:
						queries.setdefault(sample_id, []).append(query)
			return queries

		if isinstance(payload, dict) and isinstance(payload.get("data"), list):
			return self._parse_payload(payload["data"])

		if isinstance(payload, list):
			for item in payload:
				sample_id = self._sample_id_from_item(item)
				if sample_id:
					self._add_sample_queries(queries, sample_id, item)
			return queries

		return queries

	def _add_sample_queries(self, queries: Dict[str, List[QueryItem]], sample_id: str, sample_payload: Any) -> None:
		if not isinstance(sample_payload, dict):
			return
		qa_items = (
			sample_payload.get("questions")
			or sample_payload.get("qa")
			or sample_payload.get("qa_pairs")
			or []
		)
		if not isinstance(qa_items, list):
			return

		for index, qa_item in enumerate(qa_items):
			query = self._query_from_qa(sample_id, qa_item, index)
			if query:
				queries.setdefault(sample_id, []).append(query)

	def _query_from_qa(self, sample_id: str, qa_item: Any, index: int) -> Optional[QueryItem]:
		if not isinstance(qa_item, dict):
			return None

		question = qa_item.get("question") or qa_item.get("query")
		if not isinstance(question, str) or not question.strip():
			return None

		question_id = qa_item.get("question_id") or qa_item.get("id") or f"{sample_id}_q{index + 1}"
		answer = qa_item.get("answer") or qa_item.get("expected_answer") or qa_item.get("adversarial_answer") or ""
		evidence = qa_item.get("evidence") or []
		if not isinstance(evidence, list):
			evidence = [evidence]

		return QueryItem(
			sample_id=sample_id,
			question_id=str(question_id),
			question=question.strip(),
			answer=str(answer) if answer is not None else "",
			category=qa_item.get("category"),
			evidence=evidence,
		)

	def _sample_id_from_item(self, item: Any) -> Optional[str]:
		if not isinstance(item, dict):
			return None
		for key in ("sample_id", "sample", "conversation_id", "conv_id", "id"):
			value = item.get(key)
			if value is None:
				continue
			value_text = str(value)
			if value_text.startswith("conv-"):
				return value_text
			if key in {"conversation_id", "conv_id"} and value_text.isdigit():
				return f"conv-{value_text}"
			if key == "sample_id":
				return value_text
		return None


@dataclass
class LoadedSampleGraph:
	sample_id: str
	graph_dir: Path
	graph: SemanticGraph
	retriever: Any


class UnifiedGraphLoader:
	"""Load one unified SemanticGraph and prepare its MultiRetriever."""

	GRAPH_MARKERS = (
		"graph_state.json",
		"rx_graph.pkl",
		"semantic_graph.json",
		"semantic_map_data/semantic_map_meta.json",
	)

	def __init__(self, unified_dir: Path):
		self.unified_dir = unified_dir

	def discover_samples(self) -> List[str]:
		if not self.unified_dir.exists():
			return []
		sample_ids = [path.name for path in self.unified_dir.iterdir() if self._looks_like_graph_dir(path)]
		return sorted(sample_ids)

	def load(self, sample_id: str) -> LoadedSampleGraph:
		graph_dir = self.resolve_sample_dir(sample_id)
		LOGGER.info("Loading unified graph for %s from %s", sample_id, graph_dir)
		graph = SemanticGraph.load_graph(str(graph_dir))
		retriever = graph.get_multi_retriever()
		SmartSearchProfiler.instrument(retriever)
		build_stats = retriever.build_all_indexes(methods_to_build=DEFAULT_METHODS, force_rebuild=False)
		LOGGER.info("Retriever index preparation for %s: %s", sample_id, self._compact_build_stats(build_stats))
		return LoadedSampleGraph(sample_id=sample_id, graph_dir=graph_dir, graph=graph, retriever=retriever)

	def resolve_sample_dir(self, sample_id: str) -> Path:
		candidates = [
			self.unified_dir / sample_id,
			self.unified_dir / f"{sample_id}_unified",
			self.unified_dir / f"{sample_id}_semantic_graph",
			self.unified_dir / f"{sample_id}_graph",
		]
		for candidate in candidates:
			if self._looks_like_graph_dir(candidate):
				return candidate
		raise FileNotFoundError(f"No unified SemanticGraph directory found for {sample_id} under {self.unified_dir}")

	def _looks_like_graph_dir(self, path: Path) -> bool:
		if not path.is_dir():
			return False
		return any((path / marker).exists() for marker in self.GRAPH_MARKERS)

	def _compact_build_stats(self, build_stats: Any) -> Dict[str, Any]:
		if not isinstance(build_stats, dict):
			return {"raw": str(build_stats)}
		compact: Dict[str, Any] = {}
		for key, value in build_stats.items():
			if isinstance(value, dict):
				compact[str(key)] = {
					"success": value.get("success"),
					"loaded_from_disk": value.get("loaded_from_disk"),
					"built": value.get("built"),
					"error": value.get("error"),
				}
			else:
				compact[str(key)] = value
		return compact


class NoDriftScheduler:
	"""Schedule coroutines at fixed QPS without accumulating drift."""

	def __init__(self, qps: float):
		if qps <= 0:
			raise ValueError("qps must be positive")
		self.qps = float(qps)

	async def run(self, queries: Sequence[QueryItem], runner: "SmartSearchRunner") -> List[RequestRecord]:
		base_time = time.perf_counter()
		tasks: List[asyncio.Task[RequestRecord]] = []

		for index, query in enumerate(queries):
			expected_start = base_time + index / self.qps
			now = time.perf_counter()
			if expected_start > now:
				await asyncio.sleep(expected_start - now)
			actual_start = time.perf_counter()
			schedule_drift_ms = (actual_start - expected_start) * 1000.0
			scheduled_offset_s = index / self.qps
			tasks.append(
				asyncio.create_task(
					runner.run_one(
						query=query,
						index=index,
						scheduled_offset_s=scheduled_offset_s,
						actual_start=actual_start,
						schedule_drift_ms=schedule_drift_ms,
					)
				)
			)

		if not tasks:
			return []
		return await asyncio.gather(*tasks)


class SmartSearchRunner:
	"""Run MultiRetriever.smart_search using sync or true async execution."""

	def __init__(
		self,
		retriever: Any,
		sample_id: str,
		total_queries: int,
		top_k: int,
		rerank_method: Optional[str],
		search_mode: str,
	):
		self.retriever = retriever
		self.sample_id = sample_id
		self.total_queries = total_queries
		self.top_k = top_k
		self.rerank_method = rerank_method
		self.search_mode = search_mode
		self.use_async_search = self._resolve_use_async_search(search_mode)
		self._completed = 0
		self._progress_lock = asyncio.Lock()
		self._progress_step = max(1, min(20, total_queries // 10 or 1))
		LOGGER.info(
			"[%s] search execution mode: requested=%s effective=%s reranker_backend=%s rerank_method=%s",
			sample_id,
			search_mode,
			"async" if self.use_async_search else "sync",
			settings.reranker_backend,
			rerank_method or "none",
		)

	def _resolve_use_async_search(self, search_mode: str) -> bool:
		normalized = (search_mode or "auto").strip().lower()
		async_available = callable(getattr(self.retriever, "smart_search_async", None))
		if normalized == "sync":
			return False
		if normalized == "async":
			if not async_available:
				raise RuntimeError("--search-mode async requested, but retriever has no smart_search_async")
			return True
		if normalized != "auto":
			raise ValueError(f"Unsupported search mode: {search_mode}")
		return bool(async_available and self.rerank_method and settings.reranker_backend == "vllm")

	async def run_one(
		self,
		query: QueryItem,
		index: int,
		scheduled_offset_s: float,
		actual_start: float,
		schedule_drift_ms: float,
	) -> RequestRecord:
		timing = PhaseTiming()
		token = _CURRENT_TIMING.set(timing)
		started_at = datetime.now(timezone.utc).isoformat()
		error: Optional[str] = None
		success = False
		result_count = 0

		try:
			if self.use_async_search:
				response = await self._call_smart_search_async(query.question)
			else:
				response = await asyncio.to_thread(self._call_smart_search, query.question)
			success, error, result_count = self._parse_response(response)
		except Exception as exc:
			error = repr(exc)
			success = False
		finally:
			_CURRENT_TIMING.reset(token)

		completed_at = datetime.now(timezone.utc).isoformat()
		latency_ms = (time.perf_counter() - actual_start) * 1000.0
		record = RequestRecord(
			sample_id=query.sample_id,
			question_id=query.question_id,
			question=query.question,
			index=index,
			category=query.category,
			success=success,
			error=error,
			result_count=result_count,
			latency_ms=latency_ms,
			retrieval_time_ms=timing.retrieval_time_ms,
			rerank_time_ms=timing.rerank_time_ms,
			schedule_drift_ms=schedule_drift_ms,
			scheduled_offset_s=scheduled_offset_s,
			started_at=started_at,
			completed_at=completed_at,
		)
		await self._log_progress(record)
		return record

	def _call_smart_search(self, question: str) -> Any:
		return self.retriever.smart_search(
			query=question,
			methods=DEFAULT_METHODS,
			top_k=self.top_k,
			rerank_method=self.rerank_method,
			return_detailed=True,
		)

	async def _call_smart_search_async(self, question: str) -> Any:
		return await self.retriever.smart_search_async(
			query=question,
			methods=DEFAULT_METHODS,
			top_k=self.top_k,
			rerank_method=self.rerank_method,
			return_detailed=True,
		)

	def _parse_response(self, response: Any) -> Tuple[bool, Optional[str], int]:
		if isinstance(response, dict):
			if response.get("error"):
				return False, str(response["error"]), len(response.get("results") or [])
			statistics = response.get("statistics") or {}
			result_count = statistics.get("final_results_count")
			if result_count is None:
				result_count = len(response.get("results") or [])
			return True, None, int(result_count)

		if isinstance(response, list):
			return True, None, len(response)

		return False, f"Unexpected response type: {type(response).__name__}", 0

	async def _log_progress(self, record: RequestRecord) -> None:
		async with self._progress_lock:
			self._completed += 1
			if self._completed == 1 or self._completed == self.total_queries or self._completed % self._progress_step == 0:
				LOGGER.info(
					"[%s] progress %d/%d, latest latency=%.2f ms, retrieval=%.2f ms, rerank=%.2f ms, drift=%.2f ms, success=%s",
					self.sample_id,
					self._completed,
					self.total_queries,
					record.latency_ms,
					record.retrieval_time_ms,
					record.rerank_time_ms,
					record.schedule_drift_ms,
					record.success,
				)


class MetricsAggregator:
	@staticmethod
	def summarize(records: Sequence[RequestRecord]) -> RunSummary:
		total = len(records)
		successful = [record for record in records if record.success]
		failed = total - len(successful)
		duration_seconds = MetricsAggregator._duration_seconds(records)
		actual_throughput_qps = total / duration_seconds if duration_seconds > 0 else 0.0

		latencies = np.array([record.latency_ms for record in successful], dtype=np.float64)
		retrieval = np.array([record.retrieval_time_ms for record in successful], dtype=np.float64)
		rerank = np.array([record.rerank_time_ms for record in successful], dtype=np.float64)
		drift = np.array([record.schedule_drift_ms for record in records], dtype=np.float64)

		return RunSummary(
			total_requests=total,
			successful_requests=len(successful),
			failed_requests=failed,
			success_rate=(len(successful) / total) if total else 0.0,
			actual_throughput_qps=actual_throughput_qps,
			duration_seconds=duration_seconds,
			latency_mean_ms=MetricsAggregator._mean(latencies),
			latency_p50_ms=MetricsAggregator._percentile(latencies, 50),
			latency_p90_ms=MetricsAggregator._percentile(latencies, 90),
			latency_p95_ms=MetricsAggregator._percentile(latencies, 95),
			latency_p99_ms=MetricsAggregator._percentile(latencies, 99),
			retrieval_mean_ms=MetricsAggregator._mean(retrieval),
			rerank_mean_ms=MetricsAggregator._mean(rerank),
			drift_mean_ms=MetricsAggregator._mean(drift),
			drift_p95_ms=MetricsAggregator._percentile(drift, 95),
		)

	@staticmethod
	def _duration_seconds(records: Sequence[RequestRecord]) -> float:
		if not records:
			return 0.0
		starts = [datetime.fromisoformat(record.started_at).timestamp() for record in records]
		ends = [datetime.fromisoformat(record.completed_at).timestamp() for record in records]
		return max(ends) - min(starts)

	@staticmethod
	def _mean(values: np.ndarray) -> Optional[float]:
		if values.size == 0:
			return None
		return float(np.mean(values))

	@staticmethod
	def _percentile(values: np.ndarray, percentile: float) -> Optional[float]:
		if values.size == 0:
			return None
		return float(np.percentile(values, percentile))

	@staticmethod
	def log_summary(title: str, summary: RunSummary) -> None:
		LOGGER.info(
			"%s | total=%d success=%d failed=%d success_rate=%.2f%% throughput=%.3f qps duration=%.2fs",
			title,
			summary.total_requests,
			summary.successful_requests,
			summary.failed_requests,
			summary.success_rate * 100.0,
			summary.actual_throughput_qps,
			summary.duration_seconds,
		)
		LOGGER.info(
			"%s latency(ms) | mean=%s p50=%s p90=%s p95=%s p99=%s retrieval_mean=%s rerank_mean=%s drift_mean=%s drift_p95=%s",
			title,
			MetricsAggregator._fmt(summary.latency_mean_ms),
			MetricsAggregator._fmt(summary.latency_p50_ms),
			MetricsAggregator._fmt(summary.latency_p90_ms),
			MetricsAggregator._fmt(summary.latency_p95_ms),
			MetricsAggregator._fmt(summary.latency_p99_ms),
			MetricsAggregator._fmt(summary.retrieval_mean_ms),
			MetricsAggregator._fmt(summary.rerank_mean_ms),
			MetricsAggregator._fmt(summary.drift_mean_ms),
			MetricsAggregator._fmt(summary.drift_p95_ms),
		)

	@staticmethod
	def _fmt(value: Optional[float]) -> str:
		return "n/a" if value is None else f"{value:.2f}"


def request_record_to_json(record: RequestRecord) -> Dict[str, Any]:
	data = asdict(record)
	data["retrieval_time"] = record.retrieval_time_ms
	data["rerank_time"] = record.rerank_time_ms
	return data


class ResourceCleaner:
	@staticmethod
	def cleanup_sample(loaded: Optional[LoadedSampleGraph], clear_retrieval_resources: bool = True) -> None:
		if loaded is None:
			return
		LOGGER.info("Cleaning resources for %s", loaded.sample_id)
		if clear_retrieval_resources:
			cleanup_retrieval_resources(loaded.retriever)
		else:
			ResourceCleaner._light_cleanup(loaded)
		loaded.retriever = None
		loaded.graph = None
		gc.collect()

	@staticmethod
	def _light_cleanup(loaded: LoadedSampleGraph) -> None:
		try:
			if getattr(loaded.retriever, "cleanup_unused_retrievers", None):
				loaded.retriever.cleanup_unused_retrievers(keep_methods=[])
		except Exception as exc:
			LOGGER.warning("Light retriever cleanup failed for %s: %s", loaded.sample_id, exc)

	@staticmethod
	def synchronize_cuda() -> None:
		try:
			import torch

			if torch.cuda.is_available():
				torch.cuda.synchronize()
		except Exception:
			return


class SmartSearchQPSBenchmark:
	def __init__(self, config: BenchmarkConfig):
		self.config = config
		self.query_loader = LoCoMoQueryLoader(config.data_dir)
		self.graph_loader = UnifiedGraphLoader(config.unified_dir)

	async def run(self) -> Dict[str, Any]:
		started_at = datetime.now(timezone.utc).isoformat()
		hardware = detect_hardware()
		log_hardware_hint(hardware, self.config.qps)

		queries_by_sample = self.query_loader.load()
		sample_ids = self._resolve_sample_ids(queries_by_sample)

		warmup_report = await self._run_warmup(queries_by_sample)

		all_records: List[RequestRecord] = []
		sample_reports: Dict[str, Any] = {}
		for position, sample_id in enumerate(sample_ids, start=1):
			LOGGER.info("Starting timed sample %s (%d/%d)", sample_id, position, len(sample_ids))
			sample_queries = queries_by_sample.get(sample_id, [])
			if not sample_queries:
				LOGGER.warning("Skipping %s because it has no queries", sample_id)
				sample_reports[sample_id] = {"skipped": True, "reason": "no_queries"}
				continue

			loaded: Optional[LoadedSampleGraph] = None
			try:
				loaded = self.graph_loader.load(sample_id)
				sample_warmup_report = await self._run_sample_warmup(loaded, sample_queries)
				records = await self._run_sample(loaded, sample_queries)
				summary = MetricsAggregator.summarize(records)
				MetricsAggregator.log_summary(f"Sample {sample_id}", summary)
				sample_reports[sample_id] = {
					"skipped": False,
					"sample_warmup": sample_warmup_report,
					"summary": asdict(summary),
					"records": [request_record_to_json(record) for record in records],
				}
				all_records.extend(records)
			finally:
				ResourceCleaner.cleanup_sample(loaded, clear_retrieval_resources=True)

		aggregate_summary = MetricsAggregator.summarize(all_records)
		MetricsAggregator.log_summary("Aggregate", aggregate_summary)

		report = {
			"started_at": started_at,
			"finished_at": datetime.now(timezone.utc).isoformat(),
			"config": self._config_to_json(),
			"hardware": hardware,
			"warmup": warmup_report,
			"samples": sample_reports,
			"aggregate_summary": asdict(aggregate_summary),
		}

		if self.config.output_json:
			self._write_json_report(report, self.config.output_json)
		return report

	async def _run_warmup(self, queries_by_sample: Dict[str, List[QueryItem]]) -> Dict[str, Any]:
		sample_id = self.config.warmup_sample_id
		warmup_queries = queries_by_sample.get(sample_id, [])
		if not warmup_queries:
			LOGGER.warning("Warmup sample %s has no queries; skipping warmup", sample_id)
			return {"sample_id": sample_id, "skipped": True, "reason": "no_queries"}

		loaded: Optional[LoadedSampleGraph] = None
		start = time.perf_counter()
		LOGGER.info("Starting warmup on %s with %d queries", sample_id, len(warmup_queries))
		try:
			loaded = self.graph_loader.load(sample_id)
			runner = SmartSearchRunner(
				retriever=loaded.retriever,
				sample_id=sample_id,
				total_queries=len(warmup_queries),
				top_k=self.config.top_k,
				rerank_method=self.config.rerank_method,
				search_mode=self.config.search_mode,
			)
			records: List[RequestRecord] = []
			for index, query in enumerate(warmup_queries):
				actual_start = time.perf_counter()
				records.append(
					await runner.run_one(
						query=query,
						index=index,
						scheduled_offset_s=0.0,
						actual_start=actual_start,
						schedule_drift_ms=0.0,
					)
				)
			summary = MetricsAggregator.summarize(records)
			MetricsAggregator.log_summary(f"Warmup {sample_id}", summary)
			return {
				"sample_id": sample_id,
				"skipped": False,
				"duration_seconds": time.perf_counter() - start,
				"summary": asdict(summary),
			}
		finally:
			ResourceCleaner.cleanup_sample(loaded, clear_retrieval_resources=False)

	async def _run_sample_warmup(self, loaded: LoadedSampleGraph, queries: Sequence[QueryItem]) -> Dict[str, Any]:
		warmup_items = self._select_sample_warmup_queries(queries)
		if not warmup_items:
			return {"skipped": True, "reason": "disabled", "requested_requests": self.config.sample_warmup_requests}

		selected_indices = [query_index for query_index, _ in warmup_items]
		LOGGER.info(
			"Starting sample-level warmup for %s with %d representative request(s): %s",
			loaded.sample_id,
			len(warmup_items),
			selected_indices,
		)
		runner = SmartSearchRunner(
			retriever=loaded.retriever,
			sample_id=f"{loaded.sample_id}:sample_warmup",
			total_queries=len(warmup_items),
			top_k=self.config.top_k,
			rerank_method=self.config.rerank_method,
			search_mode=self.config.search_mode,
		)
		records: List[RequestRecord] = []
		start = time.perf_counter()
		for query_index, query in warmup_items:
			actual_start = time.perf_counter()
			records.append(
				await runner.run_one(
					query=query,
					index=query_index,
					scheduled_offset_s=0.0,
					actual_start=actual_start,
					schedule_drift_ms=0.0,
				)
			)

		ResourceCleaner.synchronize_cuda()
		summary = MetricsAggregator.summarize(records)
		MetricsAggregator.log_summary(f"Sample warmup {loaded.sample_id}", summary)
		return {
			"skipped": False,
			"requested_requests": self.config.sample_warmup_requests,
			"actual_requests": len(records),
			"selection_strategy": "head_quantile_longest",
			"selected_query_indices": selected_indices,
			"duration_seconds": time.perf_counter() - start,
			"summary": asdict(summary),
			"records": [request_record_to_json(record) for record in records],
		}

	def _select_sample_warmup_queries(self, queries: Sequence[QueryItem]) -> List[Tuple[int, QueryItem]]:
		target_count = min(max(0, self.config.sample_warmup_requests), len(queries))
		if target_count <= 0:
			return []

		selected_indices: List[int] = []
		seen: set[int] = set()

		def add_index(index: int) -> None:
			if len(selected_indices) >= target_count:
				return
			if index < 0 or index >= len(queries) or index in seen:
				return
			seen.add(index)
			selected_indices.append(index)

		head_count = min(3, target_count)
		long_query_reserve = 2 if target_count >= 8 else 1 if target_count >= 5 else 0
		quantile_limit = max(head_count, target_count - long_query_reserve)

		for index in range(head_count):
			add_index(index)

		for fraction in (0.1, 0.25, 0.35, 0.4, 0.5, 0.65, 0.8, 0.9, 1.0):
			if len(selected_indices) >= quantile_limit:
				break
			add_index(round((len(queries) - 1) * fraction))

		longest_indices = sorted(range(len(queries)), key=lambda index: len(queries[index].question), reverse=True)
		for index in longest_indices:
			add_index(index)

		for index in range(len(queries)):
			add_index(index)

		return [(index, queries[index]) for index in selected_indices]

	async def _run_sample(self, loaded: LoadedSampleGraph, queries: Sequence[QueryItem]) -> List[RequestRecord]:
		scheduler = NoDriftScheduler(self.config.qps)
		runner = SmartSearchRunner(
			retriever=loaded.retriever,
			sample_id=loaded.sample_id,
			total_queries=len(queries),
			top_k=self.config.top_k,
			rerank_method=self.config.rerank_method,
			search_mode=self.config.search_mode,
		)
		return await scheduler.run(queries, runner)

	def _resolve_sample_ids(self, queries_by_sample: Dict[str, List[QueryItem]]) -> List[str]:
		requested = self.config.sample_ids
		if any(sample_id.lower() == "all" for sample_id in requested):
			requested = self.graph_loader.discover_samples()
			if not requested:
				requested = sorted(queries_by_sample)

		sample_ids: List[str] = []
		seen: Dict[str, None] = {}
		for sample_id in requested:
			if sample_id == self.config.warmup_sample_id:
				LOGGER.warning("Skipping %s from timed samples because it is reserved for warmup", sample_id)
				continue
			if sample_id not in seen:
				seen[sample_id] = None
				sample_ids.append(sample_id)

		if not sample_ids:
			raise ValueError("No timed sample ids remain after filtering warmup sample")
		LOGGER.info("Timed samples: %s", ", ".join(sample_ids))
		return sample_ids

	def _config_to_json(self) -> Dict[str, Any]:
		data = asdict(self.config)
		data["unified_dir"] = str(self.config.unified_dir)
		data["data_dir"] = str(self.config.data_dir)
		data["output_json"] = str(self.config.output_json) if self.config.output_json else None
		return data

	def _write_json_report(self, report: Dict[str, Any], output_path: Path) -> None:
		output_path.parent.mkdir(parents=True, exist_ok=True)
		with output_path.open("w", encoding="utf-8") as handle:
			json.dump(report, handle, ensure_ascii=False, indent=2)
		LOGGER.info("Saved benchmark report to %s", output_path)


def detect_hardware() -> Dict[str, Any]:
	info: Dict[str, Any] = {"cuda_available": False, "devices": [], "suggested_qps": None}
	try:
		import torch

		info["cuda_available"] = bool(torch.cuda.is_available())
		if not info["cuda_available"]:
			return info
		device_count = torch.cuda.device_count()
		devices = [torch.cuda.get_device_name(index) for index in range(device_count)]
		info["devices"] = devices
		joined = " ".join(devices).lower()
		if "h800" in joined:
			info["suggested_qps"] = 10.0
		elif "5090" in joined:
			info["suggested_qps"] = 5.0
		return info
	except Exception as exc:
		info["error"] = str(exc)
		return info


def log_hardware_hint(hardware: Dict[str, Any], qps: float) -> None:
	if not hardware.get("cuda_available"):
		LOGGER.warning("CUDA is not available; fixed-QPS GPU benchmark may be unrepresentative")
		return
	LOGGER.info("CUDA devices: %s", ", ".join(hardware.get("devices") or []))
	suggested = hardware.get("suggested_qps")
	if suggested is not None and abs(float(suggested) - qps) > 1e-6:
		LOGGER.info("Hardware heuristic suggests %.1f QPS; current target is %.1f QPS", suggested, qps)


def parse_sample_ids(values: Sequence[str]) -> List[str]:
	sample_ids: List[str] = []
	for value in values:
		sample_ids.extend(token.strip() for token in value.split(",") if token.strip())
	if not sample_ids:
		raise argparse.ArgumentTypeError("--sample-ids cannot be empty")
	return sample_ids


def default_output_json_path() -> Path:
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	return DEFAULT_OUTPUT_DIR / f"qps_benchmark_report_{timestamp}.json"


def configure_threaded_torch_compile() -> None:
	os.environ.setdefault("TORCHINDUCTOR_CUDAGRAPHS", "0")
	try:
		import torch
	except Exception:
		return

	if getattr(torch, "_smart_search_qps_compile_patch", False):
		return

	original_compile = torch.compile

	def compile_without_cudagraphs(*args: Any, **kwargs: Any) -> Any:
		if kwargs.get("mode") == "reduce-overhead":
			options = dict(kwargs.get("options") or {})
			options.setdefault("triton.cudagraphs", False)
			kwargs["options"] = options
		return original_compile(*args, **kwargs)

	torch.compile = compile_without_cudagraphs
	setattr(torch, "_smart_search_qps_compile_patch", True)


def parse_args(argv: Optional[Sequence[str]] = None) -> BenchmarkConfig:
	parser = argparse.ArgumentParser(
		description="Fixed-QPS benchmark for MultiRetriever.smart_search on unified LoCoMo SemanticGraphs.",
	)
	parser.add_argument("--unified-dir", type=Path, default=DEFAULT_UNIFIED_DIR, help="Root directory containing unified graph folders by sample id.")
	parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_PATH, help="LoCoMo dataset or extracted QA directory used to load queries.")
	parser.add_argument("--qps", type=float, required=True, help="Target QPS. Suggested: RTX 5090 Laptop=5, H800=10.")
	parser.add_argument("--sample-ids", nargs="+", default=["all"], help="Timed sample ids, e.g. conv-30 conv-41. Default: all discovered samples except conv-26. Use 'all' explicitly for the same behavior.")
	parser.add_argument("--top-k", type=int, default=35, help="Number of retrieval results requested from smart_search.")
	parser.add_argument("--rerank-method", default="baai", help="Optional reranker name, e.g. qwen or baai. Omit to disable reranking.")
	parser.add_argument("--search-mode", default="auto", choices=["auto", "sync", "async"], help="Execution path for smart_search. auto uses true async when RERANKER_BACKEND=vllm and reranking is enabled; sync preserves the legacy to_thread path.")
	parser.add_argument("--sample-warmup-requests", type=int, default=12, help="Representative warmup requests to run after each timed sample graph is loaded and before fixed-QPS scheduling. Use 0 to disable.")
	parser.add_argument("--output-json", type=Path, default=None, help="Optional path to save the full benchmark report JSON. Default: timestamped file under benchmark_locomo/task_eval/results/smart_search_qps_results/.")
	parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging verbosity.")

	args = parser.parse_args(argv)
	if args.top_k <= 0:
		parser.error("--top-k must be positive")
	if args.qps <= 0:
		parser.error("--qps must be positive")
	if args.sample_warmup_requests < 0:
		parser.error("--sample-warmup-requests must be non-negative")

	rerank_method = args.rerank_method.strip() if isinstance(args.rerank_method, str) else None
	if not rerank_method:
		rerank_method = None

	return BenchmarkConfig(
		unified_dir=args.unified_dir,
		data_dir=args.data_dir,
		qps=args.qps,
		sample_ids=parse_sample_ids(args.sample_ids),
		top_k=args.top_k,
		rerank_method=rerank_method,
		search_mode=args.search_mode,
		sample_warmup_requests=args.sample_warmup_requests,
		output_json=args.output_json or default_output_json_path(),
		log_level=args.log_level,
	)


def configure_logging(level: str) -> None:
	logging.basicConfig(
		level=getattr(logging, level),
		format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
	)


def main(argv: Optional[Sequence[str]] = None) -> int:
	config = parse_args(argv)
	configure_logging(config.log_level)
	configure_threaded_torch_compile()
	benchmark = SmartSearchQPSBenchmark(config)
	asyncio.run(benchmark.run())
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
