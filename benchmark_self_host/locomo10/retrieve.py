#!/usr/bin/env python3
"""Retrieve from self-host LoCoMo10 per-sample SemanticGraphs.

The script consumes graphs produced by ``benchmark_self_host/locomo10/build_graph.py``
and mirrors the retrieval side of ``benchmark_locomo/task_eval/locomo_triple.py``:

1. First-stage direct retrieval with ``MultiRetriever.smart_search``.
2. Memory-space filtering per tower:
   - hierarchical: L0/L1/L2 textual hierarchy
   - graph: entity/relation textual structures
   - episodic: episodic facts with time injection
3. Optional second-stage reranking with the same ``tower_separate`` and
   ``unified_rerank`` strategies used by the tri-tower benchmark.
4. JSONL results saved per sample for downstream answer generation/evaluation.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _bootstrap_paths() -> Path:
	current = Path(__file__).resolve()
	for parent in [current.parent, *current.parents]:
		src = parent / "src"
		if (src / "mandol").exists():
			if str(src) not in sys.path:
				sys.path.insert(0, str(src))
			if str(parent) not in sys.path:
				sys.path.insert(0, str(parent))
			return parent
	raise RuntimeError("Could not locate repository root containing src/mandol")


REPO_ROOT = _bootstrap_paths()
SCRIPT_DIR = Path(__file__).resolve().parent

from mandol.core.memory_unit import MemoryUnit  # noqa: E402
from mandol.core.memory_space_registry import TowerSpace  # noqa: E402
from mandol.core.semantic_graph import SemanticGraph  # noqa: E402
from mandol.retrieval.advance_retriever import MultiRetriever  # noqa: E402
from mandol.retrieval.rerank_manager import RerankerManager  # noqa: E402
from mandol.retrieval.retrieval_interface import RetrievalMethod  # noqa: E402
from mandol.utils.logging_config import auto_configure_logging, setup_logging  # noqa: E402


LOGGER = logging.getLogger("locomo10_retrieve")

DEFAULT_METHODS = ["bm25", "cosine_similarity", "splade"]
DEFAULT_RERANKER_CONFIGS = {
	"baai": "BAAI/bge-reranker-v2-m3",
	"qwen": "Qwen/Qwen3-Reranker-0.6B",
	"jina": "jinaai/jina-reranker-v2-base-multilingual",
	"qwen-sili": "Qwen/Qwen3-Reranker-8B",
	"qwen-dashscope": "qwen3-rerank",
	"gte-dashscope": "gte-rerank-v2",
}

SPACE_PRESETS = {
	"l0": [TowerSpace.HIERARCHICAL_L0.value],
	"l1": [TowerSpace.HIERARCHICAL_L1.value],
	"l2": [TowerSpace.HIERARCHICAL_L2.value],
	"hierarchical": [TowerSpace.HIERARCHICAL_ROOT.value],
	"episodic": [TowerSpace.EPISODIC_ROOT.value],
	"entity": [TowerSpace.GRAPH_ENTITIES.value],
	"entity_mentions": [TowerSpace.GRAPH_MENTIONS.value],
	"entity_relations": [TowerSpace.GRAPH_RELATIONS.value],
	"graph": [TowerSpace.GRAPH_ROOT.value],
}


@dataclass
class RetrievalCase:
	sample_id: str
	question_id: str
	question: str
	category: int = 0
	expected_answer: str = ""
	evidence: List[str] = field(default_factory=list)
	source: str = "qa_dataset"


@dataclass
class LoadedSample:
	sample_id: str
	graph_dir: Path
	graph: SemanticGraph
	retriever: MultiRetriever
	build_stats: Dict[str, Any]


def configure_logging(debug: bool = False) -> None:
	level = logging.DEBUG if debug else logging.INFO
	if auto_configure_logging() is None:
		setup_logging(level=level)
	logging.getLogger().setLevel(level)
	LOGGER.setLevel(level)


def make_jsonable(value: Any) -> Any:
	if hasattr(value, "to_dict"):
		return make_jsonable(value.to_dict())
	if hasattr(value, "__dataclass_fields__"):
		return make_jsonable(asdict(value))
	if isinstance(value, Path):
		return str(value)
	if isinstance(value, dict):
		return {str(key): make_jsonable(item) for key, item in value.items()}
	if isinstance(value, (list, tuple, set)):
		return [make_jsonable(item) for item in value]
	if hasattr(value, "item") and callable(value.item):
		try:
			return value.item()
		except Exception:
			pass
	return value


def write_json(path: Path, data: Any) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as file:
		json.dump(make_jsonable(data), file, ensure_ascii=False, indent=2)


def append_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as file:
		for record in records:
			file.write(json.dumps(make_jsonable(record), ensure_ascii=False) + "\n")


def normalize_methods(methods: Sequence[str]) -> List[RetrievalMethod]:
	normalized = []
	for method in methods:
		normalized.append(RetrievalMethod(method.lower()))
	return normalized


def split_spaces(raw_spaces: Optional[Sequence[str]], preset_name: str) -> List[str]:
	if raw_spaces:
		spaces: List[str] = []
		for item in raw_spaces:
			spaces.extend(part.strip() for part in item.split(",") if part.strip())
		return spaces
	return list(SPACE_PRESETS[preset_name])


def load_cases(
	dataset_path: Path,
	sample_ids: Optional[Sequence[str]],
	query: Optional[str],
	max_questions: Optional[int],
) -> List[RetrievalCase]:
	selected = set(sample_ids or [])

	if query:
		if not selected:
			raise ValueError("--query requires --sample-ids so the target graph is unambiguous")
		return [
			RetrievalCase(
				sample_id=sample_id,
				question_id="adhoc_001",
				question=query,
				source="adhoc_query",
			)
			for sample_id in sample_ids or []
		]

	with dataset_path.open("r", encoding="utf-8") as file:
		data = json.load(file)
	if not isinstance(data, list):
		raise ValueError(f"Dataset must be a list: {dataset_path}")

	cases: List[RetrievalCase] = []
	for sample in data:
		sample_id = str(sample.get("sample_id") or sample.get("conversation_id") or "")
		if not sample_id:
			continue
		if selected and sample_id not in selected:
			continue

		qa_list = sample.get("qa") or []
		for index, qa_item in enumerate(qa_list, 1):
			if not isinstance(qa_item, dict) or not qa_item.get("question"):
				continue
			category = qa_item.get("category", 0)
			expected_answer = qa_item.get("answer") or ""
			if not expected_answer and qa_item.get("adversarial_answer"):
				expected_answer = qa_item.get("adversarial_answer") or ""
				category = 5
			cases.append(
				RetrievalCase(
					sample_id=sample_id,
					question_id=f"qa_{index:03d}",
					question=str(qa_item["question"]),
					category=int(category or 0),
					expected_answer=str(expected_answer or ""),
					evidence=qa_item.get("evidence", []) or [],
				)
			)

	if max_questions is not None:
		per_sample_counts: Dict[str, int] = {}
		limited: List[RetrievalCase] = []
		for case in cases:
			count = per_sample_counts.get(case.sample_id, 0)
			if count >= max_questions:
				continue
			limited.append(case)
			per_sample_counts[case.sample_id] = count + 1
		cases = limited

	return cases


def extract_unit_text(unit: MemoryUnit) -> str:
	raw = unit.raw_data if getattr(unit, "raw_data", None) else {}
	metadata = unit.metadata if getattr(unit, "metadata", None) else {}
	for key in ("text_content", "enhanced_content", "content", "message", "description", "original_content"):
		value = raw.get(key)
		if value:
			return str(value)
	for key in ("text_content", "content", "summary", "description"):
		value = metadata.get(key)
		if value:
			return str(value)
	return str(raw) if raw else ""


def infer_layer(unit: MemoryUnit) -> str:
	metadata = unit.metadata or {}
	raw = unit.raw_data or {}
	layer = metadata.get("layer") or metadata.get("memory_level")
	if layer:
		layer_text = str(layer)
		if layer_text.startswith("L0"):
			return "L0"
		if layer_text.startswith("L1"):
			return "L1"
		if layer_text.startswith("L2"):
			return "L2"
		return layer_text

	uid = unit.uid.lower()
	if uid.startswith("l0_") or "_l0_" in uid:
		return "L0"
	if "_l1_" in uid:
		return "L1"
	if "_l2_" in uid:
		return "L2"
	if raw.get("node_type") in {"entity", "entity_mention", "entity_relation"}:
		return "graph"
	if raw.get("fact_type") or metadata.get("fact_id"):
		return "episodic"
	return "unknown"


def infer_graph_subtype(unit: MemoryUnit) -> str:
	raw = unit.raw_data or {}
	metadata = unit.metadata or {}
	node_type = raw.get("node_type") or metadata.get("type")
	if node_type == "entity":
		return "entity"
	if node_type == "entity_mention":
		return "entity_mention"
	if node_type in {"entity_relation", "relation"}:
		return "relation"
	return str(node_type or "graph")


def extract_time_info(unit: MemoryUnit) -> str:
	metadata = unit.metadata or {}
	raw = unit.raw_data or {}
	for key in ("time_start", "time_original", "session_date", "date", "created_at"):
		value = metadata.get(key)
		if value:
			return str(value)
	for key in ("temporal_tag", "timestamp", "session_date", "date"):
		value = raw.get(key)
		if value:
			return str(value)
	time_data = raw.get("time")
	if isinstance(time_data, dict):
		for key in ("absolute_date", "range_start", "original_text"):
			value = time_data.get(key)
			if value:
				return str(value)
	return "N/A"


def unit_to_record(unit: MemoryUnit, score: float, rank: int, source: str) -> Dict[str, Any]:
	return {
		"rank": rank,
		"uid": unit.uid,
		"score": float(score),
		"source": source,
		"layer": infer_layer(unit),
		"graph_subtype": infer_graph_subtype(unit) if source == "graph" else None,
		"time": extract_time_info(unit),
		"content": extract_unit_text(unit),
		"raw_data": unit.raw_data,
		"metadata": unit.metadata,
	}


def serialize_results(results: Sequence[Tuple[MemoryUnit, float]], source: str) -> List[Dict[str, Any]]:
	return [unit_to_record(unit, score, index + 1, source) for index, (unit, score) in enumerate(results)]


def group_hierarchical(results: Sequence[Tuple[MemoryUnit, float]]) -> Dict[str, List[Dict[str, Any]]]:
	grouped = {"L0": [], "L1": [], "L2": [], "unknown": []}
	for index, (unit, score) in enumerate(results, 1):
		record = unit_to_record(unit, score, index, "hierarchical")
		grouped.setdefault(record["layer"], []).append(record)
	return grouped


def group_graph(results: Sequence[Tuple[MemoryUnit, float]]) -> Dict[str, List[Dict[str, Any]]]:
	grouped = {"entity": [], "entity_mention": [], "relation": [], "graph": []}
	for index, (unit, score) in enumerate(results, 1):
		record = unit_to_record(unit, score, index, "graph")
		grouped.setdefault(record["graph_subtype"] or "graph", []).append(record)
	return grouped


def build_episodic_context_with_time(results: Sequence[Tuple[MemoryUnit, float]]) -> str:
	parts = []
	for index, (unit, score) in enumerate(results, 1):
		content = extract_unit_text(unit)
		if not content:
			continue
		parts.append(f"Episodic Fact {index}: [Time: {extract_time_info(unit)}] {content}")
	return "\n\n".join(parts)


def extract_l0_units(hierarchical_results: Sequence[Tuple[MemoryUnit, float]]) -> List[Tuple[MemoryUnit, float]]:
	return [(unit, score) for unit, score in hierarchical_results if infer_layer(unit) == "L0"]


class LocomoSelfHostRetriever:
	def __init__(self, args: argparse.Namespace):
		self.args = args
		self.memory_dir = Path(args.memory_dir).resolve()
		self.output_dir = Path(args.output_dir).resolve()
		self.output_dir.mkdir(parents=True, exist_ok=True)
		self.methods = normalize_methods(args.methods)
		self.reranker_configs = dict(DEFAULT_RERANKER_CONFIGS)
		if args.reranker_model:
			self.reranker_configs[args.reranker_type] = args.reranker_model
		self.reranker_manager = RerankerManager()
		self.loaded_samples: Dict[str, LoadedSample] = {}

		self.l0_spaces = split_spaces(args.l0_spaces, "l0")
		self.hierarchical_spaces = split_spaces(args.hierarchical_spaces, "hierarchical")
		self.graph_spaces = split_spaces(args.graph_spaces, "graph")
		self.episodic_spaces = split_spaces(args.episodic_spaces, "episodic")
		self.topk_l0 = args.topk_l0 or args.topk_hierarchical

	def load_sample(self, sample_id: str) -> LoadedSample:
		if sample_id in self.loaded_samples:
			return self.loaded_samples[sample_id]

		graph_dir = self.memory_dir / sample_id
		if not graph_dir.exists():
			raise FileNotFoundError(f"Sample graph does not exist: {graph_dir}")
		if not ((graph_dir / "rx_graph.pkl").exists() or (graph_dir / "graph_state.json").exists()):
			raise FileNotFoundError(f"Sample graph is incomplete: {graph_dir}")

		LOGGER.info("Loading graph %s", graph_dir)
		graph = SemanticGraph.load_graph(str(graph_dir))
		retriever = MultiRetriever(
			retrieval_source=graph,
			preload_rerankers=False,
			reranker_configs=self.reranker_configs,
			reranker_manager=self.reranker_manager,
		)
		build_stats = retriever.build_all_indexes(methods_to_build=self.methods, force_rebuild=False)
		loaded = LoadedSample(
			sample_id=sample_id,
			graph_dir=graph_dir,
			graph=graph,
			retriever=retriever,
			build_stats=build_stats,
		)
		self.loaded_samples[sample_id] = loaded
		return loaded

	def run_cases(self, cases: Sequence[RetrievalCase]) -> Dict[str, Any]:
		started_at = datetime.now().isoformat()
		by_sample: Dict[str, List[RetrievalCase]] = {}
		for case in cases:
			by_sample.setdefault(case.sample_id, []).append(case)

		all_results = []
		summary = {
			"started_at": started_at,
			"memory_dir": str(self.memory_dir),
			"output_dir": str(self.output_dir),
			"config": self._config_summary(),
			"samples": {},
		}

		for sample_id, sample_cases in by_sample.items():
			sample_start = time.perf_counter()
			sample_output = self.output_dir / sample_id
			sample_records: List[Dict[str, Any]] = []
			sample_errors: List[str] = []
			try:
				loaded = self.load_sample(sample_id)
				for index, case in enumerate(sample_cases, 1):
					LOGGER.info("[%s %d/%d] %s", sample_id, index, len(sample_cases), case.question)
					record = self.run_case(loaded, case)
					sample_records.append(record)
					all_results.append(record)
			except Exception as exc:
				message = f"{type(exc).__name__}: {exc}"
				sample_errors.append(message)
				LOGGER.error("Sample %s retrieval failed: %s", sample_id, message)
				LOGGER.debug(traceback.format_exc())

			append_jsonl(sample_output / "retrieval_results.jsonl", sample_records)
			write_json(sample_output / "retrieval_results.json", sample_records)
			sample_summary = self._sample_summary(sample_id, sample_records, sample_errors)
			sample_summary["duration_seconds"] = time.perf_counter() - sample_start
			write_json(sample_output / "retrieval_summary.json", sample_summary)
			summary["samples"][sample_id] = sample_summary

		summary["finished_at"] = datetime.now().isoformat()
		summary["total_cases"] = len(cases)
		summary["successful_cases"] = sum(1 for item in all_results if item.get("success"))
		summary["failed_cases"] = summary["total_cases"] - summary["successful_cases"]
		write_json(self.output_dir / "retrieval_summary.json", summary)
		return summary

	def run_case(self, loaded: LoadedSample, case: RetrievalCase) -> Dict[str, Any]:
		start = time.perf_counter()
		record: Dict[str, Any] = {
			"success": False,
			"sample_id": case.sample_id,
			"question_id": case.question_id,
			"question": case.question,
			"category": case.category,
			"expected_answer": case.expected_answer,
			"evidence": case.evidence,
			"source": case.source,
			"graph_dir": str(loaded.graph_dir),
			"first_stage": {},
			"second_stage": {},
			"final": {},
			"errors": [],
		}

		try:
			l0_results, l0_details = self._smart_search(
				loaded.retriever,
				case.question,
				top_k=self.topk_l0,
				space_names=self.l0_spaces,
				source="hierarchical_l0",
			)
			hierarchical_results, hierarchical_details = self._smart_search(
				loaded.retriever,
				case.question,
				top_k=self.args.topk_hierarchical,
				space_names=self.hierarchical_spaces,
				source="hierarchical",
			)
			graph_results, graph_details = self._smart_search(
				loaded.retriever,
				case.question,
				top_k=self.args.topk_similarity,
				space_names=self.graph_spaces,
				source="graph",
			)
			episodic_results, episodic_details = self._smart_search(
				loaded.retriever,
				case.question,
				top_k=self.args.topk_episodic,
				space_names=self.episodic_spaces,
				source="episodic",
			)

			if not l0_results:
				l0_results = extract_l0_units(hierarchical_results)
			episodic_context = build_episodic_context_with_time(episodic_results)

			record["first_stage"] = {
				"l0_direct": {
					**l0_details,
					"by_layer": group_hierarchical(l0_results),
					"results": serialize_results(l0_results, "hierarchical_l0"),
				},
				"hierarchical": {
					**hierarchical_details,
					"by_layer": group_hierarchical(hierarchical_results),
					"results": serialize_results(hierarchical_results, "hierarchical"),
				},
				"graph": {
					**graph_details,
					"by_subtype": group_graph(graph_results),
					"results": serialize_results(graph_results, "graph"),
				},
				"episodic": {
					**episodic_details,
					"context_with_time": episodic_context,
					"results": serialize_results(episodic_results, "episodic"),
				},
				"l0_for_second_stage_count": len(l0_results),
			}

			final_l0 = l0_results
			final_graph = graph_results
			final_episodic = episodic_results

			if self.args.retrieval_mode in {"second_stage", "both"} and self.args.enable_second_stage_rerank:
				second_stage = self._perform_second_stage_rerank(
					question=case.question,
					l0_units=l0_results,
					graph_units=graph_results,
					episodic_units=episodic_results,
				)
				final_l0 = second_stage["reranked_l0"]
				final_graph = second_stage["reranked_graph"]
				final_episodic = second_stage["reranked_episodic"]
				record["second_stage"] = {
					**{key: value for key, value in second_stage.items() if not key.startswith("reranked_")},
					"reranked_l0": serialize_results(final_l0, "hierarchical_l0"),
					"reranked_graph": serialize_results(final_graph, "graph"),
					"reranked_episodic": serialize_results(final_episodic, "episodic"),
				}
			else:
				record["second_stage"] = {
					"enabled": False,
					"method": "none",
					"strategy_used": self.args.rerank_strategy,
					"rerank_time": 0.0,
					"first_stage_counts": {
						"l0": len(l0_results),
						"graph": len(graph_results),
						"episodic": len(episodic_results),
						"total": len(l0_results) + len(graph_results) + len(episodic_results),
					},
					"final_counts": {
						"l0": len(final_l0),
						"graph": len(final_graph),
						"episodic": len(final_episodic),
						"total": len(final_l0) + len(final_graph) + len(final_episodic),
					},
				}

			record["final"] = {
				"l0": serialize_results(final_l0, "hierarchical_l0"),
				"graph": serialize_results(final_graph, "graph"),
				"episodic": serialize_results(final_episodic, "episodic"),
				"episodic_context_with_time": build_episodic_context_with_time(final_episodic),
				"counts": {
					"l0": len(final_l0),
					"graph": len(final_graph),
					"episodic": len(final_episodic),
					"total": len(final_l0) + len(final_graph) + len(final_episodic),
				},
			}
			record["success"] = True

		except Exception as exc:
			message = f"{type(exc).__name__}: {exc}"
			record["errors"].append(message)
			LOGGER.error("Retrieval case failed %s/%s: %s", case.sample_id, case.question_id, message)
			LOGGER.debug(traceback.format_exc())

		record["total_retrieval_time"] = time.perf_counter() - start
		return record

	def _smart_search(
		self,
		retriever: MultiRetriever,
		question: str,
		top_k: int,
		space_names: List[str],
		source: str,
	) -> Tuple[List[Tuple[MemoryUnit, float]], Dict[str, Any]]:
		start = time.perf_counter()
		rerank_method = None if self.args.no_first_stage_rerank else self.args.reranker_type
		results = retriever.smart_search(
			query=question,
			methods=self.methods,
			fusion_method=self.args.fusion_method,
			rerank_method=rerank_method,
			top_k=top_k,
			space_names=space_names,
			return_detailed=False,
		)
		duration = time.perf_counter() - start
		return list(results), {
			"method": f"{source}_smart_search",
			"spaces": list(space_names),
			"methods": [method.value for method in self.methods],
			"fusion_method": self.args.fusion_method,
			"rerank_method": rerank_method or "none",
			"top_k": top_k,
			"retrieved_count": len(results),
			"retrieval_time": duration,
		}

	def _perform_second_stage_rerank(
		self,
		question: str,
		l0_units: List[Tuple[MemoryUnit, float]],
		graph_units: List[Tuple[MemoryUnit, float]],
		episodic_units: List[Tuple[MemoryUnit, float]],
	) -> Dict[str, Any]:
		result = {
			"enabled": True,
			"method": self.args.second_stage_rerank_method,
			"strategy_used": self.args.rerank_strategy,
			"rerank_time": 0.0,
			"first_stage_counts": {
				"l0": len(l0_units),
				"graph": len(graph_units),
				"episodic": len(episodic_units),
				"total": len(l0_units) + len(graph_units) + len(episodic_units),
			},
			"final_counts": {
				"l0": len(l0_units),
				"graph": len(graph_units),
				"episodic": len(episodic_units),
				"total": len(l0_units) + len(graph_units) + len(episodic_units),
			},
			"reranked_l0": l0_units,
			"reranked_graph": graph_units,
			"reranked_episodic": episodic_units,
		}

		if self.args.rerank_strategy == "unified_rerank":
			candidates = [(unit, score, "l0") for unit, score in l0_units]
		else:
			candidates = []
		candidates.extend((unit, score, "graph") for unit, score in graph_units)
		candidates.extend((unit, score, "episodic") for unit, score in episodic_units)

		if not candidates:
			return result

		start = time.perf_counter()
		try:
			reranker = self.reranker_manager.get_reranker(
				self.args.second_stage_rerank_method,
				self.reranker_configs.get(self.args.second_stage_rerank_method),
			)
			if reranker is None:
				LOGGER.warning("Could not get reranker %s; skip second-stage rerank", self.args.second_stage_rerank_method)
				return result

			documents = [extract_unit_text(unit) for unit, _, _ in candidates]
			scores = reranker.rerank(query=question, documents=documents)
			reranked = list(zip(candidates, scores))
			reranked.sort(key=lambda item: item[1], reverse=True)

			if self.args.threshold > 0.0:
				reranked = [(item, score) for item, score in reranked if score >= self.args.threshold]

			selected = [
				(unit, float(new_score), source)
				for (unit, _old_score, source), new_score in reranked[: self.args.final_top_k]
			]

			if self.args.rerank_strategy == "unified_rerank":
				final_l0 = [(unit, score) for unit, score, source in selected if source == "l0"]
			else:
				final_l0 = l0_units
			final_graph = [(unit, score) for unit, score, source in selected if source == "graph"]
			final_episodic = [(unit, score) for unit, score, source in selected if source == "episodic"]

			result.update(
				{
					"rerank_time": time.perf_counter() - start,
					"reranked_l0": final_l0,
					"reranked_graph": final_graph,
					"reranked_episodic": final_episodic,
					"final_counts": {
						"l0": len(final_l0),
						"graph": len(final_graph),
						"episodic": len(final_episodic),
						"total": len(final_l0) + len(final_graph) + len(final_episodic),
					},
				}
			)
			return result

		except Exception as exc:
			LOGGER.error("Second-stage rerank failed: %s", exc)
			LOGGER.debug(traceback.format_exc())
			result["rerank_time"] = time.perf_counter() - start
			result["error"] = str(exc)
			return result

	def _config_summary(self) -> Dict[str, Any]:
		return {
			"methods": [method.value for method in self.methods],
			"fusion_method": self.args.fusion_method,
			"reranker_type": self.args.reranker_type,
			"first_stage_rerank": not self.args.no_first_stage_rerank,
			"retrieval_mode": self.args.retrieval_mode,
			"second_stage_rerank": self.args.enable_second_stage_rerank,
			"second_stage_rerank_method": self.args.second_stage_rerank_method,
			"rerank_strategy": self.args.rerank_strategy,
			"final_top_k": self.args.final_top_k,
			"threshold": self.args.threshold,
			"topk_l0": self.topk_l0,
			"topk_hierarchical": self.args.topk_hierarchical,
			"topk_similarity": self.args.topk_similarity,
			"topk_episodic": self.args.topk_episodic,
			"l0_spaces": self.l0_spaces,
			"hierarchical_spaces": self.hierarchical_spaces,
			"graph_spaces": self.graph_spaces,
			"episodic_spaces": self.episodic_spaces,
		}

	@staticmethod
	def _sample_summary(sample_id: str, records: List[Dict[str, Any]], errors: List[str]) -> Dict[str, Any]:
		successful = [record for record in records if record.get("success")]
		return {
			"sample_id": sample_id,
			"total_cases": len(records),
			"successful_cases": len(successful),
			"failed_cases": len(records) - len(successful),
			"errors": errors,
			"avg_total_retrieval_time": (
				sum(record.get("total_retrieval_time", 0.0) for record in successful) / len(successful)
				if successful
				else 0.0
			),
			"total_first_stage_counts": {
				"l0": sum(record.get("first_stage", {}).get("l0_direct", {}).get("retrieved_count", 0) for record in successful),
				"hierarchical": sum(record.get("first_stage", {}).get("hierarchical", {}).get("retrieved_count", 0) for record in successful),
				"graph": sum(record.get("first_stage", {}).get("graph", {}).get("retrieved_count", 0) for record in successful),
				"episodic": sum(record.get("first_stage", {}).get("episodic", {}).get("retrieved_count", 0) for record in successful),
			},
			"total_final_counts": {
				"l0": sum(record.get("final", {}).get("counts", {}).get("l0", 0) for record in successful),
				"graph": sum(record.get("final", {}).get("counts", {}).get("graph", 0) for record in successful),
				"episodic": sum(record.get("final", {}).get("counts", {}).get("episodic", 0) for record in successful),
			},
		}


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Retrieve from self-host LoCoMo10 SemanticGraphs")
	parser.add_argument("--memory-dir", default=str(SCRIPT_DIR / "memory"), help="Directory containing per-sample SemanticGraph folders")
	parser.add_argument("--qa-dataset", default=str(SCRIPT_DIR / "dataset" / "locomo10.json"), help="LoCoMo QA dataset")
	parser.add_argument("--output-dir", default=str(SCRIPT_DIR / "retrieve"), help="Directory for retrieval outputs")
	parser.add_argument("--sample-ids", nargs="+", help="Sample IDs to retrieve, e.g. conv-30")
	parser.add_argument("--query", help="Ad-hoc query; requires --sample-ids")
	parser.add_argument("--max-questions", type=int, help="Maximum QA questions per sample")

	parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, choices=["bm25", "cosine_similarity", "splade"])
	parser.add_argument("--fusion-method", default="rrf", choices=["rrf", "weighted", "average"])
	parser.add_argument("--reranker-type", default="baai", choices=list(DEFAULT_RERANKER_CONFIGS.keys()))
	parser.add_argument("--reranker-model", help="Override model name for --reranker-type")
	parser.add_argument("--no-first-stage-rerank", action="store_true", help="Disable first-stage smart_search reranking")

	parser.add_argument("--topk-hierarchical", type=int, default=15, help="Hierarchical tower top-k")
	parser.add_argument("--topk-l0", type=int, default=None, help="L0 direct top-k; defaults to --topk-hierarchical")
	parser.add_argument("--topk-similarity", type=int, default=30, help="Graph/entity-relation tower semantic top-k")
	parser.add_argument("--topk-episodic", type=int, default=30, help="Episodic tower top-k")
	parser.add_argument("--l0-spaces", nargs="+", help="Override L0 direct memory spaces")
	parser.add_argument("--hierarchical-spaces", nargs="+", help="Override hierarchical memory spaces")
	parser.add_argument("--graph-spaces", nargs="+", help="Override graph/entity memory spaces")
	parser.add_argument("--episodic-spaces", nargs="+", help="Override episodic memory spaces")

	parser.add_argument("--retrieval-mode", choices=["direct", "second_stage", "both"], default="both")
	parser.add_argument("--enable-second-stage-rerank", action="store_true", default=True)
	parser.add_argument("--no-second-stage-rerank", action="store_true")
	parser.add_argument("--second-stage-rerank-method", default=None)
	parser.add_argument("--final-top-k", type=int, default=20)
	parser.add_argument("--threshold", type=float, default=0.0)
	parser.add_argument("--rerank-strategy", choices=["tower_separate", "unified_rerank"], default="tower_separate")
	parser.add_argument("--debug", action="store_true")
	return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
	parser = build_arg_parser()
	args = parser.parse_args(argv)
	configure_logging(args.debug)

	if args.no_second_stage_rerank:
		args.enable_second_stage_rerank = False
	if args.retrieval_mode == "direct":
		args.enable_second_stage_rerank = False
	args.second_stage_rerank_method = args.second_stage_rerank_method or args.reranker_type

	cases = load_cases(
		dataset_path=Path(args.qa_dataset).resolve(),
		sample_ids=args.sample_ids,
		query=args.query,
		max_questions=args.max_questions,
	)
	if not cases:
		LOGGER.error("No retrieval cases found")
		return 1

	LOGGER.info("Loaded %d retrieval cases", len(cases))
	runner = LocomoSelfHostRetriever(args)
	summary = runner.run_cases(cases)
	failed = summary.get("failed_cases", 0)
	if failed:
		LOGGER.error("Retrieval finished with %d failed cases", failed)
		return 1
	LOGGER.info("Retrieval finished successfully: %d cases", summary.get("successful_cases", 0))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
