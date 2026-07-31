#!/usr/bin/env python3
"""Retrieve from self-host LongMemEval per-QA SemanticGraphs.

The graphs produced by ``benchmark_self_host/longmemeval/build_graph.py`` contain
the three LongMemEval memory towers in a single SemanticGraph.  This script
mirrors the retrieval half of ``benchmark_longmemeval/task_eval/benchmark_triple.py``
by using memory-space filters to emulate the original separate graphs:

1. Sentence-level memory: ``hierarchical_memory:L0_Observation``.
2. Episodic memory: ``episodic_memory``.
3. Entity relation memory: ``entity_relation``.
4. Optional second-stage reranking over all towers with the original sentence
   bucket safeguard.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
import traceback
from collections import defaultdict
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


LOGGER = logging.getLogger("longmemeval_retrieve")

DEFAULT_METHODS = ["bm25", "splade", "cosine_similarity"]
RERANK_METHOD_CHOICES = ["baai", "qwen", "jina", "qwen-sili", "qwen-dashscope", "gte-dashscope", "none"]
DEFAULT_RERANKER_CONFIGS = {
	"baai": "BAAI/bge-reranker-v2-m3",
	"qwen": "Qwen/Qwen3-Reranker-0.6B",
	"jina": "jinaai/jina-reranker-v2-base-multilingual",
	"qwen-sili": "Qwen/Qwen3-Reranker-8B",
	"qwen-dashscope": "qwen3-rerank",
	"gte-dashscope": "gte-rerank-v2",
}

SPACE_PRESETS = {
	"sentence": [TowerSpace.HIERARCHICAL_L0.value],
	"episodic": [TowerSpace.EPISODIC_ROOT.value],
	"entity": [TowerSpace.GRAPH_ROOT.value],
}

STABLE_CATEGORIES = {
	"USER_ATTRIBUTE",
	"PREFERENCE_HABIT",
	"RELATIONSHIP_FACT",
	"KNOWLEDGE",
	"IMPLICIT_CONSTRAINT",
	"INVENTORY_ITEM",
}


@dataclass
class LongMemEvalCase:
	sample_id: str
	qa_index: int
	question_id: str
	question: str
	expected_answer: str = ""
	question_type: str = ""
	category: str = ""
	query_date: str = "Unknown Date"
	answer_session_ids: List[str] = field(default_factory=list)
	source: str = "longmemeval_dataset"


@dataclass
class LoadedSample:
	sample_id: str
	qa_index: int
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


def read_json(path: Path) -> Any:
	with path.open("r", encoding="utf-8") as file:
		return json.load(file)


def write_json(path: Path, data: Any) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as file:
		json.dump(make_jsonable(data), file, ensure_ascii=False, indent=2)


def append_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as file:
		for record in records:
			file.write(json.dumps(make_jsonable(record), ensure_ascii=False) + "\n")


def sample_id_from_index(index: int) -> str:
	return f"qa_{index}"


def qa_index_from_sample_id(sample_id: str) -> Optional[int]:
	match = re.match(r"^qa_(\d+)$", str(sample_id))
	return int(match.group(1)) if match else None


def normalize_methods(methods: Sequence[str]) -> List[RetrievalMethod]:
	return [RetrievalMethod(method.lower()) for method in methods]


def split_spaces(raw_spaces: Optional[Sequence[str]], preset_name: str) -> List[str]:
	if raw_spaces:
		spaces: List[str] = []
		for item in raw_spaces:
			spaces.extend(part.strip() for part in item.split(",") if part.strip())
		return spaces
	return list(SPACE_PRESETS[preset_name])


def select_indices(
	dataset_size: int,
	sample_ids: Optional[Sequence[str]],
	start_index: Optional[int],
	end_index: Optional[int],
	limit: Optional[int],
) -> List[int]:
	if sample_ids:
		indices: List[int] = []
		for sample_id in sample_ids:
			idx = qa_index_from_sample_id(sample_id)
			if idx is None:
				raise ValueError(f"LongMemEval sample IDs must look like qa_0, got: {sample_id}")
			if 0 <= idx < dataset_size:
				indices.append(idx)
			else:
				raise IndexError(f"Sample index out of range: {sample_id} for dataset size {dataset_size}")
		return indices[:limit] if limit is not None else indices

	start = 0 if start_index is None else max(0, int(start_index))
	end = dataset_size - 1 if end_index is None else min(dataset_size - 1, int(end_index))
	if start > end:
		return []
	indices = list(range(start, end + 1))
	return indices[:limit] if limit is not None else indices


def load_cases(
	dataset_path: Path,
	sample_ids: Optional[Sequence[str]],
	query: Optional[str],
	start_index: Optional[int],
	end_index: Optional[int],
	limit: Optional[int],
) -> List[LongMemEvalCase]:
	if query:
		if not sample_ids:
			raise ValueError("--query requires --sample-ids so the target graph is unambiguous")
		cases = []
		for sample_id in sample_ids:
			qa_index = qa_index_from_sample_id(sample_id)
			if qa_index is None:
				raise ValueError(f"LongMemEval sample IDs must look like qa_0, got: {sample_id}")
			cases.append(
				LongMemEvalCase(
					sample_id=sample_id,
					qa_index=qa_index,
					question_id="adhoc_001",
					question=query,
					source="adhoc_query",
				)
			)
		return cases[:limit] if limit is not None else cases

	data = read_json(dataset_path)
	if not isinstance(data, list):
		raise ValueError(f"Dataset must be a list: {dataset_path}")

	cases: List[LongMemEvalCase] = []
	for idx in select_indices(len(data), sample_ids, start_index, end_index, limit):
		item = data[idx]
		if not isinstance(item, dict) or not item.get("question"):
			continue
		cases.append(
			LongMemEvalCase(
				sample_id=sample_id_from_index(idx),
				qa_index=idx,
				question_id=str(item.get("question_id") or f"q_{idx}"),
				question=str(item["question"]),
				expected_answer=str(item.get("answer") or ""),
				question_type=str(item.get("question_type") or ""),
				category=str(item.get("category") or ""),
				query_date=str(item.get("question_date") or item.get("date") or item.get("session_date") or "Unknown Date"),
				answer_session_ids=[str(value) for value in (item.get("answer_session_ids") or [])],
			)
		)
	return cases


def extract_unit_text(unit: MemoryUnit) -> str:
	raw = unit.raw_data if getattr(unit, "raw_data", None) else {}
	metadata = unit.metadata if getattr(unit, "metadata", None) else {}
	for key in ("text_content", "content", "description", "message", "original_content", "enhanced_content"):
		value = raw.get(key)
		if value:
			return str(value)
	for key in ("text_content", "content", "summary", "description"):
		value = metadata.get(key)
		if value:
			return str(value)
	return str(raw) if raw else ""


def extract_time_info(unit: MemoryUnit) -> str:
	raw = unit.raw_data if getattr(unit, "raw_data", None) else {}
	metadata = unit.metadata if getattr(unit, "metadata", None) else {}
	for key in ("session_date", "event_date", "temporal_val", "time", "temporal_info", "temporal_reference", "created_at"):
		value = raw.get(key)
		if value:
			return str(value)
	for key in ("session_date", "event_date", "time_start", "time_original", "date", "created_at"):
		value = metadata.get(key)
		if value:
			return str(value)
	return "Unknown Date"


def unit_to_record(unit: MemoryUnit, score: float, rank: int, source_type: str) -> Dict[str, Any]:
	raw = unit.raw_data if getattr(unit, "raw_data", None) else {}
	metadata = unit.metadata if getattr(unit, "metadata", None) else {}
	return {
		"rank": rank,
		"uid": unit.uid,
		"score": float(score),
		"source_type": source_type,
		"content": extract_unit_text(unit),
		"text_content": str(raw.get("text_content") or extract_unit_text(unit)),
		"time": extract_time_info(unit),
		"role": metadata.get("role") or raw.get("role"),
		"session_date": metadata.get("session_date") or raw.get("session_date"),
		"session_id": metadata.get("session_id") or raw.get("session_id"),
		"category": raw.get("category") or raw.get("fact_type") or raw.get("node_type"),
		"entity_name": raw.get("entity_canonical") or raw.get("entity_text"),
		"entity_type": raw.get("entity_category") or raw.get("entity_type"),
		"raw_data": raw,
		"metadata": metadata,
	}


def serialize_results(results: Sequence[Tuple[MemoryUnit, float]], source_type: str) -> List[Dict[str, Any]]:
	return [unit_to_record(unit, score, index + 1, source_type) for index, (unit, score) in enumerate(results)]


def format_sentence_context(results: List[Tuple[MemoryUnit, float]]) -> str:
	if not results:
		return ""
	parts = []
	for index, (unit, _score) in enumerate(results, 1):
		raw = unit.raw_data if getattr(unit, "raw_data", None) else {}
		metadata = unit.metadata if getattr(unit, "metadata", None) else {}
		content = raw.get("text_content") or extract_unit_text(unit)
		role = metadata.get("role", raw.get("role", "unknown"))
		session_date = metadata.get("session_date", raw.get("session_date", "unknown"))
		parts.append(f"[Message {index}] (Date: {session_date}, Speaker: {role})\nContent: {content}")
	return "\n\n".join(parts)


def format_episodic_context(results: List[Tuple[MemoryUnit, float]]) -> str:
	if not results:
		return ""
	parts = []
	for index, (unit, _score) in enumerate(results, 1):
		raw = unit.raw_data if getattr(unit, "raw_data", None) else {}
		content = raw.get("content", raw.get("text_content", extract_unit_text(unit)))
		category = str(raw.get("category", raw.get("fact_type", raw.get("node_type", "EVENT")))).upper()
		event_date = raw.get("event_date") or raw.get("temporal_val") or raw.get("time") or raw.get("temporal_info") or "Unknown Date"
		is_stable = raw.get("is_stable")
		if is_stable is None:
			is_stable = category in STABLE_CATEGORIES
		stability = "Stable" if is_stable else "Dynamic"
		parts.append(f"[Fact {index}] Type: {category} | Time: {event_date} | Stability: {stability}\nContent: {content}")
	return "\n\n".join(parts)


def format_entity_context(results: List[Tuple[MemoryUnit, float]]) -> str:
	if not results:
		return ""
	parts = []
	for index, (unit, _score) in enumerate(results, 1):
		raw = unit.raw_data if getattr(unit, "raw_data", None) else {}
		main_content = raw.get("text_content")
		if not main_content:
			name = raw.get("entity_canonical") or raw.get("entity_text") or unit.uid
			entity_type = raw.get("entity_category") or raw.get("entity_type") or "Unknown"
			description = raw.get("content") or raw.get("description") or "No description"
			main_content = f"Entity: {name} (Type: {entity_type}) | Context: {description}"
		session_date = raw.get("session_date") or raw.get("date") or raw.get("created_at")
		date_text = f" [Date: {session_date}]" if session_date else ""
		parts.append(f"[{index}] {main_content}{date_text}")
	return "\n\n".join(parts)


def build_fused_context(sentence_context: str, episodic_context: str, entity_context: str) -> str:
	sections = []
	if sentence_context:
		sections.append(f"<conversation_history>\n{sentence_context}\n</conversation_history>")
	if episodic_context:
		sections.append(f"<episodic_facts>\n{episodic_context}\n</episodic_facts>")
	if entity_context:
		sections.append(f"<entity_knowledge>\n{entity_context}\n</entity_knowledge>")
	return "\n\n".join(sections) if sections else "[No context available]"


def facts_by_category(results: Sequence[Tuple[MemoryUnit, float]]) -> Dict[str, int]:
	counts: Dict[str, int] = defaultdict(int)
	for unit, _score in results:
		raw = unit.raw_data if getattr(unit, "raw_data", None) else {}
		category = str(raw.get("category", raw.get("fact_type", raw.get("node_type", "UNKNOWN")))).upper()
		counts[category] += 1
	return dict(counts)


def entity_types_found(results: Sequence[Tuple[MemoryUnit, float]]) -> Dict[str, int]:
	counts: Dict[str, int] = defaultdict(int)
	for unit, _score in results:
		raw = unit.raw_data if getattr(unit, "raw_data", None) else {}
		entity_type = raw.get("entity_category") or raw.get("entity_type") or "Unknown"
		counts[str(entity_type)] += 1
	return dict(counts)


class LongMemEvalSelfHostRetriever:
	def __init__(self, args: argparse.Namespace):
		self.args = args
		self.memory_dir = Path(args.memory_dir).resolve()
		self.output_dir = Path(args.output_dir).resolve()
		self.output_dir.mkdir(parents=True, exist_ok=True)
		self.methods = normalize_methods(args.methods)
		self.reranker_configs = dict(DEFAULT_RERANKER_CONFIGS)
		if args.reranker_model and args.rerank_method != "none":
			self.reranker_configs[args.rerank_method] = args.reranker_model
		self.reranker_manager = RerankerManager()
		self.loaded_samples: Dict[str, LoadedSample] = {}

		first_stage_top_k = args.first_stage_top_k
		self.sentence_top_k = first_stage_top_k or args.sentence_top_k
		self.episodic_top_k = first_stage_top_k or args.episodic_top_k
		self.entity_top_k = first_stage_top_k or args.entity_top_k

		self.sentence_spaces = split_spaces(args.sentence_spaces, "sentence")
		self.episodic_spaces = split_spaces(args.episodic_spaces, "episodic")
		self.entity_spaces = split_spaces(args.entity_spaces, "entity")

	def load_sample(self, sample_id: str, qa_index: int) -> LoadedSample:
		if sample_id in self.loaded_samples:
			return self.loaded_samples[sample_id]

		graph_dir = self.memory_dir / sample_id
		if not graph_dir.exists():
			raise FileNotFoundError(f"Sample graph does not exist: {graph_dir}")
		if not (graph_dir / "graph_state.json").exists():
			raise FileNotFoundError(f"Sample graph is incomplete: {graph_dir}")

		LOGGER.info("Loading graph %s", graph_dir)
		graph = SemanticGraph.load_graph(str(graph_dir), embedding_model_name=self.args.embedding_model)
		retriever = MultiRetriever(
			retrieval_source=graph,
			preload_rerankers=False,
			reranker_configs=self.reranker_configs,
			reranker_manager=self.reranker_manager,
		)
		build_stats = retriever.build_all_indexes(methods_to_build=self.methods, force_rebuild=False)
		loaded = LoadedSample(
			sample_id=sample_id,
			qa_index=qa_index,
			graph_dir=graph_dir,
			graph=graph,
			retriever=retriever,
			build_stats=build_stats,
		)
		self.loaded_samples[sample_id] = loaded
		return loaded

	def run_cases(self, cases: Sequence[LongMemEvalCase]) -> Dict[str, Any]:
		started_at = datetime.now().isoformat()
		all_results: List[Dict[str, Any]] = []
		summary: Dict[str, Any] = {
			"started_at": started_at,
			"memory_dir": str(self.memory_dir),
			"output_dir": str(self.output_dir),
			"config": self._config_summary(),
			"samples": {},
		}

		for ordinal, case in enumerate(cases, 1):
			sample_start = time.perf_counter()
			sample_output = self.output_dir / case.sample_id
			sample_records: List[Dict[str, Any]] = []
			sample_errors: List[str] = []
			try:
				loaded = self.load_sample(case.sample_id, case.qa_index)
				LOGGER.info("[%d/%d] Retrieving %s: %s", ordinal, len(cases), case.sample_id, case.question)
				record = self.run_case(loaded, case)
				sample_records.append(record)
				all_results.append(record)
			except Exception as exc:
				message = f"{type(exc).__name__}: {exc}"
				sample_errors.append(message)
				LOGGER.error("Sample %s retrieval failed: %s", case.sample_id, message)
				LOGGER.debug(traceback.format_exc())

			append_jsonl(sample_output / "retrieval_results.jsonl", sample_records)
			write_json(sample_output / "retrieval_results.json", sample_records)
			sample_summary = self._sample_summary(case.sample_id, sample_records, sample_errors)
			sample_summary["duration_seconds"] = time.perf_counter() - sample_start
			write_json(sample_output / "retrieval_summary.json", sample_summary)
			summary["samples"][case.sample_id] = sample_summary
			write_json(self.output_dir / "retrieval_summary.json", summary)

		summary["finished_at"] = datetime.now().isoformat()
		summary["total_cases"] = len(cases)
		summary["successful_cases"] = sum(1 for item in all_results if item.get("success"))
		summary["failed_cases"] = summary["total_cases"] - summary["successful_cases"]
		write_json(self.output_dir / "retrieval_summary.json", summary)
		return summary

	def run_case(self, loaded: LoadedSample, case: LongMemEvalCase) -> Dict[str, Any]:
		start = time.perf_counter()
		record: Dict[str, Any] = {
			"success": False,
			"sample_id": case.sample_id,
			"qa_index": case.qa_index,
			"question_id": case.question_id,
			"question": case.question,
			"expected_answer": case.expected_answer,
			"ground_truth": case.expected_answer,
			"question_type": case.question_type,
			"category": case.category,
			"query_date": case.query_date,
			"answer_session_ids": case.answer_session_ids,
			"source": case.source,
			"graph_dir": str(loaded.graph_dir),
			"first_stage": {},
			"second_stage": {},
			"final": {},
			"retrieval_details": {},
			"retrieved_contents": {"sentence": [], "episodic": [], "entity": [], "reranked": []},
			"errors": [],
		}

		try:
			sentence_results: List[Tuple[MemoryUnit, float]] = []
			episodic_results: List[Tuple[MemoryUnit, float]] = []
			entity_results: List[Tuple[MemoryUnit, float]] = []
			sentence_details: Dict[str, Any] = {"enabled": False}
			episodic_details: Dict[str, Any] = {"enabled": False}
			entity_details: Dict[str, Any] = {"enabled": False}

			if not self.args.disable_sentence:
				sentence_results, sentence_details = self._smart_search(
					loaded.retriever,
					case.question,
					top_k=self.sentence_top_k,
					space_names=self.sentence_spaces,
					source="sentence",
				)
			if not self.args.disable_episodic:
				episodic_results, episodic_details = self._smart_search(
					loaded.retriever,
					case.question,
					top_k=self.episodic_top_k,
					space_names=self.episodic_spaces,
					source="episodic",
				)
			if not self.args.disable_entity:
				entity_results, entity_details = self._smart_search(
					loaded.retriever,
					case.question,
					top_k=self.entity_top_k,
					space_names=self.entity_spaces,
					source="entity",
				)

			record["retrieved_contents"]["sentence"] = [
				self._extract_content_for_report(unit, score, "sentence") for unit, score in sentence_results
			]
			record["retrieved_contents"]["episodic"] = [
				self._extract_content_for_report(unit, score, "episodic") for unit, score in episodic_results
			]
			record["retrieved_contents"]["entity"] = [
				self._extract_content_for_report(unit, score, "entity") for unit, score in entity_results
			]

			episodic_details_with_stats = {**episodic_details, "facts_by_category": facts_by_category(episodic_results)}
			entity_details_with_stats = {**entity_details, "entity_types_found": entity_types_found(entity_results)}
			record["first_stage"] = {
				"sentence": {**sentence_details, "results": serialize_results(sentence_results, "sentence")},
				"episodic": {
					**episodic_details_with_stats,
					"results": serialize_results(episodic_results, "episodic"),
				},
				"entity": {
					**entity_details_with_stats,
					"results": serialize_results(entity_results, "entity"),
				},
			}

			final_sentence = sentence_results
			final_episodic = episodic_results
			final_entity = entity_results
			second_stage = self._direct_second_stage_summary(sentence_results, episodic_results, entity_results)

			if self.args.enable_second_stage_rerank:
				second_stage = self._perform_second_stage_rerank(
					question=case.question,
					sentence_units=sentence_results,
					episodic_units=episodic_results,
					entity_units=entity_results,
				)
				final_sentence = second_stage["reranked_sentence"]
				final_episodic = second_stage["reranked_episodic"]
				final_entity = second_stage["reranked_entity"]
				record["retrieved_contents"]["reranked"] = [
					self._extract_content_for_report(unit, score, source)
					for unit, score, source in second_stage.get("selected_with_sources", [])
				]

			sentence_context = format_sentence_context(final_sentence)
			episodic_context = format_episodic_context(final_episodic)
			entity_context = format_entity_context(final_entity)
			fused_context = build_fused_context(sentence_context, episodic_context, entity_context)

			record["second_stage"] = {
				key: value
				for key, value in second_stage.items()
				if key not in {"reranked_sentence", "reranked_episodic", "reranked_entity", "selected_with_sources"}
			}
			record["second_stage"].update(
				{
					"reranked_sentence": serialize_results(final_sentence, "sentence"),
					"reranked_episodic": serialize_results(final_episodic, "episodic"),
					"reranked_entity": serialize_results(final_entity, "entity"),
				}
			)
			record["final"] = {
				"sentence": serialize_results(final_sentence, "sentence"),
				"episodic": serialize_results(final_episodic, "episodic"),
				"entity": serialize_results(final_entity, "entity"),
				"sentence_context": sentence_context,
				"episodic_context": episodic_context,
				"entity_context": entity_context,
				"fused_context": fused_context,
				"counts": {
					"sentence": len(final_sentence),
					"episodic": len(final_episodic),
					"entity": len(final_entity),
					"total": len(final_sentence) + len(final_episodic) + len(final_entity),
				},
			}
			record["retrieval_details"] = self._retrieval_details(
				sentence_details,
				episodic_details_with_stats,
				entity_details_with_stats,
				second_stage,
				time.perf_counter() - start,
			)
			record["success"] = True

		except Exception as exc:
			message = f"{type(exc).__name__}: {exc}"
			record["errors"].append(message)
			LOGGER.error("Retrieval case failed %s: %s", case.sample_id, message)
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
		rerank_method = None if self.args.rerank_method == "none" else self.args.rerank_method
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
			"enabled": True,
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
		sentence_units: List[Tuple[MemoryUnit, float]],
		episodic_units: List[Tuple[MemoryUnit, float]],
		entity_units: List[Tuple[MemoryUnit, float]],
	) -> Dict[str, Any]:
		result = self._direct_second_stage_summary(sentence_units, episodic_units, entity_units)
		result.update({"enabled": True, "method": self.args.second_stage_rerank_method})

		if self.args.second_stage_rerank_method == "none":
			return result

		all_candidates = [(unit, score, "sentence") for unit, score in sentence_units]
		all_candidates.extend((unit, score, "episodic") for unit, score in episodic_units)
		all_candidates.extend((unit, score, "entity") for unit, score in entity_units)
		if not all_candidates:
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

			documents = [extract_unit_text(unit)[:1500] for unit, _score, _source in all_candidates]
			scores = reranker.rerank(query=question, documents=documents)
			reranked = list(zip(all_candidates, scores))
			reranked.sort(key=lambda item: item[1], reverse=True)
			if self.args.threshold > 0.0:
				reranked = [(item, score) for item, score in reranked if float(score) >= self.args.threshold]

			sentence_candidates: List[Tuple[MemoryUnit, float, str]] = []
			other_candidates: List[Tuple[MemoryUnit, float, str]] = []
			for (unit, _old_score, source), new_score in reranked:
				candidate = (unit, float(new_score), source)
				if source == "sentence":
					sentence_candidates.append(candidate)
				else:
					other_candidates.append(candidate)

			min_sentence = (self.args.final_top_k + 1) // 2
			selected = sentence_candidates[:min_sentence]
			remaining_slots = self.args.final_top_k - len(selected)
			if remaining_slots > 0:
				selected.extend(other_candidates[:remaining_slots])
			if len(selected) < self.args.final_top_k:
				slots_left = self.args.final_top_k - len(selected)
				selected.extend(sentence_candidates[min_sentence : min_sentence + slots_left])
			selected.sort(key=lambda item: item[1], reverse=True)

			final_sentence = [(unit, score) for unit, score, source in selected if source == "sentence"]
			final_episodic = [(unit, score) for unit, score, source in selected if source == "episodic"]
			final_entity = [(unit, score) for unit, score, source in selected if source == "entity"]

			result.update(
				{
					"rerank_time": time.perf_counter() - start,
					"min_sentence_keep": min_sentence,
					"final_selected_count": len(selected),
					"reranked_sentence": final_sentence,
					"reranked_episodic": final_episodic,
					"reranked_entity": final_entity,
					"selected_with_sources": selected,
					"final_counts": {
						"sentence": len(final_sentence),
						"episodic": len(final_episodic),
						"entity": len(final_entity),
						"total": len(final_sentence) + len(final_episodic) + len(final_entity),
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

	def _direct_second_stage_summary(
		self,
		sentence_units: List[Tuple[MemoryUnit, float]],
		episodic_units: List[Tuple[MemoryUnit, float]],
		entity_units: List[Tuple[MemoryUnit, float]],
	) -> Dict[str, Any]:
		return {
			"enabled": False,
			"method": "none",
			"rerank_time": 0.0,
			"first_stage_counts": {
				"sentence": len(sentence_units),
				"episodic": len(episodic_units),
				"entity": len(entity_units),
				"total": len(sentence_units) + len(episodic_units) + len(entity_units),
			},
			"final_counts": {
				"sentence": len(sentence_units),
				"episodic": len(episodic_units),
				"entity": len(entity_units),
				"total": len(sentence_units) + len(episodic_units) + len(entity_units),
			},
			"reranked_sentence": sentence_units,
			"reranked_episodic": episodic_units,
			"reranked_entity": entity_units,
			"selected_with_sources": [],
		}

	def _retrieval_details(
		self,
		sentence_details: Dict[str, Any],
		episodic_details: Dict[str, Any],
		entity_details: Dict[str, Any],
		second_stage: Dict[str, Any],
		total_time: float,
	) -> Dict[str, Any]:
		return {
			"total_retrieval_time": total_time,
			"graph_loading_time": 0.0,
			"sentence_enabled": not self.args.disable_sentence,
			"sentence_retrieved_count": int(sentence_details.get("retrieved_count", 0) or 0),
			"sentence_retrieval_time": float(sentence_details.get("retrieval_time", 0.0) or 0.0),
			"episodic_enabled": not self.args.disable_episodic,
			"episodic_retrieved_count": int(episodic_details.get("retrieved_count", 0) or 0),
			"episodic_retrieval_time": float(episodic_details.get("retrieval_time", 0.0) or 0.0),
			"episodic_facts_by_category": episodic_details.get("facts_by_category", {}),
			"entity_enabled": not self.args.disable_entity,
			"entity_retrieved_count": int(entity_details.get("retrieved_count", 0) or 0),
			"entity_retrieval_time": float(entity_details.get("retrieval_time", 0.0) or 0.0),
			"entity_types_found": entity_details.get("entity_types_found", {}),
			"second_stage_rerank_enabled": bool(second_stage.get("enabled")),
			"second_stage_rerank_method": second_stage.get("method", "none"),
			"second_stage_rerank_time": float(second_stage.get("rerank_time", 0.0) or 0.0),
			"first_stage_total_count": int(second_stage.get("first_stage_counts", {}).get("total", 0) or 0),
			"final_selected_count": int(second_stage.get("final_counts", {}).get("total", 0) or 0),
			"fusion_method": self.args.fusion_method,
			"rerank_method": self.args.rerank_method,
		}

	def _extract_content_for_report(self, unit: MemoryUnit, score: float, source_type: str) -> Dict[str, Any]:
		raw = unit.raw_data if getattr(unit, "raw_data", None) else {}
		metadata = unit.metadata if getattr(unit, "metadata", None) else {}
		content_info = {
			"uid": unit.uid,
			"score": round(float(score), 4),
			"source_type": source_type,
			"text_content": str(raw.get("text_content") or extract_unit_text(unit))[:500],
		}
		if source_type == "sentence":
			content_info.update({"role": metadata.get("role", raw.get("role", "unknown")), "session_date": metadata.get("session_date", raw.get("session_date", "unknown"))})
		elif source_type == "episodic":
			content_info.update({"category": raw.get("category", raw.get("fact_type", "UNKNOWN")), "event_date": raw.get("event_date") or raw.get("temporal_val") or raw.get("time", "unknown"), "is_stable": raw.get("is_stable", None)})
		elif source_type == "entity":
			content_info.update({"entity_name": raw.get("entity_canonical") or raw.get("entity_text", ""), "entity_type": raw.get("entity_category") or raw.get("entity_type", "Unknown"), "description": str(raw.get("content") or raw.get("description") or "")[:300], "session_date": raw.get("session_date") or raw.get("date", "")})
		return content_info

	def _config_summary(self) -> Dict[str, Any]:
		return {
			"methods": [method.value for method in self.methods],
			"fusion_method": self.args.fusion_method,
			"rerank_method": self.args.rerank_method,
			"second_stage_rerank": self.args.enable_second_stage_rerank,
			"second_stage_rerank_method": self.args.second_stage_rerank_method,
			"first_stage_top_k": self.args.first_stage_top_k,
			"sentence_top_k": self.sentence_top_k,
			"episodic_top_k": self.episodic_top_k,
			"entity_top_k": self.entity_top_k,
			"final_top_k": self.args.final_top_k,
			"threshold": self.args.threshold,
			"sentence_spaces": self.sentence_spaces,
			"episodic_spaces": self.episodic_spaces,
			"entity_spaces": self.entity_spaces,
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
				"sentence": sum(record.get("first_stage", {}).get("sentence", {}).get("retrieved_count", 0) for record in successful),
				"episodic": sum(record.get("first_stage", {}).get("episodic", {}).get("retrieved_count", 0) for record in successful),
				"entity": sum(record.get("first_stage", {}).get("entity", {}).get("retrieved_count", 0) for record in successful),
			},
			"total_final_counts": {
				"sentence": sum(record.get("final", {}).get("counts", {}).get("sentence", 0) for record in successful),
				"episodic": sum(record.get("final", {}).get("counts", {}).get("episodic", 0) for record in successful),
				"entity": sum(record.get("final", {}).get("counts", {}).get("entity", 0) for record in successful),
			},
		}


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Retrieve from self-host LongMemEval SemanticGraphs")
	parser.add_argument("--memory-dir", default=str(SCRIPT_DIR / "memory"), help="Directory containing per-qa SemanticGraph folders")
	parser.add_argument("--dataset-path", default=str(SCRIPT_DIR / "dataset" / "longmemeval_s_cleaned.json"), help="LongMemEval cleaned dataset")
	parser.add_argument("--output-dir", default=str(SCRIPT_DIR / "retrieve"), help="Directory for retrieval outputs")
	parser.add_argument("--sample-ids", nargs="+", help="Sample IDs to retrieve, e.g. qa_0")
	parser.add_argument("--query", help="Ad-hoc query; requires --sample-ids")
	parser.add_argument("--start-index", "--start-qa", dest="start_index", type=int, default=0, help="Start QA index, inclusive")
	parser.add_argument("--end-index", "--end-qa", dest="end_index", type=int, help="End QA index, inclusive")
	parser.add_argument("--limit", "--max-tests", dest="limit", type=int, help="Maximum number of QA samples")

	parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, choices=["bm25", "cosine_similarity", "splade"])
	parser.add_argument("--fusion-method", default="rrf", choices=["rrf", "weighted", "average"])
	parser.add_argument("--rerank-method", default="baai", choices=RERANK_METHOD_CHOICES, help="First-stage smart_search reranker")
	parser.add_argument("--reranker-model", help="Override model name for --rerank-method")

	parser.add_argument("--sentence-top-k", type=int, default=60, help="Sentence tower top-k")
	parser.add_argument("--episodic-top-k", type=int, default=40, help="Episodic tower top-k")
	parser.add_argument("--entity-top-k", type=int, default=40, help="Entity tower top-k")
	parser.add_argument("--first-stage-top-k", type=int, default=None, help="Override all first-stage tower top-k values")
	parser.add_argument("--sentence-spaces", nargs="+", help="Override sentence memory spaces")
	parser.add_argument("--episodic-spaces", nargs="+", help="Override episodic memory spaces")
	parser.add_argument("--entity-spaces", nargs="+", help="Override entity memory spaces")

	parser.add_argument("--disable-sentence", action="store_true", help="Disable sentence-level retrieval")
	parser.add_argument("--disable-episodic", action="store_true", help="Disable episodic retrieval")
	parser.add_argument("--disable-entity", action="store_true", help="Disable entity retrieval")
	parser.add_argument("--disable-second-stage-rerank", action="store_true", help="Disable second-stage reranking")
	parser.add_argument("--second-stage-rerank-method", default=None, choices=RERANK_METHOD_CHOICES, help="Second-stage reranker; defaults to --rerank-method")
	parser.add_argument("--final-top-k", type=int, default=25, help="Final count after second-stage reranking")
	parser.add_argument("--threshold", type=float, default=0.0, help="Optional second-stage score threshold")
	parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
	parser.add_argument("--debug", action="store_true")
	return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
	parser = build_arg_parser()
	args = parser.parse_args(argv)
	configure_logging(args.debug)

	args.enable_second_stage_rerank = not args.disable_second_stage_rerank
	args.second_stage_rerank_method = args.second_stage_rerank_method or args.rerank_method
	if args.second_stage_rerank_method == "none":
		args.enable_second_stage_rerank = False

	cases = load_cases(
		dataset_path=Path(args.dataset_path).resolve(),
		sample_ids=args.sample_ids,
		query=args.query,
		start_index=args.start_index,
		end_index=args.end_index,
		limit=args.limit,
	)
	if not cases:
		LOGGER.error("No retrieval cases found")
		return 1

	LOGGER.info("Loaded %d retrieval cases", len(cases))
	runner = LongMemEvalSelfHostRetriever(args)
	summary = runner.run_cases(cases)
	failed = summary.get("failed_cases", 0)
	if failed:
		LOGGER.error("Retrieval finished with %d failed cases", failed)
		return 1
	LOGGER.info("Retrieval finished successfully: %d cases", summary.get("successful_cases", 0))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
