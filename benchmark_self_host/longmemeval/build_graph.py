#!/usr/bin/env python3
"""Build LongMemEval per-QA SemanticGraph memories with mandol.auto_builder.

The self-host layout mirrors ``benchmark_self_host/locomo10`` while following
the LongMemEval offline makers:

1. hierarchical L0: pure user messages plus assistant chunks and REPLY_TO edges.
2. episodic: grouped session extraction with the auto_builder longmemeval prompt.
3. entity relation: grouped session entity extraction and mention-based graph units.
4. comparison: qa_x outputs are compared with the existing offline graph metadata.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import re
import shutil
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
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

from mandol.auto_builder import (  # noqa: E402
	EntityRelationAutoBuilder,
	EntityRelationBuilderConfig,
	EpisodicAutoBuilder,
	EpisodicBuilderConfig,
	HierarchicalAutoBuilder,
	HierarchicalBuilderConfig,
)
from mandol.auto_builder.entity_relation_builder import ExtractedEntity, ExtractedRelation  # noqa: E402
from mandol.auto_builder.episodic_builder import EpisodicFact  # noqa: E402
from mandol.auto_builder.episodic_prompts import EpisodicFactType  # noqa: E402
from mandol.auto_builder.graph_write_queue import GraphWriteRequest, dispatch_graph_write_requests  # noqa: E402
from mandol.auto_builder.strategy_config import STYLE_ALIASES, STYLE_STRATEGIES, PipelineStrategy  # noqa: E402
from mandol.core.memory_unit import MemoryUnit  # noqa: E402
from mandol.core.memory_space_registry import MemorySpaceRegistry, TowerSpace  # noqa: E402
from mandol.core.semantic_graph import SemanticGraph  # noqa: E402
from mandol.core.semantic_map import SemanticMap  # noqa: E402
from mandol.llm.llm_client import LLMClient  # noqa: E402
from mandol.retrieval.retrieval_interface import RetrievalMethod  # noqa: E402
from mandol.utils.logging_config import auto_configure_logging, setup_logging  # noqa: E402


LOGGER = logging.getLogger("longmemeval_build_graph")
CHECKPOINT_FILENAME = "build_checkpoint.json"
COMPLETED = "completed"
HIERARCHICAL_L0_SPACE = TowerSpace.HIERARCHICAL_L0.value
HIERARCHICAL_L1_SPACE = TowerSpace.HIERARCHICAL_L1.value
HIERARCHICAL_L2_SPACE = TowerSpace.HIERARCHICAL_L2.value
EPISODIC_SPACE = TowerSpace.EPISODIC_ROOT.value
GRAPH_ENTITY_SPACE = TowerSpace.GRAPH_ENTITIES.value
GRAPH_MENTION_SPACE = TowerSpace.GRAPH_MENTIONS.value
GRAPH_RELATION_SPACE = TowerSpace.GRAPH_RELATIONS.value
EPISODIC_SYSTEM_PROMPT = """You are a Memory Archivist AI specializing in extracting structured episodic memory facts from conversations.

Your task is to extract Atomic Memory Facts that can answer various types of questions:
- Simple facts: "What is my cat's name?"
- Aggregation: "How many model kits do I have?"
- Temporal ordering: "Which event happened first?"
- Assistant recall: "What restaurant did you recommend?"
- Personalized recommendations: "Can you suggest evening activities?"

Output valid JSON only. Preserve all details and nuances."""
ENTITY_SYSTEM_PROMPT = "You are a professional entity extraction expert. Extract entities in JSON format only."


@dataclass
class LongMemEvalQASample:
	sample_id: str
	qa_index: int
	data: Dict[str, Any]


@dataclass
class SampleBuildResult:
	sample_id: str
	qa_index: int
	output_dir: str
	success: bool
	skipped: bool = False
	resumed: bool = False
	l0_units: int = 0
	l0_edges: int = 0
	l1_units: int = 0
	l2_units: int = 0
	episodic_facts_extracted: int = 0
	episodic_facts_after_dedup: int = 0
	episodic_units_added: int = 0
	entities_extracted: int = 0
	entities_after_dedup: int = 0
	entity_mentions_added: int = 0
	relations_extracted: int = 0
	relation_units_added: int = 0
	graph_units: int = 0
	graph_edges: int = 0
	graph_saved: bool = False
	comparison_path: str = ""
	checkpoint_path: str = ""
	processing_seconds: float = 0.0
	errors: Optional[List[str]] = None

	def to_dict(self) -> Dict[str, Any]:
		data = asdict(self)
		data["errors"] = self.errors or []
		return data


def configure_logging(debug: bool = False) -> None:
	level = logging.DEBUG if debug else logging.INFO
	if auto_configure_logging() is None:
		setup_logging(level=level)
	logging.getLogger().setLevel(level)
	LOGGER.setLevel(level)


def make_jsonable(value: Any) -> Any:
	if hasattr(value, "to_dict"):
		return make_jsonable(value.to_dict())
	if hasattr(value, "__dict__") and value.__class__.__module__.startswith("mandol"):
		return make_jsonable(value.__dict__)
	if value.__class__.__module__.startswith("mandol"):
		slots = getattr(type(value), "__slots__", ())
		if isinstance(slots, str):
			slots = (slots,)
		return make_jsonable({name: getattr(value, name) for name in slots if hasattr(value, name)})
	if isinstance(value, dict):
		return {str(k): make_jsonable(v) for k, v in value.items()}
	if isinstance(value, (list, tuple, set)):
		return [make_jsonable(v) for v in value]
	return value


def read_json(path: Path, default: Any = None) -> Any:
	if not path.exists():
		return default
	with path.open("r", encoding="utf-8") as file:
		return json.load(file)


def write_json(path: Path, data: Any) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as file:
		json.dump(make_jsonable(data), file, ensure_ascii=False, indent=2)


def resolve_strategy(strategy_name: str) -> PipelineStrategy:
	strategy_key = STYLE_ALIASES.get(strategy_name, strategy_name)
	if strategy_key not in STYLE_STRATEGIES:
		raise ValueError(f"Unknown auto_builder strategy: {strategy_name}")
	return STYLE_STRATEGIES[strategy_key]


def load_text_splitter(chunk_size: int, chunk_overlap: int):
	try:
		splitter_module = importlib.import_module("langchain_text_splitters")
	except ImportError:
		splitter_module = importlib.import_module("langchain.text_splitter")
	RecursiveCharacterTextSplitter = splitter_module.RecursiveCharacterTextSplitter
	return RecursiveCharacterTextSplitter(
		chunk_size=chunk_size,
		chunk_overlap=chunk_overlap,
		length_function=len,
		separators=["\n\n", "\n", "。", ". ", "！", "! ", "？", "? ", "；", "; ", "，", ", ", " ", ""],
	)


def sanitize_content(content: str) -> str:
	if not content:
		return content
	content = content.replace("\r\n", "\n").replace("\r", "\n")
	content = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]", "", content)
	content = re.sub(r"[\u200B-\u200D\uFEFF\u2028\u2029]", "", content)
	content = re.sub(r"\\(?![\"\\/bfnrtu])", r"\\\\", content)
	content = re.sub(r"[\u0085\u000B\u000C]", " ", content)
	return content


def group_ranges(total: int, group_size: int) -> List[Tuple[int, int]]:
	group_size = max(1, int(group_size))
	return [(start, min(start + group_size, total)) for start in range(0, total, group_size)]


def sample_id_from_index(index: int) -> str:
	return f"qa_{index}"


def qa_index_from_sample_id(sample_id: str) -> Optional[int]:
	match = re.match(r"^qa_(\d+)$", sample_id)
	return int(match.group(1)) if match else None


def memory_unit_count(graph: SemanticGraph) -> int:
	return len(graph.get_all_units())


def graph_edge_count(graph: SemanticGraph) -> int:
	return graph.rx_graph.num_edges() if hasattr(graph, "rx_graph") else 0


class LongMemEvalGraphBuilder:
	def __init__(self, args: argparse.Namespace):
		self.args = args
		self.strategy = resolve_strategy(args.strategy)
		self.output_dir = Path(args.output_dir).resolve()
		self.dataset_path = Path(args.dataset_path).resolve()
		self.offline_root = Path(args.offline_root).resolve()
		self.extraction_llm: Optional[LLMClient] = None
		self.dedup_llm: Optional[LLMClient] = None

	def build_all(self) -> List[SampleBuildResult]:
		samples = self._select_samples()
		LOGGER.info("Found %d LongMemEval samples to build", len(samples))
		results: List[SampleBuildResult] = []
		for ordinal, sample in enumerate(samples, 1):
			LOGGER.info("[%d/%d] Building %s", ordinal, len(samples), sample.sample_id)
			result = self.build_sample(sample)
			results.append(result)
			write_json(self.output_dir / "build_summary.json", self._summary(results))
		return results

	def build_sample(self, sample: LongMemEvalQASample) -> SampleBuildResult:
		start_time = time.perf_counter()
		sample_dir = self.output_dir / sample.sample_id
		artifacts_dir = sample_dir / "artifacts"
		checkpoint_path = sample_dir / CHECKPOINT_FILENAME
		result = SampleBuildResult(
			sample_id=sample.sample_id,
			qa_index=sample.qa_index,
			output_dir=str(sample_dir),
			success=False,
			checkpoint_path=str(checkpoint_path),
		)

		try:
			if self.args.force and sample_dir.exists():
				shutil.rmtree(sample_dir)
			sample_dir.mkdir(parents=True, exist_ok=True)
			artifacts_dir.mkdir(parents=True, exist_ok=True)

			checkpoint = {} if self.args.no_resume else read_json(checkpoint_path, default={}) or {}
			result.resumed = bool(checkpoint.get("stages"))
			if self.args.skip_existing and checkpoint.get("status") == COMPLETED and not self.args.force:
				result.success = True
				result.skipped = True
				return result

			if self.args.dry_run:
				l0_nodes, l0_edges, l0_stats = self._build_l0_records(sample)
				write_json(artifacts_dir / "l0_nodes.json", self._nodes_output(sample, l0_nodes, l0_stats))
				write_json(artifacts_dir / "l0_edges.json", self._edges_output(sample, l0_edges))
				write_json(artifacts_dir / "l0_summary.json", self._l0_summary(sample, l0_stats))
				result.l0_units = len(l0_nodes)
				result.l0_edges = len(l0_edges)
				result.success = True
				self._write_checkpoint(checkpoint_path, sample, result, status=COMPLETED, stages={"dry_run": COMPLETED})
				return result

			self._ensure_llm_clients()
			graph = self._create_graph()
			self._ensure_spaces(graph, sample.sample_id)
			stages = checkpoint.get("stages", {}) if checkpoint else {}

			l0_nodes, l0_edges, l0_stats = self._build_l0_records(sample)
			write_json(artifacts_dir / "l0_nodes.json", self._nodes_output(sample, l0_nodes, l0_stats))
			write_json(artifacts_dir / "l0_edges.json", self._edges_output(sample, l0_edges))
			write_json(artifacts_dir / "l0_summary.json", self._l0_summary(sample, l0_stats))
			l0_units = self._add_l0_to_graph(graph, sample, l0_nodes, l0_edges)
			result.l0_units = len(l0_units)
			result.l0_edges = len(l0_edges)
			stages["l0"] = COMPLETED
			self._write_checkpoint(checkpoint_path, sample, result, status="running", stages=stages)

			if not self.args.skip_hierarchical:
				hierarchical = self._build_optional_hierarchical(graph, sample, l0_units, artifacts_dir)
				result.l1_units = len(hierarchical.get("l1_results", []))
				result.l2_units = 1 if hierarchical.get("l2_result") else 0
				stages["hierarchical"] = COMPLETED
				self._write_checkpoint(checkpoint_path, sample, result, status="running", stages=stages)

			if not self.args.skip_episodic:
				episodic = self._build_episodic(graph, sample, l0_units, artifacts_dir)
				result.episodic_facts_extracted = len(episodic.get("facts", []))
				result.episodic_facts_after_dedup = len(episodic.get("output_facts", []))
				result.episodic_units_added = len(episodic.get("added_uids", []))
				stages["episodic"] = COMPLETED
				self._write_checkpoint(checkpoint_path, sample, result, status="running", stages=stages)

			if not self.args.skip_entity_relation:
				entity_relation = self._build_entity_relation(graph, sample, l0_units, artifacts_dir)
				result.entities_extracted = len(entity_relation.get("raw_entities", []))
				result.entities_after_dedup = len(entity_relation.get("entities", []))
				result.entity_mentions_added = len(entity_relation.get("mention_uids", []))
				result.relations_extracted = len(entity_relation.get("relations", []))
				result.relation_units_added = len(entity_relation.get("relation_uids", []))
				stages["entity_relation"] = COMPLETED
				self._write_checkpoint(checkpoint_path, sample, result, status="running", stages=stages)

			retriever_methods = self._retriever_methods_to_build()
			self._finalize_graph(graph, retriever_methods)
			graph.save_graph(
				str(sample_dir),
				force_rebuild_retrievers=True,
				retriever_methods_to_build=retriever_methods,
				build_sparse_vectors=self.args.build_splade,
			)
			result.graph_saved = True
			result.graph_units = memory_unit_count(graph)
			result.graph_edges = graph_edge_count(graph)

			comparison = self._compare_with_offline(sample, result)
			comparison_path = artifacts_dir / "comparison_report.json"
			write_json(comparison_path, comparison)
			result.comparison_path = str(comparison_path)

			result.success = True
			stages["save_graph"] = COMPLETED
			stages["compare_offline"] = COMPLETED
			result.processing_seconds = time.perf_counter() - start_time
			write_json(sample_dir / "build_result.json", result.to_dict())
			self._write_checkpoint(checkpoint_path, sample, result, status=COMPLETED, stages=stages)
			return result
		except Exception as exc:
			LOGGER.error("[%s] build failed: %s", sample.sample_id, exc)
			LOGGER.debug(traceback.format_exc())
			result.success = False
			result.errors = [f"{type(exc).__name__}: {exc}"]
			result.processing_seconds = time.perf_counter() - start_time
			write_json(sample_dir / "build_result.json", result.to_dict())
			self._write_checkpoint(checkpoint_path, sample, result, status="failed", stages={})
			return result

	def _select_samples(self) -> List[LongMemEvalQASample]:
		data = read_json(self.dataset_path, default=[])
		if not isinstance(data, list):
			raise ValueError(f"Dataset must be a JSON list: {self.dataset_path}")
		selected: List[LongMemEvalQASample] = []
		sample_filter = set(self.args.sample_ids or [])
		start_index = self.args.start_index
		end_index = len(data) if self.args.end_index is None else min(self.args.end_index + 1, len(data))
		for idx in range(start_index, end_index):
			sample_id = sample_id_from_index(idx)
			if sample_filter and sample_id not in sample_filter:
				continue
			selected.append(LongMemEvalQASample(sample_id=sample_id, qa_index=idx, data=data[idx]))
			if self.args.limit and len(selected) >= self.args.limit:
				break
		return selected

	def _ensure_llm_clients(self) -> None:
		if self.extraction_llm is None:
			self.extraction_llm = LLMClient(
				model_name=self.args.extraction_model,
				max_context_ratio=self.args.max_context_ratio,
				request_timeout=self.args.extraction_request_timeout,
				request_max_retries=self.args.extraction_request_max_retries,
			)
		if self.dedup_llm is None:
			self.dedup_llm = LLMClient(
				model_name=self.args.dedup_model,
				max_context_ratio=self.args.max_context_ratio,
				request_timeout=self.args.dedup_request_timeout,
				request_max_retries=self.args.dedup_request_max_retries,
			)

	def _create_graph(self) -> SemanticGraph:
		semantic_map = SemanticMap(
			embedding_model_name=self.args.embedding_model,
			embedding_dim=self.args.embedding_dim,
			faiss_index_type=self.args.faiss_index_type,
		)
		return SemanticGraph(semantic_map_instance=semantic_map)

	def _ensure_spaces(self, graph: SemanticGraph, sample_id: str) -> None:
		MemorySpaceRegistry.initialize_spaces(graph)
		graph.create_memory_space_in_map(sample_id)

	def _build_l0_records(self, sample: LongMemEvalQASample) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
		splitter = load_text_splitter(self.args.chunk_size, self.args.chunk_overlap)
		qa_data = sample.data
		sessions = qa_data.get("haystack_sessions", [])
		session_ids = qa_data.get("haystack_session_ids", [])
		session_dates = qa_data.get("haystack_dates", [])
		nodes: List[Dict[str, Any]] = []
		edges: List[Dict[str, Any]] = []
		stats = {
			"total_messages": 0,
			"total_chunks": 0,
			"user_messages": 0,
			"assistant_messages": 0,
			"chunked_assistant_messages": 0,
			"total_edges": 0,
		}
		last_user_uid: Optional[str] = None
		for session_index, messages in enumerate(sessions):
			session_id = session_ids[session_index] if session_index < len(session_ids) else f"session_{session_index}"
			session_date = session_dates[session_index] if session_index < len(session_dates) else ""
			for message_index, message in enumerate(messages):
				role = str(message.get("role", "unknown")) if isinstance(message, dict) else "unknown"
				content = str(message.get("content", "")) if isinstance(message, dict) else str(message)
				if not content.strip():
					continue
				stats["total_messages"] += 1
				if role.lower() == "user":
					stats["user_messages"] += 1
					uid = f"qa{sample.qa_index}_s{session_index}_msg{message_index}"
					last_user_uid = uid
					nodes.append(
						{
							"uid": uid,
							"raw_data": {
								"text_content": content,
								"role": "user",
								"session_id": session_id,
								"session_index": session_index,
								"message_index": message_index,
								"session_date": session_date,
								"qa_index": sample.qa_index,
								"question_id": qa_data.get("question_id"),
								"question_type": qa_data.get("question_type", "unknown"),
							},
							"metadata": {
								"created": datetime.now().isoformat(),
								"qa_index": sample.qa_index,
								"session_id": session_id,
								"session_date": session_date,
								"role": "user",
								"node_type": "user_message",
								"is_chunk": False,
							},
						}
					)
				elif role.lower() == "assistant":
					stats["assistant_messages"] += 1
					chunks = splitter.split_text(content)
					if len(chunks) > 1:
						stats["chunked_assistant_messages"] += 1
					stats["total_chunks"] += len(chunks)
					for chunk_index, chunk_text in enumerate(chunks):
						uid = f"qa{sample.qa_index}_s{session_index}_msg{message_index}_c{chunk_index}"
						nodes.append(
							{
								"uid": uid,
								"raw_data": {
									"text_content": chunk_text,
									"role": "assistant",
									"session_id": session_id,
									"session_index": session_index,
									"message_index": message_index,
									"chunk_index": chunk_index,
									"total_chunks": len(chunks),
									"session_date": session_date,
									"qa_index": sample.qa_index,
									"question_id": qa_data.get("question_id"),
									"question_type": qa_data.get("question_type", "unknown"),
									"is_chunk": len(chunks) > 1,
									"parent_msg_id": f"qa{sample.qa_index}_s{session_index}_msg{message_index}",
								},
								"metadata": {
									"created": datetime.now().isoformat(),
									"qa_index": sample.qa_index,
									"session_id": session_id,
									"session_date": session_date,
									"role": "assistant",
									"node_type": "assistant_chunk",
									"chunk_index": chunk_index,
									"total_chunks": len(chunks),
									"is_chunk": len(chunks) > 1,
								},
							}
						)
						if last_user_uid:
							edges.append(
								{
									"source_uid": uid,
									"target_uid": last_user_uid,
									"relation_type": "REPLY_TO",
									"properties": {
										"created": datetime.now().isoformat(),
										"session_id": session_id,
										"qa_index": sample.qa_index,
										"message_index": message_index,
										"chunk_index": chunk_index,
									},
								}
							)
							stats["total_edges"] += 1
		return nodes, edges, stats

	def _add_l0_to_graph(
		self,
		graph: SemanticGraph,
		sample: LongMemEvalQASample,
		nodes: List[Dict[str, Any]],
		edges: List[Dict[str, Any]],
	) -> List[MemoryUnit]:
		units = [MemoryUnit(uid=node["uid"], raw_data=node.get("raw_data", {}), metadata=node.get("metadata", {})) for node in nodes]
		if units:
			graph.batch_add_units(
				units=units,
				batch_size=self.args.batch_size,
								space_names=[HIERARCHICAL_L0_SPACE, sample.sample_id],
				index_update_mode="none",
				generate_sparse_embedding=False,
				show_progress=False,
			)
		for edge in edges:
			source_uid = edge.get("source_uid")
			target_uid = edge.get("target_uid")
			if source_uid and target_uid and graph.get_unit(source_uid) and graph.get_unit(target_uid):
				graph.add_relationship(
					source_uid=source_uid,
					target_uid=target_uid,
					relationship_name=edge.get("relation_type", "REPLY_TO"),
					bidirectional=False,
					**(edge.get("properties") or {}),
				)
		return units

	def _build_optional_hierarchical(
		self,
		graph: SemanticGraph,
		sample: LongMemEvalQASample,
		l0_units: Sequence[MemoryUnit],
		artifacts_dir: Path,
	) -> Dict[str, Any]:
		if not (self.args.build_l1 or self.args.build_l2):
			write_json(artifacts_dir / "hierarchical_skipped.json", {"reason": "longmemeval offline hierarchical workflow is L0-only"})
			return {"l1_results": [], "l2_result": None}
		builder = HierarchicalAutoBuilder(
			semantic_system=graph,
			llm_client=self.extraction_llm,
			config=HierarchicalBuilderConfig(
				extraction_style="longmemeval",
				enable_contextual_retrieval=False,
				enable_chunking=True,
				chunk_size=self.args.chunk_size,
				chunk_overlap=self.args.chunk_overlap,
				parallel_workers=self.args.hierarchical_session_workers,
			),
		)
		l1_results = builder.extract_l1_from_l0_units(
			l0_units=list(l0_units),
			session_id=sample.sample_id,
			session_date=self._sample_reference_date(sample),
			participants=["User", "Assistant"],
		) if self.args.build_l1 else []
		l2_result = builder.aggregate_l2_from_l1(l1_results, sample.sample_id, ["User", "Assistant"]) if self.args.build_l2 and l1_results else None
		if l1_results or l2_result:
			builder.add_to_semantic_system(l1_results, l2_result, rebuild_index=False)
		write_json(artifacts_dir / "l1_results.json", l1_results)
		write_json(artifacts_dir / "l2_result.json", l2_result)
		return {"l1_results": l1_results, "l2_result": l2_result}

	def _build_episodic(
		self,
		graph: SemanticGraph,
		sample: LongMemEvalQASample,
		l0_units: Sequence[MemoryUnit],
		artifacts_dir: Path,
	) -> Dict[str, Any]:
		builder = EpisodicAutoBuilder(
			semantic_system=graph,
			llm_client=self.extraction_llm,
			config=EpisodicBuilderConfig(
				extraction_style="longmemeval",
				fact_types=EpisodicFactType.longmemeval_types(),
				enable_deduplication=not self.args.no_episodic_dedup,
				dedup_method=self.args.episodic_dedup_method,
				dbscan_eps=self.args.episodic_default_eps,
				dbscan_min_samples=self.args.episodic_default_min_samples,
								auto_optimize_dbscan=False,
				dbscan_eps_range=tuple(self.args.episodic_dbscan_eps_range),
				dbscan_min_samples_range=tuple(self.args.episodic_dbscan_min_samples_range),
				dedup_parallel_workers=self.args.episodic_dedup_workers,
				large_cluster_threshold=self.args.episodic_large_cluster_threshold,
				embedding_model=self.args.dedup_embedding_model,
			),
		)
		raw_cache_path = artifacts_dir / "episodic_facts_raw.json"
		dedup_cache_path = artifacts_dir / "episodic_facts_deduplicated.json"
		if not self.args.force and raw_cache_path.exists() and dedup_cache_path.exists():
			facts = [EpisodicFact.from_dict(item) for item in read_json(raw_cache_path, default=[])]
			output_facts = [EpisodicFact.from_dict(item) for item in read_json(dedup_cache_path, default=[])]
			added_uids = self._add_episodic_facts(graph, sample.sample_id, builder, output_facts)
			write_json(artifacts_dir / "episodic_added_uids.json", added_uids)
			return {"facts": facts, "output_facts": output_facts, "added_uids": added_uids}
		group_jobs = self._session_group_jobs(sample, self.args.episodic_sessions_per_group)
		group_cache_dir = artifacts_dir / "episodic_groups"
		group_cache_dir.mkdir(parents=True, exist_ok=True)
		requests: List[Dict[str, Any]] = []
		facts_by_group: List[Tuple[int, List[EpisodicFact], Dict[str, Any]]] = []

		def run_group(order: int, start: int, end: int) -> Tuple[int, List[EpisodicFact], Dict[str, Any]]:
			group_id = f"{sample.sample_id}_session_{start}_{end - 1}"
			content, reference_date = self._build_episodic_sessions_text(sample, start, end)
			prompt = builder.prompt_manager.get_extraction_prompt(
				style="longmemeval",
				content=content,
				reference_date=reference_date,
				source_id=group_id,
				speakers="User, Assistant",
				fact_types=EpisodicFactType.longmemeval_types(),
			)
			cache_path = group_cache_dir / f"group_{order:04d}_{start}_{end - 1}.json"
			cached = read_json(cache_path, default=None) if not self.args.force else None
			response = cached.get("response", "") if isinstance(cached, dict) else ""
			if not response:
				response = self.extraction_llm.generate_answer(
					prompt=prompt,
					temperature=0.1,
					max_tokens=self.args.episodic_extraction_max_tokens,
					json_format=False,
					system_prompt=EPISODIC_SYSTEM_PROMPT,
				)
			parsed = builder._parse_extraction_response(response)
			fact_items = self._normalize_episodic_items(parsed)
			source_uids = [unit.uid for unit in l0_units if start <= int((unit.metadata or {}).get("session_index", -1)) < end]
			facts = [builder._convert_to_episodic_fact(item, source_uids) for item in fact_items]
			for fact in facts:
				fact.metadata.update({"source_id": group_id, "session_range": [start, end - 1], "extraction_style": "longmemeval"})
			request = {
				"group_id": group_id,
				"session_range": [start, end - 1],
				"reference_date": reference_date,
				"facts": len(facts),
				"parsed_items": len(fact_items),
				"response": response,
			}
			if self.args.save_prompts:
				request["prompt"] = prompt
			write_json(cache_path, request)
			return order, facts, request

		def failed_group(order: int, start: int, end: int, exc: Exception) -> Tuple[int, List[EpisodicFact], Dict[str, Any]]:
			group_id = f"{sample.sample_id}_session_{start}_{end - 1}"
			request = {
				"group_id": group_id,
				"session_range": [start, end - 1],
				"facts": 0,
				"parsed_items": 0,
				"response": "",
				"error": f"{type(exc).__name__}: {exc}",
			}
			cache_path = group_cache_dir / f"group_{order:04d}_{start}_{end - 1}.json"
			write_json(cache_path, request)
			LOGGER.warning("[%s] episodic group %s failed; continuing with partial extraction: %s", sample.sample_id, group_id, exc)
			return order, [], request

		max_workers = min(max(1, self.args.episodic_session_workers), len(group_jobs) or 1)
		if max_workers > 1:
			with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="LongMemEpisodic") as executor:
				future_to_group = {executor.submit(run_group, order, start, end): (order, start, end) for order, start, end in group_jobs}
				for future in as_completed(future_to_group):
					order, start, end = future_to_group[future]
					try:
						facts_by_group.append(future.result())
					except Exception as exc:
						if not self.args.allow_partial_extraction:
							raise
						facts_by_group.append(failed_group(order, start, end, exc))
		else:
			for order, start, end in group_jobs:
				try:
					facts_by_group.append(run_group(order, start, end))
				except Exception as exc:
					if not self.args.allow_partial_extraction:
						raise
					facts_by_group.append(failed_group(order, start, end, exc))

		facts: List[EpisodicFact] = []
		for _, group_facts, request in sorted(facts_by_group, key=lambda item: item[0]):
			facts.extend(group_facts)
			requests.append(request)

		output_facts = facts
		if not self.args.no_episodic_dedup and facts:
			builder.llm_client = self.dedup_llm
			output_facts = builder.deduplicate_facts(facts)
			builder.llm_client = self.extraction_llm

		added_uids = self._add_episodic_facts(graph, sample.sample_id, builder, output_facts)
		write_json(artifacts_dir / "episodic_requests.json", requests)
		write_json(artifacts_dir / "episodic_facts_raw.json", facts)
		write_json(artifacts_dir / "episodic_facts_deduplicated.json", output_facts)
		write_json(artifacts_dir / "episodic_added_uids.json", added_uids)
		return {"facts": facts, "output_facts": output_facts, "added_uids": added_uids}

	def _add_episodic_facts(
		self,
		graph: SemanticGraph,
		sample_id: str,
		builder: EpisodicAutoBuilder,
		facts: Sequence[EpisodicFact],
	) -> List[str]:
		added: List[str] = []
		write_requests: List[GraphWriteRequest] = []
		for index, fact in enumerate(facts):
			unit = builder._create_memory_unit_from_fact(fact)
			if not unit.uid.startswith(f"{sample_id}_"):
				unit.uid = f"{sample_id}_{unit.uid}_{index}"
			unit.raw_data.setdefault("qa_id", sample_id)
			unit.raw_data.setdefault("node_type", str(fact.fact_type).lower() if fact.fact_type else "episodic_event")
			write_requests.append(GraphWriteRequest(
				unit=unit,
				explicit_content_for_embedding=fact.get_text_content(),
				content_type_for_embedding="text",
								space_names=[EPISODIC_SPACE, sample_id],
				index_update_mode="none",
				generate_sparse_embedding=False,
				source="longmemeval_episodic",
			))
			added.append(unit.uid)
		dispatch_graph_write_requests(graph, write_requests)
		return added

	def _build_entity_relation(
		self,
		graph: SemanticGraph,
		sample: LongMemEvalQASample,
		l0_units: Sequence[MemoryUnit],
		artifacts_dir: Path,
	) -> Dict[str, Any]:
		builder = EntityRelationAutoBuilder(
			semantic_system=graph,
			llm_client=self.extraction_llm,
			config=EntityRelationBuilderConfig(
				extraction_style="longmemeval",
				embedding_model=self.args.dedup_embedding_model,
				llm_max_tokens=self.args.entity_extraction_max_tokens,
				dbscan_eps=self.args.entity_default_eps,
				dbscan_min_samples=self.args.entity_default_min_samples,
				auto_optimize_dbscan=True,
				dbscan_eps_range=tuple(self.args.entity_dbscan_eps_range),
				dbscan_min_samples_range=tuple(self.args.entity_dbscan_min_samples_range),
				parallel_workers=self.args.entity_dedup_workers,
				large_cluster_threshold=self.args.entity_large_cluster_threshold,
				enable_relation_extraction=self.args.enable_relation_extraction,
				enable_llm_deduplication=not self.args.no_entity_dedup,
			),
		)
		group_jobs = self._session_group_jobs(sample, self.args.entity_sessions_per_group)
		group_cache_dir = artifacts_dir / "entity_groups"
		group_cache_dir.mkdir(parents=True, exist_ok=True)
		requests: List[Dict[str, Any]] = []
		raw_by_group: List[Tuple[int, List[Dict[str, Any]], Dict[str, Any]]] = []

		def run_group(order: int, start: int, end: int) -> Tuple[int, List[Dict[str, Any]], Dict[str, Any]]:
			group_id = f"{sample.sample_id}_session_{start}_{end - 1}"
			content = self._build_sessions_text(sample, start, end)
			reference_date = self._reference_date_for_range(sample, start, end)
			prompt = builder.prompt_manager.get_entity_extraction_prompt_v2(
				style="longmemeval",
				content=content,
				reference_date=reference_date,
				source_id=group_id,
				content_type="chat",
				session_type="chat",
			)
			cache_path = group_cache_dir / f"group_{order:04d}_{start}_{end - 1}.json"
			cached = read_json(cache_path, default=None) if not self.args.force else None
			response = cached.get("response", "") if isinstance(cached, dict) else ""
			if not response:
				response = self.extraction_llm.generate_answer(
					prompt=prompt,
					temperature=builder.config.llm_temperature,
					max_tokens=self.args.entity_extraction_max_tokens,
					json_format=False,
					system_prompt=ENTITY_SYSTEM_PROMPT,
				)
			parsed = builder._safe_parse_json(response)
			parse_error = parsed.get("_parse_error") if isinstance(parsed, dict) else None
			if parse_error:
				raise ValueError(f"entity extraction parse failed: {parse_error}")
			entities = self._normalize_entity_items(parsed)
			unit_uids = [unit.uid for unit in l0_units if start <= int((unit.metadata or {}).get("session_index", -1)) < end]
			session_ids = sample.data.get("haystack_session_ids", [])
			session_dates = sample.data.get("haystack_dates", [])
			for entity in entities:
				entity["source_id"] = group_id
				entity["unit_uids"] = unit_uids
				entity["reference_date"] = reference_date
				entity["session_range"] = [start, end - 1]
				if not entity.get("session_id") and start < len(session_ids):
					entity["session_id"] = session_ids[start]
				if not entity.get("session_date") and start < len(session_dates):
					entity["session_date"] = session_dates[start]
			request = {
				"group_id": group_id,
				"session_range": [start, end - 1],
				"reference_date": reference_date,
				"entities": len(entities),
				"response": response,
			}
			if self.args.save_prompts:
				request["prompt"] = prompt
			write_json(cache_path, request)
			return order, entities, request

		def failed_group(order: int, start: int, end: int, exc: Exception) -> Tuple[int, List[Dict[str, Any]], Dict[str, Any]]:
			group_id = f"{sample.sample_id}_session_{start}_{end - 1}"
			request = {
				"group_id": group_id,
				"session_range": [start, end - 1],
				"reference_date": self._reference_date_for_range(sample, start, end),
				"entities": 0,
				"response": "",
				"error": f"{type(exc).__name__}: {exc}",
			}
			cache_path = group_cache_dir / f"group_{order:04d}_{start}_{end - 1}.json"
			write_json(cache_path, request)
			LOGGER.warning("[%s] entity group %s failed; continuing with partial extraction: %s", sample.sample_id, group_id, exc)
			return order, [], request

		max_workers = min(max(1, self.args.entity_session_workers), len(group_jobs) or 1)
		if max_workers > 1:
			with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="LongMemEntity") as executor:
				future_to_group = {executor.submit(run_group, order, start, end): (order, start, end) for order, start, end in group_jobs}
				for future in as_completed(future_to_group):
					order, start, end = future_to_group[future]
					try:
						raw_by_group.append(future.result())
					except Exception as exc:
						if not self.args.allow_partial_extraction:
							raise
						raw_by_group.append(failed_group(order, start, end, exc))
		else:
			for order, start, end in group_jobs:
				try:
					raw_by_group.append(run_group(order, start, end))
				except Exception as exc:
					if not self.args.allow_partial_extraction:
						raise
					raw_by_group.append(failed_group(order, start, end, exc))

		raw_entities: List[Dict[str, Any]] = []
		for _, group_entities, request in sorted(raw_by_group, key=lambda item: item[0]):
			raw_entities.extend(group_entities)
			requests.append(request)

		entities = self._deduplicate_entities_by_type(builder, raw_entities)
		relations: List[ExtractedRelation] = []
		if self.args.enable_relation_extraction and entities:
			relations = builder.extract_relations_from_entities(list(l0_units), entities, session_type="chat")
		mention_uids = self._add_entity_mentions(graph, sample.sample_id, entities)
		relation_uids = self._add_relation_units(graph, sample.sample_id, relations)
		entity_data = {"qa_id": sample.sample_id, "total_entities": len(entities), "entities": [entity.to_dict() for entity in entities]}
		write_json(artifacts_dir / "entity_requests.json", requests)
		write_json(artifacts_dir / "entity_raw_entities.json", raw_entities)
		write_json(artifacts_dir / "entity_entities_deduplicated.json", entity_data)
		write_json(artifacts_dir / "entity_mentions_added.json", mention_uids)
		write_json(artifacts_dir / "entity_relations.json", relations)
		return {"raw_entities": raw_entities, "entities": entities, "relations": relations, "mention_uids": mention_uids, "relation_uids": relation_uids}

	def _normalize_episodic_items(self, value: Any) -> List[Dict[str, Any]]:
		if isinstance(value, dict):
			for key in ("memory_facts", "facts", "memories", "episodic_memories", "events", "items", "results"):
				nested = value.get(key)
				if isinstance(nested, list):
					return self._normalize_episodic_items(nested)
			if "content" in value or "fact_type" in value or "category" in value:
				return [self._normalize_episodic_fact_item(value)]
			normalized: List[Dict[str, Any]] = []
			for nested in value.values():
				if isinstance(nested, list):
					normalized.extend(self._normalize_episodic_items(nested))
			return normalized
		if isinstance(value, list):
			normalized: List[Dict[str, Any]] = []
			for item in value:
				normalized.extend(self._normalize_episodic_items(item))
			return normalized
		return []

	@staticmethod
	def _normalize_episodic_fact_item(item: Dict[str, Any]) -> Dict[str, Any]:
		normalized = dict(item)
		if "fact_type" not in normalized and "category" in normalized:
			normalized["fact_type"] = normalized.get("category")
		if "time" not in normalized and "temporal" in normalized:
			normalized["time"] = normalized.get("temporal")
		if "details" not in normalized and "attributes" in normalized:
			normalized["details"] = normalized.get("attributes")
		normalized.setdefault("participants", ["User"])
		normalized.setdefault("retrieval_keys", [])
		return normalized

	def _normalize_entity_items(self, value: Any) -> List[Dict[str, Any]]:
		if isinstance(value, dict):
			for key in ("entities", "entity_mentions", "mentions", "items", "results"):
				nested = value.get(key)
				if isinstance(nested, list):
					return self._normalize_entity_items(nested)
			if "name" in value or "entity" in value or "type" in value or "entity_type" in value:
				return [value]
			normalized: List[Dict[str, Any]] = []
			for nested in value.values():
				if isinstance(nested, list):
					normalized.extend(self._normalize_entity_items(nested))
			return normalized
		if isinstance(value, list):
			normalized: List[Dict[str, Any]] = []
			for item in value:
				normalized.extend(self._normalize_entity_items(item))
			return normalized
		return []

	def _deduplicate_entities_by_type(self, builder: EntityRelationAutoBuilder, raw_entities: List[Dict[str, Any]]) -> List[ExtractedEntity]:
		if not raw_entities:
			return []
		if self.args.no_entity_dedup:
			entities = [builder._convert_raw_to_extracted_entity(entity, f"E{idx}") for idx, entity in enumerate(raw_entities)]
		elif self.args.entity_dedup_by_type:
			grouped: Dict[str, List[Dict[str, Any]]] = {}
			for entity in raw_entities:
				entity_type = builder._normalize_entity_type_for_style(
					entity.get("type") or entity.get("entity_type") or "UNKNOWN"
				)
				normalized_entity = dict(entity)
				normalized_entity["type"] = entity_type
				normalized_entity["entity_type"] = entity_type
				grouped.setdefault(entity_type, []).append(normalized_entity)
			entities = []
			builder.llm_client = self.dedup_llm
			for entity_type in sorted(grouped):
				entities.extend(builder.deduplicate_entities(grouped[entity_type]))
			builder.llm_client = self.extraction_llm
		else:
			builder.llm_client = self.dedup_llm
			entities = builder.deduplicate_entities(raw_entities)
			builder.llm_client = self.extraction_llm
		for idx, entity in enumerate(entities):
			if not entity.entity_id or entity.entity_id.startswith("entity_") or entity.entity_id.startswith("merged_"):
				entity.entity_id = f"E{idx}"
		return entities

	def _add_entity_mentions(self, graph: SemanticGraph, sample_id: str, entities: Sequence[ExtractedEntity]) -> List[str]:
		mention_uids: List[str] = []
		write_requests: List[GraphWriteRequest] = []
		for entity_index, entity in enumerate(entities):
			entity_id = entity.entity_id or f"E{entity_index}"
			base_uid = entity_id if str(entity_id).startswith(f"{sample_id}_") else f"{sample_id}_{entity_id}"
			mentions = entity.mentions or []
			if not mentions:
				continue
			for mention_index, mention in enumerate(mentions, 1):
				mention_uid = f"{base_uid}_{entity_index}_mention_{mention_index}"
				content = mention.content or entity.name
				text_content = self._build_mention_text_content(entity.name, entity.entity_type, content, mention)
				raw_data = {
					"node_type": "evidence_mention",
					"qa_id": sample_id,
					"parent_entity_id": entity_id,
					"parent_hub_uid": None,
					"entity_canonical": entity.name,
					"entity_category": entity.entity_type,
					"mention_id": f"mention_{mention_index}",
					"content": content,
					"session_ids": [mention.session_id] if mention.session_id else [],
					"session_date": mention.session_date,
					"temporal_info": mention.temporal_info,
					"temporal_reference": mention.temporal_reference,
					"spatial_info": mention.spatial_info,
					"numerical_value": mention.numerical_value,
					"aliases": mention.aliases,
					"confidence": mention.confidence,
					"text_content": text_content,
					"created_at": datetime.now().isoformat(),
				}
				unit = MemoryUnit(uid=mention_uid, raw_data=raw_data)
				write_requests.append(GraphWriteRequest(
					unit=unit,
					explicit_content_for_embedding=text_content,
					content_type_for_embedding="text",
										space_names=[GRAPH_ENTITY_SPACE, GRAPH_MENTION_SPACE, sample_id],
					index_update_mode="none",
					generate_sparse_embedding=False,
					source="longmemeval_entity_mention",
				))
				mention_uids.append(mention_uid)
		dispatch_graph_write_requests(graph, write_requests)
		return mention_uids

	@staticmethod
	def _build_mention_text_content(canonical: str, entity_type: str, content: str, mention: Any) -> str:
		parts = [f"Entity: {canonical} (Type: {entity_type})"]
		if content:
			parts.append(f"Context: {content}")
		temporal_parts = []
		if getattr(mention, "temporal_info", None):
			temporal_parts.append(str(mention.temporal_info))
		if getattr(mention, "temporal_reference", None):
			temporal_parts.append(str(mention.temporal_reference))
		if temporal_parts:
			parts.append("Temporal: " + ", ".join(temporal_parts))
		attrs = []
		if getattr(mention, "spatial_info", None):
			attrs.append(f"spatial_info: {mention.spatial_info}")
		if getattr(mention, "numerical_value", None):
			attrs.append(f"numerical_value: {mention.numerical_value}")
		if attrs:
			parts.append("Attributes: " + ", ".join(attrs[:3]))
		return " | ".join(parts)

	def _add_relation_units(self, graph: SemanticGraph, sample_id: str, relations: Sequence[ExtractedRelation]) -> List[str]:
		relation_uids: List[str] = []
		write_requests: List[GraphWriteRequest] = []
		for index, relation in enumerate(relations):
			uid = f"{sample_id}_relation_{index}_{relation.relation_id}"
			content = f"{relation.source_entity_id} {relation.relation_type} {relation.target_entity_id}. {relation.context}"
			unit = MemoryUnit(
				uid=uid,
				raw_data={"text_content": content, "node_type": "entity_relation", **relation.to_dict()},
				metadata={"type": "entity_relation", "qa_id": sample_id, "created_at": datetime.now().isoformat()},
			)
			write_requests.append(GraphWriteRequest(
				unit=unit,
				explicit_content_for_embedding=content,
				content_type_for_embedding="text",
								space_names=[GRAPH_RELATION_SPACE, sample_id],
				index_update_mode="none",
				generate_sparse_embedding=False,
				source="longmemeval_relation",
			))
			relation_uids.append(uid)
		dispatch_graph_write_requests(graph, write_requests)
		return relation_uids

	def _retriever_methods_to_build(self) -> List[RetrievalMethod]:
		methods = [RetrievalMethod.BM25]
		if self.args.build_splade:
			methods.append(RetrievalMethod.SPLADE)
		return methods

	def _finalize_graph(self, graph: SemanticGraph, retriever_methods: Sequence[RetrievalMethod]) -> None:
		if self.args.build_splade:
			try:
				graph.build_sparse_embeddings(
					units=None,
					model_name=self.args.splade_model,
					batch_size=self.args.splade_batch_size,
					force_rebuild=False,
					show_progress=False,
				)
			except Exception as exc:
				LOGGER.warning("SPLADE build failed: %s", exc)
		graph.build_semantic_map_index()
		try:
			graph.get_multi_retriever().build_all_indexes(
				methods_to_build=list(retriever_methods),
				force_rebuild=True,
			)
		except Exception as exc:
			LOGGER.warning("Retriever index rebuild failed: %s", exc)

	def _session_group_jobs(self, sample: LongMemEvalQASample, sessions_per_group: int) -> List[Tuple[int, int, int]]:
		total = len(sample.data.get("haystack_sessions", []))
		return [(order, start, end) for order, (start, end) in enumerate(group_ranges(total, sessions_per_group))]

	def _build_sessions_text(self, sample: LongMemEvalQASample, start: int, end: int) -> str:
		sessions = sample.data.get("haystack_sessions", [])
		session_ids = sample.data.get("haystack_session_ids", [])
		session_dates = sample.data.get("haystack_dates", [])
		lines: List[str] = []
		for idx in range(start, end):
			session = sessions[idx]
			session_id = session_ids[idx] if idx < len(session_ids) else f"session_{idx}"
			session_date = session_dates[idx] if idx < len(session_dates) else ""
			lines.append(f"\n=== Session {idx + 1} ===")
			lines.append(f"Session ID: {session_id}")
			lines.append(f"Session Date: {session_date}")
			lines.append("Messages:")
			for message_index, message in enumerate(session, 1):
				role = str(message.get("role", "unknown")) if isinstance(message, dict) else "unknown"
				content = sanitize_content(str(message.get("content", ""))) if isinstance(message, dict) else sanitize_content(str(message))
				if len(content) > self.args.max_message_chars:
					content = content[: self.args.max_message_chars] + "\n... [content truncated due to length]"
				if content.strip():
					lines.append(f"  [{message_index}] {role}: {content}")
		return "\n".join(lines)

	def _build_episodic_sessions_text(self, sample: LongMemEvalQASample, start: int, end: int) -> Tuple[str, str]:
		sessions = sample.data.get("haystack_sessions", [])
		session_ids = sample.data.get("haystack_session_ids", [])
		session_dates = sample.data.get("haystack_dates", [])
		sessions_text = ""
		reference_date = ""
		actual_end = min(end, len(sessions))
		for idx in range(start, actual_end):
			session = sessions[idx]
			session_id = session_ids[idx] if idx < len(session_ids) else f"session_{idx}"
			session_date = session_dates[idx] if idx < len(session_dates) else "Unknown"
			if idx == actual_end - 1:
				reference_date = session_date
			sessions_text += f"\n### Session: {session_id}\n"
			sessions_text += f"**Date:** {session_date}\n\n"
			for message in session:
				role = str(message.get("role", "unknown")) if isinstance(message, dict) else "unknown"
				content = sanitize_content(str(message.get("content", ""))) if isinstance(message, dict) else sanitize_content(str(message))
				if len(content) > self.args.max_message_chars:
					content = content[: self.args.max_message_chars] + "\n... [content truncated due to length]"
				if not content.strip():
					continue
				if role.lower() == "user":
					sessions_text += f"**User:** {content}\n\n"
				elif role.lower() == "assistant":
					sessions_text += f"**Assistant:** {content}\n\n"
				else:
					sessions_text += f"**{role}:** {content}\n\n"
			sessions_text += "---\n"
		return sessions_text, reference_date or self._sample_reference_date(sample)

	def _reference_date_for_range(self, sample: LongMemEvalQASample, start: int, end: int) -> str:
		dates = sample.data.get("haystack_dates", [])
		for idx in range(end - 1, start - 1, -1):
			if idx < len(dates) and dates[idx]:
				return str(dates[idx])
		return self._sample_reference_date(sample)

	@staticmethod
	def _sample_reference_date(sample: LongMemEvalQASample) -> str:
		question_date = sample.data.get("question_date")
		if question_date:
			return str(question_date)
		dates = sample.data.get("haystack_dates", [])
		return str(dates[-1]) if dates else datetime.now().strftime("%Y-%m-%d")

	def _compare_with_offline(self, sample: LongMemEvalQASample, result: SampleBuildResult) -> Dict[str, Any]:
		offline_hier = read_json(self.offline_root / "longmemeval_hierarchical" / "step3_semantic_graphs" / sample.sample_id / "graph_metadata.json", default={}) or {}
		offline_epi = read_json(self.offline_root / "episodic_memory_graphs_new" / sample.sample_id / "meta_info.json", default={}) or {}
		offline_ent = read_json(self.offline_root / "entity_relation_graphs_new" / sample.sample_id / "meta_info.json", default={}) or {}
		l0_graph_stats = offline_hier.get("graph_stats", {})
		l0_source_stats = offline_hier.get("source_statistics", {})
		comparison = {
			"sample_id": sample.sample_id,
			"created_at": datetime.now().isoformat(),
			"hierarchical_l0": {
				"offline_node_count": l0_graph_stats.get("node_count"),
				"self_host_node_count": result.l0_units,
				"node_delta": result.l0_units - int(l0_graph_stats.get("node_count", 0) or 0),
				"offline_edge_count": l0_graph_stats.get("edge_count") or l0_source_stats.get("total_edges"),
				"self_host_edge_count": result.l0_edges,
				"edge_delta": result.l0_edges - int((l0_graph_stats.get("edge_count") or l0_source_stats.get("total_edges") or 0)),
				"matches": result.l0_units == l0_graph_stats.get("node_count") and result.l0_edges == (l0_graph_stats.get("edge_count") or l0_source_stats.get("total_edges")),
			},
			"episodic": {
				"offline_event_count": offline_epi.get("event_count"),
				"self_host_raw_fact_count": result.episodic_facts_extracted,
				"self_host_after_dedup_count": result.episodic_facts_after_dedup,
				"self_host_event_count": result.episodic_units_added,
				"event_delta": result.episodic_units_added - int(offline_epi.get("event_count", 0) or 0),
			},
			"entity_relation": {
				"offline_total_entities": offline_ent.get("total_entities"),
				"self_host_raw_entities": result.entities_extracted,
				"self_host_entities": result.entities_after_dedup,
				"entity_delta": result.entities_after_dedup - int(offline_ent.get("total_entities", 0) or 0),
				"offline_total_mentions": offline_ent.get("total_mentions"),
				"self_host_mentions": result.entity_mentions_added,
				"mention_delta": result.entity_mentions_added - int(offline_ent.get("total_mentions", 0) or 0),
			},
			"combined_graph": {
				"self_host_units": result.graph_units,
				"self_host_edges": result.graph_edges,
			},
			"offline_paths": {
				"hierarchical": str(self.offline_root / "longmemeval_hierarchical" / "step3_semantic_graphs" / sample.sample_id),
				"episodic": str(self.offline_root / "episodic_memory_graphs_new" / sample.sample_id),
				"entity_relation": str(self.offline_root / "entity_relation_graphs_new" / sample.sample_id),
			},
		}
		comparison["strict_l0_match"] = comparison["hierarchical_l0"]["matches"]
		return comparison

	def _write_checkpoint(
		self,
		checkpoint_path: Path,
		sample: LongMemEvalQASample,
		result: SampleBuildResult,
		status: str,
		stages: Dict[str, str],
	) -> None:
		state = {
			"sample_id": sample.sample_id,
			"qa_index": sample.qa_index,
			"status": status,
			"stages": stages,
			"config": self._config_summary(),
			"result": result.to_dict(),
			"updated_at": datetime.now().isoformat(),
		}
		write_json(checkpoint_path, state)

	def _config_summary(self) -> Dict[str, Any]:
		return {
			"strategy": self.args.strategy,
			"extraction_model": self.args.extraction_model,
			"dedup_model": self.args.dedup_model,
			"embedding_model": self.args.embedding_model,
						"dedup_embedding_model": self.args.dedup_embedding_model,
						"extraction_request_timeout": self.args.extraction_request_timeout,
						"dedup_request_timeout": self.args.dedup_request_timeout,
										   "extraction_request_max_retries": self.args.extraction_request_max_retries,
										   "dedup_request_max_retries": self.args.dedup_request_max_retries,
										   "allow_partial_extraction": self.args.allow_partial_extraction,
			"chunk_size": self.args.chunk_size,
			"chunk_overlap": self.args.chunk_overlap,
			"episodic_sessions_per_group": self.args.episodic_sessions_per_group,
			"entity_sessions_per_group": self.args.entity_sessions_per_group,
						"episodic_extraction_max_tokens": self.args.episodic_extraction_max_tokens,
						"entity_extraction_max_tokens": self.args.entity_extraction_max_tokens,
			"episodic_session_workers": self.args.episodic_session_workers,
			"entity_session_workers": self.args.entity_session_workers,
			"entity_dedup_workers": self.args.entity_dedup_workers,
			"build_splade": self.args.build_splade,
			"splade_model": self.args.splade_model,
		}

	@staticmethod
	def _nodes_output(sample: LongMemEvalQASample, nodes: List[Dict[str, Any]], stats: Dict[str, int]) -> Dict[str, Any]:
		return {
			"info": {
				"generated_at": datetime.now().isoformat(),
				"qa_index": sample.qa_index,
				"total_nodes": len(nodes),
				"architecture": "Pure Text + Graph Relations (Zero Redundancy)",
			},
			"qa_metadata": {
				"question_id": sample.data.get("question_id"),
				"question": sample.data.get("question", ""),
				"answer": sample.data.get("answer", ""),
				"question_type": sample.data.get("question_type", "unknown"),
				"total_sessions": len(sample.data.get("haystack_sessions", [])),
			},
			"statistics": stats,
			"nodes": nodes,
		}

	@staticmethod
	def _edges_output(sample: LongMemEvalQASample, edges: List[Dict[str, Any]]) -> Dict[str, Any]:
		return {
			"info": {
				"generated_at": datetime.now().isoformat(),
				"qa_index": sample.qa_index,
				"total_edges": len(edges),
				"edge_types": ["REPLY_TO"],
			},
			"edges": edges,
		}

	@staticmethod
	def _l0_summary(sample: LongMemEvalQASample, stats: Dict[str, int]) -> Dict[str, Any]:
		return {
			"qa_index": sample.qa_index,
			"question_id": sample.data.get("question_id"),
			"question_type": sample.data.get("question_type"),
			"generated_at": datetime.now().isoformat(),
			"statistics": stats,
			"files": {"nodes": "l0_nodes.json", "edges": "l0_edges.json"},
		}

	def _summary(self, results: Sequence[SampleBuildResult]) -> Dict[str, Any]:
		successful = [result for result in results if result.success]
		return {
			"generated_at": datetime.now().isoformat(),
			"dataset_path": str(self.dataset_path),
			"output_dir": str(self.output_dir),
			"config": self._config_summary(),
			"total_samples": len(results),
			"successful_samples": len(successful),
			"failed_samples": len(results) - len(successful),
			"total_l0_units": sum(result.l0_units for result in successful),
			"total_episodic_units": sum(result.episodic_units_added for result in successful),
			"total_entity_mentions": sum(result.entity_mentions_added for result in successful),
			"results": [result.to_dict() for result in results],
		}


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Build LongMemEval self-host SemanticGraph memories.")
	parser.add_argument("--dataset-path", default=str(SCRIPT_DIR / "dataset" / "longmemeval_s_cleaned.json"))
	parser.add_argument("--output-dir", default=str(SCRIPT_DIR / "memory"))
	parser.add_argument("--offline-root", default=str(REPO_ROOT / "benchmark_longmemeval" / "dataset" / "LongMemEval"))
	parser.add_argument("--sample-ids", nargs="+", help="Specific qa_x sample IDs to build")
	parser.add_argument("--start-index", type=int, default=0)
	parser.add_argument("--end-index", type=int)
	parser.add_argument("--limit", type=int)
	parser.add_argument("--skip-existing", action="store_true")
	parser.add_argument("--force", action="store_true")
	parser.add_argument("--no-resume", action="store_true")
	parser.add_argument("--dry-run", action="store_true")

	parser.add_argument("--strategy", default="longmemeval")
	parser.add_argument("--extraction-model")
	parser.add_argument("--dedup-model")
	parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
	parser.add_argument("--embedding-dim", type=int)
	parser.add_argument("--dedup-embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
	parser.add_argument("--faiss-index-type", default="IDMap,Flat")
	parser.add_argument("--max-context-ratio", type=float, default=0.85)
	parser.add_argument("--extraction-request-timeout", type=float, default=600.0)
	parser.add_argument("--dedup-request-timeout", type=float, default=120.0)
	parser.add_argument("--extraction-request-max-retries", type=int, default=5)
	parser.add_argument("--dedup-request-max-retries", type=int, default=5)
	parser.add_argument("--allow-partial-extraction", action="store_true")

	parser.add_argument("--chunk-size", type=int)
	parser.add_argument("--chunk-overlap", type=int)
	parser.add_argument("--batch-size", type=int, default=32)
	parser.add_argument("--max-message-chars", type=int, default=50000)
	parser.add_argument("--episodic-extraction-max-tokens", type=int, default=16384)
	parser.add_argument("--entity-extraction-max-tokens", type=int, default=16384)
	parser.add_argument("--save-prompts", action=argparse.BooleanOptionalAction, default=True)

	parser.add_argument("--hierarchical-session-workers", type=int)
	parser.add_argument("--episodic-session-workers", type=int)
	parser.add_argument("--episodic-dedup-workers", type=int)
	parser.add_argument("--entity-session-workers", type=int)
	parser.add_argument("--entity-dedup-workers", type=int)
	parser.add_argument("--relation-session-workers", type=int)
	parser.add_argument("--episodic-sessions-per-group", type=int)
	parser.add_argument("--entity-sessions-per-group", type=int)

	parser.add_argument("--episodic-dedup-method")
	parser.add_argument("--episodic-dbscan-eps-range", type=float, nargs=2)
	parser.add_argument("--episodic-dbscan-min-samples-range", type=int, nargs=2)
	parser.add_argument("--episodic-default-eps", type=float)
	parser.add_argument("--episodic-default-min-samples", type=int)
	parser.add_argument("--episodic-large-cluster-threshold", type=int)
	parser.add_argument("--entity-dbscan-eps-range", type=float, nargs=2)
	parser.add_argument("--entity-dbscan-min-samples-range", type=int, nargs=2)
	parser.add_argument("--entity-default-eps", type=float)
	parser.add_argument("--entity-default-min-samples", type=int)
	parser.add_argument("--entity-large-cluster-threshold", type=int)
	parser.add_argument("--entity-dedup-by-type", action=argparse.BooleanOptionalAction, default=True)

	parser.add_argument("--skip-hierarchical", action="store_true")
	parser.add_argument("--build-l1", action="store_true")
	parser.add_argument("--build-l2", action="store_true")
	parser.add_argument("--skip-episodic", action="store_true")
	parser.add_argument("--no-episodic-dedup", action="store_true")
	parser.add_argument("--skip-entity-relation", action="store_true")
	parser.add_argument("--no-entity-dedup", action="store_true")
	parser.add_argument("--enable-relation-extraction", action="store_true")

	parser.add_argument("--build-splade", action=argparse.BooleanOptionalAction)
	parser.add_argument("--splade-model", default="naver/splade-v3")
	parser.add_argument("--splade-batch-size", type=int)
	parser.add_argument("--debug", action="store_true")
	return parser


def apply_strategy_defaults(args: argparse.Namespace) -> argparse.Namespace:
	strategy = resolve_strategy(args.strategy)
	defaults = {
		"extraction_model": strategy.extraction_llm_name,
		"dedup_model": strategy.dedup_llm_name,
		"hierarchical_session_workers": strategy.hierarchical_session_workers,
		"episodic_session_workers": strategy.episodic_session_workers,
		"episodic_dedup_workers": strategy.episodic_dedup_workers,
		"entity_session_workers": strategy.entity_session_workers,
		"entity_dedup_workers": strategy.entity_dedup_workers,
		"relation_session_workers": strategy.relation_session_workers,
		"episodic_dedup_method": strategy.episodic_dedup_method,
		"episodic_dbscan_eps_range": list(strategy.episodic_dbscan_eps_range),
		"episodic_dbscan_min_samples_range": list(strategy.episodic_dbscan_min_samples_range),
		"episodic_default_eps": strategy.episodic_default_eps,
		"episodic_default_min_samples": strategy.episodic_default_min_samples,
		"episodic_large_cluster_threshold": strategy.episodic_large_cluster_threshold,
		"entity_dbscan_eps_range": list(strategy.entity_dbscan_eps_range),
		"entity_dbscan_min_samples_range": list(strategy.entity_dbscan_min_samples_range),
		"entity_default_eps": strategy.entity_default_eps,
		"entity_default_min_samples": strategy.entity_default_min_samples,
		"entity_large_cluster_threshold": strategy.entity_large_cluster_threshold,
		"chunk_size": strategy.chunk_size,
		"chunk_overlap": strategy.chunk_overlap,
		"episodic_sessions_per_group": strategy.episodic_sessions_per_group,
		"entity_sessions_per_group": strategy.entity_sessions_per_group,
		"build_splade": strategy.build_splade,
		"splade_batch_size": strategy.splade_batch_size,
	}
	for key, value in defaults.items():
		if getattr(args, key, None) is None:
			setattr(args, key, value)
	return args


def main(argv: Optional[Sequence[str]] = None) -> int:
	parser = build_arg_parser()
	args = apply_strategy_defaults(parser.parse_args(argv))
	configure_logging(args.debug)
	LOGGER.info("LongMemEval build_graph started")
	LOGGER.info("Dataset: %s", Path(args.dataset_path).resolve())
	LOGGER.info("Output : %s", Path(args.output_dir).resolve())
	builder = LongMemEvalGraphBuilder(args)
	results = builder.build_all()
	failures = [result for result in results if not result.success]
	if failures:
		LOGGER.error("Build finished with %d failures", len(failures))
		return 1
	LOGGER.info("Build finished successfully: %d samples", len(results))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
