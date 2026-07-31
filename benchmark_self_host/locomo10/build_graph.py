#!/usr/bin/env python3
"""Build LoCoMo10 per-sample SemanticGraph memories with mandol.auto_builder.

This script replaces the old external multi-step makers with one reproducible
entrypoint for the self-host benchmark. It keeps the old workflow shape:

1. L0: per-session contextual retrieval view for each dialogue message.
2. Hierarchical: per-session L1 extraction, sample-level L2 aggregation plus split L2 retrieval units.
3. Episodic: per-session fact extraction, with optional deduplication.
4. Entity relation: session-level entity extraction, entity deduplication,
   relation extraction, and textual entity/mention/relation units.
5. Save resumable checkpoints and one SemanticGraph per sample under ``memory/<sample_id>/``.

Graph edges are optional because the current goal is to make the textual
structures reproducible first.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed


def _bootstrap_paths() -> Path:
	"""Make the in-tree ``src/mandol`` package importable for direct runs."""
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
	EpisodicFact,
	HierarchicalAutoBuilder,
	HierarchicalBuilderConfig,
)
from mandol.auto_builder.episodic_prompts import EpisodicFactType  # noqa: E402
from mandol.auto_builder.l0_views import extract_embedding_text  # noqa: E402
from mandol.auto_builder.graph_write_queue import GraphWriteRequest, dispatch_graph_write_requests  # noqa: E402
from mandol.auto_builder.strategy_config import STYLE_ALIASES, STYLE_STRATEGIES, PipelineStrategy  # noqa: E402
from mandol.core.memory_unit import MemoryUnit  # noqa: E402
from mandol.core.memory_space_registry import MemorySpaceRegistry, TowerSpace  # noqa: E402
from mandol.core.semantic_graph import SemanticGraph  # noqa: E402
from mandol.core.semantic_map import SemanticMap  # noqa: E402
from mandol.llm.llm_client import LLMClient  # noqa: E402
from mandol.utils.logging_config import auto_configure_logging, setup_logging  # noqa: E402


LOGGER = logging.getLogger("locomo10_build_graph")
CHECKPOINT_FILENAME = "build_checkpoint.json"
COMPLETED = "completed"

HIERARCHICAL_L0_SPACE = TowerSpace.HIERARCHICAL_L0.value
HIERARCHICAL_L1_SPACE = TowerSpace.HIERARCHICAL_L1.value
HIERARCHICAL_L2_SPACE = TowerSpace.HIERARCHICAL_L2.value
EPISODIC_SPACE = TowerSpace.EPISODIC_ROOT.value
GRAPH_ENTITY_SPACE = TowerSpace.GRAPH_ENTITIES.value
GRAPH_MENTION_SPACE = TowerSpace.GRAPH_MENTIONS.value
GRAPH_RELATION_SPACE = TowerSpace.GRAPH_RELATIONS.value

L2_TIMELINE_SPACE = f"{HIERARCHICAL_L2_SPACE}:Timeline"
L2_ENTITY_PROFILE_SPACE = f"{HIERARCHICAL_L2_SPACE}:EntityProfile"
L2_CROSS_INSIGHT_SPACE = f"{HIERARCHICAL_L2_SPACE}:CrossInsight"
L2_GLOBAL_STATS_SPACE = f"{HIERARCHICAL_L2_SPACE}:GlobalStats"
L2_ACTIVITY_LEDGER_SPACE = f"{HIERARCHICAL_L2_SPACE}:ActivityLedger"
L2_NEGATIVE_CONSTRAINT_SPACE = f"{HIERARCHICAL_L2_SPACE}:NegativeConstraint"
L2_SOCIAL_GRAPH_SPACE = f"{HIERARCHICAL_L2_SPACE}:SocialGraph"
L2_SPLIT_SPACES = [
	L2_TIMELINE_SPACE,
	L2_ENTITY_PROFILE_SPACE,
	L2_CROSS_INSIGHT_SPACE,
	L2_GLOBAL_STATS_SPACE,
	L2_ACTIVITY_LEDGER_SPACE,
	L2_NEGATIVE_CONSTRAINT_SPACE,
	L2_SOCIAL_GRAPH_SPACE,
]

MONTHS = {
	"jan": 1,
	"january": 1,
	"feb": 2,
	"february": 2,
	"mar": 3,
	"march": 3,
	"apr": 4,
	"april": 4,
	"may": 5,
	"jun": 6,
	"june": 6,
	"jul": 7,
	"july": 7,
	"aug": 8,
	"august": 8,
	"sep": 9,
	"sept": 9,
	"september": 9,
	"oct": 10,
	"october": 10,
	"nov": 11,
	"november": 11,
	"dec": 12,
	"december": 12,
}


@dataclass
class LocomoSession:
	sample_id: str
	session_id: str
	session_index: int
	raw_datetime: str
	session_date: str
	session_time: str
	units: List[MemoryUnit]


@dataclass
class SampleBuildResult:
	sample_id: str
	output_dir: str
	success: bool
	skipped: bool = False
	resumed: bool = False
	l0_units: int = 0
	l1_units: int = 0
	l2_units: int = 0
	episodic_facts_extracted: int = 0
	episodic_facts_after_dedup: int = 0
	entities_extracted: int = 0
	entities_after_dedup: int = 0
	relations_extracted: int = 0
	entity_mentions_added: int = 0
	relation_units_added: int = 0
	graph_saved: bool = False
	checkpoint_path: str = ""
	processing_seconds: float = 0.0
	errors: List[str] = None

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


def parse_session_datetime(datetime_str: str) -> Tuple[str, str]:
	"""Parse LoCoMo strings like ``1:56 pm on 8 May, 2023``."""
	if not datetime_str:
		return "unknown", ""

	normalized = " ".join(str(datetime_str).replace(",", "").split())
	pattern = re.compile(
		r"(?:(?P<time>\d{1,2}:\d{2}\s*(?:am|pm))\s+on\s+)?"
		r"(?P<day>\d{1,2})\s+(?P<month>[A-Za-z]+)\s+(?P<year>\d{4})",
		re.IGNORECASE,
	)
	match = pattern.search(normalized)
	if not match:
		return normalized or "unknown", ""

	month = MONTHS.get(match.group("month").lower())
	if not month:
		return normalized, (match.group("time") or "").strip().lower()

	day = int(match.group("day"))
	year = int(match.group("year"))
	time_text = (match.group("time") or "").strip().lower()
	return f"{year:04d}-{month:02d}-{day:02d}", time_text


def session_sort_key(session_key: str) -> int:
	match = re.match(r"^session_(\d+)$", session_key)
	return int(match.group(1)) if match else 10**9


def valid_session_keys(conversation: Dict[str, Any]) -> List[str]:
	keys = [
		key
		for key, value in conversation.items()
		if re.match(r"^session_\d+$", key) and isinstance(value, list)
	]
	return sorted(keys, key=session_sort_key)


def normalize_sample_id(sample: Dict[str, Any], index: int) -> str:
	sample_id = sample.get("sample_id") or sample.get("conversation_id")
	return str(sample_id or f"sample_{index:03d}")


def build_message_text(message: Any) -> Tuple[str, str, Dict[str, Any]]:
	"""Return (combined_text, original_text, extra_fields)."""
	if not isinstance(message, dict):
		text = str(message).strip()
		return text, text, {}

	original_text = str(message.get("text") or message.get("message") or "").strip()
	parts = [original_text] if original_text else []
	extra: Dict[str, Any] = {}

	if message.get("blip_caption"):
		caption = str(message["blip_caption"]).strip()
		parts.append(f"Image caption: {caption}")
		extra["blip_caption"] = message["blip_caption"]
	if message.get("query"):
		query = str(message["query"]).strip()
		parts.append(f"Image query: {query}")
		extra["image_query"] = message["query"]
	if message.get("img_url"):
		extra["img_url"] = message["img_url"]
	if message.get("dia_id"):
		extra["original_dialogue_id"] = message["dia_id"]

	return " | ".join(part for part in parts if part), original_text, extra


def participants_from_sample(sample: Dict[str, Any]) -> List[str]:
	conversation = sample.get("conversation") or {}
	participants = [
		conversation.get("speaker_a") or "Speaker_A",
		conversation.get("speaker_b") or "Speaker_B",
	]
	return [str(name) for name in participants if name]


def speakers_text(participants: Sequence[str]) -> str:
	return ", ".join(str(name) for name in participants if name)


def safe_reference_date(sessions: Sequence[LocomoSession]) -> str:
	for session in reversed(sessions):
		if session.session_date and session.session_date != "unknown":
			return session.session_date
	return datetime.now().strftime("%Y-%m-%d")


def make_jsonable(value: Any) -> Any:
	if hasattr(value, "to_dict"):
		return make_jsonable(value.to_dict())
	if hasattr(value, "__dataclass_fields__"):
		return make_jsonable(asdict(value))
	if hasattr(value, "__dict__") and value.__class__.__module__.startswith("mandol"):
		return make_jsonable(value.__dict__)
	if value.__class__.__module__.startswith("mandol"):
		slots = getattr(type(value), "__slots__", ())
		if isinstance(slots, str):
			slots = (slots,)
		return make_jsonable({name: getattr(value, name) for name in slots if hasattr(value, name)})
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


def resolve_strategy(strategy_name: str) -> PipelineStrategy:
	strategy_key = STYLE_ALIASES.get(strategy_name, strategy_name)
	if strategy_key not in STYLE_STRATEGIES:
		raise ValueError(f"Unknown auto_builder strategy: {strategy_name}")
	return STYLE_STRATEGIES[strategy_key]


def read_json(path: Path, default: Any = None) -> Any:
	if not path.exists():
		return default
	with path.open("r", encoding="utf-8") as file:
		return json.load(file)


class LocomoGraphBuilder:
	def __init__(self, args: argparse.Namespace):
		self.args = args
		self.strategy = resolve_strategy(args.strategy)
		self.output_dir = Path(args.output_dir).resolve()
		self.output_dir.mkdir(parents=True, exist_ok=True)
		self.llm_clients = self._create_llm_clients()
		self.extraction_llm = self.llm_clients[args.extraction_model]
		self.dedup_llm = self.llm_clients[args.dedup_model]

	def _create_llm_clients(self) -> Dict[str, LLMClient]:
		clients: Dict[str, LLMClient] = {}
		for model_name in sorted({self.args.extraction_model, self.args.dedup_model}):
			clients[model_name] = LLMClient(
				model_name=model_name,
				max_context_ratio=self.args.max_context_ratio,
				request_max_retries=self.args.request_max_retries,
				request_timeout=self.args.request_timeout,
			)
		return clients

	def load_samples(self) -> List[Dict[str, Any]]:
		dataset_path = Path(self.args.dataset_file).resolve()
		with dataset_path.open("r", encoding="utf-8") as file:
			data = json.load(file)
		if not isinstance(data, list):
			raise ValueError(f"Dataset must be a list: {dataset_path}")

		if self.args.sample_ids:
			wanted = set(self.args.sample_ids)
			data = [sample for idx, sample in enumerate(data, 1) if normalize_sample_id(sample, idx) in wanted]
		if self.args.limit is not None:
			data = data[: self.args.limit]
		return data

	def build_all(self) -> Dict[str, Any]:
		samples = self.load_samples()
		LOGGER.info("Found %d samples", len(samples))

		summary = {
			"dataset_file": str(Path(self.args.dataset_file).resolve()),
			"output_dir": str(self.output_dir),
			"started_at": datetime.now().isoformat(),
			"config": self._summary_config(),
			"samples": [],
		}

		for index, sample in enumerate(samples, 1):
			sample_id = normalize_sample_id(sample, index)
			LOGGER.info("[%d/%d] Building sample %s", index, len(samples), sample_id)
			result = self.build_sample(sample, sample_id)
			summary["samples"].append(result.to_dict())
			write_json(self.output_dir / "build_summary.json", summary)

		summary["finished_at"] = datetime.now().isoformat()
		summary["success_count"] = sum(1 for item in summary["samples"] if item.get("success"))
		summary["skipped_count"] = sum(1 for item in summary["samples"] if item.get("skipped"))
		write_json(self.output_dir / "build_summary.json", summary)
		return summary

	def build_sample(self, sample: Dict[str, Any], sample_id: str) -> SampleBuildResult:
		start_time = datetime.now()
		sample_dir = self.output_dir / sample_id
		checkpoint_path = sample_dir / CHECKPOINT_FILENAME
		errors: List[str] = []

		if sample_dir.exists():
			if self.args.force:
				shutil.rmtree(sample_dir)
			elif self.args.skip_existing:
				LOGGER.info("Sample %s already exists, skipping", sample_id)
				return SampleBuildResult(
					sample_id=sample_id,
					output_dir=str(sample_dir),
					success=True,
					skipped=True,
					checkpoint_path=str(checkpoint_path),
					errors=[],
				)
			elif self.args.no_resume:
				raise FileExistsError(
					f"Output already exists: {sample_dir}. Use --force, --skip-existing, or allow resume."
				)

		sample_dir.mkdir(parents=True, exist_ok=True)
		artifacts_dir = sample_dir / "artifacts"
		checkpoint = self._load_checkpoint(sample_dir, sample_id)
		stage_status = checkpoint.setdefault("stage_status", {})

		result = SampleBuildResult(
			sample_id=sample_id,
			output_dir=str(sample_dir),
			success=False,
			resumed=bool(stage_status),
			checkpoint_path=str(checkpoint_path),
			errors=errors,
		)

		try:
			sessions = self._build_l0_sessions(sample, sample_id)
			participants = participants_from_sample(sample)
			if not sessions:
				raise ValueError(f"Sample {sample_id} has no valid LoCoMo sessions")

			if self.args.dry_run:
				result.success = True
				result.l0_units = sum(len(session.units) for session in sessions)
				result.processing_seconds = (datetime.now() - start_time).total_seconds()
				write_json(sample_dir / "build_result.json", result.to_dict())
				return result

			graph = self._create_or_load_graph(sample_dir)
			self._create_base_spaces(graph)
			if not stage_status:
				self._infer_checkpoint_stages_from_graph(graph, checkpoint)
				result.resumed = bool(checkpoint.get("stage_status"))
			self._update_graph_build_state(graph, checkpoint, status="running", current_stage="start")

			if self._stage_completed(checkpoint, "l0") and self._count_units_in_space(graph, HIERARCHICAL_L0_SPACE):
				LOGGER.info("[%s] Resuming after L0 stage", sample_id)
				sessions = self._rehydrate_sessions_from_graph(graph, sessions)
			else:
				self._enhance_and_store_l0(graph, sessions, participants)
				self._save_l0_artifacts(artifacts_dir, sessions)
				self._mark_stage(graph, sample_dir, checkpoint, "l0", {"units": sum(len(session.units) for session in sessions)})
			all_l0_units = [unit for session in sessions for unit in session.units]
			result.l0_units = len(all_l0_units)

			if self._stage_completed(checkpoint, "hierarchical"):
				LOGGER.info("[%s] Resuming after hierarchical stage", sample_id)
				h_artifacts = self._load_hierarchical_artifacts(artifacts_dir)
				added_l2_splits = self._ensure_l2_split_units_from_artifacts(graph, sample_id, artifacts_dir)
				if added_l2_splits:
					h_artifacts["l2_split_units"] = h_artifacts.get("l2_split_units", 0) + added_l2_splits
					self._save_hierarchical_artifacts(artifacts_dir, h_artifacts)
					self._mark_stage(graph, sample_dir, checkpoint, "hierarchical_l2_split", {"added": added_l2_splits})
			else:
				h_artifacts = self._build_hierarchical(graph, sessions, sample_id, participants)
				self._save_hierarchical_artifacts(artifacts_dir, h_artifacts)
				self._mark_stage(
					graph,
					sample_dir,
					checkpoint,
					"hierarchical",
					{
						"l1_units": len(h_artifacts["l1_results"]),
						"l2_units": self._count_l2_units(graph),
						"l2_split_units": h_artifacts.get("l2_split_units", 0),
					},
				)
			result.l1_units = len(h_artifacts.get("l1_results", [])) or self._count_units_in_spaces(
								graph, [HIERARCHICAL_L1_SPACE]
			)
			result.l2_units = self._count_l2_units(graph)

			if self._stage_completed(checkpoint, "episodic"):
				LOGGER.info("[%s] Resuming after episodic stage", sample_id)
				e_artifacts = self._load_episodic_artifacts(artifacts_dir)
				added_raw_facts = self._ensure_episodic_units_from_artifacts(graph, e_artifacts)
				if added_raw_facts:
					e_artifacts["deduplicated_facts"] = e_artifacts.get("facts", [])
					self._save_episodic_artifacts(artifacts_dir, e_artifacts)
					self._mark_stage(graph, sample_dir, checkpoint, "episodic_raw_restore", {"added": added_raw_facts})
			else:
				e_artifacts = self._build_episodic(graph, sessions, sample_id, participants)
				self._save_episodic_artifacts(artifacts_dir, e_artifacts)
				self._mark_stage(
					graph,
					sample_dir,
					checkpoint,
					"episodic",
					{
						"facts_extracted": len(e_artifacts["facts"]),
						"facts_after_dedup": len(e_artifacts["deduplicated_facts"]),
						"dedup_method": self.args.episodic_dedup_method,
					},
				)
			result.episodic_facts_extracted = len(e_artifacts["facts"])
			result.episodic_facts_after_dedup = len(e_artifacts["deduplicated_facts"])

			if self._stage_completed(checkpoint, "entity_relation"):
				LOGGER.info("[%s] Resuming after entity-relation stage", sample_id)
				er_artifacts = self._load_entity_relation_artifacts(artifacts_dir)
			else:
				er_artifacts = self._build_entity_relation(graph, sessions, sample_id, safe_reference_date(sessions))
				self._save_entity_relation_artifacts(artifacts_dir, er_artifacts)
				self._mark_stage(
					graph,
					sample_dir,
					checkpoint,
					"entity_relation",
					{
						"raw_entities": len(er_artifacts["raw_entities"]),
						"entities": len(er_artifacts["entities"]),
						"relations": len(er_artifacts["relations"]),
						"entity_extraction_scope": self.args.entity_extraction_scope,
						"relation_extraction_scope": self.args.relation_extraction_scope,
					},
				)
			result.entities_extracted = len(er_artifacts["raw_entities"])
			result.entities_after_dedup = len(er_artifacts["entities"])
			result.relations_extracted = len(er_artifacts["relations"])
			result.entity_mentions_added = er_artifacts["mentions_added"]
			result.relation_units_added = er_artifacts["relation_units_added"]

			graph.build_semantic_map_index()
			checkpoint.setdefault("stage_status", {})["graph_saved"] = COMPLETED
			checkpoint.setdefault("stage_details", {})["graph_saved"] = {
				"freeze_retrievers": not self.args.no_freeze_retrievers,
				"force_rebuild_retrievers": True,
			}
			self._update_graph_build_state(graph, checkpoint, status="completed", current_stage="done")
			write_json(sample_dir / CHECKPOINT_FILENAME, checkpoint)
			graph.save_graph(
				str(sample_dir),
				freeze_retrievers=not self.args.no_freeze_retrievers,
				force_rebuild_retrievers=True,
			)
			result.graph_saved = True
			result.success = True

		except Exception as exc:
			message = f"{type(exc).__name__}: {exc}"
			errors.append(message)
			LOGGER.error("Sample %s failed: %s", sample_id, message)
			LOGGER.debug(traceback.format_exc())

		result.processing_seconds = (datetime.now() - start_time).total_seconds()
		write_json(sample_dir / "build_result.json", result.to_dict())
		return result

	def _summary_config(self) -> Dict[str, Any]:
		episodic_dedup_enabled = not self.args.no_episodic_dedup and self.args.episodic_dedup_method != "none"
		return {
			"strategy": self.args.strategy,
			"extraction_model": self.args.extraction_model,
			"dedup_model": self.args.dedup_model,
			"embedding_model": self.args.embedding_model,
			"dedup_embedding_model": self.args.dedup_embedding_model,
			"contextual_workers": self.args.contextual_workers,
			"hierarchical_session_workers": self.args.hierarchical_session_workers,
			"episodic_session_workers": self.args.episodic_session_workers,
			"episodic_dedup_workers": self.args.episodic_dedup_workers,
			"entity_session_workers": self.args.entity_session_workers,
			"entity_dedup_workers": self.args.entity_dedup_workers,
			"relation_session_workers": self.args.relation_session_workers,
			"contextual_retrieval": not self.args.no_contextual_retrieval,
			"episodic_deduplication": episodic_dedup_enabled,
			"episodic_dedup_method": self.args.episodic_dedup_method,
			"entity_deduplication": not self.args.no_entity_dedup,
			"entity_extraction_scope": self.args.entity_extraction_scope,
			"relation_extraction": not self.args.no_relation_extraction,
			"relation_extraction_scope": self.args.relation_extraction_scope,
			"graph_edges": self.args.enable_graph_edges,
			"freeze_retrievers": not self.args.no_freeze_retrievers,
			"resume": not self.args.no_resume,
			"stage_graph_checkpoints": not self.args.no_stage_graph_checkpoints,
		}

	def _load_checkpoint(self, sample_dir: Path, sample_id: str) -> Dict[str, Any]:
		checkpoint = read_json(sample_dir / CHECKPOINT_FILENAME, default={}) or {}
		if not checkpoint:
			graph_state = read_json(sample_dir / "graph_state.json", default={}) or {}
			checkpoint = graph_state.get("high_level_memory_build", {}) or {}
		checkpoint.setdefault("sample_id", sample_id)
		checkpoint.setdefault("build_method", "benchmark_self_host.locomo10.build_graph")
		checkpoint.setdefault("config", self._summary_config())
		checkpoint.setdefault("stage_status", {})
		checkpoint.setdefault("stage_details", {})
		checkpoint.setdefault("created_at", datetime.now().isoformat())
		return checkpoint

	@staticmethod
	def _stage_completed(checkpoint: Dict[str, Any], stage_name: str) -> bool:
		return checkpoint.get("stage_status", {}).get(stage_name) == COMPLETED

	def _create_or_load_graph(self, sample_dir: Path) -> SemanticGraph:
		graph_state = sample_dir / "graph_state.json"
		if not self.args.no_resume and graph_state.exists():
			try:
				LOGGER.info("Loading existing SemanticGraph checkpoint: %s", sample_dir)
				return SemanticGraph.load_graph(str(sample_dir))
			except Exception as exc:
				LOGGER.warning("Could not load graph checkpoint %s, rebuilding in memory: %s", sample_dir, exc)
		return self._create_graph()

	def _update_graph_build_state(
		self,
		graph: SemanticGraph,
		checkpoint: Dict[str, Any],
		status: str,
		current_stage: str,
	) -> None:
		checkpoint["status"] = status
		checkpoint["current_stage"] = current_stage
		checkpoint["updated_at"] = datetime.now().isoformat()
		checkpoint["config"] = self._summary_config()
		graph.set_high_level_memory_build_state(checkpoint)

	def _mark_stage(
		self,
		graph: SemanticGraph,
		sample_dir: Path,
		checkpoint: Dict[str, Any],
		stage_name: str,
		details: Optional[Dict[str, Any]] = None,
		save_graph: bool = True,
	) -> None:
		checkpoint.setdefault("stage_status", {})[stage_name] = COMPLETED
		checkpoint.setdefault("stage_details", {})[stage_name] = details or {}
		self._update_graph_build_state(graph, checkpoint, status="running", current_stage=stage_name)
		write_json(sample_dir / CHECKPOINT_FILENAME, checkpoint)
		if save_graph and not self.args.no_stage_graph_checkpoints:
			try:
				graph.save_graph(str(sample_dir), freeze_retrievers=False)
			except Exception as exc:
				LOGGER.warning("Could not save stage graph checkpoint after %s: %s", stage_name, exc)

	def _space_unit_uids(self, graph: SemanticGraph, space_name: str) -> List[str]:
		space = graph.semantic_map.memory_spaces.get(space_name)
		if not space:
			return []
		return list(getattr(space, "_unit_uids", []))

	def _count_units_in_space(self, graph: SemanticGraph, space_name: str) -> int:
		return len(self._space_unit_uids(graph, space_name))

	def _count_units_in_spaces(self, graph: SemanticGraph, space_names: Sequence[str]) -> int:
		seen = set()
		for space_name in space_names:
			seen.update(self._space_unit_uids(graph, space_name))
		return len(seen)

	def _count_l2_units(self, graph: SemanticGraph) -> int:
		return self._count_units_in_spaces(
			graph,
			[
				HIERARCHICAL_L2_SPACE,
				*L2_SPLIT_SPACES,
			],
		)

	def _infer_checkpoint_stages_from_graph(self, graph: SemanticGraph, checkpoint: Dict[str, Any]) -> None:
		stage_status = checkpoint.setdefault("stage_status", {})
		stage_details = checkpoint.setdefault("stage_details", {})
		l0_count = self._count_units_in_space(graph, HIERARCHICAL_L0_SPACE)
		if l0_count:
			stage_status["l0"] = COMPLETED
			stage_details.setdefault("l0", {"units": l0_count, "inferred": True})
		l1_count = self._count_units_in_spaces(graph, [HIERARCHICAL_L1_SPACE])
		l2_count = self._count_l2_units(graph)
		if l1_count or l2_count:
			stage_status["hierarchical"] = COMPLETED
			stage_details.setdefault("hierarchical", {"l1_units": l1_count, "l2_units": l2_count, "inferred": True})
		episodic_count = self._count_units_in_space(graph, EPISODIC_SPACE)
		if episodic_count:
			stage_status["episodic"] = COMPLETED
			stage_details.setdefault("episodic", {"facts_after_dedup": episodic_count, "inferred": True})
		entity_count = self._count_units_in_spaces(
			graph,
			[GRAPH_ENTITY_SPACE, GRAPH_MENTION_SPACE, GRAPH_RELATION_SPACE],
		)
		if entity_count:
			stage_status["entity_relation"] = COMPLETED
			stage_details.setdefault("entity_relation", {"units": entity_count, "inferred": True})

	def _create_graph(self) -> SemanticGraph:
		semantic_map = SemanticMap(
			embedding_model_name=self.args.embedding_model,
			faiss_index_type=self.args.faiss_index_type,
		)
		return SemanticGraph(semantic_map_instance=semantic_map)

	def _create_base_spaces(self, graph: SemanticGraph) -> None:
		MemorySpaceRegistry.initialize_spaces(graph)
		spaces = L2_SPLIT_SPACES
		for space_name in spaces:
			graph.create_memory_space_in_map(space_name)
		for child in L2_SPLIT_SPACES:
			try:
				graph.semantic_map.add_space_to_space(child, HIERARCHICAL_L2_SPACE)
			except Exception:
				pass

	def _build_l0_sessions(self, sample: Dict[str, Any], sample_id: str) -> List[LocomoSession]:
		conversation = sample.get("conversation") or {}
		participants = participants_from_sample(sample)
		sessions: List[LocomoSession] = []

		for session_index, session_id in enumerate(valid_session_keys(conversation), 1):
			messages = conversation.get(session_id) or []
			raw_datetime = str(conversation.get(f"{session_id}_date_time") or "")
			session_date, session_time = parse_session_datetime(raw_datetime)
			units: List[MemoryUnit] = []

			for message_index, message in enumerate(messages):
				message_text, original_text, extra_fields = build_message_text(message)
				if len(message_text.strip()) < self.args.min_content_length:
					continue

				if isinstance(message, dict):
					speaker = str(message.get("speaker") or "Unknown")
					dialogue_id = str(message.get("dia_id") or f"D{session_index}:{message_index + 1}")
				else:
					speaker = participants[message_index % len(participants)] if participants else "Unknown"
					dialogue_id = f"D{session_index}:{message_index + 1}"

				uid = f"L0_{sample_id}_{session_id}_msg_{message_index}"
				raw_data = {
					"type": "conversation_message",
					"text_content": message_text,
					"original_content": original_text or message_text,
					"message": message_text,
					"speaker": speaker,
					"sample_id": sample_id,
					"session_id": session_id,
					"session_date": session_date,
					"session_time": session_time,
					"session_datetime": raw_datetime,
					"dialogue_id": dialogue_id,
					"participants": participants,
					"has_multimodal": bool(extra_fields.get("img_url") or extra_fields.get("blip_caption")),
					**extra_fields,
				}
				metadata = {
					"layer": "L0",
					"memory_level": "L0_Observation",
					"sample_id": sample_id,
					"session_id": session_id,
					"session_index": session_index,
					"message_index": message_index,
					"speaker": speaker,
					"session_date": session_date,
					"session_time": session_time,
					"dialogue_id": dialogue_id,
					"retrieval_view": "text_content",
					"source": "benchmark_self_host.locomo10.build_graph",
				}
				units.append(MemoryUnit(uid=uid, raw_data=raw_data, metadata=metadata))

			sessions.append(
				LocomoSession(
					sample_id=sample_id,
					session_id=session_id,
					session_index=session_index,
					raw_datetime=raw_datetime,
					session_date=session_date,
					session_time=session_time,
					units=units,
				)
			)

		return sessions

	def _rehydrate_sessions_from_graph(
		self,
		graph: SemanticGraph,
		fallback_sessions: Sequence[LocomoSession],
	) -> List[LocomoSession]:
		units_by_uid = graph.semantic_map.memory_units
		rehydrated: List[LocomoSession] = []
		for session in fallback_sessions:
			units = [units_by_uid[unit.uid] for unit in session.units if unit.uid in units_by_uid]
			if not units:
				units = [
					unit
					for unit in units_by_uid.values()
					if (unit.metadata or {}).get("session_id") == session.session_id
				]
			units = sorted(units, key=lambda unit: (unit.metadata or {}).get("message_index", 0))
			rehydrated.append(
				LocomoSession(
					sample_id=session.sample_id,
					session_id=session.session_id,
					session_index=session.session_index,
					raw_datetime=session.raw_datetime,
					session_date=session.session_date,
					session_time=session.session_time,
					units=units,
				)
			)
		return rehydrated

	def _enhance_and_store_l0(
		self,
		graph: SemanticGraph,
		sessions: Sequence[LocomoSession],
		participants: Sequence[str],
	) -> None:
		cr_builder = HierarchicalAutoBuilder(
			semantic_system=graph,
			llm_client=self.extraction_llm,
			config=HierarchicalBuilderConfig(
				extraction_style="locomo",
				enable_contextual_retrieval=not self.args.no_contextual_retrieval,
				contextual_parallel_workers=self.args.contextual_workers,
				llm_max_retries=1,
			),
		)

		enhanced_by_session: Dict[str, Dict[str, str]] = {}
		if not self.args.no_contextual_retrieval:
			unit_tasks = [
				(session, unit)
				for session in sessions
				for unit in session.units
				if unit.raw_data.get("text_content", "").strip()
			]
			if unit_tasks:
				max_workers = min(max(1, self.args.contextual_workers), len(unit_tasks))
				transcripts = {
					session.session_id: cr_builder._build_full_transcript(session.units)
					for session in sessions
					if session.units
				}
				LOGGER.info(
					"Starting flattened L0 Contextual Retrieval: %d units across %d sessions, workers=%d",
					len(unit_tasks),
					len(transcripts),
					max_workers,
				)

				def enhance_unit(session: LocomoSession, unit: MemoryUnit) -> Tuple[str, str, Optional[str]]:
					original_content = unit.raw_data.get("text_content", "")
					if not original_content or len(original_content.strip()) < 10:
						return session.session_id, unit.uid, None
					speaker = unit.raw_data.get("speaker", unit.metadata.get("speaker", "Unknown"))
					session_date = session.session_date if session.session_date != "unknown" else None
					prompt = cr_builder.prompt_manager.get_contextual_retrieval_prompt(
						session_date=session_date or datetime.now().strftime("%Y-%m-%d"),
						participants=list(participants) if participants else ["Speaker_A", "Speaker_B"],
						full_session_transcript=transcripts.get(session.session_id, ""),
						speaker=speaker,
						message_text=original_content,
					)
					enhanced_content = cr_builder._call_llm_with_retry(
						prompt=prompt,
						temperature=0.1,
						max_tokens=300,
						context_id=f"enhance_{unit.uid}",
					)
					return session.session_id, unit.uid, enhanced_content.strip() if enhanced_content else None

				with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="LocomoL0CR") as executor:
					future_to_unit = {
						executor.submit(enhance_unit, session, unit): unit.uid
						for session, unit in unit_tasks
					}
					for future in as_completed(future_to_unit):
						try:
							session_id, unit_uid, enhanced_content = future.result()
						except Exception:
							LOGGER.exception("L0 contextual retrieval task failed for unit %s", future_to_unit[future])
							continue
						if enhanced_content:
							enhanced_by_session.setdefault(session_id, {})[unit_uid] = enhanced_content
				LOGGER.info(
					"Flattened L0 Contextual Retrieval completed: %d/%d units enhanced",
					sum(len(items) for items in enhanced_by_session.values()),
					len(unit_tasks),
				)

		write_requests: List[GraphWriteRequest] = []

		for session in sessions:
			enhanced_by_uid = enhanced_by_session.get(session.session_id, {})

			for unit in session.units:
				enhanced_text = enhanced_by_uid.get(unit.uid)
				if enhanced_text:
					unit.raw_data["enhanced_content"] = enhanced_text
					unit.metadata["preprocessing"] = "contextual_retrieval"
					unit.metadata["retrieval_view"] = "enhanced_content"
				else:
					unit.metadata["preprocessing"] = "locomo_passthrough"

				embedding_text = extract_embedding_text(unit)
				write_requests.append(GraphWriteRequest(
					unit=unit,
					explicit_content_for_embedding=embedding_text or None,
					content_type_for_embedding="text" if embedding_text else None,
										space_names=[HIERARCHICAL_L0_SPACE],
					index_update_mode="none",
					generate_sparse_embedding=False,
					source="locomo10_l0",
				))

		dispatch_graph_write_requests(graph, write_requests)

	def _build_hierarchical(
		self,
		graph: SemanticGraph,
		sessions: Sequence[LocomoSession],
		sample_id: str,
		participants: Sequence[str],
	) -> Dict[str, Any]:
		builder = HierarchicalAutoBuilder(
			semantic_system=graph,
			llm_client=self.extraction_llm,
			config=HierarchicalBuilderConfig(
				extraction_style="locomo",
				enable_contextual_retrieval=False,
				enable_deduplication=self.args.enable_hierarchical_dedup,
				parallel_workers=self.args.hierarchical_session_workers,
				l2_max_tokens=12000,
				llm_max_retries=1,
			),
		)

		def extract_session_l1(session: LocomoSession) -> Tuple[int, List[Any]]:
			if not session.units:
				return session.session_index, []
			return session.session_index, builder.extract_l1_from_l0_units(
				l0_units=session.units,
				session_id=session.session_id,
				session_date=session.session_date if session.session_date != "unknown" else None,
				participants=list(participants),
			)

		l1_by_session: List[Tuple[int, List[Any]]] = []
		session_tasks = [session for session in sessions if session.units]
		max_workers = min(max(1, self.args.hierarchical_session_workers), len(session_tasks) or 1)
		if max_workers > 1:
			with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="LocomoL1") as executor:
				future_to_session = {executor.submit(extract_session_l1, session): session.session_id for session in session_tasks}
				for future in as_completed(future_to_session):
					l1_by_session.append(future.result())
		else:
			for session in session_tasks:
				l1_by_session.append(extract_session_l1(session))
		l1_results = [result for _, results in sorted(l1_by_session, key=lambda item: item[0]) for result in results]

		if self.args.enable_hierarchical_dedup and l1_results:
			builder.llm_client = self.dedup_llm
			l1_results = builder.deduplicate_l1(l1_results)

		builder.llm_client = self.extraction_llm
		l2_result = builder.aggregate_l2_from_l1(
			l1_results=l1_results,
			sample_id=sample_id,
			participants=list(participants),
		)
		add_stats = builder.add_to_semantic_system(
			l1_results=l1_results,
			l2_result=l2_result,
			rebuild_index=False,
		)
		l2_split_units = self._add_l2_structured_units(graph, sample_id, l2_result)
		return {
			"l1_results": l1_results,
			"l2_result": l2_result,
			"add_stats": add_stats,
			"l2_split_units": l2_split_units,
		}

	def _add_l2_structured_units(self, graph: SemanticGraph, sample_id: str, l2_result: Any) -> int:
		if not l2_result:
			return 0
		structured = self._parse_l2_content(l2_result)
		if not structured:
			return 0
		added = 0
		source_uid = getattr(l2_result, "unit_uid", None)

		for index, item in enumerate(structured.get("master_timeline") or []):
			date_text = str(item.get("date") or "unknown")
			events = item.get("events") if isinstance(item.get("events"), list) else []
			if not events:
				events = [{"event": item.get("event") or item}]
			for event_index, event in enumerate(events):
				event_text = event.get("event") if isinstance(event, dict) else str(event)
				participants = event.get("participants", []) if isinstance(event, dict) else []
				source_session = event.get("source_session") if isinstance(event, dict) else None
				content = f"Timeline event on {date_text}: {event_text}"
				if participants:
					content += f" | Participants: {', '.join(map(str, participants))}"
				if source_session:
					content += f" | Source session: {source_session}"
				added += self._add_l2_unit(
					graph,
					uid=f"{sample_id}_L2_timeline_{index}_{event_index}",
					content=content,
										space_names=[L2_TIMELINE_SPACE],
					metadata={"l2_type": "timeline", "date": date_text, "source_l2_uid": source_uid},
				)

		for index, item in enumerate(structured.get("character_status_snapshot") or []):
			person = item.get("person") if isinstance(item, dict) else f"person_{index}"
			content = f"Character profile for {person}: {json.dumps(item, ensure_ascii=False)}"
			added += self._add_l2_unit(
				graph,
				uid=f"{sample_id}_L2_profile_{index}",
				content=content,
								space_names=[L2_ENTITY_PROFILE_SPACE],
				metadata={"l2_type": "entity_profile", "person": person, "source_l2_uid": source_uid},
			)

		for index, insight in enumerate(structured.get("cross_session_insights") or []):
			content = f"Cross-session insight: {insight}"
			added += self._add_l2_unit(
				graph,
				uid=f"{sample_id}_L2_cross_insight_{index}",
				content=content,
								space_names=[L2_CROSS_INSIGHT_SPACE],
				metadata={"l2_type": "cross_session_insight", "source_l2_uid": source_uid},
			)

		for index, edge in enumerate((structured.get("relationship_graph") or {}).get("edges") or []):
			content = f"Social graph edge: {json.dumps(edge, ensure_ascii=False)}"
			added += self._add_l2_unit(
				graph,
				uid=f"{sample_id}_L2_social_graph_{index}",
				content=content,
								space_names=[L2_SOCIAL_GRAPH_SPACE],
				metadata={"l2_type": "social_graph", "source_l2_uid": source_uid},
			)

		for index, topic in enumerate(structured.get("recurring_topics") or []):
			content = f"Recurring topic/activity: {json.dumps(topic, ensure_ascii=False)}"
			added += self._add_l2_unit(
				graph,
				uid=f"{sample_id}_L2_recurring_topic_{index}",
				content=content,
								space_names=[L2_ACTIVITY_LEDGER_SPACE],
				metadata={"l2_type": "recurring_topic", "source_l2_uid": source_uid},
			)

		for index, item in enumerate(structured.get("temporal_analysis") or []):
			content = f"Temporal analysis: {item}"
			added += self._add_l2_unit(
				graph,
				uid=f"{sample_id}_L2_temporal_analysis_{index}",
				content=content,
								space_names=[L2_TIMELINE_SPACE],
				metadata={"l2_type": "temporal_analysis", "source_l2_uid": source_uid},
			)

		if structured.get("global_statistics"):
			content = f"Global conversation statistics: {json.dumps(structured['global_statistics'], ensure_ascii=False)}"
			added += self._add_l2_unit(
				graph,
				uid=f"{sample_id}_L2_global_statistics",
				content=content,
								space_names=[L2_GLOBAL_STATS_SPACE],
				metadata={"l2_type": "global_statistics", "source_l2_uid": source_uid},
			)

		return added

	def _ensure_l2_split_units_from_artifacts(self, graph: SemanticGraph, sample_id: str, artifacts_dir: Path) -> int:
		if self._count_units_in_spaces(
			graph,
			[
				L2_TIMELINE_SPACE,
				L2_ENTITY_PROFILE_SPACE,
				L2_CROSS_INSIGHT_SPACE,
				L2_GLOBAL_STATS_SPACE,
				L2_SOCIAL_GRAPH_SPACE,
				L2_ACTIVITY_LEDGER_SPACE,
			],
		):
			return 0
		l2_result = read_json(artifacts_dir / "l2_result.json", default=None)
		if not l2_result:
			return 0
		LOGGER.info("[%s] Restoring split L2 units from existing l2_result artifact", sample_id)
		return self._add_l2_structured_units(graph, sample_id, l2_result)

	@staticmethod
	def _parse_l2_content(l2_result: Any) -> Dict[str, Any]:
		if isinstance(l2_result, dict):
			content = l2_result.get("content")
		else:
			content = getattr(l2_result, "content", None)
		if isinstance(content, dict):
			return content
		if not isinstance(content, str) or not content.strip():
			return {}
		try:
			return json.loads(content)
		except json.JSONDecodeError as exc:
			try:
				from json_repair import repair_json

				return json.loads(repair_json(content))
			except Exception as repair_exc:
				raise ValueError(
					"L2 aggregation returned malformed JSON; refusing to silently skip structured L2 units"
				) from repair_exc
		except Exception as exc:
			raise ValueError("Failed to parse L2 aggregation content") from exc

	@staticmethod
	def _add_l2_unit(
		graph: SemanticGraph,
		uid: str,
		content: str,
		space_names: Sequence[str],
		metadata: Dict[str, Any],
	) -> int:
		if uid in graph.semantic_map.memory_units:
			return 0
		unit = MemoryUnit(
			uid=uid,
			raw_data={"text_content": content, "node_type": "l2_structured_unit"},
			metadata={
				"layer": "L2",
				"memory_level": "L2_Structured",
				"created_at": datetime.now().isoformat(),
				**metadata,
			},
		)
		dispatch_graph_write_requests(graph, [GraphWriteRequest(
			unit=unit,
			explicit_content_for_embedding=content,
			content_type_for_embedding="text",
			space_names=list(space_names),
			index_update_mode="none",
			generate_sparse_embedding=False,
			source="locomo10_structured_unit",
		)])
		return 1

	def _build_episodic(
		self,
		graph: SemanticGraph,
		sessions: Sequence[LocomoSession],
		sample_id: str,
		participants: Sequence[str],
	) -> Dict[str, Any]:
		enable_dedup = not self.args.no_episodic_dedup and self.args.episodic_dedup_method != "none"
		builder = EpisodicAutoBuilder(
			semantic_system=graph,
			llm_client=self.extraction_llm,
			config=EpisodicBuilderConfig(
				extraction_style="locomo",
				fact_types=EpisodicFactType.locomo_types(),
				enable_deduplication=enable_dedup,
				dedup_method=self.args.episodic_dedup_method,
				dbscan_eps=self.args.episodic_default_eps,
				dbscan_min_samples=self.args.episodic_default_min_samples,
				dbscan_eps_range=tuple(self.args.episodic_dbscan_eps_range),
				dbscan_min_samples_range=tuple(self.args.episodic_dbscan_min_samples_range),
				dedup_parallel_workers=self.args.episodic_dedup_workers,
				large_cluster_threshold=self.args.episodic_large_cluster_threshold,
				embedding_model=self.args.dedup_embedding_model,
			),
		)

		def extract_session_facts(session: LocomoSession) -> Tuple[int, List[EpisodicFact]]:
			if not session.units:
				return session.session_index, []
			return session.session_index, builder.extract_from_l0_units(
				l0_unit_uids=[unit.uid for unit in session.units],
				reference_date=session.session_date if session.session_date != "unknown" else safe_reference_date(sessions),
				source_id=f"{sample_id}_{session.session_id}",
				speakers=speakers_text(participants),
			)

		facts_by_session: List[Tuple[int, List[EpisodicFact]]] = []
		session_tasks = [session for session in sessions if session.units]
		max_workers = min(max(1, self.args.episodic_session_workers), len(session_tasks) or 1)
		builder.llm_client = self.extraction_llm
		if max_workers > 1:
			with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="LocomoEpisodic") as executor:
				future_to_session = {executor.submit(extract_session_facts, session): session.session_id for session in session_tasks}
				for future in as_completed(future_to_session):
					facts_by_session.append(future.result())
		else:
			for session in session_tasks:
				facts_by_session.append(extract_session_facts(session))
		facts = [fact for _, session_facts in sorted(facts_by_session, key=lambda item: item[0]) for fact in session_facts]

		if enable_dedup and facts:
			builder.llm_client = self.dedup_llm
			deduplicated = builder.deduplicate_facts(facts)
		else:
			deduplicated = facts

		added_uids = builder.add_to_semantic_system(deduplicated, space_name=EPISODIC_SPACE)
		return {"facts": facts, "deduplicated_facts": deduplicated, "added_uids": added_uids}

	def _ensure_episodic_units_from_artifacts(self, graph: SemanticGraph, episodic_artifacts: Dict[str, Any]) -> int:
		if self.args.episodic_dedup_method != "none":
			return 0
		facts_data = episodic_artifacts.get("facts") or []
		if not facts_data:
			return 0
		missing_facts = []
		for item in facts_data:
			if not isinstance(item, dict):
				continue
			fact_id = item.get("fact_id")
			if fact_id and fact_id not in graph.semantic_map.memory_units:
				missing_facts.append(EpisodicFact.from_dict(item))
		if not missing_facts:
			return 0
		LOGGER.info("Restoring %d raw episodic facts from artifacts", len(missing_facts))
		builder = EpisodicAutoBuilder(
			semantic_system=graph,
			llm_client=self.extraction_llm,
			config=EpisodicBuilderConfig(
				extraction_style="locomo",
				fact_types=EpisodicFactType.locomo_types(),
				enable_deduplication=False,
				embedding_model=self.args.dedup_embedding_model,
			),
		)
		added_uids = builder.add_to_semantic_system(missing_facts, space_name=EPISODIC_SPACE)
		return len(added_uids)

	def _build_entity_relation(
		self,
		graph: SemanticGraph,
		sessions: Sequence[LocomoSession],
		sample_id: str,
		reference_date: str,
	) -> Dict[str, Any]:
		builder = EntityRelationAutoBuilder(
			semantic_system=graph,
			llm_client=self.extraction_llm,
			config=EntityRelationBuilderConfig(
				extraction_style="locomo",
				embedding_model=self.args.dedup_embedding_model,
				dbscan_eps=self.args.entity_default_eps,
				dbscan_min_samples=self.args.entity_default_min_samples,
				auto_optimize_dbscan=True,
				dbscan_eps_range=tuple(self.args.entity_dbscan_eps_range),
				dbscan_min_samples_range=tuple(self.args.entity_dbscan_min_samples_range),
				parallel_workers=self.args.entity_dedup_workers,
				large_cluster_threshold=self.args.entity_large_cluster_threshold,
				enable_relation_extraction=not self.args.no_relation_extraction,
				enable_llm_deduplication=not self.args.no_entity_dedup,
			),
		)

		all_l0_units = [unit for session in sessions for unit in session.units]
		raw_entities = []
		if self.args.entity_extraction_scope == "session":
			def extract_session_entities(session: LocomoSession) -> Tuple[int, List[Dict[str, Any]]]:
				if not session.units:
					return session.session_index, []
				return session.session_index, builder.extract_entities_from_l0_units(
					l0_units=list(session.units),
					reference_date=session.session_date if session.session_date != "unknown" else reference_date,
					source_id=f"{sample_id}_{session.session_id}",
					session_type="chat",
				)

			entities_by_session: List[Tuple[int, List[Dict[str, Any]]]] = []
			session_tasks = [session for session in sessions if session.units]
			max_workers = min(max(1, self.args.entity_session_workers), len(session_tasks) or 1)
			if max_workers > 1:
				with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="LocomoEntity") as executor:
					future_to_session = {executor.submit(extract_session_entities, session): session.session_id for session in session_tasks}
					for future in as_completed(future_to_session):
						entities_by_session.append(future.result())
			else:
				for session in session_tasks:
					entities_by_session.append(extract_session_entities(session))
			raw_entities = [entity for _, session_entities in sorted(entities_by_session, key=lambda item: item[0]) for entity in session_entities]
		else:
			raw_entities = builder.extract_entities_from_l0_units(
				l0_units=list(all_l0_units),
				reference_date=reference_date,
				source_id=sample_id,
				session_type="chat",
			)

		if raw_entities and not self.args.no_entity_dedup:
			builder.llm_client = self.dedup_llm
			entities = builder.deduplicate_entities(raw_entities)
		else:
			entities = [builder._convert_raw_to_extracted_entity(entity, f"entity_{idx}") for idx, entity in enumerate(raw_entities)]

		relations = []
		if entities and not self.args.no_relation_extraction:
			builder.llm_client = self.extraction_llm
			if self.args.relation_extraction_scope == "session":
				def extract_session_relations(session: LocomoSession) -> Tuple[int, List[Any]]:
					if not session.units:
						return session.session_index, []
					session_entities = self._entities_for_session(entities, sample_id, session.session_id)
					if len(session_entities) < 2:
						session_entities = entities
					session_relations = builder.extract_relations_from_entities(
						l0_units=list(session.units),
						entities=session_entities,
						session_type="chat",
					)
					for relation_index, relation in enumerate(session_relations):
						relation.relation_id = f"{session.session_id}_{relation_index}_{relation.relation_id}"
						relation.session_id = f"{sample_id}_{session.session_id}"
					return session.session_index, session_relations

				relations_by_session: List[Tuple[int, List[Any]]] = []
				session_tasks = [session for session in sessions if session.units]
				max_workers = min(max(1, self.args.relation_session_workers), len(session_tasks) or 1)
				if max_workers > 1:
					with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="LocomoRelation") as executor:
						future_to_session = {executor.submit(extract_session_relations, session): session.session_id for session in session_tasks}
						for future in as_completed(future_to_session):
							relations_by_session.append(future.result())
				else:
					for session in session_tasks:
						relations_by_session.append(extract_session_relations(session))
				relations = [relation for _, session_relations in sorted(relations_by_session, key=lambda item: item[0]) for relation in session_relations]
			else:
				relations = builder.extract_relations_from_entities(
					l0_units=list(all_l0_units),
					entities=entities,
					session_type="chat",
				)

		entity_units_added = self._add_entity_units(graph, sample_id, entities)
		mentions_added = self._add_entity_mention_units(graph, sample_id, entities)
		relation_units_added = self._add_relation_units(graph, sample_id, entities, relations)

		if self.args.enable_graph_edges and relations:
			self._add_relation_edges(graph, sample_id, relations)

		return {
			"raw_entities": raw_entities,
			"entities": entities,
			"relations": relations,
			"entity_units_added": entity_units_added,
			"mentions_added": mentions_added,
			"relation_units_added": relation_units_added,
			"entity_extraction_scope": self.args.entity_extraction_scope,
			"relation_extraction_scope": self.args.relation_extraction_scope,
			"builder_stats": builder.get_stats(),
		}

	@staticmethod
	def _entities_for_session(entities: Sequence[Any], sample_id: str, session_id: str) -> List[Any]:
		wanted = {session_id, f"{sample_id}_{session_id}"}
		matched = []
		for entity in entities:
			entity_session_ids = set(getattr(entity, "session_ids", []) or [])
			if entity_session_ids & wanted:
				matched.append(entity)
		return matched

	def _add_entity_units(self, graph: SemanticGraph, sample_id: str, entities: Sequence[Any]) -> int:
		added = 0
		write_requests: List[GraphWriteRequest] = []
		for entity in entities:
			unit = MemoryUnit(
				uid=f"{sample_id}_{entity.entity_id}",
				raw_data={
					"node_type": "entity",
					"text_content": entity.name,
					"entity_id": entity.entity_id,
					"name": entity.name,
					"entity_type": entity.entity_type,
					"aliases": entity.aliases,
					"mentions": [mention.to_dict() for mention in entity.mentions],
				},
				metadata={
					"type": "entity",
					"source_id": sample_id,
					"entity_type": entity.entity_type,
					"confidence": entity.confidence,
					"session_ids": entity.session_ids,
					"mentions_count": len(entity.mentions),
					"created_at": datetime.now().isoformat(),
				},
			)
			write_requests.append(GraphWriteRequest(
				unit=unit,
				explicit_content_for_embedding=entity.name,
				content_type_for_embedding="text",
								space_names=[GRAPH_ENTITY_SPACE],
				index_update_mode="none",
				generate_sparse_embedding=False,
				source="locomo10_entity",
			))
			added += 1
		dispatch_graph_write_requests(graph, write_requests)
		return added

	def _add_entity_mention_units(self, graph: SemanticGraph, sample_id: str, entities: Sequence[Any]) -> int:
		added = 0
		write_requests: List[GraphWriteRequest] = []
		for entity in entities:
			for index, mention in enumerate(entity.mentions):
				mention_text = self._build_mention_text(entity, mention)
				unit = MemoryUnit(
					uid=f"{sample_id}_{entity.entity_id}_mention_{index}",
					raw_data={
						"node_type": "entity_mention",
						"text_content": mention_text,
						"entity_id": entity.entity_id,
						"entity_name": entity.name,
						"entity_type": entity.entity_type,
						"session_id": mention.session_id,
						"source_unit_uid": mention.unit_uid,
						"content": mention.content,
						"temporal_info": mention.temporal_info,
						"spatial_info": mention.spatial_info,
						"aliases": mention.aliases,
						"confidence": mention.confidence,
					},
					metadata={
						"type": "entity_mention",
						"source_id": sample_id,
						"entity_id": entity.entity_id,
						"entity_type": entity.entity_type,
						"session_id": mention.session_id,
						"created_at": datetime.now().isoformat(),
					},
				)
				write_requests.append(GraphWriteRequest(
					unit=unit,
					explicit_content_for_embedding=mention_text,
					content_type_for_embedding="text",
										space_names=[GRAPH_MENTION_SPACE],
					index_update_mode="none",
					generate_sparse_embedding=False,
					source="locomo10_entity_mention",
				))
				added += 1
		dispatch_graph_write_requests(graph, write_requests)
		return added

	def _add_relation_units(
		self,
		graph: SemanticGraph,
		sample_id: str,
		entities: Sequence[Any],
		relations: Sequence[Any],
	) -> int:
		entity_by_id = {entity.entity_id: entity for entity in entities}
		added = 0
		write_requests: List[GraphWriteRequest] = []
		for relation in relations:
			source = entity_by_id.get(relation.source_entity_id)
			target = entity_by_id.get(relation.target_entity_id)
			source_name = source.name if source else relation.source_entity_id
			target_name = target.name if target else relation.target_entity_id
			relation_text = self._build_relation_text(relation, source_name, target_name)
			unit = MemoryUnit(
				uid=f"{sample_id}_{relation.relation_id}",
				raw_data={
					"node_type": "entity_relation",
					"text_content": relation_text,
					"relation_id": relation.relation_id,
					"source_entity_id": relation.source_entity_id,
					"source_entity_name": source_name,
					"target_entity_id": relation.target_entity_id,
					"target_entity_name": target_name,
					"relation_type": relation.relation_type,
					"context": relation.context,
					"temporal_info": relation.temporal_info,
					"confidence": relation.confidence,
					"is_cross_session": relation.is_cross_session,
				},
				metadata={
					"type": "relation",
					"source_id": sample_id,
					"relation_type": relation.relation_type,
					"source_entity_id": relation.source_entity_id,
					"target_entity_id": relation.target_entity_id,
					"confidence": relation.confidence,
					"created_at": datetime.now().isoformat(),
				},
			)
			write_requests.append(GraphWriteRequest(
				unit=unit,
				explicit_content_for_embedding=relation_text,
				content_type_for_embedding="text",
								space_names=[GRAPH_RELATION_SPACE],
				index_update_mode="none",
				generate_sparse_embedding=False,
				source="locomo10_relation",
			))
			added += 1
		dispatch_graph_write_requests(graph, write_requests)
		return added

	def _add_relation_edges(self, graph: SemanticGraph, sample_id: str, relations: Sequence[Any]) -> None:
		for relation in relations:
			try:
				graph.add_relationship(
					source_uid=f"{sample_id}_{relation.source_entity_id}",
					target_uid=f"{sample_id}_{relation.target_entity_id}",
					relationship_name=relation.relation_type,
					bidirectional=False,
					context=relation.context,
					temporal_info=relation.temporal_info,
					confidence=relation.confidence,
				)
			except Exception as exc:
				LOGGER.warning("Could not add graph edge %s: %s", relation.relation_id, exc)

	@staticmethod
	def _build_mention_text(entity: Any, mention: Any) -> str:
		parts = [f"Entity: {entity.name} ({entity.entity_type})"]
		if mention.session_id:
			parts.append(f"Session: {mention.session_id}")
		if mention.content:
			parts.append(f"Context: {mention.content}")
		if mention.temporal_info:
			parts.append(f"Time: {mention.temporal_info}")
		if mention.spatial_info:
			parts.append(f"Location: {mention.spatial_info}")
		if mention.aliases:
			parts.append(f"Aliases: {', '.join(mention.aliases)}")
		return " | ".join(parts)

	@staticmethod
	def _build_relation_text(relation: Any, source_name: str, target_name: str) -> str:
		parts = [f"{source_name} {relation.relation_type} {target_name}"]
		if relation.context:
			parts.append(f"Context: {relation.context}")
		if relation.temporal_info:
			parts.append(f"Time: {relation.temporal_info}")
		return " | ".join(parts)

	def _save_l0_artifacts(self, artifacts_dir: Path, sessions: Sequence[LocomoSession]) -> None:
		write_json(
			artifacts_dir / "l0_units.json",
			[
				{"uid": unit.uid, "raw_data": unit.raw_data, "metadata": unit.metadata}
				for session in sessions
				for unit in session.units
			],
		)

	@staticmethod
	def _load_hierarchical_artifacts(artifacts_dir: Path) -> Dict[str, Any]:
		l2_split_meta = read_json(artifacts_dir / "l2_split_units.json", default={}) or {}
		return {
			"l1_results": read_json(artifacts_dir / "l1_results.json", default=[]) or [],
			"l2_result": read_json(artifacts_dir / "l2_result.json", default=None),
			"l2_split_units": l2_split_meta.get("count", 0) if isinstance(l2_split_meta, dict) else len(l2_split_meta),
		}

	@staticmethod
	def _save_hierarchical_artifacts(artifacts_dir: Path, hierarchical: Dict[str, Any]) -> None:
		write_json(artifacts_dir / "l1_results.json", hierarchical.get("l1_results", []))
		write_json(artifacts_dir / "l2_result.json", hierarchical.get("l2_result"))
		write_json(artifacts_dir / "l2_split_units.json", {"count": hierarchical.get("l2_split_units", 0)})

	@staticmethod
	def _load_episodic_artifacts(artifacts_dir: Path) -> Dict[str, Any]:
		facts = read_json(artifacts_dir / "episodic_facts_raw.json", default=[]) or []
		deduplicated = read_json(artifacts_dir / "episodic_facts_deduplicated.json", default=[]) or []
		return {"facts": facts, "deduplicated_facts": deduplicated, "added_uids": []}

	@staticmethod
	def _save_episodic_artifacts(artifacts_dir: Path, episodic: Dict[str, Any]) -> None:
		write_json(artifacts_dir / "episodic_facts_raw.json", episodic.get("facts", []))
		write_json(artifacts_dir / "episodic_facts_deduplicated.json", episodic.get("deduplicated_facts", []))
		write_json(artifacts_dir / "episodic_added_uids.json", episodic.get("added_uids", []))

	@staticmethod
	def _load_entity_relation_artifacts(artifacts_dir: Path) -> Dict[str, Any]:
		raw_entities = read_json(artifacts_dir / "entity_raw_entities.json", default=[]) or []
		entities = read_json(artifacts_dir / "entity_entities_deduplicated.json", default=[]) or []
		relations = read_json(artifacts_dir / "entity_relations.json", default=[]) or []
		stats = read_json(artifacts_dir / "entity_relation_add_stats.json", default={}) or {}
		return {
			"raw_entities": raw_entities,
			"entities": entities,
			"relations": relations,
			"entity_units_added": stats.get("entity_units_added", 0),
			"mentions_added": stats.get("mentions_added", 0),
			"relation_units_added": stats.get("relation_units_added", 0),
		}

	@staticmethod
	def _save_entity_relation_artifacts(artifacts_dir: Path, entity_relation: Dict[str, Any]) -> None:
		write_json(artifacts_dir / "entity_raw_entities.json", entity_relation.get("raw_entities", []))
		write_json(artifacts_dir / "entity_entities_deduplicated.json", entity_relation.get("entities", []))
		write_json(artifacts_dir / "entity_relations.json", entity_relation.get("relations", []))
		write_json(
			artifacts_dir / "entity_relation_add_stats.json",
			{
				"entity_units_added": entity_relation.get("entity_units_added", 0),
				"mentions_added": entity_relation.get("mentions_added", 0),
				"relation_units_added": entity_relation.get("relation_units_added", 0),
				"entity_extraction_scope": entity_relation.get("entity_extraction_scope"),
				"relation_extraction_scope": entity_relation.get("relation_extraction_scope"),
			},
		)

	def _save_artifacts(
		self,
		artifacts_dir: Path,
		sessions: Sequence[LocomoSession],
		hierarchical: Dict[str, Any],
		episodic: Dict[str, Any],
		entity_relation: Dict[str, Any],
	) -> None:
		write_json(
			artifacts_dir / "l0_units.json",
			[
				{"uid": unit.uid, "raw_data": unit.raw_data, "metadata": unit.metadata}
				for session in sessions
				for unit in session.units
			],
		)
		write_json(artifacts_dir / "l1_results.json", hierarchical["l1_results"])
		write_json(artifacts_dir / "l2_result.json", hierarchical.get("l2_result"))
		write_json(artifacts_dir / "episodic_facts_raw.json", episodic["facts"])
		write_json(artifacts_dir / "episodic_facts_deduplicated.json", episodic["deduplicated_facts"])
		write_json(artifacts_dir / "entity_raw_entities.json", entity_relation["raw_entities"])
		write_json(artifacts_dir / "entity_entities_deduplicated.json", entity_relation["entities"])
		write_json(artifacts_dir / "entity_relations.json", entity_relation["relations"])


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Build per-sample LoCoMo10 SemanticGraph memories using mandol.auto_builder."
	)
	parser.add_argument("--dataset-file", default=str(SCRIPT_DIR / "dataset" / "locomo10.json"))
	parser.add_argument("--output-dir", default=str(SCRIPT_DIR / "memory"))
	parser.add_argument("--sample-ids", nargs="+", help="Only build the specified sample IDs, e.g. conv-26 conv-30")
	parser.add_argument("--limit", type=int, default=10, help="Maximum number of samples to build after filtering")
	parser.add_argument("--force", action="store_true", help="Delete and rebuild existing sample output directories")
	parser.add_argument("--skip-existing", action="store_true", help="Skip sample directories that already exist")
	parser.add_argument("--no-resume", action="store_true", help="Disable checkpoint resume for existing sample directories")
	parser.add_argument("--no-stage-graph-checkpoints", action="store_true", help="Only write JSON checkpoints between stages; save the graph at the end")
	parser.add_argument("--dry-run", action="store_true", help="Only parse samples and count L0 units; no model calls")

	parser.add_argument("--strategy", default="locomo10", help="auto_builder strategy preset; locomo10 is aligned to offline LoCoMo makers")
	parser.add_argument("--extraction-model")
	parser.add_argument("--dedup-model")
	parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
	parser.add_argument("--dedup-embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
	parser.add_argument("--faiss-index-type", default="IDMap,Flat")
	parser.add_argument("--max-context-ratio", type=float, default=0.85)
	parser.add_argument("--request-max-retries", type=int, default=5)
	parser.add_argument("--request-timeout", type=float, default=240.0)

	parser.add_argument("--contextual-workers", type=int)
	parser.add_argument("--hierarchical-session-workers", type=int)
	parser.add_argument("--episodic-session-workers", type=int)
	parser.add_argument("--episodic-dedup-workers", type=int)
	parser.add_argument("--entity-session-workers", type=int)
	parser.add_argument("--entity-dedup-workers", type=int)
	parser.add_argument("--relation-session-workers", type=int)
	parser.add_argument("--min-content-length", type=int, default=1)
	parser.add_argument("--episodic-dedup-method", help="Use 'none' to keep extracted facts without global merging")
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
	parser.add_argument("--entity-extraction-scope", choices=["sample", "session"], default="session")
	parser.add_argument("--relation-extraction-scope", choices=["sample", "session"], default="session")

	parser.add_argument("--no-contextual-retrieval", action="store_true")
	parser.add_argument("--enable-hierarchical-dedup", action="store_true")
	parser.add_argument("--no-episodic-dedup", action="store_true")
	parser.add_argument("--no-entity-dedup", action="store_true")
	parser.add_argument("--no-relation-extraction", action="store_true")
	parser.add_argument("--enable-graph-edges", action="store_true", help="Also add relation edges to the graph structure")
	parser.add_argument("--no-freeze-retrievers", action="store_true", help="Do not build frozen BM25/SPLADE static matrices on save")
	parser.add_argument("--debug", action="store_true")
	return parser


def apply_strategy_defaults(args: argparse.Namespace) -> argparse.Namespace:
	strategy = resolve_strategy(args.strategy)
	defaults = {
		"extraction_model": strategy.extraction_llm_name,
		"dedup_model": strategy.dedup_llm_name,
		"contextual_workers": strategy.contextual_workers,
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
	}
	for key, value in defaults.items():
		if getattr(args, key, None) is None:
			setattr(args, key, value)
	return args


def main(argv: Optional[Sequence[str]] = None) -> int:
	parser = build_arg_parser()
	args = parser.parse_args(argv)
	args = apply_strategy_defaults(args)
	configure_logging(args.debug)

	if args.force and args.skip_existing:
		parser.error("--force and --skip-existing cannot be used together")

	LOGGER.info("LoCoMo10 build_graph started")
	LOGGER.info("Dataset: %s", Path(args.dataset_file).resolve())
	LOGGER.info("Output : %s", Path(args.output_dir).resolve())

	builder = LocomoGraphBuilder(args) if not args.dry_run else None
	if builder is None:
		dry_builder = object.__new__(LocomoGraphBuilder)
		dry_builder.args = args
		dry_builder.output_dir = Path(args.output_dir).resolve()
		dry_builder.output_dir.mkdir(parents=True, exist_ok=True)
		summary = dry_builder.build_all()
	else:
		summary = builder.build_all()

	failed = [item for item in summary["samples"] if not item.get("success")]
	if failed:
		LOGGER.error("Build finished with %d failed samples", len(failed))
		return 1

	LOGGER.info("Build finished successfully: %d samples", len(summary["samples"]))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
