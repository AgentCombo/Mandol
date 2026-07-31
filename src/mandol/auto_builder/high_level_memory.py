"""Unified high-level memory build API."""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..core.memory_unit import MemoryUnit
from ..core.memory_space_registry import MemorySpaceRegistry, TowerSpace
from ..core.semantic_graph import SemanticGraph
from ..llm.llm_client import LLMClient
from .entity_relation_builder import EntityRelationAutoBuilder, EntityRelationBuilderConfig
from .episodic_builder import EpisodicAutoBuilder, EpisodicBuilderConfig
from .episodic_prompts import EpisodicFactType
from .graph_write_queue import GraphWriteQueue
from .hierarchical_builder import HierarchicalAutoBuilder, HierarchicalBuilderConfig


@dataclass
class HighLevelMemoryBuildConfig:
    extraction_style: str = "default"
    sample_id: str = "sample"
    participants: List[str] = field(default_factory=list)
    reference_date: Optional[str] = None
    build_hierarchical: bool = True
    build_episodic: bool = True
    build_entity_relation: bool = True
    enable_contextual_retrieval: bool = False
    enable_hierarchical_deduplication: bool = False
    enable_episodic_deduplication: bool = False
    episodic_dedup_method: str = "none"
    enable_entity_deduplication: bool = True
    enable_relation_extraction: bool = True
    dedup_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    output_dir: Optional[str] = None
    save_after_build: bool = True
    freeze_retrievers: bool = False
    producer_workers: int = 60
    dedup_workers: int = 30
    graph_write_batch_size: int = 32
    graph_write_embedding_batch_size: int = 32
    graph_write_queue_maxsize: int = 2048
    graph_write_flush_interval: float = 0.25


@dataclass
class HighLevelMemoryBuildResult:
    sample_id: str
    l0_units: int = 0
    l1_units: int = 0
    l2_units: int = 0
    episodic_facts: int = 0
    episodic_units: int = 0
    raw_entities: int = 0
    entities: int = 0
    relations: int = 0
    saved_path: Optional[str] = None
    stage_status: Dict[str, str] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HighLevelMemoryBuilder:
    """Small public facade for building all high-level memory towers from L0 units."""

    def __init__(
        self,
        semantic_graph: SemanticGraph,
        llm_client: LLMClient,
        dedup_llm_client: Optional[LLMClient] = None,
        config: Optional[HighLevelMemoryBuildConfig] = None,
    ):
        self.semantic_graph = semantic_graph
        self.llm_client = llm_client
        self.dedup_llm_client = dedup_llm_client or llm_client
        self.config = config or HighLevelMemoryBuildConfig()

    def build_from_l0_units(self, l0_units: Sequence[MemoryUnit]) -> HighLevelMemoryBuildResult:
        result = HighLevelMemoryBuildResult(sample_id=self.config.sample_id, l0_units=len(l0_units))
        grouped_units = self._group_by_session(l0_units)
        self._ensure_spaces()
        self._set_build_state(result, status="running", current_stage="start")

        try:
            with GraphWriteQueue(
                semantic_system=self.semantic_graph,
                batch_size=self.config.graph_write_batch_size,
                embedding_batch_size=self.config.graph_write_embedding_batch_size,
                max_queue_size=self.config.graph_write_queue_maxsize,
                flush_interval=self.config.graph_write_flush_interval,
                name=f"HighLevelGraphWriter-{self.config.sample_id}",
            ) as graph_writer:
                if self.config.build_hierarchical:
                    self._set_build_state(result, current_stage="hierarchical")
                    hierarchical = self._build_hierarchical(grouped_units, graph_writer)
                    result.l1_units = len(hierarchical.get("l1_results", []))
                    result.l2_units = 1 if hierarchical.get("l2_result") else 0
                    graph_writer.drain()
                    result.stage_status["hierarchical"] = "completed"

                if self.config.build_episodic:
                    self._set_build_state(result, current_stage="episodic")
                    episodic = self._build_episodic(grouped_units, graph_writer)
                    result.episodic_facts = len(episodic.get("facts", []))
                    result.episodic_units = len(episodic.get("added_uids", []))
                    graph_writer.drain()
                    result.stage_status["episodic"] = "completed"

                if self.config.build_entity_relation:
                    self._set_build_state(result, current_stage="entity_relation")
                    entity_relation = self._build_entity_relation(l0_units, grouped_units, graph_writer)
                    result.raw_entities = len(entity_relation.get("raw_entities", []))
                    result.entities = len(entity_relation.get("entities", []))
                    result.relations = len(entity_relation.get("relations", []))
                    graph_writer.drain()
                    result.stage_status["entity_relation"] = "completed"

                graph_writer.drain()

                if self.config.output_dir and self.config.save_after_build:
                    self._set_build_state(result, current_stage="save_graph")
                    self.semantic_graph.build_semantic_map_index()
                    self.semantic_graph.save_graph(self.config.output_dir, freeze_retrievers=self.config.freeze_retrievers)
                    result.saved_path = str(Path(self.config.output_dir).resolve())
                    result.stage_status["save_graph"] = "completed"

            self._set_build_state(result, status="completed", current_stage="done")
            return result
        except Exception as exc:
            result.errors.append(f"{type(exc).__name__}: {exc}")
            self._set_build_state(result, status="failed", current_stage="failed")
            raise

    def _build_hierarchical(
        self,
        grouped_units: Dict[str, List[MemoryUnit]],
        graph_writer: GraphWriteQueue,
    ) -> Dict[str, Any]:
        builder = HierarchicalAutoBuilder(
            semantic_system=self.semantic_graph,
            llm_client=self.llm_client,
            config=HierarchicalBuilderConfig(
                extraction_style=self.config.extraction_style,
                enable_contextual_retrieval=self.config.enable_contextual_retrieval,
                contextual_parallel_workers=self.config.producer_workers,
                enable_deduplication=self.config.enable_hierarchical_deduplication,
                parallel_workers=self.config.dedup_workers,
            ),
        )
        l1_results = []
        session_items = list(grouped_units.items())

        def extract_session(session_id: str, units: List[MemoryUnit]) -> List[Any]:
            session_date = self._session_date(units)
            return builder.extract_l1_from_l0_units(
                l0_units=list(units),
                session_id=session_id,
                session_date=session_date,
                participants=self.config.participants,
            )

        if session_items:
            max_workers = min(max(1, self.config.producer_workers), len(session_items))
            if max_workers == 1:
                ordered_results = [extract_session(session_id, units) for session_id, units in session_items]
            else:
                ordered_results = [None] * len(session_items)
                with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="HighLevelL1") as executor:
                    future_to_index = {
                        executor.submit(extract_session, session_id, units): index
                        for index, (session_id, units) in enumerate(session_items)
                    }
                    for future in as_completed(future_to_index):
                        ordered_results[future_to_index[future]] = future.result()
            for session_results in ordered_results:
                l1_results.extend(session_results or [])

        if self.config.enable_hierarchical_deduplication and l1_results:
            builder.llm_client = self.dedup_llm_client
            l1_results = builder.deduplicate_l1(l1_results)
        builder.llm_client = self.llm_client
        l2_result = builder.aggregate_l2_from_l1(l1_results, self.config.sample_id, self.config.participants)
        builder.add_to_semantic_system(
            l1_results,
            l2_result,
            rebuild_index=False,
            graph_writer=graph_writer,
            wait_for_completion=False,
        )
        return {"l1_results": l1_results, "l2_result": l2_result}

    def _build_episodic(
        self,
        grouped_units: Dict[str, List[MemoryUnit]],
        graph_writer: GraphWriteQueue,
    ) -> Dict[str, Any]:
        builder = EpisodicAutoBuilder(
            semantic_system=self.semantic_graph,
            llm_client=self.llm_client,
            config=EpisodicBuilderConfig(
                extraction_style=self.config.extraction_style,
                fact_types=EpisodicFactType.locomo_types() if self.config.extraction_style == "locomo" else None,
                enable_deduplication=self.config.enable_episodic_deduplication,
                dedup_method=self.config.episodic_dedup_method,
                embedding_model=self.config.dedup_embedding_model,
                dedup_parallel_workers=self.config.dedup_workers,
            ),
        )
        facts = []
        session_items = list(grouped_units.items())

        def extract_session(session_id: str, units: List[MemoryUnit]) -> List[Any]:
            return builder.extract_from_l0_units(
                l0_unit_uids=[unit.uid for unit in units],
                reference_date=self._session_date(units) or self.config.reference_date,
                source_id=f"{self.config.sample_id}_{session_id}",
                speakers=", ".join(self.config.participants),
            )

        if session_items:
            max_workers = min(max(1, self.config.producer_workers), len(session_items))
            if max_workers == 1:
                ordered_results = [extract_session(session_id, units) for session_id, units in session_items]
            else:
                ordered_results = [None] * len(session_items)
                with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="HighLevelEpisodic") as executor:
                    future_to_index = {
                        executor.submit(extract_session, session_id, units): index
                        for index, (session_id, units) in enumerate(session_items)
                    }
                    for future in as_completed(future_to_index):
                        ordered_results[future_to_index[future]] = future.result()
            for session_results in ordered_results:
                facts.extend(session_results or [])

        if self.config.enable_episodic_deduplication and facts:
            builder.llm_client = self.dedup_llm_client
            output_facts = builder.deduplicate_facts(facts)
        else:
            output_facts = facts
        added_uids = builder.add_to_semantic_system(
            output_facts,
            space_name=TowerSpace.EPISODIC_ROOT.value,
            graph_writer=graph_writer,
            wait_for_completion=False,
        )
        return {"facts": facts, "output_facts": output_facts, "added_uids": added_uids}

    def _build_entity_relation(
        self,
        l0_units: Sequence[MemoryUnit],
        grouped_units: Dict[str, List[MemoryUnit]],
        graph_writer: GraphWriteQueue,
    ) -> Dict[str, Any]:
        builder = EntityRelationAutoBuilder(
            semantic_system=self.semantic_graph,
            llm_client=self.llm_client,
            config=EntityRelationBuilderConfig(
                extraction_style=self.config.extraction_style,
                embedding_model=self.config.dedup_embedding_model,
                enable_relation_extraction=self.config.enable_relation_extraction,
                enable_llm_deduplication=self.config.enable_entity_deduplication,
                parallel_workers=self.config.dedup_workers,
            ),
        )
        raw_entities = []
        session_items = list(grouped_units.items()) or [(self.config.sample_id, list(l0_units))]

        def extract_session(units: List[MemoryUnit]) -> List[Dict[str, Any]]:
            return builder.extract_entities_from_l0_units(
                l0_units=list(units),
                reference_date=self.config.reference_date,
                source_id=self.config.sample_id,
                session_type="chat",
            )

        max_workers = min(max(1, self.config.producer_workers), len(session_items))
        if max_workers == 1:
            ordered_results = [extract_session(units) for _, units in session_items]
        else:
            ordered_results = [None] * len(session_items)
            with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="HighLevelEntities") as executor:
                future_to_index = {
                    executor.submit(extract_session, units): index
                    for index, (_, units) in enumerate(session_items)
                }
                for future in as_completed(future_to_index):
                    ordered_results[future_to_index[future]] = future.result()
        for session_results in ordered_results:
            raw_entities.extend(session_results or [])

        if self.config.enable_entity_deduplication and raw_entities:
            builder.llm_client = self.dedup_llm_client
            entities = builder.deduplicate_entities(raw_entities)
        else:
            entities = [builder._convert_raw_to_extracted_entity(entity, f"entity_{idx}") for idx, entity in enumerate(raw_entities)]
        relations = []
        if self.config.enable_relation_extraction and entities:
            builder.llm_client = self.llm_client
            relations = builder.extract_relations_from_entities(list(l0_units), entities, session_type="chat")
        builder.add_to_semantic_system(
            entities=entities,
            relations=relations,
            source_id=self.config.sample_id,
            space_name=TowerSpace.GRAPH_ENTITIES.value,
            graph_writer=graph_writer,
            wait_for_completion=False,
        )
        return {"raw_entities": raw_entities, "entities": entities, "relations": relations}

    def _set_build_state(self, result: HighLevelMemoryBuildResult, **updates: Any) -> None:
        state = {
            "builder": "HighLevelMemoryBuilder",
            "sample_id": self.config.sample_id,
            "config": asdict(self.config),
            "result": result.to_dict(),
            "updated_at": datetime.now().isoformat(),
        }
        state.update(updates)
        self.semantic_graph.set_high_level_memory_build_state(state)

    @staticmethod
    def _group_by_session(l0_units: Sequence[MemoryUnit]) -> Dict[str, List[MemoryUnit]]:
        grouped: Dict[str, List[MemoryUnit]] = defaultdict(list)
        for unit in l0_units:
            session_id = (unit.metadata or {}).get("session_id") or (unit.raw_data or {}).get("session_id") or "session_1"
            grouped[str(session_id)].append(unit)
        return dict(grouped)

    @staticmethod
    def _session_date(units: Sequence[MemoryUnit]) -> Optional[str]:
        for unit in units:
            value = (unit.metadata or {}).get("session_date") or (unit.raw_data or {}).get("session_date")
            if value and value != "unknown":
                return str(value)
        return None

    def _ensure_spaces(self) -> None:
        MemorySpaceRegistry.initialize_spaces(self.semantic_graph)


def build_high_level_memory(
    semantic_graph: SemanticGraph,
    l0_units: Sequence[MemoryUnit],
    llm_client: LLMClient,
    dedup_llm_client: Optional[LLMClient] = None,
    config: Optional[HighLevelMemoryBuildConfig] = None,
) -> HighLevelMemoryBuildResult:
    builder = HighLevelMemoryBuilder(
        semantic_graph=semantic_graph,
        llm_client=llm_client,
        dedup_llm_client=dedup_llm_client,
        config=config,
    )
    return builder.build_from_l0_units(l0_units)
