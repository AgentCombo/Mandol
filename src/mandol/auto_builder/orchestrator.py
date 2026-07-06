# mandol/auto_builder/orchestrator.py
"""Utilities for orchestrator."""
import logging
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Any, Union, TYPE_CHECKING, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from .auto_session_assigner import AutoSessionAssigner, AutoSessionConfig
from .session_tracker import SessionTracker
from .hierarchical_builder import HierarchicalAutoBuilder, HierarchicalBuilderConfig, L1ExtractionResult, L2AggregationResult
from .hierarchical_prompts import HierarchicalPromptManager
from .episodic_builder import EpisodicAutoBuilder, EpisodicBuilderConfig, EpisodicFact
from .entity_relation_builder import EntityRelationAutoBuilder, EntityRelationBuilderConfig, ExtractedEntity, ExtractedRelation
from .graph_write_queue import GraphWriteRequest, dispatch_graph_write_requests
from .l0_views import build_l0_inference_context, extract_embedding_text, extract_original_text
from .strategy_config import PipelineStrategy, STYLE_ALIASES, STYLE_STRATEGIES
from ..core.memory_space_registry import TowerSpace
from ..utils.logging_config import create_module_logger

if TYPE_CHECKING:
    from ..core.semantic_map import SemanticMap
    from ..core.semantic_graph import SemanticGraph
    from ..core.memory_unit import MemoryUnit
    from ..llm.llm_client import LLMClient

logger = create_module_logger("auto_builder.orchestrator")





@dataclass
class OrchestratorConfig:

    raw_space_name: str = "raw"

    l0_space_name: str = TowerSpace.HIERARCHICAL_L0.value

    enable_contextual_retrieval: bool = True
    contextual_parallel_workers: int = 60

    chunk_size: int = 512
    chunk_overlap: int = 50

    
    enable_hierarchical: bool = True
    enable_episodic: bool = True
    enable_entity_relation: bool = True

    # Hierarchical
    enable_hierarchical_deduplication: bool = False

    # Episodic
    enable_episodic_deduplication: bool = True
    episodic_dedup_method: str = "dbscan_llm"

    # EntityRelation
    enable_relation_extraction: bool = True
    enable_entity_deduplication: bool = True

    add_to_system: bool = True

    # Default real-time path: automatically split raw units into sessions when
    # the caller did not provide an explicit session_id. Benchmark styles are
    # intentionally excluded.
    enable_auto_session_split: bool = True
    auto_session_split_styles: Tuple[str, ...] = ("default",)
    auto_session_config: Optional[AutoSessionConfig] = None





@dataclass
class PipelineResult:
    sample_id: str
    extraction_style: str
    success: bool = False

    raw_unit_count: int = 0
    l0_unit_count: int = 0
    l0_unit_uids: List[str] = field(default_factory=list)

    hierarchical_result: Optional[Dict[str, Any]] = None
    episodic_result: Optional[Dict[str, Any]] = None
    entity_relation_result: Optional[Dict[str, Any]] = None

    errors: List[str] = field(default_factory=list)
    processing_time: float = 0.0





class MemoryOrchestrator:
    """extraction_llm / dedup_llm。 orchestrator = MemoryOrchestrator( semantic_system=semantic_graph, llm_clients={ "qwen-3.5-plus-thinking": LLMClient("qwen-3.5-plus-thinking"), "deepseek-v3.2-dashscope": LLMClient("deepseek-v3.2-dashscope"), },."""

    def __init__(
        self,
        semantic_system: Union["SemanticMap", "SemanticGraph"],
        llm_clients: Dict[str, "LLMClient"],
        config: Optional[OrchestratorConfig] = None,
    ):
        self.semantic_system = semantic_system
        self.config = config or OrchestratorConfig()
        self.llm_clients = llm_clients
        self.extraction_llm: Optional["LLMClient"] = None
        self.dedup_llm: Optional["LLMClient"] = None
        self.auto_session_assigner = AutoSessionAssigner(self.config.auto_session_config)

        if not self.llm_clients:
            raise ValueError("llm_clients must be provided as Dict[str, LLMClient]")

        logger.info("MemoryOrchestrator initialized")
        logger.info(f"   Available LLMs: {', '.join(sorted(self.llm_clients.keys()))}")

    def _resolve_strategy(self, extraction_style: str) -> tuple[str, PipelineStrategy]:
        """Resolve requested extraction_style to a registered PipelineStrategy."""
        requested_style = extraction_style or "default"
        strategy_key = STYLE_ALIASES.get(requested_style, requested_style)
        if strategy_key not in STYLE_STRATEGIES:
            logger.warning(
                "No strategy found for extraction_style=%s; falling back to default",
                requested_style,
            )
            strategy_key = "default"
        return strategy_key, STYLE_STRATEGIES[strategy_key]

    def _select_strategy_llms(self, strategy_key: str, strategy: PipelineStrategy) -> None:
        """Select extraction/dedup LLM clients for the current strategy."""
        missing = [
            model_name
            for model_name in (strategy.extraction_llm_name, strategy.dedup_llm_name)
            if model_name not in self.llm_clients
        ]
        if missing:
            available = ", ".join(sorted(self.llm_clients.keys())) or "<empty>"
            raise ValueError(
                f"Strategy '{strategy_key}' requires LLMs {missing}, but they were not provided in llm_clients. "
                f"Available models: {available}"
            )

        self.extraction_llm = self.llm_clients[strategy.extraction_llm_name]
        self.dedup_llm = self.llm_clients[strategy.dedup_llm_name]
        logger.info(
            "   Strategy[%s]: l0_mode=%s build_l1=%s build_l2=%s extraction_llm=%s dedup_llm=%s",
            strategy_key,
            strategy.l0_mode,
            strategy.build_l1,
            strategy.build_l2,
            strategy.extraction_llm_name,
            strategy.dedup_llm_name,
        )

    
    def process_sample_from_raw(
        self,
        sample_id: str,
        extraction_style: str = "default",
        session_id: Optional[str] = None,
        session_date: Optional[str] = None,
        participants: Optional[List[str]] = None,
        speakers: Optional[str] = None,
        reference_date: Optional[str] = None,
        custom_prompts: Optional[Dict[str, str]] = None,
    ) -> PipelineResult:
        """Build all configured memory layers for one raw sample.

        Args:
            sample_id: Input sample identifier.
            extraction_style: Registered build strategy name.
            session_id: Optional session identifier used in generated units.
            session_date: Optional source-session date.
            participants: Optional participant names for prompt context.
            speakers: Optional speaker metadata.
            reference_date: Date anchor used by extraction prompts.
            custom_prompts: Optional prompt overrides for builder stages.

        Returns:
            PipelineResult with generated units, relationships, and stage stats.
        """
        start_time = datetime.now()
        explicit_session_id = session_id
        reference_date = reference_date or session_date or datetime.now().strftime("%Y-%m-%d")
        strategy_key, strategy = self._resolve_strategy(extraction_style)
        session_id = explicit_session_id or sample_id
        self._select_strategy_llms(strategy_key, strategy)

        result = PipelineResult(sample_id=sample_id, extraction_style=strategy_key)

        logger.info(f"\n{'=' * 60}")
        logger.info("MemoryOrchestrator started")
        logger.info(f"   sample_id       : {sample_id}")
        logger.info(f"   extraction_style: {extraction_style} -> {strategy_key}")
        logger.info(f"   session_id      : {session_id}")
        logger.info(f"{'=' * 60}")

        try:
            raw_units = self._fetch_raw_units(sample_id)
            result.raw_unit_count = len(raw_units)
            if not raw_units:
                msg = f"No raw memory units found for sample_id={sample_id}"
                logger.warning(msg)
                result.errors.append(msg)
                return result

            logger.info(f"Loaded {len(raw_units)} raw memory units")

            raw_units = self._assign_default_sessions_if_needed(
                raw_units=raw_units,
                sample_id=sample_id,
                strategy_key=strategy_key,
                explicit_session_id=explicit_session_id,
                session_date=session_date,
                participants=participants,
            )

            h_config, e_config, er_config = self._create_builder_configs(strategy_key, strategy)
            h_builder, e_builder, er_builder = self._create_builders(h_config, e_config, er_config)

            l0_units = self._preprocess_to_l0_by_session_if_needed(
                h_builder=h_builder,
                raw_units=raw_units,
                strategy=strategy,
                strategy_key=strategy_key,
                session_id=session_id,
                explicit_session_id=explicit_session_id,
                session_date=session_date,
                participants=participants,
                custom_prompts=custom_prompts,
            )
            result.l0_unit_count = len(l0_units)
            result.l0_unit_uids = [u.uid for u in l0_units]
            logger.info(f"L0 preprocessing completed: {len(l0_units)} L0 units")

            
            if self.config.enable_hierarchical and (strategy.build_l1 or strategy.build_l2):
                result.hierarchical_result = self._run_hierarchical_tower(
                    h_builder, l0_units, session_id, sample_id,
                    session_date, participants, custom_prompts, strategy,
                )
                if result.hierarchical_result and result.hierarchical_result.get("errors"):
                    result.errors.extend(result.hierarchical_result["errors"])
            elif self.config.enable_hierarchical:
                logger.info("[Hierarchical Tower] strategy %s disables L1/L2; skipping", strategy_key)

            if self.config.enable_episodic:
                result.episodic_result = self._run_episodic_tower(
                    e_builder, result.l0_unit_uids, reference_date,
                    sample_id, speakers,
                )
                if result.episodic_result and result.episodic_result.get("errors"):
                    result.errors.extend(result.episodic_result["errors"])

            if self.config.enable_entity_relation:
                result.entity_relation_result = self._run_entity_relation_tower(
                    er_builder, l0_units, reference_date,
                    sample_id, custom_prompts,
                )
                if result.entity_relation_result and result.entity_relation_result.get("errors"):
                    result.errors.extend(result.entity_relation_result["errors"])

            result.success = not result.errors

        except Exception as e:
            msg = f"Pipeline failed globally: {e}"
            logger.error(f" {msg}")
            result.errors.append(msg)

        result.processing_time = (datetime.now() - start_time).total_seconds()

        logger.info(f"\n{'=' * 60}")
        logger.info("Pipeline completed")
        logger.info(f"   Raw={result.raw_unit_count}  L0={result.l0_unit_count}")
        logger.info(f"   Hierarchical: {'OK' if result.hierarchical_result else 'SKIP/FAIL'}")
        logger.info(f"   Episodic    : {'OK' if result.episodic_result else 'SKIP/FAIL'}")
        logger.info(f"   EntityRel   : {'OK' if result.entity_relation_result else 'SKIP/FAIL'}")
        logger.info(f"   Elapsed: {result.processing_time:.2f}s")
        logger.info(f"{'=' * 60}\n")

        return result

    def process_raw_units(
        self,
        raw_units: List["MemoryUnit"],
        sample_id: str = "inline",
        extraction_style: str = "default",
        session_id: Optional[str] = None,
        session_date: Optional[str] = None,
        participants: Optional[List[str]] = None,
        speakers: Optional[str] = None,
        reference_date: Optional[str] = None,
        custom_prompts: Optional[Dict[str, str]] = None,
    ) -> PipelineResult:
        """Process raw units."""
        start_time = datetime.now()
        explicit_session_id = session_id
        reference_date = reference_date or session_date or datetime.now().strftime("%Y-%m-%d")
        strategy_key, strategy = self._resolve_strategy(extraction_style)
        session_id = explicit_session_id or sample_id
        self._select_strategy_llms(strategy_key, strategy)

        result = PipelineResult(
            sample_id=sample_id,
            extraction_style=strategy_key,
            raw_unit_count=len(raw_units),
        )

        if not raw_units:
            result.errors.append("raw_units is empty")
            return result

        h_config, e_config, er_config = self._create_builder_configs(strategy_key, strategy)
        h_builder, e_builder, er_builder = self._create_builders(h_config, e_config, er_config)

        try:
            raw_units = self._assign_default_sessions_if_needed(
                raw_units=raw_units,
                sample_id=sample_id,
                strategy_key=strategy_key,
                explicit_session_id=explicit_session_id,
                session_date=session_date,
                participants=participants,
            )

            l0_units = self._preprocess_to_l0_by_session_if_needed(
                h_builder=h_builder,
                raw_units=raw_units,
                strategy=strategy,
                strategy_key=strategy_key,
                session_id=session_id,
                explicit_session_id=explicit_session_id,
                session_date=session_date,
                participants=participants,
                custom_prompts=custom_prompts,
            )
            result.l0_unit_count = len(l0_units)
            result.l0_unit_uids = [u.uid for u in l0_units]

            if self.config.enable_hierarchical and (strategy.build_l1 or strategy.build_l2):
                result.hierarchical_result = self._run_hierarchical_tower(
                    h_builder, l0_units, session_id, sample_id,
                    session_date, participants, custom_prompts, strategy,
                )
            elif self.config.enable_hierarchical:
                logger.info("[Hierarchical Tower] strategy %s disables L1/L2; skipping", strategy_key)

            if self.config.enable_episodic:
                result.episodic_result = self._run_episodic_tower(
                    e_builder, result.l0_unit_uids, reference_date,
                    sample_id, speakers,
                )

            if self.config.enable_entity_relation:
                result.entity_relation_result = self._run_entity_relation_tower(
                    er_builder, l0_units, reference_date,
                    sample_id, custom_prompts,
                )

            result.success = not result.errors

        except Exception as e:
            result.errors.append(f"Pipeline failed globally: {e}")

        result.processing_time = (datetime.now() - start_time).total_seconds()
        return result

    
    
    

    def _run_hierarchical_tower(
        self,
        builder: HierarchicalAutoBuilder,
        l0_units: List["MemoryUnit"],
        session_id: str,
        sample_id: str,
        session_date: Optional[str],
        participants: Optional[List[str]],
        custom_prompts: Optional[Dict[str, str]],
        strategy: PipelineStrategy,
    ) -> Dict[str, Any]:
        """Run hierarchical tower."""
        start_time = datetime.now()
        result: Dict[str, Any] = {
            "session_id": session_id,
            "sample_id": sample_id,
            "success": False,
            "l1_results": [],
            "l2_result": None,
            "stats": {},
            "errors": [],
        }

        try:
            logger.info(
                f"\n[Hierarchical Tower] start (session={session_id}, build_l1={strategy.build_l1}, build_l2={strategy.build_l2})"
            )

            if strategy.build_l1:
                builder.llm_client = self.extraction_llm
                l1_results = builder.extract_l1_from_l0_units(
                    l0_units=l0_units,
                    session_id=session_id,
                    session_date=session_date,
                    participants=participants,
                    custom_prompts=custom_prompts,
                )

                if self.config.enable_hierarchical_deduplication and l1_results:
                    builder.llm_client = self.dedup_llm
                    l1_results = builder.deduplicate_l1(
                        l1_results=l1_results,
                        custom_prompt=custom_prompts.get("deduplication") if custom_prompts else None,
                    )

                result["l1_results"] = l1_results

            if strategy.build_l2 and result["l1_results"]:
                builder.llm_client = self.extraction_llm
                result["l2_result"] = builder.aggregate_l2_from_l1(
                    l1_results=result["l1_results"],
                    sample_id=sample_id,
                    participants=participants,
                    custom_prompt=custom_prompts.get("l2") if custom_prompts else None,
                )

            if self.config.add_to_system and builder.semantic_system is not None:
                add_stats = builder.add_to_semantic_system(
                    l1_results=result["l1_results"],
                    l2_result=result["l2_result"],
                )
                result["stats"]["added"] = add_stats

            result["success"] = True

        except Exception as e:
            msg = f"Hierarchical Tower failed: {e}"
            logger.error(f" {msg}")
            result["errors"].append(msg)

        result["build_time"] = (datetime.now() - start_time).total_seconds()
        logger.info(f"   Hierarchical L1={len(result['l1_results'])} L2={'yes' if result['l2_result'] else 'no'} ({result['build_time']:.1f}s)")
        return result

    def _run_episodic_tower(
        self,
        builder: EpisodicAutoBuilder,
        l0_unit_uids: List[str],
        reference_date: str,
        source_id: str,
        speakers: Optional[str],
    ) -> Dict[str, Any]:
        """Run episodic tower."""
        start_time = datetime.now()
        result: Dict[str, Any] = {
            "success": False,
            "facts_extracted": 0,
            "facts_after_dedup": 0,
            "units_added": 0,
            "errors": [],
        }

        try:
            logger.info(f"\n[Episodic Tower] start (source={source_id})")

            builder.llm_client = self.extraction_llm
            facts = builder.extract_from_l0_units(
                l0_unit_uids=l0_unit_uids,
                reference_date=reference_date,
                source_id=source_id,
                speakers=speakers or "",
            )
            result["facts_extracted"] = len(facts)

            if builder.config.enable_deduplication and facts:
                builder.llm_client = self.dedup_llm
                deduplicated = builder.deduplicate_facts(facts)
            else:
                deduplicated = facts
            result["facts_after_dedup"] = len(deduplicated)

            added_uids = builder.add_to_semantic_system(deduplicated)
            result["units_added"] = len(added_uids)
            result["added_uids"] = added_uids
            result["success"] = True

        except Exception as e:
            msg = f"Episodic Tower failed: {e}"
            logger.error(f" {msg}")
            result["errors"].append(msg)

        result["duration_seconds"] = (datetime.now() - start_time).total_seconds()
        logger.info(f"   Episodic extracted={result['facts_extracted']} dedup={result['facts_after_dedup']} added={result['units_added']} ({result['duration_seconds']:.1f}s)")
        return result

    def _run_entity_relation_tower(
        self,
        builder: EntityRelationAutoBuilder,
        l0_units: List["MemoryUnit"],
        reference_date: str,
        source_id: str,
        custom_prompts: Optional[Dict[str, str]],
    ) -> Dict[str, Any]:
        """Run entity relation tower."""
        start_time = datetime.now()
        custom_prompts = custom_prompts or {}
        result: Dict[str, Any] = {
            "success": False,
            "entities": [],
            "relations": [],
            "stats": {},
            "errors": [],
        }

        try:
            logger.info(f"\n[Entity-Relation Tower] start (source={source_id})")

            builder.llm_client = self.extraction_llm
            raw_entities = builder.extract_entities_from_l0_units(
                l0_units=l0_units,
                reference_date=reference_date,
                source_id=source_id,
                custom_prompt=custom_prompts.get("entity_extraction"),
            )

            if not raw_entities:
                result["errors"].append("No entities were extracted")
                return result

            builder.llm_client = self.dedup_llm
            entities = builder.deduplicate_entities(
                raw_entities=raw_entities,
                custom_prompt=custom_prompts.get("entity_deduplication"),
            )
            result["entities"] = [e.to_dict() for e in entities]

            relations: List[ExtractedRelation] = []
            if builder.config.enable_relation_extraction:
                builder.llm_client = self.extraction_llm
                relations = builder.extract_relations_from_entities(
                    l0_units=l0_units,
                    entities=entities,
                    custom_prompt=custom_prompts.get("relation_extraction"),
                )
                result["relations"] = [r.to_dict() for r in relations]

            if self.config.add_to_system and builder.semantic_system is not None:
                add_result = builder.add_to_semantic_system(
                    entities=entities,
                    relations=relations,
                    source_id=source_id,
                )
                result["add_result"] = add_result

            result["success"] = True

        except Exception as e:
            msg = f"Entity-Relation Tower failed: {e}"
            logger.error(f" {msg}")
            result["errors"].append(msg)

        result["processing_time"] = (datetime.now() - start_time).total_seconds()
        logger.info(f"   EntityRel entities={len(result['entities'])} relations={len(result['relations'])} ({result['processing_time']:.1f}s)")
        return result

    
    

    def _create_builders(self, h_config, e_config, er_config):
        """Create builders."""
        h_builder = HierarchicalAutoBuilder(
            semantic_system=self.semantic_system,
            llm_client=self.extraction_llm,
            config=h_config,
        )
        e_builder = EpisodicAutoBuilder(
            semantic_system=self.semantic_system,
            llm_client=self.extraction_llm,
            config=e_config,
        )
        er_builder = EntityRelationAutoBuilder(
            semantic_system=self.semantic_system,
            llm_client=self.extraction_llm,
            config=er_config,
        )
        return h_builder, e_builder, er_builder

    def _should_auto_split_sessions(
        self,
        strategy_key: str,
        explicit_session_id: Optional[str],
    ) -> bool:
        """Only default style without caller-provided session_id uses auto split."""
        if explicit_session_id is not None:
            return False
        if not self.config.enable_auto_session_split:
            return False
        assigner_config = self.auto_session_assigner.config
        if not assigner_config.enabled:
            return False
        if strategy_key not in self.config.auto_session_split_styles:
            return False
        if strategy_key not in assigner_config.enabled_styles:
            return False
        return strategy_key == "default"

    def _assign_default_sessions_if_needed(
        self,
        raw_units: List["MemoryUnit"],
        sample_id: str,
        strategy_key: str,
        explicit_session_id: Optional[str],
        session_date: Optional[str],
        participants: Optional[List[str]],
    ) -> List["MemoryUnit"]:
        if not self._should_auto_split_sessions(strategy_key, explicit_session_id):
            return raw_units

        logger.info(
            "   AutoSession: enabled for default style (sample_id=%s, raw_units=%d)",
            sample_id,
            len(raw_units),
        )
        return self.auto_session_assigner.assign_batch(
            raw_units,
            sample_id=sample_id,
            session_date=session_date,
            participants=participants,
            explicit_session_id=explicit_session_id,
        )

    def _unit_session_id(self, unit: "MemoryUnit") -> Optional[str]:
        metadata = unit.metadata or {}
        raw_data = unit.raw_data or {}
        value = metadata.get("session_id") or raw_data.get("session_id")
        return str(value) if value else None

    def _unit_session_date(self, unit: "MemoryUnit", fallback: Optional[str]) -> Optional[str]:
        metadata = unit.metadata or {}
        raw_data = unit.raw_data or {}
        value = metadata.get("session_date") or raw_data.get("session_date") or fallback
        return str(value) if value else None

    def _group_units_by_session(self, units: List["MemoryUnit"]) -> Dict[str, List["MemoryUnit"]]:
        grouped: "OrderedDict[str, List[MemoryUnit]]" = OrderedDict()
        for unit in units:
            session_id = self._unit_session_id(unit) or "session_1"
            grouped.setdefault(session_id, []).append(unit)
        return dict(grouped)

    def _preprocess_to_l0_by_session_if_needed(
        self,
        h_builder: HierarchicalAutoBuilder,
        raw_units: List["MemoryUnit"],
        strategy: PipelineStrategy,
        strategy_key: str,
        session_id: Optional[str],
        explicit_session_id: Optional[str],
        session_date: Optional[str] = None,
        participants: Optional[List[str]] = None,
        custom_prompts: Optional[Dict[str, str]] = None,
    ) -> List["MemoryUnit"]:
        if not self._should_auto_split_sessions(strategy_key, explicit_session_id):
            return self._preprocess_to_l0(
                h_builder=h_builder,
                raw_units=raw_units,
                strategy=strategy,
                strategy_key=strategy_key,
                session_id=session_id,
                session_date=session_date,
                participants=participants,
                custom_prompts=custom_prompts,
            )

        grouped_raw_units = self._group_units_by_session(raw_units)
        if len(grouped_raw_units) <= 1:
            only_session_id = next(iter(grouped_raw_units), session_id)
            return self._preprocess_to_l0(
                h_builder=h_builder,
                raw_units=raw_units,
                strategy=strategy,
                strategy_key=strategy_key,
                session_id=only_session_id,
                session_date=session_date,
                participants=participants,
                custom_prompts=custom_prompts,
            )

        l0_units: List["MemoryUnit"] = []
        logger.info("   AutoSession: grouped Raw -> L0 into %d sessions", len(grouped_raw_units))
        for group_session_id, session_raw_units in grouped_raw_units.items():
            session_date_for_group = (
                self._unit_session_date(session_raw_units[0], session_date)
                if session_raw_units
                else session_date
            )
            group_l0 = self._preprocess_to_l0(
                h_builder=h_builder,
                raw_units=session_raw_units,
                strategy=strategy,
                strategy_key=strategy_key,
                session_id=group_session_id,
                session_date=session_date_for_group,
                participants=participants,
                custom_prompts=custom_prompts,
            )
            l0_units.extend(group_l0)
        return l0_units

    def _builder_style_for_strategy(self, strategy_key: str) -> str:
        """Map public strategy names to bottom-layer prompt extraction styles."""
        if strategy_key == "locomo10":
            return "locomo"
        if strategy_key == "longmemeval":
            return "longmemeval"
        return "default"

    def _create_builder_configs(
        self,
        strategy_key: str,
        strategy: PipelineStrategy,
    ) -> tuple:
        """Create three tower configs while keeping workflow gates in PipelineStrategy."""
        from .episodic_prompts import EpisodicFactType

        builder_style = self._builder_style_for_strategy(strategy_key)

        if builder_style == "locomo":
            h_config = HierarchicalBuilderConfig(
                extraction_style="locomo",
                # Dataset-specific handling used by the reproduction workflow.
                enable_contextual_retrieval=False,
                contextual_parallel_workers=strategy.contextual_workers,
                enable_chunking=False,
                enable_deduplication=self.config.enable_hierarchical_deduplication,
                parallel_workers=strategy.hierarchical_session_workers,
            )
            e_config = EpisodicBuilderConfig(
                extraction_style="locomo",
                fact_types=EpisodicFactType.locomo_types(),
                enable_deduplication=self.config.enable_episodic_deduplication,
                dedup_method=strategy.episodic_dedup_method,
                dbscan_eps=strategy.episodic_default_eps,
                dbscan_min_samples=strategy.episodic_default_min_samples,
                auto_optimize_dbscan=False,
                dbscan_eps_range=strategy.episodic_dbscan_eps_range,
                dbscan_min_samples_range=strategy.episodic_dbscan_min_samples_range,
                dedup_parallel_workers=strategy.episodic_dedup_workers,
                large_cluster_threshold=strategy.episodic_large_cluster_threshold,
            )
            er_config = EntityRelationBuilderConfig(
                extraction_style="locomo",
                enable_relation_extraction=self.config.enable_relation_extraction,
                enable_llm_deduplication=self.config.enable_entity_deduplication,
                dbscan_eps=strategy.entity_default_eps,
                dbscan_min_samples=strategy.entity_default_min_samples,
                dbscan_eps_range=strategy.entity_dbscan_eps_range,
                dbscan_min_samples_range=strategy.entity_dbscan_min_samples_range,
                parallel_workers=strategy.entity_dedup_workers,
                large_cluster_threshold=strategy.entity_large_cluster_threshold,
            )

        elif builder_style == "longmemeval":
            h_config = HierarchicalBuilderConfig(
                extraction_style="longmemeval",
                enable_contextual_retrieval=False,
                enable_chunking=True,
                chunk_size=strategy.chunk_size,
                chunk_overlap=strategy.chunk_overlap,
                enable_deduplication=self.config.enable_hierarchical_deduplication,
                parallel_workers=strategy.hierarchical_session_workers,
            )
            e_config = EpisodicBuilderConfig(
                extraction_style="longmemeval",
                fact_types=EpisodicFactType.longmemeval_types(),
                enable_deduplication=self.config.enable_episodic_deduplication,
                dedup_method=strategy.episodic_dedup_method,
                dbscan_eps=strategy.episodic_default_eps,
                dbscan_min_samples=strategy.episodic_default_min_samples,
                dbscan_eps_range=strategy.episodic_dbscan_eps_range,
                dbscan_min_samples_range=strategy.episodic_dbscan_min_samples_range,
                dedup_parallel_workers=strategy.episodic_dedup_workers,
                large_cluster_threshold=strategy.episodic_large_cluster_threshold,
            )
            er_config = EntityRelationBuilderConfig(
                extraction_style="longmemeval",
                enable_relation_extraction=self.config.enable_relation_extraction,
                enable_llm_deduplication=self.config.enable_entity_deduplication,
                dbscan_eps=strategy.entity_default_eps,
                dbscan_min_samples=strategy.entity_default_min_samples,
                dbscan_eps_range=strategy.entity_dbscan_eps_range,
                dbscan_min_samples_range=strategy.entity_dbscan_min_samples_range,
                parallel_workers=strategy.entity_dedup_workers,
                large_cluster_threshold=strategy.entity_large_cluster_threshold,
            )

        else:
            h_config = HierarchicalBuilderConfig(
                extraction_style="default",
                enable_contextual_retrieval=False,
                enable_chunking=False,
                enable_deduplication=self.config.enable_hierarchical_deduplication,
                parallel_workers=strategy.hierarchical_session_workers,
            )
            e_config = EpisodicBuilderConfig(
                extraction_style="default",
                enable_deduplication=self.config.enable_episodic_deduplication,
                dedup_method=strategy.episodic_dedup_method,
                dbscan_eps=strategy.episodic_default_eps,
                dbscan_min_samples=strategy.episodic_default_min_samples,
                dbscan_eps_range=strategy.episodic_dbscan_eps_range,
                dbscan_min_samples_range=strategy.episodic_dbscan_min_samples_range,
                dedup_parallel_workers=strategy.episodic_dedup_workers,
                large_cluster_threshold=strategy.episodic_large_cluster_threshold,
            )
            er_config = EntityRelationBuilderConfig(
                extraction_style="default",
                enable_relation_extraction=self.config.enable_relation_extraction,
                enable_llm_deduplication=self.config.enable_entity_deduplication,
                dbscan_eps=strategy.entity_default_eps,
                dbscan_min_samples=strategy.entity_default_min_samples,
                dbscan_eps_range=strategy.entity_dbscan_eps_range,
                dbscan_min_samples_range=strategy.entity_dbscan_min_samples_range,
                parallel_workers=strategy.entity_dedup_workers,
                large_cluster_threshold=strategy.entity_large_cluster_threshold,
            )

        logger.info(
            "   Builder config: strategy=%s builder_style=%s l0_mode=%s build_l1=%s build_l2=%s",
            strategy_key,
            builder_style,
            strategy.l0_mode,
            strategy.build_l1,
            strategy.build_l2,
        )
        return h_config, e_config, er_config

    
    

    def _get_l0_text_splitter(self):
        try:
            try:
                from langchain_text_splitters import RecursiveCharacterTextSplitter
            except ImportError:
                from langchain.text_splitter import RecursiveCharacterTextSplitter

            return RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                length_function=len,
                separators=[
                    "\n\n",
                    "\n",
                    "。",
                    ". ",
                    "！",
                    "! ",
                    "？",
                    "? ",
                    "；",
                    "; ",
                    "，",
                    ", ",
                    " ",
                    "",
                ],
            )
        except Exception as e:
            logger.warning(f"LangChain splitter is unavailable; using character-level fallback chunking: {e}")
            return None

    def _split_long_text(self, text: str) -> List[str]:
        splitter = self._get_l0_text_splitter()
        if splitter is not None:
            chunks = [chunk for chunk in splitter.split_text(text) if chunk]
            if chunks:
                return chunks

        max_chars = max(self.config.chunk_size, 1)
        overlap_chars = min(max(self.config.chunk_overlap, 0), max_chars - 1)
        chunks: List[str] = []
        start = 0
        while start < len(text):
            end = min(start + max_chars, len(text))
            chunks.append(text[start:end])
            if end >= len(text):
                break
            start = max(end - overlap_chars, start + 1)
        return chunks

    def _speaker_for_unit(self, unit: "MemoryUnit") -> str:
        raw = unit.raw_data or {}
        meta = unit.metadata or {}
        return str(raw.get("speaker") or raw.get("role") or meta.get("speaker") or meta.get("role") or "Unknown")

    def _infer_participants(self, raw_units: List["MemoryUnit"], participants: Optional[List[str]]) -> List[str]:
        if participants:
            return participants
        speakers = []
        seen = set()
        for unit in raw_units:
            speaker = self._speaker_for_unit(unit)
            if speaker and speaker != "Unknown" and speaker not in seen:
                speakers.append(speaker)
                seen.add(speaker)
        return speakers or ["Speaker_A", "Speaker_B"]

    def _enhance_short_units_with_cr(
        self,
        short_units: List["MemoryUnit"],
        all_raw_units: List["MemoryUnit"],
        text_by_uid: Dict[str, str],
        session_date: Optional[str],
        participants: Optional[List[str]],
        custom_prompts: Optional[Dict[str, str]],
        contextual_workers: Optional[int] = None,
    ) -> Dict[str, str]:
        if not self.config.enable_contextual_retrieval or not short_units:
            return {}

        cr_config = HierarchicalBuilderConfig(
            extraction_style="locomo",
            enable_contextual_retrieval=True,
            contextual_parallel_workers=self.config.contextual_parallel_workers,
        )
        cr_builder = HierarchicalAutoBuilder(
            semantic_system=self.semantic_system,
            llm_client=self.extraction_llm,
            config=cr_config,
        )

        full_transcript = build_l0_inference_context(all_raw_units)
        session_date = session_date or datetime.now().strftime("%Y-%m-%d")
        participants = self._infer_participants(all_raw_units, participants)
        custom_prompt = custom_prompts.get("contextual_retrieval") if custom_prompts else None

        def enhance_one(unit: "MemoryUnit") -> Optional[tuple]:
            original_text = text_by_uid.get(unit.uid, "")
            if len(original_text.strip()) < 10:
                return None
            prompt = HierarchicalPromptManager.get_contextual_retrieval_prompt(
                session_date=session_date,
                participants=participants,
                full_session_transcript=full_transcript,
                speaker=self._speaker_for_unit(unit),
                message_text=original_text,
                custom_prompt=custom_prompt,
            )
            enhanced = cr_builder._call_llm_with_retry(
                prompt=prompt,
                temperature=0.1,
                max_tokens=300,
                context_id=f"l0_cr_{unit.uid}",
            )
            if enhanced and enhanced.strip():
                return unit.uid, enhanced.strip()
            return None

        enhanced_map: Dict[str, str] = {}
        max_workers = max(1, contextual_workers or self.config.contextual_parallel_workers)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(enhance_one, unit): unit for unit in short_units}
            for future in as_completed(futures):
                unit = futures[future]
                try:
                    item = future.result()
                except Exception as e:
                    logger.warning(f"L0 contextual-retrieval enhancement failed ({unit.uid}): {e}")
                    continue
                if item:
                    uid, enhanced = item
                    enhanced_map[uid] = enhanced

        logger.info(f"   L0 LoCoMo CR: {len(short_units)} raw units, {len(enhanced_map)} enhanced")
        return enhanced_map

    def _build_short_l0_unit(
        self,
        unit: "MemoryUnit",
        original_text: str,
        enhanced_content: Optional[str],
        l0_space: str,
        session_id: Optional[str],
    ) -> "MemoryUnit":
        from ..core.memory_unit import MemoryUnit as MU

        new_raw_data = dict(unit.raw_data or {})
        new_raw_data.pop("enhanced_content", None)
        if original_text:
            new_raw_data["text_content"] = original_text
            new_raw_data["original_content"] = original_text
        if enhanced_content:
            new_raw_data["enhanced_content"] = enhanced_content

        new_metadata = dict(unit.metadata or {})
        new_metadata.update({
            "layer": "L0",
            "source_raw_uid": unit.uid,
            "preprocessing": "contextual_retrieval" if enhanced_content else "locomo_passthrough",
            "retrieval_view": "enhanced_content" if enhanced_content else "text_content",
        })
        if session_id:
            new_metadata["session_id"] = session_id

        l0_unit = MU(uid=f"L0_{unit.uid}", raw_data=new_raw_data, metadata=new_metadata)
        self._add_l0_unit_to_system(l0_unit, l0_space)
        return l0_unit

    def _build_chunk_l0_units(
        self,
        unit: "MemoryUnit",
        original_text: str,
        l0_space: str,
        session_id: Optional[str],
    ) -> List["MemoryUnit"]:
        from ..core.memory_unit import MemoryUnit as MU

        chunks = self._split_long_text(original_text)
        l0_units: List["MemoryUnit"] = []

        for chunk_index, chunk_text in enumerate(chunks):
            new_raw_data = dict(unit.raw_data or {})
            new_raw_data.pop("enhanced_content", None)
            new_raw_data.update({
                "text_content": chunk_text,
                "original_content": chunk_text,
                "source_raw_uid": unit.uid,
                "source_uid": unit.uid,
            })

            new_metadata = dict(unit.metadata or {})
            new_metadata.update({
                "layer": "L0",
                "source_raw_uid": unit.uid,
                "preprocessing": "langchain_chunk",
                "retrieval_view": "text_content",
                "chunk_index": chunk_index,
                "total_chunks": len(chunks),
            })
            if session_id:
                new_metadata["session_id"] = session_id

            l0_unit = MU(
                uid=f"L0_{unit.uid}_chunk_{chunk_index}",
                raw_data=new_raw_data,
                metadata=new_metadata,
            )
            l0_units.append(l0_unit)
            self._add_l0_unit_to_system(l0_unit, l0_space)

        return l0_units

    def _preprocess_locomo_cr(
        self,
        raw_units: List["MemoryUnit"],
        l0_space: str,
        session_date: Optional[str],
        participants: Optional[List[str]],
        custom_prompts: Optional[Dict[str, str]],
        session_id: Optional[str] = None,
        contextual_workers: Optional[int] = None,
    ) -> List["MemoryUnit"]:
        """Preprocess locomo cr."""
        text_by_uid = {unit.uid: extract_original_text(unit) for unit in raw_units}
        enhanced_map = self._enhance_short_units_with_cr(
            raw_units, raw_units, text_by_uid, session_date, participants, custom_prompts, contextual_workers
        )
        l0_units = [
            self._build_short_l0_unit(unit, text_by_uid.get(unit.uid, ""), enhanced_map.get(unit.uid), l0_space, session_id)
            for unit in raw_units
        ]
        logger.info(f"   LoCoMo preprocessing: {len(raw_units)} raw -> {len(l0_units)} L0 (Contextual Retrieval)")
        return l0_units

    
    

    def _fetch_raw_units(self, sample_id: str) -> List["MemoryUnit"]:
        """Run fetch raw units."""
        raw_space_name = self.config.raw_space_name

        if hasattr(self.semantic_system, "get_units_in_memory_space"):
            all_units = self.semantic_system.get_units_in_memory_space(raw_space_name)
        elif hasattr(self.semantic_system, "semantic_map"):
            all_units = self.semantic_system.semantic_map.get_units_in_memory_space(raw_space_name)
        else:
            logger.warning(f"Could not retrieve memory space '{raw_space_name}' from semantic_system")
            return []

        matched = []
        for unit in all_units:
            meta_sample = (unit.metadata or {}).get("sample_id", "")
            if meta_sample == sample_id:
                matched.append(unit)
            elif unit.uid.startswith(f"{sample_id}_") or unit.uid.startswith(f"{sample_id}:"):
                matched.append(unit)

        logger.debug(f"raw_space contains {len(all_units)} units; matched sample_id='{sample_id}': {len(matched)}")
        return matched

    
    

    def _preprocess_to_l0(
        self,
        h_builder: HierarchicalAutoBuilder,
        raw_units: List["MemoryUnit"],
        strategy: PipelineStrategy,
        strategy_key: str,
        session_id: Optional[str] = None,
        session_date: Optional[str] = None,
        participants: Optional[List[str]] = None,
        custom_prompts: Optional[Dict[str, str]] = None,
    ) -> List["MemoryUnit"]:
        """Preprocess to L0."""
        l0_space = self.config.l0_space_name

        if strategy.l0_mode == "cr":
            logger.info("   L0 route: strategy=%s l0_mode=cr -> Contextual Retrieval", strategy_key)
            return self._preprocess_locomo_cr(
                raw_units, l0_space, session_date, participants, custom_prompts, session_id, strategy.contextual_workers
            )

        if strategy.l0_mode == "chunk":
            logger.info("   L0 route: strategy=%s l0_mode=chunk -> LangChain chunk", strategy_key)
            return self._preprocess_longmemeval(
                h_builder, raw_units, l0_space, session_id
            )

        raise ValueError(f"Unknown L0 mode: {strategy.l0_mode}")

    def _preprocess_longmemeval(
        self,
        h_builder: HierarchicalAutoBuilder,
        raw_units: List["MemoryUnit"],
        l0_space: str,
        session_id: Optional[str],
    ) -> List["MemoryUnit"]:
        """- chunk_size=512, chunk_overlap=50."""
        l0_units: List["MemoryUnit"] = []
        for unit in raw_units:
            text = extract_original_text(unit)
            if not text:
                continue
            l0_units.extend(
                self._build_chunk_l0_units(unit, text, l0_space, session_id)
            )

        logger.info(f"   LangChain chunk preprocessing: {len(raw_units)} raw -> {len(l0_units)} L0 (chunk_size={self.config.chunk_size} chars, overlap={self.config.chunk_overlap} chars)")
        return l0_units

    def _preprocess_default(
        self,
        raw_units: List["MemoryUnit"],
        l0_space: str,
    ) -> List["MemoryUnit"]:
        """Preprocess default."""
        l0_units: List["MemoryUnit"] = []
        for unit in raw_units:
            text = extract_original_text(unit)
            if not text:
                continue
            l0_units.extend(self._build_chunk_l0_units(unit, text, l0_space, None))

        logger.info(f"   default preprocessing: {len(raw_units)} raw -> {len(l0_units)} L0 (LangChain chunk)")
        return l0_units

    
    

    def _add_l0_unit_to_system(self, l0_unit: "MemoryUnit", l0_space: str):
        """Run add L0 unit to system."""
        try:
            text_content = extract_embedding_text(l0_unit)
            dispatch_graph_write_requests(
                semantic_system=self.semantic_system,
                requests=[GraphWriteRequest(
                    unit=l0_unit,
                    explicit_content_for_embedding=text_content or None,
                    content_type_for_embedding="text" if text_content else None,
                    space_names=[l0_space],
                    index_update_mode="none",
                    source="orchestrator_l0",
                )],
            )
        except Exception as e:
            logger.warning(f"Failed to add L0 unit {l0_unit.uid} to the system: {e}")
