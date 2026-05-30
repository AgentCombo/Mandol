"""Unified Fact Pipeline for entity/event extraction with coreference resolution.

This module implements the v4.2 design:
- 4 separate prompts for focused tasks
- Multi-signal entity retrieval (name/alias + BM25 + vector)
- COREF edges: dialogue_unit -> canonical_entity (represents "dialogue mentions entity")
- text_content format: "Entity {name}({type}): {description}" / "Event {name}: {description}"
- Aliases contain only short names/references, not descriptions
- Cross-session merge preserves COREF edges by migrating them to canonical entity
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING

import numpy as np

from ...domain.coref_graph_constants import (
    REL_CAUSED_BY,
    REL_CAUSES,
    REL_RELATED_TO,
)
from .._llm_retry import retry_llm_json_call
from ._utils import (
    find_matching_dialogue_uids,
    generate_entity_uid,
    generate_event_uid,
    parse_json_response,
    format_entity_text_content as _format_entity_text_content,
    format_event_text_content as _format_event_text_content,
    extract_entity_name_from_text_content as _extract_entity_name_from_text_content,
    extract_entity_desc_from_text_content as _extract_entity_desc_from_text_content,
    extract_event_name_from_text_content as _extract_event_name_from_text_content,
    extract_event_desc_from_text_content as _extract_event_desc_from_text_content,
    is_short_alias as _is_short_alias,
)
from ..services._canonical_creator import CanonicalCreator
from ..services._pipeline_graph_writer import PipelineGraphWriter
from ...domain.memory_unit import MemoryUnit
from ...domain.types import SpaceName, Uid
from ...ports.embedding_provider import EmbeddingProvider
from ...ports.llm_provider import LLMChatResponse, LLMProvider
from ...retrieval.bm25 import Bm25Retriever
from ...retrieval.text import TextExtractor, Tokenizer
from ..prompts import (
    CLUSTER_JUDGE_PROMPT,
    ENTITY_EXTRACTION_PROMPT,
    ENTITY_RELATION_PROMPT,
    EVENT_CAUSAL_PROMPT,
    EVENT_CLUSTER_JUDGE_PROMPT,
    EVENT_EXTRACTION_PROMPT,
)
from ..semantic_graph import SemanticGraphService

if TYPE_CHECKING:
    from .cross_session_coref_manager import CrossSessionCorefManager

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ExtractedEntity:
    """An entity extracted from dialogue via LLM.

    Attributes:
        entity_name: Canonical entity name.
        entity_type: Type label (PERSON, ORGANIZATION, etc.).
        linked_id: Existing entity UID this maps to, or None.
        is_new: True if this is a new entity not linked to existing ones.
        confidence: LLM confidence in [0.0, 1.0].
        mention_text: Original textual mention from the dialogue.
        new_facts: Fact strings discovered in this session.
        aliases: Short alternative names for this entity.
        reasoning: LLM-provided rationale string.
    """
    entity_name: str
    entity_type: str
    linked_id: Optional[str]
    is_new: bool
    confidence: float
    mention_text: str
    new_facts: List[str]
    aliases: List[str]
    reasoning: str = ""


@dataclass(slots=True)
class ExtractedEvent:
    """An event extracted from dialogue via LLM.

    Attributes:
        event_name: Canonical event name.
        linked_id: Existing event UID this maps to, or None.
        is_new: True if this is a new event.
        confidence: LLM confidence in [0.0, 1.0].
        description: Full-text event description.
        participants: List of participant dicts.
        location: Optional location entity name string, or None.
        inferred_time: ISO-format time string, or None.
        new_facts: Fact strings discovered in this session.
        reasoning: LLM-provided rationale string.
    """
    event_name: str
    linked_id: Optional[str]
    is_new: bool
    confidence: float
    description: str
    participants: List[Dict[str, Any]]
    location: Optional[str]
    inferred_time: Optional[str]
    new_facts: List[str]
    reasoning: str = ""


@dataclass(slots=True)
class ExtractedRelation:
    """An entity-entity relation extracted via LLM.

    Attributes:
        source: Source entity UID or name.
        target: Target entity UID or name.
        rel_type: Relationship type label (default RELATED_TO).
        subtype: Optional detailed subtype.
        confidence: LLM confidence in [0.0, 1.0].
        reasoning: LLM-provided rationale.
    """
    source: str
    target: str
    rel_type: str
    subtype: Optional[str]
    confidence: float
    reasoning: str = ""


@dataclass(slots=True)
class ExtractedCausalRelation:
    """A causal relation between two events.

    Attributes:
        cause_event: The cause event UID or name.
        effect_event: The effect event UID or name.
        confidence: LLM confidence in [0.0, 1.0].
        reasoning: LLM-provided rationale.
    """
    cause_event: str
    effect_event: str
    confidence: float
    reasoning: str = ""


@dataclass(slots=True)
class PipelineResult:
    """Aggregated output from a single pipeline process_session call.

    Attributes:
        entities: Extracted entities list.
        events: Extracted events list.
        entity_relations: Entity-entity relation list.
        causal_relations: Event-event causal relation list.
        coref_edges: Edges for dialogue-to-canonical coreference.
        evidenced_by_edges: Edges linking canonical units to source dialogue.
        involves_edges: Edges linking events to participating entities.
        related_to_edges: Edges for entity relation graph.
        causes_edges: Edges for causal (CAUSES) relationships.
    """
    entities: List[ExtractedEntity] = field(default_factory=list)
    events: List[ExtractedEvent] = field(default_factory=list)
    entity_relations: List[ExtractedRelation] = field(default_factory=list)
    causal_relations: List[ExtractedCausalRelation] = field(default_factory=list)
    coref_edges: List[Dict[str, Any]] = field(default_factory=list)
    evidenced_by_edges: List[Dict[str, Any]] = field(default_factory=list)
    involves_edges: List[Dict[str, Any]] = field(default_factory=list)
    related_to_edges: List[Dict[str, Any]] = field(default_factory=list)
    causes_edges: List[Dict[str, Any]] = field(default_factory=list)


class UnifiedFactPipeline:
    """Unified pipeline for entity/event extraction with coreference resolution.

    Each session triggers 4 separate LLM prompts (entity extraction, event
    extraction, entity relation, event causal), then coordinates with a
    CorefManager for cross-session deduplication.

    Args:
        llm_provider: LLM provider for all extraction calls.
        embedding_provider: Embedding provider for vector search retrieval.
        semantic_graph: SemanticGraphService for unit storage and graph edges.
        entity_space: Space name for entity units (default \"knowledge_entity\").
        event_space: Space name for event units (default \"episodic_event\").
        top_k_entities: Max existing entities retrieved per session (default 10).
        top_k_events: Max existing events retrieved per session (default 5).
        llm_temperature: Temperature for LLM calls (default 0.1).
        confidence_threshold: Min confidence to accept extracted results (default 0.6).
        coref_manager: Optional CrossSessionCorefManager for deduplication.
    """

    def __init__(
        self,
        *,
        llm_provider: LLMProvider,
        embedding_provider: EmbeddingProvider,
        semantic_graph: SemanticGraphService,
        entity_space: str = "knowledge_entity",
        event_space: str = "episodic_event",
        top_k_entities: int = 10,
        top_k_events: int = 5,
        llm_temperature: float = 0.1,
        confidence_threshold: float = 0.6,
        coref_manager: Optional["CrossSessionCorefManager"] = None,
        on_warning: Optional[Callable[[str], None]] = None,
    ):
        self._llm = llm_provider
        self._embedder = embedding_provider
        self._graph = semantic_graph
        self._semantic_map = semantic_graph.semantic_map
        self._entity_space = SpaceName(entity_space)
        self._event_space = SpaceName(event_space)
        self._top_k_entities = top_k_entities
        self._top_k_events = top_k_events
        self._temperature = llm_temperature
        self._confidence_threshold = confidence_threshold
        self._coref_manager = coref_manager
        self._on_warning = on_warning
        self._bm25_retriever = Bm25Retriever(
            text_extractor=TextExtractor(primary_key="text_content"),
            tokenizer=Tokenizer(use_jieba=True),
        )
        self._canonical_creator = CanonicalCreator(
            semantic_map=self._semantic_map,
            graph=self._graph,
            llm=self._llm,
            entity_space=self._entity_space,
            event_space=self._event_space,
        )
        self._graph_writer = PipelineGraphWriter(
            semantic_map=self._semantic_map,
            graph=self._graph,
        )

    def _warn(self, msg: str) -> None:
        logger.error(msg)
        if self._on_warning:
            self._on_warning(msg)

    def process_session(
        self,
        dialogue_units: Sequence[MemoryUnit],
        session_id: str,
    ) -> PipelineResult:
        """Run the full extraction pipeline for a single session.

        1. Retrieve existing entities (name/alias + BM25 + vector).
        2. Extract new entities, linked to existing ones where possible.
        3. Retrieve existing events (vector).
        4. Extract new events.
        5. Extract entity relations and causal event chains.

        Args:
            dialogue_units: MemoryUnits belonging to this session.
            session_id: Session identifier for UID generation.

        Returns:
            PipelineResult containing all extracted and linked facts.
        """
        dialogue_text = self._build_dialogue_text(dialogue_units)

        existing_entities = self._retrieve_existing_entities(dialogue_text)

        # Step 1: Entity extraction (must complete first — event/relation extraction depend on it)
        extracted_entities = self._extract_entities(dialogue_text, existing_entities)

        current_entities = [
            {"id": e.linked_id or "", "name": e.entity_name, "type": e.entity_type}
            for e in extracted_entities
        ]

        existing_events = self._retrieve_existing_events(dialogue_text)

        # Step 2: Run event extraction and entity relation extraction in parallel.
        # Both depend only on entity results, not on each other.
        extracted_events: List[ExtractedEvent] = []
        entity_relations: List[ExtractedRelation] = []
        errors: List[str] = []

        def _do_extract_events():
            return self._extract_events(dialogue_text, existing_events, current_entities)

        def _do_extract_entity_relations():
            return self._extract_entity_relations(dialogue_text, current_entities)

        with ThreadPoolExecutor(max_workers=2) as pool:
            f_events = pool.submit(_do_extract_events)
            f_relations = pool.submit(_do_extract_entity_relations)

            try:
                extracted_events = f_events.result()
            except (json.JSONDecodeError, ConnectionError, TimeoutError, OSError, RuntimeError) as e:
                # LLM call failures in the worker — log and continue with empty events.
                self._warn(f"[FALLBACK] Event extraction failed in parallel: {e}")
                errors.append(f"event_extraction: {e}")

            try:
                entity_relations = f_relations.result()
            except (json.JSONDecodeError, ConnectionError, TimeoutError, OSError, RuntimeError) as e:
                # LLM call failures in the worker — log and continue with empty relations.
                self._warn(f"[FALLBACK] Entity relation extraction failed in parallel: {e}")
                errors.append(f"entity_relations: {e}")

        current_events = [
            {"id": e.linked_id or "", "name": e.event_name}
            for e in extracted_events
        ]

        # Step 3: Causal relations (depends on event extraction results)
        causal_relations = self._extract_causal_relations(dialogue_text, current_events)

        return PipelineResult(
            entities=extracted_entities,
            events=extracted_events,
            entity_relations=entity_relations,
            causal_relations=causal_relations,
        )

    def _build_dialogue_text(self, dialogue_units: Sequence[MemoryUnit]) -> str:
        """Concatenate dialogue units into a single text block for LLM prompts.

        Args:
            dialogue_units: MemoryUnits to concatenate.

        Returns:
            Newline-joined text with optional dia_id prefixes.
        """
        lines = []
        for unit in dialogue_units:
            text = unit.raw_data.get("text_content", "")
            dia_id = unit.metadata.get("dia_id", "")
            if text:
                lines.append(f"[{dia_id}] {text}" if dia_id else text)
        return "\n".join(lines)

    def _retrieve_existing_entities(self, dialogue_text: str) -> List[Dict[str, Any]]:
        """Multi-signal retrieval of existing entities relevant to the dialogue.

        Three strategies combined:
        1. Name/alias substring match in the dialogue text.
        2. BM25 keyword search against entity text_content.
        3. Vector search against entity embeddings.

        Args:
            dialogue_text: Concatenated session dialogue text.

        Returns:
            List of entity dicts with id, name, type, aliases, up to top_k_entities.
        """
        all_entities = self._semantic_map.get_units_in_spaces([self._entity_space])
        seen_ids: Set[str] = set()
        results: List[Dict[str, Any]] = []

        name_index: Dict[str, Dict[str, Any]] = {}
        alias_index: Dict[str, Dict[str, Any]] = {}
        for e in all_entities:
            eid = str(e.uid)
            text_content = e.raw_data.get("text_content", "")
            entity_name = _extract_entity_name_from_text_content(text_content)
            etype = e.raw_data.get("entity_type", "")
            aliases = e.raw_data.get("aliases", [])
            entry = {"id": eid, "name": entity_name, "type": etype, "aliases": aliases}
            name_key = entity_name.strip().lower()
            if name_key:
                name_index[name_key] = entry
            for alias in aliases:
                if isinstance(alias, str):
                    alias_key = alias.strip().lower()
                    if alias_key:
                        alias_index[alias_key] = entry

        dialogue_lower = dialogue_text.lower()
        for name_key, entry in name_index.items():
            if name_key in dialogue_lower and entry["id"] not in seen_ids:
                results.append(entry)
                seen_ids.add(entry["id"])
        for alias_key, entry in alias_index.items():
            if alias_key in dialogue_lower and entry["id"] not in seen_ids:
                results.append(entry)
                seen_ids.add(entry["id"])

        bm25_hits = self._bm25_retriever.search(
            dialogue_text, all_entities, top_k=self._top_k_entities
        )
        for scored in bm25_hits:
            eid = str(scored.unit.uid)
            if eid in seen_ids:
                continue
            text_content = scored.unit.raw_data.get("text_content", "")
            entity_name = _extract_entity_name_from_text_content(text_content)
            seen_ids.add(eid)
            results.append({
                "id": eid,
                "name": entity_name,
                "type": scored.unit.raw_data.get("entity_type", ""),
                "aliases": scored.unit.raw_data.get("aliases", []),
            })

        try:
            query_emb = self._embedder.embed_text([dialogue_text])[0]
        except (RuntimeError, ValueError, OSError):
            return results

        vector_top_k = max(self._top_k_entities - len(results), 5)
        hits = self._semantic_map.search_by_vector(
            np.array(query_emb, dtype=np.float32),
            top_k=vector_top_k,
            space_names=[self._entity_space],
        )

        for hit, _ in hits:
            eid = str(hit.uid)
            if eid in seen_ids:
                continue
            seen_ids.add(eid)
            text_content = hit.raw_data.get("text_content", "")
            entity_name = _extract_entity_name_from_text_content(text_content)
            results.append({
                "id": eid,
                "name": entity_name,
                "type": hit.raw_data.get("entity_type", ""),
                "aliases": hit.raw_data.get("aliases", []),
            })

        return results[:self._top_k_entities]

    def _retrieve_existing_events(self, dialogue_text: str) -> List[Dict[str, Any]]:
        try:
            query_emb = self._embedder.embed_text([dialogue_text])[0]
        except (RuntimeError, ValueError, OSError):
            return []

        hits = self._semantic_map.search_by_vector(
            np.array(query_emb, dtype=np.float32),
            top_k=max(self._top_k_events, 20),
            space_names=[self._event_space],
        )

        return [
            {
                "id": str(hit.uid),
                "name": _extract_event_name_from_text_content(hit.raw_data.get("text_content", "")),
                "signature": hit.metadata.get("signature", ""),
            }
            for hit, _ in hits
        ]

    def _extract_entities(
        self,
        dialogue_text: str,
        existing_entities: List[Dict[str, Any]],
    ) -> List[ExtractedEntity]:
        prompt = ENTITY_EXTRACTION_PROMPT.format(
            dialogue_context=dialogue_text,
            existing_entities=json.dumps(existing_entities, indent=2) if existing_entities else "[]",
        )

        logger.info("Extracting entities (%d existing known)...", len(existing_entities))
        t0 = time.time()
        try:
            data = self._call_llm_json(prompt, context_label="extract_entities")
        except (json.JSONDecodeError, RuntimeError) as e:
            self._warn(
                f"[FALLBACK] Entity extraction FAILED after all retries: {e}. "
                f"Returning 0 entities — session will have no entity memory."
            )
            return []

        entities = []
        for item in data.get("entities", []):
            if not isinstance(item, dict):
                continue
            if item.get("confidence", 0) < self._confidence_threshold:
                continue
            raw_aliases = item.get("aliases", [])
            clean_aliases = [a for a in raw_aliases if _is_short_alias(a)]
            entities.append(ExtractedEntity(
                entity_name=item.get("entity_name", ""),
                entity_type=item.get("entity_type", "Concept"),
                linked_id=item.get("linked_id"),
                is_new=item.get("is_new", True),
                confidence=item.get("confidence", 0.5),
                mention_text=item.get("mention_text", ""),
                new_facts=item.get("new_facts", []),
                aliases=clean_aliases,
                reasoning=item.get("reasoning", ""),
            ))

        logger.info(
            "Entity extraction complete: %d entities in %.1fs",
            len(entities), time.time() - t0,
        )
        return entities

    def _extract_events(
        self,
        dialogue_text: str,
        existing_events: List[Dict[str, Any]],
        current_entities: List[Dict[str, Any]],
    ) -> List[ExtractedEvent]:
        prompt = EVENT_EXTRACTION_PROMPT.format(
            dialogue_context=dialogue_text,
            existing_events=json.dumps(existing_events, indent=2) if existing_events else "[]",
            current_entities=json.dumps(current_entities, indent=2) if current_entities else "[]",
        )

        logger.info("Extracting events...")
        t0 = time.time()
        try:
            data = self._call_llm_json(prompt, context_label="extract_events")
        except (json.JSONDecodeError, RuntimeError) as e:
            self._warn(
                f"[FALLBACK] Event extraction FAILED after all retries: {e}. "
                f"Returning 0 events — session will have no event memory."
            )
            return []

        events = []
        for item in data.get("events", []):
            if not isinstance(item, dict):
                continue
            if item.get("confidence", 0) < self._confidence_threshold:
                continue
            location_raw = item.get("location")
            if location_raw is not None and not isinstance(location_raw, str):
                location_raw = None
            events.append(ExtractedEvent(
                event_name=item.get("event_name", ""),
                linked_id=item.get("linked_id"),
                is_new=item.get("is_new", True),
                confidence=item.get("confidence", 0.5),
                description=item.get("description", ""),
                participants=item.get("participants", []),
                location=location_raw,
                inferred_time=item.get("inferred_time"),
                new_facts=item.get("new_facts", []),
                reasoning=item.get("reasoning", ""),
            ))

        logger.info(
            "Event extraction complete: %d events in %.1fs",
            len(events), time.time() - t0,
        )
        return events

    def _extract_entity_relations(
        self,
        dialogue_text: str,
        current_entities: List[Dict[str, Any]],
    ) -> List[ExtractedRelation]:
        prompt = ENTITY_RELATION_PROMPT.format(
            dialogue_context=dialogue_text,
            current_entities=json.dumps(current_entities, indent=2) if current_entities else "[]",
        )

        logger.info("Extracting entity relations (%d entities)...", len(current_entities))
        t0 = time.time()
        try:
            data = self._call_llm_json(prompt, context_label="extract_entity_relations")
        except (json.JSONDecodeError, RuntimeError) as e:
            self._warn(
                f"[FALLBACK] Entity relation extraction FAILED after all retries: {e}."
            )
            return []

        relations = []
        for item in data.get("relations", []):
            if not isinstance(item, dict):
                continue
            if item.get("confidence", 0) < self._confidence_threshold:
                continue
            relations.append(ExtractedRelation(
                source=item.get("source", ""),
                target=item.get("target", ""),
                rel_type=item.get("rel_type", REL_RELATED_TO),
                subtype=item.get("subtype"),
                confidence=item.get("confidence", 0.5),
                reasoning=item.get("reasoning", ""),
            ))

        logger.info(
            "Entity relation extraction complete: %d relations in %.1fs",
            len(relations), time.time() - t0,
        )
        return relations

    def _extract_causal_relations(
        self,
        dialogue_text: str,
        current_events: List[Dict[str, Any]],
    ) -> List[ExtractedCausalRelation]:
        prompt = EVENT_CAUSAL_PROMPT.format(
            dialogue_context=dialogue_text,
            current_events=json.dumps(current_events, indent=2) if current_events else "[]",
        )

        logger.info("Extracting causal relations (%d events)...", len(current_events))
        t0 = time.time()
        try:
            data = self._call_llm_json(prompt, context_label="extract_causal_relations")
        except (json.JSONDecodeError, RuntimeError) as e:
            self._warn(
                f"[FALLBACK] Causal relation extraction FAILED after all retries: {e}."
            )
            return []

        causal_relations = []
        for item in data.get("causal_relations", []):
            if not isinstance(item, dict):
                continue
            if item.get("confidence", 0) < self._confidence_threshold:
                continue
            causal_relations.append(ExtractedCausalRelation(
                cause_event=item.get("cause_event", ""),
                effect_event=item.get("effect_event", ""),
                confidence=item.get("confidence", 0.5),
                reasoning=item.get("reasoning", ""),
            ))

        logger.info(
            "Causal relation extraction complete: %d causal chains in %.1fs",
            len(causal_relations), time.time() - t0,
        )
        return causal_relations

    def _create_entity_units(
        self,
        extracted_entities: List[ExtractedEntity],
        dialogue_units: Sequence[MemoryUnit],
        session_id: str,
    ) -> Tuple[List[MemoryUnit], List[Dict[str, Any]], List[Dict[str, Any]]]:
        entity_units = []
        coref_edges = []
        evidenced_by_edges = []

        for entity in extracted_entities:
            description = ". ".join(entity.new_facts) if entity.new_facts else ""
            text_content = _format_entity_text_content(entity.entity_name, entity.entity_type, description)

            if entity.linked_id:
                canonical_uid = Uid(entity.linked_id)
                matching_dialogue_uids = find_matching_dialogue_uids(
                    entity.mention_text, dialogue_units
                )
                for dia_uid in matching_dialogue_uids:
                    coref_edges.append({
                        "source_uid": dia_uid,
                        "target_uid": canonical_uid,
                        "confidence": entity.confidence,
                        "mention_text": entity.mention_text,
                        "session_id": session_id,
                    })

                if entity.new_facts:
                    self._update_entity_description(canonical_uid, entity.new_facts)
            else:
                canonical_uid = generate_entity_uid(entity.entity_name, entity.entity_type, session_id)

                now = datetime.now(timezone.utc).isoformat()
                unit = MemoryUnit(
                    uid=canonical_uid,
                    raw_data={
                        "text_content": text_content,
                        "entity_name": entity.entity_name,
                        "entity_type": entity.entity_type,
                        "aliases": entity.aliases,
                    },
                    metadata={
                        "type": "entity",
                        "session_id": session_id,
                        "created_at": now,
                        "updated_at": now,
                    },
                )
                entity_units.append(unit)

                for dia_unit in dialogue_units:
                    if entity.mention_text in dia_unit.raw_data.get("text_content", ""):
                        evidenced_by_edges.append({
                            "source_uid": canonical_uid,
                            "target_uid": dia_unit.uid,
                            "mention_text": entity.mention_text,
                        })

        return entity_units, coref_edges, evidenced_by_edges

    def _create_event_units(
        self,
        extracted_events: List[ExtractedEvent],
        dialogue_units: Sequence[MemoryUnit],
        session_id: str,
        entity_units: List[MemoryUnit],
    ) -> Tuple[List[MemoryUnit], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        event_units = []
        coref_edges = []
        evidenced_by_edges = []
        involves_edges = []

        _entity_uid_map = {
            _extract_entity_name_from_text_content(e.raw_data.get("text_content", "")): e.uid
            for e in entity_units
        }

        for event in extracted_events:
            text_content = _format_event_text_content(event.event_name, event.description)

            if event.linked_id:
                canonical_uid = Uid(event.linked_id)
                matching_dialogue_uids = find_matching_dialogue_uids(
                    event.event_name, dialogue_units
                )
                for dia_uid in matching_dialogue_uids:
                    coref_edges.append({
                        "source_uid": dia_uid,
                        "target_uid": canonical_uid,
                        "confidence": event.confidence,
                        "mention_text": event.event_name,
                        "session_id": session_id,
                    })

                if event.new_facts:
                    self._update_event_description(canonical_uid, event.new_facts, event.participants, event.inferred_time)
            else:
                canonical_uid = generate_event_uid(event.event_name, session_id)

                now = datetime.now(timezone.utc).isoformat()
                unit = MemoryUnit(
                    uid=canonical_uid,
                    raw_data={
                        "text_content": text_content,
                        "event_name": event.event_name,
                        "description": event.description,
                        "inferred_time": event.inferred_time,
                    },
                    metadata={
                        "type": "event",
                        "session_id": session_id,
                        "signature": self._generate_event_signature(event),
                        "created_at": now,
                        "updated_at": now,
                    },
                )
                event_units.append(unit)

                for dia_unit in dialogue_units:
                    if event.event_name.lower() in dia_unit.raw_data.get("text_content", "").lower():
                        evidenced_by_edges.append({
                            "source_uid": canonical_uid,
                            "target_uid": dia_unit.uid,
                            "mention_text": event.event_name,
                        })

                for participant in event.participants:
                    linked_id = participant.get("linked_id")
                    role = participant.get("role", "participant")
                    if linked_id:
                        involves_edges.append({
                            "source_uid": canonical_uid,
                            "target_uid": Uid(linked_id),
                            "subtype": role,
                            "confidence": event.confidence,
                        })

                if event.location:
                    location_uid = _entity_uid_map.get(event.location.strip())
                    if location_uid:
                        involves_edges.append({
                            "source_uid": canonical_uid,
                            "target_uid": location_uid,
                            "subtype": "location",
                            "confidence": event.confidence,
                        })

        return event_units, coref_edges, evidenced_by_edges, involves_edges

    def _create_related_to_edges(self, relations: List[ExtractedRelation]) -> List[Dict[str, Any]]:
        edges = []
        for rel in relations:
            edges.append({
                "source_uid": Uid(rel.source),
                "target_uid": Uid(rel.target),
                "subtype": rel.subtype,
                "confidence": rel.confidence,
            })
        return edges

    def _create_causes_edges(self, causal_relations: List[ExtractedCausalRelation]) -> List[Dict[str, Any]]:
        edges = []
        for cr in causal_relations:
            edges.append({
                "source_uid": Uid(cr.cause_event),
                "target_uid": Uid(cr.effect_event),
                "rel_type": REL_CAUSES,
                "confidence": cr.confidence,
            })
            edges.append({
                "source_uid": Uid(cr.effect_event),
                "target_uid": Uid(cr.cause_event),
                "rel_type": REL_CAUSED_BY,
                "confidence": cr.confidence,
            })
        return edges

    def _call_llm(self, prompt: str) -> LLMChatResponse:
        return self._llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=self._temperature,
        )

    def _call_llm_json(
        self,
        prompt: str,
        *,
        context_label: str = "",
    ) -> dict:
        return retry_llm_json_call(
            self._llm,
            [{"role": "user", "content": prompt}],
            parse_json_response,
            temperature=self._temperature,
            max_tokens=2048,
            context_label=context_label,
        )

    def _generate_event_signature(self, event: ExtractedEvent) -> str:
        participants_str = ",".join(sorted([p.get("mention", "") for p in event.participants]))[:50]
        time_str = event.inferred_time or "unknown"
        return f"{participants_str}-{event.event_name[:30]}-{time_str}"

    def _update_entity_description(self, uid: Uid, new_facts: List[str]) -> None:
        if self._coref_manager is not None:
            self._coref_manager._update_entity_description(uid, new_facts)

    def _update_event_description(
        self,
        uid: Uid,
        new_facts: List[str],
        new_participants: List[Dict[str, Any]],
        new_time: Optional[str],
    ) -> None:
        if self._coref_manager is not None:
            self._coref_manager._update_event_description(uid, new_facts, new_participants, new_time)

    def _filter_candidates_by_type(
        self,
        candidates: List[Dict[str, Any]],
    ) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        filtered_pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        n = len(candidates)
        for i in range(n):
            for j in range(i + 1, n):
                c1, c2 = candidates[i], candidates[j]
                type1 = c1.get("type", "")
                type2 = c2.get("type", "")
                name1 = c1.get("name", "").strip().lower()
                name2 = c2.get("name", "").strip().lower()
                if not type1 or not type2 or type1 != type2:
                    continue
                if name1 and name2 and name1 == name2:
                    filtered_pairs.append((c1, c2))
                elif name1 and name2:
                    name_overlap = set(name1.split()) & set(name2.split())
                    if name_overlap:
                        filtered_pairs.append((c1, c2))
                elif type1 == type2:
                    filtered_pairs.append((c1, c2))
        return filtered_pairs

    def _compute_graph_structure_similarity(
        self,
        uid1: Uid,
        uid2: Uid,
    ) -> float:
        try:
            neighbors1 = set(
                self._graph.get_explicit_neighbors(
                    [uid1], rel_type=REL_RELATED_TO, direction="out"
                )
            )
            neighbors2 = set(
                self._graph.get_explicit_neighbors(
                    [uid2], rel_type=REL_RELATED_TO, direction="out"
                )
            )
            if not neighbors1 or not neighbors2:
                return 0.0
            shared = neighbors1 & neighbors2
            if not shared:
                return 0.0
            union = neighbors1 | neighbors2
            jaccard = len(shared) / len(union) if union else 0.0
            return min(0.5, jaccard * 0.5)
        except (KeyError, TypeError, ZeroDivisionError):
            return 0.0

    def _precluster_candidates(
        self,
        candidates: List[Dict[str, Any]],
        threshold: float = 0.75,
    ) -> List[List[Dict[str, Any]]]:
        if not candidates:
            return []
        embeddings: Dict[str, np.ndarray] = {}
        for c in candidates:
            cid = c.get("id", "")
            if not cid:
                continue
            unit = self._semantic_map.get_unit(Uid(cid))
            if unit is None or unit.embedding is None:
                continue
            embeddings[cid] = np.asarray(unit.embedding, dtype=np.float32).reshape(-1)
        candidate_ids = list(embeddings.keys())
        if not candidate_ids:
            return [[c] for c in candidates]
        id_to_candidate = {c.get("id", ""): c for c in candidates if c.get("id")}
        visited: Set[str] = set()
        clusters: List[List[Dict[str, Any]]] = []
        def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return float(np.dot(a, b) / (norm_a * norm_b))
        for cid in candidate_ids:
            if cid in visited:
                continue
            cluster: List[Dict[str, Any]] = [id_to_candidate[cid]]
            visited.add(cid)
            emb1 = embeddings[cid]
            for other_id in candidate_ids:
                if other_id in visited:
                    continue
                emb2 = embeddings[other_id]
                sim = cosine_sim(emb1, emb2)
                if sim >= threshold:
                    cluster.append(id_to_candidate[other_id])
                    visited.add(other_id)
            clusters.append(cluster)
        for c in candidates:
            cid = c.get("id", "")
            if cid and cid not in visited and cid in id_to_candidate:
                clusters.append([id_to_candidate[cid]])
                visited.add(cid)
        return clusters

    def _build_candidate_clusters(
        self,
        candidates: List[Dict[str, Any]],
        vector_threshold: float = 0.65,
        combined_threshold: float = 0.45,
    ) -> List[List[Dict[str, Any]]]:
        if not candidates:
            return []
        type_filtered_pairs = self._filter_candidates_by_type(candidates)
        if not type_filtered_pairs:
            return [[c] for c in candidates]
        embeddings: Dict[str, np.ndarray] = {}
        for c in candidates:
            cid = c.get("id", "")
            if not cid:
                continue
            unit = self._semantic_map.get_unit(Uid(cid))
            if unit is None or unit.embedding is None:
                continue
            embeddings[cid] = np.asarray(unit.embedding, dtype=np.float32).reshape(-1)
        def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return float(np.dot(a, b) / (norm_a * norm_b))
        candidate_ids = list(embeddings.keys())
        id_to_candidate = {c.get("id", ""): c for c in candidates if c.get("id")}
        parent: Dict[str, str] = {cid: cid for cid in candidate_ids}
        def find(x: str) -> str:
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        def union(x: str, y: str) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        for c1, c2 in type_filtered_pairs:
            id1, id2 = c1.get("id", ""), c2.get("id", "")
            if not id1 or not id2:
                continue
            if id1 not in embeddings or id2 not in embeddings:
                continue
            vector_sim = cosine_sim(embeddings[id1], embeddings[id2])
            if vector_sim < vector_threshold:
                continue
            graph_sim = self._compute_graph_structure_similarity(Uid(id1), Uid(id2))
            effective_threshold = combined_threshold
            if graph_sim == 0.0:
                effective_threshold = vector_threshold
            combined_score = vector_sim + graph_sim
            if combined_score >= effective_threshold:
                union(id1, id2)
        cluster_map: Dict[str, List[Dict[str, Any]]] = {}
        for cid in candidate_ids:
            root = find(cid)
            if root not in cluster_map:
                cluster_map[root] = []
            cluster_map[root].append(id_to_candidate[cid])
        result = list(cluster_map.values())
        for c in candidates:
            cid = c.get("id", "")
            if cid and cid not in parent and cid in id_to_candidate:
                result.append([id_to_candidate[cid]])
        return result

    def write_edges_to_graph(self, result: PipelineResult) -> None:
        """Write all extracted edges to the graph store.

        Args:
            result: PipelineResult containing coref, evidenced_by, involves,
                related_to, and causes edge lists.
        """
        self._graph_writer.write_edges(
            result.coref_edges,
            result.evidenced_by_edges,
            result.involves_edges,
            result.related_to_edges,
            result.causes_edges,
        )

    def _llm_cluster_judge(
        self,
        cluster: List[Dict[str, Any]],
        unit_type: str = "entity",
    ) -> Dict[str, Any]:
        if len(cluster) < 2:
            return {"should_merge": False, "confidence": 0.0, "canonical_name": None, "reasoning": "Single entity cluster"}

        cluster_info_lines = []
        for i, c in enumerate(cluster):
            cid = c.get("id", "")
            unit = self._semantic_map.get_unit(Uid(cid))
            if unit is None:
                continue

            if unit_type == "entity":
                name = _extract_entity_name_from_text_content(unit.raw_data.get("text_content", ""))
                etype = unit.raw_data.get("entity_type", "")
                desc = _extract_entity_desc_from_text_content(unit.raw_data.get("text_content", ""))
                info = f"- Entity {chr(65 + i)}: \"{name}\" ({etype})"
                if desc:
                    info += f" - \"{desc}\""
            else:
                name = _extract_event_name_from_text_content(unit.raw_data.get("text_content", ""))
                desc = unit.raw_data.get("description", "") or _extract_event_desc_from_text_content(unit.raw_data.get("text_content", ""))
                info = f"- Event {chr(65 + i)}: \"{name}\""
                if desc:
                    info += f" - \"{desc}\""
                participants = unit.raw_data.get("participants", [])
                if participants:
                    info += f", Participants: {json.dumps(participants)}"
                inferred_time = unit.raw_data.get("inferred_time")
                if inferred_time:
                    info += f", Time: {inferred_time}"

            session_id = unit.metadata.get("session_id", "unknown")
            info += f" [Session: {session_id}]"
            cluster_info_lines.append(info)

        cluster_info = "\n".join(cluster_info_lines)

        if unit_type == "entity":
            prompt = CLUSTER_JUDGE_PROMPT.format(cluster_info=cluster_info)
        else:
            prompt = EVENT_CLUSTER_JUDGE_PROMPT.format(cluster_info=cluster_info)

        try:
            data = self._call_llm_json(prompt, context_label=f"cluster_judge_{unit_type}")
            return {
                "should_merge": data.get("should_merge", False),
                "confidence": data.get("confidence", 0.0),
                "canonical_name": data.get("canonical_name"),
                "reasoning": data.get("reasoning", ""),
            }
        except json.JSONDecodeError:
            return {"should_merge": False, "confidence": 0.0, "canonical_name": None, "reasoning": "Failed to parse LLM response"}

    def _create_canonical_entity(self, cluster, canonical_name):
        return self._canonical_creator.create_canonical_entity(cluster, canonical_name)

    def _create_canonical_event(self, cluster, canonical_name):
        return self._canonical_creator.create_canonical_event(cluster, canonical_name)

    def _generate_event_signature_from_data(self, event_name, participants, inferred_time):
        return self._canonical_creator._generate_event_signature_from_data(
            event_name, participants, inferred_time
        )

    def _migrate_coref_edges_to_canonical(self, original_uids, canonical_uid, confidence):
        self._canonical_creator.migrate_coref_edges_to_canonical(
            original_uids, canonical_uid, confidence
        )

    def _delete_original_units(self, uids):
        self._canonical_creator.delete_original_units(uids)

    def merge_cross_session_entities(
        self,
        candidate_entities: Optional[List[Dict[str, Any]]] = None,
        vector_threshold: float = 0.65,
        combined_threshold: float = 0.45,
        llm_confidence_threshold: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """Merge duplicate entities across all sessions.

        Args:
            candidate_entities: Optional override entity list; gathered from store if None.
            vector_threshold: Min cosine similarity for pre-clustering.
            combined_threshold: Min combined (vector+graph) score for clustering.
            llm_confidence_threshold: Min LLM confidence to execute a merge.

        Returns:
            List of merge result dicts with canonical_uid, canonical_name, etc.
        """
        if candidate_entities is None:
            all_entities = self._semantic_map.get_units_in_spaces([self._entity_space])
            candidate_entities = [
                {
                    "id": str(e.uid),
                    "name": _extract_entity_name_from_text_content(e.raw_data.get("text_content", "")),
                    "type": e.raw_data.get("entity_type", ""),
                }
                for e in all_entities
            ]

        if not candidate_entities:
            return []

        clusters = self._build_candidate_clusters(
            candidate_entities,
            vector_threshold=vector_threshold,
            combined_threshold=combined_threshold,
        )

        merge_results: List[Dict[str, Any]] = []

        for cluster in clusters:
            if len(cluster) < 2:
                continue

            judge_result = self._llm_cluster_judge(cluster, unit_type="entity")

            if not judge_result.get("should_merge", False):
                continue

            confidence = judge_result.get("confidence", 0.0)
            if confidence < llm_confidence_threshold:
                continue

            canonical_name = judge_result.get("canonical_name")
            if not canonical_name:
                names = [c.get("name", "") for c in cluster if c.get("name")]
                canonical_name = names[0] if names else "merged_entity"

            canonical_unit = self._create_canonical_entity(cluster, canonical_name)

            self._semantic_map.upsert_unit(canonical_unit, ensure_embedding=True)

            cluster_uids: List[Uid] = []
            for c in cluster:
                cid = c.get("id", "")
                if not cid:
                    continue
                uid = Uid(cid)
                cluster_uids.append(uid)

            self._migrate_coref_edges_to_canonical(cluster_uids, canonical_unit.uid, confidence)

            self._delete_original_units(cluster_uids)

            merge_results.append({
                "canonical_uid": str(canonical_unit.uid),
                "canonical_name": canonical_name,
                "merged_count": len(cluster),
                "confidence": confidence,
                "reasoning": judge_result.get("reasoning", ""),
            })

        return merge_results

    def merge_cross_session_events(
        self,
        candidate_events: Optional[List[Dict[str, Any]]] = None,
        vector_threshold: float = 0.65,
        combined_threshold: float = 0.45,
        llm_confidence_threshold: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """Merge duplicate events across all sessions.

        Args:
            candidate_events: Optional override event list; gathered from store if None.
            vector_threshold: Min cosine similarity for pre-clustering.
            combined_threshold: Min combined score for clustering.
            llm_confidence_threshold: Min LLM confidence to execute a merge.

        Returns:
            List of merge result dicts.
        """
        if candidate_events is None:
            all_events = self._semantic_map.get_units_in_spaces([self._event_space])
            candidate_events = [
                {
                    "id": str(e.uid),
                    "name": _extract_event_name_from_text_content(e.raw_data.get("text_content", "")),
                    "signature": e.metadata.get("signature", ""),
                }
                for e in all_events
            ]

        if not candidate_events:
            return []

        clusters = self._precluster_candidates(
            candidate_events,
            threshold=vector_threshold,
        )

        merge_results: List[Dict[str, Any]] = []

        for cluster in clusters:
            if len(cluster) < 2:
                continue

            judge_result = self._llm_cluster_judge(cluster, unit_type="event")

            if not judge_result.get("should_merge", False):
                continue

            confidence = judge_result.get("confidence", 0.0)
            if confidence < llm_confidence_threshold:
                continue

            canonical_name = judge_result.get("canonical_name")
            if not canonical_name:
                names = [c.get("name", "") for c in cluster if c.get("name")]
                canonical_name = names[0] if names else "merged_event"

            canonical_unit = self._create_canonical_event(cluster, canonical_name)

            self._semantic_map.upsert_unit(canonical_unit, ensure_embedding=True)

            cluster_uids: List[Uid] = []
            for c in cluster:
                cid = c.get("id", "")
                if not cid:
                    continue
                uid = Uid(cid)
                cluster_uids.append(uid)

            self._migrate_coref_edges_to_canonical(cluster_uids, canonical_unit.uid, confidence)

            self._delete_original_units(cluster_uids)

            merge_results.append({
                "canonical_uid": str(canonical_unit.uid),
                "canonical_name": canonical_name,
                "merged_count": len(cluster),
                "confidence": confidence,
                "reasoning": judge_result.get("reasoning", ""),
            })

        return merge_results
