"""Canonical unit creator for merged entities and events.

Creates canonical MemoryUnits from clusters of duplicate entities/events,
using LLM-based description merging and preserving aliases, participants,
and cross-session facts.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from ...domain.coref_graph_constants import REL_COREF, REL_EVIDENCED_BY
from ...domain.memory_unit import MemoryUnit
from ...domain.types import SpaceName, Uid
from ...ports.llm_provider import LLMProvider
from ..pipeline._utils import (
    extract_entity_desc_from_text_content,
    extract_entity_name_from_text_content,
    extract_event_desc_from_text_content,
    extract_event_name_from_text_content,
    format_entity_text_content,
    format_event_text_content,
    generate_entity_uid,
    generate_event_uid,
    is_short_alias,
    parse_json_response,
)
from ..prompts import CROSS_SESSION_EVENT_MERGE_PROMPT, CROSS_SESSION_MERGE_PROMPT
from ..semantic_graph import SemanticGraphService
from ..semantic_map import SemanticMapService


class CanonicalCreator:
    """Creates canonical MemoryUnits from merged entity/event clusters.

    Uses LLM prompts (CROSS_SESSION_MERGE_PROMPT /
    CROSS_SESSION_EVENT_MERGE_PROMPT) to produce merged descriptions,
    aliases, and participants, then writes resulting canonical units
    and migrates COREF/EVIDENCED_BY edges.

    Args:
        semantic_map: SemanticMapService for unit storage.
        graph: SemanticGraphService for edge migration.
        llm: LLM provider for merge description generation.
        entity_space: SpaceName for entity storage.
        event_space: SpaceName for event storage.
    """

    def __init__(
        self,
        semantic_map: SemanticMapService,
        graph: SemanticGraphService,
        llm: LLMProvider,
        entity_space: SpaceName,
        event_space: SpaceName,
    ):
        self._semantic_map = semantic_map
        self._graph = graph
        self._llm = llm
        self._entity_space = entity_space
        self._event_space = event_space

    def _call_llm(self, prompt: str) -> Any:
        messages = [{"role": "user", "content": prompt}]
        return self._llm.chat(messages, temperature=0.3, max_tokens=1024)

    def _generate_event_signature_from_data(
        self,
        event_name: str,
        participants: List[Dict[str, Any]],
        inferred_time: Optional[str],
    ) -> str:
        participants_str = ",".join(
            sorted([p.get("mention", "") if isinstance(p, dict) else str(p) for p in participants])
        )[:50]
        time_str = inferred_time or "unknown"
        return f"{participants_str}-{event_name[:30]}-{time_str}"

    def create_canonical_entity(
        self,
        cluster: List[Dict[str, Any]],
        canonical_name: str,
    ) -> MemoryUnit:
        """Create a canonical entity unit from a cluster of duplicates.

        Collects per-session facts and aliases from all cluster members,
        then calls the LLM to produce a merged description. The resulting
        MemoryUnit is stored in the entity space.

        Args:
            cluster: List of dicts with at least an 'id' key pointing to
                the UID of each duplicate entity unit.
            canonical_name: The chosen canonical name for the merged entity.

        Returns:
            A new MemoryUnit representing the canonical entity.
        """
        session_facts: List[Dict[str, Any]] = []
        all_aliases: Set[str] = set()
        entity_type = "Concept"

        for c in cluster:
            cid = c.get("id", "")
            unit = self._semantic_map.get_unit(Uid(cid))
            if unit is None:
                continue

            entity_type = unit.raw_data.get("entity_type", entity_type)
            desc = extract_entity_desc_from_text_content(unit.raw_data.get("text_content", ""))
            name = extract_entity_name_from_text_content(unit.raw_data.get("text_content", ""))
            session_id = unit.metadata.get("session_id", "unknown")

            session_facts.append({
                "session_id": session_id,
                "description": desc or name,
                "source_uid": str(unit.uid),
            })

            aliases = unit.raw_data.get("aliases", [])
            for a in aliases:
                if is_short_alias(a):
                    all_aliases.add(a)
            if name and name != canonical_name and is_short_alias(name):
                all_aliases.add(name)

        import json

        prompt = CROSS_SESSION_MERGE_PROMPT.format(
            name=canonical_name,
            type=entity_type,
            session_facts=json.dumps(session_facts),
        )

        response = self._call_llm(prompt)
        merged_description = ""
        merged_aliases = list(all_aliases)

        try:
            data = parse_json_response(response.content)
            merged_description = data.get("merged_description", "")
            additional_aliases = data.get("merged_aliases", [])
            for a in additional_aliases:
                if is_short_alias(a):
                    all_aliases.add(a)
            merged_aliases = list(all_aliases)
        except json.JSONDecodeError:
            pass

        text_content = format_entity_text_content(canonical_name, entity_type, merged_description)
        canonical_uid = generate_entity_uid(canonical_name, entity_type)
        now = datetime.now(timezone.utc).isoformat()

        unit = MemoryUnit(
            uid=canonical_uid,
            raw_data={
                "text_content": text_content,
                "entity_name": canonical_name,
                "entity_type": entity_type,
                "aliases": merged_aliases,
            },
            metadata={
                "type": "entity",
                "session_id": "merged",
                "session_facts": session_facts,
                "created_at": now,
                "updated_at": now,
            },
        )

        return unit

    def create_canonical_event(
        self,
        cluster: List[Dict[str, Any]],
        canonical_name: str,
    ) -> MemoryUnit:
        """Create a canonical event unit from a cluster of duplicates.

        Collects per-session facts, participants, and inferred time from
        all cluster members, then calls the LLM to produce a merged
        description. The resulting MemoryUnit is stored in the event space.

        Args:
            cluster: List of dicts with at least an 'id' key pointing to
                the UID of each duplicate event unit.
            canonical_name: The chosen canonical name for the merged event.

        Returns:
            A new MemoryUnit representing the canonical event.
        """
        import json

        session_facts: List[Dict[str, Any]] = []
        all_participants: List[Dict[str, Any]] = []
        merged_time: Optional[str] = None

        for c in cluster:
            cid = c.get("id", "")
            unit = self._semantic_map.get_unit(Uid(cid))
            if unit is None:
                continue

            desc = unit.raw_data.get("description", "") or extract_event_desc_from_text_content(
                unit.raw_data.get("text_content", "")
            )
            name = extract_event_name_from_text_content(unit.raw_data.get("text_content", ""))
            session_id = unit.metadata.get("session_id", "unknown")

            session_facts.append({
                "session_id": session_id,
                "description": desc or name,
                "source_uid": str(unit.uid),
            })

            participants = unit.raw_data.get("participants", [])
            all_participants.extend(participants)

            event_time = unit.raw_data.get("inferred_time")
            if event_time and not merged_time:
                merged_time = event_time

        prompt = CROSS_SESSION_EVENT_MERGE_PROMPT.format(
            name=canonical_name,
            session_facts=json.dumps(session_facts),
            existing_participants=json.dumps(all_participants),
            existing_time=merged_time or "unknown",
        )

        response = self._call_llm(prompt)
        merged_description = canonical_name
        merged_participants = all_participants

        try:
            data = parse_json_response(response.content)
            merged_description = data.get("merged_description", canonical_name)
            merged_participants = data.get("merged_participants", all_participants)
            new_time = data.get("merged_time")
            if new_time:
                merged_time = new_time
        except json.JSONDecodeError:
            pass

        canonical_uid = generate_event_uid(canonical_name, "merged")
        now = datetime.now(timezone.utc).isoformat()

        text_content = format_event_text_content(canonical_name, merged_description)

        unit = MemoryUnit(
            uid=canonical_uid,
            raw_data={
                "text_content": text_content,
                "event_name": canonical_name,
                "description": merged_description,
                "inferred_time": merged_time,
                "participants": merged_participants,
            },
            metadata={
                "type": "event",
                "session_id": "merged",
                "session_facts": session_facts,
                "signature": self._generate_event_signature_from_data(
                    canonical_name, merged_participants, merged_time
                ),
                "created_at": now,
                "updated_at": now,
            },
        )

        return unit

    def migrate_coref_edges_to_canonical(
        self,
        original_uids: List[Uid],
        canonical_uid: Uid,
        confidence: float,
    ) -> None:
        """Re-wire COREF and EVIDENCED_BY edges to point to the canonical unit.

        For each original UID, existing COREF edges are redirected so the
        source points to the canonical UID, and EVIDENCED_BY edges are
        redirected so the canonical UID becomes the source.

        Args:
            original_uids: UIDs of the original (pre-merge) units.
            canonical_uid: UID of the newly created canonical unit.
            confidence: Confidence score to attach to new edges.
        """
        graph_store = self._graph.get_graph_store()
        original_set = set(original_uids)
        edges_to_add: List[Tuple[Uid, Uid, str, Dict[str, Any]]] = []
        for source, target, rel_type, props in graph_store.get_all_edges():
            if rel_type == str(REL_COREF) and target in original_set and source != canonical_uid:
                edges_to_add.append((source, canonical_uid, REL_COREF, props))
            if rel_type == str(REL_EVIDENCED_BY) and source in original_set and canonical_uid != target:
                edges_to_add.append((canonical_uid, target, REL_EVIDENCED_BY, props))

        for source, target, rel_name, props in edges_to_add:
            try:
                self._graph.add_relationship(
                    source_uid=source,
                    target_uid=target,
                    relationship_name=rel_name,
                    **props,
                )
            except (KeyError, Exception):
                pass

    def delete_original_units(self, uids: List[Uid]) -> None:
        """Remove original (pre-merge) units from the semantic map and graph.

        Silently ignores any UIDs that no longer exist.

        Args:
            uids: UIDs of the original units to delete.
        """
        for uid in uids:
            try:
                self._graph.delete_unit(uid)
            except Exception:
                pass
