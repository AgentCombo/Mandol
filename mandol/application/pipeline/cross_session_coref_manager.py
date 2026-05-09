"""Cross-session coreference management for entity/event deduplication.

Manages entity and event name indices, retrieves candidate matches across
sessions using multi-signal search (name/alias + BM25 + vector + graph),
and uses LLM judging to decide merges. Also handles description merging
and incremental index updates.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING

import numpy as np

from ...domain.coref_graph_constants import (
    REL_CAUSED_BY,
    REL_CAUSES,
    REL_COREF,
    REL_EVIDENCED_BY,
    REL_INVOLVES,
    REL_RELATED_TO,
)
from ...domain.memory_unit import MemoryUnit
from ...domain.types import SpaceName, Uid
from ...ports.llm_provider import ChatMessage, LLMProvider
from ...retrieval.bm25 import Bm25Retriever
from ...retrieval.text import TextExtractor, Tokenizer
from ..prompts import (
    DESCRIPTION_MERGE_PROMPT,
    ENTITY_MATCH_JUDGE_PROMPT,
    EVENT_DESCRIPTION_MERGE_PROMPT,
    EVENT_MATCH_JUDGE_PROMPT,
)
from ..semantic_graph import SemanticGraphService
from ..semantic_map import SemanticMapService
from ._utils import (
    extract_entity_desc_from_text_content,
    extract_entity_name_from_text_content,
    extract_entity_type_from_text_content,
    extract_event_desc_from_text_content,
    extract_event_name_from_text_content,
    find_matching_dialogue_uids,
    format_entity_text_content,
    format_event_text_content,
    generate_entity_uid,
    generate_event_uid,
    is_short_alias,
    parse_json_response,
)
from .unified_fact_pipeline import (
    ExtractedCausalRelation,
    ExtractedEntity,
    ExtractedEvent,
    ExtractedRelation,
    PipelineResult,
)

if TYPE_CHECKING:
    from ..session_manager import Session

logger = logging.getLogger(__name__)


class CrossSessionCorefManager:
    """Manages cross-session entity and event coreference resolution.

    Maintains name/alias lookup indices for both entities and events, and
    uses a multi-signal retrieval pipeline (name match, vector search, BM25,
    graph traversal) to find candidate matches for new extracted entities/events.
    LLM-based matching confirms merges with confidence scores.

    Args:
        llm_provider: LLM provider for match judging and description merging.
        semantic_map: SemanticMapService for unit storage and embedding access.
        graph: SemanticGraphService for graph traversal queries.
        naming: SpaceNamingPolicy instance for space name construction.
        root: Root SpaceName for the memory system.
        vector_threshold: Min cosine similarity for vector-based matching (default 0.45).
        llm_confidence_threshold: Min LLM confidence to accept a merge (default 0.7).
        max_candidates: Max candidates retrieved per entity/event (default 20).
        simple_concat_threshold: Below this many facts, use simple concatenation (default 2).
        entity_space: Space name for entity storage.
        event_space: Space name for event storage.
    """

    def __init__(
        self,
        *,
        llm_provider: LLMProvider,
        semantic_map: SemanticMapService,
        graph: SemanticGraphService,
        naming: Any,
        root: SpaceName,
        vector_threshold: float = 0.45,
        llm_confidence_threshold: float = 0.7,
        max_candidates: int = 20,
        simple_concat_threshold: int = 2,
        entity_space: SpaceName = SpaceName("knowledge_entity"),
        event_space: SpaceName = SpaceName("episodic_event"),
    ):
        self._llm = llm_provider
        self._semantic_map = semantic_map
        self._graph = graph
        self._naming = naming
        self._root = root
        self._vector_threshold = vector_threshold
        self._llm_confidence_threshold = llm_confidence_threshold
        self._max_candidates = max_candidates
        self._simple_concat_threshold = simple_concat_threshold
        self._entity_space = entity_space
        self._event_space = event_space

        self._lock = threading.Lock()
        self._is_initialized = False

        self._entity_name_index: Dict[str, Set[str]] = {}
        self._event_name_index: Dict[str, Set[str]] = {}

        self._bm25_retriever = Bm25Retriever(
            text_extractor=TextExtractor(primary_key="text_content"),
            tokenizer=Tokenizer(use_jieba=True),
        )

    def initialize_from_existing(self) -> None:
        """Build name/alias indices from existing entity and event stores.

        Must be called before merge_session_entities/merge_session_events.
        Sets _is_initialized to True.
        """
        entity_units = self._semantic_map.get_units_in_spaces([self._entity_space])
        event_units = self._semantic_map.get_units_in_spaces([self._event_space])

        self._entity_name_index = self._build_entity_name_index(entity_units)
        self._event_name_index = self._build_event_name_index(event_units)

        self._is_initialized = True
        logger.info(
            f"CrossSessionCorefManager initialized: "
            f"{len(entity_units)} entities, {len(event_units)} events"
        )

    def _build_entity_name_index(self, units: List[MemoryUnit]) -> Dict[str, Set[str]]:
        index: Dict[str, Set[str]] = {}
        for unit in units:
            uid_str = str(unit.uid)
            name = unit.raw_data.get("entity_name", "") or _extract_entity_name_from_text_content(
                unit.raw_data.get("text_content", "")
            )
            aliases = unit.raw_data.get("aliases", [])
            name_key = name.strip().lower()
            if name_key:
                index.setdefault(name_key, set()).add(uid_str)
            for alias in aliases:
                if isinstance(alias, str):
                    alias_key = alias.strip().lower()
                    if alias_key:
                        index.setdefault(alias_key, set()).add(uid_str)
        return index

    def _build_event_name_index(self, units: List[MemoryUnit]) -> Dict[str, Set[str]]:
        index: Dict[str, Set[str]] = {}
        for unit in units:
            uid_str = str(unit.uid)
            name = unit.raw_data.get("event_name", "") or _extract_event_name_from_text_content(
                unit.raw_data.get("text_content", "")
            )
            name_key = name.strip().lower()
            if name_key:
                index.setdefault(name_key, set()).add(uid_str)
        return index

    def _update_entity_name_index(self, uid: str, name: str, aliases: List[str]) -> None:
        uid_str = str(uid)
        name_key = name.strip().lower()
        if name_key:
            self._entity_name_index.setdefault(name_key, set()).add(uid_str)
        for alias in aliases:
            if isinstance(alias, str):
                alias_key = alias.strip().lower()
                if alias_key:
                    self._entity_name_index.setdefault(alias_key, set()).add(uid_str)

    def _update_event_name_index(self, uid: str, name: str) -> None:
        uid_str = str(uid)
        name_key = name.strip().lower()
        if name_key:
            self._event_name_index.setdefault(name_key, set()).add(uid_str)

    def _retrieve_candidate_entities(self, entity: ExtractedEntity) -> List[MemoryUnit]:
        candidates: Dict[str, MemoryUnit] = {}

        name_key = entity.entity_name.strip().lower()
        matched_uids = self._entity_name_index.get(name_key, set())
        for uid_str in matched_uids:
            unit = self._semantic_map.get_unit(Uid(uid_str))
            if unit is not None:
                candidates[uid_str] = unit

        for alias in entity.aliases:
            alias_key = alias.strip().lower()
            alias_uids = self._entity_name_index.get(alias_key, set())
            for uid_str in alias_uids:
                if uid_str not in candidates:
                    unit = self._semantic_map.get_unit(Uid(uid_str))
                    if unit is not None:
                        candidates[uid_str] = unit

        name_tokens = entity.entity_name.strip().lower().split()
        for token in name_tokens:
            if len(token) <= 1:
                continue
            token_uids = self._entity_name_index.get(token, set())
            for uid_str in token_uids:
                if uid_str not in candidates:
                    unit = self._semantic_map.get_unit(Uid(uid_str))
                    if unit is not None:
                        candidates[uid_str] = unit

        try:
            vector_hits = self._semantic_map.search_by_text(
                entity.entity_name,
                top_k=self._max_candidates,
                space_names=[self._entity_space],
            )
            for unit, score in vector_hits:
                uid_str = str(unit.uid)
                if uid_str not in candidates:
                    if score >= self._vector_threshold:
                        candidates[uid_str] = unit
        except Exception:
            logger.debug("Vector search failed for entity candidates")

        try:
            all_entity_units = self._semantic_map.get_units_in_spaces([self._entity_space])
            bm25_hits = self._bm25_retriever.search(
                entity.entity_name, all_entity_units, top_k=self._max_candidates
            )
            for scored in bm25_hits:
                uid_str = str(scored.unit.uid)
                if uid_str not in candidates:
                    candidates[uid_str] = scored.unit
        except Exception:
            logger.debug("BM25 search failed for entity candidates")

        for uid_str in list(candidates.keys())[:self._max_candidates]:
            try:
                neighbors = self._graph.get_explicit_neighbors(
                    [Uid(uid_str)],
                    rel_type=REL_RELATED_TO,
                    direction="out",
                )
                for neighbor in neighbors:
                    n_uid_str = str(neighbor.uid)
                    if n_uid_str not in candidates:
                        if neighbor.metadata.get("type") == "entity":
                            candidates[n_uid_str] = neighbor

                involves_neighbors = self._graph.get_explicit_neighbors(
                    [Uid(uid_str)],
                    rel_type=REL_INVOLVES,
                    direction="out",
                )
                for neighbor in involves_neighbors:
                    n_uid_str = str(neighbor.uid)
                    if n_uid_str not in candidates:
                        if neighbor.metadata.get("type") == "entity":
                            candidates[n_uid_str] = neighbor
            except Exception:
                pass

        result = list(candidates.values())
        return result[:self._max_candidates]

    def _retrieve_candidate_events(
        self,
        event: ExtractedEvent,
        entity_uid_map: Dict[str, Uid],
    ) -> List[MemoryUnit]:
        candidates: Dict[str, MemoryUnit] = {}

        name_key = event.event_name.strip().lower()
        matched_uids = self._event_name_index.get(name_key, set())
        for uid_str in matched_uids:
            unit = self._semantic_map.get_unit(Uid(uid_str))
            if unit is not None:
                candidates[uid_str] = unit

        try:
            vector_hits = self._semantic_map.search_by_text(
                event.event_name,
                top_k=self._max_candidates,
                space_names=[self._event_space],
            )
            for unit, score in vector_hits:
                uid_str = str(unit.uid)
                if uid_str not in candidates:
                    if score >= self._vector_threshold:
                        candidates[uid_str] = unit
        except Exception:
            logger.debug("Vector search failed for event candidates")

        for participant in event.participants:
            mention = participant.get("mention", "")
            entity_uid = entity_uid_map.get(mention)
            if entity_uid is not None:
                try:
                    involves_neighbors = self._graph.get_explicit_neighbors(
                        [entity_uid],
                        rel_type=REL_INVOLVES,
                        direction="in",
                    )
                    for neighbor in involves_neighbors:
                        n_uid_str = str(neighbor.uid)
                        if n_uid_str not in candidates:
                            if neighbor.metadata.get("type") == "event":
                                candidates[n_uid_str] = neighbor
                except Exception:
                    pass

        try:
            all_event_units = self._semantic_map.get_units_in_spaces([self._event_space])
            bm25_hits = self._bm25_retriever.search(
                event.event_name, all_event_units, top_k=self._max_candidates
            )
            for scored in bm25_hits:
                uid_str = str(scored.unit.uid)
                if uid_str not in candidates:
                    candidates[uid_str] = scored.unit
        except Exception:
            logger.debug("BM25 search failed for event candidates")

        result = list(candidates.values())
        return result[:self._max_candidates]

    def _llm_judge_entity_match(
        self,
        entity: ExtractedEntity,
        candidates: List[MemoryUnit],
    ) -> Tuple[Optional[int], float, Optional[str], List[str], str]:
        if not candidates:
            return None, 0.0, None, [], "No candidates"

        candidate_descriptions = []
        for i, unit in enumerate(candidates):
            name = unit.raw_data.get("entity_name", "") or _extract_entity_name_from_text_content(
                unit.raw_data.get("text_content", "")
            )
            etype = unit.raw_data.get("entity_type", "")
            desc = _extract_entity_desc_from_text_content(unit.raw_data.get("text_content", ""))
            aliases = unit.raw_data.get("aliases", [])
            info = f"[{i}] Name: \"{name}\", Type: \"{etype}\""
            if desc:
                info += f", Description: \"{desc}\""
            if aliases:
                info += f", Aliases: {json.dumps(aliases)}"
            candidate_descriptions.append(info)

        candidates_text = "\n".join(candidate_descriptions)

        prompt = ENTITY_MATCH_JUDGE_PROMPT.format(
            new_entity_name=entity.entity_name,
            new_entity_type=entity.entity_type,
            new_facts=json.dumps(entity.new_facts),
            mention_text=entity.mention_text,
            candidates=candidates_text,
        )

        messages: List[ChatMessage] = [
            {"role": "user", "content": prompt},
        ]

        try:
            response = self._llm.chat(
                messages,
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            data = parse_json_response(response.content)
            matched_index = data.get("matched_index")
            confidence = float(data.get("confidence", 0.0))
            canonical_name_suggestion = data.get("canonical_name_suggestion")
            new_aliases = data.get("new_aliases", [])
            reasoning = data.get("reasoning", "")

            if matched_index is not None and not isinstance(matched_index, int):
                try:
                    matched_index = int(matched_index)
                except (ValueError, TypeError):
                    matched_index = None

            if matched_index is not None and (matched_index < 0 or matched_index >= len(candidates)):
                matched_index = None

            return matched_index, confidence, canonical_name_suggestion, new_aliases, reasoning
        except Exception as e:
            logger.warning(f"LLM entity match judge failed: {e}, falling back to vector similarity")
            return self._fallback_vector_judge(entity, candidates)

    def _fallback_vector_judge(
        self,
        entity: ExtractedEntity,
        candidates: List[MemoryUnit],
    ) -> Tuple[Optional[int], float, Optional[str], List[str], str]:
        best_idx = None
        best_score = 0.0
        try:
            query_emb = self._semantic_map._embedder.embed_text([entity.entity_name])[0]
            query_vec = np.asarray(query_emb, dtype=np.float32).reshape(-1)
            for i, unit in enumerate(candidates):
                if unit.embedding is None:
                    continue
                cand_vec = np.asarray(unit.embedding, dtype=np.float32).reshape(-1)
                norm_q = np.linalg.norm(query_vec)
                norm_c = np.linalg.norm(cand_vec)
                if norm_q == 0 or norm_c == 0:
                    continue
                sim = float(np.dot(query_vec, cand_vec) / (norm_q * norm_c))
                if sim > best_score:
                    best_score = sim
                    best_idx = i
        except Exception:
            pass

        if best_idx is not None and best_score >= self._vector_threshold:
            return best_idx, best_score, None, [], f"Vector fallback (sim={best_score:.3f})"
        return None, 0.0, None, [], "No match found (vector fallback)"

    def _llm_judge_event_match(
        self,
        event: ExtractedEvent,
        candidates: List[MemoryUnit],
    ) -> Tuple[Optional[int], float, Optional[str], str]:
        if not candidates:
            return None, 0.0, None, "No candidates"

        candidate_descriptions = []
        for i, unit in enumerate(candidates):
            name = unit.raw_data.get("event_name", "") or _extract_event_name_from_text_content(
                unit.raw_data.get("text_content", "")
            )
            desc = unit.raw_data.get("description", "") or _extract_event_desc_from_text_content(
                unit.raw_data.get("text_content", "")
            )
            participants = unit.raw_data.get("participants", [])
            inferred_time = unit.raw_data.get("inferred_time")
            info = f"[{i}] Name: \"{name}\""
            if desc:
                info += f", Description: \"{desc}\""
            if participants:
                info += f", Participants: {json.dumps(participants)}"
            if inferred_time:
                info += f", Time: {inferred_time}"
            candidate_descriptions.append(info)

        candidates_text = "\n".join(candidate_descriptions)

        prompt = EVENT_MATCH_JUDGE_PROMPT.format(
            new_event_name=event.event_name,
            new_event_description=event.description,
            new_participants=json.dumps(event.participants),
            new_time=event.inferred_time or "unknown",
            new_facts=json.dumps(event.new_facts),
            candidates=candidates_text,
        )

        messages: List[ChatMessage] = [
            {"role": "user", "content": prompt},
        ]

        try:
            response = self._llm.chat(
                messages,
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            data = parse_json_response(response.content)
            matched_index = data.get("matched_index")
            confidence = float(data.get("confidence", 0.0))
            canonical_name_suggestion = data.get("canonical_name_suggestion")
            reasoning = data.get("reasoning", "")

            if matched_index is not None and not isinstance(matched_index, int):
                try:
                    matched_index = int(matched_index)
                except (ValueError, TypeError):
                    matched_index = None

            if matched_index is not None and (matched_index < 0 or matched_index >= len(candidates)):
                matched_index = None

            return matched_index, confidence, canonical_name_suggestion, reasoning
        except Exception as e:
            logger.warning(f"LLM event match judge failed: {e}, falling back to vector similarity")
            return self._fallback_event_vector_judge(event, candidates)

    def _fallback_event_vector_judge(
        self,
        event: ExtractedEvent,
        candidates: List[MemoryUnit],
    ) -> Tuple[Optional[int], float, Optional[str], str]:
        best_idx = None
        best_score = 0.0
        try:
            query_text = f"{event.event_name} {event.description}".strip()
            query_emb = self._semantic_map._embedder.embed_text([query_text])[0]
            query_vec = np.asarray(query_emb, dtype=np.float32).reshape(-1)
            for i, unit in enumerate(candidates):
                if unit.embedding is None:
                    continue
                cand_vec = np.asarray(unit.embedding, dtype=np.float32).reshape(-1)
                norm_q = np.linalg.norm(query_vec)
                norm_c = np.linalg.norm(cand_vec)
                if norm_q == 0 or norm_c == 0:
                    continue
                sim = float(np.dot(query_vec, cand_vec) / (norm_q * norm_c))
                if sim > best_score:
                    best_score = sim
                    best_idx = i
        except Exception:
            pass

        if best_idx is not None and best_score >= self._vector_threshold:
            return best_idx, best_score, None, f"Vector fallback (sim={best_score:.3f})"
        return None, 0.0, None, "No match found (vector fallback)"

    def _update_description_simple(self, existing_desc: str, new_facts: List[str]) -> str:
        parts = [existing_desc] if existing_desc else []
        parts.extend(new_facts)
        return " | ".join(parts)

    def _update_description_llm(
        self,
        existing_desc: str,
        new_facts: List[str],
        entity_name: str,
    ) -> str:
        prompt = DESCRIPTION_MERGE_PROMPT.format(
            name=entity_name,
            type="",
            existing_description=existing_desc,
            new_facts=json.dumps(new_facts),
            session_facts="[]",
        )

        messages: List[ChatMessage] = [
            {"role": "user", "content": prompt},
        ]

        try:
            response = self._llm.chat(
                messages,
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            data = parse_json_response(response.content)
            return data.get("merged_description", existing_desc)
        except Exception as e:
            logger.warning(f"LLM description merge failed: {e}, using simple concat")
            return self._update_description_simple(existing_desc, new_facts)

    def _merge_to_existing_entity(
        self,
        entity: ExtractedEntity,
        canonical_unit: MemoryUnit,
        session_id: str,
        dialogue_units: Sequence[MemoryUnit],
    ) -> Uid:
        canonical_uid = canonical_unit.uid
        entity_name = canonical_unit.raw_data.get("entity_name", "") or _extract_entity_name_from_text_content(
            canonical_unit.raw_data.get("text_content", "")
        )
        entity_type = canonical_unit.raw_data.get("entity_type", "") or _extract_entity_type_from_text_content(
            canonical_unit.raw_data.get("text_content", "")
        )
        existing_desc = _extract_entity_desc_from_text_content(
            canonical_unit.raw_data.get("text_content", "")
        )

        if entity.new_facts:
            if len(entity.new_facts) <= self._simple_concat_threshold:
                merged_desc = self._update_description_simple(existing_desc, entity.new_facts)
            else:
                merged_desc = self._update_description_llm(existing_desc, entity.new_facts, entity_name)

            canonical_unit.raw_data["text_content"] = _format_entity_text_content(
                entity_name, entity_type, merged_desc
            )
            canonical_unit.raw_data["entity_name"] = entity_name
            canonical_unit.metadata["updated_at"] = datetime.now(timezone.utc).isoformat()

        existing_aliases = canonical_unit.raw_data.get("aliases", [])
        for alias in entity.aliases:
            if alias not in existing_aliases and _is_short_alias(alias):
                existing_aliases.append(alias)
        canonical_unit.raw_data["aliases"] = existing_aliases

        session_ids = canonical_unit.metadata.get("session_ids", [])
        if isinstance(session_ids, str):
            session_ids = [session_ids]
        if session_id not in session_ids:
            session_ids.append(session_id)
        canonical_unit.metadata["session_ids"] = session_ids

        self._semantic_map.upsert_unit(canonical_unit)

        matching_dialogue_uids = find_matching_dialogue_uids(
            entity.mention_text, dialogue_units
        )
        for dia_uid in matching_dialogue_uids:
            try:
                self._graph.add_relationship(
                    source_uid=dia_uid,
                    target_uid=canonical_uid,
                    relationship_name=REL_COREF,
                    confidence=entity.confidence,
                    mention_text=entity.mention_text,
                    session_id=session_id,
                )
            except KeyError:
                pass

        self._update_entity_name_index(str(canonical_uid), entity_name, existing_aliases)

        return canonical_uid

    def _create_new_canonical_entity(
        self,
        entity: ExtractedEntity,
        session_id: str,
        dialogue_units: Sequence[MemoryUnit],
    ) -> Uid:
        canonical_uid = generate_entity_uid(entity.entity_name, entity.entity_type)

        description = ". ".join(entity.new_facts) if entity.new_facts else ""
        text_content = _format_entity_text_content(
            entity.entity_name, entity.entity_type, description
        )

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
                "session_ids": [session_id],
                "created_at": now,
                "updated_at": now,
            },
        )

        self._semantic_map.add_unit(
            unit,
            space_names=[self._entity_space],
            ensure_embedding=True,
        )

        matching_dialogue_uids = find_matching_dialogue_uids(
            entity.mention_text, dialogue_units
        )
        for dia_uid in matching_dialogue_uids:
            try:
                self._graph.add_relationship(
                    source_uid=dia_uid,
                    target_uid=canonical_uid,
                    relationship_name=REL_COREF,
                    confidence=entity.confidence,
                    mention_text=entity.mention_text,
                    session_id=session_id,
                )
            except KeyError:
                pass

        for dia_uid in matching_dialogue_uids:
            try:
                self._graph.add_relationship(
                    source_uid=canonical_uid,
                    target_uid=dia_uid,
                    relationship_name=REL_EVIDENCED_BY,
                    mention_text=entity.mention_text,
                )
            except KeyError:
                pass

        self._update_entity_name_index(str(canonical_uid), entity.entity_name, entity.aliases)

        return canonical_uid

    def _merge_to_existing_event(
        self,
        event: ExtractedEvent,
        canonical_unit: MemoryUnit,
        session_id: str,
        dialogue_units: Sequence[MemoryUnit],
        entity_uid_map: Dict[str, Uid],
    ) -> Uid:
        canonical_uid = canonical_unit.uid
        event_name = canonical_unit.raw_data.get("event_name", "") or _extract_event_name_from_text_content(
            canonical_unit.raw_data.get("text_content", "")
        )
        existing_desc = canonical_unit.raw_data.get("description", "") or _extract_event_desc_from_text_content(
            canonical_unit.raw_data.get("text_content", "")
        )

        if event.new_facts:
            if len(event.new_facts) <= self._simple_concat_threshold:
                merged_desc = self._update_description_simple(existing_desc, event.new_facts)
            else:
                merged_desc = self._update_description_llm(existing_desc, event.new_facts, event_name)

            canonical_unit.raw_data["description"] = merged_desc
            canonical_unit.raw_data["text_content"] = _format_event_text_content(event_name, merged_desc)
            canonical_unit.raw_data["event_name"] = event_name
            canonical_unit.metadata["updated_at"] = datetime.now(timezone.utc).isoformat()

        existing_participants = canonical_unit.raw_data.get("participants", [])
        if event.participants:
            existing_mentions = {p.get("mention", "") for p in existing_participants if isinstance(p, dict)}
            for p in event.participants:
                if isinstance(p, dict) and p.get("mention", "") not in existing_mentions:
                    existing_participants.append(p)
            canonical_unit.raw_data["participants"] = existing_participants

        if event.inferred_time and not canonical_unit.raw_data.get("inferred_time"):
            canonical_unit.raw_data["inferred_time"] = event.inferred_time

        session_ids = canonical_unit.metadata.get("session_ids", [])
        if isinstance(session_ids, str):
            session_ids = [session_ids]
        if session_id not in session_ids:
            session_ids.append(session_id)
        canonical_unit.metadata["session_ids"] = session_ids

        self._semantic_map.upsert_unit(canonical_unit)

        matching_dialogue_uids = find_matching_dialogue_uids(
            event.event_name, dialogue_units
        )
        for dia_uid in matching_dialogue_uids:
            try:
                self._graph.add_relationship(
                    source_uid=dia_uid,
                    target_uid=canonical_uid,
                    relationship_name=REL_COREF,
                    confidence=event.confidence,
                    mention_text=event.event_name,
                    session_id=session_id,
                )
            except KeyError:
                pass

        for participant in event.participants:
            mention = participant.get("mention", "")
            entity_uid = entity_uid_map.get(mention)
            role = participant.get("role", "participant")
            if entity_uid is not None:
                try:
                    self._graph.add_relationship(
                        source_uid=canonical_uid,
                        target_uid=entity_uid,
                        relationship_name=REL_INVOLVES,
                        subtype=role,
                        confidence=event.confidence,
                    )
                except KeyError:
                    pass

        self._update_event_name_index(str(canonical_uid), event_name)

        return canonical_uid

    def _create_new_canonical_event(
        self,
        event: ExtractedEvent,
        session_id: str,
        dialogue_units: Sequence[MemoryUnit],
        entity_uid_map: Dict[str, Uid],
    ) -> Uid:
        canonical_uid = generate_event_uid(event.event_name)

        text_content = _format_event_text_content(event.event_name, event.description)

        now = datetime.now(timezone.utc).isoformat()
        unit = MemoryUnit(
            uid=canonical_uid,
            raw_data={
                "text_content": text_content,
                "event_name": event.event_name,
                "description": event.description,
                "inferred_time": event.inferred_time,
                "participants": event.participants,
            },
            metadata={
                "type": "event",
                "session_ids": [session_id],
                "created_at": now,
                "updated_at": now,
            },
        )

        self._semantic_map.add_unit(
            unit,
            space_names=[self._event_space],
            ensure_embedding=True,
        )

        matching_dialogue_uids = find_matching_dialogue_uids(
            event.event_name, dialogue_units
        )
        for dia_uid in matching_dialogue_uids:
            try:
                self._graph.add_relationship(
                    source_uid=dia_uid,
                    target_uid=canonical_uid,
                    relationship_name=REL_COREF,
                    confidence=event.confidence,
                    mention_text=event.event_name,
                    session_id=session_id,
                )
            except KeyError:
                pass

        for dia_uid in matching_dialogue_uids:
            try:
                self._graph.add_relationship(
                    source_uid=canonical_uid,
                    target_uid=dia_uid,
                    relationship_name=REL_EVIDENCED_BY,
                    mention_text=event.event_name,
                )
            except KeyError:
                pass

        for participant in event.participants:
            mention = participant.get("mention", "")
            entity_uid = entity_uid_map.get(mention)
            role = participant.get("role", "participant")
            if entity_uid is not None:
                try:
                    self._graph.add_relationship(
                        source_uid=canonical_uid,
                        target_uid=entity_uid,
                        relationship_name=REL_INVOLVES,
                        subtype=role,
                        confidence=event.confidence,
                    )
                except KeyError:
                    pass

        if event.location:
            location_id = event.location.get("linked_id")
            if location_id:
                try:
                    self._graph.add_relationship(
                        source_uid=canonical_uid,
                        target_uid=Uid(location_id),
                        relationship_name=REL_INVOLVES,
                        subtype="location",
                        confidence=event.confidence,
                    )
                except KeyError:
                    pass

        self._update_event_name_index(str(canonical_uid), event.event_name)

        return canonical_uid

    def merge_session_entities(
        self,
        session_id: str,
        dialogue_units: Sequence[MemoryUnit],
        entities: List[ExtractedEntity],
    ) -> Dict[str, Uid]:
        """Merge extracted entities into the cross-session entity index.

        For each entity, retrieves candidate matches via multi-signal
        search (name/alias, BM25, vector, graph), then uses LLM judging
        to decide whether to merge with an existing canonical entity or
        create a new one. Handles description merging and alias updates.

        Args:
            session_id: The current session identifier.
            dialogue_units: Dialogue MemoryUnits from the session (used for
                EVIDENCED_BY edge creation).
            entities: Extracted entities from the current session.

        Returns:
            Dict mapping entity_name to the UID of the canonical entity
            (either existing or newly created).
        """
        with self._lock:
            if not self._is_initialized:
                self.initialize_from_existing()

            entity_uid_map: Dict[str, Uid] = {}

            for entity in entities:
                if entity.linked_id:
                    canonical_uid = Uid(entity.linked_id)
                    canonical_unit = self._semantic_map.get_unit(canonical_uid)
                    if canonical_unit is not None:
                        uid = self._merge_to_existing_entity(
                            entity, canonical_unit, session_id, dialogue_units
                        )
                        entity_uid_map[entity.entity_name] = uid
                    else:
                        uid = self._create_new_canonical_entity(
                            entity, session_id, dialogue_units
                        )
                        entity_uid_map[entity.entity_name] = uid
                    continue

                candidates = self._retrieve_candidate_entities(entity)

                if candidates:
                    matched_idx, confidence, canonical_name_suggestion, new_aliases, reasoning = (
                        self._llm_judge_entity_match(entity, candidates)
                    )

                    if matched_idx is not None and confidence >= self._llm_confidence_threshold:
                        canonical_unit = candidates[matched_idx]
                        if new_aliases:
                            existing_aliases = canonical_unit.raw_data.get("aliases", [])
                            for alias in new_aliases:
                                if alias not in existing_aliases and _is_short_alias(alias):
                                    existing_aliases.append(alias)
                            canonical_unit.raw_data["aliases"] = existing_aliases

                        uid = self._merge_to_existing_entity(
                            entity, canonical_unit, session_id, dialogue_units
                        )
                        entity_uid_map[entity.entity_name] = uid
                        logger.debug(
                            f"Entity '{entity.entity_name}' merged to "
                            f"'{canonical_unit.raw_data.get('entity_name', '')}' "
                            f"(conf={confidence:.2f})"
                        )
                        continue

                uid = self._create_new_canonical_entity(entity, session_id, dialogue_units)
                entity_uid_map[entity.entity_name] = uid
                logger.debug(f"Entity '{entity.entity_name}' created as new canonical")

            return entity_uid_map

    def merge_session_events(
        self,
        session_id: str,
        dialogue_units: Sequence[MemoryUnit],
        events: List[ExtractedEvent],
        entity_uid_map: Dict[str, Uid],
    ) -> Dict[str, Uid]:
        """Merge new events with cross-session event index.

        Args:
            session_id: Session identifier.
            dialogue_units: Dialogue units from the session.
            events: Newly extracted events.
            entity_uid_map: Map from entity name to canonical UID.

        Returns:
            Map from event name to canonical UID.
        """
        with self._lock:
            if not self._is_initialized:
                self.initialize_from_existing()

            event_uid_map: Dict[str, Uid] = {}

            for event in events:
                if event.linked_id:
                    canonical_uid = Uid(event.linked_id)
                    canonical_unit = self._semantic_map.get_unit(canonical_uid)
                    if canonical_unit is not None:
                        uid = self._merge_to_existing_event(
                            event, canonical_unit, session_id, dialogue_units, entity_uid_map
                        )
                        event_uid_map[event.event_name] = uid
                    else:
                        uid = self._create_new_canonical_event(
                            event, session_id, dialogue_units, entity_uid_map
                        )
                        event_uid_map[event.event_name] = uid
                    continue

                candidates = self._retrieve_candidate_events(event, entity_uid_map)

                if candidates:
                    matched_idx, confidence, canonical_name_suggestion, reasoning = (
                        self._llm_judge_event_match(event, candidates)
                    )

                    if matched_idx is not None and confidence >= self._llm_confidence_threshold:
                        canonical_unit = candidates[matched_idx]
                        uid = self._merge_to_existing_event(
                            event, canonical_unit, session_id, dialogue_units, entity_uid_map
                        )
                        event_uid_map[event.event_name] = uid
                        logger.debug(
                            f"Event '{event.event_name}' merged to "
                            f"'{canonical_unit.raw_data.get('event_name', '')}' "
                            f"(conf={confidence:.2f})"
                        )
                        continue

                uid = self._create_new_canonical_event(
                    event, session_id, dialogue_units, entity_uid_map
                )
                event_uid_map[event.event_name] = uid
                logger.debug(f"Event '{event.event_name}' created as new canonical")

            return event_uid_map

    def write_session_edges(
        self,
        dialogue_units: Sequence[MemoryUnit],
        entity_uid_map: Dict[str, Uid],
        event_uid_map: Dict[str, Uid],
        entity_relations: List[ExtractedRelation],
        causal_relations: List[ExtractedCausalRelation],
    ) -> None:
        """Write coreference, relation, and causal edges for a session.

        Creates COREF edges from dialogue → canonical entity/event,
        RELATED_TO edges for entity relations, and CAUSES/CAUSED_BY
        edges for causal event chains.

        Args:
            dialogue_units: MemoryUnits whose text_content referenced these entities/events.
            entity_uid_map: Name→canonical UID map for entities.
            event_uid_map: Name→canonical UID map for events.
            entity_relations: Extracted entity relationships.
            causal_relations: Extracted causal event relationships.
        """
        for rel in entity_relations:
            source_uid = entity_uid_map.get(rel.source)
            target_uid = entity_uid_map.get(rel.target)
            if source_uid is not None and target_uid is not None:
                try:
                    self._graph.add_relationship(
                        source_uid=source_uid,
                        target_uid=target_uid,
                        relationship_name=REL_RELATED_TO,
                        subtype=rel.subtype or "",
                        confidence=rel.confidence,
                    )
                except KeyError:
                    pass

        for cr in causal_relations:
            cause_uid = event_uid_map.get(cr.cause_event)
            effect_uid = event_uid_map.get(cr.effect_event)
            if cause_uid is not None and effect_uid is not None:
                try:
                    self._graph.add_relationship(
                        source_uid=cause_uid,
                        target_uid=effect_uid,
                        relationship_name=REL_CAUSES,
                        confidence=cr.confidence,
                    )
                except KeyError:
                    pass
                try:
                    self._graph.add_relationship(
                        source_uid=effect_uid,
                        target_uid=cause_uid,
                        relationship_name=REL_CAUSED_BY,
                        confidence=cr.confidence,
                    )
                except KeyError:
                    pass

    def _update_entity_description(self, uid: Uid, new_facts: List[str]) -> None:
        unit = self._semantic_map.get_unit(uid)
        if unit is None:
            return

        entity_name = _extract_entity_name_from_text_content(unit.raw_data.get("text_content", ""))
        entity_type = unit.raw_data.get("entity_type", "") or _extract_entity_type_from_text_content(unit.raw_data.get("text_content", ""))
        existing_desc = _extract_entity_desc_from_text_content(unit.raw_data.get("text_content", ""))

        prompt = DESCRIPTION_MERGE_PROMPT.format(
            name=entity_name,
            type=entity_type,
            existing_description=existing_desc,
            new_facts=json.dumps(new_facts),
            session_facts=json.dumps(unit.metadata.get("session_facts", [])),
        )

        response = self._llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        try:
            data = parse_json_response(response.content)
            merged_desc = data.get("merged_description", existing_desc)

            unit.raw_data["text_content"] = _format_entity_text_content(entity_name, entity_type, merged_desc)
            unit.raw_data["entity_name"] = entity_name
            unit.metadata["updated_at"] = datetime.now(timezone.utc).isoformat()
            self._semantic_map.upsert_unit(unit)
        except json.JSONDecodeError:
            pass

    def _update_event_description(
        self,
        uid: Uid,
        new_facts: List[str],
        new_participants: List[Dict[str, Any]],
        new_time: Optional[str],
    ) -> None:
        unit = self._semantic_map.get_unit(uid)
        if unit is None:
            return

        event_name = _extract_event_name_from_text_content(unit.raw_data.get("text_content", ""))
        existing_desc = unit.raw_data.get("description", "") or _extract_event_desc_from_text_content(unit.raw_data.get("text_content", ""))
        existing_participants = unit.raw_data.get("participants", [])
        existing_time = unit.raw_data.get("inferred_time")

        prompt = EVENT_DESCRIPTION_MERGE_PROMPT.format(
            name=event_name,
            existing_description=existing_desc,
            existing_participants=json.dumps(existing_participants),
            existing_time=existing_time or "unknown",
            new_facts=json.dumps(new_facts),
            new_participants=json.dumps(new_participants),
            new_time=new_time or "unknown",
        )

        response = self._llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        try:
            data = parse_json_response(response.content)

            merged_desc = data.get("merged_description", existing_desc)
            unit.raw_data["description"] = merged_desc
            unit.raw_data["text_content"] = _format_event_text_content(event_name, merged_desc)
            unit.raw_data["event_name"] = event_name
            unit.raw_data["participants"] = data.get("merged_participants", existing_participants)
            unit.raw_data["inferred_time"] = data.get("merged_time", existing_time)
            unit.metadata["updated_at"] = datetime.now(timezone.utc).isoformat()
            self._semantic_map.upsert_unit(unit)
        except json.JSONDecodeError:
            pass

    def merge_and_write(
        self,
        session: "Session",
        session_units: Sequence[MemoryUnit],
        session_space: str,
        pipeline_result: PipelineResult,
    ) -> Tuple[Dict[str, Uid], Dict[str, Uid]]:
        """Full cycle: merge entities/events, then write edges to graph.

        Args:
            session: The session object.
            session_units: MemoryUnits in this session.
            session_space: Space name for this session's units.
            pipeline_result: Pipeline output from this session.

        Returns:
            Tuple of (entity_uid_map, event_uid_map).
        """
        session_id = session.session_id
        dialogue_units = list(session_units)

        entity_uid_map = self.merge_session_entities(
            session_id, dialogue_units, pipeline_result.entities
        )

        event_uid_map = self.merge_session_events(
            session_id, dialogue_units, pipeline_result.events, entity_uid_map
        )

        self.write_session_edges(
            dialogue_units,
            entity_uid_map,
            event_uid_map,
            pipeline_result.entity_relations,
            pipeline_result.causal_relations,
        )

        logger.info(
            f"Session {session_id}: merged {len(entity_uid_map)} entities, "
            f"{len(event_uid_map)} events"
        )

        return entity_uid_map, event_uid_map
