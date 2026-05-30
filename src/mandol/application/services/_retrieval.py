"""Memory retrieval service extracted from MemorySystem.

Performs multi-group retrieval (base/entity/event/summary) with auto-build
trigger support and Cross-Encoder reranking.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from ...domain.memory_unit import MemoryUnit
from ...domain.types import SpaceName
from ...retrieval.types import (
    CausalChainResult,
    CausalStep,
    ReasoningHit,
    SearchHit,
    SessionContextResult,
    TimelineResult,
)

logger = logging.getLogger(__name__)

# Retrieval group keys for the four memory categories.
RETRIEVAL_GROUP_BASE = "base"
RETRIEVAL_GROUP_ENTITY = "entity"
RETRIEVAL_GROUP_EVENT = "event"
RETRIEVAL_GROUP_SUMMARY = "summary"


class MemoryRetrievalService:
    """Handles all retrieval operations across MemorySystem.

    Supports holistic retrieval (4-group pipeline), per-space retrieval,
    and view-based retrieval (base_memory/entity_relation/event_causal/etc.).

    Args:
        semantic_map: SemanticMapService for ANN and rerank access.
        graph: SemanticGraphService for BFS expansion access.
        naming: SpaceNamingPolicy for constructing space names.
        root: Root SpaceName.
        config: MemorySystemConfig for similarity thresholds.
    """

    def __init__(
        self,
        semantic_map,
        graph,
        naming,
        root: SpaceName,
        config,
    ):
        self._semantic_map = semantic_map
        self._graph = graph
        self._naming = naming
        self._root = root
        self._cfg = config
        self._hybrid_retriever = None
        self._subgraph_hop_retriever = None

    def _get_retrieval_groups(
        self,
    ) -> Dict[str, Tuple[str, List[SpaceName]]]:
        return {
            RETRIEVAL_GROUP_BASE: (
                "base memory",
                [self._naming.base_memory(self._root)],
            ),
            RETRIEVAL_GROUP_EVENT: (
                "events",
                [self._naming.episodic_event(self._root)],
            ),
            RETRIEVAL_GROUP_ENTITY: (
                "entities",
                [self._naming.knowledge_entity(self._root)],
            ),
            RETRIEVAL_GROUP_SUMMARY: (
                "summaries and insights",
                [
                    self._naming.episodic_summary(self._root),
                    self._naming.knowledge_summary(self._root),
                    self._naming.insights(self._root),
                ],
            ),
        }

    def _are_high_level_spaces_empty(self) -> bool:
        """Check if all high-level memory spaces (entity, event, summary) are empty.

        Returns:
            True when every high-level space contains zero units, indicating
            that build_high_level has never been called or produced no output.
        """
        high_level_spaces = [
            self._naming.episodic_event(self._root),
            self._naming.knowledge_entity(self._root),
            self._naming.episodic_summary(self._root),
            self._naming.knowledge_summary(self._root),
            self._naming.insights(self._root),
        ]
        for space_name in high_level_spaces:
            units = self._semantic_map.get_units_in_spaces([space_name])
            if units:
                return False
        return True

    def _is_base_memory_empty(self) -> bool:
        """Check if the base memory space has no units.

        Returns:
            True when base_memory contains zero units, meaning no memory
            has been ingested yet.
        """
        base_space = self._naming.base_memory(self._root)
        units = self._semantic_map.get_units_in_spaces([base_space])
        return len(units) == 0

    def holistic_retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        use_rerank: bool = True,
        auto_build_if_empty: bool = True,
        build_trigger: Optional[Callable[[], None]] = None,
        skip_views: Optional[List[str]] = None,
    ) -> List[SearchHit]:
        """Retrieve across all memory groups with optional auto-build.

        Searches base memory, entity, event, and summary spaces in
        parallel, deduplicates by UID, then optionally reranks the
        combined candidates.

        Args:
            query: Natural language search query.
            top_k: Maximum number of results to return.
            use_rerank: Enable Cross-Encoder reranking (default True).
            auto_build_if_empty: Trigger build_high_level when high-level
                spaces are empty but base memory exists.
            build_trigger: Callable that runs build_high_level (typically
                MemorySystem.build_high_level).
            skip_views: Optional list of retrieval group keys to skip.
                Valid keys: "base", "entity", "event", "summary".

        Returns:
            Ranked list of SearchHit objects ordered by final_score.
        """
        if auto_build_if_empty and build_trigger is not None:
            if not self._is_base_memory_empty() and self._are_high_level_spaces_empty():
                logger.info(
                    "High-level memory spaces are empty, triggering auto-build"
                )
                build_trigger()

        groups = self._get_retrieval_groups()
        if skip_views:
            groups = {k: v for k, v in groups.items() if k not in skip_views}

        all_hits: List[SearchHit] = []
        seen: Set[str] = set()

        for group_key, (group_label, space_names) in groups.items():
            group_hits = self._search_group(
                query,
                space_names,
                top_k=max(1, int(top_k) * 3),
                use_rerank=False,
            )
            for hit in group_hits:
                uid = str(hit.unit.uid)
                if uid not in seen:
                    seen.add(uid)
                    all_hits.append(hit)

        if not all_hits:
            return []

        if use_rerank and self._semantic_map.get_reranker() is not None:
            all_units = [hit.unit for hit in all_hits]
            reranked = self._semantic_map.get_reranker().rerank(
                query, all_units, top_k=max(1, int(top_k))
            )
            uid_to_hit = {str(hit.unit.uid): hit for hit in all_hits}
            out: List[SearchHit] = []
            for unit, rerank_score in reranked:
                original = uid_to_hit.get(str(unit.uid))
                merged_scores = dict(original.scores) if original else {}
                merged_scores["rerank"] = float(rerank_score)
                merged_ranks = dict(original.ranks) if original else {}
                out.append(
                    SearchHit(
                        unit=unit,
                        final_score=float(rerank_score),
                        scores=merged_scores,
                        ranks=merged_ranks,
                    )
                )
            return out

        all_hits.sort(
            key=lambda h: h.unit.metadata.get("timestamp", ""), reverse=True
        )
        return all_hits[: max(0, int(top_k))]

    def retrieve_in_space(
        self,
        query: str,
        space_name: str,
        *,
        top_k: int = 10,
        use_rerank: bool = True,
    ) -> List[SearchHit]:
        """Retrieve within a single named space.

        Args:
            query: Natural language search query.
            space_name: Name of the space to search.
            top_k: Maximum number of results.
            use_rerank: Enable Cross-Encoder reranking.

        Returns:
            List of SearchHit objects from the hybrid retriever.
        """
        return self._search_group(
            query,
            [SpaceName(space_name)],
            top_k=top_k,
            use_rerank=use_rerank,
        )

    def retrieve_by_view(
        self,
        query: str,
        view: str,
        *,
        top_k: int = 10,
        use_rerank: bool = True,
    ) -> List[SearchHit]:
        """Retrieve using a named view that maps to one or more spaces.

        Args:
            query: Natural language search query.
            view: View name — one of: base_memory, entity_relation,
                event_causal, emotional, episodic, knowledge, procedural,
                insights.
            top_k: Maximum number of results.
            use_rerank: Enable Cross-Encoder reranking.

        Returns:
            List of SearchHit objects from the hybrid retriever.

        Raises:
            ValueError: If *view* is not a recognised view name.
        """
        view_space_map = {
            "base_memory": [self._naming.base_memory(self._root)],
            "entity_relation": [self._naming.knowledge_entity(self._root)],
            "event_causal": [self._naming.episodic_event(self._root)],
            "emotional": [self._naming.emotional(self._root)],
            "episodic": [self._naming.episodic_summary(self._root)],
            "knowledge": [self._naming.knowledge_summary(self._root)],
            "procedural": [self._naming.procedural(self._root)],
            "insights": [self._naming.insights(self._root)],
        }

        space_names = view_space_map.get(view)
        if space_names is None:
            raise ValueError(
                f"Unknown view: {view}. "
                f"Available views: {', '.join(view_space_map.keys())}"
            )

        return self._search_group(
            query,
            space_names,
            top_k=top_k,
            use_rerank=use_rerank,
        )

    def retrieve_event_causal_chain(
        self,
        query: str,
        *,
        max_hops: int = 3,
        direction: str = "both",
        top_k: int = 5,
    ) -> "CausalChainResult":
        """Trace causal chains along CAUSES/CAUSED_BY edges.

        Retrieves the most semantically relevant event and expands along
        causal relationship edges to build an ordered causal chain.

        Args:
            query: Natural language search query for the root event.
            max_hops: Maximum number of BFS hops in the causal chain.
            direction: Traversal direction — "forward" (downstream CAUSES),
                "backward" (upstream CAUSED_BY), or "both".
            top_k: Maximum number of results per expansion step.

        Returns:
            CausalChainResult with the ordered chain, root event, and leaf
            events (events at the end of the chain with no further outgoing
            CAUSES edges).
        """
        hits = self.retrieve_by_view(query, view="event_causal", top_k=top_k)
        if not hits:
            return CausalChainResult()

        root_event = hits[0].unit
        expanded = self._graph.bfs_expand_units(
            seeds=[root_event],
            per_seed=top_k,
            hops=max_hops,
            rel_type="CAUSES",
        )

        if not expanded:
            return CausalChainResult(root_event=root_event)

        chain: List[CausalStep] = []
        all_chain_uids = {str(e.uid) for e in expanded}
        all_chain_uids.add(str(root_event.uid))

        for event in expanded:
            if str(event.uid) == str(root_event.uid):
                continue

            # Determine causal relationship between root and this event.
            fwd_rel = self._graph.get_relationship(
                root_event.uid, event.uid, "CAUSES"
            )
            bwd_rel = self._graph.get_relationship(
                event.uid, root_event.uid, "CAUSES"
            )

            if direction == "forward" and not fwd_rel:
                continue
            if direction == "backward" and not bwd_rel:
                continue

            if fwd_rel:
                causal_type = "CAUSES"
                step_direction = "forward"
            elif bwd_rel:
                causal_type = "CAUSED_BY"
                step_direction = "backward"
            else:
                # Indirect connection discovered via BFS — no direct edge
                # exists between root_event and this event. Skip instead of
                # fabricating a direct causal link.
                continue

            chain.append(
                CausalStep(
                    source_uid=root_event.uid,
                    target_uid=event.uid,
                    causal_type=causal_type,
                    confidence=1.0,
                    direction=step_direction,
                )
            )

        # Leaf events: no outgoing CAUSES edge to another event in the chain.
        leaf_events: List[MemoryUnit] = []
        for event in expanded:
            if str(event.uid) == str(root_event.uid):
                continue
            out_neighbors = self._graph.get_explicit_neighbors(
                [event.uid], rel_type="CAUSES", direction="out"
            )
            has_outgoing = any(
                str(n.uid) in all_chain_uids and str(n.uid) != str(event.uid)
                for n in out_neighbors
            )
            if not has_outgoing:
                leaf_events.append(event)

        # If no leaf events found, treat the last event in chain as leaf.
        if not leaf_events and chain:
            last_step = chain[-1]
            last_event = next(
                (e for e in expanded if str(e.uid) == str(last_step.target_uid)),
                None,
            )
            if last_event is not None:
                leaf_events.append(last_event)

        return CausalChainResult(
            chain=chain,
            root_event=root_event,
            leaf_events=leaf_events,
        )

    def retrieve_with_reasoning_path(
        self,
        query: str,
        *,
        max_hops: int = 2,
        hop_decay: float = 0.85,
        top_k: int = 5,
        rel_types: Optional[List[str]] = None,
    ) -> List["ReasoningHit"]:
        """SubgraphHopRetriever-based weighted multi-hop graph expansion.

        Combines base retrieval with graph traversal to produce results
        augmented with interpretable reasoning paths showing the traversal
        steps that led to each hit.

        Args:
            query: Natural language search query.
            max_hops: Maximum number of graph hops from each seed.
            hop_decay: Multiplicative decay factor per hop.
            top_k: Maximum number of results to return.
            rel_types: Optional list of relationship types to restrict
                traversal to. If None, all default relationship weights
                are used.

        Returns:
            List of ReasoningHit objects with final_score, scores, and
            reasoning_path.
        """
        from ...domain.coref_graph_constants import DEFAULT_REL_WEIGHTS
        from ...retrieval.subgraph_hop import (
            SubgraphHopConfig,
            SubgraphHopRetriever,
        )

        # Lazy-initialize the underlying HybridRetriever (shared).
        if self._hybrid_retriever is None:
            from ...retrieval.pipeline import HybridRetriever

            logger.debug("Lazy-initializing HybridRetriever")
            self._hybrid_retriever = HybridRetriever(
                graph=self._graph,
                reranker=self._semantic_map.get_reranker(),
            )

        # Build filtered or full relationship weights.
        if rel_types:
            rel_weights = {
                rt: DEFAULT_REL_WEIGHTS[rt]
                for rt in rel_types
                if rt in DEFAULT_REL_WEIGHTS
            }
        else:
            rel_weights = dict(DEFAULT_REL_WEIGHTS)

        config = SubgraphHopConfig(
            max_hops=max_hops,
            hop_decay=hop_decay,
            rel_weights=rel_weights,
        )

        if self._subgraph_hop_retriever is None:
            logger.debug("Lazy-initializing SubgraphHopRetriever")
            self._subgraph_hop_retriever = SubgraphHopRetriever(
                hybrid=self._hybrid_retriever,
                config=config,
            )
        else:
            # Update config for this call (parameters may differ).
            self._subgraph_hop_retriever._config = config

        subgraph_hits = self._subgraph_hop_retriever.search(
            query, top_k=top_k
        )

        out: List[ReasoningHit] = []
        for hit in subgraph_hits:
            out.append(
                ReasoningHit(
                    unit=hit.unit,
                    final_score=hit.final_score,
                    scores=hit.scores,
                    reasoning_path=list(hit.reasoning_path),
                )
            )
        return out

    def retrieve_entity_timeline(
        self,
        query: str,
        *,
        time_range: Optional[Tuple[str, str]] = None,
        top_k: int = 20,
    ) -> "TimelineResult":
        """Retrieve time-sorted events and dialogues for a specific entity.

        Searches for the entity, finds related units via explicit graph
        edges and base memory search, then sorts them chronologically.

        Args:
            query: Natural language search query describing the entity.
            time_range: Optional (start_iso, end_iso) timestamp filter.
            top_k: Maximum number of timeline entries to return.

        Returns:
            TimelineResult with chronologically ordered (unit, timestamp)
            pairs and the matched entity.
        """
        # Search for the entity in the entity_relation view.
        hits = self.retrieve_by_view(
            query, view="entity_relation", top_k=1
        )
        if not hits:
            # Fall back to base_memory.
            hits = self.retrieve_by_view(
                query, view="base_memory", top_k=1
            )
        if not hits:
            return TimelineResult()

        entity = hits[0].unit

        # Get neighbors via explicit graph edges (both directions to
        # capture events that link TO this entity via incoming edges).
        related = self._graph.get_explicit_neighbors([entity.uid], direction="both")

        # Also search base memory for units mentioning the entity.
        entity_name = entity.metadata.get("name", "") or str(
            entity.raw_data.get("text_content", "")
        )[:100]
        base_space = self._naming.base_memory(self._root)
        base_hits = self._search_group(
            entity_name,
            [base_space],
            top_k=top_k,
            use_rerank=False,
        )

        # Combine and deduplicate.
        all_units: Dict[str, MemoryUnit] = {}
        for unit in related:
            all_units[str(unit.uid)] = unit
        for hit in base_hits:
            uid = str(hit.unit.uid)
            if uid not in all_units:
                all_units[uid] = hit.unit

        # Exclude the entity itself from its own timeline events.
        all_units.pop(str(entity.uid), None)

        # Sort by timestamp descending using datetime-aware comparison.
        # Falls back to string comparison for unparseable timestamps.
        from datetime import datetime as _datetime
        from datetime import timezone as _timezone

        def _parse_ts(ts_str: str) -> Any:
            """Parse timestamp string to a UTC-aware datetime, or None."""
            if not ts_str:
                return None
            try:
                dt = _datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                # Normalize to UTC so naive and aware datetimes can be compared.
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=_timezone.utc)
                return dt
            except (ValueError, TypeError):
                return None

        events_with_ts: List[Tuple[MemoryUnit, Any, str]] = []
        for unit in all_units.values():
            ts_raw = str(unit.metadata.get("timestamp", "") or "")
            ts_parsed = _parse_ts(ts_raw)
            events_with_ts.append((unit, ts_parsed, ts_raw))

        # Sort: None (unparseable/missing) sorts last, parsed datetimes sort
        # descending (most recent first), falling back to string comparison
        # among unparseable timestamps.
        events_with_ts.sort(
            key=lambda x: (
                x[1] is None,           # False (0) = parsed, True (1) = unparseable last
                x[1] if x[1] is not None else x[2],
            ),
            reverse=True,
        )
        # Adjust: unparseable (None) should go last (reverse=True puts True first).
        # Re-sort with a stable two-pass: first by parsed/unparsed, then by value.
        parsed = [(u, t, r) for u, t, r in events_with_ts if t is not None]
        unparsed = [(u, t, r) for u, t, r in events_with_ts if t is None]
        parsed.sort(key=lambda x: x[1], reverse=True)
        unparsed.sort(key=lambda x: x[2], reverse=True)
        events_with_ts = parsed + unparsed

        # Filter by time_range if provided.
        if time_range:
            start, end = time_range
            start_dt = _parse_ts(start) if start else None
            end_dt = _parse_ts(end) if end else None
            filtered: List[Tuple[MemoryUnit, Any, str]] = []
            for unit, ts_parsed, ts_raw in events_with_ts:
                if start_dt and ts_parsed is not None and ts_parsed < start_dt:
                    continue
                if end_dt and ts_parsed is not None and ts_parsed > end_dt:
                    continue
                filtered.append((unit, ts_parsed, ts_raw))
            events_with_ts = filtered

        return TimelineResult(
            events=[(unit, ts_raw) for unit, _ts_parsed, ts_raw in events_with_ts[:max(0, int(top_k))]],
            entity=entity,
        )

    def retrieve_session_context(
        self,
        query: str,
        *,
        session_id: Optional[str] = None,
        include_adjacent: bool = True,
        top_k: int = 10,
    ) -> "SessionContextResult":
        """Retrieve full context for a session with adjacent session support.

        Searches base memory, groups units by session identifier, and
        optionally includes units from temporally adjacent sessions.

        Args:
            query: Natural language search query for the session.
            session_id: Specific session to retrieve. If None, the best
                matching session is selected by semantic relevance.
            include_adjacent: If True, include adjacent sessions'
                SessionContextResult entries.
            top_k: Maximum number of base hits considered for grouping.

        Returns:
            SessionContextResult with session_units, session_id, and
            optionally adjacent_sessions.
        """
        # Retrieve candidate units from base memory.
        hits = self.retrieve_by_view(
            query, view="base_memory", top_k=top_k
        )
        if not hits:
            return SessionContextResult()

        # Group units by session identifier found in metadata.
        # Try multiple common metadata keys.
        sessions: Dict[str, List[MemoryUnit]] = {}
        for hit in hits:
            unit = hit.unit
            sid = (
                unit.metadata.get("session_id")
                or unit.metadata.get("session")
                or unit.metadata.get("source_session")
                or ""
            )
            if sid:
                sessions.setdefault(str(sid), []).append(unit)

        if not sessions:
            # No session metadata found; treat all hits as one session.
            sessions[""] = [hit.unit for hit in hits]

        # Determine the target session.
        if session_id and session_id in sessions:
            best_sid = session_id
        elif session_id:
            # Requested session_id not found in results.
            return SessionContextResult(session_id=session_id)
        else:
            # Pick the session with the most units as best match.
            best_sid = max(sessions, key=lambda s: len(sessions[s]))

        session_units = sessions.get(best_sid, [])

        # Handle adjacent sessions.
        adjacent: List[SessionContextResult] = []
        if include_adjacent and len(sessions) > 1:
            # Order sessions by their latest timestamp.
            session_times: Dict[str, str] = {}
            for sid, units in sessions.items():
                latest = ""
                for u in units:
                    ts = u.metadata.get("timestamp", "") or ""
                    if ts > latest:
                        latest = ts
                if latest:
                    session_times[sid] = latest

            sorted_sids = sorted(
                session_times, key=lambda s: session_times[s]
            )
            try:
                idx = sorted_sids.index(best_sid)
            except ValueError:
                idx = -1

            # Include one session before and one after.
            if idx > 0:
                prev_sid = sorted_sids[idx - 1]
                adjacent.append(
                    SessionContextResult(
                        session_units=list(sessions.get(prev_sid, [])),
                        session_id=prev_sid,
                        adjacent_sessions=[],
                    )
                )
            if 0 <= idx < len(sorted_sids) - 1:
                next_sid = sorted_sids[idx + 1]
                adjacent.append(
                    SessionContextResult(
                        session_units=list(sessions.get(next_sid, [])),
                        session_id=next_sid,
                        adjacent_sessions=[],
                    )
                )

        return SessionContextResult(
            session_units=list(session_units),
            session_id=best_sid,
            adjacent_sessions=adjacent,
        )

    def _search_group(
        self,
        query: str,
        space_names: List[SpaceName],
        top_k: int,
        use_rerank: bool,
    ) -> List[SearchHit]:
        from ...retrieval.pipeline import HybridRetriever

        if self._hybrid_retriever is None:
            logger.debug("Lazy-initializing HybridRetriever")
            self._hybrid_retriever = HybridRetriever(
                graph=self._graph,
                reranker=self._semantic_map.get_reranker(),
            )
        results = self._hybrid_retriever.search(
            query, space_names=space_names, top_k=top_k, use_rerank=use_rerank,
        )
        return list(results)
