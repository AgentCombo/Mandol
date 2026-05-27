"""Legacy multi-dimensional semantic graph builder.

This module implements an older, dimension-based approach to building semantic
graphs where each dimension independently processes units and contributes edges.
For new development, prefer MemorySystem.build_high_level() which uses the
UnifiedFactPipeline approach.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import re
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Set

import numpy as np

from ..semantic_graph import SemanticGraphService
from ...domain.memory_unit import MemoryUnit
from ...domain.types import SpaceName, Uid


def _slugify(text: str, *, max_len: int = 64) -> str:
    """Normalize text for use in identifiers.

    Lowercases, replaces non-alphanumeric chars with '_', strips leading/trailing
    underscores, and truncates to 80 chars.

    Args:
        text: Input string to slugify.

    Returns:
        A slug-safe string suitable for use in identifiers.
    """
    t = str(text).strip().lower()
    t = re.sub(r"[^a-z0-9]+", "_", t)
    t = re.sub(r"_+", "_", t).strip("_")
    if not t:
        t = "x"
    return t[:max_len]


def _stable_hash(text: str, *, n: int = 12) -> str:
    h = hashlib.sha256(str(text).encode("utf-8")).hexdigest()
    return h[: max(4, int(n))]


@dataclass(frozen=True, slots=True)
class SpaceNamingPolicy:
    base_suffix: str = "_base_memory"
    high_level_suffix: str = "_high_level_memory"

    episodic_suffix: str = "_episodic"
    knowledge_suffix: str = "_knowledge"
    emotional_suffix: str = "_emotional"
    procedural_suffix: str = "_procedural"
    insights_suffix: str = "_insights"

    episodic_summary_suffix: str = "_episodic_summary"
    episodic_event_suffix: str = "_episodic_event"
    knowledge_summary_suffix: str = "_knowledge_summary"
    knowledge_entity_suffix: str = "_knowledge_entity"

    def base_memory(self, base_space_name: str) -> SpaceName:
        """Get the base memory space name from a base space name.

        Args:
            base_space_name: The base space name.

        Returns:
            SpaceName with base memory suffix.
        """
        return SpaceName(f"{base_space_name}{self.base_suffix}")

    def high_level_memory(self, base_space_name: str) -> SpaceName:
        return SpaceName(f"{base_space_name}{self.high_level_suffix}")

    def episodic(self, base_space_name: str) -> SpaceName:
        return SpaceName(f"{base_space_name}{self.episodic_suffix}")

    def knowledge(self, base_space_name: str) -> SpaceName:
        return SpaceName(f"{base_space_name}{self.knowledge_suffix}")

    def emotional(self, base_space_name: str) -> SpaceName:
        return SpaceName(f"{base_space_name}{self.emotional_suffix}")

    def procedural(self, base_space_name: str) -> SpaceName:
        return SpaceName(f"{base_space_name}{self.procedural_suffix}")

    def insights(self, base_space_name: str) -> SpaceName:
        return SpaceName(f"{base_space_name}{self.insights_suffix}")

    def episodic_summary(self, base_space_name: str) -> SpaceName:
        return SpaceName(f"{base_space_name}{self.episodic_summary_suffix}")

    def episodic_event(self, base_space_name: str) -> SpaceName:
        return SpaceName(f"{base_space_name}{self.episodic_event_suffix}")

    def knowledge_summary(self, base_space_name: str) -> SpaceName:
        return SpaceName(f"{base_space_name}{self.knowledge_summary_suffix}")

    def knowledge_entity(self, base_space_name: str) -> SpaceName:
        return SpaceName(f"{base_space_name}{self.knowledge_entity_suffix}")


@dataclass(slots=True)
class MultiDimBuildContext:
    """Shared context passed to each dimension during graph building.

    Attributes:
        graph: SemanticGraphService for unit and edge CRUD.
        base_space_name: Root space name for this build run.
        naming: SpaceNamingPolicy for constructing hierarchical names.
        config: Arbitrary config dict (from YAML or caller).
    """
    graph: SemanticGraphService
    base_space_name: SpaceName
    naming: SpaceNamingPolicy = field(default_factory=SpaceNamingPolicy)
    config: Dict[str, Any] = field(default_factory=dict)

    @property
    def input_space_name(self) -> SpaceName:
        return SpaceName(str(self.base_space_name))

    @property
    def base_memory_space_name(self) -> SpaceName:
        return self.naming.base_memory(str(self.base_space_name))

    @property
    def high_level_space_name(self) -> SpaceName:
        return self.naming.high_level_memory(str(self.base_space_name))


class DimensionBuilder(Protocol):
    """Protocol for a dimension that can augment the semantic graph.

    Attributes:
        name: Short dimension identifier string.
    """
    name: str

    def build(self, ctx: MultiDimBuildContext) -> None:  # pragma: no cover
        ...


class LayoutNormalizationDimension:
    """Creates the hierarchical space layout structure.

    Sets up base memory → high level memory → episodic/knowledge/emotional/
    procedural/insights → sub-category leaf spaces.

    Args:
        source_spaces_metadata_key: Metadata key for tracking source spaces.
    """
    name = "layout_normalization"

    def __init__(self, *, source_spaces_metadata_key: str = "_mdsg_source_spaces"):
        self._source_spaces_metadata_key = str(source_spaces_metadata_key)

    def build(self, ctx: MultiDimBuildContext) -> None:
        sm = ctx.graph.semantic_map

        input_space = sm.create_space(ctx.input_space_name)
        base_space = sm.create_space(ctx.base_memory_space_name)
        high = sm.create_space(ctx.high_level_space_name)

        sm.attach_child_space(input_space.name, base_space.name, ensure_exists=True)
        sm.attach_child_space(input_space.name, high.name, ensure_exists=True)

        episodic = sm.create_space(ctx.naming.episodic(str(ctx.base_space_name)))
        knowledge = sm.create_space(ctx.naming.knowledge(str(ctx.base_space_name)))
        emotional = sm.create_space(ctx.naming.emotional(str(ctx.base_space_name)))
        procedural = sm.create_space(ctx.naming.procedural(str(ctx.base_space_name)))
        insights = sm.create_space(ctx.naming.insights(str(ctx.base_space_name)))

        for child in [episodic, knowledge, emotional, procedural, insights]:
            sm.attach_child_space(high.name, child.name, ensure_exists=True)

        episodic_summary = sm.create_space(
            ctx.naming.episodic_summary(str(ctx.base_space_name))
        )
        episodic_event = sm.create_space(
            ctx.naming.episodic_event(str(ctx.base_space_name))
        )
        knowledge_summary = sm.create_space(
            ctx.naming.knowledge_summary(str(ctx.base_space_name))
        )
        knowledge_entity = sm.create_space(
            ctx.naming.knowledge_entity(str(ctx.base_space_name))
        )

        sm.attach_child_space(episodic.name, episodic_summary.name, ensure_exists=True)
        sm.attach_child_space(episodic.name, episodic_event.name, ensure_exists=True)
        sm.attach_child_space(knowledge.name, knowledge_summary.name, ensure_exists=True)
        sm.attach_child_space(knowledge.name, knowledge_entity.name, ensure_exists=True)

        units = sm.get_units_in_spaces([input_space.name], mode="union", recursive=True)
        for unit in units:
            uid = Uid(str(unit.uid))
            if uid not in base_space.unit_uids:
                sm.add_unit_to_space(uid, base_space.name)
                base_space = sm.get_space(base_space.name) or base_space

            src_spaces = unit.metadata.get(self._source_spaces_metadata_key)
            if not isinstance(src_spaces, list):
                src_spaces = []

            src_set: Set[str] = {str(s) for s in src_spaces if isinstance(s, str)}
            src_set.add(str(input_space.name))

            new_list = sorted(src_set)
            if unit.metadata.get(self._source_spaces_metadata_key) != new_list:
                unit.metadata[self._source_spaces_metadata_key] = new_list
                unit.touch()
                sm.upsert_unit(unit)


class SemanticSimilarityDimension:
    """Creates SEMANTIC_SIMILAR edges based on embedding cosine similarity.

    Args:
        top_k: Max similar edges to create per unit.
        similarity_threshold: Minimum cosine score to create an edge.
        recent_window: Number of recently added units to compare against.
    """
    name = "semantic_similarity"

    def __init__(
        self,
        *,
        threshold: float = 0.82,
        per_node_top_k: int = 10,
        relationship_name: str = "SEMANTIC_SIMILAR",
        edge_space_property: str = "space",
    ):
        self._threshold = float(threshold)
        self._per_node_top_k = int(per_node_top_k)
        self._relationship_name = str(relationship_name)
        self._edge_space_property = str(edge_space_property)

    def build(self, ctx: MultiDimBuildContext) -> None:
        sm = ctx.graph.semantic_map
        base_space = ctx.base_memory_space_name

        units = sm.get_units_in_spaces([base_space], mode="union", recursive=True)
        candidates: List[MemoryUnit] = [u for u in units if u.embedding is not None]
        if not candidates:
            return

        per_k = self._per_node_top_k
        threshold = self._threshold

        for u in candidates:
            if u.embedding is None:
                continue
            q = np.asarray(u.embedding, dtype=np.float32).reshape(-1)
            hits = sm.search_by_vector(q, top_k=max(1, per_k + 1), space_names=[base_space], recursive=True)

            added = 0
            for v, score in hits:
                if added >= per_k:
                    break
                if str(v.uid) == str(u.uid):
                    continue
                if score < threshold:
                    continue

                existing = ctx.graph.get_relationship(u.uid, v.uid, self._relationship_name)
                if existing is None:
                    ctx.graph.add_relationship(
                        u.uid,
                        v.uid,
                        self._relationship_name,
                        score=float(score),
                        threshold=float(threshold),
                        **{self._edge_space_property: str(base_space)},
                    )
                added += 1


class HighLevelSummaryApplicatorDimension:
    """Generates summary/memory units and applies them to the graph.

    Creates episodic, knowledge, emotional, and procedural summaries using
    LLM calls, stores them as MemoryUnits, and links them to source units
    via EVIDENCED_BY edges.

    Args:
        llm_provider: LLM provider for summary generation.
    """
    name = "high_level_summary_applicator"

    def __init__(
        self,
        *,
        config_key: str = "high_level_summary",
        evidenced_by_relationship: str = "EVIDENCED_BY",
    ) -> None:
        self._config_key = str(config_key)
        self._evidenced_by_relationship = str(evidenced_by_relationship)

    def build(self, ctx: MultiDimBuildContext) -> None:
        """Extract causal event chains and write edges.

        Args:
            ctx: Build context.
        """
        data = ctx.config.get(self._config_key) or {}
        if not isinstance(data, dict):
            return
        summaries = data.get("summaries") or []
        if not isinstance(summaries, list) or not summaries:
            return

        sm = ctx.graph.semantic_map
        base = str(ctx.base_space_name)

        episodic_summary_space = ctx.naming.episodic_summary(base)
        knowledge_summary_space = ctx.naming.knowledge_summary(base)
        insights_space = ctx.naming.insights(base)

        for i, item in enumerate(summaries):
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if not isinstance(text, str) or not text.strip():
                continue

            category = item.get("category")
            if not isinstance(category, str):
                category = ""
            category = category.strip().lower()

            level = item.get("level")
            if not isinstance(level, str):
                level = ""
            level = level.strip().lower()

            uid = item.get("uid")
            if not isinstance(uid, str) or not uid.strip():
                uid = f"{base}_summary_{_slugify(level or 'l')}_{i}_{_stable_hash(text)}"

            evidence_uids = item.get("evidence_uids") or []
            if not isinstance(evidence_uids, list):
                evidence_uids = []

            if category == "episodic":
                target_space = episodic_summary_space
            elif category == "knowledge":
                target_space = knowledge_summary_space
            elif category == "insights" or category == "insight":
                target_space = insights_space
            else:
                target_space = insights_space

            u = sm.get_unit(uid)
            if u is None:
                u = MemoryUnit(
                    uid=Uid(str(uid)),
                    raw_data={"text_content": text, "category": category, "level": level},
                    metadata={"type": "summary", "category": category, "level": level},
                )
                ctx.graph.add_unit(u, space_names=[target_space], ensure_embedding=False)
            else:
                if str(target_space) not in u.metadata.get("spaces", []):
                    sm.add_unit_to_space(Uid(str(u.uid)), target_space)

            for ev in evidence_uids:
                if not isinstance(ev, str) or not ev.strip():
                    continue
                if ctx.graph.semantic_map.get_unit(ev) is None:
                    continue
                if ctx.graph.get_relationship(uid, ev, self._evidenced_by_relationship) is None:
                    ctx.graph.add_relationship(uid, ev, self._evidenced_by_relationship)


class EventCausalApplicatorDimension:
    """Creates causal event chain edges in the graph.

    Processes events from the episodic event space and creates CAUSES/CAUSED_BY
    edges based on configured causal relation extraction.

    Args:
        config_key: Config dict key for event causal settings.
        causes_relationship: Relationship name for causal edges (default CAUSES).
        caused_by_relationship: Reverse causal edge name (default CAUSED_BY).
    """
    name = "event_causal_applicator"

    def __init__(
        self,
        *,
        config_key: str = "event_causal",
        evidenced_by_relationship: str = "EVIDENCED_BY",
        causes_relationship: str = "CAUSES",
        caused_by_relationship: str = "CAUSED_BY",
    ) -> None:
        self._config_key = str(config_key)
        self._evidenced_by_relationship = str(evidenced_by_relationship)
        self._causes_relationship = str(causes_relationship)
        self._caused_by_relationship = str(caused_by_relationship)

    def build(self, ctx: MultiDimBuildContext) -> None:
        data = ctx.config.get(self._config_key) or {}
        if not isinstance(data, dict):
            return

        events = data.get("events") or []
        links = data.get("causal_links") or []
        if not isinstance(events, list) and not isinstance(links, list):
            return

        sm = ctx.graph.semantic_map
        base = str(ctx.base_space_name)
        event_space = ctx.naming.episodic_event(base)

        sig_to_uid: Dict[str, str] = {}
        if isinstance(events, list):
            for item in events:
                if not isinstance(item, dict):
                    continue
                signature = item.get("signature")
                if not isinstance(signature, str) or not signature.strip():
                    continue
                text = item.get("text")
                if not isinstance(text, str) or not text.strip():
                    text = signature

                uid = item.get("uid")
                if not isinstance(uid, str) or not uid.strip():
                    uid = f"{base}_event_{_slugify(signature, max_len=80)}"
                sig_to_uid[signature] = uid

                u = sm.get_unit(uid)
                if u is None:
                    u = MemoryUnit(
                        uid=Uid(str(uid)),
                        raw_data={"text_content": text, "signature": signature},
                        metadata={"type": "event", "signature": signature},
                    )
                    ctx.graph.add_unit(u, space_names=[event_space], ensure_embedding=False)
                else:
                    sm.add_unit_to_space(Uid(str(u.uid)), event_space)

                evidence_uids = item.get("evidence_uids") or []
                if not isinstance(evidence_uids, list):
                    evidence_uids = []
                for ev in evidence_uids:
                    if not isinstance(ev, str) or not ev.strip():
                        continue
                    if sm.get_unit(ev) is None:
                        continue
                    if ctx.graph.get_relationship(uid, ev, self._evidenced_by_relationship) is None:
                        ctx.graph.add_relationship(uid, ev, self._evidenced_by_relationship)

        if not isinstance(links, list):
            return

        for item in links:
            if not isinstance(item, dict):
                continue
            s_sig = item.get("source_signature")
            t_sig = item.get("target_signature")
            if not isinstance(s_sig, str) or not isinstance(t_sig, str):
                continue
            s_uid = sig_to_uid.get(s_sig) or f"{base}_event_{_slugify(s_sig, max_len=80)}"
            t_uid = sig_to_uid.get(t_sig) or f"{base}_event_{_slugify(t_sig, max_len=80)}"
            if sm.get_unit(s_uid) is None or sm.get_unit(t_uid) is None:
                continue

            rel_type = item.get("type")
            if isinstance(rel_type, str):
                rel_type = rel_type.strip().upper()
            else:
                rel_type = "CAUSES"
            if rel_type not in {self._causes_relationship, self._caused_by_relationship}:
                rel_type = self._causes_relationship

            if ctx.graph.get_relationship(s_uid, t_uid, rel_type) is None:
                ctx.graph.add_relationship(s_uid, t_uid, rel_type)


class EntityRelationApplicatorDimension:
    """Creates entity relationship edges in the graph.

    Processes entities from the knowledge entity space and creates
    relationship edges based on configured entity relation extraction.

    Args:
        config_key: Config dict key for entity relation settings.
        evidenced_by_relationship: Relationship name for evidence links.
    """
    name = "entity_relation_applicator"

    def __init__(
        self,
        *,
        config_key: str = "entity_relation",
        evidenced_by_relationship: str = "EVIDENCED_BY",
    ) -> None:
        self._config_key = str(config_key)
        self._evidenced_by_relationship = str(evidenced_by_relationship)

    def build(self, ctx: MultiDimBuildContext) -> None:
        data = ctx.config.get(self._config_key) or {}
        if not isinstance(data, dict):
            return

        entities = data.get("entities") or []
        relations = data.get("relations") or []
        if not isinstance(entities, list) and not isinstance(relations, list):
            return

        sm = ctx.graph.semantic_map
        base = str(ctx.base_space_name)
        entity_space = ctx.naming.knowledge_entity(base)

        name_to_uid: Dict[str, str] = {}

        if isinstance(entities, list):
            for item in entities:
                if not isinstance(item, dict):
                    continue
                text = item.get("text")
                if not isinstance(text, str) or not text.strip():
                    continue
                uid = item.get("uid")
                if not isinstance(uid, str) or not uid.strip():
                    uid = f"{base}_entity_{_slugify(text, max_len=80)}"
                name_to_uid[text] = uid

                u = sm.get_unit(uid)
                if u is None:
                    u = MemoryUnit(
                        uid=Uid(str(uid)),
                        raw_data={"text_content": text},
                        metadata={"type": "entity", "text": text},
                    )
                    ctx.graph.add_unit(u, space_names=[entity_space], ensure_embedding=False)
                else:
                    sm.add_unit_to_space(Uid(str(u.uid)), entity_space)

                evidence_uids = item.get("evidence_uids") or []
                if not isinstance(evidence_uids, list):
                    evidence_uids = []
                for ev in evidence_uids:
                    if not isinstance(ev, str) or not ev.strip():
                        continue
                    if sm.get_unit(ev) is None:
                        continue
                    if ctx.graph.get_relationship(uid, ev, self._evidenced_by_relationship) is None:
                        ctx.graph.add_relationship(uid, ev, self._evidenced_by_relationship)

        if not isinstance(relations, list):
            return

        for item in relations:
            if not isinstance(item, dict):
                continue
            source = item.get("source")
            target = item.get("target")
            rel_type = item.get("type")
            if not isinstance(source, str) or not isinstance(target, str) or not isinstance(rel_type, str):
                continue
            rel_type = rel_type.strip().upper()
            if not rel_type:
                continue

            s_uid = name_to_uid.get(source) or f"{base}_entity_{_slugify(source, max_len=80)}"
            t_uid = name_to_uid.get(target) or f"{base}_entity_{_slugify(target, max_len=80)}"
            if sm.get_unit(s_uid) is None or sm.get_unit(t_uid) is None:
                continue

            if ctx.graph.get_relationship(s_uid, t_uid, rel_type) is None:
                ctx.graph.add_relationship(s_uid, t_uid, rel_type)

            evidence_uids = item.get("evidence_uids") or []
            if not isinstance(evidence_uids, list):
                evidence_uids = []
            for ev in evidence_uids:
                if not isinstance(ev, str) or not ev.strip():
                    continue
                if sm.get_unit(ev) is None:
                    continue
                if ctx.graph.get_relationship(s_uid, ev, self._evidenced_by_relationship) is None:
                    ctx.graph.add_relationship(s_uid, ev, self._evidenced_by_relationship)
                if ctx.graph.get_relationship(t_uid, ev, self._evidenced_by_relationship) is None:
                    ctx.graph.add_relationship(t_uid, ev, self._evidenced_by_relationship)


class MultiDimSemanticGraphBuilder:
    """Orchestrates multiple dimensions to build a semantic graph.

    Registers a list of DimensionBuilder instances and runs them in order
    against a given base space. Can optionally filter which dimensions to
    enable and pass a shared config dict.

    Args:
        graph: SemanticGraphService for unit and edge CRUD.
        naming: SpaceNamingPolicy (default created if None).
        dimensions: Initial list of DimensionBuilder instances.
    """

    def __init__(
        self,
        *,
        graph: SemanticGraphService,
        naming: Optional[SpaceNamingPolicy] = None,
        dimensions: Optional[Sequence[DimensionBuilder]] = None,
    ) -> None:
        self._graph = graph
        self._naming = naming or SpaceNamingPolicy()
        self._dimensions: List[DimensionBuilder] = list(dimensions or [])

    def register(self, dimension: DimensionBuilder) -> None:
        """Add a dimension builder to the pipeline.

        Args:
            dimension: A DimensionBuilder instance.
        """
        self._dimensions.append(dimension)

    def ensure_layout(self, base_space_name: str) -> None:
        """Create the space hierarchy layout for a base space.

        Args:
            base_space_name: The base space name to layout.
        """
        ctx = MultiDimBuildContext(graph=self._graph, base_space_name=SpaceName(base_space_name), naming=self._naming)
        LayoutNormalizationDimension().build(ctx)

    def augment_from_base(
        self,
        base_space_name: str,
        *,
        enabled_dimensions: Optional[Sequence[str]] = None,
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Run all registered dimensions against a base space.

        Args:
            base_space_name: Target base space name.
            enabled_dimensions: Optional list of dimension names to enable.
            config: Optional shared config dict for dimensions.
        """
        ctx = MultiDimBuildContext(
            graph=self._graph,
            base_space_name=SpaceName(base_space_name),
            naming=self._naming,
            config=dict(config or {}),
        )

        enabled: Optional[Set[str]]
        if enabled_dimensions is None:
            enabled = None
        else:
            enabled = {str(x) for x in enabled_dimensions}

        for dim in self._dimensions:
            if enabled is not None and getattr(dim, "name", "") not in enabled:
                continue
            dim.build(ctx)
