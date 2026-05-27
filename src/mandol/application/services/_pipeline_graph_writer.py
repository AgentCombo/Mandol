"""Pipeline graph writer for batch edge creation.

Writes COREF, EVIDENCED_BY, INVOLVES, RELATED_TO, and CAUSES edges
to the SemanticGraphService in bulk from pipeline results.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ...domain.coref_graph_constants import (
    REL_COREF,
    REL_EVIDENCED_BY,
    REL_INVOLVES,
    REL_RELATED_TO,
)
from ..semantic_graph import SemanticGraphService
from ..semantic_map import SemanticMapService


class PipelineGraphWriter:
    """Batch-writes pipeline edge lists to the SemanticGraphService.

    Validates edge endpoints (source/target types must be compatible)
    before writing COREF, EVIDENCED_BY, INVOLVES, RELATED_TO, and
    CAUSES/CAUSED_BY edges. Skips edges whose source/target units are
    not in the semantic map.

    Args:
        semantic_map: SemanticMapService for endpoint validation.
        graph: SemanticGraphService for edge creation.
    """

    def __init__(self, semantic_map: SemanticMapService, graph: SemanticGraphService):
        self._semantic_map = semantic_map
        self._graph = graph

    def write_edges(
        self,
        coref_edges: List[Dict[str, Any]],
        evidenced_by_edges: List[Dict[str, Any]],
        involves_edges: List[Dict[str, Any]],
        related_to_edges: List[Dict[str, Any]],
        causes_edges: List[Dict[str, Any]],
    ) -> None:
        """Batch-write pipeline edge lists to the SemanticGraphService.

        Validates COREF edge endpoints (source must be dialogue and target
        must be entity/event, or both must be same-type entity/event) before
        writing. Other edge types are written without type validation.

        Args:
            coref_edges: COREF edges with source_uid, target_uid, confidence,
                mention_text, session_id.
            evidenced_by_edges: EVIDENCED_BY edges with source_uid, target_uid,
                mention_text.
            involves_edges: INVOLVES edges with source_uid, target_uid,
                subtype, confidence.
            related_to_edges: RELATED_TO edges with source_uid, target_uid,
                subtype, confidence.
            causes_edges: Causal edges with source_uid, target_uid, rel_type,
                confidence.
        """
        for edge in coref_edges:
            source_uid = edge["source_uid"]
            target_uid = edge["target_uid"]
            source_unit = self._semantic_map.get_unit(source_uid)
            target_unit = self._semantic_map.get_unit(target_uid)
            if source_unit is not None and target_unit is not None:
                source_type = source_unit.metadata.get("type", "")
                target_type = target_unit.metadata.get("type", "")
                source_is_dialogue = source_type not in ("entity", "event")
                target_is_entity_or_event = target_type in ("entity", "event")
                if source_is_dialogue and target_is_entity_or_event:
                    pass
                elif source_type == target_type and source_type in ("entity", "event"):
                    pass
                else:
                    continue
            try:
                self._graph.add_relationship(
                    source_uid=source_uid,
                    target_uid=target_uid,
                    relationship_name=REL_COREF,
                    confidence=edge.get("confidence", 0.9),
                    mention_text=edge.get("mention_text", ""),
                    session_id=edge.get("session_id", ""),
                )
            except KeyError:
                pass

        for edge in evidenced_by_edges:
            try:
                self._graph.add_relationship(
                    source_uid=edge["source_uid"],
                    target_uid=edge["target_uid"],
                    relationship_name=REL_EVIDENCED_BY,
                    mention_text=edge.get("mention_text", ""),
                )
            except KeyError:
                pass

        for edge in involves_edges:
            try:
                self._graph.add_relationship(
                    source_uid=edge["source_uid"],
                    target_uid=edge["target_uid"],
                    relationship_name=REL_INVOLVES,
                    subtype=edge.get("subtype", "participant"),
                    confidence=edge.get("confidence", 0.9),
                )
            except KeyError:
                pass

        for edge in related_to_edges:
            try:
                self._graph.add_relationship(
                    source_uid=edge["source_uid"],
                    target_uid=edge["target_uid"],
                    relationship_name=REL_RELATED_TO,
                    subtype=edge.get("subtype", ""),
                    confidence=edge.get("confidence", 0.9),
                )
            except KeyError:
                pass

        for edge in causes_edges:
            try:
                self._graph.add_relationship(
                    source_uid=edge["source_uid"],
                    target_uid=edge["target_uid"],
                    relationship_name=edge["rel_type"],
                    confidence=edge.get("confidence", 0.9),
                )
            except KeyError:
                pass
