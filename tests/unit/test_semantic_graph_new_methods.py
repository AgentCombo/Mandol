"""Unit tests for new SemanticGraphService query methods.

Tests: get_edges_of_unit, get_node_neighbors, search_graph_relations,
trace_evidence, trace_coref, retrieve_entity_subgraph,
retrieve_summary_evidence_chain, retrieve_entity_involvement.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import numpy as np

from mandol.application.semantic_graph import SemanticGraphService
from mandol.domain.memory_unit import MemoryUnit
from mandol.domain.types import Uid


def _make_unit(uid: str, text: str = "content", spaces=None, metadata=None) -> MemoryUnit:
    """Create a MemoryUnit with given uid, text, spaces, and metadata."""
    _metadata = dict(metadata) if metadata else {}
    _metadata.setdefault("timestamp", "2025-01-01T00:00:00Z")
    if spaces:
        _metadata["spaces"] = list(spaces)
    return MemoryUnit(
        uid=Uid(uid),
        raw_data={"text_content": text},
        metadata=_metadata,
    )


class TestGetEdgesOfUnit(unittest.TestCase):
    """Tests for get_edges_of_unit."""

    def setUp(self):
        self.semantic_map = MagicMock()
        self.semantic_map.get_unit.return_value = None
        self.graph_store = MagicMock()
        self.graph_store.get_neighbors.return_value = []
        self.graph_store.get_relationship.return_value = None
        self.service = SemanticGraphService(
            semantic_map=self.semantic_map, graph_store=self.graph_store
        )

    def test_returns_empty_list_with_no_edges(self):
        self.graph_store.get_neighbors.return_value = []
        result = self.service.get_edges_of_unit("node1")
        self.assertEqual(result, [])

    def test_returns_all_edges_for_node(self):
        self.graph_store.get_neighbors.side_effect = lambda uid, rel_type=None, direction="out": (
            [Uid("node2"), Uid("node3")] if direction == "out" else [Uid("node4")]
        )
        self.graph_store.get_relationship.return_value = {"weight": 0.9}
        result = self.service.get_edges_of_unit("node1")
        # Outgoing: node1→node2, node1→node3 + Incoming: node4→node1 = 3
        self.assertEqual(len(result), 3)

    def test_filters_by_direction_out(self):
        self.graph_store.get_neighbors.side_effect = lambda uid, rel_type=None, direction="out": (
            [Uid("node2")] if direction == "out" else []
        )
        self.graph_store.get_relationship.return_value = {}
        result = self.service.get_edges_of_unit("node1", direction="out")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["source"], "node1")
        self.assertEqual(result[0]["target"], "node2")

    def test_filters_by_direction_in(self):
        self.graph_store.get_neighbors.side_effect = lambda uid, rel_type=None, direction="out": (
            [] if direction == "out" else [Uid("node3")]
        )
        self.graph_store.get_relationship.return_value = {}
        result = self.service.get_edges_of_unit("node1", direction="in")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["source"], "node3")
        self.assertEqual(result[0]["target"], "node1")

    def test_filters_by_rel_type(self):
        self.graph_store.get_neighbors.side_effect = lambda uid, rel_type=None, direction="out": (
            [Uid("node3")] if direction == "out" and rel_type == "CAUSES" else []
        )
        self.graph_store.get_relationship.return_value = {}
        result = self.service.get_edges_of_unit("node1", rel_type="CAUSES")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["type"], "CAUSES")


class TestGetNodeNeighbors(unittest.TestCase):
    """Tests for get_node_neighbors."""

    def setUp(self):
        self.semantic_map = MagicMock()
        self.semantic_map.get_unit.return_value = None
        self.graph_store = MagicMock()
        self.graph_store.get_neighbors.return_value = []
        self.service = SemanticGraphService(
            semantic_map=self.semantic_map, graph_store=self.graph_store
        )

    def test_returns_structural_neighbors_only(self):
        u1 = _make_unit("u1")
        u2 = _make_unit("u2")
        self.semantic_map.get_unit.side_effect = lambda u: {
            Uid("u1"): u1, Uid("u2"): u2,
        }.get(Uid(str(u))) if u else None
        self.graph_store.get_neighbors.return_value = [Uid("u2")]

        result = self.service.get_node_neighbors(
            "u1", include_semantic=False
        )
        self.assertIn("structural", result)
        self.assertNotIn("semantic", result)
        self.assertEqual(len(result["structural"]), 1)

    def test_returns_semantic_neighbors_only(self):
        u1 = _make_unit("u1")
        u1.embedding = np.random.rand(128).astype(np.float32)
        u2 = _make_unit("u2")
        u3 = _make_unit("u3")

        self.semantic_map.search_by_vector.return_value = [(u2, 0.9), (u3, 0.5)]
        self.semantic_map.get_unit.side_effect = lambda u: {
            Uid("u1"): u1, Uid("u2"): u2, Uid("u3"): u3,
        }.get(Uid(str(u))) if u else None

        result = self.service.get_node_neighbors(
            "u1", include_structural=False, similarity_threshold=0.7
        )
        self.assertNotIn("structural", result)
        self.assertIn("semantic", result)
        self.assertEqual(len(result["semantic"]), 1)


class TestSearchGraphRelations(unittest.TestCase):
    """Tests for search_graph_relations."""

    def setUp(self):
        self.semantic_map = MagicMock()
        self.semantic_map.list_units.return_value = []
        self.graph_store = MagicMock()
        self.graph_store.get_neighbors.return_value = []
        self.graph_store.get_relationship.return_value = None
        self.service = SemanticGraphService(
            semantic_map=self.semantic_map, graph_store=self.graph_store
        )

    def test_returns_empty_for_no_seeds(self):
        self.semantic_map.list_units.return_value = []
        result = self.service.search_graph_relations()
        self.assertEqual(result, [])

    def test_returns_edges_from_seeds(self):
        self.graph_store.get_neighbors.return_value = [Uid("n2")]
        self.graph_store.get_relationship.return_value = {"weight": 1.0}
        result = self.service.search_graph_relations(
            seed_nodes=["n1"], max_depth=1, limit=10
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], "n1")
        self.assertEqual(result[0][1], "n2")
        self.assertIn("rel_type", result[0][2])

    def test_respects_limit(self):
        self.graph_store.get_neighbors.side_effect = [
            [Uid("n2"), Uid("n3"), Uid("n4")],
            [Uid("n5")],
        ]
        result = self.service.search_graph_relations(
            seed_nodes=["n1"], max_depth=2, limit=2
        )
        self.assertLessEqual(len(result), 2)

    def test_uses_all_units_when_no_seeds(self):
        u1 = _make_unit("u1")
        u2 = _make_unit("u2")
        self.semantic_map.list_units.return_value = [u1, u2]
        self.graph_store.get_neighbors.return_value = [Uid("n2")]
        result = self.service.search_graph_relations(max_depth=1, limit=5)
        self.assertGreater(len(result), 0)


class TestTraceEvidence(unittest.TestCase):
    """Tests for trace_evidence."""

    def setUp(self):
        self.semantic_map = MagicMock()
        self.graph_store = MagicMock()
        self.graph_store.get_neighbors.return_value = []
        self.service = SemanticGraphService(
            semantic_map=self.semantic_map, graph_store=self.graph_store
        )

    def test_returns_empty_when_source_missing(self):
        self.semantic_map.get_unit.return_value = None
        result = self.service.trace_evidence("missing")
        self.assertIsNone(result.source)
        self.assertEqual(result.evidence, [])

    def test_traces_evidence_chain(self):
        source = _make_unit("src")
        ev1 = _make_unit("ev1")
        ev2 = _make_unit("ev2")

        self.semantic_map.get_unit.side_effect = lambda u: {
            Uid("src"): source, Uid("ev1"): ev1, Uid("ev2"): ev2,
        }.get(Uid(str(u)))

        def get_neighbors(uid, rel_type=None, direction="out"):
            if uid == Uid("src") and rel_type == "EVIDENCED_BY" and direction == "out":
                return [Uid("ev1")]
            if uid == Uid("ev1") and rel_type == "EVIDENCED_BY" and direction == "out":
                return [Uid("ev2")]
            return []

        self.graph_store.get_neighbors.side_effect = get_neighbors

        result = self.service.trace_evidence("src", max_depth=3, top_k=10)
        self.assertEqual(len(result.evidence), 2)
        self.assertEqual(result.depth_map.get("ev1"), 1)
        self.assertEqual(result.depth_map.get("ev2"), 2)


class TestTraceCoref(unittest.TestCase):
    """Tests for trace_coref."""

    def setUp(self):
        self.semantic_map = MagicMock()
        self.graph_store = MagicMock()
        self.graph_store.get_neighbors.return_value = []
        self.service = SemanticGraphService(
            semantic_map=self.semantic_map, graph_store=self.graph_store
        )

    def test_returns_empty_when_source_missing(self):
        self.semantic_map.get_unit.return_value = None
        result = self.service.trace_coref("missing")
        self.assertIsNone(result.source)
        self.assertEqual(result.canonical_entities, [])

    def test_resolves_coref_to_canonical_entity(self):
        source = _make_unit("src")
        canon_entity = _make_unit("canon", spaces=["knowledge_entity"])
        canon_event = _make_unit("cevent", spaces=["episodic_event"])

        self.semantic_map.get_unit.side_effect = lambda u: {
            Uid("src"): source,
            Uid("canon"): canon_entity,
            Uid("cevent"): canon_event,
        }.get(Uid(str(u)))

        # COREF edges from source
        def get_neighbors(uid, rel_type=None, direction="out"):
            if uid == Uid("src") and rel_type == "COREF":
                return [Uid("canon"), Uid("cevent")]
            # EVIDENCED_BY (in) for coref chain
            if uid == Uid("canon") and rel_type == "EVIDENCED_BY" and direction == "in":
                return [Uid("src")]
            if uid == Uid("cevent") and rel_type == "EVIDENCED_BY" and direction == "in":
                return [Uid("src")]
            return []

        self.graph_store.get_neighbors.side_effect = get_neighbors

        result = self.service.trace_coref("src")
        self.assertEqual(len(result.canonical_entities), 1)
        self.assertEqual(len(result.canonical_events), 1)
        self.assertGreater(len(result.coref_chain), 0)


class TestRetrieveEntitySubgraph(unittest.TestCase):
    """Tests for retrieve_entity_subgraph."""

    def setUp(self):
        self.semantic_map = MagicMock()
        self.graph_store = MagicMock()
        self.graph_store.get_neighbors.return_value = []
        self.graph_store.get_relationship.return_value = None
        self.service = SemanticGraphService(
            semantic_map=self.semantic_map, graph_store=self.graph_store
        )

    def test_returns_empty_when_no_entity_found(self):
        self.semantic_map.search_by_text.return_value = []
        result = self.service.retrieve_entity_subgraph("missing entity")
        self.assertIsNone(result.center_entity)
        self.assertEqual(result.neighbors, [])

    def test_expands_from_center_entity(self):
        center = _make_unit("center")
        neighbor = _make_unit("neighbor")

        self.semantic_map.search_by_text.return_value = [(center, 0.95)]
        self.semantic_map.get_unit.side_effect = lambda u: {
            Uid("center"): center, Uid("neighbor"): neighbor,
        }.get(Uid(str(u)))

        self.graph_store.get_neighbors.return_value = [Uid("neighbor")]
        self.graph_store.get_relationship.return_value = {"weight": 0.9}

        result = self.service.retrieve_entity_subgraph("center entity", max_depth=2)
        self.assertIsNotNone(result.center_entity)
        self.assertEqual(len(result.neighbors), 1)
        self.assertEqual(result.depth_map["neighbor"], 1)
        self.assertGreater(len(result.relationships), 0)


class TestRetrieveSummaryEvidenceChain(unittest.TestCase):
    """Tests for retrieve_summary_evidence_chain."""

    def setUp(self):
        self.semantic_map = MagicMock()
        self.graph_store = MagicMock()
        self.graph_store.get_neighbors.return_value = []
        self.service = SemanticGraphService(
            semantic_map=self.semantic_map, graph_store=self.graph_store
        )

    def test_returns_empty_when_summary_missing(self):
        self.semantic_map.get_unit.return_value = None
        result = self.service.retrieve_summary_evidence_chain("missing")
        self.assertIsNone(result.summary)

    def test_traces_summary_to_evidence(self):
        summary = _make_unit("sum")
        ev1 = _make_unit("ev1")

        self.semantic_map.get_unit.side_effect = lambda u: {
            Uid("sum"): summary, Uid("ev1"): ev1,
        }.get(Uid(str(u)))

        def get_neighbors(uid, rel_type=None, direction="out"):
            if uid == Uid("sum") and rel_type == "EVIDENCED_BY":
                return [Uid("ev1")]
            if uid == Uid("ev1") and rel_type == "COREF":
                return []
            return []

        self.graph_store.get_neighbors.side_effect = get_neighbors

        result = self.service.retrieve_summary_evidence_chain(
            "sum", include_entities=False, include_events=False
        )
        self.assertEqual(len(result.evidence_units), 1)


class TestRetrieveEntityInvolvement(unittest.TestCase):
    """Tests for retrieve_entity_involvement."""

    def setUp(self):
        self.semantic_map = MagicMock()
        self.graph_store = MagicMock()
        self.graph_store.get_neighbors.return_value = []
        self.graph_store.get_relationship.return_value = None
        self.service = SemanticGraphService(
            semantic_map=self.semantic_map, graph_store=self.graph_store
        )

    def test_returns_empty_when_entity_not_found(self):
        self.semantic_map.search_by_text.return_value = []
        result = self.service.retrieve_entity_involvement("missing")
        self.assertIsNone(result.entity)
        self.assertEqual(result.events, [])

    def test_finds_events_involving_entity(self):
        entity = _make_unit("entity")
        event1 = _make_unit("event1")

        self.semantic_map.search_by_text.return_value = [(entity, 0.95)]
        self.semantic_map.get_unit.side_effect = lambda u: {
            Uid("entity"): entity, Uid("event1"): event1,
        }.get(Uid(str(u)))

        def get_neighbors(uid, rel_type=None, direction="out"):
            if uid == Uid("entity") and rel_type == "INVOLVES" and direction == "in":
                return [Uid("event1")]
            if uid == Uid("event1"):
                return []
            return []

        self.graph_store.get_neighbors.side_effect = get_neighbors
        self.graph_store.get_relationship.return_value = {"role": "participant"}

        result = self.service.retrieve_entity_involvement("entity name")
        self.assertEqual(len(result.events), 1)
        self.assertEqual(result.events[0][1], "participant")

    def test_filters_by_role(self):
        entity = _make_unit("entity")
        event1 = _make_unit("event1")
        event2 = _make_unit("event2")

        self.semantic_map.search_by_text.return_value = [(entity, 0.95)]
        self.semantic_map.get_unit.side_effect = lambda u: {
            Uid("entity"): entity,
            Uid("event1"): event1,
            Uid("event2"): event2,
        }.get(Uid(str(u)))

        call_count = [0]

        def get_neighbors(uid, rel_type=None, direction="out"):
            if uid == Uid("entity") and rel_type == "INVOLVES" and direction == "in":
                return [Uid("event1"), Uid("event2")]
            if uid == Uid("event1") or uid == Uid("event2"):
                return []
            return []

        def get_relationship(source, target, rel_type):
            call_count[0] += 1
            if source == Uid("event1"):
                return {"role": "participant"}
            if source == Uid("event2"):
                return {"role": "organizer"}
            return None

        self.graph_store.get_neighbors.side_effect = get_neighbors
        self.graph_store.get_relationship.side_effect = get_relationship

        result = self.service.retrieve_entity_involvement("entity", role="organizer")
        self.assertEqual(len(result.events), 1)
        self.assertEqual(result.events[0][1], "organizer")


if __name__ == "__main__":
    unittest.main()
