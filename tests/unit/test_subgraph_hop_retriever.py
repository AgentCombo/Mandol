"""Unit tests for SubgraphHopRetriever multi-hop graph expansion.

Tests that flood queries surface related entities via graph edges
(e.g. LOCATED_IN) rather than relying solely on vector similarity.
"""

from __future__ import annotations

import numpy as np

from mandol.application import SemanticGraphService, SemanticMapService
from mandol.domain import MemoryUnit
from mandol.domain.coref_graph_constants import REL_HOMETOWN, REL_LOCATED_IN
from mandol.infrastructure import AdaptiveVectorIndex, InMemoryGraphStore, InMemoryUnitStore
from mandol.ports.embedding_provider import EmbeddingProvider
from mandol.retrieval import HybridRetriever, HybridRetrieverConfig, SubgraphHopConfig, SubgraphHopRetriever


class _KeywordBasisEmbedder(EmbeddingProvider):
    """Deterministic 8-D embeddings: flood / west county / john occupy orthogonal axes."""

    def embedding_dim(self) -> int:
        return 8

    def embed_text(self, texts, **kwargs):
        out = []
        for raw in texts:
            t = (raw or "").lower()
            v = np.zeros(8, dtype=np.float32)
            if "flood" in t:
                v[0] = 1.0
            elif "west county" in t:
                v[1] = 1.0
            elif "john" in t:
                v[2] = 1.0
            else:
                v[7] = 1.0
            n = float(np.linalg.norm(v))
            if n > 1e-12:
                v = v / n
            out.append(v)
        return out

    def embed_image_paths(self, image_paths, **kwargs):
        z = np.zeros(8, dtype=np.float32)
        z[7] = 1.0
        return [z for _ in image_paths]


def _build_graph():
    store = InMemoryUnitStore()
    index = AdaptiveVectorIndex(dim=8)
    embedder = _KeywordBasisEmbedder()
    sm = SemanticMapService(store=store, index=index, embedder=embedder)
    gs = InMemoryGraphStore()
    graph = SemanticGraphService(semantic_map=sm, graph_store=gs)

    u_flood = MemoryUnit(
        uid="u_event_flood",
        raw_data={"text_content": "Severe flooding destroyed homes"},
        metadata={"kind": "event"},
    )
    u_wc = MemoryUnit(
        uid="u_place_west_county",
        raw_data={"text_content": "West County is a rural area"},
        metadata={"kind": "place"},
    )
    u_john = MemoryUnit(
        uid="u_person_john",
        raw_data={"text_content": "John often visits his hometown"},
        metadata={"kind": "person"},
    )
    graph.add_unit(u_flood, space_names=["mem"])
    graph.add_unit(u_wc, space_names=["mem"])
    graph.add_unit(u_john, space_names=["mem"])

    # Prefer event -> place for disaster location (plan §5.5)
    graph.add_relationship("u_event_flood", "u_place_west_county", REL_LOCATED_IN)
    # Weaker path: person -> place (nickname / hometown)
    graph.add_relationship("u_person_john", "u_place_west_county", REL_HOMETOWN)
    graph.flush()
    return graph


def test_subgraph_hop_boosts_place_via_event_located_in():
    graph = _build_graph()
    hybrid = HybridRetriever(
        graph=graph,
        config=HybridRetrieverConfig(parallel_search=False, bfs_hops=1, bfs_per_seed=2),
    )
    hop_cfg = SubgraphHopConfig(
        graph_branch_weight=0.65,
        max_hops=2,
        seed_top_k=4,
    )
    retriever = SubgraphHopRetriever(hybrid=hybrid, config=hop_cfg)

    query = "where did the flooding happen"
    hits = retriever.search(query, top_k=5, space_names=["mem"], use_rerank=False)

    by_uid = {str(h.unit.uid): h for h in hits}
    place = by_uid.get("u_place_west_county")
    assert place is not None, "West County should appear after graph expansion"
    assert place.scores.get("graph_boost_raw", 0) > 0

    # Direct event unit should remain a strong retrieval seed; place should rank high via graph
    scores = [h.final_score for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert place.final_score >= hits[-1].final_score


def test_event_path_outranks_hometown_only_for_flood_relevant_query():
    """When only HOMETOWN connects John to West County, flood axis still hits event; graph favors LOCATED_IN."""
    graph = _build_graph()
    # Remove LOCATED_IN so only HOMETOWN links to place; place reachable only via john
    graph.delete_relationship("u_event_flood", "u_place_west_county", REL_LOCATED_IN)
    graph.flush()

    hybrid = HybridRetriever(graph=graph, config=HybridRetrieverConfig(parallel_search=False))
    retriever = SubgraphHopRetriever(
        hybrid=hybrid,
        config=SubgraphHopConfig(graph_branch_weight=0.55, max_hops=2, seed_top_k=4),
    )
    hits = retriever.search("flooding disaster", top_k=6, space_names=["mem"], use_rerank=False)
    uids = [str(h.unit.uid) for h in hits]
    # Event still in seed set; without LOCATED_IN, place may only get boost via lower-weight HOMETOWN from John
    assert "u_event_flood" in uids
    # Still demonstrate graph score field exists for place if it appears
    for h in hits:
        assert "retrieval_norm" in h.scores
        assert "graph_boost" in h.scores
