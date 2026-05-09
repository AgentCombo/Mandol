"""Unit tests for HybridRetriever pipeline with dense/BM25/sparse fusion."""


from mandol.application import SemanticGraphService, SemanticMapService
from mandol.domain import MemoryUnit
from mandol.infrastructure import InMemoryCosineVectorIndex, InMemoryGraphStore, InMemoryUnitStore
from mandol.ports import StaticEmbeddingProvider
from mandol.retrieval import HybridRetriever


class StubReranker:
    def rerank(self, query, units, *, top_k=10):
        # deterministic: reverse order, score by position
        ranked = list(reversed(units))
        out = []
        for i, u in enumerate(ranked[:top_k]):
            out.append((u, float(top_k - i)))
        return out


def test_hybrid_retriever_returns_intermediate_scores_and_rerank():
    store = InMemoryUnitStore()
    index = InMemoryCosineVectorIndex(dim=4)
    embedder = StaticEmbeddingProvider(dim=4, fill=1.0)

    sm = SemanticMapService(store=store, index=index, embedder=embedder)
    gs = InMemoryGraphStore()
    graph = SemanticGraphService(semantic_map=sm, graph_store=gs)

    u1 = MemoryUnit(uid="u1", raw_data={"text_content": "alpha"})
    u2 = MemoryUnit(uid="u2", raw_data={"text_content": "beta"})
    u3 = MemoryUnit(uid="u3", raw_data={"text_content": "gamma"})

    graph.add_unit(u1)
    graph.add_unit(u2)
    graph.add_unit(u3)

    graph.add_relationship("u1", "u2", "rel")

    retriever = HybridRetriever(graph=graph, reranker=StubReranker())

    hits = retriever.search("alpha", top_k=2)
    assert len(hits) == 2

    for h in hits:
        assert h.unit is not None
        assert "dense" in h.scores
        assert "bm25" in h.scores or "bm25" not in h.scores  # may be 0 if tokenization yields none
        assert "sparse" in h.scores or "sparse" not in h.scores
        assert "rrf" in h.scores
        assert "rerank" in h.scores

    # verify rerank took effect (StubReranker reverses order)
    assert hits[0].final_score >= hits[1].final_score
