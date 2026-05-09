"""Shared pytest fixtures for the Mandol test suite.

Provides stub embedder, reranker, and MemorySystem instances
configured for deterministic testing without external dependencies.
"""

from __future__ import annotations

import pytest

from mandol.application.memory_system import MemorySystem, MemorySystemConfig
from mandol.ports import StaticEmbeddingProvider


@pytest.fixture
def stub_embedder():
    """Provide a StaticEmbeddingProvider with dim=4 for deterministic tests."""
    return StaticEmbeddingProvider(dim=4, fill=1.0)


@pytest.fixture
def stub_reranker():
    """Provide a StubReranker that records calls and returns linear-decay scores."""
    from mandol.domain.memory_unit import MemoryUnit

    class StubReranker:
        def __init__(self):
            self.called_with: list[tuple[str, list[MemoryUnit], int]] = []

        def rerank(
            self,
            query: str,
            units: list[MemoryUnit],
            *,
            top_k: int = 10,
        ) -> list[tuple[MemoryUnit, float]]:
            self.called_with.append((query, list(units), top_k))
            scored = [(u, float(top_k - i)) for i, u in enumerate(units)]
            return scored[:top_k]

    return StubReranker()


@pytest.fixture
def ms_config():
    """Provide a MemorySystemConfig with embedder_dim=4 for tests."""
    return MemorySystemConfig(embedder_dim=4)


@pytest.fixture
def ms(ms_config, stub_embedder, stub_reranker):
    """Provide a MemorySystem with stub embedder and reranker."""
    return MemorySystem(
        config=ms_config,
        embedder=stub_embedder,
        reranker=stub_reranker,
    )


@pytest.fixture
def ms_no_reranker(ms_config, stub_embedder):
    """Provide a MemorySystem with stub embedder but no reranker."""
    return MemorySystem(
        config=ms_config,
        embedder=stub_embedder,
        reranker=None,
    )
