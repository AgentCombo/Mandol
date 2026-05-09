"""Verify that all core mandol modules can be imported without errors."""


def test_src_imports():
    """Test that key public classes are importable from their packages."""
    from mandol.application import SemanticGraphService, SemanticMapService
    from mandol.domain import MemorySpace, MemoryUnit
    from mandol.infrastructure import InMemoryGraphStore, InMemoryUnitStore
    from mandol.ports import StaticEmbeddingProvider

    assert SemanticMapService
    assert SemanticGraphService
    assert MemoryUnit
    assert MemorySpace
    assert InMemoryUnitStore
    assert InMemoryGraphStore
    assert StaticEmbeddingProvider
