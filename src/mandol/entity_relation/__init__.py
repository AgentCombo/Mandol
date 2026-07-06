"""
Entity-relation retrieval components.

Automatic entity/relation construction now lives in ``mandol.auto_builder``.
This package root exposes only retrieval-facing graph components for existing
entity-relation memory spaces.
"""

from .entity_relation_retriever import (
    EntityRelationRetriever,
    EntityRelationResult,
    EntityRelationGraphRetriever,
    GraphRetrievalConfig,
    create_graph_retriever,
)

__all__ = [
    "EntityRelationRetriever",
    "EntityRelationResult",
    "EntityRelationGraphRetriever",
    "GraphRetrievalConfig",
    "create_graph_retriever",
    "get_relation_component_status",
]


def get_relation_component_status():
    """Return availability for the retrieval-facing entity-relation package."""
    return {
        "graph_retriever": True,
        "generation_location": "mandol.auto_builder",
        "available_exports": [
            "EntityRelationGraphRetriever",
            "EntityRelationGraphRetriever.search_async",
            "EntityRelationRetriever",
            "GraphRetrievalConfig",
            "create_graph_retriever",
        ],
    }
