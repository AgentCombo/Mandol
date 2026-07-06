"""Mandol Agent Memory System.

The top-level package exports the primary core classes:

- :class:`MemorySpace`
- :class:`SemanticMap`
- :class:`SemanticGraph`
- :class:`MemoryUnit`

Additional subsystems are imported from their own modules, for example::

    from mandol.llm import LLMClient
    from mandol.retrieval import MultiRetriever
    from mandol.cluster import cluster_nodes
    from mandol.storage import DuckDBOperator

``SemanticMap`` and ``SemanticGraph`` are exposed lazily so importing
``mandol`` does not immediately import PyTorch, FAISS, or SentenceTransformers.
Accessing either semantic class still requires the complete runtime stack.
"""

from __future__ import annotations

from importlib import import_module
import sys

__version__ = "0.1.0a1"
__author__ = "AgentMemorySystem Team"
__license__ = "Apache-2.0"

if sys.version_info < (3, 12):
    raise RuntimeError("mandol requires Python 3.12 or higher")

from .core.memory_unit import MemoryUnit
from .core.memory_space import MemorySpace
from .core.memory_space_registry import MemorySpaceRegistry, TowerSpace

_LAZY_EXPORTS = {
    "SemanticMap": (".core.semantic_map", "SemanticMap"),
    "SemanticGraph": (".core.semantic_graph", "SemanticGraph"),
    "cluster_nodes": (".cluster", "cluster_nodes"),
    "ClusterMethod": (".cluster", "ClusterMethod"),
}


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        module = import_module(module_name, __name__)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "MemoryUnit",
    "MemorySpace",
    "MemorySpaceRegistry",
    "TowerSpace",
    "SemanticMap",
    "SemanticGraph",
    "cluster_nodes",
    "ClusterMethod",
    "__version__",
]
