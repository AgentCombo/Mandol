"""Mandol — a multi-dimensional semantic memory system.

Provides persistent, retrievable memory built on top of vector indexes,
graph stores, and LLM-powered extraction.  The public API surface
includes:

- :class:`MemorySystem` / :class:`MemorySystemConfig` — high-level
  add / build / retrieve / save / load interface.
- :class:`MemoryUnit` — the atomic memory record.
- :class:`Uid` / :class:`SpaceName` — domain value types.
"""
from mandol.application.memory_system import MemorySystem, MemorySystemConfig
from mandol.domain.memory_unit import MemoryUnit
from mandol.domain.types import SpaceName, Uid
from mandol.query import VectorSeededTraversalSpec

__all__ = [
    "MemorySystem",
    "MemorySystemConfig",
    "MemoryUnit",
    "SpaceName",
    "Uid",
    "VectorSeededTraversalSpec",
]
__version__ = "0.1.0"
