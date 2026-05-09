"""LoCoMo benchmark adapter for the Mandol memory system.

Re-exports :class:`LocomoMemorySystem` and :class:`LocomoMemoryConfig`
for convenient access from the benchmark scripts.
"""
from .locomo_adapter import LocomoMemorySystem
from .config import LocomoMemoryConfig

__all__ = [
    "LocomoMemorySystem",
    "LocomoMemoryConfig",
]
