#!/usr/bin/env python3
"""Quick start example for Mandol memory system.

Demonstrates the minimal workflow: create a MemorySystem, add a unit,
build high-level memories, retrieve, and persist to disk.
"""
from mandol import MemorySystem, MemoryUnit, Uid

system = MemorySystem()

unit = MemoryUnit(
    uid=Uid("dialogue_001"),
    raw_data={"text_content": "Zhang San went to Beijing on a business trip today"},
    metadata={"timestamp": "2024-01-15T10:00:00"},
)
system.add(unit)

system.build_high_level(mode="auto")

hits = system.holistic_retrieve("Where did Zhang San go?", top_k=5)

for hit in hits:
    print(f"[{hit.final_score:.3f}] {hit.unit.raw_data['text_content']}")

system.save("./memory_snapshot")
system2 = MemorySystem.load("./memory_snapshot")
