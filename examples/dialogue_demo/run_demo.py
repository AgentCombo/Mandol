#!/usr/bin/env python3
"""
Dialogue Demo - A slightly more complete example using a small dialogue dataset.

This demo shows how to:
1. Load dialogue data from a JSON file
2. Add dialogues to the memory system
3. Build high-level memories (entities, events, summaries)
4. Perform holistic retrieval

Usage:
    python run_demo.py
"""
from __future__ import annotations

import json
from pathlib import Path

from mandol import MemorySystem, MemoryUnit, Uid


def main():
    """Run the dialogue demo.

    Loads dialogue data from a JSON file, adds units to the memory system,
    builds high-level memories, and runs sample queries.
    """
    data_path = Path(__file__).parent / "demo_data.json"
    with open(data_path, "r", encoding="utf-8") as f:
        dialogues = json.load(f)

    system = MemorySystem()

    for d in dialogues:
        unit = MemoryUnit(
            uid=Uid(f"demo_{d['dia_id'].replace(':', '_')}"),
            raw_data={
                "text_content": f"[{d['dia_id']}] {d['speaker']}: {d['text']}",
                "speaker": d["speaker"],
            },
            metadata={
                "timestamp": d["session_datetime"],
                "session_number": d["session_number"],
            },
        )
        system.add(unit)

    print(f"Added {len(dialogues)} dialogue units.")

    system.build_high_level(mode="force")
    print("High-level memory construction complete.")

    queries = [
        "Where does Alice work?",
        "What did Alice do in Shanghai?",
        "Who helped Alice with the graph algorithms?",
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        hits = system.holistic_retrieve(query, top_k=3)
        for hit in hits:
            print(f"  [{hit.final_score:.3f}] {hit.unit.raw_data.get('text_content', '')[:80]}")


if __name__ == "__main__":
    main()
