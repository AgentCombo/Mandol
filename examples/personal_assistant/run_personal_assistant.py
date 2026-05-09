#!/usr/bin/env python3
"""
Personal Assistant Long-term Memory Demo

This demo shows how to build a personal assistant with long-term memory
that remembers user habits, schedules, and relationships across sessions.

Usage:
    python run_personal_assistant.py
"""
from __future__ import annotations

from mandol import MemorySystem, MemoryUnit, Uid


def main():
    """Run the personal assistant long-term memory demo.

    Adds multi-session dialogue units (work and life topics), builds
    high-level memories, and retrieves relevant context across sessions.
    """
    system = MemorySystem()

    session1 = [
        MemoryUnit(
            uid=Uid("pa_001"),
            raw_data={"text_content": "我下周二要和客户做项目汇报"},
            metadata={"timestamp": "2024-03-11T09:00:00", "speaker": "user", "session_id": "s1"},
        ),
        MemoryUnit(
            uid=Uid("pa_002"),
            raw_data={"text_content": "汇报内容是Q1的销售数据分析"},
            metadata={"timestamp": "2024-03-11T09:01:00", "speaker": "user", "session_id": "s1"},
        ),
        MemoryUnit(
            uid=Uid("pa_003"),
            raw_data={"text_content": "记得提前准备好PPT和数据图表"},
            metadata={"timestamp": "2024-03-11T09:02:00", "speaker": "assistant", "session_id": "s1"},
        ),
    ]

    session2 = [
        MemoryUnit(
            uid=Uid("pa_004"),
            raw_data={"text_content": "周末想去爬山，推荐一下附近的路线"},
            metadata={"timestamp": "2024-03-16T10:00:00", "speaker": "user", "session_id": "s2"},
        ),
        MemoryUnit(
            uid=Uid("pa_005"),
            raw_data={"text_content": "香山和百望山都不错，香山风景更好但人比较多"},
            metadata={"timestamp": "2024-03-16T10:01:00", "speaker": "assistant", "session_id": "s2"},
        ),
        MemoryUnit(
            uid=Uid("pa_006"),
            raw_data={"text_content": "那就去百望山吧，人少一点安静"},
            metadata={"timestamp": "2024-03-16T10:02:00", "speaker": "user", "session_id": "s2"},
        ),
    ]

    for unit in session1 + session2:
        system.add(unit)

    print(f"Added {len(session1) + len(session2)} personal assistant dialogue units.")
    print(f"  Session 1 (work): {len(session1)} units")
    print(f"  Session 2 (life): {len(session2)} units")

    report = system.build_high_level(mode="auto")
    print("High-level memory construction complete.")

    queries = [
        "我最近有什么安排？",
        "客户",
        "爬山",
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        hits = system.holistic_retrieve(query, top_k=5)
        for hit in hits:
            print(f"  [{hit.final_score:.3f}] {hit.unit.raw_data.get('text_content', '')[:80]}")

    system.save("./pa_memory")
    print("\nMemory saved to ./pa_memory")


if __name__ == "__main__":
    main()
