#!/usr/bin/env python3
"""
Customer Support Memory System Demo

This demo shows how to build an e-commerce customer support memory system
that remembers user orders, preferences, and complaint records.

Usage:
    python run_customer_support.py
"""
from __future__ import annotations

from mandol import MemorySystem, MemoryUnit, Uid


def main():
    """Run the customer support memory system demo.

    Adds customer support dialogue units, builds high-level memories,
    and retrieves answers to common support queries.
    """
    system = MemorySystem()

    conversations = [
        MemoryUnit(
            uid=Uid("cs_001"),
            raw_data={"text_content": "我想退昨天买的蓝色运动鞋，尺码不合适"},
            metadata={"timestamp": "2024-03-10T14:00:00", "speaker": "customer"},
        ),
        MemoryUnit(
            uid=Uid("cs_002"),
            raw_data={"text_content": "好的，已为您提交退货申请，预计3-5个工作日退款"},
            metadata={"timestamp": "2024-03-10T14:01:00", "speaker": "agent"},
        ),
        MemoryUnit(
            uid=Uid("cs_003"),
            raw_data={"text_content": "下次我想买42码的同一款，有货吗？"},
            metadata={"timestamp": "2024-03-10T14:02:00", "speaker": "customer"},
        ),
        MemoryUnit(
            uid=Uid("cs_004"),
            raw_data={"text_content": "这款42码有货，需要帮您预留吗？"},
            metadata={"timestamp": "2024-03-10T14:03:00", "speaker": "agent"},
        ),
        MemoryUnit(
            uid=Uid("cs_005"),
            raw_data={"text_content": "好的，帮我预留一双，另外我还想看看同款的黑色"},
            metadata={"timestamp": "2024-03-10T14:04:00", "speaker": "customer"},
        ),
    ]
    for unit in conversations:
        system.add(unit)

    print(f"Added {len(conversations)} customer support dialogue units.")

    report = system.build_high_level(mode="auto")
    print("High-level memory construction complete.")

    queries = [
        "这个客户之前买了什么鞋？",
        "客户喜欢什么？",
        "退货的原因是什么？",
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        hits = system.holistic_retrieve(query, top_k=3)
        for hit in hits:
            print(f"  [{hit.final_score:.3f}] {hit.unit.raw_data.get('text_content', '')[:80]}")

    system.save("./cs_memory")
    print("\nMemory saved to ./cs_memory")


if __name__ == "__main__":
    main()
