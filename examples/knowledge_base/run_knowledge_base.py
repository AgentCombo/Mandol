#!/usr/bin/env python3
"""
Knowledge Base Q&A System Demo

This demo shows how to build an enterprise knowledge base Q&A system
that supports semantic search even when user wording differs from documents.

Usage:
    python run_knowledge_base.py
"""
from __future__ import annotations

from mandol import MemorySystem, MemoryUnit, Uid


def main():
    """Run the knowledge base Q&A demo.

    Adds enterprise policy documents as memory units, builds high-level
    memories, and retrieves answers using semantic search.
    """
    system = MemorySystem()

    knowledge_units = [
        MemoryUnit(
            uid=Uid("kb_001"),
            raw_data={"text_content": "公司年假政策：入职满1年可享10天年假，满5年15天，满10年20天"},
            metadata={"timestamp": "2024-01-01T00:00:00", "source": "hr_policy"},
        ),
        MemoryUnit(
            uid=Uid("kb_002"),
            raw_data={"text_content": "报销流程：填写报销单→部门经理审批→财务审核→打款，周期约5个工作日"},
            metadata={"timestamp": "2024-01-01T00:00:00", "source": "finance_policy"},
        ),
        MemoryUnit(
            uid=Uid("kb_003"),
            raw_data={"text_content": "远程办公规定：每周最多2天远程，需提前一天在OA系统申请"},
            metadata={"timestamp": "2024-01-01T00:00:00", "source": "hr_policy"},
        ),
        MemoryUnit(
            uid=Uid("kb_004"),
            raw_data={"text_content": "加班补偿：工作日加班按1.5倍工资计算，周末加班按2倍，法定节假日按3倍"},
            metadata={"timestamp": "2024-01-01T00:00:00", "source": "hr_policy"},
        ),
        MemoryUnit(
            uid=Uid("kb_005"),
            raw_data={"text_content": "新员工入职培训为期3天，包括公司文化、制度学习和部门介绍"},
            metadata={"timestamp": "2024-01-01T00:00:00", "source": "hr_policy"},
        ),
    ]
    system.add_many(knowledge_units)

    print(f"Added {len(knowledge_units)} knowledge base units.")

    report = system.build_high_level(mode="auto")
    print("High-level memory construction complete.")

    queries = [
        "我可以在家办公吗？",
        "休假有多少天？",
        "加班费怎么算？",
        "新来的同事需要做什么？",
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        hits = system.holistic_retrieve(query, top_k=3)
        for hit in hits:
            print(f"  [{hit.final_score:.3f}] {hit.unit.raw_data.get('text_content', '')[:80]}")

    system.save("./kb_memory")
    print("\nMemory saved to ./kb_memory")


if __name__ == "__main__":
    main()
