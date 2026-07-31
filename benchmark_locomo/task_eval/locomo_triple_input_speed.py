#!/usr/bin/env python3
"""Fixed-QPS insertion benchmark for LoCoMo tri-tower memory units."""

import argparse
import gc
import json
import os
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np




from mandol.core.memory_unit import MemoryUnit  # noqa: E402
from mandol.core import paths






def _load_hierarchical_l0(data_dir: Path, sample_ids: Optional[List[str]] = None) -> List[MemoryUnit]:
    """Load hierarchical L0."""
    units: List[MemoryUnit] = []
    l0_dir = data_dir / "hierarchical_content" / "step1_l0_graphs"
    if not l0_dir.exists():
        print(f"  [WARN] L0 directory does not exist: {l0_dir}")
        return units

    for l0_file in sorted(l0_dir.glob("conv-*_l0_graph.json")):
        sample_id = l0_file.stem.replace("_l0_graph", "")
        if sample_ids and sample_id not in sample_ids:
            continue

        with open(l0_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for conv in data.get("l0_conversations", []):
            uid = conv.get("uid")
            if not uid:
                continue

            indexing_text = conv.get("indexing_text", "")
            if not indexing_text:
                raw = conv.get("raw_data", {})
                indexing_text = raw.get("message", raw.get("text_content", ""))
            if not indexing_text:
                continue

            raw_data = conv.get("raw_data", {}).copy()
            raw_data["text_content"] = indexing_text
            raw_data["original_indexing_text"] = conv.get("indexing_text", "")
            raw_data["type"] = "conversation_message"

            metadata = conv.get("metadata", {}).copy()
            metadata["memory_level"] = "L0"
            metadata["sample_id"] = sample_id

            units.append(MemoryUnit(uid=uid, raw_data=raw_data, metadata=metadata))

    return units



def _build_l1_text_content(session: Dict, participants: List[str]) -> str:
    """Build L1 text content."""
    parts = []
    topic = session.get("session_topic", "")
    if topic:
        parts.append(f"Session Topic: {topic}")

    date = session.get("session_date", "")
    if date and date != "unknown":
        parts.append(f"Date: {date}")

    key_facts = session.get("key_facts", [])
    if key_facts:
        fact_texts = []
        for fact in key_facts[:5]:
            subject = fact.get("subject", "")
            fact_text = fact.get("fact", "")
            if subject and fact_text:
                fact_texts.append(f"- {subject}: {fact_text}")
        if fact_texts:
            parts.append("Key Facts:\n" + "\n".join(fact_texts))

    if len(key_facts) < 2:
        events = session.get("structured_events", [])
        if events:
            event_texts = []
            for event in events[:3]:
                name = event.get("event_name", "")
                event_type = event.get("event_type", "")
                if name:
                    event_texts.append(f"- [{event_type}] {name}")
            if event_texts:
                parts.append("Events:\n" + "\n".join(event_texts))

    return "\n\n".join(parts) if parts else "(No session content)"


def _load_hierarchical_l1(data_dir: Path, sample_ids: Optional[List[str]] = None) -> List[MemoryUnit]:
    """Load hierarchical L1."""
    units: List[MemoryUnit] = []
    l1_dir = data_dir / "hierarchical_content" / "step2_l1_extracted"
    if not l1_dir.exists():
        print(f"  [WARN] L1 目录不存在: {l1_dir}")
        return units

    for l1_file in sorted(l1_dir.glob("conv-*_l1_extracted.json")):
        sample_id = l1_file.stem.replace("_l1_extracted", "")
        if sample_ids and sample_id not in sample_ids:
            continue

        with open(l1_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        participants = data.get("participants", ["Speaker_A", "Speaker_B"])

        for session in data.get("session_extractions", []):
            session_id = session.get("session_id", "unknown")
            uid = f"{sample_id}_{session_id}_l1_facts"

            text_content = _build_l1_text_content(session, participants)

            raw_data = {
                "text_content": text_content,
                "type": "session_facts",
                "session_id": session_id,
                "session_date": session.get("session_date", "unknown"),
                "session_topic": session.get("session_topic", ""),
                "structured_events": session.get("structured_events", []),
                "state_updates": session.get("state_updates", []),
                "countable_items": session.get("countable_items", []),
                "key_facts": session.get("key_facts", []),
                "mentioned_dates": session.get("mentioned_dates", []),
            }
            metadata = {
                "memory_level": "L1",
                "content_type": "session_facts",
                "sample_id": sample_id,
                "session_id": session_id,
                "participants": participants,
            }
            units.append(MemoryUnit(uid=uid, raw_data=raw_data, metadata=metadata))

    return units



def _load_hierarchical_l2(data_dir: Path, sample_ids: Optional[List[str]] = None) -> List[MemoryUnit]:
    """Load hierarchical L2."""
    units: List[MemoryUnit] = []
    l2_dir = data_dir / "hierarchical_content" / "step3_l2_aggregated"
    if not l2_dir.exists():
        print(f"  [WARN] L2 目录不存在: {l2_dir}")
        return units

    for l2_file in sorted(l2_dir.glob("conv-*_l2_aggregated.json")):
        sample_id = l2_file.stem.replace("_l2_aggregated", "")
        if sample_ids and sample_id not in sample_ids:
            continue

        with open(l2_file, "r", encoding="utf-8") as f:
            l2_data = json.load(f)

        # --- Activity Ledger ---
        for i, entry in enumerate(l2_data.get("activity_ledger", [])):
            activity = entry.get("activity", f"Activity_{i}")
            safe_name = re.sub(r"[^a-zA-Z0-9]", "_", activity.lower()).strip("_")
            uid = f"{sample_id}_l2_activity_{safe_name}_{i}"
            count = entry.get("count", 0)
            instances = entry.get("instances", [])
            text_parts = [f"Activity: {activity} | Count: {count} times"]
            if instances:
                text_parts.append(f"Details: {', '.join(str(inst) for inst in instances)}")
            raw_data = {"text_content": " | ".join(text_parts), "type": "activity_ledger",
                        "activity": activity, "count": count, "instances": instances}
            metadata = {"memory_level": "L2", "content_type": "activity_ledger", "sample_id": sample_id}
            units.append(MemoryUnit(uid=uid, raw_data=raw_data, metadata=metadata))

        # --- Entity Profiles ---
        for i, profile in enumerate(l2_data.get("entity_profiles", [])):
            entity = profile.get("entity", f"Entity_{i}")
            attribute = profile.get("attribute", "")
            safe_e = re.sub(r"[^a-zA-Z0-9]", "_", entity.lower()).strip("_")
            safe_a = re.sub(r"[^a-zA-Z0-9]", "_", attribute.lower()).strip("_")
            uid = f"{sample_id}_l2_entity_{safe_e}_{safe_a}_{i}"
            value = profile.get("value", "")
            context = profile.get("context", "")
            tp = [f"Entity: {entity}"]
            if attribute: tp.append(f"Attribute: {attribute}")
            if value:     tp.append(f"Value: {value}")
            if context:   tp.append(f"Context: {context}")
            raw_data = {"text_content": " | ".join(tp), "type": "entity_profile",
                        "entity": entity, "attribute": attribute, "value": value, "context": context}
            metadata = {"memory_level": "L2", "content_type": "entity_profile", "sample_id": sample_id}
            units.append(MemoryUnit(uid=uid, raw_data=raw_data, metadata=metadata))

        # --- Master Timeline ---
        for i, entry in enumerate(l2_data.get("master_timeline", [])):
            date = entry.get("date", "unknown")
            events_field = entry.get("events", [])
            if isinstance(events_field, list) and events_field:
                event_desc = "; ".join(
                    e.get("event", "") if isinstance(e, dict) else str(e) for e in events_field
                )
            else:
                event_desc = entry.get("event", str(events_field))
            is_estimated = entry.get("is_estimated", False)
            safe_date = re.sub(r"[^a-zA-Z0-9]", "", date)
            uid = f"{sample_id}_l2_timeline_{safe_date}_{i}"
            tp = [f"Date: {date}", f"Event: {event_desc}"]
            if is_estimated:
                tp.append("(estimated date)")
            raw_data = {"text_content": " | ".join(tp), "type": "timeline_event",
                        "date": date, "event": event_desc, "is_estimated": is_estimated}
            metadata = {"memory_level": "L2", "content_type": "timeline_event",
                        "sample_id": sample_id, "event_date": date}
            units.append(MemoryUnit(uid=uid, raw_data=raw_data, metadata=metadata))

        # --- Social Graph (relationship_graph.edges) ---
        rg = l2_data.get("relationship_graph", {})
        edges = rg.get("edges", []) if isinstance(rg, dict) else []
        for i, edge in enumerate(edges):
            person = edge.get("from", edge.get("person", f"Person_{i}"))
            safe_p = re.sub(r"[^a-zA-Z0-9]", "_", person.lower()).strip("_")
            uid = f"{sample_id}_l2_social_{safe_p}_{i}"
            relationship = edge.get("relationship", "")
            key_interaction = edge.get("key_interaction", "")
            to_person = edge.get("to", "")
            tp = [f"Person: {person}"]
            if to_person:       tp.append(f"To: {to_person}")
            if relationship:    tp.append(f"Relationship: {relationship}")
            if key_interaction:  tp.append(f"Key Interaction: {key_interaction}")
            raw_data = {"text_content": " | ".join(tp), "type": "social_graph",
                        "person": person, "to": to_person, "relationship": relationship}
            metadata = {"memory_level": "L2", "content_type": "social_graph", "sample_id": sample_id}
            units.append(MemoryUnit(uid=uid, raw_data=raw_data, metadata=metadata))

        # --- Negative Constraints ---
        for i, constraint in enumerate(l2_data.get("negative_constraints", [])):
            uid = f"{sample_id}_l2_negconstraint_{i}"
            text_content = constraint if isinstance(constraint, str) else str(constraint)
            raw_data = {"text_content": text_content, "type": "negative_constraint", "constraint_index": i}
            metadata = {"memory_level": "L2", "content_type": "negative_constraint", "sample_id": sample_id}
            units.append(MemoryUnit(uid=uid, raw_data=raw_data, metadata=metadata))

    return units


# ---- 1d. Entity-Relation: Hub + Mention ----

def _build_mention_context(mention: Dict[str, Any], entity_name: str, entity_type: str) -> str:
    """Build mention context."""
    parts = [f"Entity: {entity_name} (Type: {entity_type})"]
    session_id = mention.get("session_id", "")
    if session_id:
        parts.append(f"Session: {session_id}")
    main_context = mention.get("context", "")
    if main_context:
        parts.append(f"Context: {main_context}")
    temporal_info = mention.get("temporal_info")
    if temporal_info:
        parts.append(f"Time: {temporal_info}")
    spatial_info = mention.get("spatial_info")
    if spatial_info:
        parts.append(f"Location: {spatial_info}")
    aliases = mention.get("aliases", [])
    if aliases:
        parts.append(f"Also known as: {', '.join(aliases)}")
    return " | ".join(parts)


def _load_entity_relation(data_dir: Path, sample_ids: Optional[List[str]] = None) -> List[MemoryUnit]:
    """Load entity relation."""
    units: List[MemoryUnit] = []
    er_dir = data_dir / "entity_relation" / "step2_relations"
    if not er_dir.exists():
        print(f"  [WARN] 实体关系目录不存在: {er_dir}")
        return units

    for sample_dir in sorted(er_dir.iterdir()):
        if not sample_dir.is_dir():
            continue
        sample_id = sample_dir.name
        if sample_ids and sample_id not in sample_ids:
            continue

        er_file = sample_dir / f"{sample_id}_complete_entity_relation.json"
        if not er_file.exists():
            continue

        with open(er_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        conversation_id = data.get("conversation_id", sample_id)

        for entity in data.get("entities", []):
            entity_id = entity.get("entity_id", "")
            entity_name = entity.get("name", "")
            entity_type = entity.get("entity_type", "Unknown")
            confidence = entity.get("confidence", 0.0)
            mentions = entity.get("mentions", [])

            if not entity_id:
                continue

            # Hub node
            hub_uid = f"{conversation_id}_{entity_id}_hub"
            hub_unit = MemoryUnit(
                uid=hub_uid,
                raw_data={
                    "node_type": "entity_hub",
                    "conversation_id": conversation_id,
                    "original_entity_id": entity_id,
                    "name": entity_name,
                    "entity_type": entity_type,
                    "confidence": confidence,
                    "mentions_count": len(mentions),
                    "text_content": entity_name,
                    "created_at": datetime.now().isoformat(),
                },
            )
            units.append(hub_unit)

            # Mention nodes
            for mention_idx, mention in enumerate(mentions):
                mention_uid = f"{conversation_id}_{entity_id}_mention_{mention_idx}"
                mention_context = _build_mention_context(mention, entity_name, entity_type)

                mention_unit = MemoryUnit(
                    uid=mention_uid,
                    raw_data={
                        "node_type": "evidence_mention",
                        "conversation_id": conversation_id,
                        "parent_entity_id": entity_id,
                        "parent_hub_uid": hub_uid,
                        "entity_name": entity_name,
                        "entity_type": entity_type,
                        "session_id": mention.get("session_id", ""),
                        "context": mention.get("context", ""),
                        "temporal_info": mention.get("temporal_info"),
                        "spatial_info": mention.get("spatial_info"),
                        "aliases": mention.get("aliases", []),
                        "confidence": mention.get("confidence", confidence),
                        "text_content": mention_context,
                        "created_at": datetime.now().isoformat(),
                    },
                )
                units.append(mention_unit)

    return units



def _load_episodic_memory(data_dir: Path, sample_ids: Optional[List[str]] = None) -> List[MemoryUnit]:
    """Load episodic memory."""
    units: List[MemoryUnit] = []
    ep_dir = data_dir / "episodic_memory" / "step2_enhanced"
    if not ep_dir.exists():
        print(f"  [WARN] 情景记忆目录不存在: {ep_dir}")
        return units

    for ep_file in sorted(ep_dir.glob("conv-*_enhanced.json")):
        sample_id = ep_file.stem.replace("_enhanced", "")
        if sample_ids and sample_id not in sample_ids:
            continue

        with open(ep_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for fact in data.get("episodic_facts", []):
            content = fact.get("content", "")
            if not content:
                continue

            time_info = fact.get("time", {})
            if isinstance(time_info, dict):
                timestamp = time_info.get("absolute_start", "")
            else:
                timestamp = ""

            fact_id = fact.get("fact_id", "")
            uid = (
                f"episodic_{sample_id}_{fact_id}"
                if fact_id
                else f"episodic_{sample_id}_{hash(content)}"
            )

            raw_data = {
                "content": content,
                "text_content": content,
                "source": f"episodic:{sample_id}",
                "timestamp": timestamp if timestamp else datetime.now().isoformat(),
                "fact_type": fact.get("fact_type", "EVENT"),
                "participants": fact.get("participants", []),
                "location": fact.get("location", ""),
                "retrieval_keys": fact.get("retrieval_keys", []),
            }

            metadata = {
                "fact_id": fact_id,
                "fact_type": fact.get("fact_type", "EVENT"),
                "participants": fact.get("participants", []),
                "location": fact.get("location", ""),
                "source_session": fact.get("source_session_id", ""),
                "source_turns": fact.get("source_turns", []),
                "retrieval_keys": fact.get("retrieval_keys", []),
                "sample_id": sample_id,
                "memory_type": "episodic_fact",
            }
            if isinstance(time_info, dict):
                metadata["time_original"] = time_info.get("original_text", "")
                metadata["time_start"] = time_info.get("absolute_start", "")
                metadata["time_end"] = time_info.get("absolute_end", "")
                metadata["time_is_exact"] = time_info.get("is_exact", False)

            units.append(MemoryUnit(uid=uid, raw_data=raw_data, metadata=metadata))

    return units



def prepare_mixed_memory_pool(
    data_dir: str,
    sample_ids: Optional[List[str]] = None,
    pool_limit: Optional[int] = None,
    seed: int = 42,
) -> List[MemoryUnit]:
    """Run prepare mixed memory pool."""
    data_path = Path(data_dir)
    print("=" * 72)
    print("Phase 1: 准备混合记忆池 — LoCoMo 三塔 (Mixed Memory Pool)")
    print("=" * 72)

    # Hierarchical L0
    print("  [1/5] 加载 L0 对话消息 (Hierarchical L0) ...")
    l0_units = _load_hierarchical_l0(data_path, sample_ids)
    print(f"         → {len(l0_units)} units")

    # Hierarchical L1
    print("  [2/5] 加载 L1 Session 摘要 (Hierarchical L1) ...")
    l1_units = _load_hierarchical_l1(data_path, sample_ids)
    print(f"         → {len(l1_units)} units")

    # Hierarchical L2
    print("  [3/5] 加载 L2 结构化数据 (Hierarchical L2) ...")
    l2_units = _load_hierarchical_l2(data_path, sample_ids)
    print(f"         → {len(l2_units)} units")

    # Entity-Relation
    print("  [4/5] 加载实体关系 (Entity-Relation) ...")
    er_units = _load_entity_relation(data_path, sample_ids)
    print(f"         → {len(er_units)} units")

    # Episodic Memory
    print("  [5/5] 加载情景记忆 (Episodic Memory) ...")
    ep_units = _load_episodic_memory(data_path, sample_ids)
    print(f"         → {len(ep_units)} units")

    pool = l0_units + l1_units + l2_units + er_units + ep_units
    rng = random.Random(seed)
    rng.shuffle(pool)

    if pool_limit and len(pool) > pool_limit:
        pool = pool[:pool_limit]

    print(f"\n   总计记忆池大小: {len(pool)} units (已混洗)")
    print(f"     L0={len(l0_units)} | L1={len(l1_units)} | L2={len(l2_units)} "
          f"| ER={len(er_units)} | Episodic={len(ep_units)}")
    print()
    return pool





def run_10qps_benchmark(
    memory_pool: List[MemoryUnit],
    semantic_graph,
    total_requests: int = 500,
    qps: float = 10.0,
    space_name: str = "benchmark",
) -> Tuple[List[Dict[str, Any]], int, int]:
    """base_time = time.perf_counter() for i in range(total_requests): expected_start = base_time + i / qps now = time.perf_counter() if now < expected_start: time.sleep(expected_start - now) # ... call add_unit ... semantic_graph.add_unit( unit,."""
    interval = 1.0 / qps
    actual_total = min(total_requests, len(memory_pool))
    if actual_total < total_requests:
        print(
            f"  [WARN] 记忆池 ({len(memory_pool)}) 小于请求数 ({total_requests})，"
            f"实际发送 {actual_total} 次"
        )

    records: List[Dict[str, Any]] = []
    ok_count = 0
    fail_count = 0

    print("=" * 72)
    print(
        f"Phase 2+3: 严格 {qps} QPS 压测 "
        f"(total={actual_total}, interval={interval*1000:.1f}ms)"
    )
    print("=" * 72)
    print("  参数: index_update_mode=incremental, generate_sparse_embedding=True")
    print(f"  调度: base_time + i * {interval:.4f}s (无漂移)")
    print()

    gc_was_enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    base_time = time.perf_counter()
    last_progress = -1

    for i in range(actual_total):
        expected_start = base_time + i * interval
        now = time.perf_counter()
        if now < expected_start:
            time.sleep(expected_start - now)

        actual_start = time.perf_counter()
        unit = memory_pool[i]

        t0 = time.perf_counter()
        success = True
        error_msg = None
        try:
            semantic_graph.add_unit(
                unit,
                index_update_mode="incremental",
                generate_sparse_embedding=True,
            )
        except Exception as e:
            success = False
            error_msg = str(e)
        t1 = time.perf_counter()

        latency_ms = (t1 - t0) * 1000.0
        schedule_drift_ms = (actual_start - expected_start) * 1000.0

        record = {
            "index": i,
            "uid": unit.uid,
            "memory_level": unit.metadata.get("memory_level", unit.raw_data.get("node_type", "unknown")),
            "success": success,
            "latency_ms": round(latency_ms, 4),
            "schedule_drift_ms": round(schedule_drift_ms, 4),
            "expected_start_offset_s": round(i * interval, 4),
            "actual_start_offset_s": round(actual_start - base_time, 6),
        }
        if error_msg:
            record["error"] = error_msg

        records.append(record)

        if success:
            ok_count += 1
        else:
            fail_count += 1

        progress = int((i + 1) / actual_total * 10)
        if progress > last_progress:
            last_progress = progress
            elapsed = time.perf_counter() - base_time
            print(
                f"  [{progress*10:3d}%] {i+1}/{actual_total}  "
                f"elapsed={elapsed:.2f}s  ok={ok_count}  fail={fail_count}  "
                f"last_latency={latency_ms:.2f}ms"
            )

    if gc_was_enabled:
        gc.enable()
    gc.collect()

    total_elapsed = time.perf_counter() - base_time
    theoretical = actual_total / qps
    drift = total_elapsed - theoretical
    drift_pct = drift / theoretical * 100.0 if theoretical > 0 else 0.0

    print()
    print(f"  Load test complete: {total_elapsed:.3f}s (theoretical: {theoretical:.3f}s)")
    print(f"   Timing drift: {drift:+.4f}s ({drift_pct:+.2f}%)")
    print()

    return records, ok_count, fail_count





def print_statistics(records: List[Dict[str, Any]], ok_count: int, fail_count: int, qps: float):
    """Run print statistics."""
    latencies = np.array([r["latency_ms"] for r in records])
    total = len(latencies)

    mean = np.mean(latencies)
    p50 = np.percentile(latencies, 50)
    p90 = np.percentile(latencies, 90)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    min_val = np.min(latencies)
    max_val = np.max(latencies)
    std_val = np.std(latencies)

    ok_latencies = np.array([r["latency_ms"] for r in records if r["success"]])
    if len(ok_latencies) > 0:
        ok_mean = np.mean(ok_latencies)
        ok_p90 = np.percentile(ok_latencies, 90)
        ok_p99 = np.percentile(ok_latencies, 99)
    else:
        ok_mean = ok_p90 = ok_p99 = float("nan")

    drifts = np.array([r["schedule_drift_ms"] for r in records])

    print("=" * 72)
    print("Phase 4: 压测统计报告")
    print("=" * 72)
    print()
    print(f"  目标 QPS           : {qps}")
    print(f"  总请求数           : {total}")
    print(f"  成功               : {ok_count} ({ok_count/total*100:.1f}%)")
    print(f"  失败               : {fail_count} ({fail_count/total*100:.1f}%)")
    print()
    print("  ─── 全量延迟 (包含失败) ───")
    print(f"  Mean               : {mean:.2f} ms")
    print(f"  Std                : {std_val:.2f} ms")
    print(f"  Min                : {min_val:.2f} ms")
    print(f"  P50 (Median)       : {p50:.2f} ms")
    print(f"  P90                : {p90:.2f} ms")
    print(f"  P95                : {p95:.2f} ms")
    print(f"  P99                : {p99:.2f} ms")
    print(f"  Max                : {max_val:.2f} ms")
    print()

    if ok_count > 0 and fail_count > 0:
        print("  ─── 仅成功请求延迟 ───")
        print(f"  Mean               : {ok_mean:.2f} ms")
        print(f"  P90                : {ok_p90:.2f} ms")
        print(f"  P99                : {ok_p99:.2f} ms")
        print()

    interval_ms = 1000.0 / qps
    over_budget = int(np.sum(latencies > interval_ms))
    print("  ─── QPS 守约分析 ───")
    print(f"  单次预算           : {interval_ms:.1f} ms")
    print(f"  超预算请求         : {over_budget}/{total} ({over_budget/total*100:.1f}%)")
    print()

    print("  ─── 调度精度 ───")
    print(f"  调度漂移 Mean      : {np.mean(drifts):.4f} ms")
    print(f"  调度漂移 Max       : {np.max(drifts):.4f} ms")
    print(f"  调度漂移 P99       : {np.percentile(drifts, 99):.4f} ms")
    print()

    if fail_count > 0:
        print(f"  ─── 失败样本 (前 5 条) ───")
        shown = 0
        for r in records:
            if not r["success"]:
                print(f"    [{r['index']}] uid={r['uid']} error={r.get('error', 'N/A')}")
                shown += 1
                if shown >= 5:
                    break
        print()

    print("=" * 72)
    print("Done.")
    print("=" * 72)


def save_results_json(
    records: List[Dict[str, Any]],
    ok_count: int,
    fail_count: int,
    qps: float,
    output_path: str,
    extra_meta: Optional[Dict[str, Any]] = None,
):
    """Save results json."""
    latencies = np.array([r["latency_ms"] for r in records])
    drifts = np.array([r["schedule_drift_ms"] for r in records])
    total = len(latencies)

    summary = {
        "benchmark_time": datetime.now().isoformat(),
        "target_qps": qps,
        "total_requests": total,
        "ok_count": ok_count,
        "fail_count": fail_count,
        "success_rate": round(ok_count / total * 100, 2) if total > 0 else 0.0,
        "latency_ms": {
            "mean": round(float(np.mean(latencies)), 4),
            "std": round(float(np.std(latencies)), 4),
            "min": round(float(np.min(latencies)), 4),
            "p50": round(float(np.percentile(latencies, 50)), 4),
            "p90": round(float(np.percentile(latencies, 90)), 4),
            "p95": round(float(np.percentile(latencies, 95)), 4),
            "p99": round(float(np.percentile(latencies, 99)), 4),
            "max": round(float(np.max(latencies)), 4),
        },
        "schedule_drift_ms": {
            "mean": round(float(np.mean(drifts)), 4),
            "max": round(float(np.max(drifts)), 4),
            "p99": round(float(np.percentile(drifts, 99)), 4),
        },
        "over_budget_count": int(np.sum(latencies > 1000.0 / qps)),
    }
    if extra_meta:
        summary["config"] = extra_meta

    result = {
        "summary": summary,
        "records": records,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"   结果已保存到: {output_path}")



# main


def main():
    parser = argparse.ArgumentParser(
        description="LoCoMo 三塔 No-Time-Drift add_unit Benchmark (10 QPS)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(paths.LOCOMO_DATASET_DIR),
        help="LoCoMo 数据集根目录",
    )
    parser.add_argument(
        "--total-requests",
        type=int,
        default=500,
        help="压测总请求数",
    )
    parser.add_argument(
        "--qps",
        type=float,
        default=10.0,
        help="目标 QPS (默认 10)",
    )
    parser.add_argument(
        "--sample-ids",
        nargs="+",
        default=None,
        help="限制加载的 sample ID 列表 (例如: conv-26 conv-30)",
    )
    parser.add_argument(
        "--pool-limit",
        type=int,
        default=None,
        help="记忆池大小上限 (默认不限制，取 total_requests)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="Qwen/Qwen3-Embedding-0.6B",
        help="文本嵌入模型",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=None,
        help="嵌入维度 (默认从模型推断)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="结果 JSON 输出路径 (默认自动生成)",
    )
    args = parser.parse_args()

    warmup_count = 10
    pool_limit = args.pool_limit or (args.total_requests + warmup_count)

    # ---- Phase 1 ----
    memory_pool = prepare_mixed_memory_pool(
        data_dir=args.data_dir,
        sample_ids=args.sample_ids,
        pool_limit=pool_limit,
        seed=args.seed,
    )

    if len(memory_pool) == 0:
        print("[ERROR] 记忆池为空，无法执行压测。请检查数据目录。")
        sys.exit(1)

    print("=" * 72)
    print("初始化 SemanticGraph + SemanticMap (模型加载)")
    print("=" * 72)
    init_t0 = time.perf_counter()

    from mandol.core.semantic_map import SemanticMap  # noqa: E402
    from mandol.core.semantic_graph import SemanticGraph  # noqa: E402

    sm_kwargs = {"embedding_model_name": args.embedding_model}
    if args.embedding_dim is not None:
        sm_kwargs["embedding_dim"] = args.embedding_dim

    semantic_map = SemanticMap(**sm_kwargs)
    semantic_graph = SemanticGraph(semantic_map_instance=semantic_map)
    semantic_graph.create_memory_space_in_map("benchmark")

    init_elapsed = time.perf_counter() - init_t0
    print(f"   初始化完成: {init_elapsed:.2f}s")
    print()

    if len(memory_pool) < warmup_count:
        print(f"[ERROR] 记忆池不足 {warmup_count} 条，无法执行真实数据预热。")
        sys.exit(1)

    print(f"预热: 写入 {warmup_count} 条真实 warm-up units ...")
    warmup_units = [memory_pool.pop() for _ in range(warmup_count)]
    warmup_t0 = time.perf_counter()
    for warmup_unit in warmup_units:
        semantic_graph.add_unit(
            warmup_unit,
            index_update_mode="incremental",
            generate_sparse_embedding=True,
            space_names=["benchmark"],
        )
    warmup_elapsed = (time.perf_counter() - warmup_t0) * 1000.0
    print(f"   预热完成: 耗时 {warmup_elapsed:.2f} ms")
    print()

    # ---- Phase 2 + 3 ----
    records, ok_count, fail_count = run_10qps_benchmark(
        memory_pool=memory_pool,
        semantic_graph=semantic_graph,
        total_requests=args.total_requests,
        qps=args.qps,
        space_name="benchmark",
    )

    # ---- Phase 4 ----
    print_statistics(records, ok_count, fail_count, qps=args.qps)

    
    if args.output_json:
        output_path = args.output_json
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = (
            f"benchmark_locomo/task_eval/results/locomo_tri_tower_input_speed_results/"
            f"benchmark_triple_input_speed_{ts}.json"
        )
    # if args.output_json:
    #     output_path = args.output_json
    # else:
    #     ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     output_path = (
    #         f"benchmark_locomo/task_eval/results/"
    #         f"benchmark_triple_input_speed_{ts}.json"
    #     )

    extra_meta = {
        "data_dir": args.data_dir,
        "sample_ids": args.sample_ids,
        "total_requests": args.total_requests,
        "qps": args.qps,
        "pool_limit": pool_limit,
        "seed": args.seed,
        "embedding_model": args.embedding_model,
        "warmup_latency_ms": round(warmup_elapsed, 4),
        "init_time_s": round(init_elapsed, 2),
    }
    save_results_json(records, ok_count, fail_count, args.qps, output_path, extra_meta)


if __name__ == "__main__":
    main()
