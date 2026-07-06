#!/usr/bin/env python3
"""Fixed-QPS insertion benchmark for LongMemEval tri-tower memory units."""

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np




from mandol.core.memory_unit import MemoryUnit  # noqa: E402
from mandol.core import paths





def _load_hierarchical_units(data_dir: Path, max_qa: int = 5) -> List[MemoryUnit]:
    """Load hierarchical units."""
    units: List[MemoryUnit] = []
    base = data_dir / "longmemeval_hierarchical" / "step1_L0_graph"
    if not base.exists():
        print(f"  [WARN] 层级数据目录不存在: {base}")
        return units

    qa_dirs = sorted(base.iterdir(), key=lambda p: _qa_sort_key(p.name))
    for qa_dir in qa_dirs[:max_qa]:
        nodes_file = qa_dir / "nodes.json"
        if not nodes_file.exists():
            continue
        with open(nodes_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        nodes = data.get("nodes", [])
        for node in nodes:
            raw_uid = node.get("uid")
            if not raw_uid:
                continue
            unit = MemoryUnit(
                uid=raw_uid,
                raw_data=node.get("raw_data", {}),
                metadata=node.get("metadata", {}),
            )
            units.append(unit)
    return units


def _load_entity_relation_units(data_dir: Path, max_qa: int = 5) -> List[MemoryUnit]:
    """Load entity relation units."""
    units: List[MemoryUnit] = []
    base = data_dir / "entity_relation_graphs_new"
    if not base.exists():
        print(f"  [WARN] 实体关系目录不存在: {base}")
        return units

    qa_dirs = sorted(base.iterdir(), key=lambda p: _qa_sort_key(p.name))
    for qa_dir in qa_dirs[:max_qa]:
        entity_file = qa_dir / "entity_data.json"
        if not entity_file.exists():
            continue
        with open(entity_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        qa_id = data.get("qa_id", qa_dir.name)
        for entity_idx, entity in enumerate(data.get("entities", [])):
            entity_uid = entity.get("uid", entity.get("entity_id", ""))
            canonical_content = entity.get("canonical_content", entity.get("name", ""))
            category = entity.get("category", entity.get("entity_type", "UNKNOWN"))
            attributes = entity.get("attributes", {})
            mentions = entity.get("mentions", [])

            if entity_uid.startswith(f"{qa_id}_"):
                base_uid = f"{entity_uid}_{entity_idx}"
            else:
                base_uid = f"{qa_id}_{entity_uid}_{entity_idx}"

            for mention_idx, mention in enumerate(mentions, 1):
                mention_id = mention.get("mention_id", f"mention_{mention_idx}")
                mention_uid = f"{base_uid}_{mention_id}"
                content = mention.get("content", mention.get("context", ""))
                if not content:
                    continue

                raw_data = {
                    "node_type": "evidence_mention",
                    "qa_id": qa_id,
                    "entity_canonical": canonical_content,
                    "entity_category": category,
                    "content": content,
                    "session_ids": mention.get("session_ids", []),
                    "text_content": f"{canonical_content} ({category}): {content}",
                    "created_at": datetime.now().isoformat(),
                }
                units.append(MemoryUnit(uid=mention_uid, raw_data=raw_data))
    return units


def _load_episodic_units(data_dir: Path, max_qa: int = 5) -> List[MemoryUnit]:
    """Load episodic units."""
    units: List[MemoryUnit] = []
    episodic_dir = (
        _PROJECT_ROOT
        / "benchmark_longmemeval"
        / "dataset_maker"
        / "longmemeval_episodic_memory_new"
        / "deduplicated_results"
    )
    if not episodic_dir.exists():
        print(f"  [WARN] 情景记忆目录不存在: {episodic_dir}")
        return units

    qa_files = sorted(episodic_dir.glob("qa_*.json"), key=lambda p: _qa_sort_key(p.stem))
    for qa_file in qa_files[:max_qa]:
        with open(qa_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        qa_id = data.get("qa_id", qa_file.stem)
        events = data.get("facts", data.get("events", data.get("episodes", [])))
        for idx, event in enumerate(events):
            raw_event_id = event.get(
                "uid", event.get("event_id", event.get("id", f"event_{idx}"))
            )
            if str(raw_event_id).startswith(f"{qa_id}_"):
                event_uid = f"{raw_event_id}_{idx}"
            else:
                event_uid = f"{qa_id}_{raw_event_id}_{idx}"

            content = event.get(
                "canonical_content",
                event.get("content", event.get("summary", event.get("text", ""))),
            )
            if not content:
                continue

            event_date = (
                event.get("temporal_val")
                or event.get("session_date")
                or event.get("date")
                or event.get("time")
                or ""
            )
            category = event.get("category", "EPISODIC_EVENT")

            raw_data = {
                "node_type": category.lower() if category else "episodic_event",
                "qa_id": qa_id,
                "event_id": raw_event_id,
                "content": content,
                "event_date": event_date,
                "category": category,
                "text_content": content,
                "created_at": datetime.now().isoformat(),
            }
            units.append(MemoryUnit(uid=event_uid, raw_data=raw_data))
    return units


def _qa_sort_key(name: str) -> int:
    """qa_0 → 0, qa_123 → 123, fallback → 999999."""
    parts = name.replace(".json", "").split("_")
    for p in parts:
        if p.isdigit():
            return int(p)
    return 999999


def prepare_mixed_memory_pool(
    data_dir: str,
    max_qa_per_type: int = 5,
    pool_limit: Optional[int] = None,
    seed: int = 42,
) -> List[MemoryUnit]:
    """Run prepare mixed memory pool."""
    data_path = Path(data_dir)
    print("=" * 70)
    print("Phase 1: 准备混合记忆池 (Mixed Memory Pool)")
    print("=" * 70)

    print("  [1/3] 加载层级节点 (Hierarchical) ...")
    h_units = _load_hierarchical_units(data_path, max_qa_per_type)
    print(f"         → {len(h_units)} units")

    print("  [2/3] 加载实体关系 (Entity-Relation) ...")
    er_units = _load_entity_relation_units(data_path, max_qa_per_type)
    print(f"         → {len(er_units)} units")

    print("  [3/3] 加载情景记忆 (Episodic) ...")
    ep_units = _load_episodic_units(data_path, max_qa_per_type)
    print(f"         → {len(ep_units)} units")

    pool = h_units + er_units + ep_units
    rng = random.Random(seed)
    rng.shuffle(pool)

    if pool_limit and len(pool) > pool_limit:
        pool = pool[:pool_limit]

    print(f"   总计记忆池大小: {len(pool)} units (已混洗)")
    print()
    return pool





def run_10qps_benchmark(
    memory_pool: List[MemoryUnit],
    semantic_graph,
    total_requests: int = 500,
    qps: float = 10.0,
    space_name: str = "benchmark",
) -> Tuple[List[float], List[bool], List[Optional[str]]]:
    """base_time = time.perf_counter() for i in range(total_requests): expected_start = base_time + i / qps now = time.perf_counter() if now < expected_start: time.sleep(expected_start - now) # ... call add_unit ... semantic_graph.add_unit( unit,."""
    interval = 1.0 / qps
    actual_total = min(total_requests, len(memory_pool))
    if actual_total < total_requests:
        print(
            f"  [WARN] 记忆池 ({len(memory_pool)}) 小于请求数 ({total_requests})，"
            f"实际发送 {actual_total} 次"
        )

    latencies_ms: List[float] = []
    successes: List[bool] = []
    errors: List[Optional[str]] = []

    print("=" * 70)
    print(
        f"Phase 2+3: 严格 {qps} QPS 压测 "
        f"(total={actual_total}, interval={interval*1000:.1f}ms)"
    )
    print("=" * 70)
    print(
        f"  参数: index_update_mode=incremental, generate_sparse_embedding=True"
    )
    print(f"  调度: base_time + i * {interval:.4f}s (无漂移)")
    print()

    base_time = time.perf_counter()
    last_progress = -1

    for i in range(actual_total):
        expected_start = base_time + i * interval
        now = time.perf_counter()
        if now < expected_start:
            time.sleep(expected_start - now)

        unit = memory_pool[i]

        t0 = time.perf_counter()
        try:
            semantic_graph.add_unit(
                unit,
                index_update_mode="incremental",
                generate_sparse_embedding=True,
            )
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)
            successes.append(True)
            errors.append(None)
        except Exception as e:
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)
            successes.append(False)
            errors.append(str(e))

        progress = int((i + 1) / actual_total * 10)
        if progress > last_progress:
            last_progress = progress
            elapsed = time.perf_counter() - base_time
            ok_so_far = sum(successes)
            print(
                f"  [{progress*10:3d}%] {i+1}/{actual_total}  "
                f"elapsed={elapsed:.2f}s  ok={ok_so_far}  "
                f"last_latency={latencies_ms[-1]:.2f}ms"
            )

    total_elapsed = time.perf_counter() - base_time
    print()
    print(f"  Load test complete: {total_elapsed:.3f}s (theoretical: {actual_total / qps:.3f}s)")

    theoretical = actual_total / qps
    drift = total_elapsed - theoretical
    drift_pct = drift / theoretical * 100.0 if theoretical > 0 else 0.0
    print(f"   时间漂移: {drift:+.4f}s ({drift_pct:+.2f}%)")
    print()

    return latencies_ms, successes, errors





def print_statistics(
    latencies_ms: List[float],
    successes: List[bool],
    errors: List[Optional[str]],
    qps: float,
):
    """Run print statistics."""
    arr = np.array(latencies_ms)
    total = len(arr)
    ok_count = sum(successes)
    fail_count = total - ok_count

    mean = np.mean(arr)
    p50 = np.percentile(arr, 50)
    p90 = np.percentile(arr, 90)
    p95 = np.percentile(arr, 95)
    p99 = np.percentile(arr, 99)
    min_val = np.min(arr)
    max_val = np.max(arr)
    std_val = np.std(arr)

    ok_arr = arr[np.array(successes)]
    if len(ok_arr) > 0:
        ok_mean = np.mean(ok_arr)
        ok_p90 = np.percentile(ok_arr, 90)
        ok_p99 = np.percentile(ok_arr, 99)
    else:
        ok_mean = ok_p90 = ok_p99 = float("nan")

    print("=" * 70)
    print("Phase 4: 压测统计报告")
    print("=" * 70)
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
    over_budget = np.sum(arr > interval_ms)
    print(f"  ─── QPS 守约分析 ───")
    print(f"  单次预算           : {interval_ms:.1f} ms")
    print(f"  超预算请求         : {over_budget}/{total} ({over_budget/total*100:.1f}%)")
    print()

    if fail_count > 0:
        print(f"  ─── 失败样本 (前 5 条) ───")
        shown = 0
        for i, (ok, err) in enumerate(zip(successes, errors)):
            if not ok:
                print(f"    [{i}] {err}")
                shown += 1
                if shown >= 5:
                    break
        print()

    print("=" * 70)
    print("Done.")
    print("=" * 70)



# main


def main():
    parser = argparse.ArgumentParser(
        description="No-Time-Drift add_unit Benchmark (10 QPS)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(paths.LONGMEMEVAL_DATASET_DIR),
        help="LongMemEval 数据集根目录",
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
        "--max-qa-per-type",
        type=int,
        default=5,
        help="每类数据最多读取的 QA 数量",
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
    args = parser.parse_args()

    pool_limit = args.pool_limit or args.total_requests

    # ---- Phase 1 ----
    memory_pool = prepare_mixed_memory_pool(
        data_dir=args.data_dir,
        max_qa_per_type=args.max_qa_per_type,
        pool_limit=pool_limit,
        seed=args.seed,
    )

    if len(memory_pool) == 0:
        print("[ERROR] 记忆池为空，无法执行压测。请检查数据目录。")
        sys.exit(1)

    print("=" * 70)
    print("初始化 SemanticGraph + SemanticMap (模型加载)")
    print("=" * 70)
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

    print("预热: 写入 1 条 warm-up unit ...")
    warmup_unit = MemoryUnit(
        uid="__warmup__",
        raw_data={"text_content": "warmup data for benchmark", "node_type": "warmup"},
    )
    warmup_t0 = time.perf_counter()
    semantic_graph.add_unit(
        warmup_unit,
        index_update_mode="incremental",
        generate_sparse_embedding=True,
        space_names=["benchmark"],
    )
    warmup_elapsed = (time.perf_counter() - warmup_t0) * 1000.0
    print(f"   预热完成: {warmup_elapsed:.2f}ms")
    print()

    # ---- Phase 2 + 3 ----
    latencies_ms, successes, errors = run_10qps_benchmark(
        memory_pool=memory_pool,
        semantic_graph=semantic_graph,
        total_requests=args.total_requests,
        qps=args.qps,
        space_name="benchmark",
    )

    # ---- Phase 4 ----
    print_statistics(latencies_ms, successes, errors, qps=args.qps)


if __name__ == "__main__":
    main()
