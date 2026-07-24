"""Benchmark vector-seeded traversal physical-plan candidates in memory.

Run from the repository root:

    python examples/benchmark_in_memory_optimizer_candidates.py

The generated workload is deterministic and is not persisted. Index build
time is reported separately from query execution time.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mandol.domain.memory_unit import MemoryUnit
from mandol.domain.types import Uid
from mandol.infrastructure import (
    FaissHNSWVectorIndex,
    InMemoryCosineVectorIndex,
    InMemoryGraphStore,
    InMemoryUnifiedQueryStore,
    InMemoryUnitStore,
)
from mandol.query import VectorSeededTraversalSpec


@dataclass(slots=True)
class Observation:
    total_ms: float
    vector_ms: float
    join_ms: float
    traversal_ms: float
    other_ms: float
    recall: float


def _memory(
    uid: str,
    kind: str,
    *,
    embedding: np.ndarray | None = None,
    **metadata,
) -> MemoryUnit:
    return MemoryUnit(
        uid=Uid(uid),
        raw_data={"text_content": uid},
        metadata={"kind": kind, **metadata},
        embedding=embedding,
    )


def _normalized_random(rng: np.random.Generator, count: int, dim: int) -> np.ndarray:
    matrix = rng.normal(size=(count, dim)).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0.0, 1.0, norms)


def build_dataset(
    *,
    task_count: int,
    noise_vector_count: int,
    dim: int,
    seed: int,
) -> tuple[
    InMemoryUnitStore,
    InMemoryGraphStore,
    list[tuple[Uid, np.ndarray]],
    np.ndarray,
]:
    rng = np.random.default_rng(seed)
    task_vectors = _normalized_random(rng, task_count, dim)
    noise_vectors = _normalized_random(rng, noise_vector_count, dim)
    records: list[MemoryUnit] = []
    vector_items: list[tuple[Uid, np.ndarray]] = []
    graph = InMemoryGraphStore()

    for index, vector in enumerate(task_vectors):
        task_uid = Uid(f"task-{index:06d}")
        session_uid = Uid(f"session-{index:06d}")
        best_uid = Uid(f"node-best-{index:06d}")
        lower_uid = Uid(f"node-lower-{index:06d}")
        child_uid = Uid(f"child-{index:06d}")
        grandchild_uid = Uid(f"grandchild-{index:06d}")
        records.extend(
            [
                _memory(str(task_uid), "task", embedding=vector),
                _memory(str(session_uid), "session"),
                _memory(
                    str(best_uid),
                    "node",
                    fitness_score=0.9,
                    is_buggy=False,
                ),
                _memory(
                    str(lower_uid),
                    "node",
                    fitness_score=0.4,
                    is_buggy=False,
                ),
                _memory(str(child_uid), "trajectory"),
                _memory(str(grandchild_uid), "trajectory"),
            ]
        )
        vector_items.append((task_uid, vector))
        for source, target, relation in (
            (session_uid, task_uid, "BELONGS_TO"),
            (best_uid, session_uid, "IN_SESSION"),
            (lower_uid, session_uid, "IN_SESSION"),
            (best_uid, child_uid, "HAS_CHILD"),
            (child_uid, grandchild_uid, "HAS_CHILD"),
        ):
            graph.upsert_relationship(source, target, relation, {})

    for index, vector in enumerate(noise_vectors):
        uid = Uid(f"document-{index:06d}")
        records.append(_memory(str(uid), "document", embedding=vector))
        vector_items.append((uid, vector))

    units = InMemoryUnitStore()
    units.upsert_units(records)
    return units, graph, vector_items, task_vectors


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _format_table(rows: list[list[str]]) -> str:
    widths = [max(len(row[index]) for row in rows) for index in range(len(rows[0]))]
    return "\n".join(
        "  ".join(value.rjust(widths[index]) for index, value in enumerate(row))
        for row in rows
    )


def _seed_uids(execution) -> set[str]:
    return {str(row.seed_uid) for row in execution.rows}


def benchmark(args: argparse.Namespace) -> None:
    import faiss

    faiss.omp_set_num_threads(args.faiss_threads)
    dataset_started = time.perf_counter()
    units, graph, vector_items, task_vectors = build_dataset(
        task_count=args.tasks,
        noise_vector_count=args.noise_vectors,
        dim=args.dim,
        seed=args.seed,
    )
    dataset_ms = (time.perf_counter() - dataset_started) * 1000.0

    exact = InMemoryCosineVectorIndex(args.dim)
    started = time.perf_counter()
    exact.upsert(vector_items)
    exact_build_ms = (time.perf_counter() - started) * 1000.0

    ann = FaissHNSWVectorIndex(
        args.dim,
        m=args.hnsw_m,
        ef_construction=args.ef_construction,
        ef_search=args.ef_search,
    )
    started = time.perf_counter()
    ann.upsert(vector_items)
    ann_build_ms = (time.perf_counter() - started) * 1000.0

    executors = {
        "exact_full_scan": InMemoryUnifiedQueryStore(
            unit_store=units,
            vector_index=exact,
            graph_store=graph,
            vector_strategy="exact_full_scan",
        ),
        "filter_first_exact": InMemoryUnifiedQueryStore(
            unit_store=units,
            vector_index=exact,
            graph_store=graph,
            vector_strategy="filter_first_exact",
        ),
        "ann_adaptive": InMemoryUnifiedQueryStore(
            unit_store=units,
            vector_index=ann,
            graph_store=graph,
            vector_strategy="ann_adaptive",
            ann_initial_oversample=args.ann_oversample,
        ),
    }

    rng = np.random.default_rng(args.seed + 1)
    query_indices = rng.integers(0, args.tasks, size=args.warmup + args.queries)
    query_noise = rng.normal(
        scale=args.query_noise,
        size=(len(query_indices), args.dim),
    ).astype(np.float32)
    query_vectors = task_vectors[query_indices] + query_noise
    query_vectors /= np.linalg.norm(query_vectors, axis=1, keepdims=True)

    observations: dict[str, list[Observation]] = defaultdict(list)
    exact_seeds: dict[int, set[str]] = {}
    per_operator: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for query_number, query_vector in enumerate(query_vectors):
        spec = VectorSeededTraversalSpec(
            query_vector=query_vector,
            vector_k=args.vector_k,
            seed_metadata_equals={"kind": "task"},
            node_metadata_equals={"kind": "node", "is_buggy": False},
            nodes_per_seed=args.nodes_per_seed,
            min_hops=0,
            max_hops=args.max_hops,
            result_limit=max(100, args.vector_k * args.nodes_per_seed * 3),
        )
        for name, executor in executors.items():
            execution = executor.vector_seeded_graph_traversal(spec, profile=True)
            seeds = _seed_uids(execution)
            if name == "exact_full_scan":
                exact_seeds[query_number] = seeds
            baseline = exact_seeds[query_number]
            recall = 1.0 if not baseline else len(seeds & baseline) / len(baseline)
            if query_number < args.warmup:
                continue
            timings = execution.metrics.timings
            observations[name].append(
                Observation(
                    total_ms=execution.metrics.elapsed_ms,
                    vector_ms=float(timings.vector_search_cpu_ms or 0.0),
                    join_ms=float(timings.relation_join_cpu_ms or 0.0),
                    traversal_ms=float(timings.graph_traversal_cpu_ms or 0.0),
                    other_ms=float(timings.other_cpu_ms or 0.0),
                    recall=recall,
                )
            )
            for metric in execution.metrics.operators:
                per_operator[name][metric.operator_id].append(metric.elapsed_ms)

    print(
        "DATASET "
        f"tasks={args.tasks} noise_vectors={args.noise_vectors} "
        f"units={len(units.list_units())} edges={len(graph.get_all_edges())} "
        f"dim={args.dim} queries={args.queries} warmup={args.warmup}"
    )
    print(
        "BUILD "
        f"dataset_ms={dataset_ms:.2f} exact_index_ms={exact_build_ms:.2f} "
        f"hnsw_index_ms={ann_build_ms:.2f} "
        f"hnsw_m={args.hnsw_m} ef_construction={args.ef_construction} "
        f"ef_search={args.ef_search} faiss_threads={args.faiss_threads}"
    )

    table = [
        [
            "candidate",
            "total p50",
            "total p95",
            "vector p50",
            "join p50",
            "traverse p50",
            "other p50",
            "recall@k",
        ]
    ]
    for name in executors:
        values = observations[name]
        table.append(
            [
                name,
                f"{statistics.median(item.total_ms for item in values):.3f}",
                f"{_percentile([item.total_ms for item in values], 95):.3f}",
                f"{statistics.median(item.vector_ms for item in values):.3f}",
                f"{statistics.median(item.join_ms for item in values):.3f}",
                f"{statistics.median(item.traversal_ms for item in values):.3f}",
                f"{statistics.median(item.other_ms for item in values):.3f}",
                f"{statistics.mean(item.recall for item in values):.3f}",
            ]
        )
    print("\nSTAGE BREAKDOWN (milliseconds; p50 except total p95)")
    print(_format_table(table))

    print("\nOPERATOR BREAKDOWN (median milliseconds)")
    for name in executors:
        details = "  ".join(
            f"{operator_id}={statistics.median(values):.3f}"
            for operator_id, values in per_operator[name].items()
        )
        print(f"{name}: {details}")

    winner = min(
        executors,
        key=lambda name: statistics.median(
            item.total_ms for item in observations[name]
        ),
    )
    winner_ms = statistics.median(item.total_ms for item in observations[winner])
    print(f"\nWINNER by median end-to-end latency: {winner} ({winner_ms:.3f} ms)")

    if args.show_plans:
        sample_spec = VectorSeededTraversalSpec(
            query_vector=query_vectors[-1],
            vector_k=args.vector_k,
            seed_metadata_equals={"kind": "task"},
            node_metadata_equals={"kind": "node", "is_buggy": False},
            nodes_per_seed=args.nodes_per_seed,
            max_hops=args.max_hops,
        )
        print("\nPHYSICAL PLANS")
        for name, executor in executors.items():
            print(f"\n[{name}]")
            print(executor.explain_vector_seeded_graph_traversal(sample_spec))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=int, default=3000)
    parser.add_argument("--noise-vectors", type=int, default=9000)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--vector-k", type=int, default=10)
    parser.add_argument("--nodes-per-seed", type=int, default=1)
    parser.add_argument("--max-hops", type=int, default=2)
    parser.add_argument("--query-noise", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=80)
    parser.add_argument("--ef-search", type=int, default=64)
    parser.add_argument("--ann-oversample", type=int, default=4)
    parser.add_argument("--faiss-threads", type=int, default=1)
    parser.add_argument("--show-plans", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    for name in ("tasks", "dim", "queries", "vector_k", "faiss_threads"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.noise_vectors < 0 or args.warmup < 0:
        parser.error("--noise-vectors and --warmup must be non-negative")
    return args


if __name__ == "__main__":
    benchmark(parse_args())
