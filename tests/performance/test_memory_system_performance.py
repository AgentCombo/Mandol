"""Performance tests for MemorySystem core operations.

Measures throughput and memory usage for add, build, and retrieval
under various load conditions.
"""

from __future__ import annotations

import gc
import logging
import time
import tracemalloc
from datetime import datetime, timezone
from typing import List
from unittest.mock import patch

from mandol.application.memory_system import (
    MemorySystem,
    MemorySystemConfig,
    MAX_CONTEXT_UNITS,
    SESSION_CHECK_INTERVAL,
)
from mandol.domain.memory_unit import MemoryUnit
from mandol.domain.types import Uid


def create_test_units(count: int) -> List[MemoryUnit]:
    units = []
    for i in range(count):
        unit = MemoryUnit(
            uid=Uid(f"perf_unit_{i}"),
            raw_data={"text_content": f"Test content for unit {i}. " * 50},
            metadata={"timestamp": datetime.now(timezone.utc).isoformat()}
        )
        units.append(unit)
    return units


def test_add_single_unit_latency():
    print("\n=== Test: Single Unit Add Latency ===")
    ms = MemorySystem()
    units = create_test_units(100)

    latencies = []
    for unit in units:
        start = time.perf_counter()
        ms.add(unit)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)

    avg_latency = sum(latencies) / len(latencies)
    p50_latency = sorted(latencies)[len(latencies) // 2]
    p99_latency = sorted(latencies)[int(len(latencies) * 0.99)]

    print(f"  Units added: {len(latencies)}")
    print(f"  Average latency: {avg_latency:.2f} ms")
    print(f"  P50 latency: {p50_latency:.2f} ms")
    print(f"  P99 latency: {p99_latency:.2f} ms")

    ms.flush()
    return avg_latency, p50_latency, p99_latency


def test_add_many_batch_latency():
    print("\n=== Test: Batch Add (add_many) Latency ===")
    ms = MemorySystem()

    batch_sizes = [10, 50, 100, 200]
    results = []

    for batch_size in batch_sizes:
        units = create_test_units(batch_size)
        start = time.perf_counter()
        ms.add_many(units)
        end = time.perf_counter()
        latency = (end - start) * 1000
        throughput = batch_size / ((end - start))

        results.append({
            "batch_size": batch_size,
            "latency_ms": latency,
            "throughput_units_per_sec": throughput
        })
        print(f"  Batch size {batch_size}: {latency:.2f} ms, {throughput:.1f} units/sec")

        ms.flush()

    return results


def test_memory_usage():
    print("\n=== Test: Memory Usage ===")
    tracemalloc.start()

    ms = MemorySystem()
    initial_memory = tracemalloc.get_traced_memory()[0]

    units = create_test_units(1000)
    for unit in units:
        ms.add(unit)

    current_memory = tracemalloc.get_traced_memory()[0]
    peak_memory = tracemalloc.get_traced_memory()[1]

    memory_per_unit = (current_memory - initial_memory) / 1000

    print(f"  Initial memory: {initial_memory / 1024:.2f} KB")
    print(f"  After 1000 units: {current_memory / 1024:.2f} KB")
    print(f"  Peak memory: {peak_memory / 1024:.2f} KB")
    print(f"  Memory per unit: {memory_per_unit:.2f} KB")

    tracemalloc.stop()
    ms.flush()

    return {
        "initial_memory_kb": initial_memory / 1024,
        "final_memory_kb": current_memory / 1024,
        "peak_memory_kb": peak_memory / 1024,
        "memory_per_unit_kb": memory_per_unit
    }


def test_context_window_enforcement():
    print("\n=== Test: Context Window Enforcement ===")
    ms = MemorySystem()

    units = create_test_units(100)
    for unit in units:
        ms.add(unit)

    with ms._pending_lock:
        pending_count = len(ms._pending_units)
        max_expected = MAX_CONTEXT_UNITS + SESSION_CHECK_INTERVAL

    print(f"  Pending units after 100 adds: {pending_count}")
    print(f"  Max expected (MAX_CONTEXT + CHECK_INTERVAL): {max_expected}")

    enforced = pending_count <= max_expected
    print(f"  Context window enforced: {enforced}")

    ms.flush()
    return enforced


def test_session_boundary_check_efficiency():
    print("\n=== Test: Session Boundary Check Efficiency ===")

    from unittest.mock import MagicMock

    class MockLLM:
        def __init__(self):
            self.call_count = 0

        def chat(self, messages, temperature=0.1, max_tokens=512):
            self.call_count += 1
            import json
            return MagicMock(content=json.dumps({
                "should_split": False,
                "split_at_index": -1,
                "reason": "Mock response"
            }))

    ms = MemorySystem(llm_provider=MockLLM())

    units = create_test_units(100)
    for unit in units:
        ms.add(unit)

    mock_llm = ms._llm
    calls_made = getattr(mock_llm, 'call_count', 0)

    expected_max_calls = (100 // SESSION_CHECK_INTERVAL) + 1
    efficiency_ratio = calls_made / expected_max_calls if expected_max_calls > 0 else 0

    print(f"  LLM calls made: {calls_made}")
    print(f"  Expected max calls: {expected_max_calls}")
    print(f"  Efficiency ratio: {efficiency_ratio:.2%}")

    ms.flush()
    return {
        "llm_calls": calls_made,
        "expected_max": expected_max_calls,
        "efficiency": efficiency_ratio
    }


def test_threading_safety():
    print("\n=== Test: Threading Safety ===")
    import threading
    from concurrent.futures import ThreadPoolExecutor

    ms = MemorySystem()
    errors = []

    def worker(worker_id: int, count: int):
        try:
            for i in range(count):
                unit = MemoryUnit(
                    uid=Uid(f"worker{worker_id}_unit_{i}"),
                    raw_data={"text_content": f"Content from worker {worker_id}, unit {i}"},
                    metadata={"timestamp": datetime.now(timezone.utc).isoformat()}
                )
                ms.add(unit)
        except Exception as e:
            errors.append((worker_id, str(e)))

    num_workers = 4
    units_per_worker = 25

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(worker, i, units_per_worker) for i in range(num_workers)]
        for f in futures:
            f.result()
    end = time.perf_counter()

    total_units = num_workers * units_per_worker
    duration = end - start
    throughput = total_units / duration

    with ms._pending_lock:
        final_pending = len(ms._pending_units)
        final_order = len(ms._insertion_order)

    print(f"  Workers: {num_workers}")
    print(f"  Units per worker: {units_per_worker}")
    print(f"  Total units added: {total_units}")
    print(f"  Duration: {duration:.2f} sec")
    print(f"  Throughput: {throughput:.1f} units/sec")
    print(f"  Final pending count: {final_pending}")
    print(f"  Final insertion order count: {final_order}")
    print(f"  Errors: {len(errors)}")

    ms.flush()
    return {
        "errors": len(errors),
        "throughput": throughput,
        "final_pending": final_pending,
        "final_order": final_order
    }


def run_performance_tests():
    print("=" * 60)
    print("Memory System Performance Test Suite")
    print("=" * 60)

    results = {}

    try:
        avg_lat, p50_lat, p99_lat = test_add_single_unit_latency()
        results["single_unit_latency"] = {
            "avg_ms": avg_lat, "p50_ms": p50_lat, "p99_ms": p99_lat
        }
    except Exception as e:
        print(f"  ERROR: {e}")
        results["single_unit_latency"] = {"error": str(e)}

    try:
        batch_results = test_add_many_batch_latency()
        results["batch_latency"] = batch_results
    except Exception as e:
        print(f"  ERROR: {e}")
        results["batch_latency"] = {"error": str(e)}

    try:
        mem_results = test_memory_usage()
        results["memory_usage"] = mem_results
    except Exception as e:
        print(f"  ERROR: {e}")
        results["memory_usage"] = {"error": str(e)}

    try:
        enforced = test_context_window_enforcement()
        results["context_window_enforced"] = enforced
    except Exception as e:
        print(f"  ERROR: {e}")
        results["context_window_enforced"] = {"error": str(e)}

    try:
        efficiency_results = test_session_boundary_check_efficiency()
        results["session_boundary_efficiency"] = efficiency_results
    except Exception as e:
        print(f"  ERROR: {e}")
        results["session_boundary_efficiency"] = {"error": str(e)}

    try:
        thread_results = test_threading_safety()
        results["threading_safety"] = thread_results
    except Exception as e:
        print(f"  ERROR: {e}")
        results["threading_safety"] = {"error": str(e)}

    print("\n" + "=" * 60)
    print("Performance Test Summary")
    print("=" * 60)

    if "single_unit_latency" in results and "avg_ms" in results["single_unit_latency"]:
        lat = results["single_unit_latency"]
        print(f"Single Unit Latency:")
        print(f"  - Average: {lat['avg_ms']:.2f} ms")
        print(f"  - P50: {lat['p50_ms']:.2f} ms")
        print(f"  - P99: {lat['p99_ms']:.2f} ms")

    if "memory_usage" in results and "memory_per_unit_kb" in results["memory_usage"]:
        mem = results["memory_usage"]
        print(f"\nMemory Usage:")
        print(f"  - Per unit: {mem['memory_per_unit_kb']:.2f} KB")
        print(f"  - Peak: {mem['peak_memory_kb']:.2f} KB")

    if "threading_safety" in results and "errors" in results["threading_safety"]:
        thread = results["threading_safety"]
        print(f"\nThreading Safety:")
        print(f"  - Errors: {thread['errors']}")
        print(f"  - Throughput: {thread['throughput']:.1f} units/sec")

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    results = run_performance_tests()

    import json
    from pathlib import Path
    output_path = Path(__file__).parent / "performance_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")
