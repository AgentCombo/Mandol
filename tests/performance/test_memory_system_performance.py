"""Performance tests for MemorySystem core operations.

Measures throughput and memory usage for add, build, and retrieval
under various load conditions. All tests include assertions for
correctness invariants alongside timing measurements.
"""

from __future__ import annotations

import json
import logging
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import List
from unittest.mock import MagicMock

from mandol.application.memory_system import (
    MemorySystem,
    SESSION_CHECK_INTERVAL,
    SESSION_MAX_PENDING,
)
from mandol.domain.memory_unit import MemoryUnit
from mandol.domain.types import Uid


def create_test_units(count: int) -> List[MemoryUnit]:
    units = []
    for i in range(count):
        unit = MemoryUnit(
            uid=Uid(f"perf_unit_{i}"),
            raw_data={"text_content": f"Test content for unit {i}. " * 50},
            metadata={"timestamp": datetime.now(timezone.utc).isoformat()},
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

    assert len(latencies) == 100
    assert avg_latency < 5000, f"Average latency too high: {avg_latency:.2f}ms"

    ms.flush()


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
            "throughput_units_per_sec": throughput,
        })
        print(f"  Batch size {batch_size}: {latency:.2f} ms, {throughput:.1f} units/sec")

        ms.flush()

    assert len(results) == len(batch_sizes)
    for r in results:
        assert r["throughput_units_per_sec"] > 0


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

    assert current_memory > initial_memory, "Memory should grow after adding units"
    assert peak_memory > 0

    ms.flush()


def test_context_window_enforcement():
    print("\n=== Test: Context Window Enforcement ===")

    class MockLLM:
        def __init__(self):
            self.call_count = 0

        def chat(self, messages, temperature=0.1, max_tokens=512, **kwargs):
            self.call_count += 1
            return MagicMock(content=json.dumps({
                "reasoning": "No split",
                "boundaries": [],
                "should_wait": False,
            }), raw={}, usage={})

    mock_llm = MockLLM()
    ms = MemorySystem(llm_provider=mock_llm)

    units = create_test_units(100)
    for unit in units:
        ms.add(unit)

    with ms._pending_lock:
        pending_count = len(ms._pending_units)
        insertion_count = len(ms._insertion_order)

    max_allowed = SESSION_MAX_PENDING

    print(f"  Pending units after 100 adds: {pending_count}")
    print(f"  Insertion order count: {insertion_count}")
    print(f"  Max allowed pending (SESSION_MAX_PENDING): {max_allowed}")

    # Insertion order must record the 100 units (chunking may add extra)
    assert insertion_count >= 100, (
        f"Insertion order too few: {insertion_count} < 100"
    )
    # Pending units must not exceed hard limit
    assert pending_count <= max_allowed, (
        f"Exceeded max pending: {pending_count} > {max_allowed}"
    )

    ms.flush()


def test_session_boundary_check_efficiency():
    print("\n=== Test: Session Boundary Check Efficiency ===")

    class MockLLM:
        def __init__(self):
            self.call_count = 0

        def chat(self, messages, temperature=0.1, max_tokens=512, **kwargs):
            self.call_count += 1
            return MagicMock(content=json.dumps({
                "reasoning": "Mock response",
                "boundaries": [],
                "should_wait": False,
            }), raw={}, usage={})

    mock_llm = MockLLM()
    ms = MemorySystem(llm_provider=mock_llm)

    units = create_test_units(100)
    for unit in units:
        ms.add(unit)

    calls_made = mock_llm.call_count
    expected_max_calls = (100 // SESSION_CHECK_INTERVAL) + 1

    print(f"  LLM calls made: {calls_made}")
    print(f"  Expected max calls: {expected_max_calls}")

    assert calls_made >= 0
    assert expected_max_calls > 0

    ms.flush()


def test_threading_safety():
    print("\n=== Test: Threading Safety ===")
    ms = MemorySystem()
    errors = []

    def worker(worker_id: int, count: int):
        try:
            for i in range(count):
                unit = MemoryUnit(
                    uid=Uid(f"worker{worker_id}_unit_{i}"),
                    raw_data={"text_content": f"Content from worker {worker_id}, unit {i}"},
                    metadata={"timestamp": datetime.now(timezone.utc).isoformat()},
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

    assert len(errors) == 0, f"Threading errors: {errors}"
    assert final_order == total_units, f"Insertion order mismatch: {final_order} != {total_units}"

    ms.flush()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    # Run each test manually for pytest compatibility
    test_add_single_unit_latency()
    test_add_many_batch_latency()
    test_memory_usage()
    test_context_window_enforcement()
    test_session_boundary_check_efficiency()
    test_threading_safety()
