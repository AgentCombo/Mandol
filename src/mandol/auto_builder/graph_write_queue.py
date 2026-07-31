"""Producer-consumer graph write queue for high-concurrency builders.

LLM workers should create ``GraphWriteRequest`` objects and enqueue them here.
Only the consumer thread touches SemanticGraph/SemanticMap write APIs.
"""

from __future__ import annotations

import queue
import threading
import time
import traceback
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..core.memory_unit import MemoryUnit
from ..utils.logging_config import create_module_logger


logger = create_module_logger("auto_builder.graph_write_queue")


@dataclass
class GraphWriteRequest:
    """A complete deferred ``add_unit`` call.

    ``unit`` is the MemoryUnit that will be inserted. The remaining fields mirror
    the write-time arguments accepted by ``SemanticGraph.add_unit`` and are kept
    per unit so mixed L1/L2/episodic/entity batches can still be flushed together.
    """

    unit: MemoryUnit
    explicit_content_for_embedding: Optional[Any] = None
    content_type_for_embedding: Optional[str] = "text"
    space_names: Optional[List[str]] = None
    index_update_mode: str = "none"
    generate_sparse_embedding: bool = False
    sparse_model_name: str = "naver/splade-v3"
    source: str = "auto_builder"
    metadata: Dict[str, Any] = field(default_factory=dict)


class GraphWriteQueue:
    """Dedicated consumer for batched graph writes."""

    _SENTINEL = object()

    def __init__(
        self,
        semantic_system: Any,
        batch_size: int = 32,
        embedding_batch_size: int = 32,
        max_queue_size: int = 2048,
        flush_interval: float = 0.25,
        show_progress: bool = False,
        raise_on_error: bool = True,
        name: str = "GraphWriteQueue",
    ) -> None:
        if semantic_system is None:
            raise ValueError("semantic_system is required")
        if not hasattr(semantic_system, "batch_add_units"):
            raise TypeError("semantic_system must provide batch_add_units()")

        self.semantic_system = semantic_system
        self.batch_size = max(1, int(batch_size))
        self.embedding_batch_size = max(1, int(embedding_batch_size))
        self.flush_interval = max(0.01, float(flush_interval))
        self.show_progress = show_progress
        self.raise_on_error = raise_on_error
        self._queue: "queue.Queue[Any]" = queue.Queue(maxsize=max(0, int(max_queue_size)))
        self._thread = threading.Thread(target=self._consume, name=name, daemon=True)
        self._started = False
        self._closed = False
        self._stats_lock = threading.Lock()
        self._errors: List[str] = []
        self._stats: Dict[str, Any] = {
            "enqueued": 0,
            "batches": 0,
            "groups": 0,
            "added": 0,
            "skipped": 0,
            "embedding_generated": 0,
            "sparse_generated": 0,
            "graph_nodes_added": 0,
            "graph_nodes_updated": 0,
            "duration": 0.0,
        }

    def __enter__(self) -> "GraphWriteQueue":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def errors(self) -> List[str]:
        with self._stats_lock:
            return list(self._errors)

    @property
    def stats(self) -> Dict[str, Any]:
        with self._stats_lock:
            stats = dict(self._stats)
            stats["errors"] = list(self._errors)
            stats["queue_size"] = self._queue.qsize()
            return stats

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._thread.start()

    def enqueue(self, request: GraphWriteRequest) -> None:
        if self._closed:
            raise RuntimeError("GraphWriteQueue is already closed")
        if not self._started:
            self.start()
        self._queue.put(request)
        with self._stats_lock:
            self._stats["enqueued"] += 1

    def enqueue_many(self, requests: Iterable[GraphWriteRequest]) -> int:
        count = 0
        for request in requests:
            self.enqueue(request)
            count += 1
        return count

    def drain(self) -> None:
        self._queue.join()
        self._raise_errors_if_needed()

    def close(self) -> None:
        if self._closed:
            self.drain()
            return
        if not self._started:
            self.start()
        self._closed = True
        self._queue.put(self._SENTINEL)
        self._queue.join()
        self._thread.join()
        self._raise_errors_if_needed()

    def _consume(self) -> None:
        buffer: List[GraphWriteRequest] = []
        while True:
            try:
                item = self._queue.get(timeout=self.flush_interval)
            except queue.Empty:
                self._flush(buffer)
                continue

            if item is self._SENTINEL:
                self._flush(buffer)
                self._queue.task_done()
                break

            buffer.append(item)
            if len(buffer) >= self.batch_size:
                self._flush(buffer)

    def _flush(self, buffer: List[GraphWriteRequest]) -> None:
        if not buffer:
            return

        requests = list(buffer)
        started = time.perf_counter()
        try:
            for grouped_requests in self._group_compatible_requests(requests).values():
                self._write_group(grouped_requests)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            logger.error("Graph write batch failed: %s", message)
            logger.debug(traceback.format_exc())
            with self._stats_lock:
                self._errors.append(message)
        finally:
            elapsed = time.perf_counter() - started
            with self._stats_lock:
                self._stats["batches"] += 1
                self._stats["duration"] += elapsed
            for _ in requests:
                self._queue.task_done()
            buffer.clear()

    @staticmethod
    def _group_compatible_requests(
        requests: List[GraphWriteRequest],
    ) -> "OrderedDict[Tuple[str, bool, str], List[GraphWriteRequest]]":
        grouped: "OrderedDict[Tuple[str, bool, str], List[GraphWriteRequest]]" = OrderedDict()
        for request in requests:
            key = (
                request.index_update_mode,
                bool(request.generate_sparse_embedding),
                request.sparse_model_name,
            )
            grouped.setdefault(key, []).append(request)
        return grouped

    def _write_group(self, requests: List[GraphWriteRequest]) -> None:
        first = requests[0]
        units = [request.unit for request in requests]
        stats = self.semantic_system.batch_add_units(
            units=units,
            batch_size=self.embedding_batch_size,
            per_unit_space_names=[request.space_names for request in requests],
            explicit_contents_for_embedding=[request.explicit_content_for_embedding for request in requests],
            content_types_for_embedding=[request.content_type_for_embedding for request in requests],
            index_update_mode=first.index_update_mode,
            generate_sparse_embedding=first.generate_sparse_embedding,
            sparse_model_name=first.sparse_model_name,
            show_progress=self.show_progress,
        )
        with self._stats_lock:
            self._stats["groups"] += 1
            for key in (
                "added",
                "skipped",
                "embedding_generated",
                "sparse_generated",
                "graph_nodes_added",
                "graph_nodes_updated",
            ):
                self._stats[key] += int(stats.get(key, 0) or 0)

    def _raise_errors_if_needed(self) -> None:
        if self.raise_on_error and self.errors:
            raise RuntimeError("GraphWriteQueue failed: " + "; ".join(self.errors[:3]))


def dispatch_graph_write_requests(
    semantic_system: Any,
    requests: Iterable[GraphWriteRequest],
    graph_writer: Optional[GraphWriteQueue] = None,
    wait_for_completion: bool = True,
    batch_size: int = 32,
    embedding_batch_size: int = 32,
) -> Dict[str, Any]:
    """Enqueue graph writes, creating a temporary consumer when needed."""

    request_list = list(requests)
    if not request_list:
        return {"enqueued": 0, "added": 0, "skipped": 0, "errors": []}

    if graph_writer is not None:
        graph_writer.enqueue_many(request_list)
        if wait_for_completion:
            graph_writer.drain()
        return graph_writer.stats

    with GraphWriteQueue(
        semantic_system=semantic_system,
        batch_size=batch_size,
        embedding_batch_size=embedding_batch_size,
        name="GraphWriteQueueLocal",
    ) as local_writer:
        local_writer.enqueue_many(request_list)
    return local_writer.stats