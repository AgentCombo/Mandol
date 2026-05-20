"""Lightweight memory-system monitor for runtime diagnostics.

Provides a compact one-line status string and a structured dict of
metrics.  Uses ``psutil`` for process RSS when available; falls back
to ``tracemalloc`` from the standard library otherwise.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from ..application.memory_system import MemorySystem

logger = logging.getLogger(__name__)


class MemoryMonitor:
    """Lightweight runtime monitor for a :class:`MemorySystem`.

    Produces a compact one-line status string and a structured metrics
    dict.  Uses ``psutil`` for process RSS when available; falls back
    to ``tracemalloc`` from the standard library otherwise.

    Args:
        system_ref: Reference to the :class:`MemorySystem` to monitor.
    """

    def __init__(self, system_ref: "MemorySystem") -> None:
        self._sys = system_ref
        self._using_psutil = False
        self._try_init()

    def _try_init(self) -> None:
        """Attempt to import psutil; fall back to tracemalloc on failure."""
        try:
            import psutil  # noqa: F401
            self._using_psutil = True
        except ImportError:
            logger.debug("psutil not installed, falling back to tracemalloc for memory measurement")

    def _measure_rss_mb(self) -> float:
        """Return process RSS in MiB via psutil, or tracemalloc as fallback."""
        if self._using_psutil:
            try:
                import psutil
                return psutil.Process().memory_info().rss / (1024 * 1024)
            except (ImportError, AttributeError, RuntimeError) as exc:
                logger.debug("psutil RSS measurement failed: %s", exc)
        return self._measure_tracemalloc_mb()

    @staticmethod
    def _measure_tracemalloc_mb() -> float:
        """Return current traced memory in MiB via tracemalloc."""
        try:
            import tracemalloc
            if not tracemalloc.is_tracing():
                tracemalloc.start()
            current, _peak = tracemalloc.get_traced_memory()
            return current / (1024 * 1024)
        except (ValueError, RuntimeError, ImportError) as exc:
            logger.debug("tracemalloc measurement failed: %s", exc)
            return 0.0

    def status_line(self) -> str:
        """Return a compact one-line status string summarising system state."""
        try:
            return self._build_status_line()
        except (RuntimeError, TypeError, AttributeError, ValueError) as e:
            return f"[MemSys] monitor error: {e}"

    def _build_status_line(self) -> str:
        """Assemble the one-line status from subsystem stats."""
        sys = self._sys
        store = sys.semantic_map.get_store()
        graph_store = sys.graph.get_graph_store()

        total_units = len(store.list_units())
        total_spaces = len(store.list_spaces())

        try:
            g = graph_store._g
            n_nodes = g.number_of_nodes()
            n_edges = g.number_of_edges()
        except (AttributeError, RuntimeError, KeyError) as exc:
            logger.debug("Failed to read graph stats: %s", exc)
            n_nodes = 0
            n_edges = 0

        abi_stats = sys._abi.get_stats()
        promoted = abi_stats["space_faiss_total_vectors"]
        unpromoted = abi_stats["unpromoted_vector_count"]

        with sys._pending_lock:
            pend_u = len(sys._pending_units)
            pend_e = len(sys._pending_events)
            pend_et = len(sys._pending_entities)

        sessions = sys._session_manager.get_sessions()
        n_sess = len(sessions)
        avg_sess = sum(s.unit_count for s in sessions) / max(n_sess, 1)

        rss_mb = self._measure_rss_mb()
        mem_tag = "" if self._using_psutil else "(tracemalloc)"
        dirty = "DIRTY" if sys.dirty else "CLEAN"

        return (
            f"[MemSys] units={total_units} | spaces={total_spaces} | "
            f"graph:{n_nodes}n/{n_edges}e | "
            f"idx:{promoted}\u2191/{unpromoted}\u2193 | "
            f"pend:{pend_u}u/{pend_e}e/{pend_et}et | "
            f"sess:{n_sess}(avg{avg_sess:.0f}) | "
            f"mem:{rss_mb:.1f}MB{mem_tag} | "
            f"{dirty}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return all monitoring metrics as a dict for programmatic access."""
        sys = self._sys
        store = sys.semantic_map.get_store()
        graph_store = sys.graph.get_graph_store()

        try:
            g = graph_store._g
            n_nodes = g.number_of_nodes()
            n_edges = g.number_of_edges()
        except (AttributeError, RuntimeError, KeyError) as exc:
            logger.debug("Failed to read graph stats: %s", exc)
            n_nodes = 0
            n_edges = 0

        abi_stats = sys._abi.get_stats()

        with sys._pending_lock:
            pend_u = len(sys._pending_units)
            pend_e = len(sys._pending_events)
            pend_et = len(sys._pending_entities)

        sessions = sys._session_manager.get_sessions()
        n_sess = len(sessions)
        avg_sess = sum(s.unit_count for s in sessions) / max(n_sess, 1)

        total_units = len(store.list_units())

        return {
            "total_units": total_units,
            "total_spaces": len(store.list_spaces()),
            "graph_nodes": n_nodes,
            "graph_edges": n_edges,
            "vector_index_global": abi_stats["global_faiss_size"],
            "vector_index_promoted": abi_stats["space_faiss_total_vectors"],
            "vector_index_unpromoted": abi_stats["unpromoted_vector_count"],
            "pending_units": pend_u,
            "pending_events": pend_e,
            "pending_entities": pend_et,
            "total_sessions": n_sess,
            "avg_session_size": round(avg_sess, 1),
            "rss_memory_mb": round(self._measure_rss_mb(), 2),
            "memory_source": "psutil" if self._using_psutil else "tracemalloc",
            "dirty": sys.dirty,
            "persistence_enabled": sys.persistence is not None,
            "llm_model": sys._cfg.llm_model,
            "embedder_model": sys._cfg.embedder_model,
            "embedder_dim": sys._cfg.embedder_dim,
            "use_unified_pipeline": sys._cfg.use_unified_pipeline,
        }

    def __repr__(self) -> str:
        return self.status_line()

    def __str__(self) -> str:
        return self.status_line()
