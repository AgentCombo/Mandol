"""Thread-safe query feature cache for multi-backend retrieval."""

import threading
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ..utils.logging_config import create_module_logger

logger = create_module_logger("query_bundle")


class QueryBundle:
    """Cache dense, SPLADE, and BM25 query features for one query.

    QueryBundle prevents duplicate feature computation when several retrievers
    process the same query concurrently. Each feature path has its own lock and
    event so one backend can wait briefly for another backend's computation.
    """

    _NOT_STARTED = "NOT_STARTED"
    _COMPUTING = "COMPUTING"
    _DONE = "DONE"
    _WAIT_TIMEOUT_SECONDS = 1.0

    __slots__ = (
        'query_text',
        '_dense_vector', '_splade_vector', '_bm25_tokens',
        '_dense_lock', '_splade_lock', '_bm25_lock',
        '_dense_status', '_splade_status', '_bm25_status',
        '_dense_event', '_splade_event', '_bm25_event',
        '_stats',
    )

    def __init__(self, query_text: str):
        """Initialize a query feature cache.

        Args:
            query_text: Natural-language query text.
        """
        if not isinstance(query_text, str):
            raise TypeError(f"query_text must be str, got {type(query_text).__name__}")
        
        self.query_text: str = query_text
        
        
        self._dense_vector: Optional[np.ndarray] = None
        self._splade_vector: Optional[Dict[int, float]] = None
        self._bm25_tokens: Optional[List[str]] = None
        
        
        self._dense_lock = threading.Lock()
        self._splade_lock = threading.Lock()
        self._bm25_lock = threading.Lock()

        
        self._dense_status = self._NOT_STARTED
        self._splade_status = self._NOT_STARTED
        self._bm25_status = self._NOT_STARTED
        self._dense_event = threading.Event()
        self._splade_event = threading.Event()
        self._bm25_event = threading.Event()
        
        self._stats: Dict[str, Dict] = {
            'dense':  {'computed': False, 'hits': 0, 'compute_time_ms': 0.0},
            'splade': {'computed': False, 'hits': 0, 'compute_time_ms': 0.0},
            'bm25':   {'computed': False, 'hits': 0, 'compute_time_ms': 0.0},
        }

    
    # Public API: get_or_compute_*
    

    def get_or_compute_dense(
        self, compute_fn: Callable[[str], Optional[np.ndarray]]
    ) -> Optional[np.ndarray]:
        """Return or compute dense."""
        if self._dense_status == self._DONE:
            self._stats['dense']['hits'] += 1
            return self._dense_vector

        is_worker = False
        event = self._dense_event

        with self._dense_lock:
            if self._dense_status == self._DONE:
                self._stats['dense']['hits'] += 1
                return self._dense_vector

            if self._dense_status == self._COMPUTING:
                event = self._dense_event
            else:
                self._dense_status = self._COMPUTING
                self._dense_event = threading.Event()
                self._dense_event.clear()
                event = self._dense_event
                is_worker = True

        if is_worker:
            t0 = time.perf_counter()
            try:
                result = compute_fn(self.query_text)
                elapsed_ms = (time.perf_counter() - t0) * 1000

                with self._dense_lock:
                    if result is not None:
                        self._dense_vector = result
                        self._dense_status = self._DONE
                        self._stats['dense']['computed'] = True
                        self._stats['dense']['compute_time_ms'] = elapsed_ms
                    else:
                        self._dense_vector = None
                        self._dense_status = self._NOT_STARTED

                if result is not None:
                    logger.debug(
                        f"QueryBundle: dense vector computed ({elapsed_ms:.1f}ms), "
                        f"shape={result.shape}, dtype={result.dtype}"
                    )
                else:
                    logger.warning("QueryBundle: dense encoder returned None.")

                return result
            except Exception as e:
                with self._dense_lock:
                    self._dense_vector = None
                    self._dense_status = self._NOT_STARTED
                logger.error(f"QueryBundle: dense vector computation failed: {e}")
                return None
            finally:
                event.set()

        signaled = event.wait(timeout=self._WAIT_TIMEOUT_SECONDS)
        with self._dense_lock:
            if self._dense_status == self._DONE:
                self._stats['dense']['hits'] += 1
                return self._dense_vector
            if self._dense_status == self._NOT_STARTED:
                return self._dense_vector

            if not signaled:
                logger.warning("QueryBundle: timed out waiting for dense computation; returning current fallback value.")
            else:
                logger.debug("QueryBundle: dense wait returned while another computation is still running.")
            return self._dense_vector

    def get_or_compute_splade(
        self, compute_fn: Callable[[str], Dict[int, float]]
    ) -> Dict[int, float]:
        """Return or compute splade."""
        if self._splade_status == self._DONE:
            self._stats['splade']['hits'] += 1
            return self._splade_vector if self._splade_vector is not None else {}

        is_worker = False
        event = self._splade_event

        with self._splade_lock:
            if self._splade_status == self._DONE:
                self._stats['splade']['hits'] += 1
                return self._splade_vector if self._splade_vector is not None else {}

            if self._splade_status == self._COMPUTING:
                event = self._splade_event
            else:
                self._splade_status = self._COMPUTING
                self._splade_event = threading.Event()
                self._splade_event.clear()
                event = self._splade_event
                is_worker = True

        if is_worker:
            t0 = time.perf_counter()
            try:
                result = compute_fn(self.query_text)
                elapsed_ms = (time.perf_counter() - t0) * 1000

                if result is None:
                    with self._splade_lock:
                        self._splade_vector = None
                        self._splade_status = self._NOT_STARTED
                    logger.warning("QueryBundle: SPLADE encoder returned None.")
                    return {}

                cached = result if result else {}
                with self._splade_lock:
                    self._splade_vector = cached
                    self._splade_status = self._DONE
                    self._stats['splade']['computed'] = True
                    self._stats['splade']['compute_time_ms'] = elapsed_ms

                logger.debug(
                    f"QueryBundle: SPLADE vector computed ({elapsed_ms:.1f}ms), "
                    f"nonzero={len(cached)}"
                )
                return cached
            except Exception as e:
                with self._splade_lock:
                    self._splade_vector = None
                    self._splade_status = self._NOT_STARTED
                logger.error(f"QueryBundle: SPLADE vector computation failed: {e}")
                return {}
            finally:
                event.set()

        signaled = event.wait(timeout=self._WAIT_TIMEOUT_SECONDS)
        with self._splade_lock:
            if self._splade_status == self._DONE:
                self._stats['splade']['hits'] += 1
                return self._splade_vector if self._splade_vector is not None else {}
            if self._splade_status == self._NOT_STARTED:
                return self._splade_vector if self._splade_vector is not None else {}

            if not signaled:
                logger.warning("QueryBundle: timed out waiting for SPLADE computation; returning current fallback value.")
            else:
                logger.debug("QueryBundle: SPLADE wait returned while another computation is still running.")
            return self._splade_vector if self._splade_vector is not None else {}

    def get_or_compute_bm25_tokens(
        self, compute_fn: Callable[[str], List[str]]
    ) -> List[str]:
        """Return or compute bm25 tokens."""
        if self._bm25_status == self._DONE:
            self._stats['bm25']['hits'] += 1
            return self._bm25_tokens if self._bm25_tokens is not None else []

        is_worker = False
        event = self._bm25_event

        with self._bm25_lock:
            if self._bm25_status == self._DONE:
                self._stats['bm25']['hits'] += 1
                return self._bm25_tokens if self._bm25_tokens is not None else []

            if self._bm25_status == self._COMPUTING:
                event = self._bm25_event
            else:
                self._bm25_status = self._COMPUTING
                self._bm25_event = threading.Event()
                self._bm25_event.clear()
                event = self._bm25_event
                is_worker = True

        if is_worker:
            t0 = time.perf_counter()
            try:
                result = compute_fn(self.query_text)
                elapsed_ms = (time.perf_counter() - t0) * 1000

                if result is None:
                    with self._bm25_lock:
                        self._bm25_tokens = None
                        self._bm25_status = self._NOT_STARTED
                    logger.warning("QueryBundle: BM25 tokenizer returned None.")
                    return []

                cached = result if result else []
                with self._bm25_lock:
                    self._bm25_tokens = cached
                    self._bm25_status = self._DONE
                    self._stats['bm25']['computed'] = True
                    self._stats['bm25']['compute_time_ms'] = elapsed_ms

                logger.debug(
                    f"QueryBundle: BM25 tokenization completed ({elapsed_ms:.1f}ms), "
                    f"tokens={len(cached)}"
                )
                return cached
            except Exception as e:
                with self._bm25_lock:
                    self._bm25_tokens = None
                    self._bm25_status = self._NOT_STARTED
                logger.error(f"QueryBundle: BM25 tokenization failed: {e}")
                return []
            finally:
                event.set()

        signaled = event.wait(timeout=self._WAIT_TIMEOUT_SECONDS)
        with self._bm25_lock:
            if self._bm25_status == self._DONE:
                self._stats['bm25']['hits'] += 1
                return self._bm25_tokens if self._bm25_tokens is not None else []
            if self._bm25_status == self._NOT_STARTED:
                return self._bm25_tokens if self._bm25_tokens is not None else []

            if not signaled:
                logger.warning("QueryBundle: timed out waiting for BM25 tokenization; returning current fallback value.")
            else:
                logger.debug("QueryBundle: BM25 wait returned while another computation is still running.")
            return self._bm25_tokens if self._bm25_tokens is not None else []

    
    

    @property
    def has_dense(self) -> bool:
        """Return whether dense is available."""
        return self._dense_vector is not None

    @property
    def has_splade(self) -> bool:
        """Return whether splade is available."""
        return self._splade_vector is not None

    @property
    def has_bm25_tokens(self) -> bool:
        """Return whether bm25 tokens is available."""
        return self._bm25_tokens is not None

    @property
    def cached_types(self) -> List[str]:
        """Run cached types."""
        types = []
        if self.has_dense:
            types.append('dense')
        if self.has_splade:
            types.append('splade')
        if self.has_bm25_tokens:
            types.append('bm25')
        return types

    def get_stats(self) -> Dict[str, Any]:
        """Return stats."""
        total_compute_ms = sum(s['compute_time_ms'] for s in self._stats.values())
        total_hits = sum(s['hits'] for s in self._stats.values())
        
        return {
            'query_preview': (
                self.query_text[:80] + '...' 
                if len(self.query_text) > 80 
                else self.query_text
            ),
            'vectors': {k: dict(v) for k, v in self._stats.items()},
            'cached_types': self.cached_types,
            'total_compute_time_ms': round(total_compute_ms, 2),
            'total_cache_hits': total_hits,
        }

    
    

    def __repr__(self) -> str:
        cached = self.cached_types
        hits = sum(s['hits'] for s in self._stats.values())
        return (
            f"QueryBundle(query='{self.query_text[:50]}{'...' if len(self.query_text) > 50 else ''}', "
            f"cached=[{', '.join(cached) or 'none'}], hits={hits})"
        )

    def __str__(self) -> str:
        """Return the string representation."""
        return self.query_text

    def __hash__(self) -> int:
        """Return the bundle hash."""
        return hash(self.query_text)

    def __eq__(self, other) -> bool:
        if isinstance(other, QueryBundle):
            return self.query_text == other.query_text
        if isinstance(other, str):
            return self.query_text == other
        return NotImplemented
