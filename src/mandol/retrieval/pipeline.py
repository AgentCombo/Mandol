"""Main hybrid retrieval pipeline orchestrator.

Combines dense (ANN vector), BM25 (keyword), and sparse (TF-IDF) retrieval
with RRF fusion, BFS graph expansion, and Cross-Encoder reranking into a
single search interface.
"""

from __future__ import annotations

import logging
import math
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ..application.semantic_graph import SemanticGraphService
from ..domain.memory_unit import MemoryUnit
from ..domain.types import Uid
from ..ports.bm25_index import BM25Index
from ..ports.reranker import Reranker
from ..ports.sparse_index import SparseIndex
from .fusion import RankedUnit, rrf_fusion
from .types import SearchHit

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class HybridRetrieverConfig:
    """Configuration for the hybrid retrieval pipeline.

    Attributes:
        per_method_k: Top-K results from each method before fusion (default 60).
        rrf_k: RRF smoothing constant (default 60).
        bfs_per_seed: Units to collect per seed in BFS expansion (default 3).
        bfs_hops: BFS depth (default 1).
        recall_top_k_multiplier: Multiplier for recall_k in rerank pre-recall.
        parallel_search: Run dense/BM25/sparse in parallel (default True).
        max_workers: Thread pool size for parallel search (default 3).
    """
    per_method_k: int = 60
    rrf_k: int = 60
    bfs_per_seed: int = 3
    bfs_hops: int = 1
    recall_top_k_multiplier: int = 5
    parallel_search: bool = True
    max_workers: int = 3


@dataclass
class IndexedSearchResults:
    """Container for per-method retrieval results.

    Attributes:
        dense: Dense ANN search results.
        bm25: BM25 keyword search results.
        sparse: TF-IDF sparse search results.
    """
    dense: List[Tuple[MemoryUnit, float]] = field(default_factory=list)
    bm25: List[Tuple[MemoryUnit, float]] = field(default_factory=list)
    sparse: List[Tuple[MemoryUnit, float]] = field(default_factory=list)


class HybridRetriever:
    """Orchestrates dense + BM25 + sparse retrieval with fusion.

    Flow: Dense + BM25 + Sparse parallel recall → RRF fusion → BFS expansion
    → Cross-Encoder rerank → SearchHit results with per-method scores.

    Args:
        graph: SemanticGraphService for ANN search and BFS expansion.
        bm25_index: Optional BM25Index for indexed keyword search.
        sparse_index: Optional SparseIndex for indexed sparse search.
        reranker: Optional Reranker for Cross-Encoder reranking.
        config: HybridRetrieverConfig with fusion/expansion settings.
        text_extractor: Optional tokenizer callable for query tokenization.
    """

    def __init__(
        self,
        *,
        graph: SemanticGraphService,
        bm25_index: Optional[BM25Index] = None,
        sparse_index: Optional[SparseIndex] = None,
        reranker: Optional[Reranker] = None,
        config: Optional[HybridRetrieverConfig] = None,
        text_extractor: Optional[callable] = None,
    ) -> None:
        self._graph = graph
        self._bm25_index = bm25_index
        self._sparse_index = sparse_index
        self._reranker = reranker
        self._config = config or HybridRetrieverConfig()
        self._text_extractor = text_extractor

    @property
    def semantic_graph(self) -> SemanticGraphService:
        """The underlying graph service (explicit edges + semantic map)."""
        return self._graph

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        space_names: Optional[Sequence[str]] = None,
        recursive: bool = True,
        use_rerank: bool = True,
    ) -> List[SearchHit]:
        """Execute the full hybrid retrieval pipeline.

        Args:
            query: Natural language search query.
            top_k: Max results to return (default 10).
            space_names: Optional space filter (union mode).
            recursive: Whether to include child spaces.
            use_rerank: Enable Cross-Encoder reranking (default True).

        Returns:
            Ranked list of SearchHit objects with per-method scores.
        """
        if not isinstance(query, str) or not query.strip():
            return []

        t0 = time.time()
        candidates = self._candidates(space_names=space_names, recursive=recursive)
        if not candidates:
            return []

        pmk = int(self._config.per_method_k) if self._config.per_method_k > 0 else max(max(1, int(top_k)) * 3, 20)

        if self._config.parallel_search:
            indexed_results = self._parallel_search(query, pmk, space_names, recursive)
            dense_ranked = [RankedUnit(unit=u, score=float(s)) for (u, s) in indexed_results.dense]
            bm25_ranked = [RankedUnit(unit=u, score=float(s)) for (u, s) in indexed_results.bm25]
            sparse_ranked = [RankedUnit(unit=u, score=float(s)) for (u, s) in indexed_results.sparse]
        else:
            dense_results = self._graph.semantic_map.search_by_text(
                query,
                top_k=pmk,
                space_names=space_names,
                recursive=recursive,
            )
            dense_ranked = [RankedUnit(unit=u, score=float(s)) for (u, s) in dense_results]

            bm25_ranked = []
            sparse_ranked = []

            if self._bm25_index is not None:
                q_tokens = self._tokenize(query)
                bm25_hits = self._bm25_index.search(q_tokens, pmk)
                uid_to_unit = {str(u.uid): u for u in candidates}
                bm25_ranked = [
                    RankedUnit(unit=uid_to_unit[str(uid)], score=float(score))
                    for uid, score in bm25_hits
                    if str(uid) in uid_to_unit
                ]
            else:
                bm25_ranked = self._fallback_bm25_search(query, candidates, pmk)

            if self._sparse_index is not None:
                q_vec = self._compute_sparse_vector(query)
                sparse_hits = self._sparse_index.search(q_vec, pmk)
                uid_to_unit = {str(u.uid): u for u in candidates}
                sparse_ranked = [
                    RankedUnit(unit=uid_to_unit[str(uid)], score=float(score))
                    for uid, score in sparse_hits
                    if str(uid) in uid_to_unit
                ]
            else:
                sparse_ranked = self._fallback_sparse_search(query, candidates, pmk)

        logger.debug(
            "Hybrid search recall: dense=%d bm25=%d sparse=%d (candidates=%d, pmk=%d)",
            len(dense_ranked), len(bm25_ranked), len(sparse_ranked),
            len(candidates), pmk,
        )

        fused = rrf_fusion(
            [dense_ranked, bm25_ranked, sparse_ranked],
            k=int(self._config.rrf_k),
            top_k=pmk,
            method_names=["dense", "bm25", "sparse"],
        )

        dense_results_for_fallback = self._graph.semantic_map.search_by_text(
            query,
            top_k=pmk,
            space_names=space_names,
            recursive=recursive,
        ) if not self._config.parallel_search else indexed_results.dense

        if not fused:
            fused = [(u, 0.0, {}) for (u, _s) in dense_results_for_fallback]

        uid_to_scores: Dict[str, Dict[str, float]] = {}
        uid_to_unit: Dict[str, MemoryUnit] = {}

        def add_method_scores(method: str, ranked: List[RankedUnit]) -> None:
            for item in ranked:
                uid = str(item.unit.uid)
                uid_to_unit[uid] = item.unit
                scores = uid_to_scores.get(uid)
                if scores is None:
                    scores = {}
                    uid_to_scores[uid] = scores
                scores[method] = float(item.score)

        add_method_scores("dense", dense_ranked)
        add_method_scores("bm25", bm25_ranked)
        add_method_scores("sparse", sparse_ranked)

        fused_units: List[MemoryUnit] = []
        fused_rrf: Dict[str, float] = {}
        fused_ranks: Dict[str, Dict[str, int]] = {}
        for u, s, ranks in fused:
            uid = str(u.uid)
            uid_to_unit[uid] = u
            fused_units.append(u)
            fused_rrf[uid] = float(s)
            fused_ranks[uid] = dict(ranks)

        seeds = fused_units[: max(1, int(top_k))]
        expanded = self._graph.bfs_expand_units(seeds, per_seed=int(self._config.bfs_per_seed), hops=int(self._config.bfs_hops))

        expanded_uid_order: List[str] = []
        expanded_units: List[MemoryUnit] = []
        seen = set()
        for u in fused_units + expanded:
            uid = str(u.uid)
            if uid in seen:
                continue
            seen.add(uid)
            expanded_uid_order.append(uid)
            expanded_units.append(u)

        for uid in expanded_uid_order:
            scores = uid_to_scores.setdefault(uid, {})
            if uid in fused_rrf:
                scores["rrf"] = fused_rrf[uid]
            else:
                scores.setdefault("rrf", 0.0)
                scores["bfs_added"] = 1.0

        if use_rerank and self._reranker is not None:
            reranked = self._reranker.rerank(query, expanded_units, top_k=max(1, int(top_k)))
            out: List[SearchHit] = []
            for unit, rerank_score in reranked:
                uid = str(unit.uid)
                scores = dict(uid_to_scores.get(uid, {}))
                scores["rerank"] = float(rerank_score)
                ranks = fused_ranks.get(uid, {})
                out.append(
                    SearchHit(
                        unit=unit,
                        final_score=float(rerank_score),
                        scores=scores,
                        ranks=dict(ranks),
                    )
                )
            logger.debug(
                "Hybrid search complete: %d fused + %d expanded → %d reranked "
                "for '%.60s' in %.1fs",
                len(fused_units), len(expanded), len(out), query, time.time() - t0,
            )
            return out

        hits: List[SearchHit] = []
        for uid in expanded_uid_order:
            if uid not in uid_to_unit:
                u = self._graph.semantic_map.get_unit(uid)
                if u is not None:
                    uid_to_unit[uid] = u
            unit = uid_to_unit.get(uid)
            if unit is None:
                continue
            scores = dict(uid_to_scores.get(uid, {}))
            scores.setdefault("rrf", fused_rrf.get(uid, 0.0))
            ranks = fused_ranks.get(uid, {})
            hits.append(SearchHit(unit=unit, final_score=float(scores.get("rrf", 0.0)), scores=scores, ranks=dict(ranks)))

        hits.sort(key=lambda h: h.final_score, reverse=True)
        logger.debug(
            "Hybrid search complete: %d fused + %d expanded → %d hits "
            "for '%.60s' in %.1fs",
            len(fused_units), len(expanded), len(hits[:top_k]), query, time.time() - t0,
        )
        return hits[: max(0, int(top_k))]

    def _parallel_search(
        self,
        query: str,
        top_k: int,
        space_names: Optional[Sequence[str]],
        recursive: bool,
    ) -> IndexedSearchResults:
        results = IndexedSearchResults()

        def search_dense() -> List[Tuple[MemoryUnit, float]]:
            return self._graph.semantic_map.search_by_text(
                query,
                top_k=top_k,
                space_names=space_names,
                recursive=recursive,
            )

        def search_bm25() -> List[Tuple[MemoryUnit, float]]:
            if self._bm25_index is None:
                candidates = self._candidates(space_names=space_names, recursive=recursive)
                fallback = self._fallback_bm25_search(query, candidates, top_k)
                return [(r.unit, r.score) for r in fallback]
            q_tokens = self._tokenize(query)
            space_name = space_names[0] if space_names and len(space_names) == 1 else None
            if space_name:
                hits = self._bm25_index.search_in_space(q_tokens, space_name, top_k)
            else:
                hits = self._bm25_index.search(q_tokens, top_k)
            candidates = self._candidates(space_names=space_names, recursive=recursive)
            uid_to_unit = {str(u.uid): u for u in candidates}
            return [
                (uid_to_unit[str(uid)], float(score))
                for uid, score in hits
                if str(uid) in uid_to_unit
            ]

        def search_sparse() -> List[Tuple[MemoryUnit, float]]:
            if self._sparse_index is None:
                candidates = self._candidates(space_names=space_names, recursive=recursive)
                fallback = self._fallback_sparse_search(query, candidates, top_k)
                return [(r.unit, r.score) for r in fallback]
            q_vec = self._compute_sparse_vector(query)
            space_name = space_names[0] if space_names and len(space_names) == 1 else None
            if space_name:
                hits = self._sparse_index.search_in_space(q_vec, space_name, top_k)
            else:
                hits = self._sparse_index.search(q_vec, top_k)
            candidates = self._candidates(space_names=space_names, recursive=recursive)
            uid_to_unit = {str(u.uid): u for u in candidates}
            return [
                (uid_to_unit[str(uid)], float(score))
                for uid, score in hits
                if str(uid) in uid_to_unit
            ]

        with ThreadPoolExecutor(max_workers=self._config.max_workers) as executor:
            future_to_method = {
                executor.submit(search_dense): "dense",
                executor.submit(search_bm25): "bm25",
                executor.submit(search_sparse): "sparse",
            }

            for future in as_completed(future_to_method):
                method = future_to_method[future]
                try:
                    method_results = future.result()
                    if method == "dense":
                        results.dense = method_results
                    elif method == "bm25":
                        results.bm25 = method_results
                    elif method == "sparse":
                        results.sparse = method_results
                except (RuntimeError, ValueError, OSError):
                    if method == "dense":
                        results.dense = []
                    elif method == "bm25":
                        results.bm25 = []
                    elif method == "sparse":
                        results.sparse = []

        return results

    def _candidates(self, *, space_names: Optional[Sequence[str]], recursive: bool) -> List[MemoryUnit]:
        if space_names:
            return self._graph.get_units_in_spaces(space_names, mode="union", recursive=recursive)
        return self._graph.semantic_map.list_units()

    def _tokenize(self, text: str) -> List[str]:
        if self._text_extractor is not None:
            return self._text_extractor(text)
        import re
        text = text.strip().lower()
        if not text:
            return []
        tokens = re.findall(r"\w+", text, flags=re.UNICODE)
        return [w for w in tokens if len(w) > 1]

    def _compute_sparse_vector(self, text: str) -> Dict[str, float]:
        tokens = self._tokenize(text)
        if not tokens:
            return {}
        tf = Counter(tokens)
        N = 1.0
        idf: Dict[str, float] = {}
        for term, count in tf.items():
            idf[term] = math.log((N + 1.0) / (float(count) + 1.0)) + 1.0
        q_vec = {t: float(c) * idf[t] for t, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in q_vec.values()))
        if norm > 0:
            q_vec = {t: v / norm for t, v in q_vec.items()}
        return q_vec

    def _fallback_bm25_search(
        self,
        query: str,
        candidates: Sequence[MemoryUnit],
        top_k: int,
    ) -> List[RankedUnit]:
        tokens = self._tokenize(query)
        if not tokens:
            return []
        docs_tokens: List[List[str]] = []
        docs_lens: List[int] = []
        tf_list: List[Counter] = []
        df: Counter = Counter()
        for u in candidates:
            txt = self._extract_text(u)
            toks = self._tokenize(txt)
            docs_tokens.append(toks)
            docs_lens.append(len(toks))
            tf = Counter(toks)
            tf_list.append(tf)
            for term in set(toks):
                df[term] += 1

        if not docs_tokens:
            return []

        N = len(docs_tokens)
        avgdl = sum(docs_lens) / max(1.0, float(len(docs_lens)))
        k1, b = 1.5, 0.75

        def idf_func(term: str) -> float:
            n_q = df.get(term, 0)
            return math.log((N - n_q + 0.5) / (n_q + 0.5) + 1.0)

        scores: List[Tuple[MemoryUnit, float]] = []
        for idx, tf in enumerate(tf_list):
            dl = float(docs_lens[idx])
            s = 0.0
            for term in tokens:
                if term not in tf:
                    continue
                term_idf = idf_func(term)
                f = float(tf[term])
                denom = f + k1 * (1.0 - b + b * dl / (avgdl + 1e-12))
                s += term_idf * (f * (k1 + 1.0)) / (denom + 1e-12)
            if s > 0:
                scores.append((candidates[idx], float(s)))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [RankedUnit(unit=u, score=s) for u, s in scores[:top_k]]

    def _fallback_sparse_search(
        self,
        query: str,
        candidates: Sequence[MemoryUnit],
        top_k: int,
    ) -> List[RankedUnit]:
        q_vec = self._compute_sparse_vector(query)
        if not q_vec:
            return []

        q_norm = math.sqrt(sum(v * v for v in q_vec.values()))
        if q_norm <= 0:
            return []

        docs: List[Dict[str, float]] = []
        norms: List[float] = []
        for u in candidates:
            txt = self._extract_text(u)
            tokens = self._tokenize(txt)
            if not tokens:
                docs.append({})
                norms.append(0.0)
                continue
            tf = Counter(tokens)
            d_vec: Dict[str, float] = {}
            for t, c in tf.items():
                d_vec[t] = float(c)
            norm = math.sqrt(sum(v * v for v in d_vec.values()))
            norms.append(norm)
            docs.append(d_vec)

        results: List[Tuple[MemoryUnit, float]] = []
        for u, d_vec, d_norm in zip(candidates, docs, norms):
            if d_norm <= 0:
                continue
            dot = sum(q_vec.get(t, 0.0) * v for t, v in d_vec.items())
            score = dot / (q_norm * d_norm + 1e-12)
            if score > 0:
                results.append((u, float(score)))

        results.sort(key=lambda x: x[1], reverse=True)
        return [RankedUnit(unit=u, score=s) for u, s in results[:top_k]]

    def _extract_text(self, unit: MemoryUnit) -> str:
        raw = unit.raw_data or {}
        for key in ["text_content", "text", "content", "summary", "title", "message"]:
            val = raw.get(key)
            if isinstance(val, str) and val.strip():
                return val
        for k, v in raw.items():
            if isinstance(v, str) and v.strip():
                return v
        return ""

    def index_units(self, units: Sequence[MemoryUnit], space_names: Optional[Sequence[str]] = None) -> None:
        """Index units into BM25 and/or sparse persistent indices.

        If space_names are provided, units are also indexed per-space.

        Args:
            units: MemoryUnits to index.
            space_names: Optional space names for space-scoped indexing.
        """
        if self._bm25_index is not None:
            bm25_items: List[Tuple[Uid, List[str]]] = []
            for u in units:
                txt = self._extract_text(u)
                tokens = self._tokenize(txt)
                if tokens:
                    bm25_items.append((Uid(str(u.uid)), tokens))
            if bm25_items:
                self._bm25_index.upsert(bm25_items)
                if space_names:
                    for sp in space_names:
                        self._bm25_index.upsert_to_space(bm25_items, sp)

        if self._sparse_index is not None:
            sparse_items: List[Tuple[Uid, Dict[str, float]]] = []
            for u in units:
                txt = self._extract_text(u)
                tokens = self._tokenize(txt)
                if tokens:
                    tf = Counter(tokens)
                    N = 1.0
                    idf: Dict[str, float] = {}
                    for term, count in tf.items():
                        idf[term] = math.log((N + 1.0) / (float(count) + 1.0)) + 1.0
                    vec = {t: float(c) * idf[t] for t, c in tf.items()}
                    norm = math.sqrt(sum(v * v for v in vec.values()))
                    if norm > 0:
                        vec = {t: v / norm for t, v in vec.items()}
                    sparse_items.append((Uid(str(u.uid)), vec))
            if sparse_items:
                self._sparse_index.upsert(sparse_items)
                if space_names:
                    for sp in space_names:
                        self._sparse_index.upsert_to_space(sparse_items, sp)
