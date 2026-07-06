"""Score fusion and reranking utilities for Mandol retrieval."""

import logging
import traceback
import heapq
import numpy as np
from typing import Callable, Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict

from .retrieval_interface import RetrievalMethod, RetrievalResult
from .query_bundle import QueryBundle
from .rerank_manager import QWEN_AVAILABLE, RerankerManager
from .retrieval_utils import MultiRetrievalResults
from ..core.memory_unit import MemoryUnit
from ..utils.config_manager import settings
from ..utils.logging_config import create_module_logger

logger = create_module_logger("score_fusion")


class ScoreFusion:
    """Combine results from multiple retrieval backends and optional rerankers."""

    _reranker_manager: Optional[RerankerManager] = None

    @classmethod
    def get_reranker_manager(cls) -> RerankerManager:
        """Return reranker manager."""
        if cls._reranker_manager is None:
            cls._reranker_manager = RerankerManager()
            logger.info("Reranker manager created lazily.")
        return cls._reranker_manager
    
    @classmethod
    def cleanup_rerankers(cls):
        """Release associated resources."""
        if cls._reranker_manager:
            cls._reranker_manager.clear_cache()
            logger.info("Reranker cache cleared.")

    @staticmethod
    def _uses_local_neural_reranker(rerank_method: Optional[str]) -> bool:
        if not rerank_method:
            return False
        method = str(rerank_method).strip().lower()
        if method in {"", "none", "mmr"}:
            return False
        remote_markers = ("dashscope", "sili", "siliconflow", "gte", "cstcloud", "remote")
        if any(marker in method for marker in remote_markers):
            return False
        return any(marker in method for marker in ("baai", "bge", "qwen", "jina"))

    @classmethod
    def ensure_sync_rerank_allowed(cls, rerank_method: Optional[str], context: str = "sync retrieval") -> None:
        backend = str(getattr(settings, "reranker_backend", "native") or "native").lower()
        if backend == "vllm" and cls._uses_local_neural_reranker(rerank_method):
            raise RuntimeError(
                f"RERANKER_BACKEND=vllm requires the async retrieval path for rerank_method={rerank_method!r} "
                f"in {context}. Use smart_search_async()/search_async() instead of the sync API."
            )

    @staticmethod
    def is_vllm_sync_rerank_error(exc: BaseException) -> bool:
        message = str(exc).lower()
        return "vllm" in message and (
            "sync" in message or "synchronous" in message or "同步" in message
        )

    @staticmethod
    def _apply_threshold(
        results: List[Tuple[MemoryUnit, float]], 
        threshold: float, 
        top_k: int
    ) -> List[Tuple[MemoryUnit, float]]:
        """Apply threshold."""
        if not results:
            return []
        
        if threshold is not None and threshold > 0:
            original_count = len(results)
            filtered_results = [(unit, score) for unit, score in results if score >= threshold]
            filtered_count = len(filtered_results)
            
            if filtered_count < original_count:
                logger.debug(f"Threshold filter: threshold={threshold:.4f}, "
                             f"original={original_count}, kept={filtered_count}, "
                             f"removed={original_count - filtered_count}")
            
            
            return filtered_results[:top_k]
        else:
            
            return results[:top_k]

    @staticmethod
    def average_fusion(multi_results: MultiRetrievalResults) -> List[Tuple[MemoryUnit, float]]:
        """Average fusion."""
        methods_used = multi_results.get_methods_used()
        all_uids = multi_results.get_union_units()
        
        if not methods_used or not all_uids:
            return []
        
        results = []
        for uid in all_uids:
            scores = []
            for method in methods_used:
                score = multi_results.get_score_for_unit(uid, method)
                if score is not None:
                    scores.append(score)
            
            if scores:
                avg_score = sum(scores) / len(scores)
                unit = multi_results.all_units[uid]
                results.append((unit, avg_score))
        
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    @staticmethod
    def rrf_fusion(multi_results: MultiRetrievalResults, 
                   k: int = 60,
                   top_k: Optional[int] = None) -> List[Tuple[MemoryUnit, float]]:
        """Run RRF fusion."""
        methods_used = multi_results.get_methods_used()
        all_uids = multi_results.get_union_units()
        
        if not methods_used or not all_uids:
            return []
        
        
        rrf_scores = defaultdict(float)
        
        for method in methods_used:
            method_results = multi_results.results_by_method[method]
            for rank, result in enumerate(method_results, 1):
                rrf_scores[result.unit.uid] += 1.0 / (k + rank)
                
            logger.debug(f"RRF: {method.value} contributed {len(method_results)} results.")

        if top_k is not None:
            real_k = min(max(0, int(top_k)), len(all_uids))
            if real_k == 0:
                return []
            enumerated_results = [
                (ordinal, multi_results.all_units[uid], rrf_scores[uid])
                for ordinal, uid in enumerate(all_uids)
            ]
            top_results = heapq.nlargest(
                real_k,
                enumerated_results,
                key=lambda item: (item[2], -item[0]),
            )
            final_results = [(unit, score) for _, unit, score in top_results]
            logger.info(
                f"RRF fusion completed: methods={len(methods_used)}, "
                f"units={len(all_uids)}, returned={len(final_results)}"
            )
            return final_results
        
        final_results = []
        for uid in all_uids:
            unit = multi_results.all_units[uid]
            final_score = rrf_scores[uid]
            final_results.append((unit, final_score))
        
        
        final_results.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"RRF fusion completed: methods={len(methods_used)}, units={len(final_results)}")
        return final_results

    @staticmethod
    def weighted_fusion(multi_results: MultiRetrievalResults, 
                       weights: Dict[RetrievalMethod, float],
                       normalization: str = "min_max") -> List[Tuple[MemoryUnit, float]]:
        """Weighted fusion."""
        methods_used = multi_results.get_methods_used()
        all_uids = multi_results.get_union_units()
        
        if not methods_used or not all_uids:
            return []
        
        method_scores = {}
        for method in methods_used:
            scores = []
            for uid in all_uids:
                score = multi_results.get_score_for_unit(uid, method)
                scores.append(score if score is not None else 0.0)
            method_scores[method] = np.array(scores)
        
        normalized_scores = {}
        for method, scores in method_scores.items():
            if normalization == "min_max":
                if scores.max() > scores.min():
                    normalized = (scores - scores.min()) / (scores.max() - scores.min())
                else:
                    normalized = scores
            elif normalization == "z_score":
                if scores.std() > 0:
                    normalized = (scores - scores.mean()) / scores.std()
                    normalized = (normalized - normalized.min()) / (normalized.max() - normalized.min() + 1e-8)
                else:
                    normalized = scores
            else:
                normalized = scores
            
            normalized_scores[method] = normalized
        
        final_scores = np.zeros(len(all_uids))
        total_weight = 0.0
        
        for method in methods_used:
            weight = weights.get(method, 0.0)
            if weight > 0:
                final_scores += weight * normalized_scores[method]
                total_weight += weight
                logger.debug(f"Weighted fusion: {method.value} weight={weight}")
        
        if total_weight > 0:
            final_scores /= total_weight
        
        results = []
        for i, uid in enumerate(all_uids):
            unit = multi_results.all_units[uid]
            results.append((unit, float(final_scores[i])))
        
        
        results.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"Weighted fusion completed: active_weights={len([w for w in weights.values() if w > 0])}, "
                    f"units={len(results)}")
        return results
    
    @staticmethod
    def mmr_fusion(multi_results: MultiRetrievalResults, 
                   query: Union[str, QueryBundle, np.ndarray],
                   lambda_param: float = 0.5, 
                   top_k: int = 10,
                   embedding_field: str = "embedding",
                   dense_compute_fn: Optional[Callable[[str], Optional[np.ndarray]]] = None) -> List[Tuple[MemoryUnit, float]]:
        """Run MMR fusion."""
        methods_used = multi_results.get_methods_used()
        all_uids = multi_results.get_union_units()
        
        if not methods_used or not all_uids:
            return []
        
        candidate_results = ScoreFusion.rrf_fusion(multi_results)

        # candidate_results = multi_results
        
        if not candidate_results:
            return []
        
        
        candidates_with_embeddings = []
        query_embedding = None
        
        for unit, score in candidate_results:
            if hasattr(unit, embedding_field) and getattr(unit, embedding_field) is not None:
                embedding = getattr(unit, embedding_field)
                candidates_with_embeddings.append((unit, score, np.array(embedding)))
            elif unit.embedding is not None:
                candidates_with_embeddings.append((unit, score, np.array(unit.embedding)))
        
        if not candidates_with_embeddings:
            logger.warning("No candidate embeddings are available; returning original RRF results.")
            return candidate_results[:top_k]
        
        try:
            if isinstance(query, QueryBundle):
                compute_fn = dense_compute_fn if dense_compute_fn is not None else (lambda _query_text: None)
                query_embedding = query.get_or_compute_dense(compute_fn)
            elif isinstance(query, np.ndarray):
                query_embedding = query
            elif dense_compute_fn is not None:
                query_embedding = dense_compute_fn(query)
        except Exception as e:
            logger.warning(f"Failed to obtain query embedding: {e}")

        if query_embedding is None:
            logger.warning("Query embedding is unavailable; using the first candidate vector for compatibility.")
            query_embedding = candidates_with_embeddings[0][2]
        
        units = [item[0] for item in candidates_with_embeddings]
        original_scores = np.asarray([item[1] for item in candidates_with_embeddings], dtype=np.float64)
        embeddings = [np.asarray(item[2]) for item in candidates_with_embeddings]

        try:
            candidate_matrix = np.vstack(embeddings)
        except Exception as e:
            logger.warning(f"Failed to materialize candidate embedding matrix; returning original RRF results: {e}")
            return candidate_results[:top_k]

        query_embedding = np.asarray(query_embedding, dtype=candidate_matrix.dtype)
        if query_embedding.ndim != 1 or query_embedding.shape[0] != candidate_matrix.shape[1]:
            logger.warning(
                "Query embedding dimension does not match candidates; returning original RRF results: "
                f"query_shape={query_embedding.shape}, candidate_dim={candidate_matrix.shape[1]}"
            )
            return candidate_results[:top_k]
        matrix_norms = np.asarray([np.linalg.norm(row) for row in candidate_matrix])
        safe_norms = np.where(matrix_norms > 0, matrix_norms, 1.0)
        candidate_matrix_norm = candidate_matrix / safe_norms[:, None]
        candidate_matrix_norm[matrix_norms <= 0] = 0.0

        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_normed = query_embedding / query_norm
            relevance = candidate_matrix_norm @ query_normed
        else:
            relevance = np.zeros(candidate_matrix.shape[0], dtype=candidate_matrix_norm.dtype)

        selected_indices: List[int] = []
        available = np.ones(candidate_matrix.shape[0], dtype=bool)
        max_sim = np.zeros(candidate_matrix.shape[0], dtype=candidate_matrix_norm.dtype)
        real_k = min(top_k, candidate_matrix.shape[0])
        tie_epsilon = 1e-5

        def exact_best_index(candidate_indices: np.ndarray) -> int:
            best_exact_idx = int(candidate_indices[0])
            best_exact_score = float("-inf")
            for candidate_idx in candidate_indices:
                idx = int(candidate_idx)
                unit_norm = matrix_norms[idx]
                if query_norm > 0 and unit_norm > 0:
                    exact_relevance = np.dot(query_embedding, candidate_matrix[idx]) / (query_norm * unit_norm)
                else:
                    exact_relevance = 0.0
                exact_max_similarity = 0.0
                for selected_idx in selected_indices:
                    selected_norm = matrix_norms[selected_idx]
                    if unit_norm > 0 and selected_norm > 0:
                        similarity = np.dot(candidate_matrix[idx], candidate_matrix[selected_idx]) / (unit_norm * selected_norm)
                        exact_max_similarity = max(exact_max_similarity, similarity)
                exact_score = lambda_param * exact_relevance - (1 - lambda_param) * exact_max_similarity
                if exact_score > best_exact_score:
                    best_exact_score = exact_score
                    best_exact_idx = idx
            return best_exact_idx

        for _ in range(real_k):
            mmr_scores = lambda_param * relevance - (1 - lambda_param) * max_sim
            mmr_scores[~available] = -np.inf
            best_idx = int(np.argmax(mmr_scores))
            if not np.isfinite(mmr_scores[best_idx]):
                break
            # Preserve the legacy scalar tie/rounding behavior exactly while
            # keeping matrix normalization and max_sim state outside the hot loop.
            available_indices = np.flatnonzero(available)
            near_best = available_indices[np.abs(mmr_scores[available_indices] - mmr_scores[best_idx]) <= tie_epsilon]
            if near_best.size > 1:
                best_idx = exact_best_index(near_best)
            selected_indices.append(best_idx)
            available[best_idx] = False
            np.maximum(max_sim, candidate_matrix_norm @ candidate_matrix_norm[best_idx], out=max_sim)

        results = []
        total_selected = len(selected_indices)
        for rank, idx in enumerate(selected_indices):
            final_score = float(original_scores[idx]) * 0.3 + (total_selected - rank) * 0.7 / total_selected
            results.append((units[idx], final_score))
        
        logger.debug(f"MMR reranking completed: selected {len(results)} from {len(candidate_results)} candidates.")
        return results

    @staticmethod
    def advanced_fusion(multi_results: MultiRetrievalResults,
                       fusion_strategy: str = "hybrid",
                       rrf_k: int = 60,
                       weights: Optional[Dict[RetrievalMethod, float]] = None) -> List[Tuple[MemoryUnit, float]]:
        """Run advanced fusion."""
        if fusion_strategy == "hybrid":
            rrf_results = ScoreFusion.rrf_fusion(multi_results, rrf_k)
            
            if weights:
                temp_results = MultiRetrievalResults()
                for unit, score in rrf_results:
                    result = RetrievalResult(unit, score, RetrievalMethod.HYBRID, {"rrf_score": score})
                    temp_results.add_result(result)
                
                weighted_results = ScoreFusion.weighted_fusion(temp_results, weights)
                return weighted_results
            
            return rrf_results
            
        elif fusion_strategy == "adaptive":
            methods_used = multi_results.get_methods_used()
            
            return ScoreFusion.rrf_fusion(multi_results, rrf_k)
            
        else:  # rrf_weighted
            return ScoreFusion.rrf_fusion(multi_results, rrf_k)

    @staticmethod
    def BAAI_fusion(base_results: List[Tuple[MemoryUnit, float]],
                query: str,
                top_k: int = 10,
                model_name: str = "BAAI/bge-reranker-v2-m3",
                max_candidates: int = 100,
                text_field: str = "text_content",
                max_length: int = 512,
                batch_size: int = 32,
                reranker_type: str = "baai",
                threshold: float = 0.0) -> List[Tuple[MemoryUnit, float]]:
        """Run BAAI fusion."""
        if not base_results:
            logger.warning("BAAI reranking received no input results.")
            return []
        
        
        logger.debug(f"{reranker_type} pre-rerank validation: input_candidates={len(base_results)}")
        
        layer_distribution = {}
        for unit, score in base_results[:max_candidates]:
            level = unit.metadata.get("memory_level", "unknown") if unit.metadata else "unknown"
            layer_distribution[level] = layer_distribution.get(level, 0) + 1
        
        logger.debug(f"Candidate memory-level distribution: {layer_distribution}")
        
        if len(layer_distribution) > 1:
            logger.warning(f"Reranking input contains multiple memory levels: {layer_distribution}")
        
        candidates_to_rerank = base_results[:max_candidates]
        ScoreFusion.ensure_sync_rerank_allowed(reranker_type, context="ScoreFusion.BAAI_fusion")
        
        
        try:
            
            logger.debug(f"Lazy-loading {reranker_type} reranker: {model_name}")
            reranker_manager = ScoreFusion.get_reranker_manager()
            baai_reranker = reranker_manager.get_reranker(reranker_type, model_name)
            
            documents = []
            for unit, _ in candidates_to_rerank:
                text = ScoreFusion._extract_text_content(unit, text_field)
                documents.append(text)
            
            num_candidates_reranked = len(candidates_to_rerank)
            logger.debug(f"{reranker_type}: reranking {num_candidates_reranked} candidates.")
            
            
            rerank_scores = baai_reranker.rerank(
                query, documents, batch_size=batch_size, max_length=max_length
            )
            
            
            reranked_results = [
                (unit, float(score)) 
                for (unit, _), score in zip(candidates_to_rerank, rerank_scores)
            ]
            
            
            reranked_results.sort(key=lambda x: x[1], reverse=True)
            
            
            final_top_k_results = ScoreFusion._apply_threshold(reranked_results, threshold, top_k)
            
            
            final_layer_distribution = {}
            for unit, score in final_top_k_results:
                level = unit.metadata.get("memory_level", "unknown") if unit.metadata else "unknown"
                final_layer_distribution[level] = final_layer_distribution.get(level, 0) + 1
            
            logger.debug(f"Post-rerank top-{top_k} memory-level distribution: {final_layer_distribution}")
            
            logger.info(f"{reranker_type} reranking completed: candidates={num_candidates_reranked}, "
                        f"threshold={threshold}, returned={len(final_top_k_results)}")
            
            return final_top_k_results
            
        except Exception as e:
            if ScoreFusion.is_vllm_sync_rerank_error(e):
                raise
            logger.error(f"{reranker_type} reranking failed; returning original results: {e}")
            logger.debug(traceback.format_exc())
            return ScoreFusion._apply_threshold(base_results, threshold, top_k)

    # @staticmethod
    @staticmethod
    def qwen_fusion(base_results: List[Tuple[MemoryUnit, float]],
            query: str,
            top_k: int = 10,
            model_name: str = "Qwen/Qwen3-Reranker-0.6B",
            max_candidates: int = 100,
            text_field: str = "text_content",
            max_length: int = 512,
            batch_size: int = 32,
            instruction: str = None,
            reranker_type: str = "qwen",
            threshold: float = 0.0) -> List[Tuple[MemoryUnit, float]]:
            """Run Qwen fusion."""
            if not QWEN_AVAILABLE and reranker_type == "qwen":
                logger.warning("Local Qwen reranker is unavailable; returning original results.")
                return ScoreFusion._apply_threshold(base_results, threshold, top_k)
            
            if not base_results:
                logger.warning("Qwen reranking received no input results.")
                return []
            
            candidates_to_rerank = base_results[:max_candidates]
            ScoreFusion.ensure_sync_rerank_allowed(reranker_type, context="ScoreFusion.qwen_fusion")
            
            try:
                
                logger.debug(f"Lazy-loading {reranker_type} reranker: {model_name}")
                reranker_manager = ScoreFusion.get_reranker_manager()
                qwen_reranker = reranker_manager.get_reranker(reranker_type, model_name)
                
                documents = []
                for unit, _ in candidates_to_rerank:
                    text_content = ScoreFusion._extract_text_content(unit, text_field)
                    documents.append(text_content)
                
                num_candidates_reranked = len(candidates_to_rerank)
                logger.debug(f"{reranker_type}: reranking {num_candidates_reranked} candidates.")
                
                
                if reranker_type == "qwen-remote":
                    rerank_scores = qwen_reranker.rerank(
                        query, documents, batch_size=batch_size, max_length=max_length
                    )
                else:
                    rerank_scores = qwen_reranker.rerank(
                        query, documents, batch_size=batch_size, 
                        max_length=max_length, instruction=instruction
                    )
                
                
                reranked_results = [
                    (unit, float(score)) for (unit, _), score in zip(candidates_to_rerank, rerank_scores)
                ]
                
                
                reranked_results.sort(key=lambda x: x[1], reverse=True)
                
                
                final_top_k_results = ScoreFusion._apply_threshold(reranked_results, threshold, top_k)
                
                logger.info(f"{reranker_type} reranking completed: candidates={num_candidates_reranked}, "
                            f"threshold={threshold}, returned={len(final_top_k_results)}")
                
                return final_top_k_results
                
            except Exception as e:
                if ScoreFusion.is_vllm_sync_rerank_error(e):
                    raise
                logger.error(f"{reranker_type} reranking failed; returning original results: {e}")
                logger.debug(traceback.format_exc())
                return ScoreFusion._apply_threshold(base_results, threshold, top_k)
    
    @staticmethod
    def jina_fusion(base_results: List[Tuple[MemoryUnit, float]],
                query: str,
                top_k: int = 10,
                model_name: str = "jinaai/jina-reranker-v3",
                max_candidates: int = 100,
                text_field: str = "text_content",
                max_length: int = 512,
                batch_size: int = 32,
                reranker_manager: Optional[RerankerManager] = None,
                threshold: float = 0.0) -> List[Tuple[MemoryUnit, float]]:
        """Run Jina fusion."""
        if not base_results:
            return []
        ScoreFusion.ensure_sync_rerank_allowed("jina", context="ScoreFusion.jina_fusion")
        
        try:
            candidates = base_results[:max_candidates]
            
            documents = []
            for unit, _ in candidates:
                text = getattr(unit, text_field, None)
                if text is None and hasattr(unit, 'raw_data'):
                    text = unit.raw_data.get(text_field, str(unit))
                if text is None:
                    text = str(unit)
                documents.append(str(text))
            
            
            if reranker_manager is None:
                reranker_manager = RerankerManager()
            
            reranker = reranker_manager.get_reranker("jina", model_name)
            
            
            scores = reranker.rerank(
                query=query,
                documents=documents,
                batch_size=batch_size,
                max_length=max_length
            )
            
            
            reranked = []
            for (unit, original_score), new_score in zip(candidates, scores):
                reranked.append((unit, float(new_score)))
            
            
            reranked.sort(key=lambda x: x[1], reverse=True)
            
            
            final_results = ScoreFusion._apply_threshold(reranked, threshold, top_k)
            
            logger.info(f"Jina reranking completed: candidates={len(candidates)}, "
                        f"threshold={threshold}, returned={len(final_results)}")
            
            return final_results
            
        except Exception as e:
            if ScoreFusion.is_vllm_sync_rerank_error(e):
                raise
            logger.error(f"Jina reranking failed: {e}")
            logger.debug(traceback.format_exc())
            return ScoreFusion._apply_threshold(base_results, threshold, top_k)

    @staticmethod
    def _extract_text_content(unit: MemoryUnit, text_field: str) -> str:
        """Extract text content."""
        if text_field == "text_content":
            cached_text = getattr(unit, "text_cached", None)
            if isinstance(cached_text, str) and cached_text.strip():
                return cached_text

        text_content = ""
        if unit.raw_data:
            text_content = unit.raw_data.get(text_field, "")
            
            if not text_content:
                fallback_fields = ["content", "description", "summary", "title", "message"]
                for field in fallback_fields:
                    text_content = unit.raw_data.get(field, "")
                    if text_content:
                        break
            
            if not text_content:
                try:
                    
                    text_parts = []
                    for key, value in unit.raw_data.items():
                        if isinstance(value, str) and key != "embedding":
                            text_parts.append(f"{key}: {value}")
                    text_content = "; ".join(text_parts) if text_parts else f"Unit {unit.uid}"
                except Exception:
                    text_content = f"Memory unit {unit.uid}"
        
        if not text_content.strip():
            text_content = f"Memory unit {unit.uid}"
        
        return text_content
