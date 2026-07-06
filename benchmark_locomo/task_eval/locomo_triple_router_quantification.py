#!/usr/bin/env python3
"""LoCoMo routed tri-tower benchmark with cascade quantification."""

import os
import re
import sys
import json
import logging
import time
import argparse
import traceback
import gc
import hashlib
import numpy as np
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


# Avoid mutating LogRecord fields before other handlers process the record.
from mandol.utils.logging_config import setup_logging, create_module_logger, auto_configure_logging
if auto_configure_logging() is None:
    setup_logging(level=logging.INFO)
logger = create_module_logger("locomo_triple_content_mix")

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.warning("tiktoken not installed. Using character count approximation for token counting.")

from mandol.core.semantic_graph import SemanticGraph
from mandol.retrieval.rerank_manager import RerankerManager
from mandol.retrieval.advance_retriever import MultiRetriever
from mandol.retrieval.retrieval_interface import RetrievalMethod
from mandol.llm.llm_client import LLMClient
from benchmark_locomo.task_eval.evaluation import (
    calculate_comprehensive_scores,
    cleanup_evaluation_models,
    convert_numpy_types
)

from benchmark_locomo.task_eval.locomo_benchmark_hierarchical_content import (
    UnifiedHierarchicalRetriever, 
    HierarchicalContextBuilder, 
    BenchmarkConfig as HierarchicalConfig,
    MemorySpaceNames
)
from benchmark_locomo.task_eval.locomo_benchmark_entity_relation import LoCoMoEntityRelationBenchmark


from mandol.memory_router.locomo_tower_router import LocomoTowerRouter, TowerRoutingConfig

from mandol.quantification.confidence_pruner import PruneMode
from mandol.quantification.cascade_pruner import (
    CascadeConfidencePruner,
    EnhancedCandidateChunk,
    TowerSource,
    CascadePruneResult,
)
import mandol.quantification.adapters.locomo_category  # noqa: F401
from mandol.core import paths


@dataclass
class TriTowerRetrievalResult:
    """Per-question result record for LoCoMo tri-tower retrieval."""
    sample_id: str
    question: str
    category: int
    expected_answer: str
    
    
    hierarchical_context: Dict[str, Any]
    hierarchical_retrieval_time: float
    
    
    graph_retrieved_units: List[Any]
    graph_retrieval_time: float
    graph_retrieval_details: Dict[str, Any]
    
    
    episodic_retrieved_units: List[Any]
    episodic_retrieval_time: float
    episodic_context_with_time: str
    
    final_answer: str
    reasoning_process: str
    confidence_score: float
    fusion_method: str
    generation_time: float
    
    evaluation_scores: Dict[str, float]
    evaluation_success: bool = True
    
    l0_tokens: int = 0
    l1_tokens: int = 0
    l2_tokens: int = 0
    graph_tokens: int = 0
    episodic_tokens: int = 0
    system_prompt_tokens: int = 0
    question_tokens: int = 0
    total_input_tokens: int = 0
    completion_tokens: int = 0
    
    hierarchical_text: str = ""
    graph_text: str = ""
    
    total_retrieval_time: float = 0.0
    
    end_to_end_latency: float = 0.0
    
    
    second_stage_rerank_enabled: bool = False
    second_stage_rerank_method: str = "none"
    second_stage_rerank_time: float = 0.0
    first_stage_l0_count: int = 0
    first_stage_graph_count: int = 0
    first_stage_episodic_count: int = 0
    first_stage_total_count: int = 0
    final_l0_count: int = 0
    final_graph_count: int = 0
    final_episodic_count: int = 0
    final_selected_count: int = 0
    
    parallel_actual_time: float = 0.0
    parallel_sequential_theory: float = 0.0
    parallel_speedup_ratio: float = 0.0
    parallel_time_saved: float = 0.0
    
    
    routing_info: str = ""
    
    cascade_pruner_enabled: bool = False
    cascade_prune_mode: str = ""
    cascade_tokens_used: int = 0
    cascade_stage1_input: int = 0
    cascade_stage1_output: int = 0
    cascade_stage2_conflicts: int = 0
    cascade_stage2_dropped: int = 0
    cascade_stage2_output: int = 0
    cascade_stage3_mmr_iterations: int = 0
    cascade_stage3_diversity_penalties: int = 0


class LoCoMoTriTowerBenchmark:
    """Run LoCoMo tri-tower retrieval, routing, generation, and evaluation."""
    
    def __init__(self,
                 step3_graphs_dir: str,
                 enhanced_graphs_dir: str,
                 episodic_graphs_dir: str,
                 qa_dataset_path: str,
                 output_dir: str = "tri_tower_benchmark_results",
                 
                 llm_model: str = "gpt-4o-mini-closeai",
                 llm_evaluate_model: str = "gpt-4o-mini-closeai",
                 llm_client: Optional[LLMClient] = None,
                 llm_evaluate_client: Optional[LLMClient] = None,
                 
                 target_sample_ids: Optional[List[str]] = None,
                 max_questions: Optional[int] = None,
                 
                 topk_hierarchical: int = 15,  
                 hierarchical_retrieval_methods: List[str] = None,  
                 hierarchical_fusion_method: str = "rrf",
                 
                 topk_similarity: int = 30,  
                 topk_graph: int = 0,
                 use_entity_relation: bool = True,
                 
                 topk_episodic: int = 30,  
                 
                 
                 enable_second_stage_rerank: bool = True,
                 second_stage_rerank_method: Optional[str] = None,  
                 final_top_k: int = 20,  
                 rerank_threshold: float = 0.0,  
                 rerank_strategy: str = "tower_separate",  
                 
                 
                 reranker_type: str = "baai",
                 reranker_configs: Optional[Dict[str, str]] = None,
                 reranker_manager: Optional[RerankerManager] = None,
                 
                 fusion_strategy: str = "context_aware",
                 fusion_weights: Optional[Dict[str, float]] = None,
                 
                 
                 tower_router: Optional[LocomoTowerRouter] = None,
                 
                 enable_cascade_pruner: bool = False,
                 cascade_prune_mode: str = "BUDGET_MAX",
                 cascade_mad_multiplier: float = 2.5,
                 cascade_cliff_tolerance: float = 2.0,
                 cascade_absolute_min_score: float = 0.0,
                 cascade_max_context_tokens: int = 2500,
                 cascade_lambda_mmr: float = 0.6,
                 cascade_entity_overlap_penalty: float = 0.15,
                 cascade_source_duplicate_penalty: float = 0.2,
                 cascade_temporal_decay_hours: float = 720.0,
                 cascade_enable_stage2: bool = True,
                 cascade_enable_stage3_mmr: bool = True,
                 cascade_enable_stage1: bool = True,
                 cascade_cap_to_input_tokens: bool = True,
                 cascade_tower_min_ratio: Optional[Dict[str, float]] = None,
                 cascade_adaptive_dataset: Optional[str] = None,

                 save_individual_reports: bool = True,
                 
                 parallel_towers: bool = False,
                 max_workers: int = 3,
                 generation_max_tokens: int = 2000):
        """Initialize the LoCoMo tri-tower benchmark runner."""
        self.step3_graphs_dir = Path(step3_graphs_dir)
        self.enhanced_graphs_dir = Path(enhanced_graphs_dir)
        self.episodic_graphs_dir = Path(episodic_graphs_dir)
        self.qa_dataset_path = Path(qa_dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.llm_model = llm_model
        self.llm_evaluate_model = llm_evaluate_model
        self.llm_client = llm_client or LLMClient(model_name=llm_model)
        self.llm_evaluate_client = llm_evaluate_client or LLMClient(model_name=llm_evaluate_model)
        self.generation_max_tokens = generation_max_tokens
        
        self.target_sample_ids = list(target_sample_ids) if target_sample_ids else None
        self.max_questions = max_questions
        
        self.topk_hierarchical = topk_hierarchical
        self.hierarchical_retrieval_methods = hierarchical_retrieval_methods or ["bm25", "cosine_similarity", "splade"]
        self.hierarchical_fusion_method = hierarchical_fusion_method
        
        self.hierarchical_config = HierarchicalConfig(
            top_k=topk_hierarchical,
            retrieval_methods=self.hierarchical_retrieval_methods,
            fusion_method=hierarchical_fusion_method,
            rerank_method=reranker_type,
            use_all_spaces=True,
            graphs_dir=str(enhanced_graphs_dir)
        )
        
        self.topk_similarity = topk_similarity
        self.topk_graph = topk_graph
        self.use_entity_relation = use_entity_relation
        
        self.topk_episodic = topk_episodic
        
        
        self.enable_second_stage_rerank = enable_second_stage_rerank
        self.second_stage_rerank_method = second_stage_rerank_method or reranker_type
        self.final_top_k = final_top_k
        self.rerank_threshold = rerank_threshold  
        self.rerank_strategy = rerank_strategy  
        
        
        self.reranker_type = reranker_type
        self.reranker_configs = reranker_configs or {
            "baai": "BAAI/bge-reranker-v2-m3",
            "qwen": "Qwen/Qwen3-Reranker-0.6B",
            "jina": "jinaai/jina-reranker-v2-base-multilingual",
            "qwen-sili": "Qwen/Qwen3-Reranker-8B",
            "qwen-dashscope": "qwen3-rerank",
            "gte-dashscope": "gte-rerank-v2"
        }
        
        self.global_reranker_manager = reranker_manager
        if reranker_manager:
            logger.info(" 使用外部传入的重排序器管理器")
        else:
            logger.info(" 创建新的重排序器管理器")
            self.global_reranker_manager = RerankerManager()
        
        self.fusion_strategy = fusion_strategy
        self.fusion_weights = fusion_weights or {
            "hierarchical": 0.35,
            "graph": 0.35,
            "episodic": 0.30
        }
        
        
        self.tower_router = tower_router
        self.save_individual_reports = save_individual_reports
        
        self.cascade_pruner = None
        self.cascade_max_context_tokens = cascade_max_context_tokens
        if enable_cascade_pruner:
            try:
                _mode_map = {
                    "STRICT_THRESHOLD": PruneMode.STRICT_THRESHOLD,
                    "CUMULATIVE_EARLY_STOP": PruneMode.CUMULATIVE_EARLY_STOP,
                    "CLIFF_EARLY_STOP": PruneMode.CLIFF_EARLY_STOP,
                    "BUDGET_MAX": PruneMode.BUDGET_MAX,
                    "DYNAMIC_ADAPTIVE": PruneMode.DYNAMIC_ADAPTIVE,
                }
                _resolved_mode = _mode_map.get(cascade_prune_mode.upper(), PruneMode.BUDGET_MAX)
                self.cascade_pruner = CascadeConfidencePruner(
                    mode=_resolved_mode,
                    mad_multiplier=cascade_mad_multiplier,
                    cliff_tolerance=cascade_cliff_tolerance,
                    absolute_min_score=cascade_absolute_min_score,
                    max_tokens=cascade_max_context_tokens,
                    lambda_mmr=cascade_lambda_mmr,
                    entity_overlap_penalty=cascade_entity_overlap_penalty,
                    source_duplicate_penalty=cascade_source_duplicate_penalty,
                    temporal_decay_hours=cascade_temporal_decay_hours,
                    enable_stage1=cascade_enable_stage1,
                    enable_stage2=cascade_enable_stage2,
                    enable_stage3_mmr=cascade_enable_stage3_mmr,
                    cap_to_input_tokens=cascade_cap_to_input_tokens,
                    tower_min_ratio=self._parse_tower_min_ratio(cascade_tower_min_ratio),
                    adaptive_dataset=cascade_adaptive_dataset,
                )
                self.cascade_prune_mode = _resolved_mode
                self.cascade_enable_stage2 = cascade_enable_stage2
                self.cascade_enable_stage3_mmr = cascade_enable_stage3_mmr
                logger.info(f" 级联量化剪枝器初始化成功: mode={_resolved_mode.value}, "
                           f"max_tokens={cascade_max_context_tokens}, λ_mmr={cascade_lambda_mmr}")
            except Exception as e:
                logger.error(f" 级联量化剪枝器初始化失败: {e}")
                self.cascade_pruner = None
        
        self.parallel_towers = parallel_towers
        self.max_workers = max_workers
        
        self._initialize_subsystems()
        
        self.hierarchical_retrievers: Dict[str, UnifiedHierarchicalRetriever] = {}
        self.hierarchical_graphs: Dict[str, SemanticGraph] = {}
        self.episodic_graphs: Dict[str, SemanticGraph] = {}
        self.episodic_retrievers: Dict[str, MultiRetriever] = {}
        self.available_samples: List[str] = []
        
        self.test_cases: List[Dict[str, Any]] = []
        self.test_results: List[TriTowerRetrievalResult] = []
        
        self.stats = {
            "total_samples_loaded": 0,
            "total_test_cases": 0,
            "successful_tri_tower": 0,
            "failed_retrievals": 0,
            "successful_hierarchical": 0,
            "successful_graph": 0,
            "successful_episodic": 0
        }
        
        logger.info("=" * 80)
        logger.info("LoCoMo三塔召回（Tri-Tower）Benchmark测试器初始化完成")
        logger.info("=" * 80)
        logger.info(f" 知识图谱目录: {self.step3_graphs_dir}")
        logger.info(f" 分层图谱目录: {self.enhanced_graphs_dir}")
        logger.info(f" 情景记忆目录: {self.episodic_graphs_dir}")
        logger.info(f" QA数据集: {self.qa_dataset_path}")
        logger.info(f" 融合策略: {self.fusion_strategy}")
        logger.info(f"  融合权重: {self.fusion_weights}")
        logger.info(f" 并行检索: {'启用' if self.parallel_towers else '禁用'}")
        logger.info("-" * 80)
        logger.info(f" 第一阶段检索配置:")
        logger.info(f"   分层(V2统一): top_k={self.topk_hierarchical}, 方法={self.hierarchical_retrieval_methods}, 融合={self.hierarchical_fusion_method}")
        logger.info(f"   图谱: {self.topk_similarity} (实体关系: {'启用' if self.topk_graph > 0 else '禁用'}, top_k={self.topk_graph})")
        logger.info(f"   情景: {self.topk_episodic}")
        if self.enable_second_stage_rerank:
            strategy_desc = {
                "tower_separate": "分层塔直通车 + 其他两塔重排序",
                "unified_rerank": "三塔统一重排序"
            }.get(self.rerank_strategy, self.rerank_strategy)
            logger.info(f" 二次重排序:  启用")
            logger.info(f"   重排序策略: {self.rerank_strategy} ({strategy_desc})")
            logger.info(f"   重排序器: {self.second_stage_rerank_method}")
            logger.info(f"   最终Top-K: {self.final_top_k}")
            
            h_pass = self.topk_hierarchical if self.topk_hierarchical > 0 else 0
            total_budget = h_pass + self.final_top_k
            logger.info(f"    v2预算: H直通={h_pass} + rerank={self.final_top_k} = {total_budget}条 Context Budget")
        else:
            logger.info(f" 二次重排序:  禁用")
        if self.tower_router:
            logger.info(f" 塔路由器:  启用 (模型={self.tower_router.model_name}, 策略={self.tower_router.strategy})")
        else:
            logger.info(f" 塔路由器:  禁用 (使用静态配置)")
        if self.cascade_pruner is not None:
            logger.info(f" 级联量化:  启用 (CascadeConfidencePruner)")
            logger.info(f"   剪枝模式: {self.cascade_prune_mode.value}")
            logger.info(f"   max_tokens={self.cascade_max_context_tokens}, "
                        f"Stage2={'ON' if self.cascade_enable_stage2 else 'OFF'}, "
                        f"Stage3_MMR={'ON' if self.cascade_enable_stage3_mmr else 'OFF'}")
        else:
            logger.info(f" 级联量化:  禁用")
        logger.info("=" * 80)
        
        self._init_tokenizer()
    
    def _init_tokenizer(self):
        """Initialize tokenizer."""
        if TIKTOKEN_AVAILABLE:
            try:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
                logger.info(" tiktoken 编码器初始化成功 (cl100k_base)")
            except Exception as e:
                logger.warning(f"tiktoken 编码器初始化失败: {e}，使用字符估算")
                self.tokenizer = None
        else:
            self.tokenizer = None
            logger.warning("tiktoken 不可用，使用字符估算")
    
    def _count_tokens(self, text: str, model: str = "gpt-4") -> int:
        """Count tokens."""
        if not text:
            return 0
        
        if self.tokenizer is not None:
            try:
                return len(self.tokenizer.encode(text))
            except Exception as e:
                logger.warning(f"tiktoken 编码失败: {e}，使用字符估算")
        
        return len(text) // 4
    
    def _format_hierarchical_context(self, hierarchical_context: Dict[str, Any]) -> str:
        """Format hierarchical context."""
        hierarchical_enabled = hierarchical_context.get("hierarchical_enabled", False)
        if hierarchical_enabled:
            return hierarchical_context.get("hierarchical_context_text", "No hierarchical context available")
        else:
            return "Hierarchical retrieval was not available for this query."
    
    def _count_hierarchical_tokens_by_layer(self, hierarchical_context: Dict[str, Any]) -> Dict[str, int]:
        """Count hierarchical tokens by layer."""
        l0_tokens = 0
        l1_tokens = 0
        l2_tokens = 0
        
        by_layer = hierarchical_context.get("by_layer", {})
        
        
        l0_items = hierarchical_context.get("l0_results", [])
        if not l0_items:
            l0_items = hierarchical_context.get("l0_observations", [])
        if not l0_items:
            l0_items = by_layer.get("L0", [])
            
        for i, item in enumerate(l0_items, 1):
            content = ""
            if isinstance(item, dict):
                content = item.get("content", "")
                if not content and "unit" in item:
                    content = self._extract_unit_content_safe(item["unit"])
            else:
                content = str(item)
            
            if content:
                full_content = f"Observation {i}: {content}"
                l0_tokens += self._count_tokens(full_content)
        
        l1_items = hierarchical_context.get("l1_results", [])
        if not l1_items:
            l1_items = hierarchical_context.get("l1_summaries", [])
        if not l1_items:
            l1_items = by_layer.get("L1", [])
            
        for i, item in enumerate(l1_items, 1):
            content = ""
            if isinstance(item, dict):
                content = item.get("content", "")
            else:
                content = str(item)
                
            if content:
                full_content = f"Summary {i}: {content}"
                l1_tokens += self._count_tokens(full_content)
        
        l2_items = hierarchical_context.get("l2_results", [])
        if not l2_items:
            l2_items = hierarchical_context.get("l2_insights", [])
        if not l2_items:
            l2_items = by_layer.get("L2", [])
            
        for i, item in enumerate(l2_items, 1):
            content = ""
            if isinstance(item, dict):
                content = item.get("content", "")
            else:
                content = str(item)
                
            if content:
                full_content = f"Insight {i}: {content}"
                l2_tokens += self._count_tokens(full_content)
        
        if l0_tokens > 0: l0_tokens += 5
        if l1_tokens > 0: l1_tokens += 5
        if l2_tokens > 0: l2_tokens += 5
        
        return {
            "l0_tokens": l0_tokens,
            "l1_tokens": l1_tokens,
            "l2_tokens": l2_tokens
        }
    
    def _format_graph_context(self, graph_units: List[Any], max_results: int = 10) -> str:
        """Format graph context."""
        if not graph_units:
            return "No relevant entities or relationships found in the knowledge graph."
        
        context_parts = []
        context_parts.append(f"Retrieved {len(graph_units)} relevant knowledge graph units:")
        context_parts.append("")
        
        for i, (unit, score) in enumerate(graph_units[:max_results], 1):
            unit_content = self._extract_graph_unit_content(unit)
            context_parts.append(f"Graph Result {i}: {unit_content}")
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    
    
    
    _RE_PROPER_NOUN = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")
    _RE_QUOTED = re.compile(r'"([^"]{2,50})"')

    @staticmethod
    def _extract_fallback_entities(text: str) -> List[str]:
        """Extract fallback entities."""
        if not text:
            return []
        seen: set = set()
        result: List[str] = []
        for sent in re.split(r'[.!?]\s+', text):
            words = sent.split()
            for m in LoCoMoTriTowerBenchmark._RE_PROPER_NOUN.finditer(sent):
                span_start = m.start()
                if span_start == 0 and len(words) > 0:
                    continue
                key = m.group(1).lower()
                if key not in seen:
                    seen.add(key)
                    result.append(key)
        for m in LoCoMoTriTowerBenchmark._RE_QUOTED.finditer(text):
            key = m.group(1).lower()
            if key not in seen:
                seen.add(key)
                result.append(key)
        return result

    def _extract_entity_ids_for_cascade(self, unit, source_type: str) -> List[str]:
        """Extract entity IDs for cascade."""
        raw = unit.raw_data if hasattr(unit, 'raw_data') and unit.raw_data else {}
        if isinstance(unit, dict):
            raw = unit
        entity_ids: List[str] = []
        
        if source_type == "graph":
            name = raw.get("entity_canonical") or raw.get("entity_text") or raw.get("entity_name") or ""
            if name:
                entity_ids.append(name.lower().strip())
            target = raw.get("target_entity") or raw.get("related_entity") or ""
            if target:
                entity_ids.append(target.lower().strip())
        elif source_type == "episodic":
            entities = raw.get("entities") or raw.get("entity_ids") or raw.get("involved_entities") or []
            if isinstance(entities, list):
                entity_ids.extend([str(e).lower().strip() for e in entities if e])
        else:
            speaker = raw.get("speaker") or raw.get("role") or ""
            if hasattr(unit, 'metadata') and unit.metadata:
                speaker = speaker or unit.metadata.get("speaker") or unit.metadata.get("role") or ""
            if speaker and speaker.lower() not in ('unknown', 'system', 'assistant', 'user'):
                entity_ids.append(speaker.lower().strip())
        
        if not entity_ids:
            text = ""
            if hasattr(unit, 'content'):
                text = unit.content or ""
            elif isinstance(unit, dict):
                text = unit.get('content', '') or unit.get('text_content', '') or ""
            if text:
                entity_ids = [f"ner:{e}" for e in self._extract_fallback_entities(text)]
        
        return entity_ids
    
    def _extract_timestamp_for_cascade(self, unit, source_type: str) -> float:
        """Extract timestamp for cascade."""
        raw = unit.raw_data if hasattr(unit, 'raw_data') and unit.raw_data else {}
        if isinstance(unit, dict):
            raw = unit
        metadata = unit.metadata if hasattr(unit, 'metadata') and unit.metadata else {}
        
        date_str = (
            raw.get('event_date') or raw.get('temporal_val') or
            raw.get('time') or raw.get('timestamp') or
            raw.get('session_date') or raw.get('date') or
            raw.get('created_at') or
            metadata.get('session_date') or metadata.get('timestamp') or
            metadata.get('time_range') or ""
        )
        
        if not date_str or date_str in ("Unknown Date", "unknown"):
            return 0.0
        
        for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d", "%m/%d/%Y",
                     "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"):
            try:
                dt = datetime.strptime(str(date_str).strip(), fmt)
                return dt.timestamp()
            except (ValueError, TypeError):
                continue
        
        return 0.0
    
    def _extract_source_chunk_ids_for_cascade(self, unit) -> List[str]:
        """Extract source chunk IDs for cascade."""
        raw = unit.raw_data if hasattr(unit, 'raw_data') and unit.raw_data else {}
        if isinstance(unit, dict):
            raw = unit
        source_ids = raw.get("source_chunk_ids") or raw.get("source_ids") or []
        if isinstance(source_ids, list):
            return [str(s) for s in source_ids if s]
        return []

    def _perform_cascade_rerank(self,
                                question: str,
                                l0_units: List[Tuple[Any, float]],
                                graph_units: List[Tuple[Any, float]],
                                episodic_units: List[Tuple[Any, float]],
                                category: Optional[int] = None) -> Dict[str, Any]:
        """Run perform cascade rerank."""
        first_stage_counts = {
            "l0": len(l0_units),
            "graph": len(graph_units),
            "episodic": len(episodic_units),
            "total": len(l0_units) + len(graph_units) + len(episodic_units)
        }
        
        result = {
            "reranked_l0": l0_units,
            "reranked_graph": graph_units,
            "reranked_episodic": episodic_units,
            "rerank_time": 0.0,
            "first_stage_counts": first_stage_counts,
            "final_counts": {
                "l0": len(l0_units),
                "graph": len(graph_units),
                "episodic": len(episodic_units),
                "total": len(l0_units) + len(graph_units) + len(episodic_units)
            },
            "strategy_used": "router_cascade",
            "cascade_stats": {}
        }
        
        
        all_units_with_source = []
        for unit, score in l0_units:
            all_units_with_source.append((unit, score, "l0"))
        for unit, score in graph_units:
            all_units_with_source.append((unit, score, "graph"))
        for unit, score in episodic_units:
            all_units_with_source.append((unit, score, "episodic"))
        
        if not all_units_with_source:
            return result
        
        start_time = time.time()
        
        try:
            
            reranker = self.global_reranker_manager.get_reranker(self.second_stage_rerank_method)
            
            if reranker is None:
                logger.warning(f"无法获取重排序器: {self.second_stage_rerank_method}，跳过级联量化")
                return result
            
            documents = []
            for unit, _, source in all_units_with_source:
                if isinstance(unit, dict):
                    raw = unit
                else:
                    raw = unit.raw_data if hasattr(unit, 'raw_data') and unit.raw_data else {}
                text = raw.get('text_content', '')
                if not text:
                    text = raw.get('content', '') or raw.get('description', '') or str(raw)
                documents.append(text)
            
            ce_scores = reranker.rerank(query=question, documents=documents)
            
            
            
            
            _scored = list(zip(range(len(all_units_with_source)), ce_scores))
            _scored.sort(key=lambda x: x[1], reverse=True)
            
            if self.rerank_strategy == "tower_separate":
                
                _l0_indices = {i for i, (_, _, src) in enumerate(all_units_with_source) if src == "l0"}
                _ge_ranked = [(i, s) for i, s in _scored if i not in _l0_indices]
                _keep_indices = _l0_indices | {i for i, s in _ge_ranked[:self.final_top_k]}
            else:
                
                _keep_indices = {i for i, s in _scored[:self.final_top_k]}
            
            _total_before = len(all_units_with_source)
            
            all_units_with_source = [all_units_with_source[i] for i in sorted(_keep_indices)]
            documents = [documents[i] for i in sorted(_keep_indices)]
            ce_scores = [ce_scores[i] for i in sorted(_keep_indices)]
            
            logger.debug(
                f" 级联输入过滤: {_total_before} → {len(all_units_with_source)} 候选 "
                f"(final_top_k={self.final_top_k}, strategy={self.rerank_strategy})"
            )
            
            candidates: List[EnhancedCandidateChunk] = []
            chunks_meta = []
            for idx, ((unit, old_score, source), ce_score) in enumerate(
                zip(all_units_with_source, ce_scores)
            ):
                if isinstance(unit, dict):
                    chunk_id = unit.get('uid', f"{source}_{idx}")
                else:
                    chunk_id = getattr(unit, 'uid', f"{source}_{idx}")
                candidates.append(EnhancedCandidateChunk(
                    chunk_id=chunk_id,
                    text=documents[idx],
                    ce_score=ce_score,
                    rank_dense=getattr(unit, 'rank_dense', 999),
                    rank_splade=getattr(unit, 'rank_splade', 999),
                    rank_bm25=getattr(unit, 'rank_bm25', 999),
                    tower_source=TowerSource.from_source_type(source),
                    entity_ids=self._extract_entity_ids_for_cascade(unit, source),
                    timestamp=self._extract_timestamp_for_cascade(unit, source),
                    source_chunk_ids=self._extract_source_chunk_ids_for_cascade(unit) or [chunk_id],
                    memory_space=source,
                ))
                chunks_meta.append({
                    "unit": unit, "source": source,
                    "ce_score": ce_score, "chunk_id": chunk_id
                })
            
            cascade_result: CascadePruneResult = self.cascade_pruner.prune(
                candidates,
                query_category=str(category) if category is not None else None,
            )
            
            
            selected_ids = {pc.chunk.chunk_id for pc in cascade_result.selected_chunks}
            
            reranked_l0 = []
            reranked_graph = []
            reranked_episodic = []
            
            for meta in chunks_meta:
                if meta["chunk_id"] not in selected_ids:
                    continue
                pair = (meta["unit"], meta["ce_score"])
                if meta["source"] == "l0":
                    reranked_l0.append(pair)
                elif meta["source"] == "graph":
                    reranked_graph.append(pair)
                else:
                    reranked_episodic.append(pair)
            
            reranked_l0.sort(key=lambda x: x[1], reverse=True)
            reranked_graph.sort(key=lambda x: x[1], reverse=True)
            reranked_episodic.sort(key=lambda x: x[1], reverse=True)
            
            rerank_time = time.time() - start_time
            
            result = {
                "reranked_l0": reranked_l0,
                "reranked_graph": reranked_graph,
                "reranked_episodic": reranked_episodic,
                "rerank_time": rerank_time,
                "first_stage_counts": first_stage_counts,
                "final_counts": {
                    "l0": len(reranked_l0),
                    "graph": len(reranked_graph),
                    "episodic": len(reranked_episodic),
                    "total": len(reranked_l0) + len(reranked_graph) + len(reranked_episodic)
                },
                "strategy_used": "router_cascade",
                "cascade_stats": {
                    "mode": cascade_result.mode_used.value,
                    "input_candidates": len(candidates),
                    "selected_count": len(cascade_result.selected_chunks),
                    "tokens_used": cascade_result.total_tokens_used,
                    "max_tokens": self.cascade_max_context_tokens,
                    "cascade_stage1_input": cascade_result.stage1_input_count,
                    "cascade_stage1_output": cascade_result.stage1_output_count,
                    "cascade_stage2_conflicts": cascade_result.stage2_conflicts_found,
                    "cascade_stage2_dropped": cascade_result.stage2_dropped_count,
                    "cascade_stage2_output": cascade_result.stage2_output_count,
                    "cascade_stage3_mmr_iterations": cascade_result.stage3_mmr_iterations,
                    "cascade_stage3_diversity_penalties": cascade_result.stage3_diversity_penalties,
                }
            }
            
            logger.debug(
                f"级联量化: "
                f"L0: {len(l0_units)}->{len(reranked_l0)}, "
                f"Graph: {len(graph_units)}->{len(reranked_graph)}, "
                f"Episodic: {len(episodic_units)}->{len(reranked_episodic)} | "
                f"S1:{cascade_result.stage1_output_count} → "
                f"S2:{cascade_result.stage2_output_count} "
                f"(冲突={cascade_result.stage2_conflicts_found}, "
                f"丢弃={cascade_result.stage2_dropped_count}) → "
                f"S3:{len(cascade_result.selected_chunks)} "
                f"(MMR轮={cascade_result.stage3_mmr_iterations}) | "
                f"tokens={cascade_result.total_tokens_used}/{self.cascade_max_context_tokens}, "
                f"耗时 {rerank_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"级联量化失败: {e}")
            logger.error(traceback.format_exc())
            result["rerank_time"] = time.time() - start_time
            return result
    
    
    
    def _perform_second_stage_rerank(self,
                                     question: str,
                                     l0_units: List[Tuple[Any, float]],
                                     graph_units: List[Tuple[Any, float]],
                                     episodic_units: List[Tuple[Any, float]]) -> Dict[str, Any]:
        """Apply the configured second-stage reranker across enabled towers.

        Args:
            question: Evaluation question used as the rerank query.
            l0_units: Candidate observation units and scores.
            graph_units: Candidate graph units and scores.
            episodic_units: Candidate episodic units and scores.

        Returns:
            Dictionary containing per-tower reranked results, timing, input
            counts, output counts, and the strategy identifier.
        """
        result = {
            "reranked_l0": l0_units,
            "reranked_graph": graph_units,
            "reranked_episodic": episodic_units,
            "rerank_time": 0.0,
            "first_stage_counts": {
                "l0": len(l0_units),
                "graph": len(graph_units),
                "episodic": len(episodic_units),
                "total": len(l0_units) + len(graph_units) + len(episodic_units)
            },
            "final_counts": {
                "l0": len(l0_units),
                "graph": len(graph_units),
                "episodic": len(episodic_units),
                "total": len(l0_units) + len(graph_units) + len(episodic_units)
            },
            "strategy_used": self.rerank_strategy
        }
        
        
        if self.rerank_strategy == "unified_rerank":
            
            all_units_with_source = []
            for unit, score in l0_units:
                all_units_with_source.append((unit, score, "l0"))
            for unit, score in graph_units:
                all_units_with_source.append((unit, score, "graph"))
            for unit, score in episodic_units:
                all_units_with_source.append((unit, score, "episodic"))
        else:  
            
            all_units_with_source = []
            for unit, score in graph_units:
                all_units_with_source.append((unit, score, "graph"))
            for unit, score in episodic_units:
                all_units_with_source.append((unit, score, "episodic"))
        
        if not all_units_with_source:
            return result
        
        start_time = time.time()
        
        try:
            
            reranker = self.global_reranker_manager.get_reranker(self.second_stage_rerank_method)
            
            if reranker is None:
                logger.warning(f"无法获取重排序器: {self.second_stage_rerank_method}，跳过二次重排")
                return result
            
            documents = []
            for unit, _, source in all_units_with_source:
                raw = unit.raw_data if hasattr(unit, 'raw_data') and unit.raw_data else {}
                text = raw.get('text_content', '')
                if not text:
                    text = raw.get('content', '') or raw.get('description', '') or str(raw)
                documents.append(text)
            
            
            scores = reranker.rerank(query=question, documents=documents)
            
            
            reranked = list(zip(all_units_with_source, scores))
            reranked.sort(key=lambda x: x[1], reverse=True)
            
            if self.rerank_threshold > 0.0:
                filtered_count_before = len(reranked)
                reranked = [(item, score) for item, score in reranked if score >= self.rerank_threshold]
                filtered_count_after = len(reranked)
                if filtered_count_before != filtered_count_after:
                    logger.info(f" 阈值过滤: {filtered_count_before} -> {filtered_count_after} "
                              f"(threshold={self.rerank_threshold:.3f})")
            
            if self.rerank_strategy == "unified_rerank":
                
                final_selected = []
                for (unit, old_score, source), new_score in reranked[:self.final_top_k]:
                    final_selected.append((unit, new_score, source))
                
                reranked_l0 = [(u, s) for u, s, src in final_selected if src == "l0"]
                reranked_graph = [(u, s) for u, s, src in final_selected if src == "graph"]
                reranked_episodic = [(u, s) for u, s, src in final_selected if src == "episodic"]
            else:  # tower_separate
                
                reranked_l0 = l0_units
                
                final_selected = []
                for (unit, old_score, source), new_score in reranked[:self.final_top_k]:
                    final_selected.append((unit, new_score, source))
                
                reranked_graph = [(u, s) for u, s, src in final_selected if src == "graph"]
                reranked_episodic = [(u, s) for u, s, src in final_selected if src == "episodic"]
            
            rerank_time = time.time() - start_time
            
            result = {
                "reranked_l0": reranked_l0,
                "reranked_graph": reranked_graph,
                "reranked_episodic": reranked_episodic,
                "rerank_time": rerank_time,
                "first_stage_counts": {
                    "l0": len(l0_units),
                    "graph": len(graph_units),
                    "episodic": len(episodic_units),
                    "total": len(l0_units) + len(graph_units) + len(episodic_units)
                },
                "final_counts": {
                    "l0": len(reranked_l0),
                    "graph": len(reranked_graph),
                    "episodic": len(reranked_episodic),
                    "total": len(reranked_l0) + len(reranked_graph) + len(reranked_episodic)
                },
                "strategy_used": self.rerank_strategy
            }
            
            strategy_desc = "统一重排序" if self.rerank_strategy == "unified_rerank" else "分层塔直通车"
            logger.debug(f"二次重排序 ({strategy_desc}): "
                        f"L0: {len(l0_units)}->{len(reranked_l0)}, "
                        f"Graph: {len(graph_units)}->{len(reranked_graph)}, "
                        f"Episodic: {len(episodic_units)}->{len(reranked_episodic)}, "
                        f"耗时 {rerank_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"二次重排序失败: {e}")
            logger.error(traceback.format_exc())
            result["rerank_time"] = time.time() - start_time
            return result
    
    def _extract_l0_units_from_hierarchical(self, hierarchical_context: Dict[str, Any]) -> List[Tuple[Any, float]]:
        """Extract L0 units from hierarchical."""
        l0_units = []
        try:
            if not hierarchical_context.get("hierarchical_enabled", False):
                return l0_units
            
            candidates = hierarchical_context.get("l0_results", [])
            if not candidates:
                candidates = hierarchical_context.get("l0_observations", [])
            if not candidates:
                by_layer = hierarchical_context.get("by_layer", {})
                candidates = by_layer.get("L0", [])
            
            for item in candidates:
                unit = None
                score = 0.0
                
                if isinstance(item, dict):
                    if "unit" in item:
                        unit = item["unit"]
                        score = item.get("score", 0.0)
                    
                    elif "content" in item or "text_content" in item:
                        unit = item
                        score = item.get("score", 0.0)
                        
                elif isinstance(item, tuple) and len(item) >= 2:
                    unit = item[0]
                    score = item[1]
                else:
                    unit = item
                    
                if unit is not None:
                    l0_units.append((unit, score))
            
            return l0_units
            
        except Exception as e:
            logger.warning(f"提取L0单元失败: {e}")
            return l0_units
    
    
    
    
    
    def _extract_unit_content_safe(self, unit) -> str:
        """Extract unit content safe."""
        try:
            if hasattr(unit, 'raw_data') and unit.raw_data:
                raw = unit.raw_data
                return raw.get('text_content', '') or raw.get('content', '') or str(raw)
            return str(unit)
        except:
            return ""

    def _sanitize_for_json(self, obj):
        """Run sanitize for JSON."""
        if obj is None:
            return None
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, np.generic):
            return self._sanitize_for_json(obj.item())
        if isinstance(obj, float):
            if not np.isfinite(obj):
                return None
            return obj
        if isinstance(obj, (str, int, bool)):
            return obj
        if isinstance(obj, dict):
            return {str(k): self._sanitize_for_json(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [self._sanitize_for_json(item) for item in obj]
        if hasattr(obj, "raw_data") and getattr(obj, "raw_data"):
            return self._sanitize_for_json(obj.raw_data)
        return str(obj)

    def _coerce_report_unit_score(self, item) -> Tuple[Any, float]:
        """Run coerce report unit score."""
        if isinstance(item, dict):
            score = item.get("score", item.get("ce_score", 0.0))
            return item.get("unit", item), score
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            return item[0], item[1]
        return item, getattr(item, "score", 0.0)

    def _extract_content_for_report(self, unit, score: float, source_type: str) -> Dict[str, Any]:
        """Extract content for report."""
        result = {
            "score": float(score) if score is not None else 0.0,
            "source_type": source_type,
            "text_content": "",
        }
        try:
            raw = unit.raw_data if hasattr(unit, "raw_data") and unit.raw_data else unit
            metadata = unit.metadata if hasattr(unit, "metadata") and unit.metadata else {}
            if not isinstance(raw, dict):
                raw = {}
            if not isinstance(metadata, dict):
                metadata = {}

            result["text_content"] = (
                raw.get("text_content") or raw.get("content") or raw.get("description")
                or raw.get("text") or raw.get("fact") or str(unit)
            )
            result["uid"] = raw.get("uid") or raw.get("id") or getattr(unit, "uid", "")

            if "hierarchical" in source_type:
                result["memory_level"] = source_type.rsplit("_", 1)[-1]
                result["session_date"] = (
                    metadata.get("session_date") or metadata.get("date")
                    or metadata.get("time_range") or raw.get("session_date") or ""
                )
                result["role"] = raw.get("role", "") or metadata.get("role", "")
            elif source_type == "entity_relation":
                result["entity_name"] = raw.get("entity_name", "") or raw.get("source_entity", "")
                result["entity_type"] = raw.get("entity_type", "") or raw.get("relation_type", "")
            elif source_type == "episodic":
                result["event_date"] = (
                    raw.get("event_date") or raw.get("date")
                    or metadata.get("event_date") or metadata.get("date")
                    or metadata.get("session_date") or ""
                )
                result["category"] = raw.get("category", "") or metadata.get("category", "")
        except Exception as e:
            result["text_content"] = str(unit) if unit else ""
            result["extraction_error"] = str(e)
        return result

    def _collect_retrieved_contents_for_report(
        self,
        hierarchical_context: Dict[str, Any],
        graph_units: List[Any],
        episodic_units: List[Any],
        rerank_metadata: Optional[Dict[str, Any]] = None,
        l0_units_for_prompt: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """Run collect retrieved contents for report."""
        retrieved_contents = {
            "tower1_hierarchical": {
                "description": "Final hierarchical content used for the LLM prompt.",
                "by_layer": {},
            },
            "tower2_entity_relation": {
                "description": "Final graph units used for graph_text, truncated to match _format_graph_context(max_results=10).",
                "units": [],
            },
            "tower3_episodic": {
                "description": "Final episodic units used to build episodic_context_with_time.",
                "units": [],
            },
        }

        by_layer = hierarchical_context.get("by_layer", {}) if hierarchical_context else {}
        layer_keys = {
            "l0": ["l0_results", "l0_observations"],
            "l1": ["l1_results", "l1_summaries"],
            "l2": ["l2_results", "l2_insights"],
        }
        for layer_name, context_keys in layer_keys.items():
            layer_items = l0_units_for_prompt if layer_name == "l0" and l0_units_for_prompt is not None else []
            if not layer_items:
                for key in context_keys:
                    layer_items = hierarchical_context.get(key, []) if hierarchical_context else []
                    if layer_items:
                        break
            if not layer_items and isinstance(by_layer, dict):
                layer_items = by_layer.get(layer_name.upper(), []) or by_layer.get(layer_name, [])

            layer_reports = []
            for item in layer_items or []:
                unit, score = self._coerce_report_unit_score(item)
                if unit is not None:
                    layer_reports.append(
                        self._extract_content_for_report(unit, score, f"hierarchical_{layer_name}")
                    )
            retrieved_contents["tower1_hierarchical"]["by_layer"][layer_name] = {
                "count": len(layer_reports),
                "units": layer_reports,
            }

        graph_units_for_llm = graph_units[:10]
        for item in graph_units_for_llm:
            unit, score = self._coerce_report_unit_score(item)
            retrieved_contents["tower2_entity_relation"]["units"].append(
                self._extract_content_for_report(unit, score, "entity_relation")
            )
        retrieved_contents["tower2_entity_relation"]["total_before_truncation"] = len(graph_units)
        retrieved_contents["tower2_entity_relation"]["after_truncation"] = len(graph_units_for_llm)

        for item in episodic_units:
            unit, score = self._coerce_report_unit_score(item)
            retrieved_contents["tower3_episodic"]["units"].append(
                self._extract_content_for_report(unit, score, "episodic")
            )
        retrieved_contents["tower3_episodic"]["total_count"] = len(episodic_units)

        if rerank_metadata:
            retrieved_contents["rerank_metadata"] = {
                "strategy_used": rerank_metadata.get("strategy_used", self.rerank_strategy),
                "rerank_time": rerank_metadata.get("rerank_time", 0.0),
                "first_stage_counts": rerank_metadata.get("first_stage_counts", {}),
                "final_counts": rerank_metadata.get("final_counts", {}),
                "cascade_stats": rerank_metadata.get("cascade_stats", {}),
            }

        return retrieved_contents

    def _save_individual_report(
        self,
        result: TriTowerRetrievalResult,
        question_index: int,
        retrieved_contents: Dict[str, Any],
        hierarchical_text: str,
        graph_text: str,
        episodic_context_with_time: str,
        full_prompt: str,
    ) -> None:
        """Save individual report."""
        try:
            reports_dir = self.output_dir / "individual_reports" / result.sample_id
            reports_dir.mkdir(parents=True, exist_ok=True)
            full_prompt_sha256 = hashlib.sha256(full_prompt.encode("utf-8")).hexdigest()
            report_data = {
                "question_index": question_index,
                "sample_id": result.sample_id,
                "question": result.question,
                "category": result.category,
                "expected_answer": result.expected_answer,
                "routing_info": getattr(result, "routing_info", ""),
                "token_stats": {
                    "l0_tokens": result.l0_tokens,
                    "l1_tokens": result.l1_tokens,
                    "l2_tokens": result.l2_tokens,
                    "graph_tokens": result.graph_tokens,
                    "episodic_tokens": result.episodic_tokens,
                    "system_prompt_tokens": result.system_prompt_tokens,
                    "question_tokens": result.question_tokens,
                    "total_input_tokens": result.total_input_tokens,
                    "completion_tokens": getattr(result, "completion_tokens", 0),
                },
                "timing": {
                    "hierarchical_time": result.hierarchical_retrieval_time,
                    "graph_time": result.graph_retrieval_time,
                    "episodic_time": result.episodic_retrieval_time,
                    "total_retrieval_time": result.total_retrieval_time,
                    "second_stage_rerank_time": result.second_stage_rerank_time,
                    "generation_time": result.generation_time,
                    "end_to_end_latency": result.end_to_end_latency,
                },
                "rerank_stats": {
                    "enabled": result.second_stage_rerank_enabled,
                    "method": result.second_stage_rerank_method,
                    "first_stage_counts": {
                        "l0": result.first_stage_l0_count,
                        "graph": result.first_stage_graph_count,
                        "episodic": result.first_stage_episodic_count,
                        "total": result.first_stage_total_count,
                    },
                    "final_counts": {
                        "l0": result.final_l0_count,
                        "graph": result.final_graph_count,
                        "episodic": result.final_episodic_count,
                        "total": result.final_selected_count,
                    },
                },
                "retrieved_contents": retrieved_contents,
                "llm_input": {
                    "hierarchical_text": hierarchical_text,
                    "graph_text": graph_text,
                    "episodic_context_with_time": episodic_context_with_time,
                    "full_prompt": full_prompt,
                    "full_prompt_sha256": full_prompt_sha256,
                },
                "final_result": {
                    "final_answer": result.final_answer,
                    "reasoning_process": result.reasoning_process,
                    "confidence_score": result.confidence_score,
                    "fusion_method": result.fusion_method,
                    "evaluation_scores": result.evaluation_scores,
                    "evaluation_success": result.evaluation_success,
                },
            }
            if hasattr(result, "cascade_pruner_enabled"):
                report_data["cascade_pruning"] = {
                    "enabled": getattr(result, "cascade_pruner_enabled", False),
                    "mode": getattr(result, "cascade_prune_mode", ""),
                    "tokens_used": getattr(result, "cascade_tokens_used", 0),
                    "stage1_input": getattr(result, "cascade_stage1_input", 0),
                    "stage1_output": getattr(result, "cascade_stage1_output", 0),
                    "stage2_conflicts": getattr(result, "cascade_stage2_conflicts", 0),
                    "stage2_dropped": getattr(result, "cascade_stage2_dropped", 0),
                    "stage2_output": getattr(result, "cascade_stage2_output", 0),
                    "stage3_mmr_iterations": getattr(result, "cascade_stage3_mmr_iterations", 0),
                    "stage3_diversity_penalties": getattr(result, "cascade_stage3_diversity_penalties", 0),
                }

            report_file = reports_dir / f"qa_{question_index}_report.json"
            with open(report_file, "w", encoding="utf-8") as f:
                json.dump(self._sanitize_for_json(report_data), f, ensure_ascii=False, indent=2)
            logger.debug(f" 已保存 individual report: {report_file}")
        except Exception as e:
            logger.warning(f"保存 individual report 失败 (sample={result.sample_id}, q={question_index}): {e}")
        
    def _rebuild_hierarchical_text(self, l2_results: List, l1_results: List, l0_results: List) -> str:
        """Rebuild hierarchical text."""
        parts = []
        
        def get_time_str(item):
            t = ""
            if isinstance(item, dict):
                meta = item.get("metadata", {})
                if not meta and "unit" in item and hasattr(item["unit"], "metadata"):
                    meta = item["unit"].metadata or {}
                
                t = meta.get("time_range") or meta.get("timestamp") or meta.get("date")
            
            elif hasattr(item, "metadata") and item.metadata:
                t = item.metadata.get("time_range") or item.metadata.get("timestamp")
            
            return f"[Session: {t}] " if t else ""

        if l2_results:
            parts.append("=== L2 INSIGHTS ===")
            for i, item in enumerate(l2_results, 1):
                content = item.get("content", "") if isinstance(item, dict) else str(item)
                parts.append(f"Insight {i}: {content}")
            parts.append("")
        
        if l1_results:
            parts.append("=== L1 SUMMARIES ===")
            for i, item in enumerate(l1_results, 1):
                content = item.get("content", "") if isinstance(item, dict) else str(item)
                time_str = get_time_str(item)
                parts.append(f"Summary {i}: {time_str}{content}")
            parts.append("")
        
        if l0_results:
            parts.append("=== L0 OBSERVATIONS ===")
            for i, item in enumerate(l0_results, 1):
                content = item.get("content", "") if isinstance(item, dict) else str(item)
                time_str = get_time_str(item)
                parts.append(f"Observation {i}: {time_str}{content}")
            parts.append("")
        
        return "\n".join(parts)
    
    @staticmethod
    def _parse_tower_min_ratio(
        raw: Optional[Dict[str, float]],
    ) -> Optional[Dict[TowerSource, float]]:
        """Parse tower min ratio."""
        if not raw:
            return None
        _key_map = {
            "hierarchical": TowerSource.HIERARCHICAL,
            "h": TowerSource.HIERARCHICAL,
            "episodic": TowerSource.EPISODIC,
            "e": TowerSource.EPISODIC,
            "kg": TowerSource.KG,
            "k": TowerSource.KG,
        }
        result: Dict[TowerSource, float] = {}
        for k, v in raw.items():
            ts = _key_map.get(k.lower())
            if ts is None:
                ts = TowerSource.from_source_type(k)
            result[ts] = float(v)
        return result

    def _initialize_subsystems(self):
        """Initialize subsystems."""
        logger.info(" 初始化三塔子系统...")
        
        
        logger.info("   分层记忆检索 V2 (延迟加载模式)")
        logger.info(f"   分层配置: top_k={self.topk_hierarchical}, 方法={self.hierarchical_retrieval_methods}, 融合={self.hierarchical_fusion_method}")
        
        self.hierarchical_context_builder = HierarchicalContextBuilder()
        
        logger.info("   初始化知识图谱检索子系统...")
        self.graph_benchmark = LoCoMoEntityRelationBenchmark(
            semantic_graphs_dir=str(self.step3_graphs_dir),
            qa_dataset_path=str(self.qa_dataset_path),
            output_dir=str(self.output_dir / "graph_temp"),
            llm_client=self.llm_client,
            llm_evaluate_client=self.llm_evaluate_client,
            use_entity_relation=self.use_entity_relation,
            topk_similarity=self.topk_similarity,
            topk_graph=self.topk_graph,
            reranker_type=self.reranker_type,
            reranker_configs=self.reranker_configs,
            reranker_manager=self.global_reranker_manager
        )
        
        logger.info(" 三塔子系统初始化完成")
    
    def _is_sample_available(self, sample_id: str) -> bool:
        """Run is sample available."""
        hierarchical_path = self.enhanced_graphs_dir / sample_id
        hierarchical_ok = hierarchical_path.exists() and (
            (hierarchical_path / "rx_graph.pkl").exists() or 
            (hierarchical_path / "graph_state.json").exists()
        )
        
        graph_path = self.step3_graphs_dir / sample_id
        
        graph_ok = graph_path.exists() and (
            (graph_path / "rx_graph.pkl").exists() or 
            (graph_path / "graph_state.json").exists()
        )
        
        episodic_path = self.episodic_graphs_dir / sample_id
        episodic_ok = episodic_path.exists() and (
            (episodic_path / "rx_graph.pkl").exists() or 
            (episodic_path / "graph_state.json").exists()
        )
        
        if not hierarchical_ok:
            logger.debug(f"样本 {sample_id} 分层图谱不可用 (检查路径: {hierarchical_path})")
        if not graph_ok:
            logger.debug(f"样本 {sample_id} 知识图谱不可用")
        if not episodic_ok:
            logger.debug(f"样本 {sample_id} 情景记忆图谱不可用")
        
        return hierarchical_ok and graph_ok and episodic_ok
    
    # def _is_sample_available(self, sample_id: str) -> bool:
    
    #     hierarchical_path = self.enhanced_graphs_dir / sample_id
    #     hierarchical_ok = hierarchical_path.exists() and (hierarchical_path / "hierarchical_overview.json").exists()
        
    #     graph_path = self.step3_graphs_dir / sample_id
    #     graph_ok = graph_path.exists() and (graph_path / "semantic_map_data" / "semantic_map.json").exists()
        
    #     episodic_path = self.episodic_graphs_dir / sample_id
    #     episodic_ok = episodic_path.exists()
        
    #     if not hierarchical_ok:
    #     if not graph_ok:
    #     if not episodic_ok:
        
    #     return hierarchical_ok and graph_ok and episodic_ok
    
    def load_systems(self, sample_id: str):
        """Load systems."""
        logger.info(f" 加载样本 {sample_id} 的三塔系统...")
        
        if not self._is_sample_available(sample_id):
            raise RuntimeError(f"样本 {sample_id} 在一个或多个塔中不可用")
        
        
        self._load_hierarchical_system(sample_id)
        
        
        self._load_graph_system(sample_id)
        
        
        self._load_episodic_system(sample_id)
        
        self.available_samples.append(sample_id)
        self.stats["total_samples_loaded"] += 1
        
        logger.info(f" 样本 {sample_id} 三塔系统加载完成")
    
    def _load_hierarchical_system(self, sample_id: str):
        """Load hierarchical system."""
        try:
            logger.info(f"   加载 {sample_id} 分层记忆系统...")
            
            
            graph_dir = self.enhanced_graphs_dir / sample_id
            semantic_graph = SemanticGraph.load_graph(str(graph_dir))
            self.hierarchical_graphs[sample_id] = semantic_graph
            
            retriever = UnifiedHierarchicalRetriever(
                semantic_graph=semantic_graph,
                config=self.hierarchical_config,
                reranker_manager=self.global_reranker_manager
            )
            self.hierarchical_retrievers[sample_id] = retriever
            
            logger.info(f"   {sample_id} 分层记忆加载完成")
        except Exception as e:
            logger.error(f"   加载 {sample_id} 分层记忆失败: {e}")
            raise
    
    def _load_graph_system(self, sample_id: str):
        """Load graph system."""
        try:
            logger.info(f"   加载 {sample_id} 知识图谱系统...")
            
            self.graph_benchmark.target_sample_ids = {sample_id}
            self.graph_benchmark.load_semantic_graphs()
            logger.info(f"   {sample_id} 知识图谱加载完成")
        except Exception as e:
            logger.error(f"   加载 {sample_id} 知识图谱失败: {e}")
            raise
    
    def _load_episodic_system(self, sample_id: str):
        """Load episodic system."""
        try:
            logger.info(f"   加载 {sample_id} 情景记忆系统...")
            
            graph_dir = self.episodic_graphs_dir / sample_id
            semantic_graph = SemanticGraph.load_graph(str(graph_dir))
            self.episodic_graphs[sample_id] = semantic_graph
            
            multi_retriever = MultiRetriever(
                retrieval_source=semantic_graph,
                preload_rerankers=False,
                reranker_configs=self.reranker_configs,
                reranker_manager=self.global_reranker_manager
            )
            
            
            required_methods = [
                RetrievalMethod.BM25,
                RetrievalMethod.COSINE_SIMILARITY,
                RetrievalMethod.SPLADE
            ]
            
            build_stats = multi_retriever.build_all_indexes(
                methods_to_build=required_methods
            )
            
            logger.info(f"     索引构建: 成功={build_stats['built_count']}, "
                       f"跳过={build_stats['skipped_count']}, "
                       f"失败={build_stats['failed_count']}")
            
            self.episodic_retrievers[sample_id] = multi_retriever
            logger.info(f"   {sample_id} 情景记忆加载完成")
            
        except Exception as e:
            logger.error(f"   加载 {sample_id} 情景记忆失败: {e}")
            raise
    
    def unload_sample(self, sample_id: str):
        """Run unload sample."""
        logger.info(f" 清理样本 {sample_id} 的资源...")
        
        if sample_id in self.hierarchical_retrievers:
            retriever = self.hierarchical_retrievers[sample_id]
            if hasattr(retriever, 'clear_cache'):
                retriever.clear_cache()
            del self.hierarchical_retrievers[sample_id]
            logger.debug(f"   清理分层检索器")
        
        if sample_id in self.hierarchical_graphs:
            del self.hierarchical_graphs[sample_id]
            logger.debug(f"   清理分层图")
        
        if sample_id in self.graph_benchmark.multi_retrievers:
            retriever = self.graph_benchmark.multi_retrievers[sample_id]
            if hasattr(retriever, 'clear_cache'):
                retriever.clear_cache()
            del self.graph_benchmark.multi_retrievers[sample_id]
            logger.debug(f"   清理图检索器")
        
        if sample_id in self.graph_benchmark.entity_relation_retrievers:
            del self.graph_benchmark.entity_relation_retrievers[sample_id]
            logger.debug(f"   清理实体关系检索器")
        
        if sample_id in self.graph_benchmark.semantic_graphs:
            del self.graph_benchmark.semantic_graphs[sample_id]
            logger.debug(f"   清理语义图")
        
        if sample_id in self.episodic_retrievers:
            retriever = self.episodic_retrievers[sample_id]
            if hasattr(retriever, 'clear_cache'):
                retriever.clear_cache()
            del self.episodic_retrievers[sample_id]
            logger.debug(f"   清理情景记忆检索器")
        
        if sample_id in self.episodic_graphs:
            del self.episodic_graphs[sample_id]
            logger.debug(f"   清理情景记忆图")
        
        if sample_id in self.available_samples:
            self.available_samples.remove(sample_id)
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug(f"   清理GPU缓存")
        except ImportError:
            pass
        
        gc.collect()
        logger.debug(f"   执行垃圾回收")
        
        logger.info(f" 样本 {sample_id} 资源清理完成")
    
    def load_test_cases(self, sample_id: str):
        """Load test cases."""
        logger.info(f" 加载样本 {sample_id} 的测试用例...")
        
        self.test_cases = []
        
        try:
            with open(self.qa_dataset_path, 'r', encoding='utf-8') as f:
                qa_data = json.load(f)
            
            for item in qa_data:
                item_sample_id = item["sample_id"]
                
                if item_sample_id != sample_id:
                    continue
                
                qa_list = item.get("qa", [])
                for i, qa_item in enumerate(qa_list):
                    if not isinstance(qa_item, dict) or "question" not in qa_item:
                        continue
                    
                    question = qa_item["question"]
                    category = qa_item.get("category", 1)
                    
                    expected_answer = ""
                    if "answer" in qa_item and qa_item["answer"]:
                        expected_answer = qa_item["answer"]
                    elif "adversarial_answer" in qa_item:
                        expected_answer = qa_item["adversarial_answer"]
                        category = 5
                    else:
                        continue
                    
                    test_case = {
                        "sample_id": item_sample_id,
                        "question_index": i,
                        "question": question,
                        "category": category,
                        "expected_answer": expected_answer,
                        "question_id": f"{item_sample_id}_q{i+1}",
                        "evidence": qa_item.get("evidence", [])
                    }
                    
                    self.test_cases.append(test_case)
            
            logger.info(f" 样本 {sample_id} 测试用例加载完成: {len(self.test_cases)} 个")
            
        except Exception as e:
            logger.error(f"加载测试用例失败: {e}")
            raise
    
    def run_tri_tower_benchmark(self, sequential_mode: bool = True):
        """Run tri tower benchmark."""
        logger.info(" 开始运行三塔召回benchmark测试...")
        
        if not sequential_mode:
            logger.warning("  非顺序模式可能导致内存不足，建议使用sequential_mode=True")
        
        if self.target_sample_ids:
            samples_to_test = self.target_sample_ids
        else:
            samples_to_test = self._get_available_samples()
        
        logger.info(f" 共需测试 {len(samples_to_test)} 个样本")
        logger.info(f" 测试顺序: {samples_to_test}")
        
        self._initialize_progress_tracking(samples_to_test)
        
        if samples_to_test:
            logger.info(f"\n{'='*80}")
            logger.info(" [System Warmup] 正在进行系统预热...")
            logger.info(f"{'='*80}")
            try:
                self.run_single_sample_benchmark(samples_to_test[0], is_warmup=True)
                logger.info(" [System Warmup] 预热完成")
            except Exception as e:
                logger.warning(f"  [System Warmup] 预热失败: {e}，继续正式测试...")
            logger.info(f"{'='*80}\n")
        
        for i, sample_id in enumerate(samples_to_test, 1):
            logger.info(f"\n{'='*80}")
            logger.info(f" [{i}/{len(samples_to_test)}] 处理样本: {sample_id}")
            logger.info(f"{'='*80}")
            
            try:
                sample_results = self.run_single_sample_benchmark(sample_id)
                logger.info(f" 样本 {sample_id} 完成，获得 {len(sample_results)} 个结果")
                self._mark_sample_successful(sample_id, len(sample_results))
            except Exception as e:
                logger.error(f" 样本 {sample_id} 处理失败: {e}")
                self._mark_sample_failed(sample_id, str(e))
                continue
        
        logger.info(f"\n{'='*80}")
        logger.info(" 所有样本测试完成")
        logger.info(f"{'='*80}")
        
        self.generate_final_summary()
    
    def run_single_sample_benchmark(self, sample_id: str, is_warmup: bool = False) -> List[TriTowerRetrievalResult]:
        """Run single sample benchmark."""
        if is_warmup:
            logger.info(f" [Warmup] 开始预热样本: {sample_id}")
        else:
            logger.info(f" 开始测试样本: {sample_id}")
        
        sample_results = []
        previous_report_suppression = getattr(self, "_suppress_individual_reports", False)
        self._suppress_individual_reports = is_warmup
        
        try:
            
            logger.info(f" 加载样本 {sample_id} 的三塔系统...")
            self.load_systems(sample_id)
            
            
            logger.info(f" 加载样本 {sample_id} 的测试用例...")
            self.load_test_cases(sample_id)
            
            if not self.test_cases:
                logger.warning(f"  样本 {sample_id} 没有测试用例")
                return sample_results
            
            if is_warmup:
                self.test_cases = self.test_cases[:1]
                logger.info(f" [Warmup] 仅使用 1 个测试用例进行预热")
            elif self.max_questions is not None:
                original_count = len(self.test_cases)
                self.test_cases = self.test_cases[:self.max_questions]
                logger.info(
                    f" 样本 {sample_id} 使用 {len(self.test_cases)}/{original_count} 个测试用例 "
                    f"(--max-questions={self.max_questions})"
                )
            else:
                logger.info(f" 样本 {sample_id} 共有 {len(self.test_cases)} 个测试用例")
            
            test_iterator = self.test_cases if is_warmup else tqdm(self.test_cases, desc=f"测试 {sample_id}")
            for test_case in test_iterator:
                try:
                    result = self._run_single_tri_tower_test(test_case)
                    if result:
                        sample_results.append(result)
                        if not is_warmup:
                            self.test_results.append(result)
                            if result.evaluation_success:
                                self.stats["successful_tri_tower"] += 1
                            else:
                                self.stats["failed_retrievals"] += 1
                    else:
                        if not is_warmup:
                            self.stats["failed_retrievals"] += 1
                        
                except Exception as e:
                    if not is_warmup:
                        self.stats["failed_retrievals"] += 1
                    logger.error(f" 测试失败: {test_case['question_id']} - {e}")
                    continue
            
            if is_warmup:
                logger.info(f" [Warmup] 样本 {sample_id} 预热完成")
            else:
                logger.info(f" 样本 {sample_id} 测试完成: "
                           f"成功={len([r for r in sample_results if r.evaluation_success])}, "
                           f"失败={len([r for r in sample_results if not r.evaluation_success])}")
            
            
            if not is_warmup:
                self._save_sample_results(sample_id, sample_results)
            
            return sample_results
            
        except Exception as e:
            logger.error(f" 样本 {sample_id} 测试失败: {e}")
            logger.error(traceback.format_exc())
            return sample_results
        
        finally:
            self._suppress_individual_reports = previous_report_suppression
            logger.info(f" 清理样本 {sample_id} 的资源...")
            self.unload_sample(sample_id)
    
    @contextmanager
    def _routing_context(self, config: 'TowerRoutingConfig'):
        """Run routing context."""
        orig = {
            'topk_hierarchical': self.topk_hierarchical,
            'topk_similarity': self.topk_similarity,
            'topk_graph': self.topk_graph,
            'topk_episodic': self.topk_episodic,
            'use_entity_relation': self.use_entity_relation,
            'final_top_k': self.final_top_k,
            'fusion_weights': self.fusion_weights.copy(),
        }
        try:
            self.topk_hierarchical = config.topk_hierarchical
            self.topk_similarity = config.topk_similarity
            self.topk_graph = config.topk_graph
            self.topk_episodic = config.topk_episodic
            self.use_entity_relation = config.use_entity_relation
            self.final_top_k = config.final_top_k
            self.fusion_weights = {
                "hierarchical": config.weight_hierarchical,
                "graph": config.weight_graph,
                "episodic": config.weight_episodic,
            }
            yield config
        finally:
            self.topk_hierarchical = orig['topk_hierarchical']
            self.topk_similarity = orig['topk_similarity']
            self.topk_graph = orig['topk_graph']
            self.topk_episodic = orig['topk_episodic']
            self.use_entity_relation = orig['use_entity_relation']
            self.final_top_k = orig['final_top_k']
            self.fusion_weights = orig['fusion_weights']
    
    def _dispatch_tri_tower_test(self, sample_id: str, question: str,
                                 category: int, expected_answer: str,
                                 question_index: int = 0,
                                 routing_info: str = "") -> Optional[TriTowerRetrievalResult]:
        """Run dispatch tri tower test."""
        if self.parallel_towers:
            return self._run_parallel_tri_tower_test(
                sample_id, question, category, expected_answer,
                question_index=question_index,
                routing_info=routing_info,
            )
        else:
            return self._run_sequential_tri_tower_test(
                sample_id, question, category, expected_answer,
                question_index=question_index,
                routing_info=routing_info,
            )
    
    def _run_single_tri_tower_test(self, test_case: Dict[str, Any]) -> Optional[TriTowerRetrievalResult]:
        """Run single tri tower test."""
        sample_id = test_case["sample_id"]
        question = test_case["question"]
        category = test_case["category"]
        expected_answer = test_case["expected_answer"]
        question_index = test_case.get("question_index", 0)
        
        logger.debug(f" 三塔测试: {sample_id} - {question[:50]}...")
        
        try:
            
            routing_info = ""
            if self.tower_router:
                config = self.tower_router.route(question, category, enable_guidance=False)
                routing_info = (f"routed: {config.active_towers} "
                               f"({self.tower_router.strategy}, Cat{category})")
                
                h_passthrough = config.topk_hierarchical if config.topk_hierarchical > 0 else 0
                expected_budget = h_passthrough + config.final_top_k
                logger.info(f" 路由决策: Cat{category} → {config.active_towers} "
                           f"(topk: H={config.topk_hierarchical}, "
                           f"G_sim={config.topk_similarity}, "
                           f"E={config.topk_episodic}, "
                           f"final_k={config.final_top_k}, "
                           f"budget≈{expected_budget})")
                with self._routing_context(config):
                    result = self._dispatch_tri_tower_test(
                        sample_id, question, category, expected_answer,
                        question_index=question_index,
                        routing_info=routing_info,
                    )
            else:
                result = self._dispatch_tri_tower_test(
                    sample_id, question, category, expected_answer,
                    question_index=question_index,
                )
            
            
            if result and routing_info:
                result.routing_info = routing_info
            
            return result
                
        except Exception as e:
            logger.error(f"三塔测试执行失败 {sample_id}: {e}")
            logger.debug(traceback.format_exc())
            return None
    
    def _run_parallel_tri_tower_test(self,
                                     sample_id: str,
                                     question: str,
                                     category: int,
                                     expected_answer: str,
                                     question_index: int = 0,
                                     routing_info: str = "") -> Optional[TriTowerRetrievalResult]:
        """Run parallel tri tower test."""
        retrieval_start = time.perf_counter()
        
        hierarchical_result = {}
        graph_result = {}
        episodic_result = {}
        
        def run_hierarchical():
            """Run hierarchical."""
            try:
                start_time = time.time()
                context = self._run_hierarchical_retrieval(sample_id, question, category)
                elapsed = time.time() - start_time
                
                hierarchical_result['context'] = context
                hierarchical_result['time'] = elapsed
                hierarchical_result['success'] = context.get("hierarchical_enabled", False)
                
                if hierarchical_result['success']:
                    self.stats["successful_hierarchical"] += 1
                    
            except Exception as e:
                logger.error(f"并行分层检索失败 {sample_id}: {e}")
                hierarchical_result['context'] = {
                    "hierarchical_enabled": False,
                    "error": str(e),
                    "hierarchical_context_text": ""
                }
                hierarchical_result['time'] = 0.0
                hierarchical_result['success'] = False
        
        def run_graph():
            """Run graph."""
            try:
                start_time = time.time()
                units, details = self._run_graph_retrieval(sample_id, question)
                elapsed = time.time() - start_time
                
                graph_result['units'] = units
                graph_result['details'] = details
                graph_result['time'] = elapsed
                graph_result['success'] = len(units) > 0
                
                if graph_result['success']:
                    self.stats["successful_graph"] += 1
                    
            except Exception as e:
                logger.error(f"并行图检索失败 {sample_id}: {e}")
                graph_result['units'] = []
                graph_result['details'] = {"method": "failed", "error": str(e)}
                graph_result['time'] = 0.0
                graph_result['success'] = False
        
        def run_episodic():
            """Run episodic."""
            try:
                start_time = time.time()
                units, context_with_time = self._run_episodic_retrieval(sample_id, question)
                elapsed = time.time() - start_time
                
                episodic_result['units'] = units
                episodic_result['context_with_time'] = context_with_time
                episodic_result['time'] = elapsed
                episodic_result['success'] = len(units) > 0
                
                if episodic_result['success']:
                    self.stats["successful_episodic"] += 1
                    
            except Exception as e:
                logger.error(f"并行情景记忆检索失败 {sample_id}: {e}")
                episodic_result['units'] = []
                episodic_result['context_with_time'] = ""
                episodic_result['time'] = 0.0
                episodic_result['success'] = False
        
        
        parallel_start = time.time()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_hierarchical = executor.submit(run_hierarchical)
            future_graph = executor.submit(run_graph)
            future_episodic = executor.submit(run_episodic)
            
            for future in as_completed([future_hierarchical, future_graph, future_episodic]):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"并行检索任务异常: {e}")
        
        parallel_elapsed = time.time() - parallel_start
        
        if ('context' not in hierarchical_result or 
            'units' not in graph_result or 
            'units' not in episodic_result):
            logger.error(f"并行检索未完成")
            return None
        
        hierarchical_context = hierarchical_result['context']
        hierarchical_time = hierarchical_result['time']
        
        graph_units = graph_result['units']
        graph_details = graph_result['details']
        graph_time = graph_result['time']
        
        episodic_units = episodic_result['units']
        episodic_context_with_time = episodic_result['context_with_time']
        episodic_time = episodic_result['time']
        
        sequential_time = hierarchical_time + graph_time + episodic_time
        speedup_ratio = sequential_time / parallel_elapsed if parallel_elapsed > 0 else 0
        time_saved = sequential_time - parallel_elapsed
        
        logger.debug(f" 并行检索完成: 分层={hierarchical_time:.3f}s, "
                    f"图={graph_time:.3f}s, 情景={episodic_time:.3f}s")
        logger.debug(f"   并行总耗时={parallel_elapsed:.3f}s, 理论串行={sequential_time:.3f}s")
        logger.debug(f"   加速比={speedup_ratio:.2f}x, 节省时间={time_saved:.3f}s")
        
        
        second_stage_rerank_time = 0.0
        first_stage_l0_count = 0
        first_stage_graph_count = len(graph_units)
        first_stage_episodic_count = len(episodic_units)
        final_l0_count = 0
        final_graph_count = len(graph_units)
        final_episodic_count = len(episodic_units)
        cascade_stats = {}
        rerank_result = None
        
        
        l0_units = self._extract_l0_units_from_hierarchical(hierarchical_context)
        first_stage_l0_count = len(l0_units)
        final_l0_count = first_stage_l0_count
        
        
        if self.cascade_pruner is not None and (l0_units or graph_units or episodic_units):
            rerank_result = self._perform_cascade_rerank(
                question, l0_units, graph_units, episodic_units,
                category=category,
            )
            second_stage_rerank_time = rerank_result["rerank_time"]
            cascade_stats = rerank_result.get("cascade_stats", {})
            
            l0_units = rerank_result["reranked_l0"]
            graph_units = rerank_result["reranked_graph"]
            episodic_units = rerank_result["reranked_episodic"]
            
            final_l0_count = len(l0_units)
            final_graph_count = len(graph_units)
            final_episodic_count = len(episodic_units)
            
            episodic_context_with_time = self._build_episodic_context_with_time(episodic_units)
            
            logger.debug(f" 级联量化: "
                        f"L0: {first_stage_l0_count}->{final_l0_count}, "
                        f"Graph: {first_stage_graph_count}->{final_graph_count}, "
                        f"Episodic: {first_stage_episodic_count}->{final_episodic_count}")
        elif self.enable_second_stage_rerank and (l0_units or graph_units or episodic_units):
            rerank_start_inner = time.time()
            
            if self.rerank_strategy == "unified_rerank":
                
                rerank_result = self._perform_second_stage_rerank(
                    question, l0_units, graph_units, episodic_units
                )
                
                l0_units = rerank_result["reranked_l0"]
                final_l0_count = len(l0_units)
            else:  # tower_separate
                
                rerank_result = self._perform_second_stage_rerank(
                    question, [], graph_units, episodic_units
                )
            
            second_stage_rerank_time = rerank_result["rerank_time"]
            
            graph_units = rerank_result["reranked_graph"]
            episodic_units = rerank_result["reranked_episodic"]
            
            final_graph_count = len(graph_units)
            final_episodic_count = len(episodic_units)
            
            episodic_context_with_time = self._build_episodic_context_with_time(episodic_units)
            
            strategy_desc = "统一重排序" if self.rerank_strategy == "unified_rerank" else "分层塔直通车"
            logger.debug(f" 二次重排序({strategy_desc}): "
                        f"L0: {first_stage_l0_count}->{final_l0_count}, "
                        f"Graph: {first_stage_graph_count}->{final_graph_count}, "
                        f"Episodic: {first_stage_episodic_count}->{final_episodic_count}")
        
        
        if final_l0_count != first_stage_l0_count:
            pruned_l0_items = [unit for unit, score in l0_units] if l0_units else []
            l1_items = (hierarchical_context.get("l1_results") or 
                        hierarchical_context.get("l1_summaries") or 
                        hierarchical_context.get("by_layer", {}).get("L1", []))
            l2_items = (hierarchical_context.get("l2_results") or 
                        hierarchical_context.get("l2_insights") or 
                        hierarchical_context.get("by_layer", {}).get("L2", []))
            hierarchical_text = self._rebuild_hierarchical_text(l2_items, l1_items, pruned_l0_items)
            if not hierarchical_text:
                hierarchical_text = "No hierarchical context available after pruning"
        else:
            hierarchical_text = hierarchical_context.get("hierarchical_context_text", "")
            if not hierarchical_text:
                hierarchical_text = self._format_hierarchical_context(hierarchical_context)
        graph_text = self._format_graph_context(graph_units)
        
        hierarchical_layer_tokens = self._count_hierarchical_tokens_by_layer(hierarchical_context)
        l1_tokens = hierarchical_layer_tokens["l1_tokens"]
        l2_tokens = hierarchical_layer_tokens["l2_tokens"]
        if final_l0_count != first_stage_l0_count:
            total_h_tokens = self._count_tokens(hierarchical_text)
            l0_tokens = max(0, total_h_tokens - l1_tokens - l2_tokens)
        else:
            l0_tokens = hierarchical_layer_tokens["l0_tokens"]
        graph_tokens = self._count_tokens(graph_text)
        episodic_tokens = self._count_tokens(episodic_context_with_time)
        question_tokens = self._count_tokens(question)
        
        full_prompt, token_info = self._build_full_prompt(
            question, category,
            hierarchical_context, hierarchical_text,
            graph_units, graph_text,
            episodic_context_with_time
        )
        
        
        retrieval_end = time.perf_counter()
        actual_retrieval_time = retrieval_end - retrieval_start
        
        
        fusion_start = time.time()
        answer_dict, confidence_score, _ = self._fuse_and_generate_answer_with_tokens(
            question, category, 
            hierarchical_context, hierarchical_text,
            graph_units, graph_details, graph_text,
            episodic_units, episodic_context_with_time,
            pre_built_prompt=full_prompt,
            pre_built_token_info=token_info
        )
        fusion_time = time.time() - fusion_start
        
        end_to_end_latency = actual_retrieval_time + fusion_time
        
        final_answer = answer_dict.get("final_answer", "")
        reasoning_process = answer_dict.get("reasoning", "")
        
        system_prompt_tokens = token_info.get("system_prompt_tokens", 0)
        total_input_tokens = token_info.get("total_input_tokens", 0)
        completion_tokens = self._count_tokens(final_answer + reasoning_process)
        
        evaluation_result = self._evaluate_tri_tower_result(
            question=question,
            expected_answer=expected_answer,
            generated_answer=final_answer,
            reasoning=reasoning_process,
            category=category
        )
        
        evaluation_scores = evaluation_result.get("evaluation_scores", {})
        evaluation_success = evaluation_result.get("evaluation_success", False)
        
        
        result = TriTowerRetrievalResult(
            sample_id=sample_id,
            question=question,
            category=category,
            expected_answer=expected_answer,
            hierarchical_context=hierarchical_context,
            hierarchical_retrieval_time=hierarchical_time,
            graph_retrieved_units=graph_units,
            graph_retrieval_time=graph_time,
            graph_retrieval_details=graph_details,
            episodic_retrieved_units=episodic_units,
            episodic_retrieval_time=episodic_time,
            episodic_context_with_time=episodic_context_with_time,
            final_answer=final_answer,
            reasoning_process=reasoning_process,
            confidence_score=confidence_score,
            fusion_method=self.fusion_strategy,
            generation_time=fusion_time,
            evaluation_scores=evaluation_scores,
            evaluation_success=evaluation_success,
            l0_tokens=l0_tokens,
            l1_tokens=l1_tokens,
            l2_tokens=l2_tokens,
            graph_tokens=graph_tokens,
            episodic_tokens=episodic_tokens,
            system_prompt_tokens=system_prompt_tokens,
            question_tokens=question_tokens,
            total_input_tokens=total_input_tokens,
            completion_tokens=completion_tokens,
            hierarchical_text=hierarchical_text,
            graph_text=graph_text,
            total_retrieval_time=actual_retrieval_time,
            end_to_end_latency=end_to_end_latency,
            
            second_stage_rerank_enabled=self.enable_second_stage_rerank,
            second_stage_rerank_method=self.second_stage_rerank_method if self.enable_second_stage_rerank else "none",
            second_stage_rerank_time=second_stage_rerank_time,
            first_stage_l0_count=first_stage_l0_count,
            first_stage_graph_count=first_stage_graph_count,
            first_stage_episodic_count=first_stage_episodic_count,
            first_stage_total_count=first_stage_l0_count + first_stage_graph_count + first_stage_episodic_count,
            final_l0_count=final_l0_count,
            final_graph_count=final_graph_count,
            final_episodic_count=final_episodic_count,
            final_selected_count=final_l0_count + final_graph_count + final_episodic_count,
            parallel_actual_time=parallel_elapsed,
            parallel_sequential_theory=sequential_time,
            parallel_speedup_ratio=speedup_ratio,
            parallel_time_saved=time_saved,
            cascade_pruner_enabled=self.cascade_pruner is not None,
            cascade_prune_mode=cascade_stats.get("mode", ""),
            cascade_tokens_used=cascade_stats.get("tokens_used", 0),
            cascade_stage1_input=cascade_stats.get("cascade_stage1_input", 0),
            cascade_stage1_output=cascade_stats.get("cascade_stage1_output", 0),
            cascade_stage2_conflicts=cascade_stats.get("cascade_stage2_conflicts", 0),
            cascade_stage2_dropped=cascade_stats.get("cascade_stage2_dropped", 0),
            cascade_stage2_output=cascade_stats.get("cascade_stage2_output", 0),
            cascade_stage3_mmr_iterations=cascade_stats.get("cascade_stage3_mmr_iterations", 0),
            cascade_stage3_diversity_penalties=cascade_stats.get("cascade_stage3_diversity_penalties", 0),
            routing_info=routing_info,
        )

        if self.save_individual_reports and not getattr(self, "_suppress_individual_reports", False):
            retrieved_contents = self._collect_retrieved_contents_for_report(
                hierarchical_context,
                graph_units,
                episodic_units,
                rerank_metadata=rerank_result,
                l0_units_for_prompt=l0_units if final_l0_count != first_stage_l0_count else None,
            )
            self._save_individual_report(
                result,
                question_index,
                retrieved_contents,
                hierarchical_text,
                graph_text,
                episodic_context_with_time,
                full_prompt,
            )
        
        return result
    
    def _run_sequential_tri_tower_test(self,
                                       sample_id: str,
                                       question: str,
                                       category: int,
                                       expected_answer: str,
                                       question_index: int = 0,
                                       routing_info: str = "") -> Optional[TriTowerRetrievalResult]:
        """Run sequential tri tower test."""
        retrieval_start = time.perf_counter()
        
        hierarchical_start = time.time()
        hierarchical_context = self._run_hierarchical_retrieval(sample_id, question, category)
        hierarchical_time = time.time() - hierarchical_start
        
        if hierarchical_context.get("hierarchical_enabled", False):
            self.stats["successful_hierarchical"] += 1
        
        graph_start = time.time()
        graph_units, graph_details = self._run_graph_retrieval(sample_id, question)
        graph_time = time.time() - graph_start
        
        if len(graph_units) > 0:
            self.stats["successful_graph"] += 1
        
        episodic_start = time.time()
        episodic_units, episodic_context_with_time = self._run_episodic_retrieval(sample_id, question)
        episodic_time = time.time() - episodic_start
        
        if len(episodic_units) > 0:
            self.stats["successful_episodic"] += 1
        
        logger.debug(f" 串行检索完成: 分层={hierarchical_time:.3f}s, "
                    f"图={graph_time:.3f}s, 情景={episodic_time:.3f}s")
        
        
        second_stage_rerank_time = 0.0
        first_stage_l0_count = 0
        first_stage_graph_count = len(graph_units)
        first_stage_episodic_count = len(episodic_units)
        final_l0_count = 0
        final_graph_count = len(graph_units)
        final_episodic_count = len(episodic_units)
        cascade_stats = {}
        rerank_result = None
        
        
        l0_units = self._extract_l0_units_from_hierarchical(hierarchical_context)
        first_stage_l0_count = len(l0_units)
        final_l0_count = first_stage_l0_count
        
        
        if self.cascade_pruner is not None and (l0_units or graph_units or episodic_units):
            rerank_result = self._perform_cascade_rerank(
                question, l0_units, graph_units, episodic_units,
                category=category,
            )
            second_stage_rerank_time = rerank_result["rerank_time"]
            cascade_stats = rerank_result.get("cascade_stats", {})
            
            l0_units = rerank_result["reranked_l0"]
            graph_units = rerank_result["reranked_graph"]
            episodic_units = rerank_result["reranked_episodic"]
            
            final_l0_count = len(l0_units)
            final_graph_count = len(graph_units)
            final_episodic_count = len(episodic_units)
            
            episodic_context_with_time = self._build_episodic_context_with_time(episodic_units)
            
            logger.debug(f" 级联量化: "
                        f"L0: {first_stage_l0_count}->{final_l0_count}, "
                        f"Graph: {first_stage_graph_count}->{final_graph_count}, "
                        f"Episodic: {first_stage_episodic_count}->{final_episodic_count}")
        elif self.enable_second_stage_rerank and (l0_units or graph_units or episodic_units):
            rerank_start_inner = time.time()
            
            if self.rerank_strategy == "unified_rerank":
                
                rerank_result = self._perform_second_stage_rerank(
                    question, l0_units, graph_units, episodic_units
                )
                
                l0_units = rerank_result["reranked_l0"]
                final_l0_count = len(l0_units)
            else:  # tower_separate
                
                rerank_result = self._perform_second_stage_rerank(
                    question, [], graph_units, episodic_units
                )
            
            second_stage_rerank_time = rerank_result["rerank_time"]
            
            graph_units = rerank_result["reranked_graph"]
            episodic_units = rerank_result["reranked_episodic"]
            
            final_graph_count = len(graph_units)
            final_episodic_count = len(episodic_units)
            
            episodic_context_with_time = self._build_episodic_context_with_time(episodic_units)
            
            strategy_desc = "统一重排序" if self.rerank_strategy == "unified_rerank" else "分层塔直通车"
            logger.debug(f" 二次重排序({strategy_desc}): "
                        f"L0: {first_stage_l0_count}->{final_l0_count}, "
                        f"Graph: {first_stage_graph_count}->{final_graph_count}, "
                        f"Episodic: {first_stage_episodic_count}->{final_episodic_count}")
        
        
        if final_l0_count != first_stage_l0_count:
            pruned_l0_items = [unit for unit, score in l0_units] if l0_units else []
            l1_items = (hierarchical_context.get("l1_results") or 
                        hierarchical_context.get("l1_summaries") or 
                        hierarchical_context.get("by_layer", {}).get("L1", []))
            l2_items = (hierarchical_context.get("l2_results") or 
                        hierarchical_context.get("l2_insights") or 
                        hierarchical_context.get("by_layer", {}).get("L2", []))
            hierarchical_text = self._rebuild_hierarchical_text(l2_items, l1_items, pruned_l0_items)
            if not hierarchical_text:
                hierarchical_text = "No hierarchical context available after pruning"
        else:
            hierarchical_text = hierarchical_context.get("hierarchical_context_text", "")
            if not hierarchical_text:
                hierarchical_text = self._format_hierarchical_context(hierarchical_context)
        graph_text = self._format_graph_context(graph_units)
        
        hierarchical_layer_tokens = self._count_hierarchical_tokens_by_layer(hierarchical_context)
        l1_tokens = hierarchical_layer_tokens["l1_tokens"]
        l2_tokens = hierarchical_layer_tokens["l2_tokens"]
        if final_l0_count != first_stage_l0_count:
            total_h_tokens = self._count_tokens(hierarchical_text)
            l0_tokens = max(0, total_h_tokens - l1_tokens - l2_tokens)
        else:
            l0_tokens = hierarchical_layer_tokens["l0_tokens"]
        graph_tokens = self._count_tokens(graph_text)
        episodic_tokens = self._count_tokens(episodic_context_with_time)
        question_tokens = self._count_tokens(question)
        
        full_prompt, token_info = self._build_full_prompt(
            question, category,
            hierarchical_context, hierarchical_text,
            graph_units, graph_text,
            episodic_context_with_time
        )
        
        
        retrieval_end = time.perf_counter()
        actual_retrieval_time = retrieval_end - retrieval_start
        
        
        fusion_start = time.time()
        answer_dict, confidence_score, _ = self._fuse_and_generate_answer_with_tokens(
            question, category,
            hierarchical_context, hierarchical_text,
            graph_units, graph_details, graph_text,
            episodic_units, episodic_context_with_time,
            pre_built_prompt=full_prompt,
            pre_built_token_info=token_info
        )
        fusion_time = time.time() - fusion_start
        
        end_to_end_latency = actual_retrieval_time + fusion_time
        
        final_answer = answer_dict.get("final_answer", "")
        reasoning_process = answer_dict.get("reasoning", "")
        
        system_prompt_tokens = token_info.get("system_prompt_tokens", 0)
        total_input_tokens = token_info.get("total_input_tokens", 0)
        completion_tokens = self._count_tokens(final_answer + reasoning_process)
        
        evaluation_result = self._evaluate_tri_tower_result(
            question=question,
            expected_answer=expected_answer,
            generated_answer=final_answer,
            reasoning=reasoning_process,
            category=category
        )
        
        evaluation_scores = evaluation_result.get("evaluation_scores", {})
        evaluation_success = evaluation_result.get("evaluation_success", False)
        
        result = TriTowerRetrievalResult(
            sample_id=sample_id,
            question=question,
            category=category,
            expected_answer=expected_answer,
            hierarchical_context=hierarchical_context,
            hierarchical_retrieval_time=hierarchical_time,
            graph_retrieved_units=graph_units,
            graph_retrieval_time=graph_time,
            graph_retrieval_details=graph_details,
            episodic_retrieved_units=episodic_units,
            episodic_retrieval_time=episodic_time,
            episodic_context_with_time=episodic_context_with_time,
            final_answer=final_answer,
            reasoning_process=reasoning_process,
            confidence_score=confidence_score,
            fusion_method=self.fusion_strategy,
            generation_time=fusion_time,
            evaluation_scores=evaluation_scores,
            evaluation_success=evaluation_success,
            l0_tokens=l0_tokens,
            l1_tokens=l1_tokens,
            l2_tokens=l2_tokens,
            graph_tokens=graph_tokens,
            episodic_tokens=episodic_tokens,
            system_prompt_tokens=system_prompt_tokens,
            question_tokens=question_tokens,
            total_input_tokens=total_input_tokens,
            completion_tokens=completion_tokens,
            hierarchical_text=hierarchical_text,
            graph_text=graph_text,
            total_retrieval_time=actual_retrieval_time,
            end_to_end_latency=end_to_end_latency,
            
            second_stage_rerank_enabled=self.enable_second_stage_rerank,
            second_stage_rerank_method=self.second_stage_rerank_method if self.enable_second_stage_rerank else "none",
            second_stage_rerank_time=second_stage_rerank_time,
            first_stage_l0_count=first_stage_l0_count,
            first_stage_graph_count=first_stage_graph_count,
            first_stage_episodic_count=first_stage_episodic_count,
            first_stage_total_count=first_stage_l0_count + first_stage_graph_count + first_stage_episodic_count,
            final_l0_count=final_l0_count,
            final_graph_count=final_graph_count,
            final_episodic_count=final_episodic_count,
            final_selected_count=final_l0_count + final_graph_count + final_episodic_count,
            cascade_pruner_enabled=self.cascade_pruner is not None,
            cascade_prune_mode=cascade_stats.get("mode", ""),
            cascade_tokens_used=cascade_stats.get("tokens_used", 0),
            cascade_stage1_input=cascade_stats.get("cascade_stage1_input", 0),
            cascade_stage1_output=cascade_stats.get("cascade_stage1_output", 0),
            cascade_stage2_conflicts=cascade_stats.get("cascade_stage2_conflicts", 0),
            cascade_stage2_dropped=cascade_stats.get("cascade_stage2_dropped", 0),
            cascade_stage2_output=cascade_stats.get("cascade_stage2_output", 0),
            cascade_stage3_mmr_iterations=cascade_stats.get("cascade_stage3_mmr_iterations", 0),
            cascade_stage3_diversity_penalties=cascade_stats.get("cascade_stage3_diversity_penalties", 0),
            routing_info=routing_info,
        )

        if self.save_individual_reports and not getattr(self, "_suppress_individual_reports", False):
            retrieved_contents = self._collect_retrieved_contents_for_report(
                hierarchical_context,
                graph_units,
                episodic_units,
                rerank_metadata=rerank_result,
                l0_units_for_prompt=l0_units if final_l0_count != first_stage_l0_count else None,
            )
            self._save_individual_report(
                result,
                question_index,
                retrieved_contents,
                hierarchical_text,
                graph_text,
                episodic_context_with_time,
                full_prompt,
            )

        return result
    
    def _run_hierarchical_retrieval(self, sample_id: str, question: str, category: int) -> Dict[str, Any]:
        """Run hierarchical retrieval."""
        
        if self.topk_hierarchical <= 0:
            logger.debug(f"分层塔已禁用 (topk_hierarchical={self.topk_hierarchical}), 跳过检索")
            return {"hierarchical_enabled": False, "hierarchical_context_text": ""}
        
        try:
            if sample_id not in self.hierarchical_retrievers:
                logger.warning(f"分层检索器未加载: {sample_id}")
                return {"hierarchical_enabled": False, "hierarchical_context_text": ""}
            
            retriever = self.hierarchical_retrievers[sample_id]
            
            search_result = retriever.search(query=question, top_k=self.topk_hierarchical)
            
            by_layer = search_result.get("by_layer", {})
            hierarchical_context_text = self.hierarchical_context_builder.build_context(
                search_result,
                query=question, 
                category=category
            )
            # by_layer = search_result.get("by_layer", {})
            # hierarchical_context_text = self.hierarchical_context_builder.build_context(by_layer)
            # hierarchical_context_text = self.hierarchical_context_builder.build_context(
            #     by_layer, 
            #     query=question, 
            #     category=category
            # )
            
            
            return {
                "hierarchical_enabled": True,
                "hierarchical_context_text": hierarchical_context_text,
                "by_layer": by_layer,
                "retrieval_stats": search_result.get("retrieval_stats", {}),
                "total_results": len(search_result.get("results", []))
            }
            
        except Exception as e:
            logger.warning(f"分层检索失败 {sample_id}: {e}")
            return {"hierarchical_enabled": False, "error": str(e), "hierarchical_context_text": ""}
    
    def _run_graph_retrieval(self, sample_id: str, question: str) -> Tuple[List[Any], Dict[str, Any]]:
        """Run graph retrieval."""
        
        if self.topk_similarity <= 0 and self.topk_graph <= 0:
            logger.debug(f"图谱塔已禁用 (topk_similarity={self.topk_similarity}, topk_graph={self.topk_graph}), 跳过检索")
            return [], {"method": "disabled", "error": "tower_disabled"}
        
        try:
            if sample_id not in self.graph_benchmark.multi_retrievers:
                logger.warning(f"图检索器不存在: {sample_id}")
                return [], {"method": "failed", "error": "retriever_not_found"}
            
            multi_retriever = self.graph_benchmark.multi_retrievers[sample_id]
            entity_retriever = self.graph_benchmark.entity_relation_retrievers.get(sample_id)
            
            retrieved_units = []
            details = {}
            
            if self.use_entity_relation and entity_retriever:
                semantic_results = multi_retriever.smart_search(
                    query=question,
                    methods=["bm25", "cosine_similarity", "splade"],
                    fusion_method="rrf",
                    rerank_method=self.reranker_type,
                    top_k=self.topk_similarity,
                    return_detailed=False
                )
                
                entity_results = entity_retriever.search(question, self.topk_graph)
                graph_results = [(r.unit, r.score) for r in entity_results]
                
                retrieved_units = semantic_results + graph_results
                details = {
                    "method": "hybrid_semantic_graph",
                    "semantic_count": len(semantic_results),
                    "graph_count": len(graph_results)
                }
            else:
                retrieved_units = multi_retriever.smart_search(
                    query=question,
                    methods=["bm25", "cosine_similarity", "splade"],
                    fusion_method="rrf",
                    rerank_method=self.reranker_type,
                    top_k=self.topk_similarity,
                    return_detailed=False
                )
                details = {"method": "semantic_only", "semantic_count": len(retrieved_units)}
            
            return retrieved_units, details
            
        except Exception as e:
            logger.warning(f"图检索失败 {sample_id}: {e}")
            return [], {"method": "failed", "error": str(e)}
    
    def _run_episodic_retrieval(self, sample_id: str, question: str) -> Tuple[List[Any], str]:
        """Run episodic retrieval."""
        
        if self.topk_episodic <= 0:
            logger.debug(f"情景塔已禁用 (topk_episodic={self.topk_episodic}), 跳过检索")
            return [], ""
        
        try:
            if sample_id not in self.episodic_retrievers:
                logger.warning(f"情景记忆检索器不存在: {sample_id}")
                return [], ""
            
            multi_retriever = self.episodic_retrievers[sample_id]
            
            retrieved_units = multi_retriever.smart_search(
                query=question,
                methods=["bm25", "cosine_similarity", "splade"],
                fusion_method="rrf",
                rerank_method=self.reranker_type,
                top_k=self.topk_episodic,
                return_detailed=False
            )
            
            context_with_time = self._build_episodic_context_with_time(retrieved_units)
            
            return retrieved_units, context_with_time
            
        except Exception as e:
            logger.warning(f"情景记忆检索失败 {sample_id}: {e}")
            return [], ""
    
    def _extract_time_info(self, unit) -> str:
        """Extract time info."""
        try:
            if not hasattr(unit, 'metadata') or unit.metadata is None:
                return "N/A"
            
            time_start = unit.metadata.get("time_start", "")
            if time_start:
                return time_start
            
            time_original = unit.metadata.get("time_original", "")
            if time_original:
                return time_original
            
            if hasattr(unit, 'raw_data') and unit.raw_data:
                timestamp = unit.raw_data.get("timestamp", "")
                if timestamp:
                    return timestamp
            
            return "N/A"
            
        except Exception as e:
            logger.warning(f"提取时间信息失败: {e}")
            return "N/A"
    
    def _build_episodic_context_with_time(self, retrieved_units: List[Tuple[Any, float]]) -> str:
        """Build episodic context with time."""
        context_parts = []
        
        for i, (unit, score) in enumerate(retrieved_units):
            try:
                time_info = self._extract_time_info(unit)
                
                text_content = ""
                if hasattr(unit, 'raw_data') and unit.raw_data:
                    text_content = unit.raw_data.get('text_content', '')
                    if not text_content:
                        text_content = unit.raw_data.get('content', '')
                    if not text_content:
                        text_content = str(unit.raw_data)
                
                if not text_content:
                    continue
                
                time_marker = f"[Time: {time_info}]"
                enhanced_content = f"{time_marker} {text_content}"
                
                context_parts.append(f"Episodic Fact {i+1}: {enhanced_content}")
                
            except Exception as e:
                logger.warning(f"处理情景记忆单元失败: {e}")
                continue
        
        return "\n\n".join(context_parts)
    
    def _fuse_and_generate_answer(self,
                                  question: str,
                                  category: int,
                                  hierarchical_context: Dict[str, Any],
                                  graph_units: List[Any],
                                  graph_details: Dict[str, Any],
                                  episodic_units: List[Any],
                                  episodic_context_with_time: str) -> Tuple[Dict[str, str], float]:
        """Run fuse and generate answer."""
        prompt_parts = []
        
        if category == 5:
            prompt_parts.append("You are an expert conversation analyst specialized in detecting misleading or unanswerable questions.")
        else:
            prompt_parts.append("You are an expert conversation analyst with access to THREE complementary information retrieval systems.")
        
        prompt_parts.append("")
        prompt_parts.append("IMPORTANT: These are THREE DIFFERENT retrieval systems providing COMPLEMENTARY information:")
        prompt_parts.append("1. HIERARCHICAL MEMORY: Provides structured, multi-layer conversational context (summaries, insights)")
        prompt_parts.append("2. KNOWLEDGE GRAPH: Provides specific facts and entity relationships")
        prompt_parts.append("3. EPISODIC MEMORY: Provides time-stamped factual events with [Time: ...] markers")
        prompt_parts.append("")
        prompt_parts.append("Your task is to synthesize information from ALL THREE systems to provide the most accurate and complete answer.")
        prompt_parts.append("")
        
        category_guidance = self._get_tri_tower_category_guidance(category)
        prompt_parts.append(f"QUESTION: {question}")
        prompt_parts.append(f"QUESTION CATEGORY: {category} - {category_guidance}")
        prompt_parts.append("")
        
        hierarchical_enabled = hierarchical_context.get("hierarchical_enabled", False)
        prompt_parts.append("=" * 80)
        prompt_parts.append("TOWER 1: HIERARCHICAL MEMORY RESULTS")
        prompt_parts.append("=" * 80)
        if hierarchical_enabled:
            prompt_parts.append(hierarchical_context.get("hierarchical_context_text", "No hierarchical context available"))
        else:
            prompt_parts.append("Hierarchical retrieval was not available for this query.")
        prompt_parts.append("")
        
        prompt_parts.append("=" * 80)
        prompt_parts.append("TOWER 2: KNOWLEDGE GRAPH RESULTS")
        prompt_parts.append("=" * 80)
        if graph_units:
            prompt_parts.append(f"Retrieved {len(graph_units)} relevant knowledge graph units:")
            prompt_parts.append("")
            for i, (unit, score) in enumerate(graph_units[:10], 1):
                unit_content = self._extract_graph_unit_content(unit)
                prompt_parts.append(f"Graph Result {i}: {unit_content}")
                prompt_parts.append("")
        else:
            prompt_parts.append("No relevant entities or relationships found in the knowledge graph.")
        prompt_parts.append("")
        
        prompt_parts.append("=" * 80)
        prompt_parts.append("TOWER 3: EPISODIC MEMORY RESULTS (with Time Markers)")
        prompt_parts.append("=" * 80)
        if episodic_context_with_time:
            prompt_parts.append("Time-stamped facts from episodic memory:")
            prompt_parts.append("NOTE: Each fact has a [Time: ...] marker indicating when the event occurred.")
            prompt_parts.append("")
            prompt_parts.append(episodic_context_with_time)
        else:
            prompt_parts.append("No relevant episodic memories found.")
        prompt_parts.append("")
        
        prompt_parts.append("=" * 80)
        prompt_parts.append("TRI-TOWER FUSION GUIDANCE")
        prompt_parts.append("=" * 80)
        
        if category == 5:
            prompt_parts.append("SYNTHESIS INSTRUCTIONS (ADVERSARIAL):")
            prompt_parts.append("1. Cross-validate information across ALL THREE towers")
            prompt_parts.append("2. If information conflicts or is not found in any tower, state 'No information available'")
            prompt_parts.append("3. Be especially careful with [Time: ...] markers for temporal verification")
            prompt_parts.append("4. DO NOT fabricate information")
        elif category == 2:
            prompt_parts.append("SYNTHESIS INSTRUCTIONS (TEMPORAL):")
            prompt_parts.append("1. PRIORITIZE Episodic Memory [Time: ...] markers for temporal questions")
            prompt_parts.append("2. Cross-reference with Hierarchical and Knowledge Graph for context")
            prompt_parts.append("3. Extract specific dates/times from [Time: ...] markers")
        else:
            prompt_parts.append("SYNTHESIS INSTRUCTIONS:")
            prompt_parts.append("1. If ALL THREE towers provided information: Cross-validate and synthesize")
            prompt_parts.append("2. If only SOME towers worked: Prioritize based on question type")
            prompt_parts.append("3. Use [Time: ...] markers from Episodic Memory for temporal accuracy")
        
        prompt_parts.append("")
        prompt_parts.append("RESPONSE FORMAT (REQUIRED JSON):")
        prompt_parts.append("{")
        prompt_parts.append('    "reasoning": "Your synthesis process across all three towers...",')
        prompt_parts.append('    "final_answer": "Your direct, concise final answer"')
        prompt_parts.append("}")
        
        full_prompt = "\n".join(prompt_parts)
        
        try:
            raw_response = self.llm_client.generate_answer(
                prompt=full_prompt,
                temperature=0.1,
                max_tokens=self.generation_max_tokens,
                json_format=True
            )
            
            answer_dict = self._parse_structured_response(raw_response, category)
            
            confidence_score = self._calculate_tri_tower_confidence(
                hierarchical_enabled,
                len(graph_units),
                len(episodic_units)
            )
            
            return answer_dict, confidence_score
            
        except Exception as e:
            logger.error(f"三塔融合生成失败: {e}")
            return {
                "reasoning": f"Generation failed: {str(e)}",
                "final_answer": "Unable to generate answer"
            }, 0.0
    
    def _build_full_prompt(self,
                           question: str,
                           category: int,
                           hierarchical_context: Dict[str, Any],
                           hierarchical_text: str,
                           graph_units: List[Any],
                           graph_text: str,
                           episodic_context_with_time: str) -> Tuple[str, Dict[str, int]]:
        """Build full prompt."""
        
        h_active = self.topk_hierarchical > 0
        g_active = self.topk_similarity > 0 or self.topk_graph > 0
        e_active = self.topk_episodic > 0
        
        active_tower_names = []
        if h_active:
            active_tower_names.append(("HIERARCHICAL MEMORY",
                                       "Provides structured, multi-layer conversational context (summaries, insights)"))
        if g_active:
            active_tower_names.append(("KNOWLEDGE GRAPH",
                                       "Provides specific facts and entity relationships"))
        if e_active:
            active_tower_names.append(("EPISODIC MEMORY",
                                       "Provides time-stamped factual events with [Time: ...] markers"))
        num_active = len(active_tower_names)
        tower_count_word = {1: "ONE", 2: "TWO", 3: "THREE"}.get(num_active, str(num_active))
        
        
        system_prompt_parts = []
        
        if category == 5:
            system_prompt_parts.append("You are an expert conversation analyst specialized in detecting misleading or unanswerable questions.")
        else:
            system_prompt_parts.append(f"You are an expert conversation analyst with access to {tower_count_word} complementary information retrieval system{'s' if num_active > 1 else ''}.")
        
        system_prompt_parts.append("")
        system_prompt_parts.append(f"IMPORTANT: {'These are' if num_active > 1 else 'This is'} {tower_count_word} {'DIFFERENT retrieval systems' if num_active > 1 else 'retrieval system'} providing {'COMPLEMENTARY ' if num_active > 1 else ''}information:")
        for i, (name, desc) in enumerate(active_tower_names, 1):
            system_prompt_parts.append(f"{i}. {name}: {desc}")
        system_prompt_parts.append("")
        if num_active > 1:
            system_prompt_parts.append(f"Your task is to synthesize information from ALL {tower_count_word} systems to provide the most accurate and complete answer.")
        else:
            system_prompt_parts.append("Your task is to use the retrieved information to provide the most accurate and complete answer.")
        
        system_prompt_text = "\n".join(system_prompt_parts)
        
        prompt_parts = []
        prompt_parts.append(system_prompt_text)
        prompt_parts.append("")
        
        category_guidance = self._get_tri_tower_category_guidance(category)
        prompt_parts.append(f"QUESTION: {question}")
        prompt_parts.append(f"QUESTION CATEGORY: {category} - {category_guidance}")
        prompt_parts.append("")
        
        
        tower_num = 0
        hierarchical_enabled = hierarchical_context.get("hierarchical_enabled", False)
        
        if h_active:
            tower_num += 1
            prompt_parts.append("=" * 80)
            prompt_parts.append(f"TOWER {tower_num}: HIERARCHICAL MEMORY RESULTS")
            prompt_parts.append("=" * 80)
            if hierarchical_enabled:
                prompt_parts.append(hierarchical_text)
            else:
                prompt_parts.append("Hierarchical retrieval was not available for this query.")
            prompt_parts.append("")
        
        if g_active:
            tower_num += 1
            prompt_parts.append("=" * 80)
            prompt_parts.append(f"TOWER {tower_num}: KNOWLEDGE GRAPH RESULTS")
            prompt_parts.append("=" * 80)
            if graph_units:
                prompt_parts.append(graph_text)
            else:
                prompt_parts.append("No relevant entities or relationships found in the knowledge graph.")
            prompt_parts.append("")
        
        if e_active:
            tower_num += 1
            prompt_parts.append("=" * 80)
            prompt_parts.append(f"TOWER {tower_num}: EPISODIC MEMORY RESULTS (with Time Markers)")
            prompt_parts.append("=" * 80)
            if episodic_context_with_time:
                prompt_parts.append("Time-stamped facts from episodic memory:")
                prompt_parts.append("NOTE: Each fact has a [Time: ...] marker indicating when the event occurred.")
                prompt_parts.append("")
                prompt_parts.append(episodic_context_with_time)
            else:
                prompt_parts.append("No relevant episodic memories found.")
            prompt_parts.append("")
        
        
        prompt_parts.append("=" * 80)
        fusion_label = {1: "RETRIEVAL", 2: "DUAL-TOWER FUSION", 3: "TRI-TOWER FUSION"}.get(num_active, "FUSION")
        prompt_parts.append(f"{fusion_label} GUIDANCE")
        prompt_parts.append("=" * 80)
        
        if category == 5:
            prompt_parts.append("SYNTHESIS INSTRUCTIONS (ADVERSARIAL):")
            prompt_parts.append(f"1. Cross-validate information across all {num_active} active tower{'s' if num_active > 1 else ''}")
            prompt_parts.append("2. If information conflicts or is not found in any tower, state 'No information available'")
            if e_active:
                prompt_parts.append("3. Be especially careful with [Time: ...] markers for temporal verification")
                prompt_parts.append("4. DO NOT fabricate information")
            else:
                prompt_parts.append("3. DO NOT fabricate information")
        elif category == 2:
            prompt_parts.append("SYNTHESIS INSTRUCTIONS (TEMPORAL):")
            if e_active:
                prompt_parts.append("1. PRIORITIZE Episodic Memory [Time: ...] markers for temporal questions")
                prompt_parts.append("2. Cross-reference with other tower(s) for context")
                prompt_parts.append("3. Extract specific dates/times from [Time: ...] markers")
            else:
                prompt_parts.append("1. Look for temporal information in all available sources")
                prompt_parts.append("2. Cross-reference across towers for context")
                prompt_parts.append("3. Extract specific dates/times when available")
        else:
            prompt_parts.append("SYNTHESIS INSTRUCTIONS:")
            if num_active >= 2:
                prompt_parts.append(f"1. Cross-validate and synthesize information from all {num_active} towers")
            else:
                prompt_parts.append("1. Use the retrieved information to form your answer")
            prompt_parts.append("2. Prioritize based on question type")
            if e_active:
                prompt_parts.append("3. Use [Time: ...] markers from Episodic Memory for temporal accuracy")
        
        prompt_parts.append("")
        prompt_parts.append("RESPONSE FORMAT (REQUIRED JSON):")
        prompt_parts.append("{")
        tower_ref = f"all {num_active} towers" if num_active > 1 else "the retrieval system"
        prompt_parts.append(f'    "reasoning": "Your synthesis process across {tower_ref}...",')
        prompt_parts.append('    "final_answer": "Your direct, concise final answer"')
        prompt_parts.append("}")
        
        full_prompt = "\n".join(prompt_parts)
        
        system_prompt_tokens = self._count_tokens(system_prompt_text)
        total_input_tokens = self._count_tokens(full_prompt)
        
        token_info = {
            "system_prompt_tokens": system_prompt_tokens,
            "total_input_tokens": total_input_tokens,
        }
        
        return full_prompt, token_info

    def _fuse_and_generate_answer_with_tokens(self,
                                              question: str,
                                              category: int,
                                              hierarchical_context: Dict[str, Any],
                                              hierarchical_text: str,
                                              graph_units: List[Any],
                                              graph_details: Dict[str, Any],
                                              graph_text: str,
                                              episodic_units: List[Any],
                                              episodic_context_with_time: str,
                                              pre_built_prompt: Optional[str] = None,
                                              pre_built_token_info: Optional[Dict[str, int]] = None) -> Tuple[Dict[str, str], float, Dict[str, int]]:
        """Returns: (answer_dict, confidence_score, token_info)."""
        if pre_built_prompt is not None and pre_built_token_info is not None:
            full_prompt = pre_built_prompt
            token_info = pre_built_token_info
            hierarchical_enabled = hierarchical_context.get("hierarchical_enabled", False)
            
            try:
                print_prompt = f"\n{'▼'*40} SENDING PROMPT TO LLM {'▼'*40}\n{full_prompt}\n{'▲'*40} END OF PROMPT {'▲'*40}\n"
                logger.info(print_prompt)
                raw_response = self.llm_client.generate_answer(
                    prompt=full_prompt,
                    temperature=0.1,
                    max_tokens=self.generation_max_tokens,
                    json_format=True
                )
                
                answer_dict = self._parse_structured_response(raw_response, category)
                
                confidence_score = self._calculate_tri_tower_confidence(
                    hierarchical_enabled,
                    len(graph_units),
                    len(episodic_units)
                )
                
                return answer_dict, confidence_score, token_info
                
            except Exception as e:
                logger.error(f"三塔融合生成失败(pre-built prompt): {e}")
                return {
                    "reasoning": f"Generation failed: {str(e)}",
                    "final_answer": "Unable to generate answer"
                }, 0.0, token_info
        
        system_prompt_parts = []
        
        if category == 5:
            system_prompt_parts.append("You are an expert conversation analyst specialized in detecting misleading or unanswerable questions.")
        else:
            system_prompt_parts.append("You are an expert conversation analyst with access to THREE complementary information retrieval systems.")
        
        system_prompt_parts.append("")
        system_prompt_parts.append("IMPORTANT: These are THREE DIFFERENT retrieval systems providing COMPLEMENTARY information:")
        system_prompt_parts.append("1. HIERARCHICAL MEMORY: Provides structured, multi-layer conversational context (summaries, insights)")
        system_prompt_parts.append("2. KNOWLEDGE GRAPH: Provides specific facts and entity relationships")
        system_prompt_parts.append("3. EPISODIC MEMORY: Provides time-stamped factual events with [Time: ...] markers")
        system_prompt_parts.append("")
        system_prompt_parts.append("Your task is to synthesize information from ALL THREE systems to provide the most accurate and complete answer.")
        
        system_prompt_text = "\n".join(system_prompt_parts)
        
        prompt_parts = []
        prompt_parts.append(system_prompt_text)
        prompt_parts.append("")
        
        category_guidance = self._get_tri_tower_category_guidance(category)
        prompt_parts.append(f"QUESTION: {question}")
        prompt_parts.append(f"QUESTION CATEGORY: {category} - {category_guidance}")
        prompt_parts.append("")
        
        hierarchical_enabled = hierarchical_context.get("hierarchical_enabled", False)
        prompt_parts.append("=" * 80)
        prompt_parts.append("TOWER 1: HIERARCHICAL MEMORY RESULTS")
        prompt_parts.append("=" * 80)
        if hierarchical_enabled:
            prompt_parts.append(hierarchical_text)
        else:
            prompt_parts.append("Hierarchical retrieval was not available for this query.")
        prompt_parts.append("")
        
        prompt_parts.append("=" * 80)
        prompt_parts.append("TOWER 2: KNOWLEDGE GRAPH RESULTS")
        prompt_parts.append("=" * 80)
        if graph_units:
            prompt_parts.append(graph_text)
        else:
            prompt_parts.append("No relevant entities or relationships found in the knowledge graph.")
        prompt_parts.append("")
        
        prompt_parts.append("=" * 80)
        prompt_parts.append("TOWER 3: EPISODIC MEMORY RESULTS (with Time Markers)")
        prompt_parts.append("=" * 80)
        if episodic_context_with_time:
            prompt_parts.append("Time-stamped facts from episodic memory:")
            prompt_parts.append("NOTE: Each fact has a [Time: ...] marker indicating when the event occurred.")
            prompt_parts.append("")
            prompt_parts.append(episodic_context_with_time)
        else:
            prompt_parts.append("No relevant episodic memories found.")
        prompt_parts.append("")
        
        prompt_parts.append("=" * 80)
        prompt_parts.append("TRI-TOWER FUSION GUIDANCE")
        prompt_parts.append("=" * 80)
        
        if category == 5:
            prompt_parts.append("SYNTHESIS INSTRUCTIONS (ADVERSARIAL):")
            prompt_parts.append("1. Cross-validate information across ALL THREE towers")
            prompt_parts.append("2. If information conflicts or is not found in any tower, state 'No information available'")
            prompt_parts.append("3. Be especially careful with [Time: ...] markers for temporal verification")
            prompt_parts.append("4. DO NOT fabricate information")
        elif category == 2:
            prompt_parts.append("SYNTHESIS INSTRUCTIONS (TEMPORAL):")
            prompt_parts.append("1. PRIORITIZE Episodic Memory [Time: ...] markers for temporal questions")
            prompt_parts.append("2. Cross-reference with Hierarchical and Knowledge Graph for context")
            prompt_parts.append("3. Extract specific dates/times from [Time: ...] markers")
        else:
            prompt_parts.append("SYNTHESIS INSTRUCTIONS:")
            prompt_parts.append("1. If ALL THREE towers provided information: Cross-validate and synthesize")
            prompt_parts.append("2. If only SOME towers worked: Prioritize based on question type")
            prompt_parts.append("3. Use [Time: ...] markers from Episodic Memory for temporal accuracy")
        
        prompt_parts.append("")
        prompt_parts.append("RESPONSE FORMAT (REQUIRED JSON):")
        prompt_parts.append("{")
        prompt_parts.append('    "reasoning": "Your synthesis process across all three towers...",')
        prompt_parts.append('    "final_answer": "Your direct, concise final answer"')
        prompt_parts.append("}")
        
        full_prompt = "\n".join(prompt_parts)
        
        system_prompt_tokens = self._count_tokens(system_prompt_text)
        total_input_tokens = self._count_tokens(full_prompt)
        
        token_info = {
            "system_prompt_tokens": system_prompt_tokens,
            "total_input_tokens": total_input_tokens,
        }
        
        try:
            # logger.debug(f"\n{'#'*30} FULL PROMPT START {'#'*30}\n{full_prompt}\n{'#'*30} FULL PROMPT END {'#'*30}\n")
            print_prompt = f"\n{'▼'*40} SENDING PROMPT TO LLM {'▼'*40}\n{full_prompt}\n{'▲'*40} END OF PROMPT {'▲'*40}\n"
            logger.info(print_prompt)
            raw_response = self.llm_client.generate_answer(
                prompt=full_prompt,
                temperature=0.1,
                max_tokens=self.generation_max_tokens,
                json_format=True
            )
            
            answer_dict = self._parse_structured_response(raw_response, category)
            
            confidence_score = self._calculate_tri_tower_confidence(
                hierarchical_enabled,
                len(graph_units),
                len(episodic_units)
            )
            
            return answer_dict, confidence_score, token_info
            
        except Exception as e:
            logger.error(f"三塔融合生成失败(with tokens): {e}")
            return {
                "reasoning": f"Generation failed: {str(e)}",
                "final_answer": "Unable to generate answer"
            }, 0.0, token_info
    
    def _extract_graph_unit_content(self, unit) -> str:
        """Extract graph unit content."""
        if hasattr(unit, 'raw_data') and unit.raw_data:
            return unit.raw_data.get('text_content', str(unit.raw_data))
        return str(unit)
    
    def _get_tri_tower_category_guidance(self, category: int) -> str:
        """Get tri tower category gUIDance."""
        guidance_map = {
            1: "Multi-hop reasoning - Trace connections across hierarchical patterns, graph relationships, AND episodic timeline",
            2: "Temporal question - PRIORITIZE episodic [Time: ...] markers, verify with hierarchical sessions AND graph entities",
            3: "Open-domain question - Synthesize comprehensive view from all three towers",
            4: "Single-hop fact - Verify fact across hierarchical context, graph evidence, AND episodic memory",
            5: "Adversarial question - Check information existence in ALL THREE systems before answering"
        }
        return guidance_map.get(category, "General question - Use all three information sources")
    
    def _parse_structured_response(self, raw_response: str, category: int) -> Dict[str, str]:
        """Parse structured response."""
        try:
            parsed = json.loads(raw_response.strip())
            
            if isinstance(parsed, dict) and "reasoning" in parsed and "final_answer" in parsed:
                raw_reasoning = parsed["reasoning"]
                if isinstance(raw_reasoning, list):
                    raw_reasoning = ", ".join(str(x) for x in raw_reasoning)
                elif not isinstance(raw_reasoning, str):
                    raw_reasoning = str(raw_reasoning)
                
                raw_answer = parsed["final_answer"]
                if isinstance(raw_answer, list):
                    raw_answer = ", ".join(str(x) for x in raw_answer)
                elif not isinstance(raw_answer, str):
                    raw_answer = str(raw_answer)
                
                final_answer = self._post_process_answer(raw_answer, category)
                return {
                    "reasoning": raw_reasoning.strip(),
                    "final_answer": final_answer
                }
            else:
                raise ValueError("JSON格式不正确")
                
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"JSON解析失败: {e}，尝试文本解析")
            return self._parse_text_response(raw_response, category)
    
    def _parse_text_response(self, raw_response: str, category: int) -> Dict[str, str]:
        """Parse text response."""
        lines = raw_response.strip().split('\n')
        reasoning = ""
        final_answer = ""
        
        current_section = "reasoning"
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if any(keyword in line.lower() for keyword in ["answer", "final", "conclusion"]):
                current_section = "answer"
                if ":" in line:
                    final_answer = line.split(":", 1)[1].strip()
                    continue
            
            if current_section == "reasoning":
                reasoning += line + " "
            else:
                final_answer += line + " "
        
        if not final_answer.strip():
            final_answer = raw_response.strip()
            reasoning = "Unable to parse structured reasoning"
        
        final_answer = self._post_process_answer(final_answer.strip(), category)
        
        return {
            "reasoning": reasoning.strip() or "No clear reasoning provided",
            "final_answer": final_answer
        }
    
    def _post_process_answer(self, answer, category: int) -> str:
        """Run post process answer."""
        if isinstance(answer, list):
            answer = ", ".join(str(x) for x in answer)
        elif not isinstance(answer, str):
            answer = str(answer) if answer is not None else ""
        
        if not answer:
            return "No answer generated"
        
        answer = answer.strip()
        
        prefixes = ["Answer:", "ANSWER:", "Final Answer:", "Response:"]
        for prefix in prefixes:
            if answer.startswith(prefix):
                answer = answer[len(prefix):].strip()
        
        if answer and not answer[0].isupper() and not answer[0].isdigit():
            answer = answer[0].upper() + answer[1:]
        
        if category == 5:
            lower_answer = answer.lower()
            if any(phrase in lower_answer for phrase in [
                "no information", "not available", "not mentioned", 
                "not found", "insufficient information"
            ]):
                return "No information available"
        
        return answer
    
    def _calculate_tri_tower_confidence(self,
                                        hierarchical_success: bool,
                                        graph_results_count: int,
                                        episodic_results_count: int) -> float:
        """Calculate tri tower confidence."""
        base_confidence = 0.0
        
        if hierarchical_success:
            base_confidence += self.fusion_weights.get("hierarchical", 0.35)
        
        if graph_results_count > 0:
            graph_weight = self.fusion_weights.get("graph", 0.35)
            result_factor = min(graph_results_count / 10.0, 1.0)
            base_confidence += graph_weight * result_factor
        
        if episodic_results_count > 0:
            episodic_weight = self.fusion_weights.get("episodic", 0.30)
            result_factor = min(episodic_results_count / 15.0, 1.0)
            base_confidence += episodic_weight * result_factor
        
        return min(base_confidence, 1.0)
    
    def _evaluate_tri_tower_result(self,
                                   question: str,
                                   expected_answer: str,
                                   generated_answer: str,
                                   reasoning: str,
                                   category: int) -> Dict[str, Any]:
        """Run evaluate tri tower result."""
        try:
            eval_result = calculate_comprehensive_scores(
                gold_answer=expected_answer,
                response=generated_answer,
                question=question,
                reasoning=reasoning,
                llm_client=self.llm_evaluate_client,
                metrics=["exact_match", "f1", "rouge", "semantic_similarity", "llm_judge"],
                category=category,
                is_adversarial=(category == 5)
            )
            
            return {
                "evaluation_scores": eval_result.get("scores", {}),
                "evaluation_method": "unified_comprehensive",
                "evaluation_success": eval_result.get("evaluation_success", False)
            }
            
        except Exception as e:
            logger.error(f"三塔评估失败: {e}")
            return {
                "evaluation_scores": {"error": str(e)},
                "evaluation_method": "failed",
                "evaluation_success": False
            }
    
    
    
    def _save_sample_results(self, sample_id: str, sample_results: List[TriTowerRetrievalResult]):
        """Save sample results."""
        if not sample_results:
            logger.warning(f"样本 {sample_id} 没有结果需要保存")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        successful_results = [r for r in sample_results if r.evaluation_success]
        
        performance_metrics = {}
        timing_metrics = {}
        tower_success_rates = {}
        
        if successful_results:
            f1_scores = [r.evaluation_scores.get("token_f1", 0.0) for r in successful_results]
            semantic_scores = [r.evaluation_scores.get("semantic_similarity", 0.0) for r in successful_results]
            llm_scores = [r.evaluation_scores.get("llm_accuracy", 0.0) for r in successful_results]
            
            performance_metrics = {
                "avg_f1_score": float(np.mean(f1_scores)),
                "std_f1_score": float(np.std(f1_scores)),
                "avg_semantic_similarity": float(np.mean(semantic_scores)),
                "avg_llm_accuracy": float(np.mean(llm_scores)),
            }
            
            hierarchical_times = [r.hierarchical_retrieval_time for r in successful_results]
            graph_times = [r.graph_retrieval_time for r in successful_results]
            episodic_times = [r.episodic_retrieval_time for r in successful_results]
            generation_times = [r.generation_time for r in successful_results]
            total_retrieval_times = [r.total_retrieval_time for r in successful_results]
            e2e_latencies = [r.end_to_end_latency for r in successful_results]
            second_stage_rerank_times = [r.second_stage_rerank_time for r in successful_results if r.second_stage_rerank_enabled]
            
            timing_metrics = {
                "avg_hierarchical_time": float(np.mean(hierarchical_times)),
                "avg_graph_time": float(np.mean(graph_times)),
                "avg_episodic_time": float(np.mean(episodic_times)),
                "avg_generation_time": float(np.mean(generation_times)),
                
                "avg_total_retrieval_time": float(np.mean(total_retrieval_times)),
                "avg_end_to_end_latency": float(np.mean(e2e_latencies)),
                
                "avg_second_stage_rerank_time": float(np.mean(second_stage_rerank_times)) if second_stage_rerank_times else 0.0,
                "total_second_stage_rerank_time": float(sum(second_stage_rerank_times)) if second_stage_rerank_times else 0.0,
            }
            
            
            hierarchical_success = sum(1 for r in successful_results 
                if r.hierarchical_context.get("hierarchical_enabled", False))
            graph_success = sum(1 for r in successful_results if len(r.graph_retrieved_units) > 0)
            episodic_success = sum(1 for r in successful_results if len(r.episodic_retrieved_units) > 0)
            all_three_success = sum(1 for r in successful_results
                if r.hierarchical_context.get("hierarchical_enabled", False)
                and len(r.graph_retrieved_units) > 0
                and len(r.episodic_retrieved_units) > 0)
            
            tower_success_rates = {
                "hierarchical_success_rate": hierarchical_success / len(successful_results),
                "graph_success_rate": graph_success / len(successful_results),
                "episodic_success_rate": episodic_success / len(successful_results),
                "all_towers_success_rate": all_three_success / len(successful_results),
            }
            
            l0_tokens_list = [r.l0_tokens for r in successful_results]
            l1_tokens_list = [r.l1_tokens for r in successful_results]
            l2_tokens_list = [r.l2_tokens for r in successful_results]
            graph_tokens_list = [r.graph_tokens for r in successful_results]
            episodic_tokens_list = [r.episodic_tokens for r in successful_results]
            total_input_tokens_list = [r.total_input_tokens for r in successful_results]
            completion_tokens_list = [r.completion_tokens for r in successful_results]
            
            token_metrics = {
                "avg_l0_tokens": float(np.mean(l0_tokens_list)),
                "avg_l1_tokens": float(np.mean(l1_tokens_list)),
                "avg_l2_tokens": float(np.mean(l2_tokens_list)),
                "avg_graph_tokens": float(np.mean(graph_tokens_list)),
                "avg_episodic_tokens": float(np.mean(episodic_tokens_list)),
                "avg_total_input_tokens": float(np.mean(total_input_tokens_list)),
                "avg_completion_tokens": float(np.mean(completion_tokens_list)),
                "total_l0_tokens": int(sum(l0_tokens_list)),
                "total_l1_tokens": int(sum(l1_tokens_list)),
                "total_l2_tokens": int(sum(l2_tokens_list)),
                "total_graph_tokens": int(sum(graph_tokens_list)),
                "total_episodic_tokens": int(sum(episodic_tokens_list)),
                "total_input_tokens": int(sum(total_input_tokens_list)),
                "total_completion_tokens": int(sum(completion_tokens_list)),
            }
        else:
            token_metrics = {}
        
        sample_data = {
            "sample_info": {
                "sample_id": sample_id,
                "timestamp": datetime.now().isoformat(),
                "test_count": len(sample_results),
                "successful_count": len(successful_results),
                "failed_count": len(sample_results) - len(successful_results),
                "fusion_strategy": self.fusion_strategy,
                "fusion_weights": self.fusion_weights,
                "parallel_towers_enabled": self.parallel_towers,
                "hierarchical_config": {
                    "topk_hierarchical": self.topk_hierarchical,
                },
                "graph_config": {
                    "topk_similarity": self.topk_similarity,
                    "topk_graph": self.topk_graph,
                    "use_entity_relation": self.use_entity_relation,
                },
                "episodic_config": {
                    "topk_episodic": self.topk_episodic,
                },
                
                "second_stage_rerank_config": {
                    "enabled": self.enable_second_stage_rerank,
                    "method": self.second_stage_rerank_method,
                    "final_top_k": self.final_top_k,
                    "strategy": self.rerank_strategy,
                },
                
                "router_config": {
                    "enabled": self.tower_router is not None,
                    "model": self.tower_router.model_name if self.tower_router else None,
                    "strategy": self.tower_router.strategy if self.tower_router else None,
                },
                "cascade_config": {
                    "enabled": self.cascade_pruner is not None,
                    "prune_mode": self.cascade_prune_mode.value if hasattr(self, 'cascade_prune_mode') and hasattr(self.cascade_prune_mode, 'value') else "",
                    "max_context_tokens": self.cascade_max_context_tokens,
                    "stage2_enabled": getattr(self, 'cascade_enable_stage2', False),
                    "stage3_mmr_enabled": getattr(self, 'cascade_enable_stage3_mmr', False),
                },
            },
            "performance_metrics": performance_metrics,
            "timing_metrics": timing_metrics,
            "tower_success_rates": tower_success_rates,
            "token_metrics": token_metrics,
            "results": [
                {
                    "question": r.question,
                    "category": r.category,
                    "expected_answer": r.expected_answer,
                    "final_answer": r.final_answer,
                    "reasoning_process": r.reasoning_process,
                    "confidence_score": r.confidence_score,
                    "hierarchical_success": r.hierarchical_context.get("hierarchical_enabled", False),
                    "graph_results_count": len(r.graph_retrieved_units),
                    "episodic_results_count": len(r.episodic_retrieved_units),
                    "evaluation_scores": r.evaluation_scores,
                    "evaluation_success": r.evaluation_success,
                    "hierarchical_time": r.hierarchical_retrieval_time,
                    "graph_time": r.graph_retrieval_time,
                    "episodic_time": r.episodic_retrieval_time,
                    "generation_time": r.generation_time,
                    "l0_tokens": r.l0_tokens,
                    "l1_tokens": r.l1_tokens,
                    "l2_tokens": r.l2_tokens,
                    "graph_tokens": r.graph_tokens,
                    "episodic_tokens": r.episodic_tokens,
                    "system_prompt_tokens": r.system_prompt_tokens,
                    "question_tokens": r.question_tokens,
                    "total_input_tokens": r.total_input_tokens,
                    "completion_tokens": r.completion_tokens,
                    "total_retrieval_time": r.total_retrieval_time,
                    "end_to_end_latency": r.end_to_end_latency,
                    
                    "second_stage_rerank_enabled": r.second_stage_rerank_enabled,
                    "second_stage_rerank_time": r.second_stage_rerank_time,
                    "first_stage_l0_count": r.first_stage_l0_count,
                    "first_stage_graph_count": r.first_stage_graph_count,
                    "first_stage_episodic_count": r.first_stage_episodic_count,
                    "first_stage_total_count": r.first_stage_total_count,
                    "final_l0_count": r.final_l0_count,
                    "final_graph_count": r.final_graph_count,
                    "final_episodic_count": r.final_episodic_count,
                    "final_selected_count": r.final_selected_count,
                    
                    "routing_info": r.routing_info,
                    "cascade_pruner_enabled": r.cascade_pruner_enabled,
                    "cascade_prune_mode": r.cascade_prune_mode,
                    "cascade_tokens_used": r.cascade_tokens_used,
                    "cascade_stage1_input": r.cascade_stage1_input,
                    "cascade_stage1_output": r.cascade_stage1_output,
                    "cascade_stage2_conflicts": r.cascade_stage2_conflicts,
                    "cascade_stage2_dropped": r.cascade_stage2_dropped,
                    "cascade_stage2_output": r.cascade_stage2_output,
                    "cascade_stage3_mmr_iterations": r.cascade_stage3_mmr_iterations,
                    "cascade_stage3_diversity_penalties": r.cascade_stage3_diversity_penalties,
                }
                for r in sample_results
            ]
        }
        
        
        sample_file = self.output_dir / f"sample_{sample_id}_{timestamp}.json"
        with open(sample_file, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f" 样本 {sample_id} 结果已保存: {sample_file}")
        
        self._generate_sample_readable_report(sample_id, sample_results, timestamp)
        
        cumulative_file = self.output_dir / "cumulative_results.jsonl"
        with open(cumulative_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(sample_data, ensure_ascii=False) + '\n')
    
    def _generate_sample_readable_report(self, 
                                         sample_id: str, 
                                         sample_results: List[TriTowerRetrievalResult],
                                         timestamp: str):
        """Generate sample readable report."""
        lines = []
        
        lines.append("=" * 100)
        lines.append(f"样本 {sample_id} - 三塔召回测试报告")
        lines.append("=" * 100)
        lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"测试数量: {len(sample_results)}")
        lines.append(f"并行模式: {'启用' if self.parallel_towers else '禁用'}")
        lines.append("=" * 100)
        
        valid_results = [r for r in sample_results if r.evaluation_success]
        successful_count = len(valid_results)
        failed_count = len(sample_results) - successful_count
        
        if valid_results:
            avg_f1 = np.mean([r.evaluation_scores.get("token_f1", 0.0) for r in valid_results])
            avg_semantic = np.mean([r.evaluation_scores.get("semantic_similarity", 0.0) for r in valid_results])
            avg_llm = np.mean([r.evaluation_scores.get("llm_accuracy", 0.0) for r in valid_results])
            
            avg_hier_time = np.mean([r.hierarchical_retrieval_time for r in valid_results])
            avg_graph_time = np.mean([r.graph_retrieval_time for r in valid_results])
            avg_episodic_time = np.mean([r.episodic_retrieval_time for r in valid_results])
            avg_gen_time = np.mean([r.generation_time for r in valid_results])
            
            hier_success = sum(1 for r in valid_results if r.hierarchical_context.get("hierarchical_enabled", False))
            graph_success = sum(1 for r in valid_results if len(r.graph_retrieved_units) > 0)
            episodic_success = sum(1 for r in valid_results if len(r.episodic_retrieved_units) > 0)
            all_success = sum(1 for r in valid_results
                if r.hierarchical_context.get("hierarchical_enabled", False)
                and len(r.graph_retrieved_units) > 0
                and len(r.episodic_retrieved_units) > 0)
            
            lines.append(f"\n 整体统计:")
            lines.append(f"    成功测试: {successful_count} ({successful_count/len(sample_results)*100:.1f}%)")
            lines.append(f"    失败测试: {failed_count} ({failed_count/len(sample_results)*100:.1f}%)")
            
            lines.append(f"\n 性能指标:")
            lines.append(f"   - 平均F1分数: {avg_f1:.3f}")
            lines.append(f"   - 平均语义相似度: {avg_semantic:.3f}")
            lines.append(f"   - 平均LLM准确率: {avg_llm:.3f}")
            
            lines.append(f"\nTiming Performance:")
            lines.append(f"   - 平均分层检索: {avg_hier_time:.3f}s")
            lines.append(f"   - 平均图检索: {avg_graph_time:.3f}s")
            lines.append(f"   - 平均情景检索: {avg_episodic_time:.3f}s")
            lines.append(f"   - 平均答案生成: {avg_gen_time:.3f}s")
            avg_total_retrieval = np.mean([r.total_retrieval_time for r in valid_results])
            avg_e2e_latency = np.mean([r.end_to_end_latency for r in valid_results])
            lines.append(f"   - 平均真实检索耗时: {avg_total_retrieval:.3f}s (含并行开销/重排序/Token计数)")
            lines.append(f"   - 平均端到端延迟: {avg_e2e_latency:.3f}s (检索+生成)")
            
            
            if self.enable_second_stage_rerank:
                reranked_results = [r for r in valid_results if r.second_stage_rerank_enabled]
                if reranked_results:
                    avg_rerank_time = np.mean([r.second_stage_rerank_time for r in reranked_results])
                    avg_first_stage_total = np.mean([r.first_stage_total_count for r in reranked_results])
                    avg_final_l0 = np.mean([r.final_l0_count for r in reranked_results])
                    avg_final_graph = np.mean([r.final_graph_count for r in reranked_results])
                    avg_final_episodic = np.mean([r.final_episodic_count for r in reranked_results])
                    avg_final_selected = np.mean([r.final_selected_count for r in reranked_results])
                    
                    lines.append(f"\n 二次重排序统计:")
                    lines.append(f"   - 重排序方法: {self.second_stage_rerank_method}")
                    lines.append(f"   - 最终top-k: {self.final_top_k}")
                    lines.append(f"   - 平均重排序时间: {avg_rerank_time:.3f}s")
                    lines.append(f"   - 平均一阶段候选: {avg_first_stage_total:.1f}")
                    lines.append(f"   - 平均最终选中: L0={avg_final_l0:.1f} | Graph={avg_final_graph:.1f} | Episodic={avg_final_episodic:.1f}")
                    lines.append(f"   - 平均最终总数: {avg_final_selected:.1f}")
            
            lines.append(f"\n 三塔成功率:")
            lines.append(f"   - 分层检索: {hier_success}/{len(valid_results)} ({hier_success/len(valid_results)*100:.1f}%)")
            lines.append(f"   - 图检索: {graph_success}/{len(valid_results)} ({graph_success/len(valid_results)*100:.1f}%)")
            lines.append(f"   - 情景检索: {episodic_success}/{len(valid_results)} ({episodic_success/len(valid_results)*100:.1f}%)")
            lines.append(f"   - 三塔同时成功: {all_success}/{len(valid_results)} ({all_success/len(valid_results)*100:.1f}%)")
            
            cascade_results = [r for r in valid_results if r.cascade_pruner_enabled]
            if cascade_results:
                avg_cascade_tokens = np.mean([r.cascade_tokens_used for r in cascade_results])
                avg_s1_in = np.mean([r.cascade_stage1_input for r in cascade_results])
                avg_s1_out = np.mean([r.cascade_stage1_output for r in cascade_results])
                avg_s2_conflicts = np.mean([r.cascade_stage2_conflicts for r in cascade_results])
                avg_s2_dropped = np.mean([r.cascade_stage2_dropped for r in cascade_results])
                avg_s2_out = np.mean([r.cascade_stage2_output for r in cascade_results])
                avg_s3_mmr = np.mean([r.cascade_stage3_mmr_iterations for r in cascade_results])
                
                lines.append(f"\n 级联量化统计 ({len(cascade_results)} 题):")
                lines.append(f"   - 平均Token使用: {avg_cascade_tokens:.0f}/{self.cascade_max_context_tokens}")
                lines.append(f"   - Stage1: 输入={avg_s1_in:.1f} → 输出={avg_s1_out:.1f}")
                lines.append(f"   - Stage2: 冲突={avg_s2_conflicts:.1f}, 丢弃={avg_s2_dropped:.1f}, 输出={avg_s2_out:.1f}")
                lines.append(f"   - Stage3: MMR轮数={avg_s3_mmr:.1f}")
            
            avg_l0_tokens = np.mean([r.l0_tokens for r in valid_results])
            avg_l1_tokens = np.mean([r.l1_tokens for r in valid_results])
            avg_l2_tokens = np.mean([r.l2_tokens for r in valid_results])
            avg_graph_tokens = np.mean([r.graph_tokens for r in valid_results])
            avg_episodic_tokens = np.mean([r.episodic_tokens for r in valid_results])
            avg_total_input = np.mean([r.total_input_tokens for r in valid_results])
            avg_completion = np.mean([r.completion_tokens for r in valid_results])
            total_all_input = sum([r.total_input_tokens for r in valid_results])
            total_all_completion = sum([r.completion_tokens for r in valid_results])
            
            lines.append(f"\n 资源使用 (Token):")
            lines.append(f"   平均每次:")
            lines.append(f"     - 分层: L0={avg_l0_tokens:.0f} | L1={avg_l1_tokens:.0f} | L2={avg_l2_tokens:.0f}")
            lines.append(f"     - 图谱: {avg_graph_tokens:.0f} | 情景: {avg_episodic_tokens:.0f}")
            lines.append(f"     - 总输入: {avg_total_input:.0f} | 输出: {avg_completion:.0f}")
            lines.append(f"   样本累计:")
            lines.append(f"     - 总输入: {total_all_input:,} tokens | 总输出: {total_all_completion:,} tokens")
            
            category_stats = defaultdict(lambda: {"count": 0, "f1_scores": [], "llm_scores": []})
            for r in valid_results:
                cat = r.category
                category_stats[cat]["count"] += 1
                category_stats[cat]["f1_scores"].append(r.evaluation_scores.get("token_f1", 0.0))
                category_stats[cat]["llm_scores"].append(r.evaluation_scores.get("llm_accuracy", 0.0))
            
            if category_stats:
                lines.append(f"\n 类别统计:")
                category_names = {
                    1: "多跳问题", 2: "时间问题", 3: "开放域问题", 
                    4: "单跳问题", 5: "对抗性问题"
                }
                
                for cat, stats in sorted(category_stats.items()):
                    cat_name = category_names.get(cat, f"类别{cat}")
                    avg_cat_f1 = np.mean(stats["f1_scores"]) if stats["f1_scores"] else 0.0
                    avg_cat_llm = np.mean(stats["llm_scores"]) if stats["llm_scores"] else 0.0
                    
                    lines.append(f"\n   {cat_name}:")
                    lines.append(f"     - 测试数: {stats['count']}")
                    lines.append(f"     - 平均F1: {avg_cat_f1:.3f}")
                    lines.append(f"     - 平均LLM准确率: {avg_cat_llm:.3f}")
        else:
            lines.append(f"\n  警告: 所有 {len(sample_results)} 个测试都失败了")
        
        lines.append(f"\n{'='*100}")
        lines.append(f"详细测试结果")
        lines.append(f"{'='*100}")
        
        category_names = {1: "多跳", 2: "时间", 3: "开放域", 4: "单跳", 5: "对抗性"}
        
        for i, result in enumerate(sample_results, 1):
            lines.append(f"\n{'-'*100}")
            lines.append(f"测试 {i}/{len(sample_results)}")
            lines.append(f"{'-'*100}")
            
            cat_name = category_names.get(result.category, f"类别{result.category}")
            lines.append(f"类别: {cat_name} (Category {result.category})")
            lines.append(f"\n问题:\n  {result.question}")
            lines.append(f"\n标准答案:\n  {result.expected_answer}")
            lines.append(f"\n生成答案:\n  {result.final_answer}")
            lines.append(f"\n置信度: {result.confidence_score:.3f}")
            
            
            hier_status = "" if result.hierarchical_context.get("hierarchical_enabled", False) else ""
            graph_status = "" if len(result.graph_retrieved_units) > 0 else ""
            episodic_status = "" if len(result.episodic_retrieved_units) > 0 else ""
            lines.append(f"\n三塔状态:")
            lines.append(f"  分层: {hier_status} | 图谱: {graph_status} | 情景: {episodic_status}")
            
            if result.evaluation_success:
                scores = result.evaluation_scores
                lines.append(f"\n评估分数:")
                lines.append(f"  - F1: {scores.get('token_f1', 0):.3f}")
                lines.append(f"  - 语义相似度: {scores.get('semantic_similarity', 0):.3f}")
                if 'llm_accuracy' in scores:
                    lines.append(f"  - LLM准确率: {scores.get('llm_accuracy', 0):.3f}")
            else:
                lines.append(f"\n  评估失败")
            
            lines.append(f"\n资源使用:")
            lines.append(f"  时间: Hier: {result.hierarchical_retrieval_time:.2f}s | "
                        f"Graph: {result.graph_retrieval_time:.2f}s | "
                        f"Epis: {result.episodic_retrieval_time:.2f}s")
            lines.append(f"  真实检索耗时: {result.total_retrieval_time:.2f}s | "
                        f"端到端延迟: {result.end_to_end_latency:.2f}s")
            lines.append(f"  Tokens: L0={result.l0_tokens} | L1={result.l1_tokens} | L2={result.l2_tokens} | "
                        f"Graph={result.graph_tokens} | Epis={result.episodic_tokens} | "
                        f"Total Input: {result.total_input_tokens}")
            
            
            if result.second_stage_rerank_enabled:
                lines.append(f"\n  二次重排序:")
                lines.append(f"    - 重排序时间: {result.second_stage_rerank_time:.3f}s")
                lines.append(f"    - 一阶段候选: L0={result.first_stage_l0_count} | Graph={result.first_stage_graph_count} | Episodic={result.first_stage_episodic_count} | Total={result.first_stage_total_count}")
                lines.append(f"    - 二阶段选中: L0={result.final_l0_count} | Graph={result.final_graph_count} | Episodic={result.final_episodic_count} | Total={result.final_selected_count}")
            
            if result.cascade_pruner_enabled:
                lines.append(f"\n  级联量化:")
                lines.append(f"    - 模式: {result.cascade_prune_mode}")
                lines.append(f"    - Token使用: {result.cascade_tokens_used}/{self.cascade_max_context_tokens}")
                lines.append(f"    - Stage1: {result.cascade_stage1_input} → {result.cascade_stage1_output}")
                lines.append(f"    - Stage2: 冲突={result.cascade_stage2_conflicts}, 丢弃={result.cascade_stage2_dropped}, 输出={result.cascade_stage2_output}")
                lines.append(f"    - Stage3: MMR轮数={result.cascade_stage3_mmr_iterations}, 多样性惩罚={result.cascade_stage3_diversity_penalties}")
        
        lines.append(f"\n{'='*100}")
        lines.append(f"样本 {sample_id} 报告结束")
        lines.append(f"{'='*100}")
        
        
        report_file = self.output_dir / f"sample_{sample_id}_readable_{timestamp}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        print(f"\n{'='*80}")
        print(f" 样本 {sample_id} 测试完成")
        print(f"{'='*80}")
        if valid_results:
            print(f" 成功: {successful_count}/{len(sample_results)} ({successful_count/len(sample_results)*100:.1f}%)")
            print(f" 平均F1: {avg_f1:.3f} | 语义相似度: {avg_semantic:.3f} | LLM准确率: {avg_llm:.3f}")
            print(f"Average end-to-end latency: {avg_e2e_latency:.2f}s (retrieval: {avg_total_retrieval:.2f}s + generation: {avg_gen_time:.2f}s)")
            print(f" 三塔同时成功: {all_success}/{len(valid_results)} ({all_success/len(valid_results)*100:.1f}%)")
        else:
            print(f" 所有测试失败")
        print(f" 详细报告: {report_file}")
        print(f"{'='*80}\n")
        
        logger.info(f" 样本 {sample_id} 可读性报告: {report_file}")
    
    def _initialize_progress_tracking(self, samples_to_test: List[str]):
        """Initialize progress tracking."""
        self._progress_tracking = {
            'total_samples': len(samples_to_test),
            'completed_samples': 0,
            'failed_samples': [],
            'successful_samples': [],
            'start_time': time.time(),
            'samples_to_test': samples_to_test
        }
        logger.info(f" 进度跟踪已初始化，共 {len(samples_to_test)} 个样本待测试")
    
    def _mark_sample_failed(self, sample_id: str, error_message: str):
        """Run mark sample failed."""
        if hasattr(self, '_progress_tracking'):
            self._progress_tracking['failed_samples'].append({
                'sample_id': sample_id,
                'error': error_message,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            })
            self._progress_tracking['completed_samples'] += 1
            
            failed_count = len(self._progress_tracking['failed_samples'])
            total = self._progress_tracking['total_samples']
            completed = self._progress_tracking['completed_samples']
            
            logger.warning(f" 样本 {sample_id} 标记为失败 ({completed}/{total})")
            logger.warning(f"   错误: {error_message}")
            logger.info(f" 当前进度: 完成 {completed}/{total}, 失败 {failed_count}")
        else:
            logger.warning(f" 样本 {sample_id} 失败: {error_message} (未初始化进度跟踪)")
    
    def _mark_sample_successful(self, sample_id: str, result_count: int):
        """Run mark sample successful."""
        if hasattr(self, '_progress_tracking'):
            self._progress_tracking['successful_samples'].append({
                'sample_id': sample_id,
                'result_count': result_count,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            })
            self._progress_tracking['completed_samples'] += 1
            
            success_count = len(self._progress_tracking['successful_samples'])
            total = self._progress_tracking['total_samples']
            completed = self._progress_tracking['completed_samples']
            
            logger.info(f" 样本 {sample_id} 标记为成功，获得 {result_count} 个结果 ({completed}/{total})")
    
    def _get_available_samples(self) -> List[str]:
        """Get available samples."""
        available_step3 = set()
        available_enhanced = set()
        available_episodic = set()
        
        def is_valid_graph_dir(path: Path) -> bool:
            return (path / "rx_graph.pkl").exists() or (path / "graph_state.json").exists()
        
        if self.step3_graphs_dir.exists():
            for item in self.step3_graphs_dir.iterdir():
                if item.is_dir() and item.name.startswith("conv-"):
                    if is_valid_graph_dir(item):
                        available_step3.add(item.name)
        
        if self.enhanced_graphs_dir.exists():
            for item in self.enhanced_graphs_dir.iterdir():
                if item.is_dir() and item.name.startswith("conv-"):
                    # hierarchical_overview = item / "hierarchical_overview.json"
                    # if hierarchical_overview.exists() and is_valid_graph_dir(item):
                    #     available_enhanced.add(item.name)
                    if is_valid_graph_dir(item): 
                        available_enhanced.add(item.name)
        
        if self.episodic_graphs_dir.exists():
            for item in self.episodic_graphs_dir.iterdir():
                if item.is_dir() and item.name.startswith("conv-"):
                    if is_valid_graph_dir(item):
                        available_episodic.add(item.name)
        
        final_samples = list(available_step3 & available_enhanced & available_episodic)
        
        logger.info(f"发现可用样本 (New Structure): 知识图谱={len(available_step3)}, "
                   f"分层图谱={len(available_enhanced)}, "
                   f"情景记忆={len(available_episodic)}, "
                   f"交集={len(final_samples)}")
        
        if not final_samples:
             logger.warning(" 未找到任何匹配样本。请检查:")
             logger.warning(f"1. 目录路径是否正确: {self.step3_graphs_dir}")
             logger.warning(f"2. 文件夹是否以 'conv-' 开头")
             logger.warning(f"3. 是否包含 'rx_graph.pkl' 或 'graph_state.json'")
        
        return sorted(final_samples)
    
    def generate_final_summary(self):
        """Generate final summary."""
        logger.info(" 生成最终汇总报告...")
        
        sample_files = sorted(self.output_dir.glob("sample_conv-*.json"))
        sample_files = [f for f in sample_files if "_readable_" not in f.name]
        
        if not sample_files:
            logger.warning("没有找到样本结果文件")
            return
        
        all_results = []
        sample_summaries = {}
        
        for sample_file in sample_files:
            try:
                with open(sample_file, 'r', encoding='utf-8') as f:
                    sample_data = json.load(f)
                    sample_info = sample_data.get("sample_info", {})
                    sample_id = sample_info.get("sample_id", "unknown")
                    results = sample_data.get("results", [])
                    all_results.extend(results)
                    
                    sample_summaries[sample_id] = {
                        "test_count": sample_info.get("test_count", 0),
                        "successful_count": sample_info.get("successful_count", 0),
                        "failed_count": sample_info.get("failed_count", 0),
                    }
            except Exception as e:
                logger.warning(f"读取样本文件失败 {sample_file}: {e}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        final_summary = {
            "test_info": {
                "test_name": "LoCoMo Tri-Tower Retrieval Benchmark",
                "total_samples": len(sample_files),
                "total_results": len(all_results),
                "successful_results": len([r for r in all_results if r.get("evaluation_success", False)]),
                "failed_results": len([r for r in all_results if not r.get("evaluation_success", False)]),
                "timestamp": datetime.now().isoformat(),
                "fusion_strategy": self.fusion_strategy,
                "fusion_weights": self.fusion_weights,
            },
            "sample_summaries": sample_summaries,
            "aggregate_results": all_results,
        }
        
        summary_file = self.output_dir / f"final_summary_{timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(final_summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f" 最终汇总JSON已生成: {summary_file}")
        
        self._generate_final_readable_summary(final_summary, timestamp)
    
    def _generate_final_readable_summary(self, final_summary: Dict, timestamp: str):
        """Generate final readable summary."""
        lines = []
        
        lines.append("=" * 100)
        lines.append("LoCoMo三塔召回系统 - 最终汇总报告")
        lines.append("=" * 100)
        lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 100)
        
        test_info = final_summary["test_info"]
        lines.append(f"\n 测试概况:")
        lines.append(f"   - 测试样本数: {test_info['total_samples']}")
        lines.append(f"   - 总测试数: {test_info['total_results']}")
        total_results = test_info['total_results']
        if total_results > 0:
            lines.append(f"   - 成功测试: {test_info['successful_results']} ({test_info['successful_results']/total_results*100:.1f}%)")
            lines.append(f"   - 失败测试: {test_info['failed_results']} ({test_info['failed_results']/total_results*100:.1f}%)")
        lines.append(f"   - 融合策略: {test_info['fusion_strategy']}")
        lines.append(f"   - 融合权重: 分层={test_info['fusion_weights']['hierarchical']}, "
                    f"图谱={test_info['fusion_weights']['graph']}, "
                    f"情景={test_info['fusion_weights']['episodic']}")
        
        sample_summaries = final_summary.get("sample_summaries", {})
        if sample_summaries:
            lines.append(f"\n 各样本测试摘要:")
            lines.append(f"\n{'样本ID':<15} {'测试数':>8} {'成功数':>8} {'失败数':>8} {'成功率':>10}")
            lines.append(f"{'-'*60}")
            
            for sample_id, summary in sorted(sample_summaries.items()):
                test_count = summary.get('test_count', 0) or 0
                success_count = summary.get('successful_count', 0) or 0
                failed_count = summary.get('failed_count', 0) or 0
                success_rate = success_count / test_count * 100 if test_count > 0 else 0.0
                
                lines.append(f"{sample_id:<15} {test_count:>8} {success_count:>8} {failed_count:>8} {success_rate:>9.1f}%")
        
        all_results = final_summary.get("aggregate_results", [])
        valid_results = [r for r in all_results if r.get("evaluation_success", False)]
        
        if valid_results:
            def safe_get(result, key, default=0.0):
                scores = result.get("evaluation_scores", {})
                return scores.get(key, default) if scores else default
            
            avg_f1 = np.mean([safe_get(r, "token_f1") for r in valid_results])
            avg_semantic = np.mean([safe_get(r, "semantic_similarity") for r in valid_results])
            avg_llm = np.mean([safe_get(r, "llm_accuracy") for r in valid_results])
            
            avg_hier_time = np.mean([r.get("hierarchical_time", 0) for r in valid_results])
            avg_graph_time = np.mean([r.get("graph_time", 0) for r in valid_results])
            avg_episodic_time = np.mean([r.get("episodic_time", 0) for r in valid_results])
            avg_gen_time = np.mean([r.get("generation_time", 0) for r in valid_results])
            
            hier_success = sum(1 for r in valid_results if r.get("hierarchical_success", False))
            graph_success = sum(1 for r in valid_results if r.get("graph_results_count", 0) > 0)
            episodic_success = sum(1 for r in valid_results if r.get("episodic_results_count", 0) > 0)
            all_success = sum(1 for r in valid_results
                if r.get("hierarchical_success", False)
                and r.get("graph_results_count", 0) > 0
                and r.get("episodic_results_count", 0) > 0)
            
            lines.append(f"\n 整体性能指标:")
            lines.append(f"   - 平均F1分数: {avg_f1:.3f}")
            lines.append(f"   - 平均语义相似度: {avg_semantic:.3f}")
            lines.append(f"   - 平均LLM准确率: {avg_llm:.3f}")
            
            lines.append(f"\nAverage Timing Performance:")
            lines.append(f"   - 分层检索: {avg_hier_time:.3f}s")
            lines.append(f"   - 图检索: {avg_graph_time:.3f}s")
            lines.append(f"   - 情景检索: {avg_episodic_time:.3f}s")
            lines.append(f"   - 答案生成: {avg_gen_time:.3f}s")
            lines.append(f"   - 总计: {avg_hier_time + avg_graph_time + avg_episodic_time + avg_gen_time:.3f}s")
            
            lines.append(f"\n 三塔成功率:")
            lines.append(f"   - 分层检索: {hier_success}/{len(valid_results)} ({hier_success/len(valid_results)*100:.1f}%)")
            lines.append(f"   - 图检索: {graph_success}/{len(valid_results)} ({graph_success/len(valid_results)*100:.1f}%)")
            lines.append(f"   - 情景检索: {episodic_success}/{len(valid_results)} ({episodic_success/len(valid_results)*100:.1f}%)")
            lines.append(f"   - 三塔同时成功: {all_success}/{len(valid_results)} ({all_success/len(valid_results)*100:.1f}%)")
            
            category_stats = defaultdict(lambda: {"count": 0, "f1_scores": [], "llm_scores": []})
            for r in valid_results:
                cat = r.get("category", 0)
                if cat:
                    category_stats[cat]["count"] += 1
                    category_stats[cat]["f1_scores"].append(safe_get(r, "token_f1"))
                    category_stats[cat]["llm_scores"].append(safe_get(r, "llm_accuracy"))
            
            if category_stats:
                lines.append(f"\n 类别性能:")
                category_names = {
                    1: "多跳问题", 2: "时间问题", 3: "开放域问题", 
                    4: "单跳问题", 5: "对抗性问题"
                }
                
                for cat, stats in sorted(category_stats.items()):
                    if stats["count"] == 0:
                        continue
                    cat_name = category_names.get(cat, f"类别{cat}")
                    avg_cat_f1 = np.mean(stats["f1_scores"]) if stats["f1_scores"] else 0.0
                    avg_cat_llm = np.mean(stats["llm_scores"]) if stats["llm_scores"] else 0.0
                    
                    lines.append(f"\n   {cat_name} ({stats['count']}题):")
                    lines.append(f"     - 平均F1: {avg_cat_f1:.3f}")
                    lines.append(f"     - 平均LLM准确率: {avg_cat_llm:.3f}")
            
            
            routing_results = [r for r in valid_results if r.get("routing_info", "")]
            if routing_results:
                lines.append(f"\n 路由统计 ({len(routing_results)}/{len(valid_results)} 题启用路由):")
                routing_counter = defaultdict(lambda: {"count": 0, "llm_scores": []})
                for r in routing_results:
                    info = r["routing_info"]
                    routing_counter[info]["count"] += 1
                    routing_counter[info]["llm_scores"].append(safe_get(r, "llm_accuracy"))
                for info, rstat in sorted(routing_counter.items(), key=lambda x: -x[1]["count"]):
                    avg_acc = np.mean(rstat["llm_scores"]) if rstat["llm_scores"] else 0.0
                    lines.append(f"   {info}: {rstat['count']}题, LLM_Acc={avg_acc:.3f}")
        
        lines.append(f"\n{'='*100}")
        lines.append(f"报告生成完成")
        lines.append(f"{'='*100}")
        
        
        summary_file = self.output_dir / f"final_summary_readable_{timestamp}.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        print('\n'.join(lines))
        
        logger.info(f" 最终汇总报告: {summary_file}")


def _parse_tower_min_ratio_str(s: Optional[str]) -> Optional[Dict[str, float]]:
    """Parse tower min ratio str."""
    if not s:
        return None
    result: Dict[str, float] = {}
    for pair in s.split(","):
        pair = pair.strip()
        if ":" not in pair:
            continue
        k, v = pair.split(":", 1)
        result[k.strip()] = float(v.strip())
    return result


def main():
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(description="LoCoMo三塔召回Benchmark测试")
    
    parser.add_argument("--step3-graphs-dir",
                       default=str(paths.LOCOMO_ENTITY_RELATION_STEP3_DIR),
                       help="知识图谱（实体关系）数据目录")
    # Dataset-specific handling used by the reproduction workflow.
    parser.add_argument("--enhanced-graphs-dir",
                       default=str(paths.LOCOMO_HIERARCHICAL_CONTENT_STEP4_DIR),
                       help="分层图谱数据目录")
    parser.add_argument("--episodic-graphs-dir",
                       default=str(paths.LOCOMO_EPISODIC_STEP3_DIR),
                       help="情景记忆图谱数据目录")
    parser.add_argument("--qa-dataset",
                       default=str(paths.LOCOMO_RAW_FILE),
                       help="QA数据集路径")
    parser.add_argument("--output-dir",
                       default=str(paths.LOCOMO_TASK_EVAL_RESULTS_DIR / "locomo_tri_tower_benchmark_results_claude_content"),
                       help="输出目录")
    
    parser.add_argument("--llm-model",
                       default="gpt-4o-mini-closeai",
                       help="答案生成LLM模型")
    parser.add_argument("--llm-evaluate-model",
                       default="gpt-4o-mini-closeai",
                       help="答案评估LLM模型")
    parser.add_argument("--generation-max-tokens", type=int, default=2000,
                       help="Maximum output tokens for answer generation (default: 2000).")
    
    parser.add_argument("--sample-ids", nargs='+',
                       help="指定要测试的样本ID列表")
    parser.add_argument("--max-questions", type=int, default=None,
                       help="可选的每个样本最大测试问题数；用于低成本烟雾测试，默认运行全部问题")
    
    parser.add_argument("--topk-hierarchical", type=int, default=15,
                       help="分层记忆统一检索top-k（L0/L1/L2一起检索）")
    parser.add_argument("--topk-similarity", type=int, default=30,
                       help="图检索语义top-k（第一阶段）")
    parser.add_argument("--topk-graph", type=int, default=0,
                       help="图检索实体关系top-k（默认0禁用，设置>0启用）")
    parser.add_argument("--topk-episodic", type=int, default=30,
                       help="情景记忆检索top-k（第一阶段）")
    parser.add_argument("--no-entity-relation", action="store_true",
                       help="禁用实体关系检索")
    
    
    parser.add_argument("--enable-second-stage-rerank", action="store_true", default=True,
                       help="启用二次重排序（默认启用）")
    parser.add_argument("--no-second-stage-rerank", action="store_true",
                       help="禁用二次重排序")
    parser.add_argument("--second-stage-rerank-method", type=str, default=None,
                       help="二次重排序方法（默认与--reranker-type相同）")
    parser.add_argument("--final-top-k", type=int, default=20,
                       help="二次重排序后最终保留的top-k数量")
    parser.add_argument("--threshold", type=float, default=0.0,
                       help="Rerank score threshold for filtering results. Results with score < threshold will be filtered. Default 0.0 means no filtering.")
    parser.add_argument("--rerank-strategy", 
                       choices=["tower_separate", "unified_rerank"],
                       default="tower_separate",
                       help="重排序策略: tower_separate(分层塔直通车+其他两塔重排) | unified_rerank(三塔统一重排序)")
    
    
    parser.add_argument("--reranker-type",
                       choices=["baai", "qwen", "jina", "qwen-sili", "qwen-dashscope", "gte-dashscope"],
                       default="baai",
                       help="重排序器类型 (baai/qwen/jina本地, qwen-sili/qwen-dashscope/gte-dashscope云端API)")
    
    parser.add_argument("--fusion-strategy",
                       choices=["simple", "weighted", "context_aware"],
                       default="context_aware",
                       help="融合策略")
    parser.add_argument("--weight-hierarchical", type=float, default=0.35,
                       help="分层记忆权重")
    parser.add_argument("--weight-graph", type=float, default=0.35,
                       help="知识图谱权重")
    parser.add_argument("--weight-episodic", type=float, default=0.30,
                       help="情景记忆权重")
    
    parser.add_argument('--parallel', action='store_true', 
                   help='启用三塔并行检索（默认禁用，使用串行）')
    parser.add_argument("--max-workers", type=int, default=3,
                       help="最大工作线程数")
    
    
    parser.add_argument("--enable-router", action="store_true",
                       help="Enable the category router and select tower combinations by question category.")
    parser.add_argument("--router-strategy",
                       choices=["aggressive", "conservative"],
                       default="aggressive",
                       help="路由策略: aggressive(路由所有类别到最优配置) | conservative(仅路由高置信度类别)")
    
    parser.add_argument("--enable-cascade-pruner", action="store_true",
                       help="启用级联置信度剪枝器（替代二次重排序）")
    parser.add_argument("--cascade-max-context-tokens", type=int, default=2500,
                       help="级联剪枝最大上下文token数 (默认: 2500)")
    parser.add_argument("--cascade-prune-mode",
                       choices=["BUDGET_MAX", "STRICT_THRESHOLD", "CLIFF_EARLY_STOP", "DYNAMIC_ADAPTIVE"],
                       default="BUDGET_MAX",
                       help="级联剪枝模式 (默认: BUDGET_MAX)")
    parser.add_argument("--cascade-mad-multiplier", type=float, default=2.5,
                       help="Stage1 MAD离群点检测倍率，值越大越保守 (默认: 2.5)")
    parser.add_argument("--cascade-cliff-tolerance", type=float, default=2.0,
                       help="CLIFF_EARLY_STOP模式悬崖容忍度，作用于原始 logits 尺度 (默认: 2.0)")
    parser.add_argument("--cascade-absolute-min-score", type=float, default=0.0,
                       help="STRICT_THRESHOLD模式绝对最低分数阈值 (默认: 0.0)")
    parser.add_argument("--cascade-lambda-mmr", type=float, default=0.6,
                       help="Stage3 MMR多样性权重 (默认: 0.6)")
    parser.add_argument("--no-cascade-stage2", action="store_true",
                       help="禁用级联Stage2（跨塔消歧）")
    parser.add_argument("--no-cascade-stage3-mmr", action="store_true",
                       help="禁用级联Stage3（MMR多样性打包）")
    parser.add_argument("--no-cascade-stage1", action="store_true",
                       help="禁用级联Stage1（硬过滤）")
    parser.add_argument("--no-cascade-cap-to-input", action="store_true",
                       help="禁用cap_to_input_tokens（默认启用，防止级联膨胀超出输入token总量）")
    parser.add_argument("--cascade-tower-min-ratio", type=str, default=None,
                       help="Stage3 per-tower 最低配额，格式: 'H:0.50,E:0.20,KG:0.15'。"
                            "省略则禁用 tower reservation（向后兼容）")
    parser.add_argument("--cascade-adaptive-dataset", type=str, default=None,
                       help="DYNAMIC_ADAPTIVE模式使用的数据集名称，用于查找类别适配器 (e.g. 'locomo')")
    parser.add_argument(
        "--no-save-individual-reports",
        action="store_true",
        help="不保存每个问题的 individual report（默认保存，便于后续消融分析）",
    )
    
    # Avoid mutating LogRecord fields before other handlers process the record.
    parser.add_argument("--log-level",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO",
                       help="日志级别")
    
    args = parser.parse_args()

    if args.max_questions is not None and args.max_questions <= 0:
        parser.error("--max-questions must be a positive integer")
    
    # logging.basicConfig(
    #     level=getattr(logging, args.log_level),
    #     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    # )
    
    print("=" * 80)
    print(" LoCoMo三塔召回（Tri-Tower）Benchmark测试系统")
    print("=" * 80)
    print("三塔架构:")
    print("  1  分层记忆塔（Hierarchical Memory）")
    print("  2  知识图谱塔（Knowledge Graph）")
    print("  3  情景记忆塔（Episodic Memory）+ 时间注入")
    print("=" * 80)
    print(f" 知识图谱目录: {args.step3_graphs_dir}")
    print(f" 分层图谱目录: {args.enhanced_graphs_dir}")
    print(f" 情景记忆目录: {args.episodic_graphs_dir}")
    print(f" QA数据集: {args.qa_dataset}")
    print(f" 输出目录: {args.output_dir}")
    print(f" 答案生成模型: {args.llm_model}")
    print(f" 答案评估模型: {args.llm_evaluate_model}")
    print(f" 重排序器: {args.reranker_type}")
    
    
    if args.reranker_type in ['qwen-sili']:
        api_key = os.getenv("SILICONFLOW_API_KEY")
        print(f"{'' if api_key else ' '} 云端重排序 API (Siliconflow): {'已配置' if api_key else '未配置 (请设置 SILICONFLOW_API_KEY)'}")
    elif args.reranker_type in ['qwen-dashscope', 'gte-dashscope']:
        api_key = os.getenv("DASHSCOPE_API_KEY")
        print(f"{'' if api_key else ' '} 云端重排序 API (DashScope): {'已配置' if api_key else '未配置 (请设置 DASHSCOPE_API_KEY)'}")
    
    print(f" 并行检索: {'启用' if args.parallel else '禁用'}")
    print(f" 融合策略: {args.fusion_strategy}")
    print(f"  融合权重: 分层={args.weight_hierarchical}, 图谱={args.weight_graph}, 情景={args.weight_episodic}")
    print("-" * 80)
    print(f" 第一阶段检索配置:")
    print(f"   分层: top_k={args.topk_hierarchical} (L0/L1/L2统一检索)")
    print(f"   图谱: {args.topk_similarity} (实体关系: {'启用' if args.topk_graph > 0 else '禁用'}, top_k={args.topk_graph})")
    print(f"   情景: {args.topk_episodic}")
    
    
    enable_second_stage = args.enable_second_stage_rerank and not args.no_second_stage_rerank
    second_stage_method = args.second_stage_rerank_method or args.reranker_type
    
    
    use_entity_relation = (not args.no_entity_relation) and (args.topk_graph > 0)
    
    if enable_second_stage:
        strategy_desc = {
            "tower_separate": "分层塔直通车 + 其他两塔重排序",
            "unified_rerank": "三塔统一重排序"
        }.get(args.rerank_strategy, args.rerank_strategy)
        print(f" 二次重排序:  启用")
        print(f"   重排序策略: {args.rerank_strategy} ({strategy_desc})")
        print(f"   重排序器: {second_stage_method}")
        print(f"   最终Top-K: {args.final_top_k}")
    else:
        print(f" 二次重排序:  禁用")
    
    
    tower_router = None
    if args.enable_router:
        tower_router = LocomoTowerRouter(
            model_name=args.llm_model,
            strategy=args.router_strategy,
        )
        
        args.output_dir = f"{args.output_dir}_routed_{args.router_strategy}"
        print(f" 塔路由器:  启用")
        print(f"   模型: {args.llm_model}")
        print(f"   策略: {args.router_strategy}")
        print(f"   输出目录(已修改): {args.output_dir}")
    else:
        print(f" 塔路由器:  禁用 (使用静态配置)")
    
    enable_cascade = args.enable_cascade_pruner
    if enable_cascade:
        args.output_dir = f"{args.output_dir}_cascade"
        print(f" 级联量化:  启用")
        print(f"   剪枝模式: {args.cascade_prune_mode}")
        print(f"   最大上下文Token: {args.cascade_max_context_tokens}")
        print(f"   Stage2(跨塔消歧): {'启用' if not args.no_cascade_stage2 else '禁用'}")
        print(f"   Stage3(MMR打包): {'启用' if not args.no_cascade_stage3_mmr else '禁用'}")
        print(f"   输出目录(已修改): {args.output_dir}")
    else:
        print(f" 级联量化:  禁用")
    print("=" * 80)
    
    if args.sample_ids:
        print(f" 指定样本: {args.sample_ids}")
    if args.max_questions is not None:
        print(f" 每个样本最大测试问题数: {args.max_questions}")
    
    try:
        print("\n 初始化三塔Benchmark测试器...")
        
        
        reranker_manager = RerankerManager()
        
        fusion_weights = {
            "hierarchical": args.weight_hierarchical,
            "graph": args.weight_graph,
            "episodic": args.weight_episodic,
        }
        
        benchmark = LoCoMoTriTowerBenchmark(
            step3_graphs_dir=args.step3_graphs_dir,
            enhanced_graphs_dir=args.enhanced_graphs_dir,
            episodic_graphs_dir=args.episodic_graphs_dir,
            qa_dataset_path=args.qa_dataset,
            output_dir=args.output_dir,
            llm_model=args.llm_model,
            llm_evaluate_model=args.llm_evaluate_model,
            target_sample_ids=args.sample_ids,
            max_questions=args.max_questions,
            topk_hierarchical=args.topk_hierarchical,
            topk_similarity=args.topk_similarity,
            topk_graph=args.topk_graph,
            topk_episodic=args.topk_episodic,
            use_entity_relation=use_entity_relation,
            
            enable_second_stage_rerank=enable_second_stage,
            second_stage_rerank_method=second_stage_method,
            final_top_k=args.final_top_k,
            rerank_threshold=args.threshold,
            rerank_strategy=args.rerank_strategy,
            
            reranker_type=args.reranker_type,
            reranker_manager=reranker_manager,
            fusion_strategy=args.fusion_strategy,
            fusion_weights=fusion_weights,
            tower_router=tower_router,
            enable_cascade_pruner=enable_cascade,
            cascade_prune_mode=args.cascade_prune_mode,
            cascade_mad_multiplier=args.cascade_mad_multiplier,
            cascade_cliff_tolerance=args.cascade_cliff_tolerance,
            cascade_absolute_min_score=args.cascade_absolute_min_score,
            cascade_max_context_tokens=args.cascade_max_context_tokens,
            cascade_lambda_mmr=args.cascade_lambda_mmr,
            cascade_enable_stage2=not args.no_cascade_stage2,
            cascade_enable_stage3_mmr=not args.no_cascade_stage3_mmr,
            cascade_enable_stage1=not args.no_cascade_stage1,
            cascade_cap_to_input_tokens=not args.no_cascade_cap_to_input,
            cascade_tower_min_ratio=_parse_tower_min_ratio_str(args.cascade_tower_min_ratio),
            cascade_adaptive_dataset=args.cascade_adaptive_dataset,
            save_individual_reports=not args.no_save_individual_reports,
            parallel_towers=args.parallel,
            max_workers=args.max_workers,
            generation_max_tokens=args.generation_max_tokens,
        )
        
        print(" 三塔Benchmark测试器初始化完成")
        
        print("\n 开始运行三塔Benchmark测试...")
        benchmark.run_tri_tower_benchmark(sequential_mode=True)
        
        print("\n 生成最终汇总报告...")
        benchmark.generate_final_summary()
        
        print(f"\n 三塔Benchmark测试完成!")
        print(f" 结果目录: {args.output_dir}")
        
        return 0
        
    except Exception as e:
        print(f"\n Benchmark测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        try:
            cleanup_evaluation_models()
            print("\n 资源清理完成")
        except Exception as e:
            logger.warning(f"资源清理失败: {e}")


if __name__ == "__main__":
    exit(main())
