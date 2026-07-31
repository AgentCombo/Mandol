#!/usr/bin/env python3
"""LongMemEval routed tri-tower benchmark with cascade quantification."""

import os
import sys
import json
import re
import logging
import time
import argparse
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from contextlib import contextmanager
from tqdm import tqdm


# Avoid mutating LogRecord fields before other handlers process the record.
from mandol.utils.logging_config import setup_logging, create_module_logger, auto_configure_logging
if auto_configure_logging() is None:
    setup_logging(level=logging.INFO)
logger = create_module_logger("benchmark_triple_content")

from mandol.core.semantic_graph import SemanticGraph
from mandol.core.memory_unit import MemoryUnit
from mandol.llm.llm_client import LLMClient
from mandol.retrieval.retrieval_interface import RetrievalMethod
from mandol.retrieval.rerank_manager import RerankerManager
from mandol.memory_router.longmemeval_tower_router import LongMemEvalTowerRouter, LongMemEvalTowerRoutingConfig
from mandol.quantification.cascade_pruner import (
    CascadeConfidencePruner, EnhancedCandidateChunk,
    TowerSource, CascadePruneResult,
)
from mandol.quantification.confidence_pruner import PruneMode
from mandol.quantification.adapters import LongMemEvalPrunerAdapter
import mandol.quantification.adapters.longmemeval_category  # noqa: F401
from mandol.core import paths


try:
    from benchmark_longmemeval.task_eval.evaluation import (
        calculate_comprehensive_scores,
        cleanup_evaluation_models
    )
    EVALUATION_AVAILABLE = True
except ImportError:
    EVALUATION_AVAILABLE = False
    logger.warning("Evaluation module is unavailable.")

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.warning("tiktoken is not installed; approximate token counting will be used.")



@dataclass
class TripleTestCase:
    """One LongMemEval QA item used by the tri-tower benchmark."""
    qa_index: int
    question_id: str
    question: str
    answer: str
    question_type: str = ""
    category: str = ""
    query_date: str = "Unknown Date"


@dataclass
class TokenStats:
    """Token counters recorded for one benchmark example."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    context_tokens: int = 0
    sentence_context_tokens: int = 0
    episodic_context_tokens: int = 0
    entity_context_tokens: int = 0


@dataclass
class TripleRetrievalDetails:
    """Retrieval and reranking diagnostics for one benchmark example."""
    total_retrieval_time: float = 0.0
    graph_loading_time: float = 0.0  
    
    sentence_enabled: bool = True
    sentence_retrieved_count: int = 0
    sentence_retrieval_time: float = 0.0
    
    episodic_enabled: bool = True
    episodic_retrieved_count: int = 0
    episodic_retrieval_time: float = 0.0
    episodic_facts_by_category: Dict[str, int] = field(default_factory=dict)
    
    entity_enabled: bool = True
    entity_retrieved_count: int = 0
    entity_retrieval_time: float = 0.0
    entity_types_found: Dict[str, int] = field(default_factory=dict)
    
    
    second_stage_rerank_enabled: bool = False
    second_stage_rerank_method: str = "none"
    second_stage_rerank_time: float = 0.0
    first_stage_total_count: int = 0
    final_selected_count: int = 0
    
    fusion_method: str = "concatenation"
    rerank_method: str = "none"
    
    cascade_pruner_enabled: bool = False
    cascade_prune_mode: str = "none"
    cascade_tokens_used: int = 0
    cascade_stage1_input: int = 0
    cascade_stage1_output: int = 0
    cascade_stage2_conflicts: int = 0
    cascade_stage2_dropped: int = 0
    cascade_stage2_output: int = 0
    cascade_stage3_mmr_iterations: int = 0
    cascade_stage3_diversity_penalties: int = 0


@dataclass 
class TripleRetrievalResult:
    """Per-question LongMemEval tri-tower benchmark result."""
    qa_index: int
    question: str
    ground_truth: str
    generated_answer: str
    reasoning: str
    
    scores: Dict[str, float] = field(default_factory=dict)
    
    token_stats: TokenStats = field(default_factory=TokenStats)
    
    retrieval_details: TripleRetrievalDetails = field(default_factory=TripleRetrievalDetails)
    
    retrieved_contents: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    #        "episodic": [...], "entity": [...],
    
    
    
    routing_info: str = ""
    
    success: bool = False
    error_message: str = ""



class LongMemEvalTripleFusionBenchmark:
    """Run LongMemEval tri-tower retrieval, routing, generation, and evaluation."""
    
    
    VALID_RERANK_METHODS = ["baai", "qwen", "jina", "qwen-sili", "qwen-dashscope", "gte-dashscope", "none"]
    
    STABLE_CATEGORIES = [
        "USER_ATTRIBUTE", "PREFERENCE_HABIT", "RELATIONSHIP_FACT", "KNOWLEDGE", 
        "IMPLICIT_CONSTRAINT", "INVENTORY_ITEM"
    ]
    
    def __init__(self,
                 dataset_size: str = "s",
                 dataset_dir: Optional[str] = None,
                 sentence_graph_dir: Optional[str] = None,
                 episodic_graph_dir: Optional[str] = None,
                 entity_graph_dir: Optional[str] = None,
                 llm_client: Optional[LLMClient] = None,
                 llm_evaluate_client: Optional[LLMClient] = None,
                 output_dir: str = str(paths.LONGMEMEVAL_TASK_EVAL_RESULTS_DIR / "triple_fusion"),
                 
                 sentence_top_k: int = 5,
                 episodic_top_k: int = 5,
                 entity_top_k: int = 5,
                 enable_sentence: bool = True,
                 enable_episodic: bool = True,
                 enable_entity: bool = True,
                 max_tests: Optional[int] = None,
                 rerank_method: str = "baai",
                 fusion_method: str = "concatenation",
                 enable_second_stage_rerank: bool = True,  
                 second_stage_rerank_method: Optional[str] = None,  
                 first_stage_top_k: Optional[int] = None,  
                 final_top_k: int = 15,
                 start_qa: Optional[int] = None,
                 end_qa: Optional[int] = None,
                 
                 tower_router: Optional[LongMemEvalTowerRouter] = None,
                 generation_max_tokens: int = 1000,
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
                 cascade_adaptive_dataset: Optional[str] = None):
        """Initialize the LongMemEval tri-tower benchmark runner."""
        self.dataset_size = dataset_size
        
        self.dataset_dir = Path(dataset_dir) if dataset_dir else \
            Path(__file__).parent.parent / "dataset" / "LongMemEval"
        self.dataset_path = self.dataset_dir / f"longmemeval_{dataset_size}_cleaned.json"
        
        self.sentence_graph_dir = Path(sentence_graph_dir) if sentence_graph_dir else \
            self.dataset_dir / "longmemeval_hierarchical" / "step3_semantic_graphs"
        self.episodic_graph_dir = Path(episodic_graph_dir) if episodic_graph_dir else \
            self.dataset_dir / "episodic_memory_graphs"
        self.entity_graph_dir = Path(entity_graph_dir) if entity_graph_dir else \
            self.dataset_dir / "entity_relation_graphs"
        
        self.llm_client = llm_client
        self.llm_evaluate_client = llm_evaluate_client
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        
        self.sentence_top_k = sentence_top_k
        self.episodic_top_k = episodic_top_k
        self.entity_top_k = entity_top_k
        
        self.enable_sentence = enable_sentence
        self.enable_episodic = enable_episodic
        self.enable_entity = enable_entity
        
        self.max_tests = max_tests
        self.fusion_method = fusion_method
        
        self.enable_second_stage_rerank = enable_second_stage_rerank
        self.second_stage_rerank_method = second_stage_rerank_method or rerank_method
        self.first_stage_top_k = first_stage_top_k  
        self.final_top_k = final_top_k
        
        self.start_qa = start_qa
        self.end_qa = end_qa
        
        
        self.tower_router = tower_router
        self.generation_max_tokens = generation_max_tokens

        self.enable_cascade_pruner = enable_cascade_pruner
        self.cascade_pruner = None
        self.cascade_adapter = None
        self.cascade_max_context_tokens = cascade_max_context_tokens
        self.cascade_lambda_mmr = cascade_lambda_mmr
        self.cascade_enable_stage1 = cascade_enable_stage1
        self.cascade_enable_stage2 = cascade_enable_stage2
        self.cascade_enable_stage3_mmr = cascade_enable_stage3_mmr
        self.cascade_adaptive_dataset = cascade_adaptive_dataset
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
                self.cascade_adapter = LongMemEvalPrunerAdapter()
                self.cascade_prune_mode = _resolved_mode
                logger.info(f" 级联量化剪枝:  初始化成功 (mode={_resolved_mode.value}, "
                           f"max_tokens={cascade_max_context_tokens}, λ_mmr={cascade_lambda_mmr})")
            except Exception as e:
                logger.error(f" 级联量化剪枝初始化失败: {e}")
                self.enable_cascade_pruner = False
                self.cascade_pruner = None
        
        
        if rerank_method not in self.VALID_RERANK_METHODS:
            raise ValueError(f"无效的重排序方法: {rerank_method}，支持: {self.VALID_RERANK_METHODS}")
        self.rerank_method = rerank_method
        
        self.stats = {
            'total_tests': 0,
            'successful_tests': 0,
            'failed_tests': 0,
            'sentence_retrieval_count': 0,
            'episodic_retrieval_count': 0,
            'entity_retrieval_count': 0
        }
        
        self._print_config()
    
    def _print_config(self):
        """Run print config."""
        logger.info("=" * 100)
        logger.info(" LongMemEval 三重检索融合 Benchmark (Triple Fusion)")
        logger.info("=" * 100)
        logger.info(f" 数据集: {self.dataset_path}")
        logger.info(f" Sentence图谱: {self.sentence_graph_dir}")
        logger.info(f" Episodic图谱: {self.episodic_graph_dir}")
        logger.info(f" Entity图谱: {self.entity_graph_dir}")
        logger.info("-" * 100)
        logger.info(f" 检索配置:")
        logger.info(f"   Sentence-level: {' 启用' if self.enable_sentence else ' 禁用'} (Top-K={self.sentence_top_k})")
        logger.info(f"   Episodic Memory: {' 启用' if self.enable_episodic else ' 禁用'} (Top-K={self.episodic_top_k})")
        logger.info(f"   Entity Relation: {' 启用' if self.enable_entity else ' 禁用'} (Top-K={self.entity_top_k})")
        logger.info(f" 第一阶段重排序方法: {self.rerank_method}")
        logger.info(f" 融合方法: {self.fusion_method}")
        
        if self.enable_second_stage_rerank:
            first_stage_k = self.first_stage_top_k or f"({self.sentence_top_k}+{self.episodic_top_k}+{self.entity_top_k})"
            logger.info("-" * 100)
            logger.info(f" 两阶段检索策略:  启用")
            logger.info(f"   第一阶段 Top-K: {first_stage_k}")
            logger.info(f"   二次重排序方法: {self.second_stage_rerank_method}")
            logger.info(f"   最终保留数量: {self.final_top_k}")
        else:
            logger.info(f" 两阶段检索策略:  禁用")
        
        if self.tower_router:
            logger.info(f" 塔路由器:  启用 (模型={self.tower_router.model_name}, 策略={self.tower_router.strategy})")
        else:
            logger.info(f" 塔路由器:  禁用 (使用静态配置)")
        logger.info(f" 生成最大输出 Tokens: {self.generation_max_tokens}")

        if self.enable_cascade_pruner and self.cascade_pruner:
            logger.info("-" * 100)
            logger.info(f" 级联量化剪枝:  启用")
            logger.info(f"   剪枝模式: {self.cascade_prune_mode.value}")
            logger.info(f"   最大Token数: {self.cascade_max_context_tokens}")
            logger.info(f"   λ_mmr: {self.cascade_lambda_mmr}")
            logger.info(f"   Stage1硬过滤: {'启用' if self.cascade_enable_stage1 else '禁用'}")
            logger.info(f"   Stage2跨塔消歧: {'启用' if self.cascade_enable_stage2 else '禁用'}")
            logger.info(f"   Stage3 MMR: {'启用' if self.cascade_enable_stage3_mmr else '禁用'}")
        else:
            logger.info(f" 级联量化剪枝:  禁用")
        
        logger.info("=" * 100)
    
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


    _WEEKDAY_RE = re.compile(r"\s*\([A-Za-z]+\)\s*")

    def _parse_date_to_timestamp(self, date_str: str) -> float:
        """Parse date to timestamp."""
        if not date_str or not isinstance(date_str, str):
            return 0.0
        cleaned = date_str.strip()
        if cleaned.lower() in ("", "unknown date", "unknown", "n/a", "none"):
            return 0.0

        cleaned = self._WEEKDAY_RE.sub(" ", cleaned).strip()

        _FORMATS = (
            "%Y/%m/%d %H:%M",      # LongMemEval: "2023/05/30 23:40"
            "%Y/%m/%d %H:%M:%S",   # "2023/05/30 23:40:00"
            "%Y/%m/%d",            # "2023/05/30"
            "%Y-%m-%d %H:%M:%S",   # ISO datetime
            "%Y-%m-%d %H:%M",      # ISO datetime no-sec
            "%Y-%m-%d",            # ISO date
            "%Y-%m-%dT%H:%M:%S",   # ISO-T
            "%Y-%m-%dT%H:%M:%SZ",  # ISO-T-Z
        )
        for fmt in _FORMATS:
            try:
                return datetime.strptime(cleaned, fmt).timestamp()
            except (ValueError, TypeError):
                continue
        return 0.0

    def _count_tokens(self, text: str, model: str = "gpt-4") -> int:
        """Count tokens."""
        if not text:
            return 0
        if TIKTOKEN_AVAILABLE:
            try:
                encoding = tiktoken.encoding_for_model(model) if "gpt" in model else tiktoken.get_encoding("cl100k_base")
                return len(encoding.encode(text))
            except Exception:
                pass
        return int(len(text) / 4)
    
    def _get_rerank_description(self) -> str:
        """Get rerank description."""
        rerank_descriptions = {
            "baai": "BAAI重排序 (bge-reranker-v2-m3)",
            "qwen": "Qwen重排序 (Qwen3-Reranker-0.6B)",
            "jina": "Jina重排序 (jina-reranker-v3)",
            "qwen-sili": "Qwen云端重排序 (Siliconflow)",
            "qwen-dashscope": "Qwen云端重排序 (DashScope)",
            "gte-dashscope": "GTE云端重排序 (DashScope)",
            "none": "无重排序"
        }
        return rerank_descriptions.get(self.rerank_method, self.rerank_method)
    
    
    
    def _load_test_cases(self) -> List[TripleTestCase]:
        """Load test cases."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"数据集文件不存在: {self.dataset_path}")
        
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        start_idx = self.start_qa if self.start_qa is not None else 0
        end_idx = self.end_qa if self.end_qa is not None else len(dataset) - 1
        
        start_idx = max(0, min(start_idx, len(dataset) - 1))
        end_idx = max(start_idx, min(end_idx, len(dataset) - 1))
        
        test_cases = []
        count = 0
        for idx in range(start_idx, end_idx + 1):
            if self.max_tests and count >= self.max_tests:
                break
            
            item = dataset[idx]
            test_case = TripleTestCase(
                qa_index=idx,
                question_id=item.get("question_id", f"q_{idx}"),
                question=item["question"],
                answer=item["answer"],
                question_type=item.get("question_type", ""),
                category=item.get("category", ""),
                query_date=item.get("question_date", item.get("date", item.get("session_date", "Unknown Date")))
            )
            test_cases.append(test_case)
            count += 1
        
        logger.info(f" 加载 {len(test_cases)} 个测试用例 (QA {start_idx} - {end_idx})")
        return test_cases
    
    def _load_semantic_graph(self, graph_dir: Path, qa_index: int, graph_type: str) -> Optional[SemanticGraph]:
        """Load semantic graph."""
        qa_dir = graph_dir / f"qa_{qa_index}"
        if not qa_dir.exists():
            logger.debug(f" {graph_type} 图谱目录不存在: {qa_dir}")
            return None
        
        try:
            semantic_graph = SemanticGraph.load_graph(
                str(qa_dir),
                embedding_model_name="Qwen/Qwen3-Embedding-0.6B"
            )
            return semantic_graph
        except Exception as e:
            logger.error(f" 加载 {graph_type} 图谱失败 (QA {qa_index}): {e}")
            return None
    
    
    @contextmanager
    def _routing_context(self, config: LongMemEvalTowerRoutingConfig):
        """Run routing context."""
        orig = {
            'enable_sentence': self.enable_sentence,
            'enable_episodic': self.enable_episodic,
            'enable_entity': self.enable_entity,
            'sentence_top_k': self.sentence_top_k,
            'episodic_top_k': self.episodic_top_k,
            'entity_top_k': self.entity_top_k,
            'final_top_k': self.final_top_k,
        }
        try:
            self.enable_sentence = config.enable_sentence
            self.enable_episodic = config.enable_episodic
            self.enable_entity = config.enable_entity
            self.sentence_top_k = config.sentence_top_k
            self.episodic_top_k = config.episodic_top_k
            self.entity_top_k = config.entity_top_k
            self.final_top_k = config.final_top_k
            yield config
        finally:
            self.enable_sentence = orig['enable_sentence']
            self.enable_episodic = orig['enable_episodic']
            self.enable_entity = orig['enable_entity']
            self.sentence_top_k = orig['sentence_top_k']
            self.episodic_top_k = orig['episodic_top_k']
            self.entity_top_k = orig['entity_top_k']
            self.final_top_k = orig['final_top_k']
    
    def _retrieve_sentence_context(self,
                                   semantic_graph: SemanticGraph,
                                   question: str) -> Tuple[List[Tuple[MemoryUnit, float]], float]:
        """Run retrieve sentence context."""
        start_time = time.time()
        
        
        if self.sentence_top_k <= 0:
            return [], 0.0
        
        try:
            multi_retriever = semantic_graph.get_multi_retriever()
            if multi_retriever is None:
                return [], time.time() - start_time
            
            
            multi_retriever.build_all_indexes(
                methods_to_build=[
                    RetrievalMethod.BM25,
                    RetrievalMethod.SPLADE,
                    RetrievalMethod.COSINE_SIMILARITY
                ],
                force_rebuild=False
            )
            
            
            rerank_method = self.rerank_method if self.rerank_method != "none" else None
            
            results = multi_retriever.smart_search(
                query=question,
                methods=["bm25", "splade", "cosine_similarity"],
                fusion_method="rrf",
                rerank_method=rerank_method,
                top_k=self.sentence_top_k,
                return_detailed=False
            )
            
            return results, time.time() - start_time
            
        except Exception as e:
            logger.error(f"Sentence 检索失败: {e}")
            return [], time.time() - start_time
    
    def _retrieve_episodic_context(self,
                                   semantic_graph: SemanticGraph,
                                   question: str) -> Tuple[List[Tuple[MemoryUnit, float]], float, Dict[str, int]]:
        """Run retrieve episodic context."""
        start_time = time.time()
        facts_by_category = defaultdict(int)
        
        
        if self.episodic_top_k <= 0:
            return [], 0.0, {}
        
        try:
            multi_retriever = semantic_graph.get_multi_retriever()
            rerank_config = None if self.rerank_method == "none" else self.rerank_method
            
            if multi_retriever:
                methods = [
                    RetrievalMethod.BM25,
                    RetrievalMethod.SPLADE,
                    RetrievalMethod.COSINE_SIMILARITY
                ]
                
                results = multi_retriever.smart_search(
                    query=question,
                    methods=methods,
                    top_k=self.episodic_top_k,
                    fusion_method="rrf",
                    rerank_method=rerank_config,
                    return_detailed=False
                )
            else:
                results = semantic_graph.search_similarity_in_graph(
                    query_text=question, top_k=self.episodic_top_k, return_score=True
                )
            
            final_results = []
            if results:
                if isinstance(results[0], tuple):
                    final_results = results[:self.episodic_top_k]
                elif isinstance(results[0], MemoryUnit):
                    final_results = [(unit, 1.0) for unit in results[:self.episodic_top_k]]
            
            for unit, score in final_results:
                raw_data = unit.raw_data if hasattr(unit, 'raw_data') else {}
                category = raw_data.get('category', raw_data.get('node_type', 'UNKNOWN')).upper()
                facts_by_category[category] += 1
            
            return final_results, time.time() - start_time, dict(facts_by_category)
            
        except Exception as e:
            logger.error(f"Episodic 检索失败: {e}")
            return [], time.time() - start_time, {}
    
    def _retrieve_entity_context(self,
                                 semantic_graph: SemanticGraph,
                                 question: str) -> Tuple[List[Tuple[MemoryUnit, float]], float, Dict[str, int]]:
        """Run retrieve entity context."""
        start_time = time.time()
        entity_types_found = defaultdict(int)
        
        
        if self.entity_top_k <= 0:
            return [], 0.0, {}
        
        try:
            multi_retriever = semantic_graph.get_multi_retriever()
            methods = [RetrievalMethod.BM25, RetrievalMethod.SPLADE, RetrievalMethod.COSINE_SIMILARITY]
            rerank_config = None if self.rerank_method == "none" else self.rerank_method
            
            if multi_retriever:
                results = multi_retriever.smart_search(
                    query=question,
                    methods=methods,
                    top_k=self.entity_top_k,
                    fusion_method="rrf",
                    rerank_method=rerank_config,
                    return_detailed=False
                )
            else:
                results = semantic_graph.search_similarity_in_graph(
                    query_text=question, top_k=self.entity_top_k, return_score=True
                )
            
            
            if len(results) > self.entity_top_k:
                results = results[:self.entity_top_k]
            
            for unit, score in results:
                raw = unit.raw_data
                entity_type = raw.get("entity_category") or raw.get("entity_type") or "Unknown"
                entity_types_found[entity_type] += 1
            
            return results, time.time() - start_time, dict(entity_types_found)
            
        except Exception as e:
            logger.error(f"Entity 检索失败: {e}")
            return [], time.time() - start_time, {}
    
    
    def _format_sentence_context(self, results: List[Tuple[MemoryUnit, float]]) -> str:
        """Format sentence context."""
        if not results:
            return ""
        
        context_parts = []
        for i, (unit, score) in enumerate(results, 1):
            content = unit.raw_data.get('text_content', '')
            role = unit.metadata.get('role', 'unknown')
            session_date = unit.metadata.get('session_date', 'unknown')
            
            context_parts.append(
                f"[Message {i}] (Date: {session_date}, Speaker: {role})\n"
                f"Content: {content}"
            )
        
        return "\n\n".join(context_parts)
    
    def _format_episodic_context(self, results: List[Tuple[MemoryUnit, float]]) -> str:
        """Format episodic context."""
        if not results:
            return ""
        
        context_parts = []
        for i, (unit, score) in enumerate(results, 1):
            raw_data = unit.raw_data if hasattr(unit, 'raw_data') else {}
            
            content = raw_data.get('content', raw_data.get('text_content', str(raw_data)))
            category = raw_data.get('category', 'EVENT').upper()
            event_date = (
                raw_data.get('event_date') or 
                raw_data.get('temporal_val') or 
                raw_data.get('time') or 
                "Unknown Date"
            )
            
            is_stable = raw_data.get('is_stable')
            if is_stable is None:
                is_stable = category in self.STABLE_CATEGORIES
            stability_str = "Stable" if is_stable else "Dynamic"
            
            formatted_fact = (
                f"[Fact {i}] Type: {category} | Time: {event_date} | Stability: {stability_str}\n"
                f"Content: {content}"
            )
            context_parts.append(formatted_fact)
        
        return "\n\n".join(context_parts)
    
    def _format_entity_context(self, results: List[Tuple[MemoryUnit, float]]) -> str:
        """Format entity context."""
        if not results:
            return ""
        
        context_parts = []
        for i, (unit, score) in enumerate(results, 1):
            raw = unit.raw_data if hasattr(unit, 'raw_data') else {}
            
            main_content = raw.get("text_content")
            if not main_content:
                name = raw.get("entity_canonical") or raw.get("entity_text") or unit.uid
                etype = raw.get("entity_category") or "Unknown"
                desc = raw.get("content") or "No description"
                main_content = f"Entity: {name} (Type: {etype}) | Context: {desc}"
            
            session_date = raw.get("session_date") or raw.get("date") or raw.get("created_at")
            date_str = f" [Date: {session_date}]" if session_date else ""
            
            context_parts.append(f"[{i}] {main_content}{date_str}")
        
        return "\n\n".join(context_parts)
    
    def _build_fused_context(self,
                             sentence_context: str,
                             episodic_context: str,
                             entity_context: str) -> str:
        """Build fused context."""
        sections = []
        
        if self.enable_sentence and sentence_context:
            sections.append(
                f"<conversation_history>\n{sentence_context}\n</conversation_history>"
            )
        
        if self.enable_episodic and episodic_context:
            sections.append(
                f"<episodic_facts>\n{episodic_context}\n</episodic_facts>"
            )
        
        if self.enable_entity and entity_context:
            sections.append(
                f"<entity_knowledge>\n{entity_context}\n</entity_knowledge>"
            )
        
        if not sections:
            return "[No context available]"
        
        return "\n\n".join(sections)
    
    
    def _generate_answer_from_fused_context(self,
                                            question: str,
                                            fused_context: str,
                                            question_type: str = "",
                                            query_date: str = "Unknown Date") -> Tuple[str, str, TokenStats]:
        """Generate answer from fused context."""
        token_stats = TokenStats()
        
        if not self.llm_client:
            return "LLM client not configured", "", token_stats
        
        if fused_context == "[No context available]":
            return "No relevant context found.", "I don't have enough information to answer this question.", token_stats
        
        token_stats.context_tokens = self._count_tokens(fused_context)
        
        prompt = f"""You are an expert memory-augmented assistant answering questions based on retrieved multi-level context.

        # CURRENT REFERENCE TIME
        The current time for this question is: **{query_date}**
        *** CRITICAL INSTRUCTION ***
        - Treat "{query_date}" as "TODAY" or "NOW".
        - All relative time references ("yesterday", "last week", "3 days ago") MUST be calculated relative to this date.
        - Do NOT use the actual real-world date.

        # RETRIEVED CONTEXT
        You have access to three levels of memory information (organized in XML tags):
        1. **Conversation Memories** (<conversation_history>): Raw message excerpts from past conversations (Highest Priority).
        2. **Episodic Facts** (<episodic_facts>): Structured facts with temporal information.
        3. **Entity Information** (<entity_knowledge>): Entities and their relationships.

        {fused_context}

        ─────────────────────────────────────────────────────────────────────────────────

        # QUESTION
        {question}

        # REASONING INSTRUCTIONS
        1. **Cross-reference** information across all three levels when available.
        2. **Prioritize** temporal information when the question involves time ("when", "before", "after").
        3. **Use entity information** (<entity_knowledge>) to understand relationships and attributes.
        4. **Use conversation memories** (<conversation_history>) for direct quotes and specific details.
        5. **Use episodic facts** (<episodic_facts>) for structured event information.
        6. **Conflict Resolution**: If information conflicts across sources, note the discrepancy and prefer more specific/recent data.

        # IMPORTANT NOTES
        - When interpreting timestamps, use the recorded date as the reference point.
        - For "yesterday", "last week" etc., calculate based on the conversation date.
        - If the answer cannot be determined from the context, clearly state so.

        # OUTPUT FORMAT
        Please respond in strict JSON format with two fields:
        {{
            "reasoning": "Your detailed step-by-step reasoning process. Show how you analyzed information from different memory levels. Cite specific facts, messages, or entities that support your answer.",
            "final_answer": "Your direct, concise answer to the question. Be specific and avoid vague references."
        }}
        """
        
        token_stats.prompt_tokens = self._count_tokens(prompt)
        
        try:
            response = self.llm_client.generate_answer(
                prompt=prompt,
                temperature=0.0,
                max_tokens=self.generation_max_tokens,
                json_format=True
            )
            
            token_stats.completion_tokens = self._count_tokens(response)
            token_stats.total_tokens = token_stats.prompt_tokens + token_stats.completion_tokens
            
            clean_response = response.strip()
            if clean_response.startswith("```json"):
                clean_response = clean_response.replace("```json", "").replace("```", "").strip()
            
            try:
                parsed = json.loads(clean_response)
                reasoning = parsed.get("reasoning", "No reasoning provided")
                answer = parsed.get("final_answer", "Unable to answer")
                if isinstance(reasoning, list):
                    reasoning = ", ".join(str(x) for x in reasoning)
                elif not isinstance(reasoning, str):
                    reasoning = str(reasoning)
                if isinstance(answer, list):
                    answer = ", ".join(str(x) for x in answer)
                elif not isinstance(answer, str):
                    answer = str(answer)
                return reasoning, answer, token_stats
            except json.JSONDecodeError:
                import re
                reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]+)"', response, re.DOTALL)
                answer_match = re.search(r'"final_answer"\s*:\s*"([^"]+)"', response, re.DOTALL)
                
                if reasoning_match and answer_match:
                    return reasoning_match.group(1), answer_match.group(1), token_stats
                else:
                    return response[:300], response[-100:] if len(response) > 100 else response, token_stats
                    
        except Exception as e:
            logger.warning(f"LLM 生成失败: {e}")
            return f"Generation error: {str(e)}", "Unable to generate answer", token_stats
    
    
    def _evaluate_result(self,
                     test_case: TripleTestCase,
                     generated_answer: str,
                     reasoning: str) -> Dict[str, Any]:
        """Run evaluate result."""
        if not EVALUATION_AVAILABLE:
            return {"error": "Evaluation module not available"}
        
        if not self.llm_evaluate_client:
            return {"error": "Evaluation LLM client not configured"}
        
        try:
            scores = calculate_comprehensive_scores(
                question=test_case.question,
                response=generated_answer,
                gold_answer=test_case.answer,
                context=reasoning,
                question_type=test_case.question_type,
                llm_client=self.llm_evaluate_client
            )
            return scores
        except Exception as e:
            logger.error(f"评估失败: {e}")
            return {"error": str(e)}
        
    def _sanitize_for_json(self, obj: Any) -> Any:
        """Run sanitize for JSON."""
        try:
            import numpy as np
        except ImportError:
            np = None

        if np:
            if isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return self._sanitize_for_json(obj.tolist())
        
        if isinstance(obj, dict):
            return {k: self._sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._sanitize_for_json(i) for i in obj]
        elif isinstance(obj, tuple):
            return tuple(self._sanitize_for_json(i) for i in obj)
            
        return obj
    
    
    
    def _second_stage_rerank(self,
                             question: str,
                             all_results: List[Tuple[MemoryUnit, float]],
                             top_k: int) -> Tuple[List[Tuple[MemoryUnit, float]], float]:
        """Run second stage rerank."""
        if not all_results:
            return [], 0.0
        
        start_time = time.time()
        
        try:
            
            reranker_manager = RerankerManager()
            reranker = reranker_manager.get_reranker(self.second_stage_rerank_method)
            
            if reranker is None:
                logger.warning(f"无法获取重排序器: {self.second_stage_rerank_method}，跳过二次重排")
                return all_results[:top_k], time.time() - start_time
            
            documents = []
            for unit, _ in all_results:
                raw = unit.raw_data if hasattr(unit, 'raw_data') and unit.raw_data else {}
                text = raw.get('text_content', '')
                if not text:
                    text = raw.get('content', '') or raw.get('description', '') or str(raw)
                documents.append(text[:1500])
            
            
            scores = reranker.rerank(query=question, documents=documents)
            
            
            reranked = list(zip(all_results, scores))
            reranked.sort(key=lambda x: x[1], reverse=True)
            
            
            final_results = [(item[0][0], item[1]) for item in reranked[:top_k]]
            
            rerank_time = time.time() - start_time
            logger.debug(f"二次重排序完成: {len(all_results)} -> {len(final_results)}, 耗时 {rerank_time:.2f}s")
            
            return final_results, rerank_time
            
        except Exception as e:
            logger.error(f"二次重排序失败: {e}")
            return all_results[:top_k], time.time() - start_time
    
    
    def _extract_content_for_report(self, unit: MemoryUnit, score: float, source_type: str) -> Dict[str, Any]:
        """Extract content for report."""
        raw = unit.raw_data if hasattr(unit, 'raw_data') else {}
        metadata = unit.metadata if hasattr(unit, 'metadata') else {}
        
        content_info = {
            "uid": unit.uid,
            "score": round(float(score), 4),
            "source_type": source_type,
            "text_content": raw.get("text_content", "")[:500],
        }
        
        if source_type == "sentence":
            content_info.update({
                "role": metadata.get("role", "unknown"),
                "session_date": metadata.get("session_date", "unknown"),
            })
        elif source_type == "episodic":
            content_info.update({
                "category": raw.get("category", "UNKNOWN"),
                "event_date": raw.get("event_date") or raw.get("temporal_val") or raw.get("time", "unknown"),
                "is_stable": raw.get("is_stable", None),
            })
        elif source_type == "entity":
            content_info.update({
                "entity_name": raw.get("entity_canonical") or raw.get("entity_text", ""),
                "entity_type": raw.get("entity_category") or raw.get("entity_type", "Unknown"),
                "description": raw.get("content") or raw.get("description", "")[:300],
                "session_date": raw.get("session_date") or raw.get("date", ""),
            })
        
        return content_info
    
    def _test_single_qa(self, test_case: TripleTestCase, test_index: int) -> TripleRetrievalResult:
        """Run test single qa."""
        
        
        routing_info = ""
        if self.tower_router:
            _rconfig = self.tower_router.route(
                question=test_case.question,
                category=test_case.question_type,
                enable_guidance=False,
            )
            routing_info = (f"routed: {_rconfig.active_towers} "
                           f"(strategy={self.tower_router.strategy}, "
                           f"cat={test_case.question_type})")
            logger.info(f" 路由决策: {test_case.question_type} → {_rconfig.active_towers} "
                       f"(topk: S={_rconfig.sentence_top_k}, "
                       f"Ep={_rconfig.episodic_top_k}, "
                       f"Ent={_rconfig.entity_top_k}, "
                       f"final_k={_rconfig.final_top_k})")
            with self._routing_context(_rconfig):
                return self._execute_single_qa(test_case, test_index, routing_info)
        else:
            return self._execute_single_qa(test_case, test_index, routing_info)
    
    def _execute_single_qa(self, test_case: TripleTestCase, test_index: int,
                           routing_info: str = "") -> TripleRetrievalResult:
        """Execute single qa."""
        count_stats = test_index >= 0
        result = TripleRetrievalResult(
            qa_index=test_case.qa_index,
            question=test_case.question,
            ground_truth=test_case.answer,
            generated_answer="",
            reasoning=""
        )
        
        retrieval_details = TripleRetrievalDetails(
            sentence_enabled=self.enable_sentence,
            episodic_enabled=self.enable_episodic,
            entity_enabled=self.enable_entity,
            rerank_method=self.rerank_method,
            fusion_method=self.fusion_method,
            second_stage_rerank_enabled=self.enable_second_stage_rerank,
            second_stage_rerank_method=self.second_stage_rerank_method if self.enable_second_stage_rerank else "none"
        )
        
        retrieved_contents = {
            "sentence": [],
            "episodic": [],
            "entity": [],
            "reranked": []
        }
        
        try:
            graph_loading_start = time.time()
            
            sentence_graph = None
            episodic_graph = None
            entity_graph = None
            
            if self.enable_sentence:
                sentence_graph = self._load_semantic_graph(
                    self.sentence_graph_dir, test_case.qa_index, "sentence"
                )
                if sentence_graph:
                    multi_retriever = sentence_graph.get_multi_retriever()
                    if multi_retriever:
                        multi_retriever.build_all_indexes(
                            methods_to_build=[RetrievalMethod.BM25, RetrievalMethod.SPLADE, RetrievalMethod.COSINE_SIMILARITY],
                            force_rebuild=False
                        )
            
            if self.enable_episodic:
                episodic_graph = self._load_semantic_graph(
                    self.episodic_graph_dir, test_case.qa_index, "episodic"
                )
                if episodic_graph:
                    multi_retriever = episodic_graph.get_multi_retriever()
                    if multi_retriever:
                        multi_retriever.build_all_indexes(
                            methods_to_build=[RetrievalMethod.BM25, RetrievalMethod.SPLADE, RetrievalMethod.COSINE_SIMILARITY],
                            force_rebuild=False
                        )
            
            if self.enable_entity:
                entity_graph = self._load_semantic_graph(
                    self.entity_graph_dir, test_case.qa_index, "entity"
                )
                if entity_graph:
                    multi_retriever = entity_graph.get_multi_retriever()
                    if multi_retriever:
                        multi_retriever.build_all_indexes(
                            methods_to_build=[RetrievalMethod.BM25, RetrievalMethod.SPLADE, RetrievalMethod.COSINE_SIMILARITY],
                            force_rebuild=False
                        )
            
            retrieval_details.graph_loading_time = time.time() - graph_loading_start
            
            total_start_time = time.time()
            
            all_retrieved_units = []  # [(MemoryUnit, score, source_type), ...]
            
            sentence_context = ""
            sentence_results = []
            if self.enable_sentence and sentence_graph:
                sentence_results, sentence_time = self._retrieve_sentence_context(
                    sentence_graph, test_case.question
                )
                retrieval_details.sentence_retrieved_count = len(sentence_results)
                retrieval_details.sentence_retrieval_time = sentence_time
                
                for unit, score in sentence_results:
                    all_retrieved_units.append((unit, score, "sentence"))
                    retrieved_contents["sentence"].append(
                        self._extract_content_for_report(unit, score, "sentence")
                    )
                
                if not self.enable_second_stage_rerank:
                    sentence_context = self._format_sentence_context(sentence_results)
                if count_stats:
                    self.stats['sentence_retrieval_count'] += len(sentence_results)
            
            episodic_context = ""
            episodic_results = []
            if self.enable_episodic and episodic_graph:
                episodic_results, episodic_time, facts_by_category = self._retrieve_episodic_context(
                    episodic_graph, test_case.question
                )
                retrieval_details.episodic_retrieved_count = len(episodic_results)
                retrieval_details.episodic_retrieval_time = episodic_time
                retrieval_details.episodic_facts_by_category = facts_by_category
                
                for unit, score in episodic_results:
                    all_retrieved_units.append((unit, score, "episodic"))
                    retrieved_contents["episodic"].append(
                        self._extract_content_for_report(unit, score, "episodic")
                    )
                
                if not self.enable_second_stage_rerank:
                    episodic_context = self._format_episodic_context(episodic_results)
                if count_stats:
                    self.stats['episodic_retrieval_count'] += len(episodic_results)
            
            entity_context = ""
            entity_results = []
            if self.enable_entity and entity_graph:
                entity_results, entity_time, entity_types = self._retrieve_entity_context(
                    entity_graph, test_case.question
                )
                retrieval_details.entity_retrieved_count = len(entity_results)
                retrieval_details.entity_retrieval_time = entity_time
                retrieval_details.entity_types_found = entity_types
                
                for unit, score in entity_results:
                    all_retrieved_units.append((unit, score, "entity"))
                    retrieved_contents["entity"].append(
                        self._extract_content_for_report(unit, score, "entity")
                    )
                
                if not self.enable_second_stage_rerank:
                    entity_context = self._format_entity_context(entity_results)
                if count_stats:
                    self.stats['entity_retrieval_count'] += len(entity_results)
            
            
            cascade_pruned = False
            if self.enable_second_stage_rerank and all_retrieved_units:
                retrieval_details.first_stage_total_count = len(all_retrieved_units)
                
                
                uid_to_source = {}
                for unit, score, source in all_retrieved_units:
                    uid_to_source[unit.uid] = source
                
                units_for_rerank = [(unit, score) for unit, score, _ in all_retrieved_units]
                
                
                reranked_all, rerank_time = self._second_stage_rerank(
                    test_case.question,
                    units_for_rerank,
                    len(units_for_rerank)
                )
                
                retrieval_details.second_stage_rerank_time = rerank_time
                
                if self.enable_cascade_pruner and self.cascade_pruner and self.cascade_adapter:
                    try:
                        
                        ce_scores_map = {}
                        for unit, score in reranked_all:
                            ce_scores_map[unit.uid] = score
                        
                        ce_idx = 0
                        for key in ("sentence", "episodic", "entity"):
                            for item in retrieved_contents[key]:
                                uid = item.get("uid", "")
                                if uid in ce_scores_map:
                                    item["ce_score"] = ce_scores_map[uid]
                                else:
                                    item["ce_score"] = item.get("score", 0.0)
                                ce_idx += 1
                        
                        
                        
                        _sentence_cands = []
                        _other_cands = []
                        for unit, score in reranked_all:
                            _src = uid_to_source.get(unit.uid, "unknown")
                            if _src == "sentence":
                                _sentence_cands.append(unit.uid)
                            else:
                                _other_cands.append(unit.uid)
                        
                        _min_sent = (self.final_top_k + 1) // 2
                        _topk_uids = []
                        _topk_uids.extend(_sentence_cands[:_min_sent])
                        _remaining = self.final_top_k - len(_topk_uids)
                        if _remaining > 0:
                            _topk_uids.extend(_other_cands[:_remaining])
                        if len(_topk_uids) < self.final_top_k:
                            _extra = _sentence_cands[_min_sent:]
                            _topk_uids.extend(_extra[:self.final_top_k - len(_topk_uids)])
                        _selected_uids = set(_topk_uids)
                        
                        _filtered_contents = {
                            "sentence": [it for it in retrieved_contents["sentence"] if it.get("uid") in _selected_uids],
                            "episodic": [it for it in retrieved_contents["episodic"] if it.get("uid") in _selected_uids],
                            "entity":   [it for it in retrieved_contents["entity"]   if it.get("uid") in _selected_uids],
                        }
                        _total_before = sum(len(retrieved_contents[k]) for k in ("sentence", "episodic", "entity"))
                        _total_after = sum(len(v) for v in _filtered_contents.values())
                        logger.debug(
                            f" 级联输入过滤: {_total_before} → {_total_after} 候选 "
                            f"(final_top_k={self.final_top_k})"
                        )
                        
                        candidates = self.cascade_adapter.adapt(_filtered_contents)
                        
                        if candidates:
                            query_ts = self._parse_date_to_timestamp(test_case.query_date)
                            logger.debug(
                                f"[Cascade] query_date='{test_case.query_date}' → "
                                f"query_timestamp={query_ts:.0f}"
                            )
                            cascade_result: CascadePruneResult = self.cascade_pruner.prune(
                                candidates, query_timestamp=query_ts,
                                query_category=test_case.question_type,
                            )
                            
                            
                            _sent_parts, _epi_parts, _ent_parts = [], [], []
                            _sent_idx = _epi_idx = _ent_idx = 0
                            for pc in cascade_result.selected_chunks:
                                chunk = pc.chunk
                                meta = getattr(chunk, 'raw_metadata', {}) or {}
                                ts = getattr(chunk, 'tower_source', None)
                                
                                if ts == TowerSource.HIERARCHICAL:
                                    _sent_idx += 1
                                    content = meta.get('text_content', chunk.text)
                                    role = meta.get('role', 'unknown')
                                    session_date = meta.get('session_date', 'unknown')
                                    _sent_parts.append(
                                        f"[Message {_sent_idx}] (Date: {session_date}, Speaker: {role})\n"
                                        f"Content: {content}"
                                    )
                                elif ts == TowerSource.EPISODIC:
                                    _epi_idx += 1
                                    content = meta.get('content', meta.get('text_content', chunk.text))
                                    category = meta.get('category', 'EVENT').upper()
                                    event_date = (
                                        meta.get('event_date') or
                                        meta.get('temporal_val') or
                                        meta.get('time') or
                                        "Unknown Date"
                                    )
                                    is_stable = meta.get('is_stable')
                                    if is_stable is None:
                                        is_stable = category in self.STABLE_CATEGORIES
                                    stability_str = "Stable" if is_stable else "Dynamic"
                                    _epi_parts.append(
                                        f"[Fact {_epi_idx}] Type: {category} | Time: {event_date} | Stability: {stability_str}\n"
                                        f"Content: {content}"
                                    )
                                elif ts == TowerSource.KG:
                                    _ent_idx += 1
                                    main_content = meta.get('text_content', chunk.text)
                                    if not main_content:
                                        name = meta.get('entity_name', chunk.chunk_id)
                                        etype = meta.get('entity_type', 'Unknown')
                                        desc = meta.get('description', 'No description')
                                        main_content = f"Entity: {name} (Type: {etype}) | Context: {desc}"
                                    session_date = meta.get('session_date') or meta.get('date')
                                    date_str = f" [Date: {session_date}]" if session_date else ""
                                    _ent_parts.append(f"[{_ent_idx}] {main_content}{date_str}")
                            
                            
                            _sections = []
                            if _sent_parts:
                                _sections.append(
                                    f"<conversation_history>\n" +
                                    "\n\n".join(_sent_parts) +
                                    f"\n</conversation_history>"
                                )
                            if _epi_parts:
                                _sections.append(
                                    f"<episodic_facts>\n" +
                                    "\n\n".join(_epi_parts) +
                                    f"\n</episodic_facts>"
                                )
                            if _ent_parts:
                                _sections.append(
                                    f"<entity_knowledge>\n" +
                                    "\n\n".join(_ent_parts) +
                                    f"\n</entity_knowledge>"
                                )
                            fused_context = "\n\n".join(_sections) if _sections else "[No context available]"
                            cascade_pruned = True
                            
                            
                            selected_ids = {pc.chunk.chunk_id for pc in cascade_result.selected_chunks}
                            for key in ("sentence", "episodic", "entity"):
                                for item in retrieved_contents[key]:
                                    if item.get("uid") in selected_ids:
                                        retrieved_contents["reranked"].append(item)
                            
                            retrieval_details.cascade_pruner_enabled = True
                            retrieval_details.cascade_prune_mode = cascade_result.mode_used.value
                            retrieval_details.cascade_tokens_used = cascade_result.total_tokens_used
                            retrieval_details.cascade_stage1_input = cascade_result.stage1_input_count
                            retrieval_details.cascade_stage1_output = cascade_result.stage1_output_count
                            retrieval_details.cascade_stage2_conflicts = cascade_result.stage2_conflicts_found
                            retrieval_details.cascade_stage2_dropped = cascade_result.stage2_dropped_count
                            retrieval_details.cascade_stage2_output = cascade_result.stage2_output_count
                            retrieval_details.cascade_stage3_mmr_iterations = cascade_result.stage3_mmr_iterations
                            retrieval_details.cascade_stage3_diversity_penalties = cascade_result.stage3_diversity_penalties
                            retrieval_details.final_selected_count = len(cascade_result.selected_chunks)
                            
                            logger.debug(
                                f" 级联量化: {len(all_retrieved_units)} → "
                                f"S1:{cascade_result.stage1_output_count} → "
                                f"S2:{cascade_result.stage2_output_count} "
                                f"(冲突={cascade_result.stage2_conflicts_found}, "
                                f"丢弃={cascade_result.stage2_dropped_count}) → "
                                f"S3:{len(cascade_result.selected_chunks)} "
                                f"(MMR轮={cascade_result.stage3_mmr_iterations}) | "
                                f"tokens={cascade_result.total_tokens_used}/{self.cascade_max_context_tokens}"
                            )
                    
                    except Exception as e:
                        logger.error(f" 级联量化失败，回退到分桶保底: {e}")
                        logger.debug(traceback.format_exc())
                        cascade_pruned = False
                
                if not cascade_pruned:
                    sentence_candidates = []
                    other_candidates = []
                    
                    for unit, score in reranked_all:
                        source = uid_to_source.get(unit.uid, "unknown")
                        if source == "sentence":
                            sentence_candidates.append((unit, score, source))
                        else:
                            other_candidates.append((unit, score, source))
                    
                    
                    MIN_SENTENCE = (self.final_top_k + 1) // 2
                    
                    final_selected = []
                    sentence_taken = sentence_candidates[:MIN_SENTENCE]
                    final_selected.extend(sentence_taken)
                    
                    remaining_slots = self.final_top_k - len(final_selected)
                    if remaining_slots > 0:
                        final_selected.extend(other_candidates[:remaining_slots])
                    
                    
                    if len(final_selected) < self.final_top_k:
                        extra_sentence = sentence_candidates[MIN_SENTENCE:]
                        slots_left = self.final_top_k - len(final_selected)
                        final_selected.extend(extra_sentence[:slots_left])
                    
                    final_selected.sort(key=lambda x: x[1], reverse=True)
                    
                    retrieval_details.final_selected_count = len(final_selected)
                    
                    sentence_reranked = []
                    episodic_reranked = []
                    entity_reranked = []
                    
                    for unit, score, source in final_selected:
                        retrieved_contents["reranked"].append(
                            self._extract_content_for_report(unit, score, source)
                        )
                        if source == "sentence":
                            sentence_reranked.append((unit, score))
                        elif source == "episodic":
                            episodic_reranked.append((unit, score))
                        elif source == "entity":
                            entity_reranked.append((unit, score))
                    
                    sentence_context = self._format_sentence_context(sentence_reranked)
                    episodic_context = self._format_episodic_context(episodic_reranked)
                    entity_context = self._format_entity_context(entity_reranked)
                    
                    logger.debug(f"二次重排+分桶保底: {len(all_retrieved_units)} -> {len(final_selected)} "
                               f"(Sentence: {len(sentence_reranked)} [min={MIN_SENTENCE}], "
                               f"Episodic: {len(episodic_reranked)}, Entity: {len(entity_reranked)})")
            
            
            if not cascade_pruned:
                fused_context = self._build_fused_context(
                    sentence_context, episodic_context, entity_context
                )
            
            retrieval_details.total_retrieval_time = time.time() - total_start_time
            
            if not all_retrieved_units:
                logger.warning(f"QA {test_case.qa_index}: 所有检索通道均返回空结果，跳过 LLM 生成")
                reasoning = "All retrieval channels returned empty results."
                answer = "No relevant information found."
                token_stats = TokenStats()
            else:
                reasoning, answer, token_stats = self._generate_answer_from_fused_context(
                    test_case.question, fused_context, test_case.question_type,
                    query_date=test_case.query_date
                )
            
            if cascade_pruned:
                
                token_stats.sentence_context_tokens = self._count_tokens("\n\n".join(_sent_parts)) if _sent_parts else 0
                token_stats.episodic_context_tokens = self._count_tokens("\n\n".join(_epi_parts)) if _epi_parts else 0
                token_stats.entity_context_tokens = self._count_tokens("\n\n".join(_ent_parts)) if _ent_parts else 0
            else:
                token_stats.sentence_context_tokens = self._count_tokens(sentence_context)
                token_stats.episodic_context_tokens = self._count_tokens(episodic_context)
                token_stats.entity_context_tokens = self._count_tokens(entity_context)
            
            result.generated_answer = answer
            result.reasoning = reasoning
            result.token_stats = token_stats
            result.retrieval_details = retrieval_details
            result.retrieved_contents = retrieved_contents  
            result.routing_info = routing_info  
            
            if self.llm_evaluate_client:
                try:
                    scores = self._evaluate_result(test_case, answer, reasoning)
                    result.scores = scores
                except Exception as eval_e:
                    logger.warning(f"QA {test_case.qa_index} 评估异常（不影响生成结果）: {eval_e}")
                    result.scores = {"evaluation_success": False, "error": str(eval_e)}
            
            result.success = True
            if count_stats:
                self.stats['successful_tests'] += 1
            
        except Exception as e:
            result.error_message = str(e)
            result.success = False
            if count_stats:
                self.stats['failed_tests'] += 1
            logger.error(f"测试 QA {test_case.qa_index} 失败: {e}")
            logger.debug(traceback.format_exc())
        
        if count_stats:
            self.stats['total_tests'] += 1
        
        return result
    
    
    
    # def _save_single_report(self, result: TripleRetrievalResult, index: int):
    
    #     report_dir = self.output_dir / "individual_reports"
    #     report_dir.mkdir(parents=True, exist_ok=True)
        
    #     report_file = report_dir / f"qa_{result.qa_index}_report.json"
        
    #     report_data = {
    #         "qa_index": result.qa_index,
    #         "question": result.question,
    #         "ground_truth": result.ground_truth,
    #         "generated_answer": result.generated_answer,
    #         "reasoning": result.reasoning,
    #         "scores": result.scores,
    #         "token_stats": asdict(result.token_stats),
    #         "retrieval_details": asdict(result.retrieval_details),
    #         "retrieved_contents": result.retrieved_contents,
    #         "success": result.success,
    #         "error_message": result.error_message
    #     }
        
    #     with open(report_file, 'w', encoding='utf-8') as f:
    #         json.dump(report_data, f, ensure_ascii=False, indent=2)
    
    def _save_single_report(self, result: TripleRetrievalResult, index: int):
        """Save single report."""
        report_dir = self.output_dir / "individual_reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        report_file = report_dir / f"qa_{result.qa_index}_report.json"
        
        report_data = {
            "qa_index": result.qa_index,
            "question": result.question,
            "ground_truth": result.ground_truth,
            "generated_answer": result.generated_answer,
            "reasoning": result.reasoning,
            "scores": result.scores,
            "token_stats": asdict(result.token_stats),
            "retrieval_details": asdict(result.retrieval_details),
            "retrieved_contents": result.retrieved_contents,
            "routing_info": result.routing_info,
            "success": result.success,
            "error_message": result.error_message
        }
        
        safe_report_data = self._sanitize_for_json(report_data)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(safe_report_data, f, ensure_ascii=False, indent=2)
    
    def _generate_summary(self, results: List[TripleRetrievalResult], total_time: float) -> Dict[str, Any]:
        """Generate summary."""
        successful_results = [r for r in results if r.success]
        
        avg_scores = {}
        if successful_results:
            score_keys = set()
            for r in successful_results:
                if r.scores and not r.scores.get("error"):
                    actual_scores = r.scores.get("scores", r.scores)
                    for key, value in actual_scores.items():
                        if isinstance(value, (int, float)):
                            score_keys.add(key)
            
            for key in score_keys:
                values = []
                for r in successful_results:
                    if r.scores and not r.scores.get("error"):
                        actual_scores = r.scores.get("scores", r.scores)
                        if key in actual_scores:
                            value = actual_scores[key]
                            if isinstance(value, (int, float)):
                                values.append(value)
                if values:
                    avg_scores[key] = sum(values) / len(values)
        
        avg_sentence_count = sum(r.retrieval_details.sentence_retrieved_count for r in successful_results) / len(successful_results) if successful_results else 0
        avg_episodic_count = sum(r.retrieval_details.episodic_retrieved_count for r in successful_results) / len(successful_results) if successful_results else 0
        avg_entity_count = sum(r.retrieval_details.entity_retrieved_count for r in successful_results) / len(successful_results) if successful_results else 0
        
        avg_retrieval_time = sum(r.retrieval_details.total_retrieval_time for r in successful_results) / len(successful_results) if successful_results else 0
        avg_graph_loading_time = sum(r.retrieval_details.graph_loading_time for r in successful_results) / len(successful_results) if successful_results else 0
        
        total_prompt_tokens = sum(r.token_stats.prompt_tokens for r in successful_results)
        total_completion_tokens = sum(r.token_stats.completion_tokens for r in successful_results)
        
        summary = {
            "test_info": {
                "timestamp": datetime.now().isoformat(),
                "dataset_size": self.dataset_size,
                "total_time": total_time,
                "total_tests": self.stats['total_tests'],
                "successful_tests": self.stats['successful_tests'],
                "failed_tests": self.stats['failed_tests'],
                "success_rate": self.stats['successful_tests'] / self.stats['total_tests'] if self.stats['total_tests'] > 0 else 0
            },
            "config": {
                "sentence_enabled": self.enable_sentence,
                "sentence_top_k": self.sentence_top_k,
                "episodic_enabled": self.enable_episodic,
                "episodic_top_k": self.episodic_top_k,
                "entity_enabled": self.enable_entity,
                "entity_top_k": self.entity_top_k,
                "rerank_method": self.rerank_method,
                "fusion_method": self.fusion_method
            },
            
            "router_config": {
                "enabled": self.tower_router is not None,
                "model": self.tower_router.model_name if self.tower_router else None,
                "strategy": self.tower_router.strategy if self.tower_router else None,
            },
            "cascade_config": {
                "enabled": self.enable_cascade_pruner,
                "pruner_type": "CascadeConfidencePruner" if self.enable_cascade_pruner else None,
                "prune_mode": self.cascade_prune_mode.value if hasattr(self, 'cascade_prune_mode') and hasattr(self.cascade_prune_mode, 'value') else None,
                "max_context_tokens": getattr(self, 'cascade_max_context_tokens', None),
                "lambda_mmr": getattr(self, 'cascade_lambda_mmr', None),
                "stage1_enabled": getattr(self, 'cascade_enable_stage1', None),
                "stage2_enabled": getattr(self, 'cascade_enable_stage2', None),
                "stage3_mmr_enabled": getattr(self, 'cascade_enable_stage3_mmr', None),
                "adaptive_dataset": getattr(self, 'cascade_adaptive_dataset', None),
            },
            "retrieval_stats": {
                "avg_sentence_retrieved": avg_sentence_count,
                "avg_episodic_retrieved": avg_episodic_count,
                "avg_entity_retrieved": avg_entity_count,
                "total_sentence_retrieved": self.stats['sentence_retrieval_count'],
                "total_episodic_retrieved": self.stats['episodic_retrieval_count'],
                "total_entity_retrieved": self.stats['entity_retrieval_count'],
                "avg_retrieval_time": avg_retrieval_time,
                "avg_graph_loading_time": avg_graph_loading_time
            },
            "scores": avg_scores,
            "token_stats": {
                "total_prompt_tokens": total_prompt_tokens,
                "total_completion_tokens": total_completion_tokens,
                "total_tokens": total_prompt_tokens + total_completion_tokens
            }
        }
        
        if self.enable_cascade_pruner and successful_results:
            n = len(successful_results)
            cascade_active = [r for r in successful_results if r.retrieval_details.cascade_pruner_enabled]
            nc = len(cascade_active) if cascade_active else 1
            summary["cascade_stats"] = {
                "cascade_active_count": len(cascade_active),
                "avg_stage1_input": sum(r.retrieval_details.cascade_stage1_input for r in cascade_active) / nc if cascade_active else 0,
                "avg_stage1_output": sum(r.retrieval_details.cascade_stage1_output for r in cascade_active) / nc if cascade_active else 0,
                "avg_stage2_conflicts": sum(r.retrieval_details.cascade_stage2_conflicts for r in cascade_active) / nc if cascade_active else 0,
                "avg_stage2_dropped": sum(r.retrieval_details.cascade_stage2_dropped for r in cascade_active) / nc if cascade_active else 0,
                "avg_stage2_output": sum(r.retrieval_details.cascade_stage2_output for r in cascade_active) / nc if cascade_active else 0,
                "avg_stage3_mmr_iterations": sum(r.retrieval_details.cascade_stage3_mmr_iterations for r in cascade_active) / nc if cascade_active else 0,
                "avg_stage3_diversity_penalties": sum(r.retrieval_details.cascade_stage3_diversity_penalties for r in cascade_active) / nc if cascade_active else 0,
                "avg_tokens_used": sum(r.retrieval_details.cascade_tokens_used for r in cascade_active) / nc if cascade_active else 0,
            }
        
        return summary
    
    def _save_results(self, results: List[TripleRetrievalResult], summary: Dict[str, Any]):
        """Save results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        safe_summary = self._sanitize_for_json(summary)
        
        
        summary_file = self.output_dir / f"summary_{timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(safe_summary, f, ensure_ascii=False, indent=2)
        
        
        detailed_results = []
        for r in results:
            detailed_results.append({
                "qa_index": r.qa_index,
                "question": r.question,
                "ground_truth": r.ground_truth,
                "generated_answer": r.generated_answer,
                "reasoning": r.reasoning[:500] if r.reasoning else "",
                "scores": r.scores,
                "routing_info": r.routing_info,
                "success": r.success
            })
        
        safe_detailed_results = self._sanitize_for_json(detailed_results)
        
        results_file = self.output_dir / f"results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(safe_detailed_results, f, ensure_ascii=False, indent=2)
        
        self._generate_readable_report(summary, results, timestamp)
        
        logger.info(f" 结果已保存到: {self.output_dir}")
    
    # def _save_results(self, results: List[TripleRetrievalResult], summary: Dict[str, Any]):
    
    #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    
    #     summary_file = self.output_dir / f"summary_{timestamp}.json"
    #     with open(summary_file, 'w', encoding='utf-8') as f:
    #         json.dump(summary, f, ensure_ascii=False, indent=2)
        
    
    #     detailed_results = []
    #     for r in results:
    #         detailed_results.append({
    #             "qa_index": r.qa_index,
    #             "question": r.question,
    #             "ground_truth": r.ground_truth,
    #             "generated_answer": r.generated_answer,
    #             "scores": r.scores,
    #             "success": r.success
    #         })
        
    #     results_file = self.output_dir / f"results_{timestamp}.json"
    #     with open(results_file, 'w', encoding='utf-8') as f:
    #         json.dump(detailed_results, f, ensure_ascii=False, indent=2)
        
    #     self._generate_readable_report(summary, results, timestamp)
        
    
    
    def _generate_readable_report(self, summary: Dict[str, Any], results: List[TripleRetrievalResult], timestamp: str):
        """Generate readable report."""
        report_lines = [
            "=" * 100,
            " LongMemEval 三重检索融合 Benchmark 报告",
            "=" * 100,
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            " 测试概况:",
            f"   总测试数: {summary['test_info']['total_tests']}",
            f"   成功数: {summary['test_info']['successful_tests']}",
            f"   失败数: {summary['test_info']['failed_tests']}",
            f"   成功率: {summary['test_info']['success_rate']*100:.1f}%",
            f"   总耗时: {summary['test_info']['total_time']:.2f}s",
            "",
            " 配置信息:",
            f"   Sentence检索: {'' if summary['config']['sentence_enabled'] else ''} (Top-K={summary['config']['sentence_top_k']})",
            f"   Episodic检索: {'' if summary['config']['episodic_enabled'] else ''} (Top-K={summary['config']['episodic_top_k']})",
            f"   Entity检索: {'' if summary['config']['entity_enabled'] else ''} (Top-K={summary['config']['entity_top_k']})",
            f"   重排序方法: {summary['config']['rerank_method']}",
        ]
        
        
        router_cfg = summary.get('router_config', {})
        if router_cfg.get('enabled'):
            report_lines.extend([
                "",
                " 路由器信息:",
                f"   模型: {router_cfg.get('model', 'N/A')}",
                f"   策略: {router_cfg.get('strategy', 'N/A')}",
            ])
            
            routing_counts = defaultdict(int)
            for r in results:
                if r.routing_info:
                    routing_counts[r.routing_info] += 1
            if routing_counts:
                report_lines.append("   路由分布:")
                for info, count in sorted(routing_counts.items(), key=lambda x: -x[1]):
                    report_lines.append(f"     {info}: {count}次")
        
        report_lines.extend([
            "",
            " 检索统计:",
            f"   平均Sentence检索数: {summary['retrieval_stats']['avg_sentence_retrieved']:.1f}",
            f"   平均Episodic检索数: {summary['retrieval_stats']['avg_episodic_retrieved']:.1f}",
            f"   平均Entity检索数: {summary['retrieval_stats']['avg_entity_retrieved']:.1f}",
            f"   平均检索时间: {summary['retrieval_stats']['avg_retrieval_time']:.3f}s",
            "",
            " 评估分数:",
        ])
        
        for key, value in summary.get('scores', {}).items():
            report_lines.append(f"   {key}: {value:.4f}")
        
        cascade_stats = summary.get('cascade_stats', {})
        if cascade_stats:
            report_lines.extend([
                "",
                " 级联量化剪枝统计:",
                f"   激活次数: {cascade_stats.get('cascade_active_count', 0)}",
                f"   平均Stage1输入: {cascade_stats.get('avg_stage1_input', 0):.1f}",
                f"   平均Stage1输出: {cascade_stats.get('avg_stage1_output', 0):.1f}",
                f"   平均Stage2冲突组: {cascade_stats.get('avg_stage2_conflicts', 0):.1f}",
                f"   平均Stage2丢弃块: {cascade_stats.get('avg_stage2_dropped', 0):.1f}",
                f"   平均Stage2输出: {cascade_stats.get('avg_stage2_output', 0):.1f}",
                f"   平均Stage3 MMR轮数: {cascade_stats.get('avg_stage3_mmr_iterations', 0):.1f}",
                f"   平均最终Token数: {cascade_stats.get('avg_tokens_used', 0):.0f}",
            ])
        
        cascade_cfg = summary.get('cascade_config', {})
        if cascade_cfg.get('enabled'):
            report_lines.extend([
                "",
                " 级联量化配置:",
                f"   剪枝模式: {cascade_cfg.get('prune_mode', 'N/A')}",
                f"   最大Token数: {cascade_cfg.get('max_context_tokens', 'N/A')}",
                f"   λ_mmr: {cascade_cfg.get('lambda_mmr', 'N/A')}",
                f"   Stage2: {'ON' if cascade_cfg.get('stage2_enabled') else 'OFF'}",
                f"   Stage3_MMR: {'ON' if cascade_cfg.get('stage3_mmr_enabled') else 'OFF'}",
            ])
        
        report_lines.extend([
            "",
            " Token统计:",
            f"   总Prompt Tokens: {summary['token_stats']['total_prompt_tokens']:,}",
            f"   总Completion Tokens: {summary['token_stats']['total_completion_tokens']:,}",
            f"   总Tokens: {summary['token_stats']['total_tokens']:,}",
            "",
            "=" * 100
        ])
        
        report_file = self.output_dir / f"report_{timestamp}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print('\n'.join(report_lines))
    
    
    def run_benchmark(self) -> Dict[str, Any]:
        """Run benchmark."""
        logger.info(" 开始三重检索融合 Benchmark 测试...")
        
        
        test_cases = self._load_test_cases()
        
        if not test_cases:
            logger.error(" 没有测试用例可执行")
            return {}
        
        
        
        logger.info("\n [System Warmup] 正在进行系统预热，跑通所有模型和逻辑链路...")
        try:
            # Avoid mutating LogRecord fields before other handlers process the record.
            # prev_level = logging.getLogger().getEffectiveLevel()
            # logging.getLogger().setLevel(logging.WARNING)
            
            warmup_case = test_cases[0]
            start_warm = time.time()
            
            _ = self._test_single_qa(warmup_case, -1)
            
            # Avoid mutating LogRecord fields before other handlers process the record.
            # logging.getLogger().setLevel(prev_level)
            
            logger.info(f" [System Warmup] 预热完成，耗时 {time.time() - start_warm:.2f}s。开始正式测试...\n")
            
        except Exception as e:
            logger.warning(f" 预热过程中发生错误 (不影响后续测试): {e}")
        
        
        results = []
        start_time = time.time()
        
        for idx, test_case in enumerate(tqdm(test_cases, desc="测试进度")):
            logger.info(f"\n{'='*80}")
            logger.info(f" [{idx+1}/{len(test_cases)}] 测试 QA {test_case.qa_index}")
            logger.info(f"   问题: {test_case.question[:80]}...")
            
            result = self._test_single_qa(test_case, idx)
            results.append(result)
            
            
            self._save_single_report(result, idx)
            
            if result.success:
                answer_preview = str(result.generated_answer)[:50] if result.generated_answer else "N/A"
                logger.info(f"    成功 | 答案: {answer_preview}...")
                if result.scores and isinstance(result.scores, dict):
                    score_items = []
                    for k, v in list(result.scores.items())[:5]:
                        if k == "error":
                            continue
                        if isinstance(v, (int, float)):
                            score_items.append(f"{k}={v:.3f}")
                        elif isinstance(v, bool):
                            score_items.append(f"{k}={v}")
                        elif isinstance(v, str):
                            score_items.append(f"{k}={v[:20]}")
                    if score_items:
                        score_summary = ", ".join(score_items)
                        logger.info(f"    分数: {score_summary}")
            else:
                error_msg = str(result.error_message)[:50] if result.error_message else "Unknown error"
                logger.warning(f"    失败: {error_msg}...")
        
        total_time = time.time() - start_time
        
        summary = self._generate_summary(results, total_time)
        
        
        self._save_results(results, summary)
        
        logger.info("\n 三重检索融合 Benchmark 测试完成!")
        
        return summary


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
    parser = argparse.ArgumentParser(description="LongMemEval 三重检索融合 Benchmark")
    
    # Dataset-specific handling used by the reproduction workflow.
    parser.add_argument("--dataset-size", default="s", choices=["s", "m"],
                       help="数据集大小: s (small) 或 m (medium)")
    parser.add_argument("--dataset-dir", default=None,
                       help="原始数据集目录路径")
    
    parser.add_argument("--sentence-graph-dir",
                       default=str(paths.LONGMEMEVAL_HIERARCHICAL_STEP3_DIR),
                       help="消息级别图谱目录")
    parser.add_argument("--episodic-graph-dir",
                       default=str(paths.LONGMEMEVAL_EPISODIC_GRAPHS_DIR),
                       help="情景记忆图谱目录")
    parser.add_argument("--entity-graph-dir",
                       default=str(paths.LONGMEMEVAL_ENTITY_RELATION_GRAPHS_DIR),
                       help="实体关系图谱目录")
    
    
    parser.add_argument("--sentence-top-k", type=int, default=60,
                       help="消息级别检索 Top-K (默认60)")
    parser.add_argument("--episodic-top-k", type=int, default=40,
                       help="情景记忆检索 Top-K (默认40)")
    parser.add_argument("--entity-top-k", type=int, default=40,
                       help="实体关系检索 Top-K (默认40)")
    
    parser.add_argument("--disable-sentence", action="store_true",
                       help="禁用消息级别检索")
    parser.add_argument("--disable-episodic", action="store_true",
                       help="禁用情景记忆检索")
    parser.add_argument("--disable-entity", action="store_true",
                       help="禁用实体关系检索")
    
    parser.add_argument("--llm-model", 
                    #    default="gpt-4o-mini-openrouter",
                       default="gpt-4o-mini-closeai",
                       help="答案生成 LLM 模型")
    parser.add_argument("--llm-evaluate-model", 
                       #    default="gpt-4o-mini-openrouter",
                       default="gpt-4o-mini-closeai",
                       help="答案评估 LLM 模型")
    # parser.add_argument("--llm-model", 
    #                    default="gpt-4o-mini-openrouter",
    # parser.add_argument("--llm-evaluate-model", 
    #                    default="gpt-4o-mini-openrouter",
    
    
    parser.add_argument("--rerank-method", 
                       default="baai",
                       choices=["baai", "qwen", "jina", "qwen-sili", "qwen-dashscope", "gte-dashscope", "none"],
                       help="重排序方法")
    
    parser.add_argument("--fusion-method", default="concatenation",
                       choices=["concatenation", "interleaved"],
                       help="融合方法")
    
    parser.add_argument("--disable-second-stage-rerank", action="store_true",
                       help="禁用两阶段检索（默认启用对融合结果进行二次重排序）")
    parser.add_argument("--second-stage-rerank-method",
                       default=None,
                    #    default="baai",
                       choices=["baai", "qwen", "jina", "qwen-sili", "qwen-dashscope", "gte-dashscope"],
                       help="二次重排序方法（默认使用 --rerank-method）")
    parser.add_argument("--first-stage-top-k", type=int, default=None,
                       help="第一阶段检索数量（默认使用各检索器的 top-k）")
    parser.add_argument("--final-top-k", type=int, default=25,
                       help="二次重排后保留的最终数量（默认25）")
    
    parser.add_argument("--output-dir",
                       default=str(paths.LONGMEMEVAL_TASK_EVAL_RESULTS_DIR / "triple_fusion_router_cascade"),
                       help="结果输出目录")
    
    parser.add_argument("--start-qa", type=int, default=None,
                       help="起始 QA 索引 (0-indexed, 0-499)")
    parser.add_argument("--end-qa", type=int, default=None,
                       help="结束 QA 索引 (0-indexed, 0-499)")
    
    parser.add_argument("--max-tests", type=int, default=None,
                       help="最大测试数量")
    parser.add_argument("--generation-max-tokens", type=int, default=1000,
                       help="答案生成最大输出 token 数；推理模型会作为 max_completion_tokens 使用（默认1000）")
    
    
    parser.add_argument("--enable-router", action="store_true",
                       help="Enable the category router and select tower combinations by question_type.")
    parser.add_argument("--router-strategy",
                       choices=["aggressive", "conservative"],
                       default="aggressive",
                       help="路由策略: aggressive(路由所有类别到最优配置) | conservative(仅路由高置信度类别)")
    
    parser.add_argument("--enable-cascade-pruner", action="store_true",
                       help="启用级联量化剪枝器（Stage1硬过滤→Stage2跨塔去重→Stage3 MMR打包）")
    parser.add_argument("--cascade-max-context-tokens", type=int, default=2500,
                       help="级联剪枝最大上下文Token数量 (default: 2500)")
    parser.add_argument("--cascade-prune-mode",
                       choices=["BUDGET_MAX", "STRICT_THRESHOLD", "CLIFF_EARLY_STOP", "DYNAMIC_ADAPTIVE"],
                       default="BUDGET_MAX",
                       help="级联剪枝模式 (default: BUDGET_MAX)")
    parser.add_argument("--cascade-mad-multiplier", type=float, default=2.5,
                       help="Stage1 MAD离群点检测倍率，值越大越保守 (default: 2.5)")
    parser.add_argument("--cascade-cliff-tolerance", type=float, default=2.0,
                       help="CLIFF_EARLY_STOP模式悬崖容忍度，作用于原始 logits 尺度 (default: 2.0)")
    parser.add_argument("--cascade-absolute-min-score", type=float, default=0.0,
                       help="STRICT_THRESHOLD模式绝对最低分数阈值 (default: 0.0)")
    parser.add_argument("--cascade-lambda-mmr", type=float, default=0.6,
                       help="Stage3 MMR多样性权重 (default: 0.6)")
    parser.add_argument("--no-cascade-stage2", action="store_true",
                       help="禁用Stage2跨塔去重（默认启用）")
    parser.add_argument("--no-cascade-stage3-mmr", action="store_true",
                       help="禁用Stage3 MMR打包（默认启用）")
    parser.add_argument("--no-cascade-stage1", action="store_true",
                       help="禁用Stage1硬过滤（默认启用）")
    parser.add_argument("--no-cascade-cap-to-input", action="store_true",
                       help="禁用cap_to_input_tokens（默认启用，防止级联膨胀超出输入token总量）")
    parser.add_argument("--cascade-tower-min-ratio", type=str, default=None,
                       help="Stage3 per-tower 最低配额，格式: 'H:0.50,E:0.20,KG:0.15'。"
                            "省略则禁用 tower reservation（向后兼容）")
    parser.add_argument("--cascade-adaptive-dataset", type=str, default=None,
                       help="DYNAMIC_ADAPTIVE模式使用的数据集名称，用于查找类别适配器 (e.g. 'longmemeval')")
    
    args = parser.parse_args()
    
    if args.start_qa is not None and args.end_qa is not None:
        if args.start_qa > args.end_qa:
            logger.error(f" start-qa ({args.start_qa}) 必须小于等于 end-qa ({args.end_qa})")
            return 1
    if args.start_qa is not None and (args.start_qa < 0 or args.start_qa > 499):
        logger.error(f" start-qa 必须在 0-499 范围内")
        return 1
    if args.end_qa is not None and (args.end_qa < 0 or args.end_qa > 499):
        logger.error(f" end-qa 必须在 0-499 范围内")
        return 1
    
    try:
        llm_client = LLMClient(model_name=args.llm_model)
        llm_evaluate_client = LLMClient(model_name=args.llm_evaluate_model)
        
        
        tower_router = None
        if args.enable_router:
            tower_router = LongMemEvalTowerRouter(
                model_name=args.llm_model,
                strategy=args.router_strategy,
            )
            
            args.output_dir = f"{args.output_dir}_routed_{args.router_strategy}"
            logger.info(f" 塔路由器:  启用")
            logger.info(f"   模型: {args.llm_model}")
            logger.info(f"   策略: {args.router_strategy}")
            logger.info(f"   输出目录(已修改): {args.output_dir}")
        else:
            logger.info(f" 塔路由器:  禁用 (使用静态配置)")
        
        enable_cascade = args.enable_cascade_pruner
        if enable_cascade:
            args.output_dir = f"{args.output_dir}_cascade"
            logger.info(f" 级联量化剪枝:  启用")
            logger.info(f"   最大Token数: {args.cascade_max_context_tokens}")
            logger.info(f"   剪枝模式: {args.cascade_prune_mode}")
            logger.info(f"   lambda_mmr: {args.cascade_lambda_mmr}")
            logger.info(f"   Stage2去重: {'禁用' if args.no_cascade_stage2 else '启用'}")
            logger.info(f"   Stage3 MMR: {'禁用' if args.no_cascade_stage3_mmr else '启用'}")
            logger.info(f"   Stage1过滤: {'禁用' if args.no_cascade_stage1 else '启用'}")
            logger.info(f"   Cap to input: {'禁用' if args.no_cascade_cap_to_input else '启用'}")
            if args.cascade_tower_min_ratio:
                logger.info(f"   Tower Min Ratio: {args.cascade_tower_min_ratio}")
            logger.info(f"   输出目录(已修改): {args.output_dir}")
        else:
            logger.info(f" 级联量化剪枝:  禁用")
        
        benchmark = LongMemEvalTripleFusionBenchmark(
            dataset_size=args.dataset_size,
            dataset_dir=args.dataset_dir,
            sentence_graph_dir=args.sentence_graph_dir,
            episodic_graph_dir=args.episodic_graph_dir,
            entity_graph_dir=args.entity_graph_dir,
            llm_client=llm_client,
            llm_evaluate_client=llm_evaluate_client,
            output_dir=args.output_dir,
            sentence_top_k=args.sentence_top_k,
            episodic_top_k=args.episodic_top_k,
            entity_top_k=args.entity_top_k,
            enable_sentence=not args.disable_sentence,
            enable_episodic=not args.disable_episodic,
            enable_entity=not args.disable_entity,
            max_tests=args.max_tests,
            rerank_method=args.rerank_method,
            fusion_method=args.fusion_method,
            start_qa=args.start_qa,
            end_qa=args.end_qa,
            enable_second_stage_rerank=not args.disable_second_stage_rerank,
            second_stage_rerank_method=args.second_stage_rerank_method,
            first_stage_top_k=args.first_stage_top_k,
            final_top_k=args.final_top_k,
            tower_router=tower_router,
            generation_max_tokens=args.generation_max_tokens,
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
        )
        
        summary = benchmark.run_benchmark()
        
        return 0
        
    except Exception as e:
        logger.error(f" Benchmark 执行失败: {e}")
        logger.debug(traceback.format_exc())
        return 1
    
    finally:
        if EVALUATION_AVAILABLE:
            try:
                cleanup_evaluation_models()
            except Exception as e:
                logger.warning(f"清理资源失败: {e}")


if __name__ == "__main__":
    exit(main())
