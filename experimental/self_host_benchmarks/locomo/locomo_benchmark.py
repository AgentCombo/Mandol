#!/usr/bin/env python3
"""LoCoMo dual-tower retrieval benchmark.

Evaluates a dual-tower architecture that combines hierarchical
retrieval with knowledge-graph retrieval, fuses the results, and
generates answers via an LLM.  Supports multiple reranker backends
and parallel tower execution.
"""
# Environment setup before imports
import os
import sys

# Suppress unnecessary output
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['SENTENCE_TRANSFORMERS_DISABLE_PROGRESS_BAR'] = '1'

# Globally disable tqdm
# Keep tqdm enabled to avoid errors
# import tqdm
# tqdm.tqdm.disable = True

# import os
# import sys
import json
import logging
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
from dataclasses import dataclass
import numpy as np
from collections import defaultdict
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(project_root))

# Hierarchical retrieval modules
from dev.core.semantic_graph import SemanticGraph
from dev.hierarchical.hierarchical_memory_interface import HierarchicalMemoryInterface
from dev.hierarchical.hierarchical_memory_manager import MemoryLevel, SummaryType
from dev.retrieval.rerank_manager import RerankerManager

# Knowledge-graph retrieval modules
from dev.retrieval.advance_retriever import MultiRetriever
from dev.retrieval.entity_relation_retriever import EntityRelationRetriever
from dev.retrieval.retrieval_interface import RetrievalMethod

# LLM and evaluation modules
from dev.llm.llm_client import LLMClient
from benchmark_locomo.task_eval.evaluation import (
    calculate_comprehensive_scores, 
    batch_evaluate,
    cleanup_evaluation_models,
    get_model_manager
)

# Existing test components
from benchmark_locomo.task_eval.locomo_benchmark_hierarchical import (
    HierarchicalContextBuilder, 
    LocomoHierarchicalBenchmarkTester
)
from benchmark_locomo.task_eval.locomo_benchmark_entity_relation import (
    LoCoMoEntityRelationBenchmark,
    LoCoMoGraphTestCase
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DualTowerRetrievalResult:
    """Container for a single dual-tower retrieval and evaluation result.

    Attributes:
        sample_id: LoCoMo sample identifier.
        question: The question text.
        category: Question category (1–5).
        expected_answer: Ground-truth answer.
        hierarchical_context: Context retrieved by the hierarchical tower.
        hierarchical_retrieval_time: Time spent in hierarchical retrieval.
        graph_retrieved_units: Units retrieved by the graph tower.
        graph_retrieval_time: Time spent in graph retrieval.
        graph_retrieval_details: Detailed graph retrieval metadata.
        final_answer: The LLM-generated answer.
        reasoning_process: The model's chain-of-thought.
        confidence_score: Model confidence (0–1).
        fusion_method: Name of the fusion strategy used.
        generation_time: Time spent generating the answer.
        evaluation_scores: Metric scores from the evaluation module.
        evaluation_success: Whether evaluation completed without error.
    """
    sample_id: str
    question: str
    category: int
    expected_answer: str
    
    # Hierarchical retrieval results
    hierarchical_context: Dict[str, Any]
    hierarchical_retrieval_time: float
    
    # Knowledge-graph retrieval results
    graph_retrieved_units: List[Any]
    graph_retrieval_time: float
    graph_retrieval_details: Dict[str, Any]
    
    # Fusion generation results
    final_answer: str
    reasoning_process: str  # chain-of-thought
    confidence_score: float
    fusion_method: str
    generation_time: float
    
    # Evaluation results
    evaluation_scores: Dict[str, float]
    evaluation_success: bool

class LoCoMoDualTowerBenchmark:
    """Benchmark runner for the LoCoMo dual-tower retrieval system.

    Orchestrates hierarchical and graph-based retrieval, fuses results,
    generates answers, and evaluates them against ground truth.
    """
    
    def __init__(self,
        # Data paths
        enhanced_graphs_dir: str = "benchmark_locomo/dataset/locomo/hierarchical/step3_final_graphs",
        step3_graphs_dir: str = "benchmark_locomo/dataset/locomo/step3_semantic_graph", 
        qa_dataset_path: str = "benchmark_locomo/dataset/locomo/locomo10.json",
        
        # LLM configuration
        llm_client: Optional[LLMClient] = None,
        llm_evaluate_client: Optional[LLMClient] = None,
        
        # Output configuration
        output_dir: str = "benchmark_locomo/task_eval/results/locomo_dual_tower_benchmark",
        
        # Retrieval configuration
        use_entity_relation: bool = True,
        topk_hierarchical_l0: int = 15,
        topk_hierarchical_l1: int = 5,
        topk_hierarchical_l2: int = 1,
        topk_similarity: int = 15,
        topk_graph: int = 0,
        
        # Fusion configuration
        fusion_strategy: str = "context_aware",
        fusion_weights: Dict[str, float] = None,
        
        # Reranker configuration — extended support
        reranker_type: str = "baai",
        reranker_configs: Optional[Dict[str, str]] = None,
        reranker_manager: Optional[RerankerManager] = None,
        
        # Test configuration
        target_sample_ids: Optional[List[str]] = None,
        max_workers: int = 1,
        parallel_towers: bool = True):
        """Initialize the dual-tower benchmark tester.

        Args:
            enhanced_graphs_dir: Directory with enhanced graphs (hierarchical tower).
            step3_graphs_dir: Directory with step-3 semantic graphs (graph tower).
            qa_dataset_path: Path to the LoCoMo QA dataset.
            llm_client: LLM client for answer generation.
            llm_evaluate_client: LLM client for answer evaluation.
            output_dir: Directory for result files.
            use_entity_relation: Whether to enable entity-relation retrieval.
            topk_hierarchical_*: Top-k values for each hierarchical level.
            topk_similarity: Top-k for semantic similarity search.
            topk_graph: Top-k for graph-based retrieval.
            fusion_strategy: Name of the fusion strategy.
            fusion_weights: Weights for each tower in the fusion step.
            reranker_type: Reranker backend identifier.
            reranker_configs: Mapping of reranker names to model identifiers.
            reranker_manager: Pre-initialized :class:`RerankerManager`.
            target_sample_ids: Restrict evaluation to these sample IDs.
            max_workers: Thread-pool size for parallel evaluation.
            parallel_towers: Whether to run both towers concurrently.
        """
        
        # Path configuration
        self.enhanced_graphs_dir = Path(enhanced_graphs_dir)
        self.step3_graphs_dir = Path(step3_graphs_dir)
        self.qa_dataset_path = Path(qa_dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # LLM configuration
        self.llm_client = llm_client or LLMClient(model_name="gpt-4o-mini-closeai")
        self.llm_evaluate_client = llm_evaluate_client or LLMClient(model_name="deepseek-chat")
        
        # Retrieval configuration
        self.use_entity_relation = use_entity_relation
        self.topk_hierarchical_l0 = topk_hierarchical_l0
        self.topk_hierarchical_l1 = topk_hierarchical_l1
        self.topk_hierarchical_l2 = topk_hierarchical_l2
        self.topk_similarity = topk_similarity
        self.topk_graph = topk_graph
        
        # Fusion configuration
        self.fusion_strategy = fusion_strategy
        self.fusion_weights = fusion_weights or {
            "hierarchical": 0.6,
            "graph": 0.4
        }
        
        # Reranker configuration — all types
        self.reranker_type = reranker_type
        self.reranker_configs = reranker_configs or {
            "baai": "BAAI/bge-reranker-v2-m3",
            "qwen": "Qwen/Qwen3-Reranker-0.6B",
            "jina": "jinaai/jina-reranker-v3",
            "qwen-sili": "Qwen/Qwen3-Reranker-8B",
            "qwen-dashscope": "qwen3-rerank",
            "gte-dashscope": "gte-rerank-v2"
        }
        self.reranker_manager = reranker_manager
        
        # Keep target_sample_ids as list to preserve order
        self.target_sample_ids = target_sample_ids  # preserve order
        self.max_workers = max_workers
        self.parallel_towers = parallel_towers
        
        # Initialize subsystems
        self._initialize_hierarchical_system()
        self._initialize_graph_system()

        # Retriever cache
        self.hierarchical_interfaces: Dict[str, HierarchicalMemoryInterface] = {}
        self.loaded_graphs: Dict[str, SemanticGraph] = {}
        
        # Test data
        self.test_cases: List[Dict[str, Any]] = []
        self.test_results: List[DualTowerRetrievalResult] = []
        
        # Statistics
        self.stats = {
            "total_samples_loaded": 0,
            "total_test_cases": 0,
            "successful_hierarchical": 0,
            "successful_graph": 0,
            "successful_dual_tower": 0,
            "failed_retrievals": 0,
            "fusion_strategy": fusion_strategy
        }
        
        logger.info("LoCoMo dual-tower benchmark tester initialized")
        logger.info(f"Hierarchical config: L0={topk_hierarchical_l0}, L1={topk_hierarchical_l1}, L2={topk_hierarchical_l2}")
        logger.info(f"Graph config: semantic top-k={topk_similarity}, graph top-k={topk_graph}")
        logger.info(f"Fusion: strategy={fusion_strategy}, weights={self.fusion_weights}")
        logger.info(f"Reranker: {reranker_type}")
        logger.info(f"Answer generation model: {getattr(self.llm_client, 'model_name', 'unknown')}")
        logger.info(f"Answer evaluation model: {getattr(self.llm_evaluate_client, 'model_name', 'unknown')}")
        logger.info(f"Parallel towers: {'enabled' if parallel_towers else 'disabled'}")
    
    def preload_retrievers(self):
        """Pre-load retrievers for all samples (new dataset format)."""
        logger.info("Pre-loading retrievers for all samples...")
        
        if not self.available_samples:
            logger.warning("No samples available, skipping pre-load")
            return
        
        total_samples = len(self.available_samples)
        logger.info(f"Preparing to pre-load {total_samples} sample retrievers")
        
        for i, sample_id in enumerate(self.available_samples, 1):
            try:
                logger.info(f"Pre-loading [{i}/{total_samples}] {sample_id}...")
                
                # Pre-load hierarchical retrieval interface
                if sample_id not in self.hierarchical_interfaces:
                    # New format: no _enhanced suffix
                    enhanced_dir = self.enhanced_graphs_dir / sample_id
                    
                    # Fallback: try old format
                    if not enhanced_dir.exists():
                        enhanced_dir = self.enhanced_graphs_dir / f"{sample_id}_enhanced"
                    
                    if enhanced_dir.exists():
                        graph, hierarchical_interface = self.hierarchical_tester.load_enhanced_conversation_graph(
                            sample_id, str(self.enhanced_graphs_dir)
                        )
                        self.hierarchical_interfaces[sample_id] = hierarchical_interface
                        self.loaded_graphs[sample_id] = graph
                        logger.debug(f"{sample_id} hierarchical interface loaded")
                    else:
                        logger.warning(f"{sample_id} hierarchical graph directory not found")
                
                # Pre-build all graph retriever indices
                if sample_id in self.graph_benchmark.multi_retrievers:
                    multi_retriever = self.graph_benchmark.multi_retrievers[sample_id]
                    logger.info(f"Pre-building retriever indices for {sample_id}...")
                    
                    build_stats = multi_retriever.build_all_indexes(force_rebuild=False)
                    
                    logger.info(f"{sample_id} index build complete: "
                            f"built={build_stats['built_count']}, "
                            f"skipped={build_stats['skipped_count']}, "
                            f"failed={build_stats['failed_count']}, "
                            f"duration={build_stats['total_duration']:.2f}s")
                else:
                    logger.warning(f"{sample_id} graph retriever not found")
                    
            except Exception as e:
                logger.error(f"Pre-load failed for {sample_id}: {e}")
                continue
        
        logger.info(f"Pre-load complete: hierarchical={len(self.hierarchical_interfaces)}, "
                f"graph={len(self.graph_benchmark.multi_retrievers)}")

    def _initialize_hierarchical_system(self):
        """Initialize the hierarchical retrieval system."""
        logger.info("Initializing hierarchical retrieval system...")
        
        self.hierarchical_tester = LocomoHierarchicalBenchmarkTester(
            llm_client=self.llm_client,
            llm_evaluate_client=self.llm_evaluate_client,
            output_dir=str(self.output_dir / "hierarchical_temp"),
            include_llm_evaluation=False,  # evaluate at dual-tower level
            reranker_manager=self.reranker_manager
        )
        
        # Update hierarchical config
        self.hierarchical_tester.hierarchical_config.update({
            "l2_top_k": self.topk_hierarchical_l2,
            "l1_top_k": self.topk_hierarchical_l1,
            "l0_top_k": self.topk_hierarchical_l0,
            "rerank_method": self.reranker_type,
            "enable_graph_expansion": False,  # disable graph expansion in dual-tower mode
            "fusion_method": "rrf"
        })
        
        logger.info("Hierarchical retrieval system initialized")
    
    def _initialize_graph_system(self):
        """Initialize the knowledge-graph retrieval system."""
        logger.info("Initializing knowledge-graph retrieval system...")
        
        self.graph_benchmark = LoCoMoEntityRelationBenchmark(
            semantic_graphs_dir=str(self.step3_graphs_dir),
            qa_dataset_path=str(self.qa_dataset_path),
            llm_client=self.llm_client,
            llm_evaluate_client=self.llm_evaluate_client,
            output_dir=str(self.output_dir / "graph_temp"),
            use_entity_relation=self.use_entity_relation,
            target_sample_ids=list(self.target_sample_ids) if self.target_sample_ids else None,
            max_workers=1,  # concurrency controlled at dual-tower level
            topk_similarity=self.topk_similarity,
            topk_graph=self.topk_graph,
            reranker_type=self.reranker_type,
            reranker_manager=self.reranker_manager
        )
        
        logger.info("Knowledge-graph retrieval system initialized")

    def load_systems(self, max_samples: Optional[int] = None, sample_id: Optional[str] = None):
        """
        Load both retrieval systems with dataset compatibility checks.
        
        Args:
            max_samples: Max samples (batch mode).
            sample_id: Single sample ID (individual mode).
        """
        logger.info("Loading dual-tower retrieval systems...")
        
        # Validate dataset compatibility (batch mode)
        if not sample_id:
            compatibility_report = self.validate_dataset_compatibility()
            
            if compatibility_report["compatibility_status"] == "error_no_samples":
                raise RuntimeError("No samples available")
            
            if compatibility_report["compatibility_status"] == "error_enhanced_not_found":
                raise RuntimeError("Hierarchical graph directory missing or malformed")
            
            if compatibility_report["compatibility_status"] == "warning_legacy":
                logger.warning("Legacy dataset format detected; compatibility issues possible")
            
            if compatibility_report["enhanced_graphs_format"] == "old_format_with_suffix":
                logger.warning("Legacy hierarchical graphs (_enhanced suffix) detected; prefer new format")
        
        # Single-sample load mode
        if sample_id:
            logger.info(f"Single-sample mode: {sample_id}")
            
            # Validate sample availability
            if not self._is_sample_available(sample_id):
                raise RuntimeError(f"Sample {sample_id} data incomplete or unavailable")
            
            self.available_samples = [sample_id]
            self.stats["total_samples_loaded"] = 1
            
            # Load graph retriever for this sample
            self.graph_benchmark.target_sample_ids = {sample_id}
            self.graph_benchmark.load_semantic_graphs()
            
            # Load hierarchical interface for this sample
            self._load_single_sample_hierarchical(sample_id)
            
            logger.info(f"Single sample {sample_id} loaded")
            return
        
        # Original batch loading logic
        available_samples = self._get_available_samples()
        
        if self.target_sample_ids:
            if isinstance(self.target_sample_ids, (list, tuple)):
                available_samples = [s for s in self.target_sample_ids if s in available_samples]
            else:
                available_samples = [s for s in available_samples if s in self.target_sample_ids]
        
        if max_samples:
            available_samples = available_samples[:max_samples]
        
        logger.info(f"Preparing to load {len(available_samples)} samples: {available_samples}")
        
        # Load knowledge-graph system
        logger.info("Loading knowledge-graph retrieval system...")
        try:
            self.graph_benchmark.target_sample_ids = set(available_samples)
            self.graph_benchmark.load_semantic_graphs()
            logger.info(f"Knowledge-graph system loaded: {len(self.graph_benchmark.semantic_graphs)} graphs")
        except Exception as e:
            logger.error(f"Knowledge-graph system load failed: {e}")
            raise
        
        # Validate hierarchical system data availability
        logger.info("Validating hierarchical retrieval data...")
        hierarchical_available = []
        for sample_id in available_samples:
            # New format: no _enhanced suffix
            enhanced_dir = self.enhanced_graphs_dir / sample_id
            
            # Fallback: try old format
            if not enhanced_dir.exists():
                enhanced_dir = self.enhanced_graphs_dir / f"{sample_id}_enhanced"
            
            if enhanced_dir.exists():
                hierarchical_available.append(sample_id)
            else:
                logger.warning(f"⚠️  分层数据不存在: {sample_id}")
        
        logger.info(f"✅ 分层系统可用样本: {len(hierarchical_available)} 个")
        
        # 确定最终的交集样本
        final_samples = list(set(available_samples) & set(hierarchical_available))
        logger.info(f"🎯 双塔系统最终样本: {len(final_samples)} 个 - {final_samples}")
        
        if not final_samples:
            raise RuntimeError("没有找到同时支持两个检索系统的样本")
        
        self.available_samples = final_samples
        self.stats["total_samples_loaded"] = len(final_samples)

        # 预加载所有检索器
        logger.info("\n🔄 开始预加载检索器...")
        self.preload_retrievers()
        logger.info("✅ 检索器预加载完成\n")

    def _is_sample_available(self, sample_id: str) -> bool:
        """检查单个样本是否在两个图谱系统中都可用"""
        # 检查实体关系图谱（step3_graphs_dir）
        step3_dir = self.step3_graphs_dir / sample_id
        step3_semantic_map = step3_dir / "semantic_map_data" / "semantic_map.json"
        
        if not step3_dir.exists():
            logger.warning(f"⚠️  {sample_id} 实体关系图谱不存在: {step3_dir}")
            return False
        
        if not step3_semantic_map.exists():
            logger.warning(f"⚠️  {sample_id} 实体关系图谱缺少semantic_map.json")
            return False
        
        # 检查分层图谱（enhanced_graphs_dir，无_enhanced后缀）
        enhanced_dir = self.enhanced_graphs_dir / sample_id
        hierarchical_overview = enhanced_dir / "hierarchical_overview.json"
        enhanced_semantic_map = enhanced_dir / "semantic_map_data" / "semantic_map.json"
        
        if not enhanced_dir.exists():
            logger.warning(f"⚠️  {sample_id} 分层图谱不存在: {enhanced_dir}")
            return False
        
        if not hierarchical_overview.exists():
            logger.warning(f"⚠️  {sample_id} 分层图谱缺少hierarchical_overview.json")
            return False
        
        if not enhanced_semantic_map.exists():
            logger.warning(f"⚠️  {sample_id} 分层图谱缺少semantic_map.json")
            return False
        
        logger.debug(f"✅ {sample_id} 在两个图谱系统中都可用")
        return True

    def _load_single_sample_hierarchical(self, sample_id: str):
        """加载单个样本的分层接口 - 适配新数据集格式"""
        try:
            # 新数据集格式：无_enhanced后缀
            enhanced_dir = self.enhanced_graphs_dir / sample_id
            
            if not enhanced_dir.exists():
                logger.error(f"⚠️  {sample_id} 分层图谱目录不存在: {enhanced_dir}")
                raise FileNotFoundError(f"分层图谱目录不存在: {enhanced_dir}")
            
            # 验证必要文件
            hierarchical_overview = enhanced_dir / "hierarchical_overview.json"
            if not hierarchical_overview.exists():
                logger.warning(f"⚠️  {sample_id} 缺少hierarchical_overview.json，但继续加载")
            
            # 加载图谱和分层接口
            graph, hierarchical_interface = self.hierarchical_tester.load_enhanced_conversation_graph(
                sample_id, str(self.enhanced_graphs_dir)
            )
            
            self.hierarchical_interfaces[sample_id] = hierarchical_interface
            self.loaded_graphs[sample_id] = graph
            
            logger.info(f"✅ {sample_id} 分层接口加载完成")
            
        except Exception as e:
            logger.error(f"❌ 加载 {sample_id} 分层接口失败: {e}")
            raise
    
    def _get_available_samples(self) -> List[str]:
        """获取可用的样本ID列表 - 修复版：正确区分两种图谱目录"""
        available_samples_step3 = []
        available_samples_enhanced = []
        
        # 从step3_graphs_dir获取实体关系图谱样本
        if self.step3_graphs_dir.exists():
            for item in self.step3_graphs_dir.iterdir():
                if item.is_dir() and item.name.startswith("conv-"):
                    # 验证是否为有效的实体关系图谱（检查semantic_map.json）
                    semantic_map_file = item / "semantic_map_data" / "semantic_map.json"
                    if semantic_map_file.exists():
                        available_samples_step3.append(item.name)
                        logger.debug(f"实体关系图谱样本: {item.name}")
        
        # 从enhanced_graphs_dir获取分层图谱样本（无_enhanced后缀）
        if self.enhanced_graphs_dir.exists():
            for item in self.enhanced_graphs_dir.iterdir():
                if item.is_dir() and item.name.startswith("conv-"):
                    # 新数据集格式：无_enhanced后缀
                    sample_id = item.name
                    
                    # 验证是否为有效的分层图谱（检查hierarchical_overview.json）
                    hierarchical_overview = item / "hierarchical_overview.json"
                    semantic_map_file = item / "semantic_map_data" / "semantic_map.json"
                    
                    if hierarchical_overview.exists() and semantic_map_file.exists():
                        available_samples_enhanced.append(sample_id)
                        logger.debug(f"分层图谱样本: {sample_id}")
                    else:
                        logger.warning(f"⚠️  样本 {sample_id} 缺少必要文件")
        
        # 取两个集合的交集
        final_samples = list(set(available_samples_step3) & set(available_samples_enhanced))
        
        logger.info(f"发现可用样本: 实体关系图谱={len(available_samples_step3)}, "
                f"分层图谱={len(available_samples_enhanced)}, "
                f"交集={len(final_samples)}")
        
        if len(final_samples) < len(available_samples_step3) or len(final_samples) < len(available_samples_enhanced):
            missing_in_step3 = set(available_samples_enhanced) - set(available_samples_step3)
            missing_in_enhanced = set(available_samples_step3) - set(available_samples_enhanced)
            
            if missing_in_step3:
                logger.warning(f"⚠️  实体关系图谱中缺少的样本: {missing_in_step3}")
            if missing_in_enhanced:
                logger.warning(f"⚠️  分层图谱中缺少的样本: {missing_in_enhanced}")
        
        return sorted(final_samples)
    
    def load_test_cases(self, sample_id: Optional[str] = None):
        """
        加载测试用例
        
        Args:
            sample_id: 如果指定，只加载该样本的测试用例
        """
        logger.info(f"加载{'单样本' if sample_id else '双塔系统'}测试用例...")
        
        #  如果是单样本模式，先清空测试用例
        if sample_id:
            self.test_cases = []
        
        try:
            with open(self.qa_dataset_path, 'r', encoding='utf-8') as f:
                qa_data = json.load(f)
            
            for item in qa_data:
                item_sample_id = item["sample_id"]
                
                #  单样本模式：只处理指定样本
                if sample_id and item_sample_id != sample_id:
                    continue
                
                # 批量模式：只处理已加载的样本
                if not sample_id and item_sample_id not in self.available_samples:
                    continue
                
                qa_list = item.get("qa", [])
                for i, qa_item in enumerate(qa_list):
                    # 检查问题有效性
                    if not isinstance(qa_item, dict) or "question" not in qa_item:
                        continue
                    
                    question = qa_item["question"]
                    category = qa_item.get("category", 1)
                    
                    # 统一答案处理逻辑
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
                        "question": question,
                        "category": category,
                        "expected_answer": expected_answer,
                        "question_id": f"{item_sample_id}_q{i+1}",
                        "evidence": qa_item.get("evidence", [])
                    }
                    
                    self.test_cases.append(test_case)
            
            #  单样本模式不更新总数
            if not sample_id:
                self.stats["total_test_cases"] = len(self.test_cases)
            
            # Statistics
            if sample_id:
                logger.info(f"✅ 样本 {sample_id} 测试用例加载完成: {len(self.test_cases)} 个")
            else:
                category_counts = {}
                sample_counts = {}
                for case in self.test_cases:
                    category_counts[case["category"]] = category_counts.get(case["category"], 0) + 1
                    sample_counts[case["sample_id"]] = sample_counts.get(case["sample_id"], 0) + 1
                
                logger.info(f"✅ 测试用例加载完成: {len(self.test_cases)} 个")
                logger.info(f"📊 类别分布: {category_counts}")
                logger.info(f"📊 样本分布: {dict(list(sample_counts.items())[:5])}..." + 
                        (f" (共{len(sample_counts)}个样本)" if len(sample_counts) > 5 else ""))
            
        except Exception as e:
            logger.error(f"加载测试用例失败: {e}")
            raise
    
    def run_dual_tower_benchmark(self, sequential_mode: bool = False):
        """
        运行双塔召回benchmark测试
        
        Args:
            sequential_mode: 是否使用逐个样本模式（适合多样本、内存受限场景）
        """
        logger.info("🚀 开始运行双塔召回benchmark测试...")
        
        #  逐个样本模式（保持参数顺序）
        if sequential_mode:
            logger.info("📋 使用逐个样本模式（内存友好 + 增量保存）")
            
            #  获取要测试的样本列表 - 保持用户输入的顺序
            if self.target_sample_ids:
                # 如果是列表，保持原顺序；如果是集合，按原始输入顺序
                if isinstance(self.target_sample_ids, list):
                    samples_to_test = self.target_sample_ids
                else:
                    # 如果之前被转为set，需要保持原始顺序
                    samples_to_test = list(self.target_sample_ids)
                    logger.warning(f"⚠️  样本ID从集合转换，顺序可能不完全一致")
            else:
                samples_to_test = self._get_available_samples()
            
            logger.info(f"📊 共需测试 {len(samples_to_test)} 个样本（按指定顺序）")
            logger.info(f"📋 测试顺序: {samples_to_test}")
            
            # 初始化进度跟踪
            self._initialize_progress_tracking(samples_to_test)
            
            #  逐个样本测试（严格按顺序）
            for i, sample_id in enumerate(samples_to_test, 1):
                logger.info(f"\n{'='*80}")
                logger.info(f"🔄 [{i}/{len(samples_to_test)}] 处理样本: {sample_id}")
                logger.info(f"{'='*80}")
                
                try:
                    sample_results = self.run_single_sample_benchmark(sample_id)
                    logger.info(f"✓ 样本 {sample_id} 完成，获得 {len(sample_results)} 个结果")
                except Exception as e:
                    logger.error(f"✗ 样本 {sample_id} 处理失败: {e}")
                    self._mark_sample_failed(sample_id, str(e))
                    continue
            
            logger.info(f"\n{'='*80}")
            logger.info("✅ 所有样本测试完成")
            logger.info(f"{'='*80}")
            
            #  生成最终汇总报告
            self._generate_final_summary_from_incremental()
            
            return
        
        # 原有的批量测试逻辑（不推荐用于多样本）
        if not self.test_cases:
            raise RuntimeError("没有测试用例，请先调用load_test_cases()")
        
        total_tests = len(self.test_cases)
        logger.info(f"总测试数: {total_tests}")
        
        if self.max_workers == 1:
            self._run_single_threaded_dual_tower_tests()
        else:
            self._run_multi_threaded_dual_tower_tests()
        
        logger.info("双塔召回benchmark测试完成")
        
        # ✨ 新增：批量模式下也生成单样本报告
        if self.test_results:
            logger.info("\n📊 批量模式：生成单样本报告...")
            self._generate_sample_reports_from_batch()

    def run_single_sample_benchmark(self, sample_id: str) -> List[DualTowerRetrievalResult]:
        """
        运行单个样本的benchmark测试，测试完立即保存
        
        Args:
            sample_id: 样本ID
            
        Returns:
            该样本的测试结果列表
        """
        logger.info(f"🚀 开始测试样本: {sample_id}")
        
        sample_results = []
        
        try:
            # 1. 加载该样本的系统
            logger.info(f"📦 加载样本 {sample_id} 的检索系统...")
            self.load_systems(sample_id=sample_id)
            
            # 2. 加载该样本的测试用例
            logger.info(f"📋 加载样本 {sample_id} 的测试用例...")
            self.load_test_cases(sample_id=sample_id)
            
            if not self.test_cases:
                logger.warning(f"⚠️  样本 {sample_id} 没有测试用例")
                return sample_results
            
            logger.info(f"📊 样本 {sample_id} 共有 {len(self.test_cases)} 个测试用例")
            
            # 3. 运行测试
            for test_case in tqdm(self.test_cases, desc=f"测试 {sample_id}"):
                try:
                    result = self._run_single_dual_tower_test(test_case)
                    if result:
                        sample_results.append(result)
                        self.test_results.append(result)  # 同时添加到总结果
                        if result.evaluation_success:
                            self.stats["successful_dual_tower"] += 1
                        else:
                            self.stats["failed_retrievals"] += 1
                    else:
                        self.stats["failed_retrievals"] += 1
                        
                except Exception as e:
                    self.stats["failed_retrievals"] += 1
                    logger.error(f"❌ 测试失败: {test_case['question_id']} - {e}")
                    continue
            
            logger.info(f"✅ 样本 {sample_id} 测试完成: 成功={len([r for r in sample_results if r.evaluation_success])}, "
                    f"失败={len([r for r in sample_results if not r.evaluation_success])}")
            
            #  4. 立即保存该样本的结果（增量保存）
            self._save_sample_results_incrementally(sample_id, sample_results)
            
            return sample_results
            
        except Exception as e:
            logger.error(f"❌ 样本 {sample_id} 测试失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # logger.debug(traceback.format_exc())
            return sample_results
        
        finally:
            # 5. 清理该样本的资源
            logger.info(f"🧹 清理样本 {sample_id} 的资源...")
            self.unload_sample(sample_id)

    def _save_sample_results_incrementally(self, sample_id: str, sample_results: List[DualTowerRetrievalResult]):
        """
        增量保存单个样本的测试结果，并生成样本级别的可读性报告
        """
        if not sample_results:
            logger.warning(f"样本 {sample_id} 没有结果需要保存")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        #  计算样本统计指标
        successful_results = [r for r in sample_results if r.evaluation_success]
        
        # 计算平均指标（只针对成功的评估）
        performance_metrics = {}
        timing_metrics = {}
        system_success_rates = {}
        category_performance = {}
        parallel_performance = {}  #  新增：并行性能统计
        
        if successful_results:
            # 性能指标
            f1_scores = [r.evaluation_scores.get("token_f1", 0.0) for r in successful_results]
            semantic_scores = [r.evaluation_scores.get("semantic_similarity", 0.0) for r in successful_results]
            llm_scores = [r.evaluation_scores.get("llm_accuracy", 0.0) for r in successful_results]
            exact_match_scores = [r.evaluation_scores.get("exact_match", 0.0) for r in successful_results]
            confidence_scores = [r.confidence_score for r in successful_results]
            
            performance_metrics = {
                "avg_f1_score": float(np.mean(f1_scores)),
                "std_f1_score": float(np.std(f1_scores)),
                "avg_semantic_similarity": float(np.mean(semantic_scores)),
                "avg_llm_accuracy": float(np.mean(llm_scores)),
                "avg_exact_match": float(np.mean(exact_match_scores)),
                "avg_confidence": float(np.mean(confidence_scores))
            }
            
            # 时间指标
            hierarchical_times = [r.hierarchical_retrieval_time for r in successful_results]
            graph_times = [r.graph_retrieval_time for r in successful_results]
            generation_times = [r.generation_time for r in successful_results]
            
            timing_metrics = {
                "avg_hierarchical_time": float(np.mean(hierarchical_times)),
                "avg_graph_time": float(np.mean(graph_times)),
                "avg_generation_time": float(np.mean(generation_times)),
                "avg_total_time": float(np.mean([h+g+gen for h, g, gen in zip(hierarchical_times, graph_times, generation_times)]))
            }
            
            #  并行性能统计（如果启用了并行模式）
            if self.parallel_towers:
                parallel_actual_times = []
                parallel_sequential_times = []
                parallel_speedup_ratios = []
                parallel_time_saved_list = []
                
                for r in successful_results:
                    if hasattr(r, 'parallel_actual_time'):
                        parallel_actual_times.append(r.parallel_actual_time)
                        parallel_sequential_times.append(r.parallel_sequential_theory)
                        parallel_speedup_ratios.append(r.parallel_speedup_ratio)
                        parallel_time_saved_list.append(r.parallel_time_saved)
                
                if parallel_actual_times:
                    parallel_performance = {
                        "parallel_enabled": True,
                        "avg_parallel_actual_time": float(np.mean(parallel_actual_times)),
                        "avg_sequential_theory_time": float(np.mean(parallel_sequential_times)),
                        "avg_speedup_ratio": float(np.mean(parallel_speedup_ratios)),
                        "avg_time_saved": float(np.mean(parallel_time_saved_list)),
                        "total_time_saved": float(np.sum(parallel_time_saved_list)),
                        "min_speedup": float(np.min(parallel_speedup_ratios)),
                        "max_speedup": float(np.max(parallel_speedup_ratios))
                    }
            
            # 系统成功率
            hierarchical_success_count = sum(1 for r in successful_results if r.hierarchical_context.get("hierarchical_enabled", False))
            graph_success_count = sum(1 for r in successful_results if len(r.graph_retrieved_units) > 0)
            both_success_count = sum(1 for r in successful_results 
                                if r.hierarchical_context.get("hierarchical_enabled", False) and len(r.graph_retrieved_units) > 0)
            
            system_success_rates = {
                "hierarchical_success_rate": hierarchical_success_count / len(successful_results),
                "graph_success_rate": graph_success_count / len(successful_results),
                "both_systems_success_rate": both_success_count / len(successful_results),
                "dual_tower_advantage": both_success_count / max(hierarchical_success_count, graph_success_count, 1)
            }
            
            # 按类别统计
            category_stats = defaultdict(lambda: {"count": 0, "f1_scores": [], "llm_scores": []})
            for r in successful_results:
                cat = r.category
                category_stats[cat]["count"] += 1
                category_stats[cat]["f1_scores"].append(r.evaluation_scores.get("token_f1", 0.0))
                category_stats[cat]["llm_scores"].append(r.evaluation_scores.get("llm_accuracy", 0.0))
            
            category_performance = {
                str(cat): {
                    "test_count": stats["count"],
                    "avg_f1_score": float(np.mean(stats["f1_scores"])),
                    "avg_llm_accuracy": float(np.mean(stats["llm_scores"])),
                    "hierarchical_success_rate": sum(1 for r in successful_results 
                                                    if r.category == cat and r.hierarchical_context.get("hierarchical_enabled", False)) / stats["count"],
                    "graph_success_rate": sum(1 for r in successful_results 
                                            if r.category == cat and len(r.graph_retrieved_units) > 0) / stats["count"]
                }
                for cat, stats in category_stats.items()
            }
        
        #  构建完整的样本数据结构（添加并行性能）
        sample_data = {
            "sample_info": {
                "sample_id": sample_id,
                "timestamp": datetime.now().isoformat(),
                "test_count": len(sample_results),
                "successful_count": len(successful_results),
                "failed_count": len(sample_results) - len(successful_results),
                "fusion_strategy": self.fusion_strategy,
                "fusion_weights": self.fusion_weights,
                "parallel_towers_enabled": self.parallel_towers,  #  新增
                "hierarchical_config": {
                    "l0_top_k": self.topk_hierarchical_l0,
                    "l1_top_k": self.topk_hierarchical_l1,
                    "l2_top_k": self.topk_hierarchical_l2
                },
                "graph_config": {
                    "topk_similarity": self.topk_similarity,
                    "topk_graph": self.topk_graph,
                    "use_entity_relation": self.use_entity_relation
                }
            },
            "performance_metrics": performance_metrics,
            "timing_metrics": timing_metrics,
            "parallel_performance": parallel_performance,  #  新增：并行性能统计
            "system_success_rates": system_success_rates,
            "category_performance": category_performance,
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
                    "evaluation_scores": r.evaluation_scores,
                    "evaluation_success": r.evaluation_success,
                    "hierarchical_time": r.hierarchical_retrieval_time,
                    "graph_time": r.graph_retrieval_time,
                    "generation_time": r.generation_time,
                    #  添加并行时间统计
                    "parallel_stats": {
                        "actual_time": getattr(r, 'parallel_actual_time', None),
                        "sequential_theory": getattr(r, 'parallel_sequential_theory', None),
                        "speedup_ratio": getattr(r, 'parallel_speedup_ratio', None),
                        "time_saved": getattr(r, 'parallel_time_saved', None)
                    } if self.parallel_towers and hasattr(r, 'parallel_actual_time') else None
                }
                for r in sample_results
            ]
        }
        
        # 1. 保存样本级别的详细结果（JSON）
        sample_file = self.output_dir / f"sample_{sample_id}_{timestamp}.json"
        
        with open(sample_file, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 样本 {sample_id} 结果已保存: {sample_file}")
        
        #  2. 生成样本级别的可读性报告（包含并行统计）
        readable_report_file = self._generate_sample_readable_report(
            sample_id, sample_results, timestamp, parallel_performance
        )
        logger.info(f"📄 样本 {sample_id} 可读性报告: {readable_report_file}")
        
        # 3. 追加到累积结果文件
        cumulative_file = self.output_dir / "cumulative_results.jsonl"
        
        with open(cumulative_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(sample_data, ensure_ascii=False) + '\n')
        
        logger.info(f"💾 样本 {sample_id} 结果已追加到累积文件")
        
        # 4. 更新进度文件
        self._update_progress_file(sample_id, len(sample_results), 
                                len([r for r in sample_results if r.evaluation_success]))

    def _generate_sample_reports_from_batch(self):
        """
        从批量测试结果中生成单样本报告
        在批量模式下调用，为每个样本生成独立的报告文件
        """
        logger.info("🔄 从批量结果生成单样本报告...")
        
        # 按样本ID分组结果
        from collections import defaultdict
        sample_grouped_results = defaultdict(list)
        
        for result in self.test_results:
            sample_grouped_results[result.sample_id].append(result)
        
        logger.info(f"📊 发现 {len(sample_grouped_results)} 个样本的测试结果")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        generated_count = 0
        
        # 为每个样本生成报告
        for sample_id, sample_results in sample_grouped_results.items():
            try:
                logger.info(f"📝 生成样本 {sample_id} 的报告 ({len(sample_results)} 个测试)...")
                
                # 计算样本统计指标（复用现有逻辑）
                successful_results = [r for r in sample_results if r.evaluation_success]
                
                # 计算并行性能统计
                parallel_performance = {}
                if self.parallel_towers:
                    parallel_actual_times = []
                    parallel_sequential_times = []
                    parallel_speedup_ratios = []
                    parallel_time_saved_list = []
                    
                    for r in successful_results:
                        if hasattr(r, 'parallel_actual_time'):
                            parallel_actual_times.append(r.parallel_actual_time)
                            parallel_sequential_times.append(r.parallel_sequential_theory)
                            parallel_speedup_ratios.append(r.parallel_speedup_ratio)
                            parallel_time_saved_list.append(r.parallel_time_saved)
                    
                    if parallel_actual_times:
                        parallel_performance = {
                            "parallel_enabled": True,
                            "avg_parallel_actual_time": float(np.mean(parallel_actual_times)),
                            "avg_sequential_theory_time": float(np.mean(parallel_sequential_times)),
                            "avg_speedup_ratio": float(np.mean(parallel_speedup_ratios)),
                            "avg_time_saved": float(np.mean(parallel_time_saved_list)),
                            "total_time_saved": float(np.sum(parallel_time_saved_list)),
                            "min_speedup": float(np.min(parallel_speedup_ratios)),
                            "max_speedup": float(np.max(parallel_speedup_ratios))
                        }
                
                # 1. 生成JSON报告
                sample_data = self._build_sample_data_structure(
                    sample_id, sample_results, successful_results, parallel_performance
                )
                
                sample_json_file = self.output_dir / f"batch_sample_{sample_id}_{timestamp}.json"
                with open(sample_json_file, 'w', encoding='utf-8') as f:
                    json.dump(sample_data, f, ensure_ascii=False, indent=2)
                
                logger.info(f"  ✅ JSON报告: {sample_json_file.name}")
                
                # 2. 生成可读性报告
                readable_file = self._generate_sample_readable_report(
                    sample_id, sample_results, timestamp, parallel_performance
                )
                
                logger.info(f"  ✅ 可读报告: {readable_file.name}")
                
                generated_count += 1
                
            except Exception as e:
                logger.error(f"❌ 生成样本 {sample_id} 报告失败: {e}")
                continue
        
        logger.info(f"\n✅ 批量模式单样本报告生成完成: {generated_count}/{len(sample_grouped_results)} 个样本")
        
        # 生成批量处理摘要
        summary_file = self.output_dir / f"batch_samples_summary_{timestamp}.txt"
        self._generate_batch_samples_summary(sample_grouped_results, summary_file)
        logger.info(f"📋 批量样本摘要: {summary_file}")


    def _build_sample_data_structure(self, 
                                    sample_id: str, 
                                    sample_results: List[DualTowerRetrievalResult],
                                    successful_results: List[DualTowerRetrievalResult],
                                    parallel_performance: Dict[str, Any]) -> Dict[str, Any]:
        """
        构建样本数据结构（供JSON报告使用）
        复用自 _save_sample_results_incrementally 的逻辑
        """
        # 计算性能指标
        performance_metrics = {}
        timing_metrics = {}
        system_success_rates = {}
        category_performance = {}
        
        if successful_results:
            # 性能指标
            f1_scores = [r.evaluation_scores.get("token_f1", 0.0) for r in successful_results]
            semantic_scores = [r.evaluation_scores.get("semantic_similarity", 0.0) for r in successful_results]
            llm_scores = [r.evaluation_scores.get("llm_accuracy", 0.0) for r in successful_results]
            exact_match_scores = [r.evaluation_scores.get("exact_match", 0.0) for r in successful_results]
            confidence_scores = [r.confidence_score for r in successful_results]
            
            performance_metrics = {
                "avg_f1_score": float(np.mean(f1_scores)),
                "std_f1_score": float(np.std(f1_scores)),
                "avg_semantic_similarity": float(np.mean(semantic_scores)),
                "avg_llm_accuracy": float(np.mean(llm_scores)),
                "avg_exact_match": float(np.mean(exact_match_scores)),
                "avg_confidence": float(np.mean(confidence_scores))
            }
            
            # 时间指标
            hierarchical_times = [r.hierarchical_retrieval_time for r in successful_results]
            graph_times = [r.graph_retrieval_time for r in successful_results]
            generation_times = [r.generation_time for r in successful_results]
            
            timing_metrics = {
                "avg_hierarchical_time": float(np.mean(hierarchical_times)),
                "avg_graph_time": float(np.mean(graph_times)),
                "avg_generation_time": float(np.mean(generation_times)),
                "avg_total_time": float(np.mean([h+g+gen for h, g, gen in zip(hierarchical_times, graph_times, generation_times)]))
            }
            
            # 系统成功率
            hierarchical_success_count = sum(1 for r in successful_results if r.hierarchical_context.get("hierarchical_enabled", False))
            graph_success_count = sum(1 for r in successful_results if len(r.graph_retrieved_units) > 0)
            both_success_count = sum(1 for r in successful_results 
                                if r.hierarchical_context.get("hierarchical_enabled", False) and len(r.graph_retrieved_units) > 0)
            
            system_success_rates = {
                "hierarchical_success_rate": hierarchical_success_count / len(successful_results),
                "graph_success_rate": graph_success_count / len(successful_results),
                "both_systems_success_rate": both_success_count / len(successful_results),
                "dual_tower_advantage": both_success_count / max(hierarchical_success_count, graph_success_count, 1)
            }
            
            # 按类别统计
            category_stats = defaultdict(lambda: {"count": 0, "f1_scores": [], "llm_scores": []})
            for r in successful_results:
                cat = r.category
                category_stats[cat]["count"] += 1
                category_stats[cat]["f1_scores"].append(r.evaluation_scores.get("token_f1", 0.0))
                category_stats[cat]["llm_scores"].append(r.evaluation_scores.get("llm_accuracy", 0.0))
            
            category_performance = {
                str(cat): {
                    "test_count": stats["count"],
                    "avg_f1_score": float(np.mean(stats["f1_scores"])),
                    "avg_llm_accuracy": float(np.mean(stats["llm_scores"])),
                    "hierarchical_success_rate": sum(1 for r in successful_results 
                                                    if r.category == cat and r.hierarchical_context.get("hierarchical_enabled", False)) / stats["count"],
                    "graph_success_rate": sum(1 for r in successful_results 
                                            if r.category == cat and len(r.graph_retrieved_units) > 0) / stats["count"]
                }
                for cat, stats in category_stats.items()
            }
        
        # 构建完整数据结构
        return {
            "sample_info": {
                "sample_id": sample_id,
                "timestamp": datetime.now().isoformat(),
                "test_count": len(sample_results),
                "successful_count": len(successful_results),
                "failed_count": len(sample_results) - len(successful_results),
                "fusion_strategy": self.fusion_strategy,
                "fusion_weights": self.fusion_weights,
                "parallel_towers_enabled": self.parallel_towers,
                "processing_mode": "batch",  # 标记为批量模式
                "hierarchical_config": {
                    "l0_top_k": self.topk_hierarchical_l0,
                    "l1_top_k": self.topk_hierarchical_l1,
                    "l2_top_k": self.topk_hierarchical_l2
                },
                "graph_config": {
                    "topk_similarity": self.topk_similarity,
                    "topk_graph": self.topk_graph,
                    "use_entity_relation": self.use_entity_relation
                }
            },
            "performance_metrics": performance_metrics,
            "timing_metrics": timing_metrics,
            "parallel_performance": parallel_performance,
            "system_success_rates": system_success_rates,
            "category_performance": category_performance,
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
                    "evaluation_scores": r.evaluation_scores,
                    "evaluation_success": r.evaluation_success,
                    "hierarchical_time": r.hierarchical_retrieval_time,
                    "graph_time": r.graph_retrieval_time,
                    "generation_time": r.generation_time,
                    "parallel_stats": {
                        "actual_time": getattr(r, 'parallel_actual_time', None),
                        "sequential_theory": getattr(r, 'parallel_sequential_theory', None),
                        "speedup_ratio": getattr(r, 'parallel_speedup_ratio', None),
                        "time_saved": getattr(r, 'parallel_time_saved', None)
                    } if self.parallel_towers and hasattr(r, 'parallel_actual_time') else None
                }
                for r in sample_results
            ]
        }


    def _generate_batch_samples_summary(self, 
                                        sample_grouped_results: Dict[str, List], 
                                        summary_file: Path):
        """生成批量样本处理的摘要报告"""
        lines = []
        
        lines.append("=" * 100)
        lines.append("批量模式 - 样本处理摘要")
        lines.append("=" * 100)
        lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"处理样本数: {len(sample_grouped_results)}")
        lines.append("=" * 100)
        
        lines.append(f"\n{'样本ID':<15} {'测试数':>8} {'成功数':>8} {'失败数':>8} {'成功率':>10} {'平均F1':>10}")
        lines.append("-" * 80)
        
        for sample_id, results in sorted(sample_grouped_results.items()):
            test_count = len(results)
            success_count = len([r for r in results if r.evaluation_success])
            failed_count = test_count - success_count
            success_rate = (success_count / test_count * 100) if test_count > 0 else 0
            
            # 计算平均F1
            successful_results = [r for r in results if r.evaluation_success]
            avg_f1 = 0.0
            if successful_results:
                f1_scores = [r.evaluation_scores.get("token_f1", 0.0) for r in successful_results]
                avg_f1 = np.mean(f1_scores)
            
            lines.append(f"{sample_id:<15} {test_count:>8} {success_count:>8} {failed_count:>8} "
                        f"{success_rate:>9.1f}% {avg_f1:>10.3f}")
        
        lines.append("\n" + "=" * 100)
        lines.append("注意: 详细报告请查看 batch_sample_*_*.json 和 batch_sample_*_readable_*.txt")
        lines.append("=" * 100)
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        # 同时打印到控制台
        print('\n'.join(lines))

    def _generate_sample_readable_report(self, 
                            sample_id: str, 
                            sample_results: List[DualTowerRetrievalResult], 
                            timestamp: str,
                            parallel_performance: Dict[str, Any] = None) -> Path:
        """
        为单个样本生成可读性报告（包含并行性能统计）
        """
        lines = []
        
        lines.append("=" * 100)
        lines.append(f"样本 {sample_id} - 双塔召回测试报告")
        lines.append("=" * 100)
        lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"样本ID: {sample_id}")
        lines.append(f"测试数量: {len(sample_results)}")
        lines.append(f"并行模式: {'启用' if self.parallel_towers else '禁用'}")
        lines.append("=" * 100)
        
        # 计算样本统计
        successful_count = len([r for r in sample_results if r.evaluation_success])
        failed_count = len(sample_results) - successful_count
        
        # 计算平均指标
        valid_results = [r for r in sample_results if r.evaluation_success]
        
        if valid_results:
            avg_f1 = np.mean([r.evaluation_scores.get("token_f1", 0.0) for r in valid_results])
            avg_semantic = np.mean([r.evaluation_scores.get("semantic_similarity", 0.0) for r in valid_results])
            avg_llm = np.mean([r.evaluation_scores.get("llm_accuracy", 0.0) for r in valid_results])
            avg_exact_match = np.mean([r.evaluation_scores.get("exact_match", 0.0) for r in valid_results])
            avg_confidence = np.mean([r.confidence_score for r in valid_results])
            
            avg_hier_time = np.mean([r.hierarchical_retrieval_time for r in valid_results])
            avg_graph_time = np.mean([r.graph_retrieval_time for r in valid_results])
            avg_gen_time = np.mean([r.generation_time for r in valid_results])
            
            hier_success = sum(1 for r in valid_results if r.hierarchical_context.get("hierarchical_enabled", False))
            graph_success = sum(1 for r in valid_results if len(r.graph_retrieved_units) > 0)
            both_success = sum(1 for r in valid_results 
                            if r.hierarchical_context.get("hierarchical_enabled", False) and len(r.graph_retrieved_units) > 0)
            
            lines.append(f"\n📊 整体统计:")
            lines.append(f"   ✓ 成功测试: {successful_count} ({successful_count/len(sample_results)*100:.1f}%)")
            lines.append(f"   ✗ 失败测试: {failed_count} ({failed_count/len(sample_results)*100:.1f}%)")
            
            lines.append(f"\n🎯 性能指标:")
            lines.append(f"   - 平均F1分数: {avg_f1:.3f}")
            lines.append(f"   - 平均语义相似度: {avg_semantic:.3f}")
            lines.append(f"   - 平均LLM准确率: {avg_llm:.3f}")
            lines.append(f"   - 平均精确匹配: {avg_exact_match:.3f}")
            lines.append(f"   - 平均置信度: {avg_confidence:.3f}")
            
            lines.append(f"\n⏱️  时间性能:")
            lines.append(f"   - 平均分层检索: {avg_hier_time:.3f}s")
            lines.append(f"   - 平均图检索: {avg_graph_time:.3f}s")
            lines.append(f"   - 平均答案生成: {avg_gen_time:.3f}s")
            lines.append(f"   - 平均总时间: {avg_hier_time + avg_graph_time + avg_gen_time:.3f}s")
            
            #  并行性能统计部分
            if self.parallel_towers and parallel_performance and parallel_performance.get("parallel_enabled"):
                lines.append(f"\n⚡ 并行性能统计:")
                lines.append(f"   - 平均并行实际时间: {parallel_performance['avg_parallel_actual_time']:.3f}s")
                lines.append(f"   - 平均理论串行时间: {parallel_performance['avg_sequential_theory_time']:.3f}s")
                lines.append(f"   - 平均加速比: {parallel_performance['avg_speedup_ratio']:.2f}x")
                lines.append(f"   - 平均节省时间: {parallel_performance['avg_time_saved']:.3f}s")
                lines.append(f"   - 累计节省时间: {parallel_performance['total_time_saved']:.3f}s")
                lines.append(f"   - 加速比范围: {parallel_performance['min_speedup']:.2f}x ~ {parallel_performance['max_speedup']:.2f}x")
                
                # 计算效率提升百分比
                if parallel_performance['avg_sequential_theory_time'] > 0:
                    efficiency = (parallel_performance['avg_time_saved'] / parallel_performance['avg_sequential_theory_time']) * 100
                    lines.append(f"   - 效率提升: {efficiency:.1f}%")
            
            lines.append(f"\n🔄 系统成功率:")
            lines.append(f"   - 分层检索: {hier_success}/{len(valid_results)} ({hier_success/len(valid_results)*100:.1f}%)")
            lines.append(f"   - 图检索: {graph_success}/{len(valid_results)} ({graph_success/len(valid_results)*100:.1f}%)")
            lines.append(f"   - 双塔同时成功: {both_success}/{len(valid_results)} ({both_success/len(valid_results)*100:.1f}%)")
            
            # 按类别统计（添加平均LLM准确率）
            category_stats = defaultdict(lambda: {"count": 0, "f1_scores": [], "llm_scores": [], "success": 0})
            for r in valid_results:
                cat = r.category
                category_stats[cat]["count"] += 1
                category_stats[cat]["f1_scores"].append(r.evaluation_scores.get("token_f1", 0.0))
                category_stats[cat]["llm_scores"].append(r.evaluation_scores.get("llm_accuracy", 0.0))  # 收集LLM分数
                if r.evaluation_success:
                    category_stats[cat]["success"] += 1
            
            if category_stats:
                lines.append(f"\n📋 类别统计:")
                category_names = {
                    1: "多跳问题", 
                    2: "时间问题", 
                    3: "开放域问题", 
                    4: "单跳问题", 
                    5: "对抗性问题"
                }
                
                for cat, stats in sorted(category_stats.items()):
                    cat_name = category_names.get(cat, f"类别{cat}")
                    avg_cat_f1 = np.mean(stats["f1_scores"]) if stats["f1_scores"] else 0.0
                    avg_cat_llm = np.mean(stats["llm_scores"]) if stats["llm_scores"] else 0.0  # 计算平均LLM准确率
                    success_rate = stats["success"] / stats["count"] * 100 if stats["count"] > 0 else 0
                    
                    lines.append(f"\n   {cat_name}:")
                    lines.append(f"     - 测试数: {stats['count']}")
                    lines.append(f"     - 平均F1: {avg_cat_f1:.3f}")
                    lines.append(f"     - 平均LLM准确率: {avg_cat_llm:.3f}")  # 新增：显示LLM准确率
                    lines.append(f"     - 成功率: {success_rate:.1f}%")
        else:
            lines.append(f"\n⚠️  警告: 所有 {len(sample_results)} 个测试都失败了")
        
        # 详细结果列表
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
            
            lines.append(f"\n问题:")
            lines.append(f"  {result.question}")
            
            lines.append(f"\n标准答案:")
            lines.append(f"  {result.expected_answer}")
            
            lines.append(f"\n生成答案:")
            lines.append(f"  {result.final_answer}")
            
            lines.append(f"\n置信度: {result.confidence_score:.3f}")
            
            # 系统状态
            hier_status = "✓" if result.hierarchical_context.get("hierarchical_enabled", False) else "✗"
            graph_status = "✓" if len(result.graph_retrieved_units) > 0 else "✗"
            lines.append(f"\n系统状态:")
            lines.append(f"  分层检索: {hier_status} | 图检索: {graph_status}")
            
            # 评估分数
            if result.evaluation_success:
                scores = result.evaluation_scores
                lines.append(f"\n评估分数:")
                lines.append(f"  - F1: {scores.get('token_f1', 0):.3f}")
                lines.append(f"  - 语义相似度: {scores.get('semantic_similarity', 0):.3f}")
                lines.append(f"  - 精确匹配: {scores.get('exact_match', 0):.3f}")
                if 'llm_accuracy' in scores:
                    lines.append(f"  - LLM准确率: {scores.get('llm_accuracy', 0):.3f}")
            else:
                lines.append(f"\n⚠️  评估失败")
            
            # 时间统计
            lines.append(f"\n时间统计:")
            lines.append(f"  分层: {result.hierarchical_retrieval_time:.3f}s | "
                        f"图: {result.graph_retrieval_time:.3f}s | "
                        f"生成: {result.generation_time:.3f}s")
            
            #  并行时间统计（如果有）
            if self.parallel_towers and hasattr(result, 'parallel_actual_time'):
                lines.append(f"  并行统计: 实际={result.parallel_actual_time:.3f}s | "
                            f"理论串行={result.parallel_sequential_theory:.3f}s | "
                            f"加速比={result.parallel_speedup_ratio:.2f}x")
        
        lines.append(f"\n{'='*100}")
        lines.append(f"样本 {sample_id} 报告结束")
        lines.append(f"{'='*100}")
        
        # 保存报告
        report_file = self.output_dir / f"sample_{sample_id}_readable_{timestamp}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        # 同时打印摘要到控制台
        print(f"\n{'='*80}")
        print(f"📊 样本 {sample_id} 测试完成")
        print(f"{'='*80}")
        if valid_results:
            print(f"✅ 成功: {successful_count}/{len(sample_results)} ({successful_count/len(sample_results)*100:.1f}%)")
            print(f"📈 平均F1: {avg_f1:.3f} | 语义相似度: {avg_semantic:.3f} | LLM准确率: {avg_llm:.3f}")
            print(f"⏱️  平均时间: {avg_hier_time + avg_graph_time + avg_gen_time:.2f}s")
            
            #  打印并行统计摘要
            if self.parallel_towers and parallel_performance and parallel_performance.get("parallel_enabled"):
                print(f"⚡ 并行加速: {parallel_performance['avg_speedup_ratio']:.2f}x, "
                    f"节省: {parallel_performance['total_time_saved']:.2f}s")
            
            print(f"🔄 双塔成功: {both_success}/{len(valid_results)} ({both_success/len(valid_results)*100:.1f}%)")
        else:
            print(f"❌ 所有测试失败")
        print(f"📄 详细报告: {report_file}")
        print(f"{'='*80}\n")
        
        return report_file
    
    # def _save_sample_results_incrementally(self, sample_id: str, sample_results: List[DualTowerRetrievalResult]):
    #     """
    #     增量保存单个样本的测试结果
        
    #     Args:
    #         sample_id: 样本ID
    #         sample_results: 该样本的测试结果
    #     """
    #     if not sample_results:
    #         logger.warning(f"样本 {sample_id} 没有结果需要保存")
    #         return
        
    #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    #     # 1. 保存样本级别的详细结果
    #     sample_file = self.output_dir / f"sample_{sample_id}_{timestamp}.json"
        
    #     sample_data = {
    #         "sample_id": sample_id,
    #         "timestamp": datetime.now().isoformat(),
    #         "test_count": len(sample_results),
    #         "successful_count": len([r for r in sample_results if r.evaluation_success]),
    #         "failed_count": len([r for r in sample_results if not r.evaluation_success]),
    #         "results": [
    #             {
    #                 "question": r.question,
    #                 "category": r.category,
    #                 "expected_answer": r.expected_answer,
    #                 "final_answer": r.final_answer,
    #                 "reasoning_process": r.reasoning_process,
    #                 "confidence_score": r.confidence_score,
    #                 "hierarchical_success": r.hierarchical_context.get("hierarchical_enabled", False),
    #                 "graph_results_count": len(r.graph_retrieved_units),
    #                 "evaluation_scores": r.evaluation_scores,
    #                 "evaluation_success": r.evaluation_success,
    #                 "hierarchical_time": r.hierarchical_retrieval_time,
    #                 "graph_time": r.graph_retrieval_time,
    #                 "generation_time": r.generation_time
    #             }
    #             for r in sample_results
    #         ]
    #     }
        
    #     with open(sample_file, 'w', encoding='utf-8') as f:
    #         json.dump(sample_data, f, ensure_ascii=False, indent=2)
        
    #     logger.info(f"💾 样本 {sample_id} 结果已保存: {sample_file}")
        
    #     # 2. 追加到累积结果文件
    #     cumulative_file = self.output_dir / "cumulative_results.jsonl"
        
    #     with open(cumulative_file, 'a', encoding='utf-8') as f:
    #         f.write(json.dumps(sample_data, ensure_ascii=False) + '\n')
        
    #     logger.info(f"💾 样本 {sample_id} 结果已追加到累积文件")
        
    #     # 3. 更新进度文件
    #     self._update_progress_file(sample_id, len(sample_results), 
    #                             len([r for r in sample_results if r.evaluation_success]))

    def unload_sample(self, sample_id: str):
        """
        卸载单个样本的资源，包括显存和内存清理
        
        Args:
            sample_id: 样本ID
        """
        logger.info(f"🧹 清理样本 {sample_id} 的资源...")
        
        # 清理分层接口
        if sample_id in self.hierarchical_interfaces:
            del self.hierarchical_interfaces[sample_id]
            logger.debug(f"  ✓ 清理分层接口")
        
        # 清理加载的图
        if sample_id in self.loaded_graphs:
            del self.loaded_graphs[sample_id]
            logger.debug(f"  ✓ 清理加载图")
        
        # 清理图检索器
        if sample_id in self.graph_benchmark.multi_retrievers:
            retriever = self.graph_benchmark.multi_retrievers[sample_id]
            # 清理检索器的索引缓存
            if hasattr(retriever, 'clear_cache'):
                retriever.clear_cache()
            del self.graph_benchmark.multi_retrievers[sample_id]
            logger.debug(f"  ✓ 清理图检索器")
        
        if sample_id in self.graph_benchmark.entity_relation_retrievers:
            del self.graph_benchmark.entity_relation_retrievers[sample_id]
            logger.debug(f"  ✓ 清理实体关系检索器")
        
        if sample_id in self.graph_benchmark.semantic_graphs:
            del self.graph_benchmark.semantic_graphs[sample_id]
            logger.debug(f"  ✓ 清理语义图")
        
        # 从可用样本列表中移除
        if hasattr(self, 'available_samples') and sample_id in self.available_samples:
            self.available_samples.remove(sample_id)
        
        #  清理GPU显存
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug(f"  ✓ 清理GPU缓存")
        except ImportError:
            pass
        
        #  强制Python垃圾回收
        import gc
        gc.collect()
        logger.debug(f"  ✓ 执行垃圾回收")
        
        logger.info(f"✅ 样本 {sample_id} 资源清理完成")

    def _initialize_progress_tracking(self, samples_to_test: List[str]):
        """初始化进度跟踪"""
        progress_file = self.output_dir / "test_progress.json"
        
        progress_data = {
            "_summary": {
                "total_samples_planned": len(samples_to_test),
                "total_samples_completed": 0,
                "completion_rate": 0.0,
                "start_time": datetime.now().isoformat(),
                "sample_order": samples_to_test  #  记录测试顺序
            }
        }
        
        for sample_id in samples_to_test:
            progress_data[sample_id] = {
                "status": "pending",
                "test_count": 0,
                "success_count": 0,
                "failed_count": 0
            }
        
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📊 进度跟踪已初始化: {progress_file}")

    def _mark_sample_failed(self, sample_id: str, error: str):
        """标记样本测试失败"""
        progress_file = self.output_dir / "test_progress.json"
        
        if progress_file.exists():
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress_data = json.load(f)
            
            progress_data[sample_id] = {
                "status": "failed",
                "error": error,
                "failed_at": datetime.now().isoformat()
            }
            
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, ensure_ascii=False, indent=2)

    def _generate_final_summary_from_incremental(self):
        """从增量保存的结果生成最终汇总报告"""
        logger.info("📊 从增量结果生成最终汇总报告...")
        
        # 读取所有样本结果
        sample_files = sorted(self.output_dir.glob("sample_conv-*.json"))
        #  过滤掉 readable 文件
        sample_files = [f for f in sample_files if "_readable_" not in f.name]
        
        if not sample_files:
            logger.warning("没有找到增量保存的样本结果文件")
            return
        
        all_results = []
        sample_summaries = {}
        
        for sample_file in sample_files:
            try:
                with open(sample_file, 'r', encoding='utf-8') as f:
                    sample_data = json.load(f)
                    
                    #  安全获取 sample_id
                    sample_info = sample_data.get("sample_info", {})
                    sample_id = sample_info.get("sample_id", "unknown")
                    
                    # 获取结果列表
                    results = sample_data.get("results", [])
                    all_results.extend(results)
                    
                    #  收集样本级摘要 - 添加默认值处理
                    sample_summaries[sample_id] = {
                        "test_count": sample_info.get("test_count", 0),
                        "successful_count": sample_info.get("successful_count", 0),
                        "failed_count": sample_info.get("failed_count", 0),
                        "timestamp": sample_info.get("timestamp", "unknown")
                    }
                    
            except Exception as e:
                logger.warning(f"读取样本文件失败 {sample_file}: {e}")
                continue
        
        logger.info(f"✅ 从 {len(sample_files)} 个样本文件中加载了 {len(all_results)} 个结果")
        
        # 生成最终汇总JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_summary = {
            "test_info": {
                "total_samples": len(sample_files),
                "total_results": len(all_results),
                "successful_results": len([r for r in all_results if r.get("evaluation_success", False)]),
                "failed_results": len([r for r in all_results if not r.get("evaluation_success", False)]),
                "timestamp": datetime.now().isoformat(),
                "fusion_strategy": self.fusion_strategy,
                "fusion_weights": self.fusion_weights
            },
            "sample_summaries": sample_summaries,
            "sample_files": [str(f.name) for f in sample_files],
            "aggregate_results": all_results
        }
        
        summary_file = self.output_dir / f"final_summary_{timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(final_summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📊 最终汇总JSON已生成: {summary_file}")
        
        # 生成最终汇总可读性报告
        readable_summary_file = self._generate_final_readable_summary(final_summary, timestamp)
        logger.info(f"📄 最终汇总报告已生成: {readable_summary_file}")
        
        return summary_file, readable_summary_file

    def _generate_final_readable_summary(self, final_summary: Dict, timestamp: str) -> Path:
        """生成最终汇总可读性报告（包含并行性能统计）"""
        lines = []
        
        lines.append("=" * 100)
        lines.append("LoCoMo双塔召回系统 - 最终汇总报告")
        lines.append("=" * 100)
        lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 100)
        
        test_info = final_summary["test_info"]
        lines.append(f"\n📊 测试概况:")
        lines.append(f"   - 测试样本数: {test_info['total_samples']}")
        lines.append(f"   - 总测试数: {test_info['total_results']}")
        lines.append(f"   - 成功测试: {test_info['successful_results']} ({test_info['successful_results']/test_info['total_results']*100:.1f}%)")
        lines.append(f"   - 失败测试: {test_info['failed_results']} ({test_info['failed_results']/test_info['total_results']*100:.1f}%)")
        lines.append(f"   - 融合策略: {test_info['fusion_strategy']}")
        lines.append(f"   - 融合权重: 分层={test_info['fusion_weights']['hierarchical']}, 图={test_info['fusion_weights']['graph']}")
        
        # 样本级摘要
        sample_summaries = final_summary.get("sample_summaries", {})
        if sample_summaries:
            lines.append(f"\n📋 各样本测试摘要:")
            lines.append(f"\n{'样本ID':<15} {'测试数':>8} {'成功数':>8} {'失败数':>8} {'成功率':>10}")
            lines.append(f"{'-'*60}")
            
            for sample_id, summary in sorted(sample_summaries.items()):
                test_count = summary.get('test_count', 0) or 0
                success_count = summary.get('successful_count', 0) or 0
                failed_count = summary.get('failed_count', 0) or 0
                
                if test_count > 0:
                    success_rate = success_count / test_count * 100
                else:
                    success_rate = 0.0
                
                lines.append(f"{sample_id:<15} {test_count:>8} {success_count:>8} {failed_count:>8} {success_rate:>9.1f}%")
        
        # 整体性能统计
        all_results = final_summary.get("aggregate_results", [])
        valid_results = [r for r in all_results if r.get("evaluation_success", False)]
        
        if valid_results:
            def safe_get_score(result, key, default=0.0):
                scores = result.get("evaluation_scores", {})
                return scores.get(key, default) if scores else default
            
            avg_f1 = np.mean([safe_get_score(r, "token_f1") for r in valid_results])
            avg_semantic = np.mean([safe_get_score(r, "semantic_similarity") for r in valid_results])
            avg_llm = np.mean([safe_get_score(r, "llm_accuracy") for r in valid_results])
            avg_confidence = np.mean([r.get("confidence_score", 0) for r in valid_results])
            
            avg_hier_time = np.mean([r.get("hierarchical_time", 0) for r in valid_results])
            avg_graph_time = np.mean([r.get("graph_time", 0) for r in valid_results])
            avg_gen_time = np.mean([r.get("generation_time", 0) for r in valid_results])
            
            hier_success = sum(1 for r in valid_results if r.get("hierarchical_success", False))
            graph_success = sum(1 for r in valid_results if r.get("graph_results_count", 0) > 0)
            both_success = sum(1 for r in valid_results 
                            if r.get("hierarchical_success", False) and r.get("graph_results_count", 0) > 0)
            
            lines.append(f"\n🎯 整体性能指标:")
            lines.append(f"   - 平均F1分数: {avg_f1:.3f}")
            lines.append(f"   - 平均语义相似度: {avg_semantic:.3f}")
            lines.append(f"   - 平均LLM准确率: {avg_llm:.3f}")
            lines.append(f"   - 平均置信度: {avg_confidence:.3f}")
            
            lines.append(f"\n⏱️  平均时间性能:")
            lines.append(f"   - 分层检索: {avg_hier_time:.3f}s")
            lines.append(f"   - 图检索: {avg_graph_time:.3f}s")
            lines.append(f"   - 答案生成: {avg_gen_time:.3f}s")
            lines.append(f"   - 总计: {avg_hier_time + avg_graph_time + avg_gen_time:.3f}s")
            
            #  并行检索性能汇总
            parallel_actual_times = []
            parallel_sequential_times = []
            parallel_speedup_ratios = []
            parallel_time_saved_list = []
            
            for r in valid_results:
                if r.get("parallel_stats") and r["parallel_stats"]:
                    stats = r["parallel_stats"]
                    if stats.get("actual_time"):
                        parallel_actual_times.append(stats["actual_time"])
                        parallel_sequential_times.append(stats["sequential_theory"])
                        parallel_speedup_ratios.append(stats["speedup_ratio"])
                        parallel_time_saved_list.append(stats["time_saved"])
            
            if parallel_actual_times:
                lines.append(f"\n⚡ 并行检索性能汇总:")
                lines.append(f"   - 总测试数: {len(parallel_actual_times)}")
                lines.append(f"   - 平均并行时间: {np.mean(parallel_actual_times):.3f}s")
                lines.append(f"   - 平均理论串行时间: {np.mean(parallel_sequential_times):.3f}s")
                lines.append(f"   - 平均加速比: {np.mean(parallel_speedup_ratios):.2f}x")
                lines.append(f"   - 平均每次节省: {np.mean(parallel_time_saved_list):.3f}s")
                lines.append(f"   - 累计节省时间: {np.sum(parallel_time_saved_list):.2f}s ({np.sum(parallel_time_saved_list)/60:.1f}分钟)")
                lines.append(f"   - 加速比范围: {np.min(parallel_speedup_ratios):.2f}x ~ {np.max(parallel_speedup_ratios):.2f}x")
                
                # 效率提升
                total_parallel = np.sum(parallel_actual_times)
                total_sequential = np.sum(parallel_sequential_times)
                overall_efficiency = ((total_sequential - total_parallel) / total_sequential * 100) if total_sequential > 0 else 0
                lines.append(f"   - 整体效率提升: {overall_efficiency:.1f}%")
            
            lines.append(f"\n🔄 系统成功率:")
            lines.append(f"   - 分层检索: {hier_success}/{len(valid_results)} ({hier_success/len(valid_results)*100:.1f}%)")
            lines.append(f"   - 图检索: {graph_success}/{len(valid_results)} ({graph_success/len(valid_results)*100:.1f}%)")
            lines.append(f"   - 双塔同时成功: {both_success}/{len(valid_results)} ({both_success/len(valid_results)*100:.1f}%)")
            
            # 类别统计
            category_stats = defaultdict(lambda: {"count": 0, "f1_scores": [], "llm_scores": []})
            for r in valid_results:
                cat = r.get("category", 0)
                if cat:
                    category_stats[cat]["count"] += 1
                    category_stats[cat]["f1_scores"].append(safe_get_score(r, "token_f1"))
                    category_stats[cat]["llm_scores"].append(safe_get_score(r, "llm_accuracy"))
            
            if category_stats:
                lines.append(f"\n📋 类别性能:")
                category_names = {1: "多跳问题", 2: "时间问题", 3: "开放域问题", 4: "单跳问题", 5: "对抗性问题"}
                
                for cat, stats in sorted(category_stats.items()):
                    if stats["count"] == 0:
                        continue
                    cat_name = category_names.get(cat, f"类别{cat}")
                    avg_cat_f1 = np.mean(stats["f1_scores"]) if stats["f1_scores"] else 0.0
                    avg_cat_llm = np.mean(stats["llm_scores"]) if stats["llm_scores"] else 0.0
                    
                    lines.append(f"\n   {cat_name} ({stats['count']}题):")
                    lines.append(f"     - 平均F1: {avg_cat_f1:.3f}")
                    lines.append(f"     - 平均LLM准确率: {avg_cat_llm:.3f}")
        
        lines.append(f"\n{'='*100}")
        lines.append(f"报告生成完成")
        lines.append(f"详细的样本报告请查看: sample_*_readable_*.txt")
        lines.append(f"{'='*100}")
        
        # 保存报告
        summary_file = self.output_dir / f"final_summary_readable_{timestamp}.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        # 打印到控制台
        print('\n'.join(lines))
        
        return summary_file

    # def _generate_final_summary_from_incremental(self):
    #     """从增量保存的结果生成最终汇总报告"""
    #     logger.info("📊 从增量结果生成最终汇总报告...")
        
    #     # 读取所有样本结果
    #     sample_files = sorted(self.output_dir.glob("sample_*.json"))
        
    #     if not sample_files:
    #         logger.warning("没有找到增量保存的样本结果文件")
    #         return
        
    #     all_results = []
    #     for sample_file in sample_files:
    #         try:
    #             with open(sample_file, 'r', encoding='utf-8') as f:
    #                 sample_data = json.load(f)
    #                 all_results.extend(sample_data.get("results", []))
    #         except Exception as e:
    #             logger.warning(f"读取样本文件失败 {sample_file}: {e}")
        
    #     # 重建test_results用于报告生成
    #     # （注意：这里只是为了生成报告，实际结果已经增量保存）
    #     logger.info(f"✅ 从 {len(sample_files)} 个样本文件中加载了 {len(all_results)} 个结果")
        
    #     # 生成最终报告
    #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     final_summary = {
    #         "test_info": {
    #             "total_samples": len(sample_files),
    #             "total_results": len(all_results),
    #             "successful_results": len([r for r in all_results if r.get("evaluation_success", False)]),
    #             "failed_results": len([r for r in all_results if not r.get("evaluation_success", False)]),
    #             "timestamp": datetime.now().isoformat()
    #         },
    #         "sample_files": [str(f.name) for f in sample_files],
    #         "aggregate_results": all_results
    #     }
        
    #     summary_file = self.output_dir / f"final_summary_{timestamp}.json"
    #     with open(summary_file, 'w', encoding='utf-8') as f:
    #         json.dump(final_summary, f, ensure_ascii=False, indent=2)
        
    #     logger.info(f"📊 最终汇总报告已生成: {summary_file}")

    def _update_progress_file(self, sample_id: str, test_count: int, success_count: int):
        """
        更新测试进度文件
        
        Args:
            sample_id: 样本ID
            test_count: 测试数量
            success_count: 成功数量
        """
        progress_file = self.output_dir / "test_progress.json"
        
        # 读取现有进度
        progress_data = {}
        if progress_file.exists():
            try:
                with open(progress_file, 'r', encoding='utf-8') as f:
                    progress_data = json.load(f)
            except:
                progress_data = {}
        
        # 更新进度
        progress_data[sample_id] = {
            "status": "completed",
            "test_count": test_count,
            "success_count": success_count,
            "failed_count": test_count - success_count,
            "completed_at": datetime.now().isoformat()
        }
        
        # 统计总体进度
        total_completed = len([s for s, info in progress_data.items() if info.get("status") == "completed"])
        if hasattr(self, 'target_sample_ids') and self.target_sample_ids:
            total_planned = len(self.target_sample_ids)
        else:
            total_planned = len(progress_data)
        
        progress_data["_summary"] = {
            "total_samples_planned": total_planned,
            "total_samples_completed": total_completed,
            "completion_rate": total_completed / total_planned if total_planned > 0 else 0,
            "last_updated": datetime.now().isoformat()
        }
        
        # 保存进度
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📊 进度更新: {total_completed}/{total_planned} 样本完成 ({total_completed/total_planned*100:.1f}%)")
    
    def _run_single_threaded_dual_tower_tests(self):
        """单线程执行双塔测试"""
        for test_case in tqdm(self.test_cases, desc="执行双塔测试"):
            try:
                result = self._run_single_dual_tower_test(test_case)
                if result:
                    self.test_results.append(result)
                    if result.evaluation_success:
                        self.stats["successful_dual_tower"] += 1
                    else:
                        self.stats["failed_retrievals"] += 1
                else:
                    self.stats["failed_retrievals"] += 1
                    
            except Exception as e:
                self.stats["failed_retrievals"] += 1
                logger.error(f"❌ 双塔测试失败: {test_case['sample_id']} - {e}")
                continue
    
    def _run_multi_threaded_dual_tower_tests(self):
        """多线程执行双塔测试"""
        logger.info(f"使用 {self.max_workers} 个线程执行双塔测试")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_test_case = {
                executor.submit(self._run_single_dual_tower_test, test_case): test_case
                for test_case in self.test_cases
            }
            
            for future in tqdm(as_completed(future_to_test_case), total=len(self.test_cases), desc="执行双塔测试"):
                test_case = future_to_test_case[future]
                try:
                    result = future.result()
                    if result:
                        self.test_results.append(result)
                        if result.evaluation_success:
                            self.stats["successful_dual_tower"] += 1
                        else:
                            self.stats["failed_retrievals"] += 1
                    else:
                        self.stats["failed_retrievals"] += 1
                        
                except Exception as e:
                    self.stats["failed_retrievals"] += 1
                    logger.error(f"❌ 双塔测试失败: {test_case['sample_id']} - {e}")
                    continue
    
    def _run_single_dual_tower_test(self, test_case: Dict[str, Any]) -> Optional[DualTowerRetrievalResult]:
        """
        运行单个双塔测试（支持并行检索）
        
        Args:
            test_case: 测试用例
            
        Returns:
            测试结果或None
        """
        sample_id = test_case["sample_id"]
        question = test_case["question"]
        category = test_case["category"]
        expected_answer = test_case["expected_answer"]
        
        logger.debug(f"🔄 双塔测试: {sample_id} - {question[:50]}...")
        
        try:
            if self.parallel_towers:
                #  并行模式：同时执行两个塔的检索
                return self._run_parallel_dual_tower_test(
                    sample_id, question, category, expected_answer
                )
            else:
                # 串行模式：保持原有逻辑
                return self._run_sequential_dual_tower_test(
                    sample_id, question, category, expected_answer
                )
                
        except Exception as e:
            logger.error(f"双塔测试执行失败 {sample_id}: {e}")
            logger.debug(traceback.format_exc())
            return None
        
    def _run_parallel_dual_tower_test(self, 
                                sample_id: str, 
                                question: str, 
                                category: int, 
                                expected_answer: str) -> Optional[DualTowerRetrievalResult]:
        """并行执行双塔检索（核心优化）
        
        Args:
            sample_id: 样本ID
            question: 问题
            category: 类别
            expected_answer: 预期答案
            
        Returns:
            测试结果或None
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import time
        
        # 用于存储结果的容器
        hierarchical_result = {}
        graph_result = {}
        
        def run_hierarchical():
            """执行分层检索"""
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
                    "hierarchical_context_text": "",
                    "retrieval_method": "failed"
                }
                hierarchical_result['time'] = 0.0
                hierarchical_result['success'] = False
        
        def run_graph():
            """执行图检索"""
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
        
        #  使用线程池并行执行两个检索任务
        parallel_start = time.time()  #  记录并行开始时间
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            # 提交两个任务
            future_hierarchical = executor.submit(run_hierarchical)
            future_graph = executor.submit(run_graph)
            
            # 等待两个任务完成
            for future in as_completed([future_hierarchical, future_graph]):
                try:
                    future.result()  # 获取结果，如果有异常会抛出
                except Exception as e:
                    logger.error(f"并行检索任务异常: {e}")
        
        parallel_elapsed = time.time() - parallel_start  #  并行总耗时
        
        # 检查是否两个检索都完成
        if 'context' not in hierarchical_result or 'units' not in graph_result:
            logger.error(f"并行检索未完成: hierarchical={bool(hierarchical_result)}, graph={bool(graph_result)}")
            return None
        
        # 提取结果
        hierarchical_context = hierarchical_result['context']
        hierarchical_time = hierarchical_result['time']
        
        graph_units = graph_result['units']
        graph_details = graph_result['details']
        graph_time = graph_result['time']
        
        #  计算串行理论时间和加速比
        sequential_time = hierarchical_time + graph_time
        speedup_ratio = sequential_time / parallel_elapsed if parallel_elapsed > 0 else 0
        time_saved = sequential_time - parallel_elapsed
        
        logger.debug(f"⚡ 并行检索完成: 分层={hierarchical_time:.3f}s, 图={graph_time:.3f}s")
        logger.debug(f"   并行总耗时={parallel_elapsed:.3f}s, 理论串行={sequential_time:.3f}s")
        logger.debug(f"   加速比={speedup_ratio:.2f}x, 节省时间={time_saved:.3f}s")
        
        # 3. 双塔融合生成答案
        fusion_start = time.time()
        answer_dict, confidence_score = self._fuse_and_generate_answer(
            question, category, hierarchical_context, graph_units, graph_details
        )
        fusion_time = time.time() - fusion_start
        
        # 提取最终答案和推理过程
        final_answer = answer_dict.get("final_answer", "")
        reasoning_process = answer_dict.get("reasoning", "")
        
        # 4. 修改：传递reasoning到评估
        evaluation_result = self._evaluate_dual_tower_result(
            question=question,
            expected_answer=expected_answer,
            generated_answer=final_answer,
            reasoning=reasoning_process,  # 传递推理过程
            category=category
        )
        
        evaluation_scores = evaluation_result.get("evaluation_scores", {})
        evaluation_success = evaluation_result.get("evaluation_success", False)
        
        # 5. 构建完整结果
        result = DualTowerRetrievalResult(
            sample_id=sample_id,
            question=question,
            category=category,
            expected_answer=expected_answer,
            hierarchical_context=hierarchical_context,
            hierarchical_retrieval_time=hierarchical_time,
            graph_retrieved_units=graph_units,
            graph_retrieval_time=graph_time,
            graph_retrieval_details=graph_details,
            final_answer=final_answer,
            reasoning_process=reasoning_process,
            confidence_score=confidence_score,
            fusion_method=self.fusion_strategy,
            generation_time=fusion_time,
            evaluation_scores=evaluation_scores,
            evaluation_success=evaluation_success
        )
        
        #  在结果对象上附加并行时间统计（用于后续分析）
        result.parallel_actual_time = parallel_elapsed
        result.parallel_sequential_theory = sequential_time
        result.parallel_speedup_ratio = speedup_ratio
        result.parallel_time_saved = time_saved
        
        return result
    
    def _run_sequential_dual_tower_test(self, 
                                sample_id: str, 
                                question: str, 
                                category: int, 
                                expected_answer: str) -> Optional[DualTowerRetrievalResult]:
        """
        串行执行双塔检索（原有逻辑，用于对比和调试）

        Args:
            sample_id: 样本ID
            question: 问题
            category: 类别
            expected_answer: 预期答案
            
        Returns:
            测试结果或None
        """
        # 1. 分层检索
        hierarchical_start = time.time()
        hierarchical_context = self._run_hierarchical_retrieval(sample_id, question, category)
        hierarchical_time = time.time() - hierarchical_start

        hierarchical_success = hierarchical_context.get("hierarchical_enabled", False)
        if hierarchical_success:
            self.stats["successful_hierarchical"] += 1

        # 2. 知识图谱检索
        graph_start = time.time()
        graph_results, graph_details = self._run_graph_retrieval(sample_id, question)
        graph_time = time.time() - graph_start

        graph_success = len(graph_results) > 0
        if graph_success:
            self.stats["successful_graph"] += 1

        logger.debug(f"🐌 串行检索完成: 分层={hierarchical_time:.3f}s, 图={graph_time:.3f}s, "
                    f"总耗时={hierarchical_time + graph_time:.3f}s")

        # 3. 双塔融合生成答案
        fusion_start = time.time()
        answer_dict, confidence_score = self._fuse_and_generate_answer(
            question, category, hierarchical_context, graph_results, graph_details
        )
        fusion_time = time.time() - fusion_start

        # 提取最终答案和推理过程
        final_answer = answer_dict.get("final_answer", "")
        reasoning_process = answer_dict.get("reasoning", "")

        # 4. 修复：传递reasoning到评估
        evaluation_result = self._evaluate_dual_tower_result(
            question=question,
            expected_answer=expected_answer,
            generated_answer=final_answer,
            reasoning=reasoning_process,  # 传递推理过程
            category=category
        )

        evaluation_scores = evaluation_result.get("evaluation_scores", {})
        evaluation_success = evaluation_result.get("evaluation_success", False)

        # 5. 构建完整结果
        return DualTowerRetrievalResult(
            sample_id=sample_id,
            question=question,
            category=category,
            expected_answer=expected_answer,
            hierarchical_context=hierarchical_context,
            hierarchical_retrieval_time=hierarchical_time,
            graph_retrieved_units=graph_results,
            graph_retrieval_time=graph_time,
            graph_retrieval_details=graph_details,
            final_answer=final_answer,
            reasoning_process=reasoning_process,
            confidence_score=confidence_score,
            fusion_method=self.fusion_strategy,
            generation_time=fusion_time,
            evaluation_scores=evaluation_scores,
            evaluation_success=evaluation_success
        )
    
    def _run_hierarchical_retrieval(self, sample_id: str, question: str, category: int) -> Dict[str, Any]:
        """运行分层检索（使用缓存的接口）"""
        try:
            # 使用缓存的分层接口
            if sample_id not in self.hierarchical_interfaces:
                logger.warning(f"分层接口未预加载: {sample_id}，尝试动态加载...")
                graph, hierarchical_interface = self.hierarchical_tester.load_enhanced_conversation_graph(
                    sample_id, str(self.enhanced_graphs_dir)
                )
                self.hierarchical_interfaces[sample_id] = hierarchical_interface
                self.loaded_graphs[sample_id] = graph
            
            hierarchical_interface = self.hierarchical_interfaces[sample_id]
            
            # 获取分层上下文
            context_info = self.hierarchical_tester.retrieve_hierarchical_context(
                hierarchical_interface, question, sample_id, category,
                enable_graph_expansion=False,  # 在双塔模式下禁用
                graph_expansion_hops=0
            )
            
            return context_info
            
        except Exception as e:
            logger.warning(f"分层检索失败 {sample_id}: {e}")
            return {
                "hierarchical_enabled": False,
                "error": str(e),
                "hierarchical_context_text": "",
                "retrieval_method": "failed"
            }
    
    def _run_graph_retrieval(self, sample_id: str, question: str) -> Tuple[List[Any], Dict[str, Any]]:
        """运行知识图谱检索（使用已加载的检索器）"""
        try:
            # 图检索器应该已经在 graph_benchmark.load_semantic_graphs() 中加载
            if sample_id not in self.graph_benchmark.multi_retrievers:
                logger.warning(f"图检索器不存在: {sample_id}")
                return [], {"method": "failed", "error": "retriever_not_found"}
            
            # 直接使用已加载的检索器
            multi_retriever = self.graph_benchmark.multi_retrievers[sample_id]
            entity_retriever = self.graph_benchmark.entity_relation_retrievers.get(sample_id)
            
            retrieved_units = []
            details = {}
            
            if self.use_entity_relation and entity_retriever:
                # 混合检索：语义 + 实体关系
                # semantic_results = multi_retriever.smart_search(
                #     query=question,
                #     methods=["bm25", "cosine_similarity", "splade"],
                #     fusion_method="rrf",
                #     rerank_method="baai",
                #     top_k=self.topk_similarity,
                #     return_detailed=False
                # )
                semantic_results = multi_retriever.smart_search(
                    query=question,
                    methods=["bm25", "cosine_similarity", "splade"],
                    fusion_method="rrf",
                    rerank_method=self.reranker_type,  # 使用配置的重排序器类型
                    top_k=self.topk_similarity,
                    return_detailed=False
                )
                                
                entity_results = entity_retriever.search(question, self.topk_graph)
                graph_results = [(r.unit, r.score) for r in entity_results]
                
                retrieved_units = semantic_results + graph_results
                details = {
                    "method": "hybrid_semantic_graph",
                    "semantic_count": len(semantic_results),
                    "graph_count": len(graph_results),
                    "topk_similarity": self.topk_similarity,
                    "topk_graph": self.topk_graph
                }
            else:
                # 纯语义检索
                retrieved_units = multi_retriever.smart_search(
                    query=question,
                    methods=["bm25", "cosine_similarity", "splade"],
                    fusion_method="rrf",
                    rerank_method=self.reranker_type,  # 使用配置的重排序器类型
                    top_k=self.topk_similarity,
                    return_detailed=False
                )
                # retrieved_units = multi_retriever.smart_search(
                #     query=question,
                #     methods=["bm25", "cosine_similarity", "splade"],
                #     fusion_method="rrf",
                #     rerank_method="baai",
                #     top_k=self.topk_similarity,
                #     return_detailed=False
                # )
                details = {
                    "method": "semantic_only",
                    "semantic_count": len(retrieved_units),
                    "graph_count": 0
                }
            
            return retrieved_units, details
            
        except Exception as e:
            logger.warning(f"图检索失败 {sample_id}: {e}")
            return [], {"method": "failed", "error": str(e)}
    
    def _fuse_and_generate_answer(self, 
                             question: str, 
                             category: int,
                             hierarchical_context: Dict[str, Any], 
                             graph_results: List[Any], 
                             graph_details: Dict[str, Any]) -> Tuple[Dict[str, str], float]:
        """
        融合两个塔的结果并生成最终答案
        
        Returns:
            Tuple[Dict[str, str], float]: (包含reasoning和final_answer的字典, 置信度分数)
        """
        
        if self.fusion_strategy == "simple":
            return self._simple_fusion_generation(question, category, hierarchical_context, graph_results)
        elif self.fusion_strategy == "weighted":
            return self._weighted_fusion_generation(question, category, hierarchical_context, graph_results)
        elif self.fusion_strategy == "context_aware":
            return self._context_aware_fusion_generation(question, category, hierarchical_context, graph_results, graph_details)
        else:
            raise ValueError(f"未知的融合策略: {self.fusion_strategy}")
    
    def _context_aware_fusion_generation(self, 
                           question: str, 
                           category: int,
                           hierarchical_context: Dict[str, Any], 
                           graph_results: List[Any],
                           graph_details: Dict[str, Any]) -> Tuple[Dict[str, str], float]:
        """上下文感知的融合生成（推荐方法） - 返回结构化答案 - 优化版：删除冗余信息"""
        
        # 构建双塔融合提示词
        prompt_parts = []
        
        # 对抗性问题特殊指导
        if category == 5:
            prompt_parts.append("You are an expert conversation analyst specialized in detecting misleading or unanswerable questions.")
        else:
            prompt_parts.append("You are an expert conversation analyst with access to two complementary information retrieval systems.")
            
        prompt_parts.append("")
        prompt_parts.append("IMPORTANT: These are two DIFFERENT retrieval systems providing COMPLEMENTARY information:")
        prompt_parts.append("1. HIERARCHICAL MEMORY: Provides structured, multi-layer conversational context")
        prompt_parts.append("2. KNOWLEDGE GRAPH: Provides specific facts and entity relationships")
        prompt_parts.append("")
        prompt_parts.append("Your task is to synthesize information from BOTH systems to provide the most accurate and complete answer.")
        prompt_parts.append("")
        
        # 问题和类别指导
        category_guidance = self._get_dual_tower_category_guidance(category)
        prompt_parts.append(f"QUESTION: {question}")
        prompt_parts.append(f"QUESTION CATEGORY: {category} - {category_guidance}")
        prompt_parts.append("")
        
        # Hierarchical retrieval results
        hierarchical_enabled = hierarchical_context.get("hierarchical_enabled", False)
        if hierarchical_enabled:
            prompt_parts.append("=" * 80)
            prompt_parts.append("HIERARCHICAL MEMORY RESULTS")
            prompt_parts.append("=" * 80)
            prompt_parts.append(hierarchical_context.get("hierarchical_context_text", "No hierarchical context available"))
        else:
            prompt_parts.append("=" * 80)
            prompt_parts.append("HIERARCHICAL MEMORY RESULTS")
            prompt_parts.append("=" * 80)
            prompt_parts.append("Hierarchical retrieval was not available for this query.")
        
        prompt_parts.append("")

        # Knowledge-graph retrieval results
        if graph_results:
            prompt_parts.append("=" * 80)
            prompt_parts.append("KNOWLEDGE GRAPH RESULTS")
            prompt_parts.append("=" * 80)
            prompt_parts.append(f"Retrieved {len(graph_results)} relevant knowledge graph units:")
            prompt_parts.append("")
            
            for i, (unit, score) in enumerate(graph_results, 1):
                unit_content = self._extract_graph_unit_content(unit)
                prompt_parts.append(f"Graph Result {i}:")
                
                # 修复：直接使用unit_content，不再添加"Entity:"前缀
                # 因为_extract_graph_unit_content已经返回了完整格式的内容
                prompt_parts.append(f"{unit_content[:200]}")
                prompt_parts.append("")
            
        else:
            prompt_parts.append("=" * 80)
            prompt_parts.append("KNOWLEDGE GRAPH RESULTS")
            prompt_parts.append("=" * 80)
            prompt_parts.append("No relevant entities or relationships found in the knowledge graph.")
        
        prompt_parts.append("")
        
        # # Knowledge-graph retrieval results
        # if graph_results:
        #     prompt_parts.append("=" * 80)
        #     prompt_parts.append("KNOWLEDGE GRAPH RESULTS")
        #     prompt_parts.append("=" * 80)
        #     prompt_parts.append(f"Retrieved {len(graph_results)} relevant knowledge graph units:")
        #     # 删除：方法显示
        #     # prompt_parts.append(f"Method: {graph_details.get('retrieval_method', 'unknown')}")
        #     prompt_parts.append("")
            
        #     for i, (unit, score) in enumerate(graph_results[:10], 1):
        #         unit_content = self._extract_graph_unit_content(unit)
        #         # 删除：Score显示
        #         # prompt_parts.append(f"Graph Result {i} (Score: {score:.3f}):")
        #         prompt_parts.append(f"Graph Result {i}:")
                
        #         # 提取实体类型和上下文
        #         entity_type = getattr(unit, 'entity_type', 'Unknown')
        #         if entity_type != 'Unknown':
        #             prompt_parts.append(f"Entity: {unit_content[:200]} | Type: {entity_type}")
        #         else:
        #             prompt_parts.append(f"Entity: {unit_content[:200]}")
        #         prompt_parts.append("")
            
        #     # 删除：图检索详情（实体匹配、关系发现等）
        #     # if graph_details.get("entity_extraction"):
        #     #     entities = graph_details["entity_extraction"]
        #     #     prompt_parts.append(f"查询: {question}")
        #     #     prompt_parts.append(f"抽取的实体:")
        #     #     ...
            
        # else:
        #     prompt_parts.append("=" * 80)
        #     prompt_parts.append("KNOWLEDGE GRAPH RESULTS")
        #     prompt_parts.append("=" * 80)
        #     prompt_parts.append("No relevant entities or relationships found in the knowledge graph.")
        
        # prompt_parts.append("")
        
        # 融合指导（对抗性问题加强）
        if category == 5:
            prompt_parts.append("=" * 80)
            prompt_parts.append("DUAL TOWER FUSION GUIDANCE")
            prompt_parts.append("=" * 80)
            # 删除：融合策略和权重
            # prompt_parts.append(f"FUSION STRATEGY: {self.fusion_strategy}")
            # prompt_parts.append(f"HIERARCHICAL WEIGHT: {self.fusion_weights['hierarchical']}")
            # prompt_parts.append(f"KNOWLEDGE GRAPH WEIGHT: {self.fusion_weights['graph']}")
            # prompt_parts.append("")
            
            prompt_parts.append("SYNTHESIS INSTRUCTIONS:")
            prompt_parts.append("1. If BOTH systems provided information: Cross-validate and synthesize")
            prompt_parts.append("2. If ONLY one system worked: Rely on available system's information")
            prompt_parts.append("3. ADVERSARIAL QUESTION: Strictly verify entity existence in BOTH systems")
            prompt_parts.append("4. If information is contradictory or not found: State 'No information available'")
            prompt_parts.append("")
            prompt_parts.append("RESPONSE FORMAT (REQUIRED JSON):")
            prompt_parts.append("{")
            prompt_parts.append('    "reasoning": "Your synthesis process...",')
            prompt_parts.append('    "final_answer": "Your direct, concise final answer"')
            prompt_parts.append("}")
        else:
            prompt_parts.append("=" * 80)
            prompt_parts.append("DUAL TOWER FUSION GUIDANCE")
            prompt_parts.append("=" * 80)
            # 删除：融合策略和权重
            # prompt_parts.append(f"FUSION STRATEGY: {self.fusion_strategy}")
            # prompt_parts.append(f"HIERARCHICAL WEIGHT: {self.fusion_weights['hierarchical']}")
            # prompt_parts.append(f"KNOWLEDGE GRAPH WEIGHT: {self.fusion_weights['graph']}")
            # prompt_parts.append("")
            
            prompt_parts.append("SYNTHESIS INSTRUCTIONS:")
            prompt_parts.append("1. If BOTH systems provided information: Cross-validate and synthesize")
            prompt_parts.append("2. If ONLY one system worked: Rely on available system's information")
            prompt_parts.append("")
            prompt_parts.append("RESPONSE FORMAT (REQUIRED JSON):")
            prompt_parts.append("{")
            prompt_parts.append('    "reasoning": "Your synthesis process...",')
            prompt_parts.append('    "final_answer": "Your direct, concise final answer"')
            prompt_parts.append("}")
        
        full_prompt = "\n".join(prompt_parts)
        
        try:
            # 生成结构化答案
            raw_response = self.llm_client.generate_answer(
                prompt=full_prompt,
                temperature=0.1,
                max_tokens=2000,
                json_format=True
            )
            
            # 解析JSON响应
            answer_dict = self._parse_structured_dual_tower_response(raw_response, category)
            
            # 计算置信度（内部使用，不对外暴露）
            confidence_score = self._calculate_dual_tower_confidence(
                hierarchical_enabled, len(graph_results), graph_details
            )
            
            return answer_dict, confidence_score
            
        except Exception as e:
            logger.error(f"双塔融合生成失败: {e}")
            return {
                "reasoning": f"Generation failed: {str(e)}",
                "final_answer": "Unable to generate answer"
            }, 0.0

    def _parse_structured_dual_tower_response(self, raw_response: str, category: int) -> Dict[str, str]:
        """解析双塔结构化响应"""
        try:
            # 尝试解析JSON
            import json
            parsed = json.loads(raw_response.strip())
            
            if isinstance(parsed, dict) and "reasoning" in parsed and "final_answer" in parsed:
                # 后处理最终答案
                final_answer = self._post_process_dual_tower_answer(parsed["final_answer"], category)
                
                return {
                    "reasoning": parsed["reasoning"].strip(),
                    "final_answer": final_answer
                }
            else:
                raise ValueError("JSON格式不正确：缺少必需字段")
                
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"JSON解析失败: {e}，尝试文本解析")
            
            # 降级到文本解析
            return self._parse_text_dual_tower_response(raw_response, category)

    def _parse_text_dual_tower_response(self, raw_response: str, category: int) -> Dict[str, str]:
        """降级文本解析方法 - 双塔版本"""
        lines = raw_response.strip().split('\n')
        reasoning = ""
        final_answer = ""
        
        # 寻找关键词模式
        reasoning_keywords = ["reasoning", "analysis", "思考", "推理", "synthesis", "because", "since"]
        answer_keywords = ["answer", "final", "conclusion", "result", "答案", "结论"]
        
        current_section = "reasoning"
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 检查是否切换到答案部分
            if any(keyword in line.lower() for keyword in answer_keywords):
                current_section = "answer"
                if ":" in line:
                    final_answer = line.split(":", 1)[1].strip()
                    continue
            
            # 分配到相应部分
            if current_section == "reasoning":
                reasoning += line + " "
            else:
                final_answer += line + " "
        
        # 如果没有明确分割，使用最后一句作为最终答案
        if not final_answer.strip() and reasoning.strip():
            sentences = reasoning.strip().split('.')
            if len(sentences) > 1:
                final_answer = sentences[-1].strip()
                reasoning = '.'.join(sentences[:-1]).strip()
        
        # 最后的清理和后处理
        if not final_answer.strip():
            final_answer = raw_response.strip()
            reasoning = "Unable to parse structured reasoning from dual tower response"
        
        final_answer = self._post_process_dual_tower_answer(final_answer.strip(), category)
        
        return {
            "reasoning": reasoning.strip() or "No clear reasoning provided from dual tower fusion",
            "final_answer": final_answer
        }

    def _post_process_dual_tower_answer(self, answer: str, category: int) -> str:
        """后处理双塔生成的答案"""
        if not answer:
            return "No answer generated"
        
        answer = answer.strip()
        
        # 移除常见前缀
        prefixes = ["Answer:", "ANSWER:", "Final Answer:", "Response:", "Based on"]
        for prefix in prefixes:
            if answer.startswith(prefix):
                answer = answer[len(prefix):].strip()
        
        # 确保首字母大写
        if answer and not answer[0].isupper() and not answer[0].isdigit():
            answer = answer[0].upper() + answer[1:]
        
        # 对抗性问题的标准化
        if category == 5:
            lower_answer = answer.lower()
            if any(phrase in lower_answer for phrase in ["no information", "not available", "not mentioned", "not found", "insufficient information"]):
                return "No information available"
        
        return answer
    
    def _simple_fusion_generation(self, 
                             question: str, 
                             category: int,
                             hierarchical_context: Dict[str, Any], 
                             graph_results: List[Any]) -> Tuple[Dict[str, str], float]:
        """简单融合生成 - 返回结构化结果"""
        context_parts = []
        
        # 添加分层上下文
        if hierarchical_context.get("hierarchical_enabled", False):
            hierarchical_text = hierarchical_context.get("hierarchical_context_text", "")
            if hierarchical_text:
                context_parts.append(f"Hierarchical Context:\n{hierarchical_text}")
        
        # 添加图检索结果
        if graph_results:
            graph_context = []
            for i, (unit, score) in enumerate(graph_results, 1):
                if hasattr(unit, 'raw_data') and unit.raw_data:
                    text_content = unit.raw_data.get('text_content', str(unit.raw_data))
                    graph_context.append(f"{i}. {text_content[:200]}...")
            
            if graph_context:
                context_parts.append(f"Graph Knowledge:\n" + "\n".join(graph_context))
        
        if not context_parts:
            return {
                "reasoning": "No information from either retrieval system",
                "final_answer": "No relevant information found from either retrieval system."
            }, 0.0
        
        combined_context = "\n\n".join(context_parts)
        
        prompt = f"""Based on the following information from two retrieval systems, provide your answer in JSON format:

        {combined_context}

        Question: {question}
        
        Provide your response as JSON:
        {{
            "reasoning": "Your reasoning process",
            "final_answer": "Your direct answer"
        }}"""
                
        try:
            raw_response = self.llm_client.generate_answer(
                prompt=prompt, 
                temperature=0.1, 
                max_tokens=1500,
                json_format=True
            )
            parsed = self._parse_structured_dual_tower_response(raw_response, category)
            confidence = 0.7 if len(context_parts) == 2 else 0.5
            return parsed, confidence
        except Exception as e:
            return {
                "reasoning": f"Simple fusion generation failed: {str(e)}",
                "final_answer": "Generation failed"
            }, 0.0

    def _weighted_fusion_generation(self, 
                                question: str, 
                                category: int,
                                hierarchical_context: Dict[str, Any], 
                                graph_results: List[Any]) -> Tuple[Dict[str, str], float]:
        """加权融合生成 - 返回结构化结果"""
        # 当前简化实现，调用simple版本
        return self._simple_fusion_generation(question, category, hierarchical_context, graph_results)
    
    def _get_dual_tower_category_guidance(self, category: int) -> str:
        """获取双塔系统的类别指导"""
        guidance_map = {
            1: "Multi-hop reasoning - Use hierarchical patterns AND graph relationships to trace connections",
            2: "Temporal question - Use hierarchical session timing AND graph temporal entities",
            3: "Open-domain question - Synthesize comprehensive view from hierarchical insights AND graph facts",
            4: "Single-hop fact - Verify fact in hierarchical context AND confirm with graph evidence",
            5: "Adversarial question - Check information existence in BOTH systems before answering"
        }
        return guidance_map.get(category, "General question - Use both hierarchical and graph information")
    
    def _calculate_dual_tower_confidence(self, 
                                        hierarchical_success: bool, 
                                        graph_results_count: int, 
                                        graph_details: Dict[str, Any]) -> float:
        """计算双塔系统的置信度"""
        base_confidence = 0.0
        
        # 分层检索贡献
        if hierarchical_success:
            base_confidence += self.fusion_weights.get("hierarchical", 0.6)
        
        # 图检索贡献
        if graph_results_count > 0:
            graph_weight = self.fusion_weights.get("graph", 0.4)
            # 根据结果数量调整
            result_factor = min(graph_results_count / 10.0, 1.0)  # 最多10个结果为满分
            base_confidence += graph_weight * result_factor
        
        return min(base_confidence, 1.0)
    
    def _evaluate_dual_tower_result(self, 
                              question: str, 
                              expected_answer: str, 
                              generated_answer: str, 
                              reasoning: str,  # 必需参数
                              category: int) -> Dict[str, Any]:
        """评估双塔结果 - 统一处理所有类别"""
        try:
            from benchmark_locomo.task_eval.evaluation import calculate_comprehensive_scores
            
            # 统一调用
            eval_result = calculate_comprehensive_scores(
                gold_answer=expected_answer,
                response=generated_answer,
                question=question,
                reasoning=reasoning,  # 传递推理过程
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
            logger.error(f"双塔评估失败: {e}")
            return {
                "evaluation_scores": {"error": str(e)},
                "evaluation_method": "failed",
                "evaluation_success": False
            }
        
    def generate_dual_tower_report(self) -> Dict[str, Path]:
        """生成双塔benchmark报告
        
        Returns:
            Dict[str, Path]: 生成的文件路径字典
        """
        logger.info("生成双塔benchmark报告...")

        if not self.test_results:
            logger.warning("没有测试结果，无法生成报告")
            return {}
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 计算统计数据
        overall_stats = self._calculate_dual_tower_overall_stats()
        sample_performance = self._calculate_dual_tower_sample_performance()
        category_performance = self._calculate_dual_tower_category_performance()
        fusion_analysis = self._analyze_fusion_effectiveness()
        
        # 构建完整报告
        benchmark_report = {
            "benchmark_info": {
                "test_name": "LoCoMo Dual Tower Retrieval Benchmark",
                "test_type": "dual_tower_retrieval",
                "timestamp": datetime.now().isoformat(),
                "fusion_strategy": self.fusion_strategy,
                "fusion_weights": self.fusion_weights,
                "total_samples": self.stats["total_samples_loaded"],
                "total_test_cases": self.stats["total_test_cases"],
                "total_tests_run": len(self.test_results),
                "hierarchical_config": {
                    "l0_top_k": self.topk_hierarchical_l0,
                    "l1_top_k": self.topk_hierarchical_l1,
                    "l2_top_k": self.topk_hierarchical_l2
                },
                "graph_config": {
                    "topk_similarity": self.topk_similarity,
                    "topk_graph": self.topk_graph,
                    "use_entity_relation": self.use_entity_relation
                }
            },
            "overall_statistics": overall_stats,
            "sample_performance": sample_performance,
            "category_performance": category_performance,
            "fusion_analysis": fusion_analysis,
            "detailed_results": [
            {
                "sample_id": r.sample_id,
                "question": r.question,
                "category": r.category,
                "expected_answer": r.expected_answer,
                "final_answer": r.final_answer,
                "reasoning_process": r.reasoning_process,  # 新增
                "confidence_score": r.confidence_score,
                "hierarchical_success": r.hierarchical_context.get("hierarchical_enabled", False),
                "graph_results_count": len(r.graph_retrieved_units),
                "hierarchical_time": r.hierarchical_retrieval_time,
                "graph_time": r.graph_retrieval_time,
                "generation_time": r.generation_time,
                "evaluation_scores": r.evaluation_scores,
                "evaluation_success": r.evaluation_success
            }
            for r in self.test_results
        ]
        }
        
        # 保存JSON报告
        report_file = self.output_dir / f"dual_tower_benchmark_report_{timestamp}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(benchmark_report, f, ensure_ascii=False, indent=2)
        
        # 生成可读性报告
        readable_report_file = self._generate_dual_tower_readable_report(benchmark_report, timestamp)
        
        logger.info(f"双塔benchmark报告已生成: {report_file}")

        # 2. 生成检索详情报告
        retrieval_files = self.generate_retrieval_details_report()

        logger.info(f"双塔benchmark检索报告已经生成")
        logger.info(f"双塔benchmark完整报告已生成")
        
        # 返回生成的文件路径
        return {
            "main_report": report_file,
            "readable_report": readable_report_file,
            "retrieval_details_json": retrieval_files.get("json"),
            "retrieval_details_readable": retrieval_files.get("readable"),
            "timestamp": timestamp
        }
    
    def _calculate_dual_tower_overall_stats(self) -> Dict[str, Any]:
        """计算双塔整体统计"""
        if not self.test_results:
            return {}
        
        valid_results = [r for r in self.test_results if r.evaluation_success]
        
        if not valid_results:
            return {"error": "no_valid_results"}
        
        # 收集各种分数
        f1_scores = [r.evaluation_scores.get("token_f1", 0.0) for r in valid_results]
        semantic_scores = [r.evaluation_scores.get("semantic_similarity", 0.0) for r in valid_results]
        llm_scores = [r.evaluation_scores.get("llm_accuracy", 0.0) for r in valid_results]
        exact_match_scores = [r.evaluation_scores.get("exact_match", 0.0) for r in valid_results]
        confidence_scores = [r.confidence_score for r in valid_results]
        
        # 收集时间统计
        hierarchical_times = [r.hierarchical_retrieval_time for r in valid_results]
        graph_times = [r.graph_retrieval_time for r in valid_results]
        generation_times = [r.generation_time for r in valid_results]
        
        # 系统成功率统计
        hierarchical_success_count = sum(1 for r in valid_results if r.hierarchical_context.get("hierarchical_enabled", False))
        graph_success_count = sum(1 for r in valid_results if len(r.graph_retrieved_units) > 0)
        both_success_count = sum(1 for r in valid_results 
                               if r.hierarchical_context.get("hierarchical_enabled", False) and len(r.graph_retrieved_units) > 0)
        
        return {
            "total_valid_tests": len(valid_results),
            "performance_metrics": {
                "avg_f1_score": np.mean(f1_scores),
                "std_f1_score": np.std(f1_scores),
                "avg_semantic_similarity": np.mean(semantic_scores),
                "avg_llm_accuracy": np.mean(llm_scores),
                "avg_exact_match": np.mean(exact_match_scores),
                "avg_confidence": np.mean(confidence_scores)
            },
            "timing_metrics": {
                "avg_hierarchical_time": np.mean(hierarchical_times),
                "avg_graph_time": np.mean(graph_times),
                "avg_generation_time": np.mean(generation_times),
                "avg_total_time": np.mean([h+g+gen for h, g, gen in zip(hierarchical_times, graph_times, generation_times)])
            },
            "system_success_rates": {
                "hierarchical_success_rate": hierarchical_success_count / len(valid_results),
                "graph_success_rate": graph_success_count / len(valid_results),
                "both_systems_success_rate": both_success_count / len(valid_results),
                "dual_tower_advantage": both_success_count / max(hierarchical_success_count, graph_success_count, 1)
            }
        }
    
    def _calculate_dual_tower_sample_performance(self) -> Dict[str, Any]:
        """计算样本性能"""
        sample_results = defaultdict(list)
        for result in self.test_results:
            if result.evaluation_success:
                sample_results[result.sample_id].append(result)
        
        sample_performance = {}
        for sample_id, results in sample_results.items():
            f1_scores = [r.evaluation_scores.get("token_f1", 0.0) for r in results]
            sample_performance[sample_id] = {
                "test_count": len(results),
                "avg_f1_score": np.mean(f1_scores),
                "hierarchical_success_rate": sum(1 for r in results if r.hierarchical_context.get("hierarchical_enabled", False)) / len(results),
                "graph_success_rate": sum(1 for r in results if len(r.graph_retrieved_units) > 0) / len(results),
                "dual_tower_success_rate": sum(1 for r in results 
                                             if r.hierarchical_context.get("hierarchical_enabled", False) and len(r.graph_retrieved_units) > 0) / len(results)
            }
        
        return sample_performance
    
    def _calculate_dual_tower_category_performance(self) -> Dict[str, Any]:
        """计算类别性能"""
        category_results = defaultdict(list)
        for result in self.test_results:
            if result.evaluation_success:
                category_results[result.category].append(result)
        
        category_performance = {}
        for category, results in category_results.items():
            f1_scores = [r.evaluation_scores.get("token_f1", 0.0) for r in results]
            llm_scores = [r.evaluation_scores.get("llm_accuracy", 0.0) for r in results]
            
            category_performance[category] = {
                "test_count": len(results),
                "avg_f1_score": np.mean(f1_scores),
                "avg_llm_accuracy": np.mean(llm_scores),
                "hierarchical_success_rate": sum(1 for r in results if r.hierarchical_context.get("hierarchical_enabled", False)) / len(results),
                "graph_success_rate": sum(1 for r in results if len(r.graph_retrieved_units) > 0) / len(results)
            }
        
        return category_performance
    
    def _analyze_fusion_effectiveness(self) -> Dict[str, Any]:
        """分析融合效果"""
        valid_results = [r for r in self.test_results if r.evaluation_success]
        
        if not valid_results:
            return {}
        
        # 按系统组合分类结果
        both_systems = [r for r in valid_results 
                       if r.hierarchical_context.get("hierarchical_enabled", False) and len(r.graph_retrieved_units) > 0]
        hierarchical_only = [r for r in valid_results 
                           if r.hierarchical_context.get("hierarchical_enabled", False) and len(r.graph_retrieved_units) == 0]
        graph_only = [r for r in valid_results 
                     if not r.hierarchical_context.get("hierarchical_enabled", False) and len(r.graph_retrieved_units) > 0]
        neither = [r for r in valid_results 
                  if not r.hierarchical_context.get("hierarchical_enabled", False) and len(r.graph_retrieved_units) == 0]
        
        def calc_avg_f1(results):
            if not results:
                return 0.0
            return np.mean([r.evaluation_scores.get("token_f1", 0.0) for r in results])
        
        return {
            "fusion_strategy": self.fusion_strategy,
            "system_combination_analysis": {
                "both_systems": {
                    "count": len(both_systems),
                    "avg_f1": calc_avg_f1(both_systems),
                    "percentage": len(both_systems) / len(valid_results) * 100
                },
                "hierarchical_only": {
                    "count": len(hierarchical_only),
                    "avg_f1": calc_avg_f1(hierarchical_only),
                    "percentage": len(hierarchical_only) / len(valid_results) * 100
                },
                "graph_only": {
                    "count": len(graph_only),
                    "avg_f1": calc_avg_f1(graph_only),
                    "percentage": len(graph_only) / len(valid_results) * 100
                },
                "neither_system": {
                    "count": len(neither),
                    "avg_f1": calc_avg_f1(neither),
                    "percentage": len(neither) / len(valid_results) * 100
                }
            },
            "fusion_effectiveness": {
                "dual_tower_advantage": calc_avg_f1(both_systems) - max(calc_avg_f1(hierarchical_only), calc_avg_f1(graph_only)),
                "best_single_system": "hierarchical" if calc_avg_f1(hierarchical_only) > calc_avg_f1(graph_only) else "graph",
                "fusion_vs_best_single": calc_avg_f1(both_systems) - max(calc_avg_f1(hierarchical_only), calc_avg_f1(graph_only))
            }
        }
    
    def _generate_dual_tower_readable_report(self, benchmark_report: Dict[str, Any], timestamp: str):
        """生成双塔可读性报告"""
        lines = []
        
        lines.append("=" * 100)
        lines.append("LoCoMo双塔召回系统Benchmark测试报告")
        lines.append("=" * 100)
        
        # 基本信息
        info = benchmark_report["benchmark_info"]
        lines.append(f"\n📊 测试概况:")
        lines.append(f"   - 测试类型: {info['test_type']}")
        lines.append(f"   - 融合策略: {info['fusion_strategy']}")
        lines.append(f"   - 融合权重: 分层={info['fusion_weights']['hierarchical']}, 图检索={info['fusion_weights']['graph']}")
        lines.append(f"   - 总样本数: {info['total_samples']}")
        lines.append(f"   - 总测试数: {info['total_tests_run']}")
        
        # 系统配置
        lines.append(f"\n🔧 系统配置:")
        lines.append(f"   - 分层检索: L0={info['hierarchical_config']['l0_top_k']}, L1={info['hierarchical_config']['l1_top_k']}, L2={info['hierarchical_config']['l2_top_k']}")
        lines.append(f"   - 图检索: 语义top-k={info['graph_config']['topk_similarity']}, 图top-k={info['graph_config']['topk_graph']}")
        lines.append(f"   - 实体关系检索: {info['graph_config']['use_entity_relation']}")
        
        # 整体性能
        overall = benchmark_report["overall_statistics"]
        if "error" not in overall:
            perf = overall["performance_metrics"]
            timing = overall["timing_metrics"]
            success = overall["system_success_rates"]
            
            lines.append(f"\n🎯 整体性能:")
            lines.append(f"   - 平均F1分数: {perf['avg_f1_score']:.3f} ± {perf['std_f1_score']:.3f}")
            lines.append(f"   - 平均语义相似度: {perf['avg_semantic_similarity']:.3f}")
            lines.append(f"   - 平均LLM准确率: {perf['avg_llm_accuracy']:.3f}")
            lines.append(f"   - 平均精确匹配: {perf['avg_exact_match']:.3f}")
            lines.append(f"   - 平均置信度: {perf['avg_confidence']:.3f}")
            
            lines.append(f"\n⏱️  时间性能:")
            lines.append(f"   - 分层检索时间: {timing['avg_hierarchical_time']:.3f}s")
            lines.append(f"   - 图检索时间: {timing['avg_graph_time']:.3f}s")
            lines.append(f"   - 答案生成时间: {timing['avg_generation_time']:.3f}s")
            lines.append(f"   - 总平均时间: {timing['avg_total_time']:.3f}s")
            
            lines.append(f"\n🔄 系统成功率:")
            lines.append(f"   - 分层检索成功率: {success['hierarchical_success_rate']:.2%}")
            lines.append(f"   - 图检索成功率: {success['graph_success_rate']:.2%}")
            lines.append(f"   - 双塔同时成功率: {success['both_systems_success_rate']:.2%}")
            lines.append(f"   - 双塔优势系数: {success['dual_tower_advantage']:.3f}")
        
        # 融合效果分析
        fusion = benchmark_report["fusion_analysis"]
        if fusion:
            combo = fusion["system_combination_analysis"]
            effectiveness = fusion["fusion_effectiveness"]
            
            lines.append(f"\n🔀 融合效果分析:")
            lines.append(f"   - 双塔同时工作: {combo['both_systems']['count']}次 ({combo['both_systems']['percentage']:.1f}%), F1={combo['both_systems']['avg_f1']:.3f}")
            lines.append(f"   - 仅分层检索: {combo['hierarchical_only']['count']}次 ({combo['hierarchical_only']['percentage']:.1f}%), F1={combo['hierarchical_only']['avg_f1']:.3f}")
            lines.append(f"   - 仅图检索: {combo['graph_only']['count']}次 ({combo['graph_only']['percentage']:.1f}%), F1={combo['graph_only']['avg_f1']:.3f}")
            lines.append(f"   - 双系统都失败: {combo['neither_system']['count']}次 ({combo['neither_system']['percentage']:.1f}%), F1={combo['neither_system']['avg_f1']:.3f}")
            
            lines.append(f"\n📈 融合优势:")
            lines.append(f"   - 最佳单系统: {effectiveness['best_single_system']}")
            lines.append(f"   - 双塔相对优势: {effectiveness['dual_tower_advantage']:.3f}")
            lines.append(f"   - 融合vs最佳单系统: {effectiveness['fusion_vs_best_single']:.3f}")
        
        # 类别性能
        category_perf = benchmark_report["category_performance"]
        if category_perf:
            lines.append(f"\n📋 问题类别性能:")
            category_names = {1: "多跳问题", 2: "时间问题", 3: "开放域问题", 4: "单跳问题", 5: "对抗性问题"}
            for category, stats in category_perf.items():
                category_name = category_names.get(category, f"类别{category}")
                lines.append(f"\n   {category_name} ({stats['test_count']}题):")
                lines.append(f"     - F1分数: {stats['avg_f1_score']:.3f}")
                lines.append(f"     - LLM准确率: {stats['avg_llm_accuracy']:.3f}")
                lines.append(f"     - 分层成功率: {stats['hierarchical_success_rate']:.2%}")
                lines.append(f"     - 图检索成功率: {stats['graph_success_rate']:.2%}")
        
        # 保存可读性报告
        readable_report_file = self.output_dir / f"dual_tower_benchmark_readable_report_{timestamp}.txt"
        with open(readable_report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        # 打印到控制台
        print('\n'.join(lines))
        
        logger.info(f"双塔可读性报告已生成: {readable_report_file}")
        
        # 返回文件路径
        return readable_report_file

    def generate_retrieval_details_report(self) -> Dict[str, Path]:
        """生成检索详情报告 - 单独文件，包含问题、答案和检索结果
        
        Returns:
            Dict[str, Path]: 生成的文件路径字典
        """
        logger.info("生成检索详情报告...")
        
        if not self.test_results:
            logger.warning("没有测试结果，无法生成检索详情报告")
            return {}
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        details_file = self.output_dir / f"retrieval_details_{timestamp}.json"
        
        retrieval_details = []
        
        for result in self.test_results:
            # 构建单个问题的检索详情
            detail_entry = {
                "sample_id": result.sample_id,
                "question_info": {
                    "question": result.question,
                    "category": result.category,
                    "question_id": f"{result.sample_id}_{result.question.replace(' ', '_')}"
                },
                
                # 答案部分
                "answers": {
                    "expected_answer": result.expected_answer,
                    "generated_answer": result.final_answer,
                    "reasoning_process": result.reasoning_process,
                    "confidence_score": result.confidence_score
                },
                
                # 分层检索详情
                "hierarchical_retrieval": {
                    "enabled": result.hierarchical_context.get("hierarchical_enabled", False),
                    "retrieval_time": result.hierarchical_retrieval_time,
                    "retrieval_method": result.hierarchical_context.get("retrieval_method", "unknown"),
                    "l2_insights": self._extract_retrieval_summary(
                        result.hierarchical_context.get("l2_insights", []), 
                        "L2"
                    ),
                    "l1_summaries": self._extract_retrieval_summary(
                        result.hierarchical_context.get("l1_summaries", []), 
                        "L1"
                    ),
                    "l0_observations": self._extract_retrieval_summary(
                        result.hierarchical_context.get("l0_observations", []), 
                        "L0"
                    ),
                    "retrieval_stats": result.hierarchical_context.get("retrieval_stats", {})
                },
                
                # 知识图谱检索详情
                "graph_retrieval": {
                    "enabled": len(result.graph_retrieved_units) > 0,
                    "retrieval_time": result.graph_retrieval_time,
                    "retrieval_details": result.graph_retrieval_details,
                    "retrieved_units": [
                        {
                            "rank": i + 1,
                            "score": score,
                            "content": self._extract_unit_text(unit),
                            "uid": getattr(unit, 'uid', 'unknown')
                        }
                        for i, (unit, score) in enumerate(result.graph_retrieved_units)
                    ]
                },
                
                # 融合和评估信息
                "fusion_info": {
                    "fusion_method": result.fusion_method,
                    "generation_time": result.generation_time,
                    "confidence_score": result.confidence_score
                },
                
                "evaluation": {
                    "success": result.evaluation_success,
                    "scores": result.evaluation_scores
                },
                
                "timestamp": datetime.now().isoformat()
            }
            
            retrieval_details.append(detail_entry)
        
        # 保存检索详情文件
        with open(details_file, 'w', encoding='utf-8') as f:
            json.dump(retrieval_details, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ 检索详情报告已生成: {details_file}")
        
        # 同时生成易读版本
        readable_file = self._generate_readable_retrieval_details(retrieval_details, timestamp)
        
        # 返回生成的文件路径
        return {
            "json": details_file,
            "readable": readable_file,
            "timestamp": timestamp
        }

    def _extract_retrieval_summary(self, retrieval_list: List[Dict], layer_name: str) -> List[Dict]:
        """提取检索结果摘要"""
        summaries = []
        for item in retrieval_list:
            summary = {
                "uid": item.get("uid", "unknown"),
                "score": item.get("score", 0.0),
                "content_preview": item.get("content", "")[:200],  # 前200字符
            }
            
            # 添加层特定信息
            if layer_name == "L2":
                summary["core_summary"] = item.get("core_summary", "")
                summary["key_themes"] = item.get("key_themes", [])
            elif layer_name == "L1":
                summary["session_date"] = item.get("session_date", "")
                summary["main_topics"] = item.get("main_topics", [])
            elif layer_name == "L0":
                summary["speaker"] = item.get("speaker", "unknown")
                summary["session_datetime"] = item.get("session_datetime", "")
            
            summaries.append(summary)
        
        return summaries

    def _extract_unit_text(self, unit) -> str:
        """从unit提取文本内容"""
        if hasattr(unit, 'raw_data') and unit.raw_data:
            text = unit.raw_data.get('text_content', '')
            if not text:
                text = str(unit.raw_data.get('summary', ''))
            if not text:
                text = str(unit.raw_data)[:300]
            return text
        return str(unit)[:300]

    def _generate_readable_retrieval_details(self, retrieval_details: List[Dict], timestamp: str):
        """生成易读的检索详情文本文件"""
        readable_file = self.output_dir / f"retrieval_details_readable_{timestamp}.txt"
        
        lines = []
        lines.append("=" * 100)
        lines.append("LoCoMo双塔检索详情报告")
        lines.append("=" * 100)
        lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"总问题数: {len(retrieval_details)}")
        lines.append("=" * 100)
        
        for i, detail in enumerate(retrieval_details, 1):
            lines.append(f"\n{'='*100}")
            lines.append(f"问题 {i}: {detail['sample_id']}")
            lines.append(f"{'='*100}")
            
            # 问题信息
            q_info = detail["question_info"]
            lines.append(f"\n📋 问题:")
            lines.append(f"   {q_info['question']}")
            lines.append(f"   类别: {q_info['category']}")
            
            # 答案对比
            answers = detail["answers"]
            lines.append(f"\n💡 答案对比:")
            lines.append(f"   标准答案: {answers['expected_answer']}")
            lines.append(f"   生成答案: {answers['generated_answer']}")
            lines.append(f"   置信度: {answers['confidence_score']:.3f}")
            
            # 推理过程
            lines.append(f"\n🤔 推理过程:")
            reasoning = answers['reasoning_process']
            # 分段显示推理过程
            for line in reasoning.split('\n'):
                if line.strip():
                    lines.append(f"   {line.strip()}")
            
            # Hierarchical retrieval results
            hier = detail["hierarchical_retrieval"]
            lines.append(f"\n🏗️  分层检索:")
            lines.append(f"   状态: {'✓ 成功' if hier['enabled'] else '✗ 失败'}")
            lines.append(f"   耗时: {hier['retrieval_time']:.3f}s")
            
            if hier['enabled']:
                # L2洞见
                if hier['l2_insights']:
                    lines.append(f"\n   L2洞见 ({len(hier['l2_insights'])}个):")
                    for insight in hier['l2_insights']:
                        lines.append(f"     - [分数: {insight['score']:.3f}] {insight['content_preview'][:100]}...")
                
                # L1摘要
                if hier['l1_summaries']:
                    lines.append(f"\n   L1摘要 ({len(hier['l1_summaries'])}个):")
                    for summary in hier['l1_summaries']:
                        lines.append(f"     - [分数: {summary['score']:.3f}] {summary.get('session_date', '')} - {summary['content_preview'][:100]}...")
                
                # L0观察
                if hier['l0_observations']:
                    lines.append(f"\n   L0观察 ({len(hier['l0_observations'])}个):")
                    for obs in hier['l0_observations'][:5]:
                        speaker = obs.get('speaker', 'Unknown')
                        lines.append(f"     - [分数: {obs['score']:.3f}] {speaker}: {obs['content_preview'][:80]}...")
            
            # Knowledge-graph retrieval results
            graph = detail["graph_retrieval"]
            lines.append(f"\n🕸️  图谱检索:")
            lines.append(f"   状态: {'✓ 成功' if graph['enabled'] else '✗ 失败'}")
            lines.append(f"   耗时: {graph['retrieval_time']:.3f}s")
            lines.append(f"   方法: {graph['retrieval_details'].get('method', 'unknown')}")
            
            if graph['enabled'] and graph['retrieved_units']:
                lines.append(f"\n   检索单元 ({len(graph['retrieved_units'])}个):")
                for unit in graph['retrieved_units'][:5]:
                    lines.append(f"     {unit['rank']}. [分数: {unit['score']:.3f}] {unit['content'][:100]}...")
            
            # Evaluation results
            eval_info = detail["evaluation"]
            if eval_info['success']:
                scores = eval_info['scores']
                lines.append(f"\n📊 评估分数:")
                lines.append(f"   F1分数: {scores.get('token_f1', 0):.3f}")
                lines.append(f"   语义相似度: {scores.get('semantic_similarity', 0):.3f}")
                lines.append(f"   精确匹配: {scores.get('exact_match', 0):.3f}")
                if 'llm_accuracy' in scores:
                    lines.append(f"   LLM准确率: {scores.get('llm_accuracy', 0):.3f}")
        
        # 保存易读文件
        with open(readable_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        logger.info(f"✅ 易读检索详情已生成: {readable_file}")
        
        # 返回文件路径
        return readable_file

    def validate_dataset_compatibility(self) -> Dict[str, Any]:
        """
        验证数据集兼容性 - 检查新旧数据集格式
        
        Returns:
            兼容性报告
        """
        logger.info("🔍 验证数据集兼容性...")
        
        compatibility_report = {
            "dataset_version": "unknown",
            "step3_graphs_format": "unknown",
            "enhanced_graphs_format": "unknown",
            "has_hierarchical_overview": False,
            "has_step1_markers": False,
            "has_step2_markers": False,
            "space_naming_convention": "unknown",
            "compatibility_status": "unknown"
        }
        
        available_samples = self._get_available_samples()
        
        if not available_samples:
            compatibility_report["compatibility_status"] = "error_no_samples"
            return compatibility_report
        
        test_sample_id = available_samples[0]
        
        # 1. 检查实体关系图谱（step3_graphs_dir）
        step3_dir = self.step3_graphs_dir / test_sample_id
        if step3_dir.exists():
            compatibility_report["step3_graphs_format"] = "valid"
            logger.info(f"✅ 实体关系图谱格式正确: {step3_dir}")
        
        # 2. 检查分层图谱（enhanced_graphs_dir）
        enhanced_dir = self.enhanced_graphs_dir / test_sample_id
        enhanced_dir_old = self.enhanced_graphs_dir / f"{test_sample_id}_enhanced"
        
        if enhanced_dir.exists():
            compatibility_report["enhanced_graphs_format"] = "new_format_no_suffix"
            target_dir = enhanced_dir
        elif enhanced_dir_old.exists():
            compatibility_report["enhanced_graphs_format"] = "old_format_with_suffix"
            target_dir = enhanced_dir_old
            logger.warning(f"⚠️  检测到旧格式分层图谱（带_enhanced后缀）")
        else:
            compatibility_report["enhanced_graphs_format"] = "not_found"
            compatibility_report["compatibility_status"] = "error_enhanced_not_found"
            return compatibility_report
        
        # 3. 检查分层概览文件（新数据集特征）
        overview_file = target_dir / "hierarchical_overview.json"
        if overview_file.exists():
            compatibility_report["has_hierarchical_overview"] = True
            try:
                with open(overview_file, 'r', encoding='utf-8') as f:
                    overview_data = json.load(f)
                
                # 检查数据复用信息
                data_reuse_info = overview_data.get("data_reuse_info", {})
                compatibility_report["has_step1_markers"] = data_reuse_info.get("reused_l0_units", 0) > 0
                compatibility_report["has_step2_markers"] = data_reuse_info.get("new_l1_units", 0) > 0
                
                logger.info(f"📊 分层概览统计: "
                        f"L0复用={data_reuse_info.get('reused_l0_units', 0)}, "
                        f"L1新增={data_reuse_info.get('new_l1_units', 0)}, "
                        f"L2新增={data_reuse_info.get('new_l2_units', 0)}")
                
            except Exception as e:
                logger.warning(f"读取分层概览失败: {e}")
        
        # 4. 检查空间命名约定
        semantic_map_file = target_dir / "semantic_map_data" / "semantic_map.json"
        if semantic_map_file.exists():
            try:
                with open(semantic_map_file, 'r', encoding='utf-8') as f:
                    semantic_map = json.load(f)
                
                memory_spaces = semantic_map.get("memory_spaces", {})
                space_names = list(memory_spaces.keys())
                
                if space_names:
                    first_space = space_names[0]
                    if first_space.startswith("hierarchical:"):
                        compatibility_report["space_naming_convention"] = "with_prefix"
                    else:
                        compatibility_report["space_naming_convention"] = "without_prefix"
                
            except Exception as e:
                logger.warning(f"读取语义图谱失败: {e}")
        
        # 5. 确定数据集版本和兼容性
        if compatibility_report["has_hierarchical_overview"]:
            if compatibility_report["has_step1_markers"] and compatibility_report["has_step2_markers"]:
                compatibility_report["dataset_version"] = "step3_new_complete"
                compatibility_report["compatibility_status"] = "compatible"
            else:
                compatibility_report["dataset_version"] = "step3_new_partial"
                compatibility_report["compatibility_status"] = "warning_incomplete"
        else:
            compatibility_report["dataset_version"] = "legacy_or_step3_old"
            compatibility_report["compatibility_status"] = "warning_legacy"
        
        # 日志输出
        logger.info(f"📊 数据集版本: {compatibility_report['dataset_version']}")
        logger.info(f"🏗️  实体关系图谱: {compatibility_report['step3_graphs_format']}")
        logger.info(f"🏗️  分层图谱格式: {compatibility_report['enhanced_graphs_format']}")
        logger.info(f"🏷️  空间命名: {compatibility_report['space_naming_convention']}")
        logger.info(f"✅ 兼容性状态: {compatibility_report['compatibility_status']}")
        
        return compatibility_report

    #### 调试接口：获取双塔融合的完整prompt ####
    #### 不要用于测试流程，只用于调试 ####

    def debug_get_fusion_prompt(self,
                        question: str,
                        category: int,
                        sample_id: str) -> Dict[str, Any]:
        """
        🔍 调试接口：获取双塔融合的完整prompt（不调用LLM）
        
        执行与 _context_aware_fusion_generation 相同的检索流程，
        但只返回构建的prompt，不进行LLM调用。
        
        Args:
            question: 问题文本
            category: 问题类别
            sample_id: 样本ID
            
        Returns:
            Dict包含:
            - full_prompt: 完整的融合prompt文本
            - hierarchical_context: 分层检索上下文
            - graph_results: 图检索结果
            - graph_details: 图检索详情
            - prompt_stats: prompt统计信息
        """
        logger.info(f"🔍 调试模式：获取样本 {sample_id} 的融合prompt")
        
        try:
            # 1. 执行分层检索（与实际测试流程一致）
            hierarchical_start = time.time()
            hierarchical_context = self._run_hierarchical_retrieval(sample_id, question, category)
            hierarchical_time = time.time() - hierarchical_start
            
            # 2. 执行图检索（与实际测试流程一致）
            graph_start = time.time()
            graph_results, graph_details = self._run_graph_retrieval(sample_id, question)
            graph_time = time.time() - graph_start
            
            # 3. 构建完整prompt（复用 _context_aware_fusion_generation 的逻辑）
            full_prompt = self._build_fusion_prompt_for_debug(
                question=question,
                category=category,
                hierarchical_context=hierarchical_context,
                graph_results=graph_results,
                graph_details=graph_details
            )
            
            # 4. 计算prompt统计信息
            prompt_stats = self._calculate_prompt_stats(
                full_prompt=full_prompt,
                hierarchical_context=hierarchical_context,
                graph_results=graph_results
            )
            
            # 5. 构建返回结果
            debug_result = {
                "sample_id": sample_id,
                "question": question,
                "category": category,
                "full_prompt": full_prompt,
                "prompt_stats": prompt_stats,
                "retrieval_info": {
                    "hierarchical_context": {
                        "enabled": hierarchical_context.get("hierarchical_enabled", False),
                        "retrieval_time": hierarchical_time,
                        "l2_count": len(hierarchical_context.get("l2_insights", [])),
                        "l1_count": len(hierarchical_context.get("l1_summaries", [])),
                        "l0_count": len(hierarchical_context.get("l0_observations", [])),
                    },
                    "graph_results": {
                        "enabled": len(graph_results) > 0,
                        "retrieval_time": graph_time,
                        "results_count": len(graph_results),
                        "method": graph_details.get("method", "unknown")
                    }
                },
                "debug_timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"✅ Prompt获取成功: {prompt_stats['total_chars']} 字符, "
                    f"{prompt_stats['estimated_tokens']} tokens (估算)")
            
            return debug_result
            
        except Exception as e:
            logger.error(f"❌ 调试prompt获取失败: {e}")
            return {
                "error": str(e),
                "sample_id": sample_id,
                "question": question,
                "category": category,
                "debug_timestamp": datetime.now().isoformat()
            }


    def _build_fusion_prompt_for_debug(self,
                                    question: str,
                                    category: int,
                                    hierarchical_context: Dict[str, Any],
                                    graph_results: List[Any],
                                    graph_details: Dict[str, Any]) -> str:
        """
        🔍 为调试构建融合prompt（与 _context_aware_fusion_generation 逻辑一致）
        
        这个函数复用了 _context_aware_fusion_generation 的prompt构建逻辑，
        但不调用LLM，只返回prompt文本。
        """
        # 构建双塔融合提示词（完全复用现有逻辑）
        prompt_parts = []
        
        # 对抗性问题特殊指导
        if category == 5:
            prompt_parts.append("You are an expert conversation analyst specialized in detecting misleading or unanswerable questions.")
        else:
            prompt_parts.append("You are an expert conversation analyst with access to two complementary information retrieval systems.")
            
        prompt_parts.append("")
        prompt_parts.append("IMPORTANT: These are two DIFFERENT retrieval systems providing COMPLEMENTARY information:")
        prompt_parts.append("1. HIERARCHICAL MEMORY: Provides structured, multi-layer conversational context")
        prompt_parts.append("2. KNOWLEDGE GRAPH: Provides specific facts and entity relationships")
        prompt_parts.append("")
        prompt_parts.append("Your task is to synthesize information from BOTH systems to provide the most accurate and complete answer.")
        prompt_parts.append("")
        
        # 问题和类别指导
        category_guidance = self._get_dual_tower_category_guidance(category)
        prompt_parts.append(f"QUESTION: {question}")
        prompt_parts.append(f"QUESTION CATEGORY: {category} - {category_guidance}")
        prompt_parts.append("")
        
        # Hierarchical retrieval results
        hierarchical_enabled = hierarchical_context.get("hierarchical_enabled", False)
        if hierarchical_enabled:
            prompt_parts.append("=" * 80)
            prompt_parts.append("HIERARCHICAL MEMORY RESULTS")
            prompt_parts.append("=" * 80)
            prompt_parts.append(hierarchical_context.get("hierarchical_context_text", "No hierarchical context available"))
        else:
            prompt_parts.append("=" * 80)
            prompt_parts.append("HIERARCHICAL MEMORY RESULTS")
            prompt_parts.append("=" * 80)
            prompt_parts.append("Hierarchical retrieval was not available for this query.")
        
        prompt_parts.append("")
        
        # Knowledge-graph retrieval results
        if graph_results:
            prompt_parts.append("=" * 80)
            prompt_parts.append("KNOWLEDGE GRAPH RESULTS")
            prompt_parts.append("=" * 80)
            prompt_parts.append(f"Retrieved {len(graph_results)} relevant knowledge graph units:")
            prompt_parts.append("")
            
            for i, (unit, score) in enumerate(graph_results, 1):
                unit_content = self._extract_graph_unit_content(unit)
                prompt_parts.append(f"Graph Result {i}:")
                
                # 提取实体类型和上下文
                entity_type = getattr(unit, 'entity_type', 'Unknown')
                if entity_type != 'Unknown':
                    prompt_parts.append(f"Entity: {unit_content[:200]} | Type: {entity_type}")
                else:
                    prompt_parts.append(f"Entity: {unit_content[:200]}")
                prompt_parts.append("")
            
        else:
            prompt_parts.append("=" * 80)
            prompt_parts.append("KNOWLEDGE GRAPH RESULTS")
            prompt_parts.append("=" * 80)
            prompt_parts.append("No relevant entities or relationships found in the knowledge graph.")
        
        prompt_parts.append("")
        
        # 融合指导（对抗性问题加强）
        if category == 5:
            prompt_parts.append("=" * 80)
            prompt_parts.append("DUAL TOWER FUSION GUIDANCE")
            prompt_parts.append("=" * 80)
            
            prompt_parts.append("SYNTHESIS INSTRUCTIONS:")
            prompt_parts.append("1. If BOTH systems provided information: Cross-validate and synthesize")
            prompt_parts.append("2. If ONLY one system worked: Rely on available system's information")
            prompt_parts.append("3. ADVERSARIAL QUESTION: Strictly verify entity existence in BOTH systems")
            prompt_parts.append("4. If information is contradictory or not found: State 'No information available'")
            prompt_parts.append("")
            prompt_parts.append("RESPONSE FORMAT (REQUIRED JSON):")
            prompt_parts.append("{")
            prompt_parts.append('    "reasoning": "Your synthesis process...",')
            prompt_parts.append('    "final_answer": "Your direct, concise final answer"')
            prompt_parts.append("}")
        else:
            prompt_parts.append("=" * 80)
            prompt_parts.append("DUAL TOWER FUSION GUIDANCE")
            prompt_parts.append("=" * 80)
            
            prompt_parts.append("SYNTHESIS INSTRUCTIONS:")
            prompt_parts.append("1. If BOTH systems provided information: Cross-validate and synthesize")
            prompt_parts.append("2. If ONLY one system worked: Rely on available system's information")
            prompt_parts.append("")
            prompt_parts.append("RESPONSE FORMAT (REQUIRED JSON):")
            prompt_parts.append("{")
            prompt_parts.append('    "reasoning": "Your synthesis process...",')
            prompt_parts.append('    "final_answer": "Your direct, concise final answer"')
            prompt_parts.append("}")
        
        return "\n".join(prompt_parts)


    def _calculate_prompt_stats(self,
                                full_prompt: str,
                                hierarchical_context: Dict[str, Any],
                                graph_results: List[Any]) -> Dict[str, Any]:
        """
        计算prompt的统计信息
        
        Returns:
            Dict包含字符数、估算token数、各部分占比等
        """
        total_chars = len(full_prompt)
        
        # 估算token数（粗略：1 token ≈ 4 chars）
        estimated_tokens = total_chars // 4
        
        # 尝试使用tiktoken精确计算（如果可用）
        try:
            import tiktoken
            encoder = tiktoken.encoding_for_model("gpt-4")
            actual_tokens = len(encoder.encode(full_prompt))
        except ImportError:
            actual_tokens = None
        
        # 计算各部分占比
        hierarchical_text = hierarchical_context.get("hierarchical_context_text", "")
        hierarchical_chars = len(hierarchical_text)
        
        # 估算图检索部分长度
        graph_section_start = full_prompt.find("KNOWLEDGE GRAPH RESULTS")
        graph_section_end = full_prompt.find("DUAL TOWER FUSION GUIDANCE")
        if graph_section_start != -1 and graph_section_end != -1:
            graph_chars = graph_section_end - graph_section_start
        else:
            graph_chars = 0
        
        instruction_chars = total_chars - hierarchical_chars - graph_chars
        
        stats = {
            "total_chars": total_chars,
            "estimated_tokens": estimated_tokens,
            "actual_tokens": actual_tokens,
            "hierarchical_section": {
                "chars": hierarchical_chars,
                "percentage": (hierarchical_chars / total_chars * 100) if total_chars > 0 else 0
            },
            "graph_section": {
                "chars": graph_chars,
                "percentage": (graph_chars / total_chars * 100) if total_chars > 0 else 0,
                "results_count": len(graph_results)
            },
            "instruction_section": {
                "chars": instruction_chars,
                "percentage": (instruction_chars / total_chars * 100) if total_chars > 0 else 0
            }
        }
        
        return stats


    def _extract_graph_unit_content(self, unit) -> str:
        """从图单元中提取内容文本"""
        if hasattr(unit, 'raw_data') and unit.raw_data:
            # 尝试提取text_content
            text_content = unit.raw_data.get('text_content', '')
            if text_content:
                return text_content
            
            # 尝试其他字段
            for field in ['content', 'name', 'description', 'summary']:
                content = unit.raw_data.get(field, '')
                if content:
                    return str(content)
            
            # 最后兜底
            return str(unit.raw_data)[:200]
        
        # 如果没有raw_data，尝试直接转字符串
        return str(unit)[:200]

def main():
    """主函数 - 支持并行处理配置"""
    
    parser = argparse.ArgumentParser(description="LoCoMo双塔召回系统Benchmark测试")
    
    # Data paths参数
    parser.add_argument("--enhanced-graphs-dir", 
                       default="benchmark_locomo/dataset/locomo/hierarchical/step3_final_graphs",
                       help="增强图谱目录（分层检索）")
    parser.add_argument("--step3-graphs-dir", 
                       default="benchmark_locomo/dataset/locomo/entity_relation/step3_semantic_graph",
                       help="步骤3图谱目录（知识图谱检索）")
    parser.add_argument("--qa-dataset", 
                       default="benchmark_locomo/dataset/locomo/locomo10.json",
                       help="QA数据集路径")
    parser.add_argument("--output-dir", 
                       default="benchmark_locomo/task_eval/results/locomo_dual_tower_benchmark_new_dataset",
                       help="输出目录")
    
    # LLM模型参数
    parser.add_argument("--llm-model", 
                    #    default="gpt-4o-mini-closeai",
                       default="gpt-4.1-mini-closeai",
                       help="答案生成LLM模型名称")
    parser.add_argument("--llm-evaluate-model", 
                    #    default="deepseek-chat",
                       default="gpt-4o-mini-closeai",
                       help="答案评估LLM模型名称")
    
    # Retrieval configuration参数
    parser.add_argument("--topk-hierarchical-l0", type=int, default=15,
                       help="分层检索L0层top-k")
    parser.add_argument("--topk-hierarchical-l1", type=int, default=5,
                       help="分层检索L1层top-k")
    parser.add_argument("--topk-hierarchical-l2", type=int, default=1,
                       help="分层检索L2层top-k")
    parser.add_argument("--topk-similarity", type=int, default=15,
                       help="图检索语义检索top-k")
    parser.add_argument("--topk-graph", type=int, default=0,  # 修改：从5改为0，默认禁用
                       help="图检索实体关系top-k（默认0禁用，设置>0启用）")
    
    # 融合策略参数
    parser.add_argument("--fusion-strategy", 
                       choices=["simple", "weighted", "context_aware"],
                       default="context_aware",
                       help="融合策略")
    parser.add_argument("--hierarchical-weight", type=float, default=0.5,
                       help="分层检索权重")
    parser.add_argument("--graph-weight", type=float, default=0.5,
                       help="图检索权重")
    
    # Test configuration参数
    parser.add_argument("--sample-ids", nargs='+',
                       help="指定测试样本ID列表")
    parser.add_argument("--max-samples", type=int,
                       help="最大测试样本数")
    parser.add_argument("--max-workers", type=int, default=1,
                       help="并发工作线程数")
    parser.add_argument("--no-entity-relation", action="store_true",
                       default=True,  # 修改：默认设为True，即默认禁用实体关系检索
                       help="禁用实体关系检索（默认禁用）")
    parser.add_argument("--enable-entity-relation", dest="no_entity_relation", 
                       action="store_false",  # 新增：提供启用选项
                       help="启用实体关系检索")
    
    # 调试参数
    parser.add_argument("--log-level", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO",
                       help="日志级别")
    
    # ✨ 修改：逐个样本模式现在是默认启用
    parser.add_argument("--no-sequential-mode", dest="sequential_mode", action="store_false",
                       default=True,
                       help="禁用逐个样本模式，使用批量加载模式")
    parser.add_argument("--sequential-mode", dest="sequential_mode", action="store_true",
                       help="使用逐个样本模式（默认启用，内存友好）")
    
    # ✨ 修改：双塔并行检索现在默认禁用
    parser.add_argument("--parallel-towers", dest="parallel_towers", action="store_true",
                       default=False,
                       help="启用双塔并行检索")
    parser.add_argument("--no-parallel-towers", dest="parallel_towers", action="store_false",
                       help="禁用双塔并行检索，使用串行模式（默认）")

    # 重排序器参数 - 支持所有类型
    parser.add_argument('--reranker-type',
                       choices=['baai', 'qwen', 'jina', 'qwen-sili', 'qwen-dashscope', 'gte-dashscope'],
                    #    default='baai',
                       default='jina',
                       help='重排序器类型:\n'
                            '  baai: BAAI BGE本地重排序器 (默认)\n'
                            '  qwen: Qwen本地重排序器\n'
                            '  jina: Jina本地重排序器\n'
                            '  qwen-sili: Qwen云端重排序器(Siliconflow)\n'
                            '  qwen-dashscope: Qwen云端重排序器(DashScope)\n'
                            '  gte-dashscope: GTE云端重排序器(DashScope)')
    
    parser.add_argument('--reranker-model',
                       help='自定义重排序器模型名称')
    
    args = parser.parse_args()
    
    # 设置日志级别
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    print("=" * 100)
    print("🏗️ LoCoMo双塔召回系统Benchmark测试")
    print("=" * 100)
    print(f"📁 增强图谱目录: {args.enhanced_graphs_dir}")
    print(f"📁 步骤3图谱目录: {args.step3_graphs_dir}")
    print(f"📄 QA数据集: {args.qa_dataset}")
    print(f"📊 输出目录: {args.output_dir}")
    print(f"🤖 答案生成模型: {args.llm_model}")
    print(f"🔍 答案评估模型: {args.llm_evaluate_model}")
    print(f"🔀 融合策略: {args.fusion_strategy}")
    print(f"⚖️  融合权重: 分层={args.hierarchical_weight}, 图={args.graph_weight}")
    print(f"🎯 检索配置: L0={args.topk_hierarchical_l0}, L1={args.topk_hierarchical_l1}, L2={args.topk_hierarchical_l2}")
    print(f"🎯 图检索配置: 语义={args.topk_similarity}, 实体关系={'禁用' if args.topk_graph == 0 else f'{args.topk_graph}'}")
    print(f"🔧 实体关系检索: {'禁用' if args.no_entity_relation else '启用'}")
    print(f"🔧 重排序器: {args.reranker_type}")  # 新增显示
    print(f"🧵 并发线程: {args.max_workers}")
    print(f"📋 处理模式: {'逐个样本模式（内存友好）' if args.sequential_mode else '批量加载模式'}")
    print(f"⚡ 双塔并行: {'启用' if args.parallel_towers else '禁用（串行）'}")
    
    # 新增：如果使用云端重排序器，检查API密钥
    if args.reranker_type == 'qwen-remote':
        api_key = os.getenv("CSTCLOUD_API_KEY")
        if api_key:
            print(f"✅ 云端重排序API: 已配置")
        else:
            print(f"⚠️  云端重排序API: 未配置 (请设置 CSTCLOUD_API_KEY)")
    
    if args.sample_ids:
        print(f"🎯 指定样本: {args.sample_ids}")
    if args.max_samples:
        print(f"📊 最大样本数: {args.max_samples}")
    
    try:
        # 初始化LLM客户端
        print("\n🔄 初始化LLM客户端...")
        llm_client = LLMClient(model_name=args.llm_model)
        llm_evaluate_client = LLMClient(model_name=args.llm_evaluate_model)
        print(f"✅ 答案生成客户端: {args.llm_model}")
        print(f"✅ 答案评估客户端: {args.llm_evaluate_model}")
        
        # 初始化重排序器配置 - 支持所有类型
        reranker_configs = {
            "baai": args.reranker_model if args.reranker_model and args.reranker_type == "baai" 
                    else "BAAI/bge-reranker-v2-m3",
            "qwen": args.reranker_model if args.reranker_model and args.reranker_type == "qwen" 
                    else "Qwen/Qwen3-Reranker-0.6B",
            "jina": args.reranker_model if args.reranker_model and args.reranker_type == "jina" 
                    else "jinaai/jina-reranker-v3",
            "qwen-sili": args.reranker_model if args.reranker_model and args.reranker_type == "qwen-sili" 
                         else "Qwen/Qwen3-Reranker-8B",
            "qwen-dashscope": args.reranker_model if args.reranker_model and args.reranker_type == "qwen-dashscope" 
                              else "qwen3-rerank",
            "gte-dashscope": args.reranker_model if args.reranker_model and args.reranker_type == "gte-dashscope" 
                             else "gte-rerank-v2"
        }
        
        # 创建全局重排序器管理器
        print(f"\n🔄 初始化重排序器管理器 ({args.reranker_type})...")
        from dev.retrieval.rerank_manager import RerankerManager
        global_reranker_manager = RerankerManager()
        
        # # 预加载重排序器（可选）
        # try:
        #     reranker = global_reranker_manager.get_reranker(
        #         reranker_type=args.reranker_type,
        #         model_name=reranker_configs[args.reranker_type]
        #     )
        #     print(f"✅ 重排序器预加载成功: {args.reranker_type}")
        # except Exception as e:
        #     print(f"⚠️  重排序器预加载失败: {e}")
        #     print(f"   将在使用时按需加载")
        
        # 创建双塔benchmark测试器
        print("\n🔄 初始化双塔Benchmark测试器...")
        
        fusion_weights = {
            "hierarchical": args.hierarchical_weight,
            "graph": args.graph_weight
        }
        
        benchmark = LoCoMoDualTowerBenchmark(
            enhanced_graphs_dir=args.enhanced_graphs_dir,
            step3_graphs_dir=args.step3_graphs_dir,
            qa_dataset_path=args.qa_dataset,
            llm_client=llm_client,
            llm_evaluate_client=llm_evaluate_client,
            output_dir=args.output_dir,
            use_entity_relation=not args.no_entity_relation,
            topk_hierarchical_l0=args.topk_hierarchical_l0,
            topk_hierarchical_l1=args.topk_hierarchical_l1,
            topk_hierarchical_l2=args.topk_hierarchical_l2,
            topk_similarity=args.topk_similarity,
            topk_graph=args.topk_graph,
            fusion_strategy=args.fusion_strategy,
            fusion_weights=fusion_weights,
            target_sample_ids=args.sample_ids,
            max_workers=args.max_workers,
            parallel_towers=args.parallel_towers,
            reranker_type=args.reranker_type,  # 新增
            reranker_configs=reranker_configs,  # 新增
            reranker_manager=global_reranker_manager  # 新增：传入管理器
        )
        
        print("✅ 双塔Benchmark测试器初始化完成")
        
        # 新增：如果topk_graph=0，给出提示
        if args.topk_graph == 0:
            print("💡 提示：实体关系检索已禁用（topk-graph=0）")
            print("💡 如需启用：使用 --topk-graph 5 --enable-entity-relation")

        # ✨ 修改：现在默认使用逐个样本模式
        use_sequential = args.sequential_mode
        
        # 如果用户手动禁用了sequential模式，给出提示
        if not use_sequential and args.sample_ids and len(args.sample_ids) > 2:
            logger.warning(f"⚠️  检测到多个样本ID ({len(args.sample_ids)}个) 且使用批量模式")
            logger.warning(f"⚠️  建议使用默认的 --sequential-mode 以节省内存")
            response = input("是否切换到逐个样本模式？(y/N): ")
            if response.lower() == 'y':
                use_sequential = True
        
        if use_sequential:
            print("\n📋 使用逐个样本模式（默认）")
            print("💡 优点：内存占用小，适合大量样本，增量保存")
            print("💡 如需批量模式：使用 --no-sequential-mode")
            
            # 直接运行逐个样本测试
            test_start_time = time.time()
            benchmark.run_dual_tower_benchmark(sequential_mode=True)
            test_time = time.time() - test_start_time
            
            print(f"✅ 所有样本测试完成，总耗时: {test_time:.2f}s")
        else:
            print("\n📋 使用批量加载模式")
            print("💡 注意：批量模式会一次性加载所有样本到内存")
            
            # 加载系统（包括预加载检索器）
            print("\n🔄 加载双塔检索系统...")
            start_time = time.time()
            benchmark.load_systems(max_samples=args.max_samples)
            load_time = time.time() - start_time
            print(f"✅ 系统加载完成（含预加载），耗时: {load_time:.2f}s")
            
            # 加载测试用例
            print("\n🔄 加载测试用例...")
            benchmark.load_test_cases()
            print(f"✅ 测试用例加载完成: {len(benchmark.test_cases)} 个")
            
            # 运行双塔benchmark测试
            print("\n🚀 开始运行双塔benchmark测试...")
            print("💡 提示：由于检索器已预加载，后续测试将更快执行")
            test_start_time = time.time()
            benchmark.run_dual_tower_benchmark(sequential_mode=False)
            test_time = time.time() - test_start_time
            
            print(f"✅ 双塔测试完成，耗时: {test_time:.2f}s")
        
        # 生成报告
        print("\n📊 生成双塔benchmark报告...")
        generated_files = benchmark.generate_dual_tower_report()
        
        # 显示关键统计
        print(f"\n📈 关键统计:")
        print(f"   🎯 成功测试: {benchmark.stats['successful_dual_tower']}")
        print(f"   ❌ 失败测试: {benchmark.stats['failed_retrievals']}")
        print(f"   📊 分层成功: {benchmark.stats['successful_hierarchical']}")
        print(f"   🕸️  图检索成功: {benchmark.stats['successful_graph']}")
        
        total_tests = benchmark.stats['successful_dual_tower'] + benchmark.stats['failed_retrievals']
        if total_tests > 0:
            success_rate = benchmark.stats['successful_dual_tower'] / total_tests
            print(f"   📈 总体成功率: {success_rate:.2%}")
        
        if not use_sequential:
            total_time = load_time + test_time
            print(f"\n⏱️  总耗时: {total_time:.2f}s")
        else:
            print(f"\n⏱️  总耗时: {test_time:.2f}s")
        
        # 使用返回的文件路径
        if generated_files:
            print(f"\n📋 输出文件:")
            print(f"   主报告: {generated_files['main_report']}")
            print(f"   可读报告: {generated_files['readable_report']}")
            print(f"   检索详情(JSON): {generated_files['retrieval_details_json']}")
            print(f"   检索详情(可读): {generated_files['retrieval_details_readable']}")
            print(f"   时间戳: {generated_files['timestamp']}")
        
        print("\n✅ 双塔召回系统Benchmark测试完成!")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n❌ 用户中断测试")
        return 1
    except Exception as e:
        print(f"\n❌ 双塔Benchmark测试失败: {e}")
        logger.error(f"详细错误信息: {traceback.format_exc()}")
        return 1
    finally:
        # 清理资源
        try:
            cleanup_evaluation_models()
        except Exception as e:
            logger.warning(f"清理资源失败: {e}")


if __name__ == "__main__":
    sys.exit(main())
