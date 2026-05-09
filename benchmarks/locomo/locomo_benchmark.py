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
project_root = Path(__file__).parent.parent.parent
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
                logger.warning(f" : {sample_id}")
        
        logger.info(f"Sample: {len(hierarchical_available)} ")
        
        # Sample
        final_samples = list(set(available_samples) & set(hierarchical_available))
        logger.info(f"Sample: {len(final_samples)}  - {final_samples}")
        
        if not final_samples:
            raise RuntimeError("No sample found that supports both retrieval systems")
        
        self.available_samples = final_samples
        self.stats["total_samples_loaded"] = len(final_samples)

        # Pre-load
        logger.info("\n Pre-load...")
        self.preload_retrievers()
        logger.info("Pre-load\n")

    def _is_sample_available(self, sample_id: str) -> bool:
        """Check whether a sample is available in both graph systems."""
        # step3_graphs_dir
        step3_dir = self.step3_graphs_dir / sample_id
        step3_semantic_map = step3_dir / "semantic_map_data" / "semantic_map.json"
        
        if not step3_dir.exists():
            logger.warning(f" {sample_id} : {step3_dir}")
            return False
        
        if not step3_semantic_map.exists():
            logger.warning(f" {sample_id} semantic_map.json")
            return False
        
        # enhanced_graphs_dir_enhanced
        enhanced_dir = self.enhanced_graphs_dir / sample_id
        hierarchical_overview = enhanced_dir / "hierarchical_overview.json"
        enhanced_semantic_map = enhanced_dir / "semantic_map_data" / "semantic_map.json"
        
        if not enhanced_dir.exists():
            logger.warning(f" {sample_id} : {enhanced_dir}")
            return False
        
        if not hierarchical_overview.exists():
            logger.warning(f" {sample_id} hierarchical_overview.json")
            return False
        
        if not enhanced_semantic_map.exists():
            logger.warning(f" {sample_id} semantic_map.json")
            return False
        
        logger.debug(f"{sample_id} graphs")
        return True

    def _load_single_sample_hierarchical(self, sample_id: str):
        """Load hierarchical data for a single sample (new dataset format)."""
        try:
            # _enhanced
            enhanced_dir = self.enhanced_graphs_dir / sample_id
            
            if not enhanced_dir.exists():
                logger.error(f"  {sample_id} hierarchical graph directory not found: {enhanced_dir}")
                raise FileNotFoundError(f"hierarchical graph directory not found: {enhanced_dir}")
            
            # 
            hierarchical_overview = enhanced_dir / "hierarchical_overview.json"
            if not hierarchical_overview.exists():
                logger.warning(f" {sample_id} hierarchical_overview.json")
            
            # 
            graph, hierarchical_interface = self.hierarchical_tester.load_enhanced_conversation_graph(
                sample_id, str(self.enhanced_graphs_dir)
            )
            
            self.hierarchical_interfaces[sample_id] = hierarchical_interface
            self.loaded_graphs[sample_id] = graph
            
            logger.info(f"{sample_id} hierarchical interface loaded")
            
        except Exception as e:
            logger.error(f" {sample_id} failed: {e}")
            raise
    
    def _get_available_samples(self) -> List[str]:
        """Return available sample IDs, correctly distinguishing graph directory types."""
        available_samples_step3 = []
        available_samples_enhanced = []
        
        # step3_graphs_dirSample
        if self.step3_graphs_dir.exists():
            for item in self.step3_graphs_dir.iterdir():
                if item.is_dir() and item.name.startswith("conv-"):
                    # semantic_map.json
                    semantic_map_file = item / "semantic_map_data" / "semantic_map.json"
                    if semantic_map_file.exists():
                        available_samples_step3.append(item.name)
                        logger.debug(f"Sample: {item.name}")
        
        # enhanced_graphs_dirSample_enhanced
        if self.enhanced_graphs_dir.exists():
            for item in self.enhanced_graphs_dir.iterdir():
                if item.is_dir() and item.name.startswith("conv-"):
                    # _enhanced
                    sample_id = item.name
                    
                    # hierarchical_overview.json
                    hierarchical_overview = item / "hierarchical_overview.json"
                    semantic_map_file = item / "semantic_map_data" / "semantic_map.json"
                    
                    if hierarchical_overview.exists() and semantic_map_file.exists():
                        available_samples_enhanced.append(sample_id)
                        logger.debug(f"Sample: {sample_id}")
                    else:
                        logger.warning(f" Sample {sample_id} ")
        
        # 
        final_samples = list(set(available_samples_step3) & set(available_samples_enhanced))
        
        logger.info(f"Sample: ={len(available_samples_step3)}, "
                f"hierarchical={len(available_samples_enhanced)}, "
                f"交集={len(final_samples)}")
        
        if len(final_samples) < len(available_samples_step3) or len(final_samples) < len(available_samples_enhanced):
            missing_in_step3 = set(available_samples_enhanced) - set(available_samples_step3)
            missing_in_enhanced = set(available_samples_step3) - set(available_samples_enhanced)
            
            if missing_in_step3:
                logger.warning(f" Sample: {missing_in_step3}")
            if missing_in_enhanced:
                logger.warning(f" Sample: {missing_in_enhanced}")
        
        return sorted(final_samples)
    
    def load_test_cases(self, sample_id: Optional[str] = None):
        """Load test cases from the QA dataset.

        Args:
            sample_id: If specified, load only test cases for this sample.
        """
        logger.info(f"{'Sample' if sample_id else ''}...")
        
        #  Sample
        if sample_id:
            self.test_cases = []
        
        try:
            with open(self.qa_dataset_path, 'r', encoding='utf-8') as f:
                qa_data = json.load(f)
            
            for item in qa_data:
                item_sample_id = item["sample_id"]
                
                #  SampleSample
                if sample_id and item_sample_id != sample_id:
                    continue
                
                # Sample
                if not sample_id and item_sample_id not in self.available_samples:
                    continue
                
                qa_list = item.get("qa", [])
                for i, qa_item in enumerate(qa_list):
                    # 
                    if not isinstance(qa_item, dict) or "question" not in qa_item:
                        continue
                    
                    question = qa_item["question"]
                    category = qa_item.get("category", 1)
                    
                    # 
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
            
            #  Sample
            if not sample_id:
                self.stats["total_test_cases"] = len(self.test_cases)
            
            # Statistics
            if sample_id:
                logger.info(f"Sample {sample_id} loaded: {len(self.test_cases)} ")
            else:
                category_counts = {}
                sample_counts = {}
                for case in self.test_cases:
                    category_counts[case["category"]] = category_counts.get(case["category"], 0) + 1
                    sample_counts[case["sample_id"]] = sample_counts.get(case["sample_id"], 0) + 1
                
                logger.info(f"loaded: {len(self.test_cases)} ")
                logger.info(f" : {category_counts}")
                logger.info(f" Sample: {dict(list(sample_counts.items())[:5])}..." + 
                        (f" (共{len(sample_counts)}个Sample)" if len(sample_counts) > 5 else ""))
            
        except Exception as e:
            logger.error(f"failed: {e}")
            raise
    
    def run_dual_tower_benchmark(self, sequential_mode: bool = False):
        """Run the dual-tower retrieval benchmark.

        Args:
            sequential_mode: Use per-sample mode (suitable for many samples
                or memory-constrained environments).
        """
        logger.info(" benchmark...")
        
        #  Sample
        if sequential_mode:
            logger.info(" Sample + ")
            
            #  Sample - 
            if self.target_sample_ids:
                # 
                if isinstance(self.target_sample_ids, list):
                    samples_to_test = self.target_sample_ids
                else:
                    # set
                    samples_to_test = list(self.target_sample_ids)
                    logger.warning(f" SampleID")
            else:
                samples_to_test = self._get_available_samples()
            
            logger.info(f"  {len(samples_to_test)} Sample")
            logger.info(f": {samples_to_test}")
            
            # 
            self._initialize_progress_tracking(samples_to_test)
            
            #  Sample
            for i, sample_id in enumerate(samples_to_test, 1):
                logger.info(f"\n{'='*80}")
                logger.info(f"[{i}/{len(samples_to_test)}] Sample: {sample_id}")
                logger.info(f"{'='*80}")
                
                try:
                    sample_results = self.run_single_sample_benchmark(sample_id)
                    logger.info(f" Sample {sample_id}  {len(sample_results)} ")
                except Exception as e:
                    logger.error(f" Sample {sample_id} failed: {e}")
                    self._mark_sample_failed(sample_id, str(e))
                    continue
            
            logger.info(f"\n{'='*80}")
            logger.info("Sample")
            logger.info(f"{'='*80}")
            
            #  
            self._generate_final_summary_from_incremental()
            
            return
        
        # Sample
        if not self.test_cases:
            raise RuntimeError("No test cases loaded; call load_test_cases() first")
        
        total_tests = len(self.test_cases)
        logger.info(f": {total_tests}")
        
        if self.max_workers == 1:
            self._run_single_threaded_dual_tower_tests()
        else:
            self._run_multi_threaded_dual_tower_tests()
        
        logger.info("benchmark")
        
        #  Sample
        if self.test_results:
            logger.info("\n Sample...")
            self._generate_sample_reports_from_batch()

    def run_single_sample_benchmark(self, sample_id: str) -> List[DualTowerRetrievalResult]:
        """Run benchmark for a single sample, saving results immediately.

        Args:
            sample_id: The sample ID to test.

        Returns:
            List of test results for this sample.
        """
        logger.info(f" Sample: {sample_id}")
        
        sample_results = []
        
        try:
            # 1. Sample
            logger.info(f" Sample {sample_id} ...")
            self.load_systems(sample_id=sample_id)
            
            # 2. Sample
            logger.info(f"Sample {sample_id} ...")
            self.load_test_cases(sample_id=sample_id)
            
            if not self.test_cases:
                logger.warning(f" Sample {sample_id} ")
                return sample_results
            
            logger.info(f" Sample {sample_id}  {len(self.test_cases)} ")
            
            # 3. 
            for test_case in tqdm(self.test_cases, desc=f"测试 {sample_id}"):
                try:
                    result = self._run_single_dual_tower_test(test_case)
                    if result:
                        sample_results.append(result)
                        self.test_results.append(result)
                        if result.evaluation_success:
                            self.stats["successful_dual_tower"] += 1
                        else:
                            self.stats["failed_retrievals"] += 1
                    else:
                        self.stats["failed_retrievals"] += 1
                        
                except Exception as e:
                    self.stats["failed_retrievals"] += 1
                    logger.error(f"failed: {test_case['question_id']} - {e}")
                    continue
            
            logger.info(f"Sample {sample_id} : built={len([r for r in sample_results if r.evaluation_success])}, "
                    f"failed={len([r for r in sample_results if not r.evaluation_success])}")
            
            #  4. Sample
            self._save_sample_results_incrementally(sample_id, sample_results)
            
            return sample_results
            
        except Exception as e:
            logger.error(f"Sample {sample_id} failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # logger.debug(traceback.format_exc())
            return sample_results
        
        finally:
            # 5. Sample
            logger.info(f" Sample {sample_id} ...")
            self.unload_sample(sample_id)

    def _save_sample_results_incrementally(self, sample_id: str, sample_results: List[DualTowerRetrievalResult]):
        """Incrementally save a single sample's results and generate a sample-level report."""
        if not sample_results:
            logger.warning(f"Sample {sample_id} ")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        #  Sample
        successful_results = [r for r in sample_results if r.evaluation_success]
        
        # 
        performance_metrics = {}
        timing_metrics = {}
        system_success_rates = {}
        category_performance = {}
        parallel_performance = {}
        
        if successful_results:
            # 
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
            
            # 
            hierarchical_times = [r.hierarchical_retrieval_time for r in successful_results]
            graph_times = [r.graph_retrieval_time for r in successful_results]
            generation_times = [r.generation_time for r in successful_results]
            
            timing_metrics = {
                "avg_hierarchical_time": float(np.mean(hierarchical_times)),
                "avg_graph_time": float(np.mean(graph_times)),
                "avg_generation_time": float(np.mean(generation_times)),
                "avg_total_time": float(np.mean([h+g+gen for h, g, gen in zip(hierarchical_times, graph_times, generation_times)]))
            }
            
            #  
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
            
            # 
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
            
            # 
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
        
        #  Sample
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
                    #  
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
        
        # 1. SampleJSON
        sample_file = self.output_dir / f"sample_{sample_id}_{timestamp}.json"
        
        with open(sample_file, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f" Sample {sample_id} : {sample_file}")
        
        #  2. Sample
        readable_report_file = self._generate_sample_readable_report(
            sample_id, sample_results, timestamp, parallel_performance
        )
        logger.info(f" Sample {sample_id} : {readable_report_file}")
        
        # 3. 
        cumulative_file = self.output_dir / "cumulative_results.jsonl"
        
        with open(cumulative_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(sample_data, ensure_ascii=False) + '\n')
        
        logger.info(f" Sample {sample_id} ")
        
        # 4. 
        self._update_progress_file(sample_id, len(sample_results), 
                                len([r for r in sample_results if r.evaluation_success]))

    def _generate_sample_reports_from_batch(self):
        """Generate per-sample reports from batch test results.

        Called in batch mode; produces a separate report file for each sample.
        """
        logger.info("Sample...")
        
        # SampleID
        from collections import defaultdict
        sample_grouped_results = defaultdict(list)
        
        for result in self.test_results:
            sample_grouped_results[result.sample_id].append(result)
        
        logger.info(f"  {len(sample_grouped_results)} Sample")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        generated_count = 0
        
        # Sample
        for sample_id, sample_results in sample_grouped_results.items():
            try:
                logger.info(f" Sample {sample_id}  ({len(sample_results)} )...")
                
                # Sample
                successful_results = [r for r in sample_results if r.evaluation_success]
                
                # 
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
                
                # 1. JSON
                sample_data = self._build_sample_data_structure(
                    sample_id, sample_results, successful_results, parallel_performance
                )
                
                sample_json_file = self.output_dir / f"batch_sample_{sample_id}_{timestamp}.json"
                with open(sample_json_file, 'w', encoding='utf-8') as f:
                    json.dump(sample_data, f, ensure_ascii=False, indent=2)
                
                logger.info(f"   JSON: {sample_json_file.name}")
                
                # 2. 
                readable_file = self._generate_sample_readable_report(
                    sample_id, sample_results, timestamp, parallel_performance
                )
                
                logger.info(f"   : {readable_file.name}")
                
                generated_count += 1
                
            except Exception as e:
                logger.error(f"Sample {sample_id} failed: {e}")
                continue
        
        logger.info(f"\n Sample: {generated_count}/{len(sample_grouped_results)} Sample")
        
        # 
        summary_file = self.output_dir / f"batch_samples_summary_{timestamp}.txt"
        self._generate_batch_samples_summary(sample_grouped_results, summary_file)
        logger.info(f"Sample: {summary_file}")


    def _build_sample_data_structure(self, 
                                    sample_id: str, 
                                    sample_results: List[DualTowerRetrievalResult],
                                    successful_results: List[DualTowerRetrievalResult],
                                    parallel_performance: Dict[str, Any]) -> Dict[str, Any]:
        """Build sample data structure for JSON reports.

        Reuses logic from _save_sample_results_incrementally.
        """
        # 
        performance_metrics = {}
        timing_metrics = {}
        system_success_rates = {}
        category_performance = {}
        
        if successful_results:
            # 
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
            
            # 
            hierarchical_times = [r.hierarchical_retrieval_time for r in successful_results]
            graph_times = [r.graph_retrieval_time for r in successful_results]
            generation_times = [r.generation_time for r in successful_results]
            
            timing_metrics = {
                "avg_hierarchical_time": float(np.mean(hierarchical_times)),
                "avg_graph_time": float(np.mean(graph_times)),
                "avg_generation_time": float(np.mean(generation_times)),
                "avg_total_time": float(np.mean([h+g+gen for h, g, gen in zip(hierarchical_times, graph_times, generation_times)]))
            }
            
            # 
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
            
            # 
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
        
        # 
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
                "processing_mode": "batch",
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
        """Generate batch sample processing summary report."""
        lines = []
        
        lines.append("=" * 100)
        lines.append("batch mode - Sample处理摘要")
        lines.append("=" * 100)
        lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"处理Sample数: {len(sample_grouped_results)}")
        lines.append("=" * 100)
        
        lines.append(f"\n{'SampleID':<15} {'测试数':>8} {'built数':>8} {'failed数':>8} {'built率':>10} {'平均F1':>10}")
        lines.append("-" * 80)
        
        for sample_id, results in sorted(sample_grouped_results.items()):
            test_count = len(results)
            success_count = len([r for r in results if r.evaluation_success])
            failed_count = test_count - success_count
            success_rate = (success_count / test_count * 100) if test_count > 0 else 0
            
            # F1
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
        
        # 
        print('\n'.join(lines))

    def _generate_sample_readable_report(self, 
                            sample_id: str, 
                            sample_results: List[DualTowerRetrievalResult], 
                            timestamp: str,
                            parallel_performance: Dict[str, Any] = None) -> Path:
        """
        为单个Sample生成可读性报告（包含并行性能统计）
        """
        lines = []
        
        lines.append("=" * 100)
        lines.append(f"Sample {sample_id} - 双塔召回测试报告")
        lines.append("=" * 100)
        lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"SampleID: {sample_id}")
        lines.append(f"测试数量: {len(sample_results)}")
        lines.append(f"并行模式: {'enabled' if self.parallel_towers else 'disabled'}")
        lines.append("=" * 100)
        
        # Sample
        successful_count = len([r for r in sample_results if r.evaluation_success])
        failed_count = len(sample_results) - successful_count
        
        # 
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
            lines.append(f"   ✓ built测试: {successful_count} ({successful_count/len(sample_results)*100:.1f}%)")
            lines.append(f"   ✗ failed测试: {failed_count} ({failed_count/len(sample_results)*100:.1f}%)")
            
            lines.append(f"\n🎯 性能指标:")
            lines.append(f"   - 平均F1分数: {avg_f1:.3f}")
            lines.append(f"   - 平均语义相似度: {avg_semantic:.3f}")
            lines.append(f"   - 平均LLM准确率: {avg_llm:.3f}")
            lines.append(f"   - 平均精确匹配: {avg_exact_match:.3f}")
            lines.append(f"   - 平均confidence: {avg_confidence:.3f}")
            
            lines.append(f"\n⏱️  时间性能:")
            lines.append(f"   - 平均hierarchical retrieval: {avg_hier_time:.3f}s")
            lines.append(f"   - 平均图检索: {avg_graph_time:.3f}s")
            lines.append(f"   - 平均answer generation: {avg_gen_time:.3f}s")
            lines.append(f"   - 平均总时间: {avg_hier_time + avg_graph_time + avg_gen_time:.3f}s")
            
            #  
            if self.parallel_towers and parallel_performance and parallel_performance.get("parallel_enabled"):
                lines.append(f"\n⚡ 并行性能统计:")
                lines.append(f"   - 平均并行实际时间: {parallel_performance['avg_parallel_actual_time']:.3f}s")
                lines.append(f"   - 平均理论串行时间: {parallel_performance['avg_sequential_theory_time']:.3f}s")
                lines.append(f"   - 平均加速比: {parallel_performance['avg_speedup_ratio']:.2f}x")
                lines.append(f"   - 平均节省时间: {parallel_performance['avg_time_saved']:.3f}s")
                lines.append(f"   - 累计节省时间: {parallel_performance['total_time_saved']:.3f}s")
                lines.append(f"   - 加速比范围: {parallel_performance['min_speedup']:.2f}x ~ {parallel_performance['max_speedup']:.2f}x")
                
                # 
                if parallel_performance['avg_sequential_theory_time'] > 0:
                    efficiency = (parallel_performance['avg_time_saved'] / parallel_performance['avg_sequential_theory_time']) * 100
                    lines.append(f"   - 效率提升: {efficiency:.1f}%")
            
            lines.append(f"\n🔄 系统built率:")
            lines.append(f"   - hierarchical retrieval: {hier_success}/{len(valid_results)} ({hier_success/len(valid_results)*100:.1f}%)")
            lines.append(f"   - 图检索: {graph_success}/{len(valid_results)} ({graph_success/len(valid_results)*100:.1f}%)")
            lines.append(f"   - 双塔同时built: {both_success}/{len(valid_results)} ({both_success/len(valid_results)*100:.1f}%)")
            
            # LLM
            category_stats = defaultdict(lambda: {"count": 0, "f1_scores": [], "llm_scores": [], "success": 0})
            for r in valid_results:
                cat = r.category
                category_stats[cat]["count"] += 1
                category_stats[cat]["f1_scores"].append(r.evaluation_scores.get("token_f1", 0.0))
                category_stats[cat]["llm_scores"].append(r.evaluation_scores.get("llm_accuracy", 0.0))
                if r.evaluation_success:
                    category_stats[cat]["success"] += 1
            
            if category_stats:
                lines.append(f"\n📋 Category统计:")
                category_names = {
                    1: "多跳question", 
                    2: "时间question", 
                    3: "开放域question", 
                    4: "单跳question", 
                    5: "对抗性question"
                }
                
                for cat, stats in sorted(category_stats.items()):
                    cat_name = category_names.get(cat, f"Category{cat}")
                    avg_cat_f1 = np.mean(stats["f1_scores"]) if stats["f1_scores"] else 0.0
                    avg_cat_llm = np.mean(stats["llm_scores"]) if stats["llm_scores"] else 0.0
                    success_rate = stats["success"] / stats["count"] * 100 if stats["count"] > 0 else 0
                    
                    lines.append(f"\n   {cat_name}:")
                    lines.append(f"     - 测试数: {stats['count']}")
                    lines.append(f"     - 平均F1: {avg_cat_f1:.3f}")
                    lines.append(f"     - 平均LLM准确率: {avg_cat_llm:.3f}")
                    lines.append(f"     - built率: {success_rate:.1f}%")
        else:
            lines.append(f"\n⚠️  警告: 所有 {len(sample_results)} 个测试都failed了")
        
        # 
        lines.append(f"\n{'='*100}")
        lines.append(f"详细测试结果")
        lines.append(f"{'='*100}")
        
        category_names = {1: "多跳", 2: "时间", 3: "开放域", 4: "单跳", 5: "对抗性"}
        
        for i, result in enumerate(sample_results, 1):
            lines.append(f"\n{'-'*100}")
            lines.append(f"测试 {i}/{len(sample_results)}")
            lines.append(f"{'-'*100}")
            
            cat_name = category_names.get(result.category, f"Category{result.category}")
            lines.append(f"Category: {cat_name} (Category {result.category})")
            
            lines.append(f"\nquestion:")
            lines.append(f"  {result.question}")
            
            lines.append(f"\n标准answer:")
            lines.append(f"  {result.expected_answer}")
            
            lines.append(f"\nGenerate answer:")
            lines.append(f"  {result.final_answer}")
            
            lines.append(f"\nconfidence: {result.confidence_score:.3f}")
            
            # 
            hier_status = "✓" if result.hierarchical_context.get("hierarchical_enabled", False) else "✗"
            graph_status = "✓" if len(result.graph_retrieved_units) > 0 else "✗"
            lines.append(f"\n系统状态:")
            lines.append(f"  hierarchical retrieval: {hier_status} | 图检索: {graph_status}")
            
            # 
            if result.evaluation_success:
                scores = result.evaluation_scores
                lines.append(f"\nevaluation分数:")
                lines.append(f"  - F1: {scores.get('token_f1', 0):.3f}")
                lines.append(f"  - 语义相似度: {scores.get('semantic_similarity', 0):.3f}")
                lines.append(f"  - 精确匹配: {scores.get('exact_match', 0):.3f}")
                if 'llm_accuracy' in scores:
                    lines.append(f"  - LLM准确率: {scores.get('llm_accuracy', 0):.3f}")
            else:
                lines.append(f"\n⚠️  evaluationfailed")
            
            # 
            lines.append(f"\n时间统计:")
            lines.append(f"  分层: {result.hierarchical_retrieval_time:.3f}s | "
                        f"图: {result.graph_retrieval_time:.3f}s | "
                        f"生成: {result.generation_time:.3f}s")
            
            #  
            if self.parallel_towers and hasattr(result, 'parallel_actual_time'):
                lines.append(f"  并行统计: 实际={result.parallel_actual_time:.3f}s | "
                            f"理论串行={result.parallel_sequential_theory:.3f}s | "
                            f"加速比={result.parallel_speedup_ratio:.2f}x")
        
        lines.append(f"\n{'='*100}")
        lines.append(f"Sample {sample_id} 报告结束")
        lines.append(f"{'='*100}")
        
        # 
        report_file = self.output_dir / f"sample_{sample_id}_readable_{timestamp}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        # 
        print(f"\n{'='*80}")
        print(f" Sample {sample_id} ")
        print(f"{'='*80}")
        if valid_results:
            print(f" : {successful_count}/{len(sample_results)} ({successful_count/len(sample_results)*100:.1f}%)")
            print(f" F1: {avg_f1:.3f} | : {avg_semantic:.3f} | LLM: {avg_llm:.3f}")
            print(f"⏱  : {avg_hier_time + avg_graph_time + avg_gen_time:.2f}s")
            
            #  
            if self.parallel_towers and parallel_performance and parallel_performance.get("parallel_enabled"):
                print(f" : {parallel_performance['avg_speedup_ratio']:.2f}x, "
                    f"节省: {parallel_performance['total_time_saved']:.2f}s")
            
            print(f" : {both_success}/{len(valid_results)} ({both_success/len(valid_results)*100:.1f}%)")
        else:
            print(f" failed")
        print(f" : {report_file}")
        print(f"{'='*80}\n")
        
        return report_file
    
    # def _save_sample_results_incrementally(self, sample_id: str, sample_results: List[DualTowerRetrievalResult]):
    #     """
    #     Sample
        
    #     Args:
    #         sample_id: SampleID
    #         sample_results: Sample
    #     """
    #     if not sample_results:
    #         logger.warning(f"Sample {sample_id} ")
    #         return
        
    #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    #     # 1. Sample
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
        
    #     logger.info(f" Sample {sample_id} : {sample_file}")
        
    #     # 2. 
    #     cumulative_file = self.output_dir / "cumulative_results.jsonl"
        
    #     with open(cumulative_file, 'a', encoding='utf-8') as f:
    #         f.write(json.dumps(sample_data, ensure_ascii=False) + '\n')
        
    #     logger.info(f" Sample {sample_id} ")
        
    #     # 3. 
    #     self._update_progress_file(sample_id, len(sample_results), 
    #                             len([r for r in sample_results if r.evaluation_success]))

    def unload_sample(self, sample_id: str):
        """Unload a single sample's resources, including GPU and memory cleanup.

        Args:
            sample_id: The sample ID to unload.
        """
        logger.info(f" Sample {sample_id} ...")
        
        # 
        if sample_id in self.hierarchical_interfaces:
            del self.hierarchical_interfaces[sample_id]
            logger.debug(f"   ")
        
        # 
        if sample_id in self.loaded_graphs:
            del self.loaded_graphs[sample_id]
            logger.debug(f"   ")
        
        # 
        if sample_id in self.graph_benchmark.multi_retrievers:
            retriever = self.graph_benchmark.multi_retrievers[sample_id]
            # 
            if hasattr(retriever, 'clear_cache'):
                retriever.clear_cache()
            del self.graph_benchmark.multi_retrievers[sample_id]
            logger.debug(f"   ")
        
        if sample_id in self.graph_benchmark.entity_relation_retrievers:
            del self.graph_benchmark.entity_relation_retrievers[sample_id]
            logger.debug(f"   ")
        
        if sample_id in self.graph_benchmark.semantic_graphs:
            del self.graph_benchmark.semantic_graphs[sample_id]
            logger.debug(f"   ")
        
        # Sample
        if hasattr(self, 'available_samples') and sample_id in self.available_samples:
            self.available_samples.remove(sample_id)
        
        #  GPU
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug(f"   GPU")
        except ImportError:
            pass
        
        #  Python
        import gc
        gc.collect()
        logger.debug(f"   ")
        
        logger.info(f"Sample {sample_id} ")

    def _initialize_progress_tracking(self, samples_to_test: List[str]):
        """Initialize progress tracking."""
        progress_file = self.output_dir / "test_progress.json"
        
        progress_data = {
            "_summary": {
                "total_samples_planned": len(samples_to_test),
                "total_samples_completed": 0,
                "completion_rate": 0.0,
                "start_time": datetime.now().isoformat(),
                "sample_order": samples_to_test
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
        
        logger.info(f" : {progress_file}")

    def _mark_sample_failed(self, sample_id: str, error: str):
        """Mark a sample test as failed."""
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
        """Generate a final summary report from incremental results."""
        logger.info(" ...")
        
        # Sample
        sample_files = sorted(self.output_dir.glob("sample_conv-*.json"))
        #   readable 
        sample_files = [f for f in sample_files if "_readable_" not in f.name]
        
        if not sample_files:
            logger.warning("Sample")
            return
        
        all_results = []
        sample_summaries = {}
        
        for sample_file in sample_files:
            try:
                with open(sample_file, 'r', encoding='utf-8') as f:
                    sample_data = json.load(f)
                    
                    #   sample_id
                    sample_info = sample_data.get("sample_info", {})
                    sample_id = sample_info.get("sample_id", "unknown")
                    
                    # 
                    results = sample_data.get("results", [])
                    all_results.extend(results)
                    
                    #  Sample - 
                    sample_summaries[sample_id] = {
                        "test_count": sample_info.get("test_count", 0),
                        "successful_count": sample_info.get("successful_count", 0),
                        "failed_count": sample_info.get("failed_count", 0),
                        "timestamp": sample_info.get("timestamp", "unknown")
                    }
                    
            except Exception as e:
                logger.warning(f"Samplefailed {sample_file}: {e}")
                continue
        
        logger.info(f" {len(sample_files)} Sample {len(all_results)} ")
        
        # JSON
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
        
        logger.info(f" JSON: {summary_file}")
        
        # 
        readable_summary_file = self._generate_final_readable_summary(final_summary, timestamp)
        logger.info(f" : {readable_summary_file}")
        
        return summary_file, readable_summary_file

    def _generate_final_readable_summary(self, final_summary: Dict, timestamp: str) -> Path:
        """Generate final summary report (including parallel performance stats)."""
        lines = []
        
        lines.append("=" * 100)
        lines.append("LoCoMo双塔召回系统 - 最终汇总报告")
        lines.append("=" * 100)
        lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 100)
        
        test_info = final_summary["test_info"]
        lines.append(f"\n📊 测试概况:")
        lines.append(f"   - 测试Sample数: {test_info['total_samples']}")
        lines.append(f"   - 总测试数: {test_info['total_results']}")
        lines.append(f"   - built测试: {test_info['successful_results']} ({test_info['successful_results']/test_info['total_results']*100:.1f}%)")
        lines.append(f"   - failed测试: {test_info['failed_results']} ({test_info['failed_results']/test_info['total_results']*100:.1f}%)")
        lines.append(f"   - Fusion strategy: {test_info['fusion_strategy']}")
        lines.append(f"   - 融合weights: 分层={test_info['fusion_weights']['hierarchical']}, 图={test_info['fusion_weights']['graph']}")
        
        # Sample
        sample_summaries = final_summary.get("sample_summaries", {})
        if sample_summaries:
            lines.append(f"\n📋 各Sample测试摘要:")
            lines.append(f"\n{'SampleID':<15} {'测试数':>8} {'built数':>8} {'failed数':>8} {'built率':>10}")
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
        
        # 
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
            lines.append(f"   - 平均confidence: {avg_confidence:.3f}")
            
            lines.append(f"\n⏱️  平均时间性能:")
            lines.append(f"   - hierarchical retrieval: {avg_hier_time:.3f}s")
            lines.append(f"   - 图检索: {avg_graph_time:.3f}s")
            lines.append(f"   - answer generation: {avg_gen_time:.3f}s")
            lines.append(f"   - 总计: {avg_hier_time + avg_graph_time + avg_gen_time:.3f}s")
            
            #  
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
                
                # 
                total_parallel = np.sum(parallel_actual_times)
                total_sequential = np.sum(parallel_sequential_times)
                overall_efficiency = ((total_sequential - total_parallel) / total_sequential * 100) if total_sequential > 0 else 0
                lines.append(f"   - 整体效率提升: {overall_efficiency:.1f}%")
            
            lines.append(f"\n🔄 系统built率:")
            lines.append(f"   - hierarchical retrieval: {hier_success}/{len(valid_results)} ({hier_success/len(valid_results)*100:.1f}%)")
            lines.append(f"   - 图检索: {graph_success}/{len(valid_results)} ({graph_success/len(valid_results)*100:.1f}%)")
            lines.append(f"   - 双塔同时built: {both_success}/{len(valid_results)} ({both_success/len(valid_results)*100:.1f}%)")
            
            # 
            category_stats = defaultdict(lambda: {"count": 0, "f1_scores": [], "llm_scores": []})
            for r in valid_results:
                cat = r.get("category", 0)
                if cat:
                    category_stats[cat]["count"] += 1
                    category_stats[cat]["f1_scores"].append(safe_get_score(r, "token_f1"))
                    category_stats[cat]["llm_scores"].append(safe_get_score(r, "llm_accuracy"))
            
            if category_stats:
                lines.append(f"\n📋 Category性能:")
                category_names = {1: "多跳question", 2: "时间question", 3: "开放域question", 4: "单跳question", 5: "对抗性question"}
                
                for cat, stats in sorted(category_stats.items()):
                    if stats["count"] == 0:
                        continue
                    cat_name = category_names.get(cat, f"Category{cat}")
                    avg_cat_f1 = np.mean(stats["f1_scores"]) if stats["f1_scores"] else 0.0
                    avg_cat_llm = np.mean(stats["llm_scores"]) if stats["llm_scores"] else 0.0
                    
                    lines.append(f"\n   {cat_name} ({stats['count']}题):")
                    lines.append(f"     - 平均F1: {avg_cat_f1:.3f}")
                    lines.append(f"     - 平均LLM准确率: {avg_cat_llm:.3f}")
        
        lines.append(f"\n{'='*100}")
        lines.append(f"报告生成完成")
        lines.append(f"详细的Sample报告请查看: sample_*_readable_*.txt")
        lines.append(f"{'='*100}")
        
        # 
        summary_file = self.output_dir / f"final_summary_readable_{timestamp}.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        # 
        print('\n'.join(lines))
        
        return summary_file

    # def _generate_final_summary_from_incremental(self):
    #     """"""
    #     logger.info(" ...")
        
    #     # Sample
    #     sample_files = sorted(self.output_dir.glob("sample_*.json"))
        
    #     if not sample_files:
    #         logger.warning("Sample")
    #         return
        
    #     all_results = []
    #     for sample_file in sample_files:
    #         try:
    #             with open(sample_file, 'r', encoding='utf-8') as f:
    #                 sample_data = json.load(f)
    #                 all_results.extend(sample_data.get("results", []))
    #         except Exception as e:
    #             logger.warning(f"Samplefailed {sample_file}: {e}")
        
    #     # test_results
    #     # 
    #     logger.info(f" {len(sample_files)} Sample {len(all_results)} ")
        
    #     # 
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
        
    #     logger.info(f" : {summary_file}")

    def _update_progress_file(self, sample_id: str, test_count: int, success_count: int):
        """Update the test progress file.

        Args:
            sample_id: Sample ID.
            test_count: Number of tests.
            success_count: Number of successful builds.
        """
        progress_file = self.output_dir / "test_progress.json"
        
        # 
        progress_data = {}
        if progress_file.exists():
            try:
                with open(progress_file, 'r', encoding='utf-8') as f:
                    progress_data = json.load(f)
            except:
                progress_data = {}
        
        # 
        progress_data[sample_id] = {
            "status": "completed",
            "test_count": test_count,
            "success_count": success_count,
            "failed_count": test_count - success_count,
            "completed_at": datetime.now().isoformat()
        }
        
        # 
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
        
        # 
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f" : {total_completed}/{total_planned} Sample ({total_completed/total_planned*100:.1f}%)")
    
    def _run_single_threaded_dual_tower_tests(self):
        """Run dual-tower tests in single-threaded mode."""
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
                logger.error(f"failed: {test_case['sample_id']} - {e}")
                continue
    
    def _run_multi_threaded_dual_tower_tests(self):
        """Run dual-tower tests in multi-threaded mode."""
        logger.info(f" {self.max_workers} ")
        
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
                    logger.error(f"failed: {test_case['sample_id']} - {e}")
                    continue
    
    def _run_single_dual_tower_test(self, test_case: Dict[str, Any]) -> Optional[DualTowerRetrievalResult]:
        """Run a single dual-tower test (supports parallel retrieval).

        Args:
            test_case: Test case dict with sample_id, question, category, expected_answer.

        Returns:
            Test result or None.
        """
        sample_id = test_case["sample_id"]
        question = test_case["question"]
        category = test_case["category"]
        expected_answer = test_case["expected_answer"]
        
        logger.debug(f" : {sample_id} - {question[:50]}...")
        
        try:
            if self.parallel_towers:
                #  
                return self._run_parallel_dual_tower_test(
                    sample_id, question, category, expected_answer
                )
            else:
                # 
                return self._run_sequential_dual_tower_test(
                    sample_id, question, category, expected_answer
                )
                
        except Exception as e:
            logger.error(f"failed {sample_id}: {e}")
            logger.debug(traceback.format_exc())
            return None
        
    def _run_parallel_dual_tower_test(self, 
                                sample_id: str, 
                                question: str, 
                                category: int, 
                                expected_answer: str) -> Optional[DualTowerRetrievalResult]:
        """Execute dual-tower retrieval in parallel (core optimisation).

        Args:
            sample_id: Sample ID.
            question: Question text.
            category: Category code.
            expected_answer: Expected answer string.

        Returns:
            Test result or None.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import time
        
        # 
        hierarchical_result = {}
        graph_result = {}
        
        def run_hierarchical():
            """Execute hierarchical retrieval."""
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
                logger.error(f"failed {sample_id}: {e}")
                hierarchical_result['context'] = {
                    "hierarchical_enabled": False,
                    "error": str(e),
                    "hierarchical_context_text": "",
                    "retrieval_method": "failed"
                }
                hierarchical_result['time'] = 0.0
                hierarchical_result['success'] = False
        
        def run_graph():
            """Execute graph retrieval."""
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
                logger.error(f"failed {sample_id}: {e}")
                graph_result['units'] = []
                graph_result['details'] = {"method": "failed", "error": str(e)}
                graph_result['time'] = 0.0
                graph_result['success'] = False
        
        #  
        parallel_start = time.time()
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            # 
            future_hierarchical = executor.submit(run_hierarchical)
            future_graph = executor.submit(run_graph)
            
            # 
            for future in as_completed([future_hierarchical, future_graph]):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f": {e}")
        
        parallel_elapsed = time.time() - parallel_start
        
        # 
        if 'context' not in hierarchical_result or 'units' not in graph_result:
            logger.error(f": hierarchical={bool(hierarchical_result)}, graph={bool(graph_result)}")
            return None
        
        # 
        hierarchical_context = hierarchical_result['context']
        hierarchical_time = hierarchical_result['time']
        
        graph_units = graph_result['units']
        graph_details = graph_result['details']
        graph_time = graph_result['time']
        
        #  
        sequential_time = hierarchical_time + graph_time
        speedup_ratio = sequential_time / parallel_elapsed if parallel_elapsed > 0 else 0
        time_saved = sequential_time - parallel_elapsed
        
        logger.debug(f" : ={hierarchical_time:.3f}s, ={graph_time:.3f}s")
        logger.debug(f"   duration={parallel_elapsed:.3f}s, ={sequential_time:.3f}s")
        logger.debug(f"   ={speedup_ratio:.2f}x, ={time_saved:.3f}s")
        
        # 3. 
        fusion_start = time.time()
        answer_dict, confidence_score = self._fuse_and_generate_answer(
            question, category, hierarchical_context, graph_units, graph_details
        )
        fusion_time = time.time() - fusion_start
        
        # 
        final_answer = answer_dict.get("final_answer", "")
        reasoning_process = answer_dict.get("reasoning", "")
        
        # 4. reasoning
        evaluation_result = self._evaluate_dual_tower_result(
            question=question,
            expected_answer=expected_answer,
            generated_answer=final_answer,
            reasoning=reasoning_process,
            category=category
        )
        
        evaluation_scores = evaluation_result.get("evaluation_scores", {})
        evaluation_success = evaluation_result.get("evaluation_success", False)
        
        # 5. 
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
        
        #  
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
        """Execute dual-tower retrieval sequentially (original logic, for comparison and debugging).

        Args:
            sample_id: Sample ID.
            question: Question text.
            category: Category code.
            expected_answer: Expected answer string.

        Returns:
            Test result or None.
        """
        # 1. 
        hierarchical_start = time.time()
        hierarchical_context = self._run_hierarchical_retrieval(sample_id, question, category)
        hierarchical_time = time.time() - hierarchical_start

        hierarchical_success = hierarchical_context.get("hierarchical_enabled", False)
        if hierarchical_success:
            self.stats["successful_hierarchical"] += 1

        # 2. 
        graph_start = time.time()
        graph_results, graph_details = self._run_graph_retrieval(sample_id, question)
        graph_time = time.time() - graph_start

        graph_success = len(graph_results) > 0
        if graph_success:
            self.stats["successful_graph"] += 1

        logger.debug(f" : ={hierarchical_time:.3f}s, ={graph_time:.3f}s, "
                    f"总duration={hierarchical_time + graph_time:.3f}s")

        # 3. 
        fusion_start = time.time()
        answer_dict, confidence_score = self._fuse_and_generate_answer(
            question, category, hierarchical_context, graph_results, graph_details
        )
        fusion_time = time.time() - fusion_start

        # 
        final_answer = answer_dict.get("final_answer", "")
        reasoning_process = answer_dict.get("reasoning", "")

        # 4. reasoning
        evaluation_result = self._evaluate_dual_tower_result(
            question=question,
            expected_answer=expected_answer,
            generated_answer=final_answer,
            reasoning=reasoning_process,
            category=category
        )

        evaluation_scores = evaluation_result.get("evaluation_scores", {})
        evaluation_success = evaluation_result.get("evaluation_success", False)

        # 5. 
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
        """Run hierarchical retrieval (using cached interface)."""
        try:
            # 
            if sample_id not in self.hierarchical_interfaces:
                logger.warning(f"Pre-load: {sample_id}...")
                graph, hierarchical_interface = self.hierarchical_tester.load_enhanced_conversation_graph(
                    sample_id, str(self.enhanced_graphs_dir)
                )
                self.hierarchical_interfaces[sample_id] = hierarchical_interface
                self.loaded_graphs[sample_id] = graph
            
            hierarchical_interface = self.hierarchical_interfaces[sample_id]
            
            # 
            context_info = self.hierarchical_tester.retrieve_hierarchical_context(
                hierarchical_interface, question, sample_id, category,
                enable_graph_expansion=False,
                graph_expansion_hops=0
            )
            
            return context_info
            
        except Exception as e:
            logger.warning(f"failed {sample_id}: {e}")
            return {
                "hierarchical_enabled": False,
                "error": str(e),
                "hierarchical_context_text": "",
                "retrieval_method": "failed"
            }
    
    def _run_graph_retrieval(self, sample_id: str, question: str) -> Tuple[List[Any], Dict[str, Any]]:
        """Run knowledge-graph retrieval (using loaded retrievers)."""
        try:
            #  graph_benchmark.load_semantic_graphs() 
            if sample_id not in self.graph_benchmark.multi_retrievers:
                logger.warning(f": {sample_id}")
                return [], {"method": "failed", "error": "retriever_not_found"}
            
            # 
            multi_retriever = self.graph_benchmark.multi_retrievers[sample_id]
            entity_retriever = self.graph_benchmark.entity_relation_retrievers.get(sample_id)
            
            retrieved_units = []
            details = {}
            
            if self.use_entity_relation and entity_retriever:
                #  + 
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
                    rerank_method=self.reranker_type,  # 
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
                # 
                retrieved_units = multi_retriever.smart_search(
                    query=question,
                    methods=["bm25", "cosine_similarity", "splade"],
                    fusion_method="rrf",
                    rerank_method=self.reranker_type,  # 
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
            logger.warning(f"failed {sample_id}: {e}")
            return [], {"method": "failed", "error": str(e)}
    
    def _fuse_and_generate_answer(self, 
                             question: str, 
                             category: int,
                             hierarchical_context: Dict[str, Any], 
                             graph_results: List[Any], 
                             graph_details: Dict[str, Any]) -> Tuple[Dict[str, str], float]:
        """Fuse results from both towers and generate a final answer.

        Returns:
            Tuple of (dict with reasoning and final_answer, confidence score).
        """
        
        if self.fusion_strategy == "simple":
            return self._simple_fusion_generation(question, category, hierarchical_context, graph_results)
        elif self.fusion_strategy == "weighted":
            return self._weighted_fusion_generation(question, category, hierarchical_context, graph_results)
        elif self.fusion_strategy == "context_aware":
            return self._context_aware_fusion_generation(question, category, hierarchical_context, graph_results, graph_details)
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")
    
    def _context_aware_fusion_generation(self, 
                           question: str, 
                           category: int,
                           hierarchical_context: Dict[str, Any], 
                           graph_results: List[Any],
                           graph_details: Dict[str, Any]) -> Tuple[Dict[str, str], float]:
        """Context-aware fusion generation (recommended) — returns structured answer, optimised to remove redundancy."""
        
        # 
        prompt_parts = []
        
        # 
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
        
        # 
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
                
                # unit_content"Entity:"
                # _extract_graph_unit_content
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
        #     # 
        #     # prompt_parts.append(f"Method: {graph_details.get('retrieval_method', 'unknown')}")
        #     prompt_parts.append("")
            
        #     for i, (unit, score) in enumerate(graph_results[:10], 1):
        #         unit_content = self._extract_graph_unit_content(unit)
        #         # Score
        #         # prompt_parts.append(f"Graph Result {i} (Score: {score:.3f}):")
        #         prompt_parts.append(f"Graph Result {i}:")
                
        #         # 
        #         entity_type = getattr(unit, 'entity_type', 'Unknown')
        #         if entity_type != 'Unknown':
        #             prompt_parts.append(f"Entity: {unit_content[:200]} | Type: {entity_type}")
        #         else:
        #             prompt_parts.append(f"Entity: {unit_content[:200]}")
        #         prompt_parts.append("")
            
        #     # 
        #     # if graph_details.get("entity_extraction"):
        #     #     entities = graph_details["entity_extraction"]
        #     #     prompt_parts.append(f": {question}")
        #     #     prompt_parts.append(f":")
        #     #     ...
            
        # else:
        #     prompt_parts.append("=" * 80)
        #     prompt_parts.append("KNOWLEDGE GRAPH RESULTS")
        #     prompt_parts.append("=" * 80)
        #     prompt_parts.append("No relevant entities or relationships found in the knowledge graph.")
        
        # prompt_parts.append("")
        
        # 
        if category == 5:
            prompt_parts.append("=" * 80)
            prompt_parts.append("DUAL TOWER FUSION GUIDANCE")
            prompt_parts.append("=" * 80)
            # 
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
            # 
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
            # 
            raw_response = self.llm_client.generate_answer(
                prompt=full_prompt,
                temperature=0.1,
                max_tokens=2000,
                json_format=True
            )
            
            # JSON
            answer_dict = self._parse_structured_dual_tower_response(raw_response, category)
            
            # 
            confidence_score = self._calculate_dual_tower_confidence(
                hierarchical_enabled, len(graph_results), graph_details
            )
            
            return answer_dict, confidence_score
            
        except Exception as e:
            logger.error(f"failed: {e}")
            return {
                "reasoning": f"Generation failed: {str(e)}",
                "final_answer": "Unable to generate answer"
            }, 0.0

    def _parse_structured_dual_tower_response(self, raw_response: str, category: int) -> Dict[str, str]:
        """Parse dual-tower structured response."""
        try:
            # JSON
            import json
            parsed = json.loads(raw_response.strip())
            
            if isinstance(parsed, dict) and "reasoning" in parsed and "final_answer" in parsed:
                # 
                final_answer = self._post_process_dual_tower_answer(parsed["final_answer"], category)
                
                return {
                    "reasoning": parsed["reasoning"].strip(),
                    "final_answer": final_answer
                }
            else:
                raise ValueError("Invalid JSON format: missing required fields")
                
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"JSONfailed: {e}")
            
            # 
            return self._parse_text_dual_tower_response(raw_response, category)

    def _parse_text_dual_tower_response(self, raw_response: str, category: int) -> Dict[str, str]:
        """Fallback text parsing method — dual-tower version."""
        lines = raw_response.strip().split('\n')
        reasoning = ""
        final_answer = ""
        
        # 
        reasoning_keywords = ["reasoning", "analysis", "思考", "reasoning", "synthesis", "because", "since"]
        answer_keywords = ["answer", "final", "conclusion", "result", "answer", "结论"]
        
        current_section = "reasoning"
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 
            if any(keyword in line.lower() for keyword in answer_keywords):
                current_section = "answer"
                if ":" in line:
                    final_answer = line.split(":", 1)[1].strip()
                    continue
            
            # 
            if current_section == "reasoning":
                reasoning += line + " "
            else:
                final_answer += line + " "
        
        # 
        if not final_answer.strip() and reasoning.strip():
            sentences = reasoning.strip().split('.')
            if len(sentences) > 1:
                final_answer = sentences[-1].strip()
                reasoning = '.'.join(sentences[:-1]).strip()
        
        # 
        if not final_answer.strip():
            final_answer = raw_response.strip()
            reasoning = "Unable to parse structured reasoning from dual tower response"
        
        final_answer = self._post_process_dual_tower_answer(final_answer.strip(), category)
        
        return {
            "reasoning": reasoning.strip() or "No clear reasoning provided from dual tower fusion",
            "final_answer": final_answer
        }

    def _post_process_dual_tower_answer(self, answer: str, category: int) -> str:
        """Post-process dual-tower generated answer."""
        if not answer:
            return "No answer generated"
        
        answer = answer.strip()
        
        # 
        prefixes = ["Answer:", "ANSWER:", "Final Answer:", "Response:", "Based on"]
        for prefix in prefixes:
            if answer.startswith(prefix):
                answer = answer[len(prefix):].strip()
        
        # 
        if answer and not answer[0].isupper() and not answer[0].isdigit():
            answer = answer[0].upper() + answer[1:]
        
        # 
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
        """Simple fusion generation — returns structured result."""
        context_parts = []
        
        # 
        if hierarchical_context.get("hierarchical_enabled", False):
            hierarchical_text = hierarchical_context.get("hierarchical_context_text", "")
            if hierarchical_text:
                context_parts.append(f"Hierarchical Context:\n{hierarchical_text}")
        
        # 
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
        """Weighted fusion generation — returns structured result."""
        # simple
        return self._simple_fusion_generation(question, category, hierarchical_context, graph_results)
    
    def _get_dual_tower_category_guidance(self, category: int) -> str:
        """Get category-specific guidance for the dual-tower system."""
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
        """Compute confidence score for the dual-tower system."""
        base_confidence = 0.0
        
        # 
        if hierarchical_success:
            base_confidence += self.fusion_weights.get("hierarchical", 0.6)
        
        # 
        if graph_results_count > 0:
            graph_weight = self.fusion_weights.get("graph", 0.4)
            # 
            result_factor = min(graph_results_count / 10.0, 1.0)
            base_confidence += graph_weight * result_factor
        
        return min(base_confidence, 1.0)
    
    def _evaluate_dual_tower_result(self, 
                              question: str, 
                              expected_answer: str, 
                              generated_answer: str, 
                              reasoning: str,
                              category: int) -> Dict[str, Any]:
        """evaluation双塔结果 - 统一处理所有Category"""
        try:
            from benchmark_locomo.task_eval.evaluation import calculate_comprehensive_scores
            
            # 
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
            logger.error(f"failed: {e}")
            return {
                "evaluation_scores": {"error": str(e)},
                "evaluation_method": "failed",
                "evaluation_success": False
            }
        
    def generate_dual_tower_report(self) -> Dict[str, Path]:
        """Generate the dual-tower benchmark report.

        Returns:
            Dict mapping report type to the generated file path.
        """
        logger.info("benchmark...")

        if not self.test_results:
            logger.warning("")
            return {}
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 
        overall_stats = self._calculate_dual_tower_overall_stats()
        sample_performance = self._calculate_dual_tower_sample_performance()
        category_performance = self._calculate_dual_tower_category_performance()
        fusion_analysis = self._analyze_fusion_effectiveness()
        
        # 
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
                "reasoning_process": r.reasoning_process,  # added
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
        
        # JSON
        report_file = self.output_dir / f"dual_tower_benchmark_report_{timestamp}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(benchmark_report, f, ensure_ascii=False, indent=2)
        
        # 
        readable_report_file = self._generate_dual_tower_readable_report(benchmark_report, timestamp)
        
        logger.info(f"benchmark: {report_file}")

        # 2. 
        retrieval_files = self.generate_retrieval_details_report()

        logger.info(f"benchmark")
        logger.info(f"benchmark")
        
        # 
        return {
            "main_report": report_file,
            "readable_report": readable_report_file,
            "retrieval_details_json": retrieval_files.get("json"),
            "retrieval_details_readable": retrieval_files.get("readable"),
            "timestamp": timestamp
        }
    
    def _calculate_dual_tower_overall_stats(self) -> Dict[str, Any]:
        """Compute overall dual-tower statistics."""
        if not self.test_results:
            return {}
        
        valid_results = [r for r in self.test_results if r.evaluation_success]
        
        if not valid_results:
            return {"error": "no_valid_results"}
        
        # 
        f1_scores = [r.evaluation_scores.get("token_f1", 0.0) for r in valid_results]
        semantic_scores = [r.evaluation_scores.get("semantic_similarity", 0.0) for r in valid_results]
        llm_scores = [r.evaluation_scores.get("llm_accuracy", 0.0) for r in valid_results]
        exact_match_scores = [r.evaluation_scores.get("exact_match", 0.0) for r in valid_results]
        confidence_scores = [r.confidence_score for r in valid_results]
        
        # 
        hierarchical_times = [r.hierarchical_retrieval_time for r in valid_results]
        graph_times = [r.graph_retrieval_time for r in valid_results]
        generation_times = [r.generation_time for r in valid_results]
        
        # 
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
        """Compute per-sample performance."""
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
        """Compute per-category performance."""
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
        """Analyse fusion effectiveness."""
        valid_results = [r for r in self.test_results if r.evaluation_success]
        
        if not valid_results:
            return {}
        
        # 
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
        """Generate dual-tower readable report."""
        lines = []
        
        lines.append("=" * 100)
        lines.append("LoCoMo dual-tower retrieval benchmark报告")
        lines.append("=" * 100)
        
        # 
        info = benchmark_report["benchmark_info"]
        lines.append(f"\n📊 测试概况:")
        lines.append(f"   - 测试类型: {info['test_type']}")
        lines.append(f"   - Fusion strategy: {info['fusion_strategy']}")
        lines.append(f"   - 融合weights: 分层={info['fusion_weights']['hierarchical']}, 图检索={info['fusion_weights']['graph']}")
        lines.append(f"   - 总Sample数: {info['total_samples']}")
        lines.append(f"   - 总测试数: {info['total_tests_run']}")
        
        # 
        lines.append(f"\n🔧 系统配置:")
        lines.append(f"   - hierarchical retrieval: L0={info['hierarchical_config']['l0_top_k']}, L1={info['hierarchical_config']['l1_top_k']}, L2={info['hierarchical_config']['l2_top_k']}")
        lines.append(f"   - 图检索: 语义top-k={info['graph_config']['topk_similarity']}, 图top-k={info['graph_config']['topk_graph']}")
        lines.append(f"   - 实体关系检索: {info['graph_config']['use_entity_relation']}")
        
        # 
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
            lines.append(f"   - 平均confidence: {perf['avg_confidence']:.3f}")
            
            lines.append(f"\n⏱️  时间性能:")
            lines.append(f"   - hierarchical retrieval时间: {timing['avg_hierarchical_time']:.3f}s")
            lines.append(f"   - 图检索时间: {timing['avg_graph_time']:.3f}s")
            lines.append(f"   - answer generation时间: {timing['avg_generation_time']:.3f}s")
            lines.append(f"   - 总平均时间: {timing['avg_total_time']:.3f}s")
            
            lines.append(f"\n🔄 系统built率:")
            lines.append(f"   - hierarchical retrievalbuilt率: {success['hierarchical_success_rate']:.2%}")
            lines.append(f"   - 图检索built率: {success['graph_success_rate']:.2%}")
            lines.append(f"   - 双塔同时built率: {success['both_systems_success_rate']:.2%}")
            lines.append(f"   - 双塔优势系数: {success['dual_tower_advantage']:.3f}")
        
        # 
        fusion = benchmark_report["fusion_analysis"]
        if fusion:
            combo = fusion["system_combination_analysis"]
            effectiveness = fusion["fusion_effectiveness"]
            
            lines.append(f"\n🔀 融合效果分析:")
            lines.append(f"   - 双塔同时工作: {combo['both_systems']['count']}次 ({combo['both_systems']['percentage']:.1f}%), F1={combo['both_systems']['avg_f1']:.3f}")
            lines.append(f"   - 仅hierarchical retrieval: {combo['hierarchical_only']['count']}次 ({combo['hierarchical_only']['percentage']:.1f}%), F1={combo['hierarchical_only']['avg_f1']:.3f}")
            lines.append(f"   - 仅图检索: {combo['graph_only']['count']}次 ({combo['graph_only']['percentage']:.1f}%), F1={combo['graph_only']['avg_f1']:.3f}")
            lines.append(f"   - 双系统都failed: {combo['neither_system']['count']}次 ({combo['neither_system']['percentage']:.1f}%), F1={combo['neither_system']['avg_f1']:.3f}")
            
            lines.append(f"\n📈 融合优势:")
            lines.append(f"   - 最佳单系统: {effectiveness['best_single_system']}")
            lines.append(f"   - 双塔相对优势: {effectiveness['dual_tower_advantage']:.3f}")
            lines.append(f"   - 融合vs最佳单系统: {effectiveness['fusion_vs_best_single']:.3f}")
        
        # 
        category_perf = benchmark_report["category_performance"]
        if category_perf:
            lines.append(f"\n📋 questionCategory性能:")
            category_names = {1: "多跳question", 2: "时间question", 3: "开放域question", 4: "单跳question", 5: "对抗性question"}
            for category, stats in category_perf.items():
                category_name = category_names.get(category, f"Category{category}")
                lines.append(f"\n   {category_name} ({stats['test_count']}题):")
                lines.append(f"     - F1分数: {stats['avg_f1_score']:.3f}")
                lines.append(f"     - LLM准确率: {stats['avg_llm_accuracy']:.3f}")
                lines.append(f"     - 分层built率: {stats['hierarchical_success_rate']:.2%}")
                lines.append(f"     - 图检索built率: {stats['graph_success_rate']:.2%}")
        
        # 
        readable_report_file = self.output_dir / f"dual_tower_benchmark_readable_report_{timestamp}.txt"
        with open(readable_report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        # 
        print('\n'.join(lines))
        
        logger.info(f": {readable_report_file}")
        
        # 
        return readable_report_file

    def generate_retrieval_details_report(self) -> Dict[str, Path]:
        """Generate retrieval detail report — separate file with question, answer, and retrieval results.

        Returns:
            Dict mapping report type to the generated file path.
        """
        logger.info("...")
        
        if not self.test_results:
            logger.warning("")
            return {}
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        details_file = self.output_dir / f"retrieval_details_{timestamp}.json"
        
        retrieval_details = []
        
        for result in self.test_results:
            # 
            detail_entry = {
                "sample_id": result.sample_id,
                "question_info": {
                    "question": result.question,
                    "category": result.category,
                    "question_id": f"{result.sample_id}_{result.question.replace(' ', '_')}"
                },
                
                # 
                "answers": {
                    "expected_answer": result.expected_answer,
                    "generated_answer": result.final_answer,
                    "reasoning_process": result.reasoning_process,
                    "confidence_score": result.confidence_score
                },
                
                # 
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
                
                # 
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
                
                # 
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
        
        # 
        with open(details_file, 'w', encoding='utf-8') as f:
            json.dump(retrieval_details, f, ensure_ascii=False, indent=2)
        
        logger.info(f": {details_file}")
        
        # 
        readable_file = self._generate_readable_retrieval_details(retrieval_details, timestamp)
        
        # 
        return {
            "json": details_file,
            "readable": readable_file,
            "timestamp": timestamp
        }

    def _extract_retrieval_summary(self, retrieval_list: List[Dict], layer_name: str) -> List[Dict]:
        """Extract retrieval result summary."""
        summaries = []
        for item in retrieval_list:
            summary = {
                "uid": item.get("uid", "unknown"),
                "score": item.get("score", 0.0),
                "content_preview": item.get("content", "")[:200],
            }
            
            # 
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
        """Extract text content from a unit."""
        if hasattr(unit, 'raw_data') and unit.raw_data:
            text = unit.raw_data.get('text_content', '')
            if not text:
                text = str(unit.raw_data.get('summary', ''))
            if not text:
                text = str(unit.raw_data)[:300]
            return text
        return str(unit)[:300]

    def _generate_readable_retrieval_details(self, retrieval_details: List[Dict], timestamp: str):
        """Generate human-readable retrieval detail text file."""
        readable_file = self.output_dir / f"retrieval_details_readable_{timestamp}.txt"
        
        lines = []
        lines.append("=" * 100)
        lines.append("LoCoMo双塔检索详情报告")
        lines.append("=" * 100)
        lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"总question数: {len(retrieval_details)}")
        lines.append("=" * 100)
        
        for i, detail in enumerate(retrieval_details, 1):
            lines.append(f"\n{'='*100}")
            lines.append(f"question {i}: {detail['sample_id']}")
            lines.append(f"{'='*100}")
            
            # 
            q_info = detail["question_info"]
            lines.append(f"\n📋 question:")
            lines.append(f"   {q_info['question']}")
            lines.append(f"   Category: {q_info['category']}")
            
            # 
            answers = detail["answers"]
            lines.append(f"\n💡 answer对比:")
            lines.append(f"   标准answer: {answers['expected_answer']}")
            lines.append(f"   Generate answer: {answers['generated_answer']}")
            lines.append(f"   confidence: {answers['confidence_score']:.3f}")
            
            # 
            lines.append(f"\n🤔 reasoning process:")
            reasoning = answers['reasoning_process']
            # 
            for line in reasoning.split('\n'):
                if line.strip():
                    lines.append(f"   {line.strip()}")
            
            # Hierarchical retrieval results
            hier = detail["hierarchical_retrieval"]
            lines.append(f"\n🏗️  hierarchical retrieval:")
            lines.append(f"   状态: {'✓ built' if hier['enabled'] else '✗ failed'}")
            lines.append(f"   duration: {hier['retrieval_time']:.3f}s")
            
            if hier['enabled']:
                # L2
                if hier['l2_insights']:
                    lines.append(f"\n   L2洞见 ({len(hier['l2_insights'])}个):")
                    for insight in hier['l2_insights']:
                        lines.append(f"     - [分数: {insight['score']:.3f}] {insight['content_preview'][:100]}...")
                
                # L1
                if hier['l1_summaries']:
                    lines.append(f"\n   L1摘要 ({len(hier['l1_summaries'])}个):")
                    for summary in hier['l1_summaries']:
                        lines.append(f"     - [分数: {summary['score']:.3f}] {summary.get('session_date', '')} - {summary['content_preview'][:100]}...")
                
                # L0
                if hier['l0_observations']:
                    lines.append(f"\n   L0观察 ({len(hier['l0_observations'])}个):")
                    for obs in hier['l0_observations'][:5]:
                        speaker = obs.get('speaker', 'Unknown')
                        lines.append(f"     - [分数: {obs['score']:.3f}] {speaker}: {obs['content_preview'][:80]}...")
            
            # Knowledge-graph retrieval results
            graph = detail["graph_retrieval"]
            lines.append(f"\n🕸️  图谱检索:")
            lines.append(f"   状态: {'✓ built' if graph['enabled'] else '✗ failed'}")
            lines.append(f"   duration: {graph['retrieval_time']:.3f}s")
            lines.append(f"   方法: {graph['retrieval_details'].get('method', 'unknown')}")
            
            if graph['enabled'] and graph['retrieved_units']:
                lines.append(f"\n   检索单元 ({len(graph['retrieved_units'])}个):")
                for unit in graph['retrieved_units'][:5]:
                    lines.append(f"     {unit['rank']}. [分数: {unit['score']:.3f}] {unit['content'][:100]}...")
            
            # Evaluation results
            eval_info = detail["evaluation"]
            if eval_info['success']:
                scores = eval_info['scores']
                lines.append(f"\n📊 evaluation分数:")
                lines.append(f"   F1分数: {scores.get('token_f1', 0):.3f}")
                lines.append(f"   语义相似度: {scores.get('semantic_similarity', 0):.3f}")
                lines.append(f"   精确匹配: {scores.get('exact_match', 0):.3f}")
                if 'llm_accuracy' in scores:
                    lines.append(f"   LLM准确率: {scores.get('llm_accuracy', 0):.3f}")
        
        # 
        with open(readable_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        logger.info(f": {readable_file}")
        
        # 
        return readable_file

    def validate_dataset_compatibility(self) -> Dict[str, Any]:
        """Validate dataset compatibility — check old vs new dataset format.

        Returns:
            Compatibility report dict.
        """
        logger.info(" ...")
        
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
        
        # 1. step3_graphs_dir
        step3_dir = self.step3_graphs_dir / test_sample_id
        if step3_dir.exists():
            compatibility_report["step3_graphs_format"] = "valid"
            logger.info(f": {step3_dir}")
        
        # 2. enhanced_graphs_dir
        enhanced_dir = self.enhanced_graphs_dir / test_sample_id
        enhanced_dir_old = self.enhanced_graphs_dir / f"{test_sample_id}_enhanced"
        
        if enhanced_dir.exists():
            compatibility_report["enhanced_graphs_format"] = "new_format_no_suffix"
            target_dir = enhanced_dir
        elif enhanced_dir_old.exists():
            compatibility_report["enhanced_graphs_format"] = "old_format_with_suffix"
            target_dir = enhanced_dir_old
            logger.warning(f" _enhanced")
        else:
            compatibility_report["enhanced_graphs_format"] = "not_found"
            compatibility_report["compatibility_status"] = "error_enhanced_not_found"
            return compatibility_report
        
        # 3. 
        overview_file = target_dir / "hierarchical_overview.json"
        if overview_file.exists():
            compatibility_report["has_hierarchical_overview"] = True
            try:
                with open(overview_file, 'r', encoding='utf-8') as f:
                    overview_data = json.load(f)
                
                # 
                data_reuse_info = overview_data.get("data_reuse_info", {})
                compatibility_report["has_step1_markers"] = data_reuse_info.get("reused_l0_units", 0) > 0
                compatibility_report["has_step2_markers"] = data_reuse_info.get("new_l1_units", 0) > 0
                
                logger.info(f" : "
                        f"L0复用={data_reuse_info.get('reused_l0_units', 0)}, "
                        f"L1新增={data_reuse_info.get('new_l1_units', 0)}, "
                        f"L2新增={data_reuse_info.get('new_l2_units', 0)}")
                
            except Exception as e:
                logger.warning(f"failed: {e}")
        
        # 4. 
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
                logger.warning(f"failed: {e}")
        
        # 5. 
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
        
        # 
        logger.info(f" : {compatibility_report['dataset_version']}")
        logger.info(f"  : {compatibility_report['step3_graphs_format']}")
        logger.info(f"  : {compatibility_report['enhanced_graphs_format']}")
        logger.info(f"  : {compatibility_report['space_naming_convention']}")
        logger.info(f": {compatibility_report['compatibility_status']}")
        
        return compatibility_report

    #### prompt ####
    ####  ####

    def debug_get_fusion_prompt(self,
                        question: str,
                        category: int,
                        sample_id: str) -> Dict[str, Any]:
        """Debug interface: get the full fusion prompt without calling the LLM.

        Executes the same retrieval flow as _context_aware_fusion_generation,
        but only returns the constructed prompt without making an LLM call.

        Args:
            question: Question text.
            category: Question category code.
            sample_id: Sample ID.

        Returns:
            Dict containing:
            - full_prompt: Complete fusion prompt text.
            - hierarchical_context: Hierarchical retrieval context.
            - graph_results: Graph retrieval results.
            - graph_details: Graph retrieval details.
            - prompt_stats: Prompt statistics.
        """
        logger.info(f" Sample {sample_id} prompt")
        
        try:
            # 1. 
            hierarchical_start = time.time()
            hierarchical_context = self._run_hierarchical_retrieval(sample_id, question, category)
            hierarchical_time = time.time() - hierarchical_start
            
            # 2. 
            graph_start = time.time()
            graph_results, graph_details = self._run_graph_retrieval(sample_id, question)
            graph_time = time.time() - graph_start
            
            # 3. prompt _context_aware_fusion_generation 
            full_prompt = self._build_fusion_prompt_for_debug(
                question=question,
                category=category,
                hierarchical_context=hierarchical_context,
                graph_results=graph_results,
                graph_details=graph_details
            )
            
            # 4. promptStatistics
            prompt_stats = self._calculate_prompt_stats(
                full_prompt=full_prompt,
                hierarchical_context=hierarchical_context,
                graph_results=graph_results
            )
            
            # 5. 
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
            
            logger.info(f"Prompt: {prompt_stats['total_chars']} , "
                    f"{prompt_stats['estimated_tokens']} tokens (估算)")
            
            return debug_result
            
        except Exception as e:
            logger.error(f"promptfailed: {e}")
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
        """Build fusion prompt for debugging (same logic as _context_aware_fusion_generation).

        Reuses the prompt construction logic from _context_aware_fusion_generation,
        but does not call the LLM — only returns the prompt text.
        """
        # 
        prompt_parts = []
        
        # 
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
        
        # 
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
                
                # 
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
        
        # 
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
        """Compute prompt statistics.

        Returns:
            Dict with character count, estimated token count, and per-section ratios.
        """
        total_chars = len(full_prompt)
        
        # token1 token ≈ 4 chars
        estimated_tokens = total_chars // 4
        
        # tiktoken
        try:
            import tiktoken
            encoder = tiktoken.encoding_for_model("gpt-4")
            actual_tokens = len(encoder.encode(full_prompt))
        except ImportError:
            actual_tokens = None
        
        # 
        hierarchical_text = hierarchical_context.get("hierarchical_context_text", "")
        hierarchical_chars = len(hierarchical_text)
        
        # 
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
        """Extract content text from graph units."""
        if hasattr(unit, 'raw_data') and unit.raw_data:
            # text_content
            text_content = unit.raw_data.get('text_content', '')
            if text_content:
                return text_content
            
            # 
            for field in ['content', 'name', 'description', 'summary']:
                content = unit.raw_data.get(field, '')
                if content:
                    return str(content)
            
            # 
            return str(unit.raw_data)[:200]
        
        # raw_data
        return str(unit)[:200]

def main():
    """Main entry point — supports parallel processing configuration."""
    
    parser = argparse.ArgumentParser(description="LoCoMo dual-tower retrieval benchmark")
    
    # Data paths
    parser.add_argument("--enhanced-graphs-dir", 
                       default="benchmark_locomo/dataset/locomo/hierarchical/step3_final_graphs",
                       help="增强图谱目录（hierarchical retrieval）")
    parser.add_argument("--step3-graphs-dir", 
                       default="benchmark_locomo/dataset/locomo/entity_relation/step3_semantic_graph",
                       help="步骤3图谱目录（knowledge-graph retrieval）")
    parser.add_argument("--qa-dataset", 
                       default="benchmark_locomo/dataset/locomo/locomo10.json",
                       help="QA数据集路径")
    parser.add_argument("--output-dir", 
                       default="benchmark_locomo/task_eval/results/locomo_dual_tower_benchmark_new_dataset",
                       help="输出目录")
    
    # LLM
    parser.add_argument("--llm-model", 
                    #    default="gpt-4o-mini-closeai",
                       default="gpt-4.1-mini-closeai",
                       help="answer generationLLM模型名称")
    parser.add_argument("--llm-evaluate-model", 
                    #    default="deepseek-chat",
                       default="gpt-4o-mini-closeai",
                       help="answerevaluationLLM模型名称")
    
    # Retrieval configuration
    parser.add_argument("--topk-hierarchical-l0", type=int, default=15,
                       help="hierarchical retrievalL0层top-k")
    parser.add_argument("--topk-hierarchical-l1", type=int, default=5,
                       help="hierarchical retrievalL1层top-k")
    parser.add_argument("--topk-hierarchical-l2", type=int, default=1,
                       help="hierarchical retrievalL2层top-k")
    parser.add_argument("--topk-similarity", type=int, default=15,
                       help="图检索语义检索top-k")
    parser.add_argument("--topk-graph", type=int, default=0,
                       help="图检索实体关系top-k（默认0disabled，设置>0enabled）")
    
    # 
    parser.add_argument("--fusion-strategy", 
                       choices=["simple", "weighted", "context_aware"],
                       default="context_aware",
                       help="Fusion strategy")
    parser.add_argument("--hierarchical-weight", type=float, default=0.5,
                       help="hierarchical retrievalweights")
    parser.add_argument("--graph-weight", type=float, default=0.5,
                       help="图检索weights")
    
    # Test configuration
    parser.add_argument("--sample-ids", nargs='+',
                       help="指定测试SampleID列表")
    parser.add_argument("--max-samples", type=int,
                       help="最大测试Sample数")
    parser.add_argument("--max-workers", type=int, default=1,
                       help="并发工作线程数")
    parser.add_argument("--no-entity-relation", action="store_true",
                       default=True,
                       help="disabled实体关系检索（默认disabled）")
    parser.add_argument("--enable-entity-relation", dest="no_entity_relation", 
                       action="store_false",
                       help="enabled实体关系检索")
    
    # 
    parser.add_argument("--log-level", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO",
                       help="日志级别")
    
    #  Sample
    parser.add_argument("--no-sequential-mode", dest="sequential_mode", action="store_false",
                       default=True,
                       help="disabled逐个Sample模式，使用批量加载模式")
    parser.add_argument("--sequential-mode", dest="sequential_mode", action="store_true",
                       help="使用逐个Sample模式（默认enabled，内存友好）")
    
    #  
    parser.add_argument("--parallel-towers", dest="parallel_towers", action="store_true",
                       default=False,
                       help="enabledParallel towers")
    parser.add_argument("--no-parallel-towers", dest="parallel_towers", action="store_false",
                       help="disabledParallel towers，使用串行模式（默认）")

    #  - 
    parser.add_argument('--reranker-type',
                       choices=['baai', 'qwen', 'jina', 'qwen-sili', 'qwen-dashscope', 'gte-dashscope'],
                    #    default='baai',
                       default='jina',
                       help='Reranker类型:\n'
                            '  baai: BAAI BGE本地Reranker (默认)\n'
                            '  qwen: Qwen本地Reranker\n'
                            '  jina: Jina本地Reranker\n'
                            '  qwen-sili: Qwen云端Reranker(Siliconflow)\n'
                            '  qwen-dashscope: Qwen云端Reranker(DashScope)\n'
                            '  gte-dashscope: GTE云端Reranker(DashScope)')
    
    parser.add_argument('--reranker-model',
                       help='自定义Reranker模型名称')
    
    args = parser.parse_args()
    
    # 
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    print("=" * 100)
    print(" LoCoMo dual-tower retrieval benchmark")
    print("=" * 100)
    print(f" : {args.enhanced_graphs_dir}")
    print(f" 3: {args.step3_graphs_dir}")
    print(f" QA: {args.qa_dataset}")
    print(f" : {args.output_dir}")
    print(f" : {args.llm_model}")
    print(f" : {args.llm_evaluate_model}")
    print(f" : {args.fusion_strategy}")
    print(f"  : ={args.hierarchical_weight}, ={args.graph_weight}")
    print(f" Retrieval configuration: L0={args.topk_hierarchical_l0}, L1={args.topk_hierarchical_l1}, L2={args.topk_hierarchical_l2}")
    print(f" Retrieval configuration: ={args.topk_similarity}, ={'' if args.topk_graph == 0 else f'{args.topk_graph}'}")
    print(f" : {'' if args.no_entity_relation else ''}")
    print(f" : {args.reranker_type}")  # 
    print(f" : {args.max_workers}")
    print(f" : {'Sample' if args.sequential_mode else ''}")
    print(f" : {'' if args.parallel_towers else ''}")
    
    # API
    if args.reranker_type == 'qwen-remote':
        api_key = os.getenv("CSTCLOUD_API_KEY")
        if api_key:
            print(f" API: ")
        else:
            print(f"  API:  ( CSTCLOUD_API_KEY)")
    
    if args.sample_ids:
        print(f" Sample: {args.sample_ids}")
    if args.max_samples:
        print(f" Sample: {args.max_samples}")
    
    try:
        # LLM
        print("\n LLM...")
        llm_client = LLMClient(model_name=args.llm_model)
        llm_evaluate_client = LLMClient(model_name=args.llm_evaluate_model)
        print(f" : {args.llm_model}")
        print(f" : {args.llm_evaluate_model}")
        
        # Reranker configuration — all types
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
        
        # 
        print(f"\n  ({args.reranker_type})...")
        from dev.retrieval.rerank_manager import RerankerManager
        global_reranker_manager = RerankerManager()
        
        # # Pre-load
        # try:
        #     reranker = global_reranker_manager.get_reranker(
        #         reranker_type=args.reranker_type,
        #         model_name=reranker_configs[args.reranker_type]
        #     )
        #     print(f" Pre-load: {args.reranker_type}")
        # except Exception as e:
        #     print(f"  Pre-loadfailed: {e}")
        #     print(f"   ")
        
        # benchmark
        print("\n Benchmark...")
        
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
            reranker_type=args.reranker_type,  # added
            reranker_configs=reranker_configs,  # added
            reranker_manager=global_reranker_manager
        )
        
        print(" Benchmark")
        
        # topk_graph=0
        if args.topk_graph == 0:
            print(" topk-graph=0")
            print("  --topk-graph 5 --enable-entity-relation")

        #  Sample
        use_sequential = args.sequential_mode
        
        # sequential
        if not use_sequential and args.sample_ids and len(args.sample_ids) > 2:
            logger.warning(f" SampleID ({len(args.sample_ids)}) ")
            logger.warning(f"  --sequential-mode ")
            response = input("是否切换到逐个Sample模式？(y/N): ")
            if response.lower() == 'y':
                use_sequential = True
        
        if use_sequential:
            print("\n Sample")
            print(" Sample")
            print("  --no-sequential-mode")
            
            # Sample
            test_start_time = time.time()
            benchmark.run_dual_tower_benchmark(sequential_mode=True)
            test_time = time.time() - test_start_time
            
            print(f" Sample: {test_time:.2f}s")
        else:
            print("\n ")
            print(" Sample")
            
            # Pre-load
            print("\n ...")
            start_time = time.time()
            benchmark.load_systems(max_samples=args.max_samples)
            load_time = time.time() - start_time
            print(f" loadedPre-load: {load_time:.2f}s")
            
            # 
            print("\n ...")
            benchmark.load_test_cases()
            print(f" loaded: {len(benchmark.test_cases)} ")
            
            # benchmark
            print("\n benchmark...")
            print(" Pre-load")
            test_start_time = time.time()
            benchmark.run_dual_tower_benchmark(sequential_mode=False)
            test_time = time.time() - test_start_time
            
            print(f" : {test_time:.2f}s")
        
        # 
        print("\n benchmark...")
        generated_files = benchmark.generate_dual_tower_report()
        
        # 
        print(f"\n :")
        print(f"    : {benchmark.stats['successful_dual_tower']}")
        print(f"    failed: {benchmark.stats['failed_retrievals']}")
        print(f"    : {benchmark.stats['successful_hierarchical']}")
        print(f"     : {benchmark.stats['successful_graph']}")
        
        total_tests = benchmark.stats['successful_dual_tower'] + benchmark.stats['failed_retrievals']
        if total_tests > 0:
            success_rate = benchmark.stats['successful_dual_tower'] / total_tests
            print(f"    : {success_rate:.2%}")
        
        if not use_sequential:
            total_time = load_time + test_time
            print(f"\n⏱  : {total_time:.2f}s")
        else:
            print(f"\n⏱  : {test_time:.2f}s")
        
        # 
        if generated_files:
            print(f"\n :")
            print(f"   : {generated_files['main_report']}")
            print(f"   : {generated_files['readable_report']}")
            print(f"   (JSON): {generated_files['retrieval_details_json']}")
            print(f"   (): {generated_files['retrieval_details_readable']}")
            print(f"   : {generated_files['timestamp']}")
        
        print("\n Benchmark!")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n ")
        return 1
    except Exception as e:
        print(f"\n Benchmarkfailed: {e}")
        logger.error(f": {traceback.format_exc()}")
        return 1
    finally:
        # 
        try:
            cleanup_evaluation_models()
        except Exception as e:
            logger.warning(f"failed: {e}")


if __name__ == "__main__":
    sys.exit(main())