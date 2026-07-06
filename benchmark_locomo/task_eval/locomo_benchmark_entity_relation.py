"""Utilities for locomo benchmark entity relation."""

import os
import sys
import json
import logging
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass
import numpy as np
from collections import defaultdict
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm
import logging


# Avoid mutating LogRecord fields before other handlers process the record.
from mandol.utils.logging_config import setup_logging, create_module_logger, auto_configure_logging
if auto_configure_logging() is None:
    setup_logging(level=logging.INFO)
logger = create_module_logger("locomo_benchmark_entity_relation")

from mandol.core.semantic_graph import SemanticGraph
from mandol.retrieval.rerank_manager import RerankerManager
from mandol.retrieval.advance_retriever import MultiRetriever
from mandol.entity_relation.entity_relation_retriever import EntityRelationRetriever
from mandol.retrieval.retrieval_interface import RetrievalMethod
from mandol.llm.llm_client import LLMClient
from benchmark_locomo.task_eval.evaluation import (
    calculate_comprehensive_scores, 
    batch_evaluate,
    cleanup_evaluation_models,
    get_model_manager
)
from mandol.core import paths

@dataclass
class LoCoMoGraphTestCase:
    sample_id: str
    question: str
    category: int
    expected_answer: str
    question_type: str  # when, where, who, what, how, why
    evidence: List[str] = None
    context_sessions: List[str] = None
    retrieval_hints: Dict[str, Any] = None

@dataclass 
class GraphRetrievalResult:
    test_case: LoCoMoGraphTestCase
    retrieval_method: str
    retrieval_time: float
    retrieved_units: List[Any]
    final_answer: str
    confidence_score: float
    retrieval_details: Dict[str, Any]
    evaluation_scores: Dict[str, float]

class LoCoMoEntityRelationBenchmark:
    
    def __init__(self, 
    semantic_graphs_dir: str,
    qa_dataset_path: str,
    llm_client: Optional[LLMClient] = None,
    llm_evaluate_client: Optional[LLMClient] = None,
    output_dir: str = "locomo_graph_benchmark_results",
    use_entity_relation: bool = True,
    target_sample_ids: Optional[List[str]] = None,
    max_workers: int = 1,
    topk_similarity: int = 15,
    topk_graph: int = 5,
    reranker_type: str = "baai",
    reranker_configs: Optional[Dict[str, str]] = None,
    reranker_manager: Optional[RerankerManager] = None):
        self.semantic_graphs_dir = Path(semantic_graphs_dir)
        self.qa_dataset_path = Path(qa_dataset_path)
        self.llm_client = llm_client or LLMClient(model_name="deepseek-chat")
        self.llm_evaluate_client = llm_evaluate_client or LLMClient(model_name="deepseek-chat")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_entity_relation = use_entity_relation
        self.target_sample_ids = set(target_sample_ids) if target_sample_ids else None
        self.max_workers = max_workers
        
        
        self.reranker_type = reranker_type
        self.reranker_configs = reranker_configs or {
            "baai": "BAAI/bge-reranker-v2-m3",
            "qwen": "Qwen/Qwen3-Reranker-0.6B",
            "qwen-remote": "qwen3-reranker:8b"  
        }
        
        self.global_reranker_manager = reranker_manager
        if reranker_manager:
            logger.info(f" 使用外部传入的重排序器管理器")
        else:
            logger.info(f" 创建新的重排序器管理器")
            self.global_reranker_manager = RerankerManager()
        
        
        self.semantic_graphs: Dict[str, SemanticGraph] = {}
        self.multi_retrievers: Dict[str, MultiRetriever] = {}
        self.entity_relation_retrievers: Dict[str, EntityRelationRetriever] = {}
        
        self.test_cases: List[LoCoMoGraphTestCase] = []
        self.test_results: List[GraphRetrievalResult] = []
        
        self.results_lock = threading.Lock()

        self.topk_similarity = topk_similarity
        self.topk_graph = topk_graph
        
        self.stats = {
            "total_graphs_loaded": 0,
            "total_test_cases": 0,
            "successful_retrievals": 0,
            "failed_retrievals": 0,
            "average_retrieval_time": 0.0,
            "method_performance": {},
            "category_performance": {}
        }
        
        logger.info("LoCoMo实体关系Benchmark测试器初始化完成")
        logger.info(f"使用实体关系检索: {self.use_entity_relation}")
        logger.info(f"多线程工作线程数: {self.max_workers}")
        logger.info(f"语义检索top-k: {self.topk_similarity}")
        logger.info(f"图检索top-k: {self.topk_graph}")
        logger.info(f" 重排序器配置: {self.reranker_type} ({self.reranker_configs.get(self.reranker_type, 'unknown')})")
    
    def load_semantic_graphs(self, max_graphs: Optional[int] = None):
        """Load semantic graphs."""
        logger.info("开始加载语义图...")
        
        graph_dirs = [d for d in self.semantic_graphs_dir.iterdir() if d.is_dir()]
        
        if self.target_sample_ids:
            graph_dirs = [d for d in graph_dirs if d.name in self.target_sample_ids]
            logger.info(f"过滤后的样本数: {len(graph_dirs)}")
        
        if max_graphs:
            graph_dirs = graph_dirs[:max_graphs]
        
        loaded_count = 0
        failed_count = 0
        
        required_methods = [
            RetrievalMethod.BM25,
            RetrievalMethod.COSINE_SIMILARITY,
            RetrievalMethod.SPLADE
        ]
        
        for graph_dir in tqdm(graph_dirs, desc="加载语义图"):
            sample_id = graph_dir.name
            try:
                semantic_graph = SemanticGraph.load_graph(str(graph_dir))
                self.semantic_graphs[sample_id] = semantic_graph
                
                multi_retriever = MultiRetriever(
                    retrieval_source=semantic_graph,
                    preload_rerankers=False,
                    reranker_configs=self.reranker_configs,
                    reranker_manager=self.global_reranker_manager
                )
                
                
                logger.info(f" 为样本 {sample_id} 构建检索器索引...")
                build_stats = multi_retriever.build_all_indexes(
                    methods_to_build=required_methods
                )
                
                logger.info(f"   索引构建完成: 成功={build_stats['built_count']}, "
                        f"跳过={build_stats['skipped_count']}, "
                        f"失败={build_stats['failed_count']}")
                
                self.multi_retrievers[sample_id] = multi_retriever
                
                if self.use_entity_relation:
                    try:
                        entity_retriever = EntityRelationRetriever(
                            semantic_graph=semantic_graph,
                            use_bert_ner=True,
                            enable_relation_filtering=True,
                            relation_filter_threshold=0.6
                        )
                        self.entity_relation_retrievers[sample_id] = entity_retriever
                        logger.debug(f"实体关系检索器已创建: {sample_id}")
                    except Exception as e:
                        logger.warning(f"创建实体关系检索器失败 {sample_id}: {e}")
                        self.entity_relation_retrievers[sample_id] = None
                
                loaded_count += 1
                logger.debug(f" {sample_id} 语义图加载成功")
                
            except Exception as e:
                failed_count += 1
                logger.error(f" {sample_id} 语义图加载失败: {e}")
                logger.debug(traceback.format_exc())
                continue
        
        self.stats["total_graphs_loaded"] = loaded_count
        logger.info(f"语义图加载完成: 成功 {loaded_count}, 失败 {failed_count}")
        
        if loaded_count == 0:
            raise RuntimeError("没有成功加载任何语义图")
        
    def load_test_cases(self):
        """Load test cases."""
        logger.info("从locomo10.json加载测试用例...")
        
        try:
            with open(self.qa_dataset_path, 'r', encoding='utf-8') as f:
                qa_data = json.load(f)
            
            for item in qa_data:
                sample_id = item["sample_id"]
                
                
                if sample_id not in self.semantic_graphs:
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
                        logger.debug(f"跳过无答案的问题: {sample_id} 问题 {i+1}")
                        continue
                    
                    evidence = qa_item.get("evidence", [])
                    
                    question_type = self._analyze_question_type(question)
                    
                    retrieval_hints = self._extract_retrieval_hints(question, category)
                    
                    test_case = LoCoMoGraphTestCase(
                        sample_id=sample_id,
                        question=question,
                        category=category,
                        expected_answer=expected_answer,
                        question_type=question_type,
                        evidence=evidence,
                        retrieval_hints=retrieval_hints
                    )
                    
                    self.test_cases.append(test_case)
            
            self.stats["total_test_cases"] = len(self.test_cases)
            logger.info(f"测试用例加载完成: {len(self.test_cases)} 个")
            
            type_counts = {}
            category_counts = {}
            sample_counts = {}
            
            for case in self.test_cases:
                type_counts[case.question_type] = type_counts.get(case.question_type, 0) + 1
                category_counts[case.category] = category_counts.get(case.category, 0) + 1
                sample_counts[case.sample_id] = sample_counts.get(case.sample_id, 0) + 1
            
            logger.info(f"问题类型分布: {type_counts}")
            logger.info(f"问题类别分布: {category_counts}")
            logger.info(f"样本分布: {dict(list(sample_counts.items())[:5])}..." + (f" (共{len(sample_counts)}个样本)" if len(sample_counts) > 5 else ""))
            
        except Exception as e:
            logger.error(f"加载测试用例失败: {e}")
            raise
    
    def _analyze_question_type(self, question: str) -> str:
        """Run analyze question type."""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['when', 'what time', 'what date', 'how long ago']):
            return 'when'
        elif any(word in question_lower for word in ['where', 'which place', 'what location']):
            return 'where'
        elif any(word in question_lower for word in ['who', 'which person', 'whose']):
            return 'who'
        elif any(word in question_lower for word in ['what', 'which', 'what kind']):
            return 'what'
        elif any(word in question_lower for word in ['how', 'in what way']):
            return 'how'
        elif any(word in question_lower for word in ['why', 'what reason']):
            return 'why'
        else:
            return 'other'
    
    def _extract_retrieval_hints(self, question: str, category: int) -> Dict[str, Any]:
        """Extract retrieval hints."""
        hints = {
            "question_type": self._analyze_question_type(question),
            "category": category,
            "requires_multi_hop": category in [1, 4],
            "requires_temporal": "when" in question.lower() or category == 2,
            "requires_spatial": "where" in question.lower(),
            "is_adversarial": category == 5,
        }
        
        if hints["requires_temporal"]:
            hints["target_entity_types"] = ["DATE_TIME", "EVENT", "ACTIVITY"]
        elif hints["requires_spatial"]:
            hints["target_entity_types"] = ["LOCATION", "EVENT", "ACTIVITY"]
        elif "who" in question.lower():
            hints["target_entity_types"] = ["PERSON", "RELATIONSHIP", "ORGANIZATION"]
        
        return hints

    def run_comprehensive_benchmark(self):
        """Run comprehensive benchmark."""
        logger.info("开始运行综合图检索benchmark测试...")
        
        test_config = {
            "name": "Triple_Fusion_BAAI_Rerank" + ("_EntityRelation" if self.use_entity_relation else ""),
            "config": {
                "methods": ["bm25", "cosine_similarity", "splade"],
                "fusion_method": "rrf",
                "rerank_method": "baai",
                "use_entity_relation": self.use_entity_relation,
                "fusion_with_graph": self.use_entity_relation,
                "top_k": 10,  
                "topk_similarity": self.topk_similarity,
                "topk_graph": self.topk_graph
            }
        }
        
        total_tests = len(self.test_cases)
        logger.info(f"总测试数: {total_tests}")
        logger.info(f"使用方法: {test_config['name']}")
        logger.info(f"语义检索top-k: {self.topk_similarity}")  # Avoid mutating LogRecord fields before other handlers process the record.
        logger.info(f"图检索top-k: {self.topk_graph}")  # Avoid mutating LogRecord fields before other handlers process the record.
        
        if self.max_workers == 1:
            self._run_single_threaded_tests(test_config, total_tests)
        else:
            self._run_multi_threaded_tests(test_config, total_tests)
        
        logger.info("综合图检索benchmark测试完成")
    
    def _run_single_threaded_tests(self, test_config: Dict[str, Any], total_tests: int):
        """Run single threaded tests."""
        for i, test_case in enumerate(tqdm(self.test_cases, desc="执行测试"), 1):
            try:
                result = self._run_single_graph_test(test_case, test_config)
                if result:
                    self.test_results.append(result)
                    self.stats["successful_retrievals"] += 1
                    logger.debug(f" 测试成功: {test_case.sample_id}")
                else:
                    self.stats["failed_retrievals"] += 1
                    logger.warning(f" 测试返回空结果: {test_case.sample_id}")
                    
            except Exception as e:
                self.stats["failed_retrievals"] += 1
                logger.error(f" 测试失败: {test_case.sample_id} - {e}")
                continue
    
    def _run_multi_threaded_tests(self, test_config: Dict[str, Any], total_tests: int):
        """Run multi threaded tests."""
        logger.info(f"使用 {self.max_workers} 个线程执行测试")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_test_case = {
                executor.submit(self._run_single_graph_test_thread_safe, test_case, test_config): test_case
                for test_case in self.test_cases
            }
            
            for future in tqdm(as_completed(future_to_test_case), total=total_tests, desc="执行测试"):
                test_case = future_to_test_case[future]
                try:
                    result = future.result()
                    with self.results_lock:
                        if result:
                            self.test_results.append(result)
                            self.stats["successful_retrievals"] += 1
                            logger.debug(f" 测试成功: {test_case.sample_id}")
                        else:
                            self.stats["failed_retrievals"] += 1
                            logger.warning(f" 测试返回空结果: {test_case.sample_id}")
                            
                except Exception as e:
                    with self.results_lock:
                        self.stats["failed_retrievals"] += 1
                    logger.error(f" 测试失败: {test_case.sample_id} - {e}")
                    continue
    
    def _run_single_graph_test_thread_safe(self, test_case: LoCoMoGraphTestCase, method_config: Dict[str, Any]) -> Optional[GraphRetrievalResult]:
        """Run single graph test thread safe."""
        try:
            return self._run_single_graph_test(test_case, method_config)
        except Exception as e:
            logger.error(f"线程安全测试失败: {test_case.sample_id} - {e}")
            return None

    def _run_single_graph_test(self, test_case: LoCoMoGraphTestCase, method_config: Dict[str, Any]) -> Optional[GraphRetrievalResult]:
        """Run single graph test."""
        sample_id = test_case.sample_id
        question = test_case.question
        method_name = method_config["name"]
        config = method_config["config"]
        
        if sample_id not in self.multi_retrievers:
            logger.error(f"样本 {sample_id} 的检索器不存在")
            return None
        
        multi_retriever = self.multi_retrievers[sample_id]
        entity_retriever = self.entity_relation_retrievers.get(sample_id)
        
        start_time = time.time()
        retrieved_units = []
        retrieval_details = {}
        
        try:
            
            if config.get("fusion_with_graph", False) and entity_retriever:
                topk_similarity = config.get("topk_similarity", 15)
                topk_graph = config.get("topk_graph", 5)
                
                semantic_results = multi_retriever.smart_search(
                    query=question,
                    methods=config.get("methods", ["bm25", "cosine_similarity", "splade"]),
                    fusion_method=config.get("fusion_method", "rrf"),
                    rerank_method=self.reranker_type,  
                    top_k=topk_similarity,
                    return_detailed=False
                )
                
                graph_results = []
                entity_results = entity_retriever.search(question, topk_graph)
                graph_results = [(r.unit, r.score) for r in entity_results]
                
                retrieved_units = semantic_results + graph_results
                retrieved_units = retrieved_units[:config.get("top_k", 10)]
                retrieval_details["method"] = "hybrid_semantic_graph"
                retrieval_details["semantic_count"] = len(semantic_results)
                retrieval_details["graph_count"] = len(graph_results)
                retrieval_details["topk_similarity"] = topk_similarity
                retrieval_details["topk_graph"] = topk_graph
                retrieval_details["rerank_method"] = self.reranker_type  
                
            else:
                
                retrieved_units = multi_retriever.smart_search(
                    query=question,
                    methods=config.get("methods", ["bm25", "cosine_similarity", "splade"]),
                    fusion_method=config.get("fusion_method", "rrf"),
                    rerank_method=self.reranker_type,  
                    top_k=config.get("top_k", 10),
                    return_detailed=False
                )
                retrieval_details["method"] = "semantic_only"
                retrieval_details["fusion_method"] = config.get("fusion_method", "rrf")
                retrieval_details["rerank_method"] = self.reranker_type  
  
            retrieval_time = time.time() - start_time
            
            final_answer, confidence_score, reasoning = self._generate_answer_from_units(
                question, retrieved_units, test_case.retrieval_hints
            )
            
            evaluation_scores = self._evaluate_retrieval_result(
                test_case, final_answer, retrieved_units, reasoning
            )
            
            retrieval_details.update({
                "retrieved_count": len(retrieved_units),
                "total_retrieval_time": retrieval_time,
                "answer_generation_method": "llm_synthesis"
            })
            
            return GraphRetrievalResult(
                test_case=test_case,
                retrieval_method=method_name,
                retrieval_time=retrieval_time,
                # retrieved_units=retrieved_units[:5],
                retrieved_units=retrieved_units,  
                final_answer=final_answer,
                confidence_score=confidence_score,
                retrieval_details=retrieval_details,
                evaluation_scores=evaluation_scores
            )
            
        except Exception as e:
            logger.error(f"单个图检索测试执行失败: {e}")
            logger.debug(traceback.format_exc())
            return None
    
    def _generate_answer_from_units(self, question: str, retrieved_units: List[Tuple[Any, float]], hints: Dict[str, Any]) -> Tuple[str, float, str]:
        """Generate answer from units."""
        if not retrieved_units:
            return "No relevant information found.", 0.0, ""
        
        context_parts = []
        total_confidence = 0.0
        
        for i, (unit, score) in enumerate(retrieved_units):
            if hasattr(unit, 'raw_data') and unit.raw_data:
                text_content = unit.raw_data.get('text_content', '')
                if not text_content:
                    text_content = str(unit.raw_data.get('summary', ''))
                if not text_content:
                    text_content = str(unit.raw_data)
                
                # context_parts.append(f"Context {i+1} (relevance: {score:.3f}): {text_content[:200]}...")
                context_parts.append(f"Context {i+1}: {text_content[:200]}...")
                total_confidence += score
        
        if not context_parts:
            return "No textual content found in retrieved results.", 0.0, ""
        
        avg_confidence = total_confidence / len(retrieved_units)
        context = "\n\n".join(context_parts)
        
        question_type = hints.get("question_type", "other")
        is_adversarial = hints.get("is_adversarial", False)
        
        if is_adversarial:
            prompt = f"""Based on the following context, answer the question. CRITICAL: This is an ADVERSARIAL question.

            Question: {question}

            Context:
            {context}

            INSTRUCTIONS:
            1. Carefully verify if the information is EXPLICITLY present
            2. If information is not found or question is misleading, state that clearly
            3. DO NOT fabricate information

            Provide your response in JSON format:
            {{
                "reasoning": "Your verification process",
                "answer": "Your final answer"
            }}
            """
            
            try:
                response = self.llm_client.generate_answer(
                    prompt=prompt,
                    temperature=0.1,
                    max_tokens=200,
                    json_format=True
                )
                
                parsed = json.loads(response.strip())
                answer = parsed.get("answer", "Unable to answer")
                reasoning = parsed.get("reasoning", "No reasoning provided")
                
                return answer.strip(), min(avg_confidence, 1.0), reasoning
                
            except Exception as e:
                logger.error(f"答案生成失败: {e}")
                return f"Answer generation failed: {str(e)}", 0.0, str(e)
        
        elif question_type == "when":
            instruction = "Focus on temporal information, dates, and time periods."
        elif question_type == "where":
            instruction = "Focus on location information, places, and geographical details."
        elif question_type == "who":
            instruction = "Focus on people, relationships, and personal information."
        else:
            instruction = "Provide a comprehensive answer based on the available information."
        
        prompt = f"""Based on the following context, answer the question. {instruction}

        Question: {question}

        Context:
        {context}

        Please provide a direct and concise answer. If the information is not sufficient or not found, state that clearly.
        Answer:"""
        
        try:
            answer = self.llm_client.generate_answer(
                prompt=prompt,
                temperature=0.1,
                max_tokens=150
            )
            return answer.strip(), min(avg_confidence, 1.0), ""
            
        except Exception as e:
            logger.error(f"答案生成失败: {e}")
            return f"Answer generation failed: {str(e)}", 0.0, ""
    
    def _evaluate_retrieval_result(self, 
                              test_case: LoCoMoGraphTestCase, 
                              generated_answer: str, 
                              retrieved_units: List[Tuple[Any, float]],
                              reasoning: str = "") -> Dict[str, float]:
        """Run evaluate retrieval result."""
        try:
            from benchmark_locomo.task_eval.evaluation import calculate_comprehensive_scores
            
            eval_result = calculate_comprehensive_scores(
                gold_answer=test_case.expected_answer,
                response=generated_answer,
                question=test_case.question,
                reasoning=reasoning,
                llm_client=self.llm_evaluate_client,
                metrics=["exact_match", "f1", "rouge", "semantic_similarity", "llm_judge"],
                category=test_case.category,
                is_adversarial=(test_case.category == 5)
            )
            
            scores = eval_result.get("scores", {})
            
            scores["retrieval_count"] = len(retrieved_units)
            scores["avg_retrieval_score"] = np.mean([score for _, score in retrieved_units]) if retrieved_units else 0.0
            scores["top1_retrieval_score"] = retrieved_units[0][1] if retrieved_units else 0.0
            
            return scores
            
        except Exception as e:
            logger.error(f"评估失败: {e}")
            return {
                "exact_match": 0.0,
                "token_f1": 0.0,
                "semantic_similarity": 0.0,
                "llm_accuracy": 0.0,
                "retrieval_count": len(retrieved_units),
                "avg_retrieval_score": 0.0,
                "evaluation_error": str(e)
            }

    def generate_benchmark_report(self):
        """Generate benchmark report."""
        logger.info("生成benchmark报告...")
        
        if not self.test_results:
            logger.warning("没有测试结果，无法生成报告")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results_by_sample = defaultdict(list)
        for result in self.test_results:
            results_by_sample[result.test_case.sample_id].append(result)
        
        results_by_question_type = defaultdict(list)
        for result in self.test_results:
            results_by_question_type[result.test_case.question_type].append(result)
        
        results_by_category = defaultdict(list)
        for result in self.test_results:
            results_by_category[result.test_case.category].append(result)
        
        overall_stats = self._calculate_overall_statistics()
        
        sample_performance = {}
        for sample_id, sample_results in results_by_sample.items():
            sample_performance[sample_id] = self._calculate_sample_performance(sample_results)
        
        question_type_performance = {}
        for q_type, type_results in results_by_question_type.items():
            question_type_performance[q_type] = self._calculate_question_type_performance(type_results)
        
        category_performance = {}
        for category, cat_results in results_by_category.items():
            category_performance[category] = self._calculate_category_performance(cat_results)
        
        benchmark_report = {
            "benchmark_info": {
                "test_name": "LoCoMo Entity Relation Graph Benchmark",
                "test_type": "knowledge_graph_retrieval",
                "timestamp": datetime.now().isoformat(),
                "total_graphs": self.stats["total_graphs_loaded"],
                "total_test_cases": self.stats["total_test_cases"],
                "total_tests_run": len(self.test_results),
                "success_rate": self.stats["successful_retrievals"] / (self.stats["successful_retrievals"] + self.stats["failed_retrievals"]) if (self.stats["successful_retrievals"] + self.stats["failed_retrievals"]) > 0 else 0.0,
                "method_used": "Triple_Fusion_BAAI_Rerank" + ("_EntityRelation" if self.use_entity_relation else ""),
                "use_entity_relation": self.use_entity_relation,
                "max_workers": self.max_workers,
                "topk_similarity": self.topk_similarity,
                "topk_graph": self.topk_graph,
                "target_samples": list(self.target_sample_ids) if self.target_sample_ids else "all"
            },
            "overall_statistics": overall_stats,
            "sample_performance": sample_performance,
            "question_type_performance": question_type_performance,
            "category_performance": category_performance,
            "detailed_results": [
                {
                    "sample_id": r.test_case.sample_id,
                    "question": r.test_case.question,
                    "question_type": r.test_case.question_type,
                    "category": r.test_case.category,
                    "method": r.retrieval_method,
                    "expected_answer": r.test_case.expected_answer,
                    "generated_answer": r.final_answer,
                    "retrieval_time": r.retrieval_time,
                    "confidence_score": r.confidence_score,
                    "evaluation_scores": r.evaluation_scores,
                    "retrieval_details": r.retrieval_details
                }
                for r in self.test_results
            ]
        }
        
        
        report_file = self.output_dir / f"locomo_graph_benchmark_report_{timestamp}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(benchmark_report, f, ensure_ascii=False, indent=2)
        
        self._generate_readable_report(benchmark_report, timestamp)
        
        logger.info(f"Benchmark报告已生成: {report_file}")
    
    def _calculate_overall_statistics(self) -> Dict[str, Any]:
        """Calculate overall statistics."""
        if not self.test_results:
            return {}
        
        all_retrieval_times = [r.retrieval_time for r in self.test_results]
        all_f1_scores = [r.evaluation_scores.get("token_f1", 0.0) for r in self.test_results]
        all_semantic_scores = [r.evaluation_scores.get("semantic_similarity", 0.0) for r in self.test_results]
        all_llm_scores = [r.evaluation_scores.get("llm_accuracy", 0.0) for r in self.test_results]
        all_exact_match = [r.evaluation_scores.get("exact_match", 0.0) for r in self.test_results]
        
        return {
            "total_tests": len(self.test_results),
            "avg_retrieval_time": np.mean(all_retrieval_times),
            "avg_f1_score": np.mean(all_f1_scores),
            "avg_semantic_similarity": np.mean(all_semantic_scores),
            "avg_llm_accuracy": np.mean(all_llm_scores),
            "avg_exact_match": np.mean(all_exact_match),
            "std_f1_score": np.std(all_f1_scores),
            "std_semantic_similarity": np.std(all_semantic_scores)
        }
    
    def _calculate_sample_performance(self, sample_results: List[GraphRetrievalResult]) -> Dict[str, Any]:
        """Calculate sample performance."""
        if not sample_results:
            return {}
        
        f1_scores = [r.evaluation_scores.get("token_f1", 0.0) for r in sample_results]
        semantic_scores = [r.evaluation_scores.get("semantic_similarity", 0.0) for r in sample_results]
        llm_scores = [r.evaluation_scores.get("llm_accuracy", 0.0) for r in sample_results]
        retrieval_times = [r.retrieval_time for r in sample_results]
        
        return {
            "test_count": len(sample_results),
            "avg_f1_score": np.mean(f1_scores),
            "avg_semantic_similarity": np.mean(semantic_scores),
            "avg_llm_accuracy": np.mean(llm_scores),
            "avg_retrieval_time": np.mean(retrieval_times),
            "success_rate": len([r for r in sample_results if r.evaluation_scores.get("token_f1", 0.0) > 0.5]) / len(sample_results),
            "high_quality_rate": len([r for r in sample_results if r.evaluation_scores.get("llm_accuracy", 0.0) > 0.8]) / len(sample_results)
        }
    
    def _calculate_question_type_performance(self, type_results: List[GraphRetrievalResult]) -> Dict[str, Any]:
        """Calculate question type performance."""
        if not type_results:
            return {}
        
        return {
            "test_count": len(type_results),
            "avg_f1_score": np.mean([r.evaluation_scores.get("token_f1", 0.0) for r in type_results]),
            "avg_semantic_similarity": np.mean([r.evaluation_scores.get("semantic_similarity", 0.0) for r in type_results]),
            "avg_llm_accuracy": np.mean([r.evaluation_scores.get("llm_accuracy", 0.0) for r in type_results]),
            "avg_retrieval_time": np.mean([r.retrieval_time for r in type_results]),
            "success_rate": len([r for r in type_results if r.evaluation_scores.get("token_f1", 0.0) > 0.5]) / len(type_results)
        }
    
    def _calculate_category_performance(self, category_results: List[GraphRetrievalResult]) -> Dict[str, Any]:
        """Calculate category performance."""
        return self._calculate_question_type_performance(category_results)
    
    def _generate_readable_report(self, benchmark_report: Dict[str, Any], timestamp: str = None):
        """Generate readable report."""
        lines = []
        
        lines.append("=" * 80)
        lines.append("LoCoMo实体关系知识图谱Benchmark测试报告")
        lines.append("=" * 80)
        
        info = benchmark_report["benchmark_info"]
        lines.append(f"\n 测试概况:")
        lines.append(f"   - 测试类型: {info['test_type']}")
        lines.append(f"   - 测试方法: {info['method_used']}")
        lines.append(f"   - 语义图数量: {info['total_graphs']}")
        lines.append(f"   - 测试用例数: {info['total_test_cases']}")
        lines.append(f"   - 实际测试数: {info['total_tests_run']}")
        lines.append(f"   - 成功率: {info['success_rate']:.2%}")
        lines.append(f"   - 使用实体关系检索: {info['use_entity_relation']}")
        lines.append(f"   - 多线程工作数: {info['max_workers']}")
        lines.append(f"   - 语义检索top-k: {info['topk_similarity']}")
        lines.append(f"   - 图检索top-k: {info['topk_graph']}")
        
        overall = benchmark_report["overall_statistics"]
        lines.append(f"\n 整体性能:")
        lines.append(f"   - 平均检索时间: {overall['avg_retrieval_time']:.3f}s")
        lines.append(f"   - 平均F1分数: {overall['avg_f1_score']:.3f}")
        lines.append(f"   - 平均语义相似度: {overall['avg_semantic_similarity']:.3f}")
        lines.append(f"   - 平均LLM准确率: {overall['avg_llm_accuracy']:.3f}")
        lines.append(f"   - 平均精确匹配: {overall['avg_exact_match']:.3f}")
        
        sample_perf = benchmark_report["sample_performance"]
        sample_ranking = sorted(
            sample_perf.items(),
            key=lambda x: x[1]["avg_f1_score"],
            reverse=True
        )
        
        lines.append(f"\n 样本性能排名 (按F1分数，前10个):")
        for rank, (sample_id, stats) in enumerate(sample_ranking[:10], 1):
            lines.append(
                f"   {rank}. {sample_id}: F1={stats['avg_f1_score']:.3f}, "
                f"语义={stats['avg_semantic_similarity']:.3f}, "
                f"LLM={stats['avg_llm_accuracy']:.3f}, "
                f"测试数={stats['test_count']}"
            )
        
        q_type_perf = benchmark_report["question_type_performance"]
        lines.append(f"\n 问题类型性能:")
        for q_type, stats in q_type_perf.items():
            lines.append(f"\n   {q_type.upper()} 问题:")
            lines.append(f"     - 测试数量: {stats['test_count']}")
            lines.append(f"     - 平均F1: {stats['avg_f1_score']:.3f}")
            lines.append(f"     - 平均语义相似度: {stats['avg_semantic_similarity']:.3f}")
            lines.append(f"     - 平均LLM准确率: {stats['avg_llm_accuracy']:.3f}")
            lines.append(f"     - 成功率: {stats['success_rate']:.2%}")
        
        cat_perf = benchmark_report["category_performance"]
        lines.append(f"\n 问题类别性能:")
        category_names = {1: "多跳问题", 2: "时间问题", 3: "开放域问题", 4: "单跳问题", 5: "对抗性问题"}
        for category, stats in cat_perf.items():
            category_name = category_names.get(category, f"类别{category}")
            lines.append(f"\n   {category_name}:")
            lines.append(f"     - 测试数量: {stats['test_count']}")
            lines.append(f"     - 平均F1: {stats['avg_f1_score']:.3f}")
            lines.append(f"     - 平均语义相似度: {stats['avg_semantic_similarity']:.3f}")
            lines.append(f"     - 平均LLM准确率: {stats['avg_llm_accuracy']:.3f}")
            lines.append(f"     - 成功率: {stats['success_rate']:.2%}")
        
        
        if timestamp:
            readable_report_file = self.output_dir / f"locomo_graph_benchmark_readable_report_{timestamp}.txt"
        else:
            readable_report_file = self.output_dir / "locomo_graph_benchmark_readable_report.txt"
            
        with open(readable_report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        print('\n'.join(lines))
        
        logger.info(f"可读性报告已生成: {readable_report_file}")


def main():
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(description="LoCoMo实体关系知识图谱Benchmark测试")
    
    parser.add_argument("--semantic-graphs-dir", 
                       default=str(paths.LOCOMO_ENTITY_RELATION_STEP3_DIR),
                       help="语义图存储目录")
    parser.add_argument("--qa-dataset", 
                       default=str(paths.LOCOMO_RAW_FILE),
                       help="QA数据集路径")
    parser.add_argument("--output-dir", 
                       default=str(paths.LOCOMO_TASK_EVAL_RESULTS_DIR / "locomo_graph_benchmark_results"),
                       help="输出目录")

    parser.add_argument("--llm-model", 
                       default="gpt-4o-mini-closeai",
                       help="答案生成LLM模型名称（默认gpt-4o-mini-closeai）")
    
    parser.add_argument("--llm-evaluate-model", 
                       default="gpt-4o-mini-closeai",
                       help="答案评估LLM模型名称（默认gpt-4o-mini-closeai）")
    
    parser.add_argument("--max-graphs", type=int,
                       help="最大加载图数量（测试用）")
    parser.add_argument("--sample-ids", nargs='+',
                       help="指定要测试的样本ID列表，例: --sample-ids conv-26 conv-30")
    parser.add_argument("--no-entity-relation", action="store_true",
                       help="禁用实体关系检索，只使用Triple_Fusion_BAAI_Rerank")
    parser.add_argument("--max-workers", type=int, default=1,
                       help="多线程工作线程数，1表示单线程")
    
    parser.add_argument("--topk-similarity", type=int, default=15,
                       help="语义检索的top-k数量（默认15）")
    parser.add_argument("--topk-graph", type=int, default=5,
                       help="图检索的top-k数量（默认5）")
    
    
    parser.add_argument('--reranker-type',
                       choices=['baai', 'qwen', 'jina', 'qwen-sili', 'qwen-dashscope', 'gte-dashscope'],
                       default='baai',
                       help='重排序器类型 (baai/qwen/jina本地, qwen-sili/qwen-dashscope/gte-dashscope云端API)')
    
    parser.add_argument('--reranker-model',
                       help='自定义重排序器模型名称')
    
    parser.add_argument("--log-level", 
                       default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="日志级别")
    
    args = parser.parse_args()
    
    
    print("=" * 80)
    print(" LoCoMo实体关系知识图谱Benchmark测试系统")
    print("=" * 80)
    print(f" 语义图目录: {args.semantic_graphs_dir}")
    print(f" QA数据集: {args.qa_dataset}")
    print(f" 输出目录: {args.output_dir}")
    print(f" 答案生成模型: {args.llm_model}")
    print(f" 答案评估模型: {args.llm_evaluate_model}")
    print(f" 使用实体关系检索: {not args.no_entity_relation}")
    print(f" 工作线程数: {args.max_workers}")
    print(f" 语义检索top-k: {args.topk_similarity}")
    print(f" 图检索top-k: {args.topk_graph}")
    print(f" 重排序器类型: {args.reranker_type}")
    
    
    if args.reranker_type in ['qwen-sili']:
        api_key = os.getenv("SILICONFLOW_API_KEY")
        print(f"{'configured' if api_key else 'missing'} cloud reranker API (Siliconflow): {'configured' if api_key else 'set SILICONFLOW_API_KEY'}")
    elif args.reranker_type in ['qwen-dashscope', 'gte-dashscope']:
        api_key = os.getenv("DASHSCOPE_API_KEY")
        print(f"{'configured' if api_key else 'missing'} cloud reranker API (DashScope): {'configured' if api_key else 'set DASHSCOPE_API_KEY'}")
    
    if args.sample_ids:
        print(f" 指定样本: {args.sample_ids}")
    
    try:
        print("\n 初始化LLM客户端...")
        llm_client = LLMClient(model_name=args.llm_model)
        llm_evaluate_client = LLMClient(model_name=args.llm_evaluate_model)
        print(f" LLM客户端初始化完成")
        
        
        reranker_configs = {
            "baai": args.reranker_model if args.reranker_model and args.reranker_type == "baai" else "BAAI/bge-reranker-v2-m3",
            "qwen": args.reranker_model if args.reranker_model and args.reranker_type == "qwen" else "Qwen/Qwen3-Reranker-0.6B",
            "jina": args.reranker_model if args.reranker_model and args.reranker_type == "jina" else "jinaai/jina-reranker-v2-base-multilingual",
            "qwen-sili": args.reranker_model if args.reranker_model and args.reranker_type == "qwen-sili" else "Qwen/Qwen3-Reranker-8B",
            "qwen-dashscope": args.reranker_model if args.reranker_model and args.reranker_type == "qwen-dashscope" else "qwen3-rerank",
            "gte-dashscope": args.reranker_model if args.reranker_model and args.reranker_type == "gte-dashscope" else "gte-rerank-v2"
        }
        
        
        print(f"\n 初始化重排序器管理器 ({args.reranker_type})...")
        global_reranker_manager = RerankerManager()
        
        
        try:
            reranker = global_reranker_manager.get_reranker(
                reranker_type=args.reranker_type,
                model_name=reranker_configs[args.reranker_type]
            )
            print(f" 重排序器预加载成功: {args.reranker_type}")
        except Exception as e:
            print(f"  重排序器预加载失败: {e}")
            print(f"   将在使用时按需加载")
        
        print("\n 初始化Benchmark测试器...")
        benchmark = LoCoMoEntityRelationBenchmark(
            semantic_graphs_dir=args.semantic_graphs_dir,
            qa_dataset_path=args.qa_dataset,
            llm_client=llm_client,
            llm_evaluate_client=llm_evaluate_client,
            output_dir=args.output_dir,
            use_entity_relation=not args.no_entity_relation,
            target_sample_ids=args.sample_ids,
            max_workers=args.max_workers,
            topk_similarity=args.topk_similarity,
            topk_graph=args.topk_graph,
            reranker_type=args.reranker_type,
            reranker_configs=reranker_configs,
            reranker_manager=global_reranker_manager
        )
        print(" Benchmark测试器初始化完成")
        
        
        print(f"\n 加载语义图...")
        benchmark.load_semantic_graphs(max_graphs=args.max_graphs)
        print(f" 成功加载 {len(benchmark.semantic_graphs)} 个语义图")
        
        
        print(f"\n 加载测试用例...")
        benchmark.load_test_cases()
        print(f" 加载了 {len(benchmark.test_cases)} 个测试用例")
        
        print(f"\n 开始运行benchmark测试...")
        print(f"   重排序器: {args.reranker_type}")
        benchmark.run_comprehensive_benchmark()
        
        print(f"\n 生成benchmark报告...")
        benchmark.generate_benchmark_report()
        
        print(f"\n Benchmark测试完成!")
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
    
    return 0


if __name__ == "__main__":
    exit(main())
