#!/usr/bin/env python3
"""Utilities for benchmark episodic memory."""

import os
import sys
import json
import logging
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from tqdm import tqdm


# Avoid mutating LogRecord fields before other handlers process the record.
from mandol.utils.logging_config import setup_logging, create_module_logger, auto_configure_logging
if auto_configure_logging() is None:
    setup_logging(level=logging.INFO)
logger = create_module_logger("benchmark_episodic_memory")

from mandol.core.semantic_graph import SemanticGraph
from mandol.core.memory_unit import MemoryUnit
from mandol.llm.llm_client import LLMClient
from mandol.retrieval.retrieval_interface import RetrievalMethod
from mandol.core import paths


try:
    from benchmark_longmemeval.task_eval.evaluation import (
        calculate_comprehensive_scores,
        cleanup_evaluation_models
    )
    EVALUATION_AVAILABLE = True
except ImportError:
    EVALUATION_AVAILABLE = False
    logger.warning(" 评估模块不可用")

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.warning(" tiktoken 未安装，将使用近似计算")



@dataclass
class EpisodicMemoryTestCase:
    qa_index: int
    question_id: str
    question: str
    answer: str
    question_type: str = ""
    category: str = ""
    query_date: str = "Unknown Date"
    
    temporal_scope: Optional[str] = None
    fact_categories: List[str] = field(default_factory=list)


@dataclass
class TokenStats:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    context_tokens: int = 0


@dataclass
class RetrievalDetails:
    method: str = "hybrid"
    total_retrieved: int = 0
    retrieval_time: float = 0.0
    rerank_time: float = 0.0
    
    facts_by_category: Dict[str, int] = field(default_factory=dict)
    stable_facts_count: int = 0
    temporal_facts_count: int = 0
    avg_confidence: float = 0.0


@dataclass 
class EpisodicMemoryResult:
    qa_index: int
    question: str
    ground_truth: str
    generated_answer: str
    reasoning: str
    
    scores: Dict[str, float] = field(default_factory=dict)
    
    retrieval_details: RetrievalDetails = field(default_factory=RetrievalDetails)
    retrieved_facts: List[Dict[str, Any]] = field(default_factory=list)
    
    token_stats: TokenStats = field(default_factory=TokenStats)
    
    total_time: float = 0.0
    success: bool = True
    error_message: str = ""



class LongMemEvalEpisodicMemoryBenchmark:
    
    
    VALID_RERANK_METHODS = ["baai", "qwen", "jina", "qwen-sili", "qwen-dashscope", "gte-dashscope", "none"]
    
    STABLE_CATEGORIES = [
        "USER_ATTRIBUTE", "PREFERENCE_HABIT", "RELATIONSHIP_FACT", "KNOWLEDGE", 
        "IMPLICIT_CONSTRAINT", "INVENTORY_ITEM"
    ]
    
    def __init__(self,
                 dataset_size: str = "s",
                 dataset_dir: Optional[str] = None,
                 graph_data_dir: Optional[str] = None,
                 llm_client: Optional[LLMClient] = None,
                 llm_evaluate_client: Optional[LLMClient] = None,
                 output_dir: str = str(paths.LONGMEMEVAL_TASK_EVAL_RESULTS_DIR / "episodic_memory"),
                 top_k: int = 10,
                 max_tests: Optional[int] = None,
                 rerank_method: str = "baai",
                 enable_category_filter: bool = False,
                 enable_temporal_filter: bool = False):
        
        self.dataset_size = dataset_size
        
        # Dataset-specific handling used by the reproduction workflow.
        self.dataset_dir = Path(dataset_dir) if dataset_dir else \
            Path(__file__).parent.parent / "dataset" / "LongMemEval"
        self.dataset_path = self.dataset_dir / f"longmemeval_{dataset_size}_cleaned.json"
        
        self.graph_data_dir = Path(graph_data_dir) if graph_data_dir else \
            self.dataset_dir / "episodic_memory_graphs"
        
        self.llm_client = llm_client
        self.llm_evaluate_client = llm_evaluate_client
        
        self.output_dir = Path(output_dir)
        self.top_k = top_k
        self.max_tests = max_tests
        
        
        if rerank_method not in self.VALID_RERANK_METHODS:
            raise ValueError(f"无效的重排序方法: {rerank_method}，支持: {self.VALID_RERANK_METHODS}")
        self.rerank_method = rerank_method
        
        self.enable_category_filter = enable_category_filter
        self.enable_temporal_filter = enable_temporal_filter
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.stats = {
            'total_tests': 0,
            'successful_tests': 0,
            'failed_tests': 0,
            'total_facts_retrieved': 0,
            'facts_by_category': defaultdict(int),
            'stable_facts_count': 0,
            'temporal_facts_count': 0,
        }
        
        logger.info("=" * 80)
        logger.info(" LongMemEval 情景记忆 Benchmark 测试器 (Refactored)")
        logger.info("=" * 80)
        logger.info(f" 数据集: {self.dataset_path}")
        logger.info(f" 图谱目录: {self.graph_data_dir}")
        logger.info(f" 重排序: {self._get_rerank_description()}")
        logger.info(f" Top-K: {self.top_k}")
    
    def _get_rerank_description(self) -> str:
        """Get rerank description."""
        descriptions = {
            "baai": "BAAI/bge-reranker-v2-m3 (本地)",
            "qwen": "Qwen/Qwen3-Reranker-0.6B (本地)",
            "jina": "jinaai/jina-reranker-v3 (本地)",
            "qwen-sili": "Qwen/Qwen3-Reranker-8B (云端)",
            "qwen-dashscope": "qwen3-rerank (云端)",
            "gte-dashscope": "gte-rerank-v2 (云端)",
            "none": "不使用重排序"
        }
        return descriptions.get(self.rerank_method, self.rerank_method)
    
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
    
    
    
    def _load_test_cases(self) -> List[EpisodicMemoryTestCase]:
        """Load test cases."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"数据集文件不存在: {self.dataset_path}")
        
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        
        if isinstance(dataset, dict) and "qa_pairs" in dataset:
            dataset = dataset["qa_pairs"]
        
        test_cases = []
        for idx, item in enumerate(dataset):
            if self.max_tests and idx >= self.max_tests:
                break
            
            test_case = EpisodicMemoryTestCase(
                qa_index=idx,
                question_id=item.get("question_id", f"q_{idx}"),
                question=item["question"],
                answer=item["answer"],
                question_type=item.get("question_type", self._analyze_question_type(item["question"])),
                category=item.get("category", ""),
                query_date=item.get("question_date", item.get("date", item.get("session_date", "Unknown Date")))
            )
            test_cases.append(test_case)
        
        logger.info(f" 加载 {len(test_cases)} 个测试用例")
        return test_cases
    
    def _analyze_question_type(self, question: str) -> str:
        """Run analyze question type."""
        question_lower = question.lower()
        if any(kw in question_lower for kw in ['when', 'what time', 'what date', 'how long', 'how many days']): return "temporal"
        if any(kw in question_lower for kw in ['who', 'whose', 'relationship']): return "relationship"
        if any(kw in question_lower for kw in ['prefer', 'like', 'favorite']): return "preference"
        if any(kw in question_lower for kw in ['plan', 'schedule']): return "plan"
        if any(kw in question_lower for kw in ['did i', 'have i']): return "event"
        return "factual"
    
    def _load_semantic_graph_for_qa(self, qa_index: int) -> Tuple[Optional[SemanticGraph], Optional[Dict]]:
        """Load semantic graph for qa."""
        qa_dir = self.graph_data_dir / f"qa_{qa_index}"
        
        if not qa_dir.exists():
            logger.warning(f" QA {qa_index} 的图谱目录不存在: {qa_dir}")
            return None, None
        
        try:
            metadata_file = qa_dir / "meta_info.json"
            qa_metadata = {}
            embedding_model = "Qwen/Qwen3-Embedding-0.6B"
            
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    qa_metadata = json.load(f)
                    if "config" in qa_metadata:
                        embedding_model = qa_metadata["config"].get("embedding_model", embedding_model)
            
            
            
            semantic_graph = SemanticGraph.load_graph(
                str(qa_dir),
                embedding_model_name=embedding_model
            )
            
            logger.debug(f" 加载 QA {qa_index} 的 SemanticGraph 成功 (Model: {embedding_model})")
            return semantic_graph, qa_metadata
            
        except Exception as e:
            logger.error(f" 加载 QA {qa_index} 的 SemanticGraph 失败: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None, None
    
    
    def _retrieve_context(self,
                         semantic_graph: SemanticGraph,
                         question: str,
                         question_type: str = "general") -> Tuple[List[Tuple[MemoryUnit, float]], RetrievalDetails]:
        """Run retrieve context."""
        retrieval_details = RetrievalDetails(method="hybrid_rrf")
        start_time = time.time()
        
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
                    top_k=self.top_k, 
                    fusion_method="rrf",
                    rerank_method=rerank_config,
                    return_detailed=False
                )
            else:
                logger.warning(" MultiRetriever 未初始化，回退到简单搜索")
                results = semantic_graph.search_similarity_in_graph(
                    query_text=question, top_k=self.top_k, return_score=True
                )
            
            retrieval_time = time.time() - start_time
            
            final_results = []
            if results:
                if isinstance(results[0], tuple):
                    final_results = results[:self.top_k]
                elif isinstance(results[0], MemoryUnit):
                    final_results = [(unit, 1.0) for unit in results[:self.top_k]]
            
            facts_by_category = defaultdict(int)
            stable_count = 0
            temporal_count = 0
            confidence_sum = 0.0
            
            for unit, score in final_results:
                raw_data = unit.raw_data if hasattr(unit, 'raw_data') else {}
                
                category = raw_data.get('category', raw_data.get('node_type', 'UNKNOWN')).upper()
                facts_by_category[category] += 1
                
                event_date = raw_data.get('event_date') or raw_data.get('temporal_val') or raw_data.get('time')
                if event_date:
                    temporal_count += 1
                
                is_stable = raw_data.get('is_stable')
                if is_stable is None:
                    is_stable = category in self.STABLE_CATEGORIES
                if is_stable:
                    stable_count += 1
                
                confidence = float(raw_data.get('confidence', 1.0))
                confidence_sum += confidence
            
            retrieval_details.retrieval_time = retrieval_time
            retrieval_details.total_retrieved = len(final_results)
            retrieval_details.facts_by_category = dict(facts_by_category)
            retrieval_details.stable_facts_count = stable_count
            retrieval_details.temporal_facts_count = temporal_count
            retrieval_details.avg_confidence = confidence_sum / len(final_results) if final_results else 0.0
            
            return final_results, retrieval_details
            
        except Exception as e:
            logger.error(f"检索失败: {e}", exc_info=True)
            return [], retrieval_details
    
    def _generate_answer_from_context(self,
                                    question: str,
                                    retrieved_units: List[Tuple[MemoryUnit, float]],
                                    question_type: str,
                                    query_date: str = "Unknown Date") -> Tuple[str, str, TokenStats]:
        """Generate answer from context."""
        token_stats = TokenStats()
        
        if not self.llm_client:
            return "LLM client not configured", "", token_stats
        
        if not retrieved_units:
            return "I don't have enough information to answer this question.", "No relevant facts found.", token_stats
        
        context_parts = []
        for i, (unit, score) in enumerate(retrieved_units, 1):
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
                f"[Fact {i}] "
                f"Type: {category} | "
                f"Time: {event_date} | "
                f"Stability: {stability_str}\n"
                f"Content: {content}\n"
            )
            
            context_parts.append(formatted_fact)
        
        context_text = "\n".join(context_parts)
        
        # Dataset-specific handling used by the reproduction workflow.
        
        prompt = f"""You are a helpful expert assistant answering questions based on episodic memory facts.

        # CURRENT REFERENCE TIME
        The current time for this question is: **{query_date}**
        *** CRITICAL INSTRUCTION ***
        - Treat "{query_date}" as "TODAY" or "NOW".
        - All relative time references ("yesterday", "last week", "3 days ago") MUST be calculated relative to this date.
        - Do NOT use the actual real-world date.

        # RETRIEVED MEMORIES:
        You have access to episodic memory facts retrieved from a long-term memory system.
        Note that "Content" is the core event, while "Time" and "Type" are metadata to help you reason about order and validity.

        {context_text}

        # QUESTION:
        {question}

        # INSTRUCTIONS:
        1. Analyze the facts, strictly paying attention to the 'Time' field for temporal reasoning (e.g., "when", "before", "after").
        2. If facts conflict, prioritize more recent (by Time) or stable information.
        3. Think step-by-step to derive the answer.
        4. If the answer is not in the facts, state "I don't have enough information".

        # OUTPUT FORMAT (JSON):
        {{
            "reasoning": "Step-by-step analysis, explicitly citing Time and Content...",
            "final_answer": "Concise answer."
        }}
        """
        
        token_stats.prompt_tokens = self._count_tokens(prompt)
        token_stats.context_tokens = self._count_tokens(context_text)
        
        try:
            response = self.llm_client.generate_answer(prompt, temperature=0.1, json_format=True)
            token_stats.completion_tokens = self._count_tokens(response)
            token_stats.total_tokens = token_stats.prompt_tokens + token_stats.completion_tokens
            
            try:
                # Cleaning response for JSON parsing
                clean_response = response.strip()
                if clean_response.startswith("```json"):
                    clean_response = clean_response.replace("```json", "").replace("```", "")
                
                parsed = json.loads(clean_response)
                reasoning = parsed.get("reasoning", "No reasoning provided")
                answer = parsed.get("final_answer", "Unable to answer")
                return answer, reasoning, token_stats
            except json.JSONDecodeError:
                return response, "Failed to parse JSON", token_stats
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return f"Generation failed: {str(e)}", "", token_stats


    def _evaluate_result(self, test_case, generated_answer, retrieved_units, reasoning) -> Dict[str, Any]:
        """Run evaluate result."""
        if not EVALUATION_AVAILABLE:
            return {"evaluation_available": False, "scores": {}}
        
        try:
            retrieved_texts = []
            for unit, _ in retrieved_units:
                raw = unit.raw_data if hasattr(unit, 'raw_data') else {}
                content = raw.get('content', raw.get('text_content', str(raw)))
                event_date = raw.get('event_date', '')
                retrieved_texts.append(f"[{event_date}] {content}")
            
            context_text = "\n".join(retrieved_texts)
            
            return calculate_comprehensive_scores(
                gold_answer=test_case.answer,
                response=generated_answer,
                question=test_case.question,
                context=context_text,
                question_type=test_case.question_type,
                llm_client=self.llm_evaluate_client
            )
        except Exception as e:
            logger.error(f"评估失败: {e}")
            return {"error": str(e), "scores": {}}

    def _test_single_qa(self, test_case: EpisodicMemoryTestCase, test_index: int) -> Dict[str, Any]:
        """Run test single qa."""
        test_start = time.time()
        
        try:
            logger.info(f"\n 测试 {test_index} | QA: qa_{test_case.qa_index} | Type: {test_case.question_type}")
            
            
            semantic_graph, qa_metadata = self._load_semantic_graph_for_qa(test_case.qa_index)
            if not semantic_graph:
                return {"error": "Failed to load graph", "qa_index": test_case.qa_index}
            
            retrieved_units, retrieval_details = self._retrieve_context(
                semantic_graph, test_case.question, test_case.question_type
            )
            
            answer, reasoning, token_stats = self._generate_answer_from_context(
                test_case.question, retrieved_units, test_case.question_type,
                query_date=test_case.query_date
            )
            
            eval_result = self._evaluate_result(test_case, answer, retrieved_units, reasoning)
            
            result = {
                "qa_index": test_case.qa_index,
                "question": test_case.question,
                "gold_answer": test_case.answer,
                "generated_answer": answer,
                "reasoning": reasoning,
                "retrieval_details": asdict(retrieval_details),
                "evaluation": eval_result,
                "token_stats": asdict(token_stats),
                "timestamp": datetime.now().isoformat(),
                "success": True
            }
            
            self.stats['successful_tests'] += 1
            self.stats['total_facts_retrieved'] += len(retrieved_units)
            for k, v in retrieval_details.facts_by_category.items():
                self.stats['facts_by_category'][k] += v
            
            
            self._save_single_report(result, test_index)
            
            return result
            
        except Exception as e:
            logger.error(f"测试失败 qa_{test_case.qa_index}: {e}")
            self.stats['failed_tests'] += 1
            return {"qa_index": test_case.qa_index, "error": str(e), "success": False}

    def _save_single_report(self, result: Dict, index: int):
        """Save single report."""
        qa_idx = result.get('qa_index', index)
        if not hasattr(self, '_current_report_dir'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._current_report_dir = self.output_dir / f"run_{timestamp}" / "individual_reports"
            self._current_report_dir.mkdir(parents=True, exist_ok=True)
            
        file_path = self._current_report_dir / f"qa_{qa_idx}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    def run_benchmark(self) -> Dict[str, Any]:
        """Run benchmark."""
        start_time = time.time()
        test_cases = self._load_test_cases()
        self.stats['total_tests'] = len(test_cases)
        
        results = []
        for i, case in enumerate(tqdm(test_cases, desc="Running Benchmark")):
            res = self._test_single_qa(case, i+1)
            results.append(res)
        
        total_time = time.time() - start_time
        
        summary = {
            "total_time": total_time,
            "test_stats": {
                "total": self.stats['total_tests'],
                "successful": self.stats['successful_tests'],
                "failed": self.stats['failed_tests'],
                "success_rate": self.stats['successful_tests'] / max(1, self.stats['total_tests'])
            },
            "retrieval_stats": {
                "total_facts": self.stats['total_facts_retrieved'],
                "facts_by_category": dict(self.stats['facts_by_category'])
            }
        }
        
        scores_sum = defaultdict(float)
        valid_count = 0
        for r in results:
            if r.get('success') and 'evaluation' in r:
                scores = r['evaluation'].get('scores', {})
                if scores:
                    valid_count += 1
                    for k, v in scores.items():
                        scores_sum[k] += v
        
        if valid_count > 0:
            summary['average_scores'] = {k: v/valid_count for k, v in scores_sum.items()}
        
        
        summary_path = self.output_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump({"summary": summary, "results": results}, f, indent=2, ensure_ascii=False)
            
        logger.info(f" 测试完成，结果已保存至 {summary_path}")
        return summary


def main():
    parser = argparse.ArgumentParser(description="LongMemEval Episodic Memory Benchmark (Refactored)")
    parser.add_argument("--dataset-size", default="s", choices=["s", "m"], help="数据集大小")
    parser.add_argument("--dataset-dir", default=None, help="数据集目录")
    parser.add_argument("--graph-data-dir", default=str(paths.LONGMEMEVAL_EPISODIC_GRAPHS_DIR), help="图谱数据目录")
    parser.add_argument("--max-tests", type=int, default=None, help="最大测试数")
    parser.add_argument("--llm-model", default="gpt-4o-mini-closeai", help="生成模型")
    parser.add_argument("--llm-evaluate-model", default="gpt-4o-mini-closeai", help="评估模型")
    parser.add_argument("--rerank-method", default="baai", choices=["baai", "qwen", "none"], help="重排序方法")
    
    args = parser.parse_args()
    
    llm_client = LLMClient(model_name=args.llm_model)
    llm_eval = LLMClient(model_name=args.llm_evaluate_model)
    
    benchmark = LongMemEvalEpisodicMemoryBenchmark(
        dataset_size=args.dataset_size,
        dataset_dir=args.dataset_dir,
        graph_data_dir=args.graph_data_dir,
        llm_client=llm_client,
        llm_evaluate_client=llm_eval,
        max_tests=args.max_tests,
        rerank_method=args.rerank_method
    )
    
    benchmark.run_benchmark()

if __name__ == "__main__":
    main()