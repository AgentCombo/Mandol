#!/usr/bin/env python3
"""Utilities for benchmark entity relation."""

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
logger = create_module_logger("benchmark_entity_relation")

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
class EntityRelationTestCase:
    qa_index: int
    question_id: str
    question: str
    answer: str
    question_type: str = "unknown"
    query_date: str = "Unknown Date"
    relations: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class TokenStats:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    context_tokens: int = 0


@dataclass
class RetrievalDetails:
    method: str = "hybrid"
    total_entities_retrieved: int = 0
    retrieval_time: float = 0.0
    
    entity_types_found: Dict[str, int] = field(default_factory=dict)


@dataclass 
class EntityRelationResult:
    qa_index: int
    question: str
    ground_truth: str
    generated_answer: str
    reasoning: str
    
    scores: Dict[str, float] = field(default_factory=dict)
    
    retrieval_details: RetrievalDetails = field(default_factory=RetrievalDetails)
    
    token_stats: TokenStats = field(default_factory=TokenStats)
    
    total_time: float = 0.0
    success: bool = True
    error_message: str = ""



class LongMemEvalEntityRelationBenchmark:
    
    VALID_RERANK_METHODS = ["baai", "qwen", "jina", "qwen-sili", "qwen-dashscope", "gte-dashscope", "none"]
    
    def __init__(self,
                 dataset_size: str = "s",
                 dataset_dir: Optional[str] = None,
                 graph_data_dir: Optional[str] = None,
                 llm_client: Optional[LLMClient] = None,
                 llm_evaluate_client: Optional[LLMClient] = None,
                 output_dir: str = str(paths.LONGMEMEVAL_TASK_EVAL_RESULTS_DIR / "entity_relation_no_expand"),
                 top_k: int = 10,
                 max_tests: Optional[int] = None,
                 rerank_method: str = "baai"):
        
        self.dataset_size = dataset_size
        
        self.dataset_dir = Path(dataset_dir) if dataset_dir else \
            Path(__file__).parent.parent / "dataset" / "LongMemEval"
        self.dataset_path = self.dataset_dir / f"longmemeval_{dataset_size}_cleaned.json"
        
        self.graph_data_dir = Path(graph_data_dir) if graph_data_dir else \
            self.dataset_dir / "entity_relation_graphs"
        
        self.llm_client = llm_client
        self.llm_evaluate_client = llm_evaluate_client
        self.output_dir = Path(output_dir)
        self.top_k = top_k
        self.max_tests = max_tests
        
        if rerank_method not in self.VALID_RERANK_METHODS:
            raise ValueError(f"无效的重排序方法: {rerank_method}")
        self.rerank_method = rerank_method
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.stats = {
            'total_tests': 0,
            'successful_tests': 0,
            'failed_tests': 0,
            'total_entities_retrieved': 0
        }
        
        logger.info("=" * 80)
        logger.info(" LongMemEval 实体关系 Benchmark (No Graph Expansion)")
        logger.info("=" * 80)
        logger.info(f" 数据集: {self.dataset_path}")
        logger.info(f" 图谱目录: {self.graph_data_dir}")
        logger.info(f" 重排序: {rerank_method}")
        logger.info(f" Top-K: {self.top_k}")
    
    def _count_tokens(self, text: str, model: str = "gpt-4") -> int:
        if not text: return 0
        if TIKTOKEN_AVAILABLE:
            try:
                encoding = tiktoken.encoding_for_model(model) if "gpt" in model else tiktoken.get_encoding("cl100k_base")
                return len(encoding.encode(text))
            except Exception: pass
        return int(len(text) / 4)
    
    
    
    def _load_test_cases(self) -> List[EntityRelationTestCase]:
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"数据集文件不存在: {self.dataset_path}")
        
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        test_cases = []
        for idx, item in enumerate(dataset):
            if self.max_tests and idx >= self.max_tests: break
            
            relations = item.get("relations", item.get("relationships", []))
            # Dataset-specific handling used by the reproduction workflow.
            question_type = item.get("question_type", "unknown")
            test_case = EntityRelationTestCase(
                qa_index=idx,
                question_id=item.get("question_id", f"q_{idx}"),
                question=item["question"],
                answer=item["answer"],
                question_type=question_type,
                query_date=item.get("question_date", item.get("date", item.get("session_date", "Unknown Date"))),
                relations=relations
            )
            test_cases.append(test_case)
        
        logger.info(f" 加载 {len(test_cases)} 个测试用例")
        return test_cases
    
    def _load_semantic_graph_for_qa(self, qa_index: int) -> Optional[SemanticGraph]:
        qa_dir = self.graph_data_dir / f"qa_{qa_index}"
        if not qa_dir.exists():
            logger.warning(f" QA {qa_index} 的图谱目录不存在: {qa_dir}")
            return None
        
        try:
            semantic_graph = SemanticGraph.load_graph(
                str(qa_dir),
                embedding_model_name="Qwen/Qwen3-Embedding-0.6B"
            )
            logger.debug(f" 加载 QA {qa_index} 成功")
            return semantic_graph
        except Exception as e:
            logger.error(f" 加载 QA {qa_index} 失败: {e}")
            return None
    
    
    def _retrieve_context(self,
                         semantic_graph: SemanticGraph,
                         question: str) -> Tuple[List[str], RetrievalDetails]:
        """Run retrieve context."""
        retrieval_details = RetrievalDetails(method="hybrid_rrf_no_expand")
        start_time = time.time()
        
        try:
            multi_retriever = semantic_graph.get_multi_retriever()
            
            methods = [RetrievalMethod.BM25, RetrievalMethod.SPLADE, RetrievalMethod.COSINE_SIMILARITY]
            rerank_config = None if self.rerank_method == "none" else self.rerank_method
            
            if multi_retriever:
                results = multi_retriever.smart_search(
                    query=question,
                    methods=methods,
                    top_k=self.top_k,
                    fusion_method="rrf",
                    rerank_method=rerank_config,
                    return_detailed=False
                )
            else:
                results = semantic_graph.search_similarity_in_graph(
                    query_text=question, top_k=self.top_k, return_score=True
                )

            
            if len(results) > self.top_k:
                results = results[:self.top_k]
            
            retrieval_details.total_entities_retrieved = len(results)
            
            context_lines = []
            entity_types_found = defaultdict(int)
            
            for i, (unit, score) in enumerate(results, 1):
                raw = unit.raw_data
                
                main_content = raw.get("text_content")
                
                if not main_content:
                    name = raw.get("entity_canonical") or raw.get("entity_text") or unit.uid
                    etype = raw.get("entity_category") or "Unknown"
                    desc = raw.get("content") or "No description"
                    main_content = f"Entity: {name} (Type: {etype}) | Context: {desc}"
                
                entity_type = raw.get("entity_category") or raw.get("entity_type") or "Unknown"
                entity_types_found[entity_type] += 1
                
                session_date = raw.get("session_date") or raw.get("date") or raw.get("created_at")
                date_str = f" [Date: {session_date}]" if session_date else ""
                
                record = f"[{i}] {main_content}{date_str}"
                context_lines.append(record)
            
            retrieval_details.retrieval_time = time.time() - start_time
            retrieval_details.entity_types_found = dict(entity_types_found)
            
            return context_lines, retrieval_details
            
        except Exception as e:
            logger.error(f"检索失败: {e}", exc_info=True)
            return [], retrieval_details
    
    
    def _generate_answer_from_context(self,
                                    question: str,
                                    context_lines: List[str],
                                    query_date: str = "Unknown Date") -> Tuple[str, str, TokenStats]:
        """Generate answer from context."""
        token_stats = TokenStats()
        
        if not self.llm_client:
            return "LLM client not configured", "", token_stats
        
        if not context_lines:
            return "I don't have enough information.", "No relevant entities found.", token_stats
        
        context_text = "\n\n".join(context_lines)
        
        prompt = f"""You are an expert assistant answering questions based on retrieved memory records.

        # CURRENT REFERENCE TIME
        The current time for this question is: **{query_date}**
        *** CRITICAL INSTRUCTION ***
        - Treat "{query_date}" as "TODAY" or "NOW".
        - All relative time references ("yesterday", "last week", "3 days ago") MUST be calculated relative to this date.
        - Do NOT use the actual real-world date.

        # MEMORY RECORDS:
        The following are entities retrieved from the memory system. 
        Each record contains the entity name, type, a timestamp (Date), and a description.

        {context_text}

        # QUESTION:
        {question}

        # INSTRUCTIONS:
        1. Review the memory records to find information relevant to the question.
        2. Pay attention to the 'Date' field to understand the temporal context if needed.
        3. Combine information from the descriptions to form a complete answer.
        4. If the answer is not explicitly in the records, state "I don't have enough information".

        # OUTPUT FORMAT (JSON):
        {{
            "reasoning": "Analysis of the records and logical steps...",
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


    def _evaluate_result(self, test_case: EntityRelationTestCase, generated_answer: str, context_text: str) -> Dict[str, Any]:
        if not EVALUATION_AVAILABLE:
            return {"evaluation_available": False, "scores": {}}
        
        try:
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

    def _test_single_qa(self, test_case: EntityRelationTestCase, test_index: int) -> Dict[str, Any]:
        test_start = time.time()
        try:
            logger.info(f"\n 测试 {test_index} | QA: qa_{test_case.qa_index}")
            
            
            semantic_graph = self._load_semantic_graph_for_qa(test_case.qa_index)
            if not semantic_graph:
                return {"error": "Failed to load graph", "qa_index": test_case.qa_index}
            
            context_lines, retrieval_details = self._retrieve_context(
                semantic_graph, test_case.question
            )
            context_text = "\n".join(context_lines)
            
            answer, reasoning, token_stats = self._generate_answer_from_context(
                test_case.question, context_lines,
                query_date=test_case.query_date
            )
            
            eval_result = self._evaluate_result(test_case, answer, context_text)
            
            result = {
                "qa_index": test_case.qa_index,
                "question": test_case.question,
                "question_type": test_case.question_type,
                "gold_answer": test_case.answer,
                "generated_answer": answer,
                "reasoning": reasoning,
                "retrieval_details": asdict(retrieval_details),
                "evaluation": eval_result,
                "token_stats": asdict(token_stats),
                "timestamp": datetime.now().isoformat(),
                "success": True,
                "total_time": time.time() - test_start
            }
            
            self.stats['successful_tests'] += 1
            self.stats['total_entities_retrieved'] += retrieval_details.total_entities_retrieved
            
            self._save_single_report(result, test_index)
            return result
            
        except Exception as e:
            logger.error(f"测试失败 qa_{test_case.qa_index}: {e}")
            self.stats['failed_tests'] += 1
            return {"qa_index": test_case.qa_index, "error": str(e), "success": False}

    def _save_single_report(self, result: Dict, index: int):
        qa_idx = result.get('qa_index', index)
        if not hasattr(self, '_current_report_dir'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._current_report_dir = self.output_dir / f"run_{timestamp}" / "individual_reports"
            self._current_report_dir.mkdir(parents=True, exist_ok=True)
            
        file_path = self._current_report_dir / f"qa_{qa_idx}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    def run_benchmark(self) -> Dict[str, Any]:
        start_time = time.time()
        test_cases = self._load_test_cases()
        self.stats['total_tests'] = len(test_cases)
        
        results = []
        for i, case in enumerate(tqdm(test_cases, desc="Running Entity Benchmark")):
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
                "total_entities": self.stats['total_entities_retrieved']
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
    parser = argparse.ArgumentParser(description="LongMemEval Entity Relation Benchmark (No Expand)")
    parser.add_argument("--dataset-size", default="s", choices=["s", "m"], help="数据集大小")
    parser.add_argument("--dataset-dir", default=None, help="数据集目录")
    parser.add_argument("--graph-data-dir", default=str(paths.LONGMEMEVAL_ENTITY_RELATION_GRAPHS_QWEN_DIR), help="图谱数据目录")
    parser.add_argument("--max-tests", type=int, default=None, help="最大测试数")
    parser.add_argument("--llm-model", default="gpt-4o-mini-closeai", help="生成模型")
    parser.add_argument("--llm-evaluate-model", default="gpt-4o-mini-closeai", help="评估模型")
    parser.add_argument("--rerank-method", default="baai", choices=["baai", "qwen", "none"], help="重排序方法")
    
    args = parser.parse_args()
    
    llm_client = LLMClient(model_name=args.llm_model)
    llm_eval = LLMClient(model_name=args.llm_evaluate_model)
    
    benchmark = LongMemEvalEntityRelationBenchmark(
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