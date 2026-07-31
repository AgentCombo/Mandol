#!/usr/bin/env python3
"""Utilities for benchmark sentence cut."""
import os
import sys
import json
import logging
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import regex
from tqdm import tqdm
from collections import defaultdict
import traceback


# Avoid mutating LogRecord fields before other handlers process the record.
from mandol.utils.logging_config import setup_logging, create_module_logger, auto_configure_logging
if auto_configure_logging() is None:
    setup_logging(level=logging.INFO)
logger = create_module_logger("benchmark_sentence_cut")

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


class LongMemEvalL0Baseline:
    
    def __init__(self,
                dataset_size: str = "s",
                dataset_dir: str = None,
                graph_data_dir: str = None,
                llm_client: Optional[LLMClient] = None,
                llm_evaluate_client: Optional[LLMClient] = None,
                output_dir: str = str(paths.LONGMEMEVAL_TASK_EVAL_RESULTS_DIR / "l0_baseline"),
                top_k: int = 10,
                max_tests: Optional[int] = None,
                rerank_method: str = "baai"):
        self.dataset_size = dataset_size
        
        # Dataset-specific handling used by the reproduction workflow.
        self.dataset_dir = Path(dataset_dir) if dataset_dir else Path(__file__).parent.parent / "dataset" / "LongMemEval"
        self.dataset_path = self.dataset_dir / f"longmemeval_{dataset_size}_cleaned.json"
        
        self.graph_data_dir = Path(graph_data_dir) if graph_data_dir else self.dataset_dir / "raw_longmemeval_message_level"
        
        self.llm_client = llm_client or LLMClient("gpt-4o-mini-closeai")
        self.llm_evaluate_client = llm_evaluate_client or LLMClient("gpt-4o-mini-closeai")
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.top_k = top_k
        self.max_tests = max_tests
        
        
        self.rerank_method = rerank_method.lower() if rerank_method else "none"
        self._valid_rerank_methods = ["baai", "qwen", "jina", "qwen-sili", "qwen-dashscope", "gte-dashscope", "none"]
        if self.rerank_method not in self._valid_rerank_methods:
            raise ValueError(f"不支持的重排序方法: {rerank_method}，支持: {self._valid_rerank_methods}")
        
        
        self.test_cases = self._load_dataset()
        
        
        rerank_desc = self._get_rerank_description()
        
        logger.info(" LongMemEval L0 基线测试器初始化完成")
        logger.info(f"   原始数据集: {self.dataset_path.name}")
        logger.info(f"   图谱数据目录: {self.graph_data_dir}")
        logger.info(f"   测试用例数: {len(self.test_cases)}")
        logger.info(f"   检索 top-k: {self.top_k}")
        logger.info(f"   检索策略: BM25 + SPLADE + Cosine + RRF + {rerank_desc}")

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
    
    def _count_tokens(self, text: str, model: str = "gpt-4") -> int:
        """Count tokens."""
        try:
            import tiktoken
            
            if "gpt-4" in model.lower() or "gpt-3.5" in model.lower():
                encoding = tiktoken.encoding_for_model("gpt-4")
            elif "claude" in model.lower():
                encoding = tiktoken.get_encoding("cl100k_base")
            else:
                encoding = tiktoken.get_encoding("cl100k_base")
            
            tokens = encoding.encode(text)
            return len(tokens)
            
        except ImportError:
            logger.warning("tiktoken未安装，使用近似计算（1 token ≈ 4字符）")
            return len(text) // 4
        except Exception as e:
            logger.warning(f"Token计数失败: {e}，使用近似计算")
            return len(text) // 4
        
    def _load_dataset(self) -> List[Dict[str, Any]]:
        """Load dataset."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"数据集文件不存在: {self.dataset_path}")
        
        logger.info(f" 加载数据集: {self.dataset_path}")
        
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"   加载了 {len(data)} 个测试用例")
        
        for i, test_case in enumerate(data):
            if 'qa_index' not in test_case:
                test_case['qa_index'] = i
        
        if self.max_tests:
            data = data[:self.max_tests]
            logger.info(f"   限制测试数量: {len(data)}")
        
        return data
    
    def _load_semantic_graph_for_qa(self, qa_index: int) -> Optional[SemanticGraph]:
        """Load semantic graph for qa."""
        qa_dir = self.graph_data_dir / f"qa_{qa_index}"
        
        if not qa_dir.exists():
            logger.error(f" QA {qa_index} 的图谱目录不存在: {qa_dir}")
            return None
        
        try:
            logger.debug(f" 加载 QA {qa_index} 的 SemanticGraph...")
            
            
            semantic_graph = SemanticGraph.load_graph(str(qa_dir))
            
            
            unit_count = len(semantic_graph.semantic_map.memory_units)
            logger.debug(f"    成功加载 {unit_count} 个 memory units")
            
            return semantic_graph
            
        except Exception as e:
            logger.error(f" 加载 QA {qa_index} 失败: {e}")
            logger.debug(traceback.format_exc())
            return None
    
    def _retrieve_context_l0(self,
                        semantic_graph: SemanticGraph,
                        question: str,
                        top_k: int) -> Tuple[List[Tuple[MemoryUnit, float]], Dict[str, Any]]:
        """Run retrieve context L0."""
        retrieval_start = time.time()
        
        try:
            multi_retriever = semantic_graph.get_multi_retriever()
            if multi_retriever is None:
                logger.error(" 无法初始化 MultiRetriever")
                return [], {"error": "MultiRetriever initialization failed"}
            
            
            logger.debug("   构建检索索引...")
            
            build_stats = multi_retriever.build_all_indexes(
                methods_to_build=[
                    RetrievalMethod.BM25,
                    RetrievalMethod.SPLADE,
                    RetrievalMethod.COSINE_SIMILARITY
                ],
                force_rebuild=False
            )
            
            logger.debug(f"   索引构建: 成功={build_stats['built_count']}, "
                        f"跳过={build_stats['skipped_count']}, "
                        f"失败={build_stats['failed_count']}")
            
            
            rerank_method = self.rerank_method if self.rerank_method != "none" else None
            
            logger.debug(f"   执行 smart_search 检索 (rerank={rerank_method or 'None'})...")
            results = multi_retriever.smart_search(
                query=question,
                methods=["bm25", "splade", "cosine_similarity"],
                fusion_method="rrf",
                rerank_method=rerank_method,  
                top_k=top_k,
                return_detailed=False
            )
            
            retrieval_time = time.time() - retrieval_start
            
            retrieval_details = {
                "method": "smart_search",
                "retrieval_methods": ["bm25", "splade", "cosine_similarity"],
                "fusion_method": "rrf",
                "rerank_method": self.rerank_method,  
                "rerank_description": self._get_rerank_description(),
                "top_k": top_k,
                "retrieved_count": len(results),
                "retrieval_time": retrieval_time,
                "index_build_stats": build_stats
            }
            
            logger.debug(f"    L0 检索完成: {len(results)} 个结果, 耗时 {retrieval_time:.3f}s")
            
            return results, retrieval_details
            
        except Exception as e:
            logger.error(f"    L0 检索失败: {e}")
            logger.debug(traceback.format_exc())
            
            retrieval_details = {
                "method": "smart_search",
                "rerank_method": self.rerank_method,
                "error": str(e),
                "retrieval_time": time.time() - retrieval_start
            }
            
            return [], retrieval_details
    
    
    def _generate_answer_from_context(self,
                             question: str,
                             retrieved_units: List[Tuple[MemoryUnit, float]],
                             question_type: str,
                             query_date: str = "Unknown Date") -> Tuple[str, str, Dict[str, int]]:
        """Generate answer from context."""
        if not retrieved_units:
            return "No relevant context found.", "I don't have enough information to answer this question.", {
                "prompt_tokens": 0,
                "context_tokens": 0,
                "total_input_tokens": 0
            }
        
        context_parts = []
        for i, (unit, score) in enumerate(retrieved_units, 1):
            content = unit.raw_data.get('text_content', '')
            role = unit.metadata.get('role', 'unknown')
            session_id = unit.metadata.get('session_id', 'unknown')
            session_date = unit.metadata.get('session_date', 'unknown')
            
            context_parts.append(
                f"[Memory {i}] (Timestamp: {session_date}, Session: {session_id}, Relevance: {score:.3f})\n"
                f"Speaker: {role}\n"
                f"Content: {content}\n"
            )
        
        context_text = "\n".join(context_parts)
        
        prompt = f"""You are a helpful expert assistant answering questions based on conversation history.

        # CURRENT REFERENCE TIME
        The current time for this question is: **{query_date}**
        *** CRITICAL INSTRUCTION ***
        - Treat "{query_date}" as "TODAY" or "NOW".
        - All relative time references ("yesterday", "last week", "3 days ago") MUST be calculated relative to this date.
        - Do NOT use the actual real-world date.

        # CONTEXT:
        You have access to conversation memories retrieved from a long-term memory system. Each memory has a timestamp indicating when it was recorded.

        {context_text}

        # QUESTION:
        {question}

        # CLARIFICATION:
        When interpreting memories, use the timestamp to determine when the described event happened, not when someone talked about the event.

        Example:
        Memory: (2023-03-15T16:33:00Z) I went to the vet yesterday.
        Question: What day did I go to the vet?
        Correct Answer: March 15, 2023
        Explanation: The timestamp shows the event was recorded on March 15th, so the vet visit happened on that date.

        # APPROACH (Think step by step):
        1. First, examine all memories that contain information related to the question
        2. Examine the timestamps and content of these memories carefully
        3. Look for explicit mentions of dates, times, locations, or events that answer the question
        4. If the answer requires calculation (e.g., converting time references), show your work
        5. Formulate a precise, concise answer based solely on the evidence in the memories
        6. Double-check that your answer directly addresses the question asked
        7. Ensure your final answer is specific and avoids vague references

        # OUTPUT FORMAT:
        Please respond in JSON format with two fields:
        {{
            "reasoning": "Your detailed step-by-step reasoning process, showing how you analyzed the memories and arrived at the answer",
            "final_answer": "Your direct, concise answer to the question"
        }}
        """
        
        token_stats = {
            "context_tokens": self._count_tokens(context_text),
            "prompt_tokens": self._count_tokens(prompt),
            "total_input_tokens": self._count_tokens(prompt)
        }
        
        try:
            response = self.llm_client.generate_answer(
                prompt=prompt,
                temperature=0.0,
                max_tokens=500,
                json_format=True
            )
            
            parsed = json.loads(response.strip())
            reasoning = parsed.get("reasoning", "No reasoning provided")
            answer = parsed.get("final_answer", "Unable to answer")
            
            return reasoning, answer, token_stats
            
        except json.JSONDecodeError as e:
            logger.warning(f"    JSON 解析失败: {e}, 尝试文本解析")
            try:
                reasoning_match = regex.search(r'"reasoning"\s*:\s*"([^"]+)"', response)
                answer_match = regex.search(r'"final_answer"\s*:\s*"([^"]+)"', response)
                
                if reasoning_match and answer_match:
                    return reasoning_match.group(1), answer_match.group(1), token_stats
                else:
                    return response[:200], response[-100:] if len(response) > 100 else response, token_stats
            except Exception as e2:
                logger.warning(f"    文本解析也失败: {e2}")
                return "Parse error", response[:100] if response else "No response", token_stats
                
        except Exception as e:
            logger.warning(f"    LLM 生成失败: {e}")
            return "Error in generation", "Unable to generate answer", token_stats
    
    def _test_single_qa(self, test_case: Dict[str, Any], test_index: int) -> Dict[str, Any]:
        """Run test single qa."""
        test_start_time = time.time()
        
        try:
            question_id = test_case["question_id"]
            question = test_case["question"]
            gold_answer = test_case["answer"]
            question_type = test_case.get("question_type", "unknown")
            qa_index = test_case.get("qa_index", 0)
            
            logger.info(f"\n{'='*80}")
            logger.info(f" 测试 {test_index}/{len(self.test_cases)}")
            logger.info(f"   问题ID: {question_id}")
            logger.info(f"   问题类型: {question_type}")
            logger.info(f"   问题: {question}")
            logger.info(f"   标准答案: {gold_answer}")
            logger.info(f"{'='*80}")
            
            
            logger.info(" 步骤1: 加载预构建的 SemanticGraph...")
            semantic_graph = self._load_semantic_graph_for_qa(qa_index)
            
            if semantic_graph is None:
                return {
                    "question_id": question_id,
                    "test_index": test_index,
                    "qa_index": qa_index,
                    "error": "Failed to load semantic graph",
                    "test_time": time.time() - test_start_time
                }
            
            logger.info(" 步骤2: 执行 L0 基线检索...")
            retrieved_units, retrieval_details = self._retrieve_context_l0(
                semantic_graph=semantic_graph,
                question=question,
                top_k=self.top_k
            )
            
            logger.info(f"    检索完成，返回 {len(retrieved_units)} 个记忆单元")
            logger.info(f"   Retrieval time: {retrieval_details.get('retrieval_time', 0):.3f}s")
            
            logger.info(" 步骤3: 生成答案...")
            reasoning, generated_answer, token_stats = self._generate_answer_from_context(
                question=question,
                retrieved_units=retrieved_units,
                question_type=question_type,
                query_date=test_case.get("question_date", test_case.get("date", test_case.get("session_date", "Unknown Date")))
            )
            
            logger.info(f"    推理过程: {reasoning[:100]}...")
            logger.info(f"    生成答案: {generated_answer}")
            logger.info(f"    Token统计: 输入={token_stats['total_input_tokens']}, 上下文={token_stats['context_tokens']}")
            
            context_parts = []
            for i, (unit, score) in enumerate(retrieved_units, 1):
                content = unit.raw_data.get('text_content', '')
                role = unit.metadata.get('role', 'unknown')
                session_date = unit.metadata.get('session_date', 'unknown')
                context_parts.append(
                    f"[{i}] ({session_date}) {role}: {content}"
                )
            context_text = "\n".join(context_parts[:10])
            
            if EVALUATION_AVAILABLE:
                logger.info(" 步骤4: 评估答案...")
                eval_result = calculate_comprehensive_scores(
                    gold_answer=gold_answer,
                    response=generated_answer,
                    question=question,
                    context=context_text,
                    question_type=question_type,
                    llm_client=self.llm_evaluate_client,
                    metrics=["exact_match", "f1", "rouge", "bleu", "meteor", 
                            "semantic_similarity", "bert_f1", "llm_judge"]
                )
                logger.info(f"    评估完成")
            else:
                eval_result = {"error": "Evaluation module not available"}
            
            test_time = time.time() - test_start_time
            
            result = {
                "question_id": question_id,
                "question": question,
                "gold_answer": gold_answer,
                "generated_answer": generated_answer,
                "reasoning": reasoning,
                "question_type": question_type,
                "test_index": test_index,
                "qa_index": qa_index,
                "memory_stats": {
                    "total_units": len(semantic_graph.semantic_map.memory_units),
                    "retrieved_units": len(retrieved_units),
                    "graph_loaded": True
                },
                "retrieval_details": retrieval_details,
                "token_stats": token_stats,
                "evaluation": eval_result,
                "test_time": test_time,
                "timestamp": datetime.now().isoformat()
            }
            
            
            self._generate_and_save_single_qa_report(result, test_index)
            
            llm_acc = eval_result.get("scores", {}).get("llm_accuracy", 0.0)
            f1_score = eval_result.get("scores", {}).get("token_f1", 0.0)
            rouge_l = eval_result.get("scores", {}).get("rougeL_f", 0.0)
            
            logger.info(f"\n{'='*80}")
            logger.info(f" 测试完成 [{test_index}/{len(self.test_cases)}]")
            logger.info(f"    LLM准确率: {llm_acc:.2%}")
            logger.info(f"    F1分数: {f1_score:.3f}")
            logger.info(f"    ROUGE-L: {rouge_l:.3f}")
            logger.info(f"   Total time: {test_time:.2f}s")
            logger.info(f"{'='*80}\n")
            
            return result
            
        except Exception as e:
            logger.error(f" 测试失败: {e}")
            logger.debug(traceback.format_exc())
            
            return {
                "question_id": test_case.get("question_id", "unknown"),
                "test_index": test_index,
                "qa_index": test_case.get("qa_index", 0),
                "question_type": test_case.get("question_type", "unknown"),
                "error": str(e),
                "traceback": traceback.format_exc(),
                "test_time": time.time() - test_start_time
            }
            
    def _generate_and_save_single_qa_report(self, result: Dict[str, Any], test_index: int):
        """Generate and save single qa report."""
        qa_index = result.get("qa_index", test_index)
        question_id = result.get("question_id", f"unknown_{test_index}")
        
        if not hasattr(self, '_current_test_timestamp'):
            self._current_test_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        test_run_dir = self.output_dir / f"benchmark_basic_{self.dataset_size}_{self._current_test_timestamp}"
        
        single_reports_dir = test_run_dir / "individual_reports"
        single_reports_dir.mkdir(parents=True, exist_ok=True)
        
        
        json_file = single_reports_dir / f"qa_{qa_index}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        txt_file = single_reports_dir / f"qa_{qa_index}.txt"
        report_lines = []
        
        report_lines.append("=" * 100)
        report_lines.append(f"LongMemEval L0 基线 - 单个问题测试报告")
        report_lines.append("=" * 100)
        report_lines.append(f"测试时间: {result.get('timestamp', 'N/A')}")
        report_lines.append(f"QA索引: qa_{qa_index}")
        report_lines.append(f"问题ID: {question_id}")
        report_lines.append(f"测试序号: {test_index}/{len(self.test_cases)}")
        report_lines.append("")
        
        report_lines.append("【问题信息】")
        report_lines.append("-" * 100)
        report_lines.append(f"问题类型: {result.get('question_type', 'N/A')}")
        report_lines.append(f"问题: {result.get('question', 'N/A')}")
        report_lines.append(f"标准答案: {result.get('gold_answer', 'N/A')}")
        report_lines.append("")
        
        report_lines.append("【检索结果】")
        report_lines.append("-" * 100)
        retrieval_details = result.get('retrieval_details', {})
        memory_stats = result.get('memory_stats', {})
        
        report_lines.append(f"记忆库总数: {memory_stats.get('total_units', 0)}")
        report_lines.append(f"检索返回: {memory_stats.get('retrieved_units', 0)} 个记忆单元")
        report_lines.append(f"检索耗时: {retrieval_details.get('retrieval_time', 0):.3f}s")
        report_lines.append(f"检索方法: {', '.join(retrieval_details.get('retrieval_methods', []))}")
        report_lines.append(f"融合方法: {retrieval_details.get('fusion_method', 'N/A')}")
        report_lines.append(f"重排序: {retrieval_details.get('rerank_method', 'N/A')}")
        report_lines.append("")
        
        if 'retrieval_scores' in retrieval_details:
            report_lines.append("检索得分详情:")
            for method, scores in retrieval_details.get('retrieval_scores', {}).items():
                report_lines.append(f"  - {method}: {scores}")
        report_lines.append("")
        
        report_lines.append("【生成答案】")
        report_lines.append("-" * 100)
        report_lines.append("推理过程:")
        reasoning = result.get('reasoning', 'N/A')
        for i in range(0, len(reasoning), 80):
            report_lines.append(f"  {reasoning[i:i+80]}")
        report_lines.append("")
        report_lines.append(f"最终答案: {result.get('generated_answer', 'N/A')}")
        report_lines.append("")
        
        report_lines.append("【评估结果】")
        report_lines.append("-" * 100)
        evaluation = result.get('evaluation', {})
        scores = evaluation.get('scores', {})
        
        if scores:
            report_lines.append("评分指标:")
            report_lines.append(f"  - 精确匹配 (Exact Match): {scores.get('exact_match', 0):.3f}")
            report_lines.append(f"  - Token F1: {scores.get('token_f1', 0):.3f}")
            report_lines.append(f"  - ROUGE-1: {scores.get('rouge1_f', 0):.3f}")
            report_lines.append(f"  - ROUGE-2: {scores.get('rouge2_f', 0):.3f}")
            report_lines.append(f"  - ROUGE-L: {scores.get('rougeL_f', 0):.3f}")
            report_lines.append(f"  - 语义相似度: {scores.get('semantic_similarity', 0):.3f}")
            report_lines.append(f"  - BERT F1: {scores.get('bert_f1', 0):.3f}")
            report_lines.append(f"  - LLM判断准确率: {scores.get('llm_accuracy', 0):.2%}")
            report_lines.append("")
            
            if 'avg_lexical' in scores:
                report_lines.append(f"  - 词汇层面平均分: {scores['avg_lexical']:.3f}")
            if 'avg_semantic' in scores:
                report_lines.append(f"  - 语义层面平均分: {scores['avg_semantic']:.3f}")
            if 'overall_average' in scores:
                report_lines.append(f"  - 总体平均分: {scores['overall_average']:.3f}")
        else:
            report_lines.append("评分信息不可用")
        
        report_lines.append("")
        
        llm_details = evaluation.get('llm_details', {})
        if llm_details:
            report_lines.append("LLM判断详情:")
            report_lines.append(f"  - 判断结果: {llm_details.get('judgments', [])}")
            report_lines.append(f"  - 一致性: {'高' if llm_details.get('consistency', False) else '低'}")
            report_lines.append(f"  - 置信度: {llm_details.get('confidence', 'N/A')}")
        report_lines.append("")
        
        report_lines.append("【Token统计】")
        report_lines.append("-" * 100)
        token_stats = result.get('token_stats', {})
        if token_stats:
            report_lines.append(f"上下文Token数: {token_stats.get('context_tokens', 0)}")
            report_lines.append(f"完整输入Token数: {token_stats.get('total_input_tokens', 0)}")
            report_lines.append(f"提示词Token数: {token_stats.get('prompt_tokens', 0)}")
            
            input_tokens = token_stats.get('total_input_tokens', 0)
            cost_per_1k = 0.03
            estimated_cost = (input_tokens / 1000) * cost_per_1k
            report_lines.append(f"估算成本: ${estimated_cost:.6f} (基于 ${cost_per_1k}/1K tokens)")
        else:
            report_lines.append("Token统计信息不可用")
        report_lines.append("")
        report_lines.append("【性能统计】")
        report_lines.append("-" * 100)
        report_lines.append(f"总测试时间: {result.get('test_time', 0):.3f}s")
        report_lines.append(f"  - 检索时间: {retrieval_details.get('retrieval_time', 0):.3f}s")
        generation_time = result.get('test_time', 0) - retrieval_details.get('retrieval_time', 0)
        report_lines.append(f"  - 生成+评估时间: {generation_time:.3f}s")
        report_lines.append("")
        
        report_lines.append("=" * 100)
        report_lines.append("报告结束")
        report_lines.append("=" * 100)
        
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        logger.debug(f"    单个问题报告已保存:")
        logger.debug(f"      JSON: {json_file.relative_to(self.output_dir)}")
        logger.debug(f"      TXT: {txt_file.relative_to(self.output_dir)}")
    
    def run_benchmark(self) -> Dict[str, Any]:
        """Run benchmark."""
        logger.info(f"\n 开始 LongMemEval L0 基线测试")
        logger.info(f"   数据集大小: {self.dataset_size}")
        logger.info(f"   测试数量: {len(self.test_cases)}")
        logger.info(f"   检索策略: BM25 + SPLADE + Cosine + RRF + BAAI重排序")
        logger.info(f"   Top-K: {self.top_k}")
        
        
        self._current_test_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        benchmark_start = time.time()
        results = []
        
        for i, test_case in enumerate(tqdm(self.test_cases, desc="测试进度"), start=1):
            result = self._test_single_qa(test_case, i)
            results.append(result)
        
        benchmark_time = time.time() - benchmark_start
        
        summary = self._generate_summary(results, benchmark_time)
        
        
        self._save_results(results, summary)
        
        return {
            "results": results,
            "summary": summary
        }
    
    def _generate_summary(self, results: List[Dict[str, Any]], total_time: float) -> Dict[str, Any]:
        """Generate summary."""
        successful_results = [r for r in results if "error" not in r]
        
        if not successful_results:
            return {
                "error": "所有测试都失败了",
                "total_tests": len(results),
                "failed_tests": len(results)
            }
        
        type_stats = defaultdict(lambda: {
            "count": 0,
            "llm_accuracies": [],
            "f1_scores": [],
            "retrieval_times": [],
            "token_counts": []
        })
        
        all_llm_accs = []
        all_f1s = []
        all_retrieval_times = []
        all_token_counts = []
        
        for result in successful_results:
            q_type = result.get("question_type", "unknown")
            
            type_stats[q_type]["count"] += 1
            
            llm_acc = result.get("evaluation", {}).get("scores", {}).get("llm_accuracy", 0.0)
            f1 = result.get("evaluation", {}).get("scores", {}).get("token_f1", 0.0)
            ret_time = result.get("retrieval_details", {}).get("retrieval_time", 0.0)
            total_tokens = result.get("token_stats", {}).get("total_input_tokens", 0)
            
            type_stats[q_type]["llm_accuracies"].append(llm_acc)
            type_stats[q_type]["f1_scores"].append(f1)
            type_stats[q_type]["retrieval_times"].append(ret_time)
            type_stats[q_type]["token_counts"].append(total_tokens)
            
            all_llm_accs.append(llm_acc)
            all_f1s.append(f1)
            all_retrieval_times.append(ret_time)
            all_token_counts.append(total_tokens)
        
        type_stats_final = {}
        for q_type, stats in type_stats.items():
            type_stats_final[q_type] = {
                "count": stats["count"],
                "avg_llm_accuracy": sum(stats["llm_accuracies"]) / len(stats["llm_accuracies"]) if stats["llm_accuracies"] else 0.0,
                "avg_f1": sum(stats["f1_scores"]) / len(stats["f1_scores"]) if stats["f1_scores"] else 0.0,
                "avg_retrieval_time": sum(stats["retrieval_times"]) / len(stats["retrieval_times"]) if stats["retrieval_times"] else 0.0,
                "avg_tokens": sum(stats["token_counts"]) / len(stats["token_counts"]) if stats["token_counts"] else 0.0
            }
        
        summary = {
            "baseline_version": "L0",
            "dataset_size": self.dataset_size,
            "total_tests": len(results),
            "successful_tests": len(successful_results),
            "failed_tests": len(results) - len(successful_results),
            "overall_llm_accuracy": sum(all_llm_accs) / len(all_llm_accs) if all_llm_accs else 0.0,
            "overall_f1": sum(all_f1s) / len(all_f1s) if all_f1s else 0.0,
            "avg_retrieval_time": sum(all_retrieval_times) / len(all_retrieval_times) if all_retrieval_times else 0.0,
            "avg_tokens": sum(all_token_counts) / len(all_token_counts) if all_token_counts else 0.0,
            "total_benchmark_time": total_time,
            "by_question_type": type_stats_final,
            "retrieval_config": {
                "method": "smart_search",
                "retrieval_methods": ["bm25", "splade", "cosine_similarity"],
                "fusion_method": "rrf",
                "rerank_method": "baai",
                "top_k": self.top_k,
                "data_source": "prebuilt_graph"
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return summary
    
    def _save_results(self, results: List[Dict[str, Any]], summary: Dict[str, Any]):
        """Save results."""
        timestamp = self._current_test_timestamp
        
        test_run_dir = self.output_dir / f"benchmark_basic_{self.dataset_size}_{timestamp}"
        test_run_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f" 测试结果目录: {test_run_dir}")
        
        
        results_file = test_run_dir / f"benchmark_basic_{self.dataset_size}_{timestamp}.json"
        
        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "summary": summary,
                    "detailed_results": results
                }, f, indent=2, ensure_ascii=False)
            logger.info(f" 详细结果已保存: {results_file}")
        except Exception as e:
            logger.error(f" 保存详细结果失败: {e}")
            
            backup_file = test_run_dir / f"benchmark_basic_{self.dataset_size}_{timestamp}_backup.json"
            try:
                with open(backup_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "summary": summary,
                        "detailed_results": results,
                        "error": str(e)
                    }, f, indent=2, ensure_ascii=False)
                logger.info(f" 备份结果已保存: {backup_file}")
            except Exception as e2:
                logger.error(f" 备份保存也失败: {e2}")
        
        report_file = test_run_dir / f"benchmark_basic_{self.dataset_size}_{timestamp}_report.txt"
        try:
            self._generate_readable_report(summary, report_file, test_run_dir)
            logger.info(f" 可读报告已保存: {report_file}")
        except Exception as e:
            logger.error(f" 生成可读报告失败: {e}")
    
    def _generate_readable_report(self, summary: Dict[str, Any], output_file: Path, test_run_dir: Path):
        """Generate readable report."""
        lines = []
        lines.append("=" * 100)
        lines.append("LongMemEval L0 基线测试报告（优化版）")
        lines.append("=" * 100)
        lines.append(f"测试时间: {summary['timestamp']}")
        lines.append(f"数据集大小: {summary['dataset_size']}")
        lines.append(f"总测试数: {summary['total_tests']}")
        lines.append(f"成功: {summary['successful_tests']}, 失败: {summary['failed_tests']}")
        lines.append(f"总耗时: {summary['total_benchmark_time']:.2f}秒")
        lines.append("")
        
        ret_config = summary["retrieval_config"]
        lines.append("=" * 100)
        lines.append("检索配置")
        lines.append("=" * 100)
        lines.append(f"检索方法: {ret_config['method']}")
        lines.append(f"子方法: {', '.join(ret_config['retrieval_methods'])}")
        lines.append(f"融合方法: {ret_config['fusion_method']}")
        lines.append(f"重排序: {ret_config['rerank_method']}")
        lines.append(f"Top-K: {ret_config['top_k']}")
        lines.append(f"数据来源: {ret_config['data_source']}")
        lines.append("")
        
        lines.append("=" * 100)
        lines.append("整体性能指标")
        lines.append("=" * 100)
        lines.append(f"LLM准确率 (平均): {summary['overall_llm_accuracy']:.2%}")
        lines.append(f"Token F1 (平均): {summary['overall_f1']:.3f}")
        lines.append(f"检索时间 (平均): {summary['avg_retrieval_time']:.3f}秒")
        lines.append(f"输入Token数 (平均): {summary['avg_tokens']:.0f} tokens")
        lines.append("")
        
        lines.append("=" * 100)
        lines.append("按问题类型统计")
        lines.append("=" * 100)
        
        by_type = summary.get("by_question_type", {})
        if by_type:
            lines.append(f"{'问题类型':<20} {'数量':>8} {'LLM准确率':>12} {'F1分数':>10} {'检索时间(s)':>12} {'平均Token':>12}")
            lines.append("-" * 100)
            
            for q_type, stats in sorted(by_type.items()):
                lines.append(
                    f"{q_type:<20} "
                    f"{stats['count']:>8} "
                    f"{stats['avg_llm_accuracy']:>11.2%} "
                    f"{stats['avg_f1']:>10.3f} "
                    f"{stats['avg_retrieval_time']:>12.3f} "
                    f"{stats['avg_tokens']:>12.0f}"
                )
        else:
            lines.append("无问题类型统计信息")
        
        lines.append("")
        
        lines.append("=" * 100)
        lines.append("详细结果")
        lines.append("=" * 100)
        lines.append(f"详细JSON结果: {test_run_dir.name}/benchmark_basic_{summary['dataset_size']}_*.json")
        lines.append(f"单个问题报告: {test_run_dir.name}/individual_reports/qa_*.txt")
        lines.append("")
        
        lines.append("=" * 100)
        lines.append("报告结束")
        lines.append("=" * 100)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        logger.info(f" 可读报告已保存: {output_file}")


def main():
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(description="LongMemEval L0 基线测试（优化版）")
    
    # Dataset-specific handling used by the reproduction workflow.
    parser.add_argument("--dataset-size", default="s", choices=["s", "m"],
                       help="数据集大小: s (small) 或 m (medium)")
    parser.add_argument("--dataset-dir", 
                       default=None,
                       help="原始数据集目录路径")
    parser.add_argument("--graph-data-dir",
                       default=str(paths.LONGMEMEVAL_HIERARCHICAL_STEP3_DIR),
                       help="预构建的图谱数据目录（默认：benchmark_longmemeval/dataset/LongMemEval/longmemeval_hierarchical/step3_semantic_graphs）")
    parser.add_argument("--max-tests", type=int, default=None,
                       help="最大测试数量（用于快速测试）")
    parser.add_argument("--output-dir", 
                       default=str(paths.LONGMEMEVAL_TASK_EVAL_RESULTS_DIR),
                       help="结果输出目录")
    
    parser.add_argument("--llm-model", default="gpt-4o-mini-closeai",
                       help="答案生成LLM模型")
    parser.add_argument("--llm-evaluate-model", default="gpt-4o-mini-closeai",
                       help="答案评估LLM模型")
    
    parser.add_argument("--top-k", type=int, default=10,
                       help="检索返回的记忆单元数量")
    
    
    parser.add_argument("--rerank-method", default="baai",
                       choices=["baai", "qwen", "jina", "qwen-sili", "qwen-dashscope", "gte-dashscope", "none"],
                       help="重排序方法: baai (默认), qwen, jina, qwen-sili, qwen-dashscope, gte-dashscope, none (不使用)")
    
    args = parser.parse_args()
    
    try:
        llm_client = LLMClient(args.llm_model)
        llm_evaluate_client = LLMClient(args.llm_evaluate_model)
        
        baseline = LongMemEvalL0Baseline(
            dataset_size=args.dataset_size,
            dataset_dir=args.dataset_dir,
            graph_data_dir=args.graph_data_dir,
            llm_client=llm_client,
            llm_evaluate_client=llm_evaluate_client,
            output_dir=args.output_dir,
            top_k=args.top_k,
            max_tests=args.max_tests,
            rerank_method=args.rerank_method  
        )
        
        results = baseline.run_benchmark()
        
        summary = results["summary"]
        print("\n" + "="*100)
        print(" L0 基线测试完成!")
        print("="*100)
        print(f"数据集大小: {summary['dataset_size']}")
        print(f"总测试数: {summary['total_tests']}")
        print(f"成功: {summary['successful_tests']}, 失败: {summary['failed_tests']}")
        print(f"LLM准确率: {summary['overall_llm_accuracy']:.2%}")
        print(f"平均F1: {summary['overall_f1']:.3f}")
        print(f"平均检索时间: {summary['avg_retrieval_time']:.3f}秒")
        print(f"重排序方法: {args.rerank_method}")  
        print(f"总耗时: {summary['total_benchmark_time']:.2f}秒")
        print("="*100)
        
        return 0
        
    except Exception as e:
        logger.error(f" L0 基线测试失败: {e}")
        logger.debug(traceback.format_exc())
        return 1
    
    finally:
        try:
            if EVALUATION_AVAILABLE:
                cleanup_evaluation_models()
        except Exception as e:
            logger.warning(f"清理评估模型时出错: {e}")


if __name__ == "__main__":
    exit(main())
