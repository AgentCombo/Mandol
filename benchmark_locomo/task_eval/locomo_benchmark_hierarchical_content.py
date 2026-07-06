#!/usr/bin/env python3
"""Memory Spaces: - L2_CharacterSnapshot / L2_CrossInsight / L2_Timeline / L2_GlobalStats."""

import os
import sys
import json
import logging
import argparse
import traceback
import tempfile
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, field


# Avoid mutating LogRecord fields before other handlers process the record.
from mandol.utils.logging_config import setup_logging, create_module_logger, auto_configure_logging
if auto_configure_logging() is None:
    setup_logging(level=logging.INFO)
logger = create_module_logger("benchmark_hierarchical_content")

from mandol import SemanticGraph, MemoryUnit
from mandol.retrieval.advance_retriever import MultiRetriever
from mandol.retrieval.retrieval_interface import RetrievalMethod
from mandol.retrieval.rerank_manager import RerankerManager
from mandol.llm.llm_client import LLMClient
from benchmark_locomo.task_eval.evaluation import (
    calculate_f1_score, exact_match_score, llm_grader,
    calculate_rouge_score, calculate_bleu_score,
    calculate_meteor_score, calculate_semantic_similarity,
    calculate_bert_f1_score, convert_numpy_types,
    calculate_comprehensive_scores
)
from mandol.core import paths




class MemorySpaceNames:
    L0_OBSERVATION = "L0_Observation"
    L1_SESSION_FACTS = "L1_SessionFacts"
    L2_CHARACTER_SNAPSHOT = "L2_CharacterSnapshot"
    L2_CROSS_INSIGHT = "L2_CrossInsight"
    L2_TIMELINE = "L2_Timeline"
    L2_GLOBAL_STATS = "L2_GlobalStats"

    @classmethod
    def all_spaces(cls) -> List[str]:
        return [
            cls.L0_OBSERVATION,
            cls.L1_SESSION_FACTS,
            cls.L2_CHARACTER_SNAPSHOT,
            cls.L2_CROSS_INSIGHT,
            cls.L2_TIMELINE,
            cls.L2_GLOBAL_STATS
        ]

    @classmethod
    def l2_spaces(cls) -> List[str]:
        return [
            cls.L2_CHARACTER_SNAPSHOT,
            cls.L2_CROSS_INSIGHT,
            cls.L2_TIMELINE,
            cls.L2_GLOBAL_STATS
        ]




def _safe_json_serialization(obj):
    """Run safe JSON serialization."""
    if isinstance(obj, dict):
        return {k: _safe_json_serialization(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_safe_json_serialization(v) for v in obj]
    elif isinstance(obj, tuple):
        return [_safe_json_serialization(v) for v in obj]
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif hasattr(obj, '__dict__'):
        try:
            return str(obj)
        except:
            return f"<{type(obj).__name__} object>"
    else:
        try:
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            return str(obj)




@dataclass
class BenchmarkConfig:
    top_k: int = 20
    retrieval_methods: List[str] = field(default_factory=lambda: ["bm25", "cosine_similarity", "splade"])
    fusion_method: str = "rrf"
    rerank_method: Optional[str] = "baai"
    rerank_candidates_multiplier: int = 3

    use_all_spaces: bool = True
    specific_spaces: Optional[List[str]] = None

    include_llm_evaluation: bool = True
    include_semantic_metrics: bool = True
    include_lexical_metrics: bool = True

    graphs_dir: str = str(paths.LOCOMO_HIERARCHICAL_CONTENT_STEP4_DIR)
    locomo_file: str = str(paths.LOCOMO_RAW_FILE)
    output_dir: str = str(paths.LOCOMO_RESULTS_DIR / "hierarchical_content_benchmark")




class UnifiedHierarchicalRetriever:

    def __init__(self,
                 semantic_graph: SemanticGraph,
                 config: BenchmarkConfig,
                 reranker_manager: Optional[RerankerManager] = None):
        self.semantic_graph = semantic_graph
        self.config = config
        self.logger = create_module_logger("benchmark_hierarchical_content.UnifiedRetriever")

        self.multi_retriever = semantic_graph.get_multi_retriever()
        
        if reranker_manager:
            self.multi_retriever.reranker_manager = reranker_manager

        self._build_indexes()

    def _build_indexes(self):
        """Build indexes."""
        methods_to_build = []
        for method_str in self.config.retrieval_methods:
            try:
                methods_to_build.append(RetrievalMethod(method_str.lower()))
            except ValueError:
                self.logger.warning(f"未知的检索方法: {method_str}")

        if methods_to_build:
            self.multi_retriever.build_all_indexes(methods_to_build=methods_to_build, force_rebuild=False)

    def search(self, 
               query: str, 
               top_k: Optional[int] = None, 
               return_detailed: bool = False) -> Dict[str, Any]:
        """Search."""
        return self._execute_search(
            query=query, 
            top_k=top_k, 
            space_names=None, 
            return_detailed=return_detailed
        )

    def search_in_spaces(self, 
                         query: str, 
                         space_names: List[str], 
                         top_k: Optional[int] = None, 
                         return_detailed: bool = False) -> Dict[str, Any]:
        """Retrieve in spaces."""
        if not space_names:
            self.logger.warning("调用了 search_in_spaces 但 space_names 为空，自动回退到全库检索")
            return self.search(query, top_k, return_detailed)

        return self._execute_search(
            query=query, 
            top_k=top_k, 
            space_names=space_names, 
            return_detailed=return_detailed
        )

    def _execute_search(self, 
                        query: str, 
                        top_k: Optional[int], 
                        space_names: Optional[List[str]], 
                        return_detailed: bool) -> Dict[str, Any]:
        """Execute search."""
        top_k = top_k or self.config.top_k

        methods = []
        for method_str in self.config.retrieval_methods:
            try:
                methods.append(RetrievalMethod(method_str.lower()))
            except ValueError:
                pass

        try:
            if len(methods) > 1 and not self.multi_retriever.parallel_config.enable_parallel:
                self.multi_retriever.parallel_config.enable_parallel = True
                self.multi_retriever.parallel_config.max_workers = min(len(methods), 4)

            results = self.multi_retriever.smart_search(
                query=query,
                methods=methods,
                top_k=top_k,
                fusion_method=self.config.fusion_method,
                rerank_method=self.config.rerank_method,
                space_names=space_names,
                return_detailed=return_detailed
            )

            if return_detailed:
                raw_results = results.get("results", [])
            else:
                raw_results = results

            by_layer = self._group_results_by_layer(raw_results)

            return {
                "results": raw_results,
                "by_layer": by_layer,
                "retrieval_stats": {
                    "total_results": len(raw_results),
                    "mode": "Global (Fast)" if space_names is None else "Filtered (Sliced)",
                    "target_spaces": "ALL" if space_names is None else space_names,
                    "methods_used": [m.value for m in methods],
                    "l0_count": len(by_layer.get("L0", [])),
                    "l1_count": len(by_layer.get("L1", [])),
                    "l2_count": len(by_layer.get("L2", []))
                }
            }

        except Exception as e:
            self.logger.error(f"检索失败: {e}")
            traceback.print_exc()
            return {
                "results": [],
                "by_layer": {"L0": [], "L1": [], "L2": []},
                "retrieval_stats": {"error": str(e)}
            }

    def _group_results_by_layer(self, results: List[Tuple[MemoryUnit, float]]) -> Dict[str, List[Dict]]:
        """Run group results by layer."""
        by_layer = {"L0": [], "L1": [], "L2": []}

        for unit, score in results:
            memory_level = None
            
            if unit.metadata:
                memory_level = unit.metadata.get("memory_level")

            
            if not memory_level and unit.uid:
                uid_lower = unit.uid.lower()
                if "_l0_" in uid_lower or unit.uid.startswith("conv-"):
                    memory_level = "L0"
                elif "_l1_" in uid_lower:
                    memory_level = "L1"
                elif "_l2_" in uid_lower:
                    memory_level = "L2"

            session_date = None
            if unit.raw_data:
                session_date = unit.raw_data.get("session_date") or unit.raw_data.get("date")
            if not session_date and unit.metadata:
                session_date = unit.metadata.get("session_date") or unit.metadata.get("date")
            
            result_item = {
                "uid": unit.uid,
                "score": score,
                "content": self._extract_content(unit),
                "content_type": unit.metadata.get("content_type") if unit.metadata else None,
                "memory_level": memory_level,
                "session_date": session_date
            }

            if memory_level == "L0":
                by_layer["L0"].append(result_item)
            elif memory_level == "L1":
                by_layer["L1"].append(result_item)
            elif memory_level == "L2":
                by_layer["L2"].append(result_item)
            else:
                by_layer["L0"].append(result_item)

        return by_layer

    def _extract_content(self, unit: MemoryUnit) -> str:
        """Extract content."""
        if unit.raw_data:
            if "text_content" in unit.raw_data: return unit.raw_data["text_content"]
            if "content" in unit.raw_data: return unit.raw_data["content"]
            if "message" in unit.raw_data: return unit.raw_data["message"]
        return str(unit.raw_data) if unit.raw_data else ""
    
# class UnifiedHierarchicalRetriever:
#     """


#     """

#     def __init__(self,
#                  semantic_graph: SemanticGraph,
#                  config: BenchmarkConfig,
#                  reranker_manager: Optional[RerankerManager] = None):
#         self.semantic_graph = semantic_graph
#         self.config = config
#         self.logger = create_module_logger("benchmark_hierarchical_content.UnifiedRetriever")


#         self.multi_retriever = semantic_graph.get_multi_retriever()
        

#         if reranker_manager:
#             self.multi_retriever.reranker_manager = reranker_manager


#         self._build_indexes()

#     def _build_indexes(self):

#         methods_to_build = []
#         for method_str in self.config.retrieval_methods:
#             try:
#                 method = RetrievalMethod(method_str.lower())
#                 methods_to_build.append(method)
#             except ValueError:

#         if methods_to_build:

#             self.multi_retriever.build_all_indexes(
#                 methods_to_build=methods_to_build,
#                 force_rebuild=False
#             )

#     def search(self,
#                query: str,
#                top_k: Optional[int] = None,
#                space_names: Optional[List[str]] = None,
#                return_detailed: bool = False) -> Dict[str, Any]:
#         """

#         Returns:
#             {
#                 "results": [(MemoryUnit, score), ...],
#                 "by_layer": {"L0": [...], "L1": [...], "L2": [...]},
#                 "retrieval_stats": {...}
#             }
#         """
#         top_k = top_k or self.config.top_k

#         if space_names is None:
#             if self.config.use_all_spaces:
#                 space_names = MemorySpaceNames.all_spaces()
#             elif self.config.specific_spaces:
#                 space_names = self.config.specific_spaces

#         methods = []
#         for method_str in self.config.retrieval_methods:
#             try:
#                 methods.append(RetrievalMethod(method_str.lower()))
#             except ValueError:
#                 pass

#         try:
#             results = self.multi_retriever.smart_search(
#                 query=query,
#                 methods=methods,
#                 top_k=top_k,
#                 fusion_method=self.config.fusion_method,
#                 rerank_method=self.config.rerank_method,
#                 space_names=space_names,
#                 return_detailed=return_detailed
#             )

#             if return_detailed:
#                 raw_results = results.get("results", [])
#             else:
#                 raw_results = results

#             by_layer = self._group_results_by_layer(raw_results)

#             return {
#                 "results": raw_results,
#                 "by_layer": by_layer,
#                 "retrieval_stats": {
#                     "total_results": len(raw_results),
#                     "methods_used": [m.value for m in methods],
#                     "fusion_method": self.config.fusion_method,
#                     "rerank_method": self.config.rerank_method,
#                     "space_names": space_names,
#                     "l0_count": len(by_layer.get("L0", [])),
#                     "l1_count": len(by_layer.get("L1", [])),
#                     "l2_count": len(by_layer.get("L2", []))
#                 }
#             }

#         except Exception as e:
#             traceback.print_exc()
#             return {
#                 "results": [],
#                 "by_layer": {"L0": [], "L1": [], "L2": []},
#                 "retrieval_stats": {"error": str(e)}
#             }

#     def _group_results_by_layer(self,
#                                 results: List[Tuple[MemoryUnit, float]]) -> Dict[str, List[Dict]]:
#         by_layer = {"L0": [], "L1": [], "L2": []}

#         for unit, score in results:
#             memory_level = None
#             if unit.metadata:
#                 memory_level = unit.metadata.get("memory_level")


#             if not memory_level and unit.uid:
#                 if "_l0_" in unit.uid.lower() or unit.uid.startswith("conv-"):
#                     memory_level = "L0"
#                 elif "_l1_" in unit.uid.lower():
#                     memory_level = "L1"
#                 elif "_l2_" in unit.uid.lower():
#                     memory_level = "L2"

#             session_date = None
#             if unit.raw_data:
#                 session_date = unit.raw_data.get("session_date") or unit.raw_data.get("date")
#             if not session_date and unit.metadata:
#                 session_date = unit.metadata.get("session_date") or unit.metadata.get("date")
            
#             result_item = {
#                 "uid": unit.uid,
#                 "score": score,
#                 "content": self._extract_content(unit),
#                 "content_type": unit.metadata.get("content_type") if unit.metadata else None,
#                 "memory_level": memory_level,
#                 "session_date": session_date
#             }

#             if memory_level == "L0":
#                 by_layer["L0"].append(result_item)
#             elif memory_level == "L1":
#                 by_layer["L1"].append(result_item)
#             elif memory_level == "L2":
#                 by_layer["L2"].append(result_item)
#             else:
#                 by_layer["L0"].append(result_item)

#         return by_layer

#     def _extract_content(self, unit: MemoryUnit) -> str:
#         if unit.raw_data:
#             if "text_content" in unit.raw_data:
#                 return unit.raw_data["text_content"]
#             if "content" in unit.raw_data:
#                 return unit.raw_data["content"]
#             if "message" in unit.raw_data:
#                 return unit.raw_data["message"]

#         return str(unit.raw_data) if unit.raw_data else ""




class HierarchicalContextBuilder:

    def __init__(self):
        self.logger = create_module_logger("benchmark_hierarchical_content.ContextBuilder")

    def build_context(self,
                      retrieval_result: Dict[str, Any],
                      query: str,
                      category: int) -> str:
        """Build context."""
        parts = []

        by_layer = retrieval_result.get("by_layer", {})

        l2_results = by_layer.get("L2", [])
        if l2_results:
            parts.append("=== Global Insights (L2) ===")
            for i, item in enumerate(l2_results[:3], 1):
                content_type = item.get("content_type", "insight")
                content = item.get("content", "")
                parts.append(f"[{content_type}] {content}")
            parts.append("")

        l1_results = by_layer.get("L1", [])
        if l1_results:
            parts.append("=== Session Summaries (L1) ===")
            for i, item in enumerate(l1_results[:5], 1):
                content = item.get("content", "")
                parts.append(f"[Session {i}] {content}")
            parts.append("")

        l0_results = by_layer.get("L0", [])
        if l0_results:
            parts.append("=== Conversations (L0) ===")
            for i, item in enumerate(l0_results[:10], 1):
                content = item.get("content", "")
                parts.append(f"{i}. {content}")
            parts.append("")

        guidance = self._get_category_guidance(category)
        if guidance:
            parts.append(f"=== Analysis Guidance ===\n{guidance}")

        return "\n".join(parts)

    def _get_category_guidance(self, category: int) -> str:
        """Get category gUIDance."""
        guidance_map = {
            1: "For multi-hop questions: Combine evidence from multiple conversations (L0) and use L1/L2 summaries to connect related facts.",
            2: "For temporal questions: Focus on dates and time references. L1 session dates and L2 timeline provide temporal context.",
            3: "For open-domain questions: Synthesize information from all layers. L2 insights give overall patterns.",
            4: "For single-hop questions: Look for direct statements in L0 conversations first.",
            5: "For adversarial questions: Verify if information exists. If not found in any layer, state 'No information available'."
        }
        return guidance_map.get(category, "Provide a clear, factual answer based on the retrieved context.")




class HierarchicalContentBenchmarkTester:

    def __init__(self,
                 config: BenchmarkConfig,
                 llm_client: Optional[LLMClient] = None,
                 llm_evaluate_client: Optional[LLMClient] = None,
                 reranker_manager: Optional[RerankerManager] = None):
        self.config = config
        self.llm_client = llm_client
        self.llm_evaluate_client = llm_evaluate_client
        self.reranker_manager = reranker_manager
        self.logger = create_module_logger("benchmark_hierarchical_content.BenchmarkTester")

        self.output_dir = Path(config.output_dir)
        self._setup_output_dir()

        self.context_builder = HierarchicalContextBuilder()

        self.logger.info(f" Benchmark 测试器初始化完成")
        self.logger.info(f"   检索方法: {config.retrieval_methods}")
        self.logger.info(f"   融合方法: {config.fusion_method}")
        self.logger.info(f"   重排序: {config.rerank_method}")
        self.logger.info(f"   输出目录: {self.output_dir}")

    def _setup_output_dir(self):
        """Run setup output dir."""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            test_file = self.output_dir / ".write_test"
            test_file.write_text("test")
            test_file.unlink()
        except Exception as e:
            self.logger.warning(f"创建输出目录失败: {e}，使用临时目录")
            self.output_dir = Path(tempfile.gettempdir()) / "hierarchical_content_benchmark"
            self.output_dir.mkdir(exist_ok=True)

    def load_semantic_graph(self, sample_id: str) -> Tuple[SemanticGraph, UnifiedHierarchicalRetriever]:
        """Load semantic graph."""
        sample_dir = Path(self.config.graphs_dir) / sample_id

        if not sample_dir.exists():
            raise FileNotFoundError(f"样本目录不存在: {sample_dir}")

        self.logger.info(f" 加载语义图谱: {sample_id}")

        
        graph = SemanticGraph.load_graph(str(sample_dir))

        total_units = len(graph.semantic_map.memory_units) if hasattr(graph.semantic_map, 'memory_units') else 0
        self.logger.info(f"   加载了 {total_units} 个记忆单元")

        retriever = UnifiedHierarchicalRetriever(
            semantic_graph=graph,
            config=self.config,
            reranker_manager=self.reranker_manager
        )

        return graph, retriever

    def load_qa_data(self, sample_id: str) -> List[Dict]:
        """Load qa data."""
        self.logger.info(f" 加载 QA 数据: {sample_id}")

        with open(self.config.locomo_file, 'r', encoding='utf-8') as f:
            locomo_data = json.load(f)

        sample_data = None
        for item in locomo_data:
            if item.get("sample_id") == sample_id:
                sample_data = item
                break

        if not sample_data:
            raise ValueError(f"未找到样本 {sample_id} 的 QA 数据")

        qa_list = sample_data.get("qa", [])
        formatted_qa = []

        for i, qa in enumerate(qa_list):
            if not isinstance(qa, dict) or "question" not in qa:
                continue

            question = qa["question"]
            category = qa.get("category", 1)

            if "answer" in qa and qa["answer"]:
                expected_answer = qa["answer"]
                question_suffix = ""
            elif "adversarial_answer" in qa:
                expected_answer = qa["adversarial_answer"]
                question_suffix = "_adv"
                category = 5
            else:
                continue

            formatted_qa.append({
                "question_id": f"{sample_id}_q{i+1}{question_suffix}",
                "question": question,
                "answer": expected_answer,
                "category": category,
                "evidence": qa.get("evidence", [])
            })

        category_counts = defaultdict(int)
        for qa in formatted_qa:
            category_counts[qa["category"]] += 1

        self.logger.info(f"   加载了 {len(formatted_qa)} 个问题")
        self.logger.info(f"   类别分布: {dict(category_counts)}")

        return formatted_qa

    def evaluate_single_sample(self,
                               sample_id: str,
                               jsonl_file: Optional[Path] = None) -> Dict[str, Any]:
        """Evaluate single sample."""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f" 评估样本: {sample_id}")
        self.logger.info(f"{'='*60}")

        sample_start = datetime.now()

        try:
            
            graph, retriever = self.load_semantic_graph(sample_id)

            
            qa_data = self.load_qa_data(sample_id)

            qa_results = []
            for qa in tqdm(qa_data, desc=f"评估 {sample_id}"):
                qa_result = self._evaluate_single_qa(
                    qa_data=qa,
                    retriever=retriever,
                    sample_id=sample_id
                )
                qa_results.append(qa_result)

                
                if jsonl_file:
                    self._save_result_to_jsonl(qa_result, jsonl_file)

            sample_summary = self._calculate_sample_summary(qa_results)

            sample_duration = (datetime.now() - sample_start).total_seconds()

            return {
                "sample_id": sample_id,
                "evaluation_success": True,
                "qa_results": qa_results,
                "summary": sample_summary,
                "duration": sample_duration
            }

        except Exception as e:
            self.logger.error(f" 评估样本 {sample_id} 失败: {e}")
            traceback.print_exc()
            return {
                "sample_id": sample_id,
                "evaluation_success": False,
                "error": str(e)
            }

    def _evaluate_single_qa(self,
                            qa_data: Dict,
                            retriever: UnifiedHierarchicalRetriever,
                            sample_id: str) -> Dict[str, Any]:
        """Run evaluate single qa."""
        question_id = qa_data["question_id"]
        question = qa_data["question"]
        expected_answer = qa_data["answer"]
        category = qa_data["category"]

        try:
            retrieval_result = retriever.search(
                query=question,
                top_k=self.config.top_k
            )

            context = self.context_builder.build_context(
                retrieval_result=retrieval_result,
                query=question,
                category=category
            )

            generated_answer = ""
            reasoning = ""
            if self.llm_client:
                answer_result = self._generate_answer(
                    question=question,
                    context=context,
                    category=category
                )
                generated_answer = answer_result.get("answer", "")
                reasoning = answer_result.get("reasoning", "")

            scores = self._calculate_scores(
                question=question,
                prediction=generated_answer,
                reference=expected_answer,
                category=category,
                context=context,
                reasoning=reasoning
            )

            return {
                "question_id": question_id,
                "question": question,
                "category": category,
                "expected_answer": expected_answer,
                "generated_answer": generated_answer,
                "reasoning": reasoning,
                "evaluation_success": True,
                "scores": scores,
                "category_score": scores.get("combined_score", 0.0),
                "retrieval_stats": retrieval_result.get("retrieval_stats", {}),
                "context_length": len(context)
            }

        except Exception as e:
            self.logger.warning(f"评估问题 {question_id} 失败: {e}")
            return {
                "question_id": question_id,
                "question": question,
                "category": category,
                "expected_answer": expected_answer,
                "generated_answer": "",
                "evaluation_success": False,
                "error": str(e),
                "category_score": 0.0
            }

    def _generate_answer(self,
                         question: str,
                         context: str,
                         category: int) -> Dict[str, str]:
        """Generate answer."""
        
        category_guidance = self._get_answer_category_guidance(category)
        
        prompt = f"""You are an expert memory retrieval assistant analyzing a hierarchical memory structure.

        CONTEXT (Hierarchical Memory):
        {context}

        QUESTION: {question}

        CATEGORY GUIDANCE:
        {category_guidance}

        INSTRUCTIONS:
        1. Carefully analyze the provided hierarchical context (L2 insights → L1 summaries → L0 conversations)
        2. Use L2 for high-level patterns and relationships
        3. Use L1 for session-level summaries and key events
        4. Use L0 for specific dialogue details and exact quotes
        5. Cross-reference information across layers for accuracy
        6. If information is not available, clearly state so

        RESPONSE FORMAT (JSON):
        {{
            "reasoning": "Your step-by-step reasoning process, explaining how you used the different layers of context to arrive at your answer. Include specific references to L2 insights, L1 summaries, and L0 conversations as relevant.",
            "final_answer": "Your direct, concise final answer to the question"
        }}

        Respond ONLY with valid JSON:"""

        try:
            logger.info(f"生成答案提示词:\n{prompt}")
            response = self.llm_client.generate_answer(
                prompt=prompt,
                max_tokens=800,
                temperature=0.1,
                json_format=True
            )

            answer, reasoning = self._parse_structured_response(response, category)
            return {"answer": answer, "reasoning": reasoning}

        except Exception as e:
            self.logger.warning(f"生成答案失败: {e}")
            return {"answer": "", "reasoning": ""}

    def _get_answer_category_guidance(self, category: int) -> str:
        """Get answer category gUIDance."""
        guidance_map = {
            1: """Multi-hop Question: This requires combining information from multiple sources.
            - Look for connections between different conversations (L0)
            - Use L1 session summaries to identify related events
            - L2 insights may provide relationship patterns""",
            2: """Temporal Question: This involves time-related information.
            - Pay attention to dates, times, and temporal references in L0
            - L1 session dates provide temporal context
            - L2 timeline insights may help establish chronology""",
            3: """Open-domain Question: This requires comprehensive analysis.
            - Synthesize information from all three layers
            - L2 provides overall patterns and themes
            - Be thorough but concise in your answer""",
            4: """Single-hop Question: This has a direct answer.
            - Look for explicit statements in L0 conversations first
            - L1 summaries may contain the key fact
            - Answer should be straightforward and specific""",
            5: """Adversarial Question: This may be unanswerable.
            - Carefully verify if the information actually exists
            - Check all layers before concluding information is missing
            - If not found, clearly state: "The information is not available in the context." """
        }
        return guidance_map.get(category, "Provide a clear, factual answer based on the retrieved context.")

    def _parse_structured_response(self, response: str, category: int) -> Tuple[str, str]:
        """Parse structured response."""
        try:
            import re
            
            cleaned = response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            
            parsed = json.loads(cleaned)
            
            final_answer = parsed.get("final_answer", "").strip()
            reasoning = parsed.get("reasoning", "").strip()
            
            final_answer = self._post_process_answer(final_answer, category)
            
            return final_answer, reasoning
            
        except json.JSONDecodeError:
            self.logger.debug(f"JSON 解析失败，尝试文本提取")
            return self._extract_answer_from_text(response, category)

    def _extract_answer_from_text(self, response: str, category: int) -> Tuple[str, str]:
        """Extract answer from text."""
        import re
        
        reasoning = ""
        answer = response.strip()
        
        answer_match = re.search(r'"final_answer"\s*:\s*"([^"]*)"', response, re.IGNORECASE)
        if answer_match:
            answer = answer_match.group(1)
        
        reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', response, re.IGNORECASE)
        if reasoning_match:
            reasoning = reasoning_match.group(1)
        
        answer = self._post_process_answer(answer, category)
        
        return answer, reasoning

    def _post_process_answer(self, answer: str, category: int) -> str:
        """Run post process answer."""
        if not answer:
            return ""
        
        answer = answer.strip()
        
        prefixes_to_remove = [
            "Based on the context,",
            "According to the context,",
            "From the information provided,",
            "The answer is:",
            "Answer:",
        ]
        for prefix in prefixes_to_remove:
            if answer.lower().startswith(prefix.lower()):
                answer = answer[len(prefix):].strip()
        
        if category == 4:
            if len(answer) > 200 and '.' in answer:
                first_sentence = answer.split('.')[0] + '.'
                if len(first_sentence) < 150:
                    answer = first_sentence
        
        return answer

    def _calculate_scores(self,
                          question: str,
                          prediction: str,
                          reference: str,
                          category: int,
                          context: str = "",
                          reasoning: str = "") -> Dict[str, Any]:
        """Calculate scores."""
        if not prediction:
            return {
                "f1_score": 0.0,
                "exact_match": 0.0,
                "combined_score": 0.0,
                "error": "empty_prediction"
            }

        try:
            metrics = []
            
            if self.config.include_lexical_metrics:
                metrics.extend(["exact_match", "f1", "rouge", "bleu", "meteor"])
            
            if self.config.include_semantic_metrics:
                metrics.extend(["semantic_similarity", "bert_f1"])
            
            if self.config.include_llm_evaluation and self.llm_evaluate_client:
                metrics.append("llm_judge")

            eval_result = calculate_comprehensive_scores(
                gold_answer=reference,
                response=prediction,
                question=question,
                context=context,
                reasoning=reasoning,
                llm_client=self.llm_evaluate_client if self.config.include_llm_evaluation else None,
                metrics=metrics,
                category=category,
                is_adversarial=(category == 5)
            )

            scores = eval_result.get("scores", {})
            
            
            if "token_f1" in scores:
                scores["f1_score"] = scores["token_f1"]
            if "llm_accuracy" in scores:
                scores["llm_grade"] = scores["llm_accuracy"]

            score_weights = {
                "f1_score": 0.2,
                "semantic_similarity": 0.3,
                "llm_grade": 0.5
            }

            weighted_sum = 0.0
            total_weight = 0.0
            for key, weight in score_weights.items():
                val = scores.get(key)
                if val is not None and isinstance(val, (int, float)):
                    weighted_sum += float(val) * weight
                    total_weight += weight

            scores["combined_score"] = weighted_sum / total_weight if total_weight > 0 else 0.0
            
            
            if "llm_details" in eval_result:
                scores["llm_details"] = eval_result["llm_details"]

        except Exception as e:
            self.logger.warning(f"计算分数失败: {e}")
            import traceback
            traceback.print_exc()
            scores = {
                "error": str(e),
                "combined_score": 0.0
            }

        return scores

    def _calculate_sample_summary(self, qa_results: List[Dict]) -> Dict[str, Any]:
        """Calculate sample summary."""
        successful = [r for r in qa_results if r.get("evaluation_success", False)]

        if not successful:
            return {
                "total_questions": len(qa_results),
                "valid_questions": 0,
                "avg_category_score": 0.0,
                "error": "no_successful_evaluations"
            }

        category_scores = defaultdict(list)
        for r in successful:
            cat = r.get("category", 1)
            score = r.get("category_score", 0.0)
            category_scores[cat].append(score)

        category_breakdown = {}
        for cat, scores in category_scores.items():
            category_breakdown[cat] = {
                "count": len(scores),
                "avg_score": np.mean(scores) if scores else 0.0,
                "std_score": np.std(scores) if len(scores) > 1 else 0.0,
                "scores": scores
            }

        retrieval_stats = defaultdict(list)
        for r in successful:
            stats = r.get("retrieval_stats", {})
            retrieval_stats["l0_count"].append(stats.get("l0_count", 0))
            retrieval_stats["l1_count"].append(stats.get("l1_count", 0))
            retrieval_stats["l2_count"].append(stats.get("l2_count", 0))

        return {
            "total_questions": len(qa_results),
            "valid_questions": len(successful),
            "avg_category_score": np.mean([r["category_score"] for r in successful]),
            "std_category_score": np.std([r["category_score"] for r in successful]),
            "category_breakdown": category_breakdown,
            "retrieval_stats": {
                "avg_l0": np.mean(retrieval_stats["l0_count"]) if retrieval_stats["l0_count"] else 0,
                "avg_l1": np.mean(retrieval_stats["l1_count"]) if retrieval_stats["l1_count"] else 0,
                "avg_l2": np.mean(retrieval_stats["l2_count"]) if retrieval_stats["l2_count"] else 0
            }
        }

    def _save_result_to_jsonl(self, result: Dict, jsonl_file: Path):
        """Save result to JSONl."""
        try:
            with open(jsonl_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(_safe_json_serialization(result), ensure_ascii=False) + "\n")
        except Exception as e:
            self.logger.warning(f"保存结果失败: {e}")

    def run_benchmark(self,
                      sample_ids: Optional[List[str]] = None,
                      output_prefix: str = "benchmark") -> Dict[str, Any]:
        """Run benchmark."""
        start_time = datetime.now()

        if sample_ids is None:
            sample_ids = self._discover_samples()

        self.logger.info(f"\n{'='*70}")
        self.logger.info(f" 开始 Benchmark 测试")
        self.logger.info(f"{'='*70}")
        self.logger.info(f"   样本数: {len(sample_ids)}")
        self.logger.info(f"   样本: {sample_ids}")

        timestamp = start_time.strftime("%Y%m%d_%H%M%S")
        jsonl_output = self.output_dir / f"{output_prefix}_{timestamp}.jsonl"
        summary_output = self.output_dir / f"{output_prefix}_summary_{timestamp}.json"

        
        init_info = {
            "type": "benchmark_init",
            "start_time": start_time.isoformat(),
            "sample_ids": sample_ids,
            "config": {
                "retrieval_methods": self.config.retrieval_methods,
                "fusion_method": self.config.fusion_method,
                "rerank_method": self.config.rerank_method,
                "top_k": self.config.top_k
            }
        }
        self._save_result_to_jsonl(init_info, jsonl_output)

        sample_results = []
        for sample_id in sample_ids:
            result = self.evaluate_single_sample(
                sample_id=sample_id,
                jsonl_file=jsonl_output
            )
            sample_results.append(result)

            if result.get("evaluation_success"):
                summary = result["summary"]
                self.logger.info(
                    f" {sample_id}: {summary['valid_questions']}/{summary['total_questions']} 问题, "
                    f"平均分数: {summary['avg_category_score']:.3f}"
                )

        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        global_summary = self._calculate_global_summary(sample_results, total_duration)

        final_result = {
            "benchmark_info": {
                "test_type": "hierarchical_content_benchmark_v2",
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "total_duration": total_duration,
                "config": _safe_json_serialization(vars(self.config))
            },
            "sample_results": sample_results,
            "global_summary": global_summary
        }

        
        with open(summary_output, 'w', encoding='utf-8') as f:
            json.dump(_safe_json_serialization(final_result), f, indent=2, ensure_ascii=False)

        self.logger.info(f"\n 结果已保存: {jsonl_output}")
        self.logger.info(f" 汇总已保存: {summary_output}")

        self._print_summary(global_summary)

        return final_result

    def _discover_samples(self) -> List[str]:
        """Run discover samples."""
        graphs_dir = Path(self.config.graphs_dir)
        samples = []
        for item in sorted(graphs_dir.iterdir()):
            if item.is_dir() and item.name.startswith("conv-"):
                samples.append(item.name)
        return samples

    def _calculate_global_summary(self,
                                   sample_results: List[Dict],
                                   total_duration: float) -> Dict[str, Any]:
        """Calculate global summary."""
        successful = [r for r in sample_results if r.get("evaluation_success", False)]

        if not successful:
            return {
                "success_rate": 0.0,
                "total_samples": len(sample_results),
                "error": "no_successful_samples"
            }

        all_scores = []
        total_questions = 0
        valid_questions = 0
        category_aggregation = defaultdict(lambda: {"scores": [], "count": 0})

        for result in successful:
            summary = result.get("summary", {})
            total_questions += summary.get("total_questions", 0)
            valid_questions += summary.get("valid_questions", 0)

            for qa_result in result.get("qa_results", []):
                if qa_result.get("evaluation_success"):
                    score = qa_result.get("category_score", 0.0)
                    cat = qa_result.get("category", 1)
                    all_scores.append(score)
                    category_aggregation[cat]["scores"].append(score)
                    category_aggregation[cat]["count"] += 1

        category_performance = {}
        for cat, data in category_aggregation.items():
            scores = data["scores"]
            category_performance[cat] = {
                "count": data["count"],
                "avg_score": np.mean(scores) if scores else 0.0,
                "std_score": np.std(scores) if len(scores) > 1 else 0.0
            }

        return {
            "success_rate": len(successful) / len(sample_results),
            "total_samples": len(sample_results),
            "successful_samples": len(successful),
            "total_questions": total_questions,
            "valid_questions": valid_questions,
            "global_avg_score": np.mean(all_scores) if all_scores else 0.0,
            "global_std_score": np.std(all_scores) if len(all_scores) > 1 else 0.0,
            "total_duration": total_duration,
            "category_performance": category_performance
        }

    def _print_summary(self, summary: Dict[str, Any]):
        """Run print summary."""
        print("\n" + "=" * 70)
        print(" HIERARCHICAL CONTENT BENCHMARK 测试结果汇总")
        print("=" * 70)
        print(f" 样本成功率: {summary['success_rate']:.1%} ({summary['successful_samples']}/{summary['total_samples']})")
        print(f" 问题统计: {summary['valid_questions']}/{summary['total_questions']} 有效")
        print(f"Total elapsed time: {summary['total_duration']:.1f}s")
        print(f"\n 整体性能:")
        print(f"   平均分数: {summary['global_avg_score']:.3f} ± {summary.get('global_std_score', 0):.3f}")

        category_names = {
            1: "多跳问题",
            2: "时间问题",
            3: "开放域问题",
            4: "单跳问题",
            5: "对抗性问题"
        }

        if "category_performance" in summary:
            print(f"\n 各类别性能:")
            for cat, stats in sorted(summary["category_performance"].items()):
                name = category_names.get(int(cat), f"类别{cat}")
                print(f"   {name}: {stats['avg_score']:.3f} ({stats['count']}题)")

        print("=" * 70)



# CLI

def main():
    parser = argparse.ArgumentParser(
        description="Locomo 分层内容 Benchmark 测试 (V2 - Unified Retrieval)"
    )

    parser.add_argument(
        '--samples', nargs='+',
        default=None,
        help='要测试的样本ID列表（默认自动发现）'
    )

    parser.add_argument(
        '--graphs-dir',
        default=str(paths.LOCOMO_HIERARCHICAL_CONTENT_STEP4_DIR),
        help='语义图谱目录'
    )
    parser.add_argument(
        '--locomo-file',
        default=str(paths.LOCOMO_RAW_FILE),
        help='Locomo10 数据文件'
    )
    parser.add_argument(
        '--output-dir',
        default=str(paths.LOCOMO_TASK_EVAL_RESULTS_DIR / "hierarchical_content_benchmark"),
        help='输出目录'
    )

    parser.add_argument(
        '--top-k', type=int, default=20,
        help='检索结果数量'
    )
    parser.add_argument(
        '--retrieval-methods', nargs='+',
        default=["splade", "bm25", "cosine_similarity"],
        help='检索方法列表'
    )
    parser.add_argument(
        '--fusion-method',
        choices=["rrf", "weighted", "average"],
        default="rrf",
        help='融合方法'
    )
    parser.add_argument(
        '--rerank-method',
        choices=['baai', 'qwen', 'jina', 'qwen-sili', 'qwen-dashscope', 'gte-dashscope', 'none'],
        default='baai',
        help='重排序方法 (none 表示不使用重排序)'
    )

    parser.add_argument("--llm-model", 
                    #    default="gpt-4o-mini-openrouter",
                       default="gpt-4o-mini-closeai",
                       help="答案生成 LLM 模型")
    parser.add_argument("--llm-evaluate-model", 
                       #    default="gpt-4o-mini-openrouter",
                       default="gpt-4o-mini-closeai",
                       help="答案评估 LLM 模型")
    # parser.add_argument(
    #     '--llm-model', default='gpt-4o-mini-openrouter',
    # )
    # parser.add_argument(
    #     '--llm-evaluate-model', default='gpt-4o-mini-openrouter',
    # )
    parser.add_argument(
        '--disable-llm-eval', action='store_true',
        help='禁用 LLM 评估'
    )

    # Avoid mutating LogRecord fields before other handlers process the record.
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='日志级别'
    )

    args = parser.parse_args()


    print("=" * 80)
    print(" Locomo 分层内容 Benchmark 测试 (V2 - Unified Retrieval)")
    print("=" * 80)
    print(f" 图谱目录: {args.graphs_dir}")
    print(f" QA 数据: {args.locomo_file}")
    print(f" 输出目录: {args.output_dir}")
    print(f" 检索方法: {args.retrieval_methods}")
    print(f" 融合方法: {args.fusion_method}")
    print(f" 重排序: {args.rerank_method}")
    print(f" LLM 模型: {args.llm_model}")

    try:
        llm_client = None
        llm_evaluate_client = None

        if not args.disable_llm_eval:
            print("\n 初始化 LLM 客户端...")
            llm_client = LLMClient(model_name=args.llm_model)
            llm_evaluate_client = LLMClient(model_name=args.llm_evaluate_model)
            print(" LLM 客户端初始化完成")

        
        reranker_manager = None
        if args.rerank_method != 'none':
            print(f"\n 初始化重排序器: {args.rerank_method}")
            reranker_manager = RerankerManager()
            try:
                reranker_manager.get_reranker(args.rerank_method)
                print(f" 重排序器预加载成功")
            except Exception as e:
                print(f"  重排序器预加载失败: {e}")

        config = BenchmarkConfig(
            top_k=args.top_k,
            retrieval_methods=args.retrieval_methods,
            fusion_method=args.fusion_method,
            rerank_method=args.rerank_method if args.rerank_method != 'none' else None,
            include_llm_evaluation=not args.disable_llm_eval,
            graphs_dir=args.graphs_dir,
            locomo_file=args.locomo_file,
            output_dir=args.output_dir
        )

        tester = HierarchicalContentBenchmarkTester(
            config=config,
            llm_client=llm_client,
            llm_evaluate_client=llm_evaluate_client,
            reranker_manager=reranker_manager
        )

        result = tester.run_benchmark(
            sample_ids=args.samples,
            output_prefix="hierarchical_content"
        )

        print("\n Benchmark 完成!")
        return 0

    except Exception as e:
        print(f"\n Benchmark 失败: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
