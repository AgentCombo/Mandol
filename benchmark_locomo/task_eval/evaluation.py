from datetime import datetime
import re
import regex
import json
import string
import unicodedata
from typing import List, Dict, Any, Optional
import numpy as np
from collections import Counter
import os
import asyncio
import time
import logging

import sys
from pathlib import Path

# Avoid mutating LogRecord fields before other handlers process the record.
from mandol.utils.logging_config import setup_logging, create_module_logger, auto_configure_logging
if auto_configure_logging() is None:
    setup_logging(level=logging.INFO)
logger = create_module_logger("locomo_evaluation")

from bert_score import BERTScorer
from nltk.stem import PorterStemmer
from rouge import Rouge
import nltk
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel, Field

from mandol.llm.llm_client import LLMClient
from mandol.utils.model_manager import global_model_manager

# try:
#     nltk.download("wordnet", quiet=True)
#     nltk.download("punkt", quiet=True)
# except Exception as e:
#     logging.warning(f"Failed to download NLTK resources: {e}")

ps = PorterStemmer()

LENGTH_THRESHOLD = 5




class EvaluationModelAdapter:
    
    def __init__(self):
        self.logger = create_module_logger("locomo_evaluation.EvaluationModelAdapter")
        self._bert_score_available: Optional[bool] = None
    
    def get_sentence_model(self, model_name: str = "all-MiniLM-L6-v2") -> Optional[SentenceTransformer]:
        """Return sentence model."""
        def loader():
            self.logger.info(f"正在加载句子嵌入模型: {model_name}")
            return SentenceTransformer(model_name)
        
        try:
            model = global_model_manager.get_or_load_model(
                model_type="sentence_transformer",
                model_name=model_name,
                loader_func=loader
            )
            return model
        except Exception as e:
            self.logger.error(f"加载句子嵌入模型失败 {model_name}: {e}")
            return None
    
    def get_bert_scorer(self, model_type: str = "roberta-large") -> Optional[BERTScorer]:
        """Return bert scorer."""
        import torch
        
        def loader():
            self.logger.info(f" 正在加载 BERTScorer 模型: {model_type}")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            try:
                scorer = BERTScorer(
                    model_type=model_type,
                    lang="en",
                    rescale_with_baseline=True,
                    device=device
                )
                self.logger.info(f" BERTScorer 加载成功 (device={device}, model={model_type})")
                return scorer
            except Exception as e:
                self.logger.error(f"BERTScorer 加载失败: {e}")
                raise
        
        try:
            scorer = global_model_manager.get_or_load_model(
                model_type="bert_scorer",
                model_name=model_type,
                loader_func=loader
            )
            return scorer
        except Exception as e:
            self.logger.error(f"获取 BERTScorer 失败: {e}")
            return None
    
    def get_bert_score_model(self) -> bool:
        """Return bert score model."""
        if self._bert_score_available is not None:
            return self._bert_score_available
        
        try:
            scorer = self.get_bert_scorer()
            if scorer is not None:
                self._bert_score_available = True
                self.logger.info(" BERTScore 模型可用（通过全局管理器）")
                return True
            else:
                self._bert_score_available = False
                return False
        except Exception as e:
            self.logger.warning(f"BERTScore模型不可用: {e}")
            self._bert_score_available = False
            return False
    
    def clear_cache(self):
        """Remove cache."""
        self._bert_score_available = None
        self.logger.info("评估模型适配器缓存标记已清空（实际模型由 GlobalModelManager 管理）")
    
    def get_cached_models(self) -> List[str]:
        """Return cached models."""
        return list(global_model_manager.get_loaded_models().keys())


ModelManager = EvaluationModelAdapter

_model_manager: Optional[EvaluationModelAdapter] = None

def get_model_manager() -> EvaluationModelAdapter:
    """Return model manager."""
    global _model_manager
    if _model_manager is None:
        _model_manager = EvaluationModelAdapter()
    return _model_manager

def get_sentence_model(model_name: str = "all-MiniLM-L6-v2") -> Optional[SentenceTransformer]:
    """Return sentence model."""
    manager = get_model_manager()
    return manager.get_sentence_model(model_name)

def cleanup_evaluation_models():
    """Release associated resources."""
    global _model_manager
    if _model_manager:
        _model_manager.clear_cache()
        _model_manager = None




class LLMGrade(BaseModel):
    llm_judgment: str = Field(description="CORRECT or WRONG")
    llm_reasoning: str = Field(description="Explain why the answer is correct or incorrect.")

def calculate_comprehensive_scores(gold_answer: str, 
                                 response: str, 
                                 question: str = "", 
                                 context: str = "",
                                 reasoning: str = "",
                                 llm_client: Optional[LLMClient] = None,
                                 metrics: Optional[List[str]] = None,
                                 sentence_model_name: str = "all-MiniLM-L6-v2",
                                 category: int = 0,
                                 is_adversarial: bool = False,
                                 llm_judge_prompt: str = "default") -> Dict[str, Any]:
    """Compute comprehensive scores."""
    
    if category == 5 or is_adversarial:
        adversarial_result = calculate_adversarial_scores(
            question=question,
            generated_answer=response,
            reasoning=reasoning,
            adversarial_answer=gold_answer,
            context=context,
            llm_client=llm_client
        )
        
        return {
            "input_info": {
                "gold_length": len(gold_answer.split()),
                "response_length": len(response.split()),
                "context_length": len(context.split()) if context else 0,
                "category": 5,
                "is_adversarial": True
            },
            "scores": adversarial_result["scores"],
            "llm_details": adversarial_result.get("llm_details", {}),
            "evaluation_success": adversarial_result.get("evaluation_success", False)
        }
    
    if llm_client is not None and metrics is None:
        metrics = ["exact_match", "f1", "rouge", "bleu", "meteor", "semantic_similarity", "bert_f1", "llm_judge"]
    if metrics is None:
        metrics = ["exact_match", "f1", "rouge", "bleu", "meteor", "semantic_similarity", "bert_f1"]
    
    gold_answer = str(gold_answer).strip() if gold_answer else ""
    response = str(response).strip() if response else ""
    
    results = {
        "input_info": {
            "gold_length": len(gold_answer.split()),
            "response_length": len(response.split()),
            "context_length": len(context.split()) if context else 0,
            "category": category
        },
        "scores": {}
    }
    
    if "exact_match" in metrics:
        try:
            results["scores"]["exact_match"] = float(exact_match_score(gold_answer, response))
        except:
            results["scores"]["exact_match"] = 0.0
    
    if "f1" in metrics:
        try:
            results["scores"]["token_f1"] = calculate_f1_score(gold_answer, response)
        except:
            results["scores"]["token_f1"] = 0.0
    
    if "rouge" in metrics:
        try:
            rouge_scores = calculate_rouge_score(gold_answer, response)
            results["scores"].update(rouge_scores)
        except:
            pass
    
    if "semantic_similarity" in metrics:
        try:
            results["scores"]["semantic_similarity"] = calculate_semantic_similarity(
                gold_answer, response, sentence_model_name
            )
        except:
            results["scores"]["semantic_similarity"] = 0.0
    
    if "bert_f1" in metrics:
        try:
            results["scores"]["bert_f1"] = calculate_bert_f1_score(gold_answer, response)
        except Exception as e:
            logger.warning(f"BERT F1分数计算失败: {e}")
            results["scores"]["bert_f1"] = 0.0
    
    has_llm_metric = "llm_judge" in metrics
    
    if llm_client and question and has_llm_metric:
        try:
            grader_func = mem0_llm_grader if _normalize_llm_judge_prompt(llm_judge_prompt) == "mem0" else None
            llm_result = calculate_llm_judgment(
                llm_client=llm_client, 
                question=question, 
                gold_answer=gold_answer, 
                response=response, 
                num_runs=1, 
                context=context,
                grader_func=grader_func
            )
            if grader_func is not None:
                llm_result["judge_prompt"] = "mem0"
            
            results["scores"]["llm_accuracy"] = float(llm_result["accuracy"])
            results["llm_details"] = llm_result
            
        except Exception as e:
            logger.warning(f"LLM评估失败: {e}")
            results["scores"]["llm_accuracy"] = 0.0
            results["llm_details"] = {"error": str(e)}
    
    try:
        lexical_vals = [v for k, v in results["scores"].items() if k in ["exact_match", "token_f1", "rouge1_f", "rougeL_f"]]
        semantic_vals = [v for k, v in results["scores"].items() if k in ["semantic_similarity", "bert_f1"]]
        
        if lexical_vals:
            results["scores"]["avg_lexical"] = sum(lexical_vals) / len(lexical_vals)
        if semantic_vals:
            results["scores"]["avg_semantic"] = sum(semantic_vals) / len(semantic_vals)
            
        all_vals = list(results["scores"].values())
        if all_vals:
            results["scores"]["overall_average"] = sum(all_vals) / len(all_vals)
            
    except Exception:
        pass
    
    results = convert_numpy_types(results)
    results["evaluation_success"] = True
    
    return results

def batch_evaluate(questions: List[str],
                  gold_answers: List[str], 
                  predicted_answers: List[str],
                  contexts: Optional[List[str]] = None,
                  llm_client: Optional[LLMClient] = None,
                  metrics: Optional[List[str]] = None,
                  include_individual: bool = False,
                  sentence_model_name: str = "all-MiniLM-L6-v2") -> Dict[str, Any]:
    """Run batch evaluate."""
    if not (len(questions) == len(gold_answers) == len(predicted_answers)):
        raise ValueError("输入列表长度不一致")
    
    if contexts is None:
        contexts = [""] * len(questions)
    elif len(contexts) != len(questions):
        raise ValueError("上下文列表长度与问题列表不一致")
    
    results = {
        "summary": {
            "total_samples": len(questions),
            "evaluation_metrics": metrics or ["exact_match", "f1", "rouge", "bleu", "meteor", "semantic_similarity", "bert_f1"],
            "timestamp": datetime.now().isoformat(),
            "sentence_model": sentence_model_name
        },
        "aggregate_scores": {},
        "individual_results": [] if include_individual else None
    }
    
    
    manager = get_model_manager()
    if "semantic_similarity" in (metrics or []):
        logger.info(f"预加载句子嵌入模型: {sentence_model_name}")
        sentence_model = manager.get_sentence_model(sentence_model_name)
        if sentence_model is None:
            logger.warning("句子嵌入模型加载失败，语义相似度评估将被跳过")
    
    if "bert_f1" in (metrics or []):
        logger.info("检查BERTScore模型可用性")
        bert_available = manager.get_bert_score_model()
        if not bert_available:
            logger.warning("BERTScore模型不可用，BERT F1评估将被跳过")
    
    all_scores = []
    failed_count = 0
    
    for i, (question, gold_answer, predicted_answer, context) in enumerate(
        zip(questions, gold_answers, predicted_answers, contexts)
    ):
        try:
            eval_result = calculate_comprehensive_scores(
                gold_answer=gold_answer,
                response=predicted_answer,
                question=question,
                context=context,
                llm_client=llm_client,
                metrics=metrics,
                sentence_model_name=sentence_model_name
            )
            
            all_scores.append(eval_result["scores"])
            
            if include_individual:
                results["individual_results"].append({
                    "index": i,
                    "question": question,
                    "gold_answer": gold_answer,
                    "predicted_answer": predicted_answer,
                    "evaluation": eval_result
                })
                
        except Exception as e:
            logger.error(f"评估第{i+1}个样本失败: {e}")
            failed_count += 1
            
            if include_individual:
                results["individual_results"].append({
                    "index": i,
                    "question": question,
                    "gold_answer": gold_answer,
                    "predicted_answer": predicted_answer,
                    "evaluation": {"error": str(e)}
                })
        
        if (i + 1) % 100 == 0:
            logger.info(f"批量评估进度: {i + 1}/{len(questions)} ({(i + 1)/len(questions)*100:.1f}%)")
    
    if all_scores:
        metric_values = {}
        for score_dict in all_scores:
            for metric_name, value in score_dict.items():
                if isinstance(value, (int, float)):
                    if metric_name not in metric_values:
                        metric_values[metric_name] = []
                    metric_values[metric_name].append(value)
        
        for metric_name, values in metric_values.items():
            if values:
                results["aggregate_scores"][metric_name] = {
                    "mean": sum(values) / len(values),
                    "std": np.std(values).item() if len(values) > 1 else 0.0,
                    "min": min(values),
                    "max": max(values),
                    "median": np.median(values).item(),
                    "count": len(values)
                }
    
    results["summary"]["failed_evaluations"] = failed_count
    results["summary"]["success_rate"] = (len(questions) - failed_count) / len(questions) if questions else 0.0
    
    return results




def calculate_semantic_similarity(gold_answer: str, 
                                response: str, 
                                model_name: str = "all-MiniLM-L6-v2") -> float:
    """Compute semantic similarity."""
    gold_answer = str(gold_answer) if gold_answer is not None else ""
    response = str(response) if response is not None else ""
    
    if not gold_answer.strip() or not response.strip():
        return 0.0
    
    try:
        sentence_model = get_sentence_model(model_name)
        if sentence_model is None:
            return 0.0
            
        gold_embedding = sentence_model.encode([gold_answer], show_progress_bar=False)[0]
        response_embedding = sentence_model.encode([response], show_progress_bar=False)[0]
        similarity = 1 - cosine(gold_embedding, response_embedding)
        
        return max(0.0, min(1.0, similarity))
        
    except Exception as e:
        logger.error(f"Failed to calculate semantic similarity: {e}")
        return 0.0

def calculate_bert_f1_score(gold_answer: str, response: str) -> float:
    """Compute bert f1 score."""
    gold_answer = str(gold_answer) if gold_answer is not None else ""
    response = str(response) if response is not None else ""
    
    if not gold_answer.strip() or not response.strip():
        return 0.0
    
    try:
        manager = get_model_manager()
        
        scorer = manager.get_bert_scorer()
        if scorer is None:
            logger.warning("BERTScorer 不可用，跳过 BERT F1 计算")
            return 0.0
        
        _, _, f1 = scorer.score([response], [gold_answer])
        return f1.item() if f1 is not None else 0.0
        
    except Exception as e:
        logger.error(f"Failed to calculate BERT F1 score: {e}")
        return 0.0

def calculate_comprehensive_metrics(gold_answer: str, 
                                  response: str, 
                                  context: str = "", 
                                  options: Optional[List[str]] = None,
                                  sentence_model_name: str = "all-MiniLM-L6-v2") -> Dict[str, Any]:
    """Compute comprehensive metrics."""
    if options is None:
        options = ["lexical", "semantic"]

    gold_answer = str(gold_answer) if gold_answer is not None else ""
    response = str(response) if response is not None else ""

    metrics = {
        "context_tokens": len(nltk.word_tokenize(context)) if context else 0,
        "response_tokens": len(nltk.word_tokenize(response)),
        "gold_tokens": len(nltk.word_tokenize(gold_answer))
    }

    if "lexical" in options:
        metrics["lexical"] = {}
        
        metrics["lexical"]["exact_match"] = float(exact_match_score(gold_answer, response))
        metrics["lexical"]["token_f1"] = calculate_f1_score(gold_answer, response)
        
        rouge_scores = calculate_rouge_score(gold_answer, response)
        metrics["lexical"].update(rouge_scores)
        
        bleu_scores = calculate_bleu_score(gold_answer, response)
        metrics["lexical"].update(bleu_scores)
        
        metrics["lexical"]["meteor"] = calculate_meteor_score(gold_answer, response)

    if "semantic" in options:
        metrics["semantic"] = {}
        
        metrics["semantic"]["similarity"] = calculate_semantic_similarity(
            gold_answer, response, sentence_model_name
        )
        
        metrics["semantic"]["bert_f1"] = calculate_bert_f1_score(gold_answer, response)

    return metrics

def evaluate_answer_comprehensive(question: str,
                                gold_answer: str,
                                predicted_answer: str,
                                context: str = "",
                                llm_client: Optional[LLMClient] = None,
                                include_llm_judgment: bool = False,
                                evaluation_options: Optional[List[str]] = None,
                                llm_runs: int = 1,
                                sentence_model_name: str = "all-MiniLM-L6-v2") -> Dict[str, Any]:
    """Evaluate answer comprehensive."""
    if evaluation_options is None:
        evaluation_options = ["lexical", "semantic"]
    
    result = calculate_comprehensive_metrics(
        gold_answer, predicted_answer, context, evaluation_options, sentence_model_name
    )
    
    if include_llm_judgment and llm_client is not None:
        try:
            llm_result = calculate_llm_judgment(
                llm_client, question, gold_answer, predicted_answer, llm_runs
            )
            result["llm_judgment"] = llm_result
        except Exception as e:
            logger.error(f"LLM判断失败: {e}")
            result["llm_judgment"] = {
                "error": str(e),
                "accuracy": 0.0,
                "num_runs": llm_runs,
                "consistency": False
            }
    
    result = convert_numpy_types(result)
    
    return result




def generate_evaluation_report(eval_results: Dict[str, Any], 
                             output_format: str = "text",
                             save_path: Optional[str] = None) -> str:
    """Generate evaluation report."""
    if output_format == "json":
        report = json.dumps(eval_results, indent=2, ensure_ascii=False)
    elif output_format == "markdown":
        report = _generate_markdown_report(eval_results)
    else:  # text
        report = _generate_text_report(eval_results)
    
    if save_path:
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"评估报告已保存到: {save_path}")
        except Exception as e:
            logger.error(f"保存报告失败: {e}")
    
    return report

def _generate_text_report(eval_results: Dict[str, Any]) -> str:
    """Generate text report."""
    lines = []
    lines.append("="*60)
    lines.append("评估报告")
    lines.append("="*60)
    
    if "summary" in eval_results:
        summary = eval_results["summary"]
        lines.append(f"总样本数: {summary.get('total_samples', 'unknown')}")
        lines.append(f"成功率: {summary.get('success_rate', 0):.2%}")
        lines.append(f"失败数: {summary.get('failed_evaluations', 0)}")
        lines.append("")
    
    if "aggregate_scores" in eval_results:
        lines.append("聚合评估结果:")
        lines.append("-" * 40)
        
        for metric_name, stats in eval_results["aggregate_scores"].items():
            lines.append(f"{metric_name:20} | 均值: {stats['mean']:.4f} | 标准差: {stats['std']:.4f} | 范围: [{stats['min']:.4f}, {stats['max']:.4f}]")
        lines.append("")
    
    if eval_results.get("individual_results"):
        lines.append(f"包含 {len(eval_results['individual_results'])} 个单独评估结果")
    
    return "\n".join(lines)

def _generate_markdown_report(eval_results: Dict[str, Any]) -> str:
    """Generate markdown report."""
    lines = []
    lines.append("# 评估报告")
    lines.append("")
    
    if "summary" in eval_results:
        summary = eval_results["summary"]
        lines.append("## 基本信息")
        lines.append(f"- **总样本数**: {summary.get('total_samples', 'unknown')}")
        lines.append(f"- **成功率**: {summary.get('success_rate', 0):.2%}")
        lines.append(f"- **失败数**: {summary.get('failed_evaluations', 0)}")
        lines.append("")
    
    if "aggregate_scores" in eval_results:
        lines.append("## 聚合评估结果")
        lines.append("")
        lines.append("| 指标 | 均值 | 标准差 | 最小值 | 最大值 | 中位数 |")
        lines.append("|------|------|--------|--------|--------|--------|")
        
        for metric_name, stats in eval_results["aggregate_scores"].items():
            lines.append(f"| {metric_name} | {stats['mean']:.4f} | {stats['std']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} | {stats['median']:.4f} |")
        lines.append("")
    
    return "\n".join(lines)

async def llm_grader_async(llm_client: LLMClient, question: str, gold_answer: str, response: str) -> bool:
    """Run LLM grader async."""
    accuracy_prompt = f"""
    Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
        (1) a question (posed by one user to another user),
        (2) a 'gold' (ground truth) answer,
        (3) a generated answer
    which you will score as CORRECT/WRONG.

    You are an expert grader that determines if answers to questions match a gold standard answer.
    Be generous with your grading - focus on whether the core meaning and facts are correct.

    The point of the question is to ask about something one user should know about the other user based on their prior conversations.
    The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
    Question: Do you remember what I got the last time I went to Hawaii?
    Gold answer: A shell necklace
    The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

    For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

    Now it's time for the real question:
    Question: {question}
    Gold answer: {gold_answer}
    Generated answer: {response}

    First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
    Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

    Just return the label CORRECT or WRONG in a json format with the key as "label".
    """

    try:
        
        llm_response = llm_client.generate_answer(
            prompt=accuracy_prompt,
            temperature=0,
            max_tokens=100,
            json_format=True
        )
        
        return _robust_parse_grading_result(llm_response)
            
    except Exception as e:
        logger.error(f"LLM grader async failed: {e}")
        return False

MEM0_JUDGE_PROMPT = """Label the generated answer as CORRECT or WRONG.
{evidence_section}
## Rules

1. **PARTIAL CREDIT**: If the generated answer includes AT LEAST ONE correct item from the gold answer's list, mark CORRECT. Getting 1 out of 2, 2 out of 4, etc. is always acceptable. Only mark WRONG if NONE of the gold answer items appear.

2. **PARAPHRASES COUNT**: Same concept in different words is CORRECT. "Chocolate raspberry tart" = "chocolate cake with raspberries". "Shelter meal service" = "volunteering at a homeless shelter". Emotions and sentiments in the same positive/negative family count as paraphrases: "proud" = "fulfilled" = "accomplished"; "huge success" = "relieved" = "thrilled" (all express positive achievement). Judge semantic meaning, not exact wording.

3. **EXTRA DETAIL IS FINE**: A longer answer that includes the gold answer's key facts plus additional information is CORRECT. Never penalize for being more detailed or specific. If the generated answer adds extra descriptive details beyond the gold answer while still referencing the same core entity or concept, mark CORRECT.

4. **DATE TOLERANCE**: Dates within 14 days of each other are CORRECT. Durations within 50% are CORRECT (e.g., "5 months" matches "six months"; "19 days" matches "two weeks"). Relative dates ("few days before November") match specific dates in the same window. A specific date (e.g., "February 2020") that is consistent with a vague reference (e.g., "a few years ago" relative to 2023) is CORRECT. Converting "last year" to the actual year (e.g., "2022" when conversations are in 2023) is CORRECT.
{evidence_rule}
5. **SEMANTIC OVERLAP**: Judge whether the generated answer addresses the same topic and captures the core idea of the gold answer. Different wording, phrasing, or level of detail should not result in WRONG if the underlying concept matches. For EMOTIONS and FEELINGS questions, answers expressing sentiments in the same valence (positive/negative) about the same event are CORRECT — do not require the exact same emotion word.

6. **SAME REFERENT**: If the generated answer mentions or references the same named entity, character, person, or concept as the gold answer, mark CORRECT — even if the generated answer provides a different physical description or includes additional details. The key question is: does the generated answer identify the same core entity? If yes, it is CORRECT.

7. **FOCUS ON KNOWLEDGE, NOT WORDING**: The goal is to assess whether the system recalled the right fact. Minor differences in specificity, phrasing, or scope should not result in WRONG. Only mark WRONG when the generated answer demonstrates a genuinely different or incorrect understanding.

## ONLY mark WRONG if:
- The generated answer contains ZERO correct items from the gold answer{evidence_wrong_clause}
- The answer addresses a completely different topic

## Question
Question: {question}
Gold answer: {answer}
Generated answer: {response}

Return JSON with "reasoning" (one sentence) and "label" (CORRECT or WRONG). Do NOT include both labels."""

MEM0_EVIDENCE_CHUNK = """
## Evidence (actual conversation messages containing the answer)
{evidence_context}
"""

MEM0_EVIDENCE_RULE = """
5. **EVIDENCE SUPPORTS ANSWER**: If the evidence corroborates the generated answer, mark CORRECT — even when the generated answer diverges from the gold answer. The gold answer may be wrong or oversimplified; if the generated answer provides a more accurate or better-supported conclusion based on the evidence, that is acceptable. Use evidence only to ACCEPT answers, never to reject them more strictly.
"""

MEM0_EVIDENCE_WRONG_CLAUSE = " AND is not supported by evidence"

def _normalize_llm_judge_prompt(llm_judge_prompt: str) -> str:
    prompt_name = (llm_judge_prompt or "default").strip().lower()
    if prompt_name in {"default", "current", "existing"}:
        return "default"
    if prompt_name in {"mem0", "mem0_judge"}:
        return "mem0"
    raise ValueError(f"Unsupported llm_judge_prompt: {llm_judge_prompt}")

def _build_mem0_judge_prompt(question: str,
                             answer: str,
                             response: str,
                             evidence_context: str = "") -> str:
    """Build the mem0-compatible LOCOMO judge prompt."""
    if evidence_context and evidence_context.strip():
        prompt = MEM0_JUDGE_PROMPT.format(
            evidence_section=MEM0_EVIDENCE_CHUNK.format(evidence_context=evidence_context),
            evidence_rule=MEM0_EVIDENCE_RULE,
            evidence_wrong_clause=MEM0_EVIDENCE_WRONG_CLAUSE,
            question=question,
            answer=answer,
            response=response,
        )
        prompt = prompt.replace("\n5. **SEMANTIC OVERLAP", "\n6. **SEMANTIC OVERLAP")
        prompt = prompt.replace("\n6. **SAME REFERENT", "\n7. **SAME REFERENT")
        prompt = prompt.replace("\n7. **FOCUS ON KNOWLEDGE", "\n8. **FOCUS ON KNOWLEDGE")
        return prompt

    return MEM0_JUDGE_PROMPT.format(
        evidence_section="",
        evidence_rule="",
        evidence_wrong_clause="",
        question=question,
        answer=answer,
        response=response,
    )

def mem0_llm_grader(llm_client: LLMClient,
                    question: str,
                    gold_answer: str,
                    response: str,
                    context: str = "") -> bool:
    """
    mem0-compatible LLM judge for LOCOMO categories 1-4.

    The original mem0 prompt excludes adversarial category 5 from scoring. Callers
    that still evaluate category 5 should decide whether to keep the dedicated
    adversarial judge or explicitly use this unified prompt.
    """
    accuracy_prompt = _build_mem0_judge_prompt(
        question=question,
        answer=gold_answer,
        response=response,
        evidence_context=context,
    )

    try:
        llm_response = llm_client.generate_answer(
            prompt=accuracy_prompt,
            temperature=0.0,
            max_tokens=200,
            json_format=True
        )
        return _robust_parse_grading_result(llm_response)
    except Exception as e:
        logger.error(f"mem0 LLM grader failed: {e}")
        return False

def calculate_mem0_llm_judgment(llm_client: LLMClient,
                                question: str,
                                gold_answer: str,
                                response: str,
                                num_runs: int = 1,
                                context: str = "") -> Dict[str, Any]:
    """Calculate LLM judgment using the mem0-compatible judge prompt."""
    result = calculate_llm_judgment(
        llm_client=llm_client,
        question=question,
        gold_answer=gold_answer,
        response=response,
        num_runs=num_runs,
        context=context,
        grader_func=mem0_llm_grader,
    )
    result["judge_prompt"] = "mem0"
    return result

def llm_grader(llm_client: LLMClient, 
               question: str, 
               gold_answer: str, 
               response: str,
               context: str = "") -> bool:
    """Run LLM grader."""
    context_block = ""
    if context and context.strip():
        context_block = f"\nContext (Reference Information):\n{context[:1500]}\n"

    accuracy_prompt = f"""
    Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
        (1) a question (posed by one user to another user),
        (2) a 'gold' (ground truth) answer,
        (3) a generated answer
    which you will score as CORRECT/WRONG.

    You are an expert grader that determines if answers to questions match a gold standard answer.
    Be generous with your grading - focus on whether the core meaning and facts are correct.

    The point of the question is to ask about something one user should know about the other user based on their prior conversations.
    {context_block}
    The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
    Question: Do you remember what I got the last time I went to Hawaii?
    Gold answer: A shell necklace
    The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

    For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

    Now it's time for the real question:
    Question: {question}
    Gold answer: {gold_answer}
    Generated answer: {response}

    First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
    Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

    Just return the label CORRECT or WRONG in a json format with the key as "label".
    """

    try:
        llm_response = llm_client.generate_answer(
            prompt=accuracy_prompt,
            temperature=0.0, 
            max_tokens=100,   
            json_format=True  
        )

        return _robust_parse_grading_result(llm_response)
                
    except Exception as e:
        logger.error(f"LLM grader sync failed: {e}")
        return False

def _robust_parse_grading_result(llm_response: str) -> bool:
    """Run robust parse grading result."""
    if not llm_response:
        return False

    try:
        if '{' in llm_response and '}' in llm_response:
            start = llm_response.find('{')
            end = llm_response.rfind('}') + 1
            json_str = llm_response[start:end]
            
            data = json.loads(json_str)
            label = str(data.get("label", "")).strip().upper()
            
            if label in ["CORRECT", "TRUE", "YES"]:
                return True
            if label in ["WRONG", "INCORRECT", "FALSE", "NO"]:
                return False
    except (json.JSONDecodeError, AttributeError):
        logger.debug(f"JSON解析失败，回退到文本分析: {llm_response[:50]}...")

    response_upper = llm_response.strip().upper()

    
    if "INCORRECT" in response_upper:
        return False
    if "WRONG" in response_upper:
        return False
    if "NOT CORRECT" in response_upper:
        return False
    if "FALSE" in response_upper:
        return False

    if re.search(r'\bCORRECT\b', response_upper):
        return True
    if re.search(r'\bTRUE\b', response_upper):
        return True
    
    if "YES" in response_upper and "NO" not in response_upper:
        return True
        
    return False

def llm_grader_batch(llm_client: LLMClient, 
                    questions: List[str], 
                    gold_answers: List[str], 
                    responses: List[str],
                    contexts: Optional[List[str]] = None) -> List[bool]:
    """Run LLM grader batch."""
    if not (len(questions) == len(gold_answers) == len(responses)):
        raise ValueError("输入列表长度不一致")
    
    if contexts is None:
        contexts = [""] * len(questions)
    elif len(contexts) != len(questions):
        raise ValueError("上下文列表长度与问题列表不一致")
    
    results = []
    
    for i, (question, gold_answer, response, context) in enumerate(
        zip(questions, gold_answers, responses, contexts)
    ):
        logger.debug(f"批量评估 {i+1}/{len(questions)}")
        result = llm_grader(llm_client, question, gold_answer, response, context)
        results.append(result)
    
    return results

def calculate_llm_judgment(llm_client: LLMClient, 
                        question: str, 
                        gold_answer: str, 
                        response: str,
                        num_runs: int = 1,
                        context: str = "",
                        grader_func=None) -> Dict[str, Any]:
    """Compute llm judgment."""
    if grader_func is None:
        grader_func = llm_grader

    judgments = []
    
    for i in range(num_runs):
        try:
            judgment = grader_func(llm_client, question, gold_answer, response, context)
            judgments.append(judgment)
            logger.debug(f"LLM判断 {i+1}/{num_runs}: {judgment}")
        except Exception as e:
            logger.error(f"LLM judgment {i+1} failed: {e}")
            judgments.append(False)
    
    if not judgments:
        return {
            "judgments": [],
            "accuracy": 0.0,
            "num_runs": num_runs,
            "consistency": False,
            "error": "所有判断都失败了"
        }
    
    accuracy = sum(judgments) / len(judgments)
    consistency = len(set(judgments)) == 1
    
    return {
        "judgments": judgments,
        "accuracy": accuracy,
        "num_runs": num_runs,
        "consistency": consistency,
        "confidence": "high" if consistency else "low",
        "context_provided": bool(context and context.strip())
    }

def test_llm_grader(llm_client: LLMClient):
    """Run test LLM grader."""
    test_cases = [
        {
            "question": "What is Caroline's relationship status?",
            "gold_answer": "single",
            "response": "Caroline is single",
            "expected": True
        },
        {
            "question": "What did they eat for dinner?",
            "gold_answer": "pizza",
            "response": "They had Chinese food",
            "expected": False
        },
        {
            "question": "When did they meet?",
            "gold_answer": "May 7, 2023",
            "response": "They met on 7 May 2023",
            "expected": True
        }
    ]
    
    print(" 测试LLM评估器...")
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n测试案例 {i}:")
        print(f"问题: {case['question']}")
        print(f"标准答案: {case['gold_answer']}")
        print(f"生成答案: {case['response']}")
        print(f"期望结果: {case['expected']}")
        
        result = llm_grader(
            llm_client, 
            case['question'], 
            case['gold_answer'], 
            case['response']
        )
        
        print(f"实际结果: {result}")
        print(f"匹配期望: {'' if result == case['expected'] else ''}")
    
    print("\n 批量测试...")
    questions = [case['question'] for case in test_cases]
    gold_answers = [case['gold_answer'] for case in test_cases]
    responses = [case['response'] for case in test_cases]
    
    batch_results = llm_grader_batch(llm_client, questions, gold_answers, responses)
    print(f"批量结果: {batch_results}")




class SimpleTokenizer(object):
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self):
        self._regexp = regex.compile(
            '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )

    def tokenize(self, text, uncased=False):
        matches = [m for m in self._regexp.finditer(text)]
        if uncased:
            tokens = [m.group().lower() for m in matches]
        else:
            tokens = [m.group() for m in matches]
        return tokens

def _normalize(text):
    """Normalize."""
    return unicodedata.normalize('NFD', text)

def normalize_answer(s):
    """Normalize answer."""
    if s is None:
        s = ""
    elif not isinstance(s, str):
        s = str(s)
    
    s = s.replace(',', "")
    
    def remove_articles(text):
        return regex.sub(r'\b(a|an|the|and)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(gold_answer: str, response: str) -> bool:
    """Run exact match score."""
    response = str(response) if response is not None else ""
    gold_answer = str(gold_answer) if gold_answer is not None else ""
    
    response = normalize_answer(response)
    gold_answer = normalize_answer(gold_answer)
    return set(response.split()) == set(gold_answer.split())

def calculate_f1_score(gold_answer: str, response: str) -> float:
    """Compute f1 score."""
    response = str(response) if response is not None else ""
    gold_answer = str(gold_answer) if gold_answer is not None else ""
    
    response_tokens = [ps.stem(w) for w in normalize_answer(response).split()]
    gold_answer_tokens = [ps.stem(w) for w in normalize_answer(gold_answer).split()]
    
    common = Counter(response_tokens) & Counter(gold_answer_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0
    
    precision = 1.0 * num_same / len(response_tokens)
    recall = 1.0 * num_same / len(gold_answer_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1

def calculate_f1_score_multi(gold_answer: str, response: str) -> float:
    """Compute f1 score multi."""
    response = str(response) if response is not None else ""
    gold_answer = str(gold_answer) if gold_answer is not None else ""
    
    responses = [r.strip() for r in response.split(',')]
    gold_answers = [g.strip() for g in gold_answer.split(',')]
    
    return np.mean([max([calculate_f1_score(ga, resp) for resp in responses]) for ga in gold_answers])

def calculate_rouge_score(gold_answer: str, response: str) -> Dict[str, float]:
    """Compute rouge score."""
    gold_answer = str(gold_answer) if gold_answer is not None else ""
    response = str(response) if response is not None else ""
    
    metrics = {"rouge1_f": 0.0, "rouge2_f": 0.0, "rougeL_f": 0.0}
    
    try:
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        rouge_scores = scorer.score(gold_answer, response)
        metrics["rouge1_f"] = rouge_scores["rouge1"].fmeasure
        metrics["rouge2_f"] = rouge_scores["rouge2"].fmeasure
        metrics["rougeL_f"] = rouge_scores["rougeL"].fmeasure
    except Exception as e:
        logger.error(f"Failed to calculate ROUGE scores: {e}")
    
    return metrics

def calculate_bleu_score(gold_answer: str, response: str) -> Dict[str, float]:
    """Compute bleu score."""
    gold_answer = str(gold_answer) if gold_answer is not None else ""
    response = str(response) if response is not None else ""
    
    metrics = {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0}

    try:
        gold_tokens = nltk.word_tokenize(gold_answer.lower())
        response_tokens = nltk.word_tokenize(response.lower())
        
        smoothing = SmoothingFunction().method1
        weights = [(1, 0, 0, 0), (0.5, 0.5, 0, 0), (0.33, 0.33, 0.33, 0), (0.25, 0.25, 0.25, 0.25)]

        for i, weight in enumerate(weights, 1):
            metrics[f"bleu{i}"] = sentence_bleu(
                [gold_tokens], response_tokens, weights=weight, smoothing_function=smoothing
            )
    except Exception as e:
        logger.error(f"Failed to calculate BLEU scores: {e}")

    return metrics

def calculate_meteor_score(gold_answer: str, response: str) -> float:
    """Compute meteor score."""
    gold_answer = str(gold_answer) if gold_answer is not None else ""
    response = str(response) if response is not None else ""
    
    try:
        gold_tokens = nltk.word_tokenize(gold_answer.lower())
        response_tokens = nltk.word_tokenize(response.lower())
        return meteor_score([gold_tokens], response_tokens)
    except Exception as e:
        logger.error(f"Failed to calculate METEOR score: {e}")
        return 0.0

def convert_numpy_types(obj):
    """Convert numpy types."""
    if isinstance(obj, np.number):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    else:
        return obj

def ems(prediction, ground_truths):
    """Run ems."""
    prediction = str(prediction) if prediction is not None else ""
    
    safe_ground_truths = []
    for gt in ground_truths:
        if gt is not None:
            safe_ground_truths.append(str(gt))
        else:
            safe_ground_truths.append("")
    
    return max([exact_match_score(prediction, gt) for gt in safe_ground_truths])

def has_answer(answers, text, tokenizer=SimpleTokenizer()) -> bool:
    """Return whether answer is available."""
    text = _normalize(text)
    text = tokenizer.tokenize(text, uncased=True)

    for answer in answers:
        answer = _normalize(answer)
        answer = tokenizer.tokenize(answer, uncased=True)
        for i in range(0, len(text) - len(answer) + 1):
            if answer == text[i: i + len(answer)]:
                return True
    return False

def check_answer(example, tokenizer) -> List[bool]:
    """Validate answer."""
    answers = example['answers']
    ctxs = example['ctxs']

    hits = []
    for _, doc in enumerate(ctxs):
        text = doc['text']
        if text is None:
            hits.append(False)
            continue
        hits.append(has_answer(answers, text, tokenizer))

    return hits

def eval_recall(infile):
    """Evaluate recall."""
    tokenizer = SimpleTokenizer()
    lines = open(infile, 'r').readlines()[1:]

    has_answer_count = 0
    answer_lengths = []
    
    for line in lines:
        line = json.loads(line)
        answer = line['answer']
        output = ' || '.join(line['output'])

        if has_answer(answer, output, tokenizer):
            has_answer_count += 1

        answer_lengths.append(len(output.split()))

    recall = round(has_answer_count/len(lines), 4)
    lens = round(np.mean(answer_lengths), 4)

    return recall, lens

def eval_question_answering(qas, eval_key='prediction', metric='f1'):
    """Evaluate question answering."""
    all_ems = []
    all_recall = []
    
    for i, line in enumerate(qas):
        if type(line[eval_key]) == list:
            answer = str(line['answer']) if line['answer'] is not None else ""
        else:
            answer = str(line['answer']) if line['answer'] is not None else ""
            
        if line['category'] == 2:
            answer = answer.split(';')[0].strip()
        
        output = str(line[eval_key]) if line[eval_key] is not None else ""
        
        if line['category'] in [1, 2, 3, 4]:
            all_ems.append(calculate_f1_score(output, answer))
        elif line['category'] in [5]:
            output_lower = output.lower()
            if 'no information available' in output_lower or 'not mentioned' in output_lower:
                all_ems.append(1)
            else:
                all_ems.append(0)
        else:
            raise ValueError(f"未知的问题类别: {line['category']}")
        
        assert i+1 == len(all_ems)

        if eval_key + '_context' in line and len(line['evidence']) > 0:
            if line[eval_key + '_context'][0].startswith('S'):
                sessions = [e[1:] for e in line[eval_key + '_context']]
                recall_acc = float(sum([ev.split(':')[0][1:] in sessions for ev in line["evidence"]]))/len(line['evidence'])
            else:
                recall_acc = float(sum([ev in line[eval_key + '_context'] for ev in line["evidence"]]))/len(line['evidence'])
            all_recall.append(recall_acc)
        else:
            all_recall.append(1)

    print("{} 个QA样本已评估; {} 个准确率值".format(len(qas), len(all_ems)))
    return all_ems, 0.0, all_recall

def eval_fact_checking(infile):
    """Evaluate fact checking."""
    tokenizer = SimpleTokenizer()
    lines = open(infile, 'r').readlines()[1:]

    exact_match_count = 0
    answer_lengths = []
    
    for line in lines:
        line = json.loads(line)
        answer = line['answer']
        output = line['output'][0]

        if answer == ["refutes"]:
            answer = ["refutes", "no", "false"]
        if answer == ["supports"]:
            answer = ["supports", "yes", "true"]

        if has_answer(answer, output, tokenizer):
            exact_match_count += 1
        
        answer_lengths.append(len(output.split()))

    em = round(exact_match_count/len(lines), 4)
    lens = round(np.mean(answer_lengths), 4)

    return em, lens

def eval_dialogue_system(infile):
    """Evaluate dialogue system."""
    lines = open(infile, 'r').readlines()[1:]

    f1_scores = []
    rl_scores = []
    answer_lengths = []
    
    for line in lines:
        line = json.loads(line)
        answer = line['answer']
        output = line['output'][0]

        f1_scores.append(calculate_f1_score(output, answer))
        rl_scores.append(calculate_bleu_score(output, [answer]))
        answer_lengths.append(len(output.split()))

    F1 = round(np.mean(f1_scores), 4)
    RL = round(np.mean(rl_scores), 4)
    lens = round(np.mean(answer_lengths), 4)

    return F1, RL, lens




def llm_judge_adversarial(llm_client: LLMClient,
                         question: str,
                         generated_answer: str,
                         reasoning: str,
                         adversarial_answer: str,
                         context: str = "") -> Dict[str, Any]:
    """Run LLM judge adversarial."""
    
    judge_prompt = f"""You are evaluating an answer to an ADVERSARIAL question.

    IMPORTANT: Adversarial questions are designed to test if a system fabricates information or correctly identifies when information is missing/misleading.

    Question: {question}

    Generated Answer: {generated_answer}

    Reasoning Process (check if the system identified the issue): 
    {reasoning[:500] if reasoning else "No reasoning provided"}

    Known Misleading Answer (DO NOT USE AS STANDARD): {adversarial_answer}

    Context (for reference):
    {context[:1000] if context else "No context provided"}

    EVALUATION TASK:
    Determine if the generated answer is CORRECT for this adversarial question.

    A CORRECT answer should do ONE of the following:
    1. **Identify missing/misleading information**: Clearly state that information is not available/not mentioned
    2. **Refuse to fabricate**: Explicitly decline to make up information
    3. **Provide factually correct answer**: IF information IS actually present in context (verify carefully)

    CRITICAL: Check the reasoning process to see if the system recognized the adversarial nature or missing information.

    An INCORRECT answer:
    1. Fabricates information not present in context
    2. Matches or closely resembles the known misleading answer
    3. Provides a confident answer when information is missing

    KEY INDICATORS:
    - Reasoning mentions "no information", "not found", "misleading" → Usually CORRECT
    - Answer phrases like "no information available", "cannot determine" → Usually CORRECT  
    - Answer matches misleading answer without identifying the issue → WRONG
    - Confident claims without contextual support → WRONG

    Return ONLY a JSON with "label" key containing "CORRECT" or "WRONG":
    {{"label": "CORRECT"}} or {{"label": "WRONG"}}
    """
    
    try:
        response = llm_client.generate_answer(
            prompt=judge_prompt,
            temperature=0.0,
            max_tokens=100,
            json_format=True
        )
        
        try:
            if '{' in response and '}' in response:
                start = response.find('{')
                end = response.rfind('}') + 1
                json_str = response[start:end]
                result = json.loads(json_str)
                label = result.get("label", "").strip().upper()
                is_correct = (label == "CORRECT")
                
                return {
                    "is_correct": is_correct,
                    "label": label,
                    "llm_judge_raw": response,
                    "evaluation_method": "llm_judge_adversarial"
                }
            else:
                response_upper = response.upper()
                if "CORRECT" in response_upper and "WRONG" not in response_upper:
                    is_correct = True
                elif "WRONG" in response_upper and "CORRECT" not in response_upper:
                    is_correct = False
                else:
                    is_correct = False
                
                return {
                    "is_correct": is_correct,
                    "label": "CORRECT" if is_correct else "WRONG",
                    "llm_judge_raw": response,
                    "evaluation_method": "llm_judge_adversarial_fallback"
                }
                
        except json.JSONDecodeError as e:
            logger.warning(f"JSON解析失败，使用文本匹配: {response}")
            response_upper = response.upper()
            is_correct = "CORRECT" in response_upper
            
            return {
                "is_correct": is_correct,
                "label": "CORRECT" if is_correct else "WRONG",
                "llm_judge_raw": response,
                "evaluation_method": "llm_judge_adversarial_text_match",
                "parse_error": str(e)
            }
            
    except Exception as e:
        logger.error(f"LLM对抗性判断失败: {e}")
        return {
            "is_correct": False,
            "label": "WRONG",
            "error": str(e),
            "evaluation_method": "llm_judge_adversarial_failed"
        }


def calculate_adversarial_scores(question: str,
                                generated_answer: str,
                                reasoning: str,
                                adversarial_answer: str,
                                context: str = "",
                                llm_client: Optional[LLMClient] = None) -> Dict[str, Any]:
    """Compute adversarial scores."""
    
    scores = {
        "exact_match": float(exact_match_score(adversarial_answer, generated_answer)),
        "token_f1": calculate_f1_score(adversarial_answer, generated_answer),
    }
    
    try:
        rouge_scores = calculate_rouge_score(adversarial_answer, generated_answer)
        scores.update(rouge_scores)
    except Exception as e:
        logger.warning(f"ROUGE分数计算失败: {e}")
        scores.update({"rouge1_f": 0.0, "rouge2_f": 0.0, "rougeL_f": 0.0})
    
    try:
        bleu_scores = calculate_bleu_score(adversarial_answer, generated_answer)
        scores.update(bleu_scores)
    except Exception as e:
        logger.warning(f"BLEU分数计算失败: {e}")
        scores.update({"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0})
    
    try:
        scores["meteor"] = calculate_meteor_score(adversarial_answer, generated_answer)
    except Exception as e:
        logger.warning(f"METEOR分数计算失败: {e}")
        scores["meteor"] = 0.0
    
    try:
        scores["semantic_similarity"] = calculate_semantic_similarity(adversarial_answer, generated_answer)
    except Exception as e:
        logger.warning(f"语义相似度计算失败: {e}")
        scores["semantic_similarity"] = 0.0
    
    try:
        scores["bert_f1"] = calculate_bert_f1_score(adversarial_answer, generated_answer)
    except Exception as e:
        logger.warning(f"BERT F1计算失败: {e}")
        scores["bert_f1"] = 0.0
    
    if llm_client is not None:
        try:
            llm_result = llm_judge_adversarial(
                llm_client, 
                question, 
                generated_answer, 
                reasoning,
                adversarial_answer, 
                context
            )
            
            scores["llm_accuracy"] = 1.0 if llm_result["is_correct"] else 0.0
            
            
            llm_details = {
                "correct": llm_result["is_correct"],
                "label": llm_result.get("label", "UNKNOWN"),
                "evaluation_method": llm_result.get("evaluation_method", "unknown")
            }
            
            return {
                "scores": scores,
                "llm_details": llm_details,
                "evaluation_success": True
            }
            
        except Exception as e:
            logger.error(f"LLM对抗性评估失败: {e}")
            scores["llm_accuracy"] = 0.0
            
            return {
                "scores": scores,
                "llm_details": {"error": str(e)},
                "evaluation_success": False
            }
    else:
        logger.warning("对抗性问题需要LLM判断，但未提供LLM客户端")
        scores["llm_accuracy"] = 0.0
        
        return {
            "scores": scores,
            "llm_details": {"error": "no_llm_client"},
            "evaluation_success": False
        }




def example_usage():
    """Run example usage."""
    question = "What is Caroline's relationship status?"
    gold_answer = "single"
    predicted_answer = "Based on the conversation, Caroline appears to be single."
    
    basic_result = evaluate_answer_comprehensive(
        question=question,
        gold_answer=gold_answer,
        predicted_answer=predicted_answer,
        evaluation_options=["lexical", "semantic"],
        sentence_model_name="all-MiniLM-L6-v2"
    )
    
    print("Basic evaluation result:")
    print(json.dumps(basic_result, indent=2))

def example_llm_grader_usage():
    """Run example LLM grader usage."""
    
    llm_client = LLMClient("deepseek-chat")
    
    question = "What is Caroline's job?"
    gold_answer = "psychologist"
    predicted_answer = "She works as a therapist and counselor."
    
    is_correct = llm_grader(llm_client, question, gold_answer, predicted_answer)
    print(f"LLM评估结果: {is_correct}")
    
    llm_judgment = calculate_llm_judgment(
        llm_client, question, gold_answer, predicted_answer, num_runs=3
    )
    print(f"LLM判断详情: {llm_judgment}")
    
    comprehensive_result = evaluate_answer_comprehensive(
        question=question,
        gold_answer=gold_answer,
        predicted_answer=predicted_answer,
        llm_client=llm_client,
        include_llm_judgment=True,
        llm_runs=2,
        sentence_model_name="all-MiniLM-L6-v2"
    )
    print(f"综合评估结果: {json.dumps(comprehensive_result, indent=2)}")

def example_batch_evaluation():
    """Run example batch evaluation."""
    questions = [
        "What is Caroline's relationship status?",
        "What did they eat for dinner?",
        "When did they meet?"
    ]
    gold_answers = ["single", "pizza", "May 7, 2023"]
    predicted_answers = [
        "Caroline is single",
        "They had Chinese food", 
        "They met on 7 May 2023"
    ]
    
    batch_results = batch_evaluate(
        questions=questions,
        gold_answers=gold_answers,
        predicted_answers=predicted_answers,
        metrics=["exact_match", "f1", "semantic_similarity"],
        include_individual=True,
        sentence_model_name="all-MiniLM-L6-v2"
    )
    
    print("批量评估结果:")
    print(json.dumps(batch_results["summary"], indent=2))
    print("聚合分数:")
    print(json.dumps(batch_results["aggregate_scores"], indent=2))

if __name__ == "__main__":
    print(" 测试LLM评估器...")
    client = LLMClient("deepseek-chat")
    test_llm_grader(client)
    
    example_llm_grader_usage()
    example_usage()
    example_batch_evaluation()
    
    print("\n 清理模型缓存...")
    cleanup_evaluation_models()
    print(" 缓存清理完成")
