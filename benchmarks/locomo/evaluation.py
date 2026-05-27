"""Comprehensive evaluation metrics for the LoCoMo benchmark.

Provides lexical metrics (exact match, F1, ROUGE, BLEU, METEOR),
semantic metrics (cosine similarity, BERTScore), and LLM-as-a-judge
grading.  A :class:`ModelManager` singleton caches heavy models
(SentenceTransformer, BERTScore) to avoid repeated loading.
"""
from datetime import datetime
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

from bert_score import score
from nltk.stem import PorterStemmer
from rouge import Rouge
import nltk
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel, Field

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from dev.llm.llm_client import LLMClient

ps = PorterStemmer()

LENGTH_THRESHOLD = 5

# ================================
# Model manager — avoid repeated initialization
# ================================

class ModelManager:
    """Singleton manager that caches evaluation models to avoid repeated loading."""
    
    def __init__(self):
        self._models: Dict[str, Any] = {}
        self.logger = logging.getLogger(f"{__name__}.ModelManager")
    
    def get_sentence_model(self, model_name: str = "all-MiniLM-L6-v2") -> Optional[SentenceTransformer]:
        """Return a cached SentenceTransformer model, loading it on first call."""
        cache_key = f"sentence_transformer:{model_name}"
        
        if cache_key in self._models:
            self.logger.debug(f"Using cached sentence model: {model_name}")
            return self._models[cache_key]
        
        try:
            self.logger.info(f"Loading sentence embedding model: {model_name}")
            model = SentenceTransformer(model_name)
            self._models[cache_key] = model
            self.logger.info(f"Sentence embedding model loaded: {model_name}")
            return model
        except Exception as e:
            self.logger.error(f"Failed to load sentence model {model_name}: {e}")
            self._models[cache_key] = None
            return None
    
    def get_bert_score_model(self) -> bool:
        """Check whether BERTScore is available (caches the result)."""
        cache_key = "bert_score_available"
        
        if cache_key in self._models:
            return self._models[cache_key]
        
        try:
            # Check whether BERTScore is available
            from bert_score import score as bert_score
            _, _, f1 = bert_score(["test"], ["test"], lang="en", verbose=False)
            self._models[cache_key] = True
            self.logger.info("BERTScore model available")
            return True
        except Exception as e:
            self.logger.warning(f"BERTScore model unavailable: {e}")
            self._models[cache_key] = False
            return False
    
    def clear_cache(self):
        """Release cached models and free GPU memory."""
        for cache_key, model in self._models.items():
            try:
                if hasattr(model, 'cpu'):
                    model.cpu()
                if hasattr(model, 'cleanup'):
                    model.cleanup()
            except Exception as e:
                self.logger.warning(f"Failed to clear model {cache_key}: {e}")
        
        self._models.clear()
        
        # Clear GPU cache
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        
        self.logger.info("Evaluation model cache cleared")
    
    def get_cached_models(self) -> List[str]:
        """Return the list of currently cached model keys."""
        return list(self._models.keys())

# Global model manager singleton
_model_manager: Optional[ModelManager] = None

def get_model_manager() -> ModelManager:
    """Return the global :class:`ModelManager` singleton."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager

def get_sentence_model(model_name: str = "all-MiniLM-L6-v2") -> Optional[SentenceTransformer]:
    """Convenience wrapper around the global :class:`ModelManager`."""
    manager = get_model_manager()
    return manager.get_sentence_model(model_name)

def cleanup_evaluation_models():
    """Release all cached evaluation models and reset the singleton."""
    global _model_manager
    if _model_manager:
        _model_manager.clear_cache()
        _model_manager = None

# ================================
# LLM judge classes and functions
# ================================

class LLMGrade(BaseModel):
    """Pydantic model for LLM judge output."""
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
                                 is_adversarial: bool = False) -> Dict[str, Any]:
    """
    Compute a comprehensive set of evaluation scores for a single QA pair.

    Handles adversarial questions (category 5) separately via
    :func:`calculate_adversarial_scores`.  For normal questions, computes
    lexical, semantic, and optional LLM-judge metrics.

    Args:
        gold_answer: Ground-truth answer.
        response: Generated answer to evaluate.
        question: The original question text.
        context: Retrieved context (optional).
        reasoning: Model reasoning trace (used for adversarial eval).
        llm_client: LLM client for judge-based evaluation.
        metrics: List of metric names to compute.
        sentence_model_name: SentenceTransformer model for semantic similarity.
        category: Question category (1–5; 5 = adversarial).
        is_adversarial: Force adversarial evaluation path.

    Returns:
        A dict with ``input_info``, ``scores``, ``llm_details``, and
        ``evaluation_success``.
    """
    
    # 1. Adversarial question handling
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
    
    # 2. Normal question handling
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
    
    # --- Base metric computation ---
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
            pass  # keep default
    
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
            logging.warning(f"BERT F1 computation failed: {e}")
            results["scores"]["bert_f1"] = 0.0
    
    # --- LLM evaluation dispatch logic ---
    # Check whether any LLM evaluation metric is enabled
    has_llm_metric = "llm_judge" in metrics or "water_llm_judge" in metrics
    
    if llm_client and question and has_llm_metric:
        try:
            # Decide which grader function to use
            # Use lenient grader when water_llm_judge is in metrics
            target_grader = llm_grader
            if "water_llm_judge" in metrics:
                target_grader = water_llm_grader
            
            # Call calculate_llm_judgment
            llm_result = calculate_llm_judgment(
                llm_client=llm_client, 
                question=question, 
                gold_answer=gold_answer, 
                response=response, 
                num_runs=1, 
                context=context,
                grader_func=target_grader  # pass selected grader function
            )
            
            results["scores"]["llm_accuracy"] = float(llm_result["accuracy"])
            results["llm_details"] = llm_result
            
        except Exception as e:
            logging.warning(f"LLM evaluation failed: {e}")
            results["scores"]["llm_accuracy"] = 0.0
            results["llm_details"] = {"error": str(e)}
    
    # Compute aggregate score
    try:
        lexical_vals = [v for k, v in results["scores"].items() if k in ["exact_match", "token_f1", "rouge1_f", "rougeL_f"]]
        semantic_vals = [v for k, v in results["scores"].items() if k in ["semantic_similarity", "bert_f1"]]
        
        if lexical_vals:
            results["scores"]["avg_lexical"] = sum(lexical_vals) / len(lexical_vals)
        if semantic_vals:
            results["scores"]["avg_semantic"] = sum(semantic_vals) / len(semantic_vals)
            
        # Overall average
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
    """Evaluate multiple QA pairs and return aggregate statistics.

    Reuses cached models across calls for efficiency.

    Args:
        questions: List of question strings.
        gold_answers: List of ground-truth answers.
        predicted_answers: List of generated answers.
        contexts: Optional list of retrieved contexts.
        llm_client: Optional LLM client for judge-based evaluation.
        metrics: Metric names to compute.
        include_individual: Whether to include per-sample details.
        sentence_model_name: SentenceTransformer model for semantic similarity.

    Returns:
        A dict with ``summary``, ``aggregate_scores``, and optionally
        ``individual_results``.

    Raises:
        ValueError: If input list lengths are inconsistent.
    """
    if not (len(questions) == len(gold_answers) == len(predicted_answers)):
        raise ValueError("Input list lengths are inconsistent")
    
    if contexts is None:
        contexts = [""] * len(questions)
    elif len(contexts) != len(questions):
        raise ValueError("Context list length does not match question list")
    
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
    
    # Pre-load models to avoid repeated initialization
    manager = get_model_manager()
    if "semantic_similarity" in (metrics or []):
        logging.info(f"Pre-loading sentence model: {sentence_model_name}")
        sentence_model = manager.get_sentence_model(sentence_model_name)
        if sentence_model is None:
            logging.warning("Sentence model load failed; semantic similarity will be skipped")
    
    if "bert_f1" in (metrics or []):
        logging.info("Checking BERTScore availability")
        bert_available = manager.get_bert_score_model()
        if not bert_available:
            logging.warning("BERTScore unavailable; BERT F1 will be skipped")
    
    # Collect all evaluation results
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
            logging.error(f"Evaluation failed for sample {i+1}: {e}")
            failed_count += 1
            
            if include_individual:
                results["individual_results"].append({
                    "index": i,
                    "question": question,
                    "gold_answer": gold_answer,
                    "predicted_answer": predicted_answer,
                    "evaluation": {"error": str(e)}
                })
        
        # Log progress every 100 samples
        if (i + 1) % 100 == 0:
            logging.info(f"Batch progress: {i + 1}/{len(questions)} ({(i + 1)/len(questions)*100:.1f}%)")
    
    # Compute aggregate statistics
    if all_scores:
        # Collect metric values
        metric_values = {}
        for score_dict in all_scores:
            for metric_name, value in score_dict.items():
                if isinstance(value, (int, float)):
                    if metric_name not in metric_values:
                        metric_values[metric_name] = []
                    metric_values[metric_name].append(value)
        
        # Compute statistics
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

# ================================
# Optimized evaluation function
# ================================

def calculate_semantic_similarity(gold_answer: str, 
                                response: str, 
                                model_name: str = "all-MiniLM-L6-v2") -> float:
    """Compute cosine similarity between sentence embeddings of two texts.

    Args:
        gold_answer: Ground-truth answer.
        response: Generated answer.
        model_name: SentenceTransformer model identifier.

    Returns:
        Similarity score in [0, 1].
    """
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
        
        # Clamp return value to [0, 1]
        return max(0.0, min(1.0, similarity))
        
    except Exception as e:
        logging.error(f"Failed to calculate semantic similarity: {e}")
        return 0.0

def calculate_bert_f1_score(gold_answer: str, response: str) -> float:
    """Compute BERT F1 score between two texts.

    Args:
        gold_answer: Ground-truth answer.
        response: Generated answer.

    Returns:
        BERT F1 score (0.0 on failure).
    """
    gold_answer = str(gold_answer) if gold_answer is not None else ""
    response = str(response) if response is not None else ""
    
    if not gold_answer.strip() or not response.strip():
        return 0.0
    
    try:
        manager = get_model_manager()
        if not manager.get_bert_score_model():
            return 0.0
        
        _, _, f1 = score([response], [gold_answer], lang="en", rescale_with_baseline=True, verbose=False)
        return f1.item() if f1 is not None else 0.0
    except Exception as e:
        logging.error(f"Failed to calculate BERT F1 score: {e}")
        return 0.0

def calculate_comprehensive_metrics(gold_answer: str, 
                                  response: str, 
                                  context: str = "", 
                                  options: Optional[List[str]] = None,
                                  sentence_model_name: str = "all-MiniLM-L6-v2") -> Dict[str, Any]:
    """Compute a combined set of lexical and semantic metrics.

    Args:
        gold_answer: Ground-truth answer.
        response: Generated answer.
        context: Optional context text.
        options: Metric groups to include (``"lexical"``, ``"semantic"``).
        sentence_model_name: SentenceTransformer model identifier.

    Returns:
        A dict with ``context_tokens``, ``response_tokens``, ``gold_tokens``,
        and nested ``lexical`` / ``semantic`` sub-dicts.
    """
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
        
        # Base metrics
        metrics["lexical"]["exact_match"] = float(exact_match_score(gold_answer, response))
        metrics["lexical"]["token_f1"] = calculate_f1_score(gold_answer, response)
        
        # ROUGE metrics
        rouge_scores = calculate_rouge_score(gold_answer, response)
        metrics["lexical"].update(rouge_scores)
        
        # BLEU metrics
        bleu_scores = calculate_bleu_score(gold_answer, response)
        metrics["lexical"].update(bleu_scores)
        
        # METEOR metrics
        metrics["lexical"]["meteor"] = calculate_meteor_score(gold_answer, response)

    if "semantic" in options:
        metrics["semantic"] = {}
        
        # Semantic similarity (cached model)
        metrics["semantic"]["similarity"] = calculate_semantic_similarity(
            gold_answer, response, sentence_model_name
        )
        
        # BERT F1 (cached model check)
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
    """Comprehensive single-answer evaluation with optional LLM judgment.

    Args:
        question: The question text.
        gold_answer: Ground-truth answer.
        predicted_answer: Generated answer.
        context: Retrieved context.
        llm_client: Optional LLM client for judge-based evaluation.
        include_llm_judgment: Whether to run LLM-as-judge.
        evaluation_options: Metric groups (``"lexical"``, ``"semantic"``).
        llm_runs: Number of LLM judge runs for consistency.
        sentence_model_name: SentenceTransformer model identifier.

    Returns:
        A dict with lexical, semantic, and optional ``llm_judgment`` metrics.
    """
    if evaluation_options is None:
        evaluation_options = ["lexical", "semantic"]
    
    # Base metrics(cached model)
    result = calculate_comprehensive_metrics(
        gold_answer, predicted_answer, context, evaluation_options, sentence_model_name
    )
    
    # LLM judgment (if enabled and client provided)
    if include_llm_judgment and llm_client is not None:
        try:
            llm_result = calculate_llm_judgment(
                llm_client, question, gold_answer, predicted_answer, llm_runs
            )
            result["llm_judgment"] = llm_result
        except Exception as e:
            logging.error(f"LLM judgment failed: {e}")
            result["llm_judgment"] = {
                "error": str(e),
                "accuracy": 0.0,
                "num_runs": llm_runs,
                "consistency": False
            }
    
    # Convert numpy types
    result = convert_numpy_types(result)
    
    return result

# ================================
# Keep all original functions
# ================================

def generate_evaluation_report(eval_results: Dict[str, Any], 
                             output_format: str = "text",
                             save_path: Optional[str] = None) -> str:
    """Generate an evaluation report in text, JSON, or Markdown format."""
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
            logging.info(f"Report saved to: {save_path}")
        except Exception as e:
            logging.error(f"Failed to save report: {e}")
    
    return report

def _generate_text_report(eval_results: Dict[str, Any]) -> str:
    """Format evaluation results as a plain-text report."""
    lines = []
    lines.append("="*60)
    lines.append("Evaluation Report")
    lines.append("="*60)
    
    # Basic info
    if "summary" in eval_results:
        summary = eval_results["summary"]
        lines.append(f"Total samples: {summary.get('total_samples', 'unknown')}")
        lines.append(f"Success rate: {summary.get('success_rate', 0):.2%}")
        lines.append(f"Failed: {summary.get('failed_evaluations', 0)}")
        lines.append("")
    
    # Aggregate scores
    if "aggregate_scores" in eval_results:
        lines.append("Aggregate evaluation results:")
        lines.append("-" * 40)
        
        for metric_name, stats in eval_results["aggregate_scores"].items():
            lines.append(f"{metric_name:20} | mean: {stats['mean']:.4f} | std: {stats['std']:.4f} | range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        lines.append("")
    
    # Individual result summary (if available)
    if eval_results.get("individual_results"):
        lines.append(f"Contains {len(eval_results['individual_results'])} individual results")
    
    return "\n".join(lines)

def _generate_markdown_report(eval_results: Dict[str, Any]) -> str:
    """Format evaluation results as a Markdown table report."""
    lines = []
    lines.append("# Evaluation Report")
    lines.append("")
    
    # Basic info
    if "summary" in eval_results:
        summary = eval_results["summary"]
        lines.append("## Basic info")
        lines.append(f"- **Total samples**: {summary.get('total_samples', 'unknown')}")
        lines.append(f"- **Success rate**: {summary.get('success_rate', 0):.2%}")
        lines.append(f"- **Failed**: {summary.get('failed_evaluations', 0)}")
        lines.append("")
    
    # Aggregate scores table
    if "aggregate_scores" in eval_results:
        lines.append("## Aggregate Evaluation Results")
        lines.append("")
        lines.append("| Metric | Mean | Std | Min | Max | Median |")
        lines.append("|------|------|--------|--------|--------|--------|")
        
        for metric_name, stats in eval_results["aggregate_scores"].items():
            lines.append(f"| {metric_name} | {stats['mean']:.4f} | {stats['std']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} | {stats['median']:.4f} |")
        lines.append("")
    
    return "\n".join(lines)

async def llm_grader_async(llm_client: LLMClient, question: str, gold_answer: str, response: str) -> bool:
    """Async LLM-as-a-judge evaluator using an :class:`LLMClient`."""
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
        # Use LLMClient generate_answer method
        llm_response = llm_client.generate_answer(
            prompt=accuracy_prompt,
            temperature=0,
            max_tokens=100,  # short response
            json_format=True  # request JSON format
        )
        
        # Try to extract JSON
        try:
            if '{' in llm_response and '}' in llm_response:
                start = llm_response.find('{')
                end = llm_response.rfind('}') + 1
                json_str = llm_response[start:end]
                result = json.loads(json_str)
                label = result.get("label", "").strip().lower()
                return label == "correct"
            else:
                # Fall back to simple text matching
                return "correct" in llm_response.lower()
        except json.JSONDecodeError:
            # JSON parse failed, using text matching
            logging.warning(f"JSON parse failed, using text match: {llm_response}")
            return "correct" in llm_response.lower()
            
    except Exception as e:
        logging.error(f"LLM grader failed: {e}")
        return False

def llm_grader(llm_client: LLMClient, 
               question: str, 
               gold_answer: str, 
               response: str,
               context: str = "") -> bool:
    """Synchronous LLM-as-a-judge evaluator with optional context.

    Args:
        llm_client: LLM client for generating the judgment.
        question: The question text.
        gold_answer: Ground-truth answer.
        response: Generated answer to evaluate.
        context: Optional retrieved context to aid judgment.

    Returns:
        ``True`` if the LLM judges the answer as CORRECT.
    """
    # Adjust prompt based on context availability
    if context and context.strip():
        # Prompt with context
        accuracy_prompt = f"""You are an expert grader that determines if answers to questions match a gold standard answer.
        Be generous with your grading - focus on whether the core meaning and facts are correct.

        CONTEXT (for reference):
        {context[:1500]}

        Your task is to evaluate if the generated answer is CORRECT or WRONG compared to the gold answer, considering the provided context.

        Question: {question}
        Gold answer: {gold_answer}
        Generated answer: {response}

        Consider the answer CORRECT if:
        - It contains the same key information as the gold answer
        - The meaning is equivalent even if wording differs
        - For time questions, the time period matches even if format differs
        - For factual questions, the core facts are accurate
        - The answer is supported by the context (if context is relevant)

        Do NOT provide any explanations, reasoning, or any text outside of the JSON object.
        Return only a JSON with "label" key containing either "CORRECT" or "WRONG".
        Example: {{"label": "CORRECT"}} or {{"label": "WRONG"}}
        """
    else:
        # Original prompt without context
        accuracy_prompt = f"""You are an expert grader that determines if answers to questions match a gold standard answer.
        Be generous with your grading - focus on whether the core meaning and facts are correct.

        Your task is to evaluate if the generated answer is CORRECT or WRONG compared to the gold answer.

        Question: {question}
        Gold answer: {gold_answer}
        Generated answer: {response}

        Consider the answer CORRECT if:
        - It contains the same key information as the gold answer
        - The meaning is equivalent even if wording differs
        - For time questions, the time period matches even if format differs
        - For factual questions, the core facts are accurate

        Do NOT provide any explanations, reasoning, or any text outside of the JSON object.
        Return only a JSON with "label" key containing either "CORRECT" or "WRONG".
        Example: {{"label": "CORRECT"}} or {{"label": "WRONG"}}
        """

    try:
        # Use LLMClient generate_answer method
        llm_response = llm_client.generate_answer(
            prompt=accuracy_prompt,
            temperature=0.0,  # deterministic
            max_tokens=100,    # enough for JSON
            json_format=True  # request JSON format
        )

        # logging.debug(f"LLM grader response: {llm_response}")
        
        # Try to extract JSON
        try:
            if '{' in llm_response and '}' in llm_response:
                start = llm_response.find('{')
                end = llm_response.rfind('}') + 1
                json_str = llm_response[start:end]
                result = json.loads(json_str)
                label = result.get("label", "").strip().upper()
                return label == "CORRECT"
            else:
                # Fall back to keyword matching
                llm_response_upper = llm_response.upper()
                if "CORRECT" in llm_response_upper and "WRONG" not in llm_response_upper:
                    return True
                elif "WRONG" in llm_response_upper and "CORRECT" not in llm_response_upper:
                    return False
                else:
                    # Ambiguous — use lenient matching
                    return any(word in llm_response.lower() for word in ["correct", "right", "accurate", "yes"])
                    
        except json.JSONDecodeError as e:
            # JSON parse failed, using text matching
            logging.warning(f"JSON parse failed, using text match: {llm_response}, error: {e}")
            llm_response_upper = llm_response.upper()
            
            # More robust text matching
            if "CORRECT" in llm_response_upper:
                return True
            elif "WRONG" in llm_response_upper:
                return False
            else:
                # Last-resort fallback
                positive_words = ["correct", "right", "accurate", "yes", "true", "match"]
                negative_words = ["wrong", "incorrect", "false", "no", "mismatch"]
                
                response_lower = llm_response.lower()
                positive_count = sum(1 for word in positive_words if word in response_lower)
                negative_count = sum(1 for word in negative_words if word in response_lower)
                
                return positive_count > negative_count
                
    except Exception as e:
        logging.error(f"LLM grader sync failed: {e}")
        return False
    
def water_llm_grader(llm_client: LLMClient, 
               question: str, 
               gold_answer: str, 
               response: str,
               context: str = "") -> bool:
    """Lenient LLM grader aligned with stage5_eval prompt standards.

    Uses the same generous grading criteria as the LoCoMo benchmark
    reference implementation.

    Args:
        llm_client: LLM client for generating the judgment.
        question: The question text.
        gold_answer: Ground-truth answer.
        response: Generated answer to evaluate.
        context: Optional retrieved context to aid judgment.

    Returns:
        ``True`` if the LLM judges the answer as CORRECT.
    """
    # Insert context into prompt while keeping core locomo_grader instructions
    context_block = ""
    if context and context.strip():
        context_block = f"\nContext (Reference Information):\n{context[:1500]}\n"

    # Use locomo_grader prompt from stage5_eval
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
        # Use LLMClient generate_answer method
        llm_response = llm_client.generate_answer(
            prompt=accuracy_prompt,
            temperature=0.0,  # deterministic
            max_tokens=100,   # enough for JSON response
            json_format=True  # request JSON format
        )

        # Try to extract JSON
        try:
            if '{' in llm_response and '}' in llm_response:
                start = llm_response.find('{')
                end = llm_response.rfind('}') + 1
                json_str = llm_response[start:end]
                result = json.loads(json_str)
                label = result.get("label", "").strip().upper()
                return label == "CORRECT"
            else:
                # Fall back to keyword matching(lenient mode)
                llm_response_upper = llm_response.upper()
                if "CORRECT" in llm_response_upper and "WRONG" not in llm_response_upper:
                    return True
                elif "WRONG" in llm_response_upper and "CORRECT" not in llm_response_upper:
                    return False
                else:
                    # Ambiguous — use lenient matching
                    return any(word in llm_response.lower() for word in ["correct", "right", "accurate", "yes"])
                    
        except json.JSONDecodeError as e:
            logging.warning(f"JSON parse failed, using text match: {llm_response}, error: {e}")
            llm_response_upper = llm_response.upper()
            
            if "CORRECT" in llm_response_upper:
                return True
            elif "WRONG" in llm_response_upper:
                return False
            else:
                # Last-resort fallback
                positive_words = ["correct", "right", "accurate", "yes", "true", "match"]
                response_lower = llm_response.lower()
                return any(word in response_lower for word in positive_words)
                
    except Exception as e:
        logging.error(f"LLM grader sync failed: {e}")
        return False

def llm_grader_batch(llm_client: LLMClient, 
                    questions: List[str], 
                    gold_answers: List[str], 
                    responses: List[str],
                    contexts: Optional[List[str]] = None) -> List[bool]:  
    """Batch LLM grader that evaluates multiple QA pairs.

    Args:
        llm_client: LLM client for generating judgments.
        questions: List of question strings.
        gold_answers: List of ground-truth answers.
        responses: List of generated answers.
        contexts: Optional list of contexts (must match questions length).

    Returns:
        List of boolean judgments.

    Raises:
        ValueError: If input list lengths are inconsistent.
    """
    if not (len(questions) == len(gold_answers) == len(responses)):
        raise ValueError("Input list lengths are inconsistent")
    
    # Process context list
    if contexts is None:
        contexts = [""] * len(questions)
    elif len(contexts) != len(questions):
        raise ValueError("Context list length does not match question list")
    
    results = []
    
    for i, (question, gold_answer, response, context) in enumerate(
        zip(questions, gold_answers, responses, contexts)
    ):
        logging.debug(f"Batch eval {i+1}/{len(questions)}")
        result = llm_grader(llm_client, question, gold_answer, response, context)  # pass递上下文
        results.append(result)
    
    return results

def calculate_llm_judgment(llm_client: LLMClient, 
                        question: str, 
                        gold_answer: str, 
                        response: str,
                        num_runs: int = 1,
                        context: str = "",
                        grader_func=None) -> Dict[str, Any]:  # 新增 grader_func 参数
    """Run multiple LLM judge calls and compute accuracy and consistency.

    Args:
        llm_client: LLM client for generating judgments.
        question: The question text.
        gold_answer: Ground-truth answer.
        response: Generated answer.
        num_runs: Number of independent judge calls.
        context: Optional context for the judge.
        grader_func: Custom grader function (defaults to :func:`llm_grader`).

    Returns:
        A dict with ``judgments``, ``accuracy``, ``num_runs``,
        ``consistency``, ``confidence``, and ``context_provided``.
    """
    # 默认使用标准评估器
    if grader_func is None:
        grader_func = llm_grader

    judgments = []
    
    for i in range(num_runs):
        try:
            # 调用传入的评估函数
            judgment = grader_func(llm_client, question, gold_answer, response, context)
            judgments.append(judgment)
            logging.debug(f"LLM判断 {i+1}/{num_runs}: {judgment}")
        except Exception as e:
            logging.error(f"LLM judgment {i+1} failed: {e}")
            judgments.append(False)
    
    if not judgments:
        return {
            "judgments": [],
            "accuracy": 0.0,
            "num_runs": num_runs,
            "consistency": False,
            "error": "all judgments failed"
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
    """Run a small smoke test of the LLM grader with predefined cases."""
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
    
    print("🧪 测试LLM评估器...")
    
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
        print(f"匹配期望: {'✅' if result == case['expected'] else '❌'}")
    
    print("\n批量测试...")
    questions = [case['question'] for case in test_cases]
    gold_answers = [case['gold_answer'] for case in test_cases]
    responses = [case['response'] for case in test_cases]
    
    batch_results = llm_grader_batch(llm_client, questions, gold_answers, responses)
    print(f"批量结果: {batch_results}")

# ================================
# Keep all original base evaluation functions
# ================================

class SimpleTokenizer(object):
    """Unicode-aware tokenizer that splits on word boundaries."""
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
    """Apply NFD Unicode normalization to *text*."""
    return unicodedata.normalize('NFD', text)

def normalize_answer(s):
    """Lowercase, remove articles/punctuation, and collapse whitespace."""
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
    """Return ``True`` if the normalized token sets are identical."""
    response = str(response) if response is not None else ""
    gold_answer = str(gold_answer) if gold_answer is not None else ""
    
    response = normalize_answer(response)
    gold_answer = normalize_answer(gold_answer)
    return set(response.split()) == set(gold_answer.split())

def calculate_f1_score(gold_answer: str, response: str) -> float:
    """Compute token-level F1 after stemming."""
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
    """Compute F1 for comma-separated multi-answer fields."""
    response = str(response) if response is not None else ""
    gold_answer = str(gold_answer) if gold_answer is not None else ""
    
    responses = [r.strip() for r in response.split(',')]
    gold_answers = [g.strip() for g in gold_answer.split(',')]
    
    return np.mean([max([calculate_f1_score(ga, resp) for resp in responses]) for ga in gold_answers])

def calculate_rouge_score(gold_answer: str, response: str) -> Dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L F-measures."""
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
        logging.error(f"Failed to calculate ROUGE scores: {e}")
    
    return metrics

def calculate_bleu_score(gold_answer: str, response: str) -> Dict[str, float]:
    """Compute BLEU-1 through BLEU-4 scores."""
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
        logging.error(f"Failed to calculate BLEU scores: {e}")

    return metrics

def calculate_meteor_score(gold_answer: str, response: str) -> float:
    """Compute the METEOR score between two texts."""
    gold_answer = str(gold_answer) if gold_answer is not None else ""
    response = str(response) if response is not None else ""
    
    try:
        gold_tokens = nltk.word_tokenize(gold_answer.lower())
        response_tokens = nltk.word_tokenize(response.lower())
        return meteor_score([gold_tokens], response_tokens)
    except Exception as e:
        logging.error(f"Failed to calculate METEOR score: {e}")
        return 0.0

def convert_numpy_types(obj):
    """Recursively convert numpy scalars to native Python types for JSON serialization."""
    if isinstance(obj, np.number):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    else:
        return obj

def ems(prediction, ground_truths):
    """Backward-compatible multi-answer exact match."""
    prediction = str(prediction) if prediction is not None else ""
    
    safe_ground_truths = []
    for gt in ground_truths:
        if gt is not None:
            safe_ground_truths.append(str(gt))
        else:
            safe_ground_truths.append("")
    
    return max([exact_match_score(prediction, gt) for gt in safe_ground_truths])

# Keep all original file-level evaluation functions
def has_answer(answers, text, tokenizer=SimpleTokenizer()) -> bool:
    """Check whether any answer string appears in *text*."""
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
    """Search for answers across all top documents in an example."""
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
    """Evaluate answer recall from a JSONL file."""
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
    """Evaluate question-answering accuracy across multiple categories."""
    all_ems = []
    all_recall = []
    
    for i, line in enumerate(qas):
        if type(line[eval_key]) == list:
            answer = str(line['answer']) if line['answer'] is not None else ""
        else:
            answer = str(line['answer']) if line['answer'] is not None else ""
            
        if line['category'] == 2:  # Temporal questions
            answer = answer.split(';')[0].strip()
        
        output = str(line[eval_key]) if line[eval_key] is not None else ""
        
        if line['category'] in [1, 2, 3, 4]:  # Multi-hop, temporal, open-domain, single-hop questions
            all_ems.append(calculate_f1_score(output, answer))
        elif line['category'] in [5]:  # Adversarial questions
            output_lower = output.lower()
            if 'no information available' in output_lower or 'not mentioned' in output_lower:
                all_ems.append(1)
            else:
                all_ems.append(0)
        else:
            raise ValueError(f"Unknown question category: {line['category']}")
        
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

    print("{} QA samples evaluated; {} accuracy values".format(len(qas), len(all_ems)))
    return all_ems, 0.0, all_recall

def eval_fact_checking(infile):
    """Evaluate fact-checking accuracy from a JSONL file."""
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
    """Evaluate dialogue system performance from a JSONL file."""
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

# ================================
# Adversarial questions专用评估函数
# ================================

def llm_judge_adversarial(llm_client: LLMClient,
                         question: str,
                         generated_answer: str,
                         reasoning: str,
                         adversarial_answer: str,
                         context: str = "") -> Dict[str, Any]:
    """LLM judge for adversarial questions.

    Adversarial questions test whether the system correctly identifies
    missing or misleading information rather than fabricating answers.

    Args:
        llm_client: LLM client for generating the judgment.
        question: The adversarial question text.
        generated_answer: The system's generated answer.
        reasoning: The model's reasoning trace (used to detect awareness).
        adversarial_answer: The misleading answer from the dataset.
        context: Optional retrieved context.

    Returns:
        A dict with ``is_correct``, ``label``, ``llm_judge_raw``, and
        ``evaluation_method``.
    """
    
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
        
        # Parse response (same logic as normal llm_grader)
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
                # Fall back to keyword matching
                response_upper = response.upper()
                if "CORRECT" in response_upper and "WRONG" not in response_upper:
                    is_correct = True
                elif "WRONG" in response_upper and "CORRECT" not in response_upper:
                    is_correct = False
                else:
                    is_correct = False  # conservative strategy
                
                return {
                    "is_correct": is_correct,
                    "label": "CORRECT" if is_correct else "WRONG",
                    "llm_judge_raw": response,
                    "evaluation_method": "llm_judge_adversarial_fallback"
                }
                
        except json.JSONDecodeError as e:
            logging.warning(f"JSON parse failed，using text match: {response}")
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
        logging.error(f"LLM adversarial judgment failed: {e}")
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
    """Compute evaluation scores for adversarial questions.

    Returns the same score fields as normal questions so that results
    can be processed uniformly.

    Args:
        question: The adversarial question text.
        generated_answer: The system's generated answer.
        reasoning: The model's reasoning trace.
        adversarial_answer: The misleading answer from the dataset.
        context: Optional retrieved context.
        llm_client: Optional LLM client for judge-based evaluation.

    Returns:
        A dict with ``scores``, ``llm_details``, and ``evaluation_success``.
    """
    
    # Compute base metrics (same as normal questions)
    scores = {
        "exact_match": float(exact_match_score(adversarial_answer, generated_answer)),
        "token_f1": calculate_f1_score(adversarial_answer, generated_answer),
    }
    
    # Compute ROUGE metrics
    try:
        rouge_scores = calculate_rouge_score(adversarial_answer, generated_answer)
        scores.update(rouge_scores)
    except Exception as e:
        logging.warning(f"ROUGE computation failed: {e}")
        scores.update({"rouge1_f": 0.0, "rouge2_f": 0.0, "rougeL_f": 0.0})
    
    # Compute BLEU metrics
    try:
        bleu_scores = calculate_bleu_score(adversarial_answer, generated_answer)
        scores.update(bleu_scores)
    except Exception as e:
        logging.warning(f"BLEU computation failed: {e}")
        scores.update({"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0})
    
    # Compute METEOR metrics
    try:
        scores["meteor"] = calculate_meteor_score(adversarial_answer, generated_answer)
    except Exception as e:
        logging.warning(f"METEOR computation failed: {e}")
        scores["meteor"] = 0.0
    
    # Compute semantic similarity
    try:
        scores["semantic_similarity"] = calculate_semantic_similarity(adversarial_answer, generated_answer)
    except Exception as e:
        logging.warning(f"Semantic similarity failed: {e}")
        scores["semantic_similarity"] = 0.0
    
    # Compute BERT F1
    try:
        scores["bert_f1"] = calculate_bert_f1_score(adversarial_answer, generated_answer)
    except Exception as e:
        logging.warning(f"BERT F1 computation failed: {e}")
        scores["bert_f1"] = 0.0
    
    # LLM judgment (required; traditional metrics are unreliable for adversarial)
    if llm_client is not None:
        try:
            llm_result = llm_judge_adversarial(
                llm_client, 
                question, 
                generated_answer, 
                reasoning,  # pass reasoning trace
                adversarial_answer, 
                context
            )
            
            # Unified field: llm_accuracy (consistent with normal questions)
            scores["llm_accuracy"] = 1.0 if llm_result["is_correct"] else 0.0
            
            # Save detailed judgment info (optional, for debugging)
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
            logging.error(f"LLM adversarial evaluation failed: {e}")
            # Set to 0 on failure
            scores["llm_accuracy"] = 0.0
            
            return {
                "scores": scores,
                "llm_details": {"error": str(e)},
                "evaluation_success": False
            }
    else:
        # Cannot accurately evaluate adversarial questions without LLM client
        logging.warning("Adversarial questions require LLM judge but no client provided")
        scores["llm_accuracy"] = 0.0
        
        return {
            "scores": scores,
            "llm_details": {"error": "no_llm_client"},
            "evaluation_success": False
        }

# ================================
# Usage examples
# ================================

def example_usage():
    """Demonstrate basic evaluation without LLM judge."""
    question = "What is Caroline's relationship status?"
    gold_answer = "single"
    predicted_answer = "Based on the conversation, Caroline appears to be single."
    
    # 基础评估(cached model)
    basic_result = evaluate_answer_comprehensive(
        question=question,
        gold_answer=gold_answer,
        predicted_answer=predicted_answer,
        evaluation_options=["lexical", "semantic"],
        sentence_model_name="all-MiniLM-L6-v2"  # 指定模型名称
    )
    
    print("Basic evaluation result:")
    print(json.dumps(basic_result, indent=2))

def example_llm_grader_usage():
    """Demonstrate LLM grader usage with an LLMClient."""
    
    # 创建LLMClient
    llm_client = LLMClient("deepseek-chat")  # 或者使用其他模型
    
    # 单个评估
    question = "What is Caroline's job?"
    gold_answer = "psychologist"
    predicted_answer = "She works as a therapist and counselor."
    
    # 基础LLM评估
    is_correct = llm_grader(llm_client, question, gold_answer, predicted_answer)
    print(f"LLM评估结果: {is_correct}")
    
    # 多次运行的一致性检查
    llm_judgment = calculate_llm_judgment(
        llm_client, question, gold_answer, predicted_answer, num_runs=3
    )
    print(f"LLM判断详情: {llm_judgment}")
    
    # 综合评估（包含LLM判断）
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
    """Demonstrate batch evaluation with cached models."""
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
    
    # Batch eval(cached model)
    batch_results = batch_evaluate(
        questions=questions,
        gold_answers=gold_answers,
        predicted_answers=predicted_answers,
        metrics=["exact_match", "f1", "semantic_similarity"],
        include_individual=True,
        sentence_model_name="all-MiniLM-L6-v2"
    )
    
    print("Batch evaluation results:")
    print(json.dumps(batch_results["summary"], indent=2))
    print("Aggregate scores:")
    print(json.dumps(batch_results["aggregate_scores"], indent=2))

if __name__ == "__main__":
    # 运行测试
    print("🔍 测试LLM评估器...")
    client = LLMClient("deepseek-chat")
    test_llm_grader(client)
    
    # 运行示例
    example_llm_grader_usage()
    example_usage()
    example_batch_evaluation()
    
    # 最后清理缓存
    print("\n🧹 清理模型缓存...")
    cleanup_evaluation_models()
    print("缓存清理完成")