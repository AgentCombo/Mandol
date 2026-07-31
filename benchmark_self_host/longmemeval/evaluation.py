"""Utilities for evaluation."""

from datetime import datetime
import regex
import json
import string
import unicodedata
from typing import List, Dict, Any, Optional
import numpy as np
from collections import Counter
import logging

import sys
from pathlib import Path

# Avoid mutating LogRecord fields before other handlers process the record.
from mandol.utils.logging_config import setup_logging, create_module_logger, auto_configure_logging
if auto_configure_logging() is None:
    setup_logging(level=logging.INFO)
logger = create_module_logger("longmemeval_evaluation")

from bert_score import BERTScorer
from nltk.stem import PorterStemmer
from rouge import Rouge
import nltk
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer

from mandol.llm.llm_client import LLMClient
from mandol.utils.model_manager import global_model_manager

# try:
#     nltk.download("wordnet", quiet=True)
#     nltk.download("punkt", quiet=True)
# except Exception as e:
#     logging.warning(f"Failed to download NLTK resources: {e}")

ps = PorterStemmer()




class EvaluationModelAdapter:
    """Load and cache evaluation models used by LongMemEval scoring."""
    
    def __init__(self):
        self.logger = create_module_logger("longmemeval_evaluation.EvaluationModelAdapter")
        self._bert_score_available: Optional[bool] = None
    
    def get_sentence_model(self, model_name: str = "all-MiniLM-L6-v2") -> Optional[SentenceTransformer]:
        """Return sentence model."""
        def loader():
            self.logger.info(f"Loading sentence embedding model: {model_name}")
            return SentenceTransformer(model_name)
        
        try:
            model = global_model_manager.get_or_load_model(
                model_type="sentence_transformer",
                model_name=model_name,
                loader_func=loader
            )
            return model
        except Exception as e:
            self.logger.error(f"Failed to load sentence embedding model {model_name}: {e}")
            return None
    
    def get_bert_scorer(self, model_type: str = "roberta-large") -> Optional[BERTScorer]:
        """Return bert scorer."""
        import torch
        
        def loader():
            self.logger.info(f"Loading BERTScorer model: {model_type}")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            try:
                scorer = BERTScorer(
                    model_type=model_type,
                    lang="en",
                    rescale_with_baseline=True,
                    device=device
                )
                self.logger.info(f"BERTScorer loaded (device={device}, model={model_type})")
                return scorer
            except Exception as e:
                self.logger.error(f"Failed to load BERTScorer: {e}")
                raise
        
        try:
            scorer = global_model_manager.get_or_load_model(
                model_type="bert_scorer",
                model_name=model_type,
                loader_func=loader
            )
            return scorer
        except Exception as e:
            self.logger.error(f"Failed to get BERTScorer: {e}")
            return None
    
    def get_bert_score_model(self) -> bool:
        """Return bert score model."""
        if self._bert_score_available is not None:
            return self._bert_score_available
        
        try:
            scorer = self.get_bert_scorer()
            if scorer is not None:
                self._bert_score_available = True
                self.logger.info("BERTScore model is available through GlobalModelManager.")
                return True
            else:
                self._bert_score_available = False
                return False
        except Exception as e:
            self.logger.warning(f"BERTScore model is unavailable: {e}")
            self._bert_score_available = False
            return False
    
    def clear_cache(self):
        """Remove cache."""
        self._bert_score_available = None
        self.logger.info("Evaluation adapter cache markers cleared; model objects are managed by GlobalModelManager.")


ModelManager = EvaluationModelAdapter

_model_manager: Optional[EvaluationModelAdapter] = None

def get_model_manager() -> EvaluationModelAdapter:
    """Return model manager."""
    global _model_manager
    if _model_manager is None:
        _model_manager = EvaluationModelAdapter()
    return _model_manager

def cleanup_evaluation_models():
    """Release associated resources."""
    global _model_manager
    if _model_manager:
        _model_manager.clear_cache()
        _model_manager = None




def llm_grader(llm_client: LLMClient, 
               question: str, 
               gold_answer: str, 
               response: str,
               question_type: str = "default",
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

        try:
            if '{' in llm_response and '}' in llm_response:
                start = llm_response.find('{')
                end = llm_response.rfind('}') + 1
                json_str = llm_response[start:end]
                result = json.loads(json_str)
                label = result.get("label", "").strip().upper()
                return label == "CORRECT"
            else:
                llm_response_upper = llm_response.upper()
                if "CORRECT" in llm_response_upper and "WRONG" not in llm_response_upper:
                    return True
                elif "WRONG" in llm_response_upper and "CORRECT" not in llm_response_upper:
                    return False
                else:
                    return any(word in llm_response.lower() for word in ["correct", "right", "accurate", "yes"])
                    
        except json.JSONDecodeError as e:
            logger.warning(f"JSON解析失败，使用文本匹配: {llm_response}, 错误: {e}")
            llm_response_upper = llm_response.upper()
            
            if "CORRECT" in llm_response_upper:
                return True
            elif "WRONG" in llm_response_upper:
                return False
            else:
                positive_words = ["correct", "right", "accurate", "yes", "true", "match"]
                response_lower = llm_response.lower()
                return any(word in response_lower for word in positive_words)
                
    except Exception as e:
        logger.error(f"LLM grader 失败: {e}")
        return False


def calculate_llm_judgment(llm_client: LLMClient, 
                        question: str, 
                        gold_answer: str, 
                        response: str,
                        question_type: str = "default",
                        num_runs: int = 1,
                        context: str = "") -> Dict[str, Any]:
    """Compute llm judgment."""
    judgments = []
    
    for i in range(num_runs):
        try:
            result = llm_grader(llm_client, question, gold_answer, response, question_type, context)
            judgments.append(result)
        except Exception as e:
            logger.warning(f"LLM判断第 {i+1} 次失败: {e}")
            continue
    
    if not judgments:
        return {
            "judgments": [],
            "accuracy": 0.0,
            "num_runs": num_runs,
            "consistency": False,
            "question_type": question_type,
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
        "question_type": question_type,
        "context_provided": bool(context and context.strip())
    }




def calculate_comprehensive_scores(gold_answer: str, 
                                 response: str, 
                                 question: str = "", 
                                 context: str = "",
                                 question_type: str = "default",
                                 llm_client: Optional[LLMClient] = None,
                                 metrics: Optional[List[str]] = None,
                                 sentence_model_name: str = "all-MiniLM-L6-v2") -> Dict[str, Any]:
    """Compute comprehensive scores."""
    
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
            "question_type": question_type
        },
        "scores": {}
    }
    
    if "exact_match" in metrics:
        try:
            results["scores"]["exact_match"] = float(exact_match_score(gold_answer, response))
        except Exception as e:
            logger.warning(f"精确匹配计算失败: {e}")
            results["scores"]["exact_match"] = 0.0
    
    if "f1" in metrics:
        try:
            results["scores"]["token_f1"] = calculate_f1_score(gold_answer, response)
        except Exception as e:
            logger.warning(f"F1计算失败: {e}")
            results["scores"]["token_f1"] = 0.0
    
    if "rouge" in metrics:
        try:
            rouge_scores = calculate_rouge_score(gold_answer, response)
            results["scores"].update(rouge_scores)
        except Exception as e:
            logger.warning(f"ROUGE计算失败: {e}")
            results["scores"].update({"rouge1_f": 0.0, "rouge2_f": 0.0, "rougeL_f": 0.0})
    
    if "bleu" in metrics:
        try:
            bleu_scores = calculate_bleu_score(gold_answer, response)
            results["scores"].update(bleu_scores)
        except Exception as e:
            logger.warning(f"BLEU计算失败: {e}")
            results["scores"].update({"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0})
    
    if "meteor" in metrics:
        try:
            results["scores"]["meteor"] = calculate_meteor_score(gold_answer, response)
        except Exception as e:
            logger.warning(f"METEOR计算失败: {e}")
            results["scores"]["meteor"] = 0.0
    
    if "semantic_similarity" in metrics:
        try:
            results["scores"]["semantic_similarity"] = calculate_semantic_similarity(
                gold_answer, response, sentence_model_name
            )
        except Exception as e:
            logger.warning(f"语义相似度计算失败: {e}")
            results["scores"]["semantic_similarity"] = 0.0
    
    if "bert_f1" in metrics:
        try:
            results["scores"]["bert_f1"] = calculate_bert_f1_score(gold_answer, response)
        except Exception as e:
            logger.warning(f"BERT F1计算失败: {e}")
            results["scores"]["bert_f1"] = 0.0
    
    if llm_client and question and "llm_judge" in metrics:
        try:
            llm_result = calculate_llm_judgment(
                llm_client, question, gold_answer, response, 
                question_type=question_type,
                num_runs=1, 
                context=context
            )
            results["scores"]["llm_accuracy"] = llm_result["accuracy"]
            results["llm_details"] = llm_result
        except Exception as e:
            logger.warning(f"LLM评估失败: {e}")
            results["scores"]["llm_accuracy"] = 0.0
            results["llm_details"] = {"error": str(e)}

    try:
        lexical_scores = []
        semantic_scores = []
        
        for key in ["exact_match", "token_f1", "rouge1_f", "rougeL_f", "bleu4", "meteor"]:
            if key in results["scores"]:
                lexical_scores.append(results["scores"][key])
        
        for key in ["semantic_similarity", "bert_f1"]:
            if key in results["scores"]:
                semantic_scores.append(results["scores"][key])
        
        if lexical_scores:
            results["scores"]["avg_lexical"] = sum(lexical_scores) / len(lexical_scores)
        if semantic_scores:
            results["scores"]["avg_semantic"] = sum(semantic_scores) / len(semantic_scores)
        
        all_scores = lexical_scores + semantic_scores
        if all_scores:
            results["scores"]["overall_average"] = sum(all_scores) / len(all_scores)
            
    except Exception as e:
        logger.warning(f"综合分数计算失败: {e}")
    
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
        manager.get_sentence_model(sentence_model_name)
    if "bert_f1" in (metrics or []):
        manager.get_bert_score_model()
    
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
        sentence_model = get_model_manager().get_sentence_model(model_name)
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




def generate_evaluation_report(eval_results: Dict[str, Any], 
                             output_format: str = "text",
                             save_path: Optional[str] = None) -> str:
    """Generate evaluation report."""
    if output_format == "json":
        report = json.dumps(eval_results, indent=2, ensure_ascii=False)
    elif output_format == "markdown":
        report = _generate_markdown_report(eval_results)
    else:
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
    lines.append("LongMemEval 评估报告")
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
            lines.append(f"{metric_name:20} | 均值: {stats['mean']:.4f} | 标准差: {stats['std']:.4f}")
        lines.append("")
    
    return "\n".join(lines)

def _generate_markdown_report(eval_results: Dict[str, Any]) -> str:
    """Generate markdown report."""
    lines = []
    lines.append("# LongMemEval 评估报告")
    lines.append("")
    
    if "summary" in eval_results:
        summary = eval_results["summary"]
        lines.append("## 基本信息")
        lines.append(f"- **总样本数**: {summary.get('total_samples', 'unknown')}")
        lines.append(f"- **成功率**: {summary.get('success_rate', 0):.2%}")
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




if __name__ == "__main__":
    print(" 测试评估模块...")
    
    llm_client = LLMClient("deepseek-chat")
    
    question = "How long is my daily commute to work?"
    gold_answer = "45 minutes each way"
    predicted_answer = "Your daily commute takes approximately 45 minutes in each direction."
    
    result = llm_grader(llm_client, question, gold_answer, predicted_answer)
    print(f"LLM评估结果: {result}")
    
    comprehensive_result = calculate_comprehensive_scores(
        gold_answer=gold_answer,
        response=predicted_answer,
        question=question,
        llm_client=llm_client
    )
    print(f"\n综合评估结果:")
    print(json.dumps(comprehensive_result, indent=2, ensure_ascii=False))
    
    cleanup_evaluation_models()
    print("\n 测试完成")
