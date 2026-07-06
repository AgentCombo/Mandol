"""Utilities for evaluation."""

from datetime import datetime
import regex
import json
import string
import unicodedata
from typing import List, Dict, Any, Optional, Tuple
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
    
    def __init__(self):
        self.logger = create_module_logger("longmemeval_evaluation.EvaluationModelAdapter")
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




def _is_normalized_exact_match(gold_answer: str, response: str) -> bool:
    """Return True when both answers are non-empty and identical after eval normalization."""
    gold_norm = normalize_answer(gold_answer)
    response_norm = normalize_answer(response)
    return bool(gold_norm) and gold_norm == response_norm


def _coerce_grader_label(label: Any) -> Optional[bool]:
    """Convert common LLM grader label shapes into a boolean judgment."""
    if isinstance(label, bool):
        return label

    if label is None:
        return None

    label_text = str(label).strip().upper()
    label_text = regex.sub(r"[^A-Z]", "", label_text)

    if label_text in {"CORRECT", "TRUE", "YES"}:
        return True
    if label_text in {"WRONG", "FALSE", "NO", "INCORRECT"}:
        return False
    return None


def _parse_llm_grader_response(llm_response: str) -> bool:
    """Parse a CORRECT/WRONG grader response, preferring structured JSON labels."""
    response_text = str(llm_response or "")

    if "{" in response_text and "}" in response_text:
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        json_str = response_text[start:end]
        try:
            result = json.loads(json_str)
            parsed_label = _coerce_grader_label(result.get("label"))
            if parsed_label is not None:
                return parsed_label
        except (json.JSONDecodeError, AttributeError, TypeError) as e:
            logger.warning(f"JSON解析失败，使用文本匹配: {llm_response}, 错误: {e}")

    label_matches = [
        match.group(1).upper()
        for match in regex.finditer(r"\b(CORRECT|WRONG)\b", response_text, flags=regex.IGNORECASE)
    ]
    if len(label_matches) == 1:
        return label_matches[0] == "CORRECT"

    if regex.search(r"\b(incorrect|wrong|false|no|mismatch)\b", response_text, flags=regex.IGNORECASE):
        return False
    return bool(
        regex.search(r"\b(correct|right|accurate|yes|true|match)\b", response_text, flags=regex.IGNORECASE)
    )


def llm_grader(llm_client: LLMClient, 
               question: str, 
               gold_answer: str, 
               response: str,
               question_type: str = "default",
               context: str = "") -> bool:
    """Run LLM grader."""
    if _is_normalized_exact_match(gold_answer, response):
        return True

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

    Return JSON only, with exactly one field:
    {{"label": "CORRECT"}} or {{"label": "WRONG"}}
    """

    try:
        llm_response = llm_client.generate_answer(
            prompt=accuracy_prompt,
            temperature=0.0,
            max_tokens=100,
            json_format=True
        )

        return _parse_llm_grader_response(llm_response)
                
    except Exception as e:
        logger.error(f"LLM grader 失败: {e}")
        return False


MEM0_JUDGE_PROMPT = """I will give you a question, a correct answer (or rubric), and a model response. Decide whether the model response is correct.

CORE PRINCIPLE — Semantic equivalence: Judge by MEANING, not exact words. Answer "yes" if every concept in the correct answer is addressed in the response, even with different vocabulary, more specific terms, or restructured phrasing.

IMPORTANT BIAS CHECK: You have a tendency to say "no" too quickly. Before concluding "no", you MUST verify the answer is truly wrong, not just differently worded. When in doubt, lean toward "yes".

Rules:

**Equivalence & Supersets**
- Equivalent or superset responses are correct. Extra details are fine unless proven to be factually wrong. Extra qualifiers are fine unless proven to be wrong. E.g., "a blue dress and a matching necklace" is correct when the answer is "a blue dress."
- If a response captures the most specific part (exact item/place/name) but omits a broader container, it's correct.
- Same factual meaning with different phrasing = correct (e.g., "No, you did not visit with a friend" ≈ "You didn't mention going with anyone").
- Adding scope qualifiers like "regular-season" or "excluding X" is fine as long as the core value is correct. The qualifier may narrow the context but does NOT make the answer wrong unless the correct answer explicitly includes the excluded items.

**Lists & Compound Terms**
- For list answers, match each item by semantic meaning. A concept is covered if restated via synonyms, sub-concepts, or related terms. Adding methodological detail or rewording verbs to near-synonyms is acceptable.
- A broad term like "A and B significance" is covered if the response addresses the topic area through related specific terms, even without naming each component literally.
- If some items as listed as "or"s, "maybe"s and potential answers, it's okay if the answer does not include those.
- If two items in a list achieve the same purpose, listing just one of them is fine.

IMPORTANT: The "anti-preference" items are very specific!
Eg. Someone "not interested in general AI topics" could be very interested in specific AI topics in general AI *conferences*; those are not the same thing and should be accepted! topics != conferences

**Numbers & Precision**
- Hedging ("at least 3", "approximately") is fine if the core number matches. A range that includes the correct answer is correct.
Generally, if the user themself would be satisfied by the response, it is acceptable. Ie. If the answer is conditional on information they would have (eg. their birthday, some hidden dependent information), and would be correct with that information, that is acceptable.
- More precise answers are correct: "22 days" matches "3 weeks"; "over $270" matches "$270."; "9 1/2 months" matches "9 months";

- Rough answers are correct: "about nine months" ≈ "9 months; "8 months and 20 days" matches "9 months";

- Off-by-one errors on days/weeks/months are acceptable.
- Approximate unit conversions are equivalent: "14 weeks" ≈ "3 months", "6 months" ≈ "half a year."
- Round time ranges generously: 7 months and 16 days ≈ 8 months.
- Notes instead of chords are acceptable when justified
- A correct number with added context (e.g., "about 5 months ago (around December 2022)") is correct — the parenthetical date is supplementary, not a contradiction.

**Dates & Temporal**
- Date format variations are equivalent: "February 1st" = "Feb 1, 2023" = "on February 1."
- Same-day event ordering swaps are acceptable.
- Outdated info alongside the correct updated answer is acceptable if the current value is identified.
- "recent" is upto 6 years ago, which means 2017+
- References like "last weekend", "last Wednesday", etc. are imprecise - people sometimes mean the weekend/Wednesday before the latest one if they're near it. "Last 3 months" can include boundary days of the 4th month back. "Last month" includes the current month so far. Be flexible with such timestamps

**Counting Edge Cases**
- If correct answer is "0" or "nothing found," model saying "not enough information" is also correct.
- Similarly, If correct answer is "not enough information", model saying "0" or "nothing found," is also correct.

**Preference/Personalization Rubrics** (apply in order):
1. Correct if the response demonstrates awareness of user's personal context (preferences, habits, interests). Need not satisfy every rubric point.
2. Primary criterion: do main suggestions align with what the user WANTS?
3. Anti-preferences: evaluate the OVERALL thrust, not keyword scanning. If the response largely suggests correct options, minor incidental references to "not-preferred" things are fine.
4. Mentioning a phone app as a MEANS to a preferred activity (e.g., meditation app for sleep) is not "suggesting phone use." Judge by the activity, not delivery mechanism.
5. "May not prefer" = mild preference, not hard prohibition. Secondary/context-dependent inclusion is fine.
6. Explicit acknowledgment of anti-preferences (e.g., "keep screens off") strengthens correctness.
7. Context-dependent suggestions are acceptable (reading is fine on a bus even if rubric flags visual attention activities). Adjacent genres alongside preferred ones are additive, not contradictory.
8. If the rubric mentions specific user resources/tools (e.g., "Suica card", "TripIt app"), the response is correct if it demonstrates awareness of the user's MAIN personal context even if it does not name every specific tool. The rubric is a guide, not a checklist.

**Abstention Matching**
- If correct answer = unanswerable/abstention, ANY phrasing that conveys "I don't have this information" is correct, regardless of what partial context is mentioned or omitted.
- Saying "not enough information" while mentioning partial related context = correct abstention.
- Saying "no record of X" or "only have plans for X, not actual dates" = correct abstention.
- The key test: does the response REFUSE to answer the question? If yes, it matches an abstention ground truth, period.
- This is a one-way rule: if the correct answer is a concrete fact, number, date, item, link, or preference rubric, a model response that refuses to answer ("not enough information", "no record", "cannot determine", "not specified") is NOT correct.

FINAL CHECK: Before answering "no," you MUST reason through these steps:
1. What is the core factual claim or intent of the correct answer?
2. Does the model response address that same claim, even in different words?
3. Is the response a superset (correct answer + extra details)?
4. For numbers: does the core number match, ignoring hedging/qualifiers?
5. For abstentions: does the response effectively decline to answer?
Only answer "no" if, after this analysis, a core concept is entirely unaddressed or contradicted.

Question: {question}

Correct Answer: {answer}

Model Response: {response}

Return JSON only, with exactly two fields:
{{"reasoning": "brief explanation", "label": "yes"}} or {{"reasoning": "brief explanation", "label": "no"}}"""


def _normalize_llm_judge_prompt(llm_judge_prompt: str) -> str:
    prompt_name = (llm_judge_prompt or "default").strip().lower()
    if prompt_name in {"default", "current", "existing"}:
        return "default"
    if prompt_name in {"mem0", "mem0_judge"}:
        return "mem0"
    raise ValueError(f"Unsupported llm_judge_prompt: {llm_judge_prompt}")


def _build_mem0_judge_prompt(question: str, answer: str, response: str) -> str:
    """Build the mem0-compatible LongMemEval judge prompt."""
    return MEM0_JUDGE_PROMPT.format(
        question=question,
        answer=str(answer),
        response=response,
    )


def _parse_mem0_grader_response(llm_response: str) -> bool:
    """Parse mem0 LongMemEval judge output, preferring the final yes/no verdict."""
    response_text = str(llm_response or "").strip()
    if not response_text:
        return False

    if "{" in response_text and "}" in response_text:
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        try:
            result = json.loads(response_text[start:end])
            for key in ("label", "verdict", "answer"):
                parsed_label = _coerce_grader_label(result.get(key))
                if parsed_label is not None:
                    return parsed_label
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass

    tail = response_text.split("</judge_thinking>")[-1].strip()
    candidate_lines = [line.strip() for line in tail.splitlines() if line.strip()]
    if candidate_lines:
        final_line = regex.sub(r"[^A-Za-z]", "", candidate_lines[-1]).lower()
        if final_line == "yes":
            return True
        if final_line == "no":
            return False

    label_matches = [
        match.group(1).lower()
        for match in regex.finditer(r"\b(yes|no|correct|wrong|incorrect|true|false)\b", tail, flags=regex.IGNORECASE)
    ]
    if label_matches:
        last = label_matches[-1]
        return last in {"yes", "correct", "true"}

    return _parse_llm_grader_response(response_text)


def _normalized_token_stems(text: str) -> List[str]:
    """Normalize and stem answer tokens for conservative deterministic shortcuts."""
    return [ps.stem(token) for token in normalize_answer(text).split()]


def _has_negation(text: str) -> bool:
    raw_text = str(text or "").lower()
    if regex.search(
        r"\b(no|not|none|never|without|neither|nor|cannot|can't|cant|don't|dont|doesn't|doesnt|didn't|didnt|isn't|isnt|aren't|arent|won't|wont)\b",
        raw_text,
    ):
        return True
    tokens = set(normalize_answer(text).split())
    return bool(tokens & {"no", "not", "none", "never", "without", "neither", "nor", "cannot", "cant", "dont", "doesnt", "didnt"})


def _is_abstention_like(text: str) -> bool:
    normalized = normalize_answer(text)
    if not normalized:
        return True

    abstention_patterns = [
        r"\bnot enough\b",
        r"\bnot enough information\b",
        r"\binsufficient information\b",
        r"\bno information\b",
        r"\bnot specified\b",
        r"\bdoesnt specify\b",
        r"\bdoes not specify\b",
        r"\bdidnt specify\b",
        r"\bdid not specify\b",
        r"\bnot mentioned\b",
        r"\bdid not mention\b",
        r"\bdoes not mention\b",
        r"\bnot provided\b",
        r"\bunknown\b",
        r"\bcannot determine\b",
        r"\bcan not determine\b",
        r"\bdont know\b",
        r"\bdo not know\b",
        r"\bno record\b",
        r"\bnothing found\b",
        r"\bnot available\b",
    ]
    return any(regex.search(pattern, normalized) for pattern in abstention_patterns)


def _is_zero_like(text: str) -> bool:
    normalized = normalize_answer(text)
    return bool(regex.search(r"\b(0|zero|none|nothing|no)\b", normalized))


def _extract_numbers(text: str) -> List[str]:
    return regex.findall(r"\b\d+(?:\.\d+)?\b", str(text or ""))


def _mem0_deterministic_judgment(gold_answer: str, response: str) -> Optional[Tuple[bool, str]]:
    """
    Conservative mem0 shortcuts for cases the prompt explicitly treats as correct.

    These are intentionally one-way positive shortcuts. Ambiguous or negative cases
    still go to the LLM judge.
    """
    if _is_normalized_exact_match(gold_answer, response):
        return True, "normalized_exact_match"

    gold_norm = normalize_answer(gold_answer)
    response_norm = normalize_answer(response)
    if not gold_norm or not response_norm:
        return None

    if _is_abstention_like(gold_answer) and (_is_abstention_like(response) or _is_zero_like(response)):
        return True, "mem0_abstention_equivalence"
    if _is_zero_like(gold_answer) and _is_abstention_like(response):
        return True, "mem0_zero_abstention_equivalence"

    # A concise gold answer embedded in a longer generated answer is a strict
    # superset case under the mem0 rubric.
    response_is_abstention = _is_abstention_like(response)
    if (
        not _has_negation(response)
        and not response_is_abstention
        and regex.search(rf"\b{regex.escape(gold_norm)}\b", response_norm)
    ):
        return True, "mem0_gold_substring"

    gold_stems = _normalized_token_stems(gold_answer)
    response_stems = set(_normalized_token_stems(response))
    if (
        1 <= len(gold_stems) <= 8
        and not _has_negation(gold_answer)
        and not _has_negation(response)
        and not response_is_abstention
        and all(token in response_stems for token in gold_stems)
    ):
        return True, "mem0_gold_tokens_covered"

    gold_numbers = _extract_numbers(gold_answer)
    response_numbers = set(_extract_numbers(response))
    gold_tokens = normalize_answer(gold_answer).split()
    if (
        len(gold_numbers) == 1
        and gold_numbers[0] in response_numbers
        and not _has_negation(response)
        and not response_is_abstention
        and all(token == gold_numbers[0] or token in {"about", "approximately", "around", "roughly"} for token in gold_tokens)
    ):
        return True, "mem0_numeric_match"

    return None


def mem0_llm_grader_with_details(llm_client: LLMClient,
                                 question: str,
                                 gold_answer: str,
                                 response: str,
                                 question_type: str = "default",
                                 context: str = "") -> Dict[str, Any]:
    """Run the mem0-compatible judge and keep parse/debug details for re-eval reports."""
    shortcut = _mem0_deterministic_judgment(gold_answer, response)
    if shortcut is not None:
        judgment, reason = shortcut
        return {
            "judgment": judgment,
            "raw_response": "",
            "judge_prompt": "mem0",
            "short_circuit": reason,
            "question_type": question_type,
        }

    accuracy_prompt = _build_mem0_judge_prompt(
        question=question,
        answer=gold_answer,
        response=response,
    )

    try:
        llm_response = llm_client.generate_answer(
            prompt=accuracy_prompt,
            temperature=0.0,
            max_tokens=300,
            json_format=True,
        )
        judgment = _parse_mem0_grader_response(llm_response)
        return {
            "judgment": judgment,
            "raw_response": llm_response,
            "judge_prompt": "mem0",
            "question_type": question_type,
            "json_format": True,
        }
    except Exception as e:
        logger.error(f"mem0 LLM grader 失败: {e}")
        return {
            "judgment": False,
            "raw_response": "",
            "judge_prompt": "mem0",
            "question_type": question_type,
            "error": str(e),
        }


def mem0_llm_grader(llm_client: LLMClient,
                    question: str,
                    gold_answer: str,
                    response: str,
                    question_type: str = "default",
                    context: str = "") -> bool:
    """
    mem0-compatible LongMemEval LLM judge.

    question_type and context are accepted for interface compatibility. The
    mem0 LongMemEval judge prompt itself is unified across question types.
    """
    return bool(
        mem0_llm_grader_with_details(
            llm_client=llm_client,
            question=question,
            gold_answer=gold_answer,
            response=response,
            question_type=question_type,
            context=context,
        )["judgment"]
    )


def calculate_mem0_llm_judgment(llm_client: LLMClient,
                                question: str,
                                gold_answer: str,
                                response: str,
                                question_type: str = "default",
                                num_runs: int = 1,
                                context: str = "") -> Dict[str, Any]:
    """Calculate LLM judgment using the mem0-compatible LongMemEval prompt."""
    runs = max(1, num_runs)
    judgments = []
    raw_responses = []
    short_circuits = []
    errors = []

    for _ in range(runs):
        detail = mem0_llm_grader_with_details(
            llm_client=llm_client,
            question=question,
            gold_answer=gold_answer,
            response=response,
            question_type=question_type,
            context=context,
        )
        judgments.append(bool(detail.get("judgment", False)))
        raw_responses.append(detail.get("raw_response", ""))
        if detail.get("short_circuit"):
            short_circuits.append(detail["short_circuit"])
        if detail.get("error"):
            errors.append(detail["error"])

    accuracy = sum(judgments) / len(judgments) if judgments else 0.0
    consistency = len(set(judgments)) == 1 if judgments else False
    result = {
        "judgments": judgments,
        "accuracy": accuracy,
        "num_runs": runs,
        "consistency": consistency,
        "confidence": "high" if consistency else "low",
        "question_type": question_type,
        "context_provided": bool(context and context.strip()),
        "judge_prompt": "mem0",
        "raw_responses": raw_responses,
    }
    if short_circuits:
        result["short_circuit"] = short_circuits[0] if len(set(short_circuits)) == 1 else short_circuits
    if errors:
        result["errors"] = errors
    return result


def calculate_llm_judgment(llm_client: LLMClient, 
                        question: str, 
                        gold_answer: str, 
                        response: str,
                        question_type: str = "default",
                        num_runs: int = 1,
                        context: str = "",
                        grader_func=None) -> Dict[str, Any]:
    """Compute llm judgment."""
    if _is_normalized_exact_match(gold_answer, response):
        runs = max(1, num_runs)
        return {
            "judgments": [True] * runs,
            "accuracy": 1.0,
            "num_runs": runs,
            "consistency": True,
            "confidence": "deterministic",
            "question_type": question_type,
            "context_provided": bool(context and context.strip()),
            "short_circuit": "normalized_exact_match"
        }

    judgments = []
    
    for i in range(num_runs):
        try:
            if grader_func is None:
                grader_func = llm_grader
            result = grader_func(llm_client, question, gold_answer, response, question_type, context)
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
                                 sentence_model_name: str = "all-MiniLM-L6-v2",
                                 llm_judge_prompt: str = "default") -> Dict[str, Any]:
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
            if _normalize_llm_judge_prompt(llm_judge_prompt) == "mem0":
                llm_result = calculate_mem0_llm_judgment(
                    llm_client, question, gold_answer, response,
                    question_type=question_type,
                    num_runs=1,
                    context=context,
                )
            else:
                llm_result = calculate_llm_judgment(
                    llm_client, question, gold_answer, response, 
                    question_type=question_type,
                    num_runs=1, 
                    context=context,
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
