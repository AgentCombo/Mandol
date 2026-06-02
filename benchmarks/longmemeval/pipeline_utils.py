"""Shared utilities for the LongMemEval four-stage benchmark pipeline.

Provides configuration loading, dataset I/O, JSON persistence, and
preset prompt templates for generation and evaluation (shared with the
LoCoMo benchmark).

The LongMemEval dataset uses ``question_id`` as the primary key instead
of ``sample_id``, and ``question_type`` instead of ``category``.

Note: Each LongMemEval sample has exactly ONE question, so result files
are flat dicts (not wrapped in a "results" list).  Resume is simply
file-existence check.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates (same as locomo)
# ---------------------------------------------------------------------------

GENERATION_PROMPT_TEMPLATE = """
You are an intelligent memory assistant tasked with retrieving accurate information from episodic memories.

# CONTEXT:
You have access to episodic memories from conversations between two speakers. These memories contain
timestamped information that may be relevant to answering the question.

# INSTRUCTIONS:
Your goal is to synthesize information from all relevant memories to provide a comprehensive and accurate answer.
You MUST follow a structured Chain-of-Thought process to ensure no details are missed.
Actively look for connections between people, places, and events to build a complete picture. Synthesize information from different memories to answer the user's question.
It is CRITICAL that you move beyond simple fact extraction and perform logical inference. When the evidence strongly suggests a connection, you must state that connection. Do not dismiss reasonable inferences as "speculation." Your task is to provide the most complete answer supported by the available evidence.

# CRITICAL REQUIREMENTS:
1. NEVER omit specific names - use "Amy's colleague Rob" not "a colleague"
2. ALWAYS include exact numbers, amounts, prices, percentages, dates, times
3. PRESERVE frequencies exactly - "every Tuesday and Thursday" not "twice a week"
4. MAINTAIN all proper nouns and entities as they appear

# RESPONSE FORMAT (You MUST follow this structure):

## STEP 1: RELEVANT MEMORIES EXTRACTION
[List each memory that relates to the question, with its timestamp]
- Memory 1: [timestamp] - [content]
- Memory 2: [timestamp] - [content]
...

## STEP 2: KEY INFORMATION IDENTIFICATION
[Extract ALL specific details from the memories]
- Names mentioned: [list all person names, place names, company names]
- Numbers/Quantities: [list all amounts, prices, percentages]
- Dates/Times: [list all temporal information]
- Frequencies: [list any recurring patterns]
- Other entities: [list brands, products, etc.]

## STEP 3: CROSS-MEMORY LINKING
[Identify entities that appear in multiple memories and link related information. Make reasonable inferences when entities are strongly connected.]
- Shared entities: [list people, places, events mentioned across different memories]
- Connections found: [e.g., "Memory 1 mentions A moved from hometown → Memory 2 mentions A's hometown is LA → Therefore A moved from LA"]
- Inferred facts: [list any facts that require combining information from multiple memories]

## STEP 4: TIME REFERENCE CALCULATION
[If applicable, convert relative time references]
- Original reference: [e.g., "last year" from May 2022]
- Calculated actual time: [e.g., "2021"]

## STEP 5: CONTRADICTION CHECK
[If multiple memories contain different information]
- Conflicting information: [describe]
- Resolution: [explain which is most recent/reliable]

## STEP 6: DETAIL VERIFICATION CHECKLIST
- [ ] All person names included: [list them]
- [ ] All locations included: [list them]
- [ ] All numbers exact: [list them]
- [ ] All frequencies specific: [list them]
- [ ] All dates/times precise: [list them]
- [ ] All proper nouns preserved: [list them]

## STEP 7: ANSWER FORMULATION
[Explain how you're combining the information to answer the question]

## FINAL ANSWER:
[Provide the concise answer with ALL specific details preserved]

---

{context}

Question: {question}

Now, follow the Chain-of-Thought process above to answer the question:
"""

EVALUATION_PROMPT_TEMPLATE = """
You are an expert grader that determines if answers to questions match a gold standard answer.

Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
    (1) a question (posed by one user to another user),
    (2) a 'gold' (ground truth) answer,
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label".
"""


# ---------------------------------------------------------------------------
# Config & dataset helpers
# ---------------------------------------------------------------------------

def load_config(yaml_path: str) -> Dict[str, Any]:
    """Load a YAML config file, setting defaults for missing fields."""
    with open(yaml_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    experiment = cfg.setdefault("experiment", {})
    if experiment.get("config_name") is None:
        experiment["config_name"] = Path(yaml_path).stem

    return cfg


def load_dataset(
    data_path: str,
    question_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Load LongMemEval dataset items, optionally filtering by question_id.

    Args:
        data_path: Path to the JSON dataset file.
        question_ids: Optional list of question_ids to include (if None, load all).

    Returns:
        List of sample dicts.
    """
    p = Path(data_path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")

    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Dataset root must be a list")

    if question_ids is not None and len(question_ids) > 0:
        id_set = set(question_ids)
        data = [item for item in data if item.get("question_id") in id_set]
        found = {item["question_id"] for item in data}
        missing = id_set - found
        if missing:
            logger.warning("Question IDs not found in dataset: %s", missing)

    return data


def filter_questions(
    items: List[Dict[str, Any]],
    skip_types: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Filter dataset items by question_type.

    Args:
        items: List of dataset samples.
        skip_types: Question types to exclude (e.g. ["multi-session"]).

    Returns:
        Filtered list.
    """
    if not skip_types:
        return items
    skip_set = set(skip_types)
    return [q for q in items if q.get("question_type", "") not in skip_set]


# ---------------------------------------------------------------------------
# JSON persistence
# ---------------------------------------------------------------------------

def save_json(path: Path, data: Any) -> None:
    """Atomically write JSON data to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, indent=2, default=str)
    fd, tmp = tempfile.mkstemp(suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
        os.replace(tmp, str(path))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_json(path: Path) -> Any:
    """Load JSON from *path*, returning None if absent."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------

def is_graph_built(output_dir: Path, question_id: str) -> bool:
    """Return True if a manifest file for this sample exists."""
    manifest = output_dir / question_id / "graph" / "manifest.json"
    return manifest.exists()


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

def extract_final_answer(response_text: str) -> str:
    """Extract the final answer from a generation response.

    Looks for the ## FINAL ANSWER: section in the Chain-of-Thought output.
    If not found, returns the last non-empty line.

    Args:
        response_text: The raw LLM generation response.

    Returns:
        The extracted final answer string.
    """
    import re
    text = (response_text or "").strip()
    if not text:
        return ""

    # Look for the structured FINAL ANSWER section
    m = re.search(
        r"##\s*FINAL\s*ANSWER\s*:\s*\n?(.*?)$",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if m:
        answer = m.group(1).strip()
        if answer:
            return answer

    # Fallback: return the last non-empty line
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines[-1] if lines else text


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def parse_judge_label(response_text: str) -> str:
    """Parse the LLM judge response into CORRECT or WRONG.

    Args:
        response_text: The raw judge response.

    Returns:
        "CORRECT" or "WRONG".
    """
    text = response_text.strip()
    if not text:
        return "WRONG"

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "label" in parsed:
            label = str(parsed["label"]).upper().strip()
            if label in ("CORRECT", "WRONG"):
                return label
    except json.JSONDecodeError:
        pass

    upper = text.upper()
    if "CORRECT" in upper and "WRONG" not in upper:
        return "CORRECT"
    if "WRONG" in upper:
        return "WRONG"

    return "WRONG"


def aggregate_llm_judge_accuracy(
    output_dir: Path,
    question_ids: List[str],
) -> Dict[str, Any]:
    """Aggregate evaluation results across all samples.

    Each evaluation file is a flat dict (one question per sample).
    Returns a dict with overall accuracy and per-question-type breakdown.
    """
    all_results: List[Dict[str, Any]] = []
    for qid in question_ids:
        eval_path = output_dir / qid / "evaluation.json"
        data = load_json(eval_path)
        if data and isinstance(data, dict) and "llm_judge_accuracy" in data:
            all_results.append(data)

    if not all_results:
        return {"total": 0, "accuracy": 0.0, "by_type": {}}

    by_type: Dict[str, List[float]] = {}
    for r in all_results:
        qtype = r.get("question_type", "unknown")
        acc = r.get("llm_judge_accuracy", 0.0)
        by_type.setdefault(qtype, []).append(acc)

    total_acc = sum(
        r.get("llm_judge_accuracy", 0.0) for r in all_results
    ) / len(all_results)

    type_summary = {}
    for qtype, accs in sorted(by_type.items()):
        type_summary[qtype] = {
            "count": len(accs),
            "accuracy": sum(accs) / len(accs),
        }

    return {
        "total": len(all_results),
        "accuracy": total_acc,
        "by_type": type_summary,
    }


def generate_report(summary: Dict[str, Any], path: Path) -> None:
    """Write a human-readable evaluation report."""
    lines = [
        "LongMemEval Benchmark Evaluation Report",
        "=" * 45,
        f"Total questions: {summary.get('total', 0)}",
        f"Overall LLM Judge Accuracy: {summary.get('accuracy', 0.0):.4f}",
        "",
        "Per-Type Breakdown:",
        "-" * 30,
    ]
    for qtype, info in summary.get("by_type", {}).items():
        lines.append(
            f"  {qtype}: {info['accuracy']:.4f} ({info['count']} questions)"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
