"""Shared utilities for the LoCoMo four-stage benchmark pipeline.

Provides configuration loading, dataset I/O, per-query JSON persistence
with resume support, and preset prompt templates for generation and
evaluation.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)

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


def load_config(yaml_path: str) -> Dict[str, Any]:
    with open(yaml_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    experiment = cfg.setdefault("experiment", {})
    if experiment.get("config_name") is None:
        experiment["config_name"] = Path(yaml_path).stem

    return cfg


def load_dataset(data_path: str, sample_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    p = Path(data_path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")

    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Dataset root must be a list")

    if sample_ids is not None and len(sample_ids) > 0:
        id_set = set(sample_ids)
        data = [item for item in data if item.get("sample_id") in id_set]
        found = {item["sample_id"] for item in data}
        missing = id_set - found
        if missing:
            logger.warning("Sample IDs not found in dataset: %s", missing)

    return data


def filter_qa(
    qa_list: List[Dict[str, Any]],
    skip_categories: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    if not skip_categories:
        return qa_list
    skip_set = set(skip_categories)
    return [q for q in qa_list if q.get("category", 0) not in skip_set]


def save_json(path: Path, data: Any) -> None:
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
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_or_init_results(
    path: Path,
    total_queries: int,
) -> Tuple[List[Dict[str, Any]], int]:
    existing = load_json(path)
    if existing is not None and isinstance(existing, dict):
        status = existing.get("status", "")
        results = existing.get("results", [])
        if status == "completed":
            return results, len(results)
        completed = existing.get("completed_queries", 0)
        if completed > 0 and isinstance(results, list):
            return results[:completed], completed
    return [], 0


def update_results_file(
    path: Path,
    sample_id: str,
    results: List[Dict[str, Any]],
    total_queries: int,
) -> None:
    completed = len(results)
    data = {
        "sample_id": sample_id,
        "status": "in_progress" if completed < total_queries else "completed",
        "completed_queries": completed,
        "total_queries": total_queries,
        "results": results,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    save_json(path, data)


def finalize_result(path: Path) -> None:
    existing = load_json(path)
    if existing is not None and isinstance(existing, dict):
        existing["status"] = "completed"
        save_json(path, existing)


def is_graph_built(output_dir: Path, sample_id: str) -> bool:
    manifest = output_dir / sample_id / "graph" / "manifest.json"
    return manifest.exists()


def parse_judge_label(response_text: str) -> str:
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
    sample_ids: List[str],
) -> Dict[str, Any]:
    all_results: List[Dict[str, Any]] = []
    for sid in sample_ids:
        eval_path = output_dir / sid / "evaluation.json"
        data = load_json(eval_path)
        if data and isinstance(data, dict):
            all_results.extend(data.get("results", []))

    if not all_results:
        return {"total": 0, "accuracy": 0.0, "by_category": {}}

    by_category: Dict[int, List[float]] = {}
    for r in all_results:
        cat = r.get("category", 0)
        acc = r.get("llm_judge_accuracy", 0.0)
        by_category.setdefault(cat, []).append(acc)

    total_acc = sum(r.get("llm_judge_accuracy", 0.0) for r in all_results) / len(all_results)

    category_summary = {}
    for cat, accs in sorted(by_category.items()):
        category_summary[str(cat)] = {
            "count": len(accs),
            "accuracy": sum(accs) / len(accs),
        }

    return {
        "total": len(all_results),
        "accuracy": total_acc,
        "by_category": category_summary,
    }


def extract_final_answer(response_text: str) -> str:
    """Extract the final answer from a generation response.

    Looks for the ## FINAL ANSWER: section in the Chain-of-Thought output.
    If not found, returns the last non-empty line.

    Args:
        response_text: The raw LLM generation response.

    Returns:
        The extracted final answer string.
    """
    text = (response_text or "").strip()
    if not text:
        return ""

    # Look for the structured FINAL ANSWER section
    import re
    m = re.search(r"##\s*FINAL\s*ANSWER\s*:\s*\n?(.*?)$", text, re.DOTALL | re.IGNORECASE)
    if m:
        answer = m.group(1).strip()
        if answer:
            return answer

    # Fallback: return the last non-empty line
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines[-1] if lines else text


def generate_report(summary: Dict[str, Any], path: Path) -> None:
    lines = [
        "LoCoMo Benchmark Evaluation Report",
        "=" * 40,
        f"Total questions: {summary.get('total', 0)}",
        f"Overall LLM Judge Accuracy: {summary.get('accuracy', 0.0):.4f}",
        "",
        "Per-Category Breakdown:",
        "-" * 30,
    ]
    for cat, info in summary.get("by_category", {}).items():
        lines.append(
            f"  Category {cat}: {info['accuracy']:.4f} ({info['count']} questions)"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
