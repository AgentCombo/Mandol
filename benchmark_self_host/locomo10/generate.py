#!/usr/bin/env python3
"""Generate LoCoMo10 answers from self-host tri-tower retrieval results."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _bootstrap_paths() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        src = parent / "src"
        if (src / "mandol").exists():
            if str(src) not in sys.path:
                sys.path.insert(0, str(src))
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return parent
    raise RuntimeError("Could not locate repository root containing src/mandol")


REPO_ROOT = _bootstrap_paths()
SCRIPT_DIR = Path(__file__).resolve().parent

from mandol.llm.configs import MODEL_CONFIGS  # noqa: E402
from mandol.llm.llm_client import LLMClient  # noqa: E402
from mandol.utils.logging_config import auto_configure_logging, setup_logging  # noqa: E402

LOGGER = logging.getLogger("locomo10_generate")
_TOKEN_ENCODER: Any = None
GENERATION_MODEL_CHOICES = [
    "gpt-4o-mini-closeai",
    "gpt-4o-mini-openrouter",
    "gpt-4.1-mini-closeai",
    "gpt-4.1-mini-openrouter",
]
MODEL_ALIASES = {
    "gpt-4.1-mini": "gpt-4.1-mini-closeai",
    "gpt-4o-mini": "gpt-4o-mini-closeai",
}
MODEL_CHOICES_WITH_ALIASES = GENERATION_MODEL_CHOICES + sorted(MODEL_ALIASES)


def configure_logging(debug: bool = False) -> None:
    level = logging.DEBUG if debug else logging.INFO
    if auto_configure_logging() is None:
        setup_logging(level=level)
    logging.getLogger().setLevel(level)
    LOGGER.setLevel(level)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def append_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def compact_text(text: Any, max_chars: int) -> str:
    value = str(text or "").strip()
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    return value[: max_chars - 20].rstrip() + "\n...[truncated]"


def category_guidance(category: int) -> str:
    guidance_map = {
        1: "Multi-hop reasoning - Trace connections across hierarchical patterns, graph relationships, AND episodic timeline",
        2: "Temporal question - PRIORITIZE episodic [Time: ...] markers, verify with hierarchical sessions AND graph entities",
        3: "Open-domain question - Synthesize comprehensive view from all three towers",
        4: "Single-hop fact - Verify fact across hierarchical context, graph evidence, AND episodic memory",
        5: "Adversarial question - Check information existence in ALL THREE systems before answering",
    }
    return guidance_map.get(category, "General question - Use all three information sources")


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    global _TOKEN_ENCODER
    try:
        if _TOKEN_ENCODER is None:
            import tiktoken

            _TOKEN_ENCODER = tiktoken.encoding_for_model("gpt-4o-mini")
        return len(_TOKEN_ENCODER.encode(text))
    except Exception:
        pass
    return max(1, len(text) // 4)


def resolve_model_name(model_name: str) -> str:
    if model_name in MODEL_CONFIGS:
        return model_name
    alias = MODEL_ALIASES.get(model_name)
    if alias and alias in MODEL_CONFIGS:
        LOGGER.info("Resolved model alias %s -> %s", model_name, alias)
        return alias
    return model_name


def item_content(item: Dict[str, Any]) -> str:
    if not isinstance(item, dict):
        return str(item)
    content = item.get("content")
    if content:
        return str(content)
    raw_data = item.get("raw_data") or {}
    if isinstance(raw_data, dict):
        for key in ("text_content", "original_content", "message", "content"):
            if raw_data.get(key):
                return str(raw_data[key])
    return json.dumps(item, ensure_ascii=False)


def l0_item_content(item: Dict[str, Any]) -> str:
    if not isinstance(item, dict):
        return str(item)
    raw_data = item.get("raw_data") or {}
    original = item.get("content") or ""
    enhanced = ""
    if isinstance(raw_data, dict):
        enhanced = str(raw_data.get("enhanced_content") or "").strip()
        if not original:
            for key in ("text_content", "original_content", "message", "content"):
                if raw_data.get(key):
                    original = str(raw_data[key])
                    break
    original = str(original or "").strip()
    if enhanced and original and enhanced != original:
        return f"{enhanced}\nOriginal: {original}"
    return enhanced or original or item_content(item)


def build_hierarchical_text(record: Dict[str, Any], max_items: int, max_item_chars: int, max_total_chars: int) -> Tuple[str, Dict[str, Any]]:
    final_l0 = (record.get("final") or {}).get("l0") or []
    if not isinstance(final_l0, list):
        final_l0 = []
    items = [item for item in final_l0 if isinstance(item, dict)]
    included_items = items[:max_items]
    lines = []
    for index, item in enumerate(included_items, 1):
        content = compact_text(l0_item_content(item), max_item_chars)
        lines.append(f"Observation {index}: {content}")
    text = "\n\n".join(lines) if lines else "No hierarchical context available."
    text = compact_text(text, max_total_chars)
    return text, {
        "hierarchical_enabled": bool(included_items),
        "count": len(included_items),
        "available_count": len(items),
        "source": "final.l0",
    }


def build_graph_text(record: Dict[str, Any], max_items: int, max_item_chars: int, max_total_chars: int) -> Tuple[str, List[Dict[str, Any]]]:
    items = ((record.get("final") or {}).get("graph") or [])
    if not isinstance(items, list):
        items = []
    included_items = [item for item in items if isinstance(item, dict)][:max_items]
    lines = [f"Retrieved {len(items)} relevant knowledge graph units:", ""] if items else []
    for index, item in enumerate(included_items, 1):
        content = compact_text(item_content(item), max_item_chars)
        lines.append(f"Graph Result {index}: {content}")
    text = "\n\n".join(lines) if lines else "No relevant entities or relationships found in the knowledge graph."
    return compact_text(text, max_total_chars), included_items


def build_episodic_text(record: Dict[str, Any], max_items: int, max_item_chars: int, max_total_chars: int) -> Tuple[str, List[Dict[str, Any]]]:
    final = record.get("final") or {}
    items = final.get("episodic") or []
    if not isinstance(items, list):
        items = []
    context = final.get("episodic_context_with_time")
    if context:
        return compact_text(context, max_total_chars), items
    lines = []
    for index, item in enumerate(items[:max_items], 1):
        time_text = item.get("time") or "unknown"
        content = compact_text(item_content(item), max_item_chars)
        lines.append(f"Episodic Result {index}: [Time: {time_text}] {content}")
    text = "\n".join(lines) if lines else ""
    return compact_text(text, max_total_chars), items


def build_full_prompt(
    question: str,
    category: int,
    hierarchical_context: Dict[str, Any],
    hierarchical_text: str,
    graph_units: List[Dict[str, Any]],
    graph_text: str,
    episodic_context_with_time: str,
) -> Tuple[str, Dict[str, int]]:
    system_prompt_parts = []
    if category == 5:
        system_prompt_parts.append("You are an expert conversation analyst specialized in detecting misleading or unanswerable questions.")
    else:
        system_prompt_parts.append("You are an expert conversation analyst with access to THREE complementary information retrieval systems.")
    system_prompt_parts.append("")
    system_prompt_parts.append("IMPORTANT: These are THREE DIFFERENT retrieval systems providing COMPLEMENTARY information:")
    system_prompt_parts.append("1. HIERARCHICAL MEMORY: Provides structured, multi-layer conversational context (summaries, insights)")
    system_prompt_parts.append("2. KNOWLEDGE GRAPH: Provides specific facts and entity relationships")
    system_prompt_parts.append("3. EPISODIC MEMORY: Provides time-stamped factual events with [Time: ...] markers")
    system_prompt_parts.append("")
    system_prompt_parts.append("Your task is to synthesize information from ALL THREE systems to provide the most accurate and complete answer.")
    system_prompt_text = "\n".join(system_prompt_parts)

    prompt_parts = [system_prompt_text, ""]
    prompt_parts.append(f"QUESTION: {question}")
    prompt_parts.append(f"QUESTION CATEGORY: {category} - {category_guidance(category)}")
    prompt_parts.append("")

    hierarchical_enabled = hierarchical_context.get("hierarchical_enabled", False)
    prompt_parts.append("=" * 80)
    prompt_parts.append("TOWER 1: HIERARCHICAL MEMORY RESULTS")
    prompt_parts.append("=" * 80)
    if hierarchical_enabled:
        prompt_parts.append(hierarchical_text)
    else:
        prompt_parts.append("Hierarchical retrieval was not available for this query.")
    prompt_parts.append("")

    prompt_parts.append("=" * 80)
    prompt_parts.append("TOWER 2: KNOWLEDGE GRAPH RESULTS")
    prompt_parts.append("=" * 80)
    if graph_units:
        prompt_parts.append(graph_text)
    else:
        prompt_parts.append("No relevant entities or relationships found in the knowledge graph.")
    prompt_parts.append("")

    prompt_parts.append("=" * 80)
    prompt_parts.append("TOWER 3: EPISODIC MEMORY RESULTS (with Time Markers)")
    prompt_parts.append("=" * 80)
    if episodic_context_with_time:
        prompt_parts.append("Time-stamped facts from episodic memory:")
        prompt_parts.append("NOTE: Each fact has a [Time: ...] marker indicating when the event occurred.")
        prompt_parts.append("")
        prompt_parts.append(episodic_context_with_time)
    else:
        prompt_parts.append("No relevant episodic memories found.")
    prompt_parts.append("")

    prompt_parts.append("=" * 80)
    prompt_parts.append("TRI-TOWER FUSION GUIDANCE")
    prompt_parts.append("=" * 80)
    if category == 5:
        prompt_parts.append("SYNTHESIS INSTRUCTIONS (ADVERSARIAL):")
        prompt_parts.append("1. Cross-validate information across ALL THREE towers")
        prompt_parts.append("2. Treat the known answer implied by the question as potentially misleading unless the context directly supports it")
        prompt_parts.append("3. Only provide a concrete answer when the exact subject, event, and relationship in the question are explicitly supported")
        prompt_parts.append("4. If evidence belongs to a different person, date, event, or relationship, state exactly 'No information available'")
        prompt_parts.append("5. If information conflicts or is not found in any tower, state exactly 'No information available'")
        prompt_parts.append("6. Be especially careful with [Time: ...] markers for temporal verification")
        prompt_parts.append("7. DO NOT infer, transfer facts between people, or fabricate information")
    elif category == 2:
        prompt_parts.append("SYNTHESIS INSTRUCTIONS (TEMPORAL):")
        prompt_parts.append("1. PRIORITIZE Episodic Memory [Time: ...] markers for temporal questions")
        prompt_parts.append("2. Cross-reference with Hierarchical and Knowledge Graph for context")
        prompt_parts.append("3. Extract specific dates/times from [Time: ...] markers")
    else:
        prompt_parts.append("SYNTHESIS INSTRUCTIONS:")
        prompt_parts.append("1. If ALL THREE towers provided information: Cross-validate and synthesize")
        prompt_parts.append("2. If only SOME towers worked: Prioritize based on question type")
        prompt_parts.append("3. Use [Time: ...] markers from Episodic Memory for temporal accuracy")

    prompt_parts.append("")
    prompt_parts.append("RESPONSE FORMAT (REQUIRED JSON):")
    prompt_parts.append("{")
    prompt_parts.append('    "reasoning": "Your synthesis process across all three towers...",')
    prompt_parts.append('    "final_answer": "Your direct, concise final answer"')
    prompt_parts.append("}")

    full_prompt = "\n".join(prompt_parts)
    return full_prompt, {
        "system_prompt_tokens": estimate_tokens(system_prompt_text),
        "total_input_tokens": estimate_tokens(full_prompt),
    }


def post_process_answer(answer: Any, category: int) -> str:
    if isinstance(answer, list):
        answer = ", ".join(str(item) for item in answer)
    elif not isinstance(answer, str):
        answer = str(answer) if answer is not None else ""
    answer = answer.strip()
    if not answer:
        return "No answer generated"
    for prefix in ("Answer:", "ANSWER:", "Final Answer:", "Response:"):
        if answer.startswith(prefix):
            answer = answer[len(prefix):].strip()
    if answer and not answer[0].isupper() and not answer[0].isdigit():
        answer = answer[0].upper() + answer[1:]
    if category == 5:
        lower_answer = answer.lower()
        if any(phrase in lower_answer for phrase in ("no information", "not available", "not mentioned", "not found", "insufficient information")):
            return "No information available"
    return answer


def parse_text_response(raw_response: str, category: int) -> Dict[str, str]:
    lines = raw_response.strip().split("\n")
    reasoning = ""
    final_answer = ""
    current_section = "reasoning"
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if any(keyword in line.lower() for keyword in ("answer", "final", "conclusion")):
            current_section = "answer"
            if ":" in line:
                final_answer = line.split(":", 1)[1].strip()
                continue
        if current_section == "reasoning":
            reasoning += line + " "
        else:
            final_answer += line + " "
    if not final_answer.strip():
        final_answer = raw_response.strip()
        reasoning = "Unable to parse structured reasoning"
    return {
        "reasoning": reasoning.strip() or "No clear reasoning provided",
        "final_answer": post_process_answer(final_answer.strip(), category),
    }


def parse_structured_response(raw_response: str, category: int) -> Dict[str, str]:
    try:
        parsed = json.loads(raw_response.strip())
        if isinstance(parsed, dict) and "reasoning" in parsed and "final_answer" in parsed:
            reasoning = parsed["reasoning"]
            if isinstance(reasoning, list):
                reasoning = ", ".join(str(item) for item in reasoning)
            elif not isinstance(reasoning, str):
                reasoning = str(reasoning)
            return {
                "reasoning": reasoning.strip(),
                "final_answer": post_process_answer(parsed["final_answer"], category),
            }
        raise ValueError("JSON response is missing required fields")
    except Exception:
        return parse_text_response(raw_response, category)


def calculate_confidence(hierarchical_enabled: bool, graph_count: int, episodic_count: int) -> float:
    score = 0.35
    if hierarchical_enabled:
        score += 0.20
    score += min(graph_count, 5) * 0.04
    score += min(episodic_count, 5) * 0.04
    return round(min(score, 0.95), 4)


def find_retrieval_files(retrieval_dir: Path, sample_ids: Optional[Sequence[str]]) -> List[Path]:
    files: List[Path] = []
    if sample_ids:
        for sample_id in sample_ids:
            sample_dir = retrieval_dir / sample_id
            for name in ("retrieval_results.json", "retrieval_results.jsonl"):
                candidate = sample_dir / name
                if candidate.exists():
                    files.append(candidate)
                    break
    else:
        files.extend(sorted(retrieval_dir.glob("*/retrieval_results.json")))
        files.extend(sorted(retrieval_dir.glob("*/retrieval_results.jsonl")))
        for name in ("retrieval_results.json", "retrieval_results.jsonl"):
            candidate = retrieval_dir / name
            if candidate.exists():
                files.append(candidate)
    return files


def load_retrieval_records(path: Path) -> List[Dict[str, Any]]:
    if path.suffix == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    data = read_json(path)
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        if isinstance(data.get("results"), list):
            return [item for item in data["results"] if isinstance(item, dict)]
        if isinstance(data.get("samples"), list):
            records = []
            for sample in data["samples"]:
                if isinstance(sample, dict) and isinstance(sample.get("results"), list):
                    records.extend(item for item in sample["results"] if isinstance(item, dict))
            return records
    return []


def filter_records(records: List[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    filtered = records
    if args.question_ids:
        wanted = set(args.question_ids)
        filtered = [record for record in filtered if str(record.get("question_id")) in wanted]
    if args.categories:
        wanted_categories = {int(category) for category in args.categories}
        filtered = [record for record in filtered if int(record.get("category", 0) or 0) in wanted_categories]
    if args.only_successful_retrieval:
        filtered = [record for record in filtered if record.get("success", True)]
    if args.max_questions is not None:
        filtered = filtered[: args.max_questions]
    return filtered


def generate_for_record(record: Dict[str, Any], llm_client: Optional[LLMClient], args: argparse.Namespace) -> Dict[str, Any]:
    sample_id = str(record.get("sample_id") or "unknown")
    question_id = str(record.get("question_id") or f"q_{int(time.time() * 1000)}")
    question = str(record.get("question") or "")
    category = int(record.get("category", 0) or 0)

    hierarchical_text, hierarchical_context = build_hierarchical_text(
        record,
        max_items=args.max_hierarchical_items,
        max_item_chars=args.max_item_chars,
        max_total_chars=args.max_context_chars,
    )
    graph_text, graph_units = build_graph_text(
        record,
        max_items=args.max_graph_items,
        max_item_chars=args.max_item_chars,
        max_total_chars=args.max_context_chars,
    )
    episodic_text, episodic_units = build_episodic_text(
        record,
        max_items=args.max_episodic_items,
        max_item_chars=args.max_item_chars,
        max_total_chars=args.max_context_chars,
    )
    prompt, token_info = build_full_prompt(
        question=question,
        category=category,
        hierarchical_context=hierarchical_context,
        hierarchical_text=hierarchical_text,
        graph_units=graph_units,
        graph_text=graph_text,
        episodic_context_with_time=episodic_text,
    )

    raw_response = ""
    answer = {"reasoning": "Dry run: prompt assembled but LLM was not called", "final_answer": ""}
    generation_error = None
    elapsed = 0.0
    if not args.dry_run:
        if llm_client is None:
            raise RuntimeError("LLM client is required unless --dry-run is set")
        started = time.perf_counter()
        try:
            raw_response = llm_client.generate_answer(
                prompt=prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                json_format=True,
            )
            answer = parse_structured_response(raw_response, category)
        except Exception as exc:
            generation_error = f"{type(exc).__name__}: {exc}"
            answer = {"reasoning": f"Generation failed: {generation_error}", "final_answer": "Unable to generate answer"}
        elapsed = time.perf_counter() - started

    confidence = calculate_confidence(
        bool(hierarchical_context.get("hierarchical_enabled")),
        len(graph_units),
        len(episodic_units),
    )
    output = {
        "success": generation_error is None,
        "sample_id": sample_id,
        "question_id": question_id,
        "question": question,
        "category": category,
        "expected_answer": record.get("expected_answer"),
        "evidence": record.get("evidence"),
        "model": args.model,
        "requested_model": getattr(args, "requested_model", args.model),
        "answer": answer.get("final_answer", ""),
        "reasoning": answer.get("reasoning", ""),
        "confidence": confidence,
        "raw_response": raw_response,
        "generation_error": generation_error,
        "token_info": token_info,
        "context_counts": {
            "hierarchical": hierarchical_context.get("count", 0),
            "graph": len(graph_units),
            "episodic": len(episodic_units),
        },
        "retrieval_final_counts": (record.get("final") or {}).get("counts", {}),
        "generation_time": elapsed,
        "retrieval_time": record.get("total_retrieval_time"),
        "generated_at": datetime.now().isoformat(),
    }
    if args.save_prompts or args.dry_run:
        output["prompt"] = prompt
    return output


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate answers from benchmark_self_host/locomo10 retrieval results.")
    parser.add_argument("--retrieval-dir", default=str(SCRIPT_DIR / "retrieve"))
    parser.add_argument("--output-dir", default=str(SCRIPT_DIR / "generate"))
    parser.add_argument("--sample-ids", nargs="+", help="Only process these sample IDs")
    parser.add_argument("--question-ids", nargs="+", help="Only process these question IDs")
    parser.add_argument("--categories", nargs="+", type=int, help="Only process these category IDs")
    parser.add_argument("--max-questions", type=int, help="Maximum questions per full run after filtering")
    parser.add_argument("--only-successful-retrieval", action="store_true", help="Skip records marked success=false by retrieve.py")

    parser.add_argument("--model", default="gpt-4.1-mini-closeai", choices=MODEL_CHOICES_WITH_ALIASES, help="LLM model name")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--max-context-ratio", type=float, default=0.85)
    parser.add_argument("--dry-run", action="store_true", help="Assemble prompts and save outputs without calling the LLM")
    parser.add_argument("--save-prompts", action="store_true", help="Persist full prompts in generation_results.json")

    parser.add_argument("--max-hierarchical-items", type=int, default=14)
    parser.add_argument("--max-graph-items", type=int, default=10)
    parser.add_argument("--max-episodic-items", type=int, default=20)
    parser.add_argument("--max-item-chars", type=int, default=700)
    parser.add_argument("--max-context-chars", type=int, default=18000)
    parser.add_argument("--debug", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    configure_logging(args.debug)

    retrieval_dir = Path(args.retrieval_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    args.requested_model = args.model
    args.model = resolve_model_name(args.model)
    retrieval_files = find_retrieval_files(retrieval_dir, args.sample_ids)
    if not retrieval_files:
        LOGGER.error("No retrieval result files found under %s", retrieval_dir)
        return 1

    LOGGER.info("Found %d retrieval result files", len(retrieval_files))
    llm_client = None if args.dry_run else LLMClient(model_name=args.model, max_context_ratio=args.max_context_ratio)

    all_outputs: List[Dict[str, Any]] = []
    summary = {
        "retrieval_dir": str(retrieval_dir),
        "output_dir": str(output_dir),
        "model": args.model,
        "requested_model": getattr(args, "requested_model", args.model),
        "dry_run": args.dry_run,
        "started_at": datetime.now().isoformat(),
        "samples": [],
    }

    for retrieval_file in retrieval_files:
        records = filter_records(load_retrieval_records(retrieval_file), args)
        if not records:
            continue
        sample_id = str(records[0].get("sample_id") or retrieval_file.parent.name)
        LOGGER.info("Generating %d answers for %s", len(records), sample_id)
        sample_outputs = [generate_for_record(record, llm_client, args) for record in records]
        sample_dir = output_dir / sample_id
        write_json(sample_dir / "generation_results.json", sample_outputs)
        append_jsonl(sample_dir / "generation_results.jsonl", sample_outputs)
        sample_summary = {
            "sample_id": sample_id,
            "retrieval_file": str(retrieval_file),
            "output_file": str(sample_dir / "generation_results.json"),
            "questions": len(sample_outputs),
            "success_count": sum(1 for item in sample_outputs if item.get("success")),
            "failed_count": sum(1 for item in sample_outputs if not item.get("success")),
        }
        summary["samples"].append(sample_summary)
        all_outputs.extend(sample_outputs)
        write_json(output_dir / "generate_summary.json", summary)

    summary["finished_at"] = datetime.now().isoformat()
    summary["total_questions"] = len(all_outputs)
    summary["success_count"] = sum(1 for item in all_outputs if item.get("success"))
    summary["failed_count"] = sum(1 for item in all_outputs if not item.get("success"))
    write_json(output_dir / "generate_summary.json", summary)

    if summary["failed_count"]:
        LOGGER.warning("Generation finished with %d failed questions", summary["failed_count"])
    else:
        LOGGER.info("Generation finished successfully: %d questions", summary["total_questions"])
    return 0 if all_outputs else 1


if __name__ == "__main__":
    raise SystemExit(main())
