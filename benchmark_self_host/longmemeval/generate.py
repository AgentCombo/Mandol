#!/usr/bin/env python3
"""Generate LongMemEval answers from self-host retrieval results."""

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


LOGGER = logging.getLogger("longmemeval_generate")

GENERATION_MODEL_CHOICES = [
	"gpt-4o-mini-closeai",
	"gpt-4o-mini-openrouter",
	"gpt-4.1-mini-closeai",
	"gpt-4.1-mini-openrouter",
]
MODEL_ALIASES = {
	"gpt-4o-mini": "gpt-4o-mini-closeai",
	"gpt-4.1-mini": "gpt-4.1-mini-closeai",
}
MODEL_CHOICES_WITH_ALIASES = GENERATION_MODEL_CHOICES + sorted(MODEL_ALIASES)

STABLE_CATEGORIES = {
	"USER_ATTRIBUTE",
	"PREFERENCE_HABIT",
	"RELATIONSHIP_FACT",
	"KNOWLEDGE",
	"IMPLICIT_CONSTRAINT",
	"INVENTORY_ITEM",
}


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


def resolve_model_name(model_name: str) -> str:
	if model_name in MODEL_CONFIGS:
		return model_name
	alias = MODEL_ALIASES.get(model_name)
	if alias and alias in MODEL_CONFIGS:
		LOGGER.info("Resolved model alias %s -> %s", model_name, alias)
		return alias
	return model_name


def estimate_tokens(text: str) -> int:
	if not text:
		return 0
	return max(1, len(text) // 4)


def compact_text(text: Any, max_chars: int) -> str:
	value = str(text or "").strip()
	if max_chars <= 0 or len(value) <= max_chars:
		return value
	return value[: max_chars - 20].rstrip() + "\n...[truncated]"


def item_raw(item: Dict[str, Any]) -> Dict[str, Any]:
	raw = item.get("raw_data")
	return raw if isinstance(raw, dict) else {}


def item_metadata(item: Dict[str, Any]) -> Dict[str, Any]:
	metadata = item.get("metadata")
	return metadata if isinstance(metadata, dict) else {}


def item_content(item: Dict[str, Any]) -> str:
	if not isinstance(item, dict):
		return str(item)
	for key in ("content", "text_content"):
		value = item.get(key)
		if value:
			return str(value)
	raw = item_raw(item)
	for key in ("text_content", "content", "description", "message", "original_content"):
		value = raw.get(key)
		if value:
			return str(value)
	return json.dumps(item, ensure_ascii=False)


def format_sentence_context(items: List[Dict[str, Any]], max_item_chars: int) -> str:
	if not items:
		return ""
	parts = []
	for index, item in enumerate(items, 1):
		raw = item_raw(item)
		metadata = item_metadata(item)
		content = compact_text(raw.get("text_content") or item_content(item), max_item_chars)
		role = metadata.get("role", raw.get("role", item.get("role", "unknown")))
		session_date = metadata.get("session_date", raw.get("session_date", item.get("session_date", "unknown")))
		parts.append(f"[Message {index}] (Date: {session_date}, Speaker: {role})\nContent: {content}")
	return "\n\n".join(parts)


def format_episodic_context(items: List[Dict[str, Any]], max_item_chars: int) -> str:
	if not items:
		return ""
	parts = []
	for index, item in enumerate(items, 1):
		raw = item_raw(item)
		content = compact_text(raw.get("content", raw.get("text_content", item_content(item))), max_item_chars)
		category = str(raw.get("category", raw.get("fact_type", raw.get("node_type", item.get("category", "EVENT"))))).upper()
		event_date = raw.get("event_date") or raw.get("temporal_val") or raw.get("time") or raw.get("temporal_info") or item.get("time") or "Unknown Date"
		is_stable = raw.get("is_stable")
		if is_stable is None:
			is_stable = category in STABLE_CATEGORIES
		stability = "Stable" if is_stable else "Dynamic"
		parts.append(f"[Fact {index}] Type: {category} | Time: {event_date} | Stability: {stability}\nContent: {content}")
	return "\n\n".join(parts)


def format_entity_context(items: List[Dict[str, Any]], max_item_chars: int) -> str:
	if not items:
		return ""
	parts = []
	for index, item in enumerate(items, 1):
		raw = item_raw(item)
		main_content = raw.get("text_content") or item.get("text_content")
		if not main_content:
			name = raw.get("entity_canonical") or raw.get("entity_text") or item.get("entity_name") or item.get("uid", "unknown")
			entity_type = raw.get("entity_category") or raw.get("entity_type") or item.get("entity_type") or "Unknown"
			description = raw.get("content") or raw.get("description") or item_content(item) or "No description"
			main_content = f"Entity: {name} (Type: {entity_type}) | Context: {description}"
		main_content = compact_text(main_content, max_item_chars)
		session_date = raw.get("session_date") or raw.get("date") or raw.get("created_at") or item.get("session_date")
		date_text = f" [Date: {session_date}]" if session_date else ""
		parts.append(f"[{index}] {main_content}{date_text}")
	return "\n\n".join(parts)


def build_fused_context(sentence_context: str, episodic_context: str, entity_context: str) -> str:
	sections = []
	if sentence_context:
		sections.append(f"<conversation_history>\n{sentence_context}\n</conversation_history>")
	if episodic_context:
		sections.append(f"<episodic_facts>\n{episodic_context}\n</episodic_facts>")
	if entity_context:
		sections.append(f"<entity_knowledge>\n{entity_context}\n</entity_knowledge>")
	return "\n\n".join(sections) if sections else "[No context available]"


def context_from_record(record: Dict[str, Any], args: argparse.Namespace) -> Tuple[str, Dict[str, Any]]:
	final = record.get("final") if isinstance(record.get("final"), dict) else {}
	if not args.rebuild_context and final.get("fused_context"):
		fused_context = compact_text(final.get("fused_context"), args.max_context_chars)
		counts = final.get("counts") if isinstance(final.get("counts"), dict) else {}
		return fused_context, {
			"sentence": int(counts.get("sentence", 0) or 0),
			"episodic": int(counts.get("episodic", 0) or 0),
			"entity": int(counts.get("entity", 0) or 0),
		}

	sentence_items = final.get("sentence") if isinstance(final.get("sentence"), list) else []
	episodic_items = final.get("episodic") if isinstance(final.get("episodic"), list) else []
	entity_items = final.get("entity") if isinstance(final.get("entity"), list) else []
	sentence_context = format_sentence_context(sentence_items[: args.max_sentence_items], args.max_item_chars)
	episodic_context = format_episodic_context(episodic_items[: args.max_episodic_items], args.max_item_chars)
	entity_context = format_entity_context(entity_items[: args.max_entity_items], args.max_item_chars)
	fused_context = build_fused_context(sentence_context, episodic_context, entity_context)
	return compact_text(fused_context, args.max_context_chars), {
		"sentence": len(sentence_items),
		"episodic": len(episodic_items),
		"entity": len(entity_items),
	}


def build_prompt(question: str, fused_context: str, query_date: str) -> Tuple[str, Dict[str, int]]:
	if fused_context == "[No context available]":
		return fused_context, {"context_tokens": 0, "prompt_tokens": 0, "total_input_tokens": 0}

	prompt = f"""You are an expert memory-augmented assistant answering questions based on retrieved multi-level context.

        # CURRENT REFERENCE TIME
        The current time for this question is: **{query_date}**
        *** CRITICAL INSTRUCTION ***
        - Treat "{query_date}" as "TODAY" or "NOW".
        - All relative time references ("yesterday", "last week", "3 days ago") MUST be calculated relative to this date.
        - Do NOT use the actual real-world date.

        # RETRIEVED CONTEXT
        You have access to three levels of memory information (organized in XML tags):
        1. **Conversation Memories** (<conversation_history>): Raw message excerpts from past conversations (Highest Priority).
        2. **Episodic Facts** (<episodic_facts>): Structured facts with temporal information.
        3. **Entity Information** (<entity_knowledge>): Entities and their relationships.

        {fused_context}

        -----------------------------------------------------------------

        # QUESTION
        {question}

        # REASONING INSTRUCTIONS
        1. **Cross-reference** information across all three levels when available.
        2. **Prioritize** temporal information when the question involves time ("when", "before", "after").
        3. **Use entity information** (<entity_knowledge>) to understand relationships and attributes.
        4. **Use conversation memories** (<conversation_history>) for direct quotes and specific details.
        5. **Use episodic facts** (<episodic_facts>) for structured event information.
        6. **Conflict Resolution**: If information conflicts across sources, note the discrepancy and prefer more specific/recent data.

        # IMPORTANT NOTES
        - When interpreting timestamps, use the recorded date as the reference point.
        - For "yesterday", "last week" etc., calculate based on the conversation date.
        - If the answer cannot be determined from the context, clearly state so.

        # OUTPUT FORMAT
        Please respond in strict JSON format with two fields:
        {{
            "reasoning": "Your detailed step-by-step reasoning process. Show how you analyzed information from different memory levels. Cite specific facts, messages, or entities that support your answer.",
            "final_answer": "Your direct, concise answer to the question. Be specific and avoid vague references."
        }}
        """
	return prompt, {
		"context_tokens": estimate_tokens(fused_context),
		"prompt_tokens": estimate_tokens(prompt),
		"total_input_tokens": estimate_tokens(prompt),
	}


def normalize_answer_value(value: Any) -> str:
	if isinstance(value, list):
		return ", ".join(str(item) for item in value)
	if value is None:
		return ""
	return str(value).strip()


def parse_response(raw_response: str) -> Dict[str, str]:
	clean_response = raw_response.strip()
	if clean_response.startswith("```json"):
		clean_response = clean_response.replace("```json", "", 1).replace("```", "").strip()
	elif clean_response.startswith("```"):
		clean_response = clean_response.replace("```", "").strip()

	try:
		parsed = json.loads(clean_response)
		if isinstance(parsed, dict):
			return {
				"reasoning": normalize_answer_value(parsed.get("reasoning") or "No reasoning provided"),
				"final_answer": normalize_answer_value(parsed.get("final_answer") or parsed.get("answer") or "Unable to answer"),
			}
	except Exception:
		pass

	reasoning_match = re.search(r'"reasoning"\s*:\s*"(.*?)"\s*,\s*"final_answer"', clean_response, re.DOTALL)
	answer_match = re.search(r'"final_answer"\s*:\s*"(.*?)"\s*}', clean_response, re.DOTALL)
	if reasoning_match and answer_match:
		return {"reasoning": reasoning_match.group(1).strip(), "final_answer": answer_match.group(1).strip()}

	return {
		"reasoning": clean_response[:300] if clean_response else "No reasoning provided",
		"final_answer": clean_response[-100:].strip() if clean_response else "Unable to answer",
	}


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
	if args.question_types:
		wanted_types = set(args.question_types)
		filtered = [record for record in filtered if str(record.get("question_type")) in wanted_types]
	if args.only_successful_retrieval:
		filtered = [record for record in filtered if record.get("success", True)]
	if args.max_questions is not None:
		filtered = filtered[: args.max_questions]
	return filtered


def generate_for_record(record: Dict[str, Any], llm_client: Optional[LLMClient], args: argparse.Namespace) -> Dict[str, Any]:
	sample_id = str(record.get("sample_id") or "unknown")
	qa_index = record.get("qa_index")
	question_id = str(record.get("question_id") or f"q_{qa_index if qa_index is not None else int(time.time() * 1000)}")
	question = str(record.get("question") or "")
	question_type = str(record.get("question_type") or "")
	query_date = str(record.get("query_date") or "Unknown Date")
	fused_context, context_counts = context_from_record(record, args)
	prompt, token_info = build_prompt(question, fused_context, query_date)

	raw_response = ""
	parsed = {"reasoning": "Dry run: prompt assembled but LLM was not called", "final_answer": ""}
	generation_error = None
	elapsed = 0.0
	if not args.dry_run:
		if llm_client is None:
			raise RuntimeError("LLM client is required unless --dry-run is set")
		if fused_context == "[No context available]":
			parsed = {
				"reasoning": "All retrieval channels returned empty results.",
				"final_answer": "No relevant information found.",
			}
		else:
			started = time.perf_counter()
			try:
				raw_response = llm_client.generate_answer(
					prompt=prompt,
					temperature=args.temperature,
					max_tokens=args.max_tokens,
					json_format=True,
				)
				parsed = parse_response(raw_response)
			except Exception as exc:
				generation_error = f"{type(exc).__name__}: {exc}"
				parsed = {"reasoning": f"Generation failed: {generation_error}", "final_answer": "Unable to generate answer"}
			elapsed = time.perf_counter() - started

	answer = parsed.get("final_answer", "")
	output = {
		"success": generation_error is None,
		"sample_id": sample_id,
		"qa_index": qa_index,
		"question_id": question_id,
		"question": question,
		"question_type": question_type,
		"category": record.get("category"),
		"query_date": query_date,
		"expected_answer": record.get("expected_answer") or record.get("ground_truth"),
		"ground_truth": record.get("ground_truth") or record.get("expected_answer"),
		"answer": answer,
		"generated_answer": answer,
		"reasoning": parsed.get("reasoning", ""),
		"raw_response": raw_response,
		"generation_error": generation_error,
		"model": args.model,
		"requested_model": getattr(args, "requested_model", args.model),
		"token_stats": {
			**token_info,
			"completion_tokens": estimate_tokens(raw_response),
			"total_tokens": token_info.get("prompt_tokens", 0) + estimate_tokens(raw_response),
		},
		"context_counts": context_counts,
		"retrieval_details": record.get("retrieval_details", {}),
		"retrieved_contents": record.get("retrieved_contents", {}),
		"generation_time": elapsed,
		"retrieval_time": record.get("total_retrieval_time"),
		"generated_at": datetime.now().isoformat(),
	}
	if args.save_prompts or args.dry_run:
		output["prompt"] = prompt
		output["fused_context"] = fused_context
	return output


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Generate answers from benchmark_self_host/longmemeval retrieval results.")
	parser.add_argument("--retrieval-dir", default=str(SCRIPT_DIR / "retrieve"))
	parser.add_argument("--output-dir", default=str(SCRIPT_DIR / "retrieve"), help="Defaults to retrieve/ so generation sits beside retrieval outputs")
	parser.add_argument("--sample-ids", nargs="+", help="Only process these sample IDs")
	parser.add_argument("--question-ids", nargs="+", help="Only process these question IDs")
	parser.add_argument("--question-types", nargs="+", help="Only process these LongMemEval question types")
	parser.add_argument("--max-questions", type=int, help="Maximum questions after filtering")
	parser.add_argument("--only-successful-retrieval", action="store_true", help="Skip records marked success=false by retrieve.py")

	parser.add_argument("--model", default="gpt-4o-mini-closeai", choices=MODEL_CHOICES_WITH_ALIASES, help="Answer generation model")
	parser.add_argument("--temperature", type=float, default=0.0)
	parser.add_argument("--max-tokens", type=int, default=1000)
	parser.add_argument("--max-context-ratio", type=float, default=0.85)
	parser.add_argument("--dry-run", action="store_true", help="Assemble prompts and save outputs without calling the LLM")
	parser.add_argument("--save-prompts", action="store_true", help="Persist full prompts in generation_results.json")

	parser.add_argument("--rebuild-context", action="store_true", help="Rebuild XML context from final units instead of using retrieval fused_context")
	parser.add_argument("--max-sentence-items", type=int, default=60)
	parser.add_argument("--max-episodic-items", type=int, default=40)
	parser.add_argument("--max-entity-items", type=int, default=40)
	parser.add_argument("--max-item-chars", type=int, default=1800)
	parser.add_argument("--max-context-chars", type=int, default=60000)
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

	if not all_outputs:
		LOGGER.error("No retrieval records were generated")
		return 1
	if summary["failed_count"]:
		LOGGER.warning("Generation finished with %d failed questions", summary["failed_count"])
	else:
		LOGGER.info("Generation finished successfully: %d questions", summary["total_questions"])
	return 0 if summary["failed_count"] == 0 else 1


if __name__ == "__main__":
	raise SystemExit(main())
