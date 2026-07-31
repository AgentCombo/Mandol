#!/usr/bin/env python3
"""Score LongMemEval self-host generation outputs with LLM judge."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


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
from benchmark_self_host.longmemeval.evaluation import (  # noqa: E402
	calculate_comprehensive_scores,
	convert_numpy_types,
)


LOGGER = logging.getLogger("longmemeval_score")
JUDGE_MODEL_CHOICES = ["gpt-4o-mini-closeai", "gpt-4o-mini-openrouter"]
MODEL_ALIASES = {"gpt-4o-mini": "gpt-4o-mini-closeai"}
MODEL_CHOICES_WITH_ALIASES = JUDGE_MODEL_CHOICES + sorted(MODEL_ALIASES)
DEFAULT_METRICS = [
	"exact_match",
	"f1",
	"rouge",
	"bleu",
	"meteor",
	"semantic_similarity",
	"bert_f1",
	"llm_judge",
]


def configure_logging(debug: bool = False) -> None:
	level = logging.DEBUG if debug else logging.INFO
	if auto_configure_logging() is None:
		setup_logging(level=level)
	logging.getLogger().setLevel(level)
	LOGGER.setLevel(level)


def resolve_model_name(model_name: str) -> str:
	if model_name in MODEL_CONFIGS:
		return model_name
	alias = MODEL_ALIASES.get(model_name)
	if alias and alias in MODEL_CONFIGS:
		LOGGER.info("Resolved model alias %s -> %s", model_name, alias)
		return alias
	return model_name


def read_json(path: Path) -> Any:
	with path.open("r", encoding="utf-8") as file:
		return json.load(file)


def write_json(path: Path, data: Any) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as file:
		json.dump(convert_numpy_types(data), file, ensure_ascii=False, indent=2)


def append_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as file:
		for record in records:
			file.write(json.dumps(convert_numpy_types(record), ensure_ascii=False) + "\n")


def load_generation_records(path: Path) -> List[Dict[str, Any]]:
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


def find_generation_files(generation_dir: Path, sample_ids: Optional[Sequence[str]]) -> List[Path]:
	files: List[Path] = []
	if sample_ids:
		for sample_id in sample_ids:
			sample_dir = generation_dir / sample_id
			for name in ("generation_results.json", "generation_results.jsonl"):
				candidate = sample_dir / name
				if candidate.exists():
					files.append(candidate)
					break
	else:
		files.extend(sorted(generation_dir.glob("*/generation_results.json")))
		files.extend(sorted(generation_dir.glob("*/generation_results.jsonl")))
		for name in ("generation_results.json", "generation_results.jsonl"):
			candidate = generation_dir / name
			if candidate.exists():
				files.append(candidate)
	return files


def filter_records(records: List[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
	filtered = records
	if args.question_ids:
		wanted = set(args.question_ids)
		filtered = [record for record in filtered if str(record.get("question_id")) in wanted]
	if args.question_types:
		wanted_types = set(args.question_types)
		filtered = [record for record in filtered if str(record.get("question_type")) in wanted_types]
	if args.only_successful_generation:
		filtered = [record for record in filtered if record.get("success", True) and not record.get("generation_error")]
	if args.max_questions is not None:
		filtered = filtered[: args.max_questions]
	return filtered


def pick_answer(record: Dict[str, Any]) -> str:
	for key in ("answer", "generated_answer", "final_answer"):
		value = record.get(key)
		if value:
			return str(value)
	return ""


def score_record(record: Dict[str, Any], llm_client: Optional[LLMClient], args: argparse.Namespace) -> Dict[str, Any]:
	sample_id = str(record.get("sample_id") or "unknown")
	qa_index = record.get("qa_index")
	question_id = str(record.get("question_id") or "")
	question = str(record.get("question") or "")
	question_type = str(record.get("question_type") or "default")
	expected_answer = str(record.get("expected_answer") or record.get("ground_truth") or "")
	answer = pick_answer(record)
	context = str(record.get("prompt") or "") if args.use_prompt_context else str(record.get("reasoning") or "")
	started = time.perf_counter()

	result: Dict[str, Any] = {
		"success": False,
		"sample_id": sample_id,
		"qa_index": qa_index,
		"question_id": question_id,
		"question": question,
		"question_type": question_type,
		"query_date": record.get("query_date"),
		"expected_answer": expected_answer,
		"ground_truth": expected_answer,
		"answer": answer,
		"generated_answer": answer,
		"reasoning": record.get("reasoning"),
		"generation_success": bool(record.get("success", True) and not record.get("generation_error")),
		"generation_model": record.get("model"),
		"generation_time": float(record.get("generation_time", 0.0) or 0.0),
		"retrieval_time": float(record.get("retrieval_time", 0.0) or 0.0),
		"context_counts": record.get("context_counts", {}),
		"judge_model": args.model,
		"requested_judge_model": getattr(args, "requested_model", args.model),
		"evaluation_method": "llm_judge",
		"metrics": args.metrics,
		"LLM_judge": False,
		"llm_accuracy": 0.0,
		"evaluation_scores": {},
		"llm_details": {},
		"evaluation_error": None,
	}

	if args.dry_run:
		result.update({"success": True, "evaluation_method": "dry_run", "evaluation_time": 0.0})
		return result

	if llm_client is None:
		raise RuntimeError("LLM client is required unless --dry-run is set")

	try:
		evaluation = calculate_comprehensive_scores(
			gold_answer=expected_answer,
			response=answer,
			question=question,
			context=context,
			question_type=question_type,
			llm_client=llm_client,
			metrics=args.metrics,
		)
		scores = evaluation.get("scores", {})
		llm_accuracy = float(scores.get("llm_accuracy", 0.0) or 0.0)
		result.update(
			{
				"success": bool(evaluation.get("evaluation_success", False)),
				"LLM_judge": llm_accuracy >= args.correct_threshold,
				"llm_accuracy": llm_accuracy,
				"evaluation_scores": scores,
				"llm_details": evaluation.get("llm_details", {}),
				"input_info": evaluation.get("input_info", {}),
			}
		)
	except Exception as exc:
		result["evaluation_error"] = f"{type(exc).__name__}: {exc}"
		LOGGER.error("Scoring failed for %s/%s: %s", sample_id, question_id, result["evaluation_error"])

	result["evaluation_time"] = time.perf_counter() - started
	return result


def score_records(records: List[Dict[str, Any]], llm_client: Optional[LLMClient], args: argparse.Namespace) -> List[Dict[str, Any]]:
	if args.workers <= 1 or len(records) <= 1:
		return [score_record(record, llm_client, args) for record in records]

	scored: List[Optional[Dict[str, Any]]] = [None] * len(records)
	with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="LongMemScore") as executor:
		future_to_index = {executor.submit(score_record, record, llm_client, args): idx for idx, record in enumerate(records)}
		for future in as_completed(future_to_index):
			idx = future_to_index[future]
			scored[idx] = future.result()
	return [record for record in scored if record is not None]


def mean(values: Sequence[float]) -> float:
	return sum(values) / len(values) if values else 0.0


def std(values: Sequence[float]) -> float:
	if not values:
		return 0.0
	average = mean(values)
	return (sum((value - average) ** 2 for value in values) / len(values)) ** 0.5


def summarize_records(sample_id: str, records: List[Dict[str, Any]]) -> Dict[str, Any]:
	evaluated = [record for record in records if record.get("success")]
	correct = [record for record in evaluated if record.get("LLM_judge")]
	by_question_type: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"total": 0, "evaluated": 0, "correct": 0, "wrong": 0})
	f1_scores = [float(record.get("evaluation_scores", {}).get("token_f1", 0.0) or 0.0) for record in evaluated]
	semantic_scores = [float(record.get("evaluation_scores", {}).get("semantic_similarity", 0.0) or 0.0) for record in evaluated]
	llm_scores = [float(record.get("llm_accuracy", 0.0) or 0.0) for record in evaluated]
	evaluation_times = [float(record.get("evaluation_time", 0.0) or 0.0) for record in evaluated]
	generation_times = [float(record.get("generation_time", 0.0) or 0.0) for record in evaluated if record.get("generation_time") is not None]
	retrieval_times = [float(record.get("retrieval_time", 0.0) or 0.0) for record in evaluated if record.get("retrieval_time") is not None]

	for record in records:
		question_type = str(record.get("question_type") or "default")
		by_question_type[question_type]["total"] += 1
		if record.get("success"):
			by_question_type[question_type]["evaluated"] += 1
			if record.get("LLM_judge"):
				by_question_type[question_type]["correct"] += 1
			else:
				by_question_type[question_type]["wrong"] += 1

	for stats in by_question_type.values():
		stats["LLM_judge_score"] = stats["correct"] / stats["evaluated"] if stats["evaluated"] else 0.0

	return {
		"sample_id": sample_id,
		"total_questions": len(records),
		"evaluated_count": len(evaluated),
		"correct_count": len(correct),
		"wrong_count": len(evaluated) - len(correct),
		"failed_count": len(records) - len(evaluated),
		"LLM_judge_score": len(correct) / len(evaluated) if evaluated else 0.0,
		"avg_llm_accuracy": mean(llm_scores),
		"performance_metrics": {
			"avg_f1_score": mean(f1_scores),
			"std_f1_score": std(f1_scores),
			"avg_semantic_similarity": mean(semantic_scores),
			"avg_llm_accuracy": mean(llm_scores),
			"LLM_judge_score": len(correct) / len(evaluated) if evaluated else 0.0,
		},
		"timing_metrics": {
			"avg_evaluation_time": mean(evaluation_times),
			"avg_generation_time": mean(generation_times),
			"avg_retrieval_time": mean(retrieval_times),
		},
		"avg_evaluation_time": mean(evaluation_times),
		"by_question_type": dict(sorted(by_question_type.items(), key=lambda item: item[0])),
	}


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Score benchmark_self_host/longmemeval generation outputs with LLM judge.")
	parser.add_argument("--generation-dir", default=str(SCRIPT_DIR / "retrieve"), help="Directory containing qa_x/generation_results.json")
	parser.add_argument("--output-dir", default=str(SCRIPT_DIR / "score"))
	parser.add_argument("--sample-ids", nargs="+", help="Only score these sample IDs")
	parser.add_argument("--question-ids", nargs="+", help="Only score these question IDs")
	parser.add_argument("--question-types", nargs="+", help="Only score these LongMemEval question types")
	parser.add_argument("--max-questions", type=int, help="Maximum questions per sample after filtering")
	parser.add_argument("--only-successful-generation", action="store_true", help="Skip records where generation failed")
	parser.add_argument("--use-prompt-context", action="store_true", help="Pass saved generation prompt as judge context; default uses reasoning")
	parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS, help="Evaluation metrics; include llm_judge for LLM grading")
	parser.add_argument("--model", default="gpt-4o-mini-closeai", choices=MODEL_CHOICES_WITH_ALIASES, help="Judge model name")
	parser.add_argument("--max-context-ratio", type=float, default=0.85)
	parser.add_argument("--workers", type=int, default=1, help="Concurrent judge calls")
	parser.add_argument("--correct-threshold", type=float, default=0.5)
	parser.add_argument("--dry-run", action="store_true", help="Load and write score files without calling the judge LLM")
	parser.add_argument("--debug", action="store_true")
	return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
	parser = build_arg_parser()
	args = parser.parse_args(argv)
	configure_logging(args.debug)
	args.requested_model = args.model
	args.model = resolve_model_name(args.model)

	generation_dir = Path(args.generation_dir).resolve()
	output_dir = Path(args.output_dir).resolve()
	generation_files = find_generation_files(generation_dir, args.sample_ids)
	if not generation_files:
		LOGGER.error("No generation result files found under %s", generation_dir)
		return 1

	LOGGER.info("Found %d generation files", len(generation_files))
	llm_client = None if args.dry_run else LLMClient(model_name=args.model, max_context_ratio=args.max_context_ratio)
	all_scored: List[Dict[str, Any]] = []
	summary = {
		"generation_dir": str(generation_dir),
		"output_dir": str(output_dir),
		"model": args.model,
		"requested_model": getattr(args, "requested_model", args.model),
		"evaluation_method": "llm_judge",
		"metrics": args.metrics,
		"dry_run": args.dry_run,
		"started_at": datetime.now().isoformat(),
		"samples": [],
	}

	for generation_file in generation_files:
		records = filter_records(load_generation_records(generation_file), args)
		if not records:
			continue
		sample_id = str(records[0].get("sample_id") or generation_file.parent.name)
		LOGGER.info("Scoring %d answers for %s", len(records), sample_id)
		scored = score_records(records, llm_client, args)
		sample_dir = output_dir / sample_id
		write_json(sample_dir / "score_results.json", scored)
		append_jsonl(sample_dir / "score_results.jsonl", scored)
		sample_summary = summarize_records(sample_id, scored)
		sample_summary.update(
			{
				"generation_file": str(generation_file),
				"score_file": str(sample_dir / "score_results.json"),
			}
		)
		write_json(sample_dir / "score_summary.json", sample_summary)
		summary["samples"].append(sample_summary)
		all_scored.extend(scored)
		write_json(output_dir / "score_summary.json", summary)

	global_summary = summarize_records("ALL", all_scored)
	summary.update(
		{
			"finished_at": datetime.now().isoformat(),
			"total_questions": global_summary["total_questions"],
			"evaluated_count": global_summary["evaluated_count"],
			"correct_count": global_summary["correct_count"],
			"wrong_count": global_summary["wrong_count"],
			"failed_count": global_summary["failed_count"],
			"LLM_judge_score": global_summary["LLM_judge_score"],
			"avg_llm_accuracy": global_summary["avg_llm_accuracy"],
			"performance_metrics": global_summary["performance_metrics"],
			"timing_metrics": global_summary["timing_metrics"],
			"by_question_type": global_summary["by_question_type"],
		}
	)
	write_json(output_dir / "score_summary.json", summary)
	append_jsonl(output_dir / "score_results.jsonl", all_scored)

	if not all_scored:
		LOGGER.error("No generation records were scored")
		return 1
	LOGGER.info(
		"Scoring finished: %d/%d correct (LLM_judge_score=%.4f)",
		summary["correct_count"],
		summary["evaluated_count"],
		summary["LLM_judge_score"],
	)
	return 0 if summary["failed_count"] == 0 else 1


if __name__ == "__main__":
	raise SystemExit(main())
