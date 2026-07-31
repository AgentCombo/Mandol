#!/usr/bin/env python3
"""LongMemEval Quantification Ablation Runner."""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

try:
	from benchmark_longmemeval.task_eval.evaluation import cleanup_evaluation_models
except Exception:  # pragma: no cover - optional dependency in some environments
	cleanup_evaluation_models = None

from benchmark_longmemeval.task_eval.benchmark_triple_router_quantification import (
 LongMemEvalTripleFusionBenchmark,
)
from mandol.core import paths
from mandol.llm.llm_client import LLMClient
from mandol.memory_router.longmemeval_tower_router import LongMemEvalTowerRouter
from mandol.utils.logging_config import (
 auto_configure_logging,
 create_module_logger,
 setup_logging,
)

if auto_configure_logging() is None:
	setup_logging(level=logging.INFO)
logger = create_module_logger("longmemeval_quantification_ablation")


@dataclass(frozen=True)
class AblationModeConfig:

	mode: str
	router_enabled: bool
	cascade_stage1_enabled: bool
	cascade_stage2_enabled: bool
	cascade_stage3_mmr_enabled: bool
	description: str


ABLATION_ORDER = [
 "full",
 "no_context_generation",
 "no_denoising",
 "no_routing",
]

ABLATION_MODES: Dict[str, AblationModeConfig] = {
 "full": AblationModeConfig(
  mode="full",
  router_enabled=True,
  cascade_stage1_enabled=True,
  cascade_stage2_enabled=True,
  cascade_stage3_mmr_enabled=True,
  description="Full Mandol: routing + denoising + cross-tower disambiguation + MMR context generation.",
 ),
 "no_context_generation": AblationModeConfig(
  mode="no_context_generation",
  router_enabled=True,
  cascade_stage1_enabled=True,
  cascade_stage2_enabled=True,
  cascade_stage3_mmr_enabled=False,
  description="w/o Context Generation: keep denoising/disambiguation, disable Stage-3 MMR, fall back to CE-score greedy hard cut.",
 ),
 "no_denoising": AblationModeConfig(
  mode="no_denoising",
  router_enabled=True,
  cascade_stage1_enabled=False,
  cascade_stage2_enabled=False,
  cascade_stage3_mmr_enabled=False,
  description="w/o Denoising: disable Stage-1 hard filtering, Stage-2 disambiguation, and Stage-3 MMR.",
 ),
 "no_routing": AblationModeConfig(
  mode="no_routing",
  router_enabled=False,
  cascade_stage1_enabled=False,
  cascade_stage2_enabled=False,
  cascade_stage3_mmr_enabled=False,
  description="w/o Routing: fixed tri-tower configuration on top of no_denoising.",
 ),
}


def _json_safe(obj: Any) -> Any:
	"""Convert common non-JSON values into JSON-safe Python objects."""
	if obj is None:
		return None
	if isinstance(obj, Path):
		return str(obj)
	if isinstance(obj, np.generic):
		return _json_safe(obj.item())
	if isinstance(obj, float):
		return obj if math.isfinite(obj) else None
	if isinstance(obj, (str, int, bool)):
		return obj
	if is_dataclass(obj):
		return _json_safe(asdict(obj))
	if isinstance(obj, dict):
		return {str(key): _json_safe(value) for key, value in obj.items()}
	if isinstance(obj, (list, tuple, set)):
		return [_json_safe(value) for value in obj]
	return str(obj)


def _write_json(path: Path, data: Dict[str, Any]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as handle:
		json.dump(_json_safe(data), handle, ensure_ascii=False, indent=2)


def _mean(values: Iterable[float]) -> float:
	vals = [float(value) for value in values]
	return float(np.mean(vals)) if vals else 0.0


def _parse_tower_min_ratio_str(raw: Optional[str]) -> Optional[Dict[str, float]]:
	"""Parse tower min ratio str."""
	if not raw:
		return None
	result: Dict[str, float] = {}
	for pair in raw.split(","):
		pair = pair.strip()
		if not pair:
			continue
		if ":" not in pair:
			raise ValueError(f"Invalid tower ratio pair: {pair!r}")
		key, value = pair.split(":", 1)
		result[key.strip()] = float(value.strip())
	return result


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
	 description="LongMemEval quantification ablation runner (paper-use only)."
	)
	parser.add_argument(
	 "--ablation-mode",
	 choices=ABLATION_ORDER,
	 default="full",
	 help="单个消融模式；配合 --run-all 时会被忽略。",
	)
	parser.add_argument(
	 "--run-all",
	 action="store_true",
	 help="按固定顺序运行 full -> no_context_generation -> no_denoising -> no_routing。",
	)

	# Dataset-specific handling used by the reproduction workflow.
	parser.add_argument("--dataset-size", default="s", choices=["s", "m"], help="数据集大小")
	parser.add_argument("--dataset-dir", default=None, help="原始数据集目录路径")

	parser.add_argument(
	 "--sentence-graph-dir",
	 default=str(paths.LONGMEMEVAL_HIERARCHICAL_STEP3_DIR),
	 help="消息级别图谱目录",
	)
	parser.add_argument(
	 "--episodic-graph-dir",
	 default=str(paths.LONGMEMEVAL_EPISODIC_GRAPHS_DIR),
	 help="情景记忆图谱目录",
	)
	parser.add_argument(
	 "--entity-graph-dir",
	 default=str(paths.LONGMEMEVAL_ENTITY_RELATION_GRAPHS_DIR),
	 help="实体关系图谱目录",
	)

	
	parser.add_argument("--sentence-top-k", type=int, default=60, help="消息级别检索 Top-K")
	parser.add_argument("--episodic-top-k", type=int, default=40, help="情景记忆检索 Top-K")
	parser.add_argument("--entity-top-k", type=int, default=40, help="实体关系检索 Top-K")
	parser.add_argument("--final-top-k", type=int, default=25, help="二次重排后保留数量")

	parser.add_argument("--start-qa", type=int, default=None, help="起始 QA 索引 (0-indexed)")
	parser.add_argument("--end-qa", type=int, default=None, help="结束 QA 索引 (0-indexed)")
	parser.add_argument("--max-tests", type=int, default=None, help="最大测试数量")

	parser.add_argument("--llm-model", default="gpt-4o-mini-closeai", help="答案生成 LLM 模型")
	parser.add_argument("--llm-evaluate-model", default="gpt-4o-mini-closeai", help="答案评估 LLM 模型")

	
	parser.add_argument(
	 "--rerank-method",
	 default="baai",
	 choices=["baai", "qwen", "jina", "qwen-sili", "qwen-dashscope", "gte-dashscope", "none"],
	 help="第一阶段重排序方法",
	)
	parser.add_argument(
	 "--second-stage-rerank-method",
	 default=None,
	 choices=["baai", "qwen", "jina", "qwen-sili", "qwen-dashscope", "gte-dashscope"],
	 help="二次重排序方法（默认使用 --rerank-method）",
	)
	parser.add_argument("--first-stage-top-k", type=int, default=None, help="第一阶段检索数量")
	parser.add_argument("--disable-second-stage-rerank", action="store_true", help="禁用两阶段检索")
	parser.add_argument(
	 "--fusion-method",
	 default="concatenation",
	 choices=["concatenation", "interleaved"],
	 help="融合方法",
	)

	
	parser.add_argument(
	 "--router-strategy",
	 choices=["aggressive", "conservative"],
	 default="aggressive",
	 help="路由策略",
	)

	parser.add_argument("--cascade-max-context-tokens", type=int, default=2500, help="级联剪枝最大上下文 token 数")
	parser.add_argument("--cascade-mad-multiplier", type=float, default=2.5, help="Stage1 MAD 倍率")
	parser.add_argument("--cascade-cliff-tolerance", type=float, default=2.0, help="CLIFF 容忍度")
	parser.add_argument("--cascade-absolute-min-score", type=float, default=0.0, help="STRICT 阈值")
	parser.add_argument("--cascade-lambda-mmr", type=float, default=0.6, help="MMR 多样性权重")
	parser.add_argument("--no-cascade-cap-to-input", action="store_true", help="禁用 cap_to_input_tokens")
	parser.add_argument("--cascade-tower-min-ratio", default=None, help="Stage3 per-tower 最低配额，如 'H:0.50,E:0.20,KG:0.15'")

	# Avoid mutating LogRecord fields before other handlers process the record.
	parser.add_argument(
	 "--output-dir",
	 default=str(paths.LONGMEMEVAL_TASK_EVAL_RESULTS_DIR / "longmemeval_quantification_ablation"),
	 help="消融输出根目录；每个模式写入 output_dir/{mode}/。",
	)
	parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
	return parser


def _common_config(args: argparse.Namespace) -> Dict[str, Any]:
	return {
	 "dataset_size": args.dataset_size,
	 "dataset_dir": args.dataset_dir,
	 "sentence_graph_dir": args.sentence_graph_dir,
	 "episodic_graph_dir": args.episodic_graph_dir,
	 "entity_graph_dir": args.entity_graph_dir,
	 "start_qa": args.start_qa,
	 "end_qa": args.end_qa,
	 "max_tests": args.max_tests,
	 "llm_model": args.llm_model,
	 "llm_evaluate_model": args.llm_evaluate_model,
	 "rerank_method": args.rerank_method,
	 "second_stage_rerank_method": args.second_stage_rerank_method,
	 "first_stage_top_k": args.first_stage_top_k,
	 "sentence_top_k": args.sentence_top_k,
	 "episodic_top_k": args.episodic_top_k,
	 "entity_top_k": args.entity_top_k,
	 "final_top_k": args.final_top_k,
	 "fusion_method": args.fusion_method,
	 "cascade_max_context_tokens": args.cascade_max_context_tokens,
	 "cascade_mad_multiplier": args.cascade_mad_multiplier,
	 "cascade_cliff_tolerance": args.cascade_cliff_tolerance,
	 "cascade_lambda_mmr": args.cascade_lambda_mmr,
	}


def build_ablation_config(
 args: argparse.Namespace,
 mode_config: AblationModeConfig,
 mode_output_dir: Path,
) -> Dict[str, Any]:
	return {
	 "ablation_mode": mode_config.mode,
	 "description": mode_config.description,
	 "router_enabled": mode_config.router_enabled,
	 "cascade_prune_mode": "DYNAMIC_ADAPTIVE",
	 "cascade_adaptive_dataset": "longmemeval",
	 "cascade_stage1_enabled": mode_config.cascade_stage1_enabled,
	 "cascade_stage2_enabled": mode_config.cascade_stage2_enabled,
	 "cascade_stage3_mmr_enabled": mode_config.cascade_stage3_mmr_enabled,
	 "cascade_max_context_tokens": args.cascade_max_context_tokens,
	 "cascade_mad_multiplier": args.cascade_mad_multiplier,
	 "cascade_cliff_tolerance": args.cascade_cliff_tolerance,
	 "cascade_lambda_mmr": args.cascade_lambda_mmr,
	 "dataset_size": args.dataset_size,
	 "dataset_dir": args.dataset_dir,
	 "sentence_graph_dir": args.sentence_graph_dir,
	 "episodic_graph_dir": args.episodic_graph_dir,
	 "entity_graph_dir": args.entity_graph_dir,
	 "start_qa": args.start_qa,
	 "end_qa": args.end_qa,
	 "max_tests": args.max_tests,
	 "llm_model": args.llm_model,
	 "llm_evaluate_model": args.llm_evaluate_model,
	 "rerank_method": args.rerank_method,
	 "second_stage_rerank_method": args.second_stage_rerank_method,
	 "first_stage_top_k": args.first_stage_top_k,
	 "topk": {
	  "sentence": args.sentence_top_k,
	  "episodic": args.episodic_top_k,
	  "entity": args.entity_top_k,
	  "final_top_k": args.final_top_k,
	 },
	 "fusion_method": args.fusion_method,
	 "router_strategy": args.router_strategy,
	 "output_dir": str(mode_output_dir),
	 "timestamp": datetime.now().isoformat(),
	}


def create_benchmark(
 args: argparse.Namespace,
 mode_config: AblationModeConfig,
 mode_output_dir: Path,
) -> LongMemEvalTripleFusionBenchmark:
	llm_client = LLMClient(model_name=args.llm_model)
	llm_evaluate_client = LLMClient(model_name=args.llm_evaluate_model)

	tower_router = None
	if mode_config.router_enabled:
		tower_router = LongMemEvalTowerRouter(
		 model_name=args.llm_model,
		 strategy=args.router_strategy,
		)

	return LongMemEvalTripleFusionBenchmark(
	 dataset_size=args.dataset_size,
	 dataset_dir=args.dataset_dir,
	 sentence_graph_dir=args.sentence_graph_dir,
	 episodic_graph_dir=args.episodic_graph_dir,
	 entity_graph_dir=args.entity_graph_dir,
	 llm_client=llm_client,
	 llm_evaluate_client=llm_evaluate_client,
	 output_dir=str(mode_output_dir),
	 sentence_top_k=args.sentence_top_k,
	 episodic_top_k=args.episodic_top_k,
	 entity_top_k=args.entity_top_k,
	 enable_sentence=True,
	 enable_episodic=True,
	 enable_entity=True,
	 max_tests=args.max_tests,
	 rerank_method=args.rerank_method,
	 fusion_method=args.fusion_method,
	 enable_second_stage_rerank=not args.disable_second_stage_rerank,
	 second_stage_rerank_method=args.second_stage_rerank_method,
	 first_stage_top_k=args.first_stage_top_k,
	 final_top_k=args.final_top_k,
	 start_qa=args.start_qa,
	 end_qa=args.end_qa,
	 tower_router=tower_router,
	 enable_cascade_pruner=True,
	 cascade_prune_mode="DYNAMIC_ADAPTIVE",
	 cascade_mad_multiplier=args.cascade_mad_multiplier,
	 cascade_cliff_tolerance=args.cascade_cliff_tolerance,
	 cascade_absolute_min_score=args.cascade_absolute_min_score,
	 cascade_max_context_tokens=args.cascade_max_context_tokens,
	 cascade_lambda_mmr=args.cascade_lambda_mmr,
	 cascade_enable_stage1=mode_config.cascade_stage1_enabled,
	 cascade_enable_stage2=mode_config.cascade_stage2_enabled,
	 cascade_enable_stage3_mmr=mode_config.cascade_stage3_mmr_enabled,
	 cascade_cap_to_input_tokens=not args.no_cascade_cap_to_input,
	 cascade_tower_min_ratio=_parse_tower_min_ratio_str(args.cascade_tower_min_ratio),
	 cascade_adaptive_dataset="longmemeval",
	)


def _score_dict(scores: Dict[str, Any]) -> Dict[str, Any]:
	if not isinstance(scores, dict):
		return {}
	nested = scores.get("scores")
	if isinstance(nested, dict):
		return nested
	return scores


def _result_accuracy(result: Dict[str, Any]) -> float:
	scores = _score_dict(result.get("scores", {}))
	return float(scores.get("llm_accuracy", scores.get("accuracy", 0.0)) or 0.0)


def _latest_file(directory: Path, pattern: str) -> Optional[Path]:
	files = sorted(directory.glob(pattern), key=lambda path: path.stat().st_mtime)
	return files[-1] if files else None


def _load_json(path: Optional[Path]) -> Optional[Dict[str, Any]]:
	if path is None or not path.exists():
		return None
	with path.open("r", encoding="utf-8") as handle:
		data = json.load(handle)
	return data if isinstance(data, dict) else None


def _load_results(path: Optional[Path]) -> List[Dict[str, Any]]:
	if path is None or not path.exists():
		return []
	with path.open("r", encoding="utf-8") as handle:
		data = json.load(handle)
	return data if isinstance(data, list) else []


def _dataset_path(args: argparse.Namespace) -> Path:
	dataset_dir = Path(args.dataset_dir) if args.dataset_dir else Path(__file__).parent.parent / "dataset" / "LongMemEval"
	return dataset_dir / f"longmemeval_{args.dataset_size}_cleaned.json"


def _category_map(args: argparse.Namespace) -> Dict[int, str]:
	path = _dataset_path(args)
	if not path.exists():
		return {}
	try:
		with path.open("r", encoding="utf-8") as handle:
			dataset = json.load(handle)
	except Exception as exc:
		logger.warning("读取 LongMemEval 数据集失败 %s: %s", path, exc)
		return {}
	return {
	 idx: str(item.get("question_type", item.get("category", "unknown")))
	 for idx, item in enumerate(dataset)
	 if isinstance(item, dict)
	}


def _read_individual_report(mode_output_dir: Path, qa_index: int) -> Optional[Dict[str, Any]]:
	return _load_json(mode_output_dir / "individual_reports" / f"qa_{qa_index}_report.json")


def _average_report_metric(
 mode_output_dir: Path,
 results: List[Dict[str, Any]],
 path: List[str],
) -> Optional[float]:
	values: List[float] = []
	for result in results:
		qa_index = result.get("qa_index")
		if qa_index is None:
			continue
		report = _read_individual_report(mode_output_dir, int(qa_index))
		current: Any = report
		for key in path:
			if not isinstance(current, dict):
				current = None
				break
			current = current.get(key)
		if isinstance(current, (int, float)):
			values.append(float(current))
	return _mean(values) if values else None


def summarize_mode(
 mode: str,
 mode_output_dir: Path,
 summary: Dict[str, Any],
 args: argparse.Namespace,
 summary_file: Optional[Path] = None,
 results_file: Optional[Path] = None,
) -> Dict[str, Any]:
	"""Run summarize mode."""
	if not summary:
		summary_file = summary_file or _latest_file(mode_output_dir, "summary_*.json")
		summary = _load_json(summary_file) or {}
	results_file = results_file or _latest_file(mode_output_dir, "results_*.json")
	detailed_results = _load_results(results_file)

	category_lookup = _category_map(args)
	category_groups: Dict[str, List[Dict[str, Any]]] = {}
	for result in detailed_results:
		qa_index = result.get("qa_index")
		category = category_lookup.get(int(qa_index), "unknown") if qa_index is not None else "unknown"
		category_groups.setdefault(category, []).append(result)

	category_wise_accuracy = {}
	for category, category_results in sorted(category_groups.items()):
		successful = [r for r in category_results if r.get("success", False)]
		category_wise_accuracy[category] = {
		 "accuracy": _mean(_result_accuracy(r) for r in successful),
		 "accuracy_failed_as_zero": _mean(_result_accuracy(r) if r.get("success", False) else 0.0 for r in category_results),
		 "total_questions": len(category_results),
		 "success_count": len(successful),
		}

	test_info = summary.get("test_info", {}) if isinstance(summary, dict) else {}
	scores = summary.get("scores", {}) if isinstance(summary, dict) else {}
	token_stats = summary.get("token_stats", {}) if isinstance(summary, dict) else {}
	cascade_stats = summary.get("cascade_stats", {}) if isinstance(summary, dict) else {}

	total_questions = int(test_info.get("total_tests", len(detailed_results)) or 0)
	success_count = int(test_info.get("successful_tests", sum(1 for r in detailed_results if r.get("success", False))) or 0)

	average_context_tokens = _average_report_metric(mode_output_dir, detailed_results, ["token_stats", "context_tokens"])
	average_total_tokens = _average_report_metric(mode_output_dir, detailed_results, ["token_stats", "total_tokens"])
	average_cascade_tokens = _average_report_metric(
	 mode_output_dir,
	 detailed_results,
	 ["retrieval_details", "cascade_tokens_used"],
	)

	if average_total_tokens is None and success_count:
		average_total_tokens = float(token_stats.get("total_tokens", 0.0)) / success_count
	if average_cascade_tokens is None:
		average_cascade_tokens = float(cascade_stats.get("avg_tokens_used", 0.0) or 0.0)
	overall_accuracy = scores.get("llm_accuracy", scores.get("accuracy"))
	if overall_accuracy is None:
		overall_accuracy = _mean(
		 _result_accuracy(result)
		 for result in detailed_results
		 if result.get("success", False)
		)

	return {
	 "ablation_mode": mode,
	 "output_dir": str(mode_output_dir),
	 "overall_accuracy": float(overall_accuracy or 0.0),
	 "category_wise_accuracy": category_wise_accuracy,
	 "average_context_tokens": average_context_tokens if average_context_tokens is not None else 0.0,
	 "average_total_tokens": average_total_tokens if average_total_tokens is not None else 0.0,
	 "average_cascade_tokens_used": average_cascade_tokens,
	 "total_questions": total_questions,
	 "success_count": success_count,
	 "summary_file": str(summary_file) if summary_file else "",
	 "results_file": str(results_file) if results_file else "",
	}


def run_mode(args: argparse.Namespace, mode: str) -> Dict[str, Any]:
	mode_config = ABLATION_MODES[mode]
	mode_output_dir = Path(args.output_dir) / mode
	mode_output_dir.mkdir(parents=True, exist_ok=True)

	logger.info("=" * 100)
	logger.info("运行 LongMemEval quantification 消融模式: %s", mode)
	logger.info("输出目录: %s", mode_output_dir)
	logger.info("=" * 100)

	_write_json(
	 mode_output_dir / "ablation_config.json",
	 build_ablation_config(args, mode_config, mode_output_dir),
	)

	existing_summaries = set(mode_output_dir.glob("summary_*.json"))
	existing_results = set(mode_output_dir.glob("results_*.json"))
	benchmark = create_benchmark(args, mode_config, mode_output_dir)
	summary = benchmark.run_benchmark()

	new_summaries = [p for p in mode_output_dir.glob("summary_*.json") if p not in existing_summaries]
	new_results = [p for p in mode_output_dir.glob("results_*.json") if p not in existing_results]
	summary_file = max(new_summaries, key=lambda p: p.stat().st_mtime) if new_summaries else _latest_file(mode_output_dir, "summary_*.json")
	results_file = max(new_results, key=lambda p: p.stat().st_mtime) if new_results else _latest_file(mode_output_dir, "results_*.json")
	return summarize_mode(mode, mode_output_dir, summary, args, summary_file, results_file)


def write_cumulative_summary(
 args: argparse.Namespace,
 modes: List[str],
 mode_summaries: Dict[str, Dict[str, Any]],
) -> Path:
	output_dir = Path(args.output_dir)
	summary = {
	 "timestamp": datetime.now().isoformat(),
	 "output_dir": str(output_dir),
	 "modes": modes,
	 "common_config": _common_config(args),
	 "results": mode_summaries,
	}
	summary_path = output_dir / "cumulative_ablation_summary.json"
	_write_json(summary_path, summary)
	logger.info("累计消融汇总已写入: %s", summary_path)
	return summary_path


def _validate_args(args: argparse.Namespace) -> None:
	if args.start_qa is not None and args.end_qa is not None and args.start_qa > args.end_qa:
		raise ValueError("start-qa 不能大于 end-qa")
	for name in ("start_qa", "end_qa"):
		value = getattr(args, name)
		if value is not None and (value < 0 or value > 499):
			raise ValueError(f"{name.replace('_', '-')} 必须在 0-499 范围内")


def main() -> int:
	parser = build_parser()
	args = parser.parse_args()
	logging.getLogger().setLevel(getattr(logging, args.log_level))
	_validate_args(args)

	modes = ABLATION_ORDER if args.run_all else [args.ablation_mode]
	logger.info("LongMemEval quantification ablation modes: %s", modes)
	logger.info("共同实验配置: %s", _common_config(args))

	mode_summaries: Dict[str, Dict[str, Any]] = {}
	try:
		for mode in modes:
			mode_summaries[mode] = run_mode(args, mode)
		write_cumulative_summary(args, modes, mode_summaries)
		logger.info("LongMemEval quantification ablation 完成")
		return 0
	finally:
		if cleanup_evaluation_models is not None:
			try:
				cleanup_evaluation_models()
			except Exception as exc:
				logger.warning("评估资源清理失败: %s", exc)


if __name__ == "__main__":
	raise SystemExit(main())
