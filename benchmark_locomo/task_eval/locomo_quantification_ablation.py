#!/usr/bin/env python3
"""LoCoMo Quantification Ablation Runner."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from benchmark_locomo.task_eval.evaluation import cleanup_evaluation_models
from benchmark_locomo.task_eval.locomo_triple_router_quantification import (
 LoCoMoTriTowerBenchmark,
)
from mandol.core import paths
from mandol.memory_router.locomo_tower_router import LocomoTowerRouter
from mandol.retrieval.rerank_manager import RerankerManager
from mandol.utils.logging_config import (
 auto_configure_logging,
 create_module_logger,
 setup_logging,
)

if auto_configure_logging() is None:
	setup_logging(level=logging.INFO)
logger = create_module_logger("locomo_quantification_ablation")


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
	"""Small JSON sanitizer for ablation config/summary files."""
	if obj is None:
		return None
	if isinstance(obj, Path):
		return str(obj)
	if isinstance(obj, np.generic):
		return _json_safe(obj.item())
	if isinstance(obj, float):
		return obj if np.isfinite(obj) else None
	if isinstance(obj, (str, int, bool)):
		return obj
	if isinstance(obj, dict):
		return {str(k): _json_safe(v) for k, v in obj.items()}
	if isinstance(obj, (list, tuple, set)):
		return [_json_safe(v) for v in obj]
	return str(obj)


def _write_json(path: Path, data: Dict[str, Any]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as handle:
		json.dump(_json_safe(data), handle, ensure_ascii=False, indent=2)


def _mean(values: Iterable[float]) -> float:
	vals = [float(v) for v in values]
	return float(np.mean(vals)) if vals else 0.0


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
	 description="LoCoMo quantification ablation runner (paper-use only)."
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

	parser.add_argument(
	 "--step3-graphs-dir",
	 default=str(paths.LOCOMO_ENTITY_RELATION_STEP3_DIR),
	 help="知识图谱（实体关系）数据目录",
	)
	parser.add_argument(
	 "--enhanced-graphs-dir",
	 default=str(paths.LOCOMO_HIERARCHICAL_CONTENT_STEP4_DIR),
	 help="分层图谱数据目录",
	)
	parser.add_argument(
	 "--episodic-graphs-dir",
	 default=str(paths.LOCOMO_EPISODIC_STEP3_DIR),
	 help="情景记忆图谱数据目录",
	)
	parser.add_argument(
	 "--qa-dataset",
	 default=str(paths.LOCOMO_RAW_FILE),
	 help="QA 数据集路径",
	)
	parser.add_argument(
	 "--output-dir",
	 default=str(paths.LOCOMO_TASK_EVAL_RESULTS_DIR / "locomo_quantification_ablation"),
	 help="消融输出根目录；每个模式写入 output_dir/{mode}/。",
	)

	parser.add_argument("--sample-ids", nargs="+", help="指定样本 ID 列表")
	parser.add_argument("--llm-model", default="gpt-4o-mini-closeai", help="答案生成 LLM 模型")
	parser.add_argument("--llm-evaluate-model", default="gpt-4o-mini-closeai", help="答案评估 LLM 模型")
	parser.add_argument("--reranker-type", default="baai", choices=[
	 "baai", "qwen", "jina", "qwen-sili", "qwen-dashscope", "gte-dashscope",
	], help="重排序器类型")
	parser.add_argument("--topk-hierarchical", type=int, default=15, help="分层记忆统一检索 top-k")
	parser.add_argument("--topk-similarity", type=int, default=30, help="图谱语义检索 top-k")
	parser.add_argument("--topk-graph", type=int, default=0, help="图谱实体关系 top-k")
	parser.add_argument("--topk-episodic", type=int, default=30, help="情景记忆检索 top-k")
	parser.add_argument("--final-top-k", type=int, default=20, help="二次重排序后最终保留 top-k")
	parser.add_argument("--cascade-max-context-tokens", type=int, default=2500, help="级联剪枝上下文 token 预算")
	parser.add_argument("--cascade-mad-multiplier", type=float, default=2.5, help="Stage1 MAD 倍率")
	parser.add_argument("--cascade-cliff-tolerance", type=float, default=2.0, help="CLIFF 容忍度")
	parser.add_argument("--cascade-lambda-mmr", type=float, default=0.6, help="Stage3 MMR 多样性权重")

	parser.add_argument("--threshold", type=float, default=0.0, help="Rerank score threshold")
	parser.add_argument(
	 "--rerank-strategy",
	 choices=["tower_separate", "unified_rerank"],
	 default="tower_separate",
	 help="重排序策略",
	)
	parser.add_argument("--no-entity-relation", action="store_true", help="禁用实体关系检索")
	parser.add_argument("--fusion-strategy", choices=["simple", "weighted", "context_aware"], default="context_aware")
	parser.add_argument("--weight-hierarchical", type=float, default=0.35)
	parser.add_argument("--weight-graph", type=float, default=0.35)
	parser.add_argument("--weight-episodic", type=float, default=0.30)
	parser.add_argument("--router-strategy", choices=["aggressive", "conservative"], default="aggressive")
	parser.add_argument("--parallel", action="store_true", help="启用三塔并行检索")
	parser.add_argument("--max-workers", type=int, default=3, help="最大工作线程数")
	parser.add_argument(
	 "--no-save-individual-reports",
	 action="store_true",
	 help="不保存每题 individual report；默认保存，便于消融分析。",
	)
	parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
	return parser


def _common_config(args: argparse.Namespace) -> Dict[str, Any]:
	return {
	 "qa_dataset": args.qa_dataset,
	 "sample_ids": args.sample_ids,
	 "llm_model": args.llm_model,
	 "llm_evaluate_model": args.llm_evaluate_model,
	 "reranker_type": args.reranker_type,
	 "topk_hierarchical": args.topk_hierarchical,
	 "topk_similarity": args.topk_similarity,
	 "topk_graph": args.topk_graph,
	 "topk_episodic": args.topk_episodic,
	 "final_top_k": args.final_top_k,
	 "cascade_max_context_tokens": args.cascade_max_context_tokens,
	 "cascade_mad_multiplier": args.cascade_mad_multiplier,
	 "cascade_cliff_tolerance": args.cascade_cliff_tolerance,
	 "cascade_lambda_mmr": args.cascade_lambda_mmr,
	 "rerank_strategy": args.rerank_strategy,
	 "threshold": args.threshold,
	 "fusion_strategy": args.fusion_strategy,
	 "fusion_weights": {
	  "hierarchical": args.weight_hierarchical,
	  "graph": args.weight_graph,
	  "episodic": args.weight_episodic,
	 },
	 "save_individual_reports": not args.no_save_individual_reports,
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
	 "cascade_adaptive_dataset": "locomo",
	 "cascade_stage1_enabled": mode_config.cascade_stage1_enabled,
	 "cascade_stage2_enabled": mode_config.cascade_stage2_enabled,
	 "cascade_stage3_mmr_enabled": mode_config.cascade_stage3_mmr_enabled,
	 "cascade_max_context_tokens": args.cascade_max_context_tokens,
	 "cascade_mad_multiplier": args.cascade_mad_multiplier,
	 "cascade_cliff_tolerance": args.cascade_cliff_tolerance,
	 "cascade_lambda_mmr": args.cascade_lambda_mmr,
	 "llm_model": args.llm_model,
	 "llm_evaluate_model": args.llm_evaluate_model,
	 "reranker_type": args.reranker_type,
	 "topk": {
	  "hierarchical": args.topk_hierarchical,
	  "similarity": args.topk_similarity,
	  "graph": args.topk_graph,
	  "episodic": args.topk_episodic,
	  "final_top_k": args.final_top_k,
	 },
	 "qa_dataset": args.qa_dataset,
	 "sample_ids": args.sample_ids,
	 "output_dir": str(mode_output_dir),
	 "save_individual_reports": not args.no_save_individual_reports,
	 "timestamp": datetime.now().isoformat(),
	}


def create_benchmark(
 args: argparse.Namespace,
 mode_config: AblationModeConfig,
 mode_output_dir: Path,
 reranker_manager: RerankerManager,
) -> LoCoMoTriTowerBenchmark:
	tower_router = None
	if mode_config.router_enabled:
		tower_router = LocomoTowerRouter(
		 model_name=args.llm_model,
		 strategy=args.router_strategy,
		)

	fusion_weights = {
	 "hierarchical": args.weight_hierarchical,
	 "graph": args.weight_graph,
	 "episodic": args.weight_episodic,
	}
	use_entity_relation = (not args.no_entity_relation) and (args.topk_graph > 0)

	return LoCoMoTriTowerBenchmark(
	 step3_graphs_dir=args.step3_graphs_dir,
	 enhanced_graphs_dir=args.enhanced_graphs_dir,
	 episodic_graphs_dir=args.episodic_graphs_dir,
	 qa_dataset_path=args.qa_dataset,
	 output_dir=str(mode_output_dir),
	 llm_model=args.llm_model,
	 llm_evaluate_model=args.llm_evaluate_model,
	 target_sample_ids=args.sample_ids,
	 topk_hierarchical=args.topk_hierarchical,
	 topk_similarity=args.topk_similarity,
	 topk_graph=args.topk_graph,
	 topk_episodic=args.topk_episodic,
	 use_entity_relation=use_entity_relation,
	 enable_second_stage_rerank=True,
	 second_stage_rerank_method=args.reranker_type,
	 final_top_k=args.final_top_k,
	 rerank_threshold=args.threshold,
	 rerank_strategy=args.rerank_strategy,
	 reranker_type=args.reranker_type,
	 reranker_manager=reranker_manager,
	 fusion_strategy=args.fusion_strategy,
	 fusion_weights=fusion_weights,
	 tower_router=tower_router,
	 enable_cascade_pruner=True,
	 cascade_prune_mode="DYNAMIC_ADAPTIVE",
	 cascade_mad_multiplier=args.cascade_mad_multiplier,
	 cascade_cliff_tolerance=args.cascade_cliff_tolerance,
	 cascade_max_context_tokens=args.cascade_max_context_tokens,
	 cascade_lambda_mmr=args.cascade_lambda_mmr,
	 cascade_enable_stage1=mode_config.cascade_stage1_enabled,
	 cascade_enable_stage2=mode_config.cascade_stage2_enabled,
	 cascade_enable_stage3_mmr=mode_config.cascade_stage3_mmr_enabled,
	 cascade_adaptive_dataset="locomo",
	 save_individual_reports=not args.no_save_individual_reports,
	 parallel_towers=args.parallel,
	 max_workers=args.max_workers,
	)


def _result_accuracy(result: Dict[str, Any]) -> float:
	scores = result.get("evaluation_scores", {}) or {}
	return float(scores.get("llm_accuracy", 0.0) or 0.0)


def _read_results_from_files(sample_files: Iterable[Path]) -> List[Dict[str, Any]]:
	results: List[Dict[str, Any]] = []
	for sample_file in sorted(sample_files):
		try:
			with sample_file.open("r", encoding="utf-8") as handle:
				data = json.load(handle)
			for item in data.get("results", []):
				if isinstance(item, dict):
					results.append(item)
		except Exception as exc:
			logger.warning("读取样本结果失败 %s: %s", sample_file, exc)
	return results


def _summarize_results(mode: str, mode_output_dir: Path, sample_files: Iterable[Path]) -> Dict[str, Any]:
	results = _read_results_from_files(sample_files)
	successful = [r for r in results if r.get("evaluation_success", False)]
	category_groups: Dict[str, List[Dict[str, Any]]] = {}
	for result in results:
		category_groups.setdefault(str(result.get("category", "unknown")), []).append(result)

	category_wise_accuracy = {}
	for category, category_results in sorted(category_groups.items()):
		category_successful = [r for r in category_results if r.get("evaluation_success", False)]
		category_wise_accuracy[category] = {
		 "accuracy": _mean(_result_accuracy(r) for r in category_successful),
		 "accuracy_failed_as_zero": _mean(_result_accuracy(r) if r.get("evaluation_success", False) else 0.0 for r in category_results),
		 "total_questions": len(category_results),
		 "success_count": len(category_successful),
		}

	return {
	 "ablation_mode": mode,
	 "output_dir": str(mode_output_dir),
	 "overall_accuracy": _mean(_result_accuracy(r) for r in successful),
	 "overall_accuracy_failed_as_zero": _mean(_result_accuracy(r) if r.get("evaluation_success", False) else 0.0 for r in results),
	 "category_wise_accuracy": category_wise_accuracy,
	 "average_total_input_tokens": _mean(r.get("total_input_tokens", 0.0) for r in results),
	 "average_cascade_tokens_used": _mean(r.get("cascade_tokens_used", 0.0) for r in results),
	 "total_questions": len(results),
	 "success_count": len(successful),
	 "sample_result_files": [str(path) for path in sorted(sample_files)],
	}


def _sample_files(mode_output_dir: Path) -> List[Path]:
	return [
	 path for path in sorted(mode_output_dir.glob("sample_*.json"))
	 if "_readable_" not in path.name
	]


def run_mode(
 args: argparse.Namespace,
 mode: str,
 reranker_manager: RerankerManager,
) -> Dict[str, Any]:
	mode_config = ABLATION_MODES[mode]
	mode_output_dir = Path(args.output_dir) / mode
	mode_output_dir.mkdir(parents=True, exist_ok=True)

	logger.info("=" * 80)
	logger.info("运行消融模式: %s", mode)
	logger.info("输出目录: %s", mode_output_dir)
	logger.info("=" * 80)

	ablation_config = build_ablation_config(args, mode_config, mode_output_dir)
	_write_json(mode_output_dir / "ablation_config.json", ablation_config)

	existing_files = set(_sample_files(mode_output_dir))
	benchmark = create_benchmark(args, mode_config, mode_output_dir, reranker_manager)
	benchmark.run_tri_tower_benchmark(sequential_mode=True)

	new_files = [path for path in _sample_files(mode_output_dir) if path not in existing_files]
	if not new_files:
		logger.warning(
		 "模式 %s 未检测到新的 sample_*.json 文件；累计汇总将回退读取该目录下所有样本文件。",
		 mode,
		)
		new_files = _sample_files(mode_output_dir)

	return _summarize_results(mode, mode_output_dir, new_files)


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


def main() -> int:
	parser = build_parser()
	args = parser.parse_args()
	logging.getLogger().setLevel(getattr(logging, args.log_level))

	modes = ABLATION_ORDER if args.run_all else [args.ablation_mode]
	logger.info("Quantification ablation modes: %s", modes)
	logger.info("共同实验配置: %s", _common_config(args))

	reranker_manager = RerankerManager()
	mode_summaries: Dict[str, Dict[str, Any]] = {}
	try:
		for mode in modes:
			mode_summaries[mode] = run_mode(args, mode, reranker_manager)

		write_cumulative_summary(args, modes, mode_summaries)
		logger.info("Quantification ablation 完成")
		return 0
	finally:
		try:
			cleanup_evaluation_models()
		except Exception as exc:
			logger.warning("资源清理失败: %s", exc)


if __name__ == "__main__":
	raise SystemExit(main())
