"""python re_eval_llm_accuracy.py --data-dir results/triple_fusion/xxx/individual_reports."""

from datetime import datetime
import os
import sys
import json
import logging
import argparse
import re
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, List, Optional

current_file = Path(__file__).resolve()
project_root = current_file.parents[2]

# Avoid mutating LogRecord fields before other handlers process the record.
from mandol.utils.logging_config import setup_logging, create_module_logger, auto_configure_logging, set_log_level
if auto_configure_logging() is None:
    setup_logging(level=logging.INFO)
logger = create_module_logger("re_eval_llm_accuracy")

try:
    from mandol.llm.llm_client import LLMClient
    from benchmark_longmemeval.task_eval.evaluation import (
        calculate_comprehensive_scores,
        llm_grader,
        mem0_llm_grader
    )
except ImportError as e:
    print(f" 导入失败: {e}")
    print(f"当前 sys.path: {sys.path}")
    print("请确保脚本位置正确或手动调整 project_root")
    sys.exit(1)


class ReEvaluator:
    
    def __init__(self, model_name: str, llm_judge_prompt: str = "default"):
        self.llm_client = LLMClient(model_name=model_name)
        self.metrics = ["llm_judge"]
        self.llm_judge_prompt = llm_judge_prompt
            
        logger.info(f" 初始化重评测器 | 模型: {model_name} | 指标: llm_judge | prompt: {llm_judge_prompt}")

    def evaluate_item_result(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate item result."""
        question = item.get('question', '')
        gold_answer = item.get('ground_truth', '')
        response = item.get('generated_answer', '')
        
        scores_info = item.get('scores', {})
        input_info = scores_info.get('input_info', {})
        question_type = input_info.get('question_type', 'default')

        try:
            eval_result = calculate_comprehensive_scores(
                gold_answer=gold_answer,
                response=response,
                question=question,
                context="",
                question_type=question_type,
                llm_client=self.llm_client,
                metrics=self.metrics,
                llm_judge_prompt=self.llm_judge_prompt
            )
            
            return eval_result

        except Exception as e:
            logger.error(f"评估失败: {e}")
            return {
                "scores": {"llm_accuracy": 0.0},
                "llm_details": {
                    "error": str(e),
                    "judge_prompt": self.llm_judge_prompt,
                    "accuracy": 0.0,
                    "judgments": [False],
                },
            }

    def evaluate_item(self, item: Dict[str, Any]) -> float:
        """Evaluate item."""
        eval_result = self.evaluate_item_result(item)
        scores = eval_result.get("scores", {})
        return float(scores.get("llm_accuracy", 0.0))
    
    def evaluate_item_direct(self, item: Dict[str, Any]) -> float:
        """Evaluate item direct."""
        question = item.get('question', '')
        gold_answer = item.get('ground_truth', '')
        response = item.get('generated_answer', '')
        
        scores_info = item.get('scores', {})
        input_info = scores_info.get('input_info', {})
        question_type = input_info.get('question_type', 'default')
        
        try:
            grader = mem0_llm_grader if self.llm_judge_prompt == "mem0" else llm_grader
            result = grader(
                self.llm_client, question, gold_answer, response,
                question_type=question_type
            )
            return float(result)
        except Exception as e:
            logger.error(f"直接评估失败: {e}")
            return 0.0


def update_item_scores(item: Dict[str, Any],
                       new_llm_score: float,
                       llm_details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Update item scores."""
    if 'scores' not in item:
        item['scores'] = {}
    if 'scores' not in item['scores']:
        item['scores']['scores'] = {}
    
    item['scores']['scores']['llm_accuracy'] = new_llm_score
    
    if llm_details is not None:
        item['scores']['llm_details'] = llm_details
    else:
        item['scores']['llm_details'] = {
            'accuracy': new_llm_score,
            'judgments': [bool(new_llm_score)]
        }
    
    return item


def calculate_summary_statistics(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute summary statistics."""
    llm_scores = []
    question_type_scores = {}
    
    for item in items:
        scores = item.get('scores', {}).get('scores', {})
        llm_acc = scores.get('llm_accuracy')
        
        if llm_acc is not None:
            llm_scores.append(float(llm_acc))
            
            input_info = item.get('scores', {}).get('input_info', {})
            q_type = input_info.get('question_type', 'default')
            if q_type not in question_type_scores:
                question_type_scores[q_type] = []
            question_type_scores[q_type].append(float(llm_acc))
    
    stats = {
        "total_count": len(items),
        "evaluated_count": len(llm_scores),
        "avg_llm_accuracy": float(np.mean(llm_scores)) if llm_scores else 0.0,
        "std_llm_accuracy": float(np.std(llm_scores)) if llm_scores else 0.0,
        "correct_count": sum(1 for s in llm_scores if s >= 0.5),
        "accuracy_by_question_type": {}
    }
    
    for q_type, scores in question_type_scores.items():
        stats["accuracy_by_question_type"][q_type] = {
            "count": len(scores),
            "avg_accuracy": float(np.mean(scores)) if scores else 0.0,
            "correct_count": sum(1 for s in scores if s >= 0.5)
        }
    
    return stats


def natural_sort_key(path: Path) -> int:
    """Run natural sort key."""
    name = path.stem  # qa_0_report -> qa_0_report
    match = re.search(r'qa_(\d+)', name)
    if match:
        return int(match.group(1))
    return 0


def write_json_no_overwrite(data: Dict[str, Any], output_file: Path) -> Path:
    """Write JSON to output_file without overwriting an existing file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    candidate = output_file
    counter = 1

    while True:
        try:
            with candidate.open('x', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return candidate
        except FileExistsError:
            candidate = output_file.with_name(
                f"{output_file.stem}__{counter}{output_file.suffix}"
            )
            counter += 1


def sanitize_path_component(value: str) -> str:
    """Return a filesystem-friendly component while keeping model names readable."""
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "unknown")).strip("._-")
    return safe or "unknown"


def build_default_output_dir(data_dir: Path,
                             model_name: str,
                             llm_judge_prompt: str,
                             timestamp: Optional[str] = None) -> Path:
    """Build a non-ambiguous default re-evaluation run directory."""
    run_timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_model = sanitize_path_component(model_name)
    safe_protocol = sanitize_path_component(llm_judge_prompt)
    return data_dir.parent / f"reeval_{run_timestamp}_{safe_model}_{safe_protocol}"


def process_directory(data_dir: Path, 
                      evaluator: ReEvaluator, 
                      output_dir: Path,
                      file_pattern: str = "qa_*_report.json",
                      start_index: Optional[int] = None,
                      skip_existing: bool = False):
    """Process directory."""
    
    
    files = sorted(data_dir.glob(file_pattern), key=natural_sort_key)
    
    if not files:
        logger.warning(f"未在 {data_dir} 找到匹配 '{file_pattern}' 的文件")
        return
    
    total_files = len(files)
    logger.info(f" 找到 {total_files} 个文件")
    
    if start_index is not None:
        files = [f for f in files if natural_sort_key(f) >= start_index]
        logger.info(f" 从 qa_{start_index} 开始，待处理 {len(files)} 个文件")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if skip_existing:
        original_count = len(files)
        files = [f for f in files if not (output_dir / f.name).exists()]
        skipped = original_count - len(files)
        if skipped > 0:
            logger.info(f" 跳过 {skipped} 个已存在的文件，剩余 {len(files)} 个待处理")
    
    if not files:
        logger.info(" 所有文件都已处理完成")
        return
    
    all_items = []
    success_count = 0
    failed_count = 0
    skipped_count = 0
    
    for file_path in tqdm(files, desc="Re-evaluating", unit="file"):
        qa_index = natural_sort_key(file_path)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                item = json.load(f)
            
            old_score = item.get('scores', {}).get('scores', {}).get('llm_accuracy', None)
            
            eval_result = evaluator.evaluate_item_result(item)
            new_score = float(eval_result.get("scores", {}).get("llm_accuracy", 0.0))
            llm_details = eval_result.get("llm_details", {})
            
            updated_item = update_item_scores(item, new_score, llm_details=llm_details)
            
            updated_item['re_evaluation'] = {
                'original_llm_accuracy': old_score,
                'new_llm_accuracy': new_score,
                'mode': 'llm_judge',
                'llm_judge_prompt': evaluator.llm_judge_prompt,
                'changed': old_score != new_score if old_score is not None else True
            }
            
            
            output_file = output_dir / file_path.name
            saved_file = write_json_no_overwrite(updated_item, output_file)
            if saved_file != output_file:
                logger.warning(f" 输出文件已存在，已改写到不覆盖路径: {saved_file}")
            
            all_items.append(updated_item)
            success_count += 1
            
            if old_score is not None and old_score != new_score:
                change_symbol = "" if new_score > old_score else ""
                logger.debug(f"  {file_path.name}: {old_score:.1f} -> {new_score:.1f} {change_symbol}")
                
        except Exception as e:
            logger.error(f" 处理文件失败 {file_path.name}: {e}")
            failed_count += 1
            continue
    
    stats = calculate_summary_statistics(all_items)
    stats['success_count'] = success_count
    stats['failed_count'] = failed_count
    stats['mode'] = 'llm_judge'
    stats['llm_judge_prompt'] = evaluator.llm_judge_prompt
    
    
    summary_file = output_dir / "re_eval_summary.json"
    saved_summary_file = write_json_no_overwrite(stats, summary_file)
    if saved_summary_file != summary_file:
        logger.warning(f" 汇总文件已存在，已改写到不覆盖路径: {saved_summary_file}")
    
    print("\n" + "=" * 60)
    print(" 重评估完成统计")
    print("=" * 60)
    print(f"  处理文件: {success_count} 成功, {failed_count} 失败")
    print("  评估指标: llm_judge")
    print(f"  Judge Prompt: {evaluator.llm_judge_prompt}")
    print(f"  平均 LLM Accuracy: {stats['avg_llm_accuracy']:.4f}")
    print(f"  正确数量: {stats['correct_count']} / {stats['evaluated_count']}")
    print("\n  按问题类型统计:")
    for q_type, type_stats in stats['accuracy_by_question_type'].items():
        print(f"    {q_type}: {type_stats['avg_accuracy']:.4f} ({type_stats['correct_count']}/{type_stats['count']})")
    print("=" * 60)
    print(f" 结果已保存至: {output_dir}")
    print(f" 汇总报告: {saved_summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="LongMemEval 重评估脚本 - 对已有结果进行 LLM 评分重评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # LLM judge 重评估
  python re_eval_llm_accuracy.py --data-dir results/triple_fusion/xxx/individual_reports
  
  # 指定输出目录和模型
  python re_eval_llm_accuracy.py --data-dir results/xxx --output-dir results/reeval --model gpt-4o
        """
    )
    
    parser.add_argument('--data-dir', 
                        required=True,
                        help="输入目录（包含 qa_*_report.json 文件的 individual_reports 目录）")
    parser.add_argument('--output-dir', 
                        default=None,
                        help="输出目录（默认自动生成 data-dir 同级的 reeval_<时间戳>_<评估模型>_<协议> 目录）")
    parser.add_argument('--model', 
                        default="gpt-4o-mini-closeai",
                        help="评估使用的模型名称（默认 gpt-4o-mini-closeai）")
    parser.add_argument('--llm-judge-prompt',
                        choices=["default", "mem0"],
                        default="default",
                        help="LLM judge 提示词类型：default 为现有提示词；mem0 为 mem0/LongMemEval 对齐提示词")
    parser.add_argument('--file-pattern',
                        default="qa_*_report.json",
                        help="文件匹配模式（默认 qa_*_report.json）")
    parser.add_argument('--start-index', type=int,
                        default=None,
                        help="从指定的 qa_index 开始处理（用于断点续传）")
    parser.add_argument('--skip-existing',
                        action='store_true',
                        help="跳过输出目录中已存在的文件（断点续传）")
    parser.add_argument('--verbose', '-v',
                        action='store_true',
                        help="显示详细输出")
    
    args = parser.parse_args()

    # Avoid mutating LogRecord fields before other handlers process the record.
    if args.verbose:
        set_log_level(logging.DEBUG)

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f" 输入目录不存在: {data_dir}")
        sys.exit(1)
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = build_default_output_dir(
            data_dir=data_dir,
            model_name=args.model,
            llm_judge_prompt=args.llm_judge_prompt,
        )

    print("=" * 60)
    print(" LongMemEval LLM Accuracy 重评估")
    print("=" * 60)
    print(f" 输入目录: {data_dir}")
    print(f" 输出目录: {output_dir}")
    print(f" 评估模型: {args.model}")
    print(" 评估指标: llm_judge")
    print(f" Judge Prompt: {args.llm_judge_prompt}")
    print(f" 文件模式: {args.file_pattern}")
    if args.start_index is not None:
        print(f" 起始索引: qa_{args.start_index}")
    if args.skip_existing:
        print(f"  跳过已存在: 启用")
    print("=" * 60)

    evaluator = ReEvaluator(model_name=args.model, llm_judge_prompt=args.llm_judge_prompt)

    process_directory(
        data_dir, 
        evaluator, 
        output_dir, 
        args.file_pattern,
        start_index=args.start_index,
        skip_existing=args.skip_existing
    )

    logger.info(" 重评估任务完成")


if __name__ == "__main__":
    main()
