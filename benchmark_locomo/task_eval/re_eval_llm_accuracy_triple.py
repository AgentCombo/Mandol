import os
import sys
import json
import logging
import argparse
import re
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

current_file = Path(__file__).resolve()
project_root = current_file.parents[2]

# Avoid mutating LogRecord fields before other handlers process the record.
from mandol.utils.logging_config import setup_logging, create_module_logger, auto_configure_logging
from mandol.core import paths
if auto_configure_logging() is None:
    setup_logging(level=logging.INFO)
logger = create_module_logger("re_eval_llm_accuracy_triple")

try:
    from mandol.llm.llm_client import LLMClient
    from benchmark_locomo.task_eval.evaluation import calculate_comprehensive_scores
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
        
        logger.info(f" 初始化评估器 | 模型: {model_name} | 指标: llm_judge | prompt: {llm_judge_prompt}")
            
    def evaluate_item(self, item: Dict[str, Any]) -> float:
        """Evaluate item."""
        
        question = item.get('question', '')
        gold_answer = item.get('expected_answer', '')
        response = item.get('final_answer', '')
        reasoning = item.get('reasoning_process', '') 
        category = item.get('category', 1)
        
        is_adversarial = (category == 5)

        try:
            eval_result = calculate_comprehensive_scores(
                gold_answer=gold_answer,
                response=response,
                question=question,
                context="", 
                reasoning=reasoning,
                llm_client=self.llm_client,
                metrics=self.metrics,
                category=category,
                is_adversarial=is_adversarial,
                llm_judge_prompt=self.llm_judge_prompt
            )
            
            scores = eval_result.get("scores", {})
            llm_acc = scores.get("llm_accuracy", 0.0)

            return float(llm_acc)

        except Exception as e:
            logger.error(f"评估失败: {e}")
            return 0.0

def update_statistics(data: Dict[str, Any], results: List[Dict[str, Any]], filename: str):
    """Update statistics."""
    
    valid_scores = []
    for r in results:
        if 'evaluation_scores' in r:
            score = r['evaluation_scores'].get('llm_accuracy')
            if score is not None:
                valid_scores.append(float(score))
    
    avg_llm = np.mean(valid_scores) if valid_scores else 0.0
    
    if 'performance_metrics' not in data:
        data['performance_metrics'] = {}
    
    # old_acc = data['performance_metrics'].get('avg_llm_accuracy', 'N/A')
    data['performance_metrics']['avg_llm_accuracy'] = float(avg_llm)

    if 'category_performance' in data:
        for cat_key, cat_stats in data['category_performance'].items():
            cat_scores = [
                r['evaluation_scores']['llm_accuracy']
                for r in results
                if str(r.get('category')) == str(cat_key)
                and r.get('evaluation_scores', {}).get('llm_accuracy') is not None
            ]
            if cat_scores:
                cat_stats['avg_llm_accuracy'] = float(np.mean(cat_scores))
            else:
                cat_stats['avg_llm_accuracy'] = 0.0

    return avg_llm

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
                             timestamp: str = None) -> Path:
    """Build a non-ambiguous default re-evaluation run directory."""
    run_timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_model = sanitize_path_component(model_name)
    safe_protocol = sanitize_path_component(llm_judge_prompt)
    return data_dir / f"reeval_{run_timestamp}_{safe_model}_{safe_protocol}"

def process_file(file_path: Path,
                 model_name: str,
                 save_dir: Path,
                 llm_judge_prompt: str = "default",
                 disable_inner_tqdm: bool = False):
    """Process file."""
    evaluator = ReEvaluator(model_name=model_name, llm_judge_prompt=llm_judge_prompt)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'results' not in data:
            return f" Skip (No results): {file_path.name}"

        results = data['results']
        updated_count = 0
        
        iterator = results
        if not disable_inner_tqdm:
            iterator = tqdm(results, desc=f"Eval {file_path.stem}", leave=False)
        
        for item in iterator:
            new_score = evaluator.evaluate_item(item)
            if 'evaluation_scores' not in item:
                item['evaluation_scores'] = {}
            item['evaluation_scores']['llm_accuracy'] = new_score
            updated_count += 1

        avg_score = update_statistics(data, results, file_path.name)
        
        
        output_prefix = "reeval_mem0_" if llm_judge_prompt == "mem0" else "reeval_"
        output_file = save_dir / f"{output_prefix}{file_path.name}"
        saved_file = write_json_no_overwrite(data, output_file)
            
        if saved_file != output_file:
            return f" Done: {file_path.name} (Avg: {avg_score:.4f}, saved: {saved_file.name}, no overwrite)"
        return f" Done: {file_path.name} (Avg: {avg_score:.4f})"

    except Exception as e:
        import traceback
        return f" Error: {file_path.name} | {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Re-evaluate LLM Accuracy for Tri-Tower Benchmark Results (Multi-threaded)")
    parser.add_argument('--data-dir', 
                        default=str(paths.LOCOMO_TASK_EVAL_RESULTS_DIR / "locomo_tri_tower_benchmark_results"),
                        help="数据集根目录")
    parser.add_argument('--output-dir', 
                        default=None,
                        help="输出目录（默认自动生成 data-dir 下的 reeval_<时间戳>_<评估模型>_<协议> 目录）")
    parser.add_argument('--model', 
                        default="gpt-4o-mini-closeai",
                        help="评估使用的模型名称")
    parser.add_argument('--llm-judge-prompt',
                        choices=["default", "mem0"],
                        default="default",
                        help="LLM judge 提示词类型：default 为现有提示词；mem0 为 mem0/LOCOMO 对齐提示词")
    
    parser.add_argument('--target-file',
                        help="指定单个文件名进行测试")
    parser.add_argument('--max-workers',
                        type=int,
                        default=1,
                        help="并发处理的线程数 (默认为 1)")
    
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    save_dir = Path(args.output_dir) if args.output_dir else build_default_output_dir(
        data_dir=data_dir,
        model_name=args.model,
        llm_judge_prompt=args.llm_judge_prompt,
    )
    
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)

    if args.target_file:
        files = [data_dir / args.target_file]
    else:
        files = [f for f in data_dir.glob("sample_*.json") if "reeval_" not in f.name]

    if not files:
        logger.warning(f"没有在 {data_dir} 找到符合条件的文件")
        return

    num_files = len(files)
    if args.max_workers > num_files:
        logger.info(f" 指定的线程数 ({args.max_workers}) 超过文件数量 ({num_files})，自动调整为 {num_files}")
        actual_workers = num_files
    else:
        actual_workers = args.max_workers

    logger.info(
        f" 开始处理 {num_files} 个文件 | 实际线程数: {actual_workers} | "
        f"模型: {args.model} | 指标: llm_judge | prompt: {args.llm_judge_prompt}"
    )

    with ThreadPoolExecutor(max_workers=actual_workers) as executor:
        future_to_file = {
            executor.submit(
                process_file, 
                f, 
                args.model, 
                save_dir,
                args.llm_judge_prompt,
                disable_inner_tqdm=(actual_workers > 1) 
            ): f 
            for f in files
        }

        for future in tqdm(as_completed(future_to_file), total=num_files, desc="Total Progress"):
            file_path = future_to_file[future]
            try:
                result_msg = future.result()
                tqdm.write(result_msg) 
            except Exception as e:
                tqdm.write(f" Unhandled exception for {file_path.name}: {e}")

    logger.info(" 所有任务执行完毕")

if __name__ == "__main__":
    main()
