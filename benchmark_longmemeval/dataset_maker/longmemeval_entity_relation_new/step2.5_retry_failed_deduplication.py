#!/usr/bin/env python3
"""Utilities for step2.5 retry failed deduplication."""
import json
import logging
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Set, Optional

from json_repair import repair_json


from step2_entity_deduplication import LongMemEvalEntityDeduplicator
from mandol.core import paths

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def scan_missing_qa_indices(
    dedup_dir: str,
    start_index: int = 0,
    end_index: int = 500
) -> List[int]:
    """Run scan missing qa indices."""
    dedup_path = Path(dedup_dir)
    
    if not dedup_path.exists():
        logger.error(f" 去重目录不存在: {dedup_dir}")
        return list(range(start_index, end_index))
    
    existing_indices: Set[int] = set()
    
    for file_path in dedup_path.glob("qa_*_deduplicated.json"):
        try:
            
            filename = file_path.stem  # qa_123_deduplicated
            parts = filename.split('_')
            if len(parts) >= 2 and parts[0] == 'qa':
                qa_index = int(parts[1])
                existing_indices.add(qa_index)
        except (ValueError, IndexError) as e:
            logger.warning(f" 无法解析文件名: {file_path.name}, 错误: {e}")
            continue
    
    
    all_indices = set(range(start_index, end_index))
    missing_indices = sorted(all_indices - existing_indices)
    
    logger.info(f"\n{'='*80}")
    logger.info(f" 去重结果扫描报告")
    logger.info(f"{'='*80}")
    logger.info(f"扫描范围: {start_index} - {end_index-1}")
    logger.info(f"已完成: {len(existing_indices)} 个QA")
    logger.info(f"缺失: {len(missing_indices)} 个QA")
    
    if missing_indices:
        
        if len(missing_indices) <= 30:
            logger.info(f"缺失索引: {missing_indices}")
        else:
            logger.info(f"缺失索引(前20个): {missing_indices[:20]}...")
            logger.info(f"缺失索引(后10个): ...{missing_indices[-10:]}")
    
    logger.info(f"{'='*80}\n")
    
    return missing_indices


def scan_empty_or_invalid_qa(
    dedup_dir: str,
    start_index: int = 0,
    end_index: int = 500
) -> List[int]:
    """Run scan empty or invalID qa."""
    dedup_path = Path(dedup_dir)
    invalid_indices: List[int] = []
    
    for qa_index in range(start_index, end_index):
        file_path = dedup_path / f"qa_{qa_index}_deduplicated.json"
        
        if not file_path.exists():
            continue
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, dict):
                logger.warning(f" qa_{qa_index}: 不是有效的JSON对象")
                invalid_indices.append(qa_index)
                continue
            
            if 'entities' not in data:
                logger.warning(f" qa_{qa_index}: 缺少 'entities' 字段")
                invalid_indices.append(qa_index)
                continue
            
            if not isinstance(data['entities'], list):
                logger.warning(f" qa_{qa_index}: 'entities' 不是列表")
                invalid_indices.append(qa_index)
                continue
            
            if len(data['entities']) == 0:
                logger.warning(f" qa_{qa_index}: entities 为空")
                invalid_indices.append(qa_index)
                continue
                
        except json.JSONDecodeError as e:
            logger.warning(f" qa_{qa_index}: JSON解析失败 - {e}")
            invalid_indices.append(qa_index)
        except Exception as e:
            logger.warning(f" qa_{qa_index}: 读取失败 - {e}")
            invalid_indices.append(qa_index)
    
    if invalid_indices:
        logger.info(f" 发现 {len(invalid_indices)} 个无效/空文件: {invalid_indices[:20]}{'...' if len(invalid_indices) > 20 else ''}")
    
    return invalid_indices


def load_entities_for_qa_indices(
    batch_results_dir: str,
    qa_indices: List[int],
    exclude_dirs: Optional[List[str]] = None
) -> Dict[str, List[Dict]]:
    """Load entities for qa indices."""
    if exclude_dirs is None:
        exclude_dirs = ['deprecated']
    
    results_path = Path(batch_results_dir)
    if not results_path.exists():
        raise FileNotFoundError(f"批量结果目录不存在: {batch_results_dir}")
    
    target_qa_ids = {f"qa_{idx}" for idx in qa_indices}
    
    logger.info(f"\n 开始加载 {len(qa_indices)} 个QA的实体数据...")
    
    qa_entities = {}
    
    jsonl_files = []
    for file_path in results_path.glob("*.jsonl"):
        is_excluded = any(excl in file_path.parts for excl in exclude_dirs)
        if not is_excluded:
            jsonl_files.append(file_path)
    
    jsonl_files.sort()
    
    logger.info(f"找到 {len(jsonl_files)} 个JSONL文件")
    
    for jsonl_file in jsonl_files:
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    
                    try:
                        result = json.loads(line)
                        custom_id = result.get("custom_id", "")
                        
                        if not custom_id.startswith("qa_"):
                            continue
                        
                        parts = custom_id.split("_")
                        if len(parts) < 2:
                            continue
                        
                        qa_id = f"qa_{parts[1]}"
                        
                        if qa_id not in target_qa_ids:
                            continue
                        
                        try:
                            response_body = result["response"]["body"]
                            message_content = response_body["choices"][0]["message"]["content"]
                            
                            entities_data = _safe_parse_json(message_content)
                            if entities_data is None:
                                continue
                            
                            if not isinstance(entities_data, dict):
                                continue
                            
                            entities = entities_data.get("entities", [])
                            if not isinstance(entities, list):
                                continue
                            
                            valid_entities = []
                            for entity in entities:
                                if isinstance(entity, dict) and _validate_entity(entity):
                                    valid_entities.append(entity)
                            
                            if valid_entities:
                                if qa_id not in qa_entities:
                                    qa_entities[qa_id] = []
                                qa_entities[qa_id].extend(valid_entities)
                        
                        except (KeyError, IndexError, TypeError):
                            continue
                            
                    except json.JSONDecodeError:
                        continue
        
        except Exception as e:
            logger.error(f" 加载文件 {jsonl_file.name} 失败: {e}")
            continue
    
    loaded_count = len(qa_entities)
    missing_qa_ids = target_qa_ids - set(qa_entities.keys())
    
    logger.info(f" 成功加载 {loaded_count} 个QA的实体数据")
    
    if missing_qa_ids:
        logger.warning(f" {len(missing_qa_ids)} 个QA在批量结果中未找到: {sorted([int(qa.split('_')[1]) for qa in missing_qa_ids])[:20]}")
    
    return qa_entities


def _safe_parse_json(content: str) -> Optional[Dict]:
    """Run safe parse JSON."""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    
    try:
        repaired = repair_json(content)
        return json.loads(repaired)
    except Exception:
        pass
    
    try:
        start_idx = content.find('{')
        if start_idx == -1:
            return None
        
        brace_count = 0
        end_idx = -1
        for i in range(start_idx, len(content)):
            if content[i] == '{':
                brace_count += 1
            elif content[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i + 1
                    break
        
        if end_idx == -1:
            return None
        
        json_str = content[start_idx:end_idx]
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            repaired = repair_json(json_str)
            return json.loads(repaired)
        
    except Exception:
        return None


def _validate_entity(entity: Dict) -> bool:
    """Validate entity."""
    if not isinstance(entity, dict):
        return False
    
    if not entity.get('name') or not entity.get('type'):
        return False
    
    name = entity.get('name', '')
    if len(name) > 500:
        return False
    
    entity_type = entity.get('type', '')
    if len(entity_type) > 100:
        return False
    
    return True


def main():
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(
        description="LongMemEval实体去重器 - Step 2.5: 重试失败的去重",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        示例:
        # 扫描并显示缺失的QA（仅检查不处理）
        python step2.5_retry_failed_deduplication.py --scan-only

        # 重试所有缺失的QA
        python step2.5_retry_failed_deduplication.py --retry

        # 重试指定范围的缺失QA
        python step2.5_retry_failed_deduplication.py --retry --start-index 0 --end-index 100

        # 同时检查空文件和无效文件
        python step2.5_retry_failed_deduplication.py --scan-only --include-invalid

        # 使用指定的LLM模型
        python step2.5_retry_failed_deduplication.py --retry --llm-model deepseek-v3.2-dashscope

        # 指定输入输出目录
        python step2.5_retry_failed_deduplication.py --retry \
            --dedup-dir path/to/dedup \
            --results-dir path/to/batch_results
        """
    )
    
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--scan-only", action="store_true",
                           help="仅扫描缺失的QA,不进行处理")
    mode_group.add_argument("--retry", action="store_true",
                           help="重试处理缺失的QA")
    
    parser.add_argument("--dedup-dir", type=str,
                       default=str(paths.LONGMEMEVAL_ENTITY_RELATION_NEW_DEDUPLICATED_LLM_DIR),
                       help="去重结果目录")
    parser.add_argument("--results-dir", type=str,
                       default=str(paths.LONGMEMEVAL_ENTITY_RELATION_NEW_BATCH_RESULTS_DIR),
                       help="批量推理结果目录")
    parser.add_argument("--exclude-dirs", nargs="+", default=["deprecated"],
                       help="排除的子目录")
    
    parser.add_argument("--start-index", type=int, default=0,
                       help="起始QA索引(默认0)")
    parser.add_argument("--end-index", type=int, default=500,
                       help="结束QA索引(不包含,默认500)")
    
    parser.add_argument("--include-invalid", action="store_true",
                       help="同时检查并重试无效/空的QA文件")
    parser.add_argument("--specific-indices", type=int, nargs="+",
                       help="指定要重试的QA索引(覆盖自动扫描)")
    
    parser.add_argument("--llm-model", type=str, default="deepseek-v3.2-dashscope",
                       help="LLM模型名称")
    parser.add_argument("--llm-api-key", type=str, default=None,
                       help="LLM API密钥")
    parser.add_argument("--llm-base-url", type=str, default=None,
                       help="LLM API基础URL")
    
    parser.add_argument("--no-per-qa-optimization", action="store_true",
                       help="禁用逐QA参数优化")
    parser.add_argument("--no-llm-dedup", action="store_true",
                       help="禁用LLM精细去重")
    parser.add_argument("--llm-cluster-threshold", type=int, default=2,
                       help="触发LLM去重的聚类大小阈值")
    parser.add_argument("--large-cluster-threshold", type=int, default=12,
                       help="大聚类阈值")
    parser.add_argument("--parallel-workers", type=int, default=40,
                       help="并行线程数")
    
    parser.add_argument("--debug", action="store_true",
                       help="启用调试模式")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        
        if args.specific_indices:
            
            qa_indices = args.specific_indices
            logger.info(f" 使用用户指定的 {len(qa_indices)} 个QA索引")
        else:
            
            missing_indices = scan_missing_qa_indices(
                args.dedup_dir,
                args.start_index,
                args.end_index
            )
            
            if args.include_invalid:
                invalid_indices = scan_empty_or_invalid_qa(
                    args.dedup_dir,
                    args.start_index,
                    args.end_index
                )
                qa_indices = sorted(set(missing_indices + invalid_indices))
            else:
                qa_indices = missing_indices
        
        if not qa_indices:
            logger.info(f" 没有需要重试的QA,所有QA都已成功去重!")
            return 0
        
        logger.info(f"\n 需要处理的QA数量: {len(qa_indices)}")
        
        if args.scan_only:
            print(f"\n{'='*80}")
            print(f" 扫描结果汇总")
            print(f"{'='*80}")
            print(f"缺失/无效QA数量: {len(qa_indices)}")
            print(f"索引列表: {qa_indices}")
            print(f"{'='*80}\n")
            return 0
        
        logger.info(f"\n 开始重试失败的QA去重...")
        
        
        qa_entities = load_entities_for_qa_indices(
            args.results_dir,
            qa_indices,
            args.exclude_dirs
        )
        
        if not qa_entities:
            logger.error(f" 没有找到任何需要重试的QA的实体数据")
            return 1
        
        deduplicator = LongMemEvalEntityDeduplicator(
            llm_model=args.llm_model,
            llm_api_key=args.llm_api_key,
            llm_base_url=args.llm_base_url,
            optimize_per_qa=not args.no_per_qa_optimization,
            use_llm_dedup=not args.no_llm_dedup,
            llm_cluster_threshold=args.llm_cluster_threshold,
            large_cluster_threshold=args.large_cluster_threshold,
            parallel_workers=args.parallel_workers
        )
        
        summary = deduplicator.process_qa_batch(
            qa_entities,
            args.dedup_dir,
            start_index=0,
            end_index=None
        )
        
        print(f"\n{'='*80}")
        print(f" 重试完成!")
        print(f"{'='*80}")
        print(f" 统计信息:")
        print(f"  尝试重试: {len(qa_indices)} 个QA")
        print(f"  成功加载: {len(qa_entities)} 个QA")
        print(f"  成功处理: {summary.get('completed_count', 0)} 个QA")
        print(f"  失败: {summary.get('failed_count', 0)} 个QA")
        
        if summary.get('failed_qa_ids'):
            print(f"  失败列表: {summary['failed_qa_ids']}")
        
        print(f"\n 输出目录: {args.dedup_dir}")
        print(f"{'='*80}\n")
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning(f"\n 用户中断")
        return 130
        
    except Exception as e:
        logger.error(f" 程序异常: {e}")
        if args.debug:
            import traceback
            logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())