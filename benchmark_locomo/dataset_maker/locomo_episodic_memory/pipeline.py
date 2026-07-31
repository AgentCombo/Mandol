"""Utilities for pipeline."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional


from benchmark_locomo.dataset_maker.locomo_episodic_memory.step1_extract_episodic_facts import (
    EpisodicFactExtractor, EpisodicFactConfig
)
from benchmark_locomo.dataset_maker.locomo_episodic_memory.step2_deduplicate_and_enhance import (
    FactDeduplicator, DeduplicateConfig
)
from benchmark_locomo.dataset_maker.locomo_episodic_memory.deprecated.step3_load_to_retrieval import (
    EpisodicMemoryLoader, LoadConfig
)
from mandol.core import paths


def setup_logging(debug: bool = False):
    """Run setup logging."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def run_pipeline(
    input_file: str = str(paths.LOCOMO_RAW_FILE),
    output_base_dir: str = str(paths.LOCOMO_EPISODIC_DIR),
    llm_model: str = "deepseek-reasoner",
    max_workers: int = 6,
    sample_ids: List[str] = None,
    skip_step1: bool = False,
    skip_step2: bool = False,
    skip_step3: bool = False,
    debug: bool = False
) -> Dict:
    """Run pipeline."""
    setup_logging(debug)
    logger = logging.getLogger(__name__)
    
    stats = {
        'step1': None,
        'step2': None,
        'step3': None
    }
    
    step1_output = f"{output_base_dir}/step1_facts"
    step2_output = f"{output_base_dir}/step2_enhanced"
    step3_output = f"{output_base_dir}/step3_loaded"
    
    if not skip_step1:
        logger.info("=" * 80)
        logger.info(" Step 1: 情景事实抽取")
        logger.info("=" * 80)
        
        config1 = EpisodicFactConfig(
            locomo_file_path=input_file,
            output_dir=step1_output,
            llm_model=llm_model,
            max_workers=max_workers,
            sample_ids=sample_ids,
            debug_mode=debug
        )
        
        extractor = EpisodicFactExtractor(config1)
        stats['step1'] = extractor.run()
    
    if not skip_step2:
        logger.info("=" * 80)
        logger.info(" Step 2: 事实去重与增强")
        logger.info("=" * 80)
        
        config2 = DeduplicateConfig(
            input_dir=step1_output,
            output_dir=step2_output,
            llm_model="deepseek-v3.2-dashscope",
            debug_mode=debug
        )
        
        deduplicator = FactDeduplicator(config2)
        stats['step2'] = deduplicator.run()
    
    
    if not skip_step3:
        logger.info("=" * 80)
        logger.info(" Step 3: 加载到检索系统")
        logger.info("=" * 80)
        
        config3 = LoadConfig(
            input_dir=step2_output,
            output_dir=step3_output,
            debug_mode=debug
        )
        
        loader = EpisodicMemoryLoader(config3)
        stats['step3'] = loader.run()
    
    logger.info("\n" + "=" * 80)
    logger.info(" 情景记忆塔生成完成!")
    logger.info("=" * 80)
    
    if stats['step1']:
        logger.info(f" Step 1 - 抽取事实: {stats['step1'].get('facts_extracted', 0)}")
    if stats['step2']:
        logger.info(f" Step 2 - 去重后事实: {stats['step2'].get('facts_after', 0)}")
        logger.info(f" Step 2 - 累积事实: {stats['step2'].get('accumulations_created', 0)}")
        logger.info(f" Step 2 - 时间线: {stats['step2'].get('timelines_created', 0)}")
    if stats['step3']:
        logger.info(f" Step 3 - 索引条目: {stats['step3'].get('index_entries', 0)}")
    
    logger.info(f"\n 输出目录: {output_base_dir}")
    
    return stats


def main():
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(
        description="情景记忆塔生成 Pipeline"
    )
    
    parser.add_argument(
        "--input-file",
        default=str(paths.LOCOMO_RAW_FILE),
        help="输入的locomo数据文件"
    )
    parser.add_argument(
        "--output-dir",
        default=str(paths.LOCOMO_EPISODIC_DIR),
        help="输出基础目录"
    )
    parser.add_argument(
        "--llm-model",
        default="deepseek-reasoner",
        help="LLM模型名称"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=6,
        help="最大并行工作线程数"
    )
    parser.add_argument(
        "--sample-ids",
        nargs='+',
        help="指定处理的样本ID列表"
    )
    parser.add_argument(
        "--skip-step1",
        action="store_true",
        help="跳过Step 1 (使用已有的事实抽取结果)"
    )
    parser.add_argument(
        "--skip-step2",
        action="store_true",
        help="跳过Step 2 (使用已有的去重结果)"
    )
    parser.add_argument(
        "--skip-step3",
        action="store_true",
        help="跳过Step 3"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="启用调试模式"
    )
    
    args = parser.parse_args()
    
    run_pipeline(
        input_file=args.input_file,
        output_base_dir=args.output_dir,
        llm_model=args.llm_model,
        max_workers=args.max_workers,
        sample_ids=args.sample_ids,
        skip_step1=args.skip_step1,
        skip_step2=args.skip_step2,
        skip_step3=args.skip_step3,
        debug=args.debug
    )


if __name__ == "__main__":
    main()
