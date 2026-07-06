#!/usr/bin/env python3
"""Utilities for raw longmemeval generator."""
import os
import json
import sys
import argparse
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from tqdm import tqdm


from mandol.core.semantic_graph import SemanticGraph
from mandol.core.memory_unit import MemoryUnit
from mandol.core import paths

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LongMemEvalDatasetLoader:
    
    def __init__(self,
                 dataset_path: str = str(paths.LONGMEMEVAL_S_CLEANED_FILE),
                 output_dir: str = str(paths.LONGMEMEVAL_RAW_DIR)):
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f" 数据集路径: {self.dataset_path}")
        logger.info(f" 输出目录: {self.output_dir}")
    
    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load dataset."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"数据集文件不存在: {self.dataset_path}")
        
        logger.info(f" 加载数据集: {self.dataset_path}")
        
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f" 加载完成，共 {len(data)} 个 QA 样本")
        return data
    
    def create_semantic_graph_for_qa(self, 
                                 qa_data: Dict[str, Any],
                                 qa_index: int,
                                 build_splade: bool = True) -> SemanticGraph:
        """Build semantic graph for qa."""
        logger.debug(f" 为 QA {qa_index} 构建 SemanticGraph...")
        
        semantic_graph = SemanticGraph()
        
        question_id = qa_data.get("question_id", f"qa_{qa_index}")
        question = qa_data.get("question", "")
        answer = qa_data.get("answer", "")
        question_type = qa_data.get("question_type", "unknown")
        
        haystack_sessions = qa_data.get("haystack_sessions", [])
        haystack_session_ids = qa_data.get("haystack_session_ids", [])
        haystack_dates = qa_data.get("haystack_dates", [])
        
        total_sessions = len(haystack_sessions)
        total_messages = sum(len(session) for session in haystack_sessions)
        
        if len(haystack_session_ids) != total_sessions:
            logger.warning(f" QA {qa_index}: session_ids 数量不匹配")
            haystack_session_ids = haystack_session_ids + [
                f"session_{i}" for i in range(len(haystack_session_ids), total_sessions)
            ]
        
        if len(haystack_dates) != total_sessions:
            logger.warning(f" QA {qa_index}: dates 数量不匹配")
            haystack_dates = haystack_dates + [None] * (total_sessions - len(haystack_dates))
        
        message_count = 0
        
        for session_idx, session_messages in enumerate(haystack_sessions):
            session_id = haystack_session_ids[session_idx]
            session_date = haystack_dates[session_idx]
            
            for msg_idx, message in enumerate(session_messages):
                role = message.get("role", "unknown")
                content = message.get("content", "")
                
                if not content or not content.strip():
                    continue
                
                
                message_uid = f"qa{qa_index}_session{session_idx}_msg{msg_idx}"
                
                memory_unit = MemoryUnit(
                    uid=message_uid,
                    raw_data={
                        "text_content": content,
                        "role": role,
                        "session_id": session_id,
                        "session_index": session_idx,
                        "message_index": msg_idx,
                        "session_date": session_date,
                        "qa_index": qa_index,
                        "question_id": question_id,
                        "question_type": question_type
                    },
                    metadata={
                        "created": str(datetime.now()),
                        "qa_index": qa_index,
                        "session_id": session_id,
                        "session_date": session_date,
                        "role": role
                    }
                )
                
                semantic_graph.add_unit(memory_unit)
                message_count += 1
        
        
        if build_splade and message_count > 0:
            try:
                logger.debug(f" 为 QA {qa_index} 预构建稀疏向量（SPLADE）...")
                
                
                splade_stats = semantic_graph.build_sparse_embeddings(
                    text_field="text_content",
                    model_name="naver/splade-v3",
                    batch_size=32,
                    force_rebuild=False,
                    show_progress=False
                )
                
                logger.debug(
                    f"    稀疏向量构建完成: "
                    f"总计={splade_stats.get('total', 0)}, "
                    f"成功={splade_stats.get('processed', 0)}, "
                    f"失败={splade_stats.get('failed', 0)}"
                )
                
            except Exception as e:
                logger.warning(f" 为 QA {qa_index} 构建稀疏向量失败: {e}")
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(traceback.format_exc())
        
        
        semantic_graph.build_semantic_map_index()
        
        logger.debug(f" QA {qa_index} SemanticGraph 构建完成: "
                    f"{message_count} 条消息, {total_sessions} 个 sessions")
        
        return semantic_graph
    
    def save_semantic_graph(self, 
                           semantic_graph: SemanticGraph,
                           qa_index: int) -> bool:
        """Save semantic graph."""
        qa_dir = self.output_dir / f"qa_{qa_index}"
        qa_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            semantic_graph.save_graph(str(qa_dir))
            logger.debug(f" QA {qa_index} 已保存到: {qa_dir}")
            return True
        except Exception as e:
            logger.error(f" 保存 QA {qa_index} 失败: {e}")
            return False
    
    def process_dataset(self, 
                   limit: Optional[int] = None,
                   start_index: int = 0,
                   build_splade: bool = True,
                   splade_batch_mode: bool = False) -> Dict[str, Any]:
        """Process dataset."""
        start_time = datetime.now()
        
        logger.info(f"\n{'='*80}")
        logger.info(f" 开始处理 LongMemEval 数据集（消息级别粒度）")
        if build_splade:
            logger.info(f" 稀疏向量（SPLADE）预构建: 启用 (模式: {'批量' if splade_batch_mode else '逐个'})")
        else:
            logger.info(f" 稀疏向量（SPLADE）预构建: 禁用")
        logger.info(f"{'='*80}")
        
        
        qa_samples = self.load_dataset()
        
        if limit:
            qa_samples = qa_samples[:limit]
            logger.info(f" 限制处理数量: {limit}")
        
        total_samples = len(qa_samples)
        success_count = 0
        failed_count = 0
        failed_indices = []
        
        total_messages_processed = 0
        total_sessions_processed = 0
        total_splade_vectors_built = 0
        total_splade_failed = 0
        
        logger.info(f"\n 开始处理 {total_samples} 个 QA 样本...\n")
        
        for idx, qa_data in enumerate(tqdm(qa_samples, desc="处理进度"), start=start_index):
            try:
                semantic_graph = self.create_semantic_graph_for_qa(
                    qa_data=qa_data,
                    qa_index=idx,
                    build_splade=build_splade
                )
                
                messages_in_qa = len(semantic_graph.get_all_units())
                sessions_in_qa = len(qa_data.get("haystack_sessions", []))
                
                total_messages_processed += messages_in_qa
                total_sessions_processed += sessions_in_qa
                
                
                if build_splade:
                    units_with_sparse = sum(
                        1 for unit in semantic_graph.get_all_units() 
                        if unit.has_sparse_embedding()
                    )
                    total_splade_vectors_built += units_with_sparse
                    total_splade_failed += (messages_in_qa - units_with_sparse)
                
                
                if self.save_semantic_graph(semantic_graph, idx):
                    success_count += 1
                else:
                    failed_count += 1
                    failed_indices.append(idx)
                    
            except Exception as e:
                failed_count += 1
                failed_indices.append(idx)
                logger.error(f" 处理 QA {idx} 失败: {e}")
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(traceback.format_exc())
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        stats = {
            "total_samples": total_samples,
            "success_count": success_count,
            "failed_count": failed_count,
            "failed_indices": failed_indices,
            "total_messages_processed": total_messages_processed,
            "total_sessions_processed": total_sessions_processed,
            "processing_time": processing_time,
            "avg_time_per_sample": processing_time / total_samples if total_samples > 0 else 0,
            "output_dir": str(self.output_dir),
            
            "splade_enabled": build_splade,
            "total_splade_vectors_built": total_splade_vectors_built,
            "total_splade_failed": total_splade_failed,
            "splade_success_rate": (
                total_splade_vectors_built / total_messages_processed * 100 
                if total_messages_processed > 0 else 0
            )
        }
        
        logger.info(f"\n{'='*80}")
        logger.info(f" 数据集处理完成!")
        logger.info(f"{'='*80}")
        logger.info(f" 处理统计:")
        logger.info(f"   - 总 QA 样本: {total_samples}")
        logger.info(f"   - 成功: {success_count}")
        logger.info(f"   - 失败: {failed_count}")
        logger.info(f"   - 总消息数: {total_messages_processed}")
        logger.info(f"   - 总会话数: {total_sessions_processed}")
        logger.info(f"   - 平均每个 QA: {total_messages_processed/total_samples:.1f} 条消息")
        
        
        if build_splade:
            logger.info(f"\n 稀疏向量（SPLADE）统计:")
            logger.info(f"   - 成功构建: {total_splade_vectors_built}")
            logger.info(f"   - 构建失败: {total_splade_failed}")
            logger.info(f"   - 成功率: {stats['splade_success_rate']:.2f}%")
        
        logger.info(f"\nPerformance statistics:")
        logger.info(f"   - 总耗时: {processing_time:.2f} 秒")
        logger.info(f"   - 平均: {stats['avg_time_per_sample']:.2f} 秒/样本")
        
        if failed_indices:
            logger.info(f"\n 失败的索引: {failed_indices[:10]}{'...' if len(failed_indices) > 10 else ''}")
        
        logger.info(f"\n 输出目录: {self.output_dir}")
        logger.info(f"{'='*80}\n")
        
        
        self._save_processing_report(stats)
        
        return stats
    
    def _save_processing_report(self, stats: Dict[str, Any]):
        """Save processing report."""
        report_file = self.output_dir / "processing_report.json"
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            logger.info(f" 处理报告已保存: {report_file}")
        except Exception as e:
            logger.warning(f" 保存处理报告失败: {e}")


def main():
    """Run the command-line entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="LongMemEval 数据集加载器（消息级别粒度）")
    
    parser.add_argument("--dataset-path",
                       default=str(paths.LONGMEMEVAL_S_CLEANED_FILE),
                       help="数据集文件路径")
    parser.add_argument("--output-dir",
                       default=str(paths.LONGMEMEVAL_RAW_MESSAGE_LEVEL_DIR),
                       help="输出目录路径")
    
    parser.add_argument("--limit", type=int, default=None,
                       help="限制处理的 QA 数量（用于测试）")
    parser.add_argument("--start-index", type=int, default=0,
                       help="起始索引（默认从 0 开始）")
    
    
    parser.add_argument("--build-splade", action="store_true", default=True,
                       help="预构建稀疏向量（SPLADE）（默认启用）")
    parser.add_argument("--no-splade", action="store_true",
                       help="禁用稀疏向量（SPLADE）预构建")
    parser.add_argument("--splade-batch-mode", action="store_true",
                       help="使用批量模式构建稀疏向量（更快但内存占用更高）")
    
    parser.add_argument("--debug", action="store_true",
                       help="启用调试模式")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    
    build_splade = args.build_splade and not args.no_splade
    
    try:
        loader = LongMemEvalDatasetLoader(
            dataset_path=args.dataset_path,
            output_dir=args.output_dir
        )
        
        stats = loader.process_dataset(
            limit=args.limit,
            start_index=args.start_index,
            build_splade=build_splade,
            splade_batch_mode=args.splade_batch_mode
        )
        
        return 0 if stats["failed_count"] == 0 else 1
        
    except Exception as e:
        logger.error(f" 程序异常: {e}")
        if args.debug:
            import traceback
            logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())
