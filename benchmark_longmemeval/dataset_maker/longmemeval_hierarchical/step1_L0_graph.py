#!/usr/bin/env python3
"""Utilities for step1 L0 graph."""
import os
import json
import sys
import re
import argparse
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from tqdm import tqdm
from mandol.core import paths

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    
    from langchain.text_splitter import RecursiveCharacterTextSplitter

# Avoid mutating LogRecord fields before other handlers process the record.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LongMemEvalL0NodeGenerator:
    
    def __init__(self,
                 dataset_path: str = str(paths.LONGMEMEVAL_S_CLEANED_FILE),
                 output_dir: str = str(paths.LONGMEMEVAL_HIERARCHICAL_STEP1_DIR),
                 chunk_size: int = 512,
                 chunk_overlap: int = 50):
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=[
                "\n\n",
                "\n",
                "。",
                ". ",
                "！",
                "! ",
                "？",
                "? ",
                "；",
                "; ",
                "，",
                ", ",
                " ",
                ""
            ]
        )
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f" 数据集路径: {self.dataset_path}")
        logger.info(f" 输出根目录: {self.output_dir}")
        logger.info(f" 分块配置: chunk_size={chunk_size} chars, overlap={chunk_overlap} chars")
        logger.info(f" 架构策略: 纯净切分 + 图关系驱动 + 独立 QA 保存")
    
    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load dataset."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"数据集文件不存在: {self.dataset_path}")
        
        logger.info(f" 加载数据集: {self.dataset_path}")
        
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f" 加载完成，共 {len(data)} 个 QA 样本")
        return data
    
    def process_single_qa(self, 
                         qa_data: Dict[str, Any], 
                         qa_index: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
        """Process single qa."""
        nodes = []
        edges = []
        
        question_id = qa_data.get("question_id", f"qa_{qa_index}")
        question = qa_data.get("question", "")
        answer = qa_data.get("answer", "")
        question_type = qa_data.get("question_type", "unknown")
        
        sessions = qa_data.get("haystack_sessions", [])
        session_ids = qa_data.get("haystack_session_ids", [])
        session_dates = qa_data.get("haystack_dates", [])
        
        total_sessions = len(sessions)
        
        if len(session_ids) != total_sessions:
            logger.warning(f" QA {qa_index}: session_ids 数量不匹配")
            session_ids = session_ids + [
                f"session_{i}" for i in range(len(session_ids), total_sessions)
            ]
        
        if len(session_dates) != total_sessions:
            logger.warning(f" QA {qa_index}: dates 数量不匹配")
            session_dates = session_dates + [None] * (total_sessions - len(session_dates))
        
        stats = {
            "total_messages": 0,
            "total_chunks": 0,
            "user_messages": 0,
            "assistant_messages": 0,
            "chunked_assistant_messages": 0,
            "total_edges": 0
        }
        
        
        last_user_uid = None
        
        for s_idx, messages in enumerate(sessions):
            curr_session_id = session_ids[s_idx] if s_idx < len(session_ids) else f"session_{s_idx}"
            curr_date = session_dates[s_idx] if s_idx < len(session_dates) else ""
            
            for msg_idx, msg in enumerate(messages):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                
                if not content or not content.strip():
                    continue
                
                stats["total_messages"] += 1
                
                if role.lower() == "user":
                    stats["user_messages"] += 1
                    
                    node_uid = f"qa{qa_index}_s{s_idx}_msg{msg_idx}"
                    last_user_uid = node_uid  
                    
                    user_node = {
                        "uid": node_uid,
                        "raw_data": {
                            "text_content": content,
                            "role": "user",
                            "session_id": curr_session_id,
                            "session_index": s_idx,
                            "message_index": msg_idx,
                            "session_date": curr_date,
                            "qa_index": qa_index,
                            "question_id": question_id,
                            "question_type": question_type
                        },
                        "metadata": {
                            "created": str(datetime.now()),
                            "qa_index": qa_index,
                            "session_id": curr_session_id,
                            "session_date": curr_date,
                            "role": "user",
                            "node_type": "user_message",
                            "is_chunk": False
                        }
                    }
                    nodes.append(user_node)
                
                elif role.lower() == "assistant":
                    stats["assistant_messages"] += 1
                    
                    chunks = self.splitter.split_text(content)
                    
                    if len(chunks) > 1:
                        stats["chunked_assistant_messages"] += 1
                    
                    stats["total_chunks"] += len(chunks)
                    
                    for chunk_idx, chunk_text in enumerate(chunks):
                        chunk_uid = f"qa{qa_index}_s{s_idx}_msg{msg_idx}_c{chunk_idx}"
                        
                        assistant_node = {
                            "uid": chunk_uid,
                            "raw_data": {
                                "text_content": chunk_text,
                                "role": "assistant",
                                "session_id": curr_session_id,
                                "session_index": s_idx,
                                "message_index": msg_idx,
                                "chunk_index": chunk_idx,
                                "total_chunks": len(chunks),
                                "session_date": curr_date,
                                "qa_index": qa_index,
                                "question_id": question_id,
                                "question_type": question_type,
                                "is_chunk": len(chunks) > 1,
                                "parent_msg_id": f"qa{qa_index}_s{s_idx}_msg{msg_idx}"
                            },
                            "metadata": {
                                "created": str(datetime.now()),
                                "qa_index": qa_index,
                                "session_id": curr_session_id,
                                "session_date": curr_date,
                                "role": "assistant",
                                "node_type": "assistant_chunk",
                                "chunk_index": chunk_idx,
                                "total_chunks": len(chunks),
                                "is_chunk": len(chunks) > 1
                            }
                        }
                        nodes.append(assistant_node)
                        
                        if last_user_uid:
                            edge = {
                                "source_uid": chunk_uid,
                                "target_uid": last_user_uid,
                                "relation_type": "REPLY_TO",
                                "properties": {
                                    "created": str(datetime.now()),
                                    "session_id": curr_session_id,
                                    "qa_index": qa_index,
                                    "message_index": msg_idx,
                                    "chunk_index": chunk_idx
                                }
                            }
                            edges.append(edge)
                            stats["total_edges"] += 1
        
        logger.debug(f" QA {qa_index} 处理完成: "
                    f"{stats['total_messages']} 条原始消息 -> {len(nodes)} 个节点 + {len(edges)} 条边 "
                    f"(分块消息: {stats['chunked_assistant_messages']}, 总块数: {stats['total_chunks']})")
        
        return nodes, edges, stats
    
    def save_qa_data(self, 
                     qa_index: int,
                     nodes: List[Dict[str, Any]],
                     edges: List[Dict[str, Any]],
                     stats: Dict[str, int],
                     qa_metadata: Dict[str, Any]) -> bool:
        """Save qa data."""
        qa_dir = self.output_dir / f"qa_{qa_index}"
        qa_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            
            nodes_file = qa_dir / "nodes.json"
            nodes_output = {
                "info": {
                    "generated_at": str(datetime.now()),
                    "qa_index": qa_index,
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                    "total_nodes": len(nodes),
                    "architecture": "Pure Text + Graph Relations (Zero Redundancy)"
                },
                "qa_metadata": qa_metadata,
                "statistics": stats,
                "nodes": nodes
            }
            
            with open(nodes_file, 'w', encoding='utf-8') as f:
                json.dump(nodes_output, f, ensure_ascii=False, indent=2)
            
            
            edges_file = qa_dir / "edges.json"
            edges_output = {
                "info": {
                    "generated_at": str(datetime.now()),
                    "qa_index": qa_index,
                    "total_edges": len(edges),
                    "edge_types": ["REPLY_TO"]
                },
                "edges": edges
            }
            
            with open(edges_file, 'w', encoding='utf-8') as f:
                json.dump(edges_output, f, ensure_ascii=False, indent=2)
            
            
            summary_file = qa_dir / "summary.json"
            summary = {
                "qa_index": qa_index,
                "question_id": qa_metadata.get("question_id"),
                "question_type": qa_metadata.get("question_type"),
                "generated_at": str(datetime.now()),
                "statistics": stats,
                "files": {
                    "nodes": str(nodes_file.name),
                    "edges": str(edges_file.name)
                }
            }
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            logger.debug(f" QA {qa_index} 已保存到: {qa_dir}")
            return True
            
        except Exception as e:
            logger.error(f" 保存 QA {qa_index} 失败: {e}")
            return False
    
    def check_qa_exists(self, qa_index: int) -> bool:
        """Validate qa exists."""
        qa_dir = self.output_dir / f"qa_{qa_index}"
        nodes_file = qa_dir / "nodes.json"
        edges_file = qa_dir / "edges.json"
        
        return nodes_file.exists() and edges_file.exists()
    
    def process_dataset(self, 
                       limit: Optional[int] = None,
                       start_index: int = 1,
                       skip_existing: bool = True) -> Dict[str, Any]:
        """Process dataset."""
        start_time = datetime.now()
        
        logger.info(f"\n{'='*80}")
        logger.info(f" 开始处理 LongMemEval 数据集（L0 节点生成 - 纯净切分版）")
        logger.info(f" 分块配置: chunk_size={self.chunk_size} chars, overlap={self.chunk_overlap} chars")
        logger.info(f" 架构策略: 纯净文本 + 图关系驱动（零冗余）")
        logger.info(f" 保存策略: 独立 QA 目录")
        if skip_existing:
            logger.info(f" 断点续传: 启用（跳过已存在的 QA）")
        logger.info(f"{'='*80}")
        
        
        qa_samples = self.load_dataset()
        
        if limit:
            qa_samples = qa_samples[:limit]
            logger.info(f" 限制处理数量: {limit}")
        
        total_samples = len(qa_samples)
        success_count = 0
        failed_count = 0
        skipped_count = 0
        failed_indices = []
        
        global_stats = {
            "total_messages": 0,
            "total_chunks": 0,
            "total_nodes": 0,
            "total_edges": 0,
            "user_messages": 0,
            "assistant_messages": 0,
            "chunked_assistant_messages": 0,
            "total_sessions": 0
        }
        
        logger.info(f"\n 开始处理 {total_samples} 个 QA 样本...\n")
        
        for idx, qa_data in enumerate(tqdm(qa_samples, desc="处理进度"), start=start_index):
            try:
                if skip_existing and self.check_qa_exists(idx):
                    logger.debug(f" QA {idx} 已存在，跳过")
                    skipped_count += 1
                    continue
                
                qa_nodes, qa_edges, qa_stats = self.process_single_qa(qa_data, qa_index=idx)
                
                qa_metadata = {
                    "question_id": qa_data.get("question_id", f"qa_{idx}"),
                    "question": qa_data.get("question", ""),
                    "answer": qa_data.get("answer", ""),
                    "question_type": qa_data.get("question_type", "unknown"),
                    "total_sessions": len(qa_data.get("haystack_sessions", []))
                }
                
                
                save_success = self.save_qa_data(
                    qa_index=idx,
                    nodes=qa_nodes,
                    edges=qa_edges,
                    stats=qa_stats,
                    qa_metadata=qa_metadata
                )
                
                if save_success:
                    global_stats["total_messages"] += qa_stats["total_messages"]
                    global_stats["total_chunks"] += qa_stats["total_chunks"]
                    global_stats["user_messages"] += qa_stats["user_messages"]
                    global_stats["assistant_messages"] += qa_stats["assistant_messages"]
                    global_stats["chunked_assistant_messages"] += qa_stats["chunked_assistant_messages"]
                    global_stats["total_edges"] += qa_stats["total_edges"]
                    global_stats["total_sessions"] += len(qa_data.get("haystack_sessions", []))
                    global_stats["total_nodes"] += len(qa_nodes)
                    
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
        
        
        self._save_global_report(
            total_samples=total_samples,
            success_count=success_count,
            failed_count=failed_count,
            skipped_count=skipped_count,
            failed_indices=failed_indices,
            processing_time=processing_time,
            global_stats=global_stats
        )
        
        logger.info(f"\n{'='*80}")
        logger.info(f" 数据集处理完成!")
        logger.info(f"{'='*80}")
        logger.info(f" 处理统计:")
        logger.info(f"   - 总 QA 样本: {total_samples}")
        logger.info(f"   - 成功: {success_count}")
        logger.info(f"   - 跳过: {skipped_count}")
        logger.info(f"   - 失败: {failed_count}")
        logger.info(f"   - 总会话数: {global_stats['total_sessions']}")
        logger.info(f"   - 原始消息数: {global_stats['total_messages']}")
        logger.info(f"     • User 消息: {global_stats['user_messages']}")
        logger.info(f"     • Assistant 消息: {global_stats['assistant_messages']}")
        logger.info(f"   - 生成节点数: {global_stats['total_nodes']}")
        logger.info(f"   - 生成边数: {global_stats['total_edges']} (REPLY_TO)")
        logger.info(f"   - 分块统计:")
        logger.info(f"     • 被分块的 Assistant 消息: {global_stats['chunked_assistant_messages']}")
        logger.info(f"     • 生成的总块数: {global_stats['total_chunks']}")
        if global_stats['chunked_assistant_messages'] > 0:
            avg_chunks = global_stats['total_chunks'] / global_stats['chunked_assistant_messages']
            logger.info(f"     • 平均每个分块消息: {avg_chunks:.1f} 块")
        
        logger.info(f"\nPerformance statistics:")
        logger.info(f"   - 总耗时: {processing_time:.2f} 秒")
        if success_count > 0:
            logger.info(f"   - 平均: {processing_time / success_count:.2f} 秒/样本")
        
        logger.info(f"\n 架构优势:")
        logger.info(f"   -  文本零冗余（User 问题不重复存储）")
        logger.info(f"   -  向量语义纯净（无拼接污染）")
        logger.info(f"   -  检索时通过图遍历回溯上下文")
        logger.info(f"   -  独立 QA 保存，支持断点续传")
        logger.info(f"   -  适合 5000万+ token 规模")
        
        if failed_indices:
            logger.info(f"\n 失败的索引: {failed_indices[:10]}{'...' if len(failed_indices) > 10 else ''}")
        
        logger.info(f"\n 输出目录: {self.output_dir}")
        logger.info(f"   每个 QA 保存在独立的 qa_X/ 子目录中")
        logger.info(f"{'='*80}\n")
        
        return {
            "total_samples": total_samples,
            "success_count": success_count,
            "failed_count": failed_count,
            "skipped_count": skipped_count,
            "failed_indices": failed_indices,
            "processing_time": processing_time,
            "global_stats": global_stats
        }
    
    def _save_global_report(self,
                           total_samples: int,
                           success_count: int,
                           failed_count: int,
                           skipped_count: int,
                           failed_indices: List[int],
                           processing_time: float,
                           global_stats: Dict[str, int]):
        """Save global report."""
        report_file = self.output_dir / "processing_report.json"
        
        report = {
            "generated_at": str(datetime.now()),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "architecture": "Pure Text + Graph Relations (Zero Redundancy)",
            "total_samples": total_samples,
            "success_count": success_count,
            "failed_count": failed_count,
            "skipped_count": skipped_count,
            "failed_indices": failed_indices,
            "processing_time": processing_time,
            "avg_time_per_sample": processing_time / success_count if success_count > 0 else 0,
            "statistics": global_stats
        }
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            logger.info(f" 全局报告已保存: {report_file}")
        except Exception as e:
            logger.error(f" 保存全局报告失败: {e}")


def load_single_qa(qa_dir: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Load single qa."""
    nodes_file = qa_dir / "nodes.json"
    edges_file = qa_dir / "edges.json"
    
    if not nodes_file.exists() or not edges_file.exists():
        raise FileNotFoundError(f"QA 数据不完整: {qa_dir}")
    
    with open(nodes_file, 'r', encoding='utf-8') as f:
        nodes_data = json.load(f)
    
    with open(edges_file, 'r', encoding='utf-8') as f:
        edges_data = json.load(f)
    
    return nodes_data["nodes"], edges_data["edges"], nodes_data["qa_metadata"]


def main():
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(description="LongMemEval L0 节点生成器（纯净切分 + 图关系 + 独立保存）")
    
    parser.add_argument("--dataset-path",
                       default=str(paths.LONGMEMEVAL_S_CLEANED_FILE),
                       help="数据集文件路径")
    parser.add_argument("--output-dir",
                       default=str(paths.LONGMEMEVAL_HIERARCHICAL_STEP1_DIR),
                       help="输出根目录")
    
    parser.add_argument("--chunk-size", type=int, default=512,
                       help="分块大小（字符数），默认 512")
    parser.add_argument("--chunk-overlap", type=int, default=50,
                       help="重叠大小（字符数），默认 50")
    
    parser.add_argument("--limit", type=int, default=None,
                       help="限制处理的 QA 数量（用于测试）")
    parser.add_argument("--start-index", type=int, default=0,
                       help="起始索引（默认从 0 开始）")
    
    parser.add_argument("--no-skip", action="store_true",
                       help="不跳过已存在的 QA（重新处理所有）")
    
    parser.add_argument("--debug", action="store_true",
                       help="启用调试模式")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        generator = LongMemEvalL0NodeGenerator(
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
        
        result = generator.process_dataset(
            limit=args.limit,
            start_index=args.start_index,
            skip_existing=not args.no_skip
        )
        
        return 0 if result["failed_count"] == 0 else 1
        
    except Exception as e:
        logger.error(f" 程序异常: {e}")
        if args.debug:
            import traceback
            logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())
