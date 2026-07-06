#!/usr/bin/env python3
"""Utilities for step4 saved in semantic map batch."""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


from mandol.core.semantic_graph import SemanticGraph
from mandol.core.semantic_map import SemanticMap
from mandol.core.memory_unit import MemoryUnit
from mandol.core import paths

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Step4Config:
    episodic_dir: str = str(paths.LONGMEMEVAL_EPISODIC_NEW_DEDUPLICATED_DIR)
    
    output_dir: str = str(paths.LONGMEMEVAL_EPISODIC_GRAPHS_DIR)
    
    
    text_embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_dim: Optional[int] = None
    
    
    build_splade: bool = True
    splade_model: str = "naver/splade-v3"
    splade_batch_size: int = 32
    
    start_qa: int = 0
    end_qa: Optional[int] = None
    enable_parallel: bool = False
    max_workers: int = 1
    
    batch_size: int = 32  
    
    build_index: bool = True
    
    debug_mode: bool = False


class LongMemEvalEpisodicLoader:
    
    def __init__(self, config: Step4Config):
        self.config = config
        self.logger = self._setup_logging()
        
        self.stats = {
            'total_qa': 0,
            'processed_qa': 0,
            'total_events': 0,
            'failed_qa': [],
            'processing_time': 0
        }
        
        self.logger.info("=" * 80)
        self.logger.info(" LongMemEval Step 4: 情景记忆加载器 (Episodic Memory)")
        self.logger.info("=" * 80)
        self.logger.info(f" 输入目录: {self.config.episodic_dir}")
        self.logger.info(f" 输出目录: {self.config.output_dir}")
        self.logger.info(f" 嵌入模型: {self.config.text_embedding_model}")
    
    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger(f"{__name__}.EpisodicLoader")
        if self.config.debug_mode:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
        return logger
    
    def load_all_episodic_graphs(self):
        """Load all episodic graphs."""
        start_time = time.time()
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        self.logger.info(" 正在预热模型 (防止并行锁竞争)...")
        try:
            from mandol.utils.model_manager import global_model_manager
            
            temp_map = SemanticMap(
                embedding_model_name=self.config.text_embedding_model,
                embedding_dim=self.config.embedding_dim
            )
            del temp_map 
            
            if self.config.build_splade:
                global_model_manager.get_splade_model(self.config.splade_model)
            self.logger.info(" 模型预热完成")
        except Exception as e:
            self.logger.warning(f" 模型预热异常 (尝试继续): {e}")
        

        dataset_dir = Path(self.config.episodic_dir)
        if not dataset_dir.exists():
            self.logger.error(f"数据集目录不存在: {dataset_dir}")
            return
        
        
        qa_files = sorted(list(dataset_dir.glob("qa_*.json")))
        
        qa_indices = []
        file_map = {} # index -> file_path
        
        for f in qa_files:
            try:
                parts = f.stem.split('_') # ['qa', '0', ...]
                if len(parts) >= 2 and parts[0] == 'qa' and parts[1].isdigit():
                    idx = int(parts[1])
                    qa_indices.append(idx)
                    if idx in file_map:
                        if 'deduplicated' in f.name:
                            file_map[idx] = f
                    else:
                        file_map[idx] = f
            except Exception:
                continue
                
        qa_indices = sorted(list(set(qa_indices)))
        
        if self.config.start_qa is not None:
            qa_indices = [idx for idx in qa_indices if idx >= self.config.start_qa]
        if self.config.end_qa is not None:
            qa_indices = [idx for idx in qa_indices if idx <= self.config.end_qa]
            
        self.stats['total_qa'] = len(qa_indices)
        self.logger.info(f" 待处理 QA 数量: {len(qa_indices)}")
        if qa_indices:
            self.logger.info(f" 处理范围: qa_{qa_indices[0]} - qa_{qa_indices[-1]}")

        if self.config.enable_parallel:
            self._process_parallel(qa_indices, file_map)
        else:
            self._process_sequential(qa_indices, file_map)
            
        self.stats['processing_time'] = time.time() - start_time
        self._print_summary()

    def _process_parallel(self, qa_indices: List[int], file_map: Dict[int, Path]):
        self.logger.info(f" 并行处理 (Workers={self.config.max_workers})...")
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(self._load_single_qa, idx, file_map[idx]): idx
                for idx in qa_indices
            }
            with tqdm(total=len(qa_indices), desc="Processing QA") as pbar:
                for future in as_completed(futures):
                    qa_idx = futures[future]
                    try:
                        success = future.result()
                        if success: self.stats['processed_qa'] += 1
                        else: self.stats['failed_qa'].append(qa_idx)
                    except Exception as e:
                        self.logger.error(f"Error qa_{qa_idx}: {e}")
                        self.stats['failed_qa'].append(qa_idx)
                    pbar.update(1)

    def _process_sequential(self, qa_indices: List[int], file_map: Dict[int, Path]):
        self.logger.info(" 顺序处理模式...")
        for idx in tqdm(qa_indices, desc="Processing QA"):
            if self._load_single_qa(idx, file_map[idx]):
                self.stats['processed_qa'] += 1
            else:
                self.stats['failed_qa'].append(idx)

    def _load_single_qa(self, qa_index: int, file_path: Path) -> bool:
        try:
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            qa_space_name = f"qa_{qa_index}"
            semantic_map = SemanticMap(
                embedding_model_name=self.config.text_embedding_model,
                embedding_dim=self.config.embedding_dim
            )
            semantic_graph = SemanticGraph(semantic_map_instance=semantic_map)
            semantic_graph.create_memory_space_in_map(qa_space_name)
            
            
            event_count = self._load_events(
                semantic_graph, 
                data, 
                qa_index, 
                qa_space_name
            )
            
            if event_count == 0:
                self.logger.warning(f"qa_{qa_index}: 没有找到事件数据 (file: {file_path.name})")
            
            
            splade_stats = None
            if self.config.build_splade and event_count > 0:
                splade_stats = semantic_graph.build_sparse_embeddings(
                    model_name=self.config.splade_model,
                    batch_size=self.config.splade_batch_size,
                    show_progress=False
                )
            
            
            if self.config.build_index and event_count > 0:
                semantic_graph.rebuild_all_indexes()

            
            output_path = Path(self.config.output_dir) / f"qa_{qa_index}"
            output_path.mkdir(parents=True, exist_ok=True)
            semantic_graph.save_graph(str(output_path))
            
            
            self._save_metadata(output_path, qa_index, event_count, splade_stats)
            
            self.stats['total_events'] += event_count
            return True
            
        except Exception as e:
            self.logger.error(f"Failed qa_{qa_index}: {e}", exc_info=True)
            return False

    def _load_events(self, 
                    semantic_graph: SemanticGraph, 
                    data: Dict[str, Any], 
                    qa_index: int, 
                    qa_space_name: str) -> int:
        """Load events."""
        qa_id = data.get('qa_id', f'qa_{qa_index}')
        
        events = []
        if isinstance(data, list):
            events = data
        else:
            events = data.get('facts', 
                        data.get('events', 
                            data.get('episodes', 
                                data.get('history', []))))
        
        if not events:
            return 0
        
        
        units_to_add: List[MemoryUnit] = []
        
        for idx, event in enumerate(events):
            raw_event_id = event.get('uid', 
                                event.get('event_id', 
                                    event.get('id', f'event_{idx}')))
            
            
            if str(raw_event_id).startswith(f"{qa_id}_"):
                event_uid = f"{raw_event_id}_{idx}"
            else:
                event_uid = f"{qa_id}_{raw_event_id}_{idx}"
            
            content = event.get('canonical_content', 
                            event.get('content', 
                                event.get('summary', 
                                    event.get('text', ''))))
            if not content:
                continue
                
            event_date = (
                event.get('temporal_val') or
                event.get('session_date') or 
                event.get('date') or 
                event.get('time') or 
                event.get('summary_time') or 
                event.get('timestamp') or
                str(event.get('temporal_info', ''))
            )
            
            category = event.get('category', 'EPISODIC_EVENT')
            session_id = event.get('session_id', '')
            
            text_content = content
            
            raw_data = {
                "node_type": category.lower() if category else "episodic_event", 
                "qa_id": qa_id,
                "event_id": raw_event_id,
                "content": content,
                "event_date": event_date,
                "category": category,
                "session_id": session_id,
                "text_content": text_content, 
                "original_data": event,
                "created_at": datetime.now().isoformat()
            }
            
            unit = MemoryUnit(uid=event_uid, raw_data=raw_data)
            units_to_add.append(unit)
        
        if units_to_add:
            self.logger.debug(f"批量添加 {len(units_to_add)} 个事件单元...")
            stats = semantic_graph.batch_add_units(
                units=units_to_add,
                batch_size=self.config.batch_size,
                space_names=[qa_space_name],
                index_update_mode="none",  
                generate_sparse_embedding=False,  
                show_progress=False
            )
            return stats.get('added', 0)
        
        return 0

    def _save_metadata(self, output_path: Path, qa_index: int, event_count: int, splade_stats: Any):
        meta = {
            "qa_index": qa_index,
            "type": "episodic_memory",
            "event_count": event_count,
            "saved_at": datetime.now().isoformat(),
            "config": {
                "embedding_model": self.config.text_embedding_model,
                "build_splade": self.config.build_splade
            },
            "splade_stats": splade_stats
        }
        with open(output_path / "meta_info.json", 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
            
    def _print_summary(self):
        self.logger.info("\n" + "="*60)
        self.logger.info(f" 处理完成: {self.stats['processed_qa']}/{self.stats['total_qa']}")
        self.logger.info(f" 总事件数: {self.stats['total_events']}")
        self.logger.info(f"Elapsed time: {self.stats['processing_time']:.2f}s")
        if self.stats['failed_qa']:
            self.logger.info(f" 失败列表: {self.stats['failed_qa']}")
        self.logger.info("="*60 + "\n")


def main():
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(
        description="LongMemEval Step 4: 加载情景记忆 (Episodic Memory) 到 SemanticGraph",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--episodic-dir",
                       default=str(paths.LONGMEMEVAL_EPISODIC_NEW_DEDUPLICATED_DIR),
                       help="情景记忆数据目录")
    parser.add_argument("--output-dir",
                       default=str(paths.LONGMEMEVAL_EPISODIC_GRAPHS_DIR),
                       help="输出目录")
    
    parser.add_argument("--embedding-model", type=str,
                       default="Qwen/Qwen3-Embedding-0.6B",
                       help="文本嵌入模型名称")
    parser.add_argument("--embedding-dim", type=int, default=None,
                       help="嵌入向量维度（默认: 自动推断）")
    
    
    parser.add_argument("--build-splade", action="store_true", default=True,
                       help="启用 SPLADE 稀疏向量构建（默认启用）")
    parser.add_argument("--no-splade", action="store_true",
                       help="禁用 SPLADE 稀疏向量")
    parser.add_argument("--splade-model", type=str,
                       default="naver/splade-v3",
                       help="SPLADE 模型名称")
    parser.add_argument("--splade-batch-size", type=int, default=32,
                       help="SPLADE 批处理大小")
    
    parser.add_argument("--start-qa", type=int, default=0,
                       help="起始 QA 索引（默认: 0）")
    parser.add_argument("--end-qa", type=int, default=None,
                       help="结束 QA 索引（默认: None 表示处理所有）")
    
    parser.add_argument("--no-parallel", action="store_true",
                       help="禁用并行处理")
    parser.add_argument("--max-workers", type=int, default=1,
                       help="最大工作线程数（默认: 1）")
    
    parser.add_argument("--no-index", action="store_true",
                       help="不构建索引")
    parser.add_argument("--debug", action="store_true",
                       help="启用调试模式")
    
    args = parser.parse_args()
    
    
    build_splade = not args.no_splade
    
    config = Step4Config(
        episodic_dir=args.episodic_dir,
        output_dir=args.output_dir,
        text_embedding_model=args.embedding_model,
        embedding_dim=args.embedding_dim,
        build_splade=build_splade,
        splade_model=args.splade_model,
        splade_batch_size=args.splade_batch_size,
        start_qa=args.start_qa,
        end_qa=args.end_qa,
        enable_parallel=not args.no_parallel,
        max_workers=args.max_workers,
        build_index=not args.no_index,
        debug_mode=args.debug
    )
    
    loader = LongMemEvalEpisodicLoader(config)
    loader.load_all_episodic_graphs()

if __name__ == "__main__":
    main()
