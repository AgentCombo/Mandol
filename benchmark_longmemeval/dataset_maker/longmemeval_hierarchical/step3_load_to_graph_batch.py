#!/usr/bin/env python3
"""Utilities for step3 load to graph batch."""
import os
import gc
import torch
import sys
import json
import logging
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field
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
class Step3Config:
    l0_data_dir: str = str(paths.LONGMEMEVAL_HIERARCHICAL_STEP1_DIR)
    
    output_dir: str = str(paths.LONGMEMEVAL_HIERARCHICAL_SEMANTIC_GRAPHS_DIR)
    
    
    text_embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_dim: Optional[int] = None
    
    
    build_splade: bool = True  
    splade_model: str = "naver/splade-v3"  
    splade_batch_size: int = 32  
    
    start_qa: int = 0  
    end_qa: Optional[int] = None  
    enable_parallel: bool = False
    max_workers: int = 1
    
    batch_size: int = 16  
    
    build_index: bool = True
    load_edges: bool = True  
    skip_existing: bool = True
    
    gc_interval: int = 10
    clear_cuda_cache: bool = True
    
    debug_mode: bool = False


class LongMemEvalL0GraphLoader:
    
    def __init__(self, config: Step3Config):
        self.config = config
        self.logger = self._setup_logging()
        
        self.stats = {
            'total_qa': 0,
            'processed_qa': 0,
            'skipped_qa': 0,
            'total_nodes': 0,
            'total_edges': 0,
            'failed_qa': [],
            'processing_time': 0,
            'qa_details': {}
        }
        
        self._processed_count = 0
        
        self._print_config()
        
    def _cleanup_memory(self, force: bool = False):
        """Release associated resources."""
        self._processed_count += 1
        
        if force or (self._processed_count % self.config.gc_interval == 0):
            gc.collect()
            
            if self.config.clear_cuda_cache:
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                except ImportError:
                    pass
                except Exception as e:
                    self.logger.debug(f"CUDA 缓存清理失败: {e}")
            
            self.logger.debug(f" 内存清理完成 (已处理 {self._processed_count} 个 QA)")
    
    def _setup_logging(self) -> logging.Logger:
        """Run setup logging."""
        log = logging.getLogger(f"{__name__}.L0GraphLoader")
        
        if self.config.debug_mode:
            log.setLevel(logging.DEBUG)
        else:
            log.setLevel(logging.INFO)
        
        return log
    
    def _print_config(self):
        """Run print config."""
        self.logger.info("=" * 80)
        self.logger.info(" LongMemEval Step 3: L0 层级数据图谱加载器")
        self.logger.info("=" * 80)
        self.logger.info(f" L0 数据目录: {self.config.l0_data_dir}")
        self.logger.info(f" 输出目录: {self.config.output_dir}")
        self.logger.info(f" 嵌入模型: {self.config.text_embedding_model}")
        self.logger.info(f" SPLADE 向量: {'启用' if self.config.build_splade else '禁用'}")
        if self.config.build_splade:
            self.logger.info(f"   SPLADE 模型: {self.config.splade_model}")
            self.logger.info(f"   SPLADE 批大小: {self.config.splade_batch_size}")
        self.logger.info(f" 并行处理: {'启用' if self.config.enable_parallel else '禁用'}")
        self.logger.info(f" 加载边: {'启用' if self.config.load_edges else '禁用'}")
        self.logger.info(f" 构建索引: {'启用' if self.config.build_index else '禁用'}")
        self.logger.info(f" 跳过已存在: {'启用' if self.config.skip_existing else '禁用'}")
        self.logger.info("=" * 80)
    
    def scan_available_qa(self) -> List[int]:
        """Run scan available qa."""
        l0_path = Path(self.config.l0_data_dir)
        
        if not l0_path.exists():
            self.logger.error(f" L0 数据目录不存在: {l0_path}")
            return []
        
        qa_indices = []
        
        for qa_dir in l0_path.iterdir():
            if not qa_dir.is_dir():
                continue
            
            if not qa_dir.name.startswith("qa_"):
                continue
            
            nodes_file = qa_dir / "nodes.json"
            if not nodes_file.exists():
                continue
            
            try:
                qa_index = int(qa_dir.name.split("_")[1])
                qa_indices.append(qa_index)
            except (ValueError, IndexError):
                continue
        
        qa_indices.sort()
        return qa_indices
    
    def check_qa_exists(self, qa_index: int) -> bool:
        """Validate qa exists."""
        output_dir = Path(self.config.output_dir) / f"qa_{qa_index}"
        
        semantic_map_file = output_dir / "semantic_map_data" / "memory_units.pkl"
        graph_file = output_dir / "semantic_graph.gml"
        
        return output_dir.exists() and (semantic_map_file.exists() or graph_file.exists())
    
    def load_all_qa_graphs(self) -> Dict[str, Any]:
        """Load all qa graphs."""
        start_time = time.time()
        
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        available_qa = self.scan_available_qa()
        
        if not available_qa:
            self.logger.error(" 没有找到可用的 QA 数据")
            return self.stats
        
        self.logger.info(f" 找到 {len(available_qa)} 个可用的 QA 数据")
        self.logger.info(f" QA 范围: {available_qa[0]} - {available_qa[-1]}")
        
        start_idx = self.config.start_qa
        end_idx = self.config.end_qa if self.config.end_qa else max(available_qa) + 1
        
        qa_to_process = [q for q in available_qa if start_idx <= q < end_idx]
        
        if not qa_to_process:
            self.logger.warning(f" 在指定范围 [{start_idx}, {end_idx}) 内没有可用的 QA")
            return self.stats
        
        self.stats['total_qa'] = len(qa_to_process)
        
        self.logger.info(f" 将处理 QA {qa_to_process[0]} 到 QA {qa_to_process[-1]}")
        self.logger.info(f" 共 {len(qa_to_process)} 个 QA 样本")
        
        if self.config.enable_parallel:
            self._process_parallel(qa_to_process)
        else:
            self._process_sequential(qa_to_process)
        
        self.stats['processing_time'] = time.time() - start_time
        
        
        self._save_stats()
        
        self._print_summary()
        
        return self.stats
    
    def _process_sequential(self, qa_indices: List[int]):
        """Process sequential."""
        for qa_index in tqdm(qa_indices, desc="加载图谱"):
            if self.config.skip_existing and self.check_qa_exists(qa_index):
                self.logger.debug(f" 跳过已存在的 QA {qa_index}")
                self.stats['skipped_qa'] += 1
                continue
            
            try:
                success, node_count, edge_count = self._load_single_qa_graph(qa_index)
                
                if success:
                    self.stats['processed_qa'] += 1
                    self.stats['total_nodes'] += node_count
                    self.stats['total_edges'] += edge_count
                
                self._cleanup_memory()
                    
            except Exception as e:
                self.logger.error(f" 处理 QA {qa_index} 失败: {e}")
                self.stats['failed_qa'].append({
                    'qa_index': qa_index,
                    'error': str(e)
                })
        
        self._cleanup_memory(force=True)
    
    def _process_parallel(self, qa_indices: List[int]):
        """Process parallel."""
        self.logger.info(f" 使用并行模式，最大工作线程数: {self.config.max_workers}")
        
        qa_to_process = []
        for qa_index in qa_indices:
            if self.config.skip_existing and self.check_qa_exists(qa_index):
                self.logger.debug(f" 跳过已存在的 QA {qa_index}")
                self.stats['skipped_qa'] += 1
            else:
                qa_to_process.append(qa_index)
        
        if not qa_to_process:
            self.logger.info(" 所有 QA 图谱已存在，无需处理")
            return
        
        self.logger.info(f" 需要处理 {len(qa_to_process)} 个 QA（跳过 {self.stats['skipped_qa']} 个）")
        
        effective_workers = min(self.config.max_workers, 10)
        if effective_workers < self.config.max_workers:
            self.logger.warning(
                f" 为避免显存问题，将 workers 从 {self.config.max_workers} 限制为 {effective_workers}"
            )
        
        completed_count = 0
        
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            future_to_qa = {
                executor.submit(self._load_single_qa_graph, qa_index): qa_index
                for qa_index in qa_to_process
            }
            
            with tqdm(total=len(qa_to_process), desc="加载图谱") as pbar:
                for future in as_completed(future_to_qa):
                    qa_index = future_to_qa[future]
                    
                    try:
                        success, node_count, edge_count = future.result()
                        
                        if success:
                            self.stats['processed_qa'] += 1
                            self.stats['total_nodes'] += node_count
                            self.stats['total_edges'] += edge_count
                        
                    except Exception as e:
                        self.logger.error(f" 处理 QA {qa_index} 失败: {e}")
                        self.stats['failed_qa'].append({
                            'qa_index': qa_index,
                            'error': str(e)
                        })
                    
                    pbar.update(1)
                    completed_count += 1
                    
                    if completed_count % self.config.gc_interval == 0:
                        self._cleanup_memory(force=True)
        
        self._cleanup_memory(force=True)
    
    def _load_single_qa_graph(self, qa_index: int) -> Tuple[bool, int, int]:
        """Load single qa graph."""
        semantic_graph = None
        
        try:
            self.logger.debug(f" 处理 QA {qa_index}...")
            
            
            nodes_data, edges_data, summary_data = self._load_l0_data(qa_index)
            
            if not nodes_data:
                self.logger.warning(f" QA {qa_index} 没有节点数据")
                return False, 0, 0
            
            semantic_graph = self._create_semantic_graph()
            
            
            node_count = self._load_nodes(semantic_graph, nodes_data, qa_index)
            
            
            edge_count = 0
            if self.config.load_edges and edges_data:
                edge_count = self._load_edges(semantic_graph, edges_data, qa_index)
            
            
            splade_stats = None
            if self.config.build_splade:
                try:
                    splade_stats = self._build_splade_with_memory_management(
                        semantic_graph, qa_index
                    )
                except Exception as e:
                    self.logger.warning(f" QA {qa_index} SPLADE 向量构建失败: {e}")
            
            
            if self.config.build_index:
                try:
                    semantic_graph.rebuild_all_indexes()
                except Exception as e:
                    self.logger.warning(f" QA {qa_index} 构建索引失败: {e}")
            
            
            output_path = self._save_qa_graph(semantic_graph, qa_index, nodes_data, summary_data, splade_stats)
            
            qa_stats = {
                'nodes': node_count,
                'edges': edge_count,
                'output_path': output_path
            }
            
            if splade_stats:
                qa_stats['splade'] = splade_stats
            
            if summary_data:
                qa_stats['question'] = nodes_data.get('qa_metadata', {}).get('question', '')
                qa_stats['question_type'] = nodes_data.get('qa_metadata', {}).get('question_type', '')
            
            self.stats['qa_details'][f"qa_{qa_index}"] = qa_stats
            
            self.logger.debug(f" QA {qa_index} 完成: {node_count} 节点, {edge_count} 边")
            
            return True, node_count, edge_count
            
        except Exception as e:
            self.logger.error(f" 加载 QA {qa_index} 失败: {e}")
            if self.config.debug_mode:
                import traceback
                self.logger.debug(traceback.format_exc())
            return False, 0, 0
        
        finally:
            if semantic_graph is not None:
                try:
                    if hasattr(semantic_graph, 'semantic_map'):
                        semantic_graph.semantic_map.memory_units.clear()
                        if hasattr(semantic_graph.semantic_map, 'faiss_index'):
                            semantic_graph.semantic_map.faiss_index = None
                    if hasattr(semantic_graph, 'rx_graph'):
                        import rustworkx as rx
                        semantic_graph.rx_graph = rx.PyDiGraph(multigraph=True)
                        semantic_graph._uid_to_index = {}
                        semantic_graph._index_to_uid = {}
                except Exception:
                    pass
                
                del semantic_graph
            
            nodes_data = None
            edges_data = None
            summary_data = None
    
    def _load_l0_data(self, qa_index: int) -> Tuple[Optional[Dict], Optional[Dict], Optional[Dict]]:
        """Load L0 data."""
        qa_dir = Path(self.config.l0_data_dir) / f"qa_{qa_index}"
        
        nodes_file = qa_dir / "nodes.json"
        edges_file = qa_dir / "edges.json"
        summary_file = qa_dir / "summary.json"
        
        nodes_data = None
        edges_data = None
        summary_data = None
        
        
        if nodes_file.exists():
            try:
                with open(nodes_file, 'r', encoding='utf-8') as f:
                    nodes_data = json.load(f)
            except Exception as e:
                self.logger.warning(f" 加载节点数据失败: {e}")
        
        
        if edges_file.exists():
            try:
                with open(edges_file, 'r', encoding='utf-8') as f:
                    edges_data = json.load(f)
            except Exception as e:
                self.logger.warning(f" 加载边数据失败: {e}")
        
        
        if summary_file.exists():
            try:
                with open(summary_file, 'r', encoding='utf-8') as f:
                    summary_data = json.load(f)
            except Exception as e:
                self.logger.warning(f" 加载摘要数据失败: {e}")
        
        return nodes_data, edges_data, summary_data
    
    def _create_semantic_graph(self) -> SemanticGraph:
        """Create semantic graph."""
        try:
            embedding_dim = self.config.embedding_dim
            if embedding_dim is None:
                model_config = SemanticMap.MODEL_CONFIG.get(self.config.text_embedding_model)
                if model_config:
                    embedding_dim = model_config.get('dim')
                else:
                    embedding_dim = 1024
                    self.logger.warning(f" 未知模型 {self.config.text_embedding_model}，使用默认维度 {embedding_dim}")
            
            semantic_map = SemanticMap(
                embedding_model_name=self.config.text_embedding_model,
                embedding_dim=embedding_dim,
                faiss_index_type="IDMap,Flat"
            )
            
            semantic_graph = SemanticGraph(semantic_map_instance=semantic_map)
            
            return semantic_graph
            
        except Exception as e:
            self.logger.error(f"创建 SemanticGraph 失败: {e}")
            raise
        
    def _build_splade_with_memory_management(
        self, 
        semantic_graph: SemanticGraph, 
        qa_index: int
    ) -> Optional[Dict[str, Any]]:
        """Build splade with memory management."""
        try:
            import torch
            
            effective_batch_size = min(self.config.splade_batch_size, 32)
            
            splade_stats = semantic_graph.build_sparse_embeddings(
                units=None,
                # text_field="text_content",
                model_name=self.config.splade_model,
                batch_size=effective_batch_size,
                force_rebuild=False,
                show_progress=False
            )
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.logger.debug(
                f" QA {qa_index} SPLADE 构建完成: "
                f"处理 {splade_stats.get('processed', 0)} 个单元"
            )
            
            return splade_stats
            
        except ImportError:
            return semantic_graph.build_sparse_embeddings(
                units=None,
                # text_field="text_content",
                model_name=self.config.splade_model,
                batch_size=self.config.splade_batch_size,
                force_rebuild=False,
                show_progress=False
            )
    
    def _load_nodes(self, 
                    semantic_graph: SemanticGraph, 
                    nodes_data: Dict[str, Any],
                    qa_index: int) -> int:
        """Load nodes."""
        nodes = nodes_data.get('nodes', [])
        
        if not nodes:
            return 0
        
        
        units_to_add: List[MemoryUnit] = []
        
        seen_uids = set() 
        
        
        for node in nodes:
            try:
                raw_uid = node.get('uid')
                raw_data = node.get('raw_data', {})
                metadata = node.get('metadata', {})
                
                if not raw_uid:
                    continue
                
                
                uid = raw_uid
                
                if uid in seen_uids:
                    self.logger.warning(f" [QA {qa_index}] 发现重复节点 UID: {uid}，已自动跳过重复项")
                    continue
                seen_uids.add(uid)
                
                
                memory_unit = MemoryUnit(
                    uid=uid,
                    raw_data=raw_data,
                    metadata=metadata
                )
                
                units_to_add.append(memory_unit)
                
            except Exception as e:
                self.logger.warning(f" 准备节点 {node.get('uid', 'unknown')} 失败: {e}")
        
        if units_to_add:
            self.logger.debug(f"批量添加 {len(units_to_add)} 个节点...")
            stats = semantic_graph.batch_add_units(
                units=units_to_add,
                batch_size=self.config.batch_size,
                space_names=None,
                index_update_mode="none",  
                generate_sparse_embedding=False,  
                show_progress=False
            )
            return stats.get('added', 0)
        
        return 0
    
    # def _load_nodes(self, 
    #                 semantic_graph: SemanticGraph, 
    #                 nodes_data: Dict[str, Any],
    #                 qa_index: int) -> int:
    #     """
    
    
        
    #     Args:
    
        
    #     Returns:
    
    #     """
    #     nodes = nodes_data.get('nodes', [])
        
    #     if not nodes:
    #         return 0
        
    
    #     units_to_add: List[MemoryUnit] = []
        
    
    #     for node_idx, node in enumerate(nodes):
    #         try:
    #             raw_uid = node.get('uid')
    #             raw_data = node.get('raw_data', {})
    #             metadata = node.get('metadata', {})
                
    #             if not raw_uid:
    #                 continue
                
    
    
    #             uid = f"{raw_uid}_{node_idx}"
                
    #             memory_unit = MemoryUnit(
    #                 uid=uid,
    #                 raw_data=raw_data,
    #                 metadata=metadata
    #             )
                
    #             units_to_add.append(memory_unit)
                
    #         except Exception as e:
    
        
    #     if units_to_add:
    #         stats = semantic_graph.batch_add_units(
    #             units=units_to_add,
    #             batch_size=self.config.batch_size,
    
    
    #             show_progress=False
    #         )
    #         return stats.get('added', 0)
        
    #     return 0
    
    def _load_edges(self,
                    semantic_graph: SemanticGraph,
                    edges_data: Dict[str, Any],
                    qa_index: int) -> int:
        """Load edges."""
        edges = edges_data.get('edges', [])
        
        if not edges:
            return 0
        
        edge_count = 0
        
        for edge in edges:
            try:
                source_uid = edge.get('source_uid')
                target_uid = edge.get('target_uid')
                relation_type = edge.get('relation_type', 'REPLY_TO')
                properties = edge.get('properties', {})
                
                if not source_uid or not target_uid:
                    continue
                
                if semantic_graph.get_unit(source_uid) is None:
                    self.logger.debug(f"源节点不存在: {source_uid}")
                    continue
                
                if semantic_graph.get_unit(target_uid) is None:
                    self.logger.debug(f"目标节点不存在: {target_uid}")
                    continue
                
                semantic_graph.add_relationship(
                    source_uid=source_uid,
                    target_uid=target_uid,
                    relationship_name=relation_type,
                    bidirectional=False,
                    **properties
                )
                
                edge_count += 1
                
            except Exception as e:
                self.logger.warning(f" 加载边失败: {e}")
        
        return edge_count
    
    def _save_qa_graph(self,
                   semantic_graph: SemanticGraph,
                   qa_index: int,
                   nodes_data: Dict[str, Any],
                   summary_data: Optional[Dict[str, Any]],
                   splade_stats: Optional[Dict[str, Any]] = None) -> str:
        """Save qa graph."""
        output_dir = Path(self.config.output_dir) / f"qa_{qa_index}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            
            semantic_graph.save_graph(str(output_dir))
            
            
            qa_metadata = nodes_data.get('qa_metadata', {})
            info = nodes_data.get('info', {})
            statistics = nodes_data.get('statistics', {})
            
            
            units_with_splade = sum(
                1 for u in semantic_graph.get_all_units() 
                if u.has_sparse_embedding()
            )
            
            metadata = {
                'qa_index': qa_index,
                'qa_id': f"qa_{qa_index}",
                'created_at': datetime.now().isoformat(),
                'loader_version': '1.1.0',  
                'layer': 'L0',
                'qa_metadata': qa_metadata,
                'source_info': info,
                'source_statistics': statistics,
                'embedding_model': {
                    'name': self.config.text_embedding_model,
                    'dimension': semantic_graph.semantic_map.embedding_dim if hasattr(semantic_graph.semantic_map, 'embedding_dim') else None
                },
                'splade_model': {
                    'enabled': self.config.build_splade,
                    'name': self.config.splade_model if self.config.build_splade else None,
                    'units_with_splade': units_with_splade,
                    'stats': splade_stats
                },
                'config': {
                    'build_index': self.config.build_index,
                    'load_edges': self.config.load_edges,
                    'build_splade': self.config.build_splade
                },
                'graph_stats': {
                    'node_count': len(semantic_graph.get_all_units()),
                    'edge_count': semantic_graph.rx_graph.num_edges() if hasattr(semantic_graph, 'rx_graph') else 0,
                    'units_with_embedding': sum(1 for u in semantic_graph.get_all_units() if u.embedding is not None),
                    'units_with_splade': units_with_splade
                }
            }
            
            metadata_file = output_dir / "graph_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            return str(output_dir)
            
        except Exception as e:
            self.logger.error(f"保存 QA {qa_index} 图谱失败: {e}")
            raise
    
    def _save_stats(self):
        """Save stats."""
        stats_file = Path(self.config.output_dir) / "loading_stats.json"
        
        try:
            enhanced_stats = self.stats.copy()
            enhanced_stats['config'] = {
                'l0_data_dir': self.config.l0_data_dir,
                'output_dir': self.config.output_dir,
                'text_embedding_model': self.config.text_embedding_model,
                'start_qa': self.config.start_qa,
                'end_qa': self.config.end_qa,
                'enable_parallel': self.config.enable_parallel,
                'max_workers': self.config.max_workers,
                'build_index': self.config.build_index,
                'load_edges': self.config.load_edges
            }
            enhanced_stats['completion_time'] = datetime.now().isoformat()
            
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(enhanced_stats, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f" 统计信息已保存: {stats_file}")
            
        except Exception as e:
            self.logger.error(f"保存统计信息失败: {e}")
    
    def _print_summary(self):
        """Run print summary."""
        print("\n" + "=" * 80)
        print("L0 层级图谱加载摘要 (Step 3)")
        print("=" * 80)
        print(f"总 QA 数:       {self.stats['total_qa']}")
        print(f"处理成功:       {self.stats['processed_qa']}")
        print(f"跳过已存在:     {self.stats['skipped_qa']}")
        print(f"处理失败:       {len(self.stats['failed_qa'])}")
        print(f"总节点数:       {self.stats['total_nodes']}")
        print(f"总边数:         {self.stats['total_edges']}")
        print(f"处理时间:       {self.stats['processing_time']:.2f} 秒")
        
        if self.stats['processed_qa'] > 0:
            avg_time = self.stats['processing_time'] / self.stats['processed_qa']
            print(f"平均耗时:       {avg_time:.2f} 秒/QA")
        
        print(f"嵌入模型:       {self.config.text_embedding_model}")
        print(f"SPLADE 模型:    {self.config.splade_model if self.config.build_splade else '未启用'}")
        print(f"输出目录:       {self.config.output_dir}")
        
        if self.stats['failed_qa']:
            print(f"\n 失败的 QA:")
            for failed in self.stats['failed_qa'][:10]:
                print(f"  - QA {failed['qa_index']}: {failed['error']}")
            if len(self.stats['failed_qa']) > 10:
                print(f"  ... 还有 {len(self.stats['failed_qa']) - 10} 个")
        
        print("=" * 80 + "\n")


def main():
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(
        description="LongMemEval Step 3: 加载 L0 层级数据到 SemanticGraph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        示例:
            # 处理所有 QA（使用默认配置）
            python step3_load_to_graph.py

            # 处理指定范围的 QA
            python step3_load_to_graph.py --start-qa 1 --end-qa 100

            # 使用指定的嵌入模型
            python step3_load_to_graph.py --embedding-model "Qwen/Qwen3-Embedding-4B"

            # 单线程模式（推荐用于大数据集，避免显存问题）
            python step3_load_to_graph.py --no-parallel

            # 更频繁的内存清理（每 5 个 QA 清理一次）
            python step3_load_to_graph.py --gc-interval 5

            # 不构建 SPLADE 向量（大幅减少显存占用）
            python step3_load_to_graph.py --no-splade
        """
    )
    
    parser.add_argument("--l0-dir", type=str,
                       default=str(paths.LONGMEMEVAL_HIERARCHICAL_STEP1_DIR),
                       help="L0 数据目录（step1 输出）")
    parser.add_argument("--output-dir", type=str,
                       default=str(paths.LONGMEMEVAL_HIERARCHICAL_STEP3_DIR),
                       help="输出目录")
    
    parser.add_argument("--embedding-model", type=str,
                       default="Qwen/Qwen3-Embedding-0.6B",
                       help="文本嵌入模型（默认: Qwen/Qwen3-Embedding-0.6B）")
    parser.add_argument("--embedding-dim", type=int, default=None,
                       help="嵌入维度（默认自动推断）")
    
    
    parser.add_argument("--no-splade", action="store_true",
                       help="不构建 SPLADE 稀疏向量")
    parser.add_argument("--splade-model", type=str,
                       default="naver/splade-v3",
                       help="SPLADE 模型（默认: naver/splade-v3）")
    parser.add_argument("--splade-batch-size", type=int, default=16,
                       help="SPLADE 批处理大小（默认: 16）")
    
    parser.add_argument("--start-qa", type=int, default=0,
                       help="起始 QA 索引（默认: 0）")
    parser.add_argument("--end-qa", type=int, default=None,
                       help="结束 QA 索引（默认: 处理所有）")
    
    parser.add_argument("--no-parallel", action="store_true",
                       help="禁用并行处理（推荐用于避免显存问题）")
    parser.add_argument("--max-workers", type=int, default=1,
                       help="最大工作线程数（默认: 1）")
    
    parser.add_argument("--gc-interval", type=int, default=1,
                       help="垃圾回收间隔（每处理多少个 QA）（默认: 1）")
    parser.add_argument("--no-cuda-cache-clear", action="store_true",
                       help="禁用 CUDA 缓存清理")
    
    parser.add_argument("--no-index", action="store_true",
                       help="不构建索引")
    parser.add_argument("--no-edges", action="store_true",
                       help="不加载边（关系）")
    parser.add_argument("--no-skip", action="store_true",
                       help="不跳过已存在的图谱（重新处理）")
    
    parser.add_argument("--debug", action="store_true",
                       help="启用调试模式")
    
    args = parser.parse_args()
    
    config = Step3Config(
        l0_data_dir=args.l0_dir,
        output_dir=args.output_dir,
        text_embedding_model=args.embedding_model,
        embedding_dim=args.embedding_dim,
        build_splade=not args.no_splade,
        splade_model=args.splade_model,
        splade_batch_size=args.splade_batch_size,
        start_qa=args.start_qa,
        end_qa=args.end_qa,
        enable_parallel=not args.no_parallel,
        max_workers=args.max_workers,
        build_index=not args.no_index,
        load_edges=not args.no_edges,
        skip_existing=not args.no_skip,
        gc_interval=args.gc_interval,
        clear_cuda_cache=not args.no_cuda_cache_clear,
        debug_mode=args.debug
    )
    
    # Avoid mutating LogRecord fields before other handlers process the record.
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    
    loader = LongMemEvalL0GraphLoader(config)
    
    try:
        stats = loader.load_all_qa_graphs()
        
        print("\n L0 层级图谱加载完成!")
        print(f" 输出目录: {config.output_dir}")
        print(f" 统计文件: {config.output_dir}/loading_stats.json")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n 用户中断")
        return 130
        
    except Exception as e:
        print(f"\n 图谱加载失败: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1
    
    finally:
        try:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass


if __name__ == "__main__":
    sys.exit(main())