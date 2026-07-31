"""Utilities for step3 load to retrieval batch."""

from __future__ import annotations

import json
import logging
import os
import sys
import gc
import argparse
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass
from mandol.core import paths


try:
    from mandol.core.semantic_graph import SemanticGraph
    from mandol.core.semantic_map import SemanticMap
    from mandol.core.memory_unit import MemoryUnit
    SEMANTIC_GRAPH_AVAILABLE = True
    SEMANTIC_GRAPH_IMPORT_ERROR = None
except ImportError as import_error:
    SEMANTIC_GRAPH_AVAILABLE = False
    SEMANTIC_GRAPH_IMPORT_ERROR = import_error

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False



@dataclass
class LoadConfig:
    input_dir: str = str(paths.LOCOMO_EPISODIC_STEP2_DIR)
    
    output_dir: str = str(paths.LOCOMO_EPISODIC_STEP3_DIR)
    
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    
    
    build_splade: bool = True
    splade_model: str = "naver/splade-v3"
    splade_batch_size: int = 32
    
    batch_size: int = 100
    
    
    enable_index_building: bool = True
    freeze_retrievers: bool = True
    
    debug_mode: bool = False




class EpisodicMemoryLoader:
    
    def __init__(self, config: LoadConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.stats = {
            'samples_loaded': 0,
            'facts_loaded': 0,
            'accumulations_loaded': 0,
            'timelines_loaded': 0,
            'index_entries': 0,
            'failed_samples': [],
            'sample_details': {},
            'splade_stats': {
                'enabled': config.build_splade,
                'model': config.splade_model if config.build_splade else None,
                'total_processed': 0,
                'total_skipped': 0,
                'total_failed': 0
            }
        }
        
        self.logger.info(f"情景记忆加载器已初始化")
        self.logger.info(f"输入目录: {self.config.input_dir}")
        self.logger.info(f"输出目录: {self.config.output_dir}")
        self.logger.info(f"嵌入模型: {self.config.embedding_model}")
        
        if self.config.build_splade:
            self.logger.info(f" SPLADE 向量: 启用")
            self.logger.info(f"  ├─ 模型: {self.config.splade_model}")
            self.logger.info(f"  batch size: {self.config.splade_batch_size}")
        else:
            self.logger.info(f" SPLADE 向量: 禁用")
        self.logger.info(
            f" BM25/SPLADE 静态索引冻结: {'启用' if self.config.freeze_retrievers else '禁用'}"
        )
        
        if not SEMANTIC_GRAPH_AVAILABLE:
            self.logger.warning(f"SemanticGraph不可用，将仅生成JSON索引: {SEMANTIC_GRAPH_IMPORT_ERROR}")
    
    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger(f"{__name__}.EpisodicMemoryLoader")
        logger.setLevel(logging.DEBUG if self.config.debug_mode else logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _cleanup_resources(self, semantic_graph: Optional[SemanticGraph] = None):
        """Release associated resources."""
        try:
            if semantic_graph is not None:
                
                if hasattr(semantic_graph, 'semantic_map') and hasattr(semantic_graph.semantic_map, 'faiss_index'):
                    semantic_graph.semantic_map.faiss_index = None
                
                if hasattr(semantic_graph, 'semantic_map'):
                    semantic_graph.semantic_map.memory_units.clear()
                
                if hasattr(semantic_graph, 'rx_graph'):
                    import rustworkx as rx
                    semantic_graph.rx_graph = rx.PyDiGraph(multigraph=True)
                    semantic_graph._uid_to_index = {}
                    semantic_graph._index_to_uid = {}
                
                del semantic_graph
            
            gc.collect()
            
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            self.logger.debug(" 资源清理完成")
            
        except Exception as e:
            self.logger.warning(f"清理资源时出错: {e}")
    
    def _create_semantic_graph(self) -> SemanticGraph:
        """Create semantic graph."""
        try:
            model_config = SemanticMap.MODEL_CONFIG.get(self.config.embedding_model)
            
            if model_config:
                embedding_dim = model_config['dim']
                self.logger.info(f"使用预设模型: {self.config.embedding_model}, 维度: {embedding_dim}")
            else:
                self.logger.warning(f"未知模型 '{self.config.embedding_model}'，将尝试自动推断维度")
                embedding_dim = None
            
            semantic_map = SemanticMap(
                embedding_model_name=self.config.embedding_model,
                embedding_dim=embedding_dim,
                faiss_index_type="IDMap,Flat"
            )
            
            semantic_graph = SemanticGraph(semantic_map_instance=semantic_map)
            
            self.logger.info(f" SemanticGraph实例创建成功 (模型: {self.config.embedding_model}, 维度: {semantic_map.embedding_dim})")
            return semantic_graph
            
        except Exception as e:
            self.logger.error(f" 创建SemanticGraph失败: {e}")
            raise
    
    def _build_splade_embeddings(self, 
                                 semantic_graph: SemanticGraph,
                                 sample_id: str) -> Optional[Dict[str, Any]]:
        """Build splade embeddings."""
        if not self.config.build_splade:
            return None
        
        try:
            self.logger.info(f" 开始构建 SPLADE 向量: {sample_id}")
            
            
            stats = semantic_graph.build_sparse_embeddings(
                units=None,
                # text_field="text_content",
                model_name=self.config.splade_model,
                batch_size=self.config.splade_batch_size,
                force_rebuild=False,
                show_progress=True
            )
            
            self.logger.info(f" SPLADE构建完成: 处理{stats.get('processed', 0)}, "
                           f"跳过{stats.get('skipped', 0)}, 失败{stats.get('failed', 0)}")
            
            self.stats['splade_stats']['total_processed'] += stats.get('processed', 0)
            self.stats['splade_stats']['total_skipped'] += stats.get('skipped', 0)
            self.stats['splade_stats']['total_failed'] += stats.get('failed', 0)
            
            return stats
            
        except Exception as e:
            self.logger.error(f"SPLADE构建失败: {e}")
            return None
    
    def _convert_fact_to_memory_unit(self, fact: Dict, sample_id: str) -> Optional[MemoryUnit]:
        """Convert fact to memory unit."""
        if not SEMANTIC_GRAPH_AVAILABLE:
            return None
        
        content = fact.get('content', '')
        if not content:
            return None
        
        time_info = fact.get('time', {})
        if isinstance(time_info, dict):
            timestamp = time_info.get('absolute_start', '')
        else:
            timestamp = ''
        
        metadata = {
            'fact_id': fact.get('fact_id', ''),
            'fact_type': fact.get('fact_type', 'EVENT'),
            'participants': fact.get('participants', []),
            'location': fact.get('location', ''),
            'source_session': fact.get('source_session_id', ''),
            'source_turns': fact.get('source_turns', []),
            'retrieval_keys': fact.get('retrieval_keys', []),
            'sample_id': sample_id,
            'memory_type': 'episodic_fact'
        }
        
        if isinstance(time_info, dict):
            metadata['time_original'] = time_info.get('original_text', '')
            metadata['time_start'] = time_info.get('absolute_start', '')
            metadata['time_end'] = time_info.get('absolute_end', '')
            metadata['time_is_exact'] = time_info.get('is_exact', False)
        
        fact_id = fact.get('fact_id', '')
        uid = f"episodic_{sample_id}_{fact_id}" if fact_id else f"episodic_{sample_id}_{hash(content)}"
        
        raw_data = {
            'content': content,
            'text_content': content,
            'source': f"episodic:{sample_id}",
            'timestamp': timestamp if timestamp else datetime.now().isoformat(),
            'fact_type': fact.get('fact_type', 'EVENT'),
            'participants': fact.get('participants', []),
            'location': fact.get('location', ''),
            'retrieval_keys': fact.get('retrieval_keys', [])
        }
        
        try:
            unit = MemoryUnit(
                uid=uid,
                raw_data=raw_data,
                metadata=metadata
            )
            return unit
        except Exception as e:
            self.logger.warning(f"创建MemoryUnit失败: {e}")
            return None
    
    def _build_sample_index(self, data: Dict) -> Dict:
        """Build sample index."""
        sample_id = data.get('sample_id', 'unknown')
        
        index = {
            'sample_id': sample_id,
            'by_participant': {},     # participant -> [fact_ids]
            'by_time': {},            # YYYY-MM -> [fact_ids]
            'by_type': {},            # fact_type -> [fact_ids]
            'accumulations': [],      # [accumulation_ids]
            'timelines': [],          # [timeline_ids]
            'fact_lookup': {}         # fact_id -> fact_summary
        }
        
        
        for fact in data.get('episodic_facts', []):
            fact_id = fact.get('fact_id', '')
            if not fact_id:
                continue
            
            
            for p in fact.get('participants', []):
                p_lower = p.lower()
                if p_lower not in index['by_participant']:
                    index['by_participant'][p_lower] = []
                index['by_participant'][p_lower].append(fact_id)
            
            
            time_info = fact.get('time', {})
            if isinstance(time_info, dict):
                start = time_info.get('absolute_start', '')
                if start and len(start) >= 7:
                    ym = start[:7]
                    if ym not in index['by_time']:
                        index['by_time'][ym] = []
                    index['by_time'][ym].append(fact_id)
            
            
            fact_type = fact.get('fact_type', 'EVENT')
            if fact_type not in index['by_type']:
                index['by_type'][fact_type] = []
            index['by_type'][fact_type].append(fact_id)
            
            index['fact_lookup'][fact_id] = {
                'content': fact.get('content', ''),
                'fact_type': fact_type,
                'participants': fact.get('participants', []),
                'time': time_info,
                'location': fact.get('location', ''),
                'retrieval_keys': fact.get('retrieval_keys', [])
            }
        
        
        index['accumulations'] = [
            a.get('accumulation_id') for a in data.get('accumulated_facts', [])
        ]
        
        
        index['timelines'] = [
            t.get('timeline_id') for t in data.get('timelines', [])
        ]
        
        return index

    def _load_saved_retrieval_state(self, output_dir: str) -> Dict[str, Any]:
        """Load saved retrieval state."""
        state = {
            'enabled': self.config.freeze_retrievers,
            'frozen_matrices_saved': {},
            'indices_saved': {}
        }
        graph_state_file = os.path.join(output_dir, "graph_state.json")
        try:
            with open(graph_state_file, 'r', encoding='utf-8') as f:
                graph_state = json.load(f)
            retrieval = graph_state.get('retrieval', {})
            state['frozen_matrices_saved'] = retrieval.get('frozen_matrices_saved', {})
            state['indices_saved'] = retrieval.get('indices_saved', {})
        except Exception as e:
            state['error'] = str(e)
            self.logger.warning(f"读取检索索引保存状态失败: {e}")
        return state
    
    def _save_sample_graph(self, 
                           semantic_graph: SemanticGraph,
                           sample_id: str,
                           data: Dict,
                           splade_stats: Optional[Dict[str, Any]] = None) -> str:
        """Save sample graph."""
        output_dir = os.path.join(self.config.output_dir, sample_id)
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            
            semantic_graph.save_graph(output_dir, freeze_retrievers=self.config.freeze_retrievers)
            retrieval_state = self._load_saved_retrieval_state(output_dir)
            
            
            sample_index = self._build_sample_index(data)
            
            
            sample_detail = self.stats['sample_details'].get(sample_id, {})
            metadata = {
                'sample_id': sample_id,
                'content_type': 'episodic_memory',
                'build_timestamp': datetime.now().isoformat(),
                'data_source': {
                    'input_dir': self.config.input_dir,
                    'input_file': f"{sample_id}_enhanced.json"
                },
                'embedding_config': {
                    'model': self.config.embedding_model
                },
                'splade_config': {
                    'enabled': self.config.build_splade,
                    'model': self.config.splade_model if self.config.build_splade else None,
                    'batch_size': self.config.splade_batch_size if self.config.build_splade else None
                },
                'splade_stats': splade_stats if splade_stats else {},
                'static_retriever_indexes': retrieval_state,
                'statistics': {
                    'facts_count': sample_detail.get('facts', 0),
                    'units_loaded': sample_detail.get('units_loaded', 0),
                    'accumulations_count': sample_detail.get('accumulations', 0),
                    'timelines_count': sample_detail.get('timelines', 0)
                }
            }
            
            metadata_file = os.path.join(output_dir, "sample_metadata.json")
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            
            index_file = os.path.join(output_dir, "sample_index.json")
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump(sample_index, f, indent=2, ensure_ascii=False)
            
            
            data_file = os.path.join(output_dir, "episodic_facts.json")
            with open(data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f" 样本 {sample_id} 图谱已保存到: {output_dir}")
            return output_dir
            
        except Exception as e:
            self.logger.error(f"保存样本 {sample_id} 图谱失败: {e}")
            raise
    
    def process_sample(self, input_file: Path, cleanup: bool = True) -> Dict:
        """Process sample."""
        semantic_graph = None
        sample_id = "unknown"
        
        try:
            
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            sample_id = data.get('sample_id', 'unknown')
            facts = data.get('episodic_facts', [])
            accumulations = data.get('accumulated_facts', [])
            timelines = data.get('timelines', [])
            
            self.logger.info(f" 加载样本 {sample_id}: {len(facts)} facts, "
                            f"{len(accumulations)} accumulations, {len(timelines)} timelines")
            
            if SEMANTIC_GRAPH_AVAILABLE:
                semantic_graph = self._create_semantic_graph()
                
                units_to_add = []
                for fact in facts:
                    unit = self._convert_fact_to_memory_unit(fact, sample_id)
                    if unit:
                        units_to_add.append(unit)
                
                
                loaded_units = 0
                if units_to_add:
                    try:
                        batch_stats = semantic_graph.batch_add_units(
                            units=units_to_add,
                            batch_size=32,
                            space_names=[f"episodic_{sample_id}"],
                            index_update_mode="none",
                            generate_sparse_embedding=self.config.build_splade,
                            sparse_model_name=self.config.splade_model,
                            show_progress=True
                        )
                        loaded_units = batch_stats.get('added', 0)
                        self.logger.info(f" 批量添加完成: {batch_stats}")
                    except Exception as e:
                        self.logger.warning(f"批量添加到SemanticGraph失败: {e}")
                
                self.logger.info(f" 成功加载 {loaded_units}/{len(facts)} 个事实到SemanticGraph")
                
                
                splade_stats = None
                if self.config.build_splade:
                    
                    splade_stats = {
                        'enabled': True,
                        'model': self.config.splade_model,
                        'processed': loaded_units
                    }
                
                
                self.stats['sample_details'][sample_id] = {
                    'facts': len(facts),
                    'units_loaded': loaded_units,
                    'accumulations': len(accumulations),
                    'timelines': len(timelines)
                }

                if self.config.enable_index_building:
                    semantic_graph.rebuild_all_indexes()
                
                
                self._save_sample_graph(semantic_graph, sample_id, data, splade_stats)
            else:
                loaded_units = 0
                self.logger.warning(f"SemanticGraph不可用，仅保存JSON")
                
                
                output_dir = os.path.join(self.config.output_dir, sample_id)
                os.makedirs(output_dir, exist_ok=True)
                
                data_file = os.path.join(output_dir, "episodic_facts.json")
                with open(data_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                
                index = self._build_sample_index(data)
                index_file = os.path.join(output_dir, "sample_index.json")
                with open(index_file, 'w', encoding='utf-8') as f:
                    json.dump(index, f, indent=2, ensure_ascii=False)
            
            self.stats['samples_loaded'] += 1
            self.stats['facts_loaded'] += len(facts)
            self.stats['accumulations_loaded'] += len(accumulations)
            self.stats['timelines_loaded'] += len(timelines)
            self.stats['index_entries'] += len(facts)
            
            return {
                'sample_id': sample_id,
                'facts': len(facts),
                'units_loaded': loaded_units,
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f" 处理样本失败: {e}")
            self.logger.debug(traceback.format_exc())
            
            sample_id = input_file.stem.replace('_enhanced', '')
            self.stats['failed_samples'].append(sample_id)
            
            return {
                'sample_id': sample_id,
                'facts': 0,
                'units_loaded': 0,
                'success': False,
                'error': str(e)
            }
            
        finally:
            if cleanup and semantic_graph is not None:
                self.logger.debug(f" 清理 {sample_id} 的资源...")
                self._cleanup_resources(semantic_graph)
    
    def _save_loading_stats(self):
        """Save loading stats."""
        stats_file = os.path.join(self.config.output_dir, "loading_stats.json")
        
        try:
            enhanced_stats = self.stats.copy()
            enhanced_stats['loading_config'] = {
                'input_dir': self.config.input_dir,
                'output_dir': self.config.output_dir,
                'embedding_model': self.config.embedding_model,
                'build_splade': self.config.build_splade,
                'splade_model': self.config.splade_model if self.config.build_splade else None,
                'splade_batch_size': self.config.splade_batch_size if self.config.build_splade else None,
                'enable_index_building': self.config.enable_index_building,
                'freeze_retrievers': self.config.freeze_retrievers
            }
            enhanced_stats['completion_time'] = datetime.now().isoformat()
            
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(enhanced_stats, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f" 统计信息已保存: {stats_file}")
            
        except Exception as e:
            self.logger.error(f"保存统计信息失败: {e}")
    
    def run(self) -> Dict:
        """Run."""
        self.logger.info("=" * 80)
        self.logger.info(" 开始加载情景记忆 (Load Episodic Memory)")
        self.logger.info("=" * 80)
        
        input_dir = Path(self.config.input_dir)
        if not input_dir.exists():
            self.logger.error(f" 输入目录不存在: {input_dir}")
            return self.stats
        
        input_files = list(input_dir.glob("*_enhanced.json"))
        self.logger.info(f" 找到 {len(input_files)} 个输入文件")
        
        
        for i, input_file in enumerate(input_files, 1):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"[{i}/{len(input_files)}] 处理文件: {input_file.name}")
            self.logger.info(f"{'='*60}")
            
            try:
                result = self.process_sample(input_file)
                if result['success']:
                    self.logger.info(f" 样本 {result['sample_id']} 处理成功")
                else:
                    self.logger.error(f" 样本 {result['sample_id']} 处理失败")
            except Exception as e:
                self.logger.error(f" 处理文件 {input_file} 失败: {e}")
        
        
        self._save_loading_stats()
        
        self._print_summary()
        
        return self.stats
    
    def _print_summary(self):
        """Run print summary."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info(" 情景记忆加载完成")
        self.logger.info("=" * 80)
        self.logger.info(f" 加载样本数: {self.stats['samples_loaded']}")
        self.logger.info(f" 加载事实数: {self.stats['facts_loaded']}")
        self.logger.info(f" 累积事实数: {self.stats['accumulations_loaded']}")
        self.logger.info(f" 时间线数: {self.stats['timelines_loaded']}")
        self.logger.info(f" 索引条目数: {self.stats['index_entries']}")
        
        if self.config.build_splade:
            self.logger.info(f"\n SPLADE 统计:")
            self.logger.info(f"  ├─ 处理: {self.stats['splade_stats']['total_processed']}")
            self.logger.info(f"  ├─ 跳过: {self.stats['splade_stats']['total_skipped']}")
            self.logger.info(f"  failed: {self.stats['splade_stats']['total_failed']}")
        
        if self.stats['failed_samples']:
            self.logger.warning(f" 失败样本: {self.stats['failed_samples']}")
        
        self.logger.info(f" 输出目录: {self.output_dir}")



def main():
    parser = argparse.ArgumentParser(
        description="Step 3: 加载情景记忆到检索系统（SemanticGraph）"
    )
    
    parser.add_argument(
        "--input-dir",
        default=str(paths.LOCOMO_EPISODIC_STEP2_DIR),
        help="Step 2输出目录"
    )
    parser.add_argument(
        "--output-dir",
        default=str(paths.LOCOMO_EPISODIC_STEP3_DIR),
        help="输出目录"
    )
    parser.add_argument(
        "--embedding-model",
        default="Qwen/Qwen3-Embedding-0.6B",
        help="嵌入模型名称 (默认: Qwen/Qwen3-Embedding-0.6B)"
    )
    parser.add_argument(
        "--build-splade",
        action="store_true",
        default=True,
        help="是否构建SPLADE稀疏向量 (默认: 启用)"
    )
    parser.add_argument(
        "--no-splade",
        action="store_true",
        help="禁用SPLADE稀疏向量构建"
    )
    parser.add_argument(
        "--splade-model",
        default="naver/splade-v3",
        help="SPLADE模型名称 (默认: naver/splade-v3)"
    )
    parser.add_argument(
        "--splade-batch-size",
        type=int,
        default=32,
        help="SPLADE批处理大小 (默认: 32)"
    )
    parser.add_argument(
        "--no-index",
        action="store_true",
        help="禁用索引构建"
    )
    parser.add_argument(
        "--no-freeze-retrievers",
        action="store_true",
        help="禁用保存阶段的 BM25/SPLADE 静态加速索引冻结"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="启用调试模式"
    )
    
    args = parser.parse_args()
    
    
    build_splade = args.build_splade and not args.no_splade
    
    config = LoadConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        embedding_model=args.embedding_model,
        build_splade=build_splade,
        splade_model=args.splade_model,
        splade_batch_size=args.splade_batch_size,
        enable_index_building=not args.no_index,
        freeze_retrievers=not args.no_freeze_retrievers,
        debug_mode=args.debug
    )
    
    loader = EpisodicMemoryLoader(config)
    loader.run()


if __name__ == "__main__":
    main()
