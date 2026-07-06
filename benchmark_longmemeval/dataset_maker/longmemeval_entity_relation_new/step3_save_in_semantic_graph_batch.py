#!/usr/bin/env python3
"""Utilities for step3 save in semantic graph batch."""

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
class Step3Config:
    entities_dir: str = str(paths.LONGMEMEVAL_ENTITY_RELATION_NEW_DEDUPLICATED_LLM_DIR)
    
    output_dir: str = str(paths.LONGMEMEVAL_ENTITY_RELATION_GRAPHS_DIR)
    
    
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
    create_entity_hubs: bool = True
    
    debug_mode: bool = False


class LongMemEvalEntityGraphLoader:
    
    def __init__(self, config: Step3Config):
        self.config = config
        self.logger = self._setup_logging()
        
        self.stats = {
            'total_qa': 0,
            'processed_qa': 0,
            'total_entities': 0,
            'total_mentions': 0,
            'total_entity_hubs': 0,
            'failed_qa': [],
            'processing_time': 0,
            'qa_details': {}
        }
        
        self.logger.info("=" * 80)
        self.logger.info(" LongMemEval Step 3: 实体关系图谱加载器 (Mention-based)")
        self.logger.info("=" * 80)
        self.logger.info(f" 实体目录: {self.config.entities_dir}")
        self.logger.info(f" 输出目录: {self.config.output_dir}")
        self.logger.info(f" 嵌入模型: {self.config.text_embedding_model}")
        self.logger.info(f" 存储模式: 以 mention 为单位（参考 LoCoMo）")
    
    def _setup_logging(self) -> logging.Logger:
        """Run setup logging."""
        logger = logging.getLogger(f"{__name__}.EntityGraphLoader")
        
        if self.config.debug_mode:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
        
        return logger
    
    def load_all_qa_graphs(self) -> Dict[str, Any]:
        """Load all qa graphs."""
        start_time = time.time()
        
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        
        entities_dir = Path(self.config.entities_dir)
        if not entities_dir.exists():
            self.logger.error(f"实体目录不存在: {entities_dir}")
            return self.stats
        
        qa_files = sorted(entities_dir.glob("qa_*_deduplicated.json"))
        
        
        qa_indices = []
        for qa_file in qa_files:
            
            try:
                qa_idx = int(qa_file.stem.split('_')[1])
                qa_indices.append(qa_idx)
            except (ValueError, IndexError):
                continue
        
        qa_indices = sorted(qa_indices)
        
        if self.config.start_qa is not None:
            qa_indices = [idx for idx in qa_indices if idx >= self.config.start_qa]
        if self.config.end_qa is not None:
            qa_indices = [idx for idx in qa_indices if idx <= self.config.end_qa]
        
        self.stats['total_qa'] = len(qa_indices)
        
        self.logger.info(f" 找到 {len(qa_indices)} 个 QA 待处理")
        self.logger.info(f" 处理范围: qa_{qa_indices[0]} - qa_{qa_indices[-1]}")
        
        if self.config.enable_parallel:
            self._process_parallel(qa_indices)
        else:
            self._process_sequential(qa_indices)
        
        
        self.stats['processing_time'] = time.time() - start_time
        self._save_stats()
        self._print_summary()
        
        return self.stats
    
    def _process_sequential(self, qa_indices: List[int]):
        """Process sequential."""
        self.logger.info(" 使用顺序处理模式...")
        
        for qa_index in tqdm(qa_indices, desc="处理 QA"):
            try:
                success = self._load_single_qa_graph(qa_index)
                if success:
                    self.stats['processed_qa'] += 1
                else:
                    self.stats['failed_qa'].append(qa_index)
            except Exception as e:
                self.logger.error(f"处理 qa_{qa_index} 失败: {e}")
                self.stats['failed_qa'].append(qa_index)
    
    def _process_parallel(self, qa_indices: List[int]):
        """Process parallel."""
        self.logger.info(f" 使用并行处理模式 (workers={self.config.max_workers})...")
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(self._load_single_qa_graph, qa_idx): qa_idx
                for qa_idx in qa_indices
            }
            
            with tqdm(total=len(qa_indices), desc="处理 QA") as pbar:
                for future in as_completed(futures):
                    qa_idx = futures[future]
                    try:
                        success = future.result()
                        if success:
                            self.stats['processed_qa'] += 1
                        else:
                            self.stats['failed_qa'].append(qa_idx)
                    except Exception as e:
                        self.logger.error(f"处理 qa_{qa_idx} 失败: {e}")
                        self.stats['failed_qa'].append(qa_idx)
                    finally:
                        pbar.update(1)
    
    def _load_single_qa_graph(self, qa_index: int) -> bool:
        """Load single qa graph."""
        qa_start_time = time.time()
        
        try:
            
            entity_data = self._load_entity_data(qa_index)
            if not entity_data:
                self.logger.warning(f"qa_{qa_index} 无实体数据，跳过")
                return False
            
            semantic_graph = self._create_semantic_graph()
            
            qa_space_name = f"qa_{qa_index}"
            semantic_graph.create_memory_space_in_map(qa_space_name)
            
            
            mention_count, hub_count = self._load_mentions_and_hubs(
                semantic_graph, 
                entity_data, 
                qa_index,
                qa_space_name
            )
            
            
            splade_stats = None
            if self.config.build_splade:
                splade_stats = self._build_splade_embeddings(semantic_graph, qa_index)
            
            
            if self.config.build_index and (mention_count > 0 or hub_count > 0):
                semantic_graph.rebuild_all_indexes()

            
            output_path = self._save_qa_graph(
                semantic_graph, 
                qa_index, 
                entity_data,
                splade_stats
            )
            
            qa_time = time.time() - qa_start_time
            self.stats['qa_details'][f'qa_{qa_index}'] = {
                'mentions': mention_count,
                'entity_hubs': hub_count,
                'processing_time': qa_time,
                'output_path': str(output_path)
            }
            
            self.stats['total_mentions'] += mention_count
            self.stats['total_entity_hubs'] += hub_count
            self.stats['total_entities'] += len(entity_data.get('entities', []))
            
            self.logger.info(
                f" qa_{qa_index}: {mention_count} mentions, "
                f"{hub_count} hubs, {qa_time:.2f}s"
            )
            
            del semantic_graph
            
            return True
            
        except Exception as e:
            self.logger.error(f" qa_{qa_index} 处理失败: {e}", exc_info=True)
            return False
    
    def _load_entity_data(self, qa_index: int) -> Optional[Dict[str, Any]]:
        """Load entity data."""
        entity_file = Path(self.config.entities_dir) / f"qa_{qa_index}_deduplicated.json"
        
        if not entity_file.exists():
            self.logger.warning(f"实体文件不存在: {entity_file}")
            return None
        
        try:
            with open(entity_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            entities = data.get('entities', [])
            self.logger.debug(f"qa_{qa_index}: 加载 {len(entities)} 个实体")
            
            return data
            
        except Exception as e:
            self.logger.error(f"加载实体数据失败 qa_{qa_index}: {e}")
            return None
    
    def _create_semantic_graph(self) -> SemanticGraph:
        """Create semantic graph."""
        try:
            semantic_map = SemanticMap(
                embedding_model_name=self.config.text_embedding_model,
                embedding_dim=self.config.embedding_dim
            )
            
            semantic_graph = SemanticGraph(semantic_map_instance=semantic_map)
            
            self.logger.debug("SemanticGraph created successfully")
            return semantic_graph
            
        except Exception as e:
            self.logger.error(f"Failed to create SemanticGraph: {e}")
            raise
        
    def _load_mentions_and_hubs(self,
                            semantic_graph: SemanticGraph,
                            entity_data: Dict[str, Any],
                            qa_index: int,
                            qa_space_name: str) -> tuple:
        """Load entity mention and hub nodes into the QA-specific graph space.

        Args:
            semantic_graph: Target SemanticGraph.
            entity_data: Entity-relation data for one QA item.
            qa_index: Numeric QA item index.
            qa_space_name: MemorySpace name used for this QA item.

        Returns:
            Tuple of ``(mention_count, hub_count)``.
        """
        entities = entity_data.get('entities', [])
        qa_id = entity_data.get('qa_id', f'qa_{qa_index}')
        
        
        hub_units: List[MemoryUnit] = []
        mention_units: List[MemoryUnit] = []
        pending_relationships: List[tuple] = []  # (hub_uid, mention_uid)
        
        
        for entity_idx, entity in enumerate(entities):
            entity_uid = entity.get('uid', entity.get('entity_id', ''))  
            canonical_content = entity.get('canonical_content', entity.get('name', ''))  
            category = entity.get('category', entity.get('entity_type', 'UNKNOWN'))  
            attributes = entity.get('attributes', {})
            mentions = entity.get('mentions', [])
            
            
            
            if entity_uid.startswith(f"{qa_id}_"):
                base_uid = f"{entity_uid}_{entity_idx}"
            else:
                base_uid = f"{qa_id}_{entity_uid}_{entity_idx}"

            hub_uid = None
            if self.config.create_entity_hubs and canonical_content:
                hub_uid = f"{base_uid}_hub"
                
                hub_text_content = self._build_hub_text_content(
                    canonical_content, 
                    category, 
                    attributes
                )
                
                hub_raw_data = {
                    "node_type": "entity_hub",
                    "qa_id": qa_id,
                    "entity_id": entity_uid,
                    "canonical_content": canonical_content,
                    "category": category,
                    "attributes": attributes,
                    "total_mentions": len(mentions),
                    "text_content": hub_text_content,
                    "created_at": datetime.now().isoformat()
                }
                
                hub_unit = MemoryUnit(uid=hub_uid, raw_data=hub_raw_data)
                hub_units.append(hub_unit)
            
            for mention_idx, mention in enumerate(mentions, 1):
                mention_id = mention.get('mention_id', f'mention_{mention_idx}')
                mention_uid = f"{base_uid}_{mention_id}"
                
                content = mention.get('content', mention.get('context', '')) 
                session_ids = mention.get('session_ids', [])
                if isinstance(mention.get('session_id'), str):
                    session_ids = [mention.get('session_id')]
                
                session_date = mention.get('session_date', '')
                temporal = mention.get('temporal', mention.get('temporal_info', {}))
                
                if not content:
                    continue
                
                mention_text_content = self._build_mention_text_content(
                    canonical_content,
                    category,
                    content,
                    session_ids,
                    temporal,
                    attributes
                )
                
                mention_raw_data = {
                    "node_type": "evidence_mention",
                    "qa_id": qa_id,
                    "parent_entity_id": entity_uid,
                    "parent_hub_uid": hub_uid if self.config.create_entity_hubs else None,
                    "entity_canonical": canonical_content,
                    "entity_category": category,
                    "mention_id": mention_id,
                    "content": content,
                    "session_ids": session_ids,
                    "session_date": session_date,
                    "temporal_info": self._format_temporal_info(temporal),
                    "attributes": attributes,
                    "confidence": mention.get('confidence', 0.95),
                    "text_content": mention_text_content,
                    "created_at": datetime.now().isoformat()
                }
                
                mention_unit = MemoryUnit(uid=mention_uid, raw_data=mention_raw_data)
                mention_units.append(mention_unit)
                
                if self.config.create_entity_hubs and hub_uid:
                    pending_relationships.append((hub_uid, mention_uid))
        
        hub_count = 0
        if hub_units:
            self.logger.debug(f"批量添加 {len(hub_units)} 个 Hub 节点...")
            hub_stats = semantic_graph.batch_add_units(
                units=hub_units,
                batch_size=self.config.batch_size,
                space_names=[qa_space_name],
                index_update_mode="none",
                generate_sparse_embedding=False,  
                show_progress=False
            )
            hub_count = hub_stats.get('added', 0)
        
        mention_count = 0
        if mention_units:
            self.logger.debug(f"批量添加 {len(mention_units)} 个 Mention 节点...")
            mention_stats = semantic_graph.batch_add_units(
                units=mention_units,
                batch_size=self.config.batch_size,
                space_names=[qa_space_name],
                index_update_mode="none",
                generate_sparse_embedding=False,  
                show_progress=False
            )
            mention_count = mention_stats.get('added', 0)
        
        if pending_relationships:
            self.logger.debug(f"添加 {len(pending_relationships)} 条 HAS_MENTION 关系...")
            for hub_uid, mention_uid in pending_relationships:
                try:
                    semantic_graph.add_relationship(
                        source_uid=hub_uid, 
                        target_uid=mention_uid, 
                        relationship_name="HAS_MENTION"
                    )
                except Exception as e:
                    self.logger.debug(f"添加关系失败 {hub_uid} -> {mention_uid}: {e}")
        
        return mention_count, hub_count

    def _build_mention_text_content(self,
                                   canonical_content: str,
                                   category: str,
                                   content: str,
                                   session_ids: List[str],
                                   temporal: Any,
                                   attributes: Dict) -> str:
        """Build mention text content."""
        parts = []
        
        parts.append(f"Entity: {canonical_content} (Type: {category})")
        
        # if session_ids:
        #     session_str = ", ".join(session_ids) if len(session_ids) > 1 else session_ids[0]
        #     parts.append(f"Session: {session_str}")
        
        if content:
            parts.append(f"Context: {content}")
        
        if temporal:
            temporal_str = self._format_temporal_info(temporal)
            if temporal_str:
                parts.append(f"Temporal: {temporal_str}")
        
        if attributes:
            attr_strs = [f"{k}: {v}" for k, v in attributes.items() if v]
            if attr_strs:
                parts.append("Attributes: " + ", ".join(attr_strs[:3]))
        
        return " | ".join(parts)
    
    # def _load_mentions_and_hubs(self,
    #                         semantic_graph: SemanticGraph,
    #                         entity_data: Dict[str, Any],
    #                         qa_index: int,
    #                         qa_space_name: str) -> tuple:
    #     """
    
        
    #     Args:
    
        
    #     Returns:
    #         (mention_count, hub_count)
    #     """
    #     entities = entity_data.get('entities', [])
    #     qa_id = entity_data.get('qa_id', f'qa_{qa_index}')
        
    #     mention_count = 0
    #     hub_count = 0
        
    #     for entity in entities:
    
    
    
    #         attributes = entity.get('attributes', {})
    #         mentions = entity.get('mentions', [])
            
    
    
    
    #         if entity_uid.startswith(f"{qa_id}_"):
    #             base_uid = entity_uid
    #         else:
    #             base_uid = f"{qa_id}_{entity_uid}"

    #         hub_uid = None
    #         if self.config.create_entity_hubs and canonical_content:
    
    #             hub_uid = f"{base_uid}_hub"
                
    #             hub_text_content = self._build_hub_text_content(
    #                 canonical_content, 
    #                 category, 
    #                 attributes
    #             )
                
    #             hub_raw_data = {
    #                 "node_type": "entity_hub",
    #                 "qa_id": qa_id,
    #                 "entity_id": entity_uid,
    #                 "canonical_content": canonical_content,
    #                 "category": category,
    #                 "attributes": attributes,
    #                 "total_mentions": len(mentions),
    #                 "text_content": hub_text_content,
    #                 "created_at": datetime.now().isoformat()
    #             }
                
    #             hub_unit = MemoryUnit(uid=hub_uid, raw_data=hub_raw_data)
                
    #             semantic_graph.add_unit(
    #                 unit=hub_unit, 
    #             )
    #             hub_count += 1
            
    #         for mention_idx, mention in enumerate(mentions, 1):
    
    #             mention_id = mention.get('mention_id', f'mention_{mention_idx}')
                
    
    #             mention_uid = f"{base_uid}_{mention_id}"
                
    #             content = mention.get('content', mention.get('context', '')) 
    #             session_ids = mention.get('session_ids', [])
    #             if isinstance(mention.get('session_id'), str):
    #                 session_ids = [mention.get('session_id')]
                
    #             temporal = mention.get('temporal', mention.get('temporal_info', {}))
                
    #             if not content:
    #                 continue
                
    #             mention_text_content = self._build_mention_text_content(
    #                 canonical_content,
    #                 category,
    #                 content,
    #                 session_ids,
    #                 temporal,
    #                 attributes
    #             )
                
    #             mention_raw_data = {
    #                 "node_type": "evidence_mention",
    #                 "qa_id": qa_id,
    #                 "parent_entity_id": entity_uid,
    #                 "parent_hub_uid": hub_uid if self.config.create_entity_hubs else None,
    #                 "entity_canonical": canonical_content,
    #                 "entity_category": category,
    #                 "mention_id": mention_id,
    #                 "content": content,
    #                 "session_ids": session_ids,
    #                 "temporal_info": self._format_temporal_info(temporal),
    #                 "attributes": attributes,
    #                 "confidence": mention.get('confidence', 0.95),
    #                 "text_content": mention_text_content,
    #                 "created_at": datetime.now().isoformat()
    #             }
                
    #             mention_unit = MemoryUnit(uid=mention_uid, raw_data=mention_raw_data)
                
    #             semantic_graph.add_unit(
    #                 unit=mention_unit, 
    #                 space_names=[qa_space_name]
    #             )
    #             mention_count += 1
                
    #             if self.config.create_entity_hubs and hub_uid:
    #                 semantic_graph.add_relationship(
    #                     source_uid=hub_uid, 
    #                     target_uid=mention_uid, 
    #                     relationship_name="HAS_MENTION"
    #                 )
        
    #     return mention_count, hub_count
    
    def _build_hub_text_content(self, 
                               canonical_content: str,
                               category: str,
                               attributes: Dict) -> str:
        """Build hub text content."""
        parts = [f"Entity: {canonical_content} (Category: {category})"]
        
        if attributes:
            attr_strs = [f"{k}: {v}" for k, v in attributes.items() if v]
            if attr_strs:
                parts.append("Attributes: " + ", ".join(attr_strs))
        
        return " | ".join(parts)
    
    # def _build_mention_text_content(self,
    #                                canonical_content: str,
    #                                category: str,
    #                                content: str,
    #                                session_ids: List[str],
    #                                temporal: Any,
    #                                attributes: Dict) -> str:
    #     """
    #     """
    #     parts = []
        
    #     parts.append(f"Entity: {canonical_content} (Type: {category})")
        
    #     # if session_ids:
    #     #     session_str = ", ".join(session_ids) if len(session_ids) > 1 else session_ids[0]
    #     #     parts.append(f"Session: {session_str}")
        
    #     if content:
    #         parts.append(f"Context: {content}")
        
    #     if temporal:
    #         temporal_str = self._format_temporal_info(temporal)
    #         if temporal_str:
    #             parts.append(f"Temporal: {temporal_str}")
        
    #     if attributes:
    #         attr_strs = [f"{k}: {v}" for k, v in attributes.items() if v]
    #         if attr_strs:
        
    #     return " | ".join(parts)
    
    def _format_temporal_info(self, temporal: Any) -> str:
        """Format temporal info."""
        if not temporal:
            return ""
        
        if isinstance(temporal, dict):
            temporal_parts = []
            for key, value in temporal.items():
                if value:
                    temporal_parts.append(f"{key}={value}")
            return ", ".join(temporal_parts)
        else:
            return str(temporal)
    
    def _build_splade_embeddings(self, 
                                semantic_graph: SemanticGraph,
                                qa_index: int) -> Optional[Dict[str, Any]]:
        """Build splade embeddings."""
        if not self.config.build_splade:
            return None
        
        try:
            self.logger.debug(f"qa_{qa_index}: 开始构建 SPLADE 嵌入...")
            
            splade_start_time = time.time()
            
            
            splade_result = semantic_graph.build_sparse_embeddings(
                model_name=self.config.splade_model,
                batch_size=self.config.splade_batch_size,
                show_progress=False
            )
            
            splade_time = time.time() - splade_start_time
            
            if splade_result:
                splade_result['processing_time'] = splade_time
            
            self.logger.debug(
                f"qa_{qa_index}: SPLADE 构建完成 "
                f"({splade_result.get('processed', 0)} processed, {splade_time:.2f}s)"
            )
            
            return splade_result
            
        except Exception as e:
            self.logger.error(f"qa_{qa_index}: SPLADE 构建失败: {e}")
            return None
    
    def _save_qa_graph(self,
                      semantic_graph: SemanticGraph,
                      qa_index: int,
                      entity_data: Dict[str, Any],
                      splade_stats: Optional[Dict[str, Any]] = None) -> Path:
        """Save qa graph."""
        qa_output_dir = Path(self.config.output_dir) / f"qa_{qa_index}"
        qa_output_dir.mkdir(parents=True, exist_ok=True)
        
        
        semantic_graph.save_graph(str(qa_output_dir))
        
        
        meta_info = {
            'qa_index': qa_index,
            'qa_id': entity_data.get('qa_id', f'qa_{qa_index}'),
            'total_entities': entity_data.get('total_entities', 0),
            'total_mentions': sum(
                len(e.get('mentions', [])) 
                for e in entity_data.get('entities', [])
            ),
            'created_at': entity_data.get('created_at', datetime.now().isoformat()),
            'saved_at': datetime.now().isoformat(),
            'config': {
                'embedding_model': self.config.text_embedding_model,
                'build_splade': self.config.build_splade,
                'create_entity_hubs': self.config.create_entity_hubs
            }
        }
        
        if splade_stats:
            meta_info['splade_stats'] = splade_stats
        
        meta_file = qa_output_dir / "meta_info.json"
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(meta_info, f, indent=2, ensure_ascii=False)
        
        
        entity_copy_file = qa_output_dir / "entity_data.json"
        with open(entity_copy_file, 'w', encoding='utf-8') as f:
            json.dump(entity_data, f, indent=2, ensure_ascii=False)
        
        self.logger.debug(f"qa_{qa_index}: 保存到 {qa_output_dir}")
        
        return qa_output_dir
    
    def _save_stats(self):
        """Save stats."""
        stats_file = Path(self.config.output_dir) / "processing_stats.json"
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f" 统计信息已保存: {stats_file}")
    
    def _print_summary(self):
        """Run print summary."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info(" 处理摘要")
        self.logger.info("=" * 80)
        self.logger.info(f"总 QA 数: {self.stats['total_qa']}")
        self.logger.info(f"成功处理: {self.stats['processed_qa']}")
        self.logger.info(f"失败数量: {len(self.stats['failed_qa'])}")
        self.logger.info(f"总实体数: {self.stats['total_entities']}")
        self.logger.info(f"总 Mention 数: {self.stats['total_mentions']}")
        if self.config.create_entity_hubs:
            self.logger.info(f"总 Entity Hub 数: {self.stats['total_entity_hubs']}")
        self.logger.info(f"总耗时: {self.stats['processing_time']:.2f}s")
        
        if self.stats['failed_qa']:
            self.logger.warning(f"失败的 QA: {self.stats['failed_qa']}")
        
        self.logger.info("=" * 80 + "\n")


def main():
    """Run the command-line entry point."""
    
    parser = argparse.ArgumentParser(
        description="LongMemEval Step 3: 加载实体关系数据到 SemanticGraph (Mention-based)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        示例:
        # 处理所有 QA
        python step3_saved_in_semantic_graph.py

        # 处理 QA 0-99
        python step3_saved_in_semantic_graph.py --start-qa 0 --end-qa 99

        # 指定嵌入模型
        python step3_saved_in_semantic_graph.py --embedding-model "Qwen/Qwen3-Embedding-0.6B"

        # 禁用 SPLADE
        python step3_saved_in_semantic_graph.py --no-splade

        # 创建实体中心节点
        python step3_saved_in_semantic_graph.py --create-entity-hubs

        # 单线程调试模式
        python step3_saved_in_semantic_graph.py --no-parallel --debug
        """
    )
    
    parser.add_argument("--entities-dir",
                       default=str(paths.LONGMEMEVAL_ENTITY_RELATION_NEW_DEDUPLICATED_LLM_DIR),
                       help="去重实体数据目录")
    parser.add_argument("--output-dir",
                       default=str(paths.LONGMEMEVAL_ENTITY_RELATION_GRAPHS_DIR),
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
    parser.add_argument("--create-entity-hubs", action="store_true",
                       help="为每个实体创建中心节点")
    
    parser.add_argument("--debug", action="store_true",
                       help="启用调试模式")
    
    args = parser.parse_args()
    
    
    build_splade = not args.no_splade
    
    config = Step3Config(
        entities_dir=args.entities_dir,
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
        create_entity_hubs=args.create_entity_hubs,
        debug_mode=args.debug
    )
    
    print("\n" + "=" * 80)
    print(" LongMemEval Step 3: 实体关系图谱加载器 (Mention-based)")
    print("=" * 80)
    print(f" 实体目录:   {config.entities_dir}")
    print(f" 输出目录:   {config.output_dir}")
    print(f" 嵌入模型:   {config.text_embedding_model}")
    print(f" 存储模式:   以 mention 为单位（参考 LoCoMo）")
    
    if config.end_qa is not None:
        print(f" 处理范围:   QA {config.start_qa} - {config.end_qa}")
    else:
        print(f" 处理范围:   从 QA {config.start_qa} 开始（全部）")
    
    if build_splade:
        print(f" SPLADE:     启用 ({config.splade_model})")
    else:
        print(f" SPLADE:     禁用")
    
    if config.create_entity_hubs:
        print(f" 实体 Hub:   启用")
    else:
        print(f" 实体 Hub:   禁用")
    
    print("=" * 80 + "\n")
    
    
    loader = LongMemEvalEntityGraphLoader(config)
    
    try:
        
        stats = loader.load_all_qa_graphs()
        
        print("\n" + "=" * 80)
        print(" 处理完成!")
        print(f"成功: {stats['processed_qa']}/{stats['total_qa']}")
        print(f"总 Mentions: {stats['total_mentions']}")
        if config.create_entity_hubs:
            print(f"总 Entity Hubs: {stats['total_entity_hubs']}")
        print("=" * 80 + "\n")
        
        return 0
        
    except Exception as e:
        print(f"\n 处理失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
