#!/usr/bin/env python3
"""Concurrency Model: Sequential Samples, Parallel Loading per Sample."""

import json
import logging
import os
import sys
import re
import time
import traceback
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib


from mandol.core.semantic_graph import SemanticGraph
from mandol.core.semantic_map import SemanticMap
from mandol.core.memory_unit import MemoryUnit
from mandol.core import paths




class MemorySpaceNames:
    L0_OBSERVATION = "L0_Observation"
    
    L1_SESSION_FACTS = "L1_SessionFacts"
    
    L2_ACTIVITY_LEDGER = "L2_ActivityLedger"
    L2_ENTITY_PROFILE = "L2_EntityProfile"
    L2_TIMELINE = "L2_Timeline"
    L2_SOCIAL_GRAPH = "L2_SocialGraph"
    L2_NEGATIVE_CONSTRAINT = "L2_NegativeConstraint"
    
    @classmethod
    def all_spaces(cls) -> List[str]:
        return [
            cls.L0_OBSERVATION,
            cls.L1_SESSION_FACTS,
            cls.L2_ACTIVITY_LEDGER,
            cls.L2_ENTITY_PROFILE,
            cls.L2_TIMELINE,
            cls.L2_SOCIAL_GRAPH,
            cls.L2_NEGATIVE_CONSTRAINT,
        ]
    
    @classmethod
    def l2_spaces(cls) -> List[str]:
        return [
            cls.L2_ACTIVITY_LEDGER,
            cls.L2_ENTITY_PROFILE,
            cls.L2_TIMELINE,
            cls.L2_SOCIAL_GRAPH,
            cls.L2_NEGATIVE_CONSTRAINT,
        ]


@dataclass 
class Step4LoaderConfig:
    l0_graphs_dir: str = str(paths.LOCOMO_HIERARCHICAL_CONTENT_STEP1_DIR)
    l1_extracted_dir: str = str(paths.LOCOMO_HIERARCHICAL_CONTENT_STEP2_DIR)
    l2_aggregated_dir: str = str(paths.LOCOMO_HIERARCHICAL_CONTENT_STEP3_DIR)
    
    output_dir: str = str(paths.LOCOMO_HIERARCHICAL_CONTENT_STEP4_DIR)
    
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    
    
    build_splade: bool = True
    splade_model: str = "naver/splade-v3"
    splade_batch_size: int = 32
    
    enable_relation_loading: bool = True
    enable_hierarchical_relations: bool = True
    enable_index_building: bool = True
    freeze_retrievers: bool = True
    
    batch_size: int = 32
    
    
    load_l2_activity_ledger: bool = True
    load_l2_entity_profiles: bool = True
    load_l2_timeline: bool = True
    load_l2_social_graph: bool = True
    load_l2_negative_constraints: bool = True
    max_timeline_events: int = 50  
    
    debug_mode: bool = False
    save_statistics: bool = True


class Step4GraphLoader:
    
    def __init__(self, config: Step4LoaderConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        self.stats = {
            'total_samples': 0,
            'processed_samples': 0,
            'failed_samples': [],
            'l0_units_loaded': 0,
            'l0_relations_loaded': 0,
            'l1_units_loaded': 0,
            'l2_units_loaded': {
                'activity_ledger': 0,
                'entity_profiles': 0,
                'timeline_events': 0,
                'social_graph': 0,
                'negative_constraints': 0,
            },
            'hierarchical_relations': 0,
            'processing_time': 0,
            'sample_details': {},
            'splade_stats': {
                'enabled': config.build_splade,
                'total_processed': 0,
                'total_skipped': 0,
                'total_failed': 0
            }
        }
        
        self._log_config()
    
    def _setup_logging(self) -> logging.Logger:
        """Run setup logging."""
        logger = logging.getLogger(f"{__name__}.Step4GraphLoader")
        logger.setLevel(logging.DEBUG if self.config.debug_mode else logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _log_config(self):
        """Log config."""
        self.logger.info("=" * 70)
        self.logger.info(" Step 4 Graph Loader 初始化")
        self.logger.info("=" * 70)
        self.logger.info(f"   L0 输入: {self.config.l0_graphs_dir}")
        self.logger.info(f"   L1 输入: {self.config.l1_extracted_dir}")
        self.logger.info(f"   L2 输入: {self.config.l2_aggregated_dir}")
        self.logger.info(f"   输出目录: {self.config.output_dir}")
        self.logger.info(f"   嵌入模型: {self.config.embedding_model}")
        self.logger.info(f"   SPLADE: {'启用' if self.config.build_splade else '禁用'}")
        self.logger.info(f"   BM25/SPLADE 静态索引冻结: {'启用' if self.config.freeze_retrievers else '禁用'}")
        self.logger.info("=" * 70)
    
    
    
    
    
    def load_all_graphs(self) -> Dict[str, Any]:
        """Load all graphs."""
        start_time = time.time()
        self.logger.info("\n 开始加载分层语义图谱 (Step 4)")
        
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        sample_files = self._find_sample_files()
        if not sample_files:
            raise ValueError("未找到任何有效的样本文件")
        
        self.stats['total_samples'] = len(sample_files)
        self.logger.info(f" 找到 {len(sample_files)} 个样本")
        
        for i, sample_info in enumerate(sample_files, 1):
            sample_id = sample_info['sample_id']
            self.logger.info(f"\n[{i}/{len(sample_files)}] 处理样本: {sample_id}")
            
            try:
                success = self._load_single_sample(sample_info)
                if success:
                    self.stats['processed_samples'] += 1
                    self.logger.info(f" [{i}/{len(sample_files)}] {sample_id} 完成")
                else:
                    self.stats['failed_samples'].append({'sample_id': sample_id, 'error': 'Unknown'})
            except Exception as e:
                self.logger.error(f" [{i}/{len(sample_files)}] {sample_id} 失败: {e}")
                traceback.print_exc()
                self.stats['failed_samples'].append({'sample_id': sample_id, 'error': str(e)})
        
        self.stats['processing_time'] = time.time() - start_time
        
        
        if self.config.save_statistics:
            self._save_global_stats()
        
        self._print_summary()
        return self.stats
    
    def _find_sample_files(self) -> List[Dict[str, Any]]:
        """Find sample files."""
        l0_dir = Path(self.config.l0_graphs_dir)
        l1_dir = Path(self.config.l1_extracted_dir)
        l2_dir = Path(self.config.l2_aggregated_dir)
        
        sample_files = []
        
        for l0_file in sorted(l0_dir.glob("*_l0_graph.json")):
            sample_id = l0_file.stem.replace('_l0_graph', '')
            
            l1_file = l1_dir / f"{sample_id}_l1_extracted.json"
            l2_file = l2_dir / f"{sample_id}_l2_aggregated.json"
            
            if l1_file.exists() and l2_file.exists():
                sample_files.append({
                    'sample_id': sample_id,
                    'l0_file': str(l0_file),
                    'l1_file': str(l1_file),
                    'l2_file': str(l2_file)
                })
            else:
                missing = []
                if not l1_file.exists():
                    missing.append('L1')
                if not l2_file.exists():
                    missing.append('L2')
                self.logger.warning(f"样本 {sample_id} 缺少文件: {missing}")
        
        return sample_files
    
    def _load_single_sample(self, sample_info: Dict[str, Any]) -> bool:
        """Load single sample."""
        sample_id = sample_info['sample_id']
        sample_start = time.time()
        
        try:
            semantic_graph = self._create_semantic_graph()
            
            self._create_memory_spaces(semantic_graph)
            
            
            l0_stats = self._load_l0_data(semantic_graph, sample_info['l0_file'], sample_id)
            
            
            l1_stats = self._load_l1_data(semantic_graph, sample_info['l1_file'], sample_id)
            
            
            l2_stats = self._load_l2_data(semantic_graph, sample_info['l2_file'], sample_id)
            
            relation_stats = {'hierarchical': 0}
            if self.config.enable_hierarchical_relations:
                relation_stats = self._build_hierarchical_relations(
                    semantic_graph, sample_id, l0_stats, l1_stats, l2_stats
                )
            
            
            splade_stats = None
            if self.config.build_splade:
                splade_stats = self._build_splade_embeddings(semantic_graph, sample_id)
            
            
            if self.config.enable_index_building:
                semantic_graph.rebuild_all_indexes()

            
            output_path = self._save_sample_graph(semantic_graph, sample_id, {
                'l0': l0_stats,
                'l1': l1_stats,
                'l2': l2_stats,
                'relations': relation_stats,
                'splade': splade_stats
            })
            
            sample_time = time.time() - sample_start
            self.stats['sample_details'][sample_id] = {
                'l0_units': l0_stats['units_loaded'],
                'l0_relations': l0_stats['relations_loaded'],
                'l1_units': l1_stats['units_loaded'],
                'l2_units': l2_stats,
                'hierarchical_relations': relation_stats.get('hierarchical', 0),
                'processing_time': sample_time,
                'output_path': output_path
            }
            
            self.stats['l0_units_loaded'] += l0_stats['units_loaded']
            self.stats['l0_relations_loaded'] += l0_stats['relations_loaded']
            self.stats['l1_units_loaded'] += l1_stats['units_loaded']
            self.stats['hierarchical_relations'] += relation_stats.get('hierarchical', 0)
            
            for key, val in l2_stats.items():
                if key in self.stats['l2_units_loaded']:
                    self.stats['l2_units_loaded'][key] += val
            
            self.logger.info(
                f"    L0={l0_stats['units_loaded']} | L1={l1_stats['units_loaded']} | "
                f"L2={sum(l2_stats.values())} | 关系={relation_stats.get('hierarchical', 0)} | "
                f"耗时={sample_time:.2f}s"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"加载样本 {sample_id} 失败: {e}")
            traceback.print_exc()
            return False
    
    
    
    
    def _create_semantic_graph(self) -> SemanticGraph:
        """Create semantic graph."""
        model_config = SemanticMap.MODEL_CONFIG.get(self.config.embedding_model)
        embedding_dim = model_config['dim'] if model_config else None
        
        semantic_map = SemanticMap(
            embedding_model_name=self.config.embedding_model,
            embedding_dim=embedding_dim,
            faiss_index_type="IDMap,Flat"
        )
        
        semantic_graph = SemanticGraph(semantic_map_instance=semantic_map)
        self.logger.debug(f"SemanticGraph 创建完成 (dim={semantic_map.embedding_dim})")
        return semantic_graph
    
    def _create_memory_spaces(self, semantic_graph: SemanticGraph):
        """Create memory spaces."""
        for space_name in MemorySpaceNames.all_spaces():
            semantic_graph.create_memory_space_in_map(space_name)
        self.logger.debug(f"创建了 {len(MemorySpaceNames.all_spaces())} 个 Memory Spaces")
    
    
    
    
    
    def _load_l0_data(self, 
                      semantic_graph: SemanticGraph,
                      l0_file: str,
                      sample_id: str) -> Dict[str, int]:
        """Load L0 data."""
        stats = {'units_loaded': 0, 'relations_loaded': 0}
        
        with open(l0_file, 'r', encoding='utf-8') as f:
            l0_data = json.load(f)
        
        units_to_add = []
        l0_uids = set()
        
        for conv in l0_data.get('l0_conversations', []):
            try:
                uid = conv['uid']
                l0_uids.add(uid)
                
                indexing_text = conv.get('indexing_text', '')
                if not indexing_text:
                    raw_data = conv.get('raw_data', {})
                    indexing_text = raw_data.get('message', raw_data.get('text_content', ''))
                
                raw_data = conv.get('raw_data', {}).copy()
                raw_data['text_content'] = indexing_text
                raw_data['original_indexing_text'] = conv.get('indexing_text', '')
                raw_data['type'] = 'conversation_message'
                
                metadata = conv.get('metadata', {}).copy()
                metadata['memory_level'] = 'L0'
                metadata['sample_id'] = sample_id
                metadata['retrieval_field'] = 'indexing_text'
                
                unit = MemoryUnit(uid=uid, raw_data=raw_data, metadata=metadata)
                units_to_add.append(unit)
                
            except Exception as e:
                self.logger.warning(f"加载 L0 单元失败: {e}")
        
        if units_to_add:
            batch_stats = semantic_graph.batch_add_units(
                units=units_to_add,
                batch_size=self.config.batch_size,
                space_names=[MemorySpaceNames.L0_OBSERVATION],
                index_update_mode="none",
                generate_sparse_embedding=False,
                show_progress=False
            )
            stats['units_loaded'] = batch_stats.get('added', 0)
        
        
        if self.config.enable_relation_loading:
            for rel in l0_data.get('l0_relationships', []):
                try:
                    if rel['source_uid'] in l0_uids and rel['target_uid'] in l0_uids:
                        semantic_graph.add_relationship(
                            source_uid=rel['source_uid'],
                            target_uid=rel['target_uid'],
                            relationship_name=rel['type'],
                            **rel.get('properties', {})
                        )
                        stats['relations_loaded'] += 1
                except Exception as e:
                    self.logger.debug(f"加载 L0 关系失败: {e}")
        
        return stats
    
    
    
    
    
    def _load_l1_data(self,
                      semantic_graph: SemanticGraph,
                      l1_file: str,
                      sample_id: str) -> Dict[str, int]:
        """Load L1 data."""
        stats = {'units_loaded': 0, 'session_uids': []}
        
        with open(l1_file, 'r', encoding='utf-8') as f:
            l1_data = json.load(f)
        
        participants = l1_data.get('participants', ['Speaker_A', 'Speaker_B'])
        units_to_add = []
        
        for session in l1_data.get('session_extractions', []):
            try:
                session_id = session.get('session_id', 'unknown')
                uid = f"{sample_id}_{session_id}_l1_facts"
                
                text_content = self._build_l1_text_content(session, participants)
                
                raw_data = {
                    'text_content': text_content,
                    'type': 'session_facts',
                    'session_id': session_id,
                    'session_date': session.get('session_date', 'unknown'),
                    'session_topic': session.get('session_topic', ''),
                    'structured_events': session.get('structured_events', []),
                    'state_updates': session.get('state_updates', []),
                    'countable_items': session.get('countable_items', []),
                    'key_facts': session.get('key_facts', []),
                    'mentioned_dates': session.get('mentioned_dates', [])
                }
                
                metadata = {
                    'memory_level': 'L1',
                    'content_type': 'session_facts',
                    'sample_id': sample_id,
                    'session_id': session_id,
                    'participants': participants,
                    'retrieval_field': 'session_topic + key_facts',
                    'event_count': len(session.get('structured_events', [])),
                    'fact_count': len(session.get('key_facts', []))
                }
                
                unit = MemoryUnit(uid=uid, raw_data=raw_data, metadata=metadata)
                units_to_add.append(unit)
                stats['session_uids'].append(uid)
                
            except Exception as e:
                self.logger.warning(f"加载 L1 Session 失败: {e}")
        
        if units_to_add:
            batch_stats = semantic_graph.batch_add_units(
                units=units_to_add,
                batch_size=self.config.batch_size,
                space_names=[MemorySpaceNames.L1_SESSION_FACTS],
                index_update_mode="none",
                generate_sparse_embedding=False,
                show_progress=False
            )
            stats['units_loaded'] = batch_stats.get('added', 0)
        
        return stats
    
    def _build_l1_text_content(self, session: Dict, participants: List[str]) -> str:
        """Build L1 text content."""
        parts = []
        
        topic = session.get('session_topic', '')
        if topic:
            parts.append(f"Session Topic: {topic}")
        
        # 2. Session Date
        date = session.get('session_date', '')
        if date and date != 'unknown':
            parts.append(f"Date: {date}")
        
        key_facts = session.get('key_facts', [])
        if key_facts:
            fact_texts = []
            for fact in key_facts[:5]:
                subject = fact.get('subject', '')
                fact_text = fact.get('fact', '')
                if subject and fact_text:
                    fact_texts.append(f"- {subject}: {fact_text}")
            if fact_texts:
                parts.append("Key Facts:\n" + "\n".join(fact_texts))
        
        if len(key_facts) < 2:
            events = session.get('structured_events', [])
            if events:
                event_texts = []
                for event in events[:3]:
                    name = event.get('event_name', '')
                    event_type = event.get('event_type', '')
                    if name:
                        event_texts.append(f"- [{event_type}] {name}")
                if event_texts:
                    parts.append("Events:\n" + "\n".join(event_texts))
        
        return "\n\n".join(parts) if parts else "(No session content)"
    
    
    
    
    
    def _load_l2_data(self,
                      semantic_graph: SemanticGraph,
                      l2_file: str,
                      sample_id: str) -> Dict[str, int]:
        """Load L2 data."""
        stats = {
            'activity_ledger': 0,
            'entity_profiles': 0,
            'timeline_events': 0,
            'social_graph': 0,
            'negative_constraints': 0,
        }
        
        with open(l2_file, 'r', encoding='utf-8') as f:
            l2_data = json.load(f)
        
        
        if self.config.load_l2_activity_ledger:
            stats['activity_ledger'] = self._load_l2_activity_ledger(
                semantic_graph, l2_data, sample_id
            )
        
        
        if self.config.load_l2_entity_profiles:
            stats['entity_profiles'] = self._load_l2_entity_profiles(
                semantic_graph, l2_data, sample_id
            )
        
        
        if self.config.load_l2_timeline:
            stats['timeline_events'] = self._load_l2_timeline(
                semantic_graph, l2_data, sample_id
            )
        
        
        if self.config.load_l2_social_graph:
            stats['social_graph'] = self._load_l2_social_graph(
                semantic_graph, l2_data, sample_id
            )
        
        
        if self.config.load_l2_negative_constraints:
            stats['negative_constraints'] = self._load_l2_negative_constraints(
                semantic_graph, l2_data, sample_id
            )
        
        return stats
    
    def _load_l2_activity_ledger(self,
                                  semantic_graph: SemanticGraph,
                                  l2_data: Dict,
                                  sample_id: str) -> int:
        """Load L2 activity ledger."""
        ledger = l2_data.get('activity_ledger', [])
        if not ledger:
            return 0
        
        units_to_add = []
        
        for i, entry in enumerate(ledger):
            try:
                activity = entry.get('activity', f'Activity_{i}')
                safe_name = re.sub(r'[^a-zA-Z0-9]', '_', activity.lower()).strip('_')
                uid = f"{sample_id}_l2_activity_{safe_name}_{i}"
                
                count = entry.get('count', 0)
                instances = entry.get('instances', [])
                
                text_parts = [f"Activity: {activity} | Count: {count} times"]
                if instances:
                    text_parts.append(f"Details: {', '.join(str(inst) for inst in instances)}")
                text_content = " | ".join(text_parts)
                
                raw_data = {
                    'text_content': text_content,
                    'type': 'activity_ledger',
                    'activity': activity,
                    'count': count,
                    'instances': instances,
                }
                
                metadata = {
                    'memory_level': 'L2',
                    'content_type': 'activity_ledger',
                    'sample_id': sample_id,
                    'retrieval_field': 'activity + count + instances',
                }
                
                unit = MemoryUnit(uid=uid, raw_data=raw_data, metadata=metadata)
                units_to_add.append(unit)
                
            except Exception as e:
                self.logger.warning(f"加载 activity ledger entry 失败: {e}")
        
        if units_to_add:
            batch_stats = semantic_graph.batch_add_units(
                units=units_to_add,
                batch_size=self.config.batch_size,
                space_names=[MemorySpaceNames.L2_ACTIVITY_LEDGER],
                index_update_mode="none",
                generate_sparse_embedding=False,
                show_progress=False
            )
            return batch_stats.get('added', 0)
        
        return 0
    
    def _load_l2_entity_profiles(self,
                                  semantic_graph: SemanticGraph,
                                  l2_data: Dict,
                                  sample_id: str) -> int:
        """Load L2 entity profiles."""
        profiles = l2_data.get('entity_profiles', [])
        if not profiles:
            return 0
        
        units_to_add = []
        
        for i, profile in enumerate(profiles):
            try:
                entity = profile.get('entity', f'Entity_{i}')
                attribute = profile.get('attribute', '')
                safe_entity = re.sub(r'[^a-zA-Z0-9]', '_', entity.lower()).strip('_')
                safe_attr = re.sub(r'[^a-zA-Z0-9]', '_', attribute.lower()).strip('_')
                uid = f"{sample_id}_l2_entity_{safe_entity}_{safe_attr}_{i}"
                
                value = profile.get('value', '')
                context = profile.get('context', '')
                
                text_parts = [f"Entity: {entity}"]
                if attribute:
                    text_parts.append(f"Attribute: {attribute}")
                if value:
                    text_parts.append(f"Value: {value}")
                if context:
                    text_parts.append(f"Context: {context}")
                text_content = " | ".join(text_parts)
                
                raw_data = {
                    'text_content': text_content,
                    'type': 'entity_profile',
                    'entity': entity,
                    'attribute': attribute,
                    'value': value,
                    'context': context,
                }
                
                metadata = {
                    'memory_level': 'L2',
                    'content_type': 'entity_profile',
                    'sample_id': sample_id,
                    'entity_name': entity,
                    'retrieval_field': 'entity + attribute + value + context',
                }
                
                unit = MemoryUnit(uid=uid, raw_data=raw_data, metadata=metadata)
                units_to_add.append(unit)
                
            except Exception as e:
                self.logger.warning(f"加载 entity profile 失败: {e}")
        
        if units_to_add:
            batch_stats = semantic_graph.batch_add_units(
                units=units_to_add,
                batch_size=self.config.batch_size,
                space_names=[MemorySpaceNames.L2_ENTITY_PROFILE],
                index_update_mode="none",
                generate_sparse_embedding=False,
                show_progress=False
            )
            return batch_stats.get('added', 0)
        
        return 0
    
    def _load_l2_timeline(self,
                          semantic_graph: SemanticGraph,
                          l2_data: Dict,
                          sample_id: str) -> int:
        """Load L2 timeline."""
        timeline = l2_data.get('master_timeline', [])
        if not timeline:
            return 0
        
        units_to_add = []
        
        for i, entry in enumerate(timeline):
            if i >= self.config.max_timeline_events:
                break
            
            try:
                date = entry.get('date', 'unknown')
                event_desc = entry.get('event', '')
                is_estimated = entry.get('is_estimated', False)
                
                safe_date = re.sub(r'[^a-zA-Z0-9]', '', date)
                uid = f"{sample_id}_l2_timeline_{safe_date}_{i}"
                
                text_parts = [f"Date: {date}", f"Event: {event_desc}"]
                if is_estimated:
                    text_parts.append("(estimated date)")
                text_content = " | ".join(text_parts)
                
                raw_data = {
                    'text_content': text_content,
                    'type': 'timeline_event',
                    'date': date,
                    'event': event_desc,
                    'is_estimated': is_estimated,
                }
                
                metadata = {
                    'memory_level': 'L2',
                    'content_type': 'timeline_event',
                    'sample_id': sample_id,
                    'event_date': date,
                    'retrieval_field': 'date + event',
                }
                
                unit = MemoryUnit(uid=uid, raw_data=raw_data, metadata=metadata)
                units_to_add.append(unit)
                
            except Exception as e:
                self.logger.warning(f"加载 timeline event 失败: {e}")
        
        if units_to_add:
            batch_stats = semantic_graph.batch_add_units(
                units=units_to_add,
                batch_size=self.config.batch_size,
                space_names=[MemorySpaceNames.L2_TIMELINE],
                index_update_mode="none",
                generate_sparse_embedding=False,
                show_progress=False
            )
            return batch_stats.get('added', 0)
        
        return 0
    
    def _load_l2_social_graph(self,
                               semantic_graph: SemanticGraph,
                               l2_data: Dict,
                               sample_id: str) -> int:
        """Load L2 social graph."""
        social = l2_data.get('social_graph', [])
        if not social:
            return 0
        
        units_to_add = []
        
        for i, entry in enumerate(social):
            try:
                person = entry.get('person', f'Person_{i}')
                safe_person = re.sub(r'[^a-zA-Z0-9]', '_', person.lower()).strip('_')
                uid = f"{sample_id}_l2_social_{safe_person}_{i}"
                
                relationship = entry.get('relationship', '')
                key_interaction = entry.get('key_interaction', '')
                
                text_parts = [f"Person: {person}"]
                if relationship:
                    text_parts.append(f"Relationship: {relationship}")
                if key_interaction:
                    text_parts.append(f"Key Interaction: {key_interaction}")
                text_content = " | ".join(text_parts)
                
                raw_data = {
                    'text_content': text_content,
                    'type': 'social_graph',
                    'person': person,
                    'relationship': relationship,
                    'key_interaction': key_interaction,
                }
                
                metadata = {
                    'memory_level': 'L2',
                    'content_type': 'social_graph',
                    'sample_id': sample_id,
                    'person_name': person,
                    'retrieval_field': 'person + relationship + key_interaction',
                }
                
                unit = MemoryUnit(uid=uid, raw_data=raw_data, metadata=metadata)
                units_to_add.append(unit)
                
            except Exception as e:
                self.logger.warning(f"加载 social graph entry 失败: {e}")
        
        if units_to_add:
            batch_stats = semantic_graph.batch_add_units(
                units=units_to_add,
                batch_size=self.config.batch_size,
                space_names=[MemorySpaceNames.L2_SOCIAL_GRAPH],
                index_update_mode="none",
                generate_sparse_embedding=False,
                show_progress=False
            )
            return batch_stats.get('added', 0)
        
        return 0
    
    def _load_l2_negative_constraints(self,
                                       semantic_graph: SemanticGraph,
                                       l2_data: Dict,
                                       sample_id: str) -> int:
        """Load L2 negative constraints."""
        constraints = l2_data.get('negative_constraints', [])
        if not constraints:
            return 0
        
        units_to_add = []
        
        for i, constraint in enumerate(constraints):
            try:
                uid = f"{sample_id}_l2_negconstraint_{i}"
                
                text_content = constraint if isinstance(constraint, str) else str(constraint)
                
                raw_data = {
                    'text_content': text_content,
                    'type': 'negative_constraint',
                    'constraint_index': i,
                }
                
                metadata = {
                    'memory_level': 'L2',
                    'content_type': 'negative_constraint',
                    'sample_id': sample_id,
                    'retrieval_field': 'constraint_text',
                }
                
                unit = MemoryUnit(uid=uid, raw_data=raw_data, metadata=metadata)
                units_to_add.append(unit)
                
            except Exception as e:
                self.logger.warning(f"加载 negative constraint 失败: {e}")
        
        if units_to_add:
            batch_stats = semantic_graph.batch_add_units(
                units=units_to_add,
                batch_size=self.config.batch_size,
                space_names=[MemorySpaceNames.L2_NEGATIVE_CONSTRAINT],
                index_update_mode="none",
                generate_sparse_embedding=False,
                show_progress=False
            )
            return batch_stats.get('added', 0)
        
        return 0
    
    
    
    
    def _build_hierarchical_relations(self,
                                       semantic_graph: SemanticGraph,
                                       sample_id: str,
                                       l0_stats: Dict,
                                       l1_stats: Dict,
                                       l2_stats: Dict) -> Dict[str, int]:
        """Build hierarchical relations."""
        stats = {'hierarchical': 0, 'l0_to_l1': 0, 'l1_to_l2': 0}
        
        
        l1_session_uids = l1_stats.get('session_uids', [])
        
        l0_space = semantic_graph.semantic_map.memory_spaces.get(MemorySpaceNames.L0_OBSERVATION)
        if l0_space and l1_session_uids:
            for l0_uid in l0_space.get_unit_uids():
                unit = semantic_graph.semantic_map.get_unit(l0_uid)
                if not unit:
                    continue
                
                session_id = unit.raw_data.get('session_id') or \
                             (unit.metadata.get('session_id') if unit.metadata else None)
                
                if session_id:
                    target_uid = f"{sample_id}_{session_id}_l1_facts"
                    if target_uid in [uid for uid in l1_session_uids]:
                        try:
                            semantic_graph.add_relationship(
                                source_uid=l0_uid,
                                target_uid=target_uid,
                                relationship_name="belongs_to_session",
                                layer_transition="L0_to_L1"
                            )
                            stats['l0_to_l1'] += 1
                            stats['hierarchical'] += 1
                        except Exception as e:
                            self.logger.debug(f"L0->L1 关系失败: {e}")
        
        
        l2_uids = set()
        for space_name in MemorySpaceNames.l2_spaces():
            l2_space = semantic_graph.semantic_map.memory_spaces.get(space_name)
            if l2_space:
                l2_uids.update(l2_space.get_unit_uids())
        
        for l1_uid in l1_session_uids:
            for l2_uid in l2_uids:
                try:
                    semantic_graph.add_relationship(
                        source_uid=l1_uid,
                        target_uid=l2_uid,
                        relationship_name="aggregated_into",
                        layer_transition="L1_to_L2"
                    )
                    stats['l1_to_l2'] += 1
                    stats['hierarchical'] += 1
                except Exception as e:
                    self.logger.debug(f"L1->L2 关系失败: {e}")
        
        self.logger.debug(f"层级关系: L0->L1={stats['l0_to_l1']}, L1->L2={stats['l1_to_l2']}")
        return stats
    
    
    
    
    
    def _build_splade_embeddings(self,
                                  semantic_graph: SemanticGraph,
                                  sample_id: str) -> Optional[Dict[str, Any]]:
        """Build splade embeddings."""
        if not self.config.build_splade:
            return None
        
        try:
            import gc
            import torch
            
            self.logger.debug(f"构建 SPLADE 向量...")
            
            splade_stats = semantic_graph.build_sparse_embeddings(
                units=None,
                model_name=self.config.splade_model,
                batch_size=min(self.config.splade_batch_size, 32),
                force_rebuild=False,
                show_progress=False
            )
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            self.stats['splade_stats']['total_processed'] += splade_stats.get('processed', 0)
            self.stats['splade_stats']['total_skipped'] += splade_stats.get('skipped', 0)
            self.stats['splade_stats']['total_failed'] += splade_stats.get('failed', 0)
            
            return splade_stats
            
        except Exception as e:
            self.logger.warning(f"SPLADE 构建失败: {e}")
            return None
    
    
    
    

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
                           load_stats: Dict) -> str:
        """Save sample graph."""
        output_dir = os.path.join(self.config.output_dir, sample_id)
        os.makedirs(output_dir, exist_ok=True)
        
        
        semantic_graph.save_graph(output_dir, freeze_retrievers=self.config.freeze_retrievers)
        retrieval_state = self._load_saved_retrieval_state(output_dir)
        
        
        metadata = {
            'sample_id': sample_id,
            'build_timestamp': datetime.now().isoformat(),
            'memory_spaces': MemorySpaceNames.all_spaces(),
            'retrieval_fields': {
                'L0_Observation': 'indexing_text (Contextual Retrieval enhanced)',
                'L1_SessionFacts': 'session_topic + key_facts',
                'L2_ActivityLedger': 'activity + count + instances',
                'L2_EntityProfile': 'entity + attribute + value + context',
                'L2_Timeline': 'date + event',
                'L2_SocialGraph': 'person + relationship + key_interaction',
                'L2_NegativeConstraint': 'constraint_text',
            },
            'load_statistics': load_stats,
            'static_retriever_indexes': retrieval_state,
            'config': {
                'embedding_model': self.config.embedding_model,
                'splade_enabled': self.config.build_splade,
                'splade_model': self.config.splade_model if self.config.build_splade else None,
                'freeze_retrievers': self.config.freeze_retrievers
            }
        }
        
        metadata_file = os.path.join(output_dir, "sample_metadata.json")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        return output_dir
    
    def _save_global_stats(self):
        """Save global stats."""
        stats_file = os.path.join(self.config.output_dir, "loading_stats.json")
        
        output = {
            **self.stats,
            'config': {
                'embedding_model': self.config.embedding_model,
                'build_splade': self.config.build_splade,
                'splade_model': self.config.splade_model,
                'freeze_retrievers': self.config.freeze_retrievers
            },
            'completion_time': datetime.now().isoformat()
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f" 统计已保存: {stats_file}")
    
    def _print_summary(self):
        """Run print summary."""
        print("\n" + "=" * 70)
        print(" Step 4 加载摘要")
        print("=" * 70)
        print(f"总样本数:           {self.stats['total_samples']}")
        print(f"处理成功:           {self.stats['processed_samples']}")
        print(f"处理失败:           {len(self.stats['failed_samples'])}")
        print(f"L0 单元:            {self.stats['l0_units_loaded']}")
        print(f"L0 关系:            {self.stats['l0_relations_loaded']}")
        print(f"L1 单元:            {self.stats['l1_units_loaded']}")
        print(f"L2 Activities:      {self.stats['l2_units_loaded']['activity_ledger']}")
        print(f"L2 Entities:        {self.stats['l2_units_loaded']['entity_profiles']}")
        print(f"L2 Timeline:        {self.stats['l2_units_loaded']['timeline_events']}")
        print(f"L2 Social:          {self.stats['l2_units_loaded']['social_graph']}")
        print(f"L2 Constraints:     {self.stats['l2_units_loaded']['negative_constraints']}")
        print(f"层级关系:           {self.stats['hierarchical_relations']}")
        print(f"处理时间:           {self.stats['processing_time']:.2f}s")
        
        if self.config.build_splade:
            splade = self.stats['splade_stats']
            print(f"SPLADE 处理:        {splade['total_processed']}")
        
        print(f"输出目录:           {self.config.output_dir}")
        print("=" * 70 + "\n")



# CLI


def main():
    """Run the command-line entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Step 4: 加载分层JSON数据到SemanticGraph"
    )
    
    parser.add_argument(
        "--l0-dir",
        default=str(paths.LOCOMO_HIERARCHICAL_CONTENT_STEP1_DIR),
        help="L0 图谱目录"
    )
    parser.add_argument(
        "--l1-dir",
        default=str(paths.LOCOMO_HIERARCHICAL_CONTENT_STEP2_DIR),
        help="L1 提取结果目录"
    )
    parser.add_argument(
        "--l2-dir",
        default=str(paths.LOCOMO_HIERARCHICAL_CONTENT_STEP3_DIR),
        help="L2 God-Mode 结构化数据目录"
    )
    
    parser.add_argument(
        "--output-dir",
        default=str(paths.LOCOMO_HIERARCHICAL_CONTENT_STEP4_DIR),
        help="输出目录"
    )
    
    parser.add_argument(
        "--embedding-model",
        default="Qwen/Qwen3-Embedding-0.6B",
        help="嵌入模型"
    )
    
    
    parser.add_argument("--no-splade", action="store_true", help="禁用 SPLADE")
    parser.add_argument("--splade-model", default="naver/splade-v3", help="SPLADE 模型")
    parser.add_argument("--splade-batch-size", type=int, default=32, help="SPLADE 批大小")
    
    parser.add_argument("--no-relations", action="store_true", help="不加载 L0 关系")
    parser.add_argument("--no-hierarchical", action="store_true", help="不构建层级关系")
    parser.add_argument("--no-index", action="store_true", help="不构建索引")
    parser.add_argument("--no-freeze-retrievers", action="store_true", help="不在保存阶段冻结 BM25/SPLADE 静态加速索引")
    
    parser.add_argument("--no-l2-activities", action="store_true", help="不加载 L2 活动账本")
    parser.add_argument("--no-l2-entities", action="store_true", help="不加载 L2 实体属性")
    parser.add_argument("--no-l2-timeline", action="store_true", help="不加载 L2 时间线")
    parser.add_argument("--no-l2-social", action="store_true", help="不加载 L2 社交关系")
    parser.add_argument("--no-l2-constraints", action="store_true", help="不加载 L2 负面约束")
    parser.add_argument("--max-timeline-events", type=int, default=50, help="最大时间线事件数")
    
    parser.add_argument("--debug", action="store_true", help="调试模式")
    
    args = parser.parse_args()
    
    config = Step4LoaderConfig(
        l0_graphs_dir=args.l0_dir,
        l1_extracted_dir=args.l1_dir,
        l2_aggregated_dir=args.l2_dir,
        output_dir=args.output_dir,
        embedding_model=args.embedding_model,
        build_splade=not args.no_splade,
        splade_model=args.splade_model,
        splade_batch_size=args.splade_batch_size,
        enable_relation_loading=not args.no_relations,
        enable_hierarchical_relations=not args.no_hierarchical,
        enable_index_building=not args.no_index,
        freeze_retrievers=not args.no_freeze_retrievers,
        load_l2_activity_ledger=not args.no_l2_activities,
        load_l2_entity_profiles=not args.no_l2_entities,
        load_l2_timeline=not args.no_l2_timeline,
        load_l2_social_graph=not args.no_l2_social,
        load_l2_negative_constraints=not args.no_l2_constraints,
        max_timeline_events=args.max_timeline_events,
        debug_mode=args.debug
    )
    
    loader = Step4GraphLoader(config)
    
    try:
        stats = loader.load_all_graphs()
        print(" Step 4 完成!")
        return 0
    except Exception as e:
        print(f" Step 4 失败: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
