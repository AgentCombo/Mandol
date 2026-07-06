"""Utilities for step3 locomo entity relation semantic graph batch."""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
import threading
import time
import traceback
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime
import numpy as np
import gc
import torch


from mandol.core.semantic_graph import SemanticGraph
from mandol.core.semantic_map import SemanticMap
from mandol.core.memory_unit import MemoryUnit
from mandol.llm.llm_client import LLMClient
from mandol.core import paths

class LoCoMoSemanticGraphInserter:
    
    def __init__(self, 
                 text_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 llm_client: Optional[LLMClient] = None,
                 
                 build_splade: bool = True,
                 splade_model: str = "naver/splade-v3",
                 splade_batch_size: int = 32,
                 freeze_retrievers: bool = True):
        self.text_embedding_model = text_embedding_model
        self.llm_client = llm_client
        
        
        self.build_splade = build_splade
        self.splade_model = splade_model
        self.splade_batch_size = splade_batch_size
        self.freeze_retrievers = freeze_retrievers
        
        self.stats = {
            "entity_hubs_created": 0,
            "evidence_mentions_created": 0,
            "entity_relations_created": 0,
            "mention_relations_created": 0,
            "memory_spaces_created": 0,
            "processing_time": 0.0,
            "splade_stats": {},  
            "freeze_retrievers": freeze_retrievers,
            "errors": []
        }
        
        logging.info("LoCoMo SemanticGraph 插入器初始化完成")
        if build_splade:
            logging.info(f" SPLADE 默认启用: 模型={splade_model}, 批大小={splade_batch_size}")
        else:
            logging.info(" SPLADE 已禁用")
        logging.info(f" BM25/SPLADE 静态索引冻结: {'启用' if freeze_retrievers else '禁用'}")
    
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
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            logging.debug(" 资源清理完成")
            
        except Exception as e:
            logging.warning(f"清理资源时出错: {e}")

    def create_semantic_graph(self) -> SemanticGraph:
        """Build semantic graph."""
        try:
            semantic_map = SemanticMap(
                embedding_model_name=self.text_embedding_model,
                embedding_dim=None,
                faiss_index_type="IDMap,Flat"
            )
            
            semantic_graph = SemanticGraph(semantic_map_instance=semantic_map)
            
            logging.info(f"SemanticGraph创建成功，使用嵌入模型: {self.text_embedding_model}")
            return semantic_graph
            
        except Exception as e:
            logging.error(f"创建SemanticGraph失败: {e}")
            raise
        
    def _build_splade_embeddings(self, 
                                semantic_graph: SemanticGraph,
                                conversation_id: str) -> Optional[Dict[str, Any]]:
        """Build splade embeddings."""
        if not self.build_splade:
            return None
        
        try:
            logging.info(f" [{conversation_id}] 开始构建 SPLADE 稀疏向量...")
            
            effective_batch_size = min(self.splade_batch_size, 32)
            
            splade_stats = semantic_graph.build_sparse_embeddings(
                units=None,
                # text_field="text_content",
                model_name=self.splade_model,
                batch_size=effective_batch_size,
                force_rebuild=False,
                show_progress=True
            )
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logging.info(
                f" [{conversation_id}] SPLADE 构建完成: "
                f"总计 {splade_stats.get('total', 0)} | "
                f"处理 {splade_stats.get('processed', 0)} | "
                f"跳过 {splade_stats.get('skipped', 0)} | "
                f"失败 {splade_stats.get('failed', 0)}"
            )
            
            return splade_stats
            
        except Exception as e:
            logging.error(f" [{conversation_id}] SPLADE 构建失败: {e}")
            logging.debug(traceback.format_exc())
            
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
            
            return None
        
    def _find_entity_relation_files(self, input_path: Path, 
                               sample_ids: Optional[List[str]] = None) -> List[Path]:
        """Find entity relation files."""
        json_files = []
        
        if sample_ids:
            for sample_id in sample_ids:
                sample_dir = input_path / sample_id
                json_file = sample_dir / f"{sample_id}_complete_entity_relation.json"
                if json_file.exists():
                    json_files.append(json_file)
                else:
                    logging.warning(f"文件不存在: {json_file}")
        else:
            json_files = list(input_path.glob("*/*_complete_entity_relation.json"))
        
        
        json_files = sorted(json_files)
        
        return json_files
    
    def load_entity_relation_data(self, json_file_path: str) -> Dict[str, Any]:
        """Load entity relation data."""
        try:
            if not os.path.exists(json_file_path):
                raise FileNotFoundError(f"文件不存在: {json_file_path}")
            
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            conversation_id = data.get("conversation_id", "unknown")
            entities = data.get("entities", [])
            relations = data.get("relations", [])
            
            logging.info(f"数据加载完成: {conversation_id}")
            logging.info(f"  - 实体数量: {len(entities)}")
            logging.info(f"  - 关系数量: {len(relations)}")
            
            return data
            
        except Exception as e:
            logging.error(f"加载数据失败: {e}")
            raise
    
    def insert_entities_to_graph(self, semantic_graph: SemanticGraph, 
                                entities: List[Dict[str, Any]], 
                                conversation_id: str) -> Dict[str, str]:
        """Run insert entities to graph."""
        entity_hub_mapping = {}
        
        logging.info("开始插入实体数据...")
        
        conversation_space = f"conversation_{conversation_id}"
        semantic_graph.create_memory_space_in_map(conversation_space)
        self.stats["memory_spaces_created"] += 1
        
        all_units = []
        pending_relations = []  
        
        for entity in entities:
            try:
                entity_id = entity.get("entity_id", "")
                entity_name = entity.get("name", "")
                entity_type = entity.get("entity_type", "Unknown")
                confidence = entity.get("confidence", 0.0)
                mentions = entity.get("mentions", [])
                extraction_metadata = entity.get("extraction_metadata", {})
                
                if not entity_id:
                    logging.warning("实体缺少entity_id，跳过")
                    continue
                
                hub_uid = f"{conversation_id}_{entity_id}_hub"
                entity_hub_mapping[entity_id] = hub_uid
                
                hub_unit = MemoryUnit(
                    uid=hub_uid,
                    raw_data={
                        "node_type": "entity_hub",
                        "conversation_id": conversation_id,
                        "original_entity_id": entity_id,
                        "name": entity_name,
                        "entity_type": entity_type,
                        "confidence": confidence,
                        "mentions_count": len(mentions),
                        "extraction_metadata": extraction_metadata,
                        "text_content": entity_name,
                        "created_at": datetime.now().isoformat()
                    }
                )
                all_units.append(hub_unit)
                
                for mention_idx, mention in enumerate(mentions):
                    mention_uid = f"{conversation_id}_{entity_id}_mention_{mention_idx}"
                    
                    mention_context = self._build_mention_context(mention, entity_name, entity_type)
                    
                    mention_unit = MemoryUnit(
                        uid=mention_uid,
                        raw_data={
                            "node_type": "evidence_mention",
                            "conversation_id": conversation_id,
                            "parent_entity_id": entity_id,
                            "parent_hub_uid": hub_uid,
                            "entity_name": entity_name,
                            "entity_type": entity_type,
                            "session_id": mention.get("session_id", ""),
                            "context": mention.get("context", ""),
                            "temporal_info": mention.get("temporal_info"),
                            "spatial_info": mention.get("spatial_info"),
                            "aliases": mention.get("aliases", []),
                            "confidence": mention.get("confidence", confidence),
                            "text_content": mention_context,
                            "created_at": datetime.now().isoformat()
                        }
                    )
                    all_units.append(mention_unit)
                    
                    pending_relations.append({
                        "source_uid": mention_uid,
                        "target_uid": hub_uid,
                        "relationship_name": "MENTION_OF",
                        "mention_index": mention_idx,
                        "session_id": mention.get("session_id", ""),
                        "confidence": mention.get("confidence", confidence)
                    })
                
                logging.debug(f"实体 {entity_name} ({entity_id}) 已准备: 1个hub + {len(mentions)}个mentions")
                
            except Exception as e:
                error_msg = f"准备实体 {entity.get('entity_id', 'unknown')} 失败: {e}"
                logging.error(error_msg)
                self.stats["errors"].append(error_msg)
        
        if all_units:
            logging.info(f" 批量添加 {len(all_units)} 个实体单元...")
            try:
                batch_stats = semantic_graph.batch_add_units(
                    units=all_units,
                    batch_size=32,
                    space_names=[conversation_space],
                    index_update_mode="none",  
                    generate_sparse_embedding=False,  
                    show_progress=True
                )
                self.stats["entity_hubs_created"] = sum(1 for u in all_units if u.raw_data.get("node_type") == "entity_hub")
                self.stats["evidence_mentions_created"] = sum(1 for u in all_units if u.raw_data.get("node_type") == "evidence_mention")
                logging.info(f" 批量添加完成: {batch_stats}")
            except Exception as e:
                logging.error(f"批量添加实体单元失败: {e}")
                self.stats["errors"].append(f"批量添加失败: {e}")
        
        logging.info(f" 建立 {len(pending_relations)} 个mention关系...")
        for rel in pending_relations:
            try:
                semantic_graph.add_relationship(
                    source_uid=rel["source_uid"],
                    target_uid=rel["target_uid"],
                    relationship_name=rel["relationship_name"],
                    mention_index=rel["mention_index"],
                    session_id=rel["session_id"],
                    confidence=rel["confidence"]
                )
                self.stats["mention_relations_created"] += 1
            except Exception as e:
                logging.warning(f"建立关系失败: {e}")
        
        logging.info(f"实体插入完成: {self.stats['entity_hubs_created']}个hub + {self.stats['evidence_mentions_created']}个mentions")
        return entity_hub_mapping
    
    def _build_mention_context(self, mention: Dict[str, Any], 
                              entity_name: str, entity_type: str) -> str:
        """Build mention context."""
        context_parts = []
        
        context_parts.append(f"Entity: {entity_name} (Type: {entity_type})")
        
        session_id = mention.get("session_id", "")
        if session_id:
            context_parts.append(f"Session: {session_id}")
        
        main_context = mention.get("context", "")
        if main_context:
            context_parts.append(f"Context: {main_context}")
        
        temporal_info = mention.get("temporal_info")
        if temporal_info:
            context_parts.append(f"Time: {temporal_info}")
        
        spatial_info = mention.get("spatial_info")
        if spatial_info:
            context_parts.append(f"Location: {spatial_info}")
        
        aliases = mention.get("aliases", [])
        if aliases:
            context_parts.append(f"Also known as: {', '.join(aliases)}")
        
        return " | ".join(context_parts)
    
    def insert_relations_to_graph(self, semantic_graph: SemanticGraph,
                                 relations: List[Dict[str, Any]],
                                 entity_hub_mapping: Dict[str, str],
                                 conversation_id: str):
        """Run insert relations to graph."""
        logging.info("开始插入关系数据...")
        
        for relation in relations:
            try:
                relation_id = relation.get("relation_id", "")
                head_entity_id = relation.get("head_entity_id", "")
                tail_entity_id = relation.get("tail_entity_id", "")
                relation_type = relation.get("relation_type", "RELATED_TO")
                confidence = relation.get("confidence", 0.0)
                sessions = relation.get("sessions", [])
                evidence_texts = relation.get("evidence_texts", [])
                contexts = relation.get("contexts", [])
                temporal_context = relation.get("temporal_context", "")
                
                
                source_hub_uid = entity_hub_mapping.get(head_entity_id)
                target_hub_uid = entity_hub_mapping.get(tail_entity_id)
                
                if not source_hub_uid or not target_hub_uid:
                    logging.warning(f"关系 {relation_id} 引用的实体不存在: {head_entity_id} -> {tail_entity_id}")
                    continue
                
                semantic_graph.add_relationship(
                    source_uid=source_hub_uid,
                    target_uid=target_hub_uid,
                    relationship_name=relation_type,
                    relation_id=relation_id,
                    confidence=confidence,
                    sessions=sessions,
                    evidence_texts=evidence_texts,
                    contexts=contexts,
                    temporal_context=temporal_context,
                    head_entity_name=relation.get("head_entity_name", ""),
                    tail_entity_name=relation.get("tail_entity_name", ""),
                    is_cross_session=len(sessions) > 1 if sessions else False,
                    conversation_id=conversation_id
                )
                self.stats["entity_relations_created"] += 1
                
                logging.debug(f"关系添加成功: {relation.get('head_entity_name', '')} -[{relation_type}]-> {relation.get('tail_entity_name', '')}")
                
            except Exception as e:
                error_msg = f"插入关系 {relation.get('relation_id', 'unknown')} 失败: {e}"
                logging.error(error_msg)
                self.stats["errors"].append(error_msg)
        
        logging.info(f"关系插入完成: {self.stats['entity_relations_created']}个关系")

    def create_semantic_spaces(self, semantic_graph: SemanticGraph,
                          session_metadata: Dict[str, Any],
                          conversation_id: str):
        """Build semantic spaces."""
        logging.info("创建多层次语义空间...")
        
        self._create_session_spaces(semantic_graph, session_metadata, conversation_id)
        
        self._create_entity_type_spaces(semantic_graph, conversation_id)
        
        self._create_qa_optimized_spaces(semantic_graph, conversation_id)
        
        self._create_temporal_spaces(semantic_graph, conversation_id)
        
        self._create_relationship_spaces(semantic_graph, conversation_id)

    def _create_session_spaces(self, semantic_graph: SemanticGraph,
                            session_metadata: Dict[str, Any],
                            conversation_id: str):
        """Create session spaces."""
        for session_id, session_info in session_metadata.items():
            try:
                session_space_name = f"{conversation_id}_{session_id}"
                semantic_graph.create_memory_space_in_map(session_space_name)
                
                session_mention_uids = []
                for unit_uid, unit in semantic_graph.semantic_map.memory_units.items():
                    if (unit.raw_data.get("node_type") == "evidence_mention" and 
                        unit.raw_data.get("session_id") == session_id):
                        session_mention_uids.append(unit_uid)
                
                for mention_uid in session_mention_uids:
                    semantic_graph.add_unit_to_space_in_map(mention_uid, session_space_name)
                
                logging.debug(f"会话空间 {session_space_name} 创建完成，包含 {len(session_mention_uids)} 个mentions")
                self.stats["memory_spaces_created"] += 1
                
            except Exception as e:
                error_msg = f"创建会话空间 {session_id} 失败: {e}"
                logging.error(error_msg)
                self.stats["errors"].append(error_msg)

    def _create_entity_type_spaces(self, semantic_graph: SemanticGraph, conversation_id: str):
        """Create entity type spaces."""
        logging.info("创建实体类型语义空间...")
        
        supported_entity_types = [
            "PERSON", "ORGANIZATION", "EVENT", "ACTIVITY", 
            "CONCEPT", "EMOTION", "LOCATION", "DATE_TIME", 
            "NUMERICAL_VALUE", "OBJECT", "SKILL", "RELATIONSHIP", "GOAL"
        ]
        
        type_counts = {}
        
        for entity_type in supported_entity_types:
            type_space_name = f"{conversation_id}_type_{entity_type.lower()}"
            
            type_units = []
            for unit_uid, unit in semantic_graph.semantic_map.memory_units.items():
                if (unit.raw_data.get("conversation_id") == conversation_id and
                    unit.raw_data.get("entity_type") == entity_type):
                    type_units.append(unit_uid)
            
            if type_units:
                try:
                    semantic_graph.create_memory_space_in_map(type_space_name)
                    
                    for unit_uid in type_units:
                        semantic_graph.add_unit_to_space_in_map(unit_uid, type_space_name)
                    
                    type_counts[entity_type] = len(type_units)
                    self.stats["memory_spaces_created"] += 1
                    
                    logging.debug(f"实体类型空间 {type_space_name} 创建完成，包含 {len(type_units)} 个单元")
                    
                except Exception as e:
                    error_msg = f"创建实体类型空间 {entity_type} 失败: {e}"
                    logging.error(error_msg)
                    self.stats["errors"].append(error_msg)
            else:
                logging.debug(f"实体类型 {entity_type} 没有单元，跳过创建空间")
        
        logging.info(f"实体类型空间创建完成: 共创建 {len(type_counts)} 个类型空间")
        logging.info(f"类型分布: {type_counts}")

    def _create_qa_optimized_spaces(self, semantic_graph: SemanticGraph, conversation_id: str):
        """Create qa optimized spaces."""
        logging.info("创建QA优化语义空间...")
        
        qa_spaces = {
            "when_optimized": {
                "target_types": ["DATE_TIME", "EVENT", "ACTIVITY"],
                "description": "时间问答优化空间：包含时间、事件、活动实体"
            },
            
            "where_optimized": {
                "target_types": ["LOCATION", "EVENT", "ACTIVITY"],
                "description": "地点问答优化空间：包含地点、事件、活动实体"
            },
            
            "who_optimized": {
                "target_types": ["PERSON", "RELATIONSHIP", "ORGANIZATION"],
                "description": "人物问答优化空间：包含人物、关系、组织实体"
            },
            
            "what_optimized": {
                "target_types": ["ACTIVITY", "OBJECT", "CONCEPT", "SKILL", "GOAL"],
                "description": "活动对象问答优化空间：包含活动、物品、概念、技能、目标实体"
            },
            
            "how_optimized": {
                "target_types": ["SKILL", "ACTIVITY", "CONCEPT"],
                "description": "方式方法问答优化空间：包含技能、活动、概念实体"
            },
            
            "why_optimized": {
                "target_types": ["EMOTION", "CONCEPT", "GOAL", "RELATIONSHIP"],
                "description": "原因情感问答优化空间：包含情感、概念、目标、关系实体"
            }
        }
        
        qa_space_stats = {}
        
        for space_name, space_config in qa_spaces.items():
            try:
                full_space_name = f"{conversation_id}_qa_{space_name}"
                target_types = space_config["target_types"]
                
                qa_space_units = []
                
                for unit_uid, unit in semantic_graph.semantic_map.memory_units.items():
                    if unit.raw_data.get("conversation_id") != conversation_id:
                        continue
                    
                    unit_entity_type = unit.raw_data.get("entity_type", "")
                    
                    if unit_entity_type in target_types:
                        qa_space_units.append(unit_uid)
                
                if qa_space_units:
                    semantic_graph.create_memory_space_in_map(full_space_name)
                    
                    for unit_uid in qa_space_units:
                        semantic_graph.add_unit_to_space_in_map(unit_uid, full_space_name)
                    
                    qa_space_stats[space_name] = {
                        "count": len(qa_space_units),
                        "types": target_types
                    }
                    self.stats["memory_spaces_created"] += 1
                    
                    logging.debug(f"QA优化空间 {full_space_name} 创建完成")
                    logging.debug(f"  - 描述: {space_config['description']}")
                    logging.debug(f"  - 包含单元: {len(qa_space_units)} 个")
                    logging.debug(f"  - 目标类型: {', '.join(target_types)}")
                else:
                    logging.debug(f"QA优化空间 {space_name} 没有匹配单元，跳过创建")
                    
            except Exception as e:
                error_msg = f"创建QA优化空间 {space_name} 失败: {e}"
                logging.error(error_msg)
                self.stats["errors"].append(error_msg)
        
        logging.info(f"QA优化空间创建完成: 共创建 {len(qa_space_stats)} 个QA优化空间")
        for space_name, stats in qa_space_stats.items():
            logging.info(f"  - {space_name}: {stats['count']} 个单元 ({', '.join(stats['types'])})")

    def _create_temporal_spaces(self, semantic_graph: SemanticGraph, conversation_id: str):
        """Create temporal spaces."""
        logging.info("创建时间线语义空间...")
        
        try:
            temporal_entities = []
            
            for unit_uid, unit in semantic_graph.semantic_map.memory_units.items():
                if unit.raw_data.get("conversation_id") != conversation_id:
                    continue
                
                entity_type = unit.raw_data.get("entity_type", "")
                temporal_info = unit.raw_data.get("temporal_info")
                
                if entity_type == "DATE_TIME" or temporal_info:
                    temporal_entities.append({
                        "uid": unit_uid,
                        "entity_type": entity_type,
                        "temporal_info": temporal_info,
                        "session_id": unit.raw_data.get("session_id", ""),
                        "node_type": unit.raw_data.get("node_type", "")
                    })
            
            if temporal_entities:
                timeline_space = f"{conversation_id}_timeline_all"
                semantic_graph.create_memory_space_in_map(timeline_space)
                
                for entity in temporal_entities:
                    semantic_graph.add_unit_to_space_in_map(entity["uid"], timeline_space)
                
                self.stats["memory_spaces_created"] += 1
                
                session_timelines = {}
                for entity in temporal_entities:
                    session_id = entity["session_id"]
                    if session_id and session_id not in session_timelines:
                        session_timelines[session_id] = []
                    if session_id:
                        session_timelines[session_id].append(entity["uid"])
                
                for session_id, entity_uids in session_timelines.items():
                    if len(entity_uids) > 1:
                        session_timeline_space = f"{conversation_id}_timeline_{session_id}"
                        semantic_graph.create_memory_space_in_map(session_timeline_space)
                        
                        for entity_uid in entity_uids:
                            semantic_graph.add_unit_to_space_in_map(entity_uid, session_timeline_space)
                        
                        self.stats["memory_spaces_created"] += 1
                
                logging.debug(f"时间线空间创建完成: 1个总体空间 + {len(session_timelines)} 个会话时间线空间")
            
        except Exception as e:
            error_msg = f"创建时间线空间失败: {e}"
            logging.error(error_msg)
            self.stats["errors"].append(error_msg)

    def _create_relationship_spaces(self, semantic_graph: SemanticGraph, conversation_id: str):
        """Create relationship spaces."""
        logging.info("创建关系类型语义空间...")
        
        try:
            relation_groups = {
                "temporal_relations": [
                    "happens_before", "happens_after", "during", "lasts_for"
                ],
                "spatial_relations": [
                    "located_at", "travels_to", "comes_from", "near"
                ],
                "participation_relations": [
                    "participates_in", "organizes", "attends", "performs"
                ],
                "social_relations": [
                    "supports", "cares_for", "friends_with", "influences", "helps"
                ],
                "identity_relations": [
                    "is_a", "works_as", "identifies_as", "member_of"
                ],
                "possession_relations": [
                    "owns", "has", "belongs_to", "associated_with"
                ]
            }
            
            all_relations = []
            
            if hasattr(semantic_graph, 'rx_graph'):
                rx_graph = semantic_graph.rx_graph
                for edge_idx in rx_graph.edge_indices():
                    src_idx, tgt_idx = rx_graph.get_edge_endpoints_by_index(edge_idx)
                    edge_data = rx_graph.get_edge_data_by_index(edge_idx)
                    if edge_data and edge_data.get('conversation_id') == conversation_id:
                        
                        source = semantic_graph._index_to_uid.get(src_idx, '')
                        target = semantic_graph._index_to_uid.get(tgt_idx, '')
                        if source and target:
                            relation_type = edge_data.get('relationship_name', '').lower()
                            all_relations.append({
                                'source': source,
                                'target': target,
                                'type': relation_type,
                                'data': edge_data
                            })
            
            for group_name, relation_types in relation_groups.items():
                group_space_name = f"{conversation_id}_relations_{group_name}"
                
                try:
                    semantic_graph.create_memory_space_in_map(group_space_name)
                    group_units = set()
                    
                    for relation in all_relations:
                        if any(rel_type in relation['type'] for rel_type in relation_types):
                            group_units.add(relation['source'])
                            group_units.add(relation['target'])
                    
                    for unit_uid in group_units:
                        if unit_uid in semantic_graph.semantic_map.memory_units:
                            semantic_graph.add_unit_to_space_in_map(unit_uid, group_space_name)
                    
                    if group_units:
                        logging.debug(f"关系组空间 {group_space_name} 创建完成，包含 {len(group_units)} 个实体")
                        self.stats["memory_spaces_created"] += 1
                    
                except Exception as e:
                    error_msg = f"创建关系组空间 {group_name} 失败: {e}"
                    logging.error(error_msg)
                    self.stats["errors"].append(error_msg)
            
            high_confidence_space = f"{conversation_id}_relations_high_confidence"
            semantic_graph.create_memory_space_in_map(high_confidence_space)
            
            high_conf_units = set()
            for relation in all_relations:
                confidence = relation['data'].get('confidence', 0.0)
                if confidence > 0.8:
                    high_conf_units.add(relation['source'])
                    high_conf_units.add(relation['target'])
            
            for unit_uid in high_conf_units:
                if unit_uid in semantic_graph.semantic_map.memory_units:
                    semantic_graph.add_unit_to_space_in_map(unit_uid, high_confidence_space)
            
            if high_conf_units:
                self.stats["memory_spaces_created"] += 1
                logging.debug(f"高置信度关系空间创建完成，包含 {len(high_conf_units)} 个实体")
            
        except Exception as e:
            error_msg = f"创建关系空间失败: {e}"
            logging.error(error_msg)
            self.stats["errors"].append(error_msg)

    def _load_saved_retrieval_state(self, output_path: Path) -> Dict[str, Any]:
        """Load saved retrieval state."""
        state = {
            "enabled": self.freeze_retrievers,
            "frozen_matrices_saved": {},
            "indices_saved": {}
        }
        graph_state_file = output_path / "graph_state.json"
        try:
            with open(graph_state_file, 'r', encoding='utf-8') as f:
                graph_state = json.load(f)
            retrieval = graph_state.get("retrieval", {})
            state["frozen_matrices_saved"] = retrieval.get("frozen_matrices_saved", {})
            state["indices_saved"] = retrieval.get("indices_saved", {})
        except Exception as e:
            state["error"] = str(e)
            logging.warning(f"读取检索索引保存状态失败: {e}")
        return state

    def _save_processing_report(self, output_path: Path, original_data: Dict[str, Any],
                                splade_stats: Optional[Dict[str, Any]] = None):
        """Save processing report."""
        report = {
            "processing_time": datetime.now().isoformat(),
            "source_file": original_data.get("conversation_id", "unknown"),
            "insertion_stats": self.stats,
            "original_metadata": {
                "entities_count": len(original_data.get("entities", [])),
                "relations_count": len(original_data.get("relations", [])),
                "session_metadata": original_data.get("session_metadata", {})
            },
            
            "splade_config": {
                "enabled": self.build_splade,
                "model": self.splade_model if self.build_splade else None,
                "batch_size": self.splade_batch_size if self.build_splade else None
            },
            "splade_stats": splade_stats if splade_stats else {},
            "static_retriever_indexes": self._load_saved_retrieval_state(output_path),
            "storage_model": {
                "type": "中心化实体与分布式证据",
                "node_types": {
                    "entity_hub": "中心化实体节点（聚合同一实体的所有信息）",
                    "evidence_mention": "分布式证据节点（携带具体上下文信息）"
                },
                "relationship_types": {
                    "MENTION_OF": "证据提及关系（evidence_mention -> entity_hub）",
                    "ENTITY_RELATION": "实体间语义关系（entity_hub <-> entity_hub）"
                },
                "memory_spaces": {
                    "session_spaces": "Individual conversation sessions",
                    "entity_type_spaces": "Individual spaces for each entity type (PERSON, LOCATION, etc.)",
                    "qa_optimized_spaces": "Cross-type spaces optimized for specific question types",
                    "temporal_spaces": "Timeline-based entity organization",
                    "relationship_spaces": "Grouped by relationship types"
                },
                "supported_entity_types": [
                    "PERSON", "ORGANIZATION", "EVENT", "ACTIVITY", 
                    "CONCEPT", "EMOTION", "LOCATION", "DATE_TIME", 
                    "NUMERICAL_VALUE", "OBJECT", "SKILL", "RELATIONSHIP", "GOAL"
                ],
                "qa_optimization": {
                    "when_questions": "Cross-type space with DATE_TIME, EVENT, ACTIVITY entities",
                    "where_questions": "Cross-type space with LOCATION, EVENT, ACTIVITY entities", 
                    "who_questions": "Cross-type space with PERSON, RELATIONSHIP, ORGANIZATION entities",
                    "what_questions": "Cross-type space with ACTIVITY, OBJECT, CONCEPT, SKILL, GOAL entities",
                    "how_questions": "Cross-type space with SKILL, ACTIVITY, CONCEPT entities",
                    "why_questions": "Cross-type space with EMOTION, CONCEPT, GOAL, RELATIONSHIP entities"
                }
            },
            "retrieval_optimization": {
                "primary_search_targets": "evidence_mention nodes",
                "reasoning_targets": "entity_hub nodes", 
                "graph_traversal": "MENTION_OF -> entity_hub -> entity_relations",
                "space_based_filtering": "Use type-specific or QA-optimized spaces for targeted retrieval",
                "multi_hop_support": "Cross-space traversal for complex queries",
                "sparse_retrieval": "SPLADE vectors enabled" if self.build_splade else "SPLADE vectors disabled"
            }
        }
        
        report_file = output_path / "insertion_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logging.info(f"处理报告已保存: {report_file}")
    
    def process_file(self, input_file_path: str, output_dir: str, cleanup: bool = True) -> Optional[SemanticGraph]:
        """Process file."""
        start_time = datetime.now()
        semantic_graph = None
        
        try:
            
            logging.info(f"处理文件: {input_file_path}")
            data = self.load_entity_relation_data(input_file_path)
            
            conversation_id = data["conversation_id"]
            entities = data["entities"]
            relations = data["relations"]
            session_metadata = data.get("session_metadata", {})
            
            semantic_graph = self.create_semantic_graph()
            
            entity_hub_mapping = self.insert_entities_to_graph(
                semantic_graph, entities, conversation_id
            )
            
            self.insert_relations_to_graph(
                semantic_graph, relations, entity_hub_mapping, conversation_id
            )
            
            self.create_semantic_spaces(semantic_graph, session_metadata, conversation_id)
            
            
            splade_stats = None
            if self.build_splade:
                splade_stats = self._build_splade_embeddings(semantic_graph, conversation_id)
                if splade_stats:
                    self.stats["splade_stats"] = splade_stats
            
            
            logging.info("保存前重建完整索引...")
            semantic_graph.rebuild_all_indexes()

            
            output_path = Path(output_dir) / conversation_id
            output_path.mkdir(parents=True, exist_ok=True)
            
            semantic_graph.save_graph(str(output_path), freeze_retrievers=self.freeze_retrievers)
            
            end_time = datetime.now()
            self.stats["processing_time"] = (end_time - start_time).total_seconds()
            
            self._save_processing_report(output_path, data, splade_stats)
            
            logging.info(f"处理完成: {conversation_id}")
            logging.info(f"输出路径: {output_path}")
            
            if cleanup:
                logging.info(f" 清理 {conversation_id} 的资源...")
                self._cleanup_resources(semantic_graph)
                semantic_graph = None
            
            return semantic_graph
            
        except Exception as e:
            logging.error(f"处理文件失败: {e}")
            if semantic_graph is not None and cleanup:
                self._cleanup_resources(semantic_graph)
            raise

    def _build_multimodal_text_for_graph(self, raw_data: Dict[str, Any]) -> str:
        """Build multimodal text for graph."""
        text = raw_data.get("text_content", "").strip()
        
        has_multimodal = bool(
            raw_data.get("img_url") or 
            raw_data.get("blip_caption") or 
            raw_data.get("image_keywords")
        )
        
        if not has_multimodal:
            return text
        
        text_parts = []
        
        if text:
            text_parts.append(f"[content] {text}")
        
        blip_caption = raw_data.get("blip_caption", "").strip()
        if blip_caption:
            text_parts.append(f"[image_description] {blip_caption}")
        
        image_keywords = raw_data.get("image_keywords", "").strip()
        if image_keywords:
            text_parts.append(f"[image_keywords] {image_keywords}")
        
        return '\n'.join(text_parts)
    
    def _batch_process_sequential(self, json_files: List[Path], 
                             output_path: Path) -> Tuple[int, List[str], List[float]]:
        """Run batch process sequential."""
        from tqdm import tqdm
        
        success_count = 0
        failed_files = []
        processing_times = []
        
        for idx, json_file in enumerate(tqdm(json_files, desc="处理进度"), 1):
            logging.info(f"\n{'='*80}")
            logging.info(f"处理进度: {idx}/{len(json_files)} - {json_file.name}")
            logging.info(f"{'='*80}")
            
            file_start = time.time()
            
            try:
                self.process_file(
                    str(json_file), 
                    str(output_path),
                    cleanup=True
                )
                
                file_time = time.time() - file_start
                processing_times.append(file_time)
                success_count += 1
                
                logging.info(f" {json_file.name} 处理成功 (耗时: {file_time:.2f}s)")
                
                self._log_memory_usage()
                
            except Exception as e:
                file_time = time.time() - file_start
                processing_times.append(file_time)
                
                logging.error(f" {json_file.name} 处理失败: {e}")
                logging.debug(traceback.format_exc())
                failed_files.append(json_file.name)
                
                try:
                    self._cleanup_resources()
                except Exception as cleanup_error:
                    logging.warning(f"清理失败资源时出错: {cleanup_error}")
        
        return success_count, failed_files, processing_times

    def _batch_process_parallel(self, json_files: List[Path], 
                            output_path: Path,
                            max_workers: int) -> Tuple[int, List[str], List[float]]:
        """Run batch process parallel."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm
        
        success_count = 0
        failed_files = []
        processing_times = []
        
        lock = threading.Lock()
        
        def process_single_file(json_file: Path) -> Tuple[bool, str, float]:
            """Process single file."""
            file_start = time.time()
            
            try:
                thread_inserter = LoCoMoSemanticGraphInserter(
                    text_embedding_model=self.text_embedding_model,
                    llm_client=self.llm_client,
                    build_splade=self.build_splade,
                    splade_model=self.splade_model,
                    splade_batch_size=self.splade_batch_size,
                    freeze_retrievers=self.freeze_retrievers
                )
                
                thread_inserter.process_file(
                    str(json_file),
                    str(output_path),
                    cleanup=True
                )
                
                file_time = time.time() - file_start
                
                logging.info(f" {json_file.name} 处理成功 (耗时: {file_time:.2f}s)")
                
                return True, json_file.name, file_time
                
            except Exception as e:
                file_time = time.time() - file_start
                
                logging.error(f" {json_file.name} 处理失败: {e}")
                logging.debug(traceback.format_exc())
                
                return False, json_file.name, file_time
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(process_single_file, json_file): json_file 
                for json_file in json_files
            }
            
            with tqdm(total=len(json_files), desc="并行处理进度") as pbar:
                for future in as_completed(future_to_file):
                    json_file = future_to_file[future]
                    
                    try:
                        success, filename, file_time = future.result()
                        
                        with lock:
                            processing_times.append(file_time)
                            if success:
                                success_count += 1
                            else:
                                failed_files.append(filename)
                        
                    except Exception as e:
                        logging.error(f"线程执行异常 {json_file.name}: {e}")
                        with lock:
                            failed_files.append(json_file.name)
                    
                    pbar.update(1)
        
        return success_count, failed_files, processing_times

    def _log_memory_usage(self):
        """Log memory usage."""
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                logging.debug(f" GPU内存: {allocated:.2f}GB / {reserved:.2f}GB")
        except ImportError:
            pass
    
    def batch_process(self, input_dir: str, output_dir: str, 
                 sample_ids: Optional[List[str]] = None,
                 max_workers: int = 1,
                 enable_parallel: bool = False):
        """Run batch process."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        json_files = self._find_entity_relation_files(input_path, sample_ids)
        
        if not json_files:
            logging.warning(f"未找到任何实体关系文件")
            return
        
        logging.info(f"共发现 {len(json_files)} 个文件待处理")
        logging.info(f"处理模式: {'多线程并行' if enable_parallel else '单线程顺序'}")
        if enable_parallel:
            logging.info(f"最大工作线程数: {max_workers}")
        
        success_count = 0
        failed_files = []
        processing_times = []
        
        start_time = time.time()
        
        if enable_parallel and max_workers > 1:
            success_count, failed_files, processing_times = self._batch_process_parallel(
                json_files, output_path, max_workers
            )
        else:
            success_count, failed_files, processing_times = self._batch_process_sequential(
                json_files, output_path
            )
        
        total_time = time.time() - start_time
        
        batch_report = {
            "batch_processing": {
                "timestamp": datetime.now().isoformat(),
                "input_directory": str(input_path),
                "output_directory": str(output_path),
                "processing_mode": "parallel" if enable_parallel else "sequential",
                "max_workers": max_workers if enable_parallel else 1,
                "total_files": len(json_files),
                "successful_files": success_count,
                "failed_files": len(failed_files),
                "failed_file_list": failed_files,
                "success_rate": success_count / len(json_files) if json_files else 0,
                "total_time_seconds": total_time,
                "average_time_per_file": total_time / len(json_files) if json_files else 0,
                "processing_times": {
                    "min": min(processing_times) if processing_times else 0,
                    "max": max(processing_times) if processing_times else 0,
                    "mean": sum(processing_times) / len(processing_times) if processing_times else 0
                }
            },
            "cumulative_stats": self.stats
        }
        
        batch_report_file = output_path / "batch_processing_report.json"
        with open(batch_report_file, 'w', encoding='utf-8') as f:
            json.dump(batch_report, f, ensure_ascii=False, indent=2)
        
        logging.info(f"\n{'='*80}")
        logging.info(f" 批量处理完成统计")
        logging.info(f"{'='*80}")
        logging.info(f"总文件数: {len(json_files)}")
        logging.info(f"成功: {success_count}")
        logging.info(f"失败: {len(failed_files)}")
        logging.info(f"成功率: {success_count / len(json_files) * 100:.2f}%")
        logging.info(f"总耗时: {total_time:.2f}秒")
        logging.info(f"平均每个文件: {total_time / len(json_files):.2f}秒")
        if failed_files:
            logging.info(f"失败文件: {failed_files}")
        logging.info(f"详细报告: {batch_report_file}")
        logging.info(f"{'='*80}")


def main():
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(description="LoCoMo实体关系数据到SemanticGraph插入器")
    
    parser.add_argument("--input-file", 
                       help="单个实体关系JSON文件路径")
    parser.add_argument("--input-dir", 
                       default=str(paths.LOCOMO_ENTITY_RELATION_STEP2_DIR),
                       help="输入目录路径（包含多个sample子目录）")
    parser.add_argument("--output-dir", 
                       default=str(paths.LOCOMO_ENTITY_RELATION_STEP3_DIR),
                       help="输出目录路径")
    
    parser.add_argument("--text-embedding-model", 
                       default="Qwen/Qwen3-Embedding-0.6B",
                       help="文本嵌入模型名称")
    
    
    parser.add_argument("--build-splade", action="store_true", default=True,
                       help="启用 SPLADE 稀疏向量构建（默认启用）")
    parser.add_argument("--no-splade", action="store_true",
                       help="禁用 SPLADE 稀疏向量")
    parser.add_argument("--splade-model", type=str,
                       default="naver/splade-v3",
                       help="SPLADE 模型名称（默认: naver/splade-v3）")
    parser.add_argument("--splade-batch-size", type=int, default=32,
                       help="SPLADE 批处理大小（默认: 32）")
    parser.add_argument("--no-freeze-retrievers", action="store_true",
                       help="禁用保存阶段的 BM25/SPLADE 静态加速索引冻结")
    
    parser.add_argument("--batch-mode", action="store_true", default=True,
                       help="批量处理模式（默认启用，处理input-dir下的所有文件）")
    parser.add_argument("--sample-ids", nargs='+', 
                       help="指定要处理的sample ID列表（例如：--sample-ids conv-26 conv-30）")
    parser.add_argument("--enable-parallel", action="store_true",
                       help="启用多线程并行处理")
    parser.add_argument("--max-workers", type=int, default=10,
                       help="最大工作线程数（默认10，仅在--enable-parallel时生效）")
    
    # Avoid mutating LogRecord fields before other handlers process the record.
    parser.add_argument("--log-level", 
                       default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="日志级别")
    
    args = parser.parse_args()
    
    
    build_splade = not args.no_splade
    
    # Avoid mutating LogRecord fields before other handlers process the record.
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 80)
    print(" LoCoMo实体关系到SemanticGraph插入器")
    print("=" * 80)
    print(f" 存储模型: 中心化实体与分布式证据")
    print(f" 文本嵌入模型: {args.text_embedding_model}")
    print(f" 输出目录: {args.output_dir}")
    
    
    if build_splade:
        print(f" SPLADE:  启用（默认）")
        print(f"   模型:    {args.splade_model}")
        print(f"   批大小:  {args.splade_batch_size}")
    else:
        print(f" SPLADE:  已禁用（使用 --no-splade）")
    print(f" 静态检索索引冻结: {'启用（默认）' if not args.no_freeze_retrievers else '禁用'}")
    
    if args.input_file:
        print(f" 处理模式: 单文件处理")
        print(f" 输入文件: {args.input_file}")
    else:
        print(f" 处理模式: 批量处理（默认）")
        print(f" 输入目录: {args.input_dir}")
        if args.sample_ids:
            print(f" 指定样本: {args.sample_ids}")
    print("=" * 80)
    
    
    inserter = LoCoMoSemanticGraphInserter(
        text_embedding_model=args.text_embedding_model,
        build_splade=build_splade,
        splade_model=args.splade_model,
        splade_batch_size=args.splade_batch_size,
        freeze_retrievers=not args.no_freeze_retrievers
    )
    
    try:
        if args.input_file:
            print(f" 处理单文件: {args.input_file}")
            result = inserter.process_file(
                input_file_path=args.input_file,
                output_dir=args.output_dir
            )
            if result:
                print(f" 处理完成")
                return 0
            else:
                print(f" 处理失败")
                return 1
        else:
            inserter.batch_process(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                sample_ids=args.sample_ids,
                max_workers=args.max_workers,
                enable_parallel=args.enable_parallel
            )
        
        return 0
        
    except Exception as e:
        print(f"\n 处理失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())