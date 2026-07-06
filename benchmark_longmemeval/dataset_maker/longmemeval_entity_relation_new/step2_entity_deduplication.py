#!/usr/bin/env python3
"""Utilities for step2 entity deduplication."""
import json
import logging
import argparse
import sys
import traceback
import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from json_repair import repair_json


from sentence_transformers import SentenceTransformer
from mandol.llm.llm_client import LLMClient
from mandol.cluster.dbscan_method import find_clusters_with_dbscan, optimize_dbscan_parameters
from mandol.core import paths

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class EntityMention:
    session_id: str
    session_date: str
    context: str
    temporal_info: Optional[str] = None
    temporal_reference: Optional[str] = None
    spatial_info: Optional[str] = None
    numerical_value: Optional[str] = None
    aliases: List[str] = None
    confidence: float = 0.95
    
    def __post_init__(self):
        if self.aliases is None:
            self.aliases = []


@dataclass
class DeduplicatedEntity:
    entity_id: str
    name: str
    entity_type: str
    confidence: float
    mentions: List[EntityMention]
    total_mentions: int
    session_ids: List[str]
    temporal_info: Optional[str] = None
    spatial_info: Optional[str] = None
    numerical_value: Optional[str] = None
    
    def __post_init__(self):
        if not self.session_ids:
            self.session_ids = [m.session_id for m in self.mentions]
        if not self.total_mentions:
            self.total_mentions = len(self.mentions)


class LongMemEvalEntityDeduplicator:
    
    def __init__(self,
                 llm_client: Optional[LLMClient] = None,
                 llm_model: str = "deepseek-reasoner",
                 llm_api_key: Optional[str] = None,
                 llm_base_url: Optional[str] = None,
                #  embedding_model: str = "Qwen/Qwen3-Embedding-4B",
                #  embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 optimize_per_qa: bool = True,
                 use_llm_dedup: bool = True,
                 llm_cluster_threshold: int = 2,
                 large_cluster_threshold: int = 12,
                 parallel_workers: int = 10):
        self.logger = logging.getLogger(__name__)
        self.parallel_workers = parallel_workers
        self.optimize_per_qa = optimize_per_qa
        self.use_llm_dedup = use_llm_dedup
        self.llm_cluster_threshold = llm_cluster_threshold
        self.large_cluster_threshold = large_cluster_threshold
        
        self.qa_optimized_params = {}
        
        if llm_client is not None:
            self.llm_client = llm_client
        else:
            self.llm_client = LLMClient(
                model_name=llm_model,
                api_key=llm_api_key,
                base_url=llm_base_url
            )
        
        self.logger.info(f" LLM客户端初始化: {llm_model}")
        
        self.logger.info(f" 加载嵌入模型: {embedding_model}")
        self.encoder = SentenceTransformer(embedding_model)
        
        self.stats_lock = Lock()
        self.stats = {
            "qa_processed": 0,
            "requests_loaded": 0,
            "raw_entities_extracted": 0,
            "entities_after_dedup": 0,
            "clusters_processed": 0,
            "llm_merge_calls": 0,
            "param_optimizations": 0,
            "param_optimization_time": 0.0,
            "processing_time": 0.0,
        }
        
        self.dedup_prompt_template = self._prepare_llm_dedup_prompt()
        
        self.logger.info(f" LongMemEval实体去重器初始化完成")
        if self.optimize_per_qa:
            self.logger.info(f" 已启用逐QA参数优化")
        if self.use_llm_dedup:
            self.logger.info(f" 已启用LLM精细去重 (聚类大小≥{llm_cluster_threshold})")
    
    def _prepare_llm_dedup_prompt(self) -> str:
        """Run prepare LLM dedup prompt."""
        return """You are an expert in entity standardization and deduplication. The following are {cluster_size} similar entities grouped by semantic clustering. Please merge duplicates pointing to the same real object while preserving all session-specific information.

        Cluster {cluster_id} entity list:
        {entity_candidates}

        Please return deduplication results in the new mentions-based JSON format:
        {{
            "merged_entities": [
                {{
                    "entity_id": "merged_{cluster_id}_1",
                    "name": "Standard canonical entity name",
                    "entity_type": "Unified entity type",
                    "confidence": 0.95,
                    "mentions": [
                        {{
                            "session_id": "session_1",
                            "session_date": "2023/05/30 (Tue) 14:00",
                            "context": "Session-specific description from original entity",
                            "temporal_info": "time info from this session",
                            "temporal_reference": "relative time reference",
                            "spatial_info": "location info from this session", 
                            "numerical_value": "numerical value with unit",
                            "aliases": ["session-specific aliases"],
                            "confidence": 0.90
                        }},
                        {{
                            "session_id": "session_2",
                            "session_date": "2023/06/01 (Thu) 10:00", 
                            "context": "Another session's description",
                            "temporal_info": "different time context",
                            "spatial_info": "different location context",
                            "aliases": ["other aliases"],
                            "confidence": 0.88
                        }}
                    ],
                    "extraction_metadata": {{
                        "merge_method": "llm_cluster_dedup",
                        "source_count": 2,
                        "merge_reasoning": "Detailed explanation of why these entities were merged"
                    }}
                }}
            ]
        }}

        **Critical Deduplication Rules:**
        1. **Preserve ALL session contexts**: Each mention must retain the original session's specific information
        2. **Maintain temporal accuracy**: Keep different temporal_info for each session if they differ
        3. **Preserve spatial contexts**: Maintain location information specific to each session
        4. **Consolidate aliases wisely**: Merge aliases but keep session-specific ones where relevant
        5. **Calculate confidence**: Use weighted average based on original confidence scores
        6. **Standard naming**: Choose the most complete and clear canonical name
        7. **Avoid over-merging**: Only merge entities that truly refer to the same real-world object

        **Entity Matching Guidelines:**
        - These entities have been clustered by semantic similarity, focus on identifying completely identical objects
        - Unify pronoun references (I→specific person name, here→specific location) but preserve context
        - Merge obvious synonyms and variant expressions
        - Preserve semantic independence - if entities are truly different, keep them separate
        - Consider temporal evolution: same entity might have different attributes over time

        **Output Requirements:**
        - Every source entity must be represented in at least one mention
        - Session-specific information (context, temporal_info, spatial_info) must be preserved
        - Canonical name should be the most informative and standardized form
        - Confidence should reflect the quality of the merge decision
        """
            
    def _update_stats(self, **kwargs):
        """Run update stats."""
        with self.stats_lock:
            for key, value in kwargs.items():
                if key in self.stats:
                    try:
                        if isinstance(self.stats[key], (int, float)):
                            if isinstance(value, str):
                                try:
                                    value = int(value) if value.isdigit() else float(value)
                                except (ValueError, AttributeError):
                                    self.logger.error(f" 统计更新失败: {key}={value} 无法转换为数字")
                                    continue
                            
                            if isinstance(value, (int, float)):
                                self.stats[key] += value
                            else:
                                self.logger.error(f" 统计更新类型错误: {key}={value} (类型: {type(value).__name__})")
                        else:
                            self.stats[key] = value
                    except Exception as e:
                        self.logger.error(f" 统计更新异常: {key}={value}, 错误: {e}")
                else:
                    if isinstance(value, str) and value.replace('.', '', 1).replace('-', '', 1).isdigit():
                        try:
                            value = int(value) if '.' not in value else float(value)
                        except ValueError:
                            pass
                    self.stats[key] = value
    
    def optimize_parameters_for_qa(self, 
                                   entities: List[Dict[str, Any]],
                                   qa_id: str) -> Dict[str, Dict[str, float]]:
        """Run optimize parameters for qa."""
        self.logger.info(f" 开始为 {qa_id} 优化参数 ({len(entities)} 个实体)")
        
        start_time = datetime.now()
        
        type_entities = defaultdict(list)
        for entity in entities:
            entity_type = entity.get('type', 'UNKNOWN')
            type_entities[entity_type].append(entity)
        
        optimized_params = {}
        
        for entity_type, type_entities_list in type_entities.items():
            if len(type_entities_list) < 5:
                optimized_params[entity_type] = {'eps': 0.25, 'min_samples': 2}
                self.logger.info(f"  类型 {entity_type}: 样本不足({len(type_entities_list)}), 使用默认参数")
                continue
            
            self.logger.info(f"   优化类型 {entity_type}: {len(type_entities_list)} 个实体")
            
            
            texts = [self._prepare_entity_for_clustering(e) for e in type_entities_list]
            embeddings = self.encoder.encode(texts, show_progress_bar=False)
            
            class EntityWrapper:
                def __init__(self, idx, embedding):
                    self.uid = str(idx)
                    self.embedding = embedding
            
            wrapped = [EntityWrapper(i, emb) for i, emb in enumerate(embeddings)]
            
            try:
                result = optimize_dbscan_parameters(
                    wrapped,
                    eps_range=(0.15, 0.6),
                    min_samples_range=(2, min(8, len(type_entities_list) // 3)),
                    metric='cosine',
                    n_trials=16
                )
                
                best_params = result['best_params']
                best_score = result['best_score']
                
                optimized_params[entity_type] = best_params
                
                self.logger.info(f"     {entity_type}: eps={best_params['eps']:.3f}, "
                               f"min_samples={best_params['min_samples']}, score={best_score:.3f}")
                
            except Exception as e:
                self.logger.warning(f"     {entity_type} 优化失败: {e}, 使用默认参数")
                optimized_params[entity_type] = {'eps': 0.3, 'min_samples': 3}
        
        if optimized_params:
            avg_eps = np.mean([p['eps'] for p in optimized_params.values()])
            avg_min_samples = int(np.median([p['min_samples'] for p in optimized_params.values()]))
            optimized_params['_default'] = {'eps': float(avg_eps), 'min_samples': avg_min_samples}
        else:
            optimized_params['_default'] = {'eps': 0.3, 'min_samples': 3}
        
        optimization_time = (datetime.now() - start_time).total_seconds()
        self._update_stats(param_optimizations=1, param_optimization_time=optimization_time)
        
        self.logger.info(f"   {qa_id} 参数优化完成，耗时 {optimization_time:.2f}秒")
        
        self.qa_optimized_params[qa_id] = optimized_params
        
        return optimized_params
    
    def _get_params_for_type(self, entity_type: str, qa_id: str) -> Tuple[float, int]:
        """Get params for type."""
        if qa_id in self.qa_optimized_params:
            params_dict = self.qa_optimized_params[qa_id]
            
            if entity_type in params_dict:
                params = params_dict[entity_type]
                return params['eps'], params['min_samples']
            
            if '_default' in params_dict:
                params = params_dict['_default']
                return params['eps'], params['min_samples']
        
        return 0.3, 3
    
    # def load_batch_results(self, result_file: str) -> Dict[str, List[Dict]]:
    
    
        
    #     qa_entities = defaultdict(list)
    #     request_count = 0
    #     failed_parse_count = 0
    #     valid_entity_count = 0
        
    #     try:
    #         with open(result_file, 'r', encoding='utf-8') as f:
    #             for line_num, line in enumerate(f, 1):
    #                 if not line.strip():
    #                     continue
                    
    #                 try:
    #                     request_count += 1
                        
    #                     if len(line) > 1000000:  # 1MB
                        
    #                     result = json.loads(line)
                        
    #                     custom_id = result.get("custom_id", "")
    #                     if not custom_id.startswith("qa_"):
    #                         continue
                        
    #                     parts = custom_id.split("_")
    #                     if len(parts) < 2:
    #                         continue
                        
    #                     qa_index = parts[1]
    #                     qa_id = f"qa_{qa_index}"
                        
    #                     try:
    #                         response_body = result["response"]["body"]
    #                         message_content = response_body["choices"][0]["message"]["content"]
                            
    #                         if len(message_content) > 500000:  # 500KB
                            
    #                         entities_data = self._safe_parse_json(message_content, custom_id)
                            
    #                         if entities_data is None:
    #                             failed_parse_count += 1
    #                             continue
                            
    #                         if not isinstance(entities_data, dict):
    #                             failed_parse_count += 1
    #                             continue
                            
    #                         entities = entities_data.get("entities", [])
                            
    #                         if not isinstance(entities, list):
    #                             failed_parse_count += 1
    #                             continue
                            
    #                         valid_entities = []
    #                         for entity in entities:
    #                             if not isinstance(entity, dict):
    #                                 continue
                                
    #                             if self._validate_entity(entity):
    #                                 valid_entities.append(entity)
    #                             else:
                            
    #                         if valid_entities:
    #                             qa_entities[qa_id].extend(valid_entities)
    #                             valid_entity_count += len(valid_entities)
                        
    #                     except (KeyError, IndexError, TypeError) as e:
    #                         failed_parse_count += 1
    #                         continue
                            
    #                 except json.JSONDecodeError as e:
    #                     failed_parse_count += 1
    #                     continue
                        
    #                 except Exception as e:
    #                     failed_parse_count += 1
    #                     continue
            
    #         self._update_stats(
    #             requests_loaded=int(request_count),
    #             raw_entities_extracted=int(valid_entity_count)
    #         )
            
    #         if failed_parse_count > 0:
            
    
            
    #     except FileNotFoundError:
    #         raise
    #     except Exception as e:
    
    #         raise
        
    #     return dict(qa_entities)
    
    def load_batch_results(self, result_file: str) -> Dict[str, List[Dict]]:
        """Load batch results."""
        self.logger.info(f" 加载批量推理结果: {result_file}")
        
        qa_entities = defaultdict(list)
        request_count = 0
        failed_parse_count = 0
        valid_entity_count = 0
        
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    
                    try:
                        request_count += 1
                        
                        if len(line) > 1000000:  # 1MB
                            self.logger.warning(f" 行 {line_num} 过长 ({len(line)} bytes)，可能被截断")
                        
                        result = json.loads(line)
                        
                        custom_id = result.get("custom_id", "")
                        if not custom_id.startswith("qa_"):
                            continue
                        
                        parts = custom_id.split("_")
                        if len(parts) < 2:
                            self.logger.warning(f" 无效的 custom_id: {custom_id}")
                            continue
                        
                        qa_index = parts[1]
                        qa_id = f"qa_{qa_index}"
                        
                        try:
                            response_body = result["response"]["body"]
                            message_content = response_body["choices"][0]["message"]["content"]
                            
                            if len(message_content) > 500000:  # 500KB
                                self.logger.warning(f" {custom_id} 响应内容过长 ({len(message_content)} bytes)")
                            
                            entities_data = self._safe_parse_json(message_content, custom_id)
                            
                            if entities_data is None:
                                failed_parse_count += 1
                                continue
                            
                            
                            
                            
                            if isinstance(entities_data, list):
                                entities_data = {"entities": entities_data}
                            
                            if not isinstance(entities_data, dict):
                                self.logger.warning(f" {custom_id} entities_data 不是字典: {type(entities_data).__name__}")
                                failed_parse_count += 1
                                continue
                            
                            entities = entities_data.get("entities", [])
                            
                            if not isinstance(entities, list):
                                self.logger.warning(f" {custom_id} entities 不是列表: {type(entities).__name__}")
                                failed_parse_count += 1
                                continue
                            
                            valid_entities = []
                            for entity in entities:
                                if not isinstance(entity, dict):
                                    self.logger.debug(f"跳过非字典实体: {type(entity).__name__}")
                                    continue
                                
                                if self._validate_entity(entity):
                                    valid_entities.append(entity)
                                else:
                                    self.logger.debug(f"跳过无效实体: {entity.get('entity_id', 'unknown')}")
                            
                            if valid_entities:
                                qa_entities[qa_id].extend(valid_entities)
                                valid_entity_count += len(valid_entities)
                        
                        except (KeyError, IndexError, TypeError) as e:
                            self.logger.warning(f"解析响应结构失败 {custom_id} (行 {line_num}): {type(e).__name__}: {e}")
                            failed_parse_count += 1
                            continue
                            
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"解析请求失败 (行 {line_num}): {e}")
                        failed_parse_count += 1
                        continue
                        
                    except Exception as e:
                        self.logger.error(f"处理行 {line_num} 时异常: {type(e).__name__}: {e}")
                        failed_parse_count += 1
                        continue
            
            self._update_stats(
                requests_loaded=int(request_count),
                raw_entities_extracted=int(valid_entity_count)
            )
            
            if failed_parse_count > 0:
                self.logger.warning(f" 共 {failed_parse_count}/{request_count} 个请求解析失败 ({failed_parse_count/request_count*100:.1f}%)")
            
            self.logger.info(f" 加载完成: {request_count} 个请求, {len(qa_entities)} 个QA, "
                            f"{valid_entity_count} 个有效实体")
            
        except FileNotFoundError:
            self.logger.error(f" 文件不存在: {result_file}")
            raise
        except Exception as e:
            self.logger.error(f" 加载文件失败: {type(e).__name__}: {e}")
            raise
        
        return dict(qa_entities)
    
    def _safe_parse_json(self, content: str, custom_id: str) -> Optional[Dict]:
        """Run safe parse JSON."""
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        
        try:
            repaired = repair_json(content)
            return json.loads(repaired)
        except Exception as e:
            self.logger.debug(f"json_repair 修复失败 {custom_id}: {e}")
        
        try:
            start_idx = content.find('{')
            if start_idx == -1:
                raise ValueError("未找到 JSON 起始符号")
            
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
                raise ValueError("未找到匹配的 JSON 结束符号")
            
            json_str = content[start_idx:end_idx]
            
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                repaired = repair_json(json_str)
                return json.loads(repaired)
            
        except Exception as e:
            self.logger.debug(f"JSON 修复失败 {custom_id}: {e}")
            return None


    def _validate_entity(self, entity: Dict[str, Any]) -> bool:
        """Validate entity."""
        if not isinstance(entity, dict):
            return False

        required_fields = ['name', 'type']
        
        for field in required_fields:
            if not entity.get(field):
                return False
        
        name = entity.get('name', '')
        if len(name) > 500:
            return False
        
        entity_type = entity.get('type', '')
        if len(entity_type) > 100:
            return False
        
        return True

    def load_all_batch_results(self, results_dir: str, exclude_dirs: Optional[List[str]] = None) -> Dict[str, List[Dict]]:
        """Load all batch results."""
        if exclude_dirs is None:
            exclude_dirs = ['deprecated']
        
        results_path = Path(results_dir)
        if not results_path.exists():
            raise FileNotFoundError(f"结果目录不存在: {results_dir}")
        
        jsonl_files = []
        for file_path in results_path.glob("*.jsonl"):
            is_excluded = any(excl in file_path.parts for excl in exclude_dirs)
            if not is_excluded:
                jsonl_files.append(file_path)
        
        if not jsonl_files:
            raise FileNotFoundError(f"在 {results_dir} 中未找到 JSONL 文件")
        
        jsonl_files.sort()
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f" 自动加载批量推理结果")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"结果目录: {results_dir}")
        self.logger.info(f"找到文件数: {len(jsonl_files)}")
        for f in jsonl_files:
            self.logger.info(f"  - {f.name}")
        self.logger.info(f"排除目录: {exclude_dirs}")
        self.logger.info(f"{'='*80}\n")
        
        qa_entities = defaultdict(list)
        total_requests = 0
        total_entities = 0
        total_files = len(jsonl_files)
        failed_files = []
        
        for file_idx, jsonl_file in enumerate(jsonl_files, 1):
            self.logger.info(f" [{file_idx}/{total_files}] 处理文件: {jsonl_file.name}")
            
            try:
                
                stats_before = {
                    'requests': int(self.stats.get('requests_loaded', 0)),
                    'entities': int(self.stats.get('raw_entities_extracted', 0))
                }
                
                
                file_qa_entities = self.load_batch_results(str(jsonl_file))
                
                try:
                    stats_after_requests = int(self.stats.get('requests_loaded', 0))
                    stats_after_entities = int(self.stats.get('raw_entities_extracted', 0))
                    
                    file_requests = stats_after_requests - stats_before['requests']
                    file_entities = stats_after_entities - stats_before['entities']
                except (ValueError, TypeError) as e:
                    self.logger.error(f" 统计计算错误: {e}")
                    self.logger.error(f"   stats_loaded: {self.stats.get('requests_loaded')} (类型: {type(self.stats.get('requests_loaded')).__name__})")
                    self.logger.error(f"   stats_extracted: {self.stats.get('raw_entities_extracted')} (类型: {type(self.stats.get('raw_entities_extracted')).__name__})")
                    file_requests = len([e for entities in file_qa_entities.values() for e in entities])
                    file_entities = file_requests
                
                for qa_id, entities in file_qa_entities.items():
                    qa_entities[qa_id].extend(entities)
                
                total_requests += file_requests
                total_entities += file_entities
                
                self.logger.info(f"   {jsonl_file.name}: {len(file_qa_entities)} 个QA, "
                                f"{file_entities} 个实体\n")
            
            except Exception as e:
                self.logger.error(f" 文件 {jsonl_file.name} 处理失败: {e}")
                failed_files.append(jsonl_file.name)
                if self.logger.isEnabledFor(logging.DEBUG):
                    import traceback
                    self.logger.debug(traceback.format_exc())
                continue
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f" 所有文件加载完成")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"总文件数: {total_files}")
        self.logger.info(f"成功: {total_files - len(failed_files)}")
        self.logger.info(f"失败: {len(failed_files)}")
        if failed_files:
            self.logger.info(f"失败文件: {failed_files}")
        self.logger.info(f"总请求数: {total_requests}")
        self.logger.info(f"总实体数: {total_entities}")
        self.logger.info(f"QA数量: {len(qa_entities)}")
        if qa_entities:
            self.logger.info(f"平均每个QA: {total_entities / len(qa_entities):.1f} 个实体")
        self.logger.info(f"{'='*80}\n")
        
        return dict(qa_entities)
    
    def _prepare_entity_for_clustering(self, entity: Dict[str, Any]) -> str:
        """Run prepare entity for clustering."""
        parts = [
            f"Name: {entity.get('name', '')}",
            f"Type: {entity.get('type', '')}",
            f"Content: {entity.get('content', '')}",
        ]
        
        aliases = entity.get('aliases', [])
        if aliases:
            parts.append(f"Aliases: {', '.join(aliases)}")
        
        temporal = entity.get('temporal_info')
        if temporal:
            parts.append(f"Time: {temporal}")
        
        spatial = entity.get('spatial_info')
        if spatial:
            parts.append(f"Location: {spatial}")
        
        return " | ".join(parts)
    
    def _deduplicate_entities_dbscan(self, 
                                     entities: List[Dict[str, Any]],
                                     qa_id: str) -> List[DeduplicatedEntity]:
        """Deduplicate entities dbscan."""
        if not entities:
            return []
        
        self.logger.info(f" 开始去重 {qa_id}: {len(entities)} 个原始实体")
        
        if self.optimize_per_qa:
            self.optimize_parameters_for_qa(entities, qa_id)
        
        type_groups = defaultdict(list)
        for idx, entity in enumerate(entities):
            entity['_temp_idx'] = idx
            entity_type = entity.get('type', 'UNKNOWN')
            type_groups[entity_type].append(entity)
        
        deduplicated_entities = []
        
        for entity_type, type_entities in type_groups.items():
            if len(type_entities) == 1:
                deduplicated_entities.append(
                    self._convert_to_deduplicated(type_entities[0], f"{qa_id}_E{type_entities[0]['_temp_idx']}")
                )
                continue
            
            eps, min_samples = self._get_params_for_type(entity_type, qa_id)
            
            self.logger.info(f"  处理类型 {entity_type}: {len(type_entities)} 个实体 "
                           f"(eps={eps:.3f}, min_samples={min_samples})")
            
            
            texts = [self._prepare_entity_for_clustering(e) for e in type_entities]
            embeddings = self.encoder.encode(texts, show_progress_bar=False)
            
            class EntityWrapper:
                def __init__(self, entity, embedding, temp_idx):
                    self.uid = str(temp_idx)
                    self.embedding = embedding
                    self.entity = entity
            
            wrapped_entities = [
                EntityWrapper(e, emb, e['_temp_idx']) 
                for e, emb in zip(type_entities, embeddings)
            ]
            
            try:
                clusters = find_clusters_with_dbscan(
                    wrapped_entities,
                    eps=eps,
                    min_samples=min_samples,
                    metric='cosine'
                )
                
                self.logger.info(f"    发现 {len(clusters)} 个聚类 (包含噪声)")
                
            except Exception as e:
                self.logger.error(f"    DBSCAN失败: {e}, 保留原始实体")
                for entity in type_entities:
                    deduplicated_entities.append(
                        self._convert_to_deduplicated(entity, f"{qa_id}_E{entity['_temp_idx']}")
                    )
                continue
            
            self._update_stats(clusters_processed=len(clusters))
            
            for cluster_id, uids in clusters.items():
                cluster_entities = [w.entity for w in wrapped_entities if w.uid in uids]
                
                if cluster_id == -1:
                    for entity in cluster_entities:
                        deduplicated_entities.append(
                            self._convert_to_deduplicated(entity, f"{qa_id}_E{entity['_temp_idx']}")
                        )
                else:
                    if len(cluster_entities) >= self.llm_cluster_threshold and self.use_llm_dedup:
                        if len(cluster_entities) >= self.large_cluster_threshold:
                            self.logger.info(f"     聚类 {cluster_id} 包含 {len(cluster_entities)} 个实体, 分批LLM去重")
                            try:
                                llm_merged = self._process_large_cluster(cluster_entities, cluster_id, qa_id)
                                deduplicated_entities.extend(llm_merged)
                            except Exception as e:
                                self.logger.error(f"    大聚类LLM去重失败: {e}, 回退到规则合并")
                                merged = self._merge_entity_cluster(
                                    cluster_entities, 
                                    f"{qa_id}_C{cluster_id}",
                                    qa_id
                                )
                                deduplicated_entities.append(merged)
                        else:
                            self.logger.info(f"     聚类 {cluster_id} 包含 {len(cluster_entities)} 个实体, 使用LLM去重")
                            try:
                                llm_merged = self._llm_deduplicate_cluster(cluster_entities, cluster_id, qa_id)
                                deduplicated_entities.extend(llm_merged)
                            except Exception as e:
                                self.logger.error(f"    LLM去重失败: {e}, 回退到规则合并")
                                merged = self._merge_entity_cluster(
                                    cluster_entities, 
                                    f"{qa_id}_C{cluster_id}",
                                    qa_id
                                )
                                deduplicated_entities.append(merged)
                    else:
                        merged = self._merge_entity_cluster(
                            cluster_entities, 
                            f"{qa_id}_C{cluster_id}",
                            qa_id
                        )
                        deduplicated_entities.append(merged)
        
        self.logger.info(f" 去重完成 {qa_id}: {len(entities)} -> {len(deduplicated_entities)}")
        return deduplicated_entities
    
    def _llm_deduplicate_cluster(self, 
                                 cluster_entities: List[Dict[str, Any]], 
                                 cluster_id: int,
                                 qa_id: str) -> List[DeduplicatedEntity]:
        """Run LLM deduplicate cluster."""
        try:
            entity_candidates_text = ""
            for i, entity in enumerate(cluster_entities, 1):
                entity_candidates_text += f"{i}. Name: \"{entity.get('name', '')}\"\n"
                entity_candidates_text += f"   Type: {entity.get('type', 'UNKNOWN')}\n"
                entity_candidates_text += f"   Session: {entity.get('session_id', '')}\n"
                entity_candidates_text += f"   Session Date: {entity.get('session_date', '')}\n"
                entity_candidates_text += f"   Context: {entity.get('content', '')}\n"
                entity_candidates_text += f"   Temporal: {entity.get('temporal_info', 'N/A')}\n"
                entity_candidates_text += f"   Temporal Reference: {entity.get('temporal_reference', 'N/A')}\n"
                entity_candidates_text += f"   Spatial: {entity.get('spatial_info', 'N/A')}\n"
                entity_candidates_text += f"   Numerical: {entity.get('numerical_value', 'N/A')}\n"
                entity_candidates_text += f"   Aliases: {entity.get('aliases', [])}\n"
                entity_candidates_text += f"   Confidence: {entity.get('confidence', 0.95)}\n\n"
            
            prompt = self.dedup_prompt_template.format(
                cluster_size=len(cluster_entities),
                cluster_id=cluster_id,
                entity_candidates=entity_candidates_text
            )
            
            response = self.llm_client.generate_answer(
                prompt,
                temperature=0.05,
                json_format=True
            )
            
            self._update_stats(llm_merge_calls=1)
            
            try:
                result = json.loads(response)
                merged_entities = result.get("merged_entities", [])
                
                extracted_entities = []
                for i, merged in enumerate(merged_entities):
                    extracted_entity = self._create_merged_entity_from_llm(
                        merged, 
                        cluster_entities, 
                        f"{qa_id}_LLM_C{cluster_id}_{i}"
                    )
                    extracted_entities.append(extracted_entity)
                
                return extracted_entities
                
            except json.JSONDecodeError as e:
                self.logger.warning(f" 聚类 {cluster_id} JSON解析失败: {e}, 尝试修复")
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    try:
                        json_str = json_match.group()
                        result = json.loads(json_str)
                        merged_entities = result.get("merged_entities", [])
                        
                        extracted_entities = []
                        for i, merged in enumerate(merged_entities):
                            extracted_entity = self._create_merged_entity_from_llm(
                                merged, 
                                cluster_entities, 
                                f"{qa_id}_LLM_C{cluster_id}_{i}"
                            )
                            extracted_entities.append(extracted_entity)
                        
                        return extracted_entities
                    except Exception:
                        pass
                
                self.logger.warning(f" JSON修复失败,回退到规则合并")
                return [self._merge_entity_cluster(cluster_entities, f"{qa_id}_C{cluster_id}", qa_id)]
                
        except Exception as e:
            self.logger.error(f" 聚类 {cluster_id} LLM去重失败: {e}")
            return [self._merge_entity_cluster(cluster_entities, f"{qa_id}_C{cluster_id}", qa_id)]
    
    def _process_large_cluster(self, 
                               cluster_entities: List[Dict[str, Any]], 
                               cluster_id: int,
                               qa_id: str) -> List[DeduplicatedEntity]:
        """Process large cluster."""
        batch_size = self.large_cluster_threshold
        batches = []
        
        for i in range(0, len(cluster_entities), batch_size):
            batch = cluster_entities[i:i + batch_size]
            batch_id = f"{cluster_id}_batch_{i//batch_size}"
            batches.append((batch, batch_id))
        
        self.logger.debug(f" 大聚类 {cluster_id} 分为 {len(batches)} 个批次处理")
        
        all_merged = []
        
        for batch, batch_id in batches:
            try:
                batch_merged = self._llm_deduplicate_cluster(batch, batch_id, qa_id)
                all_merged.extend(batch_merged)
            except Exception as e:
                self.logger.error(f"批次 {batch_id} LLM去重失败: {e}, 使用规则合并")
                merged = self._merge_entity_cluster(batch, f"{qa_id}_{batch_id}", qa_id)
                all_merged.append(merged)
        
        self.logger.debug(f" 大聚类 {cluster_id} 处理完成: {len(cluster_entities)} -> {len(all_merged)}")
        return all_merged
    
    def _create_merged_entity_from_llm(self, 
                                      merged_data: Dict[str, Any], 
                                      source_entities: List[Dict[str, Any]], 
                                      entity_id: str) -> DeduplicatedEntity:
        """Create merged entity from LLM."""
        mentions = []
        
        if 'mentions' in merged_data:
            for mention_data in merged_data['mentions']:
                mention = EntityMention(
                    session_id=mention_data.get('session_id', ''),
                    session_date=mention_data.get('session_date', ''),
                    context=mention_data.get('context', ''),
                    temporal_info=mention_data.get('temporal_info'),
                    temporal_reference=mention_data.get('temporal_reference'),
                    spatial_info=mention_data.get('spatial_info'),
                    numerical_value=mention_data.get('numerical_value'),
                    aliases=mention_data.get('aliases', []),
                    confidence=float(mention_data.get('confidence', 0.95))
                )
                mentions.append(mention)
        else:
            for entity in source_entities:
                mention = EntityMention(
                    session_id=entity.get('session_id', ''),
                    session_date=entity.get('session_date', ''),
                    context=entity.get('content', ''),
                    temporal_info=entity.get('temporal_info'),
                    temporal_reference=entity.get('temporal_reference'),
                    spatial_info=entity.get('spatial_info'),
                    numerical_value=entity.get('numerical_value'),
                    aliases=entity.get('aliases', []),
                    confidence=float(entity.get('confidence', 0.95))
                )
                mentions.append(mention)
        
        avg_confidence = sum(m.confidence for m in mentions) / len(mentions) if mentions else 0.95
        
        return DeduplicatedEntity(
            entity_id=entity_id,
            name=merged_data.get("name", source_entities[0].get('name', '')),
            entity_type=merged_data.get("entity_type", source_entities[0].get('type', 'UNKNOWN')),
            confidence=float(merged_data.get("confidence", avg_confidence)),
            mentions=mentions,
            total_mentions=len(mentions),
            session_ids=list(set([m.session_id for m in mentions])),
            temporal_info=merged_data.get('temporal_info'),
            spatial_info=merged_data.get('spatial_info'),
            numerical_value=merged_data.get('numerical_value')
        )
    
    def _merge_entity_cluster(self, 
                             entities: List[Dict[str, Any]], 
                             new_id: str,
                             qa_id: str) -> DeduplicatedEntity:
        """Run merge entity cluster."""
        if len(entities) == 1:
            return self._convert_to_deduplicated(entities[0], new_id)
        
        standard_name = max([e.get('name', '') for e in entities], key=len)
        entity_type = entities[0].get('type', 'UNKNOWN')
        
        all_mentions = []
        confidences = []
        
        for entity in entities:
            mention = EntityMention(
                session_id=entity.get('session_id', ''),
                session_date=entity.get('session_date', ''),
                context=entity.get('content', ''),
                temporal_info=entity.get('temporal_info'),
                temporal_reference=entity.get('temporal_reference'),
                spatial_info=entity.get('spatial_info'),
                numerical_value=entity.get('numerical_value'),
                aliases=entity.get('aliases', []),
                confidence=float(entity.get('confidence', 0.95))
            )
            all_mentions.append(mention)
            confidences.append(mention.confidence)
        
        temporal_info = None
        spatial_info = None
        numerical_value = None
        
        for e in entities:
            if not temporal_info and e.get('temporal_info'):
                temporal_info = e['temporal_info']
            if not spatial_info and e.get('spatial_info'):
                spatial_info = e['spatial_info']
            if not numerical_value and e.get('numerical_value'):
                numerical_value = e['numerical_value']
        
        merged = DeduplicatedEntity(
            entity_id=new_id,
            name=standard_name,
            entity_type=entity_type,
            confidence=np.mean(confidences) if confidences else 0.95,
            mentions=all_mentions,
            total_mentions=len(all_mentions),
            session_ids=list(set([m.session_id for m in all_mentions])),
            temporal_info=temporal_info,
            spatial_info=spatial_info,
            numerical_value=numerical_value
        )
        
        return merged
    
    def _convert_to_deduplicated(self, 
                                entity: Dict[str, Any], 
                                new_id: str) -> DeduplicatedEntity:
        """Convert to deduplicated."""
        mention = EntityMention(
            session_id=entity.get('session_id', ''),
            session_date=entity.get('session_date', ''),
            context=entity.get('content', ''),
            temporal_info=entity.get('temporal_info'),
            temporal_reference=entity.get('temporal_reference'),
            spatial_info=entity.get('spatial_info'),
            numerical_value=entity.get('numerical_value'),
            aliases=entity.get('aliases', []),
            confidence=entity.get('confidence', 0.95)
        )
        
        return DeduplicatedEntity(
            entity_id=new_id,
            name=entity.get('name', ''),
            entity_type=entity.get('type', 'UNKNOWN'),
            confidence=entity.get('confidence', 0.95),
            mentions=[mention],
            total_mentions=1,
            session_ids=[entity.get('session_id', '')],
            temporal_info=entity.get('temporal_info'),
            spatial_info=entity.get('spatial_info'),
            numerical_value=entity.get('numerical_value')
        )
    
    def process_qa_batch(self, 
                    qa_entities: Dict[str, List[Dict]], 
                    output_dir: str,
                    start_index: int = 0,
                    end_index: Optional[int] = None) -> Dict[str, Any]:
        """Process qa batch."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        qa_ids = sorted(qa_entities.keys(), key=lambda x: int(x.split('_')[1]))
        if end_index is not None:
            qa_ids = qa_ids[start_index:end_index]
        else:
            qa_ids = qa_ids[start_index:]
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f" 开始批量去重处理")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"处理范围: {len(qa_ids)} 个QA")
        self.logger.info(f"并行线程数: {self.parallel_workers}")
        self.logger.info(f"LLM去重: {'启用' if self.use_llm_dedup else '禁用'}")
        self.logger.info(f"逐QA优化: {'启用' if self.optimize_per_qa else '禁用'}")
        self.logger.info(f"{'='*80}\n")
        
        start_time = datetime.now()
        
        results = {}
        failed_qa_ids = []
        completed_count = 0
        total_qa_count = len([qa_id for qa_id in qa_ids if qa_entities.get(qa_id)])
        
        with ThreadPoolExecutor(max_workers=self.parallel_workers) as executor:
            future_to_qa = {}
            for qa_id in qa_ids:
                entities = qa_entities.get(qa_id, [])
                if not entities:
                    self.logger.debug(f" 跳过空QA: {qa_id}")
                    continue
                
                future = executor.submit(self._process_single_qa_safe, qa_id, entities, output_path)
                future_to_qa[future] = qa_id
            
            self.logger.info(f"Waiting for {len(future_to_qa)} tasks to complete...\n")
            
            try:
                for future in as_completed(future_to_qa):
                    qa_id = future_to_qa[future]
                    
                    try:
                        deduplicated_entities = future.result()
                        
                        if deduplicated_entities is not None:
                            results[qa_id] = deduplicated_entities
                            completed_count += 1
                            
                            self._update_stats(
                                qa_processed=1,
                                entities_after_dedup=len(deduplicated_entities)
                            )
                            
                            progress = (completed_count / total_qa_count) * 100
                            self.logger.info(f" [{completed_count}/{total_qa_count}] {qa_id} 完成: "
                                        f"{len(deduplicated_entities)} 个实体 ({progress:.1f}%)")
                        else:
                            failed_qa_ids.append(qa_id)
                            self.logger.error(f" {qa_id} 处理失败")
                    
                    except Exception as e:
                        failed_qa_ids.append(qa_id)
                        self.logger.error(f" {qa_id} 处理异常: {e}")
                        if self.logger.isEnabledFor(logging.DEBUG):
                            self.logger.debug(traceback.format_exc())
            
            except KeyboardInterrupt:
                self.logger.warning(f"\n 收到中断信号，正在安全关闭...")
                for future in future_to_qa:
                    if not future.done():
                        future.cancel()
                raise
        
        processing_time = (datetime.now() - start_time).total_seconds()
        self._update_stats(processing_time=processing_time)
        
        summary = self._generate_summary(results)
        summary['failed_qa_ids'] = failed_qa_ids
        summary['failed_count'] = len(failed_qa_ids)
        summary['completed_count'] = completed_count
        summary['total_qa_count'] = total_qa_count
        
        summary_file = output_path / "deduplication_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f" 批量去重完成!")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"处理QA数: {completed_count}/{total_qa_count}")
        self.logger.info(f"失败QA数: {len(failed_qa_ids)}")
        if failed_qa_ids:
            self.logger.info(f"失败列表: {failed_qa_ids[:10]}{'...' if len(failed_qa_ids) > 10 else ''}")
        self.logger.info(f"原始实体数: {self.stats['raw_entities_extracted']}")
        self.logger.info(f"去重后实体数: {self.stats['entities_after_dedup']}")
        if self.stats['raw_entities_extracted'] > 0:
            dedup_rate = (1 - self.stats['entities_after_dedup']/self.stats['raw_entities_extracted'])*100
            self.logger.info(f"去重率: {dedup_rate:.1f}%")
        if self.stats.get('param_optimization_time', 0) > 0:
            self.logger.info(f"参数优化耗时: {self.stats['param_optimization_time']:.2f}秒")
        if self.stats.get('llm_merge_calls', 0) > 0:
            self.logger.info(f"LLM合并调用: {self.stats['llm_merge_calls']} 次")
        self.logger.info(f"总耗时: {processing_time:.2f}秒 ({processing_time/60:.1f}分钟)")
        if completed_count > 0:
            avg_time = processing_time / completed_count
            self.logger.info(f"平均耗时: {avg_time:.2f}秒/QA")
        self.logger.info(f"输出目录: {output_path}")
        self.logger.info(f"{'='*80}\n")
        
        return summary


    def _process_single_qa_safe(self, qa_id: str, entities: List[Dict], output_path: Path) -> Optional[List[DeduplicatedEntity]]:
        """Process single qa safe."""
        try:
            self.logger.debug(f" 开始处理 {qa_id}...")
            
            deduplicated_entities = self._deduplicate_entities_dbscan(entities, qa_id)
            
            
            output_file = output_path / f"{qa_id}_deduplicated.json"
            self._save_qa_result(qa_id, deduplicated_entities, output_file)
            
            return deduplicated_entities
            
        except KeyboardInterrupt:
            raise
            
        except Exception as e:
            self.logger.error(f" {qa_id} 处理异常: {type(e).__name__}: {e}")
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(traceback.format_exc())
            return None
    
    def _process_single_qa(self, qa_id: str, entities: List[Dict], output_path: Path) -> List[DeduplicatedEntity]:
        """Process single qa."""
        return self._deduplicate_entities_dbscan(entities, qa_id)
    
    def _save_qa_result(self, qa_id: str, entities: List[DeduplicatedEntity], output_file: Path):
        """Save qa result."""
        data = {
            "qa_id": qa_id,
            "total_entities": len(entities),
            "entities": [self._entity_to_dict(e) for e in entities],
            "created_at": datetime.now().isoformat()
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _entity_to_dict(self, entity: DeduplicatedEntity) -> Dict:
        """Run entity to dict."""
        return {
            "entity_id": entity.entity_id,
            "name": entity.name,
            "entity_type": entity.entity_type,
            "confidence": entity.confidence,
            "total_mentions": entity.total_mentions,
            "session_ids": entity.session_ids,
            "temporal_info": entity.temporal_info,
            "spatial_info": entity.spatial_info,
            "numerical_value": entity.numerical_value,
            "mentions": [asdict(m) for m in entity.mentions]
        }
    
    def _generate_summary(self, results: Dict[str, List[DeduplicatedEntity]]) -> Dict:
        """Generate summary."""
        type_counts = defaultdict(int)
        mention_counts = []
        
        for qa_id, entities in results.items():
            for entity in entities:
                type_counts[entity.entity_type] += 1
                mention_counts.append(entity.total_mentions)
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "statistics": self.stats,
            "entity_type_distribution": dict(type_counts),
            "mention_statistics": {
                "total": sum(mention_counts),
                "mean": float(np.mean(mention_counts)) if mention_counts else 0,
                "median": float(np.median(mention_counts)) if mention_counts else 0,
                "max": int(max(mention_counts)) if mention_counts else 0
            }
        }
        
        if self.qa_optimized_params:
            summary['optimization_info'] = {
                'optimized_qa_count': len(self.qa_optimized_params),
                'per_qa_optimization': self.optimize_per_qa,
                'avg_optimization_time': self.stats['param_optimization_time'] / max(self.stats['param_optimizations'], 1)
            }
        
        return summary


def main():
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(
        description="LongMemEval实体去重器 - Step 2 (支持LLM精细去重)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        示例:
        # 自动加载所有文件并处理
        python step2_entity_deduplication.py --auto-load

        # 默认模式: 逐QA优化 + LLM精细去重
        python step2_entity_deduplication.py --result-file batch_results/batch_xxx.jsonl

        # 指定LLM模型
        python step2_entity_deduplication.py --auto-load --llm-model deepseek-reasoner

        # 使用自定义LLM服务
        python step2_entity_deduplication.py --auto-load \
            --llm-model deepseek-ai/DeepSeek-V3.2-Exp \
            --llm-base-url https://api.siliconflow.cn/v1

        # 禁用逐QA优化（加快速度）
        python step2_entity_deduplication.py --auto-load --no-per-qa-optimization

        # 禁用LLM去重（仅使用规则合并）
        python step2_entity_deduplication.py --auto-load --no-llm-dedup

        # 单线程处理（最稳定）
        python step2_entity_deduplication.py --auto-load --parallel-workers 1

        # 调整并行度
        python step2_entity_deduplication.py --auto-load --parallel-workers 5
        """
    )
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--result-file", type=str,
                            help="单个批量推理结果文件(JSONL)")
    input_group.add_argument("--auto-load", action="store_true",
                            help="自动加载results-dir目录下所有JSONL文件")
    
    parser.add_argument("--results-dir", type=str,
                       default=str(paths.LONGMEMEVAL_ENTITY_RELATION_NEW_BATCH_RESULTS_DIR),
                       help="批量结果目录(默认: batch_results)")
    parser.add_argument("--exclude-dirs", nargs="+", default=["deprecated"],
                       help="自动加载时要排除的子目录(默认: deprecated)")
    
    parser.add_argument("--output-dir",
                       default=str(paths.LONGMEMEVAL_ENTITY_RELATION_NEW_DEDUPLICATED_LLM_DIR),
                       help="输出目录")
    
    parser.add_argument("--start-index", type=int, default=0,
                       help="起始QA索引")
    parser.add_argument("--end-index", type=int, default=None,
                       help="结束QA索引")
    
    # parser.add_argument("--llm-model", type=str, default="deepseek-reasoner",
    parser.add_argument("--dedup-model", type=str, default="deepseek-v3.2-dashscope",
                       help="去重模型名称")
    parser.add_argument("--llm-api-key", type=str, default=None,
                       help="LLM API密钥")
    parser.add_argument("--llm-base-url", type=str, default=None,
                       help="LLM API基础URL")
    
    parser.add_argument("--no-per-qa-optimization", action="store_true",
                       help="禁用逐QA参数优化")
    parser.add_argument("--no-llm-dedup", action="store_true",
                       help="禁用LLM精细去重")
    parser.add_argument("--llm-cluster-threshold", type=int, default=2,
                       help="触发LLM去重的聚类大小阈值(默认2,即所有聚类都用LLM)")
    parser.add_argument("--large-cluster-threshold", type=int, default=12,
                       help="大聚类阈值(需要分批处理)")
    
    parser.add_argument("--parallel-workers", type=int, default=40,
                       help="并行线程数(建议1-10，大数据集建议用1)")
    
    parser.add_argument("--debug", action="store_true",
                       help="启用调试模式")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        deduplicator = LongMemEvalEntityDeduplicator(
            llm_model=args.dedup_model,
            llm_api_key=args.llm_api_key,
            llm_base_url=args.llm_base_url,
            optimize_per_qa=not args.no_per_qa_optimization,
            use_llm_dedup=not args.no_llm_dedup,
            llm_cluster_threshold=args.llm_cluster_threshold,
            large_cluster_threshold=args.large_cluster_threshold,
            parallel_workers=args.parallel_workers
        )
        
        
        if args.auto_load:
            logger.info(f" 自动加载模式")
            qa_entities = deduplicator.load_all_batch_results(
                args.results_dir,
                exclude_dirs=args.exclude_dirs
            )
        else:
            logger.info(f" 单文件模式")
            qa_entities = deduplicator.load_batch_results(args.result_file)
        
        summary = deduplicator.process_qa_batch(
            qa_entities,
            args.output_dir,
            start_index=args.start_index,
            end_index=args.end_index
        )
        
        print(f"\n{'='*80}")
        print(f" 实体去重完成!")
        print(f"{'='*80}")
        print(f" 统计信息:")
        print(f"  处理QA数: {summary.get('completed_count', 0)}/{summary.get('total_qa_count', 0)}")
        print(f"  失败QA数: {summary.get('failed_count', 0)}")
        print(f"  原始实体数: {summary['statistics']['raw_entities_extracted']}")
        print(f"  去重后实体数: {summary['statistics']['entities_after_dedup']}")
        print(f"  聚类数: {summary['statistics']['clusters_processed']}")
        if summary['statistics']['raw_entities_extracted'] > 0:
            dedup_rate = (1 - summary['statistics']['entities_after_dedup']/summary['statistics']['raw_entities_extracted'])*100
            print(f"  去重率: {dedup_rate:.1f}%")
        if summary['statistics'].get('llm_merge_calls', 0) > 0:
            print(f"  LLM合并调用: {summary['statistics']['llm_merge_calls']} 次")
        
        print(f"\n 输出目录: {args.output_dir}")
        print(f"{'='*80}\n")
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning(f"\n 用户中断，程序退出")
        return 130
        
    except Exception as e:
        logger.error(f" 程序异常: {e}")
        if args.debug:
            import traceback
            logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())
