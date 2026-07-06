#!/usr/bin/env python3
"""Utilities for step2 entity deduplication without llm."""
import json
import sys
import argparse
import logging
import numpy as np
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
            self.session_ids = list(set([m.session_id for m in self.mentions]))
        if not self.total_mentions:
            self.total_mentions = len(self.mentions)


class LongMemEvalEntityDeduplicator:
    
    def __init__(self,
                 llm_client: Optional[LLMClient] = None,
                 embedding_model: str = "Qwen/Qwen3-Embedding-4B",
                #  embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 dbscan_eps: Optional[float] = None,
                 dbscan_min_samples: Optional[int] = None,
                 auto_optimize_params: bool = False,
                 optimization_sample_size: int = 3,
                 parallel_workers: int = 10):
        self.logger = logging.getLogger(__name__)
        self.parallel_workers = parallel_workers
        self.auto_optimize_params = auto_optimize_params or (dbscan_eps is None and dbscan_min_samples is None)
        self.optimization_sample_size = optimization_sample_size
        
        self.dbscan_eps = dbscan_eps if dbscan_eps is not None else 0.3
        self.dbscan_min_samples = dbscan_min_samples if dbscan_min_samples is not None else 2
        
        self.optimized_params = {}
        
        self.llm_client = llm_client or LLMClient(
            model_name="deepseek-reasoner",
        )
        # self.llm_client = llm_client or LLMClient(
        #     model_name="deepseek-ai/DeepSeek-V3.2-Exp",
        #     api_key=None,
        #     base_url="https://api.siliconflow.cn/v1"
        # )
        
        self.logger.info(f"加载嵌入模型: {embedding_model}")
        self.encoder = SentenceTransformer(embedding_model)
        
        self.stats_lock = Lock()
        self.stats = {
            "qa_processed": 0,
            "requests_loaded": 0,
            "raw_entities_extracted": 0,
            "entities_after_dedup": 0,
            "clusters_processed": 0,
            "llm_merge_calls": 0,
            "param_optimization_time": 0.0,
            "processing_time": 0.0,
        }
        
        self.logger.info(f" LongMemEval实体去重器初始化完成")
        if self.auto_optimize_params:
            self.logger.info(f" 已启用自动参数优化 (样本数={optimization_sample_size})")
    
    def _update_stats(self, **kwargs):
        """Run update stats."""
        with self.stats_lock:
            for key, value in kwargs.items():
                if key in self.stats:
                    self.stats[key] += value
    
    def optimize_parameters_on_sample(self, 
                                     qa_entities: Dict[str, List[Dict]],
                                     n_samples: Optional[int] = None) -> Dict[str, Dict[str, float]]:
        """Args: qa_entities: {qa_id: [entities]} Returns:."""
        if n_samples is None:
            n_samples = self.optimization_sample_size
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f" 开始DBSCAN参数优化")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"样本QA数: {n_samples}")
        self.logger.info(f"{'='*80}\n")
        
        start_time = datetime.now()
        
        qa_ids = list(qa_entities.keys())[:n_samples]
        
        type_entities = defaultdict(list)
        for qa_id in qa_ids:
            entities = qa_entities[qa_id]
            for entity in entities:
                entity_type = entity.get('type', 'UNKNOWN')
                type_entities[entity_type].append(entity)
        
        optimized_params = {}
        
        for entity_type, entities in type_entities.items():
            if len(entities) < 10:
                self.logger.info(f"  跳过类型 {entity_type}: 样本太少({len(entities)})")
                continue
            
            self.logger.info(f" 优化类型 {entity_type}: {len(entities)} 个实体")
            
            
            texts = [self._prepare_entity_for_clustering(e) for e in entities]
            embeddings = self.encoder.encode(texts, show_progress_bar=False)
            
            class EntityWrapper:
                def __init__(self, idx, embedding):
                    self.uid = f"temp_{idx}"
                    self.embedding = embedding
                    self.metadata = {}
            
            wrapped = [EntityWrapper(i, emb) for i, emb in enumerate(embeddings)]
            
            try:
                result = optimize_dbscan_parameters(
                    wrapped,
                    eps_range=(0.15, 0.6),
                    min_samples_range=(2, min(8, len(entities) // 3)),
                    metric='cosine',
                    n_trials=16
                )
                
                best_params = result['best_params']
                best_score = result['best_score']
                
                optimized_params[entity_type] = best_params
                
                self.logger.info(f"   {entity_type}: eps={best_params['eps']:.3f}, "
                               f"min_samples={best_params['min_samples']}, "
                               f"score={best_score:.3f}")
                
            except Exception as e:
                self.logger.warning(f"   {entity_type} 优化失败: {e}, 使用默认参数")
                optimized_params[entity_type] = {
                    'eps': self.dbscan_eps,
                    'min_samples': self.dbscan_min_samples
                }
        
        if optimized_params:
            avg_eps = np.mean([p['eps'] for p in optimized_params.values()])
            avg_min_samples = int(np.median([p['min_samples'] for p in optimized_params.values()]))
            
            optimized_params['_default'] = {
                'eps': float(avg_eps),
                'min_samples': avg_min_samples
            }
            
            self.logger.info(f"\n 全局默认参数: eps={avg_eps:.3f}, min_samples={avg_min_samples}")
        else:
            optimized_params['_default'] = {
                'eps': self.dbscan_eps,
                'min_samples': self.dbscan_min_samples
            }
        
        optimization_time = (datetime.now() - start_time).total_seconds()
        self._update_stats(param_optimization_time=optimization_time)
        
        self.logger.info(f"\n 参数优化完成，耗时 {optimization_time:.2f}秒")
        self.logger.info(f"{'='*80}\n")
        
        self.optimized_params = optimized_params
        
        return optimized_params
    
    def _get_params_for_type(self, entity_type: str) -> Tuple[float, int]:
        """Get params for type."""
        if not self.optimized_params:
            return self.dbscan_eps, self.dbscan_min_samples
        
        if entity_type in self.optimized_params:
            params = self.optimized_params[entity_type]
            return params['eps'], params['min_samples']
        
        if '_default' in self.optimized_params:
            params = self.optimized_params['_default']
            return params['eps'], params['min_samples']
        
        return self.dbscan_eps, self.dbscan_min_samples
    
    def load_batch_results(self, result_file: str) -> Dict[str, List[Dict]]:
        """Load batch results."""
        self.logger.info(f" 加载批量推理结果: {result_file}")
        
        qa_entities = defaultdict(list)
        request_count = 0
        
        with open(result_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                
                request_count += 1
                result = json.loads(line)
                
                custom_id = result.get("custom_id", "")
                if not custom_id.startswith("qa_"):
                    continue
                
                parts = custom_id.split("_")
                qa_index = parts[1]
                qa_id = f"qa_{qa_index}"
                
                try:
                    response_body = result["response"]["body"]
                    message_content = response_body["choices"][0]["message"]["content"]
                    
                    try:
                        entities_data = json.loads(message_content)
                    except json.JSONDecodeError:
                        repaired = repair_json(message_content)
                        entities_data = json.loads(repaired)
                    
                    entities = entities_data.get("entities", [])
                    
                    if entities:
                        qa_entities[qa_id].extend(entities)
                        self._update_stats(raw_entities_extracted=len(entities))
                    
                except (KeyError, json.JSONDecodeError) as e:
                    self.logger.warning(f"解析实体失败 {custom_id}: {e}")
                    continue
        
        self._update_stats(requests_loaded=request_count)
        self.logger.info(f" 加载完成: {request_count} 个请求, {len(qa_entities)} 个QA")
        
        return dict(qa_entities)
    
    def _prepare_entity_for_clustering(self, entity: Dict[str, Any]) -> str:
        """Run prepare entity for clustering."""
        parts = [
            f"Name: {entity.get('name', '')}",
            f"Type: {entity.get('type', '')}",
            f"Content: {entity.get('content', '')[:200]}",
        ]
        
        aliases = entity.get('aliases', [])
        if aliases:
            parts.append(f"Aliases: {', '.join(aliases)}")
        
        return " | ".join(parts)
    
    def _deduplicate_entities_dbscan(self, 
                                     entities: List[Dict[str, Any]],
                                     qa_id: str) -> List[DeduplicatedEntity]:
        """Deduplicate entities dbscan."""
        if not entities:
            return []
        
        self.logger.info(f" 开始去重 {qa_id}: {len(entities)} 个原始实体")
        
        type_groups = defaultdict(list)
        for idx, entity in enumerate(entities):
            entity['_temp_idx'] = idx
            entity_type = entity.get('type', 'UNKNOWN')
            type_groups[entity_type].append(entity)
        
        deduplicated_entities = []
        
        for entity_type, type_entities in type_groups.items():
            if len(type_entities) == 1:
                deduplicated_entities.append(
                    self._convert_to_deduplicated(type_entities[0], f"{qa_id}_E{len(deduplicated_entities)+1}")
                )
                continue
            
            eps, min_samples = self._get_params_for_type(entity_type)
            
            self.logger.info(f"  处理类型 {entity_type}: {len(type_entities)} 个实体 "
                           f"(eps={eps:.3f}, min_samples={min_samples})")
            
            
            texts = [self._prepare_entity_for_clustering(e) for e in type_entities]
            embeddings = self.encoder.encode(texts, show_progress_bar=False)
            
            class EntityWrapper:
                def __init__(self, entity, embedding, idx):
                    self.uid = f"temp_{idx}"
                    self.embedding = embedding
                    self.entity = entity
                    self.metadata = {}
            
            wrapped_entities = [
                EntityWrapper(e, emb, e['_temp_idx']) 
                for e, emb in zip(type_entities, embeddings)
            ]
            
            try:
                clusters = find_clusters_with_dbscan(
                    wrapped_entities,
                    eps=eps,
                    min_samples=min_samples,
                    metric='cosine',
                    normalize_embeddings=True
                )
            except Exception as e:
                self.logger.warning(f"DBSCAN聚类失败 {qa_id} {entity_type}: {e}")
                for entity in type_entities:
                    deduplicated_entities.append(
                        self._convert_to_deduplicated(entity, f"{qa_id}_E{len(deduplicated_entities)+1}")
                    )
                continue
            
            self._update_stats(clusters_processed=len(clusters))
            
            for cluster_id, uids in clusters.items():
                if cluster_id == -1:
                    for uid in uids:
                        idx = int(uid.split('_')[1])
                        entity = next(e for e in type_entities if e['_temp_idx'] == idx)
                        deduplicated_entities.append(
                            self._convert_to_deduplicated(entity, f"{qa_id}_E{len(deduplicated_entities)+1}")
                        )
                else:
                    cluster_entities = []
                    for uid in uids:
                        idx = int(uid.split('_')[1])
                        entity = next(e for e in type_entities if e['_temp_idx'] == idx)
                        cluster_entities.append(entity)
                    
                    merged = self._merge_entity_cluster(
                        cluster_entities, 
                        f"{qa_id}_E{len(deduplicated_entities)+1}",
                        qa_id
                    )
                    deduplicated_entities.append(merged)
        
        self.logger.info(f" 去重完成 {qa_id}: {len(entities)} -> {len(deduplicated_entities)}")
        return deduplicated_entities
    
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
                confidence=entity.get('confidence', 0.95)
            )
            all_mentions.append(mention)
            confidences.append(entity.get('confidence', 0.95))
        
        temporal_info = None
        spatial_info = None
        numerical_value = None
        
        for e in entities:
            if not temporal_info and e.get('temporal_info'):
                temporal_info = e.get('temporal_info')
            if not spatial_info and e.get('spatial_info'):
                spatial_info = e.get('spatial_info')
            if not numerical_value and e.get('numerical_value'):
                numerical_value = e.get('numerical_value')
        
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
            qa_ids = [qid for qid in qa_ids 
                     if start_index <= int(qid.split('_')[1]) <= end_index]
        
        if self.auto_optimize_params and not self.optimized_params:
            self.optimize_parameters_on_sample(qa_entities, self.optimization_sample_size)
            
            
            params_file = output_path / "optimized_dbscan_params.json"
            with open(params_file, 'w', encoding='utf-8') as f:
                json.dump(self.optimized_params, f, indent=2)
            self.logger.info(f" 优化参数已保存: {params_file}")
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f" 开始批量去重处理")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"处理范围: {len(qa_ids)} 个QA")
        self.logger.info(f"并行线程数: {self.parallel_workers}")
        if self.optimized_params:
            self.logger.info(f"参数优化: 已启用({len(self.optimized_params)-1} 种类型)")
        else:
            self.logger.info(f"DBSCAN参数: eps={self.dbscan_eps}, min_samples={self.dbscan_min_samples}")
        self.logger.info(f"{'='*80}\n")
        
        start_time = datetime.now()
        
        results = {}
        with ThreadPoolExecutor(max_workers=self.parallel_workers) as executor:
            future_to_qa = {
                executor.submit(self._deduplicate_entities_dbscan, qa_entities[qa_id], qa_id): qa_id
                for qa_id in qa_ids
            }
            
            for future in as_completed(future_to_qa):
                qa_id = future_to_qa[future]
                try:
                    deduplicated = future.result()
                    results[qa_id] = deduplicated
                    
                    self._update_stats(
                        qa_processed=1,
                        entities_after_dedup=len(deduplicated)
                    )
                    
                    
                    qa_output_file = output_path / f"{qa_id}_deduplicated.json"
                    self._save_qa_result(qa_id, deduplicated, qa_output_file)
                    
                    if self.stats['qa_processed'] % 10 == 0:
                        self.logger.info(f"进度: {self.stats['qa_processed']}/{len(qa_ids)} QA已完成")
                    
                except Exception as e:
                    self.logger.error(f"处理失败 {qa_id}: {e}")
                    import traceback
                    self.logger.error(traceback.format_exc())
        
        processing_time = (datetime.now() - start_time).total_seconds()
        self._update_stats(processing_time=processing_time)
        
        summary = self._generate_summary(results)
        summary_file = output_path / "deduplication_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f" 批量去重完成!")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"处理QA数: {self.stats['qa_processed']}")
        self.logger.info(f"原始实体数: {self.stats['raw_entities_extracted']}")
        self.logger.info(f"去重后实体数: {self.stats['entities_after_dedup']}")
        self.logger.info(f"去重率: {(1 - self.stats['entities_after_dedup']/self.stats['raw_entities_extracted'])*100:.1f}%")
        if self.stats['param_optimization_time'] > 0:
            self.logger.info(f"参数优化耗时: {self.stats['param_optimization_time']:.2f}秒")
        self.logger.info(f"去重处理耗时: {processing_time:.2f}秒")
        self.logger.info(f"输出目录: {output_path}")
        self.logger.info(f"{'='*80}\n")
        
        return summary
    
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
                "mean": np.mean(mention_counts) if mention_counts else 0,
                "median": np.median(mention_counts) if mention_counts else 0,
                "max": max(mention_counts) if mention_counts else 0
            }
        }
        
        if self.optimized_params:
            summary["optimized_parameters"] = self.optimized_params
        
        return summary
    
    def load_all_batch_results(self, results_dir: str, exclude_dirs: Optional[List[str]] = None) -> Dict[str, List[Dict]]:
        """Load all batch results."""
        if exclude_dirs is None:
            exclude_dirs = ['deprecated']
        
        results_path = Path(results_dir)
        if not results_path.exists():
            raise FileNotFoundError(f"结果目录不存在: {results_dir}")
        
        jsonl_files = []
        for file_path in results_path.glob("*.jsonl"):
            if any(excluded in file_path.parts for excluded in exclude_dirs):
                self.logger.info(f"  跳过排除目录中的文件: {file_path.name}")
                continue
            jsonl_files.append(file_path)
        
        if not jsonl_files:
            raise ValueError(f"在 {results_dir} 中未找到JSONL文件")
        
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
        
        for jsonl_file in jsonl_files:
            self.logger.info(f" 处理文件: {jsonl_file.name}")
            
            file_requests = 0
            file_entities = 0
            
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    
                    file_requests += 1
                    result = json.loads(line)
                    
                    custom_id = result.get("custom_id", "")
                    if not custom_id.startswith("qa_"):
                        continue
                    
                    parts = custom_id.split("_")
                    qa_index = parts[1]
                    qa_id = f"qa_{qa_index}"
                    
                    try:
                        response_body = result["response"]["body"]
                        message_content = response_body["choices"][0]["message"]["content"]
                        
                        try:
                            entities_data = json.loads(message_content)
                        except json.JSONDecodeError:
                            repaired = repair_json(message_content)
                            entities_data = json.loads(repaired)
                        
                        entities = entities_data.get("entities", [])
                        
                        if entities:
                            qa_entities[qa_id].extend(entities)
                            file_entities += len(entities)
                        
                    except (KeyError, json.JSONDecodeError) as e:
                        self.logger.warning(f"解析实体失败 {custom_id}: {e}")
                        continue
            
            total_requests += file_requests
            total_entities += file_entities
            
            self.logger.info(f"   {jsonl_file.name}: {file_requests} 个请求, {file_entities} 个实体")
        
        self._update_stats(
            requests_loaded=total_requests,
            raw_entities_extracted=total_entities
        )
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f" 所有文件加载完成")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"总文件数: {len(jsonl_files)}")
        self.logger.info(f"总请求数: {total_requests}")
        self.logger.info(f"总实体数: {total_entities}")
        self.logger.info(f"QA数量: {len(qa_entities)}")
        self.logger.info(f"{'='*80}\n")
        
        return dict(qa_entities)

def main():
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(
        description="LongMemEval实体去重器 - Step 2 (支持自动参数优化)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        示例:
        # 自动读取batch_results目录下所有JSONL文件
        python step2_entity_deduplication_without_llm.py --auto-load

        # 自动读取并优化参数
        python step2_entity_deduplication_without_llm.py --auto-load --auto-optimize

        # 指定结果目录自动读取
        python step2_entity_deduplication_without_llm.py --results-dir batch_results --auto-load

        # 自动读取并指定排除目录
        python step2_entity_deduplication_without_llm.py --auto-load --exclude-dirs deprecated backup

        # 单文件处理(原有功能)
        python step2_entity_deduplication_without_llm.py --result-file batch_results/batch_xxx.jsonl

        # 自动优化参数并去重
        python step2_entity_deduplication_without_llm.py --result-file batch_results/batch_xxx.jsonl --auto-optimize

        # 指定参数去重
        python step2_entity_deduplication_without_llm.py --result-file batch_results/batch_xxx.jsonl --eps 0.25 --min-samples 3

        # 加载已保存的优化参数
        python step2_entity_deduplication_without_llm.py --auto-load --load-params optimized_dbscan_params.json
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
                       default=str(paths.LONGMEMEVAL_ENTITY_RELATION_NEW_DEDUPLICATED_DIR),
                       help="输出目录")
    
    parser.add_argument("--start-index", type=int, default=0,
                       help="起始QA索引")
    parser.add_argument("--end-index", type=int, default=None,
                       help="结束QA索引")
    
    parser.add_argument("--auto-optimize", action="store_true",
                       help="自动优化DBSCAN参数")
    parser.add_argument("--optimization-samples", type=int, default=3,
                       help="参数优化使用的样本QA数量")
    parser.add_argument("--load-params", type=str, default=None,
                       help="加载已保存的优化参数文件")
    
    parser.add_argument("--eps", type=float, default=None,
                       help="DBSCAN邻域半径(不指定则自动优化)")
    parser.add_argument("--min-samples", type=int, default=None,
                       help="DBSCAN最小样本数(不指定则自动优化)")
    
    parser.add_argument("--parallel-workers", type=int, default=10,
                       help="并行线程数")
    
    parser.add_argument("--debug", action="store_true",
                       help="启用调试模式")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        deduplicator = LongMemEvalEntityDeduplicator(
            dbscan_eps=args.eps,
            dbscan_min_samples=args.min_samples,
            auto_optimize_params=args.auto_optimize,
            optimization_sample_size=args.optimization_samples,
            parallel_workers=args.parallel_workers
        )
        
        
        if args.load_params:
            params_file = Path(args.load_params)
            if params_file.exists():
                with open(params_file, 'r') as f:
                    deduplicator.optimized_params = json.load(f)
                logger.info(f" 已加载优化参数: {params_file}")
            else:
                logger.warning(f"参数文件不存在: {params_file}")
        
        
        if args.auto_load:
            
            qa_entities = deduplicator.load_all_batch_results(
                args.results_dir,
                exclude_dirs=args.exclude_dirs
            )
        else:
            
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
        print(f"  处理QA数: {summary['statistics']['qa_processed']}")
        print(f"  原始实体数: {summary['statistics']['raw_entities_extracted']}")
        print(f"  去重后实体数: {summary['statistics']['entities_after_dedup']}")
        print(f"  聚类数: {summary['statistics']['clusters_processed']}")
        print(f"  去重率: {(1 - summary['statistics']['entities_after_dedup']/summary['statistics']['raw_entities_extracted'])*100:.1f}%")
        
        if 'optimized_parameters' in summary:
            print(f"\n 优化参数:")
            for entity_type, params in summary['optimized_parameters'].items():
                if entity_type != '_default':
                    print(f"  {entity_type}: eps={params['eps']:.3f}, min_samples={params['min_samples']}")
        
        print(f"\n 输出目录: {args.output_dir}")
        print(f"{'='*80}\n")
        
        return 0
        
    except Exception as e:
        logger.error(f" 程序异常: {e}")
        if args.debug:
            import traceback
            logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())
