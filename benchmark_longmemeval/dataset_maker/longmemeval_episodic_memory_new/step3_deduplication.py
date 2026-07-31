#!/usr/bin/env python3
"""Utilities for step3 deduplication."""

import argparse
import json
import logging
import os
import re
import sys
import hashlib
from pathlib import Path

from json_repair import repair_json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict, Counter
from threading import Lock


import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score

from mandol.llm.llm_client import LLMClient
from mandol.core import paths

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



@dataclass
class FactMention:
    mention_id: str
    source_file: str
    source_session_id: str
    original_fact_id: str        # "F1", "F2"...
    content: str
    category: str
    temporal: Optional[Dict]
    attributes: Dict
    created_at: str
    
    def __post_init__(self):
        if self.temporal is None:
            self.temporal = {}
        if self.attributes is None:
            self.attributes = {}


@dataclass
class UniqueMemoryFact:
    uid: str
    canonical_content: str
    category: str
    mentions: List[Dict]
    aggregated_count: int
    temporal_val: Optional[str]
    is_stable: bool
    confidence: float = 0.95
    session_ids: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Run to dict."""
        return {
            "uid": self.uid,
            "canonical_content": self.canonical_content,
            "category": self.category,
            "mentions": self.mentions,
            "aggregated_count": self.aggregated_count,
            "temporal_val": self.temporal_val,
            "is_stable": self.is_stable,
            "confidence": self.confidence,
            "session_ids": self.session_ids
        }



def compute_safe_cosine_distance(embeddings: np.ndarray) -> Optional[np.ndarray]:
    """Compute safe cosine distance."""
    try:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        
        zero_mask = norms.flatten() == 0
        if np.any(zero_mask):
            logger.warning(f"发现 {np.sum(zero_mask)} 个零向量，将其替换为小随机值")
            embeddings[zero_mask] = np.random.normal(0, 1e-8, (np.sum(zero_mask), embeddings.shape[1]))
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        normalized = embeddings / norms
        
        cosine_sim = np.dot(normalized, normalized.T)
        
        distance_matrix = 1 - cosine_sim
        
        np.fill_diagonal(distance_matrix, 0)
        distance_matrix = np.maximum(distance_matrix, 0)
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        
        if np.any(np.isnan(distance_matrix)) or np.any(np.isinf(distance_matrix)):
            logger.error("距离矩阵包含NaN或无穷值")
            return None
        
        return distance_matrix
        
    except Exception as e:
        logger.error(f"计算余弦距离矩阵失败: {e}")
        return None


def evaluate_clustering_quality(labels: np.ndarray, features: np.ndarray) -> float:
    """Evaluate clustering quality."""
    unique_labels = set(labels)
    if len(unique_labels) <= 1 or (len(unique_labels) == 2 and -1 in unique_labels):
        return 0.0
    
    try:
        mask = labels != -1
        if np.sum(mask) < 2:
            return 0.0
        
        filtered_labels = labels[mask]
        filtered_features = features[mask]
        
        if len(set(filtered_labels)) < 2:
            return 0.0
        
        silhouette = silhouette_score(filtered_features, filtered_labels)
        
        ch_score = calinski_harabasz_score(filtered_features, filtered_labels)
        ch_normalized = min(ch_score / 1000.0, 1.0)
        
        noise_ratio = list(labels).count(-1) / len(labels)
        noise_penalty = 1.0 - noise_ratio
        
        final_score = (silhouette * 0.5 + ch_normalized * 0.3 + noise_penalty * 0.2)
        
        return max(0.0, final_score)
        
    except Exception as e:
        logger.debug(f"聚类质量评估失败: {e}")
        return simple_clustering_evaluation(labels)


def simple_clustering_evaluation(labels: np.ndarray) -> float:
    """Run simple clustering evaluation."""
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = list(labels).count(-1)
    
    if n_clusters == 0:
        return 0.0
    
    cluster_score = min(n_clusters / 10.0, 1.0)
    noise_penalty = 1.0 - (n_noise / len(labels))
    
    return (cluster_score * 0.6 + noise_penalty * 0.4)


def optimize_dbscan_parameters(embeddings: np.ndarray,
                               eps_range: Tuple[float, float] = (0.1, 0.6),
                               min_samples_range: Tuple[int, int] = (1, 5),
                               metric: str = 'cosine',
                               n_trials: int = 20) -> Dict[str, Any]:
    """Run optimize dbscan parameters."""
    n_samples = embeddings.shape[0]
    
    if n_samples < 10:
        logger.warning("样本数量太少，使用默认参数")
        return {
            'best_params': {'eps': 0.15, 'min_samples': 1},
            'best_score': 0.0,
            'evaluation': 'insufficient_data'
        }
    
    logger.info(f" 开始DBSCAN参数优化，样本数量: {n_samples}")
    
    distance_matrix = None
    if metric == 'cosine':
        distance_matrix = compute_safe_cosine_distance(embeddings.copy())
        if distance_matrix is None:
            logger.warning("余弦距离计算失败，改用欧几里得距离")
            metric = 'euclidean'
    
    best_params = None
    best_score = -1
    evaluation_results = []
    
    eps_values = np.linspace(eps_range[0], eps_range[1], max(n_trials // 4, 5))
    min_samples_max = min(min_samples_range[1] + 1, max(n_samples // 5, 2))
    min_samples_values = range(min_samples_range[0], min_samples_max)
    
    for eps in eps_values:
        for min_samples in min_samples_values:
            try:
                if metric == 'cosine' and distance_matrix is not None:
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
                    labels = dbscan.fit_predict(distance_matrix)
                else:
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
                    labels = dbscan.fit_predict(embeddings)
                
                score = evaluate_clustering_quality(labels, embeddings)
                
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = int(list(labels).count(-1))
                
                evaluation_results.append({
                    'eps': float(eps),
                    'min_samples': int(min_samples),
                    'score': float(score),
                    'n_clusters': n_clusters,
                    'n_noise': n_noise
                })
                
                if score > best_score:
                    best_score = score
                    best_params = {'eps': float(eps), 'min_samples': int(min_samples)}
                    
            except Exception as e:
                logger.debug(f"参数 eps={eps:.3f}, min_samples={min_samples} 失败: {e}")
                continue
    
    if best_params:
        logger.info(f" 参数优化完成，最佳得分: {best_score:.3f}")
        logger.info(f"   最佳参数: eps={best_params['eps']:.3f}, min_samples={best_params['min_samples']}")
    else:
        best_params = {'eps': 0.15, 'min_samples': 1}
        logger.warning(f" 参数优化失败，使用默认参数: {best_params}")
    
    return {
        'best_params': best_params,
        'best_score': float(best_score),
        'evaluation_results': evaluation_results,
        'evaluation': 'completed'
    }



class FactDeduplicator:
    
    def __init__(self,
             model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
             eps: float = 0.15,
             min_samples: int = 1,
             workers: int = 4,
             auto_optimize: bool = True,
             eps_range: Tuple[float, float] = (0.1, 0.6),
             min_samples_range: Tuple[int, int] = (1, 5),
             n_trials: int = 20,
             optimize_per_category: bool = True,
             use_llm_dedup: bool = True,
             llm_client: Optional[LLMClient] = None,
             llm_model: str = "deepseek-v3.2-dashscope",
             llm_api_key: Optional[str] = None,
             llm_base_url: Optional[str] = None,
             llm_cluster_threshold: int = 2,
             large_cluster_threshold: int = 15):
        self.eps = eps
        self.min_samples = min_samples
        self.workers = workers
        self.auto_optimize = auto_optimize
        self.eps_range = eps_range
        self.min_samples_range = min_samples_range
        self.n_trials = n_trials
        self.optimize_per_category = optimize_per_category
        
        self.use_llm_dedup = use_llm_dedup
        self.llm_cluster_threshold = llm_cluster_threshold
        self.large_cluster_threshold = large_cluster_threshold
        
        self.category_optimized_params = {}
        self.global_optimized_params = None
        
        self.stats_lock = Lock()
        self.stats = {
            "files_loaded": 0,
            "total_mentions": 0,
            "unique_facts": 0,
            "total_unique_facts": 0,
            "clusters_processed": 0,
            "events_split_by_date": 0,
            "attributes_merged": 0,
            "processing_time": 0.0,
            "param_optimizations": 0,
            "param_optimization_time": 0.0,
            "qa_processed": 0,
            "failed_qa_count": 0,
            "llm_merge_calls": 0,
            "llm_merge_time": 0.0
        }
        
        
        logger.info(f" 加载嵌入模型: {model_name}")
        self.encoder = SentenceTransformer(model_name)
        
        if use_llm_dedup:
            if llm_client is not None:
                self.llm_client = llm_client
            else:
                self.llm_client = LLMClient(
                    model_name=llm_model,
                    api_key=llm_api_key,
                    base_url=llm_base_url
                )
            logger.info(f" LLM客户端初始化: {llm_model}")
            
            self.dedup_prompt_template = self._prepare_llm_dedup_prompt()
        else:
            self.llm_client = None
            self.dedup_prompt_template = None
        
        logger.info(f" FactDeduplicator 初始化完成")
        logger.info(f"   - eps: {eps}, min_samples: {min_samples}, workers: {workers}")
        if auto_optimize:
            logger.info(f"   - 自动参数优化: 启用")
            logger.info(f"   - eps范围: {eps_range}, min_samples范围: {min_samples_range}")
            logger.info(f"   - 按类别优化: {'启用' if optimize_per_category else '禁用'}")
        if use_llm_dedup:
            logger.info(f"   - LLM精细去重: 启用 (阈值: {llm_cluster_threshold})")
            
    def _prepare_llm_dedup_prompt(self) -> str:
        """Run prepare LLM dedup prompt."""
        return """You are an expert in memory fact deduplication and standardization. The following are {cluster_size} similar memory facts grouped by semantic clustering. Please merge duplicates while preserving all session-specific information.

        Cluster {cluster_id} fact list:
        {fact_candidates}

        Please return deduplication results in JSON format:
        {{
            "merged_facts": [
                {{
                    "canonical_content": "The most complete and accurate description of this fact",
                    "category": "EPISODIC_EVENT | USER_ATTRIBUTE | USER_PREFERENCE | etc.",
                    "temporal_val": "Date if applicable (e.g., '2023-05-30') or null",
                    "confidence": 0.95,
                    "mentions": [
                        {{
                            "source_session_id": "session_id_from_original",
                            "original_content": "Original fact content from this session",
                            "temporal": {{"absolute_date": "2023-05-30"}},
                            "confidence": 0.90
                        }}
                    ],
                    "merge_reasoning": "Explanation of why these facts were merged or kept separate"
                }}
            ]
        }}

        **Critical Deduplication Rules:**
        1. **Preserve temporal accuracy**: Facts with different dates should generally NOT be merged
        2. **Maintain session traceability**: Each mention must retain the original session's information
        3. **Handle EPISODIC_EVENT carefully**: Events on different dates are DIFFERENT facts even if similar
        4. **Merge USER_ATTRIBUTE/PREFERENCE**: These can be merged if they describe the same attribute
        5. **Select best canonical content**: Choose the most complete and informative description
        6. **Calculate confidence**: Use weighted average based on original confidence scores
        7. **Avoid over-merging**: When in doubt, keep facts separate

        **Category-specific Guidelines:**
        - EPISODIC_EVENT: Split by date; same event on different dates = different facts
        - USER_ATTRIBUTE: Merge if describing the same attribute (e.g., "User likes coffee" + "User prefers coffee")
        - USER_PREFERENCE: Merge similar preferences about the same topic
        - Other categories: Merge if semantically identical

        **Output Requirements:**
        - Every source fact must be represented in at least one mention
        - Canonical content should be the most informative form
        - Keep temporal_val consistent within each merged fact
        - Confidence should reflect the quality of the merge decision
        """
        
    def _llm_deduplicate_cluster(self, 
                                cluster_mentions: List[FactMention], 
                                cluster_id: int,
                                qa_id: str) -> List[UniqueMemoryFact]:
        """Run LLM deduplicate cluster."""
        try:
            start_time = datetime.now()
            
            fact_candidates_text = ""
            for i, mention in enumerate(cluster_mentions, 1):
                fact_candidates_text += f"{i}. Content: \"{mention.content}\"\n"
                fact_candidates_text += f"   Category: {mention.category}\n"
                fact_candidates_text += f"   Session ID: {mention.source_session_id}\n"
                fact_candidates_text += f"   Temporal: {json.dumps(mention.temporal) if mention.temporal else 'N/A'}\n"
                fact_candidates_text += f"   Confidence: {mention.attributes.get('confidence', 0.95)}\n"
                fact_candidates_text += f"   Source File: {mention.source_file}\n\n"
            
            prompt = self.dedup_prompt_template.format(
                cluster_size=len(cluster_mentions),
                cluster_id=cluster_id,
                fact_candidates=fact_candidates_text
            )
            
            response = self.llm_client.generate_answer(
                prompt,
                temperature=0.05,
                json_format=True
            )
            
            llm_time = (datetime.now() - start_time).total_seconds()
            self._update_stats(llm_merge_calls=1, llm_merge_time=llm_time)
            
            result = None
            try:
                result = json.loads(response)
            except json.JSONDecodeError:
                try:
                    repaired = repair_json(response)
                    result = json.loads(repaired)
                    logger.debug(f"聚类 {cluster_id} 使用 json_repair 修复成功")
                except Exception:
                    pass
            
            if result is None:
                logger.warning(f" 聚类 {cluster_id} JSON解析失败，尝试提取 JSON 块")
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    try:
                        result = json.loads(json_str)
                    except json.JSONDecodeError:
                        try:
                            repaired = repair_json(json_str)
                            result = json.loads(repaired)
                        except Exception:
                            pass
            
            if result is not None:
                merged_facts = result.get("merged_facts", [])
                
                extracted_facts = []
                for i, merged in enumerate(merged_facts):
                    fact = self._create_fact_from_llm(
                        merged, 
                        cluster_mentions, 
                        f"{qa_id}_LLM_C{cluster_id}_{i}"
                    )
                    extracted_facts.append(fact)
                
                logger.debug(f" 聚类 {cluster_id} LLM去重: {len(cluster_mentions)} -> {len(extracted_facts)}")
                return extracted_facts
            else:
                logger.warning(f" JSON修复失败，回退到规则合并")
                return self._apply_aggregation_rules(cluster_mentions, cluster_id)
                
        except Exception as e:
            logger.error(f" 聚类 {cluster_id} LLM去重失败: {e}")
            return self._apply_aggregation_rules(cluster_mentions, cluster_id)
    
    def _update_stats(self, **kwargs):
        """Run update stats."""
        with self.stats_lock:
            for key, value in kwargs.items():
                if key in self.stats:
                    if isinstance(self.stats[key], (int, float)):
                        self.stats[key] += value
                    else:
                        self.stats[key] = value
    
    def _generate_mention_id(self, source_file: str, session_id: str, fact_id: str) -> str:
        """Generate mention id."""
        return f"{Path(source_file).stem}_{session_id}_{fact_id}"
    
    def _generate_fact_uid(self, content: str, category: str, temporal_val: Optional[str]) -> str:
        """Generate fact UID."""
        hash_input = f"{content.lower().strip()}|{category}|{temporal_val or 'none'}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]
    
    def _extract_temporal_value(self, temporal: Optional[Dict]) -> Optional[str]:
        """Extract temporal value."""
        if not temporal:
            return None
        
        if "absolute_date" in temporal:
            return temporal["absolute_date"]
        
        if "date" in temporal:
            return temporal["date"]
        
        for key in ["timestamp", "time", "datetime"]:
            if key in temporal:
                return temporal[key]
        
        return None
    
    def _get_params_for_category(self, category: str) -> Tuple[float, int]:
        """Get params for category."""
        if self.optimize_per_category and category in self.category_optimized_params:
            params = self.category_optimized_params[category]
            return params['eps'], params['min_samples']
        
        if self.global_optimized_params:
            return self.global_optimized_params['eps'], self.global_optimized_params['min_samples']
        
        return self.eps, self.min_samples
    
    
    
    def load_and_flatten_by_qa(self, input_dir: str) -> Dict[str, List[FactMention]]:
        """Load and flatten by qa."""
        input_path = Path(input_dir)
        
        if not input_path.exists():
            logger.error(f" 输入目录不存在: {input_dir}")
            return {}
        
        all_files = []
        for f in input_path.glob("*.jsonl"):
            if f.is_file():
                all_files.append(f)
        
        if not all_files:
            logger.warning(f" 目录中没有找到 JSONL 文件: {input_dir}")
            return {}
        
        main_files = []
        retry_files = []
        for f in all_files:
            if 'retry' in f.name.lower():
                retry_files.append(f)
            else:
                main_files.append(f)
        
        logger.info(f" 找到 {len(main_files)} 个主文件, {len(retry_files)} 个retry文件")
        
        qa_mentions: Dict[str, List[FactMention]] = defaultdict(list)
        
        
        logger.info(f" 加载主文件...")
        for file_path in main_files:
            file_qa_mentions = self._load_single_file_by_qa(file_path, is_retry=False)
            for qa_id, mentions in file_qa_mentions.items():
                qa_mentions[qa_id].extend(mentions)
            self._update_stats(files_loaded=1)
        
        
        if retry_files:
            logger.info(f" 加载retry文件并合并...")
            for file_path in retry_files:
                file_qa_mentions = self._load_single_file_by_qa(file_path, is_retry=True)
                for qa_id, mentions in file_qa_mentions.items():
                    qa_mentions[qa_id].extend(mentions)
                    logger.debug(f"   合并 {len(mentions)} 个retry facts到 {qa_id}")
                self._update_stats(files_loaded=1)
        
        total_mentions = sum(len(mentions) for mentions in qa_mentions.values())
        self._update_stats(total_mentions=total_mentions)
        
        logger.info(f" 加载完成: {len(qa_mentions)} 个QA, 共 {total_mentions} 个事实提及")
        
        for qa_id, mentions in sorted(qa_mentions.items()):
            source_files = set(m.source_file for m in mentions)
            if len(source_files) > 1:
                logger.debug(f"   {qa_id}: {len(mentions)} 个facts (来自 {len(source_files)} 个文件)")
        
        return dict(qa_mentions)
    
    def _load_single_file_by_qa(self, file_path: Path, is_retry: bool = False) -> Dict[str, List[FactMention]]:
        """Load single file by qa."""
        qa_mentions: Dict[str, List[FactMention]] = defaultdict(list)
        file_name = file_path.name
        
        logger.info(f"    加载{'retry' if is_retry else '主'}文件: {file_name}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                line_count = 0
                success_count = 0
                
                for line in f:
                    line_count += 1
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            repaired = repair_json(line)
                            record = json.loads(repaired)
                            logger.debug(f"   使用 json_repair 修复行 {line_count}")
                        
                        qa_id, mentions = self._extract_facts_from_record_with_qa(
                            record, file_name, is_retry=is_retry
                        )
                        
                        # if qa_id and mentions:
                        #     qa_mentions[qa_id].extend(mentions)
                        #     success_count += 1
                        
                        if qa_id:
                            success_count += 1 
                            if mentions:
                                qa_mentions[qa_id].extend(mentions)
                            
                    except (json.JSONDecodeError, Exception) as e:
                        logger.warning(f"    JSON解析错误 (行 {line_count}): {e}")
                    except Exception as e:
                        logger.warning(f"    处理记录失败 (行 {line_count}): {e}")
                
                logger.info(f"      成功处理 {success_count}/{line_count} 条记录, "
                          f"提取 {sum(len(m) for m in qa_mentions.values())} 个facts")
                
        except Exception as e:
            logger.error(f"    读取文件失败: {e}")
        
        return dict(qa_mentions)
    
    def _extract_facts_from_record_with_qa(self, record: Dict, file_name: str, 
                                        is_retry: bool = False) -> Tuple[Optional[str], List[FactMention]]:
        """Extract facts from record with qa."""
        try:
            custom_id = record.get("custom_id", "")
            if not custom_id:
                return None, []
            
            qa_id = self._extract_qa_id_from_custom_id(custom_id, is_retry)
            
            if not qa_id:
                logger.warning(f"无法从 custom_id 提取 QA ID: {custom_id}")
                return None, []
            
            response = record.get("response", {})
            body = response.get("body", {})
            choices = body.get("choices", [])
            
            if not choices:
                return qa_id, []
            
            content = choices[0].get("message", {}).get("content", "")
            if not content:
                return qa_id, []
            
            parsed = self._safe_parse_json(content)
            if not parsed:
                return qa_id, []
            
            if not isinstance(parsed, dict):
                logger.warning(f"解析结果不是字典: {type(parsed).__name__}")
                return qa_id, []
            
            memory_facts = parsed.get("memory_facts", [])
            if not memory_facts:
                return qa_id, []
            
            if not isinstance(memory_facts, list):
                logger.warning(f"memory_facts 不是列表: {type(memory_facts).__name__}")
                return qa_id, []
            
            facts = []
            for idx, fact in enumerate(memory_facts):
                if not isinstance(fact, dict):
                    logger.debug(f"跳过非字典类型的 fact: {type(fact).__name__}")
                    continue
                    
                mention = self._create_fact_mention(
                    fact, file_name, custom_id, idx,
                    is_retry=is_retry
                )
                if mention:
                    facts.append(mention)
            
            return qa_id, facts
            
        except Exception as e:
            logger.error(f"提取记录失败: {e}")
            return None, []
    
    def _extract_qa_id_from_custom_id(self, custom_id: str, is_retry: bool = False) -> Optional[str]:
        """Extract qa ID from custom id."""
        if not custom_id:
            return None
        
        working_id = custom_id
        if is_retry and working_id.startswith("retry_"):
            working_id = working_id[6:]
        
        import re
        match = re.match(r'(qa_\d+)', working_id)
        if match:
            return match.group(1)
        
        return working_id
    
    # def _safe_parse_json(self, content: str) -> Optional[Dict]:
    #     """
        
    #     """
    #     if not content or not content.strip():
    #         return None
        
    #     def _ensure_dict(result) -> Optional[Dict]:
    #         if isinstance(result, dict):
    #             return result
    #         elif isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
    #             return result[0]
    #         return None
        
    #     try:
    #         parsed = json.loads(content)
    #         result = _ensure_dict(parsed)
    #         if result is not None:
    #             return result
    #     except json.JSONDecodeError:
    #         pass
    #     except Exception as e:
        
    #     try:
    #         repaired = repair_json(content)
    #             parsed = json.loads(repaired)
    #             result = _ensure_dict(parsed)
    #             if result is not None:
    #                 return result
    #     except Exception as e:
        
    #     json_patterns = [
    #         r'```json\s*(\{.*\})\s*```',
    #         r'```\s*(\{.*\})\s*```',
    #     ]
    #     for pattern in json_patterns:
    #         match = re.search(pattern, content, re.DOTALL)
    #         if match:
    #             json_str = match.group(1)
    #             try:
    #                 parsed = json.loads(json_str)
    #                 result = _ensure_dict(parsed)
    #                 if result is not None:
    #                     return result
    #             except json.JSONDecodeError:
    #                 pass
    #             try:
    #                 repaired = repair_json(json_str)
    #                 if repaired:
    #                     parsed = json.loads(repaired)
    #                     result = _ensure_dict(parsed)
    #                     if result is not None:
    #                         return result
    #             except Exception:
    #                 continue
        
    #     try:
    #         start_idx = content.find('{')
    #         if start_idx != -1:
    #             brace_count = 0
    #             end_idx = -1
    #             for i in range(start_idx, len(content)):
    #                 if content[i] == '{':
    #                     brace_count += 1
    #                 elif content[i] == '}':
    #                     brace_count -= 1
    #                     if brace_count == 0:
    #                         end_idx = i + 1
    #                         break
                
    #             if end_idx != -1:
    #                 json_str = content[start_idx:end_idx]
    #                 try:
    #                     parsed = json.loads(json_str)
    #                     result = _ensure_dict(parsed)
    #                     if result is not None:
    #                         return result
    #                 except json.JSONDecodeError:
    #                     try:
    #                         repaired = repair_json(json_str)
    #                         if repaired:
    #                             parsed = json.loads(repaired)
    #                             result = _ensure_dict(parsed)
    #                             if result is not None:
    #                                 return result
    #                     except Exception:
    #                         pass
    #     except Exception as e:
        
    #     return None
    
    def _safe_parse_json(self, content: str) -> Optional[Dict]:
        """Run safe parse JSON."""
        if not content or not content.strip():
            return None
        
        def _ensure_dict(result) -> Optional[Dict]:
            if isinstance(result, dict):
                return result
            elif isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
                return result[0]
            return None

        
        
        cleaned_content = re.sub(r'^```json\s*', '', content, flags=re.MULTILINE | re.IGNORECASE)
        cleaned_content = re.sub(r'^```\s*', '', cleaned_content, flags=re.MULTILINE)
        cleaned_content = re.sub(r'```\s*$', '', cleaned_content, flags=re.MULTILINE)
        cleaned_content = cleaned_content.strip()

        try:
            parsed = json.loads(cleaned_content)
            result = _ensure_dict(parsed)
            if result: return result
        except Exception:
            pass

        
        
        try:
            parsed = repair_json(cleaned_content)
            parsed = json.loads(parsed)
            result = _ensure_dict(parsed)
            if result: return result
        except Exception:
            pass

        
        
        try:
            start_idx = content.find('{')
            end_idx = content.rfind('}')
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_candidate = content[start_idx : end_idx + 1]
                parsed_str = repair_json(json_candidate)
                parsed = json.loads(parsed_str)
                result = _ensure_dict(parsed)
                if result: return result
        except Exception:
            pass

        return None
    
    def _create_fact_mention(self, fact: Dict, file_name: str, 
                        session_id: str, idx: int,
                        is_retry: bool = False) -> Optional[FactMention]:
        """Create fact mention."""
        try:
            if not isinstance(fact, dict):
                logger.debug(f"fact 不是字典类型: {type(fact).__name__}")
                return None
            
            content = fact.get("content", "")
            if not content:
                return None
            
            category = fact.get("category", "UNKNOWN")
            fact_id = fact.get("fact_id", f"F{idx + 1}")
            
            retry_marker = "_retry" if is_retry else ""
            mention_id = self._generate_mention_id(file_name, session_id, f"{fact_id}{retry_marker}")
            
            temporal = fact.get("temporal", {})
            if temporal is None:
                temporal = {}
            
            attributes = {
                "confidence": fact.get("confidence", 0.95),
                "source": fact.get("source", "unknown"),
                "session_date": fact.get("session_date", ""),
                "qa_relevance": fact.get("qa_relevance", []),
                "is_from_retry": is_retry
            }
            
            if "attributes" in fact and isinstance(fact["attributes"], dict):
                attributes.update(fact["attributes"])
            
            return FactMention(
                mention_id=mention_id,
                source_file=file_name,
                source_session_id=session_id,
                original_fact_id=fact_id,
                content=content,
                category=category,
                temporal=temporal,
                attributes=attributes,
                created_at=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.warning(f"创建 FactMention 失败: {e}")
            return None
    
    
    def optimize_parameters_for_qa(self, mentions: List[FactMention], qa_id: str) -> Dict[str, Dict[str, float]]:
        """Run optimize parameters for qa."""
        if not self.auto_optimize:
            return {}
        
        if len(mentions) < 5:
            logger.info(f"   {qa_id}: 样本不足({len(mentions)}), 使用默认参数")
            return {'_default': {'eps': self.eps, 'min_samples': self.min_samples}}
        
        logger.info(f" {qa_id}: 开始参数优化 ({len(mentions)} 个事实提及)...")
        start_time = datetime.now()
        
        texts = [m.content for m in mentions]
        embeddings = self.encoder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        
        
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized_embeddings = embeddings / norms
        
        optimized_params = {}
        
        if self.optimize_per_category:
            category_indices = defaultdict(list)
            for idx, m in enumerate(mentions):
                category_indices[m.category].append(idx)
            
            for category, indices in category_indices.items():
                if len(indices) < 5:
                    optimized_params[category] = {'eps': self.eps, 'min_samples': self.min_samples}
                    logger.debug(f"     类别 {category}: 样本不足({len(indices)}), 使用默认参数")
                    continue
                
                cat_embeddings = normalized_embeddings[indices]
                
                try:
                    result = optimize_dbscan_parameters(
                        cat_embeddings,
                        eps_range=self.eps_range,
                        min_samples_range=self.min_samples_range,
                        metric='cosine',
                        n_trials=self.n_trials
                    )
                    
                    best_params = result['best_params']
                    best_score = result['best_score']
                    
                    optimized_params[category] = best_params
                    logger.debug(f"     类别 {category}: eps={best_params['eps']:.3f}, "
                            f"min_samples={best_params['min_samples']}, score={best_score:.3f}")
                    
                except Exception as e:
                    logger.warning(f"     类别 {category} 优化失败: {e}, 使用默认参数")
                    optimized_params[category] = {'eps': self.eps, 'min_samples': self.min_samples}
            
            if optimized_params:
                avg_eps = np.mean([p['eps'] for p in optimized_params.values()])
                avg_min_samples = int(np.median([p['min_samples'] for p in optimized_params.values()]))
                optimized_params['_default'] = {'eps': float(avg_eps), 'min_samples': avg_min_samples}
            else:
                optimized_params['_default'] = {'eps': self.eps, 'min_samples': self.min_samples}
        
        else:
            try:
                result = optimize_dbscan_parameters(
                    normalized_embeddings,
                    eps_range=self.eps_range,
                    min_samples_range=self.min_samples_range,
                    metric='cosine',
                    n_trials=self.n_trials
                )
                
                best_params = result['best_params']
                best_score = result['best_score']
                
                optimized_params['_default'] = best_params
                logger.info(f"   {qa_id}: eps={best_params['eps']:.3f}, "
                        f"min_samples={best_params['min_samples']}, score={best_score:.3f}")
                
            except Exception as e:
                logger.warning(f"   {qa_id} 优化失败: {e}, 使用默认参数")
                optimized_params['_default'] = {'eps': self.eps, 'min_samples': self.min_samples}
        
        optimization_time = (datetime.now() - start_time).total_seconds()
        self._update_stats(param_optimizations=1, param_optimization_time=optimization_time)
        
        return optimized_params
    
    
    def cluster_mentions(self, mentions: List[FactMention]) -> Dict[int, List[FactMention]]:
        """Run cluster mentions."""
        if not mentions:
            return {}
        
        logger.info(f" 开始聚类 {len(mentions)} 个事实提及...")
        
        eps, min_samples = self._get_params_for_category('_default')
        logger.info(f"   使用参数: eps={eps:.3f}, min_samples={min_samples}")
        
        texts = [m.content for m in mentions]
        
        
        logger.info("   正在生成嵌入向量...")
        embeddings = self.encoder.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        
        
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized_embeddings = embeddings / norms
        
        logger.info("   正在计算距离矩阵...")
        cosine_sim = np.dot(normalized_embeddings, normalized_embeddings.T)
        distance_matrix = 1 - cosine_sim
        distance_matrix = np.maximum(distance_matrix, 0)
        np.fill_diagonal(distance_matrix, 0)
        
        logger.info(f"   正在执行 DBSCAN 聚类 (eps={eps}, min_samples={min_samples})...")
        clusterer = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric='precomputed'
        )
        labels = clusterer.fit_predict(distance_matrix)
        
        clusters = defaultdict(list)
        for idx, label in enumerate(labels):
            clusters[label].append(mentions[idx])
        
        n_clusters = len([k for k in clusters.keys() if k != -1])
        n_noise = len(clusters.get(-1, []))
        
        logger.info(f" 聚类完成: {n_clusters} 个聚类, {n_noise} 个噪声点")
        self._update_stats(clusters_processed=n_clusters)
        
        return dict(clusters)
    
    def cluster_mentions_by_category(self, mentions: List[FactMention]) -> Dict[int, List[FactMention]]:
        """Run cluster mentions by category."""
        if not mentions:
            return {}
        
        if not self.optimize_per_category:
            return self.cluster_mentions(mentions)
        
        logger.info(f" 按类别聚类 {len(mentions)} 个事实提及...")
        
        category_mentions = defaultdict(list)
        for m in mentions:
            category_mentions[m.category].append(m)
        
        all_clusters = {}
        cluster_id_offset = 0
        
        for category, cat_mentions in category_mentions.items():
            if len(cat_mentions) < 2:
                if -1 not in all_clusters:
                    all_clusters[-1] = []
                all_clusters[-1].extend(cat_mentions)
                continue
            
            eps, min_samples = self._get_params_for_category(category)
            logger.info(f"   处理类别 {category}: {len(cat_mentions)} 个提及 "
                       f"(eps={eps:.3f}, min_samples={min_samples})")
            
            
            texts = [m.content for m in cat_mentions]
            embeddings = self.encoder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            normalized_embeddings = embeddings / norms
            
            cosine_sim = np.dot(normalized_embeddings, normalized_embeddings.T)
            distance_matrix = 1 - cosine_sim
            distance_matrix = np.maximum(distance_matrix, 0)
            np.fill_diagonal(distance_matrix, 0)
            
            clusterer = DBSCAN(
                eps=eps,
                min_samples=min_samples,
                metric='precomputed'
            )
            labels = clusterer.fit_predict(distance_matrix)
            
            for idx, label in enumerate(labels):
                if label == -1:
                    if -1 not in all_clusters:
                        all_clusters[-1] = []
                    all_clusters[-1].append(cat_mentions[idx])
                else:
                    adjusted_label = label + cluster_id_offset
                    if adjusted_label not in all_clusters:
                        all_clusters[adjusted_label] = []
                    all_clusters[adjusted_label].append(cat_mentions[idx])
            
            max_label = max([l for l in labels if l != -1], default=-1)
            cluster_id_offset += max_label + 1
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            logger.info(f"     发现 {n_clusters} 个聚类")
        
        n_total_clusters = len([k for k in all_clusters.keys() if k != -1])
        n_noise = len(all_clusters.get(-1, []))
        
        logger.info(f" 按类别聚类完成: {n_total_clusters} 个聚类, {n_noise} 个噪声点")
        self._update_stats(clusters_processed=n_total_clusters)
        
        return all_clusters
    
    
    def aggregate_clusters(self, clusters: Dict[int, List[FactMention]], qa_id: str = "") -> List[UniqueMemoryFact]:
        """Run aggregate clusters."""
        logger.info(" 开始聚合聚类结果...")
        
        unique_facts = []
        
        for cluster_id, mentions in clusters.items():
            if cluster_id == -1:
                for mention in mentions:
                    fact = self._create_single_fact(mention)
                    unique_facts.append(fact)
            else:
                if (self.use_llm_dedup and 
                    self.llm_client is not None and 
                    len(mentions) >= self.llm_cluster_threshold):
                    
                    if len(mentions) > self.large_cluster_threshold:
                        aggregated = self._process_large_cluster(mentions, cluster_id, qa_id)
                    else:
                        aggregated = self._llm_deduplicate_cluster(mentions, cluster_id, qa_id)
                    unique_facts.extend(aggregated)
                else:
                    aggregated = self._apply_aggregation_rules(mentions, cluster_id)
                    unique_facts.extend(aggregated)
        
        logger.info(f" 聚合完成: {len(unique_facts)} 个唯一事实")
        
        return unique_facts
    
    def _apply_aggregation_rules(self, mentions: List[FactMention], 
                                 cluster_id: int) -> List[UniqueMemoryFact]:
        """Apply aggregation rules."""
        results = []
        
        by_category = defaultdict(list)
        for m in mentions:
            by_category[m.category].append(m)
        
        for category, category_mentions in by_category.items():
            if category == "EPISODIC_EVENT":
                results.extend(self._split_events_by_date(category_mentions))
            elif category in ["USER_ATTRIBUTE", "USER_PREFERENCE", "ATTRIBUTE"]:
                fact = self._merge_attributes(category_mentions)
                results.append(fact)
                self._update_stats(attributes_merged=1)
            else:
                fact = self._merge_general_facts(category_mentions)
                results.append(fact)
        
        return results
    
    def _split_events_by_date(self, mentions: List[FactMention]) -> List[UniqueMemoryFact]:
        """Run split events by date."""
        results = []
        
        by_date = defaultdict(list)
        for m in mentions:
            date_val = self._extract_temporal_value(m.temporal)
            by_date[date_val].append(m)
        
        for date_val, date_mentions in by_date.items():
            fact = self._create_merged_fact(date_mentions, temporal_val=date_val)
            results.append(fact)
            
            if len(by_date) > 1:
                self._update_stats(events_split_by_date=1)
        
        return results
    
    def _merge_attributes(self, mentions: List[FactMention]) -> UniqueMemoryFact:
        """Run merge attributes."""
        return self._create_merged_fact(mentions, temporal_val=None)
    
    def _merge_general_facts(self, mentions: List[FactMention]) -> UniqueMemoryFact:
        """Run merge general facts."""
        temporal_vals = [self._extract_temporal_value(m.temporal) for m in mentions]
        temporal_vals = [t for t in temporal_vals if t]
        
        temporal_val = None
        if temporal_vals:
            temporal_val = Counter(temporal_vals).most_common(1)[0][0]
        
        return self._create_merged_fact(mentions, temporal_val=temporal_val)
    
    def _create_merged_fact(self, mentions: List[FactMention], 
                           temporal_val: Optional[str]) -> UniqueMemoryFact:
        """Create merged fact."""
        canonical_content = self._select_canonical_content(mentions)
        
        category = mentions[0].category
        
        
        uid = self._generate_fact_uid(canonical_content, category, temporal_val)
        
        session_ids = list(set(m.source_session_id for m in mentions))
        
        is_stable = len(session_ids) > 1
        
        confidences = []
        for m in mentions:
            conf = m.attributes.get("confidence", 0.95)
            if isinstance(conf, (int, float)):
                confidences.append(conf)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.95
        
        mention_dicts = [asdict(m) for m in mentions]
        
        return UniqueMemoryFact(
            uid=uid,
            canonical_content=canonical_content,
            category=category,
            mentions=mention_dicts,
            aggregated_count=len(mentions),
            temporal_val=temporal_val,
            is_stable=is_stable,
            confidence=avg_confidence,
            session_ids=session_ids
        )
    
    def _create_single_fact(self, mention: FactMention) -> UniqueMemoryFact:
        """Create single fact."""
        temporal_val = self._extract_temporal_value(mention.temporal)
        uid = self._generate_fact_uid(mention.content, mention.category, temporal_val)
        
        confidence = mention.attributes.get("confidence", 0.95)
        if not isinstance(confidence, (int, float)):
            confidence = 0.95
        
        return UniqueMemoryFact(
            uid=uid,
            canonical_content=mention.content,
            category=mention.category,
            mentions=[asdict(mention)],
            aggregated_count=1,
            temporal_val=temporal_val,
            is_stable=False,
            confidence=confidence,
            session_ids=[mention.source_session_id]
        )
    
    def _select_canonical_content(self, mentions: List[FactMention]) -> str:
        """Run select canonical content."""
        if not mentions:
            return ""
        
        if len(mentions) == 1:
            return mentions[0].content
        
        
        content_counts = Counter(m.content for m in mentions)
        most_common = content_counts.most_common(1)[0]
        
        if most_common[1] > 1:
            return most_common[0]
        
        return max(mentions, key=lambda m: len(m.content)).content
    

    def process_single_qa(self, qa_id: str, mentions: List[FactMention]) -> List[UniqueMemoryFact]:
        """Process single qa."""
        logger.info(f" 处理 {qa_id}: {len(mentions)} 个事实提及")
        
        if not mentions:
            return []
        
        if self.auto_optimize:
            self.optimize_parameters_for_qa(mentions, qa_id)
        
        if self.optimize_per_category:
            clusters = self.cluster_mentions_by_category(mentions)
        else:
            clusters = self.cluster_mentions(mentions)
        
        unique_facts = self.aggregate_clusters(clusters, qa_id=qa_id)
        
        logger.info(f" {qa_id} 处理完成: {len(mentions)} -> {len(unique_facts)} 个唯一事实")
        
        return unique_facts
    
    
    # def process(self, input_dir: str, output_dir: str) -> Dict[str, Any]:
    #     """
        
    #     Args:
        
    #     Returns:
    #     """
    #     start_time = datetime.now()
        
    #     logger.info("=" * 80)
    #     logger.info("=" * 80)
    #     if self.auto_optimize:
        
    
    #     logger.info("\n" + "=" * 40)
    
    #     logger.info("=" * 40)
    #     qa_mentions = self.load_and_flatten_by_qa(input_dir)
        
    #     if not qa_mentions:
    #         return self.stats
        
    #     output_path = Path(output_dir)
    #     output_path.mkdir(parents=True, exist_ok=True)
        
    #     logger.info("\n" + "=" * 40)
    #     logger.info("=" * 40)
        
    #     all_results = {}
    #     total_unique_facts = 0
    #     failed_qa_ids = []
        
    
    #     sorted_qa_ids = sorted(qa_mentions.keys(), key=lambda x: int(x.split('_')[1]) if x.split('_')[1].isdigit() else 0)
        
    #     for idx, qa_id in enumerate(sorted_qa_ids, 1):
    #         mentions = qa_mentions[qa_id]
            
    #         try:
                
    #             unique_facts = self.process_single_qa(qa_id, mentions)
                
    
    #             self._save_qa_result(qa_id, unique_facts, output_path)
                
    #             all_results[qa_id] = unique_facts
    #             total_unique_facts += len(unique_facts)
                
    #             self._update_stats(qa_processed=1)
                
    #         except Exception as e:
    #             failed_qa_ids.append(qa_id)
    #             import traceback
    #             logger.debug(traceback.format_exc())
    #             continue
        
    
    #     logger.info("\n" + "=" * 40)
    
    #     logger.info("=" * 40)
    #     self._save_summary(all_results, output_path, failed_qa_ids)
        
    #     processing_time = (datetime.now() - start_time).total_seconds()
    #     self.stats["processing_time"] = processing_time
    #     self.stats["total_unique_facts"] = total_unique_facts
    #     self.stats["failed_qa_count"] = len(failed_qa_ids)
        
    #     self._print_summary()
        
    #     return self.stats
    
    def process(self, input_dir: str, output_dir: str) -> Dict[str, Any]:
        """Process."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        start_time = datetime.now()
        
        logger.info("=" * 80)
        logger.info(f" 开始事实去重处理（并行数: {self.workers}）")
        logger.info("=" * 80)
        logger.info(f"   输入目录: {input_dir}")
        logger.info(f"   输出目录: {output_dir}")
        
        
        logger.info("\n" + "=" * 40)
        logger.info(" Step A: 加载与扁平化（按QA分组）")
        logger.info("=" * 40)
        qa_mentions = self.load_and_flatten_by_qa(input_dir)
        
        if not qa_mentions:
            logger.error(" 未找到任何事实提及，退出处理")
            return self.stats
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("\n" + "=" * 40)
        logger.info(" Step B/C/D: 并行执行参数优化、聚类和聚合")
        logger.info("=" * 40)
        
        all_results = {}
        total_unique_facts = 0
        failed_qa_ids = []
        
        sorted_qa_ids = sorted(qa_mentions.keys(), key=lambda x: int(x.split('_')[1]) if x.split('_')[1].isdigit() else 0)
        total_tasks = len(sorted_qa_ids)

        def process_task(qa_id):
            mentions = qa_mentions[qa_id]
            try:
                unique_facts = self.process_single_qa(qa_id, mentions)
                
                self._save_qa_result(qa_id, unique_facts, output_path)
                return qa_id, unique_facts, None
            except Exception as e:
                return qa_id, None, e

        logger.info(f" 启动线程池，工作线程数: {self.workers}")
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            future_to_qa = {executor.submit(process_task, q_id): q_id for q_id in sorted_qa_ids}
            
            for i, future in enumerate(as_completed(future_to_qa), 1):
                qa_id, unique_facts, error = future.result()
                
                if error:
                    logger.error(f" [{i}/{total_tasks}] {qa_id} 处理失败: {error}")
                    failed_qa_ids.append(qa_id)
                    # logger.error(traceback.format_exc()) 
                else:
                    logger.info(f" [{i}/{total_tasks}] 完成 {qa_id} (提取 {len(unique_facts)} 个事实)")
                    all_results[qa_id] = unique_facts
                    total_unique_facts += len(unique_facts)
                    self._update_stats(qa_processed=1)

        
        logger.info("\n" + "=" * 40)
        logger.info(" 保存汇总结果")
        logger.info("=" * 40)
        self._save_summary(all_results, output_path, failed_qa_ids)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        self.stats["processing_time"] = processing_time
        self.stats["total_unique_facts"] = total_unique_facts
        self.stats["failed_qa_count"] = len(failed_qa_ids)
        
        self._print_summary()
        
        return self.stats
    
    def _process_large_cluster(self, 
                            cluster_mentions: List[FactMention], 
                            cluster_id: int,
                            qa_id: str) -> List[UniqueMemoryFact]:
        """Process large cluster."""
        batch_size = self.large_cluster_threshold
        batches = []
        
        for i in range(0, len(cluster_mentions), batch_size):
            batch = cluster_mentions[i:i + batch_size]
            batch_id = f"{cluster_id}_batch_{i // batch_size}"
            batches.append((batch, batch_id))
        
        logger.debug(f" 大聚类 {cluster_id} 分为 {len(batches)} 个批次处理")
        
        all_merged = []
        
        for batch, batch_id in batches:
            try:
                batch_merged = self._llm_deduplicate_cluster(batch, int(batch_id.split('_')[0]), qa_id)
                all_merged.extend(batch_merged)
            except Exception as e:
                logger.error(f"批次 {batch_id} LLM去重失败: {e}, 使用规则合并")
                merged = self._apply_aggregation_rules(batch, int(batch_id.split('_')[0]))
                all_merged.extend(merged)
        
        logger.debug(f" 大聚类 {cluster_id} 处理完成: {len(cluster_mentions)} -> {len(all_merged)}")
        return all_merged
    
    def _create_fact_from_llm(self, 
                            merged_data: Dict[str, Any], 
                            source_mentions: List[FactMention], 
                            fact_id: str) -> UniqueMemoryFact:
        """Create fact from LLM."""
        canonical_content = merged_data.get("canonical_content", "")
        category = merged_data.get("category", source_mentions[0].category if source_mentions else "UNKNOWN")
        temporal_val = merged_data.get("temporal_val")
        confidence = float(merged_data.get("confidence", 0.95))
        
        if not canonical_content and source_mentions:
            canonical_content = max(source_mentions, key=lambda m: len(m.content)).content
        
        mention_dicts = []
        if 'mentions' in merged_data and merged_data['mentions']:
            for llm_mention in merged_data['mentions']:
                source_session_id = llm_mention.get('source_session_id', '')
                matching_source = None
                for src in source_mentions:
                    if src.source_session_id == source_session_id:
                        matching_source = src
                        break
                
                if matching_source:
                    mention_dicts.append(asdict(matching_source))
                else:
                    mention_dicts.append({
                        "mention_id": f"{fact_id}_{len(mention_dicts)}",
                        "source_file": "",
                        "source_session_id": source_session_id,
                        "original_fact_id": "",
                        "content": llm_mention.get('original_content', canonical_content),
                        "category": category,
                        "temporal": llm_mention.get('temporal', {}),
                        "attributes": {"confidence": llm_mention.get('confidence', confidence)},
                        "created_at": datetime.now().isoformat()
                    })
        else:
            for src in source_mentions:
                mention_dicts.append(asdict(src))
        
        session_ids = list(set(
            m.get('source_session_id', '') for m in mention_dicts if m.get('source_session_id')
        ))
        if not session_ids:
            session_ids = list(set(src.source_session_id for src in source_mentions))
        
        is_stable = len(session_ids) > 1
        
        
        uid = self._generate_fact_uid(canonical_content, category, temporal_val)
        
        return UniqueMemoryFact(
            uid=uid,
            canonical_content=canonical_content,
            category=category,
            mentions=mention_dicts,
            aggregated_count=len(mention_dicts),
            temporal_val=temporal_val,
            is_stable=is_stable,
            confidence=confidence,
            session_ids=session_ids
        )
    
    def _save_qa_result(self, qa_id: str, unique_facts: List[UniqueMemoryFact], output_path: Path):
        """Save qa result."""
        output_file = output_path / f"{qa_id}_deduplicated.json"
        
        data = {
            "qa_id": qa_id,
            "total_facts": len(unique_facts),
            "facts": [fact.to_dict() for fact in unique_facts],
            "created_at": datetime.now().isoformat()
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.debug(f"   保存: {output_file}")
    
    def _save_summary(self, all_results: Dict[str, List[UniqueMemoryFact]], 
                output_path: Path, failed_qa_ids: List[str]):
        """Save summary."""
        total_unique_facts = sum(len(facts) for facts in all_results.values())
        
        category_counts = defaultdict(int)
        stable_count = 0
        retry_source_count = 0
        
        for qa_id, facts in all_results.items():
            for fact in facts:
                category_counts[fact.category] += 1
                if fact.is_stable:
                    stable_count += 1
                for mention in fact.mentions:
                    if mention.get('attributes', {}).get('is_from_retry', False):
                        retry_source_count += 1
                        break
        
        summary = {
            "generated_at": datetime.now().isoformat(),
            "total_qa_count": len(all_results),
            "total_unique_facts": total_unique_facts,
            "failed_qa_count": len(failed_qa_ids),
            "failed_qa_ids": failed_qa_ids,
            "facts_with_retry_source": retry_source_count,
            "category_distribution": dict(category_counts),
            "stable_facts_count": stable_count,
            "stats": self.stats,
            "config": {
                "eps": self.eps,
                "min_samples": self.min_samples,
                "auto_optimize": self.auto_optimize,
                "optimize_per_category": self.optimize_per_category,
                "use_llm_dedup": self.use_llm_dedup,
                "llm_cluster_threshold": self.llm_cluster_threshold if self.use_llm_dedup else None,
                "large_cluster_threshold": self.large_cluster_threshold if self.use_llm_dedup else None
            },
            "qa_summary": {
                qa_id: {
                    "unique_facts": len(facts),
                    "categories": dict(Counter(f.category for f in facts)),
                    "stable_facts": sum(1 for f in facts if f.is_stable),
                    "has_retry_source": any(
                        m.get('attributes', {}).get('is_from_retry', False)
                        for f in facts for m in f.mentions
                    )
                }
                for qa_id, facts in all_results.items()
            }
        }
        
        summary_file = output_path / "deduplication_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f" 保存汇总信息: {summary_file}")
        logger.info(f"   - 包含retry来源的facts: {retry_source_count}")


    def _print_summary(self):
        """Run print summary."""
        logger.info("\n" + "=" * 60)
        logger.info(" 处理摘要")
        logger.info("=" * 60)
        logger.info(f"   文件加载数: {self.stats['files_loaded']}")
        logger.info(f"   总提及数: {self.stats['total_mentions']}")
        logger.info(f"   唯一事实数: {self.stats['total_unique_facts']}")
        logger.info(f"   聚类处理数: {self.stats['clusters_processed']}")
        logger.info(f"   事件按日期分裂数: {self.stats['events_split_by_date']}")
        logger.info(f"   属性合并数: {self.stats['attributes_merged']}")
        logger.info(f"   QA处理数: {self.stats['qa_processed']}")
        logger.info(f"   失败QA数: {self.stats['failed_qa_count']}")
        if self.auto_optimize:
            logger.info(f"   参数优化次数: {self.stats['param_optimizations']}")
            logger.info(f"   参数优化耗时: {self.stats['param_optimization_time']:.2f}秒")
        if self.use_llm_dedup:
            logger.info(f"   LLM去重调用数: {self.stats['llm_merge_calls']}")
            logger.info(f"   LLM去重耗时: {self.stats['llm_merge_time']:.2f}秒")
        logger.info(f"   总处理耗时: {self.stats['processing_time']:.2f}秒")
        logger.info("=" * 60)



def main():
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(
        description="LongMemEval 情景记忆事实去重器 - Step 3 (支持自动参数优化和LLM精细去重)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        示例:
            # 使用默认参数运行（默认启用自动优化、按类别优化和LLM去重）
            python step3_deduplication.py

            # 禁用LLM精细去重（仅使用规则合并）
            python step3_deduplication.py --no-llm-dedup

            # 禁用自动参数优化（使用手动参数）
            python step3_deduplication.py --no-auto-optimize

            # 禁用按类别优化（使用全局优化参数）
            python step3_deduplication.py --no-optimize-per-category

            # 指定输入输出目录
            python step3_deduplication.py --input_dir ./my_batch_results --output_dir ./my_output

            # 调整 DBSCAN 参数（手动模式，需配合 --no-auto-optimize）
            python step3_deduplication.py --no-auto-optimize --eps 0.10 --min_samples 2

            # 使用自定义LLM服务
            python step3_deduplication.py --llm-model deepseek-reasoner --llm-base-url https://api.deepseek.com/v1

            # 调整LLM去重阈值
            python step3_deduplication.py --llm-cluster-threshold 3 --large-cluster-threshold 20

            # 使用不同的嵌入模型
            python step3_deduplication.py --model_name paraphrase-MiniLM-L6-v2
        """
    )
    
    parser.add_argument(
        "--input_dir", 
        type=str,
        default=str(paths.LONGMEMEVAL_EPISODIC_NEW_BATCH_RESULTS_DIR),
        help="输入目录，包含 LLM 批量推理结果 (默认: benchmark_longmemeval/dataset_maker/longmemeval_episodic_memory_new/batch_results)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(paths.LONGMEMEVAL_EPISODIC_NEW_DEDUPLICATED_DIR),
        help="输出目录 (默认: benchmark_longmemeval/dataset_maker/longmemeval_episodic_memory_new/deduplicated_results)"
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer 嵌入模型名称 (默认: all-MiniLM-L6-v2)"
    )
    
    parser.add_argument(
        "--eps",
        type=float,
        default=0.15,
        help="DBSCAN epsilon（邻域半径），越小越严格 (默认: 0.15，仅在禁用自动优化时有效)"
    )
    
    parser.add_argument(
        "--min_samples",
        type=int,
        default=1,
        help="DBSCAN min_samples（最小样本数）(默认: 1，仅在禁用自动优化时有效)"
    )
    
    parser.add_argument(
        "--no-auto-optimize",
        action="store_true",
        help="禁用自动 DBSCAN 参数优化（默认启用自动优化）"
    )
    
    parser.add_argument(
        "--no-optimize-per-category",
        action="store_true",
        help="禁用按类别单独优化参数（默认启用按类别优化）"
    )
    
    parser.add_argument(
        "--eps-min",
        type=float,
        default=0.1,
        help="eps参数搜索下限 (默认: 0.1)"
    )
    
    parser.add_argument(
        "--eps-max",
        type=float,
        default=0.6,
        help="eps参数搜索上限 (默认: 0.6)"
    )
    
    parser.add_argument(
        "--min-samples-min",
        type=int,
        default=1,
        help="min_samples参数搜索下限 (默认: 1)"
    )
    
    parser.add_argument(
        "--min-samples-max",
        type=int,
        default=5,
        help="min_samples参数搜索上限 (默认: 5)"
    )
    
    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="参数优化尝试次数 (默认: 20)"
    )
    
    parser.add_argument(
        "--no-llm-dedup",
        action="store_true",
        help="禁用LLM精细去重（默认启用LLM去重）"
    )
    
    parser.add_argument(
        "--dedup-model",
        type=str,
        default="deepseek-v3.2-dashscope",
        help="去重模型名称 (默认: deepseek-v3.2-dashscope)"
    )
    
    parser.add_argument(
        "--llm-api-key",
        type=str,
        default=None,
        help="LLM API密钥 (默认从环境变量读取)"
    )
    
    parser.add_argument(
        "--llm-base-url",
        type=str,
        default=None,
        help="LLM API基础URL"
    )
    
    parser.add_argument(
        "--llm-cluster-threshold",
        type=int,
        default=2,
        help="触发LLM去重的聚类大小阈值 (默认: 2，即所有>=2的聚类都用LLM)"
    )
    
    parser.add_argument(
        "--large-cluster-threshold",
        type=int,
        default=15,
        help="大聚类阈值，需要分批处理 (默认: 15)"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=40,
        help="并行工作线程数 (默认: 40)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="启用调试模式"
    )
    
    args = parser.parse_args()
    
    # Avoid mutating LogRecord fields before other handlers process the record.
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    auto_optimize = not args.no_auto_optimize
    optimize_per_category = not args.no_optimize_per_category
    use_llm_dedup = not args.no_llm_dedup
    
    if not auto_optimize:
        optimize_per_category = False
    
    try:
        logger.info("\n" + "=" * 80)
        logger.info(" 配置信息")
        logger.info("=" * 80)
        logger.info(f"   输入目录: {args.input_dir}")
        logger.info(f"   输出目录: {args.output_dir}")
        logger.info(f"   嵌入模型: {args.model_name}")
        logger.info(f"   自动参数优化: {'启用' if auto_optimize else '禁用'}")
        if auto_optimize:
            logger.info(f"   按类别优化: {'启用' if optimize_per_category else '禁用'}")
            logger.info(f"   eps范围: [{args.eps_min}, {args.eps_max}]")
            logger.info(f"   min_samples范围: [{args.min_samples_min}, {args.min_samples_max}]")
            logger.info(f"   优化尝试次数: {args.n_trials}")
        else:
            logger.info(f"   手动eps: {args.eps}")
            logger.info(f"   手动min_samples: {args.min_samples}")
        logger.info(f"   LLM精细去重: {'启用' if use_llm_dedup else '禁用'}")
        if use_llm_dedup:
            logger.info(f"   LLM模型: {args.dedup_model}")
            logger.info(f"   LLM聚类阈值: {args.llm_cluster_threshold}")
            logger.info(f"   大聚类阈值: {args.large_cluster_threshold}")
        logger.info(f"   并行线程数: {args.workers}")
        logger.info("=" * 80 + "\n")
        
        deduplicator = FactDeduplicator(
            model_name=args.model_name,
            eps=args.eps,
            min_samples=args.min_samples,
            workers=args.workers,
            auto_optimize=auto_optimize,
            eps_range=(args.eps_min, args.eps_max),
            min_samples_range=(args.min_samples_min, args.min_samples_max),
            n_trials=args.n_trials,
            optimize_per_category=optimize_per_category,
            use_llm_dedup=use_llm_dedup,
            llm_model=args.dedup_model,
            llm_api_key=args.llm_api_key,
            llm_base_url=args.llm_base_url,
            llm_cluster_threshold=args.llm_cluster_threshold,
            large_cluster_threshold=args.large_cluster_threshold
        )
        
        results = deduplicator.process(args.input_dir, args.output_dir)
        
        logger.info("\n" + "=" * 80)
        logger.info(" 最终统计")
        logger.info("=" * 80)
        for key, value in results.items():
            if isinstance(value, float):
                logger.info(f"   {key}: {value:.2f}")
            else:
                logger.info(f"   {key}: {value}")
        logger.info("=" * 80)
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("\n 用户中断处理")
        return 1
        
    except Exception as e:
        logger.error(f"\n 处理失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())