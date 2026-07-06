"""Utilities for step2 deduplicate and enhance."""

import json
import logging
import os
import sys
import re
import argparse
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict, field
from datetime import datetime
from collections import defaultdict
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from sentence_transformers import SentenceTransformer


from mandol.llm.llm_client import LLMClient
from mandol.core.memory_unit import MemoryUnit
from mandol.core import paths

try:
    from mandol.cluster.dbscan_method import find_clusters_with_dbscan, optimize_dbscan_parameters
    DBSCAN_AVAILABLE = True
except ImportError:
    DBSCAN_AVAILABLE = False
    logging.warning("DBSCAN模块不可用，将使用简单相似度聚类")

try:
    from json_repair import repair_json
    JSON_REPAIR_AVAILABLE = True
except ImportError:
    JSON_REPAIR_AVAILABLE = False



@dataclass
class DeduplicateConfig:
    input_dir: str = str(paths.LOCOMO_EPISODIC_STEP1_DIR)
    output_dir: str = str(paths.LOCOMO_EPISODIC_STEP2_DIR)
    
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    auto_optimize_dbscan: bool = True
    eps_range: Tuple[float, float] = (0.15, 0.5)
    min_samples_range: Tuple[int, int] = (2, 5)  # Dataset-specific handling used by the reproduction workflow.
    default_eps: float = 0.25
    default_min_samples: int = 2  # Dataset-specific handling used by the reproduction workflow.
    
    llm_model: str = "deepseek-v3.2-dashscope"
    enable_llm_merge: bool = True
    llm_cluster_threshold: int = 2
    large_cluster_threshold: int = 15
    
    enable_accumulation: bool = True
    enable_timeline: bool = True
    
    max_workers: int = 8
    
    debug_mode: bool = False



@dataclass
class FactMention:
    mention_id: str
    content: str
    fact_type: str
    participants: List[str]
    time: Dict
    location: str
    details: Dict
    source_session_id: str
    source_turns: List[str]
    retrieval_keys: List[str]
    confidence: float
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class MergedFact:
    fact_id: str
    content: str
    fact_type: str
    participants: List[str]
    time: Dict
    location: str
    details: Dict
    
    mentions: List[Dict] = field(default_factory=list)
    source_sessions: List[str] = field(default_factory=list)
    source_turns: List[str] = field(default_factory=list)
    retrieval_keys: List[str] = field(default_factory=list)
    
    merge_mode: str = "SINGLE"                # SINGLE, INFO_MERGE, FREQUENCY_COUNT, STATE_EVOLUTION
    merge_count: Optional[int] = None
    date_list: Optional[List[str]] = None
    state_evolution: Optional[Dict] = None
    
    confidence: float = 1.0
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class AccumulatedFact:
    accumulation_id: str
    description: str
    fact_type: str = "ACCUMULATED"
    
    count: int = 0
    subject: str = ""
    action: str = ""
    
    component_fact_ids: List[str] = field(default_factory=list)
    time_span_start: str = ""
    time_span_end: str = ""
    retrieval_keys: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Timeline:
    timeline_id: str
    subject: str
    description: str
    
    ordered_fact_ids: List[str] = field(default_factory=list)
    state_changes: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return asdict(self)



LLM_DEDUP_PROMPT = """You are an expert in memory fact deduplication and fusion. The following are {cluster_size} semantically similar facts from a conversation memory system. Please analyze and merge them intelligently.

## Cluster {cluster_id} - Fact List:
{fact_candidates}

## Your Task:
For each group of similar facts, choose ONE fusion mode:

**Mode A - Information Merge**: Facts describe the SAME event with COMPLEMENTARY details.
  - Combine all details into one comprehensive fact
  - Example: "Jon got a job" + "to pay rent" → "Jon got a job to pay rent"

**Mode B - Frequency Count**: Facts describe the SAME repeated action on DIFFERENT dates.
  - Create one fact with count and date list
  - Example: 3x "Jon went to gym" → "Jon went to gym 3 times (dates: 2023-05-01, 2023-05-08, 2023-05-15)"

**Mode C - State Evolution**: Facts show a STATUS CHANGE over time.
  - Capture the state transition with timeline
  - Example: "Jon is unemployed" + "Jon got a job" → "Jon transitioned from unemployed to employed"

## Output (JSON):
{{
    "merged_facts": [
        {{
            "canonical_content": "The most complete and accurate description",
            "fact_type": "EVENT | STATE_CHANGE | ACTIVITY | OPINION | PREFERENCE | ACCUMULATED",
            "merge_mode": "A" | "B" | "C",
            "merge_count": null | number,
            "date_list": null | ["date1", "date2", ...],
            "state_evolution": null | {{"from": "state1", "to": "state2", "time_from": "...", "time_to": "..."}},
            "confidence": 0.95,
            "source_fact_indices": [1, 2, ...],
            "merge_reasoning": "Brief explanation"
        }}
    ]
}}

## Critical Rules:
1. **Temporal Sensitivity**: Facts on DIFFERENT dates are usually DIFFERENT facts (unless Mode B applies)
2. **Preserve Details**: Never lose important information when merging
3. **Session Traceability**: Track which sessions each merged fact came from
4. **Conservative Merging**: When in doubt, keep facts separate
5. **Participant Consistency**: Only merge facts about the SAME participants

## Output Requirements:
- Every input fact MUST be represented in at least one merged_fact
- source_fact_indices should reference the fact numbers (1-indexed) from input
- canonical_content should be the most informative version
"""



class FactDeduplicator:
    
    def __init__(self, config: DeduplicateConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        self.logger.info(f"加载嵌入模型: {config.embedding_model}")
        self.encoder = SentenceTransformer(config.embedding_model)
        
        if config.enable_llm_merge:
            self.llm = LLMClient(model_name=config.llm_model)
            self.logger.info(f"LLM客户端已初始化: {config.llm_model}")
        else:
            self.llm = None
        
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.stats_lock = Lock()
        self.stats = {
            'samples_processed': 0,
            'facts_before': 0,
            'facts_after': 0,
            'facts_merged': 0,
            'clusters_total': 0,
            'clusters_llm_processed': 0,
            'llm_calls': 0,
            'accumulations_created': 0,
            'timelines_created': 0
        }
    
    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger(f"{__name__}.FactDeduplicator")
        logger.setLevel(logging.DEBUG if self.config.debug_mode else logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _update_stats(self, **kwargs):
        """Run update stats."""
        with self.stats_lock:
            for key, value in kwargs.items():
                if key in self.stats:
                    self.stats[key] += value
    
    def _generate_fact_uid(self, content: str, fact_type: str, time_val: Optional[str]) -> str:
        """Generate fact UID."""
        hash_input = f"{content}_{fact_type}_{time_val or 'no_time'}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]
    
    
    def _prepare_facts_for_clustering(self, facts: List[Dict]) -> Tuple[List, List[Dict]]:
        """Run prepare facts for clustering."""
        try:
            
            units = []
            valid_facts = []
            
            for fact in facts:
                content = fact.get('content', '')
                if not content:
                    continue
                
                embedding = self.encoder.encode(content)
                
                unit = MemoryUnit(
                    uid=fact.get('fact_id', f"fact_{len(units)}"),
                    raw_data={
                        'content': content,
                        'fact_type': fact.get('fact_type', 'EVENT'),
                        'participants': fact.get('participants', []),
                        'time': fact.get('time', {}),
                        'location': fact.get('location', ''),
                    },
                    metadata={
                        'source_session_id': fact.get('source_session_id', ''),
                        'source_turns': fact.get('source_turns', []),
                    }
                )
                unit.embedding = embedding.tolist()
                
                units.append(unit)
                valid_facts.append(fact)
            
            return units, valid_facts
            
        except Exception as e:
            self.logger.warning(f"准备聚类数据失败: {e}，使用简化方法")
            return [], facts
    
    def _cluster_facts_dbscan(self, facts: List[Dict]) -> Dict[int, List[Dict]]:
        """Run cluster facts dbscan."""
        if not facts:
            return {}
        
        if not DBSCAN_AVAILABLE:
            self.logger.warning("DBSCAN不可用，使用简单相似度聚类")
            return self._cluster_facts_simple(facts)
        
        units, valid_facts = self._prepare_facts_for_clustering(facts)
        
        if len(units) < 2:
            return {i: [f] for i, f in enumerate(facts)}
        
        if self.config.auto_optimize_dbscan and len(units) >= 10:
            self.logger.info(" 自动优化DBSCAN参数...")
            try:
                optimization_result = optimize_dbscan_parameters(
                    units,
                    eps_range=self.config.eps_range,
                    min_samples_range=self.config.min_samples_range,
                    metric='cosine',
                    n_trials=15
                )
                best_params = optimization_result.get('best_params', {})
                eps = best_params.get('eps', self.config.default_eps)
                min_samples = best_params.get('min_samples', self.config.default_min_samples)
                self.logger.info(f" 最优参数: eps={eps:.3f}, min_samples={min_samples}")
            except Exception as e:
                self.logger.warning(f"参数优化失败: {e}，使用默认参数")
                eps = self.config.default_eps
                min_samples = self.config.default_min_samples
        else:
            eps = self.config.default_eps
            min_samples = self.config.default_min_samples
        
        try:
            clusters_by_uid = find_clusters_with_dbscan(
                units,
                eps=eps,
                min_samples=min_samples,
                metric='cosine',
                normalize_embeddings=True,
                use_metadata_features=False
            )
            
            uid_to_fact = {f.get('fact_id'): f for f in valid_facts}
            clusters = {}
            
            for cluster_id, uids in clusters_by_uid.items():
                cluster_facts = [uid_to_fact[uid] for uid in uids if uid in uid_to_fact]
                if cluster_facts:
                    clusters[cluster_id] = cluster_facts
            
            self.logger.info(f" DBSCAN聚类完成: {len(clusters)} 个聚类")
            return clusters
            
        except Exception as e:
            self.logger.error(f"DBSCAN聚类失败: {e}，回退到简单聚类")
            return self._cluster_facts_simple(facts)
    
    def _cluster_facts_simple(self, facts: List[Dict]) -> Dict[int, List[Dict]]:
        """Run cluster facts simple."""
        if not facts:
            return {}
        
        contents = [f.get('content', '') for f in facts]
        embeddings = self.encoder.encode(contents)
        
        clusters = {}
        used = set()
        cluster_id = 0
        
        similarity_threshold = 0.85
        
        for i, fact_i in enumerate(facts):
            if i in used:
                continue
            
            cluster = [fact_i]
            used.add(i)
            
            for j, fact_j in enumerate(facts):
                if j in used or j <= i:
                    continue
                
                similarity = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                
                if similarity >= similarity_threshold:
                    cluster.append(fact_j)
                    used.add(j)
            
            clusters[cluster_id] = cluster
            cluster_id += 1
        
        return clusters
    
    
    def _llm_deduplicate_cluster(self, cluster_facts: List[Dict], cluster_id: int) -> List[MergedFact]:
        """Run LLM deduplicate cluster."""
        if not self.llm or not cluster_facts:
            return self._fallback_merge(cluster_facts, cluster_id)
        
        try:
            fact_candidates_text = ""
            for i, fact in enumerate(cluster_facts, 1):
                time_info = fact.get('time', {})
                time_str = time_info.get('absolute_start', 'unknown') if isinstance(time_info, dict) else 'unknown'
                
                fact_candidates_text += f"{i}. Content: \"{fact.get('content', '')}\"\n"
                fact_candidates_text += f"   Type: {fact.get('fact_type', 'EVENT')}\n"
                fact_candidates_text += f"   Time: {time_str}\n"
                fact_candidates_text += f"   Participants: {fact.get('participants', [])}\n"
                fact_candidates_text += f"   Session: {fact.get('source_session_id', 'unknown')}\n"
                fact_candidates_text += f"   Location: {fact.get('location', '')}\n\n"
            
            prompt = LLM_DEDUP_PROMPT.format(
                cluster_size=len(cluster_facts),
                cluster_id=cluster_id,
                fact_candidates=fact_candidates_text
            )
            
            response = self.llm.generate_answer(
                prompt=prompt,
                temperature=0.1,
                json_format=True
            )
            
            self._update_stats(llm_calls=1)
            
            result = self._parse_llm_response(response, cluster_id)
            
            if result and 'merged_facts' in result:
                merged_facts = []
                for i, merged in enumerate(result['merged_facts']):
                    fact = self._create_merged_fact(merged, cluster_facts, cluster_id, i)
                    merged_facts.append(fact)
                
                self.logger.debug(f" 聚类 {cluster_id} LLM去重: {len(cluster_facts)} -> {len(merged_facts)}")
                return merged_facts
            else:
                return self._fallback_merge(cluster_facts, cluster_id)
                
        except Exception as e:
            self.logger.warning(f" 聚类 {cluster_id} LLM去重失败: {e}")
            return self._fallback_merge(cluster_facts, cluster_id)
    
    def _parse_llm_response(self, response: str, cluster_id: int) -> Optional[Dict]:
        """Parse LLM response."""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        if JSON_REPAIR_AVAILABLE:
            try:
                repaired = repair_json(response)
                return json.loads(repaired)
            except Exception:
                pass
        
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                json_str = json_match.group()
                try:
                    return json.loads(json_str)
                except:
                    if JSON_REPAIR_AVAILABLE:
                        repaired = repair_json(json_str)
                        return json.loads(repaired)
        except Exception:
            pass
        
        self.logger.warning(f"聚类 {cluster_id} JSON解析失败")
        return None
    
    def _create_merged_fact(self, merged_data: Dict, source_facts: List[Dict], 
                           cluster_id: int, fact_index: int) -> MergedFact:
        """Create merged fact."""
        
        source_indices = merged_data.get('source_fact_indices', list(range(1, len(source_facts) + 1)))
        selected_facts = [source_facts[i-1] for i in source_indices if 0 < i <= len(source_facts)]
        
        if not selected_facts:
            selected_facts = source_facts
        
        all_participants = set()
        all_sessions = set()
        all_turns = []
        all_keys = set()
        
        for f in selected_facts:
            all_participants.update(f.get('participants', []))
            if f.get('source_session_id'):
                all_sessions.add(f['source_session_id'])
            all_turns.extend(f.get('source_turns', []))
            all_keys.update(f.get('retrieval_keys', []))
        
        mentions = []
        for f in selected_facts:
            mentions.append({
                'original_content': f.get('content', ''),
                'source_session_id': f.get('source_session_id', ''),
                'source_turns': f.get('source_turns', []),
                'fact_id': f.get('fact_id', '')
            })
        
        merge_mode = merged_data.get('merge_mode', 'A')
        mode_map = {'A': 'INFO_MERGE', 'B': 'FREQUENCY_COUNT', 'C': 'STATE_EVOLUTION'}
        
        time_info = selected_facts[0].get('time', {}) if selected_facts else {}
        
        canonical_content = merged_data.get('canonical_content', selected_facts[0].get('content', ''))
        
        return MergedFact(
            fact_id=f"merged_{cluster_id}_{fact_index}",
            content=canonical_content,
            fact_type=merged_data.get('fact_type', selected_facts[0].get('fact_type', 'EVENT')),
            participants=list(all_participants),
            time=time_info,
            location=selected_facts[0].get('location', ''),
            details=selected_facts[0].get('details', {}),
            mentions=mentions,
            source_sessions=list(all_sessions),
            source_turns=list(set(all_turns)),
            retrieval_keys=list(all_keys),
            merge_mode=mode_map.get(merge_mode, 'INFO_MERGE'),
            merge_count=merged_data.get('merge_count'),
            date_list=merged_data.get('date_list'),
            state_evolution=merged_data.get('state_evolution'),
            confidence=merged_data.get('confidence', 0.95)
        )
    
    def _fallback_merge(self, cluster_facts: List[Dict], cluster_id: int) -> List[MergedFact]:
        """Run fallback merge."""
        if not cluster_facts:
            return []
        
        base = max(cluster_facts, key=lambda f: len(f.get('content', '')))
        
        all_participants = set()
        all_sessions = set()
        all_turns = []
        all_keys = set()
        mentions = []
        
        for f in cluster_facts:
            all_participants.update(f.get('participants', []))
            if f.get('source_session_id'):
                all_sessions.add(f['source_session_id'])
            all_turns.extend(f.get('source_turns', []))
            all_keys.update(f.get('retrieval_keys', []))
            mentions.append({
                'original_content': f.get('content', ''),
                'source_session_id': f.get('source_session_id', ''),
                'source_turns': f.get('source_turns', []),
                'fact_id': f.get('fact_id', '')
            })
        
        return [MergedFact(
            fact_id=f"fallback_{cluster_id}_0",
            content=base.get('content', ''),
            fact_type=base.get('fact_type', 'EVENT'),
            participants=list(all_participants),
            time=base.get('time', {}),
            location=base.get('location', ''),
            details=base.get('details', {}),
            mentions=mentions,
            source_sessions=list(all_sessions),
            source_turns=list(set(all_turns)),
            retrieval_keys=list(all_keys),
            merge_mode='FALLBACK',
            confidence=0.8
        )]
    
    def _process_large_cluster(self, cluster_facts: List[Dict], cluster_id: int) -> List[MergedFact]:
        """Process large cluster."""
        batch_size = 12
        all_merged = []
        
        for i in range(0, len(cluster_facts), batch_size):
            batch = cluster_facts[i:i + batch_size]
            sub_cluster_id = f"{cluster_id}_batch_{i // batch_size}"
            merged = self._llm_deduplicate_cluster(batch, sub_cluster_id)
            all_merged.extend(merged)
        
        return all_merged
    
    
    def deduplicate_facts(self, facts: List[Dict], sample_id: str) -> List[MergedFact]:
        """Deduplicate facts."""
        if not facts:
            return []
        
        self.logger.info(f" 开始DBSCAN+LLM去重: {len(facts)} 个事实")
        
        clusters = self._cluster_facts_dbscan(facts)
        self._update_stats(clusters_total=len(clusters))
        
        all_merged_facts = []
        llm_tasks = []      # (cluster_facts, cluster_id)
        direct_facts = []
        
        for cluster_id, cluster_facts in clusters.items():
            if cluster_id == -1:
                for f in cluster_facts:
                    merged = self._convert_single_fact(f, f"noise_{len(direct_facts)}")
                    direct_facts.append(merged)
            elif len(cluster_facts) == 1:
                merged = self._convert_single_fact(cluster_facts[0], f"single_{cluster_id}")
                direct_facts.append(merged)
            elif len(cluster_facts) >= self.config.large_cluster_threshold:
                llm_tasks.append((cluster_facts, cluster_id, "large"))
            else:
                llm_tasks.append((cluster_facts, cluster_id, "normal"))
        
        all_merged_facts.extend(direct_facts)
        
        if llm_tasks and self.config.enable_llm_merge:
            self.logger.info(f" 开始多线程LLM去重: {len(llm_tasks)} 个聚类")
            self._update_stats(clusters_llm_processed=len(llm_tasks))
            
            with ThreadPoolExecutor(max_workers=min(self.config.max_workers, len(llm_tasks))) as executor:
                future_to_task = {}
                
                for cluster_facts, cluster_id, task_type in llm_tasks:
                    if task_type == "large":
                        future = executor.submit(self._process_large_cluster, cluster_facts, cluster_id)
                    else:
                        future = executor.submit(self._llm_deduplicate_cluster, cluster_facts, cluster_id)
                    
                    future_to_task[future] = (cluster_id, task_type, len(cluster_facts))
                
                for future in as_completed(future_to_task):
                    cluster_id, task_type, original_size = future_to_task[future]
                    try:
                        result = future.result()
                        all_merged_facts.extend(result)
                        self.logger.debug(f" 聚类 {cluster_id}: {original_size} -> {len(result)}")
                    except Exception as e:
                        self.logger.error(f" 聚类 {cluster_id} 处理失败: {e}")
        
        elif llm_tasks:
            for cluster_facts, cluster_id, _ in llm_tasks:
                merged = self._fallback_merge(cluster_facts, cluster_id)
                all_merged_facts.extend(merged)
        
        self.logger.info(f" 去重完成: {len(facts)} -> {len(all_merged_facts)}")
        return all_merged_facts
    
    def _convert_single_fact(self, fact: Dict, fact_id: str) -> MergedFact:
        """Convert single fact."""
        return MergedFact(
            fact_id=fact_id,
            content=fact.get('content', ''),
            fact_type=fact.get('fact_type', 'EVENT'),
            participants=fact.get('participants', []),
            time=fact.get('time', {}),
            location=fact.get('location', ''),
            details=fact.get('details', {}),
            mentions=[{
                'original_content': fact.get('content', ''),
                'source_session_id': fact.get('source_session_id', ''),
                'source_turns': fact.get('source_turns', []),
                'fact_id': fact.get('fact_id', '')
            }],
            source_sessions=[fact.get('source_session_id', '')] if fact.get('source_session_id') else [],
            source_turns=fact.get('source_turns', []),
            retrieval_keys=fact.get('retrieval_keys', []),
            merge_mode='SINGLE',
            confidence=fact.get('confidence', 1.0)
        )
    
    
    def _generate_accumulations(self, facts: List[MergedFact], sample_id: str) -> List[AccumulatedFact]:
        """Generate accumulations."""
        accumulations = []
        
        action_groups = defaultdict(list)
        
        for fact in facts:
            if fact.fact_type in ['EVENT', 'ACHIEVEMENT', 'ACTIVITY']:
                for participant in fact.participants:
                    content = fact.content.lower()
                    
                    countable_patterns = [
                        ('organized', 'organized'),
                        ('won', 'won'),
                        ('played', 'played'),
                        ('visited', 'visited'),
                        ('met', 'met'),
                        ('recommended', 'recommended'),
                        ('shared', 'shared'),
                        ('went to', 'visited'),
                        ('attended', 'attended')
                    ]
                    
                    for pattern, action in countable_patterns:
                        if pattern in content:
                            key = (participant.lower(), action)
                            action_groups[key].append(fact)
                            break
        
        for (subject, action), group_facts in action_groups.items():
            if len(group_facts) >= 2:
                times = []
                for f in group_facts:
                    time_info = f.time
                    if isinstance(time_info, dict):
                        start = time_info.get('absolute_start')
                        if start and isinstance(start, str) and start.strip():
                            times.append(start)
                
                
                times = sorted([t for t in times if t])
                
                acc = AccumulatedFact(
                    accumulation_id=f"{sample_id}_acc_{subject}_{action}",
                    description=f"{subject.title()} has {action} {len(group_facts)} times",
                    count=len(group_facts),
                    subject=subject.title(),
                    action=action,
                    component_fact_ids=[f.fact_id for f in group_facts],
                    time_span_start=times[0] if times else '',
                    time_span_end=times[-1] if times else '',
                    retrieval_keys=[
                        f"{subject} {action}",
                        f"how many {action}",
                        f"{subject} times",
                        f"count {action}"
                    ]
                )
                accumulations.append(acc)
        
        return accumulations
    
    def _build_timelines(self, facts: List[MergedFact], sample_id: str) -> List[Timeline]:
        """Build timelines."""
        timelines = []
        
        participant_facts = defaultdict(list)
        
        for fact in facts:
            for participant in fact.participants:
                participant_facts[participant.lower()].append(fact)
        
        for participant, p_facts in participant_facts.items():
            if len(p_facts) < 3:
                continue
            
            def get_time(f):
                time_info = f.time
                if isinstance(time_info, dict):
                    val = time_info.get('absolute_start')
                    
                    if val is None or val == '':
                        return '9999-99-99'
                    return str(val)
                return '9999-99-99'
            
            sorted_facts = sorted(p_facts, key=get_time)
            
            state_changes = []
            state_facts = [f for f in sorted_facts if f.fact_type == 'STATE_CHANGE' or f.merge_mode == 'STATE_EVOLUTION']
            
            for i in range(len(state_facts) - 1):
                state_changes.append({
                    'from_fact_id': state_facts[i].fact_id,
                    'to_fact_id': state_facts[i + 1].fact_id,
                    'from_state': state_facts[i].content[:100],
                    'to_state': state_facts[i + 1].content[:100]
                })
            
            timeline = Timeline(
                timeline_id=f"{sample_id}_timeline_{participant}",
                subject=participant.title(),
                description=f"Timeline of events for {participant.title()}",
                ordered_fact_ids=[f.fact_id for f in sorted_facts],
                state_changes=state_changes
            )
            timelines.append(timeline)
        
        return timelines
    
    def _build_temporal_index(self, facts: List[MergedFact]) -> Dict:
        """Build temporal index."""
        index = defaultdict(list)
        for fact in facts:
            time_info = fact.time
            if isinstance(time_info, dict):
                start = time_info.get('absolute_start', '')
                if start and len(start) >= 7:
                    year_month = start[:7]
                    index[year_month].append(fact.fact_id)
        return dict(index)
    
    def _build_participant_index(self, facts: List[MergedFact]) -> Dict:
        """Build participant index."""
        index = defaultdict(list)
        for fact in facts:
            for p in fact.participants:
                index[p.lower()].append(fact.fact_id)
        return dict(index)
    
    
    def process_sample(self, input_file: Path) -> Dict:
        """Process sample."""
        
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        sample_id = data.get('sample_id', 'unknown')
        facts = data.get('episodic_facts', [])
        
        self.logger.info(f"处理样本 {sample_id}: {len(facts)} 个事实")
        
        facts_before = len(facts)
        
        deduplicated_facts = self.deduplicate_facts(facts, sample_id)
        
        facts_merged = facts_before - len(deduplicated_facts)
        
        accumulations = []
        if self.config.enable_accumulation:
            accumulations = self._generate_accumulations(deduplicated_facts, sample_id)
        
        timelines = []
        if self.config.enable_timeline:
            timelines = self._build_timelines(deduplicated_facts, sample_id)
        
        
        temporal_index = self._build_temporal_index(deduplicated_facts)
        participant_index = self._build_participant_index(deduplicated_facts)
        
        output_data = {
            "sample_id": sample_id,
            "metadata": {
                **data.get('metadata', {}),
                "deduplication_time": datetime.now().isoformat(),
                "deduplication_method": "DBSCAN+LLM",
                "facts_before": facts_before,
                "facts_after": len(deduplicated_facts),
                "facts_merged": facts_merged,
                "accumulations_count": len(accumulations),
                "timelines_count": len(timelines)
            },
            "episodic_facts": [f.to_dict() for f in deduplicated_facts],
            "accumulated_facts": [a.to_dict() for a in accumulations],
            "timelines": [t.to_dict() for t in timelines],
            "indices": {
                "temporal": temporal_index,
                "participants": participant_index
            }
        }
        
        
        output_file = self.output_dir / f"{sample_id}_enhanced.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        self._update_stats(
            samples_processed=1,
            facts_before=facts_before,
            facts_after=len(deduplicated_facts),
            facts_merged=facts_merged,
            accumulations_created=len(accumulations),
            timelines_created=len(timelines)
        )
        
        self.logger.info(f" {sample_id}: {facts_before} -> {len(deduplicated_facts)} facts, "
                        f"{len(accumulations)} accumulations, {len(timelines)} timelines")
        
        return {
            'sample_id': sample_id,
            'facts_before': facts_before,
            'facts_after': len(deduplicated_facts),
            'accumulations': len(accumulations),
            'timelines': len(timelines)
        }
    
    def run(self) -> Dict:
        """Run."""
        self.logger.info("=" * 80)
        self.logger.info(" 开始事实去重与增强 (DBSCAN + LLM)")
        self.logger.info("=" * 80)
        
        input_dir = Path(self.config.input_dir)
        if not input_dir.exists():
            self.logger.error(f" 输入目录不存在: {input_dir}")
            return self.stats
        
        input_files = list(input_dir.glob("*_episodic_facts.json"))
        self.logger.info(f" 找到 {len(input_files)} 个输入文件")
        
        for input_file in input_files:
            try:
                self.process_sample(input_file)
            except Exception as e:
                self.logger.error(f" 处理文件 {input_file} 失败: {e}")
                if self.config.debug_mode:
                    import traceback
                    self.logger.error(traceback.format_exc())
        
        
        stats_file = self.output_dir / "enhancement_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)
        
        self._print_summary()
        
        return self.stats
    
    def _print_summary(self):
        """Run print summary."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info(" 事实去重与增强完成 (DBSCAN + LLM)")
        self.logger.info("=" * 80)
        self.logger.info(f" 处理样本数: {self.stats['samples_processed']}")
        self.logger.info(f" 事实数变化: {self.stats['facts_before']} -> {self.stats['facts_after']}")
        self.logger.info(f" 合并事实数: {self.stats['facts_merged']}")
        self.logger.info(f" 聚类总数: {self.stats['clusters_total']}")
        self.logger.info(f" LLM处理聚类: {self.stats['clusters_llm_processed']}")
        self.logger.info(f" LLM调用次数: {self.stats['llm_calls']}")
        self.logger.info(f" 累积事实数: {self.stats['accumulations_created']}")
        self.logger.info(f" 时间线数: {self.stats['timelines_created']}")
        self.logger.info(f" 输出目录: {self.output_dir}")



def main():
    parser = argparse.ArgumentParser(
        description="Step 2: 事实去重与增强 (DBSCAN + LLM)"
    )
    
    parser.add_argument(
        "--input-dir",
        default=str(paths.LOCOMO_EPISODIC_STEP1_DIR),
        help="Step 1输出目录"
    )
    parser.add_argument(
        "--output-dir",
        default=str(paths.LOCOMO_EPISODIC_STEP2_DIR),
        help="输出目录"
    )
    parser.add_argument(
        "--dedup-model",
        default="deepseek-v3.2-dashscope",
        help="去重模型名称"
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="禁用LLM去重（仅使用DBSCAN）"
    )
    parser.add_argument(
        "--no-accumulation",
        action="store_true",
        help="禁用累积事实生成"
    )
    parser.add_argument(
        "--no-timeline",
        action="store_true",
        help="禁用时间线构建"
    )
    parser.add_argument(
        "--auto-optimize",
        action="store_true",
        default=True,
        help="自动优化DBSCAN参数"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="并行工作线程数"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="启用调试模式"
    )
    
    args = parser.parse_args()
    
    config = DeduplicateConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        llm_model=args.dedup_model,
        enable_llm_merge=not args.no_llm,
        enable_accumulation=not args.no_accumulation,
        enable_timeline=not args.no_timeline,
        auto_optimize_dbscan=args.auto_optimize,
        max_workers=args.workers,
        debug_mode=args.debug
    )
    
    deduplicator = FactDeduplicator(config)
    deduplicator.run()


if __name__ == "__main__":
    main()
