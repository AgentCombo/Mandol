# mandol/auto_builder/episodic_builder.py
"""Utilities for episodic builder."""
import logging
from ..utils.logging_config import create_module_logger
import hashlib
import json
from typing import Dict, List, Optional, Any, Union, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from .episodic_prompts import EpisodicPromptTemplateManager, EpisodicFactType
from .l0_views import build_l0_inference_context, extract_original_text
from .graph_write_queue import GraphWriteQueue, GraphWriteRequest, dispatch_graph_write_requests
from ..core.memory_space_registry import TowerSpace

if TYPE_CHECKING:
    from ..core.semantic_map import SemanticMap
    from ..core.semantic_graph import SemanticGraph
    from ..core.memory_unit import MemoryUnit
    from ..llm.llm_client import LLMClient

logger = create_module_logger("auto_builder.episodic_builder")



@dataclass
class TimeInfo:
    original_text: str = ""
    absolute_date: Optional[str] = None
    is_range: bool = False
    range_start: Optional[str] = None
    range_end: Optional[str] = None
    is_future: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TimeInfo":
        if not data:
            return cls()
        return cls(
            original_text=data.get("original_text", ""),
            absolute_date=data.get("absolute_date"),
            is_range=data.get("is_range", False),
            range_start=data.get("range_start"),
            range_end=data.get("range_end"),
            is_future=data.get("is_future", False)
        )


@dataclass
class EpisodicFact:
    fact_id: str
    content: str
    fact_type: str
    participants: List[str] = field(default_factory=list)
    time_info: Optional[TimeInfo] = None
    location: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    source_unit_uids: List[str] = field(default_factory=list)
    retrieval_keys: List[str] = field(default_factory=list)
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if self.time_info:
            result["time_info"] = self.time_info.to_dict()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EpisodicFact":
        time_info = TimeInfo.from_dict(data.get("time_info") or data.get("time", {}))
        return cls(
            fact_id=data.get("fact_id", ""),
            content=data.get("content", ""),
            fact_type=data.get("fact_type", EpisodicFactType.EVENT),
            participants=data.get("participants", []),
            time_info=time_info,
            location=data.get("location"),
            details=data.get("details", {}),
            source_unit_uids=data.get("source_unit_uids", []),
            retrieval_keys=data.get("retrieval_keys", []),
            confidence=data.get("confidence", 1.0),
            metadata=data.get("metadata", {})
        )
    
    def get_text_content(self) -> str:
        """Return text content."""
        return self.content


@dataclass
class MergedFact:
    canonical_content: str
    fact_type: str
    merge_mode: str
    merge_count: Optional[int] = None
    date_list: Optional[List[str]] = None
    state_evolution: Optional[Dict[str, str]] = None
    confidence: float = 1.0
    source_fact_ids: List[str] = field(default_factory=list)
    merge_reasoning: str = ""


@dataclass
class EpisodicBuilderConfig:
    extraction_style: str = "default"  # "default", "locomo", "longmemeval"
    fact_types: Optional[List[str]] = None
    
    enable_deduplication: bool = True
    dedup_method: str = "dbscan_llm"  # "dbscan_llm", "dbscan_only", "embedding_only"
    dbscan_eps: float = 0.15
    dbscan_min_samples: int = 1
    auto_optimize_dbscan: bool = True
    dbscan_eps_range: Tuple[float, float] = (0.1, 0.5)
    dbscan_min_samples_range: Tuple[int, int] = (1, 3)
    llm_dedup_cluster_threshold: int = 2
    dedup_parallel_workers: int = 30
    large_cluster_threshold: int = 15
    
    episodic_space_name: str = TowerSpace.EPISODIC_ROOT.value
    add_to_graph: bool = True
    
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    debug_mode: bool = False



class EpisodicAutoBuilder:
    """```python builder = EpisodicAutoBuilder( semantic_system=semantic_graph, llm_client=llm_client, config=EpisodicBuilderConfig(extraction_style="locomo") ) facts = builder.extract_from_l0_units( l0_unit_uids=["unit1", "unit2"], reference_date."""
    
    def __init__(self,
                 semantic_system: Union["SemanticMap", "SemanticGraph"],
                 llm_client: "LLMClient",
                 config: Optional[EpisodicBuilderConfig] = None):
        self.semantic_system = semantic_system
        self.llm_client = llm_client
        self.config = config or EpisodicBuilderConfig()
        
        self.prompt_manager = EpisodicPromptTemplateManager()
        
        self._encoder = None
        self._encoder_lock = Lock()
        
        self._stats_lock = Lock()
        self._stats = {
            "facts_extracted": 0,
            "facts_before_dedup": 0,
            "facts_after_dedup": 0,
            "facts_merged": 0,
            "llm_calls": 0,
            "units_added": 0
        }
        
        logger.info("EpisodicAutoBuilder initialized")
        logger.info(f"   - extraction_style: {self.config.extraction_style}")
        logger.info(f"   - dedup_method: {self.config.dedup_method if self.config.enable_deduplication else 'disabled'}")
    
    def _get_encoder(self):
        """Get encoder."""
        if self._encoder is None:
            with self._encoder_lock:
                if self._encoder is None:
                    from sentence_transformers import SentenceTransformer
                    self._encoder = SentenceTransformer(self.config.embedding_model)
                    logger.info(f"Embedding model loaded: {self.config.embedding_model}")
        return self._encoder
    
    def _update_stats(self, **kwargs):
        """Run update stats."""
        with self._stats_lock:
            for key, value in kwargs.items():
                if key in self._stats:
                    self._stats[key] += value
    
    def _generate_fact_id(self, content: str, fact_type: str, source_uid: str = "") -> str:
        """Generate fact id."""
        hash_input = f"{content}_{fact_type}_{source_uid}"
        return f"ef_{hashlib.md5(hash_input.encode()).hexdigest()[:12]}"
    
    
    def extract_from_l0_units(self,
                             l0_unit_uids: List[str],
                             reference_date: str = "",
                             source_id: str = "",
                             speakers: str = "",
                             custom_prompt: Optional[str] = None,
                             batch_size: int = 10) -> List[EpisodicFact]:
        """Extract from l0 units."""
        logger.info(f"Starting episodic fact extraction from {len(l0_unit_uids)} L0 units...")
        
        all_facts: List[EpisodicFact] = []
        
        
        l0_units = []
        for uid in l0_unit_uids:
            unit = self._get_unit_by_uid(uid)
            if unit:
                l0_units.append(unit)
        
        if not l0_units:
            logger.warning("No valid L0 units found")
            return []

        batch_content = self._merge_l0_content(l0_units)
        if not batch_content:
            logger.warning("L0 units do not contain source text for episodic extraction")
            return []

        if not reference_date:
            for unit in l0_units:
                metadata = unit.metadata or {}
                meta_date = metadata.get('session_date') or metadata.get('date')
                if meta_date:
                    reference_date = meta_date
                    break

        if not reference_date:
            reference_date = datetime.now().strftime("%Y-%m-%d")

        prompt = self.prompt_manager.get_extraction_prompt(
            style=self.config.extraction_style,
            content=batch_content,
            reference_date=reference_date,
            source_id=source_id or "l0_reconstructed",
            speakers=speakers,
            fact_types=self.config.fact_types,
            custom_prompt=custom_prompt
        )

        try:
            response = self.llm_client.generate_answer(
                prompt=prompt,
                temperature=0.1,
                max_tokens=4000,
                json_format=True
            )

            self._update_stats(llm_calls=1)

            facts_data = self._parse_extraction_response(response)
            source_uids = [unit.uid for unit in l0_units]
            for fact_data in facts_data:
                fact = self._convert_to_episodic_fact(fact_data, source_uids)
                all_facts.append(fact)

        except Exception as e:
            logger.error(f"Episodic fact extraction failed: {e}")
        
        self._update_stats(facts_extracted=len(all_facts))
        logger.info(f"Extraction completed with {len(all_facts)} facts")
        
        return all_facts
    
    def extract_from_text(self,
                         text: str,
                         reference_date: str = "",
                         source_id: str = "",
                         speakers: str = "",
                         custom_prompt: Optional[str] = None) -> List[EpisodicFact]:
        """Extract from text."""
        if not reference_date:
            reference_date = datetime.now().strftime("%Y-%m-%d")
        
        prompt = self.prompt_manager.get_extraction_prompt(
            style=self.config.extraction_style,
            content=text,
            reference_date=reference_date,
            source_id=source_id,
            speakers=speakers,
            fact_types=self.config.fact_types,
            custom_prompt=custom_prompt
        )
        
        try:
            response = self.llm_client.generate_answer(
                prompt=prompt,
                temperature=0.1,
                max_tokens=4000,
                json_format=True
            )
            
            self._update_stats(llm_calls=1)
            
            facts_data = self._parse_extraction_response(response)
            
            facts = []
            for fact_data in facts_data:
                fact = self._convert_to_episodic_fact(fact_data, [source_id] if source_id else [])
                facts.append(fact)
            
            self._update_stats(facts_extracted=len(facts))
            return facts
            
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            return []
    
    
    def deduplicate_facts(self,
                         facts: List[EpisodicFact],
                         custom_prompt: Optional[str] = None) -> List[EpisodicFact]:
        """Deduplicate facts."""
        if not self.config.enable_deduplication:
            return facts
        
        if len(facts) < 2:
            return facts
        
        self._update_stats(facts_before_dedup=len(facts))
        logger.info(f"Starting deduplication for {len(facts)} facts...")

        if self.config.extraction_style == "longmemeval":
            deduplicated = self._deduplicate_longmemeval_facts(facts, custom_prompt)
            self._update_stats(
                facts_after_dedup=len(deduplicated),
                facts_merged=len(facts) - len(deduplicated)
            )
            logger.info(f"Deduplication completed: {len(facts)} -> {len(deduplicated)} facts")
            return deduplicated
        
        clusters = self._cluster_facts(facts)
        logger.info(f"Clustering completed with {len(clusters)} clusters")
        
        deduplicated = []
        llm_tasks = []

        for cluster_id, cluster_facts in clusters.items():
            if len(cluster_facts) == 1:
                deduplicated.append(cluster_facts[0])
            elif len(cluster_facts) >= self.config.llm_dedup_cluster_threshold and self.config.dedup_method == "dbscan_llm":
                llm_tasks.append((cluster_id, cluster_facts))
            else:
                merged = self._simple_merge_cluster(cluster_facts)
                deduplicated.extend(merged)

        if llm_tasks:
            max_workers = min(max(1, self.config.dedup_parallel_workers), len(llm_tasks))
            if max_workers == 1:
                for cluster_id, cluster_facts in llm_tasks:
                    deduplicated.extend(self._deduplicate_cluster_with_size_guard(cluster_id, cluster_facts, custom_prompt))
            else:
                with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="EpisodicDedup") as executor:
                    future_to_cluster = {
                        executor.submit(
                            self._deduplicate_cluster_with_size_guard,
                            cluster_id,
                            cluster_facts,
                            custom_prompt,
                        ): cluster_id
                        for cluster_id, cluster_facts in llm_tasks
                    }
                    for future in as_completed(future_to_cluster):
                        cluster_id = future_to_cluster[future]
                        try:
                            deduplicated.extend(future.result())
                        except Exception as exc:
                            logger.warning(f"LLM deduplication failed for cluster {cluster_id}: {exc}; keeping original facts")
                            for fallback_cluster_id, fallback_facts in llm_tasks:
                                if fallback_cluster_id == cluster_id:
                                    deduplicated.extend(fallback_facts)
                                    break
        
        self._update_stats(
            facts_after_dedup=len(deduplicated),
            facts_merged=len(facts) - len(deduplicated)
        )
        logger.info(f"Deduplication completed: {len(facts)} -> {len(deduplicated)} facts")
        
        return deduplicated

    def _deduplicate_longmemeval_facts(
            self,
            facts: List[EpisodicFact],
            custom_prompt: Optional[str] = None) -> List[EpisodicFact]:
        grouped: Dict[str, List[EpisodicFact]] = defaultdict(list)
        for fact in facts:
            grouped[fact.fact_type or "UNKNOWN"].append(fact)

        deduplicated: List[EpisodicFact] = []
        llm_tasks: List[Tuple[str, int, List[EpisodicFact]]] = []

        for fact_type in sorted(grouped):
            type_facts = grouped[fact_type]
            if len(type_facts) < 2:
                deduplicated.extend(type_facts)
                continue

            clusters = self._cluster_facts(type_facts)
            logger.info(f"LongMemEval fact type {fact_type}: {len(type_facts)} facts, {len(clusters)} clusters")
            for cluster_id, cluster_facts in clusters.items():
                if len(cluster_facts) == 1:
                    deduplicated.append(cluster_facts[0])
                elif len(cluster_facts) >= self.config.llm_dedup_cluster_threshold and self.config.dedup_method == "dbscan_llm":
                    llm_tasks.append((fact_type, cluster_id, cluster_facts))
                else:
                    deduplicated.extend(self._simple_merge_cluster(cluster_facts))

        def run_llm_task(fact_type: str, cluster_id: int, cluster_facts: List[EpisodicFact]) -> List[EpisodicFact]:
            merged = self._deduplicate_cluster_with_size_guard(cluster_id, cluster_facts, custom_prompt)
            source_types = {fact.fact_type for fact in cluster_facts if fact.fact_type}
            fallback_type = fact_type or (cluster_facts[0].fact_type if cluster_facts else "UNKNOWN")
            for fact in merged:
                if fact.fact_type not in source_types:
                    fact.fact_type = fallback_type
            return merged or cluster_facts

        if llm_tasks:
            max_workers = min(max(1, self.config.dedup_parallel_workers), len(llm_tasks))
            if max_workers == 1:
                for fact_type, cluster_id, cluster_facts in llm_tasks:
                    deduplicated.extend(run_llm_task(fact_type, cluster_id, cluster_facts))
            else:
                with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="EpisodicDedup") as executor:
                    future_to_task = {
                        executor.submit(run_llm_task, fact_type, cluster_id, cluster_facts): (fact_type, cluster_id, cluster_facts)
                        for fact_type, cluster_id, cluster_facts in llm_tasks
                    }
                    for future in as_completed(future_to_task):
                        fact_type, cluster_id, cluster_facts = future_to_task[future]
                        try:
                            deduplicated.extend(future.result())
                        except Exception as exc:
                            logger.warning(f"LongMemEval {fact_type} cluster {cluster_id} deduplication failed: {exc}; keeping original facts")
                            deduplicated.extend(cluster_facts)

        return deduplicated
    
    def _cluster_facts(self, facts: List[EpisodicFact]) -> Dict[int, List[EpisodicFact]]:
        """Run cluster facts."""
        try:
            from sklearn.cluster import DBSCAN
            import numpy as np
        except ImportError:
            logger.warning("sklearn is unavailable; using simple clustering")
            return {i: [f] for i, f in enumerate(facts)}
        
        encoder = self._get_encoder()
        contents = [f.content for f in facts]
        embeddings = encoder.encode(contents, show_progress_bar=False)
        
        eps = self.config.dbscan_eps
        min_samples = self.config.dbscan_min_samples
        
        if self.config.auto_optimize_dbscan and len(facts) >= 10:
            optimized = self._optimize_dbscan_params(embeddings)
            if optimized:
                eps = optimized.get("eps", eps)
                min_samples = optimized.get("min_samples", min_samples)
                logger.info(f"DBSCAN parameter tuning: eps={eps:.3f}, min_samples={min_samples}")
        
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        labels = clustering.fit_predict(embeddings)
        
        clusters: Dict[int, List[EpisodicFact]] = defaultdict(list)
        noise_idx = 0
        for i, label in enumerate(labels):
            if label == -1:
                clusters[f"noise_{noise_idx}"] = [facts[i]]
                noise_idx += 1
            else:
                clusters[label].append(facts[i])
        
        return dict(clusters)
    
    def _optimize_dbscan_params(self, embeddings) -> Optional[Dict[str, Any]]:
        """Run optimize dbscan params."""
        try:
            from sklearn.metrics import silhouette_score
            import numpy as np
            
            best_score = -1
            best_params = None
            
            eps_low, eps_high = self.config.dbscan_eps_range
            min_low, min_high = self.config.dbscan_min_samples_range
            for eps in np.linspace(eps_low, eps_high, 9):
                for min_samples in range(min_low, min_high + 1):
                    from sklearn.cluster import DBSCAN
                    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
                    labels = clustering.fit_predict(embeddings)
                    
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    if n_clusters < 2:
                        continue
                    
                    try:
                        score = silhouette_score(embeddings, labels)
                        if score > best_score:
                            best_score = score
                            best_params = {"eps": eps, "min_samples": min_samples}
                    except:
                        continue
            
            return best_params
            
        except Exception as e:
            logger.debug(f"Parameter tuning failed: {e}")
            return None

    def _deduplicate_cluster_with_size_guard(
        self,
        cluster_id: int,
        facts: List[EpisodicFact],
        custom_prompt: Optional[str] = None,
    ) -> List[EpisodicFact]:
        """Run LLM dedup in offline-compatible batches for very large clusters."""
        if len(facts) < self.config.large_cluster_threshold:
            return self._llm_deduplicate_cluster(cluster_id, facts, custom_prompt)

        merged: List[EpisodicFact] = []
        batch_size = 12
        for start in range(0, len(facts), batch_size):
            batch = facts[start : start + batch_size]
            batch_id = f"{cluster_id}_batch_{start // batch_size}"
            merged.extend(self._llm_deduplicate_cluster(batch_id, batch, custom_prompt))
        return merged
    
    def _llm_deduplicate_cluster(self,
                                cluster_id: int,
                                facts: List[EpisodicFact],
                                custom_prompt: Optional[str] = None) -> List[EpisodicFact]:
        """Run LLM deduplicate cluster."""
        candidates = []
        for i, fact in enumerate(facts, 1):
            time_str = ""
            if fact.time_info and fact.time_info.absolute_date:
                time_str = f" [Date: {fact.time_info.absolute_date}]"
            candidates.append(f"{i}. [{fact.fact_type}]{time_str}: {fact.content}")
        
        fact_candidates = "\n".join(candidates)
        
        prompt = self.prompt_manager.get_deduplication_prompt(
            cluster_id=cluster_id,
            fact_candidates=fact_candidates,
            cluster_size=len(facts),
            custom_prompt=custom_prompt
        )
        
        try:
            response = self.llm_client.generate_answer(
                prompt=prompt,
                temperature=0.1,
                max_tokens=2000,
                json_format=True
            )
            
            self._update_stats(llm_calls=1)
            
            merged_data = self._parse_dedup_response(response)
            
            result = []
            for merged in merged_data:
                source_indices = merged.get("source_fact_indices", [])
                source_facts = [facts[i-1] for i in source_indices if 0 < i <= len(facts)]
                
                source_uids = []
                for sf in source_facts:
                    source_uids.extend(sf.source_unit_uids)
                
                fact = EpisodicFact(
                    fact_id=self._generate_fact_id(
                        merged.get("canonical_content", ""),
                        merged.get("fact_type", EpisodicFactType.EVENT),
                        "_".join(source_uids[:3]) if source_uids else ""
                    ),
                    content=merged.get("canonical_content", ""),
                    fact_type=merged.get("fact_type", EpisodicFactType.EVENT),
                    participants=list(set(p for sf in source_facts for p in sf.participants)),
                    time_info=source_facts[0].time_info if source_facts else None,
                    source_unit_uids=list(set(source_uids)),
                    retrieval_keys=list(set(k for sf in source_facts for k in sf.retrieval_keys)),
                    confidence=merged.get("confidence", 1.0),
                    metadata={
                        "merge_mode": merged.get("merge_mode", "A"),
                        "merge_count": merged.get("merge_count"),
                        "merge_reasoning": merged.get("merge_reasoning", "")
                    }
                )
                result.append(fact)
            
            return result if result else facts
            
        except Exception as e:
            logger.warning(f"LLM deduplication failed: {e}; keeping original facts")
            return facts
    
    def _simple_merge_cluster(self, facts: List[EpisodicFact]) -> List[EpisodicFact]:
        """Run simple merge cluster."""
        if not facts:
            return []
        
        
        sorted_facts = sorted(facts, key=lambda f: len(f.content), reverse=True)
        best = sorted_facts[0]
        
        all_participants = set(best.participants)
        all_keys = set(best.retrieval_keys)
        all_uids = set(best.source_unit_uids)
        
        for f in sorted_facts[1:]:
            all_participants.update(f.participants)
            all_keys.update(f.retrieval_keys)
            all_uids.update(f.source_unit_uids)
        
        best.participants = list(all_participants)
        best.retrieval_keys = list(all_keys)
        best.source_unit_uids = list(all_uids)
        
        return [best]
    
    
    def add_to_semantic_system(self,
                              facts: List[EpisodicFact],
                              space_name: Optional[str] = None,
                              graph_writer: Optional[GraphWriteQueue] = None,
                              wait_for_completion: bool = True) -> List[str]:
        """Add to semantic system."""
        if not self.config.add_to_graph:
            logger.info("add_to_graph is disabled; skipping insertion")
            return []
        
        space_name = space_name or self.config.episodic_space_name
        
        logger.info(f"Adding {len(facts)} facts to {space_name}...")
        
        added_uids = []
        write_requests: List[GraphWriteRequest] = []
        
        for fact in facts:
            try:
                unit = self._create_memory_unit_from_fact(fact)
                
                write_requests.append(GraphWriteRequest(
                    unit=unit,
                    explicit_content_for_embedding=fact.get_text_content(),
                    content_type_for_embedding="text",
                    space_names=[space_name],
                    index_update_mode="none",
                    generate_sparse_embedding=False,
                    source="episodic_fact",
                    metadata={"fact_id": fact.fact_id},
                ))
                added_uids.append(unit.uid)
                
            except Exception as e:
                logger.warning(f"Failed to add fact {fact.fact_id}: {e}")
                continue

        if write_requests:
            dispatch_graph_write_requests(
                semantic_system=self.semantic_system,
                requests=write_requests,
                graph_writer=graph_writer,
                wait_for_completion=wait_for_completion,
            )
        
        self._update_stats(units_added=len(added_uids))
        logger.info(f"Added {len(added_uids)} memory units")
        
        return added_uids
    
    
    def run_full_pipeline(self,
                         l0_unit_uids: List[str],
                         reference_date: str = "",
                         source_id: str = "",
                         speakers: str = "",
                         custom_extraction_prompt: Optional[str] = None,
                         custom_dedup_prompt: Optional[str] = None,
                         space_name: Optional[str] = None,
                         graph_writer: Optional[GraphWriteQueue] = None) -> Dict[str, Any]:
        """Run full pipeline."""
        start_time = datetime.now()
        logger.info(f"\n{'='*60}")
        logger.info("Starting episodic memory extraction pipeline")
        logger.info(f"{'='*60}")
        
        facts = self.extract_from_l0_units(
            l0_unit_uids=l0_unit_uids,
            reference_date=reference_date,
            source_id=source_id,
            speakers=speakers,
            custom_prompt=custom_extraction_prompt
        )
        
        if self.config.enable_deduplication and facts:
            deduplicated = self.deduplicate_facts(facts, custom_prompt=custom_dedup_prompt)
        else:
            deduplicated = facts
        
        added_uids = self.add_to_semantic_system(
            deduplicated,
            space_name=space_name,
            graph_writer=graph_writer,
            wait_for_completion=graph_writer is None,
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        result = {
            "success": True,
            "duration_seconds": duration,
            "facts_extracted": len(facts),
            "facts_after_dedup": len(deduplicated),
            "units_added": len(added_uids),
            "added_uids": added_uids,
            "stats": dict(self._stats)
        }
        
        logger.info(f"\n{'='*60}")
        logger.info("Pipeline completed")
        logger.info(f"   - elapsed: {duration:.2f}s")
        logger.info(f"   - extracted: {len(facts)} facts")
        logger.info(f"   - deduplicated: {len(deduplicated)} facts")
        logger.info(f"   - added: {len(added_uids)} units")
        logger.info(f"{'='*60}\n")
        
        return result
    
    
    def _get_unit_by_uid(self, uid: str) -> Optional["MemoryUnit"]:
        """Get unit by UID."""
        try:
            if hasattr(self.semantic_system, 'get_unit'):
                return self.semantic_system.get_unit(uid)
            elif hasattr(self.semantic_system, 'semantic_map'):
                return self.semantic_system.semantic_map.get_unit(uid)
            elif hasattr(self.semantic_system, 'units'):
                return self.semantic_system.units.get(uid)
        except Exception as e:
            logger.debug(f"Failed to get unit {uid}: {e}")
        return None
    
    def _extract_unit_content(self, unit: "MemoryUnit") -> str:
        """Extract unit content."""
        content = extract_original_text(unit)
        if content:
            return content

        raw_data = dict(unit.raw_data or {})
        raw_data.pop("enhanced_content", None)
        
        for key in ["text_content", "content", "text", "dialogue", "message"]:
            if key in raw_data and raw_data[key]:
                return str(raw_data[key])
        
        if "dialogues" in raw_data:
            dialogues = raw_data["dialogues"]
            if isinstance(dialogues, list):
                lines = []
                for turn in dialogues:
                    if isinstance(turn, dict):
                        speaker = turn.get("speaker", "")
                        text = turn.get("text", "")
                        lines.append(f"{speaker}: {text}")
                    else:
                        lines.append(str(turn))
                return "\n".join(lines)
        
        return json.dumps(raw_data, ensure_ascii=False)

    def _merge_l0_content(self, l0_units: List["MemoryUnit"]) -> str:
        """Run merge L0 content."""
        return build_l0_inference_context(l0_units)
    
    def _parse_extraction_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse extraction response."""
        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            
            data = json.loads(response)
            
            if isinstance(data, dict) and "memory_facts" in data:
                return data["memory_facts"]
            elif isinstance(data, dict) and "facts" in data:
                return data["facts"]
            elif isinstance(data, list):
                return data
            else:
                return []
                
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing failed: {e}")
            try:
                from json_repair import repair_json
                repaired = repair_json(response)
                data = json.loads(repaired)
                if isinstance(data, dict) and "memory_facts" in data:
                    return data["memory_facts"]
                if isinstance(data, dict) and "facts" in data:
                    return data["facts"]
                return data if isinstance(data, list) else []
            except:
                return []
    
    def _parse_dedup_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse dedup response."""
        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            
            data = json.loads(response)
            
            if isinstance(data, dict) and "merged_facts" in data:
                return data["merged_facts"]
            elif isinstance(data, list):
                return data
            else:
                return []
                
        except json.JSONDecodeError:
            try:
                from json_repair import repair_json
                repaired = repair_json(response)
                data = json.loads(repaired)
                if isinstance(data, dict) and "merged_facts" in data:
                    return data["merged_facts"]
                return data if isinstance(data, list) else []
            except:
                return []
    
    def _convert_to_episodic_fact(self,
                                  data: Dict[str, Any],
                                  source_uids: List[str]) -> EpisodicFact:
        """Convert to episodic fact."""
        time_data = data.get("time", {}) or data.get("time_info", {})
        time_info = TimeInfo.from_dict(time_data) if time_data else None
        
        content = data.get("content", "")
        fact_type = data.get("fact_type") or data.get("category", EpisodicFactType.EVENT)
        
        return EpisodicFact(
            fact_id=self._generate_fact_id(content, fact_type, source_uids[0] if source_uids else ""),
            content=content,
            fact_type=fact_type,
            participants=data.get("participants", []),
            time_info=time_info,
            location=data.get("location"),
            details=data.get("details", {}),
            source_unit_uids=source_uids,
            retrieval_keys=data.get("retrieval_keys", []),
            confidence=data.get("confidence", 1.0),
            metadata={
                "source_turns": data.get("source_turns", []),
                "extraction_style": self.config.extraction_style
            }
        )
    
    def _create_memory_unit_from_fact(self, fact: EpisodicFact) -> "MemoryUnit":
        """Create memory unit from fact."""
        from ..core.memory_unit import MemoryUnit
        
        raw_data = {
            "text_content": fact.content,
            "fact_type": fact.fact_type,
            "participants": fact.participants,
            "location": fact.location,
            "details": fact.details,
            "retrieval_keys": fact.retrieval_keys
        }
        
        if fact.time_info:
            raw_data["time"] = fact.time_info.to_dict()
            if fact.time_info.absolute_date:
                raw_data["temporal_tag"] = fact.time_info.absolute_date
        
        metadata = {
            "fact_id": fact.fact_id,
            "source_unit_uids": fact.source_unit_uids,
            "confidence": fact.confidence,
            "created": datetime.now().isoformat(),
            "episodic_builder_version": "1.0.0",
            **fact.metadata
        }
        
        return MemoryUnit(
            uid=fact.fact_id,
            raw_data=raw_data,
            metadata=metadata
        )
    
    
    def get_stats(self) -> Dict[str, int]:
        """Return stats."""
        with self._stats_lock:
            return dict(self._stats)
    
    def reset_stats(self):
        """Run reset stats."""
        with self._stats_lock:
            for key in self._stats:
                self._stats[key] = 0
    
    def set_custom_prompt(self, prompt_name: str, prompt: str):
        """Set custom prompt."""
        self.prompt_manager.register_custom_prompt(prompt_name, prompt)
        logger.info(f"Registered custom prompt: {prompt_name}")
