# mandol/auto_builder/entity_relation_builder.py
"""Utilities for entity relation builder."""
import logging
from ..utils.logging_config import create_module_logger
import json
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple, TYPE_CHECKING
from datetime import datetime
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from .entity_relation_prompts import EntityRelationPromptManager, EntityType
from .l0_views import build_l0_inference_context
from .graph_write_queue import GraphWriteQueue, GraphWriteRequest, dispatch_graph_write_requests
from ..core.memory_space_registry import TowerSpace

if TYPE_CHECKING:
    from ..core.semantic_map import SemanticMap
    from ..core.semantic_graph import SemanticGraph
    from ..core.memory_unit import MemoryUnit
    from ..llm.llm_client import LLMClient

logger = create_module_logger("auto_builder.entity_relation_builder")

try:
    from sklearn.cluster import DBSCAN
    DBSCAN_AVAILABLE = True
except ImportError:
    DBSCAN_AVAILABLE = False
    logger.warning("DBSCAN clustering is unavailable; using simple similarity-threshold deduplication")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False
    logger.warning("SentenceTransformer is unavailable; entity deduplication will be limited")



@dataclass
class EntityMention:
    session_id: str
    unit_uid: str
    content: str
    session_date: Optional[str] = None
    temporal_info: Optional[str] = None
    temporal_reference: Optional[str] = None
    spatial_info: Optional[str] = None
    numerical_value: Optional[str] = None
    aliases: List[str] = field(default_factory=list)
    confidence: float = 0.8
    
    def __post_init__(self):
        if self.aliases is None:
            self.aliases = []
    
    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "unit_uid": self.unit_uid,
            "content": self.content,
            "session_date": self.session_date,
            "temporal_info": self.temporal_info,
            "temporal_reference": self.temporal_reference,
            "spatial_info": self.spatial_info,
            "numerical_value": self.numerical_value,
            "aliases": self.aliases,
            "confidence": self.confidence
        }


@dataclass
class ExtractedEntity:
    entity_id: str
    name: str
    entity_type: str
    confidence: float
    mentions: List[EntityMention] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)
    related_entities: List[str] = field(default_factory=list)
    merge_reasoning: Optional[str] = None
    
    def __post_init__(self):
        if self.mentions is None:
            self.mentions = []
        if self.aliases is None:
            self.aliases = []
        if self.related_entities is None:
            self.related_entities = []
    
    @property
    def session_ids(self) -> List[str]:
        """Run session IDs."""
        return list(set(m.session_id for m in self.mentions))
    
    @property
    def unit_uids(self) -> List[str]:
        """Run unit UIDs."""
        return list(set(m.unit_uid for m in self.mentions))
    
    def get_all_temporal_info(self) -> List[str]:
        """Return all temporal info."""
        return [m.temporal_info for m in self.mentions if m.temporal_info]
    
    def get_all_spatial_info(self) -> List[str]:
        """Return all spatial info."""
        return [m.spatial_info for m in self.mentions if m.spatial_info]
    
    def to_dict(self) -> Dict:
        return {
            "entity_id": self.entity_id,
            "name": self.name,
            "entity_type": self.entity_type,
            "confidence": self.confidence,
            "aliases": self.aliases,
            "related_entities": self.related_entities,
            "merge_reasoning": self.merge_reasoning,
            "session_ids": self.session_ids,
            "mentions": [m.to_dict() for m in self.mentions]
        }


@dataclass
class ExtractedRelation:
    relation_id: str
    source_entity_id: str
    relation_type: str
    target_entity_id: str
    context: str
    session_id: Optional[str] = None
    unit_uid: Optional[str] = None
    temporal_info: Optional[str] = None
    confidence: float = 0.8
    is_cross_session: bool = False
    
    def to_dict(self) -> Dict:
        return {
            "relation_id": self.relation_id,
            "source_entity_id": self.source_entity_id,
            "relation_type": self.relation_type,
            "target_entity_id": self.target_entity_id,
            "context": self.context,
            "session_id": self.session_id,
            "unit_uid": self.unit_uid,
            "temporal_info": self.temporal_info,
            "confidence": self.confidence,
            "is_cross_session": self.is_cross_session
        }


@dataclass
class MergedEntity:
    entity_id: str
    name: str
    entity_type: str
    confidence: float
    mentions: List[EntityMention]
    merge_reasoning: str
    original_entity_ids: List[str] = field(default_factory=list)



@dataclass
class EntityRelationBuilderConfig:
    
    extraction_style: str = "default"          # "default", "locomo", "longmemeval"
    
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    dbscan_eps: float = 0.3
    dbscan_min_samples: int = 1
    auto_optimize_dbscan: bool = False
    dbscan_eps_range: Tuple[float, float] = (0.15, 0.6)
    dbscan_min_samples_range: Tuple[int, int] = (2, 6)
    similarity_threshold: float = 0.85
    
    relation_confidence_threshold: float = 0.6
    
    llm_temperature: float = 0.3
    llm_max_tokens: int = 4096
    
    parallel_workers: int = 30
    batch_size: int = 10
    large_cluster_threshold: int = 12
    
    entity_space_name: str = TowerSpace.GRAPH_ENTITIES.value
    relation_space_name: str = TowerSpace.GRAPH_RELATIONS.value
    
    enable_relation_extraction: bool = True
    enable_cross_session_relations: bool = False
    enable_llm_deduplication: bool = True
    
    def to_dict(self) -> Dict:
        return asdict(self)



class EntityRelationAutoBuilder:
    
    ENTITY_RELATION_SPACE_NAME = TowerSpace.GRAPH_ROOT.value
    LONGMEMEVAL_ADDITIONAL_ENTITY_TYPES = {
        "SERVICE",
        "FEATURE",
        "ATTRIBUTE_SET",
        "PROJECT",
        "INDUSTRY",
        "DESIGNATION",
        "SESSION_ID",
    }
    LONGMEMEVAL_ENTITY_TYPE_ALIASES = {
        "PRODUCT_CATEGORY": EntityType.PRODUCT,
        "PRODUCT_LINE": EntityType.PRODUCT,
        "EVENT_CATEGORY": EntityType.EVENT,
        "PROCEDURE": EntityType.ACTIVITY,
        "STRATEGY": EntityType.CONCEPT,
        "GAME_MECHANIC_CONCEPT": EntityType.CONCEPT,
        "LEGAL_CONCEPT": EntityType.CONCEPT,
        "STATISTICAL_AGGREGATE": EntityType.NUMERICAL_VALUE,
        "COMPOSITE_NUMERICAL_VALUE_SET": EntityType.NUMERICAL_VALUE,
        "SESSION": "SESSION_ID",
        "ACT": EntityType.ACTIVITY,
        "GROUP": EntityType.ORGANIZATION,
    }
    
    def __init__(self,
                 semantic_system: Optional[Union["SemanticMap", "SemanticGraph"]] = None,
                 llm_client: Optional["LLMClient"] = None,
                 config: Optional[EntityRelationBuilderConfig] = None):
        self.semantic_system = semantic_system
        self.llm_client = llm_client
        self.config = config or EntityRelationBuilderConfig()
        
        self.prompt_manager = EntityRelationPromptManager()
        
        self.entity_encoder = None
        if SENTENCE_TRANSFORMER_AVAILABLE:
            try:
                logger.info(f"Loading entity embedding model: {self.config.embedding_model}")
                self.entity_encoder = SentenceTransformer(self.config.embedding_model)
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")
        
        self.stats_lock = Lock()
        self.stats = {
            "l0_units_processed": 0,
            "entities_extracted": 0,
            "entities_after_dedup": 0,
            "relations_extracted": 0,
            "relations_after_filter": 0,
            "entities_added_to_graph": 0,
            "relations_added_to_graph": 0,
            "llm_calls": 0,
            "errors": 0
        }
        
        logger.info("EntityRelationAutoBuilder initialized")
        logger.info(f"   - extraction_style: {self.config.extraction_style}")
        logger.info(f"   - DBSCAN eps: {self.config.dbscan_eps}")
        logger.info(f"   - relation_confidence_threshold: {self.config.relation_confidence_threshold}")
    
    def _update_stats(self, **kwargs):
        """Run update stats."""
        with self.stats_lock:
            for key, value in kwargs.items():
                if key in self.stats:
                    self.stats[key] += value
    
    def get_stats(self) -> Dict[str, Any]:
        """Return stats."""
        with self.stats_lock:
            return self.stats.copy()
    
    def reset_stats(self):
        """Run reset stats."""
        with self.stats_lock:
            for key in self.stats:
                self.stats[key] = 0
    
    
    def extract_entities_from_l0_units(self,
                                       l0_units: List["MemoryUnit"],
                                       reference_date: Optional[str] = None,
                                       source_id: str = "default",
                                       session_type: str = "default",
                                       custom_prompt: Optional[str] = None) -> List[Dict[str, Any]]:
        """Extract entities from l0 units."""
        if not l0_units:
            logger.warning("No L0 units were provided")
            return []
        
        logger.info(f"Starting entity extraction from {len(l0_units)} L0 units...")
        
        content = self._merge_l0_content(l0_units)
        unit_uids = [u.uid for u in l0_units]
        
        prompt = self.prompt_manager.get_entity_extraction_prompt_v2(
            style=self.config.extraction_style,
            content=content,
            reference_date=reference_date or datetime.now().strftime("%Y-%m-%d"),
            source_id=source_id,
            content_type=session_type,
            session_type=session_type,
            custom_prompt=custom_prompt
        )
        
        try:
            response = self.llm_client.generate_answer(
                prompt=prompt,
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens,
                json_format=True
            )
            self._update_stats(llm_calls=1)
            
            parsed = self._safe_parse_json(response)
            raw_entities = parsed.get("entities", [])
            
            for entity in raw_entities:
                entity["source_id"] = source_id
                entity["unit_uids"] = unit_uids
                entity["reference_date"] = reference_date
            
            self._update_stats(
                l0_units_processed=len(l0_units),
                entities_extracted=len(raw_entities)
            )
            
            logger.info(f"    Extracted {len(raw_entities)} raw entities")
            return raw_entities
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            self._update_stats(errors=1)
            return []
    
    def deduplicate_entities(self,
                            raw_entities: List[Dict[str, Any]],
                            custom_prompt: Optional[str] = None) -> List[ExtractedEntity]:
        """Deduplicate entities."""
        if not raw_entities:
            return []
        
        logger.info(f"Starting entity deduplication (raw={len(raw_entities)})")
        
        entity_texts = [self._entity_clustering_text(e) for e in raw_entities]
        
        if self.entity_encoder and DBSCAN_AVAILABLE and self.config.enable_llm_deduplication:
            logger.debug("   Using DBSCAN clustering...")
            embeddings = self.entity_encoder.encode(entity_texts)
            
            clusters = self._cluster_entity_embeddings(embeddings)
            
            logger.debug(f"   DBSCAN produced {len(clusters)} clusters")
            
            deduplicated = []
            llm_tasks = []
            
            for cluster_id, cluster_indices in enumerate(clusters):
                cluster_entities = [raw_entities[i] for i in cluster_indices]
                
                if len(cluster_entities) == 1:
                    deduplicated.append(
                        self._convert_raw_to_extracted_entity(
                            cluster_entities[0], 
                            f"entity_{cluster_id}"
                        )
                    )
                else:
                    llm_tasks.append((cluster_id, cluster_entities))

            if llm_tasks:
                max_workers = min(max(1, self.config.parallel_workers), len(llm_tasks))
                if max_workers == 1:
                    for cluster_id, cluster_entities in llm_tasks:
                        deduplicated.extend(self._merge_entity_cluster_with_size_guard(cluster_entities, cluster_id, custom_prompt))
                else:
                    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="EntityDedup") as executor:
                        future_to_cluster = {
                            executor.submit(
                                self._merge_entity_cluster_with_size_guard,
                                cluster_entities,
                                cluster_id,
                                custom_prompt,
                            ): (cluster_id, cluster_entities)
                            for cluster_id, cluster_entities in llm_tasks
                        }
                        for future in as_completed(future_to_cluster):
                            cluster_id, cluster_entities = future_to_cluster[future]
                            try:
                                deduplicated.extend(future.result())
                            except Exception as exc:
                                logger.warning(f"Concurrent deduplication failed for entity cluster {cluster_id}: {exc}")
                                deduplicated.extend(
                                    self._convert_raw_to_extracted_entity(entity, f"entity_{cluster_id}_{i}")
                                    for i, entity in enumerate(cluster_entities)
                                )
            
            self._update_stats(entities_after_dedup=len(deduplicated))
            logger.info(f"    Deduplicated entities: {len(deduplicated)}")
            return deduplicated
            
        else:
            logger.debug("   Using simple name-based deduplication...")
            deduplicated = self._simple_name_dedup(raw_entities)
            self._update_stats(entities_after_dedup=len(deduplicated))
            logger.info(f"    Deduplicated entities: {len(deduplicated)}")
            return deduplicated

    def _cluster_entity_embeddings(self, embeddings: Any) -> List[List[int]]:
        """Run cluster entity embeddings."""
        embedding_array = np.asarray(embeddings)
        if embedding_array.ndim != 2 or embedding_array.shape[0] == 0:
            return []

        eps = self.config.dbscan_eps
        min_samples = self.config.dbscan_min_samples
        if self.config.auto_optimize_dbscan and embedding_array.shape[0] >= 10:
            optimized = self._optimize_dbscan_params(embedding_array)
            if optimized:
                eps = optimized.get("eps", eps)
                min_samples = optimized.get("min_samples", min_samples)
                logger.info(f"   DBSCAN parameter tuning: eps={eps:.3f}, min_samples={min_samples}")

        labels = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric="cosine",
        ).fit_predict(embedding_array)

        clusters_by_label: Dict[Union[int, str], List[int]] = defaultdict(list)
        noise_count = 0
        for entity_index, raw_label in enumerate(labels):
            label = int(raw_label)
            if label == -1:
                label_key: Union[int, str] = f"noise_{noise_count}"
                noise_count += 1
            else:
                label_key = label
            clusters_by_label[label_key].append(entity_index)

        return list(clusters_by_label.values())

    @staticmethod
    def _entity_clustering_text(entity: Dict[str, Any]) -> str:
        """Build the same rich entity text used by the offline LoCoMo maker."""
        name = str(entity.get("name", ""))
        entity_type = str(entity.get("type") or entity.get("entity_type") or "")
        content = str(entity.get("content") or entity.get("context") or "")[:100]
        return " ".join(part for part in (name, entity_type, content) if part).strip() or name

    def _optimize_dbscan_params(self, embeddings: Any) -> Optional[Dict[str, Any]]:
        """Small sklearn-only DBSCAN parameter search for online LoCoMo alignment."""
        try:
            from sklearn.cluster import DBSCAN
            from sklearn.metrics import silhouette_score
        except Exception:
            return None

        embedding_array = np.asarray(embeddings)
        best_score = -1.0
        best_params: Optional[Dict[str, Any]] = None
        eps_low, eps_high = self.config.dbscan_eps_range
        min_low, min_high = self.config.dbscan_min_samples_range
        for eps in np.linspace(eps_low, eps_high, 10):
            for min_samples in range(min_low, min_high + 1):
                labels = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit_predict(embedding_array)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                if n_clusters < 2:
                    continue
                try:
                    score = silhouette_score(embedding_array, labels)
                except Exception:
                    continue
                if score > best_score:
                    best_score = float(score)
                    best_params = {"eps": float(eps), "min_samples": int(min_samples)}
        return best_params
    
    def extract_relations_from_entities(self,
                                        l0_units: List["MemoryUnit"],
                                        entities: List[ExtractedEntity],
                                        session_type: str = "default",
                                        custom_prompt: Optional[str] = None) -> List[ExtractedRelation]:
        """Extract relations from entities."""
        if not entities:
            logger.warning("No entities; skipping relation extraction")
            return []
        
        if not self.config.enable_relation_extraction:
            logger.info("Relation extraction is disabled")
            return []
        
        logger.info(f"Starting relation extraction ({len(entities)} entities)")
        
        content = self._merge_l0_content(l0_units)
        
        entity_list = [
            {"id": e.entity_id, "name": e.name, "type": e.entity_type}
            for e in entities
        ]
        
        prompt = self.prompt_manager.get_relation_extraction_prompt_v2(
            content=content,
            entities=entity_list,
            session_type=session_type,
            custom_prompt=custom_prompt
        )
        
        try:
            response = self.llm_client.generate_answer(
                prompt=prompt,
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens,
                json_format=True
            )
            self._update_stats(llm_calls=1)
            
            parsed = self._safe_parse_json(response)
            relations_data = parsed.get("relations", [])
            
            entity_name_to_id = {e.name.lower(): e.entity_id for e in entities}
            relations = []
            
            for i, rel_data in enumerate(relations_data):
                source_id = self._resolve_entity_reference(
                    rel_data.get("source_entity", ""), entity_name_to_id
                )
                target_id = self._resolve_entity_reference(
                    rel_data.get("target_entity", ""), entity_name_to_id
                )
                
                if source_id and target_id:
                    relation = ExtractedRelation(
                        relation_id=f"rel_{i}",
                        source_entity_id=source_id,
                        relation_type=rel_data.get("relation_type", "related_to"),
                        target_entity_id=target_id,
                        context=rel_data.get("context", ""),
                        temporal_info=rel_data.get("temporal_info"),
                        confidence=rel_data.get("confidence", 0.8)
                    )
                    relations.append(relation)
            
            relations = [
                r for r in relations 
                if r.confidence >= self.config.relation_confidence_threshold
            ]
            
            self._update_stats(
                relations_extracted=len(relations_data),
                relations_after_filter=len(relations)
            )
            
            logger.info(f"    Extracted {len(relations)} valid relations")
            return relations
            
        except Exception as e:
            logger.error(f"Relation extraction failed: {e}")
            self._update_stats(errors=1)
            return []
    
    def add_to_semantic_system(self,
                               entities: List[ExtractedEntity],
                               relations: Optional[List[ExtractedRelation]] = None,
                               source_id: str = "default",
                               space_name: Optional[str] = None,
                               graph_writer: Optional[GraphWriteQueue] = None,
                               wait_for_completion: bool = True) -> Dict[str, Any]:
        """Add to semantic system."""
        if not self.semantic_system:
            logger.warning("semantic_system is not configured; skipping insertion")
            return {"entities_added": 0, "relations_added": 0}
        
        space = space_name or self.config.entity_space_name
        
        logger.info(f"Adding to SemanticSystem (space={space})")
        
        result = {
            "entities_added": 0,
            "relations_added": 0,
            "errors": []
        }
        
        from ..core.memory_unit import MemoryUnit
        write_requests: List[GraphWriteRequest] = []
        
        for entity in entities:
            try:
                unit = MemoryUnit(
                    uid=f"{source_id}_{entity.entity_id}",
                    raw_data={
                        "text_content": entity.name,
                        "entity_type": entity.entity_type,
                        "aliases": entity.aliases,
                        "mentions": [m.to_dict() for m in entity.mentions]
                    },
                    metadata={
                        "type": "entity",
                        "entity_type": entity.entity_type,
                        "confidence": entity.confidence,
                        "source_id": source_id,
                        "session_ids": entity.session_ids,
                        "mentions_count": len(entity.mentions),
                        "created_at": datetime.now().isoformat()
                    }
                )
                
                write_requests.append(GraphWriteRequest(
                    unit=unit,
                    explicit_content_for_embedding=entity.name,
                    content_type_for_embedding="text",
                    space_names=[space],
                    index_update_mode="none",
                    generate_sparse_embedding=False,
                    source="entity_relation_entity",
                    metadata={"entity_id": entity.entity_id, "source_id": source_id},
                ))
                result["entities_added"] += 1
                
            except Exception as e:
                error_msg = f"Failed to add entity {entity.entity_id}: {e}"
                logger.warning(f" {error_msg}")
                result["errors"].append(error_msg)

        if write_requests:
            dispatch_graph_write_requests(
                semantic_system=self.semantic_system,
                requests=write_requests,
                graph_writer=graph_writer,
                wait_for_completion=wait_for_completion or bool(relations),
            )
        
        if relations and hasattr(self.semantic_system, 'add_relationship'):
            for relation in relations:
                try:
                    self.semantic_system.add_relationship(
                        source_uid=f"{source_id}_{relation.source_entity_id}",
                        target_uid=f"{source_id}_{relation.target_entity_id}",
                        relationship_name=relation.relation_type,
                        bidirectional=False,
                        context=relation.context,
                        temporal_info=relation.temporal_info,
                        confidence=relation.confidence
                    )
                    result["relations_added"] += 1
                    
                except Exception as e:
                    error_msg = f"Failed to add relation {relation.relation_id}: {e}"
                    logger.warning(f" {error_msg}")
                    result["errors"].append(error_msg)
        
        
        should_rebuild_index = graph_writer is None or wait_for_completion or bool(relations)
        if should_rebuild_index and hasattr(self.semantic_system, 'build_semantic_map_index'):
            try:
                self.semantic_system.build_semantic_map_index()
            except Exception as e:
                logger.warning(f"Index rebuild failed: {e}")
        
        self._update_stats(
            entities_added_to_graph=result["entities_added"],
            relations_added_to_graph=result["relations_added"]
        )
        
        logger.info(f"    Insertion completed: {result['entities_added']} entities, {result['relations_added']} relations")
        return result
    
    def run_full_pipeline(self,
                          l0_units: List["MemoryUnit"],
                          reference_date: Optional[str] = None,
                          source_id: str = "default",
                          session_type: str = "default",
                          add_to_system: bool = True,
                          custom_prompts: Optional[Dict[str, str]] = None,
                          graph_writer: Optional[GraphWriteQueue] = None) -> Dict[str, Any]:
        """Run full pipeline."""
        start_time = datetime.now()
        custom_prompts = custom_prompts or {}
        
        logger.info(f"\n{'='*60}")
        logger.info("Starting entity-relation extraction pipeline")
        logger.info(f"   - L0 units: {len(l0_units)}")
        logger.info(f"   - source_id: {source_id}")
        logger.info(f"   - extraction_style: {self.config.extraction_style}")
        logger.info(f"{'='*60}")
        
        result = {
            "success": False,
            "source_id": source_id,
            "entities": [],
            "relations": [],
            "stats": {},
            "errors": [],
            "processing_time": 0
        }
        
        try:
            logger.info("\nStep 1: extract entities")
            raw_entities = self.extract_entities_from_l0_units(
                l0_units=l0_units,
                reference_date=reference_date,
                source_id=source_id,
                session_type=session_type,
                custom_prompt=custom_prompts.get("entity_extraction")
            )
            
            if not raw_entities:
                result["errors"].append("No entities were extracted")
                return result
            
            logger.info("\nStep 2: deduplicate entities")
            entities = self.deduplicate_entities(
                raw_entities=raw_entities,
                custom_prompt=custom_prompts.get("entity_deduplication")
            )
            
            result["entities"] = [e.to_dict() for e in entities]
            
            if self.config.enable_relation_extraction:
                logger.info("\nStep 3: extract relations")
                relations = self.extract_relations_from_entities(
                    l0_units=l0_units,
                    entities=entities,
                    session_type=session_type,
                    custom_prompt=custom_prompts.get("relation_extraction")
                )
                result["relations"] = [r.to_dict() for r in relations]
            else:
                relations = []
            
            if add_to_system and self.semantic_system:
                logger.info("\nStep 4: add to SemanticSystem")
                add_result = self.add_to_semantic_system(
                    entities=entities,
                    relations=relations,
                    source_id=source_id,
                    graph_writer=graph_writer,
                    wait_for_completion=graph_writer is None,
                )
                result["add_result"] = add_result
            
            result["success"] = True
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            result["errors"].append(str(e))
        
        result["processing_time"] = (datetime.now() - start_time).total_seconds()
        result["stats"] = self.get_stats()
        
        logger.info(f"\n{'='*60}")
        logger.info("Pipeline completed")
        logger.info(f"   - entities: {len(result['entities'])}")
        logger.info(f"   - relations: {len(result['relations'])}")
        logger.info(f"   - elapsed: {result['processing_time']:.2f}s")
        logger.info(f"{'='*60}\n")
        
        return result
    
    
    def _safe_parse_json(self, response: str) -> Dict[str, Any]:
        """Run safe parse JSON."""
        if not response or not response.strip():
            logger.warning("Could not parse JSON response")
            return {"entities": [], "relations": [], "_parse_error": "empty_response"}

        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            
            try:
                from json_repair import repair_json
                repaired = repair_json(response)
                return json.loads(repaired)
            except Exception:
                pass
            
            logger.warning("Could not parse JSON response")
            return {"entities": [], "relations": [], "_parse_error": "invalid_json"}
    
    def _convert_raw_to_extracted_entity(self, 
                                         entity_data: Dict[str, Any], 
                                         entity_id: str) -> ExtractedEntity:
        """Convert raw to extracted entity."""
        entity_type = self._normalize_entity_type_for_style(
            entity_data.get("type", entity_data.get("entity_type", "UNKNOWN"))
        )
        mention = EntityMention(
            session_id=entity_data.get("source_id", "unknown"),
            unit_uid=",".join(entity_data.get("unit_uids", [])) if entity_data.get("unit_uids") else "unknown",
            content=entity_data.get("content", entity_data.get("context", "")),
            session_date=entity_data.get("session_date"),
            temporal_info=entity_data.get("temporal_info"),
            temporal_reference=entity_data.get("temporal_reference"),
            spatial_info=entity_data.get("spatial_info"),
            numerical_value=entity_data.get("numerical_value"),
            aliases=entity_data.get("aliases", []),
            confidence=entity_data.get("confidence", 0.8)
        )
        
        return ExtractedEntity(
            entity_id=entity_id,
            name=entity_data.get("name", "Unknown"),
            entity_type=entity_type,
            confidence=entity_data.get("confidence", 0.8),
            mentions=[mention],
            aliases=entity_data.get("aliases", []),
            related_entities=entity_data.get("related_entities", [])
        )

    def _normalize_entity_type_for_style(self, entity_type: Any, fallback: Optional[str] = None) -> str:
        """Normalize dataset-specific entity type variants without letting LLM merge invent categories."""
        normalized_type = str(entity_type or "").strip().upper()
        fallback_type = str(fallback or "").strip().upper()
        if self.config.extraction_style != "longmemeval":
            return normalized_type or fallback_type or "UNKNOWN"

        allowed_types = set(EntityType.longmemeval_types()) | self.LONGMEMEVAL_ADDITIONAL_ENTITY_TYPES

        def canonical(value: str) -> str:
            return self.LONGMEMEVAL_ENTITY_TYPE_ALIASES.get(value, value)

        candidate = canonical(normalized_type)
        if candidate in allowed_types:
            return candidate

        fallback_candidate = canonical(fallback_type)
        if fallback_candidate in allowed_types:
            return fallback_candidate

        return EntityType.CONCEPT

    def _dominant_input_entity_type(self, cluster_entities: List[Dict[str, Any]]) -> str:
        type_counts: Dict[str, int] = defaultdict(int)
        type_order: List[str] = []
        for entity_data in cluster_entities:
            raw_type = entity_data.get("type", entity_data.get("entity_type", "UNKNOWN"))
            normalized_type = self._normalize_entity_type_for_style(raw_type)
            if normalized_type not in type_counts:
                type_order.append(normalized_type)
            type_counts[normalized_type] += 1
        if not type_order:
            return EntityType.CONCEPT if self.config.extraction_style == "longmemeval" else "UNKNOWN"
        return max(type_order, key=lambda item: type_counts[item])
    
    def _llm_merge_entity_cluster(self,
                                   cluster_entities: List[Dict[str, Any]],
                                   cluster_id: int,
                                   custom_prompt: Optional[str] = None) -> List[ExtractedEntity]:
        """Run LLM merge entity cluster."""
        try:
            entity_candidates = json.dumps(cluster_entities, indent=2, ensure_ascii=False)
            
            prompt = self.prompt_manager.get_entity_deduplication_prompt(
                cluster_id=cluster_id,
                entity_candidates=entity_candidates,
                cluster_size=len(cluster_entities),
                custom_prompt=custom_prompt
            )
            
            response = self.llm_client.generate_answer(
                prompt=prompt,
                temperature=0.2,
                max_tokens=2000,
                json_format=True
            )
            self._update_stats(llm_calls=1)
            
            parsed = self._safe_parse_json(response)
            merged_entities = parsed.get("merged_entities", [])
            
            if not merged_entities:
                return [
                    self._convert_raw_to_extracted_entity(e, f"entity_{cluster_id}_{i}")
                    for i, e in enumerate(cluster_entities)
                ]
            
            result = []
            preferred_entity_type = self._dominant_input_entity_type(cluster_entities)
            for i, merged in enumerate(merged_entities):
                entity_id = merged.get("entity_id", f"merged_{cluster_id}_{i}")
                merged_entity_type = self._normalize_entity_type_for_style(
                    merged.get("entity_type", merged.get("type", "UNKNOWN")),
                    fallback=preferred_entity_type,
                )
                if self.config.extraction_style == "longmemeval" and preferred_entity_type:
                    merged_entity_type = preferred_entity_type
                
                mentions = []
                for mention_data in merged.get("mentions", []):
                    mention = EntityMention(
                        session_id=mention_data.get("session_id", "unknown"),
                        unit_uid=mention_data.get("unit_uid", "unknown"),
                        content=mention_data.get("content", ""),
                        session_date=mention_data.get("session_date"),
                        temporal_info=mention_data.get("temporal_info"),
                        temporal_reference=mention_data.get("temporal_reference"),
                        spatial_info=mention_data.get("spatial_info"),
                        numerical_value=mention_data.get("numerical_value"),
                        aliases=mention_data.get("aliases", []),
                        confidence=mention_data.get("confidence", 0.8)
                    )
                    mentions.append(mention)
                
                if not mentions:
                    for orig in cluster_entities:
                        mention = EntityMention(
                            session_id=orig.get("source_id", "unknown"),
                            unit_uid=",".join(orig.get("unit_uids", [])) if orig.get("unit_uids") else "unknown",
                            content=orig.get("content", ""),
                            session_date=orig.get("session_date"),
                            temporal_info=orig.get("temporal_info"),
                            temporal_reference=orig.get("temporal_reference"),
                            spatial_info=orig.get("spatial_info"),
                            numerical_value=orig.get("numerical_value"),
                            aliases=orig.get("aliases", []),
                            confidence=orig.get("confidence", 0.8)
                        )
                        mentions.append(mention)
                
                entity = ExtractedEntity(
                    entity_id=entity_id,
                    name=merged.get("name", merged.get("canonical_content", "Unknown")),
                    entity_type=merged_entity_type,
                    confidence=merged.get("confidence", 0.8),
                    mentions=mentions,
                    aliases=merged.get("aliases", []),
                    merge_reasoning=merged.get("merge_reasoning")
                )
                result.append(entity)
            
            return result
            
        except Exception as e:
            logger.warning(f"LLM merge failed: {e}")
            return [
                self._convert_raw_to_extracted_entity(e, f"entity_{cluster_id}_{i}")
                for i, e in enumerate(cluster_entities)
            ]

    def _merge_entity_cluster_with_size_guard(
        self,
        cluster_entities: List[Dict[str, Any]],
        cluster_id: int,
        custom_prompt: Optional[str] = None,
    ) -> List[ExtractedEntity]:
        """Split large entity clusters like the offline LoCoMo maker before LLM merge."""
        if len(cluster_entities) <= self.config.large_cluster_threshold:
            return self._llm_merge_entity_cluster(cluster_entities, cluster_id, custom_prompt)

        merged: List[ExtractedEntity] = []
        batch_size = self.config.large_cluster_threshold
        for start in range(0, len(cluster_entities), batch_size):
            batch = cluster_entities[start : start + batch_size]
            batch_id = f"{cluster_id}_batch_{start // batch_size}"
            merged.extend(self._llm_merge_entity_cluster(batch, batch_id, custom_prompt))
        return merged
    
    def _simple_name_dedup(self, raw_entities: List[Dict[str, Any]]) -> List[ExtractedEntity]:
        """Run simple name dedup."""
        unique_entities = []
        seen_names = set()
        
        for i, entity in enumerate(raw_entities):
            name = entity.get("name", "").lower().strip()
            if name and name not in seen_names:
                seen_names.add(name)
                unique_entities.append(
                    self._convert_raw_to_extracted_entity(entity, f"entity_{i}")
                )
        
        return unique_entities
    
    def _resolve_entity_reference(self, 
                                  reference: str, 
                                  name_to_id: Dict[str, str]) -> Optional[str]:
        """Resolve entity reference."""
        if not reference:
            return None
        
        if reference.startswith("entity_") or reference.startswith("E"):
            return reference
        
        reference_lower = reference.lower()
        if reference_lower in name_to_id:
            return name_to_id[reference_lower]
        
        for name, entity_id in name_to_id.items():
            if name in reference_lower or reference_lower in name:
                return entity_id
        
        logger.debug(f"Could not resolve entity reference: {reference}")
        return None
    
    def _merge_l0_content(self, l0_units: List["MemoryUnit"]) -> str:
        """Run merge L0 content."""
        return build_l0_inference_context(l0_units)
    
    
    
    def get_build_stats(self) -> Dict:
        """Return build stats."""
        return self.get_stats()
    
    def extract_and_build_for_units(self,
                                    l0_units: List["MemoryUnit"],
                                    source_id: str = "default",
                                    **kwargs) -> Dict[str, Any]:
        """Extract and build for units."""
        return self.run_full_pipeline(
            l0_units=l0_units,
            source_id=source_id,
            **kwargs
        )
