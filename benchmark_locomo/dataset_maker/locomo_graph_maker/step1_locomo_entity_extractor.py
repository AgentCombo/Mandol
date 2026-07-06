"""Utilities for step1 locomo entity extractor."""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import numpy as np
from sentence_transformers import SentenceTransformer


from mandol.llm.llm_client import LLMClient
from mandol.cluster.dbscan_method import find_clusters_with_dbscan, optimize_dbscan_parameters
from mandol.core import paths

class EntityType:
    
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    
    EVENT = "EVENT"
    ACTIVITY = "ACTIVITY"
    
    CONCEPT = "CONCEPT"
    EMOTION = "EMOTION"
    
    LOCATION = "LOCATION"
    DATE_TIME = "DATE_TIME"
    NUMERICAL_VALUE = "NUMERICAL_VALUE"
    
    OBJECT = "OBJECT"
    SKILL = "SKILL"
    RELATIONSHIP = "RELATIONSHIP"
    GOAL = "GOAL"

    @classmethod
    def get_all_types(cls) -> List[str]:
        """Return all types."""
        return [
            cls.PERSON, cls.ORGANIZATION, cls.EVENT, cls.ACTIVITY,
            cls.CONCEPT, cls.EMOTION, cls.LOCATION, cls.DATE_TIME,
            cls.NUMERICAL_VALUE, cls.OBJECT, cls.SKILL, cls.RELATIONSHIP, cls.GOAL
        ]


@dataclass
class EntityMention:
    session_id: str
    context: str
    temporal_info: Optional[str] = None
    spatial_info: Optional[str] = None
    aliases: List[str] = None
    confidence: float = 0.8
    
    def __post_init__(self):
        if self.aliases is None:
            self.aliases = []

@dataclass
class ExtractedEntity:
    entity_id: str
    name: str
    entity_type: str
    confidence: float
    mentions: List[EntityMention]
    extraction_metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.extraction_metadata is None:
            self.extraction_metadata = {}
    
    @property
    def sessions(self) -> List[str]:
        """Run sessions."""
        return [mention.session_id for mention in self.mentions]
    
    @property
    def contexts(self) -> List[str]:
        """Run contexts."""
        return [mention.context for mention in self.mentions]
    
    @property
    def all_aliases(self) -> List[str]:
        """Run all aliases."""
        aliases = set()
        for mention in self.mentions:
            aliases.update(mention.aliases)
        return sorted(list(aliases))

class LoCoMoEntityExtractor:
    
    def __init__(self,
                 llm_client: Optional[LLMClient] = None,
                 dedup_llm_client: Optional[LLMClient] = None,
                 entity_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 similarity_threshold: float = 0.85,
                 enable_cross_session_dedup: bool = True,
                 parallel_workers: int = 10):
        self.logger = logging.getLogger(__name__)
        self.similarity_threshold = similarity_threshold
        self.enable_cross_session_dedup = enable_cross_session_dedup
        self.parallel_workers = parallel_workers
        
        self.llm_client = llm_client or LLMClient(model_name="qwen-3.5-plus-thinking")
        self.dedup_llm_client = dedup_llm_client or LLMClient(model_name="deepseek-v3.2-dashscope")
        
        self.logger.info(f"加载实体嵌入模型: {entity_embedding_model}")
        self.entity_encoder = SentenceTransformer(entity_embedding_model)
        
        self.stats_lock = Lock()
        self.stats = {
            "conversations_processed": 0,
            "sessions_processed": 0,
            "raw_entities_extracted": 0,
            "entities_after_dedup": 0,
            "llm_calls_made": 0,
            "entity_extraction_calls": 0,
            "entity_merging_calls": 0,
            "clusters_processed": 0,
            "processing_time": 0.0,
        }
        
        self.prompts = self._prepare_entity_extraction_prompts()
        
        self.logger.info(f" LoCoMo实体抽取器初始化完成")
    
    def _update_stats(self, **kwargs):
        """Run update stats."""
        with self.stats_lock:
            for key, value in kwargs.items():
                if key in self.stats:
                    if isinstance(self.stats[key], (int, float)):
                        self.stats[key] += value
                    else:
                        self.stats[key] = value

    def _prepare_entity_extraction_prompts(self) -> Dict[str, str]:
        """Run prepare entity extraction prompts."""
        
        entity_extraction = """
        You are a professional entity recognition expert specializing in multi-hop question answering tasks. 
        
        Your task is to extract ALL important entities from the given conversation session text, following a rich and specific entity type schema designed for question answering scenarios.

        **Session Information:**
        Session ID: {session_id}
        Session Time: {session_time}
        Participants: {speakers}

        **Conversation Text:**
        {conversation_text}

        **Entity Type Schema:**
        {entity_types_description}

        **Task Requirements:**
        1. Extract entities that are crucial for answering Who/What/When/Where/Why/How questions
        2. Pay special attention to LOCATION, DATE_TIME, and NUMERICAL_VALUE entities - these are key for multi-hop reasoning
        3. Standardize entity names (e.g., convert relative time to absolute time when possible)
        4. Provide rich content descriptions for each entity

        **Output Format (JSON):**
        {{
            "entities": [
                {{
                    "entity_id": "E1",
                    "name": "Standardized entity name",
                    "content": "Rich description of the entity with context",
                    "type": "ENTITY_TYPE",
                    "confidence": 0.95,
                    "temporal_info": "time information if applicable",
                    "spatial_info": "location information if applicable",
                    "aliases": ["alternative names or mentions"]
                }}
            ]
        }}

        **Critical Guidelines:**
        - For DATE_TIME entities: Convert relative time references to absolute dates when session time is available
        - For LOCATION entities: Extract specific geographic references, venue names, addresses
        - For NUMERICAL_VALUE entities: Include meaningful numbers with their context (durations, quantities, ages, etc.)
        - For PERSON entities: Include both explicit names and role-based references
        - For EVENT entities: Focus on specific occurrences with participants, time, and location
        
        Ensure all extracted entities are relevant for potential question answering scenarios.
        """

        entity_deduplication = """
        You are an expert in entity standardization and deduplication. The following are {cluster_size} similar entities grouped by semantic clustering. Please merge duplicates pointing to the same real object while preserving all session-specific information.

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
                            "context": "Session-specific description from original entity",
                            "temporal_info": "time info from this session",
                            "spatial_info": "location info from this session", 
                            "aliases": ["session-specific aliases"],
                            "confidence": 0.90
                        }},
                        {{
                            "session_id": "session_2", 
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

        return {
            "entity_extraction": entity_extraction,
            "entity_deduplication": entity_deduplication
        }

    def _get_entity_types_description(self) -> str:
        """Get entity types description."""
        return """
        **Core Entities:**
        - PERSON: Key individuals in conversations (names, roles, participants)
        - ORGANIZATION: Institutions, groups, companies, teams

        **Events & Activities:**
        - EVENT: Specific occurrences with time, place, participants (conferences, parades, meetings)
        - ACTIVITY: Ongoing or repeated actions, hobbies, practices (pottery, camping, counseling)

        **Concepts & Themes:**
        - CONCEPT: Abstract ideas, themes, values (mental health, identity, acceptance)
        - EMOTION: Expressed feelings, moods, emotional states (supportive, excited, nervous)

        **Key Attributes (HIGH PRIORITY for multi-hop QA):**
        - LOCATION: Geographic places, venues, addresses (Sweden, LGBTQ center, museum)
        - DATE_TIME: Specific dates, times, durations (May 20 2023, last week, 5 years)
        - NUMERICAL_VALUE: Meaningful numbers (ages, quantities, durations, scores)

        **Other Important Types:**
        - OBJECT: Physical items, tools, artifacts (necklace, painting, book)
        - SKILL: Abilities, expertise, talents (counseling, art, writing)
        - RELATIONSHIP: Social connections, roles (friend, family, colleague)
        - GOAL: Objectives, plans, aspirations (adoption, career change, helping others)
        """

    def load_conversation_data(self, conversation_file: str) -> Dict[str, Any]:
        """Load conversation data."""
        try:
            with open(conversation_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list) and len(data) > 0:
                return data[0]
            elif isinstance(data, dict):
                return data
            else:
                raise ValueError("Invalid conversation data format")
                
        except Exception as e:
            self.logger.error(f" Failed to load conversation data: {e}")
            return None

    def extract_session_content(self, conversation_data: Dict[str, Any], session_key: str) -> Dict[str, Any]:
        """Extract session content."""
        try:
            conversation = conversation_data.get("conversation", {})
            session_dialogues = conversation.get(session_key, [])
            session_time = conversation.get(f"{session_key}_date_time", "Unknown")
            
            if not session_dialogues:
                return None
            
            conversation_text = ""
            speakers = set()
            
            for dialogue in session_dialogues:
                speaker = dialogue.get("speaker", "Unknown")
                dia_id = dialogue.get("dia_id", "")
                
                speakers.add(speaker)
                
                enhanced_text = self._build_multimodal_dialogue_text(dialogue)
                
                conversation_text += f"{dia_id} | {speaker}: {enhanced_text}\n"
            
            return {
                "session_id": session_key,
                "session_time": session_time,
                "speakers": list(speakers),
                "conversation_text": conversation_text.strip(),
                "dialogues": session_dialogues,
                "dialogue_count": len(session_dialogues)
            }
            
        except Exception as e:
            self.logger.error(f" Failed to extract session content {session_key}: {e}")
            return None

    def _build_multimodal_dialogue_text(self, dialogue: Dict[str, Any]) -> str:
        """Build a text block that preserves dialogue and image metadata.

        The returned string appends available image captions and retrieval
        queries so entity extraction can use multimodal evidence without
        changing the downstream prompt interface.
        """
        text = dialogue.get("text", "").strip()
        
        has_multimodal = bool(
            dialogue.get("img_url") or 
            dialogue.get("blip_caption") or 
            dialogue.get("query")
        )
        
        if not has_multimodal:
            return text
        
        text_parts = []
        
        if text:
            text_parts.append(f"[dialogue] {text}")
        
        blip_caption = dialogue.get("blip_caption", "").strip()
        if blip_caption:
            text_parts.append(f"[image_description] {blip_caption}")
        
        image_keywords = dialogue.get("query", "").strip()
        if image_keywords:
            text_parts.append(f"[image_keywords] {image_keywords}")
        
        enhanced_text = '\n'.join(text_parts)
        
        return enhanced_text

    def extract_entities_from_session(self, session_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract entities from session."""
        try:
            session_id = session_data["session_id"]
            self.logger.info(f" Extracting entities from {session_id}")
            
            prompt = self.prompts["entity_extraction"].format(
                session_id=session_data["session_id"],
                session_time=session_data["session_time"],
                speakers=", ".join(session_data["speakers"]),
                conversation_text=session_data["conversation_text"],
                entity_types_description=self._get_entity_types_description()
            )
            
            response = self.llm_client.generate_answer(
                prompt,
                temperature=0.1,
                json_format=True
            )
            
            self._update_stats(llm_calls_made=1, entity_extraction_calls=1)
            
            try:
                result = json.loads(response)
                entities = result.get("entities", [])
                
                for entity in entities:
                    original_eid = entity.get("entity_id", "unknown_id")
                    entity['entity_id'] = f"{session_id}_{original_eid}"
                    
                    entity["session"] = session_id
                    entity["session_time"] = session_data["session_time"]
                    entity["original_entity_id"] = original_eid
                
                self.logger.info(f" Extracted {len(entities)} entities from {session_id}")
                return entities
                
            except json.JSONDecodeError as e:
                self.logger.error(f" JSON parsing failed for {session_id}: {e}")
                try:
                    import re
                    json_match = re.search(r'\{.*\}', response, re.DOTALL)
                    if json_match:
                        json_str = json_match.group()
                        result = json.loads(json_str)
                        entities = result.get("entities", [])
                        
                        for entity in entities:
                            original_eid = entity.get("entity_id", "unknown_id")
                            entity['entity_id'] = f"{session_id}_{original_eid}"
                            entity["session"] = session_id
                            entity["session_time"] = session_data["session_time"]
                            entity["original_entity_id"] = original_eid
                        
                        self.logger.info(f" Recovered {len(entities)} entities from {session_id}")
                        return entities
                    else:
                        return []
                except Exception:
                    return []
                    
        except Exception as e:
            self.logger.error(f" Entity extraction failed for {session_data.get('session_id', 'unknown')}: {e}")
            return []
        
    def deduplicate_entities_dbscan(self, all_entities: List[Dict[str, Any]]) -> List[ExtractedEntity]:
        """Deduplicate entities dbscan."""
        if not all_entities:
            return []
        
        try:
            self.logger.info(f" 开始DBSCAN实体去重: {len(all_entities)} 个实体")
            
            if not self.enable_cross_session_dedup:
                return self._convert_to_extracted_entities(all_entities)
            
            entities_with_embeddings = self._prepare_entities_for_clustering(all_entities)
            if len(entities_with_embeddings) < 2:
                self.logger.warning("实体数量太少，跳过聚类")
                return self._convert_to_extracted_entities(all_entities)
            
            optimization_result = optimize_dbscan_parameters(
                entities_with_embeddings,
                eps_range=(0.15, 0.6),
                min_samples_range=(2, 6),
                metric='cosine',
                n_trials=15
            )
            
            best_params = optimization_result['best_params']
            self.logger.info(f" DBSCAN最优参数: eps={best_params['eps']:.3f}, min_samples={best_params['min_samples']}")
            
            clusters = find_clusters_with_dbscan(
                entities_with_embeddings,
                eps=best_params['eps'],
                min_samples=best_params['min_samples'],
                metric='cosine',
                normalize_embeddings=True,
                use_metadata_features=True
            )
            
            self.logger.info(f" 聚类完成: {len(clusters)} 个聚类（包含噪声）")
            
            deduplicated_entities = []
            uid_to_entity = {self._generate_entity_temp_uid(all_entities[i]): all_entities[i] for i in range(len(all_entities))}
            
            llm_tasks = []  # (cluster_entities, cluster_id, task_type)
            direct_entities = []
            
            for cluster_id, entity_uids in clusters.items():
                cluster_entities = [uid_to_entity[uid] for uid in entity_uids if uid in uid_to_entity]
                
                if cluster_id == -1:
                    for entity_data in cluster_entities:
                        extracted_entity = self._convert_single_entity(entity_data, f"noise_{len(direct_entities)}")
                        direct_entities.append(extracted_entity)
                elif len(cluster_entities) <= 1:
                    for entity_data in cluster_entities:
                        extracted_entity = self._convert_single_entity(entity_data, f"single_{len(direct_entities)}")
                        direct_entities.append(extracted_entity)
                elif len(cluster_entities) <= 12:
                    llm_tasks.append((cluster_entities, cluster_id, "small_cluster"))
                else:
                    llm_tasks.append((cluster_entities, cluster_id, "large_cluster"))
            
            deduplicated_entities.extend(direct_entities)
            
            if llm_tasks:
                self.logger.info(f" 开始多线程LLM去重处理: {len(llm_tasks)} 个聚类")
                
                with ThreadPoolExecutor(max_workers=min(self.parallel_workers, len(llm_tasks)), 
                                    thread_name_prefix="LLMDedup") as executor:
                    future_to_task = {}
                    
                    for cluster_entities, cluster_id, task_type in llm_tasks:
                        if task_type == "small_cluster":
                            future = executor.submit(self._llm_deduplicate_cluster, cluster_entities, cluster_id)
                        else:  # large_cluster
                            future = executor.submit(self._process_large_cluster, cluster_entities, cluster_id)
                        
                        future_to_task[future] = (cluster_id, task_type)
                    
                    for future in as_completed(future_to_task):
                        cluster_id, task_type = future_to_task[future]
                        try:
                            cluster_result = future.result()
                            deduplicated_entities.extend(cluster_result)
                            self.logger.debug(f" 聚类 {cluster_id} ({task_type}) 处理完成: {len(cluster_result)} 个实体")
                        except Exception as e:
                            self.logger.error(f" 聚类 {cluster_id} ({task_type}) 处理失败: {e}")
            
            if len(deduplicated_entities) > 30:
                self.logger.info(" 执行最终相似度检查...")
                final_entities = self._final_similarity_check(deduplicated_entities)
            else:
                final_entities = deduplicated_entities
            
            self._update_stats(
                entities_after_dedup=len(final_entities),
                clusters_processed=len(clusters)
            )
            
            self.logger.info(f" DBSCAN去重完成: {len(all_entities)} -> {len(final_entities)}")
            self.logger.info(f"    LLM任务处理: {len(llm_tasks)} 个聚类")
            return final_entities
            
        except Exception as e:
            self.logger.error(f" DBSCAN去重失败: {e}")
            return self._convert_to_extracted_entities(all_entities)

    def _prepare_entities_for_clustering(self, entities: List[Dict[str, Any]]) -> List:
        """Run prepare entities for clustering."""
        try:
            from mandol.core.memory_unit import MemoryUnit
        except ImportError:
            class SimpleMemoryUnit:
                def __init__(self, uid, raw_data, metadata):
                    self.uid = uid
                    self.raw_data = raw_data
                    self.metadata = metadata
                    self.embedding = None
            MemoryUnit = SimpleMemoryUnit
        
        entities_with_embeddings = []
        
        for i, entity in enumerate(entities):
            entity_text = f"{entity['name']} {entity['type']}"
            if entity.get('content'):
                entity_text += f" {entity['content'][:100]}"
            
            try:
                embedding = self.entity_encoder.encode([entity_text], convert_to_numpy=True)[0]
            except Exception as e:
                self.logger.warning(f"实体 {entity['name']} 嵌入生成失败: {e}")
                continue
            
            temp_uid = self._generate_entity_temp_uid(entity)
            metadata = {
                'content_type': 'entity',
                'entity_type': entity['type'],
                'session': entity['session'],
                'confidence': entity.get('confidence', 0.8)
            }
            
            memory_unit = MemoryUnit(
                uid=temp_uid,
                raw_data={'text': entity['name'], 'type': entity['type']},
                metadata=metadata
            )
            memory_unit.embedding = embedding
            entities_with_embeddings.append(memory_unit)
        
        return entities_with_embeddings

    def _generate_entity_temp_uid(self, entity: Dict[str, Any]) -> str:
        """Generate entity temp UID."""
        import hashlib
        content = f"{entity['name']}_{entity['type']}_{entity['session']}"
        return f"temp_entity_{hashlib.md5(content.encode()).hexdigest()[:8]}"


    def _llm_deduplicate_cluster(self, cluster_entities: List[Dict[str, Any]], cluster_id: int) -> List[ExtractedEntity]:
        """Run LLM deduplicate cluster."""
        try:
            entity_candidates_text = ""
            for i, entity in enumerate(cluster_entities, 1):
                entity_candidates_text += f"{i}. Name: \"{entity['name']}\"\n"
                entity_candidates_text += f"   Type: {entity['type']}\n"
                entity_candidates_text += f"   Session: {entity['session']}\n"
                entity_candidates_text += f"   Context: {entity.get('content', '')[:100]}...\n"
                entity_candidates_text += f"   Temporal: {entity.get('temporal_info', '')}\n"
                entity_candidates_text += f"   Spatial: {entity.get('spatial_info', '')}\n"
                entity_candidates_text += f"   Aliases: {entity.get('aliases', [])}\n"
                entity_candidates_text += f"   Confidence: {entity.get('confidence', 0.8)}\n\n"
            
            prompt = self.prompts["entity_deduplication"].format(
                cluster_size=len(cluster_entities),
                cluster_id=cluster_id,
                entity_candidates=entity_candidates_text
            )
            
            response = self.dedup_llm_client.generate_answer(
                prompt,
                temperature=0.05,
                json_format=True
            )
            
            self._update_stats(llm_calls_made=1, entity_merging_calls=1)
            
            try:
                result = json.loads(response)
                merged_entities = result.get("merged_entities", [])
                
                extracted_entities = []
                for i, merged in enumerate(merged_entities):
                    extracted_entity = self._create_merged_entity(merged, cluster_entities, f"cluster_{cluster_id}_{i}")
                    extracted_entities.append(extracted_entity)
                
                return extracted_entities
                    
            except json.JSONDecodeError as e:
                self.logger.warning(f" 聚类 {cluster_id} JSON解析失败: {e}")
                try:
                    import re
                    json_match = re.search(r'\{.*\}', response, re.DOTALL)
                    if json_match:
                        json_str = json_match.group()
                        result = json.loads(json_str)
                        merged_entities = result.get("merged_entities", [])
                        
                        extracted_entities = []
                        for i, merged in enumerate(merged_entities):
                            extracted_entity = self._create_merged_entity(merged, cluster_entities, f"cluster_{cluster_id}_{i}")
                            extracted_entities.append(extracted_entity)
                        
                        return extracted_entities
                    else:
                        return [self._convert_single_entity(entity, f"fallback_{cluster_id}_{i}") 
                            for i, entity in enumerate(cluster_entities)]
                except Exception:
                    return [self._convert_single_entity(entity, f"fallback_{cluster_id}_{i}") 
                        for i, entity in enumerate(cluster_entities)]
                    
        except Exception as e:
            self.logger.error(f" 聚类 {cluster_id} LLM去重失败: {e}")
            return [self._convert_single_entity(entity, f"error_{cluster_id}_{i}") 
                for i, entity in enumerate(cluster_entities)]
        
    def _process_large_cluster(self, cluster_entities: List[Dict[str, Any]], cluster_id: int) -> List[ExtractedEntity]:
        """Process large cluster."""
        batch_size = 12
        batches = []
        
        for i in range(0, len(cluster_entities), batch_size):
            batch = cluster_entities[i:i + batch_size]
            batch_id = f"{cluster_id}_batch_{i//batch_size}"
            batches.append((batch, batch_id))
        
        self.logger.debug(f" 大聚类 {cluster_id} 分为 {len(batches)} 个批次处理")
        
        all_merged = []
        
        max_batch_workers = min(10, len(batches))
        with ThreadPoolExecutor(max_workers=max_batch_workers, 
                            thread_name_prefix=f"BatchDedup_{cluster_id}") as executor:
            future_to_batch = {
                executor.submit(self._llm_deduplicate_cluster, batch, batch_id): batch_id
                for batch, batch_id in batches
            }
            
            for future in as_completed(future_to_batch):
                batch_id = future_to_batch[future]
                try:
                    batch_result = future.result()
                    all_merged.extend(batch_result)
                    self.logger.debug(f" 批次 {batch_id} 处理完成: {len(batch_result)} 个实体")
                except Exception as e:
                    self.logger.error(f" 批次 {batch_id} 处理失败: {e}")
        
        if len(all_merged) > 20:
            self.logger.debug(f" 大聚类 {cluster_id} 批处理后仍有 {len(all_merged)} 个实体，进行相似度合并")
            final_entities = self._final_similarity_check(all_merged)
        else:
            final_entities = all_merged
        
        self.logger.debug(f" 大聚类 {cluster_id} 处理完成: {len(cluster_entities)} -> {len(final_entities)}")
        return final_entities

    def _final_similarity_check(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Run final similarity check."""
        try:
            if not entities or len(entities) <= 1:
                return entities
            
            entity_texts = [f"{entity.name} {entity.entity_type}" for entity in entities]
            embeddings = self.entity_encoder.encode(entity_texts, convert_to_numpy=True)
            
            try:
                from sklearn.metrics.pairwise import cosine_similarity
                similarity_matrix = cosine_similarity(embeddings)
            except ImportError:
                return entities
            
            final_entities = []
            used_indices = set()
            merge_threshold = 0.92
            
            for i in range(len(entities)):
                if i in used_indices:
                    continue
                    
                merge_group = [entities[i]]
                used_indices.add(i)
                
                for j in range(i + 1, len(entities)):
                    if j in used_indices:
                        continue
                    if (similarity_matrix[i][j] > merge_threshold and 
                        entities[i].entity_type == entities[j].entity_type):
                        merge_group.append(entities[j])
                        used_indices.add(j)
                
                if len(merge_group) == 1:
                    final_entities.append(merge_group[0])
                else:
                    merged = self._merge_entity_group(merge_group, len(final_entities))
                    final_entities.append(merged)
            
            return final_entities
            
        except Exception as e:
            self.logger.error(f" 最终相似度检查失败: {e}")
            return entities

    def _create_merged_entity(self, merged_data: Dict[str, Any], source_entities: List[Dict[str, Any]], entity_id: str) -> ExtractedEntity:
        """Create merged entity."""
        mentions = []
        
        if 'mentions' in merged_data:
            for mention_data in merged_data['mentions']:
                mention = EntityMention(
                    session_id=mention_data.get('session_id', ''),
                    context=mention_data.get('context', ''),
                    temporal_info=mention_data.get('temporal_info'),
                    spatial_info=mention_data.get('spatial_info'),
                    aliases=mention_data.get('aliases', []),
                    confidence=float(mention_data.get('confidence', 0.8))
                )
                mentions.append(mention)
        else:
            for src_entity in source_entities:
                mention = EntityMention(
                    session_id=src_entity['session'],
                    context=src_entity.get('content', ''),
                    temporal_info=src_entity.get('temporal_info'),
                    spatial_info=src_entity.get('spatial_info'),
                    aliases=src_entity.get('aliases', []),
                    confidence=float(src_entity.get('confidence', 0.8))
                )
                mentions.append(mention)
        
        avg_confidence = sum(mention.confidence for mention in mentions) / len(mentions) if mentions else 0.8
        
        return ExtractedEntity(
            entity_id=entity_id,
            name=merged_data.get("name", source_entities[0]['name']),
            entity_type=merged_data.get("entity_type", source_entities[0]['type']),
            confidence=float(merged_data.get("confidence", avg_confidence)),
            mentions=mentions,
            extraction_metadata=merged_data.get("extraction_metadata", {
                "merge_reasoning": "LLM cluster deduplication",
                "source_count": len(source_entities),
                "merge_method": "llm_cluster_dedup"
            })
        )
    
    def _convert_single_entity(self, entity_data: Dict[str, Any], entity_id: str) -> ExtractedEntity:
        """Convert single entity."""
        mention = EntityMention(
            session_id=entity_data['session'],
            context=entity_data.get('content', ''),
            temporal_info=entity_data.get('temporal_info'),
            spatial_info=entity_data.get('spatial_info'),
            aliases=entity_data.get('aliases', []),
            confidence=float(entity_data.get('confidence', 0.8))
        )
        
        return ExtractedEntity(
            entity_id=entity_id,
            name=entity_data['name'],
            entity_type=entity_data['type'],
            confidence=float(entity_data.get('confidence', 0.8)),
            mentions=[mention],
            extraction_metadata={
                "merge_method": "no_merge_needed",
                "source_count": 1
            }
        )

    def _convert_to_extracted_entities(self, entities: List[Dict[str, Any]]) -> List[ExtractedEntity]:
        """Convert to extracted entities."""
        return [self._convert_single_entity(entity, f"entity_{i}") for i, entity in enumerate(entities)]


    def _merge_entity_group(self, entities: List[ExtractedEntity], new_id: int) -> ExtractedEntity:
        """Run merge entity group."""
        canonical_text = max(entities, key=lambda e: len(e.name)).name
        
        all_mentions = []
        for entity in entities:
            all_mentions.extend(entity.mentions)
            
            if entity.name != canonical_text and all_mentions:
                if entity.name not in all_mentions[0].aliases:
                    all_mentions[0].aliases.append(entity.name)
        
        avg_confidence = sum(entity.confidence for entity in entities) / len(entities)
        
        return ExtractedEntity(
            entity_id=f"merged_{new_id}",
            name=canonical_text,
            entity_type=entities[0].entity_type,
            confidence=avg_confidence,
            mentions=all_mentions,
            extraction_metadata={
                "merge_method": "similarity_final_check",
                "source_count": len(entities)
            }
        )

    def process_conversation(self, conversation_file: str) -> Dict[str, Any]:
        """Process conversation."""
        start_time = datetime.now()
        
        self.logger.info(f" Processing conversation for entity extraction: {conversation_file}")
        
        
        conversation_data = self.load_conversation_data(conversation_file)
        if not conversation_data:
            return None
        
        conversation_id = conversation_data.get("sample_id", Path(conversation_file).stem)
        conversation = conversation_data.get("conversation", {})
        
        all_session_data = []
        for key in conversation.keys():
            if key.startswith("session_") and not key.endswith("_date_time"):
                session_data = self.extract_session_content(conversation_data, key)
                if session_data and session_data["dialogue_count"] > 0:
                    all_session_data.append(session_data)
        
        if not all_session_data:
            self.logger.warning(f" No valid sessions found in {conversation_id}")
            return None
        
        self.logger.info(f" Found {len(all_session_data)} sessions for entity extraction")
        
        all_entities = []
        session_metadata = {}
        
        with ThreadPoolExecutor(max_workers=self.parallel_workers) as executor:
            future_to_session = {
                executor.submit(self.extract_entities_from_session, session_data): session_data["session_id"]
                for session_data in all_session_data
            }
            
            for future in as_completed(future_to_session):
                session_id = future_to_session[future]
                try:
                    session_entities = future.result()
                    all_entities.extend(session_entities)
                    
                    session_data = next(s for s in all_session_data if s["session_id"] == session_id)
                    session_metadata[session_id] = {
                        "session_time": session_data["session_time"],
                        "speakers": session_data["speakers"],
                        "dialogue_count": session_data["dialogue_count"],
                        "entities_extracted": len(session_entities)
                    }
                    
                except Exception as e:
                    self.logger.error(f" Entity extraction failed for {session_id}: {e}")
        
        self.logger.info(f" Raw extraction completed: {len(all_entities)} entities")
        self._update_stats(raw_entities_extracted=len(all_entities))
        
        self.logger.info(" Starting DBSCAN deduplication...")

        deduplicated_entities = self.deduplicate_entities_dbscan(all_entities)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        self._update_stats(
            conversations_processed=1,
            sessions_processed=len(all_session_data),
            processing_time=processing_time
        )
        
        result = {
            "conversation_id": conversation_id,
            "extraction_metadata": {
                "extraction_method": "dbscan_clustering_deduplication_with_mentions",
                "llm_model": getattr(self.llm_client, 'model_name', 'unknown'),
                "processing_time_seconds": processing_time,
                "created_at": datetime.now().isoformat(),
                "raw_entities_count": len(all_entities),
                "deduplicated_entities_count": len(deduplicated_entities),
                "reduction_ratio": (len(all_entities) - len(deduplicated_entities)) / len(all_entities) if all_entities else 0,
                "data_format": "mentions_based"
            },
            "session_metadata": session_metadata,
            "entities": deduplicated_entities,
            "extraction_stats": self.stats.copy(),
            "entity_schema": {
                "types": EntityType.get_all_types(),
                "version": "v2.0_mentions_based"  
            }
        }
        
        return result

    def export_entities(self, entities_data: Dict[str, Any], output_path: str) -> bool:
        """Save entities."""
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            serializable_entities = []
            for entity in entities_data.get('entities', []):
                if isinstance(entity, ExtractedEntity):
                    entity_dict = {
                        "entity_id": entity.entity_id,
                        "name": entity.name,
                        "entity_type": entity.entity_type,
                        "confidence": entity.confidence,
                        "mentions": [
                            {
                                "session_id": mention.session_id,
                                "context": mention.context,
                                "temporal_info": mention.temporal_info,
                                "spatial_info": mention.spatial_info,
                                "aliases": mention.aliases,
                                "confidence": mention.confidence
                            }
                            for mention in entity.mentions
                        ],
                        "extraction_metadata": entity.extraction_metadata
                    }
                    serializable_entities.append(entity_dict)
                else:
                    serializable_entities.append(entity)
            
            entities_data['entities'] = serializable_entities
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(entities_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f" Entities exported to: {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f" Failed to export entities: {e}")
            return False

def main():
    """Run the command-line entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="LoCoMo实体抽取器 - 第一步：实体抽取和去重")
    
    parser.add_argument("--input-file", 
                       default=str(paths.LOCOMO_RAW_FILE),
                       help="输入的LoCoMo数据文件")
    parser.add_argument("--output-dir", 
                       default=str(paths.LOCOMO_ENTITY_RELATION_STEP1_DIR), 
                       help="输出目录路径")
    
    # Dataset-specific handling used by the reproduction workflow.
    parser.add_argument("--sample-id", 
                       type=str, 
                       nargs='+',
                       default=None,
                       help="处理特定样本ID（可指定多个，用空格分隔）")
    
    parser.add_argument("--limit", type=int, default=None,
                       help="限制处理的对话数量")
    
    parser.add_argument("--extract-model", 
                       default="qwen-3.5-plus-thinking",
                       help="抽取模型名称")
    parser.add_argument("--dedup-model", 
                       default="deepseek-v3.2-dashscope",
                       help="去重模型名称")
    parser.add_argument("--no-cross-session-dedup", action="store_true",
                       help="禁用跨会话去重")
    
    parser.add_argument("--parallel-workers", type=int, default=10,
                       help="并行处理工作线程数")
    
    parser.add_argument("--debug", action="store_true",
                       help="启用调试模式")
    
    args = parser.parse_args()
    
    # Avoid mutating LogRecord fields before other handlers process the record.
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 80)
    print(" LoCoMo实体抽取器 - 第一步：实体抽取和去重")
    print("=" * 80)
    print(f" 输入文件: {args.input_file}")
    print(f" 输出目录: {args.output_dir}")
    print(f" 抽取模型: {args.extract_model}")
    print(f" 去重模型: {args.dedup_model}")
    print(f" 跨会话去重: {not args.no_cross_session_dedup}")
    
    if args.sample_id:
        print(f" 指定样本ID: {', '.join(args.sample_id)}")
    
    try:
        extract_llm_client = LLMClient(model_name=args.extract_model)
        dedup_llm_client = LLMClient(model_name=args.dedup_model)
        extractor = LoCoMoEntityExtractor(
            llm_client=extract_llm_client,
            dedup_llm_client=dedup_llm_client,
            enable_cross_session_dedup=not args.no_cross_session_dedup,
            parallel_workers=args.parallel_workers
        )
        
        
        with open(args.input_file, 'r', encoding='utf-8') as f:
            locomo_data = json.load(f)
        
        conversations = locomo_data if isinstance(locomo_data, list) else [locomo_data]
        
        # Dataset-specific handling used by the reproduction workflow.
        if args.sample_id:
            sample_id_set = set(args.sample_id)
            conversations = [conv for conv in conversations 
                           if conv.get("sample_id") in sample_id_set]
            print(f" 找到 {len(conversations)} 个匹配的对话（共指定 {len(args.sample_id)} 个ID）")
            
            found_ids = {conv.get("sample_id") for conv in conversations}
            missing_ids = sample_id_set - found_ids
            if missing_ids:
                print(f"  未找到的样本ID: {', '.join(missing_ids)}")
            
        if args.limit:
            conversations = conversations[:args.limit]
        
        print(f" 将处理 {len(conversations)} 个对话")
        
        if len(conversations) == 0:
            print(" 没有对话需要处理，退出")
            return 1
        
        successful = 0
        failed = 0
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for i, conv_data in enumerate(conversations, 1):
            sample_id = conv_data.get("sample_id", f"conv_{i}")
            print(f"\n{'='*80}")
            print(f" 处理对话 {i}/{len(conversations)}: {sample_id}")
            print(f"{'='*80}")
            
            try:
                temp_file = output_path / f"{sample_id}_temp.json"
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(conv_data, f, ensure_ascii=False, indent=2)
                
                entities_result = extractor.process_conversation(str(temp_file))
                
                if entities_result:
                    entity_file = output_path / f"{sample_id}_entities.json"
                    success = extractor.export_entities(entities_result, str(entity_file))
                    
                    if success:
                        successful += 1
                        print(f" 对话 {sample_id} 实体抽取成功")
                        print(f"    原始实体数: {entities_result['extraction_metadata']['raw_entities_count']}")
                        print(f"    最终实体数: {len(entities_result['entities'])}")
                        print(f"    去重率: {entities_result['extraction_metadata']['reduction_ratio']:.2%}")
                        print(f"   Processing time: {entities_result['extraction_metadata']['processing_time_seconds']:.2f}s")
                    else:
                        failed += 1
                        print(f" 对话 {sample_id} 导出失败")
                else:
                    failed += 1
                    print(f" 对话 {sample_id} 实体抽取失败")
                
                if temp_file.exists():
                    temp_file.unlink()
                
            except Exception as e:
                failed += 1
                print(f" 对话 {sample_id} 处理异常: {e}")
                if args.debug:
                    import traceback
                    print(f"详细错误: {traceback.format_exc()}")
        
        print(f"\n{'='*80}")
        print(f" 第一步实体抽取完成!")
        print(f"{'='*80}")
        print(f" 处理总数: {len(conversations)}")
        print(f" 成功: {successful}")
        print(f" 失败: {failed}")
        print(f" 输出目录: {output_path}")
        print(f"{'='*80}")
        
        return 0 if failed == 0 else 1
        
    except Exception as e:
        print(f" 程序异常: {e}")
        if args.debug:
            import traceback
            print(f"详细错误: {traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    exit(main())
