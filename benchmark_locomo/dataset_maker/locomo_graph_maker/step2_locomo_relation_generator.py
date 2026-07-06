"""Utilities for step2 locomo relation generator."""

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
import networkx as nx


from mandol.llm.llm_client import LLMClient
from mandol.core import paths

@dataclass
class ExtractedRelation:
    relation_id: str
    head_entity_id: str
    tail_entity_id: str
    head_entity_name: str
    tail_entity_name: str
    relation_type: str
    confidence: float
    sessions: List[str]
    evidence_texts: List[str]
    contexts: List[str]
    temporal_context: Optional[str] = None
    extraction_metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.extraction_metadata is None:
            self.extraction_metadata = {}

@dataclass
class EntityRelationGraph:
    conversation_id: str
    entities: Dict[str, Dict[str, Any]]
    relations: List[ExtractedRelation]
    entity_graph: nx.DiGraph
    session_metadata: Dict[str, Any]
    extraction_metadata: Dict[str, Any]

class LoCoMoRelationGenerator:
    
    def __init__(self,
                 llm_client: Optional[LLMClient] = None,
                 parallel_workers: int = 10):
        self.logger = logging.getLogger(__name__)
        self.parallel_workers = parallel_workers
        
        self.llm_client = llm_client or LLMClient(model_name="qwen-3.5-plus-thinking")
        
        self.stats_lock = Lock()
        self.stats = {
            "conversations_processed": 0,
            "sessions_processed": 0,
            "total_relations_extracted": 0,
            "cross_session_relations_extracted": 0,
            "llm_calls_made": 0,
            "relation_extraction_calls": 0,
            "cross_session_extraction_calls": 0,
            "processing_time": 0.0,
        }
        
        self.prompts = self._prepare_relation_extraction_prompts()
        
        self.logger.info(f" LoCoMo关系生成器初始化完成 (mentions格式兼容)")
    
    def _update_stats(self, **kwargs):
        """Run update stats."""
        with self.stats_lock:
            for key, value in kwargs.items():
                if key in self.stats:
                    if isinstance(self.stats[key], (int, float)):
                        self.stats[key] += value
                    else:
                        self.stats[key] = value

    def _prepare_relation_extraction_prompts(self) -> Dict[str, str]:
        """Run prepare relation extraction prompts."""
        
        relation_extraction = """
        You are a professional relation extraction expert. Based on the conversation session and the extracted entities with their specific mentions in this session, discover and extract relationships between entities that are crucial for multi-hop question answering.

        **Session Information:**
        Session ID: {session_id}
        Session Time: {session_time}
        
        **Original Conversation Text:**
        {conversation_text}

        **Entities with Session-Specific Context:**
        {entities_list}

        **Task: Extract relationships that connect these entities in meaningful ways for question answering.**

        **Key Relationship Types for Multi-hop QA:**
        - Temporal Relations: "happens_before", "happens_after", "during", "lasts_for"
        - Spatial Relations: "located_at", "travels_to", "comes_from", "near"
        - Participation: "participates_in", "organizes", "attends", "performs"
        - Identity & Role: "is_a", "works_as", "identifies_as", "member_of"
        - Possession & Association: "owns", "has", "belongs_to", "associated_with"
        - Emotional & Social: "supports", "cares_for", "friends_with", "influences"
        - Causal Relations: "causes", "leads_to", "results_in", "enables"
        - Comparative: "similar_to", "different_from", "better_than"
        - Descriptive: "characterized_by", "described_as", "known_for"

        **Output Format (JSON):**
        {{
            "relations": [
                {{
                    "head_entity": "EXACT Entity ID from the list above (e.g., merged_0, noise_123)",
                    "tail_entity": "EXACT Entity ID from the list above (e.g., merged_1, noise_456)",
                    "relation_type": "relationship type",
                    "confidence": 0.90,
                    "evidence": "Direct quote or paraphrase supporting this relation",
                    "temporal_context": "when this relationship holds/occurred",
                    "reasoning": "Why this relationship is important for QA"
                }}
            ]
        }}

        **CRITICAL REQUIREMENTS:**
        - **MUST USE EXACT ENTITY IDs**: The 'head_entity' and 'tail_entity' values MUST be the exact Entity IDs shown in the entities list above
        - **DO NOT use entity names**: Never use entity names like "Caroline" or "pottery class" in head_entity/tail_entity fields
        - **ONLY use provided IDs**: Only use IDs that appear in the entities list (like merged_0, noise_123, etc.)
        - Focus on relationships that enable multi-hop reasoning chains
        - Connect temporal entities (DATE_TIME) with events and activities
        - Link spatial entities (LOCATION) with people, events, and activities
        - Establish clear causal and temporal sequences
        - Ensure evidence text directly supports the claimed relationship
        - Prioritize relationships that answer implicit questions in the conversation

        Extract relationships that form coherent reasoning paths for complex question answering.
        """

        cross_session_relation_extraction = """
        You are an expert in identifying cross-session relationships. Based on entities that appear across multiple conversation sessions with rich mention information, extract relationships that span different time periods or contexts.

        **Cross-Session Entity Groups:**
        {cross_session_entities}

        **Session Contexts:**
        {session_contexts}

        **Task: Extract relationships that connect entities across different sessions, utilizing the rich mention contexts.**

        **Focus on Cross-Session Relationships:**
        - Temporal evolution: "evolves_into", "continues_from", "develops_over_time"
        - Consistency tracking: "remains_consistent", "changes_attitude_towards"
        - Cross-reference connections: "mentioned_in_both", "recurring_topic"
        - Relationship development: "strengthens_over_time", "builds_upon"
        - Progress tracking: "progresses_in", "improves_over_time", "maintains_through"

        **Output Format (JSON):**
        {{
            "cross_session_relations": [
                {{
                    "head_entity": "EXACT Entity ID from the groups above",
                    "tail_entity": "EXACT Entity ID from the groups above",
                    "relation_type": "cross-session relationship type",
                    "confidence": 0.85,
                    "evidence": "Evidence from multiple sessions with specific contexts",
                    "sessions": ["session_1", "session_2"],
                    "temporal_context": "timespan of the relationship",
                    "reasoning": "Why this cross-session relationship is significant for multi-hop QA"
                }}
            ]
        }}

        **CRITICAL REQUIREMENTS:**
        - **MUST USE EXACT ENTITY IDs**: Use only the exact Entity IDs shown in the entity groups
        - **DO NOT use entity names**: Never use entity names in head_entity/tail_entity fields
        - Only extract relationships supported by evidence from multiple sessions
        - Focus on relationships that show progression, consistency, or development
        - Ensure the relationship adds value beyond single-session connections
        - Utilize the rich context information from different mentions
        - Consider temporal progression between sessions
        """

        return {
            "relation_extraction": relation_extraction,
            "cross_session_relation_extraction": cross_session_relation_extraction
        }

    def load_entities_data(self, entities_file: str) -> Dict[str, Any]:
        """Load entities data."""
        try:
            with open(entities_file, 'r', encoding='utf-8') as f:
                entities_data = json.load(f)
            
            self.logger.info(f" Loaded entities data: {len(entities_data.get('entities', []))} entities")
            
            entities = entities_data.get('entities', [])
            if entities:
                sample_entity = entities[0]
                data_format = entities_data.get('extraction_metadata', {}).get('data_format', 'unknown')
                
                if 'mentions' in sample_entity:
                    self.logger.info(f" 检测到mentions格式数据 (format: {data_format})")
                else:
                    self.logger.info(" 检测到传统格式数据，将进行兼容处理")
            
            return entities_data
            
        except Exception as e:
            self.logger.error(f" Failed to load entities data: {e}")
            return None

    def load_original_conversation(self, conversation_file: str) -> Dict[str, Any]:
        """Load original conversation."""
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
                "dialogues": session_dialogues
            }
            
        except Exception as e:
            self.logger.error(f" Failed to extract session content {session_key}: {e}")
            return None

    def _build_multimodal_dialogue_text(self, dialogue: Dict[str, Any]) -> str:
        """Build a text block that preserves dialogue and image metadata.

        The returned string appends available image captions and retrieval
        queries so relation generation can use multimodal evidence without
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

    def _get_entity_sessions(self, entity: Dict[str, Any]) -> List[str]:
        """Get entity sessions."""
        if 'mentions' in entity:
            return list(set([mention.get('session_id', '') for mention in entity['mentions'] if mention.get('session_id')]))
        else:
            
            return entity.get('sessions', [])

    def _get_entity_session_context(self, entity: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """Get entity session context."""
        if 'mentions' in entity:
            session_mentions = [m for m in entity['mentions'] if m.get('session_id') == session_id]
            if session_mentions:
                mention = session_mentions[0]
                return {
                    'context': mention.get('context', ''),
                    'temporal_info': mention.get('temporal_info'),
                    'spatial_info': mention.get('spatial_info'),
                    'aliases': mention.get('aliases', []),
                    'confidence': mention.get('confidence', entity.get('confidence', 0.8))
                }
        else:
            
            return {
                'context': entity.get('content', ''),
                'temporal_info': entity.get('temporal_info'),
                'spatial_info': entity.get('spatial_info'),
                'aliases': entity.get('aliases', []),
                'confidence': entity.get('confidence', 0.8)
            }
        
        return {
            'context': '',
            'temporal_info': None,
            'spatial_info': None,
            'aliases': [],
            'confidence': 0.8
        }

    def extract_relations_from_session(self, session_data: Dict[str, Any], 
                                 session_entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract relations from session."""
        try:
            session_id = session_data["session_id"]
            self.logger.info(f" Extracting relations from {session_id}")
            
            relevant_entities = []
            entity_id_mapping = {}
            entity_name_mapping = {}
            
            for entity in session_entities:
                entity_sessions = self._get_entity_sessions(entity)
                
                if session_id in entity_sessions:
                    session_context = self._get_entity_session_context(entity, session_id)
                    
                    relevant_entities.append({
                        'entity_id': entity['entity_id'],
                        'name': entity['name'],
                        'entity_type': entity['entity_type'],
                        'context': session_context['context'],
                        'temporal_info': session_context['temporal_info'],
                        'spatial_info': session_context['spatial_info'],
                        'aliases': session_context['aliases'],
                        'confidence': session_context['confidence']
                    })
                    
                    entity_id_mapping[entity['entity_id']] = entity['name']
                    entity_name_mapping[entity['name'].lower().strip()] = entity['entity_id']
                    
                    for alias in session_context.get('aliases', []):
                        if alias and alias.strip():
                            entity_name_mapping[alias.lower().strip()] = entity['entity_id']
            
            if len(relevant_entities) < 2:
                self.logger.info(f" Not enough entities ({len(relevant_entities)}) for relation extraction in {session_id}")
                return []
            
            entities_list = ""
            for entity in relevant_entities:
                entities_list += f"- ID: {entity['entity_id']} | Name: {entity['name']} | Type: {entity['entity_type']}\n"
                entities_list += f"  Context: {entity['context'][:150]}...\n"
                if entity.get('temporal_info'):
                    entities_list += f"  Temporal Info: {entity['temporal_info']}\n"
                if entity.get('spatial_info'):
                    entities_list += f"  Spatial Info: {entity['spatial_info']}\n"
                if entity.get('aliases'):
                    entities_list += f"  Aliases: {', '.join(entity['aliases'])}\n"
                entities_list += f"  Confidence: {entity['confidence']}\n\n"
        
            prompt = self.prompts["relation_extraction"].format(
                session_id=session_data["session_id"],
                session_time=session_data["session_time"],
                conversation_text=session_data["conversation_text"],
                entities_list=entities_list
            )
            
            response = self.llm_client.generate_answer(
                prompt,
                temperature=0.1,
                json_format=True
            )
            
            self._update_stats(llm_calls_made=1, relation_extraction_calls=1)
            
            try:
                result = json.loads(response)
                relations = result.get("relations", [])
                
                
                valid_relations = []
                for relation in relations:
                    head_entity_ref = relation.get('head_entity')
                    tail_entity_ref = relation.get('tail_entity')
                    
                    head_entity_id = self._resolve_entity_reference(head_entity_ref, entity_id_mapping, entity_name_mapping)
                    tail_entity_id = self._resolve_entity_reference(tail_entity_ref, entity_id_mapping, entity_name_mapping)
                    
                    if head_entity_id and tail_entity_id and head_entity_id != tail_entity_id:
                        relation["head_entity"] = head_entity_id
                        relation["tail_entity"] = tail_entity_id
                        relation["session"] = session_id
                        relation["session_time"] = session_data["session_time"]
                        relation["head_entity_name"] = entity_id_mapping[head_entity_id]
                        relation["tail_entity_name"] = entity_id_mapping[tail_entity_id]
                        valid_relations.append(relation)
                    else:
                        self.logger.warning(f" 无法解析实体引用，关系被跳过: {head_entity_ref} -> {tail_entity_ref}")
                
                self.logger.info(f" Extracted {len(valid_relations)} valid relations from {session_id}")
                return valid_relations
                
            except json.JSONDecodeError as e:
                self.logger.error(f" JSON parsing failed for relations in {session_id}: {e}")
                try:
                    import re
                    json_match = re.search(r'\{.*\}', response, re.DOTALL)
                    if json_match:
                        json_str = json_match.group()
                        result = json.loads(json_str)
                        relations = result.get("relations", [])
                        
                        valid_relations = []
                        for relation in relations:
                            head_entity_ref = relation.get('head_entity')
                            tail_entity_ref = relation.get('tail_entity')
                            
                            head_entity_id = self._resolve_entity_reference(head_entity_ref, entity_id_mapping, entity_name_mapping)
                            tail_entity_id = self._resolve_entity_reference(tail_entity_ref, entity_id_mapping, entity_name_mapping)
                            
                            if head_entity_id and tail_entity_id and head_entity_id != tail_entity_id:
                                relation["head_entity"] = head_entity_id
                                relation["tail_entity"] = tail_entity_id
                                relation["session"] = session_id
                                relation["session_time"] = session_data["session_time"]
                                relation["head_entity_name"] = entity_id_mapping[head_entity_id]
                                relation["tail_entity_name"] = entity_id_mapping[tail_entity_id]
                                valid_relations.append(relation)
                        
                        self.logger.info(f" Recovered {len(valid_relations)} relations from {session_id}")
                        return valid_relations
                    else:
                        return []
                except Exception:
                    return []
                    
        except Exception as e:
            self.logger.error(f" Relation extraction failed for {session_data.get('session_id', 'unknown')}: {e}")
            return []
        
    def _resolve_entity_reference(self, reference: str, 
                            id_mapping: Dict[str, str], 
                            name_mapping: Dict[str, str]) -> Optional[str]:
        """Resolve entity reference."""
        if not reference or not isinstance(reference, str):
            return None
        
        reference = reference.strip()
        
        if reference in id_mapping:
            return reference
        
        reference_lower = reference.lower()
        if reference_lower in name_mapping:
            return name_mapping[reference_lower]
        
        for name_lower, entity_id in name_mapping.items():
            if reference_lower in name_lower or name_lower in reference_lower:
                similarity = len(reference_lower) / max(len(name_lower), 1)
                if similarity > 0.6:
                    self.logger.debug(f" 部分匹配实体: '{reference}' -> '{name_lower}' (ID: {entity_id})")
                    return entity_id
        
        available_ids = list(id_mapping.keys())[:5]  # Avoid mutating LogRecord fields before other handlers process the record.
        available_names = list(name_mapping.keys())[:5]
        
        self.logger.debug(f" 无法解析实体引用: '{reference}'")
        self.logger.debug(f"   可用ID示例: {available_ids}")
        self.logger.debug(f"   可用名称示例: {available_names}")
        
        return None

    def _extract_cross_session_entity_group(self, entity_group: List[Dict[str, Any]], 
                                      session_contents: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract cross session entity group."""
        try:
            if len(entity_group) < 2:
                return []
            
            entity_id_mapping = {entity['entity_id']: entity['name'] for entity in entity_group}
            entity_name_mapping = {entity['name'].lower().strip(): entity['entity_id'] for entity in entity_group}
            
            entity_group_text = f"Entity Group: {entity_group[0]['name']}\n"
            entity_sessions = set()
            
            for entity in entity_group:
                sessions = self._get_entity_sessions(entity)
                entity_sessions.update(sessions)
                
                entity_group_text += f"- ID: {entity['entity_id']} | Name: {entity['name']} | Type: {entity['entity_type']}\n"
                entity_group_text += f"  Sessions: {sessions}\n"
                
                if 'mentions' in entity:
                    entity_group_text += f"  Mentions:\n"
                    for mention in entity['mentions']:
                        entity_group_text += f"    {mention['session_id']}: {mention.get('context', '')[:100]}...\n"
                        if mention.get('temporal_info'):
                            entity_group_text += f"      Time: {mention['temporal_info']}\n"
                        if mention.get('spatial_info'):
                            entity_group_text += f"      Location: {mention['spatial_info']}\n"
                entity_group_text += "\n"
            
            session_contexts_text = ""
            for session_id in sorted(entity_sessions):
                if session_id in session_contents:
                    session_data = session_contents[session_id]
                    session_contexts_text += f"Session: {session_id}\n"
                    session_contexts_text += f"Time: {session_data.get('session_time', 'Unknown')}\n"
                    session_contexts_text += f"Content: {session_data.get('conversation_text', '')[:300]}...\n\n"
            
            prompt = self.prompts["cross_session_relation_extraction"].format(
                cross_session_entities=entity_group_text,
                session_contexts=session_contexts_text
            )
            
            response = self.llm_client.generate_answer(
                prompt,
                temperature=0.1,
                json_format=True
            )
            
            self._update_stats(llm_calls_made=1, cross_session_extraction_calls=1)
            
            try:
                result = json.loads(response)
                cross_relations = result.get("cross_session_relations", [])
                
                valid_relations = []
                for relation in cross_relations:
                    head_entity_ref = relation.get('head_entity')
                    tail_entity_ref = relation.get('tail_entity')
                    
                    head_entity_id = self._resolve_entity_reference(head_entity_ref, entity_id_mapping, entity_name_mapping)
                    tail_entity_id = self._resolve_entity_reference(tail_entity_ref, entity_id_mapping, entity_name_mapping)
                    
                    if head_entity_id and tail_entity_id and head_entity_id != tail_entity_id:
                        relation["head_entity"] = head_entity_id
                        relation["tail_entity"] = tail_entity_id
                        relation["head_entity_name"] = entity_id_mapping[head_entity_id]
                        relation["tail_entity_name"] = entity_id_mapping[tail_entity_id]
                        relation["entity_group_sessions"] = sorted(list(entity_sessions))
                        
                        valid_relations.append(relation)
                    else:
                        self.logger.warning(f" 跨会话关系实体引用解析失败: {head_entity_ref} -> {tail_entity_ref}")
                
                return valid_relations
                
            except json.JSONDecodeError as e:
                self.logger.warning(f" Cross-session JSON parsing failed: {e}")
                return []
                
        except Exception as e:
            self.logger.error(f" Cross-session entity group processing failed: {e}")
            return []

    def extract_cross_session_relations(self, entities_data: Dict[str, Any], 
                                      session_contents: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract cross session relations."""
        try:
            self.logger.info(" Extracting cross-session relations...")
            
            entities_by_name = defaultdict(list)
            for entity in entities_data.get('entities', []):
                entity_name = entity['name'].lower().strip()
                entities_by_name[entity_name].append(entity)
            
            cross_session_entity_groups = []
            for name, entity_list in entities_by_name.items():
                if len(entity_list) > 1:
                    all_sessions = set()
                    for entity in entity_list:
                        sessions = self._get_entity_sessions(entity)
                        all_sessions.update(sessions)
                    
                    if len(all_sessions) > 1:
                        cross_session_entity_groups.append(entity_list)
            
            if not cross_session_entity_groups:
                self.logger.info(" No cross-session entities found")
                return []
            
            self.logger.info(f" Found {len(cross_session_entity_groups)} cross-session entity groups")
            
            all_cross_relations = []
            
            max_workers = min(self.parallel_workers, len(cross_session_entity_groups))
            with ThreadPoolExecutor(max_workers=max_workers, 
                                  thread_name_prefix="CrossSessionWorker") as executor:
                future_to_group = {
                    executor.submit(self._extract_cross_session_entity_group, group, session_contents): i
                    for i, group in enumerate(cross_session_entity_groups)
                }
                
                for future in as_completed(future_to_group):
                    group_id = future_to_group[future]
                    try:
                        group_relations = future.result()
                        all_cross_relations.extend(group_relations)
                        
                        entity_group = cross_session_entity_groups[group_id]
                        self.logger.debug(f" Cross-session group {group_id} ({entity_group[0]['name']}) processed: {len(group_relations)} relations")
                        
                    except Exception as e:
                        self.logger.error(f" Cross-session group {group_id} processing failed: {e}")
            
            self._update_stats(cross_session_relations_extracted=len(all_cross_relations))
            self.logger.info(f" Extracted {len(all_cross_relations)} cross-session relations")
            return all_cross_relations
                
        except Exception as e:
            self.logger.error(f" Cross-session relation extraction failed: {e}")
            return []

    def convert_to_relation_objects(self, relations_data: List[Dict[str, Any]], 
                                   entities_dict: Dict[str, Dict[str, Any]]) -> List[ExtractedRelation]:
        """Convert to relation objects."""
        relation_objects = []
        
        for i, relation in enumerate(relations_data):
            try:
                sessions = relation.get("sessions", [])
                if not sessions:
                    session = relation.get("session")
                    if session:
                        sessions = [session]
                    else:
                        sessions = relation.get("entity_group_sessions", ["unknown"])
                
                relation_obj = ExtractedRelation(
                    relation_id=f"R_{i}",
                    head_entity_id=relation.get("head_entity", ""),
                    tail_entity_id=relation.get("tail_entity", ""),
                    head_entity_name=relation.get("head_entity_name", ""),
                    tail_entity_name=relation.get("tail_entity_name", ""),
                    relation_type=relation.get("relation_type", "unknown"),
                    confidence=float(relation.get("confidence", 0.8)),
                    sessions=sessions,
                    evidence_texts=[relation.get("evidence", "")],
                    contexts=[relation.get("reasoning", "")],
                    temporal_context=relation.get("temporal_context"),
                    extraction_metadata={
                        "extraction_method": "llm_based_mentions_compatible",
                        "session": relation.get("session"),
                        "session_time": relation.get("session_time"),
                        "is_cross_session": len(sessions) > 1 if sessions else False
                    }
                )
                relation_objects.append(relation_obj)
                
            except Exception as e:
                self.logger.warning(f" Failed to convert relation {i}: {e}")
                continue
        
        return relation_objects

    def build_entity_graph(self, entities_dict: Dict[str, Dict[str, Any]], 
                          relations: List[ExtractedRelation]) -> nx.DiGraph:
        """Build entity graph."""
        graph = nx.DiGraph()
        
        for entity_id, entity in entities_dict.items():
            
            sessions = self._get_entity_sessions(entity)
            
            all_contexts = []
            all_aliases = []
            
            if 'mentions' in entity:
                for mention in entity['mentions']:
                    if mention.get('context'):
                        all_contexts.append(mention['context'])
                    all_aliases.extend(mention.get('aliases', []))
            else:
                
                if entity.get('content'):
                    all_contexts.append(entity['content'])
                all_aliases.extend(entity.get('aliases', []))
            
            node_attrs = {
                'name': entity.get('name', ''),
                'entity_type': entity.get('entity_type', ''),
                'confidence': entity.get('confidence', 0.8),
                'sessions': ','.join(sessions),
                'aliases': ','.join(list(set(all_aliases))),
                'contexts_count': len(all_contexts),
                'total_mentions': len(entity.get('mentions', [])) if 'mentions' in entity else 1
            }
            
            if all_contexts:
                node_attrs['primary_context'] = all_contexts[0][:200]
            
            graph.add_node(entity_id, **node_attrs)
        
        for relation in relations:
            if (graph.has_node(relation.head_entity_id) and 
                graph.has_node(relation.tail_entity_id)):
                
                edge_attrs = {
                    'relation_id': relation.relation_id,
                    'relation_type': relation.relation_type,
                    'confidence': relation.confidence,
                    'sessions': ','.join(relation.sessions),
                    'evidence': '; '.join(relation.evidence_texts)[:300],
                    'temporal_context': relation.temporal_context or "",
                    'is_cross_session': len(relation.sessions) > 1
                }
                graph.add_edge(relation.head_entity_id, relation.tail_entity_id, **edge_attrs)
        
        return graph

    def process_entities_to_relations(self, entities_file: str, 
                                    original_conversation_file: str) -> EntityRelationGraph:
        """Process entities to relations."""
        start_time = datetime.now()
        
        self.logger.info(f" Processing entities to generate relations (mentions compatible)")
        self.logger.info(f" Entities file: {entities_file}")
        self.logger.info(f" Original conversation: {original_conversation_file}")
        
        
        entities_data = self.load_entities_data(entities_file)
        if not entities_data:
            return None
        
        
        conversation_data = self.load_original_conversation(original_conversation_file)
        if not conversation_data:
            return None
        
        conversation_id = entities_data.get("conversation_id", Path(entities_file).stem)
        
        entities_dict = {entity['entity_id']: entity for entity in entities_data.get('entities', [])}
        
        conversation = conversation_data.get("conversation", {})
        session_contents = {}
        all_session_data = []
        
        for key in conversation.keys():
            if key.startswith("session_") and not key.endswith("_date_time"):
                session_data = self.extract_session_content(conversation_data, key)
                if session_data:
                    session_contents[key] = session_data
                    all_session_data.append(session_data)
        
        if not all_session_data:
            self.logger.warning(f" No valid sessions found")
            return None
        
        self.logger.info(f" Found {len(all_session_data)} sessions for relation extraction")
        
        all_relations = []
        session_metadata = {}
        
        with ThreadPoolExecutor(max_workers=self.parallel_workers, 
                              thread_name_prefix="SessionRelationWorker") as executor:
            future_to_session = {}
            
            for session_data in all_session_data:
                session_id = session_data["session_id"]
                
                
                session_entities = []
                for entity in entities_data.get('entities', []):
                    sessions = self._get_entity_sessions(entity)
                    if session_id in sessions:
                        session_entities.append(entity)
                
                if session_entities:
                    future = executor.submit(self.extract_relations_from_session, session_data, session_entities)
                    future_to_session[future] = session_id
                else:
                    self.logger.debug(f" No entities found for session {session_id}")
            
            for future in as_completed(future_to_session):
                session_id = future_to_session[future]
                try:
                    session_relations = future.result()
                    all_relations.extend(session_relations)
                    
                    session_data = next(s for s in all_session_data if s["session_id"] == session_id)
                    session_entities_count = len([e for e in entities_data.get('entities', []) 
                                                if session_id in self._get_entity_sessions(e)])
                    
                    session_metadata[session_id] = {
                        "session_time": session_data["session_time"],
                        "speakers": session_data["speakers"],
                        "entities_count": session_entities_count,
                        "relations_count": len(session_relations)
                    }
                    
                except Exception as e:
                    self.logger.error(f" Relation extraction failed for {session_id}: {e}")
        
        cross_session_relations = self.extract_cross_session_relations(entities_data, session_contents)
        all_relations.extend(cross_session_relations)
        
        self.logger.info(f" Total relations extracted: {len(all_relations)}")
        self.logger.info(f"    Session relations: {len(all_relations) - len(cross_session_relations)}")
        self.logger.info(f"    Cross-session relations: {len(cross_session_relations)}")
        
        relation_objects = self.convert_to_relation_objects(all_relations, entities_dict)
        
        entity_graph = self.build_entity_graph(entities_dict, relation_objects)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        self._update_stats(
            conversations_processed=1,
            sessions_processed=len(all_session_data),
            total_relations_extracted=len(relation_objects),
            processing_time=processing_time
        )
        
        result = EntityRelationGraph(
            conversation_id=conversation_id,
            entities=entities_dict,
            relations=relation_objects,
            entity_graph=entity_graph,
            session_metadata=session_metadata,
            extraction_metadata={
                "extraction_method": "mentions_compatible_relation_generator",
                "llm_model": getattr(self.llm_client, 'model_name', 'unknown'),
                "processing_time_seconds": processing_time,
                "created_at": datetime.now().isoformat(),
                "relation_extraction_calls": self.stats["relation_extraction_calls"],
                "cross_session_extraction_calls": self.stats["cross_session_extraction_calls"],
                "total_llm_calls": self.stats["llm_calls_made"],
                "entities_loaded": len(entities_dict),
                "relations_generated": len(relation_objects),
                "cross_session_relations": len(cross_session_relations),
                "data_format_compatibility": "mentions_and_legacy_supported"
            }
        )
        
        self.logger.info(f" Relation generation completed: {conversation_id}")
        self.logger.info(f"    Entities: {len(entities_dict)}, Relations: {len(relation_objects)}")
        self.logger.info(f"    Cross-session relations: {len(cross_session_relations)}")
        self.logger.info(f"   Processing time: {processing_time:.2f}s")
        
        return result

    def export_relation_graph(self, graph: EntityRelationGraph, output_dir: str) -> bool:
        """Save relation graph."""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            complete_data = {
                "conversation_id": graph.conversation_id,
                "entities": list(graph.entities.values()),
                "relations": [asdict(relation) for relation in graph.relations],
                "session_metadata": graph.session_metadata,
                "extraction_metadata": graph.extraction_metadata,
                "export_time": datetime.now().isoformat(),
                "schema_info": {
                    "extraction_pipeline": "two_step_entity_first_relation_second",
                    "step": "2_relation_generation_mentions_compatible",
                    "entity_format": "mentions_based",
                    "relation_format": "enhanced_with_cross_session"
                }
            }
            
            
            json_path = output_path / f"{graph.conversation_id}_complete_entity_relation.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(complete_data, f, ensure_ascii=False, indent=2)
            
            
            relations_data = {
                "conversation_id": graph.conversation_id,
                "relations": [asdict(relation) for relation in graph.relations],
                "relation_statistics": self._generate_relation_statistics(graph.relations),
                "extraction_metadata": graph.extraction_metadata
            }
            
            relations_path = output_path / f"{graph.conversation_id}_relations.json"
            with open(relations_path, 'w', encoding='utf-8') as f:
                json.dump(relations_data, f, ensure_ascii=False, indent=2)
            
            
            try:
                gml_path = output_path / f"{graph.conversation_id}_knowledge_graph.gml"
                nx.write_gml(graph.entity_graph, gml_path)
            except Exception as e:
                self.logger.warning(f" Failed to save GML format: {e}")
                try:
                    graphml_path = output_path / f"{graph.conversation_id}_knowledge_graph.graphml"
                    nx.write_graphml(graph.entity_graph, graphml_path)
                except Exception as e2:
                    self.logger.warning(f" Failed to save GraphML format: {e2}")
            
            stats = self._generate_complete_stats(graph)
            stats_path = output_path / f"{graph.conversation_id}_complete_stats.json"
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            
            self._generate_relation_readme(graph, output_path)
            
            self.logger.info(f" Relation graph exported to: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f" Failed to export relation graph: {e}")
            return False

    def _generate_relation_statistics(self, relations: List[ExtractedRelation]) -> Dict[str, Any]:
        """Generate relation statistics."""
        relation_types = Counter([r.relation_type for r in relations])
        confidence_stats = [r.confidence for r in relations]
        cross_session_count = sum(1 for r in relations if len(r.sessions) > 1)
        
        return {
            "total_relations": len(relations),
            "cross_session_relations": cross_session_count,
            "intra_session_relations": len(relations) - cross_session_count,
            "relation_types": dict(relation_types),
            "confidence_stats": {
                "mean": sum(confidence_stats) / len(confidence_stats) if confidence_stats else 0,
                "min": min(confidence_stats) if confidence_stats else 0,
                "max": max(confidence_stats) if confidence_stats else 0
            },
            "sessions_with_relations": len(set([session for r in relations for session in r.sessions]))
        }

    def _generate_complete_stats(self, graph: EntityRelationGraph) -> Dict[str, Any]:
        """Generate complete stats."""
        entity_types = Counter([entity['entity_type'] for entity in graph.entities.values()])
        relation_stats = self._generate_relation_statistics(graph.relations)
        
        mentions_stats = {"total_mentions": 0, "entities_with_multiple_mentions": 0}
        for entity in graph.entities.values():
            if 'mentions' in entity:
                mention_count = len(entity['mentions'])
                mentions_stats["total_mentions"] += mention_count
                if mention_count > 1:
                    mentions_stats["entities_with_multiple_mentions"] += 1
        
        return {
            "conversation_id": graph.conversation_id,
            "pipeline_summary": {
                "extraction_method": "two_step_pipeline_mentions_compatible",
                "step1": "entity_extraction_and_deduplication_with_mentions",
                "step2": "relation_generation_cross_session_enhanced",
                "total_entities": len(graph.entities),
                "total_relations": len(graph.relations),
                "sessions_processed": len(graph.session_metadata)
            },
            "entity_analysis": {
                "by_type": dict(entity_types),
                "mentions_statistics": mentions_stats,
                "entities_per_session": {
                    session_id: sum(1 for entity in graph.entities.values() 
                                   if session_id in self._get_entity_sessions(entity))
                    for session_id in graph.session_metadata.keys()
                }
            },
            "relation_analysis": relation_stats,
            "graph_metrics": {
                "nodes": graph.entity_graph.number_of_nodes(),
                "edges": graph.entity_graph.number_of_edges(),
                "density": nx.density(graph.entity_graph),
                "connected_components": nx.number_weakly_connected_components(graph.entity_graph),
                "average_clustering": nx.average_clustering(graph.entity_graph.to_undirected()) if graph.entity_graph.number_of_nodes() > 0 else 0
            },
            "extraction_metadata": graph.extraction_metadata
        }

    def _generate_relation_readme(self, graph: EntityRelationGraph, output_path: Path):
        """Generate relation readme."""
        stats = self._generate_complete_stats(graph)
        
        readme_content = f"""# Complete Entity-Relation Graph - {graph.conversation_id}

        ## Overview
        - **Conversation ID**: {graph.conversation_id}  
        - **Extraction Pipeline**: Two-Step Method (Entity First → Relation Second)
        - **Data Format**: Mentions-based entities with enhanced relation extraction
        - **Generated**: {graph.extraction_metadata['created_at']}
        - **Processing Time**: {graph.extraction_metadata['processing_time_seconds']:.2f}s

        ## Enhanced Pipeline Architecture

        This graph was generated using an improved two-step extraction pipeline with mentions compatibility:

        ### Step 1: Entity Extraction & Deduplication (Mentions Format)
        - Entities were extracted from conversation sessions with rich context
        - **Mentions Structure**: Each entity maintains session-specific context, temporal, and spatial information
        - Cross-session deduplication using DBSCAN clustering while preserving all mentions
        - Entities were standardized and unified across sessions

        ### Step 2: Enhanced Relation Generation (This Step)
        - **Session-Specific Context**: Relations extracted using session-specific entity contexts
        - **Cross-Session Relations**: Multi-threaded extraction of relationships spanning multiple sessions
        - **Temporal Progression**: Tracking entity evolution and relationship development over time
        - Relationships validated against entity references with mentions support

        ## Statistics

        ### Pipeline Summary
        - **Total Entities**: {stats['pipeline_summary']['total_entities']}
        - **Total Relations**: {stats['pipeline_summary']['total_relations']}
        - **Sessions Processed**: {stats['pipeline_summary']['sessions_processed']}
        - **Cross-Session Relations**: {stats['relation_analysis']['cross_session_relations']}
        - **Intra-Session Relations**: {stats['relation_analysis']['intra_session_relations']}

        ### Entity Analysis
        - **Total Mentions**: {stats['entity_analysis']['mentions_statistics']['total_mentions']}
        - **Multi-Mention Entities**: {stats['entity_analysis']['mentions_statistics']['entities_with_multiple_mentions']}

        #### Entity Type Distribution
        """
        
        for entity_type, count in stats['entity_analysis']['by_type'].items():
            readme_content += f"- **{entity_type}**: {count}\n"
        
        readme_content += f"""
        #### Relation Type Distribution
        """
        
        for relation_type, count in stats['relation_analysis']['relation_types'].items():
            readme_content += f"- **{relation_type}**: {count}\n"
        
        readme_content += f"""

        ### Graph Metrics
        - **Nodes**: {stats['graph_metrics']['nodes']}
        - **Edges**: {stats['graph_metrics']['edges']}
        - **Graph Density**: {stats['graph_metrics']['density']:.3f}
        - **Connected Components**: {stats['graph_metrics']['connected_components']}
        - **Average Clustering**: {stats['graph_metrics']['average_clustering']:.3f}

        ## Key Features

        ###  **Cross-Session Relationship Tracking**
        - Automatically identifies entities appearing across multiple sessions
        - Extracts temporal evolution and consistency relationships
        - Multi-threaded processing for efficient cross-session analysis

        ###  **Mentions-Based Entity Context**
        - Each entity maintains session-specific contexts and information
        - Preserves temporal and spatial information per mention
        - Enables precise relation extraction using relevant context

        ###  **Performance Optimizations**
        - Multi-threaded session relation extraction
        - Parallel cross-session relationship processing
        - Efficient entity-session mapping with format compatibility

        ## Files Generated
        - `{graph.conversation_id}_complete_entity_relation.json`: Complete entity-relation data
        - `{graph.conversation_id}_relations.json`: Relations-focused data  
        - `{graph.conversation_id}_knowledge_graph.gml/graphml`: NetworkX graph format
        - `{graph.conversation_id}_complete_stats.json`: Detailed statistics
        - `README.md`: This documentation

        ## Enhanced Data Structure

        ### Entity Format (Mentions-Based)
        ```json
        {{
            "entity_id": "merged_0",
            "name": "Caroline",
            "entity_type": "PERSON", 
            "confidence": 0.95,
            "mentions": [
                {{
                    "session_id": "session_1",
                    "context": "Session-specific description",
                    "temporal_info": "May 8, 2023", 
                    "spatial_info": null,
                    "aliases": ["Caroline", "she"],
                    "confidence": 0.95
                }}
            ]
        }}
        ```

        ### Enhanced Relation Format
        ```json
        {{
            "relation_id": "R_1",
            "head_entity_id": "merged_0",
            "tail_entity_id": "merged_1",
            "relation_type": "supports",
            "confidence": 0.90,
            "sessions": ["session_1", "session_2"],
            "evidence_texts": ["Caroline provides emotional support to Melanie"],
            "temporal_context": "Throughout their friendship",
            "extraction_metadata": {{
                "is_cross_session": true,
                "extraction_method": "llm_based_mentions_compatible"
            }}
        }}
        ```

        ## Loading and Analysis

        ### Python Example
        ```python
        import json
        import networkx as nx

        # Load complete data
        with open('{graph.conversation_id}_complete_entity_relation.json', 'r') as f:
            data = json.load(f)

        # Load graph
        try:
            graph = nx.read_gml('{graph.conversation_id}_knowledge_graph.gml')
        except:
            graph = nx.read_graphml('{graph.conversation_id}_knowledge_graph.graphml')

        # Access enhanced data
        entities = data['entities']
        relations = data['relations']

        # Analyze cross-session relations
        cross_session_relations = [r for r in relations if r['extraction_metadata']['is_cross_session']]
        print(f"Cross-session relations: {{len(cross_session_relations)}}")

        # Query entities with multiple mentions
        multi_mention_entities = [e for e in entities if len(e.get('mentions', [])) > 1]
        print(f"Entities with multiple contexts: {{len(multi_mention_entities)}}")

        # Find entity evolution patterns
        for entity in multi_mention_entities:
            print(f"Entity: {{entity['name']}}")
            for mention in entity['mentions']:
                print(f"  {{mention['session_id']}}: {{mention.get('temporal_info', 'N/A')}}")
        ```

        ### Relation Analysis
        ```python
        # Group relations by type
        from collections import Counter
        relation_types = Counter([r['relation_type'] for r in relations])

        # Find temporal progressions
        temporal_relations = [r for r in relations if 'time' in r.get('temporal_context', '').lower()]

        # Analyze cross-session patterns
        cross_session_types = Counter([r['relation_type'] for r in cross_session_relations])
        ```

        ## Performance Metrics
        - **LLM Calls**: {graph.extraction_metadata['total_llm_calls']}
        - **Session Extraction Calls**: {graph.extraction_metadata['relation_extraction_calls']}  
        - **Cross-Session Extraction Calls**: {graph.extraction_metadata['cross_session_extraction_calls']}
        - **Processing Time**: {graph.extraction_metadata['processing_time_seconds']:.2f} seconds
        - **Parallel Workers Used**: Multi-threaded processing enabled

        ---
        **Generated by Enhanced LoCoMo Relation Generator v2.0**  
        *Compatible with mentions-based entity format*  
        *{datetime.now().isoformat()}*
        """
                
        readme_path = output_path / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)

    def process_entities_to_relations_from_data(self, entities_file: str, 
                                              conversation_data: Dict[str, Any]) -> EntityRelationGraph:
        """Process entities to relations from data."""
        start_time = datetime.now()
        
        self.logger.info(f" Processing entities to generate relations (from data)")
        self.logger.info(f" Entities file: {entities_file}")
        
        
        entities_data = self.load_entities_data(entities_file)
        if not entities_data:
            return None
        
        conversation_id = entities_data.get("conversation_id", Path(entities_file).stem.replace('_entities', ''))
        
        if not conversation_data:
            self.logger.error(" No conversation data provided")
            return None
        
        entities_dict = {entity['entity_id']: entity for entity in entities_data.get('entities', [])}
        
        conversation = conversation_data.get("conversation", {})
        session_contents = {}
        all_session_data = []
        
        for key in conversation.keys():
            if key.startswith("session_") and not key.endswith("_date_time"):
                session_data = self.extract_session_content(conversation_data, key)
                if session_data:
                    session_contents[key] = session_data
                    all_session_data.append(session_data)
        
        if not all_session_data:
            self.logger.warning(f" No valid sessions found")
            return None
        
        self.logger.info(f" Found {len(all_session_data)} sessions for relation extraction")
        
        all_relations = []
        session_metadata = {}
        
        with ThreadPoolExecutor(max_workers=self.parallel_workers, 
                              thread_name_prefix="SessionRelationWorker") as executor:
            future_to_session = {}
            
            for session_data in all_session_data:
                session_id = session_data["session_id"]
                
                
                session_entities = []
                for entity in entities_data.get('entities', []):
                    sessions = self._get_entity_sessions(entity)
                    if session_id in sessions:
                        session_entities.append(entity)
                
                if session_entities:
                    future = executor.submit(self.extract_relations_from_session, session_data, session_entities)
                    future_to_session[future] = session_id
                else:
                    self.logger.debug(f" No entities found for session {session_id}")
            
            for future in as_completed(future_to_session):
                session_id = future_to_session[future]
                try:
                    session_relations = future.result()
                    all_relations.extend(session_relations)
                    
                    session_data = next(s for s in all_session_data if s["session_id"] == session_id)
                    session_entities_count = len([e for e in entities_data.get('entities', []) 
                                                if session_id in self._get_entity_sessions(e)])
                    
                    session_metadata[session_id] = {
                        "session_time": session_data["session_time"],
                        "speakers": session_data["speakers"],
                        "entities_count": session_entities_count,
                        "relations_count": len(session_relations)
                    }
                    
                except Exception as e:
                    self.logger.error(f" Relation extraction failed for {session_id}: {e}")
        
        cross_session_relations = self.extract_cross_session_relations(entities_data, session_contents)
        all_relations.extend(cross_session_relations)
        
        self.logger.info(f" Total relations extracted: {len(all_relations)}")
        self.logger.info(f"    Session relations: {len(all_relations) - len(cross_session_relations)}")
        self.logger.info(f"    Cross-session relations: {len(cross_session_relations)}")
        
        relation_objects = self.convert_to_relation_objects(all_relations, entities_dict)
        
        entity_graph = self.build_entity_graph(entities_dict, relation_objects)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        self._update_stats(
            conversations_processed=1,
            sessions_processed=len(all_session_data),
            total_relations_extracted=len(relation_objects),
            processing_time=processing_time
        )
        
        result = EntityRelationGraph(
            conversation_id=conversation_id,
            entities=entities_dict,
            relations=relation_objects,
            entity_graph=entity_graph,
            session_metadata=session_metadata,
            extraction_metadata={
                "extraction_method": "mentions_compatible_relation_generator_from_data",
                "llm_model": getattr(self.llm_client, 'model_name', 'unknown'),
                "processing_time_seconds": processing_time,
                "created_at": datetime.now().isoformat(),
                "relation_extraction_calls": self.stats["relation_extraction_calls"],
                "cross_session_extraction_calls": self.stats["cross_session_extraction_calls"],
                "total_llm_calls": self.stats["llm_calls_made"],
                "entities_loaded": len(entities_dict),
                "relations_generated": len(relation_objects),
                "cross_session_relations": len(cross_session_relations),
                "data_format_compatibility": "mentions_and_legacy_supported"
            }
        )
        
        self.logger.info(f" Relation generation completed: {conversation_id}")
        self.logger.info(f"    Entities: {len(entities_dict)}, Relations: {len(relation_objects)}")
        self.logger.info(f"    Cross-session relations: {len(cross_session_relations)}")
        self.logger.info(f"   Processing time: {processing_time:.2f}s")
        
        return result

    
def load_original_conversations(conversation_file: str) -> Dict[str, Dict[str, Any]]:
    """Load original conversations."""
    try:
        with open(conversation_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Dataset-specific handling used by the reproduction workflow.
        conversations = {}
        if isinstance(data, list):
            for item in data:
                sample_id = item.get('sample_id', item.get('conversation_id'))
                if sample_id:
                    conversations[sample_id] = item
        elif isinstance(data, dict):
            sample_id = data.get('sample_id', data.get('conversation_id', 'single'))
            conversations[sample_id] = data
        
        return conversations
        
    except Exception as e:
        print(f" 加载原始对话数据失败: {e}")
        return {}

def process_single_sample(generator: LoCoMoRelationGenerator, 
                         sample_id: str, 
                         entities_dir: str,
                         conversations: Dict[str, Dict[str, Any]],
                         output_dir: str) -> bool:
    """Process single sample."""
    from pathlib import Path
    
    logger = logging.getLogger(__name__)
    
    try:
        entities_file = Path(entities_dir) / f"{sample_id}_entities.json"
        
        if not entities_file.exists():
            logger.error(f" 实体文件不存在: {entities_file}")
            return False
        
        if sample_id not in conversations:
            logger.error(f" 原始对话数据中没有找到 sample_id: {sample_id}")
            return False
        
        conversation_data = conversations[sample_id]
        
        # Dataset-specific handling used by the reproduction workflow.
        sample_output_dir = Path(output_dir) / sample_id
        sample_output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f" 处理sample: {sample_id}")
        logger.info(f"    实体文件: {entities_file}")
        logger.info(f"    输出目录: {sample_output_dir}")
        
        relation_graph = generator.process_entities_to_relations_from_data(
            str(entities_file), 
            conversation_data
        )
        
        if relation_graph:
            # Dataset-specific handling used by the reproduction workflow.
            success = generator.export_relation_graph(relation_graph, str(sample_output_dir))
            
            if success:
                logger.info(f" Sample {sample_id} 处理完成")
                logger.info(f"    实体: {len(relation_graph.entities)}, 关系: {len(relation_graph.relations)}")
                logger.info(f"    跨会话关系: {sum(1 for r in relation_graph.relations if len(r.sessions) > 1)}")
                return True
            else:
                logger.error(f" Sample {sample_id} 结果导出失败")
                return False
        else:
            logger.error(f" Sample {sample_id} 关系生成失败")
            return False
            
    except Exception as e:
        logger.error(f" Sample {sample_id} 处理异常: {e}")
        return False

def process_multiple_samples(generator: LoCoMoRelationGenerator,
                           sample_ids: List[str],
                           args) -> tuple[int, int]:
    """Process multiple samples."""
    
    logger = logging.getLogger(__name__)
    
    
    logger.info(" 加载原始对话数据...")
    conversations = load_original_conversations(args.original_conversation)
    
    if not conversations:
        logger.error(" 无法加载原始对话数据")
        return 0, len(sample_ids)
    
    logger.info(f" 加载了 {len(conversations)} 个对话数据")
    
    # Dataset-specific handling used by the reproduction workflow.
    success_count = 0
    total_count = len(sample_ids)
    
    for i, sample_id in enumerate(sample_ids, 1):
        print(f"\n{'-'*60}")
        print(f" 处理进度: {i}/{total_count} - {sample_id}")
        print(f"{'-'*60}")
        
        success = process_single_sample(
            generator, 
            sample_id, 
            args.entities_dir,
            conversations,
            args.output_dir
        )
        
        if success:
            success_count += 1
            print(f" {sample_id} 处理成功")
        else:
            print(f" {sample_id} 处理失败")
    
    return success_count, total_count

def main():
    """Run the command-line entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="LoCoMo关系生成器 - 第二步：基于实体生成关系 (兼容mentions格式)")
    
    # Dataset-specific handling used by the reproduction workflow.
    parser.add_argument("--entities-dir", 
                       default=str(paths.LOCOMO_ENTITY_RELATION_STEP1_DIR),
                       help="第一步输出的实体文件目录路径（默认：benchmark_locomo/dataset/locomo/entity_relation/step1_entities）")
    parser.add_argument("--original-conversation", 
                       default=str(paths.LOCOMO_RAW_FILE),
                       help="原始对话数据文件路径（默认：benchmark_locomo/dataset/locomo/locomo10.json）")
    parser.add_argument("--output-dir", 
                       default=str(paths.LOCOMO_ENTITY_RELATION_STEP2_DIR),
                       help="输出目录路径（默认：benchmark_locomo/dataset/locomo/entity_relation/step2_relations）")

    # Dataset-specific handling used by the reproduction workflow.
    parser.add_argument("--sample-ids", nargs='+', 
                       help="指定要处理的sample ID列表，如 --sample-ids conv-26 conv-30（默认处理所有）")
    
    parser.add_argument("--extract-model", 
                       default="qwen-3.5-plus-thinking",
                       help="抽取模型名称")
    parser.add_argument("--parallel-workers", type=int, default=10,
                       help="并行处理的工作线程数（默认10，支持session和cross-session并行）")
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
    print(" LoCoMo关系生成器 - 第二步：基于实体生成关系 (Mentions兼容版)")
    print("=" * 80)
    print(f" 实体目录: {args.entities_dir}")
    print(f" 原始对话: {args.original_conversation}")
    print(f" 输出目录: {args.output_dir}")
    print(f" LLM模型: {args.extract_model}")
    print(f" 并行线程数: {args.parallel_workers}")
    print(f" 支持功能: Sessions并行 + Cross-session多线程 + Mentions兼容")
    
    try:
        llm_client = LLMClient(model_name=args.extract_model)
        generator = LoCoMoRelationGenerator(
            llm_client=llm_client,
            parallel_workers=args.parallel_workers
        )
        
        # Dataset-specific handling used by the reproduction workflow.
        sample_ids_to_process = determine_sample_ids(args)
        
        if not sample_ids_to_process:
            print(" 没有找到要处理的sample ID")
            return 1
        
        print(f" 准备处理 {len(sample_ids_to_process)} 个sample: {sample_ids_to_process}")
        
        # Dataset-specific handling used by the reproduction workflow.
        success_count, total_count = process_multiple_samples(
            generator, sample_ids_to_process, args
        )
        
        print(f"\n{'='*60}")
        print(f" 批量处理完成!")
        print(f" 成功处理: {success_count}/{total_count}")
        if success_count < total_count:
            print(f" 处理失败: {total_count - success_count}")
        print(f" 输出目录: {args.output_dir}")
        
        return 0 if success_count == total_count else 1
        
    except Exception as e:
        print(f" 程序异常: {e}")
        if args.debug:
            import traceback
            print(f"详细错误: {traceback.format_exc()}")
        return 1


def determine_sample_ids(args) -> List[str]:
    """Determine sample IDs."""
    from pathlib import Path
    import glob
    
    entities_dir = Path(args.entities_dir)
    
    # Dataset-specific handling used by the reproduction workflow.
    if args.sample_ids:
        return args.sample_ids
    else:
        entity_files = glob.glob(str(entities_dir / "*_entities.json"))
        sample_ids = []
        for file_path in entity_files:
            file_name = Path(file_path).name
            # Dataset-specific handling used by the reproduction workflow.
            if file_name.endswith('_entities.json'):
                sample_id = file_name[:-len('_entities.json')]
                sample_ids.append(sample_id)
        if not sample_ids:
            print(" 在entities目录中未找到实体文件")
        else:
            print(f" 默认处理所有 {len(sample_ids)} 个samples")
        return sorted(sample_ids)

if __name__ == "__main__":
    exit(main())
