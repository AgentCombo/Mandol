"""Utilities for step1 generate l0 graph."""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import re
import numpy as np


from mandol.llm.llm_client import LLMClient
from mandol.core.semantic_map import SemanticMap
from mandol.core.memory_unit import MemoryUnit
from mandol.core import paths



#  Contextual Retrieval Prompt Template V2 (Full Session Global Context)
# Purpose: Fix "Time Anchoring" and "Pronoun Resolution" using FULL session context
# Key Change: "God Mode" - LLM sees entire session transcript (not just past history)


CONTEXTUAL_RETRIEVAL_PROMPT = """
<system_instructions>
You are a Data Enrichment Expert for a RAG system. 
Your task is to rewrite a specific message from a dialogue into a **standalone, self-contained indexing string**.
The goal is to allow a vector search engine to find this message without needing the surrounding context window.

### MANDATORY TRANSFORMATION RULES (Apply Strictly):

1. **Global Context Resolution (God Mode)**:
   - You have access to the **FULL transcript** (Past and Future context).
   - Resolve ALL pronouns (it, he, she, that, they) to specific entity names.
   - If a message relies on context from 10 turns later to be understood, incorporate that future context NOW.
   - If the message is a short reaction (e.g., "I agree", "No way"), rewrite it to include exactly *what* is being agreed to or denied.

2. **Absolute Time Enforcement**:
   - The **Session Date** is provided in the context below.
   - **NEVER** use relative terms like "today", "tomorrow", "next Friday", "last week".
   - **CALCULATE** and output the specific absolute date (YYYY-MM-DD) for any event mentioned.
   - *Example*: If Session Date is 2023-01-01 and text says "I'm leaving next Friday", write "leaving on 2023-01-06".

3. **Event & State Classification**:
   - **Occurrence**: If an event is happening *now* or is a specific plan, phrase it as a factual occurrence.
   - **Reference/Recall**: If they are discussing a *past* event, phrase it as "Speaker recalls/discusses [Event] from [Date]".
   - **State Change**: If a key status changes (e.g., Job, Location, Relationship), explicitly state: "Speaker's [Attribute] changed from [Old] to [New]".

4. **Noise Filtering**:
   - If the message is pure chitchat (e.g., "Haha", "Okay") with no factual content, output a minimal string or just the intent. Focus on **Entities, Actions, Dates, and Facts**.

</system_instructions>

<document_context>
Session Reference Date: {date}
Participants: {participants}

=== BEGIN FULL TRANSCRIPT (Static Context) ===
{full_session_transcript}
=== END FULL TRANSCRIPT ===
</document_context>

<target_message_locator>
Target Speaker: {speaker}
Target Message Content: "{message_text}"
</target_message_locator>

<output_requirement>
Output ONLY the contextualized string. Do not output explanations.
Format: [Date: YYYY-MM-DD] [Speaker Name] [Event/Fact/Action] [Details with Resolved Entities]
</output_requirement>
"""

# CONTEXTUAL_RETRIEVAL_PROMPT = """
# <document_context>
# Session Date: {date}
# Participants: {participants}
# Full Transcript:
# {full_session_transcript}
# </document_context>

# <target_message_locator>
# Target Speaker: {speaker}
# Target Message Content: "{message_text}"
# </target_message_locator>

# Task: Generate a standalone "Indexing String" for the <target_message> to facilitate accurate vector retrieval.
# Target Audience: A search engine looking for specific facts, events, and opinions.

# Mandatory Requirements:
# 1. **Global Context Awareness**: Use the FULL transcript to resolve pronouns (it, he, that project) and ambiguous references. Look *ahead* in the transcript if the explanation comes later.
# 2. **Absolute Time Resolution**: Do NOT use relative terms like "today", "tomorrow", "next week". Calculate and explicitly state the estimated absolute date based on the Session Date "{date}".
# 3. **Event Classification**: Explicitly state if the message describes an event *happening now*, a *future plan*, or a *past memory*.
# 4. **Context Integration**: If the message is a short response (e.g., "Yes", "I agree"), rewrite it to include what is being agreed to.

# Output Format:
# [Date: YYYY-MM-DD] [Speaker Name] [Action/Intent] [Details with Resolved Entities]
# """


@dataclass
class L0GraphConfig:
    locomo_file_path: str = "benchmark/dataset/locomo/locomo10.json"
    output_dir: str = "benchmark/dataset/locomo/hierarchical/step1_l0_graphs"
    
    # Dataset-specific handling used by the reproduction workflow.
    # Sample-level: Sequential by default (easier debugging)
    enable_sample_parallel: bool = False  # Deprecated: use sequential samples by default
    max_workers_sample: int = 1  # Legacy: kept for backwards compatibility
    
    # Message-level: Parallel LLM calls within each session
    enable_message_parallel: bool = True  # Enable parallel LLM context generation
    max_workers_per_session: int = 10  # Max concurrent LLM calls per session
    
    enable_relation_extraction: bool = False
    enable_entity_extraction: bool = False
    
    similarity_threshold: float = 0.6
    top_k_candidates: int = 5  
    
    llm_model: str = "qwen-3.5-plus-thinking"  
    llm_temperature: float = 0.1
    llm_confidence_threshold: float = 0.6
    
    min_content_length: int = 15
    max_sessions_per_sample: Optional[int] = None
    
    sample_ids: Optional[List[str]] = None  # Dataset-specific handling used by the reproduction workflow.
    
    debug_mode: bool = False


class L0GraphGenerator:
    
    def __init__(self, config: L0GraphConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        self.stats = {
            'total_samples': 0,
            'processed_samples': 0,
            'total_conversations': 0,
            'total_relations': 0,
            'total_entities': 0,
            'similarity_filtered_relations': 0,
            'llm_confirmed_relations': 0,
            'processing_time': 0,
            'failed_samples': [],
            'sample_details': {}
        }
        
        self.llm_client = LLMClient(
            model_name=self.config.llm_model,
            max_context_ratio=0.8
        )
        
        self.logger.info(f"L0图谱生成器已初始化（改进版：语义相似度+LLM）")
        self.logger.info(f"输出目录: {self.config.output_dir}")
        self.logger.info(f"相似度阈值: {self.config.similarity_threshold}")
        self.logger.info(f"并发模式: Sequential Sample, {'Parallel' if self.config.enable_message_parallel else 'Sequential'} Messages")
        if self.config.enable_message_parallel:
            self.logger.info(f"消息级并发数: {self.config.max_workers_per_session}")
    
    def _setup_logging(self) -> logging.Logger:
        """Run setup logging."""
        logger = logging.getLogger(f"{__name__}.L0GraphGenerator")
        
        if self.config.debug_mode:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    
    
    def generate_all_l0_graphs(self) -> Dict[str, Any]:
        """Generate all l0 graphs."""
        start_time = time.time()
        self.logger.info("=" * 80)
        self.logger.info(" 开始生成L0层图谱（改进版：语义相似度+LLM）")
        self.logger.info("=" * 80)
        
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        locomo_data = self._load_locomo_data()
        if not locomo_data:
            raise ValueError("无法加载LOCOMO数据")
        
        # Dataset-specific handling used by the reproduction workflow.
        if self.config.sample_ids:
            id_set = set(self.config.sample_ids)
            locomo_data = [s for s in locomo_data if s.get('sample_id') in id_set]
            self.logger.info(f" 筛选指定样本: {self.config.sample_ids} → {len(locomo_data)} 个匹配")
        
        self.stats['total_samples'] = len(locomo_data)
        self.logger.info(f" 加载了 {len(locomo_data)} 个样本")
        
        #  V2: Always process samples sequentially for better control and debugging
        # Parallel processing happens at message level within each session
        if self.config.enable_sample_parallel and self.config.max_workers_sample > 1:
            self.logger.warning("  Sample-level parallelism is deprecated. Using sequential mode for samples.")
        self._generate_sequential(locomo_data)
        
        self.stats['processing_time'] = time.time() - start_time
        self._save_generation_stats()
        
        self.logger.info(f"\n L0图谱生成完成，耗时 {self.stats['processing_time']:.2f} 秒")
        self._print_summary()
        
        return self.stats
    
    def _load_locomo_data(self) -> List[Dict]:
        """Load locomo data."""
        try:
            with open(self.config.locomo_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.logger.info(f" 成功加载LOCOMO数据: {len(data)} 个样本")
            return data
        except Exception as e:
            self.logger.error(f" 加载LOCOMO数据失败: {e}")
            return []
    
    def _generate_sequential(self, locomo_data: List[Dict]):
        """Generate sequential."""
        for i, sample_data in enumerate(locomo_data, 1):
            sample_id = sample_data.get('sample_id', 'unknown')
            try:
                self.logger.info(f"\n{'='*60}")
                self.logger.info(f" 处理样本 {i}/{len(locomo_data)}: {sample_id}")
                self.logger.info(f"{'='*60}")
                
                l0_graph = self.generate_l0_graph(sample_data, sample_id)
                
                if l0_graph:
                    self.save_l0_graph(l0_graph, sample_id)
                    self.stats['processed_samples'] += 1
                    self.logger.info(f" 样本 {sample_id} 处理成功 ({i}/{len(locomo_data)})")
                else:
                    self.logger.warning(f"  样本 {sample_id} 处理失败 ({i}/{len(locomo_data)})")
            except Exception as e:
                self.logger.error(f" 处理样本 {sample_id} 失败: {e}")
                self.stats['failed_samples'].append({'sample_id': sample_id, 'error': str(e)})
            
            if i % 5 == 0 or i == len(locomo_data):
                progress = (i / len(locomo_data)) * 100
                self.logger.info(f"\n 处理进度: {i}/{len(locomo_data)} ({progress:.1f}%) - "
                            f"成功: {self.stats['processed_samples']}, 失败: {len(self.stats['failed_samples'])}")
    
    def generate_l0_graph(self, sample_data: Dict, sample_id: str) -> Dict[str, Any]:
        """Generate l0 graph."""
        try:
            l0_graph = {
                "sample_id": sample_id,
                "data_version": "1.0",
                "generation_time": datetime.now().isoformat(),
                "l0_conversations": [],
                "l0_relationships": [],
                "entities": [],
                "metadata": {
                    "total_sessions": 0,
                    "total_messages": 0,
                    "speakers": [],
                    "has_multimodal": False,
                    "relation_extraction_method": "similarity_llm_hybrid"
                }
            }
            
            conversation = sample_data.get('conversation', {})
            if not conversation:
                self.logger.warning(f"样本 {sample_id} 没有对话数据")
                return None
            
            speaker_a = conversation.get('speaker_a', 'Speaker_A')
            speaker_b = conversation.get('speaker_b', 'Speaker_B')
            l0_graph["metadata"]["speakers"] = [speaker_a, speaker_b]
            
            temp_semantic_map = self._create_temp_semantic_map()
            
            session_count = 0
            all_messages = []
            session_unit_map = {}  # {session_id: [MemoryUnit]}
            
            for key, messages in conversation.items():
                if not key.startswith('session_') or not isinstance(messages, list):
                    continue
                
                session_count += 1
                if (self.config.max_sessions_per_sample and 
                    session_count > self.config.max_sessions_per_sample):
                    break
                
                datetime_key = key + '_date_time'
                session_datetime = conversation.get(datetime_key, '')
                date, time = self._parse_session_datetime(session_datetime)
                
                session_units = []
                
                #  Contextual Retrieval V2: Build FULL session transcript upfront ("God Mode")
                # This allows the LLM to resolve pronouns using both past AND future context
                full_session_transcript_parts = []
                for msg_idx_pre, msg in enumerate(messages):
                    if isinstance(msg, dict):
                        msg_speaker = msg.get('speaker', 'Unknown')
                        msg_text = msg.get('text', str(msg))
                    else:
                        msg_speaker = speaker_a if msg_idx_pre % 2 == 0 else speaker_b
                        msg_text = str(msg)
                    # Include message index for easier target message identification
                    full_session_transcript_parts.append(f"[{msg_idx_pre+1}] {msg_speaker}: {msg_text}")
                full_session_transcript = "\n".join(full_session_transcript_parts)
                
                #  Step A: Prepare task arguments for all messages in this session
                message_tasks = []
                for msg_idx, message in enumerate(messages):
                    enhanced_text, has_multimodal = self._build_multimodal_text(message)
                    
                    if has_multimodal:
                        l0_graph["metadata"]["has_multimodal"] = True
                    
                    speaker = "unknown"
                    if isinstance(message, dict):
                        speaker = message.get('speaker', 'unknown')
                    else:
                        speaker = speaker_a if msg_idx % 2 == 0 else speaker_b
                    
                    if len(enhanced_text.strip()) < self.config.min_content_length:
                        continue
                    
                    unit_id = f"{sample_id}_{key}_msg_{msg_idx}"
                    
                    # Store task arguments for parallel/sequential LLM processing
                    message_tasks.append({
                        'msg_idx': msg_idx,
                        'unit_id': unit_id,
                        'speaker': speaker,
                        'enhanced_text': enhanced_text,
                        'has_multimodal': has_multimodal,
                        'original_message': message,
                        'session_date': date,
                        'session_time': time,
                        'session_datetime': session_datetime,
                        'session_id': key,
                        'session_count': session_count,
                        'full_session_transcript': full_session_transcript,
                        'participants': l0_graph["metadata"]["speakers"]
                    })
                
                #  Step B: Parallel LLM context generation (thread-safe)
                if self.config.enable_message_parallel and len(message_tasks) > 1:
                    indexing_results = self._process_session_messages_parallel(
                        message_tasks, 
                        max_workers=self.config.max_workers_per_session
                    )
                else:
                    # Sequential fallback
                    indexing_results = self._process_session_messages_sequential(message_tasks)
                
                #  Step C: Sequential graph building (NOT thread-safe: SemanticMap, list appends)
                for task, indexing_text in zip(message_tasks, indexing_results):
                    msg_idx = task['msg_idx']
                    unit_id = task['unit_id']
                    speaker = task['speaker']
                    enhanced_text = task['enhanced_text']
                    has_multimodal = task['has_multimodal']
                    message = task['original_message']
                    
                    conv_unit = {
                        "uid": unit_id,
                        "raw_data": {
                            "type": "conversation_message",
                            "speaker": speaker,
                            "message": enhanced_text,
                            "session_id": key,
                            "session_datetime": task['session_datetime'],
                            "date": task['session_date'],
                            "time": task['session_time'],
                            "dialogue_id": f"D{task['session_count']}:{msg_idx+1}",
                            "sample_id": sample_id,
                            "text_content": f"{speaker}: {enhanced_text}",
                            "has_multimodal": has_multimodal
                        },
                        "metadata": {
                            "memory_level": "L0_Observation",
                            "sample_id": sample_id,
                            "session_id": key,
                            "speaker": speaker,
                            "is_locomo_specialized": False,
                            "message_index": msg_idx,
                            "has_image": has_multimodal
                        },
                        #  NEW: Contextualized indexing text for improved retrieval
                        "indexing_text": indexing_text
                    }
                    
                    
                    if isinstance(message, dict):
                        if message.get('img_url'):
                            conv_unit["raw_data"]['img_url'] = message.get('img_url')
                        if message.get('blip_caption'):
                            conv_unit["raw_data"]['blip_caption'] = message.get('blip_caption')
                        if message.get('query'):
                            conv_unit["raw_data"]['image_keywords'] = message.get('query')
                        if message.get('dia_id'):
                            conv_unit["raw_data"]['original_dialogue_id'] = message.get('dia_id')
                        conv_unit["raw_data"]['original_text'] = message.get('text', '')
                    
                    l0_graph["l0_conversations"].append(conv_unit)
                    all_messages.append(conv_unit)
                    
                    memory_unit = MemoryUnit(
                        uid=unit_id,
                        raw_data=conv_unit["raw_data"],
                        metadata=conv_unit["metadata"]
                    )
                    
                    
                    temp_semantic_map.add_unit(
                        unit=memory_unit,
                        explicit_content_for_embedding=enhanced_text,
                        content_type_for_embedding='text',
                        rebuild_index_immediately=False
                    )
                    
                    session_units.append(memory_unit)
                
                self.logger.debug(f"  Session {key}: {len(message_tasks)} messages processed")
                
                if session_units:
                    session_unit_map[key] = session_units
            
            l0_graph["metadata"]["total_sessions"] = session_count
            l0_graph["metadata"]["total_messages"] = len(l0_graph["l0_conversations"])
            
            if not l0_graph["l0_conversations"]:
                self.logger.warning(f"样本 {sample_id} 没有有效的对话单元")
                return None
            
            
            temp_semantic_map.build_index()
            self.logger.debug(f"临时SemanticMap构建完成，包含 {len(temp_semantic_map.memory_units)} 个单元")
            
            if self.config.enable_relation_extraction:
                self._extract_intra_session_relations(l0_graph, all_messages, sample_id)
                
                self._extract_cross_session_relations_with_similarity(
                    l0_graph, session_unit_map, temp_semantic_map, sample_id
                )
            
            if self.config.enable_entity_extraction:
                self._extract_entities(l0_graph, all_messages)
            
            self.stats['total_conversations'] += len(l0_graph["l0_conversations"])
            self.stats['total_relations'] += len(l0_graph["l0_relationships"])
            self.stats['total_entities'] += len(l0_graph["entities"])
            
            self.stats['sample_details'][sample_id] = {
                'conversations': len(l0_graph["l0_conversations"]),
                'relations': len(l0_graph["l0_relationships"]),
                'entities': len(l0_graph["entities"]),
                'sessions': session_count,
                'has_multimodal': l0_graph["metadata"]["has_multimodal"]
            }
            
            return l0_graph
            
        except Exception as e:
            self.logger.error(f"生成样本 {sample_id} 的L0图谱失败: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return None
    
    def _create_temp_semantic_map(self) -> SemanticMap:
        """Create temp semantic map."""
        try:
            semantic_map = SemanticMap(
                embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
                embedding_dim=384,
                faiss_index_type="IDMap,Flat"
            )
            self.logger.debug("临时SemanticMap创建成功")
            return semantic_map
        except Exception as e:
            self.logger.error(f"创建临时SemanticMap失败: {e}")
            raise
    
    
    #  Message-Level Parallel Processing Methods
    
    
    def _process_session_messages_parallel(self, 
                                           message_tasks: List[Dict], 
                                           max_workers: int) -> List[str]:
        """
         Parallel LLM context generation for messages in a session.
        
        This method is thread-safe as it only calls the stateless LLM client.
        Graph building and SemanticMap updates happen sequentially in the main thread.
        
        Args:
            message_tasks: List of task dicts with message info
            max_workers: Max concurrent LLM calls
            
        Returns:
            List of indexing_text strings in the same order as message_tasks
        """
        results = [None] * len(message_tasks)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all LLM tasks
            future_to_idx = {
                executor.submit(
                    self._process_message_llm,
                    task['session_date'],
                    task['participants'],
                    task['full_session_transcript'],
                    task['speaker'],
                    task['enhanced_text']
                ): idx
                for idx, task in enumerate(message_tasks)
            }
            
            # Collect results maintaining order
            completed = 0
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                completed += 1
                try:
                    results[idx] = future.result()
                except Exception as e:
                    task = message_tasks[idx]
                    self.logger.warning(f"  LLM call failed for msg_{task['msg_idx']}: {e}")
                    # Fallback to original message
                    date_prefix = f"[{task['session_date']}] " if task['session_date'] != "unknown" else ""
                    results[idx] = f"{date_prefix}{task['speaker']}: {task['enhanced_text']}"
        
        return results
    
    def _process_session_messages_sequential(self, message_tasks: List[Dict]) -> List[str]:
        """
        Sequential LLM context generation (fallback when parallel is disabled).
        """
        results = []
        for task in message_tasks:
            indexing_text = self._process_message_llm(
                task['session_date'],
                task['participants'],
                task['full_session_transcript'],
                task['speaker'],
                task['enhanced_text']
            )
            results.append(indexing_text)
        return results
    
    def _process_message_llm(self,
                             session_date: str,
                             participants: List[str],
                             full_session_transcript: str,
                             speaker: str,
                             message_text: str) -> str:
        """
         Pure, thread-safe LLM call for context generation.
        
        This is the core LLM calling logic extracted for parallel execution.
        It has no side effects and only depends on its arguments.
        
        Args:
            session_date: Session date string
            participants: List of participant names
            full_session_transcript: Full session transcript
            speaker: Message speaker
            message_text: Original message text
            
        Returns:
            Contextualized indexing_text string
        """
        try:
            participants_str = ", ".join(participants) if participants else "Unknown participants"
            
            prompt = CONTEXTUAL_RETRIEVAL_PROMPT.format(
                date=session_date if session_date != "unknown" else "Unknown date",
                participants=participants_str,
                full_session_transcript=full_session_transcript,
                speaker=speaker,
                message_text=message_text
            )
            
            context_description = self.llm_client.generate_answer(
                prompt=prompt,
                temperature=0.1,
                max_tokens=250
            )
            
            context_description = context_description.strip()
            indexing_text = f"{context_description}\nOriginal: {message_text}"
            
            return indexing_text
            
        except Exception as e:
            self.logger.debug(f"LLM context generation failed: {e}")
            date_prefix = f"[{session_date}] " if session_date != "unknown" else ""
            return f"{date_prefix}{speaker}: {message_text}"
    
    def _generate_node_context(self, 
                               session_date: str,
                               participants: List[str],
                               full_session_transcript: str,
                               speaker: str,
                               message_text: str) -> str:
        """
         Generate contextualized indexing text using Full Session Global Context (V2).
        
        Key Change from V1: "God Mode" - The LLM sees the ENTIRE session transcript,
        allowing it to resolve pronouns using both past AND future context.
        
        This method creates a self-contained "indexing_text" that:
        1. Anchors the message to a specific absolute date/time
        2. Resolves pronouns using the full session (including future messages)
        3. Captures the intent and classifies the event type
        
        Args:
            session_date: The date of the session (e.g., "2023-08-15")
            participants: List of participant names
            full_session_transcript: Complete session transcript (all messages with speaker IDs)
            speaker: The speaker of the target message
            message_text: The original message text
            
        Returns:
            Contextualized string: "{generated_context}\nOriginal: {original_message}"
        """
        try:
            # Format participants
            participants_str = ", ".join(participants) if participants else "Unknown participants"
            
            #  V2: Use full session transcript (no truncation by default)
            # Safety check: if transcript is extremely long (>15000 chars), log a warning but still use it
            if len(full_session_transcript) > 15000:
                self.logger.debug(f"Long session transcript ({len(full_session_transcript)} chars), using full context")
            
            # Build the prompt using the V2 template
            prompt = CONTEXTUAL_RETRIEVAL_PROMPT.format(
                date=session_date if session_date != "unknown" else "Unknown date",
                participants=participants_str,
                full_session_transcript=full_session_transcript,
                speaker=speaker,
                message_text=message_text
            )
            
            # Call LLM to generate contextualized description
            context_description = self.llm_client.generate_answer(
                prompt=prompt,
                temperature=0.1,
                max_tokens=250  # Slightly increased for better context resolution
            )
            
            # Clean up the response
            context_description = context_description.strip()
            
            # Combine: contextualized description + original message
            indexing_text = f"{context_description}\nOriginal: {message_text}"
            
            self.logger.debug(f"Generated indexing_text for message: {indexing_text[:100]}...")
            return indexing_text
            
        except Exception as e:
            self.logger.warning(f"  Failed to generate node context: {e}. Falling back to original message.")
            # Fallback: return original message with basic date prefix
            date_prefix = f"[{session_date}] " if session_date != "unknown" else ""
            return f"{date_prefix}{speaker}: {message_text}"
    
    def _extract_cross_session_relations_with_similarity(self,
                                                    l0_graph: Dict,
                                                    session_unit_map: Dict[str, List[MemoryUnit]],
                                                    semantic_map: SemanticMap,
                                                    sample_id: str):
        """Extract cross session relations with similarity."""
        if len(session_unit_map) < 2:
            self.logger.debug(f"样本 {sample_id} 只有 {len(session_unit_map)} 个会话，跳过跨会话关系抽取")
            return
        
        self.logger.debug(f"开始跨会话关系抽取（相似度阈值: {self.config.similarity_threshold}）")
        
        session_summaries = {}
        
        for session_id, units in session_unit_map.items():
            if len(units) < 3:
                continue
            
            summary_text = self._generate_session_summary_with_llm([
                {
                    "raw_data": {"speaker": u.raw_data.get('speaker'), "message": u.raw_data.get('message')},
                    "metadata": {}
                } 
                for u in units
            ])
            
            if not summary_text:
                continue
            
            try:
                summary_embedding = semantic_map._get_text_embedding(summary_text)
                
                if summary_embedding is not None:
                    session_summaries[session_id] = {
                        'summary_text': summary_text,
                        'embedding': summary_embedding,
                        'representative_unit': units[0],
                        'all_units': units
                    }
                    self.logger.debug(f"会话 {session_id} 摘要embedding生成成功")
                else:
                    self.logger.warning(f"会话 {session_id} 摘要embedding生成失败")
            except Exception as e:
                self.logger.warning(f"会话 {session_id} embedding计算失败: {e}")
        
        if len(session_summaries) < 2:
            self.logger.debug(f"有效会话摘要不足2个，跳过跨会话关系")
            return
        
        session_list = list(session_summaries.keys())
        similarity_filtered_count = 0
        llm_confirmed_count = 0
        
        for i in range(len(session_list)):
            for j in range(i + 1, len(session_list)):
                session1 = session_list[i]
                session2 = session_list[j]
                
                similarity = self._compute_cosine_similarity(
                    session_summaries[session1]['embedding'],
                    session_summaries[session2]['embedding']
                )
                
                self.logger.debug(f"会话对 ({session1}, {session2}) 相似度: {similarity:.3f}")
                
                if similarity >= self.config.similarity_threshold:
                    similarity_filtered_count += 1
                    self.logger.debug(f" 相似度通过筛选: {session1} <-> {session2} ({similarity:.3f})")
                    
                    try:
                        relation = self._analyze_session_relation_with_llm(
                            session_summaries[session1]['summary_text'],
                            session_summaries[session2]['summary_text'],
                            session1, session2
                        )
                        
                        if (relation['has_relation'] and 
                            relation['confidence'] >= self.config.llm_confidence_threshold):
                            
                            llm_confirmed_count += 1
                            
                            repr_unit1 = session_summaries[session1]['representative_unit']
                            repr_unit2 = session_summaries[session2]['representative_unit']
                            
                            l0_graph["l0_relationships"].append({
                                "source_uid": repr_unit1.uid,
                                "target_uid": repr_unit2.uid,
                                "type": relation['relation_type'],
                                "properties": {
                                    "sample_id": sample_id,
                                    "semantic_similarity": float(similarity),
                                    "llm_confidence": relation['confidence'],
                                    "description": relation['description'],
                                    "llm_generated": True,
                                    "cross_session": True,
                                    "relation_scope": "intra_sample_cross_session",
                                    "extraction_method": "similarity_llm_hybrid"
                                }
                            })
                            
                            self.logger.debug(
                                f" 添加跨会话关系: {repr_unit1.uid} -> {repr_unit2.uid} "
                                f"(相似度: {similarity:.3f}, LLM置信度: {relation['confidence']:.3f})"
                            )
                        else:
                            self.logger.debug(
                                f" LLM未确认关系: {session1} <-> {session2} "
                                f"(has_relation: {relation['has_relation']}, "
                                f"confidence: {relation['confidence']:.3f})"
                            )
                    
                    except Exception as e:
                        self.logger.warning(f"LLM分析会话对 ({session1}, {session2}) 失败: {e}")
                else:
                    self.logger.debug(f" 相似度未通过筛选: {session1} <-> {session2} ({similarity:.3f})")
        
        self.stats['similarity_filtered_relations'] += similarity_filtered_count
        self.stats['llm_confirmed_relations'] += llm_confirmed_count
        
        self.logger.info(
            f"样本 {sample_id} 跨会话关系抽取完成: "
            f"相似度筛选={similarity_filtered_count}, LLM确认={llm_confirmed_count}"
        )
    
    def _compute_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity."""
        try:
            a = vec1.astype(np.float32)
            b = vec2.astype(np.float32)
            
            dot_product = float(np.dot(a, b))
            norm_a = float(np.linalg.norm(a))
            norm_b = float(np.linalg.norm(b))
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
            
            cosine_sim = dot_product / (norm_a * norm_b + 1e-8)
            
            return (cosine_sim + 1.0) / 2.0
        
        except Exception as e:
            self.logger.warning(f"计算余弦相似度失败: {e}")
            return 0.0
    
    
    def _parse_session_datetime(self, datetime_str: str) -> Tuple[str, str]:
        """Parse session datetime."""
        try:
            match = re.match(r'(\d{1,2}):(\d{2})\s*(am|pm)\s*on\s*(\d{1,2})\s*(\w+),?\s*(\d{4})', datetime_str)
            if match:
                hour, minute, ampm, day, month, year = match.groups()
                month_map = {
                    'january': '01', 'february': '02', 'march': '03', 'april': '04',
                    'may': '05', 'june': '06', 'july': '07', 'august': '08',
                    'september': '09', 'october': '10', 'november': '11', 'december': '12'
                }
                month_num = month_map.get(month.lower(), '01')
                hour = int(hour)
                if ampm.lower() == 'pm' and hour != 12:
                    hour += 12
                elif ampm.lower() == 'am' and hour == 12:
                    hour = 0
                date = f"{year}-{month_num.zfill(2)}-{day.zfill(2)}"
                time = f"{hour:02d}:{minute}"
                return date, time
            return "unknown", "unknown"
        except Exception as e:
            self.logger.error(f"解析时间失败: {e}")
            return "unknown", "unknown"
    
    def _build_multimodal_text(self, message: Any) -> Tuple[str, bool]:
        """Build multimodal text."""
        if isinstance(message, str):
            return message, False
        if not isinstance(message, dict):
            return str(message), False
        
        dialogue_text = message.get('text', '').strip()
        has_image_info = bool(message.get('img_url') or message.get('blip_caption') or message.get('query'))
        
        if not has_image_info:
            return dialogue_text, False
        
        text_parts = []
        if dialogue_text:
            text_parts.append(f"[dialogue] {dialogue_text}")
        
        blip_caption = message.get('blip_caption', '').strip()
        if blip_caption:
            text_parts.append(f"[image_description] {blip_caption}")
        
        image_keywords = message.get('query', '').strip()
        if image_keywords:
            text_parts.append(f"[image_keywords] {image_keywords}")
        
        return '\n'.join(text_parts), True
    
    def _extract_intra_session_relations(self, l0_graph: Dict, all_messages: List[Dict], sample_id: str):
        """Extract intra session relations."""
        session_groups = {}
        for msg in all_messages:
            session_id = msg["raw_data"]["session_id"]
            if session_id not in session_groups:
                session_groups[session_id] = []
            session_groups[session_id].append(msg)
        
        for session_id, messages in session_groups.items():
            if len(messages) < 2:
                continue
            messages.sort(key=lambda x: x["metadata"]["message_index"])
            
            for i in range(len(messages) - 1):
                current_msg = messages[i]
                next_msg = messages[i + 1]
                
                try:
                    relation = self._analyze_conversation_relation_with_llm(current_msg, next_msg)
                    
                    if relation['has_relation']:
                        l0_graph["l0_relationships"].append({
                            "source_uid": current_msg["uid"],
                            "target_uid": next_msg["uid"],
                            "type": relation['relation_type'],
                            "properties": {
                                "session_id": session_id,
                                "sample_id": sample_id,
                                "temporal_distance": 1,
                                "confidence": relation['confidence'],
                                "description": relation['description'],
                                "llm_generated": True,
                                "relation_scope": "intra_session"
                            }
                        })
                except Exception as e:
                    self.logger.warning(f"LLM关系分析失败: {e}")
    
    def _analyze_conversation_relation_with_llm(self, msg1: Dict, msg2: Dict) -> Dict[str, Any]:
        """Run analyze conversation relation with LLM."""
        speaker1 = msg1["raw_data"]["speaker"]
        speaker2 = msg2["raw_data"]["speaker"]
        text1 = msg1["raw_data"]["message"]
        text2 = msg2["raw_data"]["message"]
        
        prompt = f"""Analyze the relationship between the following two consecutive dialogues.

        Dialogue 1 ({speaker1}): "{text1}"
        Dialogue 2 ({speaker2}): "{text2}"

        Please determine if there is a meaningful relationship between these two dialogues. If there is, please:
        1. Determine the relationship type (response, question, topic transition, emotional support, agreement, disagreement, supplementary explanation, etc.)
        2. Give a confidence score from 1-10
        3. Provide a brief relationship description

        Please respond in JSON format:
        {{
            "has_relation": true/false,
            "relation_type": "relationship type",
            "confidence": 0.0-1.0,
            "description": "relationship description"
        }}

        Generate a response in JSON format. All textual content within the JSON must be in English."""

        try:
            response = self.llm_client.generate_answer(prompt=prompt, temperature=self.config.llm_temperature)
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:-3]
            elif response.startswith('```'):
                response = response[3:-3]
            result = json.loads(response)
            return {
                'has_relation': result.get('has_relation', False),
                'relation_type': result.get('relation_type', 'DIALOGUE_FLOW'),
                'confidence': min(max(result.get('confidence', 0.5), 0.0), 1.0),
                'description': result.get('description', 'Dialogue flow relationship')
            }
        except Exception as e:
            self.logger.warning(f"LLM关系分析失败: {e}")
            return {'has_relation': True, 'relation_type': 'DIALOGUE_FLOW', 'confidence': 0.5, 'description': 'Dialogue flow relationship'}
    
    def _generate_session_summary_with_llm(self, messages: List[Dict]) -> str:
        """Generate session summary with LLM."""
        dialogue_text = ""
        for msg in messages:
            speaker = msg["raw_data"]["speaker"]
            text = msg["raw_data"]["message"]
            dialogue_text += f"{speaker}: {text}\n"
        
        prompt = f"""Please generate a concise topic summary (within 30 words) for the following dialogue:

        {dialogue_text}

        The summary should capture the main topic and content of the dialogue. Return only the summary text, no other content."""

        try:
            summary = self.llm_client.generate_answer(prompt=prompt, temperature=0.1)
            return summary.strip()
        except Exception as e:
            self.logger.warning(f"生成会话摘要失败: {e}")
            return ""
    
    def _analyze_session_relation_with_llm(self, summary1: str, summary2: str, session1: str, session2: str) -> Dict[str, Any]:
        """Run analyze session relation with LLM."""
        prompt = f"""Analyze the thematic relationship between the following two dialogue sessions:

        Session {session1} Summary: "{summary1}"
        Session {session2} Summary: "{summary2}"

        Please determine if these two sessions discuss related topics. If they are related, please:
        1. Determine the relationship type (topic continuation, topic regression, related discussion, etc.)
        2. Give a confidence score (only consider it as related if above 0.6)
        3. Provide relationship description

        Respond in JSON format:
        {{
            "has_relation": true/false,
            "relation_type": "relationship type",
            "confidence": 0.0-1.0,
            "description": "relationship description"
        }}

        Generate a response in JSON format. All textual content within the JSON must be in English."""

        try:
            response = self.llm_client.generate_answer(prompt=prompt, temperature=0.1)
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:-3]
            elif response.startswith('```'):
                response = response[3:-3]
            result = json.loads(response)
            confidence = result.get('confidence', 0.0)
            has_relation = result.get('has_relation', False) and confidence >= self.config.llm_confidence_threshold
            return {
                'has_relation': has_relation,
                'relation_type': result.get('relation_type', 'TOPIC_RELATED'),
                'confidence': confidence,
                'description': result.get('description', 'Topic related')
            }
        except Exception as e:
            self.logger.warning(f"会话关系分析失败: {e}")
            return {'has_relation': False, 'relation_type': '', 'confidence': 0.0, 'description': ''}
    
    def _extract_entities(self, l0_graph: Dict, all_messages: List[Dict]):
        """Extract entities."""
        speakers = set()
        for msg in all_messages:
            speakers.add(msg["raw_data"]["speaker"])
        for speaker in speakers:
            entity_uid = f"entity_{speaker}"
            l0_graph["entities"].append({
                "uid": entity_uid,
                "type": "person",
                "name": speaker,
                "attributes": {}
            })
    
    def save_l0_graph(self, l0_graph: Dict, sample_id: str):
        """Save l0 graph."""
        output_file = os.path.join(self.config.output_dir, f"{sample_id}_l0_graph.json")
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(l0_graph, f, indent=2, ensure_ascii=False)
            self.logger.info(f" L0图谱已保存: {output_file}")
        except Exception as e:
            self.logger.error(f"保存L0图谱失败: {e}")
            raise
    
    def _save_generation_stats(self):
        """Save generation stats."""
        stats_file = os.path.join(self.config.output_dir, "l0_generation_stats.json")
        try:
            enhanced_stats = self.stats.copy()
            enhanced_stats['generation_config'] = {
                'enable_relation_extraction': self.config.enable_relation_extraction,
                'enable_entity_extraction': self.config.enable_entity_extraction,
                'llm_model': self.config.llm_model,
                'similarity_threshold': self.config.similarity_threshold,
                'llm_confidence_threshold': self.config.llm_confidence_threshold,
                'top_k_candidates': self.config.top_k_candidates,
                'enable_message_parallel': self.config.enable_message_parallel,
                'max_workers_per_session': self.config.max_workers_per_session,
                'concurrency_model': 'sequential_sample_parallel_messages',
                'extraction_method': 'similarity_llm_hybrid'
            }
            enhanced_stats['completion_time'] = datetime.now().isoformat()
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(enhanced_stats, f, indent=2, ensure_ascii=False)
            self.logger.info(f" 统计信息已保存: {stats_file}")
        except Exception as e:
            self.logger.error(f"保存统计信息失败: {e}")
    
    def _print_summary(self):
        """Run print summary."""
        print("\n" + "=" * 80)
        print("L0图谱生成摘要 (Step 1 - 改进版)")
        print("=" * 80)
        print(f"总样本数:           {self.stats['total_samples']}")
        print(f"处理成功:           {self.stats['processed_samples']}")
        print(f"处理失败:           {len(self.stats['failed_samples'])}")
        print(f"总对话单元:         {self.stats['total_conversations']}")
        print(f"总关系数:           {self.stats['total_relations']}")
        print(f"  - 相似度筛选:     {self.stats['similarity_filtered_relations']}")
        print(f"  - LLM确认:        {self.stats['llm_confirmed_relations']}")
        print(f"总实体数:           {self.stats['total_entities']}")
        print(f"处理时间:           {self.stats['processing_time']:.2f} 秒")
        print(f"LLM模型:            {self.config.llm_model}")
        print(f"相似度阈值:         {self.config.similarity_threshold}")
        print(f"LLM置信度阈值:      {self.config.llm_confidence_threshold}")
        print(f"并发模式:           Sequential Sample, {'Parallel' if self.config.enable_message_parallel else 'Sequential'} Messages")
        if self.config.enable_message_parallel:
            print(f"消息级并发数:       {self.config.max_workers_per_session}")
        print(f"输出目录:           {self.config.output_dir}")
        print("=" * 80 + "\n")


def main():
    """Run the command-line entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Step 1: 生成L0层对话图谱（改进版：语义相似度+LLM）")
    parser.add_argument("--locomo-file", default=str(paths.LOCOMO_RAW_FILE), help="locomo10数据文件路径")
    parser.add_argument("--output-dir", default=str(paths.LOCOMO_HIERARCHICAL_CONTENT_STEP1_DIR), help="输出目录")
    
    #  V2 Concurrency: Sequential Sample, Parallel Messages
    parser.add_argument("--no-message-parallel", action="store_true", help="禁用消息级并行LLM调用")
    parser.add_argument("--max-workers-per-session", type=int, default=40, help="每个Session的最大并发LLM调用数")
    
    # Legacy (deprecated, kept for backwards compatibility)
    parser.add_argument("--no-parallel", action="store_true", help="[Deprecated] 已弃用，使用 --no-message-parallel")
    parser.add_argument("--max-workers", type=int, default=1, help="[Deprecated] 已弃用，使用 --max-workers-per-session")
    
    parser.add_argument("--sample-ids", nargs='+', help="指定要处理的sample ID列表，如 --sample-ids conv-26 conv-30")
    parser.add_argument("--enable-relations", action="store_true", help="启用关系边生成")
    parser.add_argument("--enable-entities", action="store_true", help="启用实体提取")
    parser.add_argument("--extract-model", default="qwen-3.5-plus-thinking", help="抽取模型名称")
    parser.add_argument("--similarity-threshold", type=float, default=0.75, help="语义相似度阈值")
    parser.add_argument("--llm-confidence-threshold", type=float, default=0.6, help="LLM置信度阈值")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    
    args = parser.parse_args()
    
    # Handle legacy args
    enable_msg_parallel = not args.no_message_parallel
    if args.no_parallel:
        print("  Warning: --no-parallel is deprecated. Use --no-message-parallel instead.")
        enable_msg_parallel = False
    
    max_workers = args.max_workers_per_session
    if args.max_workers != 1:
        print("  Warning: --max-workers is deprecated. Use --max-workers-per-session instead.")
        max_workers = args.max_workers
    
    config = L0GraphConfig(
        locomo_file_path=args.locomo_file,
        output_dir=args.output_dir,
        enable_sample_parallel=False,  # Always sequential at sample level
        max_workers_sample=1,
        enable_message_parallel=enable_msg_parallel,
        max_workers_per_session=max_workers,
        enable_relation_extraction=args.enable_relations,
        enable_entity_extraction=args.enable_entities,
        llm_model=args.extract_model,
        similarity_threshold=args.similarity_threshold,
        llm_confidence_threshold=args.llm_confidence_threshold,
        sample_ids=args.sample_ids,
        debug_mode=args.debug
    )
    
    generator = L0GraphGenerator(config)
    
    try:
        stats = generator.generate_all_l0_graphs()
        
        print("\n L0图谱生成完成（改进版）!")
        print(f" 输出目录: {config.output_dir}")
        print(f" 统计文件: {config.output_dir}/l0_generation_stats.json")
        print(f"\n 关系抽取统计:")
        print(f"   - 相似度筛选候选: {stats['similarity_filtered_relations']}")
        print(f"   - LLM最终确认:    {stats['llm_confirmed_relations']}")
        print(f"   - 筛选效率:       {stats['llm_confirmed_relations']/max(stats['similarity_filtered_relations'], 1)*100:.1f}%")
        
        return 0
    except Exception as e:
        print(f"\n L0图谱生成失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())