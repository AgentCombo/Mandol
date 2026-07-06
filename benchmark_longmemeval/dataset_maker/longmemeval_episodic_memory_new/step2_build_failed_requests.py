#!/usr/bin/env python3
"""Utilities for step2 build failed requests."""

import json
import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime
from string import Template
from collections import defaultdict
from mandol.core import paths


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EpisodicMemoryCategory:
    
    USER_ATTRIBUTE = "USER_ATTRIBUTE"
    EPISODIC_EVENT = "EPISODIC_EVENT"
    INVENTORY_ITEM = "INVENTORY_ITEM"
    PREFERENCE_HABIT = "PREFERENCE_HABIT"
    RELATIONSHIP_FACT = "RELATIONSHIP_FACT"
    QUANTITATIVE_FACT = "QUANTITATIVE_FACT"
    TEMPORAL_MARKER = "TEMPORAL_MARKER"
    ASSISTANT_KNOWLEDGE = "ASSISTANT_KNOWLEDGE"
    AGGREGATABLE_ITEM = "AGGREGATABLE_ITEM"
    IMPLICIT_CONSTRAINT = "IMPLICIT_CONSTRAINT"
    
    @classmethod
    def get_all_categories(cls) -> List[str]:
        return [
            cls.USER_ATTRIBUTE,
            cls.EPISODIC_EVENT,
            cls.INVENTORY_ITEM,
            cls.PREFERENCE_HABIT,
            cls.RELATIONSHIP_FACT,
            cls.QUANTITATIVE_FACT,
            cls.TEMPORAL_MARKER,
            cls.ASSISTANT_KNOWLEDGE,
            cls.AGGREGATABLE_ITEM,
            cls.IMPLICIT_CONSTRAINT
        ]


class ErrorRequestRegenerator:
    
    def __init__(self,
                 dataset_path: str = str(paths.LONGMEMEVAL_S_CLEANED_FILE),
                 error_dir: str = str(paths.LONGMEMEVAL_EPISODIC_NEW_BATCH_RESULTS_ERROR_DIR),
                 output_dir: str = str(paths.LONGMEMEVAL_EPISODIC_NEW_BATCH_REQUESTS_ERROR_DIR),
                 sessions_per_group: int = 1):
        self.dataset_path = Path(dataset_path)
        self.error_dir = Path(error_dir)
        self.output_dir = Path(output_dir)
        self.sessions_per_group = sessions_per_group
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f" 数据集路径: {self.dataset_path}")
        logger.info(f" Error目录: {self.error_dir}")
        logger.info(f" 输出目录: {self.output_dir}")
        logger.info(f" 每组session数: {self.sessions_per_group}")
    
    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load dataset."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"数据集文件不存在: {self.dataset_path}")
        
        logger.info(f" 加载数据集: {self.dataset_path}")
        
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f" 加载完成，共 {len(data)} 个 QA 样本")
        return data
    
    def parse_custom_id(self, custom_id: str) -> Optional[Tuple[int, int, int, int]]:
        """Parse custom id."""
        pattern = r'qa_(\d+)_group_(\d+)_sessions_(\d+)_(\d+)'
        match = re.match(pattern, custom_id)
        if match:
            qa_idx = int(match.group(1))
            group_idx = int(match.group(2))
            start_session = int(match.group(3))
            end_session = int(match.group(4))
            return (qa_idx, group_idx, start_session, end_session)
        return None
    
    def load_error_requests(self, error_file: Path) -> List[Dict[str, Any]]:
        """Returns: [(custom_id, qa_index, group_idx, start_session, end_session), ...]."""
        errors = []
        
        if not error_file.exists():
            logger.warning(f" Error文件不存在: {error_file}")
            return errors
        
        with open(error_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    error_data = json.loads(line)
                    custom_id = error_data.get("custom_id", "")
                    
                    parsed = self.parse_custom_id(custom_id)
                    if parsed:
                        qa_idx, group_idx, start_session, end_session = parsed
                        errors.append({
                            "custom_id": custom_id,
                            "qa_index": qa_idx,
                            "group_idx": group_idx,
                            "start_session": start_session,
                            "end_session": end_session,
                            "error_message": error_data.get("error", {}).get("message", "Unknown error")
                        })
                    else:
                        if "retry" not in custom_id:
                            logger.warning(f" 无法解析 custom_id: {custom_id}")
                        
                except json.JSONDecodeError as e:
                    logger.error(f" JSON解析错误: {e}")
                    continue
        
        return errors
    
    def load_all_error_requests(self) -> Dict[int, List[Dict[str, Any]]]:
        """Returns: {qa_index: [error_info, ...], ...}."""
        all_errors = defaultdict(list)
        
        error_files = list(self.error_dir.glob("*_error.jsonl"))
        
        if not error_files:
            logger.warning(f" 在 {self.error_dir} 下没有找到任何 error.jsonl 文件")
            return dict(all_errors)
        
        logger.info(f" 找到 {len(error_files)} 个 error 文件")
        
        for error_file in sorted(error_files):
            logger.info(f"  - {error_file.name}")
            errors = self.load_error_requests(error_file)
            
            for error_info in errors:
                qa_idx = error_info["qa_index"]
                all_errors[qa_idx].append(error_info)
        
        logger.info(f"\n 总共加载 {sum(len(v) for v in all_errors.values())} 个失败请求")
        logger.info(f"   涉及 {len(all_errors)} 个不同的 QA")
        
        return dict(all_errors)
    
    def _build_episodic_memory_extraction_prompt(self) -> Template:
        """Build episodic memory extraction prompt."""
        
        template_string = """You are a **Memory Archivist** specializing in personal episodic memory preservation for AI assistants. Your task is to extract **Atomic Memory Facts** from the user's conversation history.

        ## CRITICAL UNDERSTANDING

        This is NOT a Named Entity Recognition task. This is an **Episodic Memory Extraction** task designed to answer questions like:
        - "What is X?" (simple fact)
        - "How many X did I do?" (aggregation)
        - "Which happened first, X or Y?" (temporal ordering)
        - "What did you recommend for X?" (assistant knowledge recall)
        - "Can you suggest X for me?" (preference-based recommendation)

         WRONG approach (NER-style):
        - "Target" → Organization
        - "45 minutes" → Duration
        - "gray" → Color

         CORRECT approach (Episodic Memory):
        - "User redeemed a $$5 coupon on coffee creamer at Target" → EPISODIC_EVENT
        - "User's daily commute is 45 minutes each way" → QUANTITATIVE_FACT + PREFERENCE_HABIT
        - "User repainted bedroom walls to a lighter shade of gray" → EPISODIC_EVENT (preserve exact color!)

        ## CONVERSATION SESSIONS

        $sessions_text

        ## SESSION REFERENCE DATE: $reference_date

        Use this date to convert ALL relative time expressions to absolute dates.

        ---

        ## MEMORY FACT CATEGORIES (10 Types)

        ### 1. USER_ATTRIBUTE (用户属性)
        Static profile information that defines who the user is.

        **Extract:**
        - Educational background: degrees, schools, majors, graduation years
        - Professional history: CURRENT job, PREVIOUS jobs (mark temporal state!)
        - Personal identity: ethnicity, nationality, age, birthdate
        - Physical attributes: height, weight
        - Beliefs/Stances: religious views, political views (with temporal state if changed!)

        **Examples:**
        ```json
        {
        "category": "USER_ATTRIBUTE",
        "content": "User graduated with a Business Administration degree",
        "attributes": {"degree": "Business Administration", "state": "completed"}
        }
        ```

        ### 2. EPISODIC_EVENT (情景事件)
        Specific actions the user performed at a specific time/place.

        **Extract the COMPLETE event structure:**
        - WHAT action was performed
        - WHERE it happened (preserve exact location names!)
        - WHEN it happened (convert relative to absolute!)
        - WHO was involved
        - HOW it was done
        - WHY (if mentioned)

        **Examples:**
        ```json
        {
        "category": "EPISODIC_EVENT",
        "content": "User attended The Glass Menagerie at the local community theater",
        "attributes": {
            "action": "attended play",
            "object": "The Glass Menagerie",
            "location": "local community theater"
        },
        "temporal": {"absolute_date": "2023-05-23"}
        }
        ```

        ### 3. INVENTORY_ITEM (物品清单)
        Things the user owns, bought, or possesses.

        **CRITICAL: Preserve ALL details:**
        - Brand/Model: Nike, iPhone 13 Pro, Fitbit Inspire HR
        - Specifications: 500 Mbps, 16GB RAM, 55-inch
        - Color/Style: "a lighter shade of gray", "yellow dress"
        - Quantity: "three bikes", "20 playlists"
        - Acquisition details: where bought, when bought, price paid
        - Pet details: name, breed, age

        **Examples:**
        ```json
        {
        "category": "INVENTORY_ITEM",
        "content": "User owns a cat named Luna",
        "attributes": {"type": "pet", "species": "cat", "name": "Luna"}
        }
        ```

        ### 4. PREFERENCE_HABIT (偏好习惯)
        User's likes, dislikes, routines, and regular behaviors.

        **Extract:**
        - Explicit preferences: "favorite", "prefer", "like", "hate", "love"
        - Routines: "usually", "always", "every [day/week]", "typically"
        - Time-based habits: wake up time, work schedule, exercise routine
        - Consumption habits: brands used, services subscribed

        **Examples:**
        ```json
        {
        "category": "PREFERENCE_HABIT",
        "content": "User stops checking work emails at 7 pm",
        "attributes": {"habit": "no work emails after 7 pm", "frequency": "daily"}
        }
        ```

        ### 5. RELATIONSHIP_FACT (人际关系)
        Social connections and their attributes.

        **Extract:**
        - Relationship type: sister, friend, colleague, cousin, therapist
        - Person's name (if mentioned)
        - Person's attributes: where they live, what they do, their characteristics
        - Shared experiences with the user

        **Examples:**
        ```json
        {
        "category": "RELATIONSHIP_FACT",
        "content": "User's sister Emily lives in Denver",
        "attributes": {"relation": "sister", "name": "Emily", "location": "Denver"}
        }
        ```

        ### 6. QUANTITATIVE_FACT (数值事实)
        Numerical information with full context.

        **CRITICAL: ALWAYS include units and context!**

        **Extract:**
        - Prices: "$$800", "$$5 coupon"
        - Durations: "45 minutes each way", "4 hours", "two weeks"
        - Quantities: "20 playlists", "7 shirts", "three bikes"
        - Measurements: "500 Mbps", "16GB", "55-inch"
        - Percentages/Ratios: "10% discount", "3:1 ratio"
        - Scores/Records: "132 points", "25:50 time"

        **Examples:**
        ```json
        {
        "category": "QUANTITATIVE_FACT",
        "content": "User's daily commute is 45 minutes each way",
        "attributes": {
            "quantity": "45 minutes each way",
            "type": "commute duration",
            "note": "90 minutes total round trip"
        }
        }
        ```

        ### 7. TEMPORAL_MARKER (时间标记)
        Temporal information for reasoning about event sequences.

        **CRITICAL: Convert relative to absolute dates AND track event ordering!**

        **Extract:**
        - Relative references: "last Tuesday", "two weeks ago", "recently"
        - Absolute dates: "February 14th", "2023/05/30"
        - Duration spans: "for two weeks", "over a year"
        - Sequence markers: "before", "after", "first", "then"
        - Event ordering: which events happened in what sequence

        **Examples:**
        ```json
        {
        "category": "TEMPORAL_MARKER",
        "content": "User was in Japan for two weeks",
        "attributes": {"duration": "two weeks", "location": "Japan"},
        "temporal": {
            "relative_expression": "two weeks",
            "event_sequence_tag": "japan_trip"
        }
        }
        ```

        ### 8. ASSISTANT_KNOWLEDGE (Assistant 提供的知识)
        Information, recommendations, or advice that the ASSISTANT provided to the user.

        **CRITICAL: This captures what the AI assistant said, not what the user said!**

        **Extract:**
        - Recommendations: restaurants, hotels, products, services
        - Factual information: historical facts, technical details, procedures
        - Lists provided: steps, items, options
        - Specific details: phone numbers, addresses, names mentioned by assistant
        - Creative content: stories, songs, scripts written by assistant

        **Examples:**
        ```json
        {
        "category": "ASSISTANT_KNOWLEDGE",
        "content": "Assistant recommended Roscioli restaurant in Rome for Italian food",
        "attributes": {
            "type": "recommendation",
            "topic": "restaurant",
            "name": "Roscioli",
            "location": "Rome",
            "cuisine": "Italian"
        },
        "source": "assistant_response"
        }
        ```

        ### 9. AGGREGATABLE_ITEM (可聚合项目)
        Items that belong to a countable/summable group for answering aggregation questions.

        **CRITICAL: Tag items with aggregation keys so they can be counted/summed later!**

        **Use for questions like:**
        - "How many model kits have I worked on?" → count items with aggregation_key="model_kits"
        - "How much total money did I spend on X?" → sum items with aggregation_key="X_expenses"
        - "How many weddings did I attend?" → count items with aggregation_key="weddings_attended"

        **Examples:**
        ```json
        {
        "category": "AGGREGATABLE_ITEM",
        "content": "User bought Tamiya 1/48 scale Spitfire Mk.V model kit",
        "attributes": {
            "item": "Tamiya 1/48 scale Spitfire Mk.V",
            "type": "model kit"
        },
        "aggregation": {
            "key": "model_kits_owned",
            "operation": "count",
            "value": 1
        }
        }
        ```

        ### 10. IMPLICIT_CONSTRAINT (隐含约束)
        Implicit preferences, limitations, or constraints that affect recommendations.

        **CRITICAL: Capture information that would affect personalized suggestions!**

        **Extract:**
        - Negative preferences: things user wants to avoid
        - Limitations: time constraints, budget limits, dietary restrictions
        - Context for recommendations: existing equipment, skill level, past experiences
        - Soft constraints: "prefer X over Y", "looking for something like Z"

        **Examples:**
        ```json
        {
        "category": "IMPLICIT_CONSTRAINT",
        "content": "User wants to avoid using phone or watching TV in the evening as they affect sleep",
        "attributes": {
            "constraint_type": "avoidance",
            "avoid": ["phone use", "TV watching"],
            "reason": "affects sleep quality",
            "time_context": "evening"
        },
        "affects_recommendations": ["evening activities", "relaxation suggestions"]
        }
        ```

        ---

        ## SPECIAL EXTRACTION RULES

        ### Rule 1: NUANCE PRESERVATION (细节保留)
        **NEVER simplify or generalize details!**

         "gray" 
         "a lighter shade of gray"

         "45 minutes"
         "45 minutes each way"

         "my job"
         "Marketing specialist at a small startup" (and mark if PREVIOUS or CURRENT)

        ### Rule 2: TEMPORAL RESOLUTION (时间转换)
        **Convert ALL relative time to absolute dates using the session reference date.**

        If reference date is 2023/05/30 (Tuesday):
        - "last Tuesday" → 2023/05/23
        - "yesterday" → 2023/05/29
        - "two weeks ago" → 2023/05/16
        - "last month" → approximately 2023/04/30

        ### Rule 3: EVENT CHAIN TRACKING (事件链追踪)
        **For events that might need ordering, assign sequence tags!**

        When you see multiple related events, assign them to a chain:
        ```json
        {
        "temporal": {
            "absolute_date": "2023-05-15",
            "event_chain": "museum_visits_2023",
            "sequence_order": 1
        }
        }
        ```

        ### Rule 4: AGGREGATION TAGGING (聚合标记)
        **For countable/summable items, always add aggregation metadata!**

        Ask yourself: "Could the user ask 'how many' or 'how much total' about this?"
        If yes, add:
        ```json
        {
        "aggregation": {
            "key": "descriptive_key_name",
            "operation": "count|sum",
            "value": 1
        }
        }
        ```

        ### Rule 5: ASSISTANT CONTENT EXTRACTION (Assistant内容抽取)
        **Extract what the ASSISTANT said, not just what the user said!**

        When assistant provides:
        - Restaurant names → ASSISTANT_KNOWLEDGE
        - Phone numbers → ASSISTANT_KNOWLEDGE
        - Step-by-step instructions → ASSISTANT_KNOWLEDGE
        - Technical recommendations → ASSISTANT_KNOWLEDGE
        - Creative writing (stories, songs) → ASSISTANT_KNOWLEDGE

        ### Rule 6: STATE TRACKING (状态追踪)
        **Track temporal states for attributes that change over time.**

        - "I was an atheist" → state: "previous"
        - "I am now more spiritual" → state: "current"
        - "My previous job was..." → state: "previous"
        - "I currently work as..." → state: "current"

        ### Rule 7: IMPLICIT CONSTRAINT DETECTION (隐含约束检测)
        **Detect constraints that would affect future recommendations!**

        Look for:
        - "I don't like..." / "I want to avoid..."
        - "I already have..." / "I'm using..."
        - "I prefer..." / "I'm looking for something that..."
        - Time/budget/dietary/accessibility constraints

        ---

        ## OUTPUT FORMAT (JSON)

        ```json
        {
        "memory_facts": [
            {
            "fact_id": "F1",
            "category": "CATEGORY_NAME",
            "content": "Complete, detailed description preserving all nuances",
            "attributes": {
                "key": "value pairs specific to the category"
            },
            "temporal": {
                "relative_expression": "original expression if any",
                "absolute_date": "YYYY-MM-DD",
                "duration": "time span if applicable",
                "event_chain": "chain_id for ordering",
                "sequence_order": 1
            },
            "aggregation": {
                "key": "aggregation_group_key",
                "operation": "count|sum",
                "value": 1,
                "unit": "optional unit for sums"
            },
            "source": "user_statement|assistant_response",
            "session_id": "source session ID",
            "session_date": "session date",
            "confidence": 0.95,
            "qa_relevance": ["types of questions this fact can answer"]
            }
        ],
        "event_chains": [
            {
            "chain_id": "museum_visits_2023",
            "description": "Sequence of museum visits in 2023",
            "events": ["F1", "F5", "F12"],
            "ordered_by": "date"
            }
        ],
        "aggregation_groups": [
            {
            "key": "model_kits_owned",
            "description": "Model kits user has worked on or bought",
            "facts": ["F3", "F7", "F15", "F22", "F31"],
            "total_count": 5
            }
        ],
        "extraction_summary": {
            "total_facts": 0,
            "by_category": {
            "USER_ATTRIBUTE": 0,
            "EPISODIC_EVENT": 0,
            "INVENTORY_ITEM": 0,
            "PREFERENCE_HABIT": 0,
            "RELATIONSHIP_FACT": 0,
            "QUANTITATIVE_FACT": 0,
            "TEMPORAL_MARKER": 0,
            "ASSISTANT_KNOWLEDGE": 0,
            "AGGREGATABLE_ITEM": 0,
            "IMPLICIT_CONSTRAINT": 0
            },
            "event_chains_count": 0,
            "aggregation_groups_count": 0
        }
        }
        ```

        ---

        ## NOW EXTRACT MEMORY FACTS FROM THE SESSIONS ABOVE

        Remember:
        1. **Preserve ALL details** - "a lighter shade of gray" not "gray"
        2. **Convert relative time** - use session dates to calculate absolute dates
        3. **Extract complete events** - not just nouns
        4. **Track state changes** - mark "previous" vs "current"
        5. **Include units with numbers** - "45 minutes each way" not "45"
        6. **Tag aggregatable items** - add aggregation keys for countable/summable items
        7. **Build event chains** - link related events for ordering questions
        8. **Extract assistant knowledge** - capture what the AI recommended/said
        9. **Detect implicit constraints** - note preferences that affect recommendations
        10. **Calculate temporal distances** - include days_before_reference for "how many days ago" questions

        Extract memory facts now:"""
        
        return Template(template_string)
    
    def _sanitize_content(self, content: str) -> str:
        """Run sanitize content."""
        if not content:
            return ""
        
        content = content.replace('\r\n', '\n').replace('\r', '\n')
        
        content = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', content)
        
        content = re.sub(r'[\u200B-\u200D\uFEFF\u2028\u2029]', '', content)
        
        content = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', content)
        
        content = re.sub(r'[\u0085\u000B\u000C]', ' ', content)
        
        return content
    
    def _build_sessions_text(self,
                            haystack_sessions: List[List[Dict]],
                            haystack_session_ids: List[str],
                            haystack_dates: List[str],
                            start_idx: int,
                            end_idx: int) -> tuple:
        """Build sessions text."""
        sessions_text = ""
        reference_date = ""
        
        actual_end_idx = min(end_idx, len(haystack_sessions))
        
        for idx in range(start_idx, actual_end_idx):
            session = haystack_sessions[idx]
            session_id = haystack_session_ids[idx] if idx < len(haystack_session_ids) else f"session_{idx}"
            date_str = haystack_dates[idx] if idx < len(haystack_dates) else "Unknown"
            
            if not reference_date:
                reference_date = date_str
            
            sessions_text += f"\n{'='*60}\n"
            sessions_text += f"Session: {session_id}\n"
            sessions_text += f"Date: {date_str}\n"
            sessions_text += f"{'='*60}\n\n"
            
            for msg_idx, message in enumerate(session):
                role = message.get("role", "unknown")
                content = message.get("content", "")
                
                content = self._sanitize_content(content)
                
                sessions_text += f"[{role.upper()}]: {content}\n\n"
        
        return sessions_text, reference_date
    
    def regenerate_failed_requests(self,
                                  qa_indices: Optional[List[int]] = None,
                                  model: str = "qwen-plus-latest",
                                  enable_thinking: bool = False,
                                  thinking_budget: int = 1024) -> str:
        """Run regenerate failed requests."""
        
        qa_samples = self.load_dataset()
        
        
        all_errors = self.load_all_error_requests()
        
        if not all_errors:
            logger.warning(" 没有找到任何失败的请求")
            return None
        
        if qa_indices is None:
            qa_indices_to_process = sorted(all_errors.keys())
        else:
            qa_indices_to_process = [idx for idx in qa_indices if idx in all_errors]
        
        if not qa_indices_to_process:
            logger.warning(" 没有需要处理的 QA")
            return None
        
        logger.info(f"\n{'='*80}")
        logger.info(f" 重新生成失败的请求")
        logger.info(f"{'='*80}")
        logger.info(f"处理的QA数量: {len(qa_indices_to_process)}")
        logger.info(f"使用模型: {model}")
        logger.info(f"Thinking模式: {'启用 (budget={})'.format(thinking_budget) if enable_thinking else '禁用'}")
        logger.info(f"每组session数: {self.sessions_per_group}")
        logger.info(f"{'='*80}\n")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        qa_range = f"{min(qa_indices_to_process)}-{max(qa_indices_to_process)}"
        output_file = self.output_dir / f"retry_qa{qa_range}_sessions{self.sessions_per_group}_{timestamp}.jsonl"
        
        total_sessions = 0
        total_messages = 0
        total_requests = 0
        
        prompt_template = self._build_episodic_memory_extraction_prompt()
        
        # System prompt
        system_content = """You are a Memory Archivist AI specializing in extracting structured episodic memory facts from conversations. 

        Your task is to extract Atomic Memory Facts that can answer various types of questions:
        - Simple facts: "What is my cat's name?"
        - Aggregation: "How many model kits do I have?"
        - Temporal ordering: "Which event happened first?"
        - Assistant recall: "What restaurant did you recommend?"
        - Personalized recommendations: "Can you suggest evening activities?"

        Output valid JSON only. Preserve all details and nuances."""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for qa_idx in qa_indices_to_process:
                if qa_idx >= len(qa_samples):
                    logger.warning(f" QA {qa_idx} 超出数据集范围，跳过")
                    continue
                
                qa_data = qa_samples[qa_idx]
                
                question_id = qa_data.get("question_id", f"qa_{qa_idx}")
                haystack_sessions = qa_data.get("haystack_sessions", [])
                haystack_session_ids = qa_data.get("haystack_session_ids", [])
                haystack_dates = qa_data.get("haystack_dates", [])
                
                if not haystack_sessions:
                    logger.warning(f" QA {qa_idx} 没有 haystack_sessions，跳过")
                    continue
                
                failed_requests = all_errors[qa_idx]
                
                session_ranges_to_process = set()
                for error_info in failed_requests:
                    start_s = error_info["start_session"]
                    end_s = error_info["end_session"]
                    for s in range(start_s, end_s + 1):
                        session_ranges_to_process.add(s)
                
                sorted_sessions = sorted(session_ranges_to_process)
                
                logger.info(f" 处理 QA {qa_idx}: 需要重新处理 {len(sorted_sessions)} 个 sessions")
                
                total_sessions += len(sorted_sessions)
                total_messages += sum(len(haystack_sessions[s]) for s in sorted_sessions if s < len(haystack_sessions))
                
                group_idx = 0
                i = 0
                while i < len(sorted_sessions):
                    group_sessions = sorted_sessions[i:i+self.sessions_per_group]
                    start_session_idx = group_sessions[0]
                    end_session_idx = group_sessions[-1]
                    
                    try:
                        sessions_text, reference_date = self._build_sessions_text(
                            haystack_sessions,
                            haystack_session_ids,
                            haystack_dates,
                            start_session_idx,
                            end_session_idx + 1
                        )
                        
                        if not sessions_text.strip():
                            logger.warning(f" QA {qa_idx} group {group_idx} 会话文本为空，跳过")
                            i += self.sessions_per_group
                            group_idx += 1
                            continue
                        
                        user_prompt = prompt_template.substitute(
                            sessions_text=sessions_text,
                            reference_date=reference_date if reference_date else "Unknown"
                        )
                        
                        custom_id = f"retry_qa_{qa_idx}_group_{group_idx}_sessions_{start_session_idx}_{end_session_idx}"
                        
                        messages = [
                            {
                                "role": "system",
                                "content": system_content
                            },
                            {
                                "role": "user",
                                "content": user_prompt
                            }
                        ]
                        
                        body = {
                            "model": model,
                            "messages": messages,
                            "enable_thinking": enable_thinking
                        }
                        if enable_thinking:
                            body["thinking_budget"] = thinking_budget

                        request = {
                            "custom_id": custom_id,
                            "method": "POST",
                            "url": "/v1/chat/completions",
                            "body": body
                        }
                        
                        json_line = json.dumps(request, ensure_ascii=True)
                        f.write(json_line + '\n')
                        total_requests += 1
                        
                    except Exception as e:
                        logger.error(f" 处理 QA {qa_idx} group {group_idx} 时出错: {e}")
                        
                    i += self.sessions_per_group
                    group_idx += 1
        
        logger.info(f"\n{'='*80}")
        logger.info(f" 重新生成完成!")
        logger.info(f"{'='*80}")
        logger.info(f"输出文件: {output_file}")
        logger.info(f"文件大小: {output_file.stat().st_size / 1024:.2f} KB")
        logger.info(f"\n 统计信息:")
        logger.info(f"  处理的QA数量: {len(qa_indices_to_process)}")
        logger.info(f"  生成的请求数: {total_requests}")
        logger.info(f"  总Session数: {total_sessions}")
        logger.info(f"  总消息数: {total_messages}")
        
        if len(qa_indices_to_process) > 0:
            logger.info(f"\n 平均统计:")
            logger.info(f"  平均请求/QA: {total_requests / len(qa_indices_to_process):.1f}")
        logger.info(f"{'='*80}\n")
        
        
        metadata = {
            "created_at": datetime.now().isoformat(),
            "dataset_path": str(self.dataset_path),
            "error_dir": str(self.error_dir),
            "qa_indices": qa_indices_to_process,
            "sessions_per_group": self.sessions_per_group,
            "total_qa_count": len(qa_indices_to_process),
            "total_requests": total_requests,
            "total_sessions": total_sessions,
            "total_messages": total_messages,
            "model": model,
            "enable_thinking": enable_thinking,
            "thinking_budget": thinking_budget if enable_thinking else None,
            "memory_categories": EpisodicMemoryCategory.get_all_categories(),
            "prompt_version": "V2",
            "note": "Retry requests for failed QA samples with smaller session groups"
        }
        
        metadata_file = output_file.with_suffix('.meta.json')
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f" 元数据文件: {metadata_file}")
        
        return str(output_file)


def main():
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(
        description="LongMemEval Episodic Memory V2 - 失败请求重新生成器 (Step 2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        示例:
        # 重新生成所有失败的请求，每组1个session
        python step2_regenerate_error_requests.py
        
        # 重新生成所有失败的请求，每组2个session
        python step2_regenerate_error_requests.py --sessions-per-group 2
        
        # 只重新生成特定 QA 的失败请求
        python step2_regenerate_error_requests.py --qa-indices 19 27 44
        
        # 使用不同模型
        python step2_regenerate_error_requests.py --model qwen-max

        特点:
        - 自动扫描 batch_results/error/ 目录下的所有 *_error.jsonl 文件
        - 默认使用 sessions_per_group=1 (更保守，避免内容审查问题)
        - 输出到 batch_requests/error/ 目录
        - custom_id 前缀为 "retry_" 便于区分
        """
    )
    
    parser.add_argument("--dataset-path",
                       default=str(paths.LONGMEMEVAL_S_CLEANED_FILE),
                       help="数据集文件路径")
    parser.add_argument("--error-dir",
                       default=str(paths.LONGMEMEVAL_EPISODIC_NEW_BATCH_RESULTS_ERROR_DIR),
                       help="error.jsonl 文件所在目录")
    parser.add_argument("--output-dir",
                       default=str(paths.LONGMEMEVAL_EPISODIC_NEW_BATCH_REQUESTS_ERROR_DIR),
                       help="输出目录路径")
    
    parser.add_argument("--qa-indices", type=int, nargs='+',
                       help="要处理的 QA 索引列表 (例如: 19 27 44), 不指定则处理所有失败的 QA")
    
    parser.add_argument("--sessions-per-group", type=int, default=1,
                       help="每组处理的session数量 (默认1，更保守)")
    
    parser.add_argument("--enable-thinking", action="store_true",
                       help="启用模型的推理/思考模式 (默认禁用，以防 Qwen 3.5 默认开启导致 token 浪费)")
    parser.add_argument("--thinking-budget", type=int, default=2048,
                       help="思考过程的 Token 预算（仅在启用 enable-thinking 时生效）")
    
    parser.add_argument("--model",
                       default="qwen-plus-latest",
                    #    default="qwen3.5-plus",
                       choices=[
                           "qwen-plus-latest",
                           "qwen-max-latest",
                           "qwen-turbo-latest",
                           "qwen-long",
                           "qwen3.5-plus"
                       ],
                       help="使用的模型名称 (默认: qwen3.5-plus)")
    # parser.add_argument("--model",
    #                    default="qwen-plus-latest",
    
    parser.add_argument("--debug", action="store_true",
                       help="启用调试模式")
    
    args = parser.parse_args()
    
    # Avoid mutating LogRecord fields before other handlers process the record.
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        regenerator = ErrorRequestRegenerator(
            dataset_path=args.dataset_path,
            error_dir=args.error_dir,
            output_dir=args.output_dir,
            sessions_per_group=args.sessions_per_group
        )
        
        output_file = regenerator.regenerate_failed_requests(
            qa_indices=args.qa_indices,
            model=args.model,
            enable_thinking=args.enable_thinking,
            thinking_budget=args.thinking_budget
        )
        
        if output_file:
            print(f"\n{'='*80}")
            print(f" 成功重新生成失败请求!")
            print(f"{'='*80}")
            print(f" 文件路径: {output_file}")
            print(f"\n 特点:")
            print(f"  - Sessions已按{args.sessions_per_group}个一组分批处理（更保守）")
            print(f"  - custom_id前缀: retry_qa_X_group_Y_sessions_A_B")
            print(f"  - 使用模型: {args.model}")
            print(f"\n下一步:")
            print(f"  1. 上传文件到阿里云百炼进行批量推理")
            print(f"  2. 检查是否还有失败的请求")
            print(f"  3. 如有需要，进一步降低 sessions-per-group")
            print(f"{'='*80}\n")
        
        return 0
        
    except FileNotFoundError as e:
        logger.error(f" 文件未找到: {e}")
        return 1
    except ValueError as e:
        logger.error(f" 参数错误: {e}")
        return 1
    except Exception as e:
        logger.error(f" 程序异常: {e}")
        if args.debug:
            import traceback
            logger.debug(traceback.format_exc())
            
    return 1

if __name__ == "__main__":
    sys.exit(main())
