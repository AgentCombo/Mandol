#!/usr/bin/env python3
"""Utilities for design 2 episodic memory extraction."""

import json
import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from string import Template
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


class LongMemEvalEpisodicMemoryExtractorV2:
    
    def __init__(self,
                 dataset_path: str = str(paths.LONGMEMEVAL_S_CLEANED_FILE),
                 output_dir: str = str(paths.LONGMEMEVAL_EPISODIC_LEGACY_BATCH_REQUESTS_V2_DIR),
                 sessions_per_group: int = 10):
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.sessions_per_group = sessions_per_group
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f" 数据集路径: {self.dataset_path}")
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
    
    def _build_episodic_memory_extraction_prompt_v2(self) -> Template:
        """Build episodic memory extraction prompt v2."""
        
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
        - "User redeemed a $5 coupon on coffee creamer at Target" → EPISODIC_EVENT
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
        - Prices: "$800", "$5 coupon"
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

        ```json
        {
        "category": "ASSISTANT_KNOWLEDGE",
        "content": "Assistant provided phone number +49 (0) 62 32 / 14 23 - 0 for Speyer tourism board",
        "attributes": {
            "type": "contact_info",
            "entity": "Speyer tourism board",
            "phone": "+49 (0) 62 32 / 14 23 - 0"
        },
        "source": "assistant_response"
        }
        ```

        ```json
        {
        "category": "ASSISTANT_KNOWLEDGE",
        "content": "Assistant suggested using Ruby, Python, or PHP for back-end programming",
        "attributes": {
            "type": "technical_recommendation",
            "topic": "back-end programming languages",
            "options": ["Ruby", "Python", "PHP"]
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

        ```json
        {
        "category": "AGGREGATABLE_ITEM",
        "content": "User spent $50 on bike tire replacement",
        "attributes": {
            "item": "bike tire replacement",
            "amount": "$50"
        },
        "aggregation": {
            "key": "bike_expenses",
            "operation": "sum",
            "value": 50,
            "unit": "USD"
        }
        }
        ```

        ```json
        {
        "category": "AGGREGATABLE_ITEM",
        "content": "User attended Rachel and Mike's wedding",
        "attributes": {
            "event": "wedding",
            "couple": "Rachel and Mike"
        },
        "aggregation": {
            "key": "weddings_attended_2023",
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

        ```json
        {
        "category": "IMPLICIT_CONSTRAINT",
        "content": "User has experience with Adobe Premiere Pro for video editing",
        "attributes": {
            "constraint_type": "context",
            "software": "Adobe Premiere Pro",
            "skill_area": "video editing"
        },
        "affects_recommendations": ["video editing resources should focus on Premiere Pro"]
        }
        ```

        ```json
        {
        "category": "IMPLICIT_CONSTRAINT",
        "content": "User prefers stand-up comedy specials with storytelling style on Netflix",
        "attributes": {
            "constraint_type": "preference",
            "genre": "stand-up comedy",
            "style": "storytelling",
            "platform": "Netflix"
        },
        "affects_recommendations": ["show/movie recommendations"]
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

        This enables answering: "What is the order of museums I visited?"

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
        - "Before I bought X, I had Y" → track both states!

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

        ## FEW-SHOT EXAMPLES

        ### Example 1: Aggregatable Items (for "How many X?" questions)
        **Input:** User mentions buying 5 different model kits across conversations:
        - "I'm working on a Revell F-15 Eagle model"
        - "Just finished my Tamiya 1/48 scale Spitfire Mk.V"
        - "Started the 1/16 scale German Tiger I tank"
        - "Bought a 1/72 scale B-29 bomber kit"
        - "Working on my 1/24 scale '69 Camaro"

        **Output:**
        ```json
        {
        "memory_facts": [
            {
            "fact_id": "F1",
            "category": "AGGREGATABLE_ITEM",
            "content": "User is working on Revell F-15 Eagle model kit",
            "attributes": {"item": "Revell F-15 Eagle", "type": "aircraft model", "status": "in progress"},
            "aggregation": {"key": "model_kits_owned", "operation": "count", "value": 1}
            },
            {
            "fact_id": "F2", 
            "category": "AGGREGATABLE_ITEM",
            "content": "User finished Tamiya 1/48 scale Spitfire Mk.V model kit",
            "attributes": {"item": "Tamiya 1/48 scale Spitfire Mk.V", "type": "aircraft model", "scale": "1/48", "status": "completed"},
            "aggregation": {"key": "model_kits_owned", "operation": "count", "value": 1}
            }
        ],
        "aggregation_groups": [
            {
            "key": "model_kits_owned",
            "description": "Model kits user has worked on",
            "facts": ["F1", "F2", "F3", "F4", "F5"],
            "total_count": 5
            }
        ]
        }
        ```

        ### Example 2: Event Chain (for "Which happened first?" questions)
        **Input:** User mentions visiting multiple museums:
        - "Went to the Science Museum last month"
        - "Visited Museum of Contemporary Art two weeks ago"
        - "Just came back from the Natural History Museum yesterday"

        **Session Date:** 2023/05/30

        **Output:**
        ```json
        {
        "memory_facts": [
            {
            "fact_id": "F1",
            "category": "EPISODIC_EVENT",
            "content": "User visited the Science Museum",
            "attributes": {"action": "visited", "location": "Science Museum"},
            "temporal": {
                "relative_expression": "last month",
                "absolute_date": "2023-04-30",
                "event_chain": "museum_visits",
                "sequence_order": 1
            }
            },
            {
            "fact_id": "F2",
            "category": "EPISODIC_EVENT", 
            "content": "User visited Museum of Contemporary Art",
            "attributes": {"action": "visited", "location": "Museum of Contemporary Art"},
            "temporal": {
                "relative_expression": "two weeks ago",
                "absolute_date": "2023-05-16",
                "event_chain": "museum_visits",
                "sequence_order": 2
            }
            },
            {
            "fact_id": "F3",
            "category": "EPISODIC_EVENT",
            "content": "User visited the Natural History Museum",
            "attributes": {"action": "visited", "location": "Natural History Museum"},
            "temporal": {
                "relative_expression": "yesterday",
                "absolute_date": "2023-05-29",
                "event_chain": "museum_visits",
                "sequence_order": 3
            }
            }
        ],
        "event_chains": [
            {
            "chain_id": "museum_visits",
            "description": "Sequence of museum visits",
            "events": ["F1", "F2", "F3"],
            "ordered_by": "date",
            "order": ["Science Museum", "Museum of Contemporary Art", "Natural History Museum"]
            }
        ]
        }
        ```

        ### Example 3: Assistant Knowledge (for "What did you recommend?" questions)
        **Input:** 
        User: "Can you recommend a romantic restaurant in Rome?"
        Assistant: "I'd recommend Roscioli - it's a famous deli near the Vatican that serves excellent cured meats and Italian cuisine."

        **Output:**
        ```json
        {
        "fact_id": "F1",
        "category": "ASSISTANT_KNOWLEDGE",
        "content": "Assistant recommended Roscioli restaurant in Rome, describing it as a famous deli near the Vatican serving excellent cured meats and Italian cuisine",
        "attributes": {
            "type": "restaurant_recommendation",
            "name": "Roscioli",
            "location": "Rome, near the Vatican",
            "cuisine": "Italian",
            "specialty": "cured meats",
            "description": "famous deli"
        },
        "source": "assistant_response",
        "qa_relevance": [
            "What restaurant did you recommend in Rome?",
            "What's the name of the deli near the Vatican?",
            "Where can I get good Italian food in Rome?"
        ]
        }
        ```

        ### Example 4: Implicit Constraint (for personalized recommendations)
        **Input:**
        User: "I've been having trouble sleeping. I think using my phone and watching TV in the evening is affecting my sleep quality."

        **Output:**
        ```json
        {
        "fact_id": "F1",
        "category": "IMPLICIT_CONSTRAINT",
        "content": "User wants to avoid phone use and TV watching in the evening because they affect sleep quality",
        "attributes": {
            "constraint_type": "avoidance",
            "avoid": ["phone use in evening", "TV watching in evening"],
            "reason": "affects sleep quality",
            "health_concern": "sleep problems"
        },
        "affects_recommendations": [
            "evening activity suggestions should NOT include phone or TV",
            "relaxation suggestions should be screen-free"
        ],
        "qa_relevance": [
            "Can you suggest activities for the evening?",
            "What should I do to relax before bed?"
        ]
        }
        ```

        ### Example 5: State Change with Temporal Tracking
        **Input:**
        "When I started as Senior Software Engineer, I led 4 engineers. Now I lead 5 engineers."

        **Output:**
        ```json
        [
        {
            "fact_id": "F1",
            "category": "USER_ATTRIBUTE",
            "content": "User initially led 4 engineers when starting as Senior Software Engineer",
            "attributes": {
            "role": "Senior Software Engineer",
            "team_size": 4,
            "state": "initial"
            },
            "qa_relevance": ["How many engineers did I lead when I started?"]
        },
        {
            "fact_id": "F2",
            "category": "USER_ATTRIBUTE",
            "content": "User currently leads 5 engineers as Senior Software Engineer",
            "attributes": {
            "role": "Senior Software Engineer", 
            "team_size": 5,
            "state": "current"
            },
            "qa_relevance": ["How many engineers do I lead now?"]
        }
        ]
        ```

        ### Example 6: Temporal Calculation (for "How many days ago?" questions)
        **Input:** "I bought a smoker last Wednesday"
        **Session Date:** 2023/05/30 (Tuesday)

        **Output:**
        ```json
        {
        "fact_id": "F1",
        "category": "EPISODIC_EVENT",
        "content": "User bought a smoker",
        "attributes": {
            "action": "bought",
            "item": "smoker",
            "type": "kitchen appliance"
        },
        "temporal": {
            "relative_expression": "last Wednesday",
            "absolute_date": "2023-05-24",
            "days_before_reference": 6
        },
        "qa_relevance": [
            "How many days ago did I buy a smoker?",
            "What kitchen appliance did I buy recently?",
            "What did I buy last week?"
        ]
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
            return content
        
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
        
        for idx in range(start_idx, min(end_idx, len(haystack_sessions))):
            session = haystack_sessions[idx]
            session_id = haystack_session_ids[idx] if idx < len(haystack_session_ids) else f"session_{idx}"
            session_date = haystack_dates[idx] if idx < len(haystack_dates) else "Unknown"
            
            if idx == end_idx - 1:
                reference_date = session_date
            
            sessions_text += f"\n### Session: {session_id}\n"
            sessions_text += f"**Date:** {session_date}\n\n"
            
            for msg in session:
                role = msg.get('role', 'unknown')
                content = self._sanitize_content(msg.get('content', ''))
                
                if role == 'user':
                    sessions_text += f"**User:** {content}\n\n"
                elif role == 'assistant':
                    sessions_text += f"**Assistant:** {content}\n\n"
                else:
                    sessions_text += f"**{role}:** {content}\n\n"
            
            sessions_text += "---\n"
        
        return sessions_text, reference_date
    
    def _split_sessions_into_groups(self, total_sessions: int) -> List[tuple]:
        """Run split sessions into groups."""
        groups = []
        for i in range(0, total_sessions, self.sessions_per_group):
            end = min(i + self.sessions_per_group, total_sessions)
            groups.append((i, end))
        return groups
    
    def generate_batch_requests(self,
                                start_index: int = 0,
                                end_index: Optional[int] = None,
                                model: str = "deepseek-ai/DeepSeek-V3",
                                max_tokens: Optional[int] = 16384,
                                temperature: float = 0.1) -> str:
        """Generate batch requests."""
        qa_samples = self.load_dataset()
        
        if end_index is None:
            end_index = len(qa_samples) - 1
        
        if start_index < 0 or start_index >= len(qa_samples):
            raise ValueError(f"start_index {start_index} 超出范围 [0, {len(qa_samples)-1}]")
        if end_index < start_index or end_index >= len(qa_samples):
            end_index = len(qa_samples) - 1
        
        selected_samples = qa_samples[start_index:end_index+1]
        
        logger.info(f"\n{'='*80}")
        logger.info(f" 生成 Episodic Memory V2 批量推理请求")
        logger.info(f"{'='*80}")
        logger.info(f"处理范围: QA {start_index} - {end_index}")
        logger.info(f"样本数量: {len(selected_samples)}")
        logger.info(f"使用模型: {model}")
        logger.info(f"每组session数: {self.sessions_per_group}")
        logger.info(f"{'='*80}\n")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"episodic_memory_v2_qa{start_index}_to_qa{end_index}_{timestamp}.jsonl"
        
        total_requests = 0
        prompt_template = self._build_episodic_memory_extraction_prompt_v2()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample_idx, sample in enumerate(selected_samples):
                global_idx = start_index + sample_idx
                question_id = sample.get('question_id', f'q_{global_idx}')
                
                haystack_sessions = sample.get('haystack_sessions', [])
                haystack_session_ids = sample.get('haystack_session_ids', [])
                haystack_dates = sample.get('haystack_dates', [])
                
                if not haystack_sessions:
                    logger.warning(f" QA {global_idx} 没有 haystack_sessions，跳过")
                    continue
                
                groups = self._split_sessions_into_groups(len(haystack_sessions))
                
                for group_idx, (start_sess, end_sess) in enumerate(groups):
                    sessions_text, reference_date = self._build_sessions_text(
                        haystack_sessions, 
                        haystack_session_ids,
                        haystack_dates,
                        start_sess,
                        end_sess
                    )
                    
                    if not sessions_text.strip():
                        continue
                    
                    prompt = prompt_template.substitute(
                        sessions_text=sessions_text,
                        reference_date=reference_date or "Unknown"
                    )
                    
                    request_id = f"qa{global_idx}_group{group_idx}_{question_id}"
                    
                    request = {
                        "custom_id": request_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": model,
                            "messages": [
                                {
                                    "role": "system",
                                    "content": "You are a Memory Archivist AI that extracts structured memory facts from conversations. Output valid JSON only."
                                },
                                {
                                    "role": "user",
                                    "content": prompt
                                }
                            ],
                            "max_tokens": max_tokens,
                            "temperature": temperature,
                            "response_format": {"type": "json_object"}
                        }
                    }
                    
                    f.write(json.dumps(request, ensure_ascii=False) + '\n')
                    total_requests += 1
                
                if (sample_idx + 1) % 50 == 0:
                    logger.info(f" 已处理 {sample_idx + 1}/{len(selected_samples)} 个 QA 样本")
        
        logger.info(f"\n{'='*80}")
        logger.info(f" 批量请求生成完成")
        logger.info(f"{'='*80}")
        logger.info(f"输出文件: {output_file}")
        logger.info(f"总请求数: {total_requests}")
        logger.info(f"{'='*80}\n")
        
        return str(output_file)


def main():
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(
        description="LongMemEval Episodic Memory Extractor V2 - 生成情景记忆抽取批量请求 (增强版)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 处理前10个QA
  python step_option_episodic_memory_extraction.py --start 0 --end 9
  
  # 处理所有QA
  python step_option_episodic_memory_extraction.py
  
  # 使用自定义模型
  python step_option_episodic_memory_extraction.py --model gpt-4o

V2 增强特性:
  - 聚合关联标记: 支持 "How many X?" 类问题
  - 时序事件链: 支持 "Which happened first?" 类问题  
  - Assistant知识抽取: 支持 "What did you recommend?" 类问题
  - 隐含约束检测: 支持个性化推荐问题
        """
    )
    
    parser.add_argument("--dataset", type=str,
                       default=str(paths.LONGMEMEVAL_S_CLEANED_FILE),
                       help="数据集文件路径")
    parser.add_argument("--output-dir", type=str,
                       default=str(paths.LONGMEMEVAL_EPISODIC_LEGACY_BATCH_REQUESTS_V2_DIR),
                       help="输出目录")
    parser.add_argument("--start", type=int, default=0,
                       help="起始QA索引")
    parser.add_argument("--end", type=int, default=None,
                       help="结束QA索引")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-V3",
                       help="模型名称")
    parser.add_argument("--max-tokens", type=int, default=16384,
                       help="最大输出token数")
    parser.add_argument("--sessions-per-group", type=int, default=10,
                       help="每组处理的session数量")
    
    args = parser.parse_args()
    
    try:
        extractor = LongMemEvalEpisodicMemoryExtractorV2(
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            sessions_per_group=args.sessions_per_group
        )
        
        output_file = extractor.generate_batch_requests(
            start_index=args.start,
            end_index=args.end,
            model=args.model,
            max_tokens=args.max_tokens
        )
        
        logger.info(f" 完成! 输出文件: {output_file}")
        return 0
        
    except Exception as e:
        logger.error(f" 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())