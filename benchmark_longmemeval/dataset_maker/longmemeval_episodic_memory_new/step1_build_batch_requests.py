#!/usr/bin/env python3
"""{"custom_id":"request-1","method":"POST","url":"/v1/chat/completions","body":{"model":"qwen-plus-latest","messages":[...]}}."""

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
    
    @classmethod
    def get_category_description(cls) -> str:
        """Return category description."""
        return """
        Memory Fact Categories (10 Types):
        1. USER_ATTRIBUTE: Static user profile (education, job, identity)
        2. EPISODIC_EVENT: Specific events with time/place/action
        3. INVENTORY_ITEM: Things user owns/bought (with full specs)
        4. PREFERENCE_HABIT: Likes, dislikes, routines
        5. RELATIONSHIP_FACT: Social connections
        6. QUANTITATIVE_FACT: Numbers with units and context
        7. TEMPORAL_MARKER: Time information for event ordering
        8. ASSISTANT_KNOWLEDGE: What the AI assistant said/recommended
        9. AGGREGATABLE_ITEM: Items that can be counted/summed
        10. IMPLICIT_CONSTRAINT: Hidden preferences affecting recommendations
        """


class LongMemEvalEpisodicMemoryBatchExtractor:
    
    def __init__(self,
                 dataset_path: str = str(paths.LONGMEMEVAL_S_CLEANED_FILE),
                 output_dir: str = str(paths.LONGMEMEVAL_EPISODIC_NEW_BATCH_REQUESTS_DIR),
                 sessions_per_group: int = 1):
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
    
    # def _build_episodic_memory_extraction_prompt(self) -> Template:
    #     """
        
        
    #     Returns:
    #     """
        
    #     template_string = """You are a **Memory Archivist** specializing in personal episodic memory preservation for AI assistants. Your task is to extract **Atomic Memory Facts** from the user's conversation history.

    #     ## CRITICAL UNDERSTANDING

    #     This is NOT a Named Entity Recognition task. This is an **Episodic Memory Extraction** task designed to answer questions like:
    #     - "What is X?" (simple fact)
    #     - "How many X did I do?" (aggregation)
    #     - "Which happened first, X or Y?" (temporal ordering)
    #     - "What did you recommend for X?" (assistant knowledge recall)
    #     - "Can you suggest X for me?" (preference-based recommendation)

    #      WRONG approach (NER-style):
    #     - "Target" → Organization
    #     - "45 minutes" → Duration
    #     - "gray" → Color

    #      CORRECT approach (Episodic Memory):
    #     - "User redeemed a $$5 coupon on coffee creamer at Target" → EPISODIC_EVENT
    #     - "User's daily commute is 45 minutes each way" → QUANTITATIVE_FACT + PREFERENCE_HABIT
    #     - "User repainted bedroom walls to a lighter shade of gray" → EPISODIC_EVENT (preserve exact color!)

    #     ## CONVERSATION SESSIONS

    #     $sessions_text

    #     ## SESSION REFERENCE DATE: $reference_date

    #     Use this date to convert ALL relative time expressions to absolute dates.

    #     ---

    #     ## MEMORY FACT CATEGORIES (10 Types)

    #     Static profile information that defines who the user is.

    #     **Extract:**
    #     - Educational background: degrees, schools, majors, graduation years
    #     - Professional history: CURRENT job, PREVIOUS jobs (mark temporal state!)
    #     - Personal identity: ethnicity, nationality, age, birthdate
    #     - Physical attributes: height, weight
    #     - Beliefs/Stances: religious views, political views (with temporal state if changed!)

    #     **Examples:**
    #     ```json
    #     {
    #     "category": "USER_ATTRIBUTE",
    #     "content": "User graduated with a Business Administration degree",
    #     "attributes": {"degree": "Business Administration", "state": "completed"}
    #     }
    #     ```

    #     Specific actions the user performed at a specific time/place.

    #     **Extract the COMPLETE event structure:**
    #     - WHAT action was performed
    #     - WHERE it happened (preserve exact location names!)
    #     - WHEN it happened (convert relative to absolute!)
    #     - WHO was involved
    #     - HOW it was done
    #     - WHY (if mentioned)

    #     **Examples:**
    #     ```json
    #     {
    #     "category": "EPISODIC_EVENT",
    #     "content": "User attended The Glass Menagerie at the local community theater",
    #     "attributes": {
    #         "action": "attended play",
    #         "object": "The Glass Menagerie",
    #         "location": "local community theater"
    #     },
    #     "temporal": {"absolute_date": "2023-05-23"}
    #     }
    #     ```

    #     Things the user owns, bought, or possesses.

    #     **CRITICAL: Preserve ALL details:**
    #     - Brand/Model: Nike, iPhone 13 Pro, Fitbit Inspire HR
    #     - Specifications: 500 Mbps, 16GB RAM, 55-inch
    #     - Color/Style: "a lighter shade of gray", "yellow dress"
    #     - Quantity: "three bikes", "20 playlists"
    #     - Acquisition details: where bought, when bought, price paid
    #     - Pet details: name, breed, age

    #     **Examples:**
    #     ```json
    #     {
    #     "category": "INVENTORY_ITEM",
    #     "content": "User owns a cat named Luna",
    #     "attributes": {"type": "pet", "species": "cat", "name": "Luna"}
    #     }
    #     ```

    #     User's likes, dislikes, routines, and regular behaviors.

    #     **Extract:**
    #     - Explicit preferences: "favorite", "prefer", "like", "hate", "love"
    #     - Routines: "usually", "always", "every [day/week]", "typically"
    #     - Time-based habits: wake up time, work schedule, exercise routine
    #     - Consumption habits: brands used, services subscribed

    #     **Examples:**
    #     ```json
    #     {
    #     "category": "PREFERENCE_HABIT",
    #     "content": "User stops checking work emails at 7 pm",
    #     "attributes": {"habit": "no work emails after 7 pm", "frequency": "daily"}
    #     }
    #     ```

    #     Social connections and their attributes.

    #     **Extract:**
    #     - Relationship type: sister, friend, colleague, cousin, therapist
    #     - Person's name (if mentioned)
    #     - Person's attributes: where they live, what they do, their characteristics
    #     - Shared experiences with the user

    #     **Examples:**
    #     ```json
    #     {
    #     "category": "RELATIONSHIP_FACT",
    #     "content": "User's sister Emily lives in Denver",
    #     "attributes": {"relation": "sister", "name": "Emily", "location": "Denver"}
    #     }
    #     ```

    #     Numerical information with full context.

    #     **CRITICAL: ALWAYS include units and context!**

    #     **Extract:**
    #     - Prices: "$$800", "$$5 coupon"
    #     - Durations: "45 minutes each way", "4 hours", "two weeks"
    #     - Quantities: "20 playlists", "7 shirts", "three bikes"
    #     - Measurements: "500 Mbps", "16GB", "55-inch"
    #     - Percentages/Ratios: "10% discount", "3:1 ratio"
    #     - Scores/Records: "132 points", "25:50 time"

    #     **Examples:**
    #     ```json
    #     {
    #     "category": "QUANTITATIVE_FACT",
    #     "content": "User's daily commute is 45 minutes each way",
    #     "attributes": {
    #         "quantity": "45 minutes each way",
    #         "type": "commute duration",
    #         "note": "90 minutes total round trip"
    #     }
    #     }
    #     ```

    #     Temporal information for reasoning about event sequences.

    #     **CRITICAL: Convert relative to absolute dates AND track event ordering!**

    #     **Extract:**
    #     - Relative references: "last Tuesday", "two weeks ago", "recently"
    #     - Absolute dates: "February 14th", "2023/05/30"
    #     - Duration spans: "for two weeks", "over a year"
    #     - Sequence markers: "before", "after", "first", "then"
    #     - Event ordering: which events happened in what sequence

    #     **Examples:**
    #     ```json
    #     {
    #     "category": "TEMPORAL_MARKER",
    #     "content": "User was in Japan for two weeks",
    #     "attributes": {"duration": "two weeks", "location": "Japan"},
    #     "temporal": {
    #         "relative_expression": "two weeks",
    #         "event_sequence_tag": "japan_trip"
    #     }
    #     }
    #     ```

    #     Information, recommendations, or advice that the ASSISTANT provided to the user.

    #     **CRITICAL: This captures what the AI assistant said, not what the user said!**

    #     **Extract:**
    #     - Recommendations: restaurants, hotels, products, services
    #     - Factual information: historical facts, technical details, procedures
    #     - Lists provided: steps, items, options
    #     - Specific details: phone numbers, addresses, names mentioned by assistant
    #     - Creative content: stories, songs, scripts written by assistant

    #     **Examples:**
    #     ```json
    #     {
    #     "category": "ASSISTANT_KNOWLEDGE",
    #     "content": "Assistant recommended Roscioli restaurant in Rome for Italian food",
    #     "attributes": {
    #         "type": "recommendation",
    #         "topic": "restaurant",
    #         "name": "Roscioli",
    #         "location": "Rome",
    #         "cuisine": "Italian"
    #     },
    #     "source": "assistant_response"
    #     }
    #     ```

    #     Items that belong to a countable/summable group for answering aggregation questions.

    #     **CRITICAL: Tag items with aggregation keys so they can be counted/summed later!**

    #     **Use for questions like:**
    #     - "How many model kits have I worked on?" → count items with aggregation_key="model_kits"
    #     - "How much total money did I spend on X?" → sum items with aggregation_key="X_expenses"
    #     - "How many weddings did I attend?" → count items with aggregation_key="weddings_attended"

    #     **Examples:**
    #     ```json
    #     {
    #     "category": "AGGREGATABLE_ITEM",
    #     "content": "User bought Tamiya 1/48 scale Spitfire Mk.V model kit",
    #     "attributes": {
    #         "item": "Tamiya 1/48 scale Spitfire Mk.V",
    #         "type": "model kit"
    #     },
    #     "aggregation": {
    #         "key": "model_kits_owned",
    #         "operation": "count",
    #         "value": 1
    #     }
    #     }
    #     ```

    #     Implicit preferences, limitations, or constraints that affect recommendations.

    #     **CRITICAL: Capture information that would affect personalized suggestions!**

    #     **Extract:**
    #     - Negative preferences: things user wants to avoid
    #     - Limitations: time constraints, budget limits, dietary restrictions
    #     - Context for recommendations: existing equipment, skill level, past experiences
    #     - Soft constraints: "prefer X over Y", "looking for something like Z"

    #     **Examples:**
    #     ```json
    #     {
    #     "category": "IMPLICIT_CONSTRAINT",
    #     "content": "User wants to avoid using phone or watching TV in the evening as they affect sleep",
    #     "attributes": {
    #         "constraint_type": "avoidance",
    #         "avoid": ["phone use", "TV watching"],
    #         "reason": "affects sleep quality",
    #         "time_context": "evening"
    #     },
    #     "affects_recommendations": ["evening activities", "relaxation suggestions"]
    #     }
    #     ```

    #     ---

    #     ## SPECIAL EXTRACTION RULES

    #     **NEVER simplify or generalize details!**

    #      "gray" 
    #      "a lighter shade of gray"

    #      "45 minutes"
    #      "45 minutes each way"

    #      "my job"
    #      "Marketing specialist at a small startup" (and mark if PREVIOUS or CURRENT)

    #     **Convert ALL relative time to absolute dates using the session reference date.**

    #     If reference date is 2023/05/30 (Tuesday):
    #     - "last Tuesday" → 2023/05/23
    #     - "yesterday" → 2023/05/29
    #     - "two weeks ago" → 2023/05/16
    #     - "last month" → approximately 2023/04/30

    #     **For events that might need ordering, assign sequence tags!**

    #     When you see multiple related events, assign them to a chain:
    #     ```json
    #     {
    #     "temporal": {
    #         "absolute_date": "2023-05-15",
    #         "event_chain": "museum_visits_2023",
    #         "sequence_order": 1
    #     }
    #     }
    #     ```

    #     **For countable/summable items, always add aggregation metadata!**

    #     Ask yourself: "Could the user ask 'how many' or 'how much total' about this?"
    #     If yes, add:
    #     ```json
    #     {
    #     "aggregation": {
    #         "key": "descriptive_key_name",
    #         "operation": "count|sum",
    #         "value": 1
    #     }
    #     }
    #     ```

    #     **Extract what the ASSISTANT said, not just what the user said!**

    #     When assistant provides:
    #     - Restaurant names → ASSISTANT_KNOWLEDGE
    #     - Phone numbers → ASSISTANT_KNOWLEDGE
    #     - Step-by-step instructions → ASSISTANT_KNOWLEDGE
    #     - Technical recommendations → ASSISTANT_KNOWLEDGE
    #     - Creative writing (stories, songs) → ASSISTANT_KNOWLEDGE

    #     **Track temporal states for attributes that change over time.**

    #     - "I was an atheist" → state: "previous"
    #     - "I am now more spiritual" → state: "current"
    #     - "My previous job was..." → state: "previous"
    #     - "I currently work as..." → state: "current"

    #     **Detect constraints that would affect future recommendations!**

    #     Look for:
    #     - "I don't like..." / "I want to avoid..."
    #     - "I already have..." / "I'm using..."
    #     - "I prefer..." / "I'm looking for something that..."
    #     - Time/budget/dietary/accessibility constraints

    #     ---

    #     ## OUTPUT FORMAT (JSON)

    #     ```json
    #     {
    #     "memory_facts": [
    #         {
    #         "fact_id": "F1",
    #         "category": "CATEGORY_NAME",
    #         "content": "Complete, detailed description preserving all nuances",
    #         "attributes": {
    #             "key": "value pairs specific to the category"
    #         },
    #         "temporal": {
    #             "relative_expression": "original expression if any",
    #             "absolute_date": "YYYY-MM-DD",
    #             "duration": "time span if applicable",
    #             "event_chain": "chain_id for ordering",
    #             "sequence_order": 1
    #         },
    #         "aggregation": {
    #             "key": "aggregation_group_key",
    #             "operation": "count|sum",
    #             "value": 1,
    #             "unit": "optional unit for sums"
    #         },
    #         "source": "user_statement|assistant_response",
    #         "session_id": "source session ID",
    #         "session_date": "session date",
    #         "confidence": 0.95,
    #         "qa_relevance": ["types of questions this fact can answer"]
    #         }
    #     ],
    #     "event_chains": [
    #         {
    #         "chain_id": "museum_visits_2023",
    #         "description": "Sequence of museum visits in 2023",
    #         "events": ["F1", "F5", "F12"],
    #         "ordered_by": "date"
    #         }
    #     ],
    #     "aggregation_groups": [
    #         {
    #         "key": "model_kits_owned",
    #         "description": "Model kits user has worked on or bought",
    #         "facts": ["F3", "F7", "F15", "F22", "F31"],
    #         "total_count": 5
    #         }
    #     ],
    #     "extraction_summary": {
    #         "total_facts": 0,
    #         "by_category": {
    #         "USER_ATTRIBUTE": 0,
    #         "EPISODIC_EVENT": 0,
    #         "INVENTORY_ITEM": 0,
    #         "PREFERENCE_HABIT": 0,
    #         "RELATIONSHIP_FACT": 0,
    #         "QUANTITATIVE_FACT": 0,
    #         "TEMPORAL_MARKER": 0,
    #         "ASSISTANT_KNOWLEDGE": 0,
    #         "AGGREGATABLE_ITEM": 0,
    #         "IMPLICIT_CONSTRAINT": 0
    #         },
    #         "event_chains_count": 0,
    #         "aggregation_groups_count": 0
    #     }
    #     }
    #     ```

    #     ---

    #     ## NOW EXTRACT MEMORY FACTS FROM THE SESSIONS ABOVE

    #     Remember:
    #     1. **Preserve ALL details** - "a lighter shade of gray" not "gray"
    #     2. **Convert relative time** - use session dates to calculate absolute dates
    #     3. **Extract complete events** - not just nouns
    #     4. **Track state changes** - mark "previous" vs "current"
    #     5. **Include units with numbers** - "45 minutes each way" not "45"
    #     6. **Tag aggregatable items** - add aggregation keys for countable/summable items
    #     7. **Build event chains** - link related events for ordering questions
    #     8. **Extract assistant knowledge** - capture what the AI recommended/said
    #     9. **Detect implicit constraints** - note preferences that affect recommendations
    #     10. **Calculate temporal distances** - include days_before_reference for "how many days ago" questions

    #     Extract memory facts now:"""
        
    #     return Template(template_string)
    
    def _build_episodic_memory_extraction_prompt(self) -> Template:
        """
        Builds the Episodic Memory Extraction Prompt V2 (English Version).
        Optimized for Single-Session High-Precision Extraction with ONE-SHOT EXAMPLE.
        """
        
        template_string = """You are a **High-Precision Memory Archivist** specializing in personal episodic memory preservation. Your task is to extract **Atomic Memory Facts** from the user's conversation session.

        ## CONVERSATION SESSION

        $sessions_text

        ## CURRENT SESSION DATE: $reference_date

        **CRITICAL TEMPORAL INSTRUCTION:**
        The conversation above takes place strictly on **$reference_date**.
        - Treat this date as **"TODAY"**.
        - If the user says "yesterday", calculate exactly 1 day before $reference_date.
        - If the user says "last Tuesday", calculate the date relative to $reference_date.
        - **ALL** extracted dates must be absolute (YYYY-MM-DD).

        ---

        ## MEMORY FACT CATEGORIES (10 Types)

        ### 1. USER_ATTRIBUTE
        Static profile information that defines who the user is.
        *Extract:* Education, current/previous jobs, identity, nationality, age, physical attributes, beliefs.
        *Example:* User is a "Marketing specialist" (preserve exact title).

        ### 2. EPISODIC_EVENT
        Specific actions the user performed at a specific time/place.
        *Extract:* WHAT action, WHERE it happened, WHEN (absolute date), WHO was involved, HOW it was done.
        *Constraint:* Do not merge events. If user went to the gym twice, extract 2 separate event facts.

        ### 3. INVENTORY_ITEM
        Things the user owns, bought, or possesses.
        *Extract:* Full specifications, Brand/Model, Color, Price, Quantity, Acquisition details.
        *Example:* "User bought a yellow raincoat" (preserve 'yellow').

        ### 4. PREFERENCE_HABIT
        User's likes, dislikes, routines, and regular behaviors.
        *Extract:* Explicit preferences ("favorite", "love", "hate"), Routines ("usually", "every morning").

        ### 5. RELATIONSHIP_FACT
        Social connections and their attributes.
        *Extract:* Person's name, Relationship type (sister, friend), Attributes of that person (where they live, what they do).

        ### 6. QUANTITATIVE_FACT
        Numerical information with full context and units.
        *Extract:* Prices ($$), Durations (minutes/hours), Quantities, Measurements, Scores.
        *Example:* "User spent $$45.50 on dinner", "Commute is 45 mins".

        ### 7. TEMPORAL_MARKER
        Time information for reasoning about event sequences.
        *Extract:* Duration spans ("lived there for 2 years"), Sequence orders ("event A happened before event B").

        ### 8. ASSISTANT_KNOWLEDGE
        Information, recommendations, or advice that the ASSISTANT provided.
        *Extract:* Recommendations (restaurants, books), Factual info provided by AI, Specific instructions given.
        *Source:* Capture what the AI said.

        ### 9. AGGREGATABLE_ITEM
        Items that belong to a countable/summable group.
        *Extract:* Any item that contributes to a "How many" or "How much total" question.
        *Constraint:* Record the value for THIS session only. Do not sum with past knowledge.

        ### 10. IMPLICIT_CONSTRAINT
        Implicit preferences or limitations affecting recommendations.
        *Extract:* "avoid", "don't like", dietary restrictions, budget limits, time constraints.

        ---

        ## SPECIAL EXTRACTION RULES (STRICT)

        ### Rule 1: NO SUMMARIZATION
        **Extract Atomic Facts only.** Do not generalize. If user mentions 3 separate purchases, create 3 facts. Preserve exact adjectives and numbers.

        ### Rule 2: ABSOLUTE TIME RESOLUTION
        **Convert ALL relative time to absolute dates using CURRENT SESSION DATE ($reference_date).** Never output relative words like "yesterday" in the output.

        ### Rule 3: ATOMIC AGGREGATION
        **Record counts/sums for THIS SESSION only.** Do not try to calculate a running total from previous history.

        ### Rule 4: STATE TRACKING
        **Track changes in user status.** Use `state: "previous"` vs `state: "current"` for attributes that changed.

        ---

        ## ONE-SHOT DEMONSTRATION (LEARN FROM THIS)

        **Input Session:**
        Date: 2023-05-20
        User: "Yesterday I bought a red Gibson guitar for $$1200. I used to play piano, but now I focus on strings."

        **Correct Output Logic:**
        1. "Yesterday" (relative to 2023-05-20) -> **2023-05-19** (Absolute Date).
        2. "Red Gibson guitar" -> INVENTORY_ITEM (Brand: Gibson, Color: Red).
        3. "$$1200" -> QUANTITATIVE_FACT (Unit: USD).
        4. "Used to play piano" -> USER_ATTRIBUTE (State: **previous**).
        5. "Now focus on strings" -> USER_ATTRIBUTE (State: **current**).

        **Target JSON Output:**
        ```json
        {
        "memory_facts": [
            {
            "category": "EPISODIC_EVENT",
            "content": "User bought a red Gibson guitar",
            "attributes": { "action": "bought", "object": "Gibson guitar", "color": "red" },
            "temporal": { "relative_expression": "Yesterday", "absolute_date": "2023-05-19" },
            "confidence": 1.0
            },
            {
            "category": "QUANTITATIVE_FACT",
            "content": "User spent $$1200 on the guitar",
            "attributes": { "value": 1200, "unit": "USD", "context": "guitar purchase" },
            "temporal": { "absolute_date": "2023-05-19" },
            "confidence": 1.0
            },
            {
            "category": "INVENTORY_ITEM",
            "content": "User owns a red Gibson guitar",
            "attributes": { "item": "Gibson guitar", "brand": "Gibson", "color": "red", "price": 1200 },
            "temporal": { "absolute_date": "2023-05-19" },
            "confidence": 1.0
            },
            {
            "category": "USER_ATTRIBUTE",
            "content": "User plays the piano",
            "attributes": { "skill": "piano", "state": "previous" },
            "temporal": { "absolute_date": "2023-05-20" },
            "confidence": 0.9
            },
            {
            "category": "USER_ATTRIBUTE",
            "content": "User focuses on string instruments",
            "attributes": { "skill": "strings", "state": "current" },
            "temporal": { "absolute_date": "2023-05-20" },
            "confidence": 0.9
            }
        ]
        }
        ```

        ---

        ## OUTPUT FORMAT (JSON)

        Please strictly follow the JSON structure shown in the demonstration above.

        Extract high-precision memory facts now:"""
        
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
        
        actual_end_idx = min(end_idx, len(haystack_sessions))
        
        for idx in range(start_idx, actual_end_idx):
            session = haystack_sessions[idx]
            session_id = haystack_session_ids[idx] if idx < len(haystack_session_ids) else f"session_{idx}"
            session_date = haystack_dates[idx] if idx < len(haystack_dates) else "Unknown"
            
            if idx == actual_end_idx - 1:
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
            start_idx = i
            end_idx = min(i + self.sessions_per_group, total_sessions)
            groups.append((start_idx, end_idx))
        return groups
    
    def generate_batch_requests(self,
                                start_index: int = 0,
                                end_index: Optional[int] = None,
                                model: str = "qwen-plus-latest",
                                enable_thinking: bool = False,
                                thinking_budget: int = 1024) -> str:
        """Generate batch requests."""
        
        qa_samples = self.load_dataset()
        
        if end_index is None:
            end_index = len(qa_samples) - 1
        
        if start_index < 0 or start_index >= len(qa_samples):
            raise ValueError(f"start_index {start_index} 超出范围 [0, {len(qa_samples)-1}]")
        if end_index < start_index or end_index >= len(qa_samples):
            raise ValueError(f"end_index {end_index} 超出范围 [{start_index}, {len(qa_samples)-1}]")
        
        selected_samples = qa_samples[start_index:end_index+1]
        
        logger.info(f"\n{'='*80}")
        logger.info(f" 生成 Episodic Memory V2 批量推理请求")
        logger.info(f"{'='*80}")
        logger.info(f"处理范围: QA {start_index} - {end_index}")
        logger.info(f"样本数量: {len(selected_samples)}")
        logger.info(f"使用模型: {model}")
        logger.info(f"每组session数: {self.sessions_per_group}")
        logger.info(f"Thinking模式: {'启用 (budget={})'.format(thinking_budget) if enable_thinking else '禁用'}")
        logger.info(f"{'='*80}\n")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"episodic_memory_v2_qa{start_index}_to_qa{end_index}_{timestamp}.jsonl"
        
        total_sessions = 0
        total_messages = 0
        total_groups = 0
        total_requests = 0
        skipped_empty = 0
        
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
            for idx, qa_data in enumerate(selected_samples):
                qa_index = start_index + idx
                
                question_id = qa_data.get("question_id", f"qa_{qa_index}")
                question = qa_data.get("question", "")
                question_type = qa_data.get("question_type", "unknown")
                
                haystack_sessions = qa_data.get("haystack_sessions", [])
                haystack_session_ids = qa_data.get("haystack_session_ids", [])
                haystack_dates = qa_data.get("haystack_dates", [])
                
                if not haystack_sessions:
                    logger.warning(f" QA {qa_index} ({question_id}) 没有 haystack_sessions，跳过")
                    skipped_empty += 1
                    continue
                
                session_count = len(haystack_sessions)
                message_count = sum(len(session) for session in haystack_sessions)
                total_sessions += session_count
                total_messages += message_count
                
                session_groups = self._split_sessions_into_groups(session_count)
                total_groups += len(session_groups)
                
                logger.debug(f" 处理 QA {qa_index}: {session_count} sessions, {len(session_groups)} groups")
                
                for group_idx, (start_session_idx, end_session_idx) in enumerate(session_groups):
                    group_session_count = end_session_idx - start_session_idx
                    
                    try:
                        sessions_text, reference_date = self._build_sessions_text(
                            haystack_sessions,
                            haystack_session_ids,
                            haystack_dates,
                            start_session_idx,
                            end_session_idx
                        )
                        
                        if not sessions_text.strip():
                            logger.warning(f" QA {qa_index} group {group_idx} 会话文本为空，跳过")
                            continue
                        
                        user_prompt = prompt_template.substitute(
                            sessions_text=sessions_text,
                            reference_date=reference_date if reference_date else "Unknown"
                        )
                        
                        custom_id = f"qa_{qa_index}_group_{group_idx}_sessions_{start_session_idx}_{end_session_idx-1}"
                        
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
                        
                        if '\n' in json_line:
                            logger.warning(f" {custom_id} 的JSON中包含换行符")
                        
                        f.write(json_line + '\n')
                        total_requests += 1
                        
                    except Exception as e:
                        logger.error(f" 处理 QA {qa_index} group {group_idx} 时出错: {e}")
                        logger.error(f"   跳过此组并继续...")
                        continue
                
                if (idx + 1) % 10 == 0:
                    logger.info(f" 进度: {idx + 1}/{len(selected_samples)} QA 已处理, {total_requests} 请求已生成")
        
        logger.info(f"\n{'='*80}")
        logger.info(f" 批量请求文件生成完成!")
        logger.info(f"{'='*80}")
        logger.info(f"输出文件: {output_file}")
        logger.info(f"文件大小: {output_file.stat().st_size / 1024:.2f} KB")
        logger.info(f"\n 统计信息:")
        logger.info(f"  处理的QA数量: {len(selected_samples)}")
        logger.info(f"  跳过的空QA: {skipped_empty}")
        logger.info(f"  生成的请求数: {total_requests}")
        logger.info(f"  Session组数: {total_groups}")
        logger.info(f"  总Session数: {total_sessions}")
        logger.info(f"  总消息数: {total_messages}")
        
        if len(selected_samples) > skipped_empty:
            actual_processed = len(selected_samples) - skipped_empty
            logger.info(f"\n 平均统计:")
            logger.info(f"  平均Session/QA: {total_sessions / actual_processed:.1f}")
            logger.info(f"  平均消息/QA: {total_messages / actual_processed:.1f}")
            logger.info(f"  平均请求/QA: {total_requests / actual_processed:.1f}")
            if total_groups > 0:
                logger.info(f"  平均Session/组: {total_sessions / total_groups:.1f}")
        logger.info(f"{'='*80}\n")
        
        
        metadata = {
            "created_at": datetime.now().isoformat(),
            "dataset_path": str(self.dataset_path),
            "start_index": start_index,
            "end_index": end_index,
            "sessions_per_group": self.sessions_per_group,
            "total_qa_count": len(selected_samples),
            "skipped_empty_qa": skipped_empty,
            "total_requests": total_requests,
            "total_groups": total_groups,
            "total_sessions": total_sessions,
            "total_messages": total_messages,
            "model": model,
            "enable_thinking": enable_thinking,
            "thinking_budget": thinking_budget if enable_thinking else None,
            "memory_categories": EpisodicMemoryCategory.get_all_categories(),
            "prompt_version": "V2",
            "note": "Episodic Memory Extraction V2 - supports aggregation, event chains, assistant knowledge, and implicit constraints"
        }
        
        metadata_file = output_file.with_suffix('.meta.json')
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f" 元数据文件: {metadata_file}")
        
        return str(output_file)


def main():
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(
        description="LongMemEval Episodic Memory V2 批量抽取 JSONL 生成器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        示例:
        # 生成前10个QA的请求
        python step1_build_batch_requests.py --start-index 0 --end-index 9
        
        # 生成所有500个QA的请求，每5个session一组
        python step1_build_batch_requests.py --start-index 0 --end-index 499 --sessions-per-group 5
        
        # 生成第100-199个QA的请求，使用不同模型
        python step1_build_batch_requests.py --start-index 100 --end-index 199 --model qwen-max
        
        # 处理所有QA（不指定end-index）
        python step1_build_batch_requests.py --start-index 0

        V2 增强特性:
        - 聚合关联标记: 支持 "How many X?" 类问题
        - 时序事件链: 支持 "Which happened first?" 类问题
        - Assistant知识抽取: 支持 "What did you recommend?" 类问题
        - 隐含约束检测: 支持个性化推荐问题

        批量请求格式 (阿里云百炼/Qwen):
        {"custom_id":"...","method":"POST","url":"/v1/chat/completions","body":{"model":"qwen-plus","messages":[...]}}
        """
    )
    
    parser.add_argument("--dataset-path",
                       default=str(paths.LONGMEMEVAL_S_CLEANED_FILE),
                       help="数据集文件路径")
    parser.add_argument("--output-dir",
                       default=str(paths.LONGMEMEVAL_EPISODIC_NEW_BATCH_REQUESTS_DIR),
                       help="输出目录路径")
    
    
    parser.add_argument("--start-index", type=int, required=True,
                       help="起始QA索引 (0-499)")
    parser.add_argument("--end-index", type=int, default=None,
                       help="结束QA索引 (0-499), 默认处理到最后")
    
    parser.add_argument("--sessions-per-group", type=int, default=1,
                       help="每组处理的session数量 (默认1)")
    
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
    
    parser.add_argument("--enable-thinking", action="store_true",
                       help="启用模型的推理/思考模式 (默认禁用，以防 Qwen 3.5 默认开启导致 token 浪费)")
    parser.add_argument("--thinking-budget", type=int, default=2048,
                       help="思考过程的 Token 预算（仅在启用 enable-thinking 时生效）")
    
    parser.add_argument("--debug", action="store_true",
                       help="启用调试模式")
    
    args = parser.parse_args()
    
    # Avoid mutating LogRecord fields before other handlers process the record.
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        extractor = LongMemEvalEpisodicMemoryBatchExtractor(
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            sessions_per_group=args.sessions_per_group
        )
        
        output_file = extractor.generate_batch_requests(
            start_index=args.start_index,
            end_index=args.end_index,
            model=args.model,
            enable_thinking=args.enable_thinking,
            thinking_budget=args.thinking_budget
        )
        
        print(f"\n{'='*80}")
        print(f" 成功生成批量推理请求文件!")
        print(f"{'='*80}")
        print(f" 文件路径: {output_file}")
        print(f"\n 注意:")
        print(f"  - Sessions已按{args.sessions_per_group}个一组分批处理")
        print(f"  - custom_id格式: qa_X_group_Y_sessions_A_B")
        print(f"  - 后续需要合并同一QA的多个请求结果")
        print(f"  - 使用模型: {args.model}")
        print(f"\n V2 增强特性:")
        print(f"  - 聚合关联标记: 支持 'How many X?' 类问题")
        print(f"  - 时序事件链: 支持 'Which happened first?' 类问题")
        print(f"  - Assistant知识抽取: 支持 'What did you recommend?' 类问题")
        print(f"  - 隐含约束检测: 支持个性化推荐问题")
        print(f"\n下一步:")
        print(f"  1. 上传文件到阿里云百炼进行批量推理")
        print(f"  2. 下载推理结果")
        print(f"  3. 合并同一QA的多个请求结果")
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
    exit(main())