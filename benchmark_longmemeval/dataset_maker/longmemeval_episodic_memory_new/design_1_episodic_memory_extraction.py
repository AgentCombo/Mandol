#!/usr/bin/env python3
"""Utilities for design 1 episodic memory extraction."""

import json
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from string import Template
import re
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
    
    @classmethod
    def get_all_categories(cls) -> List[str]:
        return [
            cls.USER_ATTRIBUTE,
            cls.EPISODIC_EVENT,
            cls.INVENTORY_ITEM,
            cls.PREFERENCE_HABIT,
            cls.RELATIONSHIP_FACT,
            cls.QUANTITATIVE_FACT,
            cls.TEMPORAL_MARKER
        ]


class LongMemEvalEpisodicMemoryExtractor:
    
    def __init__(self,
                 dataset_path: str = str(paths.LONGMEMEVAL_S_CLEANED_FILE),
                 output_dir: str = str(paths.LONGMEMEVAL_ENTITY_RELATION_LEGACY_BATCH_REQUESTS_EPISODIC_DIR),
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
    
    def _build_episodic_memory_extraction_prompt(self) -> Template:
        """Build episodic memory extraction prompt."""
        
        template_string = """You are a **Memory Archivist** specializing in personal episodic memory preservation for AI assistants. Your task is to extract **Atomic Memory Facts** from the user's conversation history.

        ## CRITICAL UNDERSTANDING

        This is NOT a Named Entity Recognition task. This is an **Episodic Memory Extraction** task.

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

        ## MEMORY FACT CATEGORIES

        ### 1. USER_ATTRIBUTE (用户属性)
        Static profile information that defines who the user is.

        **Extract:**
        - Educational background: degrees, schools, majors
        - Professional history: CURRENT job, PREVIOUS jobs (mark temporal state!)
        - Personal identity: ethnicity, nationality, age
        - Physical attributes: height, weight
        - Beliefs/Stances: religious views, political views (with temporal state if changed!)

        **Examples:**
        - "I graduated with a Business Administration degree" → USER_ATTRIBUTE
        - "I WAS a marketing specialist at a startup" (previous) → USER_ATTRIBUTE (state: "previous")
        - "I AM now a project manager" (current) → USER_ATTRIBUTE (state: "current")
        - "I'm a mix of Irish and Italian" → USER_ATTRIBUTE

        ### 2. EPISODIC_EVENT (情景事件)
        Specific actions the user performed at a specific time/place.

        **Extract the COMPLETE event structure:**
        - WHAT action was performed
        - WHERE it happened
        - WHEN it happened (convert relative to absolute!)
        - WHO was involved
        - HOW it was done
        - WHY (if mentioned)

        **Examples:**
        - "I redeemed a $5 coupon on coffee creamer at Target" → 
        - action: "redeemed coupon"
        - object: "$5 coupon on coffee creamer"
        - location: "Target"
        
        - "Last Tuesday I attended The Glass Menagerie at the community theater" →
        - action: "attended play"
        - object: "The Glass Menagerie"
        - location: "local community theater"
        - absolute_date: "[Convert 'last Tuesday' using reference date]"

        ### 3. INVENTORY_ITEM (物品清单)
        Things the user owns, bought, or possesses.

        **CRITICAL: Preserve ALL details:**
        - Brand/Model: Nike, iPhone 13 Pro, Fitbit Inspire HR
        - Specifications: 500 Mbps, 16GB RAM, 55-inch
        - Color/Style: "a lighter shade of gray", "yellow dress"
        - Quantity: "three bikes", "20 playlists"
        - Acquisition details: where bought, when bought, price paid

        **Examples:**
        - "My new internet plan is 500 Mbps" → INVENTORY_ITEM (spec: "500 Mbps")
        - "I bought a yellow dress for my sister" → INVENTORY_ITEM (color: "yellow", purpose: "gift for sister")
        - "I have three bikes" → INVENTORY_ITEM (quantity: 3, type: "bikes")
        - "My cat Luna" → INVENTORY_ITEM (type: "pet/cat", name: "Luna")

        ### 4. PREFERENCE_HABIT (偏好习惯)
        User's likes, dislikes, routines, and regular behaviors.

        **Extract:**
        - Preferences: "favorite", "prefer", "like", "hate"
        - Routines: "usually", "always", "every [day/week]", "typically"
        - Time-based habits: wake up time, work schedule, exercise routine
        - Consumption habits: brands used, services subscribed

        **Examples:**
        - "My daily commute is 45 minutes each way" → PREFERENCE_HABIT (commute_duration: "45 minutes each way")
        - "I stop checking work emails at 7 pm" → PREFERENCE_HABIT (habit: "no work emails after 7 pm")
        - "My preferred gin-to-vermouth ratio is 3:1" → PREFERENCE_HABIT (preference: "gin martini ratio 3:1")
        - "I usually get home at 6:30 pm on weeknights" → PREFERENCE_HABIT (routine: "home by 6:30 pm on weeknights")

        ### 5. RELATIONSHIP_FACT (人际关系)
        Social connections and their attributes.

        **Extract:**
        - Relationship type: sister, friend, colleague, cousin
        - Person's name (if mentioned)
        - Person's attributes: where they live, what they do, their characteristics
        - Shared experiences with the user

        **Examples:**
        - "My sister Emily lives in Denver" → RELATIONSHIP_FACT
        - relation: "sister"
        - name: "Emily"
        - attribute: "lives in Denver"
        
        - "Sarah and I had a conversation about destiny" → RELATIONSHIP_FACT
        - person: "Sarah"
        - interaction: "conversation about destiny"

        ### 6. QUANTITATIVE_FACT (数值事实)
        Numerical information with full context.

        **CRITICAL: ALWAYS include units and context!**

        **Extract:**
        - Prices: "$800", "$5 coupon"
        - Durations: "45 minutes each way", "4 hours", "two weeks"
        - Quantities: "20 playlists", "7 shirts", "three bikes"
        - Measurements: "500 Mbps", "16GB", "55-inch"
        - Percentages/Ratios: "10% discount", "3:1 ratio"
        - Distances: "45 minutes commute"

        **Examples:**
        - "I spent $800 on a designer handbag" → QUANTITATIVE_FACT (amount: "$800", item: "designer handbag")
        - "It took 4 hours to assemble the IKEA bookshelf" → QUANTITATIVE_FACT (duration: "4 hours", task: "assemble IKEA bookshelf")
        - "I packed 7 shirts for my 5-day trip to Costa Rica" → QUANTITATIVE_FACT (quantity: "7 shirts", context: "5-day trip to Costa Rica")

        ### 7. TEMPORAL_MARKER (时间标记)
        Temporal information for reasoning about event sequences.

        **CRITICAL: Convert relative to absolute dates!**

        **Extract:**
        - Relative references: "last Tuesday", "two weeks ago", "recently"
        - Absolute dates: "February 14th", "2023/05/30"
        - Duration spans: "for two weeks", "over a year"
        - Sequence markers: "before", "after", "first", "then"

        **Examples:**
        - "I was in Japan for two weeks" → TEMPORAL_MARKER (duration: "two weeks", location: "Japan")
        - "Last Tuesday I went shopping" (ref date: 2023/05/30 Tue) → TEMPORAL_MARKER
        - relative: "last Tuesday"
        - absolute: "2023/05/23"

        ---

        ## EXTRACTION RULES

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
        - "last month" → 2023/04/30 (approximately)

        ### Rule 3: ACTION-CENTRICITY (事件中心)
        **Don't just extract nouns. Extract the COMPLETE event.**

         Extract: "Target" (noun)
         Extract: "User redeemed a $5 coupon on coffee creamer at Target" (complete event)

         Extract: "yoga" (noun)
         Extract: "User takes yoga classes at Serenity Yoga" (complete fact)

        ### Rule 4: STATE TRACKING (状态追踪)
        **Track temporal states for attributes that change over time.**

        - "I was an atheist" → state: "previous"
        - "I am now more spiritual" → state: "current"
        - "My previous job was..." → state: "previous"
        - "I currently work as..." → state: "current"

        ---

        ## OUTPUT FORMAT (JSON)

        ```json
        {
        "memory_facts": [
            {
            "fact_id": "F1",
            "category": "EPISODIC_EVENT|USER_ATTRIBUTE|INVENTORY_ITEM|PREFERENCE_HABIT|RELATIONSHIP_FACT|QUANTITATIVE_FACT|TEMPORAL_MARKER",
            "content": "Complete, detailed description of the memory fact (preserve all nuances!)",
            "attributes": {
                "action": "verb phrase (for events)",
                "object": "what was acted upon",
                "location": "where it happened",
                "person": "who was involved",
                "quantity": "numerical value with unit",
                "specification": "detailed specs (brand, model, color, size)",
                "state": "current|previous|ongoing (for temporal attributes)"
            },
            "temporal": {
                "relative_expression": "original expression (e.g., 'last Tuesday')",
                "absolute_date": "YYYY-MM-DD or specific date",
                "duration": "time span if applicable"
            },
            "session_id": "source session ID",
            "session_date": "session date for reference",
            "confidence": 0.95,
            "qa_relevance": ["types of questions this fact can answer"]
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
            "TEMPORAL_MARKER": 0
            }
        }
        }
        ```

        ---

        ## FEW-SHOT EXAMPLES

        ### Example 1: Complex Event
        **Input:** "Last Tuesday I redeemed a $5 coupon on coffee creamer at Target."
        **Session Date:** 2023/05/30 (Tuesday)

        **Output:**
        ```json
        {
        "fact_id": "F1",
        "category": "EPISODIC_EVENT",
        "content": "User redeemed a $5 coupon on coffee creamer at Target",
        "attributes": {
            "action": "redeemed coupon",
            "object": "$5 coupon on coffee creamer",
            "location": "Target"
        },
        "temporal": {
            "relative_expression": "last Tuesday",
            "absolute_date": "2023-05-23"
        },
        "qa_relevance": ["Where did I redeem a coupon?", "What did I buy at Target?"]
        }
        ```

        ### Example 2: State Change
        **Input:** "I used to be a marketing specialist at a small startup. Now I'm a project manager at TechCorp."

        **Output:**
        ```json
        [
        {
            "fact_id": "F2",
            "category": "USER_ATTRIBUTE",
            "content": "User was previously a marketing specialist at a small startup",
            "attributes": {
            "role": "marketing specialist",
            "organization": "a small startup",
            "state": "previous"
            },
            "qa_relevance": ["What was my previous occupation?"]
        },
        {
            "fact_id": "F3",
            "category": "USER_ATTRIBUTE", 
            "content": "User is currently a project manager at TechCorp",
            "attributes": {
            "role": "project manager",
            "organization": "TechCorp",
            "state": "current"
            },
            "qa_relevance": ["What is my current job?", "Where do I work?"]
        }
        ]
        ```

        ### Example 3: Quantitative with Full Detail
        **Input:** "My daily commute is 45 minutes each way."

        **Output:**
        ```json
        {
        "fact_id": "F4",
        "category": "QUANTITATIVE_FACT",
        "content": "User's daily commute is 45 minutes each way",
        "attributes": {
            "quantity": "45 minutes each way",
            "type": "commute duration",
            "frequency": "daily",
            "note": "each way means 45 min to work AND 45 min back = 90 min total"
        },
        "qa_relevance": ["How long is my commute?", "How long do I spend commuting?"]
        }
        ```

        ### Example 4: Inventory with Specifications
        **Input:** "I repainted my bedroom walls to a lighter shade of gray."

        **Output:**
        ```json
        {
        "fact_id": "F5",
        "category": "EPISODIC_EVENT",
        "content": "User repainted bedroom walls to a lighter shade of gray",
        "attributes": {
            "action": "repainted",
            "object": "bedroom walls",
            "specification": "a lighter shade of gray"
        },
        "qa_relevance": ["What color did I paint my bedroom?", "What did I do to my bedroom?"]
        }
        ```

        ### Example 5: Relationship
        **Input:** "My sister Emily moved to Denver last month."

        **Output:**
        ```json
        {
        "fact_id": "F6",
        "category": "RELATIONSHIP_FACT",
        "content": "User's sister Emily lives in Denver (moved recently)",
        "attributes": {
            "relation": "sister",
            "name": "Emily",
            "location": "Denver",
            "action": "moved"
        },
        "temporal": {
            "relative_expression": "last month",
            "note": "Convert using session reference date"
        },
        "qa_relevance": ["Where does my sister live?", "Who is Emily?"]
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
        
        for idx in range(start_idx, end_idx):
            session = haystack_sessions[idx]
            session_id = haystack_session_ids[idx]
            session_date = haystack_dates[idx]
            
            reference_date = session_date
            
            sessions_text += f"\n=== Session {idx + 1} ===\n"
            sessions_text += f"Session ID: {session_id}\n"
            sessions_text += f"Session Date: {session_date}\n"
            sessions_text += f"Messages:\n"
            
            for msg_idx, message in enumerate(session, start=1):
                role = message.get("role", "unknown")
                content = message.get("content", "")
                
                content = self._sanitize_content(content)
                
                MAX_CONTENT_LENGTH = 50000
                if len(content) > MAX_CONTENT_LENGTH:
                    content = content[:MAX_CONTENT_LENGTH] + "\n... [content truncated]"
                
                if content and content.strip():
                    sessions_text += f"  [{msg_idx}] {role}: {content}\n"
        
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
                                model: str = "deepseek-ai/DeepSeek-V3",
                                max_tokens: Optional[int] = 16384,
                                temperature: float = 0.1) -> str:
        """Generate batch requests."""
        qa_samples = self.load_dataset()
        
        if end_index is None:
            end_index = len(qa_samples) - 1
        
        if start_index < 0 or start_index >= len(qa_samples):
            raise ValueError(f"start_index {start_index} 超出范围")
        if end_index < start_index or end_index >= len(qa_samples):
            raise ValueError(f"end_index {end_index} 超出范围")
        
        selected_samples = qa_samples[start_index:end_index+1]
        
        logger.info(f"\n{'='*80}")
        logger.info(f" 生成 Episodic Memory 批量推理请求")
        logger.info(f"{'='*80}")
        logger.info(f"处理范围: QA {start_index} - {end_index}")
        logger.info(f"样本数量: {len(selected_samples)}")
        logger.info(f"使用模型: {model}")
        logger.info(f"每组session数: {self.sessions_per_group}")
        logger.info(f"{'='*80}\n")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"episodic_memory_qa{start_index}_to_qa{end_index}_{timestamp}.jsonl"
        
        total_requests = 0
        prompt_template = self._build_episodic_memory_extraction_prompt()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for idx, qa_data in enumerate(selected_samples):
                qa_index = start_index + idx
                
                question_id = qa_data.get("question_id", f"qa_{qa_index}")
                haystack_sessions = qa_data.get("haystack_sessions", [])
                haystack_session_ids = qa_data.get("haystack_session_ids", [])
                haystack_dates = qa_data.get("haystack_dates", [])
                
                session_count = len(haystack_sessions)
                session_groups = self._split_sessions_into_groups(session_count)
                
                for group_idx, (start_idx, end_idx) in enumerate(session_groups):
                    sessions_text, reference_date = self._build_sessions_text(
                        haystack_sessions, haystack_session_ids, haystack_dates,
                        start_idx, end_idx
                    )
                    
                    if not sessions_text.strip():
                        continue
                    
                    prompt = prompt_template.safe_substitute(
                        sessions_text=sessions_text,
                        reference_date=reference_date
                    )
                    
                    request_id = f"qa{qa_index}_group{group_idx}_sessions{start_idx}-{end_idx-1}"
                    
                    request = {
                        "custom_id": request_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": model,
                            "messages": [
                                {
                                    "role": "system",
                                    "content": "You are a Memory Archivist AI that extracts structured episodic memory facts from conversations. Output valid JSON only."
                                },
                                {
                                    "role": "user",
                                    "content": prompt
                                }
                            ],
                            "temperature": temperature,
                            "response_format": {"type": "json_object"}
                        }
                    }
                    
                    if max_tokens is not None:
                        request["body"]["max_tokens"] = max_tokens
                    
                    f.write(json.dumps(request, ensure_ascii=False) + "\n")
                    total_requests += 1
                
                if (idx + 1) % 50 == 0:
                    logger.info(f"  已处理 {idx + 1}/{len(selected_samples)} 个QA样本")
        
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
        description="LongMemEval Episodic Memory Extractor - 生成情景记忆抽取批量请求",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        示例用法:
        # 处理前10个QA
        python step1_episodic_memory_extractor.py --start 0 --end 9
        
        # 处理所有QA
        python step1_episodic_memory_extractor.py
        
        # 使用自定义模型
        python step1_episodic_memory_extractor.py --model gpt-4o
        """
    )
    
    parser.add_argument("--dataset", type=str,
                       default=str(paths.LONGMEMEVAL_S_CLEANED_FILE),
                       help="数据集文件路径")
    parser.add_argument("--output-dir", type=str,
                       default=str(paths.LONGMEMEVAL_ENTITY_RELATION_NEW_DIR / "batch_requests_episodic"),
                       help="输出目录")
    parser.add_argument("--start", type=int, default=0,
                       help="起始QA索引")
    parser.add_argument("--end", type=int, default=None,
                       help="结束QA索引")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-V3",
                       help="模型名称")
    parser.add_argument("--max-tokens", type=int, default=16384,
                       help="最大输出token数")
    parser.add_argument("--sessions-per-group", type=int, default=1,
                       help="每组处理的session数量")
    
    args = parser.parse_args()
    
    try:
        extractor = LongMemEvalEpisodicMemoryExtractor(
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
        
        logger.info(f" 完成！输出文件: {output_file}")
        return 0
        
    except Exception as e:
        logger.error(f" 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())