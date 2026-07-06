#!/usr/bin/env python3
"""Utilities for step1 entity batch requests."""
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


class LongMemEvalEntityType:
    
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    PRODUCT = "PRODUCT"
    
    DATE_TIME = "DATE_TIME"
    TEMPORAL_REFERENCE = "TEMPORAL_REFERENCE"
    LOCATION = "LOCATION"
    
    NUMERICAL_VALUE = "NUMERICAL_VALUE"
    DURATION = "DURATION"
    MEASUREMENT = "MEASUREMENT"
    
    EVENT = "EVENT"
    ACTIVITY = "ACTIVITY"
    
    RELATIONSHIP = "RELATIONSHIP"
    OBJECT = "OBJECT"
    ATTRIBUTE = "ATTRIBUTE"  # Avoid mutating LogRecord fields before other handlers process the record.
    
    PREFERENCE = "PREFERENCE"
    HABIT = "HABIT"
    
    SKILL = "SKILL"
    EDUCATION = "EDUCATION"
    OCCUPATION = "OCCUPATION"
    
    CONCEPT = "CONCEPT"
    GOAL = "GOAL"
    
    @classmethod
    def get_all_types(cls) -> List[str]:
        """Return all types."""
        return [
            cls.PERSON, cls.ORGANIZATION, cls.PRODUCT,
            cls.DATE_TIME, cls.TEMPORAL_REFERENCE, cls.LOCATION,
            cls.NUMERICAL_VALUE, cls.DURATION, cls.MEASUREMENT,
            cls.EVENT, cls.ACTIVITY,
            cls.RELATIONSHIP, cls.OBJECT, cls.ATTRIBUTE,
            cls.PREFERENCE, cls.HABIT,
            cls.SKILL, cls.EDUCATION, cls.OCCUPATION,
            cls.CONCEPT, cls.GOAL
        ]
    
    @classmethod
    def get_priority_description(cls) -> str:
        """Return priority description."""
        return """
        **HIGH PRIORITY Entities (Critical for QA):**

        1. **PERSON**: Names, roles, identities
        - Examples: "John", "my sister Emily", "my friend Rachel", "the manager"
        - Include both explicit names and role-based references

        2. **RELATIONSHIP**: Human relationships
        - Examples: "sister", "friend", "colleague", "cousin", "neighbor"
        - Important for understanding "who" in questions

        3. **DATE_TIME**: Absolute dates and times
        - Examples: "2023/05/30", "February 14th", "6:30 pm", "last Tuesday (2023/05/23)"
        - CRITICAL: Convert relative time to absolute dates using session dates

        4. **TEMPORAL_REFERENCE**: Relative time expressions
        - Examples: "last week", "two months ago", "recently", "yesterday"
        - Must preserve the original expression AND provide absolute conversion

        5. **LOCATION**: Geographic locations and venues
        - Examples: "Target", "downtown", "University of Melbourne", "Hawaii"
        - Include both specific addresses and general locations

        6. **NUMERICAL_VALUE**: All meaningful numbers
        - Examples: "$800", "45 minutes", "20 people", "500 Mbps"
        - MUST include the unit (dollars, minutes, people, etc.)

        7. **DURATION**: Time spans
        - Examples: "45 minutes each way", "3 weeks", "5 years"
        - Different from NUMERICAL_VALUE - represents a span of time

        8. **MEASUREMENT**: Units and measurements
        - Examples: "miles", "pounds", "GB", "Mbps"
        - Often paired with NUMERICAL_VALUE

        9. **PRODUCT**: Brands, models, services
        - Examples: "iPhone 13 Pro", "Nike running shoes", "Spotify"

        10. **ORGANIZATION**: Companies, institutions
            - Examples: "Target", "UCLA", "TechCorp"

        **MEDIUM PRIORITY Entities:**

        11. **PREFERENCE**: User preferences and favorites
            - Examples: "favorite", "prefer", "usually order"
            - Important for recommendation questions

        12. **HABIT**: Regular behaviors and routines
            - Examples: "usually wake up at", "every Tuesday", "always"

        13. **EVENT**: Specific events
            - Examples: "cousin's wedding", "conference", "Super Bowl"

        14. **ACTIVITY**: Hobbies and actions
            - Examples: "yoga classes", "tennis", "cooking"

        15. **OBJECT**: Physical items
            - Examples: "yellow dress", "tennis racket", "laptop"

        16. **ATTRIBUTE**: Descriptive properties
            - Examples: "lighter shade of gray", "45 minutes each way"

        **LOW PRIORITY Entities:**

        17. **OCCUPATION**: Jobs and positions
        18. **EDUCATION**: Degrees and courses
        19. **SKILL**: Abilities and talents
        20. **CONCEPT**: Abstract concepts
        21. **GOAL**: Plans and intentions
        """


class LongMemEvalBatchEntityExtractor:
    
    def __init__(self,
                 dataset_path: str = str(paths.LONGMEMEVAL_S_CLEANED_FILE),
                 output_dir: str = str(paths.LONGMEMEVAL_ENTITY_RELATION_LEGACY_BATCH_REQUESTS_DIR),
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
    
    def _build_unified_extraction_prompt(self) -> Template:
        """Build unified extraction prompt."""
        
        template_string = """You are a professional entity extraction expert specializing in conversational data analysis for question-answering systems.

        **Task:** Extract ALL important entities from the given conversation sessions that could be used to answer various types of questions about the conversations, including factual questions, temporal reasoning, preferences, and recommendations.

        **Conversation Sessions:**
        $sessions_text

        **Entity Type Schema:**
        $entity_types_description

        **CRITICAL Extraction Guidelines:**

        1. **Temporal Conversion (HIGHEST PRIORITY):**
        - Convert ALL relative time references to absolute dates using session dates
        - Example: If session date is "2023/05/30" and text says "last Tuesday", extract:
            * TEMPORAL_REFERENCE: "last Tuesday"
            * DATE_TIME: "2023/05/23 (Tue)"
        - Extract both relative expressions AND their absolute conversions

        2. **Relationship Extraction:**
        - Extract ALL relationship references: "my sister Emily", "my friend Rachel"
        - Create separate entities for:
            * PERSON: "Emily"
            * RELATIONSHIP: "sister"
        - Link them with contextual information

        3. **Numerical Values with Units:**
        - ALWAYS include units with numbers
        - "$$800" → NUMERICAL_VALUE with content "800 dollars"
        - "45 minutes" → DURATION with content "45 minutes"
        - "500 Mbps" → NUMERICAL_VALUE with content "500 Mbps"

        4. **Preference and Habit Extraction:**
        - Extract user preferences: "favorite", "usually", "prefer"
        - Extract regular behaviors: "every Tuesday", "always"
        - These are critical for recommendation questions

        5. **Comprehensive Coverage:**
        - Extract entities from ALL sessions in this group
        - Don't skip "obvious" information - it might be needed for questions

        6. **Standardization:**
        - Use consistent naming: "John Smith" not "john", "Smith", "Mr. Smith"
        - Standardize locations: "University of California, Los Angeles (UCLA)"

        **Output Format (JSON):**
        {
            "entities": [
                {
                    "entity_id": "E1",
                    "name": "Standardized entity name",
                    "type": "ENTITY_TYPE",
                    "content": "Detailed description with full context",
                    "session_id": "session_X",
                    "session_date": "YYYY/MM/DD (Day) HH:MM",
                    "temporal_info": "Absolute date if time-related (e.g., 2023/05/23)",
                    "temporal_reference": "Original relative expression (e.g., last Tuesday)",
                    "spatial_info": "Location reference if applicable",
                    "numerical_value": "Number with unit if applicable",
                    "related_entities": ["E2", "E3"],
                    "aliases": ["alternative names or mentions"],
                    "confidence": 0.95
                }
            ]
        }

        **Priority Examples:**

        HIGH: "my sister Emily moved to Denver" →
        - PERSON: "Emily" (content: "User's sister, moved to Denver")
        - RELATIONSHIP: "sister" (content: "Emily is the user's sister")
        - LOCATION: "Denver" (content: "City where Emily moved to")

        HIGH: "last Tuesday I went to Target" (session date: 2023/05/30 Tue) →
        - TEMPORAL_REFERENCE: "last Tuesday" (temporal_info: "2023/05/23 (Tue)")
        - DATE_TIME: "2023/05/23 (Tue)" (content: "Date when user went to Target")
        - ORGANIZATION: "Target" (content: "Store visited on 2023/05/23")

        HIGH: "my daily commute is 45 minutes each way" →
        - DURATION: "45 minutes each way" (numerical_value: "45", content: "User's one-way commute time")
        - HABIT: "daily commute" (content: "User commutes 45 minutes each way daily")

        MEDIUM: "I usually order coffee from Starbucks" →
        - PREFERENCE: "usually order coffee" (content: "User's regular beverage preference")
        - ORGANIZATION: "Starbucks" (content: "User's usual coffee source")

        **Remember:**
        - Questions may ask about ANY detail from the conversations
        - Temporal questions require precise date extraction and conversion
        - Relationship questions need clear entity linking
        - Numerical questions need values WITH units
        - Recommendation questions need preferences and habits

        Extract entities now:"""
        
        return Template(template_string)
    
    def _sanitize_content(self, content: str) -> str:
        """Run sanitize content."""
        if not content:
            return content
        
        import re
        
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
                        end_idx: int) -> str:
        """Build sessions text."""
        sessions_text = ""
        
        for idx in range(start_idx, end_idx):
            session = haystack_sessions[idx]
            session_id = haystack_session_ids[idx]
            session_date = haystack_dates[idx]
            
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
                    content = content[:MAX_CONTENT_LENGTH] + "\n... [content truncated due to length]"
                
                if content and content.strip():
                    sessions_text += f"  [{msg_idx}] {role}: {content}\n"
        
        return sessions_text
    
    def _split_sessions_into_groups(self, 
                                    total_sessions: int) -> List[tuple]:
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
        logger.info(f" 生成批量推理请求 (阿里云百炼格式)")
        logger.info(f"{'='*80}")
        logger.info(f"处理范围: QA {start_index} - {end_index}")
        logger.info(f"样本数量: {len(selected_samples)}")
        logger.info(f"使用模型: {model}")
        logger.info(f"每组session数: {self.sessions_per_group}")
        logger.info(f"Thinking模式: {'启用 (budget={})'.format(thinking_budget) if enable_thinking else '禁用'}")
        logger.info(f"{'='*80}\n")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"entity_extraction_qa{start_index}_to_qa{end_index}_{timestamp}.jsonl"
        
        total_sessions = 0
        total_messages = 0
        total_groups = 0
        total_requests = 0
        
        unified_prompt_template = self._build_unified_extraction_prompt()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for idx, qa_data in enumerate(selected_samples):
                qa_index = start_index + idx
                
                question_id = qa_data.get("question_id", f"qa_{qa_index}")
                question = qa_data.get("question", "")
                question_type = qa_data.get("question_type", "unknown")
                
                haystack_sessions = qa_data.get("haystack_sessions", [])
                haystack_session_ids = qa_data.get("haystack_session_ids", [])
                haystack_dates = qa_data.get("haystack_dates", [])
                
                session_count = len(haystack_sessions)
                message_count = sum(len(session) for session in haystack_sessions)
                total_sessions += session_count
                total_messages += message_count
                
                session_groups = self._split_sessions_into_groups(session_count)
                total_groups += len(session_groups)
                
                logger.info(f"\n 处理 QA {qa_index}:")
                logger.info(f"  总session数: {session_count}")
                logger.info(f"  总消息数: {message_count}")
                logger.info(f"  分组数: {len(session_groups)}")
                
                for group_idx, (start_session_idx, end_session_idx) in enumerate(session_groups):
                    group_session_count = end_session_idx - start_session_idx
                    
                    logger.info(f"  组 {group_idx + 1}: session {start_session_idx}-{end_session_idx-1} (共{group_session_count}个)")
                    
                    try:
                        sessions_text = self._build_sessions_text(
                            haystack_sessions,
                            haystack_session_ids,
                            haystack_dates,
                            start_session_idx,
                            end_session_idx
                        )
                        
                        user_prompt = unified_prompt_template.substitute(
                            sessions_text=sessions_text,
                            entity_types_description=LongMemEvalEntityType.get_priority_description()
                        )
                        
                        messages = [
                            {
                                "role": "system",
                                "content": "You are a professional entity extraction expert. Extract entities in JSON format only."
                            },
                            {
                                "role": "user",
                                "content": user_prompt
                            }
                        ]
                        
                        custom_id = f"qa_{qa_index}_session_{start_session_idx}_{end_session_idx-1}"
                        
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
                            logger.warning(f" 检测到 {custom_id} 的JSON中包含换行符，这不应该发生")
                            # json_line = json_line.replace('\n', '\\n')
                        
                        f.write(json_line + '\n')
                        total_requests += 1
                        
                    except Exception as e:
                        logger.error(f" 处理 {custom_id} 时出错: {e}")
                        logger.error(f"   跳过此组并继续...")
                        continue
                
                if (idx + 1) % 5 == 0:
                    logger.info(f"\n 已生成 {idx + 1}/{len(selected_samples)} 个QA的请求 (共 {total_requests} 个请求)")
        
        logger.info(f"\n{'='*80}")
        logger.info(f" 批量请求文件生成完成!")
        logger.info(f"{'='*80}")
        logger.info(f"输出文件: {output_file}")
        logger.info(f"文件大小: {output_file.stat().st_size / 1024:.2f} KB")
        logger.info(f"\n 统计信息:")
        logger.info(f"  处理的QA数量: {len(selected_samples)}")
        logger.info(f"  生成的请求数: {total_requests}")
        logger.info(f"  Session组数: {total_groups}")
        logger.info(f"  总Session数: {total_sessions}")
        logger.info(f"  总消息数: {total_messages}")
        logger.info(f"\n 平均统计:")
        logger.info(f"  平均Session/QA: {total_sessions / len(selected_samples):.1f}")
        logger.info(f"  平均消息/QA: {total_messages / len(selected_samples):.1f}")
        logger.info(f"  平均请求/QA: {total_requests / len(selected_samples):.1f}")
        logger.info(f"  平均Session/组: {total_sessions / total_groups:.1f}")
        logger.info(f"{'='*80}\n")
        
        metadata = {
            "created_at": datetime.now().isoformat(),
            "dataset_path": str(self.dataset_path),
            "start_index": start_index,
            "end_index": end_index,
            "sessions_per_group": self.sessions_per_group,
            "total_qa_count": len(selected_samples),
            "total_requests": total_requests,
            "total_groups": total_groups,
            "total_sessions": total_sessions,
            "total_messages": total_messages,
            "model": model,
            "enable_thinking": enable_thinking,
            "thinking_budget": thinking_budget if enable_thinking else None,
            "batch_api": "阿里云百炼 (qwen)",
            "entity_types": LongMemEvalEntityType.get_all_types(),
            "note": "Unified prompt for all QAs - completely question-agnostic extraction. Sessions grouped to avoid context length limits. Using Template.substitute() to avoid format conflicts. 使用阿里云百炼批量推理接口格式。"
        }
        
        metadata_file = output_file.with_suffix('.json')
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f" 元数据文件: {metadata_file}")
        
        return str(output_file)

def main():
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(
        description="LongMemEval 批量实体抽取 JSONL 生成器（分组处理，阿里云百炼格式）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        示例:
        # 生成前10个QA的请求
        python step1_entity_batch_requests.py --start-index 0 --end-index 9
        
        # 生成所有500个QA的请求，每5个session一组
        python step1_entity_batch_requests.py --start-index 0 --end-index 499 --sessions-per-group 5
        
        # 生成第100-199个QA的请求
        python step1_entity_batch_requests.py --start-index 100 --end-index 199
        
        # 使用指定模型
        python step1_entity_batch_requests.py --start-index 0 --end-index 99 --model qwen-max-latest
        """
    )
    
    parser.add_argument("--dataset-path",
                       default=str(paths.LONGMEMEVAL_S_CLEANED_FILE),
                       help="数据集文件路径")
    parser.add_argument("--output-dir",
                       default=str(paths.LONGMEMEVAL_ENTITY_RELATION_NEW_BATCH_REQUESTS_DIR),
                       help="输出目录路径")
    
    parser.add_argument("--start-index", type=int, required=True,
                       help="起始QA索引 (0-499)")
    parser.add_argument("--end-index", type=int, default=None,
                       help="结束QA索引 (0-499), 默认处理到最后")
    
    parser.add_argument("--sessions-per-group", type=int, default=1,
                       help="每组处理的session数量 (默认1)")
    
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
                           "qwen3.5-plus",
                       ],
                       help="使用的模型 (默认: qwen3.5-plus)")
    
    parser.add_argument("--debug", action="store_true",
                       help="启用调试模式")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        extractor = LongMemEvalBatchEntityExtractor(
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
        print(f"  - custom_id格式: qa_X_session_Y_Z")
        print(f"  - 后续需要合并同一QA的多个请求结果")
        print(f"\n下一步:")
        print(f"1. 上传文件到阿里云百炼:")
        print(f"   参考文档: https://help.aliyun.com/document_detail/2780160.html")
        print(f"\n2. 或使用 SDK:")
        print(f"   from openai import OpenAI")
        print(f"   client = OpenAI(api_key='YOUR_KEY', base_url='https://dashscope.aliyuncs.com/compatible-mode/v1')")
        print(f"   file = client.files.create(file=open('{output_file}', 'rb'), purpose='batch')")
        print(f"\n3. 登录阿里云百炼控制台查看和管理批量任务")
        print(f"{'='*80}\n")
        
        return 0
        
    except Exception as e:
        logger.error(f" 程序异常: {e}")
        if args.debug:
            import traceback
            logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())