# mandol/auto_builder/entity_relation_prompts.py
"""Utilities for entity relation prompts."""
from typing import Dict, Optional, List, Any
from string import Template
import json



class EntityType:
    
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
    EMOTION = "EMOTION"
    GOAL = "GOAL"
    
    @classmethod
    def all_types(cls) -> List[str]:
        """Run all types."""
        return [
            cls.PERSON, cls.ORGANIZATION, cls.PRODUCT,
            cls.DATE_TIME, cls.TEMPORAL_REFERENCE, cls.LOCATION,
            cls.NUMERICAL_VALUE, cls.DURATION, cls.MEASUREMENT,
            cls.EVENT, cls.ACTIVITY,
            cls.RELATIONSHIP, cls.OBJECT, cls.ATTRIBUTE,
            cls.PREFERENCE, cls.HABIT,
            cls.SKILL, cls.EDUCATION, cls.OCCUPATION,
            cls.CONCEPT, cls.EMOTION, cls.GOAL
        ]
    
    @classmethod
    def locomo_types(cls) -> List[str]:
        """Run locomo types."""
        return [
            cls.PERSON, cls.ORGANIZATION,
            cls.EVENT, cls.ACTIVITY,
            cls.CONCEPT, cls.EMOTION,
            cls.LOCATION, cls.DATE_TIME, cls.NUMERICAL_VALUE,
            cls.OBJECT, cls.SKILL, cls.RELATIONSHIP, cls.GOAL
        ]
    
    @classmethod
    def longmemeval_types(cls) -> List[str]:
        """Run longmemeval types."""
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



class EntityRelationPromptManager:
    
    
    DEFAULT_ENTITY_EXTRACTION_PROMPT = """You are a professional entity extraction expert for question-answering systems. Your task is to extract ALL important entities from the given content.

## Context
- **Reference Date**: $reference_date
- **Source ID**: $source_id
- **Content Type**: $content_type

## Content to Process
$content

## Entity Type Schema
$entity_types_description

## Extraction Guidelines

### CRITICAL: Temporal Conversion
- Convert ALL relative time references to absolute dates using the reference date
- "last Friday" + Reference 2023-11-17 → "2023-11-10"
- Extract BOTH the relative expression AND the absolute conversion

### Numerical Values
- ALWAYS include units with numbers
- "$800" → NUMERICAL_VALUE: "800 dollars"
- "45 minutes" → DURATION: "45 minutes"

### Relationships
- Extract relationship references: "my sister Emily", "my friend Rachel"
- Create separate entities for PERSON and RELATIONSHIP

### Standardization
- Use consistent naming: "John Smith" not "john", "Smith"
- Standardize locations: "University of California, Los Angeles (UCLA)"

## Output Format (JSON)
{
    "entities": [
        {
            "entity_id": "E1",
            "name": "Standardized entity name",
            "type": "ENTITY_TYPE",
            "content": "Rich description with context",
            "temporal_info": "Absolute date if applicable",
            "spatial_info": "Location if applicable",
            "aliases": ["alternative names"],
            "confidence": 0.95
        }
    ]
}
"""

    
    
    LOCOMO_ENTITY_EXTRACTION_PROMPT = """You are a professional entity recognition expert specializing in multi-hop question answering tasks.

**Session Information:**
- Session ID: $source_id
- Session Time: $reference_date
- Speakers: $speakers

**Conversation Text:**
$content

**Entity Type Schema:**
$entity_types_description

**Task Requirements:**
1. Extract entities crucial for answering Who/What/When/Where/Why/How questions
2. Pay special attention to LOCATION, DATE_TIME, and NUMERICAL_VALUE entities
3. Standardize entity names (convert relative time to absolute when possible)
4. Provide rich content descriptions for each entity

**Output Format (JSON):**
{
    "entities": [
        {
            "entity_id": "E1",
            "name": "Standardized entity name",
            "content": "Rich description of the entity with context",
            "type": "ENTITY_TYPE",
            "confidence": 0.95,
            "temporal_info": "time information if applicable",
            "spatial_info": "location information if applicable",
            "aliases": ["alternative names or mentions"]
        }
    ]
}

**Critical Guidelines:**
- For DATE_TIME: Convert relative time to absolute dates using session time
- For LOCATION: Extract specific geographic references, venue names
- For NUMERICAL_VALUE: Include meaningful numbers with context
- For PERSON: Include both explicit names and role-based references
- For EVENT: Focus on specific occurrences with participants, time, location
"""

    
    
    LONGMEMEVAL_ENTITY_EXTRACTION_PROMPT = """You are a professional entity extraction expert specializing in conversational data analysis for question-answering systems.

**Task:** Extract ALL important entities from the given conversation sessions that could be used to answer various types of questions about the conversations, including factual questions, temporal reasoning, preferences, and recommendations.

**Conversation Sessions:**
$content

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
- "$$800" -> NUMERICAL_VALUE with content "800 dollars"
- "45 minutes" -> DURATION with content "45 minutes"
- "500 Mbps" -> NUMERICAL_VALUE with content "500 Mbps"

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

HIGH: "my sister Emily moved to Denver" ->
- PERSON: "Emily" (content: "User's sister, moved to Denver")
- RELATIONSHIP: "sister" (content: "Emily is the user's sister")
- LOCATION: "Denver" (content: "City where Emily moved to")

HIGH: "last Tuesday I went to Target" (session date: 2023/05/30 Tue) ->
- TEMPORAL_REFERENCE: "last Tuesday" (temporal_info: "2023/05/23 (Tue)")
- DATE_TIME: "2023/05/23 (Tue)" (content: "Date when user went to Target")
- ORGANIZATION: "Target" (content: "Store visited on 2023/05/23")

HIGH: "my daily commute is 45 minutes each way" ->
- DURATION: "45 minutes each way" (numerical_value: "45", content: "User's one-way commute time")
- HABIT: "daily commute" (content: "User commutes 45 minutes each way daily")

MEDIUM: "I usually order coffee from Starbucks" ->
- PREFERENCE: "usually order coffee" (content: "User's regular beverage preference")
- ORGANIZATION: "Starbucks" (content: "User's usual coffee source")

**Remember:**
- Questions may ask about ANY detail from the conversations
- Temporal questions require precise date extraction and conversion
- Relationship questions need clear entity linking
- Numerical questions need values WITH units
- Recommendation questions need preferences and habits

Extract entities now:"""

    
    ENTITY_DEDUPLICATION_PROMPT = """You are an expert in entity standardization and deduplication. The following are {cluster_size} similar entities grouped by semantic clustering. Please merge duplicates while preserving all session-specific information.

## Cluster {cluster_id} - Entity List:
{entity_candidates}

## Your Task:
Analyze these entities and determine if they should be merged:
- **Merge if**: They refer to the SAME real-world object/person/place
- **Keep separate if**: They are truly different entities despite similar names

## Output Format (JSON):
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
                    "content": "Session-specific description",
                    "temporal_info": "time info from this session",
                    "spatial_info": "location info from this session",
                    "aliases": ["session-specific aliases"],
                    "confidence": 0.90
                }}
            ],
            "merge_reasoning": "Why these entities were merged"
        }}
    ]
}}

## Critical Rules:
1. **Preserve ALL session contexts**: Each mention retains original session info
2. **Maintain temporal accuracy**: Keep different temporal_info if they differ
3. **Standard naming**: Choose the most complete and clear canonical name
4. **Avoid over-merging**: Only merge truly identical real-world objects
5. **Confidence calculation**: Use weighted average of original scores
"""

    
    
    ENTITY_MERGING_PROMPT = """你是一个实体去重和合并专家。请分析以下实体是否指向同一个真实世界对象，如果是则合并它们。

实体列表：
{entities}

要求：
1. 判断这些实体是否应该合并
2. 如果合并，提供统一的标准名称
3. 整合所有别名、上下文信息
4. 选择最合适的实体类型
5. 综合考虑所有提及的置信度

请以JSON格式返回：
{{
    "should_merge": true/false,
    "merged_entity": {{
        "name": "标准化名称",
        "type": "实体类型",
        "aliases": ["所有别名"],
        "contexts": ["所有上下文"],
        "temporal_info": "综合时间信息",
        "spatial_info": "综合空间信息",
        "confidence": 平均置信度
    }},
    "reasoning": "合并理由"
}}"""

    
    DEFAULT_RELATION_EXTRACTION_PROMPT = """You are a professional relation extraction expert. Extract relationships between the given entities from the content.

## Content:
$content

## Identified Entities:
$entities

## Relation Types to Extract:
- works_at, employed_by: Employment relationships
- located_in, lives_in: Location relationships  
- knows, friend_of, family_of: Social relationships
- participated_in, attended: Event participation
- owns, possesses: Ownership relationships
- prefers, likes, dislikes: Preference relationships
- caused_by, resulted_in: Causal relationships
- happened_on, occurred_at: Temporal relationships
- Any other meaningful semantic relationship

## Output Format (JSON):
{
    "relations": [
        {
            "source_entity": "Entity name or ID",
            "relation_type": "relationship_type",
            "target_entity": "Entity name or ID",
            "context": "Context describing this relationship",
            "temporal_info": "When this relationship was valid",
            "confidence": 0.9
        }
    ]
}

## Guidelines:
1. Only extract relations between identified entities
2. Prefer specific relation types over generic ones
3. Include temporal information when available
4. Provide context for each relationship
"""

    
    
    RELATION_EXTRACTION_PROMPT = """你是一个专业的关系抽取助手。请从以下对话中抽取实体之间的关系。

对话内容：
{content}

已识别的实体：
{entities}

要求：
1. 只抽取已识别实体之间的关系
2. 每个关系需要提供：
   - source_entity: 关系的起始实体（使用实体ID或名称）
   - relation_type: 关系类型（如：works_at, located_in, participated_in, knows等）
   - target_entity: 关系的目标实体（使用实体ID或名称）
   - context: 关系的上下文描述
   - temporal_info: 时间信息（如果有）
   - confidence: 置信度（0-1之间）

请以JSON格式返回：
{{
    "relations": [
        {{
            "source_entity": "实体名称或ID",
            "relation_type": "关系类型",
            "target_entity": "实体名称或ID",
            "context": "关系上下文",
            "temporal_info": "时间信息",
            "confidence": 0.9
        }}
    ]
}}"""

    
    CROSS_SESSION_RELATION_PROMPT = """You are an expert in analyzing entity evolution across sessions. Analyze how the same entity changes or relates across different sessions.

## Entity Information:
{entity_info}

## Session Contexts:
{session_contexts}

## Task:
Identify cross-session relationships such as:
- **State evolution**: Entity's state/attribute changed over time
- **Temporal progression**: Events happened in sequence
- **Causal chain**: One session's event caused another

## Output Format (JSON):
{{
    "cross_session_relations": [
        {{
            "source_entity": "entity name",
            "relation_type": "evolved_to | caused | followed_by | replaced_by",
            "target_entity": "entity name (could be same entity in different state)",
            "context": "Description of the cross-session relationship",
            "temporal_span": "from session X to session Y",
            "confidence": 0.85
        }}
    ]
}}
"""

    
    RELATION_FILTERING_PROMPT = """You are a relation quality evaluator. Assess the following relations and filter out low-quality ones.

## Relations to Evaluate:
{relations}

## Evaluation Criteria:
1. **Semantic clarity**: Is the relation type meaningful and specific?
2. **Evidence support**: Is there sufficient context support?
3. **Confidence level**: Is the confidence score justified?
4. **Redundancy**: Is this relation duplicate or implied by others?
5. **QA relevance**: Would this relation help answer questions?

## Output Format (JSON):
{{
    "filtered_relations": [
        {{
            "relation": {{original relation object}},
            "keep": true | false,
            "reason": "Why to keep or filter",
            "adjusted_confidence": 0.85
        }}
    ]
}}
"""

    
    ENTITY_TYPE_DESCRIPTIONS = {
        "default": """
### Core Entities (High Priority for QA):
- PERSON: People mentioned (names, roles, participants)
- ORGANIZATION: Companies, institutions, groups, teams
- PRODUCT: Products, services, brands, models

### Temporal Information (Critical for When questions):
- DATE_TIME: Absolute dates and times (2023-05-30, 3pm)
- TEMPORAL_REFERENCE: Relative time expressions (last week, recently)
- DURATION: Time periods (45 minutes, 3 weeks, 5 years)

### Spatial Information (Critical for Where questions):
- LOCATION: Geographic places, venues, addresses

### Numerical Information (Critical for How many questions):
- NUMERICAL_VALUE: Numbers with context (prices, quantities, ages)
- MEASUREMENT: Measurements with units (miles, GB, calories)

### Events and Activities:
- EVENT: Specific occurrences with time/place/participants
- ACTIVITY: Ongoing activities, hobbies, practices

### Relationships and Attributes:
- RELATIONSHIP: Social connections (sister, friend, colleague)
- OBJECT: Physical items, tools, possessions
- ATTRIBUTE: Characteristics, properties, states

### Preferences and Behaviors:
- PREFERENCE: Likes, dislikes, favorites
- HABIT: Regular behaviors, routines

### Other:
- SKILL: Abilities, expertise
- CONCEPT: Abstract ideas, themes
- GOAL: Plans, intentions, objectives
- EMOTION: Feelings, moods
""",
        "locomo": """
### Core Entities:
- PERSON: Key individuals in conversations
- ORGANIZATION: Institutions, groups, companies

### Events & Activities:
- EVENT: Occurrences with time, place, participants
- ACTIVITY: Ongoing activities, hobbies

### Concepts & Themes:
- CONCEPT: Abstract ideas, themes, values
- EMOTION: Expressed feelings, moods

### Key Attributes (HIGH PRIORITY):
- LOCATION: Geographic places, venues
- DATE_TIME: Specific dates, times, durations
- NUMERICAL_VALUE: Meaningful numbers with context

### Other:
- OBJECT: Physical items, tools
- SKILL: Skills, abilities
- RELATIONSHIP: Social connections
- GOAL: Plans, objectives
""",
        "longmemeval": """
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

    11. **PREFERENCE**: User preferences and tastes
    - Examples: "favorite restaurant", "preferred workout time", "likes spicy food"

    12. **HABIT**: Regular behaviors and routines
    - Examples: "morning run", "weekly grocery shopping", "daily commute"

    13. **ACTIVITY**: Activities, hobbies, actions
    - Examples: "running", "cooking", "studying", "meeting"

    14. **EVENT**: Specific events and occasions
    - Examples: "birthday party", "job interview", "vacation"

    15. **OBJECT**: Physical items and possessions
    - Examples: "laptop", "car", "guitar", "book"

    16. **ATTRIBUTE**: Properties, characteristics, states
    - Examples: "red", "expensive", "broken", "current job"

    **LOW PRIORITY Entities:**

    17. **SKILL**: Abilities and competencies
    18. **EDUCATION**: Degrees, schools, courses
    19. **OCCUPATION**: Jobs, roles, professions
    20. **CONCEPT**: Abstract ideas, topics, themes
    21. **GOAL**: Plans, objectives, intentions
    """
    }

    
    
    ENTITY_TYPES_DESCRIPTION = """
核心实体 (Core Entities):
- PERSON: 对话中的核心人物
- ORGANIZATION: 组织机构

事件与活动 (Events & Activities):
- EVENT: 具有特定时间、地点和参与者的事件
- ACTIVITY: 个人或团体参与的日常活动或爱好

概念与主题 (Concepts & Themes):
- CONCEPT: 抽象概念或主题
- EMOTION: 明确表达的情绪或氛围

关键属性 (Key Attributes):
- LOCATION: 明确的地理位置
- DATE_TIME: 绝对的时间信息
- NUMERICAL_VALUE: 具有意义的数字

其他重要实体:
- OBJECT: 具体物品或工具
- SKILL: 技能或能力
- RELATIONSHIP: 关系描述
- GOAL: 目标或计划
"""

    
    SESSION_TYPE_CONTEXT = {
        "meeting": "This is a meeting transcript. Focus on participants, topics, decisions, and action items.",
        "chat": "This is a conversation. Focus on participants, topics, and emotional context.",
        "task": "This is a task execution log. Focus on tasks, executors, tools, and results.",
        "document": "This is document content. Focus on concepts, definitions, and knowledge.",
        "default": "This is conversational content."
    }

    def __init__(self):
        self._custom_prompts: Dict[str, str] = {}
    
    
    def get_entity_extraction_prompt_v2(self,
                                       style: str = "default",
                                       content: str = "",
                                       reference_date: str = "",
                                       source_id: str = "",
                                       content_type: str = "conversation",
                                       speakers: str = "",
                                       entity_types: Optional[List[str]] = None,
                                       session_type: str = "default",
                                       custom_prompt: Optional[str] = None) -> str:
        """Return entity extraction prompt v2."""
        if custom_prompt:
            template = Template(custom_prompt)
        else:
            if style == "locomo":
                template = Template(self.LOCOMO_ENTITY_EXTRACTION_PROMPT)
            elif style == "longmemeval":
                template = Template(self.LONGMEMEVAL_ENTITY_EXTRACTION_PROMPT)
            else:
                template = Template(self.DEFAULT_ENTITY_EXTRACTION_PROMPT)
        
        entity_types_desc = self.ENTITY_TYPE_DESCRIPTIONS.get(
            style, self.ENTITY_TYPE_DESCRIPTIONS["default"]
        )
        
        context_prefix = self.SESSION_TYPE_CONTEXT.get(session_type, "")
        if context_prefix and style == "default":
            content = f"{context_prefix}\n\n{content}"
        
        try:
            return template.safe_substitute(
                content=content,
                reference_date=reference_date or "Unknown",
                source_id=source_id or "unknown",
                content_type=content_type,
                speakers=speakers,
                entity_types_description=entity_types_desc
            )
        except Exception:
            return template.template if hasattr(template, 'template') else str(template)
    
    def get_entity_deduplication_prompt(self,
                                        cluster_id: int,
                                        entity_candidates: str,
                                        cluster_size: int,
                                        custom_prompt: Optional[str] = None) -> str:
        """Return entity deduplication prompt."""
        if custom_prompt:
            prompt_template = custom_prompt
        else:
            prompt_template = self.ENTITY_DEDUPLICATION_PROMPT
        
        return prompt_template.format(
            cluster_id=cluster_id,
            entity_candidates=entity_candidates,
            cluster_size=cluster_size
        )
    
    
    def get_relation_extraction_prompt_v2(self,
                                          content: str,
                                          entities: List[Dict],
                                          session_type: str = "default",
                                          custom_prompt: Optional[str] = None) -> str:
        """Return relation extraction prompt v2."""
        if custom_prompt:
            template = Template(custom_prompt)
        else:
            template = Template(self.DEFAULT_RELATION_EXTRACTION_PROMPT)
        
        entities_text = json.dumps(entities, indent=2, ensure_ascii=False)
        
        context_prefix = self.SESSION_TYPE_CONTEXT.get(session_type, "")
        if context_prefix:
            content = f"{context_prefix}\n\n{content}"
        
        return template.safe_substitute(
            content=content,
            entities=entities_text
        )
    
    def get_cross_session_relation_prompt_v2(self,
                                             entity_info: Dict,
                                             session_contexts: List[Dict],
                                             custom_prompt: Optional[str] = None) -> str:
        """Return cross session relation prompt v2."""
        if custom_prompt:
            prompt_template = custom_prompt
        else:
            prompt_template = self.CROSS_SESSION_RELATION_PROMPT
        
        entity_text = json.dumps(entity_info, indent=2, ensure_ascii=False)
        contexts_text = json.dumps(session_contexts, indent=2, ensure_ascii=False)
        
        return prompt_template.format(
            entity_info=entity_text,
            session_contexts=contexts_text
        )
    
    def get_relation_filtering_prompt_v2(self,
                                         relations: List[Dict],
                                         custom_prompt: Optional[str] = None) -> str:
        """Return relation filtering prompt v2."""
        if custom_prompt:
            prompt_template = custom_prompt
        else:
            prompt_template = self.RELATION_FILTERING_PROMPT
        
        relations_text = json.dumps(relations, indent=2, ensure_ascii=False)
        return prompt_template.format(relations=relations_text)
    
    
    
    @classmethod
    def get_entity_extraction_prompt(cls,
                                    content: str,
                                    session_type: str = "default",
                                    custom_prompt: Optional[str] = None) -> str:
        """Return entity extraction prompt."""
        if custom_prompt:
            return custom_prompt.format(
                content=content,
                entity_types=cls.ENTITY_TYPES_DESCRIPTION
            )
        
        context_prefix = cls.SESSION_TYPE_CONTEXT.get(session_type, "")
        if context_prefix:
            content = f"{context_prefix}\n\n{content}"
        
        
        old_prompt = """你是一个专业的实体抽取助手。请从以下对话中抽取所有重要的实体。

对话内容：
{content}

实体类型定义：
{entity_types}

要求：
1. 抽取所有符合类型定义的实体
2. 每个实体需要提供：
   - name: 实体名称（标准化）
   - type: 实体类型（从上述类型中选择）
   - aliases: 其他称呼方式（可选）
   - context: 实体在对话中的上下文
   - temporal_info: 时间信息（如果有）
   - spatial_info: 空间信息（如果有）
   - confidence: 置信度（0-1之间）

请以JSON格式返回：
{{
    "entities": [
        {{
            "name": "实体名称",
            "type": "实体类型",
            "aliases": ["别名1", "别名2"],
            "context": "上下文描述",
            "temporal_info": "时间信息",
            "spatial_info": "地点信息",
            "confidence": 0.9
        }}
    ]
}}"""
        
        return old_prompt.format(
            content=content,
            entity_types=cls.ENTITY_TYPES_DESCRIPTION
        )
    
    @classmethod
    def get_entity_merging_prompt(cls,
                                  entities: List[Dict],
                                  custom_prompt: Optional[str] = None) -> str:
        """Return entity merging prompt."""
        if custom_prompt:
            return custom_prompt.format(entities=entities)
        
        entities_text = json.dumps(entities, indent=2, ensure_ascii=False)
        return cls.ENTITY_MERGING_PROMPT.format(entities=entities_text)
    
    @classmethod
    def get_relation_extraction_prompt(cls,
                                      content: str,
                                      entities: List[Dict],
                                      session_type: str = "default",
                                      custom_prompt: Optional[str] = None) -> str:
        """Return relation extraction prompt."""
        if custom_prompt:
            return custom_prompt.format(
                content=content,
                entities=entities
            )
        
        context_prefix = cls.SESSION_TYPE_CONTEXT.get(session_type, "")
        if context_prefix:
            content = f"{context_prefix}\n\n{content}"
        
        entities_text = json.dumps(entities, indent=2, ensure_ascii=False)
        
        return cls.RELATION_EXTRACTION_PROMPT.format(
            content=content,
            entities=entities_text
        )
    
    @classmethod
    def get_cross_session_relation_prompt(cls,
                                         entity_info: Dict,
                                         session_contexts: List[Dict],
                                         custom_prompt: Optional[str] = None) -> str:
        """Return cross session relation prompt."""
        if custom_prompt:
            return custom_prompt.format(
                entity_info=entity_info,
                session_contexts=session_contexts
            )
        
        entity_text = json.dumps(entity_info, indent=2, ensure_ascii=False)
        contexts_text = json.dumps(session_contexts, indent=2, ensure_ascii=False)
        
        return cls.CROSS_SESSION_RELATION_PROMPT.format(
            entity_info=entity_text,
            session_contexts=contexts_text
        )
    
    @classmethod
    def get_relation_filtering_prompt(cls,
                                      relations: List[Dict],
                                      custom_prompt: Optional[str] = None) -> str:
        """Return relation filtering prompt."""
        if custom_prompt:
            return custom_prompt.format(relations=relations)
        
        relations_text = json.dumps(relations, indent=2, ensure_ascii=False)
        return cls.RELATION_FILTERING_PROMPT.format(relations=relations_text)
    
    
    def register_custom_prompt(self, name: str, prompt: str):
        """Register custom prompt."""
        self._custom_prompts[name] = prompt
    
    def get_custom_prompt(self, name: str) -> Optional[str]:
        """Return custom prompt."""
        return self._custom_prompts.get(name)
    
    def list_registered_prompts(self) -> List[str]:
        """Run list registered prompts."""
        return list(self._custom_prompts.keys())
