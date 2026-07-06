# mandol/auto_builder/episodic_prompts.py
"""Utilities for episodic prompts."""
from typing import Dict, Optional, List, Any
from string import Template
from dataclasses import dataclass, field



class EpisodicFactType:
    
    EVENT = "EVENT"
    STATE_CHANGE = "STATE_CHANGE"
    ACTIVITY = "ACTIVITY"
    
    PLAN = "PLAN"
    ACHIEVEMENT = "ACHIEVEMENT"
    
    RECOMMENDATION = "RECOMMENDATION"
    OPINION = "OPINION"
    PREFERENCE = "PREFERENCE"
    
    RELATIONSHIP = "RELATIONSHIP"
    POSSESSION = "POSSESSION"
    ATTRIBUTE = "ATTRIBUTE"
    
    NUMERICAL = "NUMERICAL"
    TEMPORAL_MARKER = "TEMPORAL_MARKER"
    
    ASSISTANT_KNOWLEDGE = "ASSISTANT_KNOWLEDGE"
    AGGREGATABLE_ITEM = "AGGREGATABLE_ITEM"
    IMPLICIT_CONSTRAINT = "IMPLICIT_CONSTRAINT"
    
    @classmethod
    def all_types(cls) -> List[str]:
        """Run all types."""
        return [
            cls.EVENT, cls.STATE_CHANGE, cls.ACTIVITY, cls.PLAN,
            cls.ACHIEVEMENT, cls.RECOMMENDATION, cls.OPINION, cls.PREFERENCE,
            cls.RELATIONSHIP, cls.POSSESSION, cls.ATTRIBUTE, cls.NUMERICAL,
            cls.TEMPORAL_MARKER, cls.ASSISTANT_KNOWLEDGE, 
            cls.AGGREGATABLE_ITEM, cls.IMPLICIT_CONSTRAINT
        ]
    
    @classmethod
    def locomo_types(cls) -> List[str]:
        """Run locomo types."""
        return [
            cls.EVENT, cls.STATE_CHANGE, cls.ACTIVITY, cls.PLAN,
            cls.ACHIEVEMENT, cls.RECOMMENDATION, cls.OPINION, cls.PREFERENCE,
            cls.RELATIONSHIP, cls.POSSESSION, cls.ATTRIBUTE, cls.NUMERICAL
        ]
    
    @classmethod
    def longmemeval_types(cls) -> List[str]:
        """Run longmemeval types."""
        return [
            cls.ATTRIBUTE,           # USER_ATTRIBUTE
            cls.EVENT,               # EPISODIC_EVENT
            cls.POSSESSION,          # INVENTORY_ITEM
            cls.PREFERENCE,          # PREFERENCE_HABIT
            cls.RELATIONSHIP,        # RELATIONSHIP_FACT
            cls.NUMERICAL,           # QUANTITATIVE_FACT
            cls.TEMPORAL_MARKER,     # TEMPORAL_MARKER
            cls.ASSISTANT_KNOWLEDGE, # ASSISTANT_KNOWLEDGE
            cls.AGGREGATABLE_ITEM,   # AGGREGATABLE_ITEM
            cls.IMPLICIT_CONSTRAINT  # IMPLICIT_CONSTRAINT
        ]



class EpisodicPromptTemplateManager:
    
    
    DEFAULT_EXTRACTION_PROMPT = """You are an expert fact extractor for a Question-Answering memory system. Your task is to extract ALL answerable facts from the given memory content.

## Context
- **Reference Date**: $reference_date
- **Source ID**: $source_id
- **Content Type**: $content_type

## Extraction Goals
Extract facts that can answer these question types:
1. **When** questions: "When did X happen?" → Need precise time
2. **What** questions: "What did X do/say/recommend?" → Need specific details
3. **How many** questions: "How many times did X?" → Need countable events
4. **Who** questions: "Who did X meet/help/talk to?" → Need participants
5. **Where** questions: "Where did X go?" → Need locations
6. **What kind/type** questions: "What kind of X?" → Need specific names/details

## Time Resolution Rules (CRITICAL)
- Reference Date: $reference_date
- Convert ALL relative time expressions to absolute dates
- "last Friday" + Reference 2023-11-17 → "2023-11-10"
- "next weekend" → calculate from reference date
- If time is unclear, use the reference date as default

## Content to Process
$content

## Fact Types to Extract
$fact_types_description

## Output Format (JSON)
{
    "facts": [
        {
            "content": "Complete, self-contained description of the fact. Include WHO, WHAT, WHEN context.",
            "fact_type": "$fact_type_list",
            "participants": ["Person1", "Person2"],
            "time": {
                "original_text": "last Friday",
                "absolute_date": "2023-11-10",
                "is_range": false,
                "range_start": null,
                "range_end": null,
                "is_future": false
            },
            "location": "Location name or null",
            "details": {
                "what": "action description",
                "specific_items": [],
                "numerical_value": null
            },
            "retrieval_keys": ["key1", "key2", "key3"]
        }
    ]
}

## IMPORTANT Rules
1. Extract EVERY fact that could be asked about, even small details
2. Each fact should be self-contained and understandable without context
3. For numerical facts, extract the exact number with units
4. Generate multiple retrieval_keys for each fact (synonyms, related terms)
5. Preserve specific names, brands, and details - do NOT generalize
"""

    
    
    LOCOMO_EXTRACTION_PROMPT = """You are an expert fact extractor for a Question-Answering system. Your task is to extract ALL answerable facts from this conversation session.

## Context
- **Session Date (Reference)**: $reference_date
- **Speakers**: $speakers
- **Session ID**: $source_id

## Extraction Goals
Extract facts that can answer these question types:
1. **When** questions: "When did X happen?" → Need precise time
2. **What** questions: "What did X do/say/recommend?" → Need specific details
3. **How many** questions: "How many times did X?" → Need countable events
4. **Who** questions: "Who did X meet/help/talk to?" → Need participants
5. **Where** questions: "Where did X go?" → Need locations
6. **What kind/type** questions: "What kind of food?" → Need specific names

## Time Resolution Rules (CRITICAL)
- Reference Date: $reference_date
- "last Friday" + Reference 2023-11-17 → 2023-11-10
- "Thursday before December 17" → 2023-12-14
- "towards the end of summer" → 2023-08-15 to 2023-09-01
- "next weekend" → calculate from reference date
- If time is unclear, use the session date as default

## Dialogue to Process
$content

## Output Format (JSON)
{
    "facts": [
        {
            "content": "Complete, self-contained description of the fact. Include WHO, WHAT, WHEN context. Example: 'Sam fell in love with a Canadian woman towards the end of summer 2023.'",
            "fact_type": "EVENT|STATE_CHANGE|ACTIVITY|PLAN|ACHIEVEMENT|RECOMMENDATION|OPINION|PREFERENCE|RELATIONSHIP|POSSESSION|ATTRIBUTE|NUMERICAL",
            "participants": ["Person1", "Person2"],
            "time": {
                "original_text": "towards the end of summer",
                "absolute_date": "2023-08-15",
                "is_range": true,
                "range_start": "2023-08-15",
                "range_end": "2023-09-01",
                "is_future": false
            },
            "location": "Canada (or null if unknown)",
            "details": {
                "what": "fell in love",
                "with_whom": "Canadian woman",
                "specific_items": [],
                "numerical_value": null,
                "advice_content": null
            },
            "source_turns": ["Turn 5"],
            "retrieval_keys": ["Sam love", "Canadian woman", "summer 2023", "Sam relationship"]
        }
    ]
}

## IMPORTANT Rules
1. Extract EVERY fact that could be asked about, even small details
2. For recommendations/suggestions, include the EXACT content of what was recommended
3. For food/games/books, include SPECIFIC names, not generic descriptions
4. For numerical facts (how many, how long), extract the exact number
5. Each fact should be self-contained and understandable without context
6. Generate multiple retrieval_keys for each fact (synonyms, related terms)
7. If a speaker shares something (photo, recipe), describe WHAT they shared specifically
"""

    
    
    LONGMEMEVAL_EXTRACTION_PROMPT = """You are a **High-Precision Memory Archivist** specializing in personal episodic memory preservation. Your task is to extract **Atomic Memory Facts** from the user's conversation session.

## CONVERSATION SESSION

$content

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

    
    DEDUPLICATION_PROMPT = """You are an expert in memory fact deduplication and fusion. The following are {cluster_size} semantically similar facts from a conversation memory system. Please analyze and merge them intelligently.

## Cluster {cluster_id} - Fact List:
{fact_candidates}

## Your Task:
For each group of similar facts, choose ONE fusion mode:

**Mode A - Information Merge**: Facts describe the SAME event with COMPLEMENTARY details.
  - Combine all details into one comprehensive fact
  - Example: "Jon got a job" + "to pay rent" → "Jon got a job to pay rent"

**Mode B - Frequency Count**: Facts describe the SAME repeated action on DIFFERENT dates.
  - Create one fact with count and date list
  - Example: 3x "Jon went to gym" → "Jon went to gym 3 times (dates: 2023-05-01, 2023-05-08, 2023-05-15)"

**Mode C - State Evolution**: Facts show a STATUS CHANGE over time.
  - Capture the state transition with timeline
  - Example: "Jon is unemployed" + "Jon got a job" → "Jon transitioned from unemployed to employed"

## Output (JSON):
{
    "merged_facts": [
        {
            "canonical_content": "The most complete and accurate description",
            "fact_type": "EVENT|STATE_CHANGE|ACTIVITY|...",
            "merge_mode": "A" | "B" | "C",
            "merge_count": null | number,
            "date_list": null | ["date1", "date2"],
            "state_evolution": null | {"from": "state1", "to": "state2"},
            "confidence": 0.95,
            "source_fact_indices": [1, 2],
            "merge_reasoning": "Brief explanation"
        }
    ]
}

## Critical Rules:
1. **Temporal Sensitivity**: Facts on DIFFERENT dates are usually DIFFERENT facts (unless Mode B applies)
2. **Preserve Details**: Never lose important information when merging
3. **Conservative Merging**: When in doubt, keep facts separate
4. **Participant Consistency**: Only merge facts about the SAME participants
"""

    
    FACT_TYPE_DESCRIPTIONS = {
        "default": """
### Fact Types:
- EVENT: Specific events that occurred
- STATE_CHANGE: Status transitions (e.g., got a job, moved to)
- ACTIVITY: Ongoing activities (e.g., studying, traveling)
- PLAN: Future intentions
- ACHIEVEMENT: Milestones or accomplishments
- RECOMMENDATION: Suggestions or advice given
- OPINION: Viewpoints or evaluations
- PREFERENCE: Likes, dislikes, preferences
- RELATIONSHIP: Social connections
- POSSESSION: Owned items or belongings
- ATTRIBUTE: Static characteristics
- NUMERICAL: Quantitative facts with numbers
""",
        "locomo": """
### LoCoMo Fact Types:
- EVENT: Concrete events with time/place
- STATE_CHANGE: Life changes (job, relationship, location)
- ACTIVITY: Ongoing activities
- PLAN: Future plans
- ACHIEVEMENT: Accomplishments
- RECOMMENDATION: Advice from conversation partner
- OPINION: Personal views
- PREFERENCE: Likes and habits
- RELATIONSHIP: People connections
- POSSESSION: Items owned
- ATTRIBUTE: Personal attributes
- NUMERICAL: Numbers and measurements
""",
        "longmemeval": """
### LongMemEval Memory Categories (10 Types):
1. ATTRIBUTE: User profile (education, job, identity)
2. EVENT: Specific actions at time/place
3. POSSESSION: Items owned/bought
4. PREFERENCE: Habits and preferences
5. RELATIONSHIP: Social connections
6. NUMERICAL: Numbers with context
7. TEMPORAL_MARKER: Time information
8. ASSISTANT_KNOWLEDGE: AI assistant's recommendations
9. AGGREGATABLE_ITEM: Countable items
10. IMPLICIT_CONSTRAINT: User constraints for recommendations
"""
    }
    
    def __init__(self):
        self._custom_prompts: Dict[str, str] = {}
    
    def get_extraction_prompt(self,
                             style: str = "default",
                             content: str = "",
                             reference_date: str = "",
                             source_id: str = "",
                             content_type: str = "conversation",
                             speakers: str = "",
                             fact_types: Optional[List[str]] = None,
                             custom_prompt: Optional[str] = None) -> str:
        """Return extraction prompt."""
        if custom_prompt:
            template = Template(custom_prompt)
        else:
            if style == "locomo":
                template = Template(self.LOCOMO_EXTRACTION_PROMPT)
            elif style == "longmemeval":
                template = Template(self.LONGMEMEVAL_EXTRACTION_PROMPT)
            else:
                template = Template(self.DEFAULT_EXTRACTION_PROMPT)
        
        fact_types_desc = self.FACT_TYPE_DESCRIPTIONS.get(style, self.FACT_TYPE_DESCRIPTIONS["default"])
        
        if fact_types:
            fact_type_list = "|".join(fact_types)
        else:
            fact_type_list = "|".join(EpisodicFactType.all_types())
        
        try:
            return template.safe_substitute(
                content=content,
                reference_date=reference_date or "Unknown",
                source_id=source_id or "unknown",
                content_type=content_type,
                speakers=speakers,
                fact_types_description=fact_types_desc,
                fact_type_list=fact_type_list
            )
        except Exception as e:
            return template.template if hasattr(template, 'template') else str(template)
    
    def get_deduplication_prompt(self,
                                cluster_id: int,
                                fact_candidates: str,
                                cluster_size: int,
                                custom_prompt: Optional[str] = None) -> str:
        """Return deduplication prompt."""
        values = {
            "cluster_id": str(cluster_id),
            "fact_candidates": fact_candidates,
            "cluster_size": str(cluster_size),
        }

        if custom_prompt:
            try:
                return custom_prompt.format(**values)
            except (KeyError, ValueError):
                return Template(custom_prompt).safe_substitute(**values)

        prompt = self.DEDUPLICATION_PROMPT
        for key, value in values.items():
            prompt = prompt.replace(f"{{{key}}}", value)
        return prompt
    
    def register_custom_prompt(self, name: str, prompt: str):
        """Register custom prompt."""
        self._custom_prompts[name] = prompt
    
    def get_custom_prompt(self, name: str) -> Optional[str]:
        """Return custom prompt."""
        return self._custom_prompts.get(name)
    
    def list_registered_prompts(self) -> List[str]:
        """Run list registered prompts."""
        return list(self._custom_prompts.keys())
