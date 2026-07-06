"""Utilities for multi query prompts."""

OPTIMIZED_QUERY_EXPANSION_PROMPT = """You are an expert Search Engine Optimizer and Query Engineer.
Your goal is to generate 3 specialized, complementary search queries to find missing information that the initial retrieval failed to find.

**Input Context**:
- Original Query: "{original_query}"
- Missing Information: "{missing_info}"

**Core Strategy (Hybrid Retrieval)**:
1. **The Anchor (Lexical/BM25)**: Focus on EXACT surface forms of entities, IDs, numbers, and rigid constraints (time, location).
2. **The Semantic (Dense/Vector)**: Rephrase the intent using synonyms, aliases, and descriptive language to capture meaning.
3. **The Gap-Filler (Targeted)**: A specific, open-ended question directly targeting the "{missing_info}".

**CRITICAL INSTRUCTIONS (Must Follow)**:
1. **No Pronouns**: NEVER use "he", "she", "it", "they", "his". Always replace them with the full entity name (e.g., use "Alice's project", not "her project").
2. **Date & Number Variants**: If dates/numbers exist, use different formats in different queries (e.g., "Nov 1 2025" in Query 1, "2025-11-01" in Query 2).
3. **No Boolean/Quotes**: Do not use operators like AND, OR, NOT or quotation marks.
4. **No Yes/No Questions**: Convert "Did Alice go to Paris?" -> "When and where did Alice travel?".
5. **Entity Specificity**: Repeat the exact names of people, places, and brands.
6. **Synonym Expansion**: If the query mentions "laptop", Query 2 should mention "computer" or "device".

**Output Format** (STRICT JSON):
{{
  "queries": [
    "Query 1 (High-Precision Anchor: exact names, dates, constraints)",
    "Query 2 (Semantic Expansion: synonyms, aliases, descriptive intent)",
    "Query 3 (Gap-Filler: specific question about the missing info)"
  ],
  "reasoning": "Brief explanation of how these queries handle dates, entities, and synonyms."
}}

**Examples**:

Example 1:
Original: "Did he finish the task on 10/5?"
Missing: "Task completion status for Bob"
Output:
{{
  "queries": [
    "Bob task completion status October 5th 2024",
    "Did Bob finish his assignment on 2024-10-05?", 
    "What is the status of the task Bob was working on?"
  ],
  "reasoning": "Q1 uses text-form date for BM25. Q2 uses ISO date. Q3 targets the status semantically. Pronouns replaced with 'Bob'."
}}

Example 2:
Original: "Tom's opinion on Dr. Seuss"
Missing: "Specific books Tom likes"
Output:
{{
  "queries": [
    "books in Tom's collection Dr. Seuss",
    "What children's literature authors or titles does Tom prefer?",
    "List of specific Dr. Seuss books Tom has mentioned liking"
  ],
  "reasoning": "Anchor query focuses on keywords. Semantic query broadens 'Dr. Seuss' to 'children's literature'. Gap-filler asks for a list."
}}
"""

# OPTIMIZED_QUERY_EXPANSION_PROMPT = """You are a Search Engine Optimizer.
# Your goal is to generate 3 specialized search queries to find missing information that the initial retrieval failed to find.

# **Input Context**:
# - User Query: "{original_query}"
# - Missing Information: "{missing_info}"
# - Current Status: Initial retrieval was insufficient.

# **Strategy**:
# 1. **The Anchor (Lexical)**: Focus on exact entities, names, IDs, and constraints. Good for keyword search (BM25/SPLADE).
# 2. **The Semantic (Dense)**: Rephrase the intent using synonyms and descriptive language. Good for vector search.
# 3. **The Gap-Filler (Targeted)**: A specific question directly targeting the "{missing_info}".

# **Constraints**:
# - Output STRICT JSON only.
# - No yes/no questions.
# - Keep queries under 20 words.
# - Do NOT make up facts.

# **Output Format**:
# {{
#   "queries": [
#     "Query 1 (Anchor)",
#     "Query 2 (Semantic)",
#     "Query 3 (Gap-Filler)"
#   ],
#   "reasoning": "Brief explanation of the strategy"
# }}

# **Examples**:

# Example 1:
# User Query: "Alice's hobbies"
# Missing Info: "Specific details about Alice's free time activities"
# Output:
# {{
#   "queries": [
#     "Alice hobbies interests pastimes",
#     "What does Alice enjoy doing in her leisure time?",
#     "List of activities Alice participates in"
#   ],
#   "reasoning": "Mixed keyword list for BM25 and natural language for Dense retrieval."
# }}

# Example 2:
# User Query: "Meeting on Oct 5th"
# Missing Info: "Location and participants of the meeting"
# Output:
# {{
#   "queries": [
#     "Meeting October 5th 2023 attendees location",
#     "Who participated in the meeting held on Oct 5th and where was it?",
#     "Venue and participant list for the October 5th event"
#   ],
#   "reasoning": "Targeted specific missing details (venue, attendees) while keeping date constraints."
# }}

# Now generate for:
# """




TARGETED_SEARCH_PROMPT = """You are a Precision Search Expert.
The initial retrieval failed to answer the user's query because of specific missing information.
Your goal is to generate **ONE single, highly targeted search query** solely to find that missing piece.

**Input Context**:
- Original Query: "{original_query}"
- Missing Information Identified: "{missing_info}"

**Instructions**:
1. **Target the Gap**: Do NOT search for the whole topic again. Focus ONLY on the missing part.
2. **Be Specific**: Use specific entity names (no pronouns like "he/she").
3. **Question Format**: Formulate the query as a specific question or a precise keyword string.
4. **Standalone**: The query must make sense without context.

**Examples**:

Example 1:
Original: "Alice's flight to Paris"
Missing: "Specific date of the flight"
Output:
{{
  "targeted_query": "What is the specific date of Alice's flight to Paris?",
  "reasoning": "Directly targets the missing date entity."
}}

Example 2:
Original: "Tom's opinion on Dr. Seuss"
Missing: "List of specific books mentioned"
Output:
{{
  "targeted_query": "List of Dr. Seuss books Tom likes",
  "reasoning": "Uses keywords to hit book titles associated with Tom."
}}

**Output Format** (STRICT JSON):
{{
  "targeted_query": "The generated query string",
  "reasoning": "Why this query will find the missing info"
}}
"""