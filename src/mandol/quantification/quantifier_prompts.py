
"""
Quantifier Prompts for LoCoMo System
"""



FAST_BINARY_PROMPT = """Task: Determine if the provided Documents contain the specific answer to the Query.
Output ONLY 'T' (for True/Sufficient) or 'F' (for False/Insufficient).

Examples:
Query: "What is the capital of France?" | Doc: "Paris is the capital of France." -> T
Query: "Who is the CEO?" | Doc: "The company was founded in 1990." -> F
Query: "Does Alice have a dog?" | Doc: "Alice mentions she is a cat person and doesn't have time for a dog." -> T
Query: "What is Bob's phone number?" | Doc: "Bob works as an engineer in Seattle." -> F
Query: "What did Sarah say about the meeting?" | Doc: "Sarah: The meeting was postponed to next Monday." -> T

Now evaluate:
Query: "{query}"
Documents:
{context}

Output (T or F):"""




# QUANTIFICATION_PROMPT = """You are a Pragmatic Information Retrieval Evaluator for the LoCoMo memory system.
# Your task is to assess whether the retrieved memory context contains ENOUGH information to reasonably answer the user's query.

# **User Query**:
# {query}

# **Retrieved Memory Context**:
# {context}

# **Evaluation Instructions (Chain of Thought)**:
# Please follow these steps to ensure a balanced and efficient evaluation:

# 1.  **Identify Core Intent**: What is the *main* thing the user wants to know? Distinguish between the "Must-Have" information and "Nice-to-Have" details.
# 2.  **Scan for Semantic Coverage**: Check if the context covers the gist of the question.
#     * **L0/L1/L2**: Information in summaries (L1) or insights (L2) is treated as FACTUAL and sufficient; raw dialogue is not always required.
# 3.  **Apply Pragmatic Judgment (CRITICAL)**:
#     * **Reasonable Inference**: If the answer isn't stated verbatim but can be easily inferred or synthesized from the context, mark as **SUFFICIENT**.
#     * **Avoid Pedantry**: Do NOT mark as MISSING for minor details (e.g., exact timestamps, middle names, specific adjectives) unless the user *explicitly* asked for them (e.g., "What implies..." vs "What is the exact date...").
#     * **Implicit Negation**: If the context describes a situation where the target entity is logically absent, count this as answering the query (the answer is "None").
# 4.  **Sufficiency Judgment**:
#     * **SUFFICIENT**: The user can form a coherent, helpful answer based on these documents. The core intent is satisfied.
#     * **MISSING**: Only if the *central* piece of information is completely absent, making it impossible to answer the main question.

# **Few-Shot Examples**:

# Example 1 (Strong Match - L0 + Graph):
# Query: "Where did Calvin and Dave meet Frank Ocean to start collaborating?"
# Context:
# [Doc 1] Entity: 2 August 2023... Time period when Calvin and Frank Ocean met at a festival...
# [Doc 2] Entity: Frank Ocean... met at a music festival in Tokyo...
# [Doc 6] Entity: Tokyo (Type: LOCATION)... City where Calvin met Frank Ocean...
# Response:
# {{
#   "thought_process": "1. Intent: The user wants to know the specific location where Calvin, Dave, and Frank Ocean met to start collaborating. Key entities: Calvin, Dave, Frank Ocean. 2. Evidence: Doc 2 mentions Frank Ocean met at a music festival in Tokyo. Doc 6 confirms Tokyo is the city where Calvin met Frank Ocean. The location 'music festival in Tokyo' is explicitly present. 3. Implicit Negation: Not applicable as positive evidence exists. 4. Conclusion: The documents provide the specific location (Tokyo, music festival), so the information is sufficient.",
#   "status": "SUFFICIENT",
#   "missing_info": ""
# }}

# Example 2 (Simple Dialogue - L0):
# Query: "What does Jon tell Gina he won't do?"
# Context:
# [Doc 1] Jon: Sure thing, Gina! Your help means a lot to me. I'm not giving up.
# Response:
# {{
#   "thought_process": "1. Intent: The user is asking for a specific negative statement made by Jon to Gina ('what he won't do'). 2. Evidence: In Doc 1 (L0), Jon explicitly says 'I'm not giving up'. This directly answers what he will not do (give up). 3. Implicit Negation: Not needed, direct quote found. 4. Conclusion: The dialogue contains the exact answer.",
#   "status": "SUFFICIENT",
#   "missing_info": ""
# }}

# Example 3 (Partial Missing - False):
# Query: "What specific date did Alice fly to Paris?"
# Context:
# [Doc 1] Summary: Alice discussed her upcoming trip to Paris planned for next month.
# [Doc 2] Entity: Trip to Paris (Type: EVENT) | Status: Planning phase.
# [Doc 3] Alice: I'm so excited about going to France!
# Response:
# {{
#   "thought_process": "1. Intent: The user needs the 'specific date' of Alice's flight to Paris. 2. Evidence: Doc 1 (Summary) mentions the trip is 'planned for next month'. Doc 2 (Entity) says status is 'Planning phase'. Doc 3 is an expression of excitement. None of the layers contain a specific calendar date (e.g., 'October 12th'). 3. Implicit Negation: The documents do not imply a date doesn't exist; they just don't contain it yet. 4. Conclusion: The specific date is critical to the query but absent in the retrieved context.",
#   "status": "MISSING",
#   "missing_info": "Specific date of the flight"
# }}

# Example 4 (Summary as Evidence - True):
# Query: "How did the user feel about the project outcome?"
# Context:
# [Doc 1] Summary: The session concluded with the user expressing high satisfaction and relief regarding the project's completion.
# [Doc 2] Entity: Project X (Type: WORK) | Status: Completed.
# Response:
# {{
#   "thought_process": "1. Intent: The user wants to know the user's emotional state regarding the project outcome. 2. Evidence: Doc 1 is an L1 Summary stating the user expressed 'high satisfaction and relief'. According to the instructions, L1 Summaries are factual evidence. 3. Implicit Negation: N/A. 4. Conclusion: The summary directly answers the question about feelings, so it is sufficient even without the raw dialogue.",
#   "status": "SUFFICIENT",
#   "missing_info": ""
# }}

# Example 5 (Implicit Negation - True):
# Query: "What is the name of Caroline's dog?"
# Context:
# [Doc 1] Entity: Luna (Type: PET) | Context: Caroline's cat.
# [Doc 2] Summary: Caroline talked about her life living alone with her cat, mentioning she is more of a cat person than a dog person.
# [Doc 3] Caroline: I don't think I have time for a dog right now.
# Response:
# {{
#   "thought_process": "1. Intent: The user is asking for the name of Caroline's dog. 2. Evidence: Doc 1 mentions a pet 'Luna' but clarifies it is a cat. Doc 2 (Summary) says she is a 'cat person'. Doc 3 (L0) explicitly states 'I don't think I have time for a dog right now'. 3. Implicit Negation: The evidence strongly implies the non-existence of a dog. The answer to 'What is the name of her dog?' is 'She does not have one'. 4. Conclusion: The absence of the entity is confirmed by the context, making the context sufficient to answer the query negatively.",
#   "status": "SUFFICIENT",
#   "missing_info": ""
# }}

# **Output Format**:
# Respond in strict JSON format:
# {{
#   "thought_process": "Step-by-step analysis. 1. Intent... 2. Evidence... 3. Pragmatic check... 4. Conclusion.",
#   "status": "SUFFICIENT" or "MISSING",
#   "missing_info": "Specific missing details (only if status is MISSING)"
# }}
# """

"""
Quantifier Prompts for LoCoMo System
"""

QUANTIFICATION_PROMPT = """You are a meticulous Information Retrieval Evaluator for the LoCoMo memory system.
Your task is to critically assess whether the retrieved memory context contains SUFFICIENT information to answer the user's query.

**User Query**:
{query}

**Retrieved Memory Context**:
{context}

**Evaluation Instructions (Chain of Thought)**:
Please follow these steps to ensure an accurate evaluation:

1.  **Deconstruct Intent**: Break down the user's query into core informational needs (Who, What, When, Where, Why) and identify key entities.
2.  **Scan All Layers**: Systematically check for evidence in all memory layers:
    *   **L0 (Observations)**: Specific dialogue details.
    *   **L1 (Summaries)**: Session-level summaries. **CRITICAL**: Information in summaries is considered FACTUAL evidence. You do NOT need raw dialogue if the summary explicitly answers the question.
    *   **L2 (Insights)**: High-level patterns/attributes. Valid for abstract questions.
    *   **Graph/Episodic**: Entity relations and specific events.
3.  **Check for Implicit Negation**:
    *   If the context provides a complete list, timeline, or description where the target entity/event is absent, this counts as **SUFFICIENT** (the answer is "Not found", "None", or "Did not happen").
    *   Do NOT mark as MISSING just because the specific item isn't there, if the context implies it *should* be there if it existed.
4.  **Sufficiency Judgment**:
    *   **SUFFICIENT**: If the query can be directly answered (including negative answers) based on the evidence.
    *   **MISSING**: If critical information is completely absent or ambiguous across all layers.

**Few-Shot Examples**:

Example 1 (Strong Match - L0 + Graph):
Query: "Where did Calvin and Dave meet Frank Ocean to start collaborating?"
Context:
[Doc 1] Entity: 2 August 2023... Time period when Calvin and Frank Ocean met at a festival...
[Doc 2] Entity: Frank Ocean... met at a music festival in Tokyo...
[Doc 6] Entity: Tokyo (Type: LOCATION)... City where Calvin met Frank Ocean...
Response:
{{
  "thought_process": "1. Intent: The user wants to know the specific location where Calvin, Dave, and Frank Ocean met to start collaborating. Key entities: Calvin, Dave, Frank Ocean. 2. Evidence: Doc 2 mentions Frank Ocean met at a music festival in Tokyo. Doc 6 confirms Tokyo is the city where Calvin met Frank Ocean. The location 'music festival in Tokyo' is explicitly present. 3. Implicit Negation: Not applicable as positive evidence exists. 4. Conclusion: The documents provide the specific location (Tokyo, music festival), so the information is sufficient.",
  "status": "SUFFICIENT",
  "missing_info": ""
}}

Example 2 (Simple Dialogue - L0):
Query: "What does Jon tell Gina he won't do?"
Context:
[Doc 1] Jon: Sure thing, Gina! Your help means a lot to me. I'm not giving up.
Response:
{{
  "thought_process": "1. Intent: The user is asking for a specific negative statement made by Jon to Gina ('what he won't do'). 2. Evidence: In Doc 1 (L0), Jon explicitly says 'I'm not giving up'. This directly answers what he will not do (give up). 3. Implicit Negation: Not needed, direct quote found. 4. Conclusion: The dialogue contains the exact answer.",
  "status": "SUFFICIENT",
  "missing_info": ""
}}

Example 3 (Partial Missing - False):
Query: "What specific date did Alice fly to Paris?"
Context:
[Doc 1] Summary: Alice discussed her upcoming trip to Paris planned for next month.
[Doc 2] Entity: Trip to Paris (Type: EVENT) | Status: Planning phase.
[Doc 3] Alice: I'm so excited about going to France!
Response:
{{
  "thought_process": "1. Intent: The user needs the 'specific date' of Alice's flight to Paris. 2. Evidence: Doc 1 (Summary) mentions the trip is 'planned for next month'. Doc 2 (Entity) says status is 'Planning phase'. Doc 3 is an expression of excitement. None of the layers contain a specific calendar date (e.g., 'October 12th'). 3. Implicit Negation: The documents do not imply a date doesn't exist; they just don't contain it yet. 4. Conclusion: The specific date is critical to the query but absent in the retrieved context.",
  "status": "MISSING",
  "missing_info": "Specific date of the flight"
}}

Example 4 (Summary as Evidence - True):
Query: "How did the user feel about the project outcome?"
Context:
[Doc 1] Summary: The session concluded with the user expressing high satisfaction and relief regarding the project's completion.
[Doc 2] Entity: Project X (Type: WORK) | Status: Completed.
Response:
{{
  "thought_process": "1. Intent: The user wants to know the user's emotional state regarding the project outcome. 2. Evidence: Doc 1 is an L1 Summary stating the user expressed 'high satisfaction and relief'. According to the instructions, L1 Summaries are factual evidence. 3. Implicit Negation: N/A. 4. Conclusion: The summary directly answers the question about feelings, so it is sufficient even without the raw dialogue.",
  "status": "SUFFICIENT",
  "missing_info": ""
}}

Example 5 (Implicit Negation - True):
Query: "What is the name of Caroline's dog?"
Context:
[Doc 1] Entity: Luna (Type: PET) | Context: Caroline's cat.
[Doc 2] Summary: Caroline talked about her life living alone with her cat, mentioning she is more of a cat person than a dog person.
[Doc 3] Caroline: I don't think I have time for a dog right now.
Response:
{{
  "thought_process": "1. Intent: The user is asking for the name of Caroline's dog. 2. Evidence: Doc 1 mentions a pet 'Luna' but clarifies it is a cat. Doc 2 (Summary) says she is a 'cat person'. Doc 3 (L0) explicitly states 'I don't think I have time for a dog right now'. 3. Implicit Negation: The evidence strongly implies the non-existence of a dog. The answer to 'What is the name of her dog?' is 'She does not have one'. 4. Conclusion: The absence of the entity is confirmed by the context, making the context sufficient to answer the query negatively.",
  "status": "SUFFICIENT",
  "missing_info": ""
}}

**Output Format**:
Respond in strict JSON format:
{{
  "thought_process": "Step-by-step analysis. 1. Intent... 2. Evidence in L1/L2/Graph... 3. Implicit negation check... 4. Conclusion.",
  "status": "SUFFICIENT" or "MISSING",
  "missing_info": "Specific missing details (only if status is MISSING)"
}}
"""
