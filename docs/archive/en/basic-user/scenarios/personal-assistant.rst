Personal Assistant Long-Term Memory
========================================

Remember user habits, schedules, and interpersonal relationships across multiple days.

Complete Code (Runnable)
-------------------------

Runnable example: ``examples/personal_assistant/run_personal_assistant.py``

.. code-block:: python

   from mandol import MemorySystem, MemoryUnit, Uid

   system = MemorySystem.from_yaml_config("config.yaml")

   session1 = [
       ("pa_1", "I have a project presentation with the client next Tuesday", "2024-03-11T09:00:00"),
       ("pa_2", "The presentation is about Q1 sales data analysis", "2024-03-11T09:01:00"),
   ]
   session2 = [
       ("pa_3", "I want to go hiking this weekend, recommend some nearby trails", "2024-03-16T10:00:00"),
       ("pa_4", "Xiangshan and Baiwangshan are both good, Xiangshan has better scenery but more crowded", "2024-03-16T10:01:00"),
   ]
   for uid, text, ts in session1 + session2:
       system.add(MemoryUnit(
           uid=Uid(uid),
           raw_data={"text_content": text},
           metadata={"timestamp": ts},
       ))

   system.build_high_level(mode="auto")

   hits = system.holistic_retrieve("What are my recent plans?", top_k=5)
   entities = system.retrieve_in_space("client", space_name="root_knowledge_entity", top_k=5)

   system.save("./pa_memory")

Expected Output
----------------

**holistic_retrieve("What are my recent plans?")**:

.. code-block::

   [0.923] I have a project presentation with the client next Tuesday          ← Work plan (from session 1)
   [0.876] The presentation is about Q1 sales data analysis
   [0.845] I want to go hiking this weekend, recommend some nearby trails      ← Life plan (from session 2)
   [0.812] Xiangshan and Baiwangshan are both good

The two sessions are 5 days apart; the system automatically identifies them as different sessions, but retrieval still returns comprehensive results across sessions.

**retrieve_in_space("client", space_name="root_knowledge_entity")**:

.. code-block::

   [0.901] Entity: Client - Presentation target, involving Q1 sales data analysis

Underlying Mechanism for Cross-Session Memory
-----------------------------------------------

1. **Topic identification**: The system automatically detects topic changes through LLM semantic analysis, grouping conversations by different topics
2. **Entity merging**: The same entity mentioned across different topics is automatically merged
3. **Unified retrieval**: ``holistic_retrieve`` retrieves in parallel across all four groups, naturally cross-topic
