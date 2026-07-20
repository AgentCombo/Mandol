Your First Complete Memory
============================

This chapter walks you through a complete **create → add → build → retrieve → save → load** loop from scratch.

Complete Code (Runnable)
-------------------------

.. code-block:: python

   from mandol import MemorySystem, MemoryUnit, Uid

   # 1. Create system
   system = MemorySystem.from_yaml_config("config.yaml")

   # 2. Add some conversations
   conversations = [
       ("msg_1", "I have a meeting with Mr. Li next Tuesday to discuss Q2 plans", "2024-03-11T09:00:00"),
       ("msg_2", "Mr. Li wants to focus on overseas market growth data", "2024-03-11T09:01:00"),
       ("msg_3", "I want to take the kids to the science museum this weekend, any recommendations?", "2024-03-16T10:00:00"),
       ("msg_4", "The science museum has an aerospace exhibition right now, great for kids", "2024-03-16T10:01:00"),
   ]
   for uid, text, ts in conversations:
       system.add(MemoryUnit(
           uid=Uid(uid),
           raw_data={"text_content": text},
           metadata={"timestamp": ts},
       ))

   print("Added 4 memories")

   # 3. Build high-level memories
   report = system.build_high_level(mode="auto")
   print(f"Built {report.sessions_processed} sessions")

   # 4. Retrieve
   queries = [
       "What's on my schedule next week?",
       "What data does Mr. Li want to see?",
       "Where to go this weekend?",
   ]
   for q in queries:
       hits = system.holistic_retrieve(q, top_k=3)
       print(f"\nQuery: {q}")
       if hits:
           best = hits[0]
           print(f"  Most relevant: {best.unit.raw_data['text_content'][:80]}")
           print(f"  Confidence: {best.final_score:.3f}")

   # 5. Check system status
   print(system.monitor)

   # 6. Save
   system.save("./my_first_memory")

   # 7. Load and verify
   system2 = MemorySystem.load("./my_first_memory")
   hits = system2.holistic_retrieve("Mr. Li", top_k=2)
   print(f"\nRetrieved 'Mr. Li' after loading from file:")
   for hit in hits:
       print(f"  {hit.final_score:.3f} | {hit.unit.raw_data['text_content'][:80]}")

Expected Output Reference
--------------------------

.. code-block::

   Added 4 memories
   Built 2 sessions
   [MemSys] units=8 | spaces=5 | graph:12n/8e | idx:8↑/0↓ | pend:0u/0e/0et | sess:2(avg2) | mem:156.6MB | DIRTY

   Query: What's on my schedule next week?
     Most relevant: I have a meeting with Mr. Li next Tuesday to discuss Q2 plans
     Confidence: 0.923

   Query: What data does Mr. Li want to see?
     Most relevant: Mr. Li wants to focus on overseas market growth data
     Confidence: 0.951

   Query: Where to go this weekend?
     Most relevant: I want to take the kids to the science museum this weekend, any recommendations?
     Confidence: 0.912

   Retrieved 'Mr. Li' after loading from file:
     0.968 | I have a meeting with Mr. Li next Tuesday to discuss Q2 plans
     0.923 | Mr. Li wants to focus on overseas market growth data

What Happened?
---------------

1. The system detected through LLM semantic analysis that work discussion and weekend plans are different topics → split into 2 sessions
2. ``build_high_level`` extracted entities (Mr. Li, science museum), events (meeting, visit), and summaries
3. When retrieving "where to go this weekend," the system understood the semantic connection between "science museum" and "aerospace exhibition"
4. After save and load, memories are fully restored without needing to rebuild

Next Steps
----------

- :doc:`understanding-results` — Understand what each field in the retrieval results means
- :doc:`scenarios/index` — See examples for specific business scenarios
