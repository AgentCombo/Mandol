Knowledge Base Q&A
======================

Import documents/FAQs into the memory system, supporting semantic retrieval.

Complete Code (Runnable)
-------------------------

Runnable example: ``examples/knowledge_base/run_knowledge_base.py``

.. code-block:: python

   from mandol import MemorySystem, MemoryUnit, Uid

   system = MemorySystem.from_yaml_config("config.yaml")

   docs = [
       ("kb_1", "Company annual leave policy: 10 days after 1 year, 15 days after 5 years, 20 days after 10 years"),
       ("kb_2", "Reimbursement process: Fill form → Department manager approval → Finance review → Payment, about 5 business days"),
       ("kb_3", "Remote work policy: Maximum 2 days per week, must apply in OA system one day in advance"),
       ("kb_4", "Overtime compensation: 1.5x on weekdays, 2x on weekends, 3x on public holidays"),
   ]
   system.add_many([
       MemoryUnit(
           uid=Uid(uid),
           raw_data={"text_content": text},
           metadata={"timestamp": "2024-01-01T00:00:00", "source": "hr_policy"},
       )
       for uid, text in docs
   ])

   system.build_high_level(mode="auto")

   hits = system.holistic_retrieve("Can I work from home?", top_k=3)
   knowledge = system.retrieve_by_view("How many vacation days?", view="knowledge", top_k=3)

   system.save("./kb_memory")

Expected Output
----------------

**holistic_retrieve("Can I work from home?")**:

.. code-block::

   Most relevant: Remote work policy: Maximum 2 days per week, must apply in OA system one day in advance
   Confidence: 0.934

Even though you asked about "working from home," the system understands "working from home" = "remote work" and gives a precise match.

**retrieve_by_view("How many vacation days?", view="knowledge")**:

.. code-block::

   Most relevant: Knowledge summary: Annual leave policy - 10 days after 1 year, 15 days after 5 years, 20 days after 10 years
   Confidence: 0.918

Semantic Retrieval vs Keyword Search
--------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 37 38

   * - Feature
     - Keyword Search
     - Mandol Semantic Retrieval
   * - "work from home" matches "remote work"
     - ❌ No results
     - ✅ Precise match
   * - "overtime pay" matches "overtime compensation"
     - ❌ Requires exact wording
     - ✅ Automatic association
   * - Multi-path recall
     - Single path
     - Dense + BM25 + Sparse

Batch Import Tips
-------------------

- Use ``add_many`` for bulk document imports
- Keep default ``chunk_max_tokens: 512`` for long documents (best results in practice)
- In knowledge base scenarios, the system still segments by semantic topic changes — "session" is not limited to conversations but is a semantically coherent topic unit

Understanding "Sessions"
-------------------------

A "Session" in Mandol is not just a conversation concept. Its essence is a **semantic topic boundary**:

- In conversation scenarios, one session = one coherent conversation
- In knowledge base scenarios, one session = a group of topically related documents
- In log analysis scenarios, one session = a sequence of related events

The system automatically detects topic boundaries through LLM semantic analysis, not just time intervals. Therefore, non-conversation scenarios like knowledge bases are equally applicable.
