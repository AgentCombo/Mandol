Basic Depth: Understanding the Memory Pipeline in Three Sentences
===================================================================

Mandol works like an assistant who proactively takes notes.

Step 1: Feed It Conversations
------------------------------

You tell the system your conversation records, and it automatically understands and remembers them.

.. code-block:: python

   unit = MemoryUnit(
       uid=Uid("msg_001"),
       raw_data={"text_content": "Zhang San went to Beijing on a business trip today"},
   )
   system.add(unit)

Step 2: Let It Digest and Organize
-----------------------------------

After adding a batch of data, call the "digest" command once. The system will automatically identify topic boundaries, extract key people, events, and knowledge points.

.. code-block:: python

   system.build_high_level(mode="auto")

.. important::

   If you skip this step, the system hasn't organized the memories yet, and retrieving entities/events/summaries will return empty results. It's like someone who took notes but hasn't reviewed them — the information is in their head, but they can't quickly recall it. If you only need to retrieve raw conversations (BASE group), this step is not needed.

Step 3: Ask It Questions
-------------------------

Once the system has digested the data, you can ask questions in natural language as if asking a person.

.. code-block:: python

   hits = system.holistic_retrieve("Where did Zhang San go?", top_k=5)

   for hit in hits:
       print(f"Relevance {hit.final_score:.2f}: {hit.unit.raw_data['text_content']}")

The entire process is just these three steps: **feed data → let it digest → ask it questions**.
