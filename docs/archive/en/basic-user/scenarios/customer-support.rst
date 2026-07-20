Customer Service Conversation Memory
========================================

Remember user order history, preferences, and complaint records to provide personalized service.

Complete Code (Runnable)
-------------------------

Runnable example: ``examples/customer_support/run_customer_support.py``

.. code-block:: python

   from mandol import MemorySystem, MemoryUnit, Uid

   system = MemorySystem.from_yaml_config("config.yaml")

   conversations = [
       ("cs_1", "I want to return the blue sneakers I bought yesterday, the size doesn't fit", "2024-03-10T14:00:00"),
       ("cs_2", "Sure, I've submitted the return request for you, refund expected in 3-5 business days", "2024-03-10T14:01:00"),
       ("cs_3", "Next time I want to buy the same model in size 42, is it in stock?", "2024-03-10T14:02:00"),
   ]
   for uid, text, ts in conversations:
       system.add(MemoryUnit(
           uid=Uid(uid),
           raw_data={"text_content": text},
           metadata={"timestamp": ts},
       ))

   system.build_high_level(mode="auto")

   # Global retrieval
   hits = system.holistic_retrieve("What shoes did this customer buy before?", top_k=3)
   # Knowledge perspective retrieval
   knowledge = system.retrieve_by_view("What does the customer like?", view="knowledge", top_k=5)
   # Event causal retrieval
   events = system.retrieve_by_view("What was the reason for the return?", view="event_causal", top_k=5)

   system.save("./cs_memory")

Expected Output
----------------

**holistic_retrieve("What shoes did this customer buy before?")**:

.. code-block::

   [0.941] I want to return the blue sneakers I bought yesterday, the size doesn't fit

**retrieve_by_view("What does the customer like?", view="knowledge")**:

.. code-block::

   [0.887] Knowledge Entity: Blue Sneakers - User prefers this brand/style, size preference 42

**retrieve_by_view("What was the reason for the return?", view="event_causal")**:

.. code-block::

   [0.912] Event Causal: Size doesn't fit → Return request

Mandol's Value in Customer Service
------------------------------------

- **Automatic entity extraction**: No manual annotation needed, automatically extracts product names, sizes from conversations
- **Event causal chains**: Automatically builds "size doesn't fit → return" causal relationships
- **Cross-session memory**: Can retrieve historical conversations when the same user consults multiple times
