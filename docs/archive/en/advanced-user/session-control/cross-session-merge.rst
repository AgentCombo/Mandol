Cross-Session Merging
=======================

``build_high_level()`` internally triggers cross-session entity and event merging automatically.

Entity Merging
--------------

.. code-block:: python

   # Automatically called (within build_high_level)
   # Manual call (for debugging):
   merged = system.semantic_graph._cross_session_entity_merge()

Merge detection logic: LLM determines whether two entity names refer to the same concept. For example, "Zhang San", "Boss Zhang", "Old Zhang" → merged into "Zhang San".

Event Merging
-------------

.. code-block:: python

   merged = system.semantic_graph._cross_session_event_merge()

Merge detection logic: High event similarity + close timing → merged as the same event.

Verifying Merge Results
------------------------

.. code-block:: python

   # Before merging: query returns multiple results for same-named entities
   # After merging: same-named entities are unified
   hits = system.retrieve_in_space("Zhang San", space_name="root_knowledge_entity")
   for hit in hits:
       print(hit.unit.raw_data.get("text_content", ""))

Tuning Merge Parameters
-------------------------

.. code-block:: yaml

   system:
     max_entities_per_llm: 50     # Candidates per LLM dedup call
     max_events_per_llm: 50      # Same (for events)

Increasing these values → more thorough deduplication but higher LLM costs.
