Index Parameters
==================

promote_threshold
-------------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Default
     - Description
   * - ``promote_threshold``
     - 100
     - Upgrades from brute-force search to index at this count

- < 100 units: Brute-force search (exact but slower)
- >= 100 units: FAISS/BM25/TF-IDF index

LLM Call Parameters
---------------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
   * - ``max_entities_per_llm``
     - 50
     - Maximum candidates per LLM call for entity deduplication
   * - ``max_events_per_llm``
     - 50
     - Maximum candidates per LLM call for event deduplication

- Increase (100-200): More thorough deduplication but higher cost
- Decrease (20-30): Saves money but may miss duplicates

Flush and Rebuild
-------------------

.. code-block:: python

   system.flush()                               # Flush cache to disk
   system.semantic_map.rebuild_index_from_store()  # Rebuild index from store
