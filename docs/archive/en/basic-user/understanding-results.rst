Understanding Retrieval Results
================================

Each call to ``holistic_retrieve`` returns a list of ``SearchHit`` objects. This section explains each field.

SearchHit Structure
-------------------

.. code-block:: python

   hit: SearchHit

   # The matched memory unit
   hit.unit          # MemoryUnit object

   # Final composite score (0~1)
   hit.final_score   # float, after Cross-Encoder reranking

   # Individual scores from each retriever
   hit.scores        # {"dense": 0.92, "bm25": 0.78, "sparse": 0.65}

   # Internal rankings from each retriever
   hit.ranks         # {"dense": 3, "bm25": 5, "sparse": 12}

Relationship Between the Three Scores
--------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Field
     - Meaning
   * - ``scores``
     - **Raw scores** from each retriever. Dense is cosine similarity (0~1), BM25 has no upper bound. **Cannot be directly compared across different retrievers.**
   * - ``ranks``
     - **Rankings** within each retriever (lower is better). Used for RRF fusion, comparable across retrievers.
   * - ``final_score``
     - **Final score** after Cross-Encoder reranking (0~1). This is the most reliable sorting criterion.

Common Usage Patterns
---------------------

.. code-block:: python

   hits = system.holistic_retrieve("Mr. Li", top_k=5)

   # View the most relevant result
   best = hits[0]
   print(f"Best match: {best.unit.raw_data['text_content']}")
   print(f"Composite score: {best.final_score:.3f}")

   # Check why a specific result was recalled
   for hit in hits:
       if "overseas market" in hit.unit.raw_data.get("text_content", ""):
           print(f"Dense score: {hit.scores.get('dense', 'N/A')}")
           print(f"Dense rank: {hit.ranks.get('dense', 'N/A')}")

   # Filter results above a threshold
   good_hits = [h for h in hits if h.final_score > 0.7]

What If Results Are Empty?
---------------------------

The most common reason for empty retrieval results is **forgetting to call ``build_high_level()``**.

Make sure you've executed:

.. code-block:: python

   system.build_high_level(mode="auto")
   hits = system.holistic_retrieve("...")

If you've already called it and still get empty results, see :doc:`troubleshooting`.

From SearchHit to Natural Language Response
--------------------------------------------

``holistic_retrieve`` returns a structured ``SearchHit`` list. If you need a natural language response, use the ``ask()`` method:

.. code-block:: python

   answer = system.ask("Where did Zhang San go?")
   print(answer)

``ask()`` internally calls ``holistic_retrieve`` automatically, then sends the retrieval results as context to the LLM to generate a natural language response.

If you want to control the retrieval process yourself (e.g., using ``retrieve_in_space`` or ``retrieve_by_view``), you can pass the retrieval results to ``ask_with_hits()``:

.. code-block:: python

   hits = system.retrieve_by_view("Zhang San", view="entity_relation", top_k=3)
   answer = system.ask_with_hits("Where did Zhang San go?", hits)
   print(answer)

You can also customize the system prompt:

.. code-block:: python

   custom_prompt = "You are a concise assistant. Answer the question based only on the following memories, in one sentence.\n\nMemories:\n{context}"
   answer = system.ask("Where did Zhang San go?", system_prompt=custom_prompt)
