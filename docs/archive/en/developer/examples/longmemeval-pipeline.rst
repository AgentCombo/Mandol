LongMemEval Pipeline Details
=================================

LongMemEval evaluation framework's data adaptation and evaluation pipeline.

Data Adaptation
-----------------

.. code-block:: python

   # Raw data format
   {
       "passage": "A ML has been applied to healthcare...",
       "qa_pairs": [
           {
               "category": "Single-Session Preference",
               "question": "What was the first FDA-approved AI diagnostic tool?",
               "answer": "IDx-DR (2018)",
               "relevant_sentence": "...the FDA approved IDx-DR in 2018"
           }
       ]
   }

   # Convert to MemoryUnit
   chunks = split_passage(passage)
   units = [MemoryUnit(uid=Uid(f"chunk_{i}"), raw_data={"text_content": c}) for i, c in enumerate(chunks)]
   system.add_many(units)

Evaluation Framework
----------------------

.. code-block:: python

   def evaluate(system, qa_pairs):
       results = []
       for qa in qa_pairs:
           hits = system.holistic_retrieve(qa["question"], top_k=5)
           best = hits[0]
           correct = qa["answer"].lower() in best.unit.raw_data["text_content"].lower()
           results.append({**qa, "correct": correct, "score": best.final_score})
       return results

Per-Category Analysis
-----------------------

.. code-block:: python

   from collections import Counter
   by_cat = Counter()
   cat_correct = Counter()
   for r in results:
       cat = r["category"]
       by_cat[cat] += 1
       if r["correct"]:
           cat_correct[cat] += 1
   for cat, total in by_cat.items():
       print(f"{cat}: {cat_correct[cat]}/{total} ({100*cat_correct[cat]/total:.1f}%)")

Extending the Evaluation Framework
-------------------------------------

- Add new evaluation metrics (MRR, NDCG, etc.)
- Support custom similarity judgment (based on Embedding instead of substring matching)
- Load full LongMemEval dataset from HuggingFace instead of synthetic data
