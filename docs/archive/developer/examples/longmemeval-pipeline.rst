LongMemEval 管线详解
=========================

LongMemEval 评估框架的数据适配和评估管线。

数据适配
--------

.. code-block:: python

   # 原始数据格式
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

   # 转换为 MemoryUnit
   chunks = split_passage(passage)
   units = [MemoryUnit(uid=Uid(f"chunk_{i}"), raw_data={"text_content": c}) for i, c in enumerate(chunks)]
   system.add_many(units)

评估框架
--------

.. code-block:: python

   def evaluate(system, qa_pairs):
       results = []
       for qa in qa_pairs:
           hits = system.holistic_retrieve(qa["question"], top_k=5)
           best = hits[0]
           correct = qa["answer"].lower() in best.unit.raw_data["text_content"].lower()
           results.append({**qa, "correct": correct, "score": best.final_score})
       return results

Per-category 分析
------------------

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

扩展评估框架
------------

- 添加新的评估指标（MRR, NDCG 等）
- 支持自定义相似度判断（基于 Embedding 而非子串匹配）
- 从 HuggingFace 加载完整 LongMemEval 数据集替代合成数据
