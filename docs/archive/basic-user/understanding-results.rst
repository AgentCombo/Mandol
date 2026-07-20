理解检索结果
==============

每次调用 ``holistic_retrieve`` 返回的是 ``SearchHit`` 对象列表。本节解释每个字段。

SearchHit 结构
--------------

.. code-block:: python

   hit: SearchHit

   # 命中的记忆单元
   hit.unit          # MemoryUnit 对象

   # 最终综合得分（0~1）
   hit.final_score   # float，经过 Cross-Encoder 精排

   # 各检索器分项得分
   hit.scores        # {"dense": 0.92, "bm25": 0.78, "sparse": 0.65}

   # 各检索器内部排名
   hit.ranks         # {"dense": 3, "bm25": 5, "sparse": 12}

三个分数的关系
--------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - 字段
     - 含义
   * - ``scores``
     - 各检索器的**原始分数**。Dense 是余弦相似度（0~1），BM25 无上界。**不同检索器之间不能直接比较**。
   * - ``ranks``
     - 各检索器内部的**排名**（越小越好）。用于 RRF 融合，不同检索器之间可以比较。
   * - ``final_score``
     - Cross-Encoder 精排后的**最终得分**（0~1）。这是最可靠的排序依据。

常见使用模式
------------

.. code-block:: python

   hits = system.holistic_retrieve("李总", top_k=5)

   # 查看最相关的结果
   best = hits[0]
   print(f"最佳匹配: {best.unit.raw_data['text_content']}")
   print(f"综合得分: {best.final_score:.3f}")

   # 检查某条结果为什么被召回
   for hit in hits:
       if "海外市场" in hit.unit.raw_data.get("text_content", ""):
           print(f"Dense 得分: {hit.scores.get('dense', 'N/A')}")
           print(f"Dense 排名: {hit.ranks.get('dense', 'N/A')}")

   # 只看得分超过阈值的结果
   good_hits = [h for h in hits if h.final_score > 0.7]

结果为空怎么办？
----------------

检索返回空列表，最常见的原因是**忘记调用 ``build_high_level()``**。

确认你已执行：

.. code-block:: python

   system.build_high_level(mode="auto")
   hits = system.holistic_retrieve("...")

如果已调用仍返回空，请翻阅 :doc:`troubleshooting`。
