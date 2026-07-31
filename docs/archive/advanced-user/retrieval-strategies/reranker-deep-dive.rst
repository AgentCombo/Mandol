Reranker 深度解析
======================

什么是 Cross-Encoder Reranker
------------------------------

与 Dense Embedding（Bi-Encoder）不同，Cross-Encoder 将查询和文档**同时输入**一个 Transformer 模型，获得更精确的相关性判断。

::

   Bi-Encoder（召回阶段）:
   query → [Encoder] → vec_q ─┐
   doc   → [Encoder] → vec_d ─┤→ cosine(vec_q, vec_d) → 速度快但精度一般

   Cross-Encoder（精排阶段）:
   [CLS] query [SEP] doc [SEP] → [Transformer] → score → 速度慢但精度高

配置
----

.. code-block:: yaml

   reranker:
     model: "Qwen/Qwen3-Reranker-4B"
     device: "cuda"

使用开关
--------

.. code-block:: python

   # 开启（默认）
   hits = system.holistic_retrieve("...", use_rerank=True)

   # 关闭（更快但精度降低）
   hits = system.holistic_retrieve("...", use_rerank=False)

性能 vs 质量
------------

.. list-table::
   :header-rows: 1
   :widths: 15 22 31 32

   * -
     - 开启 Rerank
     - 关闭 Rerank
     - 建议
   * - 延迟
     - +100-300ms
     - 0ms
     - 实时场景可关
   * - Top-1 精度
     - ~95%
     - ~85%
     - 精度敏感必开
   * - GPU 占用
     - ~2-4GB
     - 0
     - CPU 模式可选轻量模型
