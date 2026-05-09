Reranker 配置与使用
======================

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
