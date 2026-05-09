Reranker Configuration and Usage
===================================

Configuration
-------------

.. code-block:: yaml

   reranker:
     model: "Qwen/Qwen3-Reranker-4B"
     device: "cuda"

Usage Toggle
-------------

.. code-block:: python

   # Enable (default)
   hits = system.holistic_retrieve("...", use_rerank=True)

   # Disable (faster but less accurate)
   hits = system.holistic_retrieve("...", use_rerank=False)

Performance vs Quality
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 15 22 31 32

   * -
     - Rerank Enabled
     - Rerank Disabled
     - Recommendation
   * - Latency
     - +100-300ms
     - 0ms
     - Can disable for real-time scenarios
   * - Top-1 Accuracy
     - ~95%
     - ~85%
     - Must enable for accuracy-sensitive use
   * - GPU Usage
     - ~2-4GB
     - 0
     - Lightweight model option for CPU mode
