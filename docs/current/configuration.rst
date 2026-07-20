Configuration
=============

Model presets
-------------

``SemanticMap`` reads its model registry from
``src/mandol/core/semantic_map.py``. Common presets:

.. list-table::
   :header-rows: 1

   * - Name
     - Type
     - Dim
     - Notes
   * - ``Qwen/Qwen3-Embedding-0.6B``
     - local
     - 1024
     - Default text model
   * - ``Qwen/Qwen3-Embedding-4B``
     - local
     - 2560
     - Larger text model
   * - ``Qwen/Qwen3-Embedding-8B``
     - local
     - 4096
     - Larger text model
   * - ``Qwen/Qwen3-Embedding-0.6B-remote``
     - cloud
     - 1024
     - SiliconFlow adapter
   * - ``BAAI/bge-m3`` / ``bge-m3``
     - local
     - 1024
     - BGE text model
   * - ``all-MiniLM-L6-v2``
     - local
     - 384
     - Lightweight CPU-friendly option
   * - ``jinaai/jina-clip-v2``
     - local
     - 1024
     - Text and image modalities
   * - ``jinaai/jina-embeddings-v4``
     - local
     - 2048
     - Text and image modalities

Environment variables
---------------------

Runtime settings are centralized in ``mandol.utils.config_manager.settings``.
The module reads the repository root ``.env`` file and system environment
variables.

Common keys:

* ``SILICONFLOW_API_KEY``
* ``SILICONFLOW_EMBEDDINGS_URL``
* ``SILICONFLOW_RERANK_URL``
* ``DEEPSEEK_API_KEY`` and ``DEEPSEEK_BASE_URL``
* ``CLOSEAI_API_KEY`` and ``CLOSEAI_BASE_URL``
* ``OPENAI_API_KEY`` and ``OPENAI_BASE_URL``
* ``OPENROUTER_API_KEY`` and ``OPENROUTER_BASE_URL``
* ``DASHSCOPE_API_KEY`` and ``DASHSCOPE_BASE_URL``
* ``CSTCLOUD_API_KEY`` and ``CSTCLOUD_BASE_URL``
* ``HF_TOKEN`` or ``HUGGINGFACE_TOKEN``
* ``HF_ENDPOINT`` and ``HF_HOME``
* ``RERANKER_BACKEND=native|vllm``
* ``VLLM_API_URL``, ``VLLM_API_KEY``, ``VLLM_GPU_MEMORY_UTILIZATION``,
  ``VLLM_TIMEOUT_SECONDS`` and ``VLLM_MAX_RETRIES``

Optional runtime tuning
-----------------------

``MANDOL_BM25_SPACY_MAX_PROCESSES`` limits spaCy worker processes during BM25
tokenization. ``MANDOLIN_MODEL_DIR`` selects the root directory for locally
fine-tuned quantification models. ``SKIP_AUTO_LOGGING=true`` disables Mandol's
automatic root-logging setup when another application owns logging.

Use ``env.template`` as the authoritative environment-variable template. Pass
model names, index options and runtime parameters directly to ``SemanticMap``,
``SemanticGraph``, retrieval classes and builder configuration objects.
