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
* ``OPENAI_API_KEY`` and ``OPENAI_BASE_URL``
* ``DASHSCOPE_API_KEY`` and ``DASHSCOPE_BASE_URL``
* ``CSTCLOUD_API_KEY`` and ``CSTCLOUD_BASE_URL``
* ``HF_TOKEN`` or ``HUGGINGFACE_TOKEN``
* ``HF_ENDPOINT`` and ``HF_HOME``
* ``RERANKER_BACKEND=native|vllm``
* ``VLLM_API_URL``, ``VLLM_API_KEY``, ``VLLM_TIMEOUT_SECONDS``,
  ``VLLM_MAX_RETRIES``

Repository config.yaml
----------------------

The root ``config.yaml`` is retained as a reference configuration for model,
storage and system parameters. The currently maintained lower-level APIs do not
expose ``MemorySystem.from_yaml_config``; pass model names and runtime options
directly to ``SemanticMap``, ``SemanticGraph`` and retrieval components, and use
``.env`` for secrets.
