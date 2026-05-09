Configuration Guide
===================

.. note::

   Basic configuration has been migrated to :doc:`/basic-user/configuration-simple`. Full parameter tuning has been migrated to :doc:`/advanced-user/parameter-tuning/index`. This page will be removed in a future version, please update your bookmarks.

Mandol's configuration follows a three-level priority mechanism, flexibly supporting different deployment scenarios.

Configuration Priority
----------------------

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Priority
     - Source
     - Description
   * - 1 (highest)
     - ``.env`` environment variables
     - Sensitive information (API Keys, passwords, etc.), always overrides other sources
   * - 2
     - ``config.yaml``
     - Non-sensitive configuration (model names, devices, thresholds, etc.)
   * - 3 (lowest)
     - Code defaults
     - Built-in defaults, runs without any configuration

That is: ``.env environment variables > config.yaml > code defaults``.

Environment Variables
---------------------

All configurable environment variables in the ``.env`` file:

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Variable Name
     - Default
     - Description
   * - ``OPENAI_API_KEY``
     - (empty)
     - OpenAI API key, for LLM calls
   * - ``OPENAI_API_BASE``
     - ``https://api.openai.com/v1``
     - LLM API base URL, can be replaced with compatible interfaces
   * - ``MANDOL_LLM_MODEL``
     - ``gpt-4o-mini``
     - LLM model name
   * - ``MANDOL_LLM_TIMEOUT_S``
     - ``60``
     - LLM request timeout (seconds)
   * - ``MANDOL_EMBEDDER_MODEL``
     - ``Qwen/Qwen3-Embedding-4B``
     - Embedding model name
   * - ``MANDOL_EMBEDDER_DEVICE``
     - ``cpu``
     - Embedding device, ``cpu`` or ``cuda``
   * - ``USE_REMOTE_EMBEDDER``
     - ``false``
     - Whether to use remote Embedder service
   * - ``MANDOL_EMBEDDER_BASE_URL``
     - ``http://localhost:8000/v1``
     - Remote Embedder service URL
   * - ``MANDOL_EMBEDDER_API_PATH``
     - ``/embeddings``
     - Remote Embedder API path
   * - ``MANDOL_EMBEDDER_API_KEY``
     - (empty)
     - Remote Embedder API key
   * - ``MANDOL_RERANKER_MODEL``
     - ``Qwen/Qwen3-Reranker-4B``
     - Reranker model name
   * - ``MANDOL_RERANKER_DEVICE``
     - ``cpu``
     - Reranker device, ``cpu`` or ``cuda``
   * - ``USE_REMOTE_RERANKER``
     - ``false``
     - Whether to use remote Reranker service
   * - ``MANDOL_RERANKER_BASE_URL``
     - (empty)
     - Remote Reranker service URL
   * - ``MANDOL_RERANKER_API_PATH``
     - ``/v1/rerank``
     - Remote Reranker API path
   * - ``MANDOL_RERANKER_API_KEY``
     - (empty)
     - Remote Reranker API key

YAML Configuration
------------------

The ``config.yaml`` file is divided into five configuration sections: llm, embedder, reranker, system, and storage.

LLM Configuration
^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 15 25 40

   * - Parameter
     - Type
     - Default
     - Description
   * - ``base_url``
     - str
     - ``https://api.openai.com/v1``
     - LLM API base URL, can be replaced with any OpenAI-compatible interface
   * - ``model``
     - str
     - ``gpt-4o-mini``
     - LLM model name

Embedder Configuration
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 15 25 40

   * - Parameter
     - Type
     - Default
     - Description
   * - ``model``
     - str
     - ``Qwen/Qwen3-Embedding-4B``
     - Embedding model name (uses sentence-transformers in local mode)
   * - ``device``
     - str
     - ``cpu``
     - Inference device, ``cpu`` or ``cuda``
   * - ``dimension``
     - int
     - ``2560``
     - Embedding vector dimension
   * - ``use_remote``
     - bool
     - ``false``
     - Whether to use remote Embedder service
   * - ``base_url``
     - str
     - ``http://localhost:8000/v1``
     - Remote Embedder service URL (remote mode only)
   * - ``api_path``
     - str
     - ``/embeddings``
     - Remote Embedder API path (remote mode only)
   * - ``api_key``
     - str
     - (empty)
     - Remote Embedder API key (remote mode only, recommend setting via .env)
   * - ``timeout``
     - int
     - ``30``
     - Remote Embedder request timeout (seconds, remote mode only)

Reranker Configuration
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 15 25 40

   * - Parameter
     - Type
     - Default
     - Description
   * - ``model``
     - str
     - ``Qwen/Qwen3-Reranker-4B``
     - Reranker model name (uses CrossEncoder in local mode)
   * - ``device``
     - str
     - ``cpu``
     - Inference device, ``cpu`` or ``cuda``
   * - ``use_remote``
     - bool
     - ``false``
     - Whether to use remote Reranker service
   * - ``base_url``
     - str
     - (empty)
     - Remote Reranker service URL (remote mode only)
   * - ``api_path``
     - str
     - ``/v1/rerank``
     - Remote Reranker API path (remote mode only)
   * - ``api_key``
     - str
     - (empty)
     - Remote Reranker API key (remote mode only, recommend setting via .env)
   * - ``timeout``
     - int
     - ``30``
     - Remote Reranker request timeout (seconds, remote mode only)

System Configuration
^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 15 20 35

   * - Parameter
     - Type
     - Default
     - Description
   * - ``chunk_max_tokens``
     - int
     - ``512``
     - Maximum tokens per memory chunk
   * - ``session_time_gap_seconds``
     - int
     - ``1800``
     - Time-interval-based session segmentation threshold (seconds), i.e., 30 minutes
   * - ``session_check_interval``
     - int
     - ``20``
     - Number of memories to accumulate before triggering session boundary detection
   * - ``session_max_pending``
     - int
     - ``100``
     - Maximum pending memories, forces segmentation when exceeded
   * - ``similarity_top_k``
     - int
     - ``5``
     - Number of candidates returned by vector retrieval
   * - ``similarity_threshold``
     - float
     - ``0.7``
     - Semantic similarity edge creation threshold
   * - ``similarity_recent_window``
     - int
     - ``20``
     - Recent memory window size for similarity computation
   * - ``bfs_expansion_per_seed``
     - int
     - ``3``
     - Number of neighbors expanded per seed node during BFS expansion
   * - ``bfs_expansion_hops``
     - int
     - ``1``
     - Number of hops for BFS expansion
   * - ``max_context_units``
     - int
     - ``20``
     - Maximum context memories per LLM call
   * - ``max_entities_per_llm``
     - int
     - ``50``
     - Maximum candidates per LLM call for entity deduplication
   * - ``max_events_per_llm``
     - int
     - ``50``
     - Maximum candidates per LLM call for event deduplication
   * - ``promote_threshold``
     - int
     - ``100``
     - FAISS/BM25/TF-IDF index upgrade threshold

Storage Configuration
^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 15 25 35

   * - Parameter
     - Type
     - Default
     - Description
   * - ``root``
     - str
     - ``null``
     - Persistent storage root path, e.g., ``./data/memory``
   * - ``enable_persistence``
     - bool
     - ``false``
     - Whether to enable auto-persistence
   * - ``auto_save_interval``
     - int
     - ``300``
     - Auto-save interval (seconds)

Remote Mode vs Local Mode
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 37 38

   * - Feature
     - Remote Mode
     - Local Mode
   * - Embedding
     - Calls remote service via OpenAI-compatible API
     - Local Sentence-Transformers inference
   * - Reranker
     - Calls remote service via OpenAI-compatible API
     - Local CrossEncoder inference
   * - API Key
     - Requires ``OPENAI_API_KEY``
     - No API Key needed
   * - GPU Requirements
     - None (server-side GPU)
     - GPU recommended, CPU works but slower
   * - First Launch
     - No model download needed
     - Requires model download (~2-4 GB)
   * - Use Case
     - Production, quick experimentation
     - Offline environments, privacy-sensitive scenarios
   * - Configuration
     - ``USE_REMOTE_EMBEDDER=true`` + ``USE_REMOTE_RERANKER=true``
     - ``USE_REMOTE_EMBEDDER=false`` + ``USE_REMOTE_RERANKER=false``

Common Configuration Patterns
-----------------------------

Minimal CPU Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^

For quick experimentation, all models running on CPU:

.. code-block:: yaml

   llm:
     base_url: "https://api.openai.com/v1"
     model: "gpt-4o-mini"

   embedder:
     model: "Qwen/Qwen3-Embedding-4B"
     device: "cpu"
     dimension: 2560
     use_remote: false

   reranker:
     model: "Qwen/Qwen3-Reranker-4B"
     device: "cpu"
     use_remote: false

   storage:
     root: null
     enable_persistence: false

   system:
     chunk_max_tokens: 512
     session_time_gap_seconds: 1800
     similarity_top_k: 5
     similarity_threshold: 0.7

GPU Acceleration Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For development environments with GPU, local models using CUDA acceleration:

.. code-block:: yaml

   llm:
     base_url: "https://api.openai.com/v1"
     model: "gpt-4o-mini"

   embedder:
     model: "Qwen/Qwen3-Embedding-4B"
     device: "cuda"
     dimension: 2560
     use_remote: false

   reranker:
     model: "Qwen/Qwen3-Reranker-4B"
     device: "cuda"
     use_remote: false

   storage:
     root: "./data/memory"
     enable_persistence: true
     auto_save_interval: 300

   system:
     chunk_max_tokens: 512
     session_time_gap_seconds: 1800
     similarity_top_k: 5
     similarity_threshold: 0.7

Remote API Configuration
^^^^^^^^^^^^^^^^^^^^^^^^

For using remote API services:

.. code-block:: yaml

   llm:
     base_url: "https://api.openai.com/v1"
     model: "gpt-4o-mini"

   embedder:
     model: "Qwen/Qwen3-Embedding-4B"
     device: "cpu"
     dimension: 2560
     use_remote: true
     base_url: "http://localhost:8000/v1"
     api_path: "/embeddings"
     timeout: 30

   reranker:
     model: "Qwen/Qwen3-Reranker-4B"
     device: "cpu"
     use_remote: true
     base_url: "https://your-reranker-api-endpoint.com"
     api_path: "/v1/rerank"
     timeout: 30

   storage:
     root: "./data/memory"
     enable_persistence: true
     auto_save_interval: 300

   system:
     chunk_max_tokens: 512
     session_time_gap_seconds: 1800
     session_check_interval: 20
     session_max_pending: 100
     similarity_top_k: 5
     similarity_threshold: 0.7
     similarity_recent_window: 20
     bfs_expansion_per_seed: 3
     bfs_expansion_hops: 1
     max_context_units: 20
     max_entities_per_llm: 50
     max_events_per_llm: 50
     promote_threshold: 100

Corresponding ``.env`` file:

.. code-block:: bash

   OPENAI_API_KEY=sk-your-api-key-here
   USE_REMOTE_EMBEDDER=true
   MANDOL_EMBEDDER_BASE_URL=http://localhost:8000/v1
   MANDOL_EMBEDDER_API_KEY=your-embedder-api-key
   USE_REMOTE_RERANKER=true
   MANDOL_RERANKER_BASE_URL=https://your-reranker-api-endpoint.com
   MANDOL_RERANKER_API_KEY=your-reranker-api-key
