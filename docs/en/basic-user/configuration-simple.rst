Minimal Configuration
=====================

For 90% of basic users, you only need to care about 4 configurations.

Configuration 1: API Key
-------------------------

Write it in the ``.env`` file:

.. code-block:: bash

   OPENAI_API_KEY=sk-your-key-here

Configuration 2: Model Selection
---------------------------------

Write it in ``config.yaml``:

.. code-block:: yaml

   llm:
     model: "gpt-4o-mini"          # LLM model (for session segmentation, entity/event extraction)
     base_url: "https://api.openai.com/v1"

   embedder:
     model: "Qwen/Qwen3-Embedding-4B"   # Embedding model
     device: "cpu"                       # or "cuda"

   reranker:
     model: "Qwen/Qwen3-Reranker-4B"    # Reranker model (improves retrieval precision)
     device: "cpu"                       # or "cuda"

.. tip::

   The Embedder converts text to vectors for retrieval, and the Reranker fine-tunes retrieval results. Both support local models or remote API modes. If GPU memory is limited, consider using a local model for the Embedder and a remote API for the Reranker.

Configuration 3: Chunk Size
---------------------------

Controls how long each text segment is before processing:

.. code-block:: yaml

   system:
     chunk_max_tokens: 512    # Default value, best results in practice

Configuration 4: Remote API Toggle
-----------------------------------

If you don't want to download local models, you can use remote APIs:

.. code-block:: yaml

   embedder:
     use_remote: true
     base_url: "http://your-api-endpoint/v1"
     api_path: "/embeddings"
     api_key: "your-key"

   reranker:
     use_remote: true
     base_url: "http://your-api-endpoint"
     api_path: "/v1/rerank"
     api_key: "your-key"

Other Configurations You Don't Need to Change
----------------------------------------------

The following configurations work with their default values for most scenarios:

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Configuration
     - Default
     - When to change
   * - ``similarity_threshold``
     - 0.7
     - Increase to 0.8 for more precise similarity matching
   * - ``bfs_expansion_hops``
     - 1
     - Set to 2 for more graph expansion results
   * - ``session_check_interval``
     - 20
     - Controls how often LLM topic boundary detection is triggered

For the full configuration reference, see :doc:`/advanced-user/parameter-tuning/index`.

Using Preset Configurations
----------------------------

If you don't want to write YAML by hand, you can use presets directly:

.. code-block:: python

   from mandol import MemorySystem

   # No arguments = all defaults (local model mode)
   system = MemorySystem()

   # Load from YAML file
   system = MemorySystem.from_yaml_config("config.yaml")
