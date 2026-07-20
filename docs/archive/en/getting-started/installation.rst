Installation Guide
==================

.. note::

   This document has been migrated to :doc:`/basic-user/installation`. This page will be removed in a future version, please update your bookmarks.

Prerequisites
-------------

Before installing Mandol, ensure your environment meets the following requirements:

- **Python** 3.9 or higher
- **pip** or **conda** package manager

You can check your Python version with:

.. code-block:: bash

   python --version

Basic Installation
------------------

Install the latest stable version with pip:

.. code-block:: bash

   pip install mandol

Installing from Source
----------------------

If you need the latest development version or want to contribute, install from source:

.. code-block:: bash

   git clone https://github.com/your-org/mandol.git
   cd mandol
   pip install -e .

The ``-e`` flag installs in editable mode, so changes to the source code take effect without reinstalling.

Optional Dependencies
---------------------

Mandol follows a lean-core, extend-as-needed dependency strategy. Install optional dependency groups based on your use case:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Dependency Group
     - Install Command
     - Description
   * - faiss
     - ``pip install mandol[faiss]``
     - FAISS vector index, for large-scale vector retrieval
   * - sentence-transformers
     - ``pip install mandol[sentence-transformers]``
     - Local Embedding and Reranker models, no API Key needed
   * - openai
     - ``pip install mandol[openai]``
     - OpenAI-compatible API client, for LLM and remote Embedding
   * - milvus
     - ``pip install mandol[milvus]``
     - Milvus vector database, for production persistent storage
   * - neo4j
     - ``pip install mandol[neo4j]``
     - Neo4j graph database, for large-scale semantic graph storage
   * - all
     - ``pip install mandol[all]``
     - Install all optional dependencies
   * - dev
     - ``pip install mandol[dev]``
     - Development tools (pytest, pre-commit, ruff, etc.)
   * - docs
     - ``pip install mandol[docs]``
     - Documentation build tools (Sphinx, furo, myst-parser, etc.)

You can also combine multiple dependency groups:

.. code-block:: bash

   pip install mandol[faiss,openai,sentence-transformers]

Environment Variable Configuration
-----------------------------------

Mandol uses a ``.env`` file to manage sensitive configurations (like API Keys). A template file is provided in the project root:

.. code-block:: bash

   cp .env.example .env

Edit the ``.env`` file and fill in your actual configuration. Key environment variables:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Variable Name
     - Description
   * - ``OPENAI_API_KEY``
     - OpenAI API key, for LLM calls and remote Embedding/Reranker services
   * - ``USE_REMOTE_EMBEDDER``
     - Whether to use remote Embedder service, set to ``true`` to enable, default ``false``
   * - ``USE_REMOTE_RERANKER``
     - Whether to use remote Reranker service, set to ``true`` to enable, default ``false``

For more environment variables, see :doc:`configuration`.

Verify Installation
-------------------

After installation, run the following command to verify:

.. code-block:: bash

   python -c "from mandol import MemorySystem, MemoryUnit, Uid; print('Mandol installed successfully!')"

If the output is ``Mandol installed successfully!``, Mandol is correctly installed.

Common Issues
-------------

pip command not found
^^^^^^^^^^^^^^^^^^^^^

**Symptom**: ``pip: command not found`` when running ``pip install``

**Solution**:

.. code-block:: bash

   python -m pip install mandol

faiss-cpu installation fails
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Symptom**: Error when running ``pip install mandol[faiss]``

**Solution**:

faiss-cpu requires build environment support. If installation fails, try using conda:

.. code-block:: bash

   conda install -c conda-forge faiss-cpu
   pip install mandol

Permission denied
^^^^^^^^^^^^^^^^^

**Symptom**: ``Permission denied`` during installation

**Solution**:

Install to user directory with the ``--user`` flag:

.. code-block:: bash

   pip install --user mandol

Or use a virtual environment:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   pip install mandol
