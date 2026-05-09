Installing Mandol
=================

Prerequisites
-------------

- **Python** 3.9 or higher (3.10 / 3.11 recommended)

.. code-block:: bash

   python --version

Installation Methods
--------------------

**Method 1: pip install (simplest)**

.. code-block:: bash

   pip install mandol

**Method 2: Virtual environment install (recommended)**

.. code-block:: bash

   # venv
   python -m venv .venv
   source .venv/bin/activate
   pip install mandol

   # conda
   conda create -n mandol python=3.10
   conda activate mandol
   pip install mandol

**Method 3: Install from source (developers)**

.. code-block:: bash

   git clone https://github.com/your-org/mandol.git
   cd mandol
   pip install -e .

Optional Dependencies
---------------------

.. code-block:: bash

   pip install mandol[faiss]                    # FAISS vector index acceleration
   pip install mandol[sentence-transformers]    # Local Embedding/Reranker models
   pip install mandol[openai]                   # OpenAI API support
   pip install mandol[all]                      # Install all optional dependencies

Choose Your Runtime Mode
------------------------

**Mode A: Remote API (easiest, no model downloads needed)**

.. code-block:: bash

   pip install mandol[openai]

   cp .env.example .env
   # Edit .env, fill in OPENAI_API_KEY

**Mode B: Local models (no API Key needed, requires model downloads)**

.. code-block:: bash

   pip install mandol[sentence-transformers]

   Models will be automatically downloaded on first run (~8 GB), cached for subsequent use.

Verify Installation
-------------------

.. code-block:: bash

   python -c "from mandol import MemorySystem, MemoryUnit, Uid; print('Mandol installed successfully!')"

If the output is ``Mandol installed successfully!``, the installation is correct.

Common Installation Issues
--------------------------

**pip install error "No matching distribution found"**

Please confirm Python version >= 3.9, and try ``pip install --upgrade pip``.

**faiss-cpu installation fails**

Try ``conda install -c conda-forge faiss-cpu`` or ``pip install faiss-cpu --no-deps``.

**Permission denied**

Add the ``--user`` flag or use a virtual environment.

Next Steps
----------

Once installed, head to :doc:`five-minute-start` to get started.
