Installation
============

Mandol currently targets Python ``>=3.12,<3.13``. Use ``uv`` from the repository
root so the environment is resolved from ``pyproject.toml`` and ``uv.lock``.

Recommended development environment
-----------------------------------

.. code-block:: bash

   uv sync --extra dev --extra docs --group spacy-model

For paper reproduction and throughput comparisons, install the full artifact
stack when the machine has a compatible CUDA / flash-attention environment:

.. code-block:: bash

   uv sync --extra dev --extra cuda --group spacy-model

Runtime-only environment
------------------------

.. code-block:: bash

   uv sync

Optional groups and extras
--------------------------

.. code-block:: bash

   uv sync --group spacy-model
   uv sync --extra cuda

``spacy-model`` installs the optional ``en-core-web-lg`` wheel. ``cuda`` installs
the configured ``flash-attn`` wheel and should only be used on a matching Linux
CUDA environment. Accuracy reproduction can run without ``cuda``, but retrieval
and reranking throughput may differ from the paper performance environment.

Verify the install
------------------

Use a light import check first. It does not instantiate embedding models.

.. code-block:: bash

   uv run python -c "import mandol; print(mandol.__version__)"

The expected package version in this checkout is ``0.1.0a1``.

Important dependency note
-------------------------

The default dependency list is intentionally the complete research/runtime
stack. It includes Torch, transformers, sentence-transformers, FAISS CPU,
DuckDB, graph libraries, LLM clients, retrieval/rerank dependencies and
benchmark tooling. This differs from older docs that described many optional
``pip install mandol[...]`` extras. In the current ``pyproject.toml``, the
maintained extras are ``dev``, ``docs`` and ``cuda``.
