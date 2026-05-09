Development Environment Setup
=================================

Clone the Code
----------------

.. code-block:: bash

   git clone https://github.com/your-org/mandol.git
   cd mandol

Create Virtual Environment
----------------------------

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate

   pip install -e ".[dev]"

Configure pre-commit
-----------------------

.. code-block:: bash

   pip install pre-commit
   pre-commit install

Configure .env
-----------------

.. code-block:: bash

   cp .env.example .env
   # Edit .env and fill in necessary API Keys

Verify Environment
--------------------

.. code-block:: bash

   python -c "from mandol import MemorySystem; print('Development environment ready')"
   make test
