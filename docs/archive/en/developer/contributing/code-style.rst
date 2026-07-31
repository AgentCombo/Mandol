Code Style
=============

Use ruff for formatting and code quality checks.

.. code-block:: bash

   ruff check .          # Check
   ruff format .         # Format
   ruff check --fix .    # Auto-fix

Rules
------

- Follow PEP 8
- Type annotations: All public methods must have type annotations
- Documentation: All public methods must have docstrings
- Exceptions: Use custom exception classes, no bare ``except Exception``
