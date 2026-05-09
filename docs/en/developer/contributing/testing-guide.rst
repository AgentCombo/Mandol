Testing Guide
=================

Running Tests
---------------

.. code-block:: bash

   make test           # All
   make test-unit      # Unit tests
   make test-integration  # Integration tests

Test framework: pytest

Writing Tests
---------------

.. code-block:: python

   # tests/test_semantic_map.py
   import pytest
   from mandol.application.semantic_map import SemanticMapService

   class TestSemanticMapService:
       def test_create_space(self, semantic_map):
           space = semantic_map.create_space("test")
           assert space.name == "test"

Fixture location: ``tests/conftest.py``
