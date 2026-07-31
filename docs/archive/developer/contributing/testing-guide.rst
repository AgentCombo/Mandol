测试指南
========

运行测试
--------

.. code-block:: bash

   make test           # 全部
   make test-unit      # 单元测试
   make test-integration  # 集成测试

测试框架：pytest

编写测试
--------

.. code-block:: python

   # tests/test_semantic_map.py
   import pytest
   from mandol.application.semantic_map import SemanticMapService

   class TestSemanticMapService:
       def test_create_space(self, semantic_map):
           space = semantic_map.create_space("test")
           assert space.name == "test"

Fixture 位置：``tests/conftest.py``
