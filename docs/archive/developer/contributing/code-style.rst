代码规范
========

使用 ruff 进行格式化和代码质量检查。

.. code-block:: bash

   ruff check .          # 检查
   ruff format .         # 格式化
   ruff check --fix .    # 自动修复

规则
----

- 遵循 PEP 8
- 类型注解：所有公开方法必须有类型注解
- 文档：公开方法必须有 docstring
- 异常：使用自定义异常类，不裸露 ``except Exception``
