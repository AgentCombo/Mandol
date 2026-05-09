贡献指南
========

感谢你对 Mandol 项目的关注！本指南介绍如何参与项目贡献。

开发环境搭建
------------

.. code-block:: bash

   git clone https://github.com/your-org/mandol.git
   cd mandol
   pip install -e ".[dev]"

代码风格
--------

Mandol 使用 ruff 进行代码格式化和 lint 检查：

.. code-block:: bash

   ruff check mandol/
   ruff format mandol/

测试
----

.. code-block:: bash

   pytest tests/ -v

提交规范
--------

提交信息遵循 `Conventional Commits <https://www.conventionalcommits.org/>`_ 规范：

- ``feat: 添加新功能``
- ``fix: 修复 bug``
- ``docs: 文档更新``
- ``refactor: 代码重构``
- ``test: 测试相关``
- ``chore: 构建/工具变更``

文档贡献
--------

文档使用 Sphinx + RST 格式，位于 ``docs/`` 目录。

构建文档：

.. code-block:: bash

   pip install -e ".[docs]"
   cd docs && make html

文档写作规范：

- 使用中文撰写
- 代码块中不添加注释
- 使用 ``.. warning::`` 标注预想接口
- 使用 ``.. caution::`` 标注内部接口
- 使用 ``list-table`` RST 指令创建表格

问题反馈
--------

- 在 GitHub Issues 中提交 bug 报告或功能建议
- 提交时请包含：Python 版本、操作系统、复现步骤、预期行为与实际行为
