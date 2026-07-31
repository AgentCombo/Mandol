开发环境搭建
==============

克隆代码
--------

.. code-block:: bash

   git clone https://github.com/your-org/mandol.git
   cd mandol

创建虚拟环境
------------

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate

   pip install -e ".[dev]"

配置 pre-commit
---------------

.. code-block:: bash

   pip install pre-commit
   pre-commit install

配置 .env
---------

.. code-block:: bash

   cp .env.example .env
   # 编辑 .env 填入必要的 API Key

验证环境
--------

.. code-block:: bash

   python -c "from mandol import MemorySystem; print('开发环境就绪')"
   make test
