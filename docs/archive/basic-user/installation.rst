安装 Mandol
============

前置条件
--------

- **Python** 3.9 或更高版本（推荐 3.10 / 3.11）

.. code-block:: bash

   python --version

安装方式
--------

**方式一：pip 安装**

.. code-block:: bash

   pip install mandol

**方式二：虚拟环境安装（推荐）**

.. code-block:: bash

   # venv
   python -m venv .venv
   source .venv/bin/activate
   pip install mandol

   # conda
   conda create -n mandol python=3.10
   conda activate mandol
   pip install mandol

**方式三：从源码安装（开发者）**

.. code-block:: bash

   git clone https://github.com/your-org/mandol.git
   cd mandol
   pip install -e .

可选依赖
--------

.. code-block:: bash

   pip install mandol[faiss]                    # FAISS 向量索引加速
   pip install mandol[sentence-transformers]    # 本地 Embedding/Reranker 模型
   pip install mandol[openai]                   # OpenAI API 支持
   pip install mandol[all]                      # 安装所有可选依赖

选择运行模式
------------

**模式 A：远程 API（无需下载模型）**

.. code-block:: bash

   pip install mandol[openai]

   cp .env.example .env
   # 编辑 .env，填入 OPENAI_API_KEY

**模式 B：本地模型（无需 API Key，需下载模型）**

.. code-block:: bash

   pip install mandol[sentence-transformers]

   首次运行会自动下载模型（约 8 GB），后续使用缓存。

验证安装
--------

.. code-block:: bash

   python -c "from mandol import MemorySystem, MemoryUnit, Uid; print('Mandol 安装成功！')"

如果输出 ``Mandol 安装成功！``，说明安装正确。

常见安装问题
------------

**pip install 报错 "No matching distribution found"**

请确认 Python 版本 >= 3.9，并尝试 ``pip install --upgrade pip``。

**安装 faiss-cpu 失败**

尝试 ``conda install -c conda-forge faiss-cpu`` 或 ``pip install faiss-cpu --no-deps``。

**权限不足**

加 ``--user`` 参数或使用虚拟环境。

下一步
------

安装完成后，前往 :doc:`five-minute-start` 开始使用。
