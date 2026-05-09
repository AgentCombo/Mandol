安装指南
========

.. note::

   本文档已迁移至 :doc:`/basic-user/installation`。本页面将在后续版本中移除，请更新你的书签。

前置条件
--------

在安装 Mandol 之前，请确保你的环境满足以下要求：

- **Python** 3.9 或更高版本
- **pip** 或 **conda** 包管理器

你可以通过以下命令检查 Python 版本：

.. code-block:: bash

   python --version

基础安装
--------

使用 pip 安装最新稳定版：

.. code-block:: bash

   pip install mandol

从源码安装
----------

如果你需要最新的开发版本或希望参与贡献，可以从源码安装：

.. code-block:: bash

   git clone https://github.com/your-org/mandol.git
   cd mandol
   pip install -e .

``-e`` 参数表示以可编辑模式安装，修改源码后无需重新安装即可生效。

可选依赖
--------

Mandol 采用核心精简、按需扩展的依赖策略。以下可选依赖组可根据你的使用场景安装：

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - 依赖组
     - 安装命令
     - 说明
   * - faiss
     - ``pip install mandol[faiss]``
     - FAISS 向量索引，适用于大规模向量检索场景
   * - sentence-transformers
     - ``pip install mandol[sentence-transformers]``
     - 本地 Embedding 和 Reranker 模型，无需 API Key 即可运行
   * - openai
     - ``pip install mandol[openai]``
     - OpenAI 兼容 API 客户端，用于 LLM 和远程 Embedding
   * - milvus
     - ``pip install mandol[milvus]``
     - Milvus 向量数据库，适用于生产环境持久化存储
   * - neo4j
     - ``pip install mandol[neo4j]``
     - Neo4j 图数据库，适用于大规模语义图存储
   * - all
     - ``pip install mandol[all]``
     - 安装所有可选依赖
   * - dev
     - ``pip install mandol[dev]``
     - 开发工具（pytest、pre-commit、ruff 等）
   * - docs
     - ``pip install mandol[docs]``
     - 文档构建工具（Sphinx、furo、myst-parser 等）

你也可以组合安装多个依赖组：

.. code-block:: bash

   pip install mandol[faiss,openai,sentence-transformers]

环境变量配置
------------

Mandol 使用 ``.env`` 文件管理敏感配置（如 API Key）。项目根目录提供了模板文件：

.. code-block:: bash

   cp .env.example .env

编辑 ``.env`` 文件，填入你的实际配置。以下为关键环境变量：

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 变量名
     - 说明
   * - ``OPENAI_API_KEY``
     - OpenAI API 密钥，用于 LLM 调用和远程 Embedding/Reranker 服务
   * - ``USE_REMOTE_EMBEDDER``
     - 是否使用远程 Embedder 服务，设为 ``true`` 启用，默认 ``false``
   * - ``USE_REMOTE_RERANKER``
     - 是否使用远程 Reranker 服务，设为 ``true`` 启用，默认 ``false``

更多环境变量详见 :doc:`configuration`。

验证安装
--------

安装完成后，运行以下命令验证：

.. code-block:: bash

   python -c "from mandol import MemorySystem, MemoryUnit, Uid; print('Mandol 安装成功！')"

如果输出 ``Mandol 安装成功！``，说明 Mandol 已正确安装。

常见问题
--------

pip 命令未找到
^^^^^^^^^^^^^^

**现象**：执行 ``pip install`` 时提示 ``pip: command not found``

**解决方案**：

.. code-block:: bash

   python -m pip install mandol

faiss-cpu 安装失败
^^^^^^^^^^^^^^^^^^

**现象**：执行 ``pip install mandol[faiss]`` 时报错

**解决方案**：

faiss-cpu 需要编译环境支持。如果安装失败，可以尝试使用 conda 安装：

.. code-block:: bash

   conda install -c conda-forge faiss-cpu
   pip install mandol

权限不足
^^^^^^^^

**现象**：安装时提示 ``Permission denied``

**解决方案**：

使用 ``--user`` 参数安装到用户目录：

.. code-block:: bash

   pip install --user mandol

或使用虚拟环境：

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   pip install mandol
