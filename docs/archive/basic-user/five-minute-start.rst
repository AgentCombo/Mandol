5 分钟快速开始
=================

两种模式，选择适合的运行方式。

模式一：远程 API（推荐）
------------------------

**前提**：已配置 ``OPENAI_API_KEY`` 环境变量和 ``config.yaml``。

.. code-block:: python

   from mandol import MemorySystem, MemoryUnit, Uid

   system = MemorySystem.from_yaml_config("config.yaml")

   system.add(MemoryUnit(
       uid=Uid("msg_1"),
       raw_data={"text_content": "张三今天去北京出差了"},
       metadata={"timestamp": "2024-01-15T10:00:00"},
   ))
   system.add(MemoryUnit(
       uid=Uid("msg_2"),
       raw_data={"text_content": "李四说下周要去上海开会"},
       metadata={"timestamp": "2024-01-15T10:05:00"},
   ))

   system.build_high_level(mode="auto")

   hits = system.holistic_retrieve("张三去了哪里？", top_k=5)
   for hit in hits:
       print(f"[{hit.final_score:.3f}] {hit.unit.raw_data['text_content']}")

**预期输出**：

.. code-block::

   [0.947] 张三今天去北京出差了

获取自然语言回复：

.. code-block:: python

   answer = system.ask("张三去了哪里？")
   print(answer)

**预期输出**：

.. code-block::

   根据记忆记录，张三今天去北京出差了。

**保存和恢复**：

.. code-block:: python

   system.save("./memory_snapshot")
   system2 = MemorySystem.load("./memory_snapshot")

模式二：本地模型（无需 API Key）
--------------------------------

.. code-block:: bash

   pip install mandol[sentence-transformers]

.. code-block:: python

   from mandol import MemorySystem, MemoryUnit, Uid

   system = MemorySystem()

   system.add(MemoryUnit(
       uid=Uid("msg_1"),
       raw_data={"text_content": "张三今天去北京出差了"},
       metadata={"timestamp": "2024-01-15T10:00:00"},
   ))
   system.add(MemoryUnit(
       uid=Uid("msg_2"),
       raw_data={"text_content": "李四说下周要去上海开会"},
       metadata={"timestamp": "2024-01-15T10:05:00"},
   ))

   system.build_high_level(mode="auto")

   hits = system.holistic_retrieve("张三去了哪里？", top_k=5)
   for hit in hits:
       print(f"[{hit.final_score:.3f}] {hit.unit.raw_data['text_content']}")

**预期输出**：

.. code-block::

   [0.947] 张三今天去北京出差了

获取自然语言回复：

.. code-block:: python

   answer = system.ask("张三去了哪里？")
   print(answer)

.. note::

   首次运行需下载模型（约 2-4 GB），请确保网络畅通。后续运行使用缓存，无需重复下载。

三个关键方法
------------

.. list-table::
   :widths: 60 40

   * - 方法
     - 作用
   * - ``system.add(unit)``
     - 添加一段记忆
   * - ``system.build_high_level(mode="auto")``
     - 构建高阶记忆
   * - ``system.holistic_retrieve(query)``
     - 检索相关记忆（返回 SearchHit 列表）
   * - ``system.ask(query)``
     - 用自然语言提问，返回自然语言回复

.. tip::

   ``raw_data`` 中的 ``text_content`` 是系统检索的核心字段。字段名必须是 ``text_content``，内容必须是纯文本。详见 :doc:`/shared/data-format-guide`。

快速健康检查
------------

使用 ``system.monitor`` 查看系统状态：

.. code-block:: python

   print(system.monitor)

输出示例：

.. code-block::

   [MemSys] units=2 | spaces=1 | graph:2n/0e | idx:2↑/0↓ | pend:0u/0e/0et | sess:1(avg2) | mem:52.3MB | CLEAN

.. important::

   ``add()`` 后系统会异步构建高阶记忆，但少量数据时可能未完成。手动调用 ``build_high_level()`` 可确保高阶记忆（实体/事件/摘要）立即可用。仅检索原始对话（BASE 组）时无需等待。

下一步
------

- :doc:`your-first-memory` — 完整的端到端流程（包含保存和加载）
- :doc:`configuration-simple` — 最常用的 4 个配置项
