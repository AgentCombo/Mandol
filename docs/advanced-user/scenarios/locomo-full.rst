LoCoMo 完整复现
===================

全量 19 个会话、419 轮对话的完整处理与检索。

数据全景
--------

.. list-table::
   :header-rows: 0
   :widths: 25 75

   * - 数据集
     - LoCoMo conv-26: Caroline & Melanie
   * - 规模
     - 19 sessions, 419 turns, 199 QA pairs
   * - 特征
     - 高度个人化的长对话，包含身份探索、情感表达

运行完整流程
------------

.. code-block:: bash

   cd examples/locomo
   python run_example.py --full

图关系构建
----------

LoCoMo 示例中手动构建了三种时序关系：

.. code-block:: python

   # 时序前驱
   system.semantic_graph.add_relationship(
       prev_uid, curr_uid, "PRECEDES"
   )
   # 时序后继
   system.semantic_graph.add_relationship(
       curr_uid, prev_uid, "FOLLOWS"
   )

增量更新策略
------------

.. code-block:: python

   # 处理完前 3 个会话
   system.build_high_level(mode="auto")
   system.flush()

   # 新增会话 4-6
   load_sessions(4, 6)
   system.build_high_level(mode="auto")  # 仅增量
   system.flush()

Flush 策略：``flush()`` 将内存中的索引和存储持久化。每批会话处理完建议 flush 一次。

空间组织
--------

.. code-block::

   locomo_conv_26
   ├── locomo_conv_26_session_1
   ├── locomo_conv_26_session_2
   └── ...

每个 session 一个独立空间，以 ``{base_root}_session_{n}`` 命名。

检索统计
--------

.. code-block::

   MEMORY STATISTICS
     Total units: 835
     Total spaces: 57
     Dialogues processed: 419
     Sessions processed: 19
