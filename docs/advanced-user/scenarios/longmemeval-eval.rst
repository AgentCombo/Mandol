LongMemEval 完整评估
=========================

使用 HuggingFace 全量数据集进行 6 类问题评估。

获取全量数据
------------

.. code-block:: bash

   cd examples/longmemeval
   python download_data.py

6 类评估详解
------------

.. list-table::
   :header-rows: 1
   :widths: 22 28 50

   * - 类别
     - 全称
     - 考察
   * - SS-Pref
     - Single-Session Preference
     - 用户偏好/观点
   * - SS-Asst
     - Single-Session Assistant
     - 精确事实提取
   * - Temporal
     - Temporal
     - 时间关系推理
   * - Multi-S
     - Multi-Session
     - 跨会话综合
   * - Know.Upd.
     - Knowledge Update
     - 知识变化追踪
   * - SS-User
     - Single-Session User
     - 用户特定细节

运行评估
--------

.. code-block:: bash

   python run_example.py --eval

Per-Category 分析
------------------

.. code-block::

   Category Breakdown:
     SS-Pref:      180/200 (90.0%)
     SS-Asst:      195/200 (97.5%)
     Temporal:     170/200 (85.0%)
     Multi-S:       88/100 (88.0%)
     Know.Upd.:     75/100 (75.0%)
     SS-User:      185/200 (92.5%)
   Total:         893/1000 (89.3%)

自定义评估
----------

.. code-block:: python

   # 仅评估特定类别
   python run_example.py --eval --category temporal,knowledge-update

   # 自定义查询
   python run_example.py --query "What happened in 2018?"
