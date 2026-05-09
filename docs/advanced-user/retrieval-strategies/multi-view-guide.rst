多视图检索指南
================

8 种视图的选择决策树。

视图一览
--------

.. list-table::
   :header-rows: 1
   :widths: 22 28 50

   * - 视图
     - 检索内容
     - 适用问题类型
   * - ``base_memory``
     - 原始对话
     - 「上次聊了什么」「原话是什么」
   * - ``knowledge``
     - 知识摘要
     - 「系统知道什么关于 X」
   * - ``entity_relation``
     - 实体节点
     - 「谁认识谁」「张三是做什么的」
   * - ``event_causal``
     - 事件因果
     - 「发生了什么」「为什么」
   * - ``episodic``
     - 情景摘要
     - 「那段时间发生了什么」
   * - ``emotional``
     - 情感摘要
     - 「用户情绪怎么样」
   * - ``procedural``
     - 程序总结
     - 「怎么做」「流程是什么」
   * - ``insights``
     - 洞察
     - 「有什么深层模式」

选择决策树
----------

.. code-block::

   你想知道什么？
   ├── 原始对话原话
   │   └── view="base_memory"
   ├── 关于某主题的知识
   │   └── view="knowledge"
   ├── 人物/地点/概念的信息
   │   └── view="entity_relation"
   ├── 事件及前因后果
   │   └── view="event_causal"
   ├── 某段时间的概括
   │   └── view="episodic"
   ├── 用户的情感态度
   │   └── view="emotional"
   ├── 操作步骤/流程
   │   └── view="procedural"
   └── 深层模式/规律
       └── view="insights"
