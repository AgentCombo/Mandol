空间层级
========

空间支持父子层级嵌套，可用于组织多用户、多项目的记忆结构。

创建子空间
----------

.. code-block:: python

   # 先创建父空间
   system.semantic_map.create_space("项目-2024")

   # 创建子空间（保证父空间存在，不存在则自动创建）
   system.semantic_map.ensure_child_space("项目-2024", "Q1")
   system.semantic_map.ensure_child_space("项目-2024", "Q2")

   # 直接挂载
   system.semantic_map.attach_child_space("项目-2024", "Q3")

查看层级
--------

.. code-block:: python

   space = system.semantic_map.get_space("项目-2024")
   print(f"子空间: {[s.name for s in space.child_spaces]}")

   child = system.semantic_map.get_space("项目-2024/Q1")
   print(f"父空间: {child.parent_space.name}")

典型层级模式
------------

**模式一：用户 → 会话**

.. code-block::

   客服-用户A
   ├── 会话-20240301
   ├── 会话-20240305
   └── 会话-20240308

适合多用户的客服系统，顶层按用户分组，底层按会话分组。

**模式二：项目 → 阶段 → 模块**

.. code-block::

   项目-2024
   ├── Q1
   │   ├── 需求分析
   │   └── 原型设计
   └── Q2
       ├── 开发
       └── 测试

适合知识管理场景，结构化的文档按项目层级组织。

**模式三：全局共享 + 用户私有**

.. code-block::

   全局知识库
   └── （共享文档、FAQ 等）
   用户-001
   └── （用户私有对话、偏好等）
   用户-002
   └── （用户私有对话、偏好等）

多用户系统中，共享知识全局可见，私有记忆按用户隔离。
