单元与空间的映射
================

添加时指定空间
--------------

.. code-block:: python

   unit = MemoryUnit(
       uid=Uid("msg_1"),
       raw_data={"text_content": "用户询问了退货政策"},
   )

   # 添加到指定空间
   system.add(unit, space_names=["客服-用户A"])

   # 添加到多个空间
   system.add(unit, space_names=["客服-用户A", "紧急-待处理"])

事后分配空间
------------

对于已存在的单元，可以事后分配到空间：

.. code-block:: python

   # 分配到现有空间
   system.semantic_map.add_unit_to_space("msg_1", "客服-用户A")

   # 获取某个单元当前所在的空间
   # 通过 MemorySpace 的 units 列表检查

空间迁移
--------

.. code-block:: python

   # 从 A 移除 → 添加到 B
   system.semantic_map.remove_unit_from_space("msg_1", "旧空间")
   system.semantic_map.add_unit_to_space("msg_1", "新空间")

批量操作
--------

.. code-block:: python

   units = [unit1, unit2, unit3]
   system.add_many(units, space_names=["批量导入"])

   # 先添加数据，再分配空间
   system.add_many(units)
   for uid in [u.uid for u in units]:
       system.semantic_map.add_unit_to_space(uid, "批量导入")

单元与空间的检索关系
--------------------

- 一个单元可以属于多个空间
- ``retrieve_in_space(query, space_name="X")`` 只检索空间 X 下的单元
- ``holistic_retrieve(query)`` 检索全部空间
- 构建高阶记忆时，每个会话的高阶结构（实体、摘要）自动分配到对应的空间
