创建与管理空间
============

所有空间操作通过 ``system.semantic_map`` 访问。

创建空间
--------

.. code-block:: python

   from mandol.domain.types import SpaceName

   space = system.semantic_map.create_space("客服-用户A")
   # 返回 MemorySpace 对象

   # 或直接通过 SpaceName 创建
   system.semantic_map.create_space(SpaceName("客服-用户A"))

获取空间
--------

.. code-block:: python

   space = system.semantic_map.get_space("客服-用户A")
   # 返回 MemorySpace 或 None（不存在时）

   if space:
       print(f"空间名称: {space.name}")
       print(f"父空间: {space.parent_space}")
       print(f"子空间数: {len(space.child_spaces)}")

列出所有空间
------------

.. code-block:: python

   all_spaces = system.semantic_map.list_spaces()
   for sp in all_spaces:
       print(sp.name)

统计
----

.. code-block:: python

   # 全局统计
   total = system.semantic_map.count_units()
   print(f"总记忆数: {total}")

   # 按空间统计
   in_space = system.semantic_map.count_units(space_name="客服-用户A")
   print(f"客服-用户A 中的记忆数: {in_space}")

删除空间
--------

.. code-block:: python

   # 仅删除空空间
   system.semantic_map.delete_space("临时空间")

   # 级联删除（删除空间 + 所有单元 + 图关系 + 索引）
   system.semantic_map.delete_space("废弃项目", cascade=True)

.. caution::

   级联删除不可逆。删除前请确认空间下的数据不再需要。

列出空间下的单元
-----------------

.. code-block:: python

   units = system.semantic_map.list_units_in_space("客服-用户A")
   for u in units:
       print(f"[{u.uid}] {u.raw_data.get('text_content', '')[:60]}")

从空间移除单元
---------------

.. code-block:: python

   # 从空间移除（不删除单元本身）
   system.semantic_map.remove_unit_from_space("msg_001", "客服-用户A")
