MemorySpace 参考
=======================

MemorySpace 是记忆单元的逻辑容器，支持层级嵌套。

构造函数
--------

.. code-block:: python

   MemorySpace(
       name: SpaceName,
       parent_space: SpaceName | None = None,
   )

字段
----

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - 字段
     - 类型
     - 说明
   * - ``name``
     - SpaceName
     - 空间名称
   * - ``parent_space``
     - SpaceName | None
     - 父空间
   * - ``unit_uids``
     - Set[Uid]
     - 包含的记忆单元 UID 集合
   * - ``child_spaces``
     - Set[SpaceName]
     - 子空间名称集合
   * - ``summary_text``
     - Optional[str]
     - 空间摘要文本
   * - ``summary_embedding``
     - Optional[numpy.ndarray]
     - 摘要向量
   * - ``metadata``
     - Dict[str, Any]
     - 元数据字典

方法
----

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - 方法
     - 类型
     - 说明
   * - ``add_unit(uid)``
     - 实例方法
     - 添加记忆单元到空间
   * - ``remove_unit(uid)``
     - 实例方法
     - 从空间移除记忆单元
   * - ``add_child_space(name)``
     - 实例方法
     - 添加子空间
   * - ``remove_child_space(name)``
     - 实例方法
     - 移除子空间
   * - ``set_summary(text, embedding)``
     - 实例方法
     - 设置空间摘要及向量
   * - ``get_all_unit_uids(recursive=True, resolver=...)``
     - 实例方法
     - 递归获取所有单元 UID
   * - ``get_all_child_space_names(recursive=True, resolver=...)``
     - 实例方法
     - 递归获取所有子空间名称
   * - ``to_dict()`` / ``from_dict(data)``
     - 实例/类方法
     - 序列化与反序列化
   * - ``touch()``
     - 实例方法
     - 更新时间戳

使用示例
--------

.. code-block:: python

   space = system.semantic_map.create_space("项目-A")

   child = system.semantic_map.attach_child_space("项目-A", "Q1")

   units = system.semantic_map.list_units_in_space("项目-A")
