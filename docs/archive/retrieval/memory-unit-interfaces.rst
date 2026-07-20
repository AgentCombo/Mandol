MemoryUnit 检索接口
=======================

以下接口由 ``SemanticMap`` 提供，适用于单个记忆单元的访问与管理。

适用于 MemoryUnit 的接口
------------------------

以下接口由 ``SemanticMap`` 提供，适用于单个记忆单元的访问与管理。

get_unit
^^^^^^^^

根据 UID 获取特定记忆单元。

**签名**：

.. code-block:: python

   def get_unit(uid: str) -> Optional[MemoryUnit]

**使用示例**：

.. code-block:: python

   unit = system.semantic_map.get_unit("dialogue_001")

get_all_units
^^^^^^^^^^^^^

获取所有记忆单元。

**签名**：

.. code-block:: python

   def get_all_units() -> List[MemoryUnit]

**使用示例**：

.. code-block:: python

   all_units = system.semantic_map.get_all_units()

filter_memory_units
^^^^^^^^^^^^^^^^^^^

按条件过滤记忆单元，支持嵌套字段查询。

**签名**：

.. code-block:: python

   def filter_memory_units(
       candidate_units: Optional[List[MemoryUnit]] = None,
       filter_condition: Optional[dict] = None,
       ms_names: Optional[List[str]] = None,
       recursive: bool = True
   ) -> List[MemoryUnit]

**使用示例**：

.. code-block:: python

   # 过滤特定空间的单元
   units = system.semantic_map.filter_memory_units(
       ms_names=["root_knowledge_entity"]
   )

   # 按元数据条件过滤
   units = system.semantic_map.filter_memory_units(
       filter_condition={"metadata.entity_type": {"eq": "Person"}}
   )

.. _retrieval-semantic-map:
