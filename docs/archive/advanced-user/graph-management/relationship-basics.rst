关系基础操作
==============

所有图操作通过 ``system.semantic_graph`` 访问。

添加关系
--------

.. code-block:: python

   # 基础用法
   system.semantic_graph.add_relationship(
       source=Uid("msg_1"),
       target=Uid("msg_2"),
       rel_type="RELATED_TO",
   )

   # 带属性
   system.semantic_graph.add_relationship(
       source=Uid("entity_张三"),
       target=Uid("entity_李四"),
       rel_type="RELATED_TO",
       props={"sub_type": "colleague", "confidence": 0.92},
   )

支持的关系类型
--------------

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - 类型
     - 含义
   * - ``PRECEDES``
     - 时序前驱（A 在 B 之前发生）
   * - ``FOLLOWS``
     - 时序后继（A 在 B 之后发生）
   * - ``SEMANTIC_SIMILAR``
     - 语义相似关联
   * - ``RELATED_TO``
     - 通用关系（含 located_in / works_at 等子类型）
   * - ``COREF``
     - 共指关系（对话→实体）
   * - ``CAUSES`` / ``CAUSED_BY``
     - 事件因果
   * - ``INVOLVES``
     - 事件参与
   * - ``EVIDENCED_BY``
     - 高阶记忆→原始数据

查看关系
--------

.. code-block:: python

   rel = system.semantic_graph.get_relationship(
       source=Uid("msg_1"),
       target=Uid("msg_2"),
       rel_type="RELATED_TO",
   )
   if rel:
       print(f"关系属性: {rel}")

列出某单元的所有关系
---------------------

.. code-block:: python

   # 所有方向
   all_rels = system.semantic_graph.list_relationships(Uid("msg_1"))

   # 仅出边
   outgoing = system.semantic_graph.list_relationships(
       Uid("msg_1"), direction="outgoing"
   )

   # 仅入边
   incoming = system.semantic_graph.list_relationships(
       Uid("msg_1"), direction="incoming"
   )

   for rel in all_rels:
       src = rel.get("source", "?")
       tgt = rel.get("target", "?")
       rtype = rel.get("rel_type", "?")
       print(f"  {src} --[{rtype}]--> {tgt}")

删除关系
--------

.. code-block:: python

   # 删除指定类型关系
   system.semantic_graph.delete_relationship(
       source=Uid("msg_1"),
       target=Uid("msg_2"),
       rel_type="RELATED_TO",
   )

   # 删除两节点间所有关系
   system.semantic_graph.delete_relationship(
       source=Uid("msg_1"),
       target=Uid("msg_2"),
       rel_type=None,  # None = 所有类型
   )
