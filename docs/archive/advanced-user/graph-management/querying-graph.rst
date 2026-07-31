查询图结构
============

直接邻居查询
------------

.. code-block:: python

   # 显式邻居（只查一级）
   neighbors = system.semantic_graph.get_explicit_neighbors(
       Uid("entity_张三")
   )
   for uid in neighbors:
       print(f"邻居: {uid}")

   # 按关系类型过滤
   colleagues = system.semantic_graph.get_explicit_neighbors(
       Uid("entity_张三"),
       rel_type="RELATED_TO",
   )

   # 指定方向
   successors = system.semantic_graph.get_explicit_neighbors(
       Uid("msg_1"),
       rel_type="PRECEDES",
       direction="outgoing",  # 只查 A→B
   )
   predecessors = system.semantic_graph.get_explicit_neighbors(
       Uid("msg_3"),
       rel_type="PRECEDES",
       direction="incoming",  # 只查 X→A
   )

隐式邻居查询（预想）
--------------------

.. warning:: ⚠️ Planned — 此接口尚未实现，签名可能变更

.. code-block:: python

   # 通过实体共指找隐式关联
   implicit = system.semantic_graph.get_implicit_neighbors(
       Uid("msg_1")
   )
   # 返回通过实体桥接的对话单元

连通分量分析
------------

.. code-block:: python

   components = system.semantic_graph.get_connected_components()
   for i, comp in enumerate(components):
       print(f"连通分量 {i}: {len(comp)} 个节点")

子图导出
--------

.. code-block:: python

   subgraph = system.semantic_graph.get_subgraph(
       uids=[Uid("entity_张三"), Uid("entity_李四")],
       hops=2,
   )
   print(f"子图节点: {len(subgraph.get('nodes', []))}")
   print(f"子图边: {len(subgraph.get('edges', []))}")
