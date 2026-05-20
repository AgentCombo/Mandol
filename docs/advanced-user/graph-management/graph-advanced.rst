高级图操作
============

图修剪
------

手动删除不再需要的节点和关系：

.. code-block:: python

   # 删除节点（会同时删除相关关系）
   system.semantic_graph.delete_unit(Uid("entity_过时"))

   # 批量删除关系
   for uid in outdated_nodes:
       all_rels = system.semantic_graph.list_relationships(uid)
       for rel in all_rels:
           system.semantic_graph.delete_relationship(
               source=Uid(rel["source"]),
               target=Uid(rel["target"]),
               rel_type=rel["rel_type"],
           )

图合并
------

.. code-block:: python

   # 跨会话实体合并（在 build_high_level 中自动调用）
   # 手动触发：
   system.semantic_graph._cross_session_entity_merge()
   <!-- TODO: 验证 API 签名 -->

   # 事件合并
   system.semantic_graph._cross_session_event_merge()
   <!-- TODO: 验证 API 签名 -->

图状态导出
----------

.. code-block:: python

   # 查看图的整体状态
   graph_store = system.semantic_graph.get_graph_store()
   all_relationships = graph_store.get_all_relationships()

   # 统计
   unique_nodes = set()
   edge_count = 0
   for rel in all_relationships:
       unique_nodes.add(rel["source"])
       unique_nodes.add(rel["target"])
       edge_count += 1

   print(f"图统计: {len(unique_nodes)} 个节点, {edge_count} 条边")
   print(f"图密度: {edge_count / max(1, len(unique_nodes)):.2f}")

性能考虑
--------

- 内存图存储（InMemory）：10万节点内性能良好
- 超出建议切换到外部图数据库（Neo4j 等）
- BFS 扩展是检索延迟的主要贡献者之一（每次扩展 ≈ 50-200ms）
- 不相关的边会降低检索精度——定期修剪无效关系
