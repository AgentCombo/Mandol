Advanced Graph Operations
===========================

Graph Pruning
--------------

Manually delete nodes and relationships that are no longer needed:

.. code-block:: python

   # Delete a node (also deletes related relationships)
   system.semantic_graph.delete_unit(Uid("entity_outdated"))

   # Batch delete relationships
   for uid in outdated_nodes:
       all_rels = system.semantic_graph.list_relationships(uid)
       for rel in all_rels:
           system.semantic_graph.delete_relationship(
               source=Uid(rel["source"]),
               target=Uid(rel["target"]),
               rel_type=rel["rel_type"],
           )

Graph Merging
--------------

.. code-block:: python

   # Cross-session entity merging (automatically called in build_high_level)
   # Manual trigger:
   system.semantic_graph._cross_session_entity_merge()

   # Event merging
   system.semantic_graph._cross_session_event_merge()

Graph State Export
-------------------

.. code-block:: python

   # View overall graph state
   graph_store = system.semantic_graph.get_graph_store()
   all_relationships = graph_store.get_all_relationships()

   # Statistics
   unique_nodes = set()
   edge_count = 0
   for rel in all_relationships:
       unique_nodes.add(rel["source"])
       unique_nodes.add(rel["target"])
       edge_count += 1

   print(f"Graph stats: {len(unique_nodes)} nodes, {edge_count} edges")
   print(f"Graph density: {edge_count / max(1, len(unique_nodes)):.2f}")

Performance Considerations
---------------------------

- In-memory graph store (InMemory): Good performance within 100K nodes
- Beyond that, consider switching to an external graph database (Neo4j, etc.)
- BFS expansion is one of the main contributors to retrieval latency (~50-200ms per expansion)
- Irrelevant edges reduce retrieval precision — regularly prune invalid relationships
