Querying Graph Structure
==========================

Direct Neighbor Queries
------------------------

.. code-block:: python

   # Explicit neighbors (one level only)
   neighbors = system.semantic_graph.get_explicit_neighbors(
       Uid("entity_zhangsan")
   )
   for uid in neighbors:
       print(f"Neighbor: {uid}")

   # Filter by relationship type
   colleagues = system.semantic_graph.get_explicit_neighbors(
       Uid("entity_zhangsan"),
       rel_type="RELATED_TO",
   )

   # Specify direction
   successors = system.semantic_graph.get_explicit_neighbors(
       Uid("msg_1"),
       rel_type="PRECEDES",
       direction="outgoing",  # Only A→B
   )
   predecessors = system.semantic_graph.get_explicit_neighbors(
       Uid("msg_3"),
       rel_type="PRECEDES",
       direction="incoming",  # Only X→A
   )

Implicit Neighbor Queries (Planned)
-------------------------------------

.. code-block:: python

   # Find implicit associations through entity coreference
   implicit = system.semantic_graph.get_implicit_neighbors(
       Uid("msg_1")
   )
   # Returns conversation units bridged through entities

Connected Component Analysis
-----------------------------

.. code-block:: python

   components = system.semantic_graph.get_connected_components()
   for i, comp in enumerate(components):
       print(f"Connected component {i}: {len(comp)} nodes")

Subgraph Export
---------------

.. code-block:: python

   subgraph = system.semantic_graph.get_subgraph(
       uids=[Uid("entity_zhangsan"), Uid("entity_lisi")],
       hops=2,
   )
   print(f"Subgraph nodes: {len(subgraph.get('nodes', []))}")
   print(f"Subgraph edges: {len(subgraph.get('edges', []))}")
