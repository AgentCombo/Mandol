Relationship Basic Operations
================================

All graph operations are accessed through ``system.semantic_graph``.

Adding Relationships
---------------------

.. code-block:: python

   # Basic usage
   system.semantic_graph.add_relationship(
       source=Uid("msg_1"),
       target=Uid("msg_2"),
       rel_type="RELATED_TO",
   )

   # With properties
   system.semantic_graph.add_relationship(
       source=Uid("entity_zhangsan"),
       target=Uid("entity_lisi"),
       rel_type="RELATED_TO",
       props={"sub_type": "colleague", "confidence": 0.92},
   )

Supported Relationship Types
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Type
     - Meaning
   * - ``PRECEDES``
     - Temporal predecessor (A occurs before B)
   * - ``FOLLOWS``
     - Temporal successor (A occurs after B)
   * - ``SEMANTIC_SIMILAR``
     - Semantic similarity association
   * - ``RELATED_TO``
     - Generic relationship (includes subtypes like located_in / works_at)
   * - ``COREF``
     - Coreference relationship (conversation → entity)
   * - ``CAUSES`` / ``CAUSED_BY``
     - Event causality
   * - ``INVOLVES``
     - Event participation
   * - ``EVIDENCED_BY``
     - High-level memory → raw data

Viewing Relationships
----------------------

.. code-block:: python

   rel = system.semantic_graph.get_relationship(
       source=Uid("msg_1"),
       target=Uid("msg_2"),
       rel_type="RELATED_TO",
   )
   if rel:
       print(f"Relationship properties: {rel}")

Listing All Relationships for a Unit
--------------------------------------

.. code-block:: python

   # All directions
   all_rels = system.semantic_graph.list_relationships(Uid("msg_1"))

   # Outgoing only
   outgoing = system.semantic_graph.list_relationships(
       Uid("msg_1"), direction="outgoing"
   )

   # Incoming only
   incoming = system.semantic_graph.list_relationships(
       Uid("msg_1"), direction="incoming"
   )

   for rel in all_rels:
       src = rel.get("source", "?")
       tgt = rel.get("target", "?")
       rtype = rel.get("rel_type", "?")
       print(f"  {src} --[{rtype}]--> {tgt}")

Deleting Relationships
-----------------------

.. code-block:: python

   # Delete a specific relationship type
   system.semantic_graph.delete_relationship(
       source=Uid("msg_1"),
       target=Uid("msg_2"),
       rel_type="RELATED_TO",
   )

   # Delete all relationships between two nodes
   system.semantic_graph.delete_relationship(
       source=Uid("msg_1"),
       target=Uid("msg_2"),
       rel_type=None,  # None = all types
   )
