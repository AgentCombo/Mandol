SemanticGraphService Complete Reference
==========================================

Responsible for relationship modeling and graph traversal between memory units.

Access
-------

Accessed through ``system.semantic_graph``.

All Public Methods
-------------------

Node Management:

- ``add_unit(uid: Uid) -> None``
- ``delete_unit(uid: Uid) -> None``

Relationship Management:

- ``add_relationship(source: Uid, target: Uid, rel_type: str, props=None) -> None``
- ``get_relationship(source: Uid, target: Uid, rel_type: str) -> dict | None``
- ``delete_relationship(source: Uid, target: Uid, rel_type: str | None = None) -> None``
- ``list_relationships(uid: Uid, direction: str = "both") -> list[dict]``

Queries:

- ``get_explicit_neighbors(uid: Uid, rel_type=None, direction="both") -> list[Uid]``
- ``get_implicit_neighbors(uid: Uid) -> list[Uid]``
- ``get_units_in_spaces(space_names: list[str]) -> list[Uid]``

Graph Traversal:

- ``bfs_expand_units(seeds: list[Uid], per_seed=3, hops=1) -> list[Uid]``

Maintenance:

- ``flush() -> None``
- ``get_graph_store()``

Advanced:

- ``get_connected_components() -> list[list[str]]``
- ``get_subgraph(uids: list[Uid], hops=1) -> dict``

Usage Example
--------------

.. code-block:: python

   system.semantic_graph.add_relationship(
       Uid("msg_1"), Uid("msg_2"), "RELATED_TO"
   )

   neighbors = system.semantic_graph.get_explicit_neighbors(Uid("msg_1"))
   expanded = system.semantic_graph.bfs_expand_units([Uid("msg_1")])

Graph Structure Composition
-----------------------------

SemanticGraph consists of two parts: explicit relationships and implicit semantic relationships.

Explicit Relationships
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 40 42

   * - Edge Type
     - Description
     - Subtypes
   * - ``RELATED_TO``
     - Entity relationship edge
     - hometown, lives_in, works_at, located_in, part_of
   * - ``CAUSES``
     - Event causal relationship (A causes B)
     -
   * - ``CAUSED_BY``
     - Event causal relationship (B is caused by A)
     -
   * - ``INVOLVES``
     - Event-entity edge
     - participant, location, organizer, victim
   * - ``EVIDENCED_BY``
     - Traceability edge (points to raw conversation unit)
     -
   * - ``COREF``
     - Coreference edge (conversation unit → global entity reference)
     -
   * - ``ALIAS_OF``
     - Alias edge (entity alias relationship)
     -
   * - ``PRECEDES``
     - Temporal edge (chronological order of conversations/events)
     -
   * - ``FOLLOWS``
     - Temporal edge (chronological order of conversations/events)
     -

Implicit Semantic Relationships
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``SEMANTIC_SIMILAR`` edges built based on vector similarity
- Incrementally built during ``add()`` via ``_build_immediate_similarity_edges()``
- Edge weight is vector cosine similarity

For how each edge type is used in high-level memory, see the "Multi-Perspective Memory Representation" section in :doc:`/shared/memory-pipeline/detailed-flow`.
