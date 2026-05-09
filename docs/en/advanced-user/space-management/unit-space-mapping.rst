Unit-Space Mapping
=====================

Specifying Spaces When Adding
-------------------------------

.. code-block:: python

   unit = MemoryUnit(
       uid=Uid("msg_1"),
       raw_data={"text_content": "User asked about the return policy"},
   )

   # Add to a specific space
   system.add(unit, space_names=["Support-UserA"])

   # Add to multiple spaces
   system.add(unit, space_names=["Support-UserA", "Urgent-Pending"])

Post-Hoc Space Assignment
---------------------------

For existing units, you can assign them to spaces after creation:

.. code-block:: python

   # Assign to an existing space
   system.semantic_map.add_unit_to_space("msg_1", "Support-UserA")

   # Get the spaces a unit currently belongs to
   # Check via MemorySpace's units list

Space Migration
----------------

.. code-block:: python

   # Remove from A → add to B
   system.semantic_map.remove_unit_from_space("msg_1", "Old Space")
   system.semantic_map.add_unit_to_space("msg_1", "New Space")

Batch Operations
-----------------

.. code-block:: python

   units = [unit1, unit2, unit3]
   system.add_many(units, space_names=["Batch Import"])

   # Add data first, then assign to spaces
   system.add_many(units)
   for uid in [u.uid for u in units]:
       system.semantic_map.add_unit_to_space(uid, "Batch Import")

Unit-Space Retrieval Relationship
-----------------------------------

- A unit can belong to multiple spaces
- ``retrieve_in_space(query, space_name="X")`` only retrieves units in space X
- ``holistic_retrieve(query)`` retrieves across all spaces
- During high-level memory construction, each session's high-level structures (entities, summaries) are automatically assigned to corresponding spaces
