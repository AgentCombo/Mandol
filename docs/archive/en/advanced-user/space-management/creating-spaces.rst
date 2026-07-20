Creating and Managing Spaces
==============================

All space operations are accessed through ``system.semantic_map``.

Creating Spaces
----------------

.. code-block:: python

   from mandol.domain.types import SpaceName

   space = system.semantic_map.create_space("Support-UserA")
   # Returns MemorySpace object

   # Or create via SpaceName directly
   system.semantic_map.create_space(SpaceName("Support-UserA"))

Getting a Space
----------------

.. code-block:: python

   space = system.semantic_map.get_space("Support-UserA")
   # Returns MemorySpace or None (when not found)

   if space:
       print(f"Space name: {space.name}")
       print(f"Parent space: {space.parent_space}")
       print(f"Child spaces: {len(space.child_spaces)}")

Listing All Spaces
-------------------

.. code-block:: python

   all_spaces = system.semantic_map.list_spaces()
   for sp in all_spaces:
       print(sp.name)

Statistics
-----------

.. code-block:: python

   # Global statistics
   total = system.semantic_map.count_units()
   print(f"Total memories: {total}")

   # Per-space statistics
   in_space = system.semantic_map.count_units(space_name="Support-UserA")
   print(f"Memories in Support-UserA: {in_space}")

Deleting Spaces
----------------

.. code-block:: python

   # Delete empty space only
   system.semantic_map.delete_space("Temporary")

   # Cascade delete (delete space + all units + graph relationships + indexes)
   system.semantic_map.delete_space("Deprecated Project", cascade=True)

.. caution::

   Cascade deletion is irreversible. Confirm that data under the space is no longer needed before deleting.

Listing Units in a Space
--------------------------

.. code-block:: python

   units = system.semantic_map.list_units_in_space("Support-UserA")
   for u in units:
       print(f"[{u.uid}] {u.raw_data.get('text_content', '')[:60]}")

Removing Units from a Space
-----------------------------

.. code-block:: python

   # Remove from space (does not delete the unit itself)
   system.semantic_map.remove_unit_from_space("msg_001", "Support-UserA")
