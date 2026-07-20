MemorySpace Reference
========================

MemorySpace is a logical container for memory units, supporting hierarchical nesting.

Constructor
------------

.. code-block:: python

   MemorySpace(
       name: SpaceName,
       parent_space: SpaceName | None = None,
   )

Fields
-------

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Field
     - Type
     - Description
   * - ``name``
     - SpaceName
     - Space name
   * - ``parent_space``
     - SpaceName | None
     - Parent space
   * - ``unit_uids``
     - Set[Uid]
     - Set of contained memory unit UIDs
   * - ``child_spaces``
     - Set[SpaceName]
     - Set of child space names
   * - ``summary_text``
     - Optional[str]
     - Space summary text
   * - ``summary_embedding``
     - Optional[numpy.ndarray]
     - Summary vector
   * - ``metadata``
     - Dict[str, Any]
     - Metadata dictionary

Methods
--------

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Method
     - Type
     - Description
   * - ``add_unit(uid)``
     - Instance
     - Add memory unit to space
   * - ``remove_unit(uid)``
     - Instance
     - Remove memory unit from space
   * - ``add_child_space(name)``
     - Instance
     - Add child space
   * - ``remove_child_space(name)``
     - Instance
     - Remove child space
   * - ``set_summary(text, embedding)``
     - Instance
     - Set space summary and vector
   * - ``get_all_unit_uids(recursive=True, resolver=...)``
     - Instance
     - Recursively get all unit UIDs
   * - ``get_all_child_space_names(recursive=True, resolver=...)``
     - Instance
     - Recursively get all child space names
   * - ``to_dict()`` / ``from_dict(data)``
     - Instance/Class
     - Serialization and deserialization
   * - ``touch()``
     - Instance
     - Update timestamp

Usage Example
--------------

.. code-block:: python

   space = system.semantic_map.create_space("Project-A")

   child = system.semantic_map.attach_child_space("Project-A", "Q1")

   units = system.semantic_map.list_units_in_space("Project-A")
