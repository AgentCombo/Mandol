Space Management Best Practices
==================================

Multi-User Space Isolation
---------------------------

.. code-block:: python

   users = ["user_001", "user_002", "user_003"]

   for uid in users:
       system.semantic_map.create_space(f"User-{uid}")
       user_units = load_user_history(uid)
       system.add_many(user_units, space_names=[f"User-{uid}"])
       system.build_high_level(mode="auto")

   # Only search a specific user's memories
   hits = system.retrieve_in_space(
       "recent orders", space_name="User-001"
   )

Business Space Layering
------------------------

.. code-block:: python

   system.semantic_map.create_space("Business")
   system.semantic_map.ensure_child_space("Business", "Orders")
   system.semantic_map.ensure_child_space("Business", "After-Sales")
   system.semantic_map.ensure_child_space("Business", "Inquiries")

   # Categorize customer service conversations by business type
   system.add(order_unit, space_names=["Business/Orders"])
   system.add(after_sale_unit, space_names=["Business/After-Sales"])

   # Query all memories for a specific business type
   hits = system.retrieve_in_space(
       "user feedback", space_name="Business/After-Sales"
   )

Temporary and Persistent Spaces
---------------------------------

.. code-block:: python

   # Temporary analysis space (delete when done)
   system.semantic_map.create_space("Analysis-Temporary")
   system.add_many(analysis_units, space_names=["Analysis-Temporary"])
   hits = system.retrieve_in_space("patterns", space_name="Analysis-Temporary")
   # ... after analysis ...
   system.semantic_map.delete_space("Analysis-Temporary", cascade=True)

Performance Tips for Many Spaces
-----------------------------------

- 100+ spaces: Minimal impact
- 1000+ spaces: Consider periodically cleaning up unused empty spaces
- ``list_spaces()`` returns the complete list; for large numbers of spaces, consider using ``get_space()`` on demand
