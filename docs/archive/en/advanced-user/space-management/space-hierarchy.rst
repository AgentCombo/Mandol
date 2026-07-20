Space Hierarchy
=================

Spaces support parent-child hierarchical nesting, useful for organizing multi-user, multi-project memory structures.

Creating Child Spaces
-----------------------

.. code-block:: python

   # Create parent space first
   system.semantic_map.create_space("Project-2024")

   # Create child space (ensures parent exists, auto-creates if missing)
   system.semantic_map.ensure_child_space("Project-2024", "Q1")
   system.semantic_map.ensure_child_space("Project-2024", "Q2")

   # Direct attachment
   system.semantic_map.attach_child_space("Project-2024", "Q3")

Viewing Hierarchy
-------------------

.. code-block:: python

   space = system.semantic_map.get_space("Project-2024")
   print(f"Child spaces: {[s.name for s in space.child_spaces]}")

   child = system.semantic_map.get_space("Project-2024/Q1")
   print(f"Parent space: {child.parent_space.name}")

Typical Hierarchy Patterns
----------------------------

**Pattern 1: User → Session**

.. code-block::

   Support-UserA
   ├── Session-20240301
   ├── Session-20240305
   └── Session-20240308

Suitable for multi-user customer service systems, top-level grouped by user, bottom-level grouped by session.

**Pattern 2: Project → Phase → Module**

.. code-block::

   Project-2024
   ├── Q1
   │   ├── Requirements Analysis
   │   └── Prototype Design
   └── Q2
       ├── Development
       └── Testing

Suitable for knowledge management scenarios, structured documents organized by project hierarchy.

**Pattern 3: Global Shared + User Private**

.. code-block::

   Global Knowledge Base
   └── (Shared documents, FAQs, etc.)
   User-001
   └── (User private conversations, preferences, etc.)
   User-002
   └── (User private conversations, preferences, etc.)

In multi-user systems, shared knowledge is globally visible, private memories are isolated per user.
