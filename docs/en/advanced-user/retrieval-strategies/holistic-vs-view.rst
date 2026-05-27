Holistic vs By-View vs In-Space
==================================

Comparison of three retrieval approaches.

holistic_retrieve — Easiest
----------------------------

.. code-block:: python

   hits = system.holistic_retrieve("refund policy", top_k=10)

Automatically handles all four groups, three-way retrieval, RRF fusion, BFS expansion, and Rerank. Suitable when you're not sure where the data is.

retrieve_by_view — By Semantic Perspective
-------------------------------------------

.. code-block:: python

   hits = system.retrieve_by_view("refund policy", view="knowledge", top_k=10)

Only retrieves knowledge-type memories, skipping conversations/events/emotions. Suitable when you specifically want to know "what the system knows."

Available views:

- ``knowledge`` / ``entity_relation`` / ``event_causal`` / ``emotional`` / ``episodic`` / ``procedural`` / ``insights`` / ``base_memory``

retrieve_in_space — By Space Scope
------------------------------------

.. code-block:: python

   hits = system.retrieve_in_space("refund policy", space_name="Support-UserA")

Only retrieves within the specified space. Suitable for precise queries when you know where the data is.

search() (Planned) — Most Flexible
------------------------------------

.. code-block:: python

   hits = system.search("refund policy", retriever_types=["dense"], use_graph_expansion=True)

Customizable retriever combinations and graph expansion. Suitable when you need to control the retrieval pipeline.

Selection Flowchart
-------------------

.. code-block::

   Know which space the data is in?
   ├── Yes → retrieve_in_space(query, space_name=X)
   └── No → Need to filter by type?
             ├── Yes → retrieve_by_view(query, view=X)
             └── No → holistic_retrieve(query)
