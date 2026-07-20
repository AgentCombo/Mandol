Multi-View Retrieval Guide
============================

Decision tree for choosing among 8 views.

View Overview
-------------

.. list-table::
   :header-rows: 1
   :widths: 22 28 50

   * - View
     - Retrieved Content
     - Suitable Question Types
   * - ``base_memory``
     - Raw conversations
     - "What was discussed last time" "What were the exact words"
   * - ``knowledge``
     - Knowledge summaries
     - "What does the system know about X"
   * - ``entity_relation``
     - Entity nodes
     - "Who knows whom" "What does Zhang San do"
   * - ``event_causal``
     - Event causality
     - "What happened" "Why"
   * - ``episodic``
     - Episodic summaries
     - "What happened during that period"
   * - ``emotional``
     - Emotional summaries
     - "How is the user feeling"
   * - ``procedural``
     - Procedural summaries
     - "How to do it" "What's the process"
   * - ``insights``
     - Insights
     - "What deep patterns are there"

Selection Decision Tree
------------------------

.. code-block::

   What do you want to know?
   ├── Exact words from original conversation
   │   └── view="base_memory"
   ├── Knowledge about a topic
   │   └── view="knowledge"
   ├── Information about people/places/concepts
   │   └── view="entity_relation"
   ├── Events and their causes/effects
   │   └── view="event_causal"
   ├── Summary of a time period
   │   └── view="episodic"
   ├── User's emotional attitudes
   │   └── view="emotional"
   ├── Operation steps/processes
   │   └── view="procedural"
   └── Deep patterns/regularities
       └── view="insights"
