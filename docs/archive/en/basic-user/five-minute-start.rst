5-Minute Quick Start
=====================

Two modes — pick one, copy and run.

Mode 1: Remote API (Recommended)
---------------------------------

**Prerequisite**: ``OPENAI_API_KEY`` environment variable and ``config.yaml`` are configured.

.. code-block:: python

   from mandol import MemorySystem, MemoryUnit, Uid

   system = MemorySystem.from_yaml_config("config.yaml")

   system.add(MemoryUnit(
       uid=Uid("msg_1"),
       raw_data={"text_content": "Zhang San went to Beijing on a business trip today"},
       metadata={"timestamp": "2024-01-15T10:00:00"},
   ))
   system.add(MemoryUnit(
       uid=Uid("msg_2"),
       raw_data={"text_content": "Li Si said they're going to Shanghai for a meeting next week"},
       metadata={"timestamp": "2024-01-15T10:05:00"},
   ))

   system.build_high_level(mode="auto")

   hits = system.holistic_retrieve("Where did Zhang San go?", top_k=5)
   for hit in hits:
       print(f"[{hit.final_score:.3f}] {hit.unit.raw_data['text_content']}")

**Expected output**:

.. code-block::

   [0.947] Zhang San went to Beijing on a business trip today

Get a natural language response:

.. code-block:: python

   answer = system.ask("Where did Zhang San go?")
   print(answer)

**Expected output**:

.. code-block::

   According to the memory records, Zhang San went to Beijing on a business trip today.

**Save and restore**:

.. code-block:: python

   system.save("./memory_snapshot")
   system2 = MemorySystem.load("./memory_snapshot")

Mode 2: Local Models (No API Key Needed)
-----------------------------------------

.. code-block:: bash

   pip install mandol[sentence-transformers]

.. code-block:: python

   from mandol import MemorySystem, MemoryUnit, Uid

   system = MemorySystem()

   system.add(MemoryUnit(
       uid=Uid("msg_1"),
       raw_data={"text_content": "Zhang San went to Beijing on a business trip today"},
       metadata={"timestamp": "2024-01-15T10:00:00"},
   ))
   system.add(MemoryUnit(
       uid=Uid("msg_2"),
       raw_data={"text_content": "Li Si said they're going to Shanghai for a meeting next week"},
       metadata={"timestamp": "2024-01-15T10:05:00"},
   ))

   system.build_high_level(mode="auto")

   hits = system.holistic_retrieve("Where did Zhang San go?", top_k=5)
   for hit in hits:
       print(f"[{hit.final_score:.3f}] {hit.unit.raw_data['text_content']}")

**Expected output**:

.. code-block::

   [0.947] Zhang San went to Beijing on a business trip today

Get a natural language response:

.. code-block:: python

   answer = system.ask("Where did Zhang San go?")
   print(answer)

.. note::

   First run requires downloading models (~2-4 GB), ensure a stable internet connection. Subsequent runs use cached models, no repeated downloads needed.

Three Key Methods
-----------------

.. list-table::
   :widths: 60 40

   * - Method
     - Purpose
   * - ``system.add(unit)``
     - Add a memory
   * - ``system.build_high_level(mode="auto")``
     - Let the system digest and organize
   * - ``system.holistic_retrieve(query)``
     - Retrieve relevant memories (returns a list of SearchHit)
   * - ``system.ask(query)``
     - Ask in natural language, get a natural language response

Quick Health Check
------------------

Check system status at any time with one line:

.. code-block:: python

   print(system.monitor)

Example output:

.. code-block::

   [MemSys] units=2 | spaces=1 | graph:2n/0e | idx:2↑/0↓ | pend:0u/0e/0et | sess:1(avg2) | mem:52.3MB | CLEAN

.. important::

   After ``add()``, the system builds high-level memories asynchronously, but with small amounts of data it may not complete immediately. Manually calling ``build_high_level()`` ensures high-level memories (entities/events/summaries) are available right away. If you only need to retrieve raw conversations (BASE group), no waiting is needed.

Next Steps
----------

- :doc:`your-first-memory` — Complete end-to-end flow (including save and load)
- :doc:`configuration-simple` — The only 4 configurations you need to know
