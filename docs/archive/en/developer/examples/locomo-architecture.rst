LoCoMo Architecture Details
===============================

LocomoMemorySystem wrapper layer design.

Design Intent
---------------

``LocomoMemorySystem`` is a business wrapper around ``MemorySystem``, providing data loading and memory building workflows specific to the LoCoMo dataset.

Core Classes
-------------

.. code-block:: python

   class LocomoMemorySystem:
       config: LocomoMemoryConfig
       system: MemorySystem

       def load_and_process_samples(self) -> dict:
           """Load LoCoMo JSON data and add as MemoryUnit"""
           pass

       def build_high_level_memories(self, mode="auto") -> BuildReport:
           """Build high-level memories"""
           pass

       def get_memory_stats(self) -> dict:
           """Statistics"""
           pass

       def search(self, query, top_k=5) -> list[SearchHit]:
           """Search using memory_system"""
           pass

       def run_query_set(self, queries) -> list[dict]:
           """Run preset query set"""
           pass

Wrapper Layer Pattern
-----------------------

.. mermaid::

   graph LR
       A[LocomoMemorySystem] -->|wraps| B[MemorySystem]
       A -->|configures| C[LocomoMemoryConfig]
       B --> D[SemanticMapService]
       B --> E[SemanticGraphService]
       B --> F[SessionManager]

This wrapper pattern is suitable for:
- Specific datasets with fixed processing workflows
- Need for predefined query evaluation sets
- Providing simplified APIs for specific business scenarios

Provider Switching
--------------------

``LocomoMemorySystem`` can switch providers via config during construction:

.. code-block:: python

   config = LocomoMemoryConfig(
       llm_provider="openai",
       embedder_provider="sentence_transformers",
   )
   system = LocomoMemorySystem(config=config)
