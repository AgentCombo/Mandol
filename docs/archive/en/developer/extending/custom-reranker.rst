Custom Reranker
==================

Implement the ``Reranker`` interface.

.. code-block:: python

   from mandol.ports.reranker import Reranker
   from mandol.domain.memory_unit import MemoryUnit

   class MyCustomReranker(Reranker):
       def rerank(
           self,
           query: str,
           units: list[MemoryUnit],
           top_k: int,
       ) -> list[tuple[MemoryUnit, float]]:
           scores = []
           for unit in units:
               score = your_rerank_function(query, unit.raw_data["text_content"])
               scores.append((unit, score))
           scores.sort(key=lambda x: x[1], reverse=True)
           return scores[:top_k]

Injection
----------

.. code-block:: python

   system = MemorySystem(reranker=MyCustomReranker())
   system.semantic_map.set_reranker(MyCustomReranker())
