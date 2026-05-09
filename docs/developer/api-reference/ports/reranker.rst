Reranker
==================

.. code-block:: python

   class Reranker(ABC):
       def rerank(
           self,
           query: str,
           units: list[MemoryUnit],
           top_k: int,
       ) -> list[tuple[MemoryUnit, float]]: ...

实现：``SentenceTransformerReranker``。
