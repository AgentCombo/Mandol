Custom Embedder
==================

Implement the ``EmbeddingProvider`` interface.

.. code-block:: python

   from mandol.ports.embedding_provider import EmbeddingProvider
   import numpy as np

   class MyCustomEmbedder(EmbeddingProvider):
       def embedding_dim(self) -> int:
           return 768

       def embed_text(self, texts: list[str]) -> list[np.ndarray]:
           vectors = []
           for text in texts:
               vec = your_encode_function(text)
               vectors.append(np.array(vec, dtype=np.float32))
           return vectors

       def embed_image_paths(self, paths: list[str]) -> list[np.ndarray]:
           raise NotImplementedError("Image not supported")

Injection
----------

.. code-block:: python

   from mandol import MemorySystem

   system = MemorySystem(embedder=MyCustomEmbedder())

   # Or replace at runtime
   system.semantic_map.set_embedder(MyCustomEmbedder())
