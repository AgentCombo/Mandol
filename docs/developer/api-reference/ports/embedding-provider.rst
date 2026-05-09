EmbeddingProvider
==================

.. code-block:: python

   class EmbeddingProvider(ABC):
       def embedding_dim(self) -> int: ...

       def embed_text(self, texts: list[str]) -> list[np.ndarray]: ...

       def embed_image_paths(self, paths: list[str]) -> list[np.ndarray]: ...

实现：``SentenceTransformersEmbedder``、``OpenAIEmbedder``。
