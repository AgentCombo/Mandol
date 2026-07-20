LLMProvider
===============

.. code-block:: python

   class LLMProvider(ABC):
       def chat(
           self,
           messages: list[ChatMessage],
           temperature: float = 0.1,
           max_tokens: int = 1024,
       ) -> ChatResponse: ...

Implementation: ``OpenAICompatibleLLMProvider``.
