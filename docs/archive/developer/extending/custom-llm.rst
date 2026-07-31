自定义 LLM Provider
========================

实现 ``LLMProvider`` 接口。

.. code-block:: python

   from mandol.ports.llm_provider import LLMProvider, ChatMessage, ChatResponse

   class MyCustomLLM(LLMProvider):
       def chat(
           self,
           messages: list[ChatMessage],
           temperature: float = 0.1,
           max_tokens: int = 1024,
           **kwargs,
       ) -> ChatResponse:
           prompt = format_messages(messages)
           raw = your_llm_api_call(prompt, temperature, max_tokens)
           return ChatResponse(content=raw)

注入方式
--------

.. code-block:: python

   system = MemorySystem(llm_provider=MyCustomLLM())
