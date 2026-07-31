import json
from typing import Optional
from .base import BaseProvider

class ReasoningProvider(BaseProvider):
    
    def generate(self, 
                 prompt: str, 
                 max_tokens: Optional[int] = None,
                 temperature: float = 0.1,
                 generate_strategy: str = "default",
                 json_format: bool = False,
                 **kwargs) -> str:
        
        if max_tokens is None:
            if generate_strategy == "max":
                max_tokens = self.max_output_tokens
            else:
                max_tokens = self.default_output_tokens
        else:
            max_tokens = min(max_tokens, self.max_output_tokens)
        
        prompt_tokens = self.count_tokens(prompt)
        max_prompt_tokens = self.context_length - max_tokens - 100
        
        if prompt_tokens > max_prompt_tokens:
            self.logger.warning(f"Prompt is too long ({prompt_tokens} > {max_prompt_tokens}); truncating.")
            prompt = self.truncate_context(prompt, max_prompt_tokens)
            prompt_tokens = self.count_tokens(prompt)
            self.logger.info(f"Prompt after truncation: {prompt_tokens} tokens")
            
        try:
            request_params = {
                "model": self.actual_model,
                "messages": [{"role": "user", "content": prompt}]
            }
            
            if temperature != 1.0:
                self.logger.warning(f"Model {self.model_name} is a reasoning model; ignoring temperature={temperature} and using 1.0")
            
            
            request_params["max_completion_tokens"] = max_tokens
            self.logger.debug(f"Using max_completion_tokens={max_tokens}")
            
            if "reasoning_effort" in kwargs:
                request_params["reasoning_effort"] = kwargs["reasoning_effort"]
                self.logger.debug(f"Using reasoning_effort={kwargs['reasoning_effort']}")
            
            
            if json_format and self.supports_json_format:
                request_params["response_format"] = {"type": "json_object"}
                if "json" not in prompt.lower():
                    prompt += "\n\nPlease respond in JSON format."
                    request_params["messages"][0]["content"] = prompt
                self.logger.debug("JSON response format enabled")
            
            extra_body = {**self.model_config.get("extra_body", {}), **kwargs.pop("extra_body", {})}
            if extra_body:
                request_params["extra_body"] = extra_body
                self.logger.debug(f" extra_body: {extra_body}")
            
            self.logger.debug(f"Sending reasoning request: {prompt_tokens} tokens -> max_completion {max_tokens}")
            
            response = self._create_chat_completion_with_retry(request_params)
            content = response.choices[0].message.content
            if content is None:
                answer = ""
            elif isinstance(content, str):
                answer = content.strip()
            else:
                answer = json.dumps(content, ensure_ascii=False)
            
            if hasattr(response, 'usage') and response.usage:
                usage = response.usage
                details = getattr(usage, "completion_tokens_details", None)
                reasoning_tokens = getattr(details, "reasoning_tokens", 0) if details else 0
                
                self.logger.debug(f"Token usage: input={usage.prompt_tokens}, "
                                f"output={usage.completion_tokens} (reasoning={reasoning_tokens}), "
                                f"total={usage.total_tokens}")
            else:
                estimated_output = self.count_tokens(answer)
                self.logger.debug(f"Estimated tokens: input~={prompt_tokens}, output~={estimated_output}")
                
            return answer
            
        except Exception as e:
            self.logger.error(f"Generation failed (reasoning): {e}")
            return f"Generation failed: {str(e)}"

    async def generate_async(self, 
                             prompt: str, 
                             max_tokens: Optional[int] = None,
                             temperature: float = 0.1,
                             generate_strategy: str = "default",
                             json_format: bool = False,
                             **kwargs) -> str:
        """Generate async."""
        if max_tokens is None:
            if generate_strategy == "max":
                max_tokens = self.max_output_tokens
            else:
                max_tokens = self.default_output_tokens
        else:
            max_tokens = min(max_tokens, self.max_output_tokens)
        
        prompt_tokens = self.count_tokens(prompt)
        max_prompt_tokens = self.context_length - max_tokens - 100
        
        if prompt_tokens > max_prompt_tokens:
            self.logger.warning(f"[Async] Prompt is too long ({prompt_tokens} > {max_prompt_tokens}); truncating.")
            prompt = self.truncate_context(prompt, max_prompt_tokens)
            prompt_tokens = self.count_tokens(prompt)
            self.logger.info(f"[Async] Prompt after truncation: {prompt_tokens} tokens")
            
        try:
            request_params = {
                "model": self.actual_model,
                "messages": [{"role": "user", "content": prompt}]
            }
            
            if temperature != 1.0:
                self.logger.warning(f"[Async] Model {self.model_name} is a reasoning model; ignoring temperature={temperature}")
            
            request_params["max_completion_tokens"] = max_tokens
            
            # reasoning_effort
            if "reasoning_effort" in kwargs:
                request_params["reasoning_effort"] = kwargs["reasoning_effort"]
            
            if json_format and self.supports_json_format:
                request_params["response_format"] = {"type": "json_object"}
                if "json" not in prompt.lower():
                    prompt += "\n\nPlease respond in JSON format."
                    request_params["messages"][0]["content"] = prompt
            
            extra_body = {**self.model_config.get("extra_body", {}), **kwargs.pop("extra_body", {})}
            if extra_body:
                request_params["extra_body"] = extra_body
                self.logger.debug(f" [Async] extra_body: {extra_body}")
            
            self.logger.debug(f"[Async] Sending reasoning request: {prompt_tokens} tokens -> max_completion {max_tokens}")
            
            response = await self._create_chat_completion_with_retry_async(request_params)
            content = response.choices[0].message.content
            if content is None:
                answer = ""
            elif isinstance(content, str):
                answer = content.strip()
            else:
                answer = json.dumps(content, ensure_ascii=False)
            
            if hasattr(response, 'usage') and response.usage:
                usage = response.usage
                details = getattr(usage, "completion_tokens_details", None)
                reasoning_tokens = getattr(details, "reasoning_tokens", 0) if details else 0
                
                self.logger.debug(f"[Async] Token usage: input={usage.prompt_tokens}, "
                                f"output={usage.completion_tokens} (reasoning={reasoning_tokens}), "
                                f"total={usage.total_tokens}")
            else:
                estimated_output = self.count_tokens(answer)
                self.logger.debug(f"[Async] Estimated tokens: input~={prompt_tokens}, output~={estimated_output}")
                
            return answer
            
        except Exception as e:
            self.logger.error(f"[Async] Generation failed (reasoning): {e}")
            return f"Generation failed: {str(e)}"
