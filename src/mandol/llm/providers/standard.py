import json
from typing import Optional
from .base import BaseProvider

class StandardProvider(BaseProvider):
    
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
        
        system_prompt = kwargs.pop("system_prompt", None)
        prompt_for_count = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        prompt_tokens = self.count_tokens(prompt_for_count)
        max_prompt_tokens = self.context_length - max_tokens - 100
        
        if prompt_tokens > max_prompt_tokens:
            self.logger.warning(f"Prompt is too long ({prompt_tokens} > {max_prompt_tokens}); truncating.")
            system_tokens = self.count_tokens(system_prompt) if system_prompt else 0
            prompt = self.truncate_context(prompt, max(1, max_prompt_tokens - system_tokens - 10))
            prompt_for_count = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            prompt_tokens = self.count_tokens(prompt_for_count)
            self.logger.info(f"Prompt after truncation: {prompt_tokens} tokens")
            
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            request_params = {
                "model": self.actual_model,
                "messages": messages
            }
            
            request_params["temperature"] = temperature
            self.logger.debug(f"Using temperature={temperature}")
            
            
            if self.model_config.get("uses_max_completion_tokens", False):
                request_params["max_completion_tokens"] = max_tokens
            else:
                request_params["max_tokens"] = max_tokens
            
            supported_params = ['frequency_penalty', 'presence_penalty', 'top_p', 'stop']
            for param in supported_params:
                if param in kwargs:
                    request_params[param] = kwargs[param]
            
            if json_format and self.supports_json_format:
                request_params["response_format"] = {"type": "json_object"}
                if "json" not in prompt.lower():
                    prompt += "\n\nPlease respond in JSON format."
                    request_params["messages"][-1]["content"] = prompt
                self.logger.debug("JSON response format enabled")
            elif json_format and not self.supports_json_format:
                self.logger.warning(f"Model {self.model_name} does not support JSON response format; ignoring json_format")
            
            extra_body = {**self.model_config.get("extra_body", {}), **kwargs.pop("extra_body", {})}
            if extra_body:
                request_params["extra_body"] = extra_body
                self.logger.debug(f" extra_body: {extra_body}")
            
            self.logger.debug(f"Sending standard request: {prompt_tokens} tokens -> max {max_tokens}")
            
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
                self.logger.debug(f"Token usage: input={usage.prompt_tokens}, "
                                f"output={usage.completion_tokens}, "
                                f"total={usage.total_tokens}")
            else:
                estimated_output = self.count_tokens(answer)
                self.logger.debug(f"Estimated tokens: input~={prompt_tokens}, output~={estimated_output}")
                
            return answer
            
        except Exception as e:
            self.logger.error(f"Generation failed (standard): {e}")
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
        
        system_prompt = kwargs.pop("system_prompt", None)
        prompt_for_count = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        prompt_tokens = self.count_tokens(prompt_for_count)
        max_prompt_tokens = self.context_length - max_tokens - 100
        
        if prompt_tokens > max_prompt_tokens:
            self.logger.warning(f"[Async] Prompt is too long ({prompt_tokens} > {max_prompt_tokens}); truncating.")
            system_tokens = self.count_tokens(system_prompt) if system_prompt else 0
            prompt = self.truncate_context(prompt, max(1, max_prompt_tokens - system_tokens - 10))
            prompt_for_count = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            prompt_tokens = self.count_tokens(prompt_for_count)
            self.logger.info(f"[Async] Prompt after truncation: {prompt_tokens} tokens")
            
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            request_params = {
                "model": self.actual_model,
                "messages": messages
            }
            
            request_params["temperature"] = temperature
            
            if self.model_config.get("uses_max_completion_tokens", False):
                request_params["max_completion_tokens"] = max_tokens
            else:
                request_params["max_tokens"] = max_tokens
            
            supported_params = ['frequency_penalty', 'presence_penalty', 'top_p', 'stop']
            for param in supported_params:
                if param in kwargs:
                    request_params[param] = kwargs[param]
            
            if json_format and self.supports_json_format:
                request_params["response_format"] = {"type": "json_object"}
                if "json" not in prompt.lower():
                    prompt += "\n\nPlease respond in JSON format."
                    request_params["messages"][-1]["content"] = prompt
            elif json_format and not self.supports_json_format:
                self.logger.warning(f"Model {self.model_name} does not support JSON response format; ignoring json_format")
            
            extra_body = {**self.model_config.get("extra_body", {}), **kwargs.pop("extra_body", {})}
            if extra_body:
                request_params["extra_body"] = extra_body
                self.logger.debug(f" [Async] extra_body: {extra_body}")
            
            self.logger.debug(f"[Async] Sending standard request: {prompt_tokens} tokens -> max {max_tokens}")
            
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
                self.logger.debug(f"[Async] Token usage: input={usage.prompt_tokens}, "
                                f"output={usage.completion_tokens}, "
                                f"total={usage.total_tokens}")
            else:
                estimated_output = self.count_tokens(answer)
                self.logger.debug(f"[Async] Estimated tokens: input~={prompt_tokens}, output~={estimated_output}")
                
            return answer
            
        except Exception as e:
            self.logger.error(f"[Async] Generation failed (standard): {e}")
            return f"Generation failed: {str(e)}"
