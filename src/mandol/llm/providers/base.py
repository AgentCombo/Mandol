import os
import logging
from ...utils.logging_config import create_module_logger
import re
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
import httpx
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    OpenAI,
    RateLimitError,
)
from tenacity import (
    AsyncRetrying,
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)


DEFAULT_REQUEST_MAX_RETRIES = 5
DEFAULT_REQUEST_TIMEOUT_SECONDS = 120.0
DEFAULT_CONNECT_TIMEOUT_SECONDS = 30.0
RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logging.warning("tiktoken is not installed; falling back to approximate token counting. Install with: pip install tiktoken")

class BaseProvider(ABC):
    def __init__(self, 
                 model_name: str, 
                 model_config: Dict[str, Any], 
                 provider_config: Dict[str, Any],
                 api_key: str, 
                 base_url: str,
                 max_context_ratio: float = 0.85,
                 request_max_retries: int = DEFAULT_REQUEST_MAX_RETRIES,
                 request_timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS):
        
        self.logger = create_module_logger("llm.providers.base.BaseProvider")
        self.model_name = model_name
        self.model_config = model_config
        self.provider_config = provider_config
        self.api_key = api_key
        self.base_url = base_url
        self.max_context_ratio = max_context_ratio
        configured_retries = self.model_config.get("request_max_retries", request_max_retries)
        try:
            self.request_max_retries = max(0, int(configured_retries))
        except (TypeError, ValueError):
            self.request_max_retries = DEFAULT_REQUEST_MAX_RETRIES
        configured_timeout = self.model_config.get("request_timeout", request_timeout)
        try:
            self.request_timeout = max(1.0, float(configured_timeout))
        except (TypeError, ValueError):
            self.request_timeout = DEFAULT_REQUEST_TIMEOUT_SECONDS
        self._client_timeout = httpx.Timeout(
            timeout=self.request_timeout,
            connect=min(DEFAULT_CONNECT_TIMEOUT_SECONDS, self.request_timeout),
            read=self.request_timeout,
            write=min(DEFAULT_CONNECT_TIMEOUT_SECONDS, self.request_timeout),
            pool=min(DEFAULT_CONNECT_TIMEOUT_SECONDS, self.request_timeout),
        )
        
        self.actual_model = self.model_config.get("actual_model", model_name)
        
        self.context_length = self.model_config["context_length"]
        self.max_output_tokens = self.model_config["max_output"]
        self.default_output_tokens = self.model_config["default_output"]
        self.max_context_tokens = int(self.context_length * max_context_ratio)
        
        self.tokenizer = self._init_tokenizer()
        self.client = self._init_client()
        self.async_client = self._init_async_client()
        
        self.supports_json_format = self.provider_config["supports_json_format"]
        self._log_initialization()

    def _init_tokenizer(self):
        """Initialize tokenizer."""
        if not TIKTOKEN_AVAILABLE:
            return None
        
        try:
            encoding_name = self.model_config.get("encoding", "cl100k_base")
            tokenizer = tiktoken.get_encoding(encoding_name)
            self.logger.debug(f"Using tiktoken encoding: {encoding_name}")
            return tokenizer
        except Exception as e:
            self.logger.warning(f"tiktoken initialization failed: {e}; using approximate counting")
            return None
    
    def _init_client(self):
        """Initialize client."""
        try:
            
            if "openrouter" in self.base_url:
                client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    max_retries=0,
                    timeout=self._client_timeout,
                    default_headers={
                        "HTTP-Referer": "https://github.com/AgentCombo/Mandol",
                        "X-Title": "Mandol"
                    }
                )
                self.logger.debug(f"OpenRouter client initialized: {self.base_url}")
            else:
                client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    max_retries=0,
                    timeout=self._client_timeout
                )
                self.logger.debug(f"OpenAI client initialized: {self.base_url}")
            return client
        except Exception as e:
            self.logger.error(f"OpenAI client initialization failed: {e}")
            raise

    def _init_async_client(self):
        """Initialize async client."""
        try:
            if "openrouter" in self.base_url:
                async_client = AsyncOpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    max_retries=0,
                    timeout=self._client_timeout,
                    default_headers={
                        "HTTP-Referer": "https://github.com/AgentCombo/Mandol",
                        "X-Title": "Mandol"
                    }
                )
                self.logger.debug(f"AsyncOpenAI client initialized for OpenRouter: {self.base_url}")
            else:
                async_client = AsyncOpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    max_retries=0,
                    timeout=self._client_timeout
                )
                self.logger.debug(f"AsyncOpenAI client initialized: {self.base_url}")
            return async_client
        except Exception as e:
            self.logger.error(f"AsyncOpenAI client initialization failed: {e}")
            raise

    def _is_retryable_llm_error(self, exc: BaseException) -> bool:
        """Run is retryable LLM error."""
        if isinstance(
            exc,
            (APIConnectionError, APITimeoutError, RateLimitError, TimeoutError, ConnectionError),
        ):
            return True

        status_code = getattr(exc, "status_code", None)
        if isinstance(exc, APIStatusError):
            return status_code in RETRYABLE_STATUS_CODES

        exc_name = exc.__class__.__name__.lower()
        transient_markers = (
            "timeout",
            "connection",
            "ratelimit",
            "rate_limit",
            "temporary",
            "server",
            "overloaded",
        )
        if any(marker in exc_name for marker in transient_markers):
            return status_code is None or status_code in RETRYABLE_STATUS_CODES

        return False

    def _log_retry_sleep(self, retry_state) -> None:
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        sleep_seconds = retry_state.next_action.sleep if retry_state.next_action else 0.0
        exc_name = exc.__class__.__name__ if exc else "UnknownError"
        self.logger.warning(
            "LLM request failed; retrying in %.2fs (%s/%s): %s: %s",
            sleep_seconds,
            retry_state.attempt_number,
            self.request_max_retries,
            exc_name,
            exc,
        )

    def _retry_config(self) -> Dict[str, Any]:
        return {
            "stop": stop_after_attempt(self.request_max_retries + 1),
            "wait": wait_random_exponential(multiplier=1, min=1, max=30),
            "retry": retry_if_exception(self._is_retryable_llm_error),
            "before_sleep": self._log_retry_sleep,
            "reraise": True,
        }

    def _create_chat_completion_with_retry(self, request_params: Dict[str, Any]) -> Any:
        for attempt in Retrying(**self._retry_config()):
            with attempt:
                return self.client.chat.completions.create(**request_params)
        raise RuntimeError("LLM retry loop terminated unexpectedly")

    async def _create_chat_completion_with_retry_async(self, request_params: Dict[str, Any]) -> Any:
        async for attempt in AsyncRetrying(**self._retry_config()):
            with attempt:
                return await self.async_client.chat.completions.create(**request_params)
        raise RuntimeError("Async LLM retry loop terminated unexpectedly")

    def _log_initialization(self):
        """Log initialization."""
        self.logger.info(f"Initializing provider: {self.__class__.__name__}")
        self.logger.info(f"   Model: {self.model_name} (type: {self.model_config.get('model_type', 'standard')})")
        self.logger.info(f"   API base URL: {self.base_url}")
        self.logger.info(f"   Context length: {self.context_length:,}")
        self.logger.info(f"   Request retries: {self.request_max_retries} with exponential backoff")
        self.logger.info(f"   Request timeout: {self.request_timeout:.1f}s")

    def count_tokens(self, text: str) -> int:
        """Count tokens."""
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception:
                pass
        
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        other_chars = len(text) - chinese_chars - english_chars
        
        estimated_tokens = int(
            chinese_chars * 0.6 +
            english_chars * 0.3 +
            other_chars * 0.4
        )
        
        return max(estimated_tokens, 1)

    def truncate_context(self, context_text: str, max_tokens: Optional[int] = None) -> str:
        """Run truncate context."""
        if max_tokens is None:
            max_tokens = self.max_context_tokens
        
        current_tokens = self.count_tokens(context_text)
        
        if current_tokens <= max_tokens:
            return context_text
        
        lines = context_text.split('\n')
        selected_lines = []
        current_tokens = 0
        
        for line in reversed(lines):
            line_tokens = self.count_tokens(line + '\n')
            if current_tokens + line_tokens <= max_tokens:
                selected_lines.insert(0, line)
                current_tokens += line_tokens
            else:
                break
        
        result = '\n'.join(selected_lines)
        final_tokens = self.count_tokens(result)
        
        if final_tokens != current_tokens:
            self.logger.info(f"Context truncated: {self.count_tokens(context_text)} -> {final_tokens} tokens")
        
        return result

    @abstractmethod
    def generate(self, 
                 prompt: str, 
                 max_tokens: Optional[int] = None,
                 temperature: float = 0.1,
                 generate_strategy: str = "default",
                 json_format: bool = False,
                 **kwargs) -> str:
        """Generate."""
        pass

    @abstractmethod
    async def generate_async(self, 
                             prompt: str, 
                             max_tokens: Optional[int] = None,
                             temperature: float = 0.1,
                             generate_strategy: str = "default",
                             json_format: bool = False,
                             **kwargs) -> str:
        """Generate async."""
        pass
