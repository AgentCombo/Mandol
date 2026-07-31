from typing import Dict, Any

from ..utils.config_manager import settings

API_PROVIDERS = {
    "deepseek": {
        "base_url": settings.deepseek_base_url,
        "api_key_env": "DEEPSEEK_API_KEY",
        "supports_json_format": True
    },
    "cstcloud":{
        "base_url": settings.cstcloud_base_url,
        "api_key_env": "CSTCLOUD_API_KEY",
        "supports_json_format": True
    },
    "openai": {
        "base_url": settings.openai_base_url,
        "api_key_env": "OPENAI_API_KEY",
        "supports_json_format": True
    },
    "openai-proxy": {
        "base_url": settings.closeai_base_url,
        "api_key_env": "CLOSEAI_API_KEY",
        "fallback_env": "OPENAI_API_KEY",
        "supports_json_format": True
    },
    "openrouter": {  
        "base_url": settings.openrouter_base_url,
        "api_key_env": "OPENROUTER_API_KEY",
        "supports_json_format": True
    },
    "dashscope": {
        "base_url": settings.dashscope_base_url,
        "api_key_env": "DASHSCOPE_API_KEY",
        "supports_json_format": True
    }
}

MODEL_CONFIGS = {
    "deepseek-v4-flash-thinking": {
        "provider": "deepseek",
        "context_length": 1000000,
        "max_output": 384000,
        "default_output": 32768,
        "encoding": "cl100k_base",
        "actual_model": "deepseek-v4-flash",
        "uses_max_completion_tokens": False,
        "supports_temperature": True,
        "model_type": "standard",
        "extra_body": {"thinking": {"type": "enabled"}}
    },
    "deepseek-v4-flash": {
        "provider": "deepseek",
        "context_length": 1000000,
        "max_output": 384000,
        "default_output": 8192,
        "encoding": "cl100k_base",
        "actual_model": "deepseek-v4-flash",
        "uses_max_completion_tokens": False,
        "supports_temperature": True,
        "model_type": "standard",
        "extra_body": {"thinking": {"type": "disabled"}}
    },
    "deepseek-v4-pro-thinking": {
        "provider": "deepseek",
        "context_length": 1000000,
        "max_output": 384000,
        "default_output": 32768,
        "encoding": "cl100k_base",
        "actual_model": "deepseek-v4-pro",
        "uses_max_completion_tokens": False,
        "supports_temperature": True,
        "model_type": "standard",
        "extra_body": {"thinking": {"type": "enabled"}}
    },
    "deepseek-v4-pro": {
        "provider": "deepseek",
        "context_length": 1000000,
        "max_output": 384000,
        "default_output": 8192,
        "encoding": "cl100k_base",
        "actual_model": "deepseek-v4-pro",
        "uses_max_completion_tokens": False,
        "supports_temperature": True,
        "model_type": "standard",
        "extra_body": {"thinking": {"type": "disabled"}}
    },
    
    "deepseek-r1:671b-0528": {
        "provider": "cstcloud",
        "context_length": 65536,
        "max_output": 8192,
        "default_output": 8192,
        "encoding": "cl100k_base",
        "uses_max_completion_tokens": False,
        "supports_temperature": True,
        "model_type": "standard"
    },
    "deepseek-v3:671b": {
        "provider": "cstcloud",
        "context_length": 65536,
        "max_output": 8192,
        "default_output": 4096,
        "encoding": "cl100k_base",
        "uses_max_completion_tokens": False,
        "supports_temperature": True,
        "model_type": "standard"
    },
    
    "gpt-4o-mini": {
        "provider": "openai",
        "context_length": 128000,
        "max_output": 16384,
        "default_output": 8192,
        "encoding": "o200k_base",
        "uses_max_completion_tokens": False,
        "supports_temperature": True,
        "model_type": "standard"
    },
    "gpt-4o": {
        "provider": "openai",
        "context_length": 128000,
        "max_output": 16384,
        "default_output": 8192,
        "encoding": "o200k_base",
        "uses_max_completion_tokens": False,
        "supports_temperature": True,
        "model_type": "standard"
    },
    "gpt-4-turbo": {
        "provider": "openai",
        "context_length": 128000,
        "max_output": 4096,
        "default_output": 2048,
        "encoding": "o200k_base",
        "uses_max_completion_tokens": False,
        "supports_temperature": True,
        "model_type": "standard"
    },
    
    "gpt-4o-mini-closeai": {
        "provider": "openai-proxy",
        "context_length": 128000,
        "max_output": 16384,
        "default_output": 8192,
        "encoding": "o200k_base",
        "actual_model": "gpt-4o-mini",
        "uses_max_completion_tokens": False,
        "supports_temperature": True,
        "model_type": "standard"
    },
    "gpt-4o-closeai": {
        "provider": "openai-proxy",
        "context_length": 128000,
        "max_output": 16384,
        "default_output": 8192,
        "encoding": "o200k_base",
        "actual_model": "gpt-4o",
        "uses_max_completion_tokens": False,
        "supports_temperature": True,
        "model_type": "standard"
    },
    "gpt-4.1-nano-closeai": {
        "provider": "openai-proxy",
        "context_length": 1047576,
        "max_output": 32768,
        "default_output": 8192,
        "encoding": "o200k_base",
        "actual_model": "gpt-4.1-nano",
        "uses_max_completion_tokens": False,
        "supports_temperature": True,
        "model_type": "standard"
    },
    "gpt-4.1-mini-closeai": {
        "provider": "openai-proxy",
        "context_length": 1047576,
        "max_output": 32768,
        "default_output": 8192,
        "encoding": "o200k_base",
        "actual_model": "gpt-4.1-mini",
        "uses_max_completion_tokens": False,
        "supports_temperature": True,
        "model_type": "standard"
    },
    "gpt-5-nano-closeai": {
        "provider": "openai-proxy",
        "context_length": 400000,
        "max_output": 128000,
        "default_output": 8192,
        "encoding": "o200k_base",
        "actual_model": "gpt-5-nano",
        "uses_max_completion_tokens": True,
        "supports_temperature": False,
        "model_type": "reasoning"
    },
    "gpt-5-closeai": {
        "provider": "openai-proxy",
        "context_length": 400000,
        "max_output": 128000,
        "default_output": 8192,
        "encoding": "o200k_base",
        "actual_model": "gpt-5",
        "uses_max_completion_tokens": True,
        "supports_temperature": False,
        "model_type": "reasoning"
    },
    "o1-preview": {
        "provider": "openai",
        "context_length": 128000,
        "max_output": 32768,
        "default_output": 4096,
        "encoding": "o200k_base",
        "uses_max_completion_tokens": True,
        "supports_temperature": False,
        "model_type": "reasoning"
    },
    "o1-mini": {
        "provider": "openai",
        "context_length": 128000,
        "max_output": 65536,
        "default_output": 4096,
        "encoding": "o200k_base",
        "uses_max_completion_tokens": True,
        "supports_temperature": False,
        "model_type": "reasoning"
    },
    
    
    # DeepSeek V3.2 via OpenRouter
    "deepseek-v3.2-openrouter": {
        "provider": "openrouter",
        "context_length": 131072,
        "max_output": 8192,
        "default_output": 4096,
        "encoding": "cl100k_base",
        "actual_model": "deepseek/deepseek-chat",
        "uses_max_completion_tokens": False,
        "supports_temperature": True,
        "model_type": "standard"
    },
    # Claude haiku 4.5 via OpenRouter
    "claude-haiku-4.5-openrouter": {
        "provider": "openrouter",
        "context_length": 200000,
        "max_output": 8192,
        "default_output": 4096,
        "encoding": "cl100k_base",
        "actual_model": "anthropic/claude-haiku-4.5",
        "uses_max_completion_tokens": False,
        "supports_temperature": True,
        "model_type": "standard"
    },
    
    # Gemini 3 Flash via OpenRouter
    "gemini-3-flash-openrouter": {
        "provider": "openrouter",
        "context_length": 200000,
        "max_output": 8192,
        "default_output": 4096,
        "encoding": "cl100k_base",
        "actual_model": "google/gemini-3-flash-preview",
        "uses_max_completion_tokens": False,
        "supports_temperature": True,
        "model_type": "standard"
    },
    # GPT-4o-mini via OpenRouter
    "gpt-4o-mini-openrouter": {
        "provider": "openrouter",
        "context_length": 128000,
        "max_output": 16384,
        "default_output": 8192,
        "encoding": "o200k_base",
        "actual_model": "openai/gpt-4o-mini",
        "uses_max_completion_tokens": False,
        "supports_temperature": True,
        "model_type": "standard"
    },
    # GPT-4.1-mini via OpenRouter
    "gpt-4.1-mini-openrouter": {
        "provider": "openrouter",
        "context_length": 1047576,
        "max_output": 32768,
        "default_output": 8192,
        "encoding": "o200k_base",
        "actual_model": "openai/gpt-4.1-mini",
        "uses_max_completion_tokens": False,
        "supports_temperature": True,
        "model_type": "standard"
    },
    # GPT-5-nano via OpenRouter (Reasoning Model)
    "gpt-5-nano-openrouter": {
        "provider": "openrouter",
        "context_length": 400000,
        "max_output": 128000,
        "default_output": 8192,
        "encoding": "o200k_base",
        "actual_model": "openai/gpt-5-nano",
        "uses_max_completion_tokens": True,
        "supports_temperature": False,
        "model_type": "reasoning"
    },
    "gpt-5-openrouter": {
        "provider": "openrouter",
        "context_length": 400000,
        "max_output": 128000,
        "default_output": 8192,
        "encoding": "o200k_base",
        "actual_model": "openai/gpt-5",
        "uses_max_completion_tokens": True,
        "supports_temperature": False,
        "model_type": "reasoning"
    },
    
    "llama-free-openrouter": {
        "provider": "openrouter",
        "context_length": 8192,
        "max_output": 4096,
        "default_output": 2048,
        "encoding": "cl100k_base",
        "actual_model": "meta-llama/llama-3.3-70b-instruct:free",
        "uses_max_completion_tokens": False,
        "supports_temperature": True,
        "model_type": "standard"
    },
    
    "deepseek-v3.2-dashscope": {
        "provider": "dashscope",
        "context_length": 131072,
        "max_output": 8192,
        "default_output": 4096,
        "encoding": "cl100k_base",
        "actual_model": "deepseek-v3.2",
        "uses_max_completion_tokens": False,
        "supports_temperature": True,
        "model_type": "standard",
        "extra_body": {"enable_thinking": False}
    },
    
    "deepseek-v3.2-dashscope-thinking": {
        "provider": "dashscope",
        "context_length": 131072,
        "max_output": 65536,
        "default_output": 16384,
        "encoding": "cl100k_base",
        "actual_model": "deepseek-v3.2",
        "uses_max_completion_tokens": False,
        "supports_temperature": True,
        "model_type": "standard",
        "extra_body": {"enable_thinking": True}
    },

    
    "qwen-3.5-plus": {
        "provider": "dashscope",
        "context_length": 1000000,
        "max_output": 65536,
        "default_output": 8192,
        "encoding": "cl100k_base",
        "actual_model": "qwen3.5-plus",
        "uses_max_completion_tokens": False,
        "supports_temperature": True,
        "model_type": "standard",
        "extra_body": {"enable_thinking": False}
    },
    "qwen-3.5-plus-thinking": {
        "provider": "dashscope",
        "context_length": 1000000,
        "max_output": 65536,
        "default_output": 16384,
        "encoding": "cl100k_base",
        "actual_model": "qwen3.5-plus",
        "uses_max_completion_tokens": False,
        "supports_temperature": True,
        "model_type": "standard",
        "extra_body": {"enable_thinking": True}
    },
    
    "qwen-3.5-plus-2026-02-15": {
        "provider": "dashscope",
        "context_length": 1000000,
        "max_output": 65536,
        "default_output": 8192,
        "encoding": "cl100k_base",
        "actual_model": "qwen3.5-plus-2026-02-15",
        "uses_max_completion_tokens": False,
        "supports_temperature": True,
        "model_type": "standard",
        "extra_body": {"enable_thinking": False}
    },
    
    "qwen-3.5-plus-2026-02-15-thinking": {
        "provider": "dashscope",
        "context_length": 1000000,
        "max_output": 65536,
        "default_output": 16384,
        "encoding": "cl100k_base",
        "actual_model": "qwen3.5-plus-2026-02-15",
        "uses_max_completion_tokens": False,
        "supports_temperature": True,
        "model_type": "standard",
        "extra_body": {"enable_thinking": True}
    },
    
    "qwen-3-plus": {
        "provider": "dashscope",
        "context_length": 1000000,
        "max_output": 32768,
        "default_output": 8192,
        "encoding": "cl100k_base",
        "actual_model": "qwen-plus",
        "uses_max_completion_tokens": False,
        "supports_temperature": True,
        "model_type": "standard",
        "extra_body": {"enable_thinking": False}
    },
    
    "qwen-3-plus-thinking": {
        "provider": "dashscope",
        "context_length": 1000000,
        "max_output": 32768,
        "default_output": 16384,
        "encoding": "cl100k_base",
        "actual_model": "qwen-plus",
        "uses_max_completion_tokens": False,
        "supports_temperature": True,
        "model_type": "standard",
        "extra_body": {"enable_thinking": True}
    },
    "qwen-3-plus-latest": {
        "provider": "dashscope",
        "context_length": 1000000,
        "max_output": 32768,
        "default_output": 8192,
        "encoding": "cl100k_base",
        "actual_model": "qwen-plus-latest",
        "uses_max_completion_tokens": False,
        "supports_temperature": True,
        "model_type": "standard",
        "extra_body": {"enable_thinking": False}
    },
    "qwen-3-plus-latest-thinking": {
        "provider": "dashscope",
        "context_length": 1000000,
        "max_output": 32768,
        "default_output": 16384,
        "encoding": "cl100k_base",
        "actual_model": "qwen-plus-latest",
        "uses_max_completion_tokens": False,
        "supports_temperature": True,
        "model_type": "standard",
        "extra_body": {"enable_thinking": True}
    },
    
    "qwen-3-plus-2025-12-01": {
        "provider": "dashscope",
        "context_length": 1000000,
        "max_output": 32768,
        "default_output": 8192,
        "encoding": "cl100k_base",
        "actual_model": "qwen-plus-2025-12-01",
        "uses_max_completion_tokens": False,
        "supports_temperature": True,
        "model_type": "standard",
        "extra_body": {"enable_thinking": False}
    },
    "qwen-3-plus-2025-12-01-thinking": {
        "provider": "dashscope",
        "context_length": 1000000,
        "max_output": 32768,
        "default_output": 16384,
        "encoding": "cl100k_base",
        "actual_model": "qwen-plus-2025-12-01",
        "uses_max_completion_tokens": False,
        "supports_temperature": True,
        "model_type": "standard",
        "extra_body": {"enable_thinking": True}
    }
}