"""Package exports for llm."""

from typing import Optional, Dict, Any

from .llm_client import (
    LLMClient,
    create_llm_client,
    create_deepseek_client,
    estimate_tokens,
    MODEL_CONFIGS,
    API_PROVIDERS
)

from .local_llm_client import LocalLLMClient


def create_openai_client(
    api_key: Optional[str] = None,
    model_name: str = "gpt-4o",
    **kwargs
) -> LLMClient:
    """Build openai client."""
    return LLMClient(
        model_name=model_name,
        api_key=api_key,
        **kwargs
    )

def create_claude_client(
    api_key: Optional[str] = None,
    model_name: str = "claude-3-5-sonnet-20241022",
    **kwargs
) -> LLMClient:
    """Build claude client."""
    return LLMClient(
        model_name=model_name,
        api_key=api_key,
        **kwargs
    )


def get_llm_status() -> Dict[str, Any]:
    """Return llm status."""
    return {
        "available_providers": list(API_PROVIDERS.keys()),
        "available_models": list(MODEL_CONFIGS.keys()),
        "engine_version": "2.0 (Refactored)",
        "cache_advisor_available": False,
        "batch_processor_available": False
    }

deepseek = create_deepseek_client
openai = create_openai_client
claude = create_claude_client

__all__ = [
    # Core
    'LLMClient',
    # Configs
    'MODEL_CONFIGS',
    'API_PROVIDERS',
    
    # Factories
    'create_llm_client',
    'create_deepseek_client',
    'create_openai_client',
    'create_claude_client',
    'estimate_tokens',
    
    # Helpers
    'get_llm_status',
    
    # Aliases
    'deepseek',
    'openai',
    'claude', 
    'LocalLLMClient'
]
