"""Package exports for quantification."""

import os
import logging
from ..utils.logging_config import create_module_logger
from typing import Optional, Union
from ..llm.llm_client import LLMClient

from .semantic_quantifier import SemanticQuantifier, create_semantic_quantifier
from .finetune_semantic_quantifier import FastFinetunedQuantifier

from .confidence_pruner import ConfidenceAwarePruner, CandidateChunk, PrunedChunk, PruneResult, ConfidenceLevel

from .cascade_pruner import (
    CascadeConfidencePruner, EnhancedCandidateChunk, CascadePruneResult,
    TowerSource, ConflictResolution, DisambiguationRecord,
    create_cascade_pruner, DEFAULT_TOWER_MIN_RATIO,
)

from .query_expander import QueryExpander, create_query_expander

from .quantifier_prompts import QUANTIFICATION_PROMPT
from .multi_query_prompts import OPTIMIZED_QUERY_EXPANSION_PROMPT, TARGETED_SEARCH_PROMPT

logger = create_module_logger("quantification")

MODEL_PATHS = {
    "Qwen/Qwen3-4B-locomo-4o": "qwen3-4B-Instruct-locomo-gpt-4o-mini-baai",
    "Qwen/Qwen3-4B-locomo-4.1": "qwen3-4B-Instruct-locomo-gpt-4.1-mini-baai",
    "Qwen/Qwen3-4B-longmemeval-4.1": "qwen3-4B-Instruct-longmemeval-gpt-4.1-mini-baai",
}

def _resolve_model_path(model_alias: str) -> str:
    """Resolve model path."""
    if model_alias in MODEL_PATHS:
        model_root = os.getenv(
            "MANDOLIN_MODEL_DIR",
            os.path.join(os.getcwd(), "finetune_model"),
        )
        return os.path.join(model_root, MODEL_PATHS[model_alias])
    return model_alias

def get_semantic_quantifier(
    model_alias: str = "Qwen/Qwen3-4B-locomo",
    device: Optional[str] = None,
    llm_client: Optional[LLMClient] = None
) -> Union[SemanticQuantifier, FastFinetunedQuantifier]:
    """Return semantic quantifier."""
    full_path = _resolve_model_path(model_alias)
    
    is_finetuned = "locomo" in model_alias or "longmemeval" in model_alias
    
    if is_finetuned:
        logger.info(f"[Quantifier] fine-tuned acceleration mode enabled: {model_alias}")
        return FastFinetunedQuantifier(model_source="local", model_name=full_path, device=device, llm_client=llm_client)
    else:
        logger.info(f"[Quantifier] general mode enabled: {model_alias}")
        return SemanticQuantifier(model_source="local", model_name=full_path, device=device, llm_client=llm_client)


try:
    import torch
    import transformers
    LOCAL_MODEL_AVAILABLE = True
except ImportError:
    LOCAL_MODEL_AVAILABLE = False
    logger.warning("torch or transformers is not installed; local model features are unavailable")

__all__ = [
    'SemanticQuantifier',
    'FastFinetunedQuantifier',
    'get_semantic_quantifier',
    
    'ConfidenceAwarePruner',
    'CandidateChunk',
    'PrunedChunk',
    'PruneResult',
    'ConfidenceLevel',
    
    'CascadeConfidencePruner',
    'EnhancedCandidateChunk',
    'CascadePruneResult',
    'TowerSource',
    'ConflictResolution',
    'DisambiguationRecord',
    'create_cascade_pruner',
    
    'QueryExpander',
    'create_query_expander',
    
    'QUANTIFICATION_PROMPT',
    'OPTIMIZED_QUERY_EXPANSION_PROMPT',
    'TARGETED_SEARCH_PROMPT',
    
    'MODEL_PATHS',
    
    'LOCAL_MODEL_AVAILABLE',
]

__version__ = "2.0.0"
