"""Adaptive cascade-pruning parameters by dataset category."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

from ..utils.logging_config import create_module_logger

logger = create_module_logger("quantification.adaptive_category")


class EvidenceComplexity(Enum):
    """Evidence-complexity buckets used by adaptive cascade pruning."""

    AGGRESSIVE_CUT = "AGGRESSIVE_CUT"
    """Single-fact categories where the answer is concentrated in a few chunks."""

    MODERATE_CUT = "MODERATE_CUT"
    """Moderate evidence categories requiring several complementary chunks."""

    EVIDENCE_CHAIN = "EVIDENCE_CHAIN"
    """Evidence-chain categories where retrieval must preserve long-tail context."""


@dataclass(frozen=True)
class AdaptiveParams:
    """Cascade-pruning parameters for one evidence-complexity bucket."""

    complexity: EvidenceComplexity
    cliff_tolerance_multiplier: float
    gradient_threshold: float
    min_chunks: int
    budget_fraction: float
    max_stage2_drops: Optional[int] = None



_PARAMS_BY_COMPLEXITY: Dict[EvidenceComplexity, AdaptiveParams] = {
    EvidenceComplexity.AGGRESSIVE_CUT: AdaptiveParams(
        complexity=EvidenceComplexity.AGGRESSIVE_CUT,
        cliff_tolerance_multiplier=0.6,
        gradient_threshold=1.2,
        min_chunks=2,
        budget_fraction=0.80,
        max_stage2_drops=None,
    ),
    EvidenceComplexity.MODERATE_CUT: AdaptiveParams(
        complexity=EvidenceComplexity.MODERATE_CUT,
        cliff_tolerance_multiplier=1.0,
        gradient_threshold=1.8,
        min_chunks=4,
        budget_fraction=0.90,
        max_stage2_drops=5,
    ),
    EvidenceComplexity.EVIDENCE_CHAIN: AdaptiveParams(
        complexity=EvidenceComplexity.EVIDENCE_CHAIN,
        cliff_tolerance_multiplier=1.8,
        gradient_threshold=2.5,
        min_chunks=6,
        budget_fraction=1.0,
        max_stage2_drops=None,
    ),
}

_DEFAULT_PARAMS = _PARAMS_BY_COMPLEXITY[EvidenceComplexity.MODERATE_CUT]


# Dataset-specific handling used by the reproduction workflow.
_CATEGORY_REGISTRY: Dict[str, Dict[str, EvidenceComplexity]] = {}


def register_category_mapping(
    dataset: str,
    mapping: Dict[str, EvidenceComplexity],
) -> None:
    """Parameters dataset : str mapping : Dict[str, EvidenceComplexity]."""
    _CATEGORY_REGISTRY[dataset.lower()] = mapping
    logger.info(
        f"[AdaptiveCategory] registered {dataset} category mapping: "
        f"{len(mapping)} categories -> {set(c.value for c in mapping.values())}"
    )


def get_adaptive_params(
    query_category: Optional[str] = None,
    dataset: Optional[str] = None,
) -> AdaptiveParams:
    """Parameters query_category : Optional[str] dataset : Optional[str] Returns AdaptiveParams."""
    if query_category is None:
        return _DEFAULT_PARAMS

    cat_str = str(query_category).strip().lower()

    # Dataset-specific handling used by the reproduction workflow.
    if dataset:
        ds_key = dataset.lower()
        if ds_key in _CATEGORY_REGISTRY:
            mapping = _CATEGORY_REGISTRY[ds_key]
            if cat_str in mapping:
                complexity = mapping[cat_str]
                params = _PARAMS_BY_COMPLEXITY[complexity]
                logger.debug(
                    f"[AdaptiveParams] {dataset}/{cat_str} -> {complexity.value}"
                )
                return params

    for ds_name, mapping in _CATEGORY_REGISTRY.items():
        if cat_str in mapping:
            complexity = mapping[cat_str]
            params = _PARAMS_BY_COMPLEXITY[complexity]
            logger.debug(
                f"[AdaptiveParams] {cat_str} (found in {ds_name}) -> {complexity.value}"
            )
            return params

    # Dataset-specific handling used by the reproduction workflow.
    complexity = _keyword_fallback(cat_str)
    if complexity is not None:
        params = _PARAMS_BY_COMPLEXITY[complexity]
        logger.debug(
            f"[AdaptiveParams] {cat_str} (keyword fallback) -> {complexity.value}"
        )
        return params

    logger.debug(
        f"[AdaptiveParams] unknown category '{query_category}'; using default MODERATE_CUT"
    )
    return _DEFAULT_PARAMS


def _keyword_fallback(cat_str: str) -> Optional[EvidenceComplexity]:
    """Run keyword fallback."""
    evidence_chain_keywords = [
        "multi-session", "multi_session", "multisession",
        "temporal-reasoning", "temporal_reasoning",
        "aggregation", "adversarial",
    ]
    for kw in evidence_chain_keywords:
        if kw in cat_str:
            return EvidenceComplexity.EVIDENCE_CHAIN

    aggressive_keywords = [
        "single-session-user", "single-session-preference",
        "single_session_user", "single_session_preference",
        "simple_fact", "simplefact", "open_domain", "opendomain",
    ]
    for kw in aggressive_keywords:
        if kw in cat_str:
            return EvidenceComplexity.AGGRESSIVE_CUT

    moderate_keywords = [
        "knowledge-update", "knowledge_update",
        "single-session-assistant", "single_session_assistant",
        "temporal",
    ]
    for kw in moderate_keywords:
        if kw in cat_str:
            return EvidenceComplexity.MODERATE_CUT

    return None
