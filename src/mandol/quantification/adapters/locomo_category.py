"""Cat4 SimpleFact: -1.6pp → AGGRESSIVE_CUT Cat2 Temporal: -3.4pp → MODERATE_CUT Cat1 Aggregation: -4.9pp → EVIDENCE_CHAIN Cat5 Adversarial: -7.6pp → EVIDENCE_CHAIN."""

from ..adaptive_category import (
    EvidenceComplexity,
    register_category_mapping,
)

_LOCOMO_MAPPING = {
    "1": EvidenceComplexity.EVIDENCE_CHAIN,
    "aggregation": EvidenceComplexity.EVIDENCE_CHAIN,
    "5": EvidenceComplexity.EVIDENCE_CHAIN,
    "adversarial": EvidenceComplexity.EVIDENCE_CHAIN,
    "2": EvidenceComplexity.MODERATE_CUT,
    "temporal": EvidenceComplexity.MODERATE_CUT,
    "3": EvidenceComplexity.AGGRESSIVE_CUT,
    "open_domain": EvidenceComplexity.AGGRESSIVE_CUT,
    "opendomain": EvidenceComplexity.AGGRESSIVE_CUT,
    "4": EvidenceComplexity.AGGRESSIVE_CUT,
    "simple_fact": EvidenceComplexity.AGGRESSIVE_CUT,
    "simplefact": EvidenceComplexity.AGGRESSIVE_CUT,
}

register_category_mapping("locomo", _LOCOMO_MAPPING)
