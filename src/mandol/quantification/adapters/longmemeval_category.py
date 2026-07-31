"""single-session-user: +1.5pp → AGGRESSIVE_CUT single-session-preference: 0.0pp → AGGRESSIVE_CUT single-session-assistant: -10.7pp → MODERATE_CUT knowledge-update: -17.9pp → MODERATE_CUT multi-session: -26.3pp → EVIDENCE_CHAIN temporal-reason."""

from ..adaptive_category import (
    EvidenceComplexity,
    register_category_mapping,
)

_LONGMEMEVAL_MAPPING = {
    "multi-session": EvidenceComplexity.EVIDENCE_CHAIN,
    "temporal-reasoning": EvidenceComplexity.EVIDENCE_CHAIN,
    "knowledge-update": EvidenceComplexity.MODERATE_CUT,
    "single-session-assistant": EvidenceComplexity.MODERATE_CUT,
    "single-session-user": EvidenceComplexity.AGGRESSIVE_CUT,
    "single-session-preference": EvidenceComplexity.AGGRESSIVE_CUT,
}

register_category_mapping("longmemeval", _LONGMEMEVAL_MAPPING)
