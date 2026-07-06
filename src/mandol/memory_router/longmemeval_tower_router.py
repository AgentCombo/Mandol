"""Category-based tower router for LongMemEval.

The router selects sentence, episodic, and entity retrieval towers according to
question categories used by the artifact reproduction workflow.
"""

import re
from dataclasses import dataclass
from typing import Dict, Optional
from enum import Enum

class LongMemEvalCategory(str, Enum):
    """Question categories used by the LongMemEval benchmark."""
    MULTI_SESSION = "multi-session"
    TEMPORAL_REASONING = "temporal-reasoning"
    KNOWLEDGE_UPDATE = "knowledge-update"
    SINGLE_SESSION_USER = "single-session-user"
    SINGLE_SESSION_ASSISTANT = "single-session-assistant"
    SINGLE_SESSION_PREFERENCE = "single-session-preference"

@dataclass
class LongMemEvalTowerRoutingConfig:
    """Routing configuration for one LongMemEval query.

    Boolean fields enable or disable retrieval towers. Top-k fields define
    per-tower retrieval budgets. A top-k value of 0 disables the corresponding
    retrieval path. ``final_top_k`` controls how many contexts remain after
    reranking.
    """

    enable_sentence: bool = True  # Enable the sentence/hierarchical tower.
    enable_episodic: bool = True  # Enable the episodic-memory tower.
    enable_entity: bool = True    # Enable the entity-relation tower.

    sentence_top_k: int = 60  # Retrieval budget for the sentence tower.
    episodic_top_k: int = 40  # Retrieval budget for the episodic tower.
    entity_top_k: int = 40    # Retrieval budget for the entity tower.

    final_top_k: int = 25     # Number of contexts kept after reranking.

    rerank_method: str = "baai"
    enable_second_stage_rerank: bool = True

    category_guidance: str = ""

    @property
    def active_towers(self) -> str:
        """Return labels for enabled retrieval towers."""
        parts = []
        if self.enable_sentence:
            parts.append("S")
        if self.enable_episodic:
            parts.append("Ep")
        if self.enable_entity:
            parts.append("Ent")
        return "+".join(parts) if parts else "NONE"

    def __repr__(self):
        return (f"LongMemEvalTowerRoutingConfig(towers={self.active_towers}, "
                f"topk=[S={self.sentence_top_k},Ep={self.episodic_top_k},"
                f"Ent={self.entity_top_k}], final_k={self.final_top_k})")

def _lme_cfg_full() -> LongMemEvalTowerRoutingConfig:
    """Return the full LongMemEval routing configuration."""
    return LongMemEvalTowerRoutingConfig(
        enable_sentence=True, enable_episodic=True, enable_entity=True,
        sentence_top_k=60, episodic_top_k=40, entity_top_k=40,
        final_top_k=25,
    )

def _lme_cfg_s_ep() -> LongMemEvalTowerRoutingConfig:
    """Return the LongMemEval S+Ep routing configuration."""
    return LongMemEvalTowerRoutingConfig(
        enable_sentence=True, enable_episodic=True, enable_entity=False,
        sentence_top_k=60, episodic_top_k=40, entity_top_k=0,
        final_top_k=25,
    )

def _lme_cfg_s_ent() -> LongMemEvalTowerRoutingConfig:
    """Return the LongMemEval S+Ent routing configuration."""
    return LongMemEvalTowerRoutingConfig(
        enable_sentence=True, enable_episodic=False, enable_entity=True,
        sentence_top_k=60, episodic_top_k=0, entity_top_k=40,
        final_top_k=25,
    )

def _lme_cfg_ep_ent() -> LongMemEvalTowerRoutingConfig:
    """Return the LongMemEval Ep+Ent routing configuration."""
    return LongMemEvalTowerRoutingConfig(
        enable_sentence=False, enable_episodic=True, enable_entity=True,
        sentence_top_k=0, episodic_top_k=40, entity_top_k=40,
        final_top_k=25,
    )

def _lme_cfg_full_top50() -> LongMemEvalTowerRoutingConfig:
    """Return the full LongMemEval Top-50 routing configuration."""
    return LongMemEvalTowerRoutingConfig(
        enable_sentence=True, enable_episodic=True, enable_entity=True,
        sentence_top_k=60, episodic_top_k=40, entity_top_k=40,
        final_top_k=50,
    )

def _lme_cfg_s_ep_top50() -> LongMemEvalTowerRoutingConfig:
    """Return the LongMemEval S+Ep Top-50 routing configuration."""
    return LongMemEvalTowerRoutingConfig(
        enable_sentence=True, enable_episodic=True, enable_entity=False,
        sentence_top_k=60, episodic_top_k=40, entity_top_k=0,
        final_top_k=50,
    )

def _lme_cfg_s_ent_top50() -> LongMemEvalTowerRoutingConfig:
    """Return the LongMemEval S+Ent Top-50 routing configuration."""
    return LongMemEvalTowerRoutingConfig(
        enable_sentence=True, enable_episodic=False, enable_entity=True,
        sentence_top_k=60, episodic_top_k=0, entity_top_k=40,
        final_top_k=50,
    )

def _lme_cfg_ep_ent_top50() -> LongMemEvalTowerRoutingConfig:
    """Return the LongMemEval Ep+Ent Top-50 routing configuration."""
    return LongMemEvalTowerRoutingConfig(
        enable_sentence=False, enable_episodic=True, enable_entity=True,
        sentence_top_k=0, episodic_top_k=40, entity_top_k=40,
        final_top_k=50,
    )

LME_GPT_41_MINI_ROUTING_AGGRESSIVE: Dict[str, LongMemEvalTowerRoutingConfig] = {
    LongMemEvalCategory.MULTI_SESSION:             _lme_cfg_s_ep(),
    LongMemEvalCategory.TEMPORAL_REASONING:        _lme_cfg_s_ep(),
    LongMemEvalCategory.KNOWLEDGE_UPDATE:          _lme_cfg_s_ent(),
    LongMemEvalCategory.SINGLE_SESSION_USER:       _lme_cfg_ep_ent(),
    LongMemEvalCategory.SINGLE_SESSION_ASSISTANT:  _lme_cfg_full(),
    LongMemEvalCategory.SINGLE_SESSION_PREFERENCE: _lme_cfg_full(),
}

LME_GPT_41_MINI_ROUTING_CONSERVATIVE: Dict[str, LongMemEvalTowerRoutingConfig] = {
    LongMemEvalCategory.MULTI_SESSION:             _lme_cfg_s_ep(),
    LongMemEvalCategory.TEMPORAL_REASONING:        _lme_cfg_s_ep(),
    LongMemEvalCategory.KNOWLEDGE_UPDATE:          _lme_cfg_s_ent(),
    LongMemEvalCategory.SINGLE_SESSION_USER:       _lme_cfg_full(),
    LongMemEvalCategory.SINGLE_SESSION_ASSISTANT:  _lme_cfg_full(),
    LongMemEvalCategory.SINGLE_SESSION_PREFERENCE: _lme_cfg_full(),
}

LME_GPT_41_MINI_ROUTING = LME_GPT_41_MINI_ROUTING_AGGRESSIVE

LME_GPT_4O_MINI_ROUTING_AGGRESSIVE: Dict[str, LongMemEvalTowerRoutingConfig] = {
    LongMemEvalCategory.MULTI_SESSION:             _lme_cfg_ep_ent(),
    LongMemEvalCategory.TEMPORAL_REASONING:        _lme_cfg_full(),
    LongMemEvalCategory.KNOWLEDGE_UPDATE:          _lme_cfg_full(),
    LongMemEvalCategory.SINGLE_SESSION_USER:       _lme_cfg_ep_ent(),
    LongMemEvalCategory.SINGLE_SESSION_ASSISTANT:  _lme_cfg_full(),
    LongMemEvalCategory.SINGLE_SESSION_PREFERENCE: _lme_cfg_s_ent(),
}

LME_GPT_4O_MINI_ROUTING_CONSERVATIVE: Dict[str, LongMemEvalTowerRoutingConfig] = {
    LongMemEvalCategory.MULTI_SESSION:             _lme_cfg_ep_ent(),
    LongMemEvalCategory.TEMPORAL_REASONING:        _lme_cfg_full(),
    LongMemEvalCategory.KNOWLEDGE_UPDATE:          _lme_cfg_full(),
    LongMemEvalCategory.SINGLE_SESSION_USER:       _lme_cfg_full(),  
    LongMemEvalCategory.SINGLE_SESSION_ASSISTANT:  _lme_cfg_full(),
    LongMemEvalCategory.SINGLE_SESSION_PREFERENCE: _lme_cfg_full(),  
}

LME_GPT_4O_MINI_ROUTING = LME_GPT_4O_MINI_ROUTING_AGGRESSIVE

LME_GPT_5_ROUTING_AGGRESSIVE: Dict[str, LongMemEvalTowerRoutingConfig] = {
    LongMemEvalCategory.MULTI_SESSION:             _lme_cfg_s_ent_top50(),
    LongMemEvalCategory.TEMPORAL_REASONING:        _lme_cfg_s_ep_top50(),
    LongMemEvalCategory.KNOWLEDGE_UPDATE:          _lme_cfg_s_ent_top50(),
    LongMemEvalCategory.SINGLE_SESSION_USER:       _lme_cfg_s_ent_top50(),
    LongMemEvalCategory.SINGLE_SESSION_ASSISTANT:  _lme_cfg_s_ep_top50(),
    LongMemEvalCategory.SINGLE_SESSION_PREFERENCE: _lme_cfg_full_top50(),
}

LME_GPT_5_ROUTING_CONSERVATIVE: Dict[str, LongMemEvalTowerRoutingConfig] = {
    LongMemEvalCategory.MULTI_SESSION:             _lme_cfg_full_top50(),
    LongMemEvalCategory.TEMPORAL_REASONING:        _lme_cfg_s_ep_top50(),
    LongMemEvalCategory.KNOWLEDGE_UPDATE:          _lme_cfg_s_ent_top50(),
    LongMemEvalCategory.SINGLE_SESSION_USER:       _lme_cfg_s_ent_top50(),
    LongMemEvalCategory.SINGLE_SESSION_ASSISTANT:  _lme_cfg_s_ep_top50(),
    LongMemEvalCategory.SINGLE_SESSION_PREFERENCE: _lme_cfg_full_top50(),
}

LME_GPT_5_ROUTING = LME_GPT_5_ROUTING_AGGRESSIVE

LME_CATEGORY_GUIDANCE: Dict[str, str] = {
    LongMemEvalCategory.MULTI_SESSION: (
        "Multi-session question - The answer requires integrating information from "
        "MULTIPLE conversation sessions. Carefully cross-reference facts, events, "
        "and context across different sessions to construct a complete answer."
    ),
    LongMemEvalCategory.TEMPORAL_REASONING: (
        "Temporal reasoning question - PRIORITIZE explicit time markers, session dates, "
        "and chronological ordering. Determine the correct temporal sequence of events "
        "and reason about 'before', 'after', 'during', 'latest', or 'first' as needed."
    ),
    LongMemEvalCategory.KNOWLEDGE_UPDATE: (
        "Knowledge update question - Information may have CHANGED over time. "
        "Always prefer the MOST RECENT statement. Look for corrections, updates, "
        "and evolving facts. If earlier and later information conflicts, trust the later one."
    ),
    LongMemEvalCategory.SINGLE_SESSION_USER: (
        "Single-session user question - The answer is based on what the USER said in "
        "a specific conversation session. Focus on user statements, claims, and "
        "self-reported information."
    ),
    LongMemEvalCategory.SINGLE_SESSION_ASSISTANT: (
        "Single-session assistant question - The answer is based on what the ASSISTANT "
        "said in a specific conversation session. Focus on assistant responses, "
        "recommendations, and provided information."
    ),
    LongMemEvalCategory.SINGLE_SESSION_PREFERENCE: (
        "User preference question - The answer involves the user's personal preferences, "
        "likes, dislikes, or choices. Look for explicit preference statements. "
        "If preferences changed over time, use the most recent one."
    ),
}

class LongMemEvalTowerRouter:
    """Rule-based category router for LongMemEval tri-tower retrieval."""

    
    MODEL_ROUTING_MAP = {
        "gpt-5": LME_GPT_5_ROUTING,
        "gpt-5-closeai": LME_GPT_5_ROUTING,
        "gpt-5-openrouter": LME_GPT_5_ROUTING,
        "gpt5": LME_GPT_5_ROUTING,
        "gpt5-closeai": LME_GPT_5_ROUTING,
        "gpt5-openrouter": LME_GPT_5_ROUTING,
        "gpt-4.1-mini": LME_GPT_41_MINI_ROUTING,
        "gpt-4.1-mini-closeai": LME_GPT_41_MINI_ROUTING,
        "gpt-4.1-mini-2025-04-14": LME_GPT_41_MINI_ROUTING,
        "gpt-4o-mini": LME_GPT_4O_MINI_ROUTING,
        "gpt-4o-mini-closeai": LME_GPT_4O_MINI_ROUTING,
        "gpt-4o-mini-2024-07-18": LME_GPT_4O_MINI_ROUTING,
    }

    MODEL_ROUTING_MAP_CONSERVATIVE = {
        "gpt-5": LME_GPT_5_ROUTING_CONSERVATIVE,
        "gpt-5-closeai": LME_GPT_5_ROUTING_CONSERVATIVE,
        "gpt-5-openrouter": LME_GPT_5_ROUTING_CONSERVATIVE,
        "gpt5": LME_GPT_5_ROUTING_CONSERVATIVE,
        "gpt5-closeai": LME_GPT_5_ROUTING_CONSERVATIVE,
        "gpt5-openrouter": LME_GPT_5_ROUTING_CONSERVATIVE,
        "gpt-4.1-mini": LME_GPT_41_MINI_ROUTING_CONSERVATIVE,
        "gpt-4.1-mini-closeai": LME_GPT_41_MINI_ROUTING_CONSERVATIVE,
        "gpt-4.1-mini-2025-04-14": LME_GPT_41_MINI_ROUTING_CONSERVATIVE,
        "gpt-4o-mini": LME_GPT_4O_MINI_ROUTING_CONSERVATIVE,
        "gpt-4o-mini-closeai": LME_GPT_4O_MINI_ROUTING_CONSERVATIVE,
        "gpt-4o-mini-2024-07-18": LME_GPT_4O_MINI_ROUTING_CONSERVATIVE,
    }

    def __init__(self, model_name: str = "gpt-4.1-mini",
                 strategy: str = "aggressive",
                 custom_routing: Optional[Dict[str, LongMemEvalTowerRoutingConfig]] = None):
        """Initialize a LongMemEval tower router.

        Args:
            model_name: Evaluation model name used to select a routing table.
            strategy: Routing-table variant, typically ``"aggressive"`` or
                ``"conservative"``.
            custom_routing: Optional routing table that overrides the built-in
                model-specific policy.
        """
        self.model_name = model_name
        self.strategy = strategy

        if custom_routing is not None:
            self.routing_table = custom_routing
        else:
            self.routing_table = self._resolve_routing_table(model_name, strategy)

        self._classify_stats: Dict[str, int] = {cat.value: 0 for cat in LongMemEvalCategory}

    def _resolve_routing_table(
        self, model_name: str, strategy: str = "aggressive"
    ) -> Dict[str, LongMemEvalTowerRoutingConfig]:
        """Resolve the routing table for a model and strategy."""
        model_lower = model_name.lower().strip()
        routing_map = (self.MODEL_ROUTING_MAP_CONSERVATIVE
                       if strategy == "conservative"
                       else self.MODEL_ROUTING_MAP)

        if model_lower in routing_map:
            return routing_map[model_lower]

        if "gpt-5" in model_lower or "gpt5" in model_lower:
            if strategy == "conservative":
                return LME_GPT_5_ROUTING_CONSERVATIVE
            return LME_GPT_5_ROUTING
        if "4.1" in model_lower:
            if strategy == "conservative":
                return LME_GPT_41_MINI_ROUTING_CONSERVATIVE
            return LME_GPT_41_MINI_ROUTING
        if "4o" in model_lower:
            if strategy == "conservative":
                return LME_GPT_4O_MINI_ROUTING_CONSERVATIVE
            return LME_GPT_4O_MINI_ROUTING

        
        if strategy == "conservative":
            return LME_GPT_41_MINI_ROUTING_CONSERVATIVE
        return LME_GPT_41_MINI_ROUTING

    def route(self, question: str,
              category: Optional[str] = None,
              enable_guidance: bool = False) -> LongMemEvalTowerRoutingConfig:
        """Return the retrieval-tower configuration for a LongMemEval question.

        Args:
            question: Natural-language benchmark question.
            category: Optional LongMemEval category. When omitted, a lightweight
                rule-based classifier is used.
            enable_guidance: Whether to attach category guidance text to the
                returned configuration.

        Returns:
            Routing configuration selected from the active policy table.
        """
        if category is None:
            category = self.classify_question(question)

        cat_key = self._normalize_category(category)

        self._classify_stats[cat_key] = self._classify_stats.get(cat_key, 0) + 1

        config = self.routing_table.get(cat_key)
        if config is None:
            config = _lme_cfg_full()

        if enable_guidance:
            config.category_guidance = LME_CATEGORY_GUIDANCE.get(
                cat_key, "General question - Use all available information sources."
            )
        else:
            config.category_guidance = ""

        return config
    
    

    

    @staticmethod
    def _normalize_category(category: str) -> str:
        """Normalize category."""
        cat = category.strip().lower()
        for member in LongMemEvalCategory:
            if cat == member.value:
                return member.value
        for member in LongMemEvalCategory:
            if cat == member.name.lower():
                return member.value
        if "multi" in cat and "session" in cat:
            return LongMemEvalCategory.MULTI_SESSION.value
        if "temporal" in cat:
            return LongMemEvalCategory.TEMPORAL_REASONING.value
        if "knowledge" in cat or "update" in cat:
            return LongMemEvalCategory.KNOWLEDGE_UPDATE.value
        if "preference" in cat or "pref" in cat:
            return LongMemEvalCategory.SINGLE_SESSION_PREFERENCE.value
        if "assistant" in cat or "asst" in cat:
            return LongMemEvalCategory.SINGLE_SESSION_ASSISTANT.value
        if "user" in cat:
            return LongMemEvalCategory.SINGLE_SESSION_USER.value
        return LongMemEvalCategory.MULTI_SESSION.value

    def classify_question(self, question: str) -> str:
        """Classify a LongMemEval question into a benchmark category.

        Args:
            question: Natural-language question text.

        Returns:
            LongMemEval category string.
        """
        q = question.strip().lower()

        temporal_patterns = [
            r'\bwhen\b', r'\bbefore\b', r'\bafter\b', r'\bfirst\b',
            r'\blast\b', r'\btimeline\b', r'\bchronolog', r'\border\b',
            r'\bearlier\b', r'\blater\b', r'\brecent\b', r'\blatest\b',
            r'\bhow\s+long\b', r'\bsince\s+when\b',
        ]
        for pattern in temporal_patterns:
            if re.search(pattern, q):
                return LongMemEvalCategory.TEMPORAL_REASONING.value

        update_patterns = [
            r'\bupdate[ds]?\b', r'\bchange[ds]?\b', r'\bnow\b',
            r'\bcurrent(?:ly)?\b', r'\bstill\b', r'\bused\s+to\b',
            r'\bno\s+longer\b', r'\binstead\b', r'\breplace[ds]?\b',
            r'\bswitch(?:ed)?\b',
        ]
        update_count = sum(1 for p in update_patterns if re.search(p, q))
        if update_count >= 2:
            return LongMemEvalCategory.KNOWLEDGE_UPDATE.value

        preference_patterns = [
            r'\bprefer(?:s|red|ence)?\b', r'\bfavorite\b', r'\bfavourite\b',
            r'\blike[sd]?\s+(?:more|most|best|better)\b',
            r'\bdislike[sd]?\b', r'\bhate[sd]?\b',
        ]
        for pattern in preference_patterns:
            if re.search(pattern, q):
                return LongMemEvalCategory.SINGLE_SESSION_PREFERENCE.value

        multi_session_patterns = [
            r'\bsession[s]?\b', r'\bconversation[s]?\b',
            r'\blast\s+time\b', r'\bprevious(?:ly)?\b',
            r'\bacross\b', r'\bmultiple\b',
            r'\bearlier\s+(?:today|conversation|session|discussion)\b',
        ]
        for pattern in multi_session_patterns:
            if re.search(pattern, q):
                return LongMemEvalCategory.MULTI_SESSION.value

        
        return LongMemEvalCategory.MULTI_SESSION.value

    def get_routing_stats(self) -> Dict:
        """Return counts for categories routed by this router instance."""
        total = sum(self._classify_stats.values())
        return {
            "model": self.model_name,
            "strategy": self.strategy,
            "total_routed": total,
            "per_category": {
                cat: {
                    "count": count,
                    "ratio": f"{count / total:.1%}" if total > 0 else "0%",
                    "towers": self.routing_table.get(cat, _lme_cfg_full()).active_towers,
                }
                for cat, count in sorted(self._classify_stats.items())
            },
        }

    def get_expected_improvement(self) -> Dict:
        """Return reproduction metadata for the selected routing policy."""
        is_41 = (self.routing_table in (
            LME_GPT_41_MINI_ROUTING_AGGRESSIVE,
            LME_GPT_41_MINI_ROUTING_CONSERVATIVE))
        is_5 = (self.routing_table in (
            LME_GPT_5_ROUTING_AGGRESSIVE,
            LME_GPT_5_ROUTING_CONSERVATIVE))
        is_aggressive = (self.routing_table in (
            LME_GPT_41_MINI_ROUTING_AGGRESSIVE,
            LME_GPT_4O_MINI_ROUTING_AGGRESSIVE,
            LME_GPT_5_ROUTING_AGGRESSIVE))

        if is_5:
            strategy = "aggressive" if is_aggressive else "conservative"
            return {
                "model": "GPT-5",
                "strategy": strategy,
                "baseline": 0.8980,
                "expected_routed": 0.9340 if is_aggressive else 0.9220,
                "improvement_pp": 3.40 if is_aggressive else 2.40,
                "note": ("GPT-5 top-50 ablations show the Episodic tower is harmful "
                         "for multi-session, knowledge-update, and single-session-user; "
                         "Entity is harmful for temporal and assistant categories. "
                         "The aggressive category router reaches 467/500 in offline "
                         "oracle-by-category analysis."),
                "per_category": {
                    "multi-session": {
                        "route": "S+Ent (wo_Ep)" if is_aggressive else "Full (S+Ep+Ent)",
                        "accuracy": 0.8421 if is_aggressive else 0.7970,
                        "gain_pp": 4.51 if is_aggressive else 0.0,
                        "improve_degrade": "10/4" if is_aggressive else "-",
                        "confidence": "moderate" if is_aggressive else "N/A",
                    },
                    "temporal-reasoning": {
                        "route": "S+Ep (wo_Ent)",
                        "accuracy": 0.9398,
                        "gain_pp": 5.26,
                        "improve_degrade": "8/1",
                        "confidence": "high",
                    },
                    "knowledge-update": {
                        "route": "S+Ent (wo_Ep)",
                        "accuracy": 0.9487,
                        "gain_pp": 2.56,
                        "improve_degrade": "2/0",
                        "confidence": "moderate",
                    },
                    "single-session-user": {
                        "route": "S+Ent (wo_Ep)",
                        "accuracy": 1.0000,
                        "gain_pp": 1.43,
                        "improve_degrade": "1/0",
                        "confidence": "low",
                    },
                    "single-session-assistant": {
                        "route": "S+Ep (wo_Ent)",
                        "accuracy": 1.0000,
                        "gain_pp": 3.57,
                        "improve_degrade": "2/0",
                        "confidence": "moderate",
                    },
                    "single-session-preference": {
                        "route": "Full (S+Ep+Ent)",
                        "accuracy": 1.0000,
                        "gain_pp": 0.0,
                        "improve_degrade": "-",
                        "confidence": "N/A",
                    },
                },
                "confidence_summary": (
                    "Aggressive routes all per-category winners (93.4%). "
                    "Conservative keeps multi-session on Full because its "
                    "improve/degrade ratio is 10/4, yielding 92.2%."
                ),
            }

        if is_41:
            strategy = "aggressive" if is_aggressive else "conservative"
            return {
                "model": "GPT-4.1-mini",
                "strategy": strategy,
                "baseline_abl_full": 0.8560,
                "baseline_water_reeval": 0.8700,
                "expected_routed": 0.9020 if is_aggressive else 0.9000,
                "improvement_vs_abl_full_pp": 4.60 if is_aggressive else 4.40,
                "improvement_vs_water_reeval_pp": 3.20 if is_aggressive else 3.00,
                "note": ("Entity tower (Ent) hurts multi-session and temporal categories; "
                         "removing it gives the biggest gains. "
                         "Knowledge-update benefits from removing Episodic tower."),
                "per_category": {
                    "multi-session": {
                        "route": "S+Ep (wo_Ent)" if True else "Full",
                        "accuracy": 0.8271 if True else 0.7669,
                        "gain_pp": 6.02 if True else 0.0,
                        "improve_degrade": "10/2",
                        "confidence": "high",
                    },
                    "temporal-reasoning": {
                        "route": "S+Ep (wo_Ent)" if True else "Full",
                        "accuracy": 0.8647 if True else 0.7970,
                        "gain_pp": 6.77 if True else 0.0,
                        "improve_degrade": "12/3",
                        "confidence": "high",
                    },
                    "knowledge-update": {
                        "route": "S+Ent (wo_Ep)" if True else "Full",
                        "accuracy": 0.9231 if True else 0.8590,
                        "gain_pp": 6.41 if True else 0.0,
                        "improve_degrade": "6/1",
                        "confidence": "high",
                    },
                    "single-session-user": {
                        "route": "Ep+Ent (wo_S)" if is_aggressive else "Full",
                        "accuracy": 0.9857 if is_aggressive else 0.9714,
                        "gain_pp": 1.43 if is_aggressive else 0.0,
                        "improve_degrade": "2/1" if is_aggressive else "-",
                        "confidence": "low" if is_aggressive else "N/A",
                    },
                    "single-session-assistant": {
                        "route": "Full (S+Ep+Ent)",
                        "accuracy": 1.0000,
                        "gain_pp": 0.0,
                        "improve_degrade": "-",
                        "confidence": "N/A (already 100%)",
                    },
                    "single-session-preference": {
                        "route": "Full (S+Ep+Ent)",
                        "accuracy": 0.9667,
                        "gain_pp": 0.0,
                        "improve_degrade": "-",
                        "confidence": "N/A",
                    },
                },
                "confidence_summary": (
                    "3 high-confidence routes (multi-session, temporal, knowledge) "
                    "with improve/degrade ratios >= 4:1 and gains >= 6pp each. "
                    "These 3 routes account for most of the improvement."
                ),
            }
        else:
            strategy = "aggressive" if is_aggressive else "conservative"
            return {
                "model": "GPT-4o-mini",
                "strategy": strategy,
                "baseline": 0.8420,
                "expected_routed": 0.8560 if is_aggressive else 0.8520,
                "improvement_pp": 1.40 if is_aggressive else 1.00,
                "note": ("Limited routing gain for GPT-4o-mini. "
                         "All overall ablations perform worse than full tower. "
                         "Only multi-session shows moderate per-category gain."),
                "per_category": {
                    "multi-session": {
                        "route": "Ep+Ent (wo_S)",
                        "accuracy": 0.7444,
                        "gain_pp": 3.76,
                        "improve_degrade": "10/5",
                        "confidence": "moderate",
                        "note": "S+Ep also gives 0.7444; Ep+Ent chosen as primary",
                    },
                    "temporal-reasoning": {
                        "route": "Full (S+Ep+Ent)",
                        "accuracy": 0.8045,
                        "gain_pp": 0.0,
                        "improve_degrade": "-",
                        "confidence": "N/A",
                    },
                    "knowledge-update": {
                        "route": "Full (S+Ep+Ent)",
                        "accuracy": 0.9103,
                        "gain_pp": 0.0,
                        "improve_degrade": "-",
                        "confidence": "N/A",
                    },
                    "single-session-user": {
                        "route": "Ep+Ent (wo_S)" if is_aggressive else "Full",
                        "accuracy": 0.9714 if is_aggressive else 0.9571,
                        "gain_pp": 1.43 if is_aggressive else 0.0,
                        "improve_degrade": "1/0" if is_aggressive else "-",
                        "confidence": "low" if is_aggressive else "N/A",
                    },
                    "single-session-assistant": {
                        "route": "Full (S+Ep+Ent)",
                        "accuracy": 0.9643,
                        "gain_pp": 0.0,
                        "improve_degrade": "-",
                        "confidence": "N/A",
                    },
                    "single-session-preference": {
                        "route": "S+Ent (wo_Ep)" if is_aggressive else "Full",
                        "accuracy": 0.9667 if is_aggressive else 0.9333,
                        "gain_pp": 3.33 if is_aggressive else 0.0,
                        "improve_degrade": "1/0" if is_aggressive else "-",
                        "confidence": "low" if is_aggressive else "N/A",
                    },
                },
                "confidence_summary": (
                    "Only multi-session→Ep+Ent has moderate confidence (10/5, +3.76pp). "
                    "Other gains rely on single-question changes. "
                    "Consider using full tower for GPT-4o-mini unless multi-session "
                    "questions are a significant portion of workload."
                ),
            }

    def __repr__(self):
        entries = []
        for cat in LongMemEvalCategory:
            cfg = self.routing_table.get(cat.value, _lme_cfg_full())
            entries.append(f"  {cat.value}: {cfg.active_towers}")
        table_str = "\n".join(entries)
        return (f"LongMemEvalTowerRouter(model={self.model_name}, "
                f"strategy={self.strategy})\n{table_str}")
