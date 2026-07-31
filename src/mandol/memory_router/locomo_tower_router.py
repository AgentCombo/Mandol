"""Category-based tower router for LoCoMo.

The router selects hierarchical, graph, and episodic retrieval towers according
to question categories used by the artifact reproduction workflow.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
from enum import IntEnum

class LoCoMoCategory(IntEnum):
    """Question categories used by the LoCoMo benchmark."""
    AGGREGATION = 1
    TEMPORAL = 2
    OPEN_DOMAIN = 3
    SIMPLE_FACT = 4
    ADVERSARIAL = 5

@dataclass
class TowerRoutingConfig:
    """Routing configuration for one LoCoMo query.

    Boolean fields enable or disable retrieval towers. Top-k fields define
    per-tower retrieval budgets. A top-k value of 0 disables the corresponding
    retrieval path. Weights are used when combining enabled tower scores.
    """

    enable_hierarchical: bool = True  # Enable the hierarchical-memory tower.
    enable_graph: bool = True         # Enable the entity-relation graph tower.
    enable_episodic: bool = True      # Enable the episodic-memory tower.

    topk_hierarchical: int = 15  # Retrieval budget for the hierarchical tower.
    topk_similarity: int = 30    # Semantic retrieval budget for the graph tower.
    topk_graph: int = 0          # Entity-relation graph budget; 0 disables it.
    topk_episodic: int = 30      # Retrieval budget for the episodic tower.
    use_entity_relation: bool = True

    final_top_k: int = 20        # Number of contexts kept after reranking.

    weight_hierarchical: float = 0.34
    weight_graph: float = 0.33
    weight_episodic: float = 0.33

    category_guidance: str = ""

    @property
    def active_towers(self) -> str:
        """Return labels for enabled retrieval towers."""
        parts = []
        if self.enable_hierarchical:
            parts.append("H")
        if self.enable_graph:
            parts.append("G")
        if self.enable_episodic:
            parts.append("E")
        return "+".join(parts) if parts else "NONE"

    def __repr__(self):
        return (f"TowerRoutingConfig(towers={self.active_towers}, "
                f"topk=[H={self.topk_hierarchical},G_sim={self.topk_similarity},"
                f"G_ent={self.topk_graph},E={self.topk_episodic}], "
                f"final_k={self.final_top_k}, "
                f"w=[{self.weight_hierarchical:.2f},{self.weight_graph:.2f},{self.weight_episodic:.2f}])")

def _cfg_full() -> TowerRoutingConfig:
    """Return the full LoCoMo routing configuration."""
    return TowerRoutingConfig(
        enable_hierarchical=True, enable_graph=True, enable_episodic=True,
        topk_hierarchical=15, topk_similarity=30, topk_graph=0, topk_episodic=30,
        use_entity_relation=False,  
        final_top_k=20,
        weight_hierarchical=0.34, weight_graph=0.33, weight_episodic=0.33,
    )

def _cfg_h_g() -> TowerRoutingConfig:
    """Return the LoCoMo H+G routing configuration."""
    return TowerRoutingConfig(
        enable_hierarchical=True, enable_graph=True, enable_episodic=False,
        topk_hierarchical=15, topk_similarity=30, topk_graph=0, topk_episodic=0,
        use_entity_relation=False,  
        final_top_k=20,
        weight_hierarchical=0.5, weight_graph=0.5, weight_episodic=0.0,
    )

def _cfg_h_e() -> TowerRoutingConfig:
    """Return the LoCoMo H+E routing configuration."""
    return TowerRoutingConfig(
        enable_hierarchical=True, enable_graph=False, enable_episodic=True,
        topk_hierarchical=15, topk_similarity=0, topk_graph=0, topk_episodic=30,
        use_entity_relation=False,
        final_top_k=20,
        weight_hierarchical=0.5, weight_graph=0.0, weight_episodic=0.5,
    )

def _cfg_g_e() -> TowerRoutingConfig:
    """Return the LoCoMo G+E routing configuration."""
    return TowerRoutingConfig(
        enable_hierarchical=False, enable_graph=True, enable_episodic=True,
        topk_hierarchical=0, topk_similarity=30, topk_graph=0, topk_episodic=30,
        use_entity_relation=False,  
        final_top_k=35,
        weight_hierarchical=0.0, weight_graph=0.5, weight_episodic=0.5,
    )

GPT_41_MINI_ROUTING_AGGRESSIVE: Dict[int, TowerRoutingConfig] = {
    LoCoMoCategory.AGGREGATION: _cfg_full(),  
    LoCoMoCategory.TEMPORAL: _cfg_full(),  
    LoCoMoCategory.OPEN_DOMAIN: _cfg_full(),  
    LoCoMoCategory.SIMPLE_FACT: _cfg_full(),  
    LoCoMoCategory.ADVERSARIAL: _cfg_full(),  
}

GPT_41_MINI_ROUTING_CONSERVATIVE: Dict[int, TowerRoutingConfig] = {
    LoCoMoCategory.AGGREGATION: _cfg_full(),  
    LoCoMoCategory.TEMPORAL: _cfg_full(),  
    LoCoMoCategory.OPEN_DOMAIN: _cfg_full(),  
    LoCoMoCategory.SIMPLE_FACT: _cfg_full(),  
    LoCoMoCategory.ADVERSARIAL: _cfg_full(),  
}

GPT_41_MINI_ROUTING = GPT_41_MINI_ROUTING_AGGRESSIVE

GPT_4O_MINI_ROUTING_AGGRESSIVE: Dict[int, TowerRoutingConfig] = {
    LoCoMoCategory.AGGREGATION: _cfg_h_e(),  
    LoCoMoCategory.TEMPORAL: _cfg_h_g(),
    LoCoMoCategory.OPEN_DOMAIN: _cfg_full(),
    LoCoMoCategory.SIMPLE_FACT: _cfg_full(),
    LoCoMoCategory.ADVERSARIAL: _cfg_g_e(),
}

GPT_4O_MINI_ROUTING_CONSERVATIVE: Dict[int, TowerRoutingConfig] = {
    LoCoMoCategory.AGGREGATION: _cfg_full(),  
    LoCoMoCategory.TEMPORAL: _cfg_full(),  
    LoCoMoCategory.OPEN_DOMAIN: _cfg_full(),
    LoCoMoCategory.SIMPLE_FACT: _cfg_full(),
    LoCoMoCategory.ADVERSARIAL: _cfg_g_e(),
}

GPT_4O_MINI_ROUTING = GPT_4O_MINI_ROUTING_AGGRESSIVE

CATEGORY_GUIDANCE: Dict[int, str] = {
    LoCoMoCategory.AGGREGATION: (
        "Multi-hop aggregation - The answer requires tracing connections across multiple "
        "facts and sessions. Combine evidence from different sources to form a complete answer. "
        "Pay attention to cross-session references and cumulative information."
    ),
    LoCoMoCategory.TEMPORAL: (
        "Temporal reasoning - PRIORITIZE explicit time markers like [Time: ...] and session dates. "
        "Cross-verify temporal claims across sources. Express time as specifically as possible."
    ),
    LoCoMoCategory.OPEN_DOMAIN: (
        "Open-domain reasoning - The question requires inference or speculation based on known facts. "
        "Synthesize evidence from available sources and provide a well-reasoned conclusion."
    ),
    LoCoMoCategory.SIMPLE_FACT: (
        "Single-hop fact retrieval - The answer is a single, directly stated fact. "
        "Look for a clear, concise factual statement. If multiple sources agree, trust them."
    ),
    LoCoMoCategory.ADVERSARIAL: (
        "Adversarial verification - This question may contain entity confusion or ask about "
        "non-existent information. Carefully verify each claim against ALL available sources. "
        "If the information is not found in memory, answer 'I don't have that information.'"
    ),
}

class LocomoTowerRouter:
    """Rule-based category router for LoCoMo tri-tower retrieval."""

    
    MODEL_ROUTING_MAP = {
        "gpt-4.1-mini": GPT_41_MINI_ROUTING,
        "gpt-4.1-mini-closeai": GPT_41_MINI_ROUTING,
        "gpt-4.1-mini-openrouter": GPT_41_MINI_ROUTING,
        "gpt-4.1-mini-2025-04-14": GPT_41_MINI_ROUTING,
        "gpt-4o-mini": GPT_4O_MINI_ROUTING,
        "gpt-4o-mini-closeai": GPT_4O_MINI_ROUTING,
        "gpt-4o-mini-openrouter": GPT_4O_MINI_ROUTING,
        "gpt-4o-mini-2024-07-18": GPT_4O_MINI_ROUTING,
    }

    
    MODEL_ROUTING_MAP_CONSERVATIVE = {
        "gpt-4.1-mini": GPT_41_MINI_ROUTING_CONSERVATIVE,
        "gpt-4.1-mini-closeai": GPT_41_MINI_ROUTING_CONSERVATIVE,
        "gpt-4.1-mini-openrouter": GPT_41_MINI_ROUTING_CONSERVATIVE,
        "gpt-4.1-mini-2025-04-14": GPT_41_MINI_ROUTING_CONSERVATIVE,
        "gpt-4o-mini": GPT_4O_MINI_ROUTING_CONSERVATIVE,
        "gpt-4o-mini-closeai": GPT_4O_MINI_ROUTING_CONSERVATIVE,
        "gpt-4o-mini-openrouter": GPT_4O_MINI_ROUTING_CONSERVATIVE,
        "gpt-4o-mini-2024-07-18": GPT_4O_MINI_ROUTING_CONSERVATIVE,
    }

    def __init__(self, model_name: str = "gpt-4.1-mini",
                 strategy: str = "aggressive",
                 custom_routing: Optional[Dict[int, TowerRoutingConfig]] = None):
        """Initialize a LoCoMo tower router.

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

        self._classify_stats = {cat: 0 for cat in LoCoMoCategory}

    def _resolve_routing_table(self, model_name: str, strategy: str = "aggressive") -> Dict[int, TowerRoutingConfig]:
        """Resolve the routing table for a model and strategy."""
        model_lower = model_name.lower().strip()
        routing_map = (self.MODEL_ROUTING_MAP_CONSERVATIVE
                       if strategy == "conservative"
                       else self.MODEL_ROUTING_MAP)

        if model_lower in routing_map:
            return routing_map[model_lower]

        if "4.1" in model_lower:
            if strategy == "conservative":
                return GPT_41_MINI_ROUTING_CONSERVATIVE
            return GPT_41_MINI_ROUTING
        if "4o" in model_lower:
            if strategy == "conservative":
                return GPT_4O_MINI_ROUTING_CONSERVATIVE
            return GPT_4O_MINI_ROUTING

        if strategy == "conservative":
            return GPT_41_MINI_ROUTING_CONSERVATIVE
        return GPT_41_MINI_ROUTING

    def route(self, question: str,
              category: Optional[int] = None,
              enable_guidance: bool = False) -> TowerRoutingConfig:
        """Return the retrieval-tower configuration for a LoCoMo question.

        Args:
            question: Natural-language benchmark question.
            category: Optional LoCoMo category ID. When omitted, a lightweight
                rule-based classifier is used.
            enable_guidance: Whether to attach category guidance text to the
                returned configuration.

        Returns:
            Routing configuration selected from the active policy table.
        """
        if category is None:
            category = self.classify_question(question)

        if isinstance(category, int) and not isinstance(category, LoCoMoCategory):
            try:
                category = LoCoMoCategory(category)
            except ValueError:
                pass

        self._classify_stats[category] = self._classify_stats.get(category, 0) + 1

        config = self.routing_table.get(category)
        if config is None:
            config = _cfg_full()

        if enable_guidance:
            config.category_guidance = CATEGORY_GUIDANCE.get(
                category, "General question - Use all available information sources."
            )
        else:
            config.category_guidance = ""

        return config
    
    

    

    def classify_question(self, question: str) -> int:
        """Classify a LoCoMo question into a benchmark category.

        Args:
            question: Natural-language question text.

        Returns:
            A LoCoMoCategory value represented as an integer enum member.
        """
        q = question.strip().lower()

        if re.match(r'\bwhen\b', q):
            return LoCoMoCategory.TEMPORAL

        reasoning_patterns = [
            r'\bwould\b',
            r'\bcould\b',
            r'\blikely\b',
            r'\bprobably\b',
            r'\bif\s+(?:she|he|they|it)\b',
            r'\bwhat\s+(?:would|could|might)\b',
            r'\bdo\s+you\s+think\b',
            r'\bis\s+it\s+(?:likely|possible|probable)\b',
            r'\bconsidered\s+a\b',  # "Would X be considered a ..."
        ]
        for pattern in reasoning_patterns:
            if re.search(pattern, q):
                return LoCoMoCategory.OPEN_DOMAIN

        simple_fact_patterns = [
            r'^what\s+is\b',            # "What is X's Y?"
            r'^what\s+was\b',           # "What was X's Y?"
            r'^what\s+does\b',          # "What does X do?"
            r'^who\s+is\b',             # "Who is X?"
            r'^who\s+was\b',            # "Who was X?"
            r'^where\s+is\b',           # "Where is X?"
            r'^where\s+does\b',         # "Where does X live?"
            r'^where\s+did\b.*\bmove\b', # "Where did X move from?"
        ]

        for pattern in simple_fact_patterns:
            if re.search(pattern, q):
                word_count = len(q.split())
                if word_count <= 12:
                    return LoCoMoCategory.SIMPLE_FACT

        aggregation_patterns = [
            r'^how\b',                  # "How does X do Y?"
            r'^why\b',                  # "Why did X do Y?"
            r'^what\s+(?:did|are|were)\b',  # "What did X do after Y?"
        ]
        for pattern in aggregation_patterns:
            if re.search(pattern, q):
                return LoCoMoCategory.AGGREGATION

        return LoCoMoCategory.SIMPLE_FACT

    def get_routing_stats(self) -> Dict[str, any]:
        """Return counts for categories routed by this router instance."""
        total = sum(self._classify_stats.values())
        return {
            "model": self.model_name,
            "total_routed": total,
            "per_category": {
                f"Cat{cat}({cat.name})": {
                    "count": count,
                    "ratio": f"{count / total:.1%}" if total > 0 else "0%",
                    "towers": self.routing_table.get(cat, _cfg_full()).active_towers,
                }
                for cat, count in sorted(self._classify_stats.items())
            },
        }

    def get_expected_improvement(self) -> Dict[str, any]:
        """Return reproduction metadata for the selected routing policy."""
        is_41 = (self.routing_table is GPT_41_MINI_ROUTING_AGGRESSIVE
                 or self.routing_table is GPT_41_MINI_ROUTING_CONSERVATIVE)
        is_4o_agg = self.routing_table is GPT_4O_MINI_ROUTING_AGGRESSIVE
        is_4o_con = self.routing_table is GPT_4O_MINI_ROUTING_CONSERVATIVE

        if is_41:
            return {
                "model": "GPT-4.1-mini",
                "strategy": self.strategy,
                "data_version": "v2 (locomo10_ablation_separate, tower_separate rerank)",
                "baseline_reeval": 0.9366,
                "expected_routed": 0.9366,
                "improvement_pp": 0.0,
                "note": (
                    "Matched-budget tower ablations select the full tri-tower route "
                    "for every category. Earlier apparent gains were sensitive to "
                    "evaluation and budget mismatches."
                ),
                "per_category": {
                    "Cat1_Aggregation": "Full is selected; reduced-tower variants regress.",
                    "Cat2_Temporal": "Full is selected; H+G is lower by 0.62pp.",
                    "Cat3_OpenDomain": "Full is selected; reduced-tower variants regress.",
                    "Cat4_SimpleFact": "Full is selected as the conservative tied policy.",
                    "Cat5_Adversarial": "Full is selected; G+E is lower by 0.23pp.",
                },
                "confidence": {
                    "Cat1": "N/A (Full optimal)",
                    "Cat2": "N/A (Full optimal)",
                    "Cat3": "N/A (Full optimal)",
                    "Cat4": "N/A (Full=H+E tied, net=0)",
                    "Cat5": "N/A (Full optimal)",
                },
            }
        else:
            strategy = "aggressive" if is_4o_agg else "conservative"
            return {
                "model": "GPT-4o-mini",
                "strategy": strategy,
                "data_version": "v2 (locomo10_ablation_separate, tower_separate rerank)",
                "baseline_reeval": 0.8474,
                "expected_routed": 0.9104 if is_4o_agg else 0.9094,
                "improvement_pp": 6.29 if is_4o_agg else 6.19,
                "note": ("Cat5 -> G+E accounts for most of the observed gain "
                         "(+27.58pp, 129/6 improve/degrade). "
                         "Removing the hierarchical tower reduces interference on adversarial questions."
                         + (" Cat1 -> H+E is tied with Full and reduces latency; "
                            "Cat2 -> H+G has a marginal +0.62pp gain."
                            if is_4o_agg else "")),
                "per_category": {
                    "Cat1_Aggregation": ("H+E ties Full and avoids graph-tower latency."
                                         if is_4o_agg else "Full is selected conservatively."),
                    "Cat2_Temporal": ("H+G is selected with a marginal +0.62pp gain."
                                      if is_4o_agg else "Full is selected conservatively."),
                    "Cat3_OpenDomain": "Full is selected; reduced-tower variants regress.",
                    "Cat4_SimpleFact": "Full is selected.",
                    "Cat5_Adversarial": "G+E(0.9439) exceeds Full(0.6682), +27.58pp, 129/6 high confidence",
                },
                "confidence": {
                    "Cat1": "N/A (tied with Full)" if is_4o_agg else "N/A (Full)",
                    "Cat2": "low (10/8 improve/degrade, +0.62pp)" if is_4o_agg else "N/A (Full)",
                    "Cat3": "N/A (Full optimal)",
                    "Cat4": "N/A (Full optimal)",
                    "Cat5": "very high (129/6, +27.58pp, repeated across policy sweeps)",
                },
            }

    def __repr__(self):
        entries = []
        for cat in sorted(self.routing_table):
            cfg = self.routing_table[cat]
            entries.append(f"  Cat{cat}: {cfg.active_towers}")
        table_str = "\n".join(entries)
        return f"LocomoTowerRouter(model={self.model_name}, strategy={self.strategy})\n{table_str}"
