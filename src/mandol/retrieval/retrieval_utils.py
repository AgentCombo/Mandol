"""Shared configuration and result containers for multi-backend retrieval."""

import time
import logging
from typing import Dict, List, Set, Any, Optional
from dataclasses import dataclass
from collections import defaultdict

from .retrieval_interface import RetrievalMethod, RetrievalResult
from ..core.memory_unit import MemoryUnit
from ..utils.logging_config import create_module_logger

logger = create_module_logger("retrieval_utils")


@dataclass
class ParallelRetrievalConfig:
    """Execution settings for parallel retrieval backends."""
    enable_parallel: bool = True
    max_workers: int = 3
    timeout_seconds: float = 30.0
    consistency_check: bool = True
    fallback_on_error: bool = True


class RetrievalSnapshot:
    """Lightweight consistency snapshot for a retrieval source."""
    
    def __init__(self, retrieval_source, snapshot_id: str = None):
        self.snapshot_id = snapshot_id or f"snapshot_{int(time.time() * 1000)}"
        self.creation_time = time.time()
        self.retrieval_source = retrieval_source
        
        self._snapshot_state = self._create_snapshot_state()
        
    def _create_snapshot_state(self) -> Dict[str, Any]:
        """Capture source counters used for consistency checks."""
        state = {
            'snapshot_id': self.snapshot_id,
            'creation_time': self.creation_time,
        }
        
        if hasattr(self.retrieval_source, 'memory_units'):
            state['units_count'] = len(self.retrieval_source.memory_units)
            state['units_hash'] = hash(tuple(self.retrieval_source.memory_units.keys()))
        
        if hasattr(self.retrieval_source, 'semantic_map'):
            state['semantic_units_count'] = len(self.retrieval_source.semantic_map.memory_units)
            state['faiss_index_size'] = self.retrieval_source.semantic_map.faiss_index.ntotal if self.retrieval_source.semantic_map.faiss_index else 0
        
        return state
    
    def validate_consistency(self) -> bool:
        """Return whether the retrieval source still matches the snapshot."""
        current_state = self._create_snapshot_state()
        
        for key in ['units_count', 'units_hash', 'semantic_units_count', 'faiss_index_size']:
            if key in self._snapshot_state and key in current_state:
                if self._snapshot_state[key] != current_state[key]:
                    logger.warning(
                        f"Retrieval source changed during search: {key} "
                        f"{self._snapshot_state[key]} -> {current_state[key]}"
                    )
                    return False
        
        return True


class FlexibleRetrievalConfig:
    """Fusion settings for a multi-backend retrieval request."""
    
    def __init__(self, 
                 methods: List[RetrievalMethod],
                 fusion_method: str = "rrf",
                 weights: Optional[Dict[RetrievalMethod, float]] = None,
                 rrf_k: int = 60,
                 normalization: str = "min_max",
                 top_k_multiplier: float = 2.0):
        """Initialize fusion settings.

        Args:
            methods: Retrieval methods enabled for the request.
            fusion_method: Score fusion method, such as ``"rrf"``.
            weights: Optional per-method weights for weighted fusion.
            rrf_k: RRF smoothing constant.
            normalization: Score normalization strategy.
            top_k_multiplier: Multiplier for per-backend candidate budgets.
        """
        self.methods = methods
        self.fusion_method = fusion_method.lower()
        self.weights = weights
        self.rrf_k = rrf_k
        self.normalization = normalization
        self.top_k_multiplier = top_k_multiplier
        
        self._validate_config()
    
    def _validate_config(self):
        """Validate retrieval method and fusion settings."""
        if not self.methods:
            raise ValueError("At least one retrieval method must be specified.")
        
        if self.fusion_method not in ["rrf", "weighted", "average"]:
            raise ValueError(f"Unsupported fusion method: {self.fusion_method}")
        
        if self.fusion_method == "weighted" and not self.weights:
            logger.warning("Weighted fusion requested without weights; uniform weights will be used.")
            
    
    def get_display_name(self) -> str:
        """Return display name."""
        method_names = [method.value for method in self.methods]
        if len(self.methods) == 1:
            return method_names[0]
        else:
            return f"{'+'.join(method_names)}({self.fusion_method})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize the retrieval configuration to a dictionary."""
        result = {
            "methods": [method.value for method in self.methods],
            "fusion_method": self.fusion_method,
            "rrf_k": self.rrf_k,
            "normalization": self.normalization,
            "top_k_multiplier": self.top_k_multiplier
        }
        if self.weights:
            result["weights"] = {method.value: weight for method, weight in self.weights.items()}
        return result


class MultiRetrievalResults:
    """Accumulator for results returned by multiple retrieval backends."""
    
    def __init__(self):
        self.results_by_method: Dict[RetrievalMethod, List[RetrievalResult]] = defaultdict(list)
        self.results_by_uid: Dict[str, Dict[RetrievalMethod, RetrievalResult]] = defaultdict(dict)
        self.all_units: Dict[str, MemoryUnit] = {}
        
    def add_result(self, result: RetrievalResult):
        """Add result."""
        self.results_by_method[result.method].append(result)
        self.results_by_uid[result.unit.uid][result.method] = result
        self.all_units[result.unit.uid] = result.unit
        
    def add_results(self, results: List[RetrievalResult]):
        """Add results."""
        for result in results:
            self.add_result(result)
    
    def get_methods_used(self) -> Set[RetrievalMethod]:
        """Return methods used."""
        return set(self.results_by_method.keys())
    
    def get_units_found_by_method(self, method: RetrievalMethod) -> List[MemoryUnit]:
        """Return units found by method."""
        return [result.unit for result in self.results_by_method[method]]
    
    def get_intersection_units(self) -> List[str]:
        """Return intersection units."""
        if not self.results_by_method:
            return []
        
        method_uids = [
            set(result.unit.uid for result in results) 
            for results in self.results_by_method.values()
        ]
        return list(set.intersection(*method_uids))
    
    def get_union_units(self) -> List[str]:
        """Return union units."""
        return list(self.all_units.keys())
    
    def get_score_for_unit(self, uid: str, method: RetrievalMethod) -> Optional[float]:
        """Return score for unit."""
        if uid in self.results_by_uid and method in self.results_by_uid[uid]:
            return self.results_by_uid[uid][method].score
        return None
