"""Cascade pruning for assembling bounded multi-tower evidence contexts.

The pruner applies score filtering, cross-tower disambiguation, and optional
MMR diversification before converting candidates into a prompt context. It is
used after retrieval and reranking, and does not modify source indexes or memory
objects.
"""

from __future__ import annotations

import math
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import tiktoken

from .adaptive_category import AdaptiveParams, get_adaptive_params

from .confidence_pruner import (
    CandidateChunk,
    ConfidenceLevel,
    PruneMode,
    PrunedChunk,
    PruneResult,
)
from ..utils.logging_config import create_module_logger

logger = create_module_logger("quantification.cascade_pruner")




class TowerSource(Enum):
    """Source tower labels used to balance retained evidence chunks."""

    KG = "KG"  
    EPISODIC = "EPISODIC"  
    HIERARCHICAL = "HIERARCHICAL"  

    
    @classmethod
    def from_source_type(cls, source_type: str) -> "TowerSource":
        """Map benchmark source labels to the canonical tower taxonomy."""
        _mapping = {
            "entity": cls.KG,
            "kg": cls.KG,
            "graph": cls.KG,
            "episodic": cls.EPISODIC,
            "sentence": cls.HIERARCHICAL,
            "hierarchical": cls.HIERARCHICAL,
            "l0": cls.HIERARCHICAL,
            "l1": cls.HIERARCHICAL,
            "l2": cls.HIERARCHICAL,
        }
        key = source_type.strip().lower()
        if key in _mapping:
            return _mapping[key]
        logger.warning("Unknown source_type=%r; defaulting to HIERARCHICAL.", source_type)
        return cls.HIERARCHICAL


class ConflictResolution(Enum):

    KEPT = "KEPT"
    DROPPED_STALE = "DROPPED_STALE"  
    DROPPED_LOW_SCORE = "DROPPED_LOW_SCORE"
    DROPPED_REDUNDANT = "DROPPED_REDUNDANT"
    MERGED = "MERGED"  


@dataclass
class EnhancedCandidateChunk(CandidateChunk):
    """Candidate chunk enriched with tower and provenance metadata."""

    tower_source: TowerSource = TowerSource.HIERARCHICAL
    source_chunk_ids: List[str] = field(default_factory=list)
    timestamp: float = 0.0
    entity_ids: List[str] = field(default_factory=list)
    memory_space: str = ""
    fact_type: str = ""
    raw_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DisambiguationRecord:
    """Record how conflicting chunks were resolved during stage 2."""

    group_key: str
    group_size: int
    kept_chunk_ids: List[str]
    dropped_chunk_ids: List[str]
    resolution_reasons: Dict[str, str]       # chunk_id -> ConflictResolution.value
    tower_distribution: Dict[str, int]       # tower -> count


@dataclass
class CascadePruneResult:
    """Pruning result with per-stage accounting for reproducibility."""

    prompt_context: str
    selected_chunks: List[PrunedChunk]
    total_tokens_used: int
    mode_used: PruneMode

    stage1_input_count: int = 0
    stage1_output_count: int = 0
    stage2_conflicts_found: int = 0
    stage2_dropped_count: int = 0
    stage2_output_count: int = 0
    stage3_mmr_iterations: int = 0
    stage3_diversity_penalties: int = 0

    disambiguation_log: List[DisambiguationRecord] = field(default_factory=list)

    def to_base_result(self) -> PruneResult:
        """Drop cascade diagnostics and return the base pruning result."""
        return PruneResult(
            prompt_context=self.prompt_context,
            selected_chunks=self.selected_chunks,
            total_tokens_used=self.total_tokens_used,
            mode_used=self.mode_used,
        )







DEFAULT_TOWER_PRIORITY: Dict[TowerSource, float] = {
    TowerSource.HIERARCHICAL: 1.0,
    TowerSource.EPISODIC: 0.8,
    TowerSource.KG: 0.6,
}


# DEFAULT_TOWER_PRIORITY: Dict[TowerSource, float] = {



# }







DEFAULT_TOWER_MIN_RATIO: Dict[TowerSource, float] = {
    TowerSource.HIERARCHICAL: 0.50,
    TowerSource.EPISODIC: 0.20,
    TowerSource.KG: 0.15,
}


class CascadeConfidencePruner:
    """Three-stage context pruner for routed multi-tower retrieval results.

    Args:
        mode: Base token-budget policy used by the inherited pruning result.
        mad_multiplier: Robust score cutoff multiplier for stage 1.
        cliff_tolerance: Gap threshold for score-cliff filtering.
        max_tokens: Maximum prompt-token budget.
        separator: String inserted between retained chunks.
        tower_priority: Optional source-priority weights for conflict handling.
        lambda_mmr: MMR relevance-diversity tradeoff in stage 3.
        entity_overlap_penalty: Penalty for entity overlap during diversification.
        source_duplicate_penalty: Penalty for duplicate source chunks.
        temporal_decay_hours: Time window used when ranking stale conflicts.
        enable_stage1: Enable robust score filtering.
        enable_stage2: Enable cross-tower conflict and duplicate handling.
        enable_stage3_mmr: Enable MMR-style final ordering.
        cap_to_input_tokens: Bound output by the input token footprint.
        tower_min_ratio: Optional minimum retained share per tower.
        absolute_min_score: Hard lower score bound.
        adaptive_dataset: Optional dataset name for adaptive pruning parameters.
    """

    def __init__(
        self,
        mode: PruneMode = PruneMode.BUDGET_MAX,
        mad_multiplier: float = 2.5,
        cliff_tolerance: float = 2.0,
        max_tokens: int = 4096,
        separator: str = "\n\n",
        tower_priority: Optional[Dict[TowerSource, float]] = None,
        lambda_mmr: float = 0.6,
        entity_overlap_penalty: float = 0.15,
        source_duplicate_penalty: float = 0.2,
        temporal_decay_hours: float = 720.0,
        enable_stage1: bool = True,
        enable_stage2: bool = True,
        enable_stage3_mmr: bool = True,
        cap_to_input_tokens: bool = True,
        tower_min_ratio: Optional[Dict[TowerSource, float]] = None,
        absolute_min_score: float = 0.0,
        adaptive_dataset: Optional[str] = None,
    ):
        if not 0.0 <= lambda_mmr <= 1.0:
            raise ValueError(f"lambda_mmr must be in [0, 1], got {lambda_mmr}")
        if entity_overlap_penalty < 0:
            raise ValueError(f"entity_overlap_penalty must be non-negative: {entity_overlap_penalty}")
        if source_duplicate_penalty < 0:
            raise ValueError(f"source_duplicate_penalty must be non-negative: {source_duplicate_penalty}")
        if mad_multiplier < 0:
            raise ValueError(f"mad_multiplier must be non-negative: {mad_multiplier}")
        if cliff_tolerance < 0:
            raise ValueError(f"cliff_tolerance must be non-negative: {cliff_tolerance}")
        if tower_min_ratio is not None:
            _ratio_sum = sum(tower_min_ratio.values())
            if _ratio_sum > 1.0:
                raise ValueError(
                    f"tower_min_ratio values sum to {_ratio_sum:.2f}, which is greater than 1.0"
                )
            if any(v < 0 for v in tower_min_ratio.values()):
                raise ValueError("tower_min_ratio values must be non-negative")

        self.mode = mode
        self.mad_multiplier = mad_multiplier
        self.cliff_tolerance = cliff_tolerance
        self.max_tokens = max_tokens
        self.separator = separator
        self.tower_priority = tower_priority or dict(DEFAULT_TOWER_PRIORITY)
        self.lambda_mmr = lambda_mmr
        self.entity_overlap_penalty = entity_overlap_penalty
        self.source_duplicate_penalty = source_duplicate_penalty
        self.temporal_decay_hours = temporal_decay_hours
        self.enable_stage1 = enable_stage1
        self.enable_stage2 = enable_stage2
        self.enable_stage3_mmr = enable_stage3_mmr
        self.cap_to_input_tokens = cap_to_input_tokens
        self.tower_min_ratio = tower_min_ratio
        self.absolute_min_score = absolute_min_score
        self.adaptive_dataset = adaptive_dataset

        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        _tmr_str = (
            f"tower_min_ratio={{{', '.join(f'{k.value}:{v:.2f}' for k, v in tower_min_ratio.items())}}}"
            if tower_min_ratio else "tower_min_ratio=OFF"
        )
        logger.info(
            f"[CascadeConfidencePruner] initialized: "
            f"mode={mode.value}, max_tokens={max_tokens}, "
            f"mad_multiplier={mad_multiplier}, cliff_tolerance={cliff_tolerance}, "
            f"lambda_mmr={lambda_mmr}, entity_penalty={entity_overlap_penalty}, "
            f"source_penalty={source_duplicate_penalty}, "
            f"temporal_decay={temporal_decay_hours}h, "
            f"stage1={'ON' if enable_stage1 else 'OFF'}, "
            f"stage2={'ON' if enable_stage2 else 'OFF'}, "
            f"stage3_mmr={'ON' if enable_stage3_mmr else 'OFF'}, "
            f"cap_to_input={'ON' if cap_to_input_tokens else 'OFF'}, "
            f"absolute_min_score={absolute_min_score}, "
            f"{_tmr_str}"
        )

    def prune(
        self,
        candidates: List[EnhancedCandidateChunk],
        query_timestamp: float = 0.0,
        query_category: Optional[str] = None,
    ) -> CascadePruneResult:
        """Prune reranked candidates into a bounded prompt context.

        Args:
            candidates: Reranked candidates from one or more retrieval towers.
            query_timestamp: Optional timestamp used by temporal decay rules.
            query_category: Optional benchmark category used for adaptive
                pruning parameters.

        Returns:
            CascadePruneResult with selected chunks and stage-level diagnostics.
        """
        if not candidates:
            logger.warning("[CascadePrune] candidate list is empty.")
            return CascadePruneResult(
                prompt_context="",
                selected_chunks=[],
                total_tokens_used=0,
                mode_used=self.mode,
            )

        stage1_input_count = len(candidates)
        logger.info('-' * 60)
        logger.info("[CascadePrune] starting cascade pruning | input candidates: %d", stage1_input_count)
        self._log_tower_distribution(candidates, "input")

        # Dataset-specific handling used by the reproduction workflow.
        if query_timestamp > 0.0:
            reference_time = query_timestamp
            _ref_source = "query_timestamp"
        else:
            valid_ts = [c.timestamp for c in candidates if c.timestamp > 0.0]
            if valid_ts:
                reference_time = max(valid_ts)
                _ref_source = "max(chunk.timestamp)"
            else:
                reference_time = time.time()
                _ref_source = "time.time()"
        logger.info(
            f"[CascadePrune] reference_time={reference_time:.0f} "
            f"(source={_ref_source})"
        )

        if self.enable_stage1:
            valid_chunks = self._stage1_hard_filter(candidates)
        else:
            logger.info("[Stage 1] skipped (enable_stage1=False)")
            valid_chunks = [
                (PrunedChunk(chunk=c, confidence=ConfidenceLevel.HIGH, promoted=False), c)
                for c in candidates
            ]
        stage1_output_count = len(valid_chunks)

        s2_max_drops: Optional[int] = None
        if self.mode == PruneMode.DYNAMIC_ADAPTIVE and query_category is not None:
            _s2_adaptive = get_adaptive_params(
                query_category=query_category,
                dataset=self.adaptive_dataset,
            )
            s2_max_drops = _s2_adaptive.max_stage2_drops
            if s2_max_drops is not None:
                logger.info(
                    f"[Stage 2]  DYNAMIC_ADAPTIVE: category={query_category!r}, "
                    f"complexity={_s2_adaptive.complexity.value}, "
                    f"max_stage2_drops={s2_max_drops}"
                )

        disambiguated, disambiguation_log, s2_stats = self._stage2_disambiguate(
            valid_chunks, reference_time=reference_time,
            max_drops=s2_max_drops,
        )
        stage2_output_count = len(disambiguated)

        sep_tokens = self._estimate_tokens(self.separator)
        chunk_costs: List[int] = [
            self._estimate_tokens(ec.text) for _, ec in disambiguated
        ]
        input_total_tokens = sum(chunk_costs)
        if len(disambiguated) > 1:
            input_total_tokens += sep_tokens * (len(disambiguated) - 1)

        if self.cap_to_input_tokens:
            effective_budget = min(self.max_tokens, input_total_tokens)
            if effective_budget < self.max_tokens:
                logger.info(
                    f"[CascadePrune] cap_to_input: input_tokens={input_total_tokens} "
                    f"< max_tokens={self.max_tokens} -> effective_budget={effective_budget} "
                    f"(prevents context expansion)"
                )
            else:
                logger.info(
                    f"[CascadePrune] input_tokens={input_total_tokens} "
                    f">= max_tokens={self.max_tokens} -> effective_budget={effective_budget}"
                )
        else:
            effective_budget = self.max_tokens
            logger.info(
                f"[CascadePrune] cap_to_input=OFF -> effective_budget={effective_budget}, "
                f"input_tokens={input_total_tokens}"
            )

        selected, tokens_used, s3_stats = self._stage3_mmr_pack(
            disambiguated, effective_budget=effective_budget,
            query_category=query_category,
            chunk_costs=chunk_costs,
            sep_tokens=sep_tokens,
        )

        prompt_context = self.separator.join(pc.chunk.text for pc in selected)

        logger.info("-" * 60)
        logger.info(
            f" [CascadePrune Summary] "
            f"Stage1: {stage1_input_count}->{stage1_output_count} | "
            f"Stage2: {stage1_output_count}->{stage2_output_count} "
            f"(conflict_groups={s2_stats['conflicts']}, dropped={s2_stats['dropped']}) | "
            f"Stage3: {stage2_output_count}->{len(selected)} "
            f"(mmr_iterations={s3_stats['iterations']}, penalties={s3_stats['penalties']}) | "
            f"Token: {tokens_used}/{effective_budget}"
            f"{f' (capped from {self.max_tokens})' if effective_budget < self.max_tokens else ''}"
        )
        self._log_tower_distribution_pruned(selected, "final output")
        logger.info('-' * 60)

        return CascadePruneResult(
            prompt_context=prompt_context,
            selected_chunks=selected,
            total_tokens_used=tokens_used,
            mode_used=self.mode,
            stage1_input_count=stage1_input_count,
            stage1_output_count=stage1_output_count,
            stage2_conflicts_found=s2_stats["conflicts"],
            stage2_dropped_count=s2_stats["dropped"],
            stage2_output_count=stage2_output_count,
            stage3_mmr_iterations=s3_stats["iterations"],
            stage3_diversity_penalties=s3_stats["penalties"],
            disambiguation_log=disambiguation_log,
        )


    def _stage1_hard_filter(
        self, candidates: List[EnhancedCandidateChunk]
    ) -> List[Tuple[PrunedChunk, EnhancedCandidateChunk]]:
        """Drop robust low-score outliers while preserving tower accounting."""
        logger.info("[Stage 1] MAD outlier detection started.")

        if len(candidates) < 2:
            logger.info("[Stage 1] fewer than two candidates; keeping all.")
            return [
                (PrunedChunk(chunk=c, confidence=ConfidenceLevel.HIGH), c)
                for c in candidates
            ]

        scores = [c.ce_score for c in candidates]
        med_s = statistics.median(scores)
        mad_s = statistics.median([abs(s - med_s) for s in scores])

        epsilon = 1e-6
        if mad_s < epsilon:
            logger.info(
                f"[Stage 1] MAD~=0 (median={med_s:.4f}), "
                f"all candidate scores are nearly identical; keeping all."
            )
            return [
                (PrunedChunk(chunk=c, confidence=ConfidenceLevel.HIGH), c)
                for c in candidates
            ]

        bottom_line = med_s - self.mad_multiplier * mad_s

        valid: List[Tuple[PrunedChunk, EnhancedCandidateChunk]] = []
        tower_kept: Dict[str, int] = defaultdict(int)
        tower_dropped: Dict[str, int] = defaultdict(int)

        for c in candidates:
            tower_name = c.tower_source.value if isinstance(c, EnhancedCandidateChunk) else "UNKNOWN"
            if c.ce_score >= bottom_line:
                valid.append(
                    (PrunedChunk(chunk=c, confidence=ConfidenceLevel.HIGH), c)
                )
                tower_kept[tower_name] += 1
            else:
                tower_dropped[tower_name] += 1

        dropped = len(candidates) - len(valid)
        logger.info(
            f"[Stage 1]  MAD: median={med_s:.4f}, MAD={mad_s:.4f}, "
            f"bottom_line={bottom_line:.4f} "
            f"(median - {self.mad_multiplier}*MAD)"
        )
        logger.info(
            f"[Stage 1] complete: kept={len(valid)}, dropped={dropped} | "
            f"kept_distribution={dict(tower_kept)} | dropped_distribution={dict(tower_dropped)}"
        )
        return valid

    

    def _stage2_disambiguate(
        self,
        valid_chunks: List[Tuple[PrunedChunk, EnhancedCandidateChunk]],
        *,
        reference_time: float = 0.0,
        max_drops: Optional[int] = None,
    ) -> Tuple[
        List[Tuple[PrunedChunk, EnhancedCandidateChunk]],
        List[DisambiguationRecord],
        Dict[str, int],
    ]:
        """Resolve cross-tower duplicate and stale-evidence conflicts.

        The stage uses entity and source identifiers to avoid retaining multiple
        towers' versions of the same fact unless the adaptive policy caps the
        number of drops.
        """
        if not self.enable_stage2:
            logger.info("[Stage 2] skipped (enable_stage2=False)")
            return valid_chunks, [], {"conflicts": 0, "dropped": 0}

        logger.info("[Stage 2] cross-tower disambiguation started (global greedy).")

        if reference_time <= 0.0:
            reference_time = max(
                (ec.timestamp for _, ec in valid_chunks if ec.timestamp > 0),
                default=time.time(),
            )

        
        scored: List[Tuple[float, int]] = []
        for idx, (pc, ec) in enumerate(valid_chunks):
            arb_score = self._compute_arbitration_score(pc, ec, reference_time)
            scored.append((arb_score, idx))

        scored.sort(key=lambda x: x[0], reverse=True)

        
        accepted_entity_towers: Dict[str, Set[TowerSource]] = defaultdict(set)
        accepted_source_towers: Dict[str, Set[TowerSource]] = defaultdict(set)
        entity_first_owner: Dict[str, int] = {}
        source_first_owner: Dict[str, int] = {}

        drop_indices: Set[int] = set()
        disambiguation_log: List[DisambiguationRecord] = []
        total_conflicts = 0

        for arb_score, idx in scored:
            pc, ec = valid_chunks[idx]

            
            conflict_key: Optional[str] = None
            conflict_owner_idx: Optional[int] = None  
            for eid in ec.entity_ids:
                if eid in accepted_entity_towers:
                    if ec.tower_source not in accepted_entity_towers[eid]:
                        conflict_key = f"entity:{eid}"
                        conflict_owner_idx = entity_first_owner.get(eid)
                        break

            
            if conflict_key is None:
                for sid in ec.source_chunk_ids:
                    if sid in accepted_source_towers:
                        if ec.tower_source not in accepted_source_towers[sid]:
                            conflict_key = f"source:{sid}"
                            conflict_owner_idx = source_first_owner.get(sid)
                            break

            if conflict_key is not None:
                if max_drops is not None and len(drop_indices) >= max_drops:
                    for eid in ec.entity_ids:
                        accepted_entity_towers[eid].add(ec.tower_source)
                        if eid not in entity_first_owner:
                            entity_first_owner[eid] = idx
                    for sid in ec.source_chunk_ids:
                        accepted_source_towers[sid].add(ec.tower_source)
                        if sid not in source_first_owner:
                            source_first_owner[sid] = idx
                    total_conflicts += 1
                    logger.debug(
                        f"[Stage 2] max_drops={max_drops} reached; "
                        f"keeping chunk_id={ec.chunk_id} "
                        f"(conflict={conflict_key}, tower={ec.tower_source.value})"
                    )
                    continue

                
                drop_indices.add(idx)
                total_conflicts += 1

                if conflict_owner_idx is not None:
                    keeper_ec = valid_chunks[conflict_owner_idx][1]
                    if (self.temporal_decay_hours > 0
                            and keeper_ec.timestamp > 0
                            and ec.timestamp > 0
                            and keeper_ec.timestamp - ec.timestamp
                            > self.temporal_decay_hours * 3600):
                        reason = ConflictResolution.DROPPED_STALE.value
                    elif ec.ce_score < keeper_ec.ce_score * 0.5:
                        reason = ConflictResolution.DROPPED_LOW_SCORE.value
                    else:
                        reason = ConflictResolution.DROPPED_REDUNDANT.value
                    keeper_id = keeper_ec.chunk_id
                else:
                    reason = ConflictResolution.DROPPED_REDUNDANT.value
                    keeper_id = "unknown"

                disambiguation_log.append(DisambiguationRecord(
                    group_key=conflict_key,
                    group_size=2,
                    kept_chunk_ids=[keeper_id],
                    dropped_chunk_ids=[ec.chunk_id],
                    resolution_reasons={
                        keeper_id: ConflictResolution.KEPT.value,
                        ec.chunk_id: reason,
                    },
                    tower_distribution={ec.tower_source.value: 1},
                ))

                logger.debug(
                    f"[Stage 2] cross-tower conflict {conflict_key!r}: "
                    f"dropping chunk_id={ec.chunk_id} "
                    f"(tower={ec.tower_source.value}, score={arb_score:.4f}, reason={reason})"
                )
            else:
                for eid in ec.entity_ids:
                    accepted_entity_towers[eid].add(ec.tower_source)
                    if eid not in entity_first_owner:
                        entity_first_owner[eid] = idx
                for sid in ec.source_chunk_ids:
                    accepted_source_towers[sid].add(ec.tower_source)
                    if sid not in source_first_owner:
                        source_first_owner[sid] = idx

        result = [
            (pc, ec)
            for idx, (pc, ec) in enumerate(valid_chunks)
            if idx not in drop_indices
        ]

        logger.info(
            f"[Stage 2] complete: conflicts={total_conflicts}, "
            f"dropped={len(drop_indices)}, remaining={len(result)}"
            f"{f', max_drops={max_drops} (capped={total_conflicts - len(drop_indices)} kept)' if max_drops is not None else ''}"
        )

        return result, disambiguation_log, {
            "conflicts": total_conflicts,
            "dropped": len(drop_indices),
        }

    def _compute_arbitration_score(
        self,
        pc: PrunedChunk,
        ec: EnhancedCandidateChunk,
        reference_time: float = 0.0,
    ) -> float:
        """Compute the conflict-arbitration score used by stage 2.

        The score combines reranker relevance, tower priority, and temporal
        freshness. Future timestamps receive a reduced temporal bonus so clock
        skew does not dominate conflict resolution.
        """
        w_relevance = 0.5
        w_tower = 0.3
        w_temporal = 0.2

        relevance = ec.ce_score

        
        tower_trust = self.tower_priority.get(ec.tower_source, 0.5)

        if ec.timestamp > 0 and self.temporal_decay_hours > 0 and reference_time > 0:
            if ec.timestamp <= reference_time:
                delta_hours = (reference_time - ec.timestamp) / 3600.0
                temporal_bonus = math.exp(-delta_hours / self.temporal_decay_hours)
            else:
                delta_hours = (ec.timestamp - reference_time) / 3600.0
                temporal_bonus = math.exp(-delta_hours / self.temporal_decay_hours) * 0.2
        else:
            temporal_bonus = 0.5

        score = (w_relevance * relevance
                 + w_tower * tower_trust
                 + w_temporal * temporal_bonus)
        return score

    def _stage3_mmr_pack(
        self,
        disambiguated: List[Tuple[PrunedChunk, EnhancedCandidateChunk]],
        *,
        effective_budget: Optional[int] = None,
        query_category: Optional[str] = None,
        chunk_costs: Optional[List[int]] = None,
        sep_tokens: Optional[int] = None,
    ) -> Tuple[List[PrunedChunk], int, Dict[str, int]]:
        """Pack candidates into the token budget with MMR diversification.

        Stage 3 keeps relevance as the primary signal while penalizing entity,
        source, and tower redundancy. The selected chunks are marked as included
        in-place so downstream prompt assembly can preserve pruning metadata.
        """
        budget = effective_budget if effective_budget is not None else self.max_tokens
        logger.info("[Stage 3] multi-dimensional MMR packing started (budget=%s).", budget)

        if not disambiguated:
            return [], 0, {"iterations": 0, "penalties": 0}

        if not self.enable_stage3_mmr:
            logger.info("[Stage 3] MMR disabled; using greedy packing.")
            return self._greedy_pack(
                disambiguated, budget=budget,
                query_category=query_category,
                chunk_costs=chunk_costs,
                sep_tokens=sep_tokens,
            )

        if sep_tokens is None:
            sep_tokens = self._estimate_tokens(self.separator)
        if chunk_costs is None:
            chunk_costs = [
                self._estimate_tokens(ec.text) for _, ec in disambiguated
            ]

        entity_sets: List[FrozenSet[str]] = [
            frozenset(ec.entity_ids) for _, ec in disambiguated
        ]
        source_sets: List[FrozenSet[str]] = [
            frozenset(ec.source_chunk_ids) for _, ec in disambiguated
        ]
        tower_sources: List[TowerSource] = [
            ec.tower_source for _, ec in disambiguated
        ]
        max_redundancy_cache: List[float] = [0.0] * len(disambiguated)

        n = len(disambiguated)
        selected_indices: List[int] = []
        selected_set: Set[int] = set()
        tokens_used = 0
        penalty_count = 0

        def _accept(idx: int) -> None:
            """Accept one candidate and update redundancy caches."""
            nonlocal tokens_used
            selected_indices.append(idx)
            selected_set.add(idx)
            cost = chunk_costs[idx] + (sep_tokens if len(selected_indices) > 1 else 0)
            tokens_used += cost
            pc, ec = disambiguated[idx]
            pc.included_in_context = True
            for i in range(n):
                if i in selected_set:
                    continue
                redundancy = self._compute_pair_redundancy(
                    i, idx, entity_sets, source_sets, tower_sources
                )
                if redundancy > max_redundancy_cache[i]:
                    max_redundancy_cache[i] = redundancy
            logger.debug(
                f"[Stage 3] iteration {len(selected_indices)}: "
                f"selected chunk_id={ec.chunk_id}, "
                f"tower={ec.tower_source.value}, "
                f"ce_score={ec.ce_score:.4f}, tokens={tokens_used}/{budget}"
            )

        def _run_mmr_phase(
            eligible_filter,
            adaptive_params: Optional[AdaptiveParams] = None,
        ) -> None:
            """Run one MMR phase over candidates accepted by ``eligible_filter``."""
            nonlocal penalty_count
            prev_score: Optional[float] = None

            for _ in range(n):
                best_idx, round_penalties = self._mmr_iteration(
                    disambiguated, eligible_filter, chunk_costs,
                    entity_sets, source_sets, sep_tokens,
                    selected_indices, selected_set, tokens_used,
                    max_redundancy_cache=max_redundancy_cache,
                    budget=budget,
                )
                penalty_count += round_penalties
                if best_idx < 0:
                    break
                _accept(best_idx)

                current_score = disambiguated[best_idx][1].ce_score

                if self.mode == PruneMode.CLIFF_EARLY_STOP:
                    top_score = disambiguated[selected_indices[0]][1].ce_score
                    if (top_score - current_score) > self.cliff_tolerance:
                        logger.debug(
                            f"[Stage 3] CLIFF_EARLY_STOP triggered: "
                            f"top_score={top_score:.4f} - "
                            f"current_score={current_score:.4f} = "
                            f"{top_score - current_score:.4f} > "
                            f"cliff_tolerance={self.cliff_tolerance}"
                        )
                        break

                
                #
                
                #
                
                
                
                #
                
                
                
                if adaptive_params is not None:
                    budget_fraction_limit = int(
                        self.max_tokens * adaptive_params.budget_fraction
                    )
                    budget_exceeded = tokens_used >= budget_fraction_limit
                    enough_chunks = (
                        len(selected_indices) >= adaptive_params.min_chunks
                    )

                    should_stop = enough_chunks and budget_exceeded

                    if should_stop:
                        logger.debug(
                            f"[Stage 3] DYNAMIC_ADAPTIVE early stop: "
                            f"complexity={adaptive_params.complexity.value}, "
                            f"chunks={len(selected_indices)}, "
                            f"tokens={tokens_used}"
                            f">={budget_fraction_limit}"
                            f"(frac={adaptive_params.budget_fraction})"
                        )
                        break

                prev_score = current_score

        
        
        
        
        if self.tower_min_ratio:
            tower_groups: Dict[TowerSource, List[int]] = defaultdict(list)
            for idx, (_, ec) in enumerate(disambiguated):
                tower_groups[ec.tower_source].append(idx)

            
            sorted_towers = sorted(
                self.tower_min_ratio.items(), key=lambda kv: kv[1]
            )
            for tower, ratio in sorted_towers:
                if ratio <= 0 or tower not in tower_groups:
                    continue
                reserved_budget = int(ratio * budget)
                
                indices = sorted(
                    tower_groups[tower],
                    key=lambda i: disambiguated[i][1].ce_score,
                    reverse=True,
                )
                tower_tokens_used = 0
                for idx in indices:
                    if idx in selected_set:
                        continue
                    cost = chunk_costs[idx] + (
                        sep_tokens if selected_indices else 0
                    )
                    if tower_tokens_used + cost > reserved_budget:
                        continue
                    if tokens_used + cost > budget:
                        continue  
                    _accept(idx)
                    tower_tokens_used += cost

            _tower_dist = defaultdict(int)
            for i in selected_indices:
                _tower_dist[disambiguated[i][1].tower_source.value] += 1
            logger.info(
                f"[Stage 3] tower reservation complete: "
                f"reserved={len(selected_indices)} chunks, "
                f"tokens={tokens_used}/{budget}, "
                f"tower_distribution={dict(_tower_dist)}"
            )

        if self.mode == PruneMode.STRICT_THRESHOLD:
            def strict_filter(i: int) -> bool:
                return disambiguated[i][1].ce_score >= self.absolute_min_score
            _run_mmr_phase(strict_filter)

        elif self.mode == PruneMode.BUDGET_MAX:
            
            _run_mmr_phase(lambda i: True)

        elif self.mode == PruneMode.CLIFF_EARLY_STOP:
            _run_mmr_phase(lambda i: True)

        elif self.mode == PruneMode.DYNAMIC_ADAPTIVE:
            _adaptive = get_adaptive_params(
                query_category=query_category,
                dataset=self.adaptive_dataset,
            )
            logger.info(
                f"[Stage 3]  DYNAMIC_ADAPTIVE: "
                f"category={query_category!r}, "
                f"complexity={_adaptive.complexity.value}, "
                f"cliff_mult={_adaptive.cliff_tolerance_multiplier}, "
                f"grad_thresh={_adaptive.gradient_threshold}, "
                f"min_chunks={_adaptive.min_chunks}, "
                f"budget_frac={_adaptive.budget_fraction}"
            )
            _run_mmr_phase(lambda i: True, adaptive_params=_adaptive)

        else:
            _run_mmr_phase(lambda i: True)

        selected = [disambiguated[i][0] for i in selected_indices]

        logger.info(
            f"[Stage 3] complete: selected={len(selected)}, "
            f"mmr_iterations={len(selected_indices)}, "
            f"diversity_penalties={penalty_count}, "
            f"tower_distribution={self._count_towers_pruned(selected)}"
        )

        return selected, tokens_used, {
            "iterations": len(selected_indices),
            "penalties": penalty_count,
        }

    def _mmr_iteration(
        self,
        disambiguated: List[Tuple[PrunedChunk, EnhancedCandidateChunk]],
        candidate_filter,
        chunk_costs: List[int],
        entity_sets: List[FrozenSet[str]],
        source_sets: List[FrozenSet[str]],
        sep_tokens: int,
        selected_indices: List[int],
        selected_set: Set[int],
        tokens_used: int,
        *,
        max_redundancy_cache: Optional[List[float]] = None,
        budget: Optional[int] = None,
    ) -> Tuple[int, int]:
        """Select one feasible candidate with the highest MMR score."""
        effective_max = budget if budget is not None else self.max_tokens
        n = len(disambiguated)
        best_idx = -1
        best_mmr = -float("inf")
        penalties = 0

        for i in range(n):
            if i in selected_set:
                continue
            if not candidate_filter(i):
                continue

            cost = chunk_costs[i] + (sep_tokens if selected_indices else 0)
            if tokens_used + cost > effective_max:
                continue

            _, ec_i = disambiguated[i]
            relevance = ec_i.ce_score

            if not selected_indices:
                redundancy = 0.0
            elif max_redundancy_cache is not None:
                redundancy = max_redundancy_cache[i]
            else:
                redundancy = self._compute_max_redundancy(
                    i, ec_i, entity_sets, source_sets,
                    selected_indices, disambiguated,
                    set(), set(),
                )

            mmr = self.lambda_mmr * relevance - (1 - self.lambda_mmr) * redundancy

            if redundancy > 0.01:
                penalties += 1

            if mmr > best_mmr:
                best_mmr = mmr
                best_idx = i

        return best_idx, penalties

    def _compute_pair_redundancy(
        self,
        candidate_idx: int,
        selected_idx: int,
        entity_sets: List[FrozenSet[str]],
        source_sets: List[FrozenSet[str]],
        tower_sources: List[TowerSource],
    ) -> float:
        """Compute pair redundancy."""
        alpha = 0.4
        beta = 0.4
        gamma = 0.2  

        cand_entities = entity_sets[candidate_idx]
        sel_entities = entity_sets[selected_idx]
        if cand_entities or sel_entities:
            entity_intersection = len(cand_entities & sel_entities)
            entity_union = len(cand_entities | sel_entities)
            entity_overlap = entity_intersection / entity_union if entity_union > 0 else 0.0
        else:
            entity_overlap = 0.0

        cand_sources = source_sets[candidate_idx]
        sel_sources = source_sets[selected_idx]
        if cand_sources:
            source_intersection = len(cand_sources & sel_sources)
            source_overlap = source_intersection / len(cand_sources)
        else:
            source_overlap = 0.0

        tower_same = 1.0 if tower_sources[candidate_idx] == tower_sources[selected_idx] else 0.0

        return (
            alpha * entity_overlap
            + (self.entity_overlap_penalty if entity_overlap > 0 else 0.0)
            + beta * source_overlap
            + (self.source_duplicate_penalty if source_overlap > 0 else 0.0)
            + gamma * tower_same
        )

    def _compute_max_redundancy(
        self,
        candidate_idx: int,
        candidate_ec: EnhancedCandidateChunk,
        entity_sets: List[FrozenSet[str]],
        source_sets: List[FrozenSet[str]],
        selected_indices: List[int],
        disambiguated: List[Tuple[PrunedChunk, EnhancedCandidateChunk]],
        selected_entities: Set[str],
        selected_sources: Set[str],
    ) -> float:
        """Redundancy(d_i, d_j) = α · EntityOverlap(d_i, d_j) + (entity_overlap_penalty if EntityOverlap > 0 else 0) + β · SourceOverlap(d_i, d_j) + (source_duplicate_penalty if SourceOverlap > 0 else 0) + γ · TowerSameness(d_i, d_j)."""
        tower_sources: List[TowerSource] = [
            ec.tower_source for _, ec in disambiguated
        ]

        max_redundancy = 0.0

        for j in selected_indices:
            redundancy = self._compute_pair_redundancy(
                candidate_idx, j, entity_sets, source_sets, tower_sources
            )

            max_redundancy = max(max_redundancy, redundancy)

        return max_redundancy

    def _greedy_pack(
        self,
        disambiguated: List[Tuple[PrunedChunk, EnhancedCandidateChunk]],
        *,
        budget: Optional[int] = None,
        query_category: Optional[str] = None,
        chunk_costs: Optional[List[int]] = None,
        sep_tokens: Optional[int] = None,
    ) -> Tuple[List[PrunedChunk], int, Dict[str, int]]:
        """Run greedy pack."""
        effective_max = budget if budget is not None else self.max_tokens
        if sep_tokens is None:
            sep_tokens = self._estimate_tokens(self.separator)
        if chunk_costs is None:
            chunk_costs = [
                self._estimate_tokens(ec.text) for _, ec in disambiguated
            ]

        ordered_indices = sorted(
            range(len(disambiguated)),
            key=lambda i: disambiguated[i][1].ce_score,
            reverse=True,
        )

        selected: List[PrunedChunk] = []
        tokens_used = 0
        top_score: Optional[float] = None
        prev_score: Optional[float] = None

        adaptive_params: Optional[AdaptiveParams] = None
        if self.mode == PruneMode.DYNAMIC_ADAPTIVE:
            adaptive_params = get_adaptive_params(
                query_category=query_category,
                dataset=self.adaptive_dataset,
            )

        for idx in ordered_indices:
            pc, ec = disambiguated[idx]
            if self.mode == PruneMode.STRICT_THRESHOLD:
                if ec.ce_score < self.absolute_min_score:
                    continue

            cost = chunk_costs[idx] + (sep_tokens if selected else 0)
            if tokens_used + cost > effective_max:
                if self.mode in (PruneMode.BUDGET_MAX, PruneMode.DYNAMIC_ADAPTIVE):
                    continue
                break

            pc.included_in_context = True
            selected.append(pc)
            tokens_used += cost

            if top_score is None:
                top_score = ec.ce_score

            if self.mode == PruneMode.CLIFF_EARLY_STOP and top_score is not None:
                if (top_score - ec.ce_score) > self.cliff_tolerance:
                    break

            
            if adaptive_params is not None:
                budget_fraction_limit = int(
                    self.max_tokens * adaptive_params.budget_fraction
                )
                budget_exceeded = tokens_used >= budget_fraction_limit
                enough_chunks = len(selected) >= adaptive_params.min_chunks
                if enough_chunks and budget_exceeded:
                    break

            prev_score = ec.ce_score

        return selected, tokens_used, {"iterations": len(selected), "penalties": 0}


    def _estimate_tokens(self, text: str) -> int:
        """Estimate prompt tokens using the configured tokenizer."""
        if not text:
            return 0
        return len(self.tokenizer.encode(text))

    def _log_tower_distribution(
        self,
        candidates: List[EnhancedCandidateChunk],
        label: str,
    ) -> None:
        """Log candidate counts by retrieval tower."""
        dist: Dict[str, int] = defaultdict(int)
        for c in candidates:
            dist[c.tower_source.value] += 1
        logger.info(f"[CascadePrune] {label} tower_distribution: {dict(dist)}")

    def _log_tower_distribution_pruned(
        self,
        selected: List[PrunedChunk],
        label: str,
    ) -> None:
        """Log selected chunk counts by retrieval tower."""
        dist = self._count_towers_pruned(selected)
        logger.info(f"[CascadePrune] {label} tower_distribution: {dist}")

    @staticmethod
    def _count_towers_pruned(selected: List[PrunedChunk]) -> Dict[str, int]:
        """Count towers pruned."""
        dist: Dict[str, int] = defaultdict(int)
        for pc in selected:
            chunk = pc.chunk
            if isinstance(chunk, EnhancedCandidateChunk):
                dist[chunk.tower_source.value] += 1
            else:
                dist["UNKNOWN"] += 1
        return dict(dist)




def create_cascade_pruner(
    max_tokens: int = 2000,
    lambda_mmr: float = 0.6,
    mode: str = "budget_max",
    mad_multiplier: float = 2.5,
    cliff_tolerance: float = 2.0,
    absolute_min_score: float = 0.0,
    adaptive_dataset: Optional[str] = None,
    **kwargs,
) -> CascadeConfidencePruner:
    """Parameters max_tokens : int lambda_mmr : float mode : str "budget_max" / "dynamic_adaptive")。 mad_multiplier : float cliff_tolerance : float absolute_min_score : float adaptive_dataset : Optional[str] **kwargs Returns CascadeConfidencePrune."""
    mode_enum = PruneMode(mode.upper())
    return CascadeConfidencePruner(
        mode=mode_enum,
        max_tokens=max_tokens,
        lambda_mmr=lambda_mmr,
        mad_multiplier=mad_multiplier,
        cliff_tolerance=cliff_tolerance,
        absolute_min_score=absolute_min_score,
        adaptive_dataset=adaptive_dataset,
        **kwargs,
    )
