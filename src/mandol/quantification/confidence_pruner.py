"""Utilities for confidence pruner."""

from dataclasses import dataclass
from enum import Enum
from typing import List

import tiktoken

from ..utils.logging_config import create_module_logger

logger = create_module_logger("quantification.confidence_pruner")


class ConfidenceLevel(Enum):

    HIGH = "HIGH"
    MED = "MED"
    DROP = "DROP"


class PruneMode(Enum):

    STRICT_THRESHOLD = "STRICT_THRESHOLD"
    CUMULATIVE_EARLY_STOP = "CUMULATIVE_EARLY_STOP"
    CLIFF_EARLY_STOP = "CLIFF_EARLY_STOP"
    BUDGET_MAX = "BUDGET_MAX"
    DYNAMIC_ADAPTIVE = "DYNAMIC_ADAPTIVE"


@dataclass
class CandidateChunk:

    chunk_id: str
    text: str
    ce_score: float
    rank_dense: int
    rank_splade: int
    rank_bm25: int


@dataclass
class PrunedChunk:

    chunk: CandidateChunk
    confidence: ConfidenceLevel
    promoted: bool = False
    included_in_context: bool = False


@dataclass
class PruneResult:

    prompt_context: str
    selected_chunks: List[PrunedChunk]
    total_tokens_used: int
    mode_used: PruneMode


class ConfidenceAwarePruner:

    def __init__(
        self,
        mode: PruneMode = PruneMode.CUMULATIVE_EARLY_STOP,
        tau_high: float = 0.7,
        tau_med: float = 0.3,
        mu: int = 10,
        target_confidence_sum: float = 2.5,
        max_tokens: int = 4096,
        decay_factor: float = 0.9,
        separator: str = "\n\n",
        use_relative_scoring: bool = False,
    ):
        if tau_med > tau_high:
            raise ValueError(
                f"tau_med ({tau_med}) cannot exceed tau_high ({tau_high}); otherwise the MED interval is empty."
            )
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than 0")
        if mu <= 0:
            raise ValueError("mu must be greater than 0")

        self.mode = mode
        self.tau_high = tau_high
        self.tau_med = tau_med
        self.mu = mu
        self.target_confidence_sum = target_confidence_sum
        self.max_tokens = max_tokens
        self.decay_factor = decay_factor
        self.separator = separator
        self.use_relative_scoring = use_relative_scoring

        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            raise RuntimeError(
                f"tiktoken initialization failed (encoding='cl100k_base'): {e}. "
                "Install tiktoken with: pip install tiktoken"
            ) from e

        logger.info(
            "[ConfidenceAwarePruner] initialized: "
            f"mode={mode.value}, tau_high={tau_high}, tau_med={tau_med}, "
            f"mu={mu}, target_confidence_sum={target_confidence_sum}, "
            f"max_tokens={max_tokens}, decay_factor={decay_factor}, "
            f"use_relative_scoring={use_relative_scoring}"
        )

    def prune(self, candidates: List[CandidateChunk]) -> PruneResult:
        """Prune."""
        if not candidates:
            logger.warning("[Prune] candidates is empty")
            return PruneResult(
                prompt_context="",
                selected_chunks=[],
                total_tokens_used=0,
                mode_used=self.mode,
            )

        if self.use_relative_scoring:
            scores = [c.ce_score for c in candidates]
            min_s = min(scores)
            max_s = max(scores)
            if max_s > min_s:
                score_range = max_s - min_s
                for c in candidates:
                    c.ce_score = (c.ce_score - min_s) / score_range
                normalized_min = 0.0
                normalized_max = 1.0
            else:
                for c in candidates:
                    c.ce_score = 1.0
                normalized_min = 1.0
                normalized_max = 1.0
            logger.info(
                "[Prune] relative score normalization: min_s=%.4f, max_s=%.4f, "
                "normalized ce_score range=[%.4f, %.4f]",
                min_s,
                max_s,
                normalized_min,
                normalized_max,
            )

        valid_chunks: List[PrunedChunk] = []
        promoted_count = 0
        for c in candidates:
            pc = self._classify(c)
            if pc.promoted:
                promoted_count += 1
            if pc.confidence != ConfidenceLevel.DROP:
                valid_chunks.append(pc)
        if promoted_count > 0:
            logger.info(f"[Prune] cross-tower consensus promoted MED candidates to HIGH: {promoted_count}")

        valid_chunks.sort(key=lambda pc: pc.chunk.ce_score, reverse=True)
        token_costs = {
            id(pc): self._estimate_tokens(pc.chunk.text)
            for pc in valid_chunks
        }

        selected: List[PrunedChunk] = []
        tokens_used = 0
        confidence_sum = 0.0
        sep_tokens = self._estimate_tokens(self.separator)

        if self.mode == PruneMode.STRICT_THRESHOLD:
            for pc in valid_chunks:
                if pc.chunk.ce_score < self.tau_high and not pc.promoted:
                    continue

                cost = token_costs[id(pc)] + (sep_tokens if selected else 0)
                if tokens_used + cost > self.max_tokens:
                    logger.debug(" [Prune] Token budget reached (%s/%s)", tokens_used, self.max_tokens)
                    break

                pc.included_in_context = True
                selected.append(pc)
                tokens_used += cost

        elif self.mode == PruneMode.CUMULATIVE_EARLY_STOP:
            for i, pc in enumerate(valid_chunks):
                cost = token_costs[id(pc)] + (sep_tokens if selected else 0)
                if tokens_used + cost > self.max_tokens:
                    logger.debug("[Prune] token budget reached (%s/%s)", tokens_used, self.max_tokens)
                    break

                pc.included_in_context = True
                selected.append(pc)
                tokens_used += cost
                confidence_sum += pc.chunk.ce_score * (self.decay_factor ** i)

                if confidence_sum >= self.target_confidence_sum:
                    logger.info(
                        f"[Prune] Early stop at chunk {i + 1}: "
                        f"cumulative confidence reached the target "
                        f"({confidence_sum:.2f} >= {self.target_confidence_sum})"
                    )
                    break

        elif self.mode == PruneMode.BUDGET_MAX:
            high_chunks = []
            med_chunks = []
            for pc in valid_chunks:
                if pc.confidence == ConfidenceLevel.HIGH:
                    high_chunks.append(pc)
                elif pc.confidence == ConfidenceLevel.MED:
                    med_chunks.append(pc)

            for pc in high_chunks:
                cost = token_costs[id(pc)] + (sep_tokens if selected else 0)
                if tokens_used + cost > self.max_tokens:
                    break
                pc.included_in_context = True
                selected.append(pc)
                tokens_used += cost

            for pc in med_chunks:
                cost = token_costs[id(pc)] + (sep_tokens if selected else 0)
                if tokens_used + cost > self.max_tokens:
                    break
                pc.included_in_context = True
                selected.append(pc)
                tokens_used += cost

        else:
            raise ValueError(f"Unsupported pruning mode: {self.mode}")

        prompt_context = self.separator.join(pc.chunk.text for pc in selected)

        logger.info(
            f"[Prune Summary] mode={self.mode.value} | input={len(candidates)} | "
            f"valid={len(valid_chunks)} | selected={len(selected)} | "
            f"tokens={tokens_used}/{self.max_tokens}"
        )

        return PruneResult(
            prompt_context=prompt_context,
            selected_chunks=selected,
            total_tokens_used=tokens_used,
            mode_used=self.mode,
        )
    

    def _classify(self, candidate: CandidateChunk) -> PrunedChunk:
        """Classify."""
        score = candidate.ce_score

        if score >= self.tau_high:
            return PrunedChunk(chunk=candidate, confidence=ConfidenceLevel.HIGH)

        if score < self.tau_med:
            return PrunedChunk(chunk=candidate, confidence=ConfidenceLevel.DROP)

        votes = sum([
            candidate.rank_dense <= self.mu,
            candidate.rank_splade <= self.mu,
            candidate.rank_bm25 <= self.mu,
        ])
        if votes >= 2:
            return PrunedChunk(
                chunk=candidate,
                confidence=ConfidenceLevel.HIGH,
                promoted=True,
            )

        return PrunedChunk(chunk=candidate, confidence=ConfidenceLevel.MED)

    def _estimate_tokens(self, text: str) -> int:
        """Estimate tokens."""
        if not text:
            return 0
        return len(self.tokenizer.encode(text))
