"""Utilities for locomo adapter."""

from __future__ import annotations

import hashlib
import logging
import re
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List

from ..cascade_pruner import EnhancedCandidateChunk, TowerSource

logger = logging.getLogger("quantification.adapters.locomo_adapter")




class BasePrunerAdapter(ABC):


    @staticmethod
    def parse_timestamp(date_str: str) -> float:
        """Parse timestamp."""
        if not date_str or not isinstance(date_str, str):
            return 0.0
        date_str = date_str.strip()
        if date_str.lower() in ("", "unknown date", "unknown", "n/a", "none"):
            return 0.0

        _FORMATS = (
            "%d %B, %Y",           # "7 May, 2023"
            "%d %B %Y",            # "7 May 2023"
            "%B %d, %Y",           # "May 7, 2023"
            "%Y-%m-%d",            # "2023-05-07"
            "%Y-%m-%d %H:%M:%S",   # "2023-05-07 12:00:00"
            "%Y/%m/%d",            # "2023/05/07"
            "%m/%d/%Y",            # "05/07/2023"
            "%Y-%m-%dT%H:%M:%S",   # ISO
            "%Y-%m-%dT%H:%M:%SZ",  # ISO-Z
            "%d %b %Y",            # "7 May 2023" (abbreviated month)
            "%d %b, %Y",           # "7 May, 2023"
        )
        for fmt in _FORMATS:
            try:
                return datetime.strptime(date_str, fmt).timestamp()
            except (ValueError, TypeError):
                continue

        if date_str.isdigit() and len(date_str) == 4:
            try:
                return datetime(int(date_str), 1, 1).timestamp()
            except (ValueError, TypeError):
                pass

        return 0.0

    @staticmethod
    def generate_deterministic_id(text: str) -> str:
        """Generate deterministic id."""
        digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:10]
        return f"gen_{digest}"


    _RE_PROPER_NOUN = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")
    _RE_QUOTED = re.compile(r'"([^"]{2,50})"')

    @staticmethod
    def _extract_fallback_entities(text: str) -> List[str]:
        """Extract fallback entities."""
        if not text or len(text) < 3:
            return []

        entities: List[str] = []

        for sentence in re.split(r'[.!?\n]', text):
            sentence = sentence.strip()
            if not sentence:
                continue
            first_space = sentence.find(' ')
            if first_space < 0:
                continue
            after_first = sentence[first_space + 1:]
            for m in BasePrunerAdapter._RE_PROPER_NOUN.finditer(after_first):
                entities.append(m.group(1).strip().lower())

        for m in BasePrunerAdapter._RE_QUOTED.finditer(text):
            entities.append(m.group(1).strip().lower())

        seen: set = set()
        unique: List[str] = []
        for e in entities:
            if e and e not in seen:
                seen.add(e)
                unique.append(e)
        return unique


    @abstractmethod
    def adapt(self, retrieved_contents: Dict[str, Any]) -> List[EnhancedCandidateChunk]:
        """Adapt."""
        ...





_SOURCE_TYPE_MAP: Dict[str, TowerSource] = {
    "hierarchical": TowerSource.HIERARCHICAL,
    "episodic": TowerSource.EPISODIC,
    "entity_relation": TowerSource.KG,
}


def _resolve_tower_source(source_type: str) -> TowerSource:
    """Resolve tower source."""
    st_lower = source_type.lower()
    for keyword, tower in _SOURCE_TYPE_MAP.items():
        if keyword in st_lower:
            return tower
    return TowerSource.HIERARCHICAL


class LocomoPrunerAdapter(BasePrunerAdapter):
    """``List[EnhancedCandidateChunk]``。 { "tower1_hierarchical": { "by_layer": { "l0": {"count": N, "units": [...]}, "l1": {"count": N, "units": [...]}, "l2": {"count": N, "units": [...]}, } }, "tower2_entity_relation": { "units": [...] }, "tower."""

    def adapt(self, retrieved_contents: Dict[str, Any]) -> List[EnhancedCandidateChunk]:
        """Adapt."""
        chunks: List[EnhancedCandidateChunk] = []

        
        tower1 = retrieved_contents.get("tower1_hierarchical", {})
        by_layer = tower1.get("by_layer", {})
        for layer_name in ("l0", "l1", "l2"):
            layer_data = by_layer.get(layer_name, {})
            units = layer_data.get("units", []) if isinstance(layer_data, dict) else []
            for unit in units:
                chunks.append(self._convert_hierarchical_unit(unit, layer_name))

        
        tower2 = retrieved_contents.get("tower2_entity_relation", {})
        for unit in tower2.get("units", []):
            chunks.append(self._convert_entity_relation_unit(unit))

        
        tower3 = retrieved_contents.get("tower3_episodic", {})
        for unit in tower3.get("units", []):
            chunks.append(self._convert_episodic_unit(unit))

        logger.debug(
            f"[LocomoPrunerAdapter] conversion complete: "
            f"hier={sum(1 for c in chunks if c.tower_source == TowerSource.HIERARCHICAL)}, "
            f"kg={sum(1 for c in chunks if c.tower_source == TowerSource.KG)}, "
            f"epi={sum(1 for c in chunks if c.tower_source == TowerSource.EPISODIC)}, "
            f"total={len(chunks)}"
        )
        return chunks

    

    def _convert_hierarchical_unit(
        self, unit: Dict[str, Any], layer_name: str
    ) -> EnhancedCandidateChunk:
        text = unit.get("text_content", "") or ""
        uid = unit.get("uid", "") or ""
        score = float(unit.get("score", 0.0))
        session_date = unit.get("session_date", "") or ""

        chunk_id = uid if uid else self.generate_deterministic_id(text)
        timestamp = self.parse_timestamp(session_date)

        return EnhancedCandidateChunk(
            chunk_id=chunk_id,
            text=text,
            ce_score=score,
            rank_dense=9999,
            rank_splade=9999,
            rank_bm25=9999,
            tower_source=TowerSource.HIERARCHICAL,
            source_chunk_ids=[chunk_id],
            timestamp=timestamp,
            entity_ids=[e.lower().strip() for e in self._extract_fallback_entities(text)],
            memory_space=f"hierarchical_{layer_name}",
            fact_type="",
        )

    def _convert_entity_relation_unit(
        self, unit: Dict[str, Any]
    ) -> EnhancedCandidateChunk:
        text = unit.get("text_content", "") or ""
        uid = unit.get("uid", "") or ""
        score = float(unit.get("score", 0.0))
        entity_name = unit.get("entity_name", "") or ""

        chunk_id = uid if uid else self.generate_deterministic_id(text)

        entity_ids = [entity_name.lower().strip()] if entity_name else []

        timestamp = self._extract_kg_timestamp(unit)

        return EnhancedCandidateChunk(
            chunk_id=chunk_id,
            text=text,
            ce_score=score,
            rank_dense=9999,
            rank_splade=9999,
            rank_bm25=9999,
            tower_source=TowerSource.KG,
            source_chunk_ids=[chunk_id],
            timestamp=timestamp,
            entity_ids=entity_ids,
            memory_space="entity_relation",
            fact_type="",
        )

    def _convert_episodic_unit(
        self, unit: Dict[str, Any]
    ) -> EnhancedCandidateChunk:
        text = unit.get("text_content", "") or ""
        uid = unit.get("uid", "") or ""
        score = float(unit.get("score", 0.0))
        event_date = unit.get("event_date", "") or ""

        chunk_id = uid if uid else self.generate_deterministic_id(text)
        timestamp = self.parse_timestamp(event_date)

        entity_ids = [e.lower().strip() for e in self._extract_fallback_entities(text)]

        return EnhancedCandidateChunk(
            chunk_id=chunk_id,
            text=text,
            ce_score=score,
            rank_dense=9999,
            rank_splade=9999,
            rank_bm25=9999,
            tower_source=TowerSource.EPISODIC,
            source_chunk_ids=[chunk_id],
            timestamp=timestamp,
            entity_ids=entity_ids,
            memory_space="episodic",
            fact_type="",
        )


    def _extract_kg_timestamp(self, unit: Dict[str, Any]) -> float:
        """Extract kg timestamp."""
        entity_name = unit.get("entity_name", "") or ""
        ts = self.parse_timestamp(entity_name)
        if ts > 0.0:
            return ts

        text = unit.get("text_content", "") or ""
        if "Time:" in text:
            after_time = text.split("Time:")[-1]
            time_part = after_time.split("|")[0].strip().rstrip(".")
            ts = self.parse_timestamp(time_part)
            if ts > 0.0:
                return ts

        return 0.0
