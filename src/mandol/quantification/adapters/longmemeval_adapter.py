"""Utilities for longmemeval adapter."""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any, Dict, List

from ..cascade_pruner import EnhancedCandidateChunk, TowerSource
from .locomo_adapter import BasePrunerAdapter

logger = logging.getLogger("quantification.adapters.longmemeval_adapter")


class LongMemEvalPrunerAdapter(BasePrunerAdapter):
    """``List[EnhancedCandidateChunk]``。 { "sentence": [ {"uid": "...", "score": 0.75, "ce_score": 0.82, "text_content": "...", "source_type": "sentence", "role": "Alice", "session_date": "2023/05/30 (Tue) 17:27"}, ... ], "episodic": [ {"uid": ".."""

    _WEEKDAY_RE = re.compile(r"\s*\([A-Za-z]+\)\s*")

    def adapt(self, retrieved_contents: Dict[str, Any]) -> List[EnhancedCandidateChunk]:
        """Adapt."""
        chunks: List[EnhancedCandidateChunk] = []

        for item in retrieved_contents.get("sentence", []):
            chunks.append(self._convert_sentence_item(item))

        for item in retrieved_contents.get("episodic", []):
            chunks.append(self._convert_episodic_item(item))

        for item in retrieved_contents.get("entity", []):
            chunks.append(self._convert_entity_item(item))

        logger.debug(
            f"[LongMemEvalPrunerAdapter] conversion complete: "
            f"hier={sum(1 for c in chunks if c.tower_source == TowerSource.HIERARCHICAL)}, "
            f"epi={sum(1 for c in chunks if c.tower_source == TowerSource.EPISODIC)}, "
            f"kg={sum(1 for c in chunks if c.tower_source == TowerSource.KG)}, "
            f"total={len(chunks)}"
        )
        return chunks

    

    def _convert_sentence_item(self, item: Dict[str, Any]) -> EnhancedCandidateChunk:
        text = item.get("text_content", "") or ""
        uid = item.get("uid", "") or ""
        score = float(item.get("ce_score", item.get("score", 0.0)))
        session_date = item.get("session_date", "") or ""
        role = item.get("role", "") or ""

        chunk_id = uid if uid else self.generate_deterministic_id(text)
        timestamp = self._parse_longmemeval_date(session_date)

        entity_ids: List[str] = []
        if role and role.lower() not in ("unknown", "system", "assistant", "user"):
            entity_ids.append(role.lower().strip())

        for e in self._extract_fallback_entities(text):
            ner_id = e.lower().strip()
            if ner_id not in entity_ids:
                entity_ids.append(ner_id)

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
            entity_ids=entity_ids,
            memory_space="sentence",
            fact_type="",
            raw_metadata=dict(item),
        )

    def _convert_episodic_item(self, item: Dict[str, Any]) -> EnhancedCandidateChunk:
        text = item.get("text_content", "") or ""
        uid = item.get("uid", "") or ""
        score = float(item.get("ce_score", item.get("score", 0.0)))
        event_date = item.get("event_date", "") or ""

        chunk_id = uid if uid else self.generate_deterministic_id(text)
        timestamp = self._parse_longmemeval_date(event_date)

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
            raw_metadata=dict(item),
        )

    def _convert_entity_item(self, item: Dict[str, Any]) -> EnhancedCandidateChunk:
        text = item.get("text_content", "") or ""
        uid = item.get("uid", "") or ""
        score = float(item.get("ce_score", item.get("score", 0.0)))
        entity_name = item.get("entity_name", "") or ""
        session_date = item.get("session_date", "") or ""

        if not text:
            entity_type = item.get("entity_type", "Unknown")
            description = item.get("description", "")
            text = f"Entity: {entity_name} (Type: {entity_type}) | Context: {description}"

        chunk_id = uid if uid else self.generate_deterministic_id(text)
        timestamp = self._parse_longmemeval_date(session_date)

        entity_ids = [entity_name.lower().strip()] if entity_name else []

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
            memory_space="entity",
            fact_type="",
            raw_metadata=dict(item),
        )


    def _parse_longmemeval_date(self, date_str: str) -> float:
        """Parse longmemeval date."""
        if not date_str or not isinstance(date_str, str):
            return 0.0

        cleaned = self._WEEKDAY_RE.sub(" ", date_str).strip()

        try:
            return datetime.strptime(cleaned, "%Y/%m/%d %H:%M").timestamp()
        except (ValueError, TypeError):
            pass

        return self.parse_timestamp(cleaned)
