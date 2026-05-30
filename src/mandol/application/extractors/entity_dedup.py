"""Entity deduplication using LLM-based cluster judging.

Identifies duplicate entities within and across sessions by clustering
candidates via vector similarity + graph structure overlap, then asking
an LLM to confirm merges and select canonical names.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ...domain.memory_unit import MemoryUnit
from ...domain.types import Uid
from ...ports.embedding_provider import EmbeddingProvider
from ...ports.llm_provider import ChatMessage, LLMProvider

logger = logging.getLogger(__name__)

ENTITY_DEDUP_SYSTEM_PROMPT = """You are a professional entity standardization expert. Identify different expressions pointing to the same real entity.

Entity candidate list:
{entity_list}

Output JSON:
{
    "merged_entities": [
        {
            "canonical_form": "Standardized entity name",
            "entity_type": "Unified entity type",
            "confidence": 0.95,
            "source_entities": [1, 3, 5],
            "aliases": ["Variant 1", "Variant 2"],
            "sessions": ["session_1", "session_3"],
            "reasoning": "Merge reasoning"
        }
    ]
}

Principles:
- Same entity, different expressions → Merge
- Different entities, similar expressions → Do NOT merge
- Preserve aliases and compositional relationships
- Output only valid JSON"""


@dataclass
class EntityCandidate:
    index: int
    text: str
    entity_type: str
    context: str
    sessions: List[str]
    source_uids: List[str]


class EntityDeduplicator:
    """Clusters and merges duplicate entities using embedding + LLM judging.

    Uses vector similarity pre-clustering (cosine) and LLM decision-making
    to detect and merge entity duplicates within and across sessions.

    Args:
        llm_provider: LLM provider for cluster judge calls.
        embedder: Optional embedding provider for vector pre-clustering.
        similarity_threshold: Minimum cosine similarity for clustering (default 0.85).
        max_candidates_per_llm: Maximum candidates per LLM call (default 30).
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        embedder: Optional[EmbeddingProvider] = None,
        similarity_threshold: float = 0.85,
        max_candidates_per_llm: int = 30,
    ):
        self._llm = llm_provider
        self._embedder = embedder
        self._threshold = float(similarity_threshold)
        self._max_candidates = int(max_candidates_per_llm)

    def deduplicate(
        self,
        entities: Sequence[MemoryUnit],
    ) -> List[MemoryUnit]:
        """Deduplicate a batch of entity memory units.

        Extracts candidates, preclusters by embedding similarity, sends
        clusters to LLM for merge decisions, and returns canonical units.

        Args:
            entities: Raw entity MemoryUnits to deduplicate.

        Returns:
            List of canonical MemoryUnits after merging.
        """
        if not entities:
            return []

        candidates = self._extract_candidates(entities)
        if not candidates:
            return []

        clusters = self._precluster_by_embedding(candidates)

        merged_entities: List[Dict[str, Any]] = []
        for cluster in clusters:
            if len(cluster) == 1:
                c = cluster[0]
                merged_entities.append({
                    "canonical_form": c.text,
                    "entity_type": c.entity_type,
                    "confidence": 0.9,
                    "source_entities": [c.index],
                    "aliases": [],
                    "sessions": c.sessions,
                    "source_uids": c.source_uids,
                })
            else:
                chunked = [cluster[i:i + self._max_candidates]
                          for i in range(0, len(cluster), self._max_candidates)]
                for chunk in chunked:
                    result = self._llm_merge(chunk)
                    merged_entities.extend(result)

        result_units = []
        for me in merged_entities:
            uid = f"entity:{me['canonical_form'][:50].lower().replace(' ', '_')}"
            source_uids = me.get("source_uids", [])
            unit = MemoryUnit(
                uid=Uid(uid),
                raw_data={
                    "text_content": me["canonical_form"],
                    "entity_type": me.get("entity_type", ""),
                    "aliases": me.get("aliases", []),
                },
                metadata={
                    "type": "entity",
                    "entity_type": me.get("entity_type", ""),
                    "aliases": me.get("aliases", []),
                    "merged_from_uids": source_uids,
                    "evidence_uids": source_uids,
                },
            )
            result_units.append(unit)

        return result_units

    def _extract_candidates(
        self,
        entities: Sequence[MemoryUnit],
    ) -> List[EntityCandidate]:
        """Convert raw MemoryUnit entities to EntityCandidate dataclasses.

        Args:
            entities: Raw entity MemoryUnits.

        Returns:
            List of EntityCandidate objects for clustering.
        """
        candidates = []
        for i, entity in enumerate(entities):
            text = entity.raw_data.get("text_content", "") or entity.raw_data.get("text", "")
            if not text:
                continue
            entity_type = entity.metadata.get("entity_type", "unknown")
            context = entity.raw_data.get("description", "")
            sessions = list(entity.metadata.get("sessions", []))
            if not sessions:
                sessions = [entity.metadata.get("session_id", "unknown")]
            source_uids = [str(entity.uid)]

            candidates.append(EntityCandidate(
                index=i + 1,
                text=text,
                entity_type=entity_type,
                context=context[:200] if context else "",
                sessions=sessions,
                source_uids=source_uids,
            ))
        return candidates

    def _precluster_by_embedding(
        self,
        candidates: List[EntityCandidate],
    ) -> List[List[EntityCandidate]]:
        """Group candidates by cosine similarity of their embeddings.

        Computes pairwise cosine similarity matrix, then greedily assigns
        each candidate to the cluster with the highest similarity score,
        creating a new cluster when similarity falls below threshold.

        Args:
            candidates: EntityCandidates to cluster.

        Returns:
            List of clusters, each a list of EntityCandidates.
        """
        if self._embedder is None:
            return [candidates]

        texts = [c.text for c in candidates]
        try:
            embeddings = self._embedder.embed_text(texts)
        except (RuntimeError, ValueError, OSError) as e:
            logger.warning("Embedding failed: %s, skipping preclustering", e)
            return [candidates]

        n = len(candidates)
        emb_matrix = np.array(embeddings, dtype=np.float32)
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normalized = emb_matrix / norms

        similarity_matrix = normalized @ normalized.T

        clusters: List[List[int]] = [[0]]
        used = {0}
        for i in range(1, n):
            best_sim = 0.0
            best_cluster = 0
            for j, cluster in enumerate(clusters):
                for member in cluster:
                    sim = float(similarity_matrix[i, member])
                    if sim > best_sim:
                        best_sim = sim
                        best_cluster = j
            if best_sim >= self._threshold:
                clusters[best_cluster].append(i)
                used.add(i)
            else:
                clusters.append([i])
                used.add(i)

        for i in range(n):
            if i not in used:
                clusters.append([i])

        return [[candidates[idx] for idx in cluster] for cluster in clusters if cluster]

    def _llm_merge(
        self,
        cluster: List[EntityCandidate],
    ) -> List[Dict[str, Any]]:
        """Send a candidate cluster to the LLM for merge decision.

        Formats entity list and invokes ENTITY_DEDUP_SYSTEM_PROMPT,
        parsing the JSON response for canonical entities.

        Args:
            cluster: EntityCandidates in one similarity cluster.

        Returns:
            List of dicts with canonical_form, entity_type, confidence, etc.
        """
        if len(cluster) == 1:
            c = cluster[0]
            return [{
                "canonical_form": c.text,
                "entity_type": c.entity_type,
                "confidence": 0.9,
                "source_entities": [c.index],
                "aliases": [],
                "sessions": c.sessions,
                "source_uids": c.source_uids,
            }]

        entity_list = self._format_entity_list(cluster)
        prompt_content = ENTITY_DEDUP_SYSTEM_PROMPT.format(entity_list=entity_list)

        messages: List[ChatMessage] = [
            {"role": "user", "content": prompt_content},
        ]

        try:
            response = self._llm.chat(messages, temperature=0.1, max_tokens=2048)
            return self._parse_merge_response(response.content, cluster)
        except (json.JSONDecodeError, ConnectionError, TimeoutError, OSError, RuntimeError) as e:
            logger.error("Entity dedup LLM failed: %s", e)
            return [{
                "canonical_form": cluster[0].text,
                "entity_type": cluster[0].entity_type,
                "confidence": 0.5,
                "source_entities": [c.index for c in cluster],
                "aliases": [],
                "sessions": list(set(s for c in cluster for s in c.sessions)),
                "source_uids": [u for c in cluster for u in c.source_uids],
            }]

    def _format_entity_list(self, cluster: List[EntityCandidate]) -> str:
        """Format cluster candidates into a numbered text list for the LLM.

        Args:
            cluster: EntityCandidates to format.

        Returns:
            Newline-separated string like \"1. name (type) | ctx=... | sessions=...\".
        """
        lines = []
        for c in cluster:
            ctx = c.context[:100] if c.context else "no context"
            lines.append(f"{c.index}. {c.text} ({c.entity_type}) | ctx={ctx} | sessions={','.join(c.sessions)}")
        return "\n".join(lines)

    def _parse_merge_response(
        self,
        response: str,
        cluster: List[EntityCandidate],
    ) -> List[Dict[str, Any]]:
        """Parse the LLM's JSON merge response.

        Args:
            response: Raw LLM response string (JSON).
            cluster: Original EntityCandidates for index mapping.

        Returns:
            List of dicts with canonical entity info, or a fallback dict
            using the first candidate on parse failure.
        """
        try:
            data = json.loads(response)
            merged = data.get("merged_entities", [])
            results = []
            for me in merged:
                source_indices = me.get("source_entities", [])
                source_uids = []
                sessions = set()
                for idx in source_indices:
                    if 0 < idx <= len(cluster):
                        source_uids.extend(cluster[idx - 1].source_uids)
                        sessions.update(cluster[idx - 1].sessions)
                results.append({
                    "canonical_form": me.get("canonical_form", cluster[0].text),
                    "entity_type": me.get("entity_type", cluster[0].entity_type),
                    "confidence": float(me.get("confidence", 0.8)),
                    "source_entities": source_indices,
                    "aliases": list(me.get("aliases", [])),
                    "sessions": list(sessions),
                    "source_uids": source_uids,
                })
            return results
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse entity dedup response: {response[:200]}")
            return [{
                "canonical_form": cluster[0].text,
                "entity_type": cluster[0].entity_type,
                "confidence": 0.5,
                "source_entities": [c.index for c in cluster],
                "aliases": [],
                "sessions": list(set(s for c in cluster for s in c.sessions)),
                "source_uids": [u for c in cluster for u in c.source_uids],
            }]
