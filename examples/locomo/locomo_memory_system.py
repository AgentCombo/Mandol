"""LoCoMo memory system adapter for the Mandol framework.

Wraps :class:`MemorySystem` with LoCoMo-specific logic for loading
multi-session dialogue datasets, processing samples incrementally,
and building high-level memories (entities, events, summaries).
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from mandol.application.memory_system import MemorySystem, MemorySystemConfig
from mandol.application.semantic_graph import SemanticGraphService
from mandol.domain.memory_unit import MemoryUnit
from mandol.domain.types import SpaceName, Uid
from mandol.infrastructure.in_memory_graph_store import InMemoryGraphStore
from mandol.infrastructure.in_memory_unit_store import InMemoryUnitStore
from mandol.infrastructure.openai_compatible_embedding_provider import (
    OpenAICompatibleEmbeddingConfig,
    OpenAICompatibleEmbeddingProvider,
)
from mandol.infrastructure.openai_compatible_llm_provider import (
    OpenAICompatibleLLMConfig,
    OpenAICompatibleLLMProvider,
)
from mandol.infrastructure.openai_compatible_reranker import (
    OpenAICompatibleRerankConfig,
    OpenAICompatibleReranker,
)
from mandol.infrastructure.sentence_transformers_embedding_provider import (
    SentenceTransformersEmbeddingProvider,
)
from mandol.retrieval.types import SearchHit

from .config import LocomoMemoryConfig, MemorySystemSettings, RemoteProviderSettings

logger = logging.getLogger(__name__)


def _parse_session_number(key: str) -> Optional[int]:
    """Extract the numeric session index from a key like ``"session_3"``.

    Returns:
        The session number, or ``None`` if *key* does not match the pattern.
    """
    m = re.match(r"^session_(\d+)$", str(key))
    if m is None:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _parse_dialogue_index(dia_id: str, fallback: int) -> int:
    """Extract the dialogue index from a ``dia_id`` like ``"D3:5"``.

    Falls back to *fallback* when parsing fails.

    Returns:
        The zero-based dialogue index within its session.
    """
    try:
        if ":" in str(dia_id):
            return int(str(dia_id).split(":")[-1])
    except Exception:
        pass
    return int(fallback)


class LocomoMemorySystem:
    """High-level adapter that drives a :class:`MemorySystem` with LoCoMo data.

    Handles provider setup (remote or local), dataset loading, per-sample
    processing (session splitting, PRECEDES/FOLLOWS edges), incremental
    updates, and high-level memory construction.

    Args:
        config: LoCoMo-specific configuration.  Defaults to
            :data:`LocomoMemoryConfig` with environment-variable overrides.
    """

    def __init__(self, config: Optional[LocomoMemoryConfig] = None) -> None:
        self._config = config or LocomoMemoryConfig()
        self._memory_system: Optional[MemorySystem] = None
        self._processed_sample_ids: Set[str] = set()
        self._loaded_dataset_hash: Optional[str] = None
        self._dataset_samples: List[Dict[str, Any]] = []
        self._stats = {
            "samples_processed": 0,
            "dialogues_processed": 0,
            "sessions_processed": 0,
            "units_added": 0,
            "high_level_memories_built": 0,
            "incremental_updates": 0,
            "total_time_seconds": 0.0,
        }

    def _setup_providers(self) -> Tuple[Any, Any, Any]:
        """Instantiate embedder, reranker, and LLM providers from config.

        Returns:
            A ``(embedder, reranker, llm_provider)`` tuple.
        """
        cfg = self._config
        mem_settings = cfg.memory_settings

        if cfg.use_remote_embedder:
            remote = cfg.remote_provider or RemoteProviderSettings()
            if remote.embedding_require_token:
                token_env = remote.embedding_token_env
            else:
                token_env = ""
            emb_config = OpenAICompatibleEmbeddingConfig(
                base_url=remote.embedding_base_url,
                api_path=remote.embedding_api_path,
                token_env=token_env,
                timeout_s=remote.embedding_timeout_s,
            )
            embedder = OpenAICompatibleEmbeddingProvider(
                model=mem_settings.embedder_model,
                dim=mem_settings.embedder_dim,
                config=emb_config,
                token="dummy-token" if not remote.embedding_require_token else None,
            )
        else:
            embedder = SentenceTransformersEmbeddingProvider(
                model=mem_settings.embedder_model,
                device=mem_settings.embedder_device,
            )

        if cfg.use_remote_reranker:
            remote = cfg.remote_provider or RemoteProviderSettings()
            rerank_config = OpenAICompatibleRerankConfig(
                base_url=remote.reranker_base_url,
                api_path=remote.reranker_api_path,
                token_env=remote.reranker_token_env,
                timeout_s=remote.reranker_timeout_s,
            )
            reranker = OpenAICompatibleReranker(
                model=mem_settings.reranker_model,
                config=rerank_config,
            )
        else:
            from mandol.infrastructure.sentence_transformers_reranker import (
                SentenceTransformersCrossEncoderReranker,
            )
            reranker = SentenceTransformersCrossEncoderReranker(
                model=mem_settings.reranker_model,
                device=mem_settings.reranker_device,
            )

        if cfg.use_remote_llm:
            remote = cfg.remote_provider or RemoteProviderSettings()
            llm_config = OpenAICompatibleLLMConfig(
                base_url=remote.llm_base_url,
                api_key_env=remote.llm_api_key_env,
                timeout_s=remote.llm_timeout_s,
            )
            llm_provider = OpenAICompatibleLLMProvider(
                model=mem_settings.llm_model,
                config=llm_config,
            )
        else:
            from mandol.infrastructure.stub_llm_provider import StubLLMProvider
            llm_provider = StubLLMProvider()

        return embedder, reranker, llm_provider

    def initialize(self) -> None:
        """Create and configure the underlying :class:`MemorySystem`.

        Sets up providers, builds a :class:`MemorySystemConfig`, and
        instantiates the :class:`MemorySystem`.
        """
        cfg = self._config
        mem_settings = cfg.memory_settings

        embedder, reranker, llm_provider = self._setup_providers()

        memory_config = MemorySystemConfig(
            embedder_model=mem_settings.embedder_model,
            embedder_device=mem_settings.embedder_device,
            reranker_model=mem_settings.reranker_model,
            reranker_device=mem_settings.reranker_device,
            llm_model=mem_settings.llm_model,
            embedder_dim=mem_settings.embedder_dim,
            chunk_max_tokens=mem_settings.chunk_max_tokens,
            session_time_gap_seconds=mem_settings.session_time_gap_seconds,
            similarity_top_k=mem_settings.similarity_top_k,
            similarity_threshold=mem_settings.similarity_threshold,
            similarity_recent_window=mem_settings.similarity_recent_window,
            bfs_expansion_per_seed=mem_settings.bfs_expansion_per_seed,
            bfs_expansion_hops=mem_settings.bfs_expansion_hops,
        )

        self._memory_system = MemorySystem(
            config=memory_config,
            embedder=embedder,
            reranker=reranker,
            llm_provider=llm_provider,
        )
        logger.info("LocomoMemorySystem initialized successfully")

    def _compute_dataset_hash(self, data: List[Dict[str, Any]]) -> str:
        """Return an MD5 hash of the sorted sample IDs for change detection."""
        import hashlib
        sample_ids = [item.get("sample_id", "") for item in data]
        hash_str = json.dumps(sample_ids, sort_keys=True)
        return hashlib.md5(hash_str.encode()).hexdigest()

    def _load_dataset(self) -> List[Dict[str, Any]]:
        """Load the LoCoMo JSON dataset from the configured path.

        Returns:
            A list of sample dictionaries.

        Raises:
            FileNotFoundError: If the dataset file does not exist.
            ValueError: If the dataset root is not a list.
        """
        dataset_path = Path(self._config.dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("Dataset root must be a list")

        return data

    def load_and_process_samples(
        self,
        sample_ids: Optional[List[str]] = None,
        force_reload: bool = False,
    ) -> Dict[str, Any]:
        """Load the dataset and process each sample into memory units.

        Skips already-processed samples unless *force_reload* is ``True``.

        Args:
            sample_ids: Optional whitelist of sample IDs to process.
            force_reload: Re-process samples even if their hash is unchanged.

        Returns:
            A dict with ``status``, ``samples_processed``, ``total_samples``,
            and ``elapsed_time``.
        """
        if self._memory_system is None:
            self.initialize()

        start_time = time.time()
        data = self._load_dataset()

        self._dataset_samples = data
        current_hash = self._compute_dataset_hash(data)

        if not force_reload and self._loaded_dataset_hash == current_hash:
            logger.info("Dataset unchanged, skipping reload")
            return {"status": "unchanged", "samples_skipped": len(data)}

        samples_to_process = data
        if sample_ids is not None:
            samples_to_process = [s for s in data if s.get("sample_id") in set(sample_ids)]

        newly_processed = 0
        for sample in samples_to_process:
            sample_id = sample.get("sample_id", "")
            if sample_id in self._processed_sample_ids and not force_reload:
                continue

            self._process_sample(sample)
            self._processed_sample_ids.add(sample_id)
            newly_processed += 1

            if newly_processed % self._config.progress_report_interval == 0:
                logger.info(f"Processed {newly_processed}/{len(samples_to_process)} samples")

        self._loaded_dataset_hash = current_hash
        self._stats["samples_processed"] = len(self._processed_sample_ids)
        self._stats["total_time_seconds"] = time.time() - start_time

        return {
            "status": "completed",
            "samples_processed": newly_processed,
            "total_samples": len(samples_to_process),
            "elapsed_time": time.time() - start_time,
        }

    def _process_sample(self, sample: Dict[str, Any]) -> None:
        """Convert a single LoCoMo sample into memory units and graph edges.

        Each session becomes a separate space; consecutive dialogues within
        a session are linked with PRECEDES / FOLLOWS edges.

        Args:
            sample: A dict with ``sample_id`` and ``conversation`` keys.

        Raises:
            RuntimeError: If the memory system has not been initialized.
            ValueError: If required fields are missing.
        """
        if self._memory_system is None:
            raise RuntimeError("MemorySystem not initialized")

        sample_id = str(sample.get("sample_id", "")).strip()
        if not sample_id:
            raise ValueError("sample.sample_id is required")

        base_root = sample_id

        conv = sample.get("conversation") or {}
        if not isinstance(conv, dict):
            raise ValueError("sample.conversation must be a dict")

        sessions: List[Tuple[int, str, List[Dict[str, Any]]]] = []
        for k, v in conv.items():
            n = _parse_session_number(k)
            if n is None:
                continue
            if not isinstance(v, list):
                continue
            dt = conv.get(f"session_{n}_date_time", "")
            sessions.append((n, str(dt or ""), [x for x in v if isinstance(x, dict)]))
        sessions.sort(key=lambda x: x[0])

        dialogue_count = 0
        for sess_n, sess_dt, dialogues in sessions:
            session_space = f"{base_root}_session_{sess_n}"

            ordered: List[Tuple[int, str]] = []
            for i, d in enumerate(dialogues):
                dia_id = str(d.get("dia_id") or "").strip()
                speaker = str(d.get("speaker") or "").strip()
                text = d.get("text")
                if text is None or str(text).strip() == "":
                    text = d.get("text_content") or d.get("content") or ""
                text = str(text)
                if not text.strip():
                    continue

                didx = _parse_dialogue_index(dia_id, i)
                if not dia_id:
                    dia_id = f"D{sess_n}:{didx}"

                unit_uid = f"{base_root}_dialogue_{dia_id}"

                existing_unit = self._memory_system.semantic_map.get_unit(unit_uid)
                if existing_unit is not None:
                    ordered.append((didx, unit_uid))
                    continue

                blip_caption = d.get("blip_caption")
                img_url = d.get("img_url")

                unit = MemoryUnit(
                    uid=Uid(unit_uid),
                    raw_data={
                        "type": "dialogue",
                        "dia_id": dia_id,
                        "speaker": speaker,
                        "text": text,
                        "dialogue_index": didx,
                        "session_datetime": sess_dt,
                        "session_number": sess_n,
                        "img_url": img_url,
                        "blip_caption": blip_caption,
                        "query": d.get("query"),
                        "text_content": f"Dialogue {dia_id} [Time {sess_dt}]: {speaker}{f' sent {blip_caption} and' if blip_caption else ''} said: {text}",
                    },
                    metadata={
                        "unit_type": "dialogue",
                        "session_number": sess_n,
                        "sample_id": sample_id,
                    },
                    embedding=None,
                )

                self._memory_system.add(unit)
                ordered.append((didx, unit_uid))
                dialogue_count += 1
                self._stats["units_added"] += 1

            ordered.sort(key=lambda x: x[0])
            for (_, a), (_, b) in zip(ordered, ordered[1:]):
                self._memory_system.graph.add_relationship(
                    source_uid=a,
                    target_uid=b,
                    relationship_name="PRECEDES",
                    score=1.0,
                )
                self._memory_system.graph.add_relationship(
                    source_uid=b,
                    target_uid=a,
                    relationship_name="FOLLOWS",
                    score=1.0,
                )

            self._stats["sessions_processed"] += 1

        self._stats["dialogues_processed"] += dialogue_count

    def incremental_update(self, new_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process additional or updated samples without reloading the full dataset.

        Args:
            new_samples: List of sample dicts to add or update.

        Returns:
            A dict with ``status``, ``new_samples_processed``,
            ``updated_samples``, and ``elapsed_time``.
        """
        if self._memory_system is None:
            self.initialize()

        start_time = time.time()
        newly_processed = 0
        updated_samples = 0

        for sample in new_samples:
            sample_id = sample.get("sample_id", "")
            if not sample_id:
                continue

            existing_sample_ids = {s.get("sample_id") for s in self._dataset_samples}
            if sample_id in existing_sample_ids:
                updated_samples += 1
                self._stats["incremental_updates"] += 1

            self._process_sample(sample)
            self._processed_sample_ids.add(sample_id)
            newly_processed += 1

            if sample_id not in existing_sample_ids:
                self._dataset_samples.append(sample)

        self._loaded_dataset_hash = self._compute_dataset_hash(self._dataset_samples)
        self._stats["total_time_seconds"] += time.time() - start_time

        return {
            "status": "completed",
            "new_samples_processed": newly_processed,
            "updated_samples": updated_samples,
            "elapsed_time": time.time() - start_time,
        }

    def build_high_level_memories(self, mode: str = "auto", sample_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Trigger high-level memory construction on the underlying system.

        Args:
            mode: Build mode — ``"auto"`` or ``"force"``.
            sample_ids: Reserved for future per-sample filtering.

        Returns:
            The build result dict augmented with ``elapsed_time``.

        Raises:
            RuntimeError: If the memory system has not been initialized.
        """
        if self._memory_system is None:
            raise RuntimeError("MemorySystem not initialized")

        start_time = time.time()

        if sample_ids is not None:
            logger.info(f"Building high-level memories for samples: {sample_ids}")
        else:
            logger.info("Building high-level memories for all processed samples")

        result = self._memory_system.build_high_level(mode=mode)

        self._stats["high_level_memories_built"] += 1
        self._stats["total_time_seconds"] += time.time() - start_time

        result["elapsed_time"] = time.time() - start_time
        return result

    def search(self, query: str, *, top_k: int = 10, use_rerank: bool = True, sample_ids: Optional[List[str]] = None) -> List[SearchHit]:
        """Search the memory system and optionally filter by sample IDs.

        Args:
            query: Natural-language query string.
            top_k: Maximum number of results to return.
            use_rerank: Whether to apply the reranker.
            sample_ids: If provided, only return hits whose unit belongs to
                one of these samples.

        Returns:
            A list of :class:`SearchHit` objects.

        Raises:
            RuntimeError: If the memory system has not been initialized.
        """
        if self._memory_system is None:
            raise RuntimeError("MemorySystem not initialized")

        hits = self._memory_system.search(query, top_k=top_k, use_rerank=use_rerank)

        if sample_ids is not None:
            sample_id_set = set(sample_ids)
            hits = [h for h in hits if h.unit.metadata.get("sample_id") in sample_id_set]

        return hits

    def get_memory_stats(self) -> Dict[str, Any]:
        """Return aggregate statistics about spaces, units, and processing.

        Returns:
            A dict with ``status``, ``total_spaces``, ``total_units``,
            ``processed_sample_ids``, ``space_details``, and
            ``processing_stats``.
        """
        if self._memory_system is None:
            return {"status": "not_initialized"}

        spaces = self._memory_system.semantic_map.list_spaces()
        units = self._memory_system.semantic_map.list_units()

        space_info = {}
        for sp in spaces:
            space_info[str(sp.name)] = {
                "unit_count": len(sp.unit_uids),
                "child_spaces": [str(c) for c in sp.child_spaces],
            }

        return {
            "status": "ok",
            "total_spaces": len(spaces),
            "total_units": len(units),
            "processed_sample_ids": sorted(self._processed_sample_ids),
            "space_details": space_info,
            "processing_stats": dict(self._stats),
        }

    def flush(self) -> None:
        """Persist any buffered data in the underlying memory system."""
        if self._memory_system is not None:
            self._memory_system.flush()
