#!/usr/bin/env python3
"""Build one per-sample SemanticGraph from LoCoMo tri-tower sample graphs.

The merge path intentionally bypasses SemanticGraph.add_unit()/SemanticMap.add_unit()
so existing dense and sparse vectors are reused instead of regenerated.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import logging
import sys
import time
import types
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import rustworkx as rx


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mandol.core.memory_unit import MemoryUnit
from mandol.core.semantic_graph import SemanticGraph
from mandol.core.semantic_map import SemanticMap
from mandol.retrieval.advance_retriever import MultiRetriever
from mandol.retrieval.bm25_retriever import BM25Retriever
from mandol.retrieval.retrieval_interface import RetrievalMethod
from mandol.retrieval.splade_retriever import SPLADERetriever


LOGGER = logging.getLogger("build_unified_graph")


@dataclass(frozen=True)
class BuilderConfig:
    episodic_dir: Path
    entity_dir: Path
    hierarchical_dir: Path
    output_dir: Path
    sample_ids: Optional[List[str]]
    embedding_model: str
    splade_model: str
    image_embedding_model: str = "clip-ViT-B-32"
    embedding_dim: Optional[int] = None
    faiss_index_type: str = "IDMap,Flat"
    freeze_retrievers: bool = True


@dataclass(frozen=True)
class TowerSpec:
    name: str
    root: Path


@dataclass
class MergeStats:
    sample_id: str
    tower: str
    source_path: str = ""
    loaded: bool = False
    skipped: bool = False
    units: int = 0
    spaces: int = 0
    edges: int = 0
    uid_collisions: int = 0
    missing_dense_embeddings: int = 0
    invalid_dense_embeddings: int = 0
    missing_sparse_embeddings: int = 0
    duration_seconds: float = 0.0
    warning: Optional[str] = None


@dataclass
class SampleBuildStats:
    sample_id: str
    output_dir: str = ""
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    finished_at: Optional[str] = None
    towers_attempted: int = 0
    towers_loaded: int = 0
    towers_skipped: int = 0
    units: int = 0
    spaces: int = 0
    edges: int = 0
    uid_collisions: int = 0
    missing_dense_embeddings: int = 0
    invalid_dense_embeddings: int = 0
    missing_sparse_embeddings: int = 0
    rebuild_seconds: float = 0.0
    save_seconds: float = 0.0
    total_seconds: float = 0.0
    retrieval_state: Dict[str, Any] = field(default_factory=dict)
    tower_stats: List[MergeStats] = field(default_factory=list)


@dataclass
class BuildStats:
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    finished_at: Optional[str] = None
    samples_requested: int = 0
    samples_with_any_tower: int = 0
    towers_attempted: int = 0
    towers_loaded: int = 0
    towers_skipped: int = 0
    total_units: int = 0
    total_spaces: int = 0
    total_edges: int = 0
    uid_collisions: int = 0
    missing_dense_embeddings: int = 0
    invalid_dense_embeddings: int = 0
    missing_sparse_embeddings: int = 0
    rebuild_seconds: float = 0.0
    save_seconds: float = 0.0
    total_seconds: float = 0.0
    sample_stats: List[SampleBuildStats] = field(default_factory=list)
    tower_stats: List[MergeStats] = field(default_factory=list)


class UnifiedGraphBuilder:
    """Merge each sample's tri-tower SemanticGraph outputs into its own graph."""

    GRAPH_MARKERS = (
        "graph_state.json",
        "rx_graph.pkl",
        "semantic_graph.json",
        "semantic_map_data/semantic_map_meta.json",
    )

    def __init__(self, config: BuilderConfig):
        self.config = config
        self.logger = LOGGER
        self.unified_graph: Optional[SemanticGraph] = None
        self._uid_sources: Dict[str, Tuple[str, str, str]] = {}

    def build(self) -> BuildStats:
        start_time = time.time()
        sample_ids = self.config.sample_ids or self._discover_sample_ids()
        stats = BuildStats(samples_requested=len(sample_ids))

        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("Starting per-sample unified graph build")
        self.logger.info("Samples: %s", ", ".join(sample_ids) if sample_ids else "<none>")
        self.logger.info("Output root directory: %s", self.config.output_dir)
        self.logger.info("BM25/SPLADE static index freeze: %s", "enabled" if self.config.freeze_retrievers else "disabled")

        if not sample_ids:
            self.logger.warning("No sample ids to process. No per-sample graph will be saved.")

        for sample_id in sample_ids:
            sample_stats = self._build_sample_graph(sample_id)
            stats.sample_stats.append(sample_stats)
            stats.tower_stats.extend(sample_stats.tower_stats)
            stats.towers_attempted += sample_stats.towers_attempted
            stats.towers_loaded += sample_stats.towers_loaded
            stats.towers_skipped += sample_stats.towers_skipped
            stats.total_units += sample_stats.units
            stats.total_spaces += sample_stats.spaces
            stats.total_edges += sample_stats.edges
            stats.uid_collisions += sample_stats.uid_collisions
            stats.missing_dense_embeddings += sample_stats.missing_dense_embeddings
            stats.invalid_dense_embeddings += sample_stats.invalid_dense_embeddings
            stats.missing_sparse_embeddings += sample_stats.missing_sparse_embeddings
            stats.rebuild_seconds += sample_stats.rebuild_seconds
            stats.save_seconds += sample_stats.save_seconds

            if sample_stats.towers_loaded:
                stats.samples_with_any_tower += 1

        stats.finished_at = datetime.now().isoformat()
        stats.total_seconds = time.time() - start_time
        self._write_report(stats)
        self._log_summary(stats)
        return stats

    def _build_sample_graph(self, sample_id: str) -> SampleBuildStats:
        sample_start = time.time()
        sample_output_dir = self.config.output_dir / sample_id
        sample_stats = SampleBuildStats(sample_id=sample_id, output_dir=str(sample_output_dir))
        self.unified_graph = self._create_unified_graph()
        self._uid_sources = {}

        self.logger.info("Processing sample: %s", sample_id)
        try:
            for tower in self._tower_specs():
                sample_stats.towers_attempted += 1
                merge_stats = self._process_tower(sample_id, tower)
                sample_stats.tower_stats.append(merge_stats)

                if merge_stats.loaded:
                    sample_stats.towers_loaded += 1
                    sample_stats.units += merge_stats.units
                    sample_stats.spaces += merge_stats.spaces
                    sample_stats.edges += merge_stats.edges
                    sample_stats.uid_collisions += merge_stats.uid_collisions
                    sample_stats.missing_dense_embeddings += merge_stats.missing_dense_embeddings
                    sample_stats.invalid_dense_embeddings += merge_stats.invalid_dense_embeddings
                    sample_stats.missing_sparse_embeddings += merge_stats.missing_sparse_embeddings
                elif merge_stats.skipped:
                    sample_stats.towers_skipped += 1

            if not sample_stats.towers_loaded:
                self.logger.warning("Sample %s had no loadable tower graph; output is skipped", sample_id)
                return sample_stats

            rebuild_start = time.time()
            self._rebuild_indexes_once()
            sample_stats.rebuild_seconds = time.time() - rebuild_start

            save_start = time.time()
            sample_stats.retrieval_state = self._save_unified_graph(sample_output_dir)
            sample_stats.save_seconds = time.time() - save_start
            sample_stats.finished_at = datetime.now().isoformat()
            sample_stats.total_seconds = time.time() - sample_start
            self._write_sample_metadata(sample_stats, sample_output_dir)
            return sample_stats
        finally:
            if sample_stats.finished_at is None:
                sample_stats.finished_at = datetime.now().isoformat()
                sample_stats.total_seconds = time.time() - sample_start
            if self.unified_graph is not None:
                self._cleanup_source_graph(self.unified_graph)
                self.unified_graph = None

    def _tower_specs(self) -> List[TowerSpec]:
        return [
            TowerSpec("episodic", self.config.episodic_dir),
            TowerSpec("entity", self.config.entity_dir),
            TowerSpec("hierarchical", self.config.hierarchical_dir),
        ]

    def _create_unified_graph(self) -> SemanticGraph:
        self.logger.info("Initializing empty per-sample SemanticGraph")
        semantic_map = SemanticMap(
            embedding_model_name=self.config.embedding_model,
            embedding_dim=self.config.embedding_dim,
            faiss_index_type=self.config.faiss_index_type,
        )
        return SemanticGraph(semantic_map_instance=semantic_map)

    def _discover_sample_ids(self) -> List[str]:
        discovered: Dict[str, None] = {}
        for tower in self._tower_specs():
            if not tower.root.exists():
                self.logger.warning("Cannot discover samples from missing directory: %s", tower.root)
                continue
            for child in sorted(tower.root.iterdir()):
                if not child.is_dir() or not self._looks_like_graph_dir(child):
                    continue
                sample_id = child.name
                if sample_id.endswith("_enhanced"):
                    sample_id = sample_id[: -len("_enhanced")]
                discovered.setdefault(sample_id, None)

        sample_ids = list(discovered.keys())
        self.logger.info("Discovered %d sample ids from tower directories", len(sample_ids))
        return sample_ids

    def _process_tower(self, sample_id: str, tower: TowerSpec) -> MergeStats:
        start_time = time.time()
        graph_dir = self._resolve_sample_graph_dir(tower.root, sample_id)
        merge_stats = MergeStats(sample_id=sample_id, tower=tower.name)

        if graph_dir is None:
            merge_stats.skipped = True
            merge_stats.warning = f"Missing {tower.name} graph for sample {sample_id}"
            self.logger.warning("%s", merge_stats.warning)
            return merge_stats

        merge_stats.source_path = str(graph_dir)
        source_graph: Optional[SemanticGraph] = None
        try:
            self.logger.info("Loading %s graph: %s", tower.name, graph_dir)
            source_graph = self._load_source_graph(graph_dir)
            merge_stats = self._merge_source_graph(source_graph, sample_id, tower.name, graph_dir)
            merge_stats.loaded = True
            merge_stats.duration_seconds = time.time() - start_time
            self.logger.info(
                "Merged %s/%s: units=%d spaces=%d edges=%d collisions=%d",
                sample_id,
                tower.name,
                merge_stats.units,
                merge_stats.spaces,
                merge_stats.edges,
                merge_stats.uid_collisions,
            )
            return merge_stats
        except Exception as exc:
            merge_stats.skipped = True
            merge_stats.warning = f"Failed to merge {tower.name} graph for {sample_id}: {exc}"
            merge_stats.duration_seconds = time.time() - start_time
            self.logger.exception("%s", merge_stats.warning)
            return merge_stats
        finally:
            if source_graph is not None:
                self._cleanup_source_graph(source_graph)

    def _resolve_sample_graph_dir(self, root: Path, sample_id: str) -> Optional[Path]:
        candidates = [
            root / sample_id,
            root / f"{sample_id}_enhanced",
            root / f"{sample_id}_semantic_graph",
            root / f"{sample_id}_graph",
        ]
        for candidate in candidates:
            if self._looks_like_graph_dir(candidate):
                return candidate
        return None

    def _looks_like_graph_dir(self, path: Path) -> bool:
        if not path.is_dir():
            return False
        return any((path / marker).exists() for marker in self.GRAPH_MARKERS)

    def _load_source_graph(self, graph_dir: Path) -> SemanticGraph:
        return SemanticGraph.load_graph(
            str(graph_dir),
            embedding_model_name=self.config.embedding_model,
        )

    def _merge_source_graph(
        self,
        source_graph: SemanticGraph,
        sample_id: str,
        tower_name: str,
        graph_dir: Path,
    ) -> MergeStats:
        source_map = source_graph.semantic_map
        target_map = self.unified_graph.semantic_map
        stats = MergeStats(sample_id=sample_id, tower=tower_name, source_path=str(graph_dir))
        uid_mapping: Dict[str, str] = {}
        unit_spaces = self._collect_unit_spaces(source_graph)

        for space_name, source_space in source_map.memory_spaces.items():
            created = space_name not in target_map.memory_spaces
            target_space = target_map.create_memory_space(space_name)
            if getattr(target_space, "_faiss_index_type", None) is None:
                target_space._faiss_index_type = getattr(source_space, "_faiss_index_type", None)
            if created:
                stats.spaces += 1

        for space_name, source_space in source_map.memory_spaces.items():
            target_space = target_map.create_memory_space(space_name)
            target_space._child_space_names.update(source_space.get_child_space_names())

        for old_uid, unit in source_map.memory_units.items():
            new_uid, collided = self._allocate_uid(old_uid, sample_id, tower_name)
            uid_mapping[old_uid] = new_uid
            if collided:
                stats.uid_collisions += 1

            new_unit = self._clone_unit(unit, new_uid, sample_id, tower_name, graph_dir, collided)
            target_map.memory_units[new_uid] = new_unit
            target_map._modified_units.add(new_uid)
            target_map._access_counts.setdefault(new_uid, 0)
            self._uid_sources[new_uid] = (sample_id, tower_name, old_uid)

            spaces_for_unit = unit_spaces.get(old_uid, [])
            for space_name in spaces_for_unit:
                target_map.add_unit_to_space(new_uid, space_name)

            node_attrs = self._source_node_attrs(source_graph, old_uid)
            self._ensure_graph_node(new_uid, new_unit, spaces_for_unit, node_attrs, sample_id, tower_name, old_uid)

            stats.units += 1
            if new_unit.embedding is None:
                stats.missing_dense_embeddings += 1
            elif not self._is_valid_dense_embedding(new_unit.embedding):
                stats.invalid_dense_embeddings += 1
            if new_unit.sparse_embedding is None:
                stats.missing_sparse_embeddings += 1

        for space_name, source_space in source_map.memory_spaces.items():
            target_space = target_map.create_memory_space(space_name)
            for old_uid in source_space.get_unit_uids():
                new_uid = uid_mapping.get(old_uid)
                if new_uid is not None:
                    target_space.add_unit(new_uid)

        graph_node_mapping = self._copy_non_unit_graph_nodes(source_graph, uid_mapping)
        stats.edges = self._copy_edges(source_graph, graph_node_mapping)
        return stats

    def _collect_unit_spaces(self, graph: SemanticGraph) -> Dict[str, List[str]]:
        unit_spaces: Dict[str, List[str]] = {}
        for space_name, space in graph.semantic_map.memory_spaces.items():
            for uid in space.get_unit_uids():
                unit_spaces.setdefault(uid, []).append(space_name)
        return unit_spaces

    def _allocate_uid(self, original_uid: str, sample_id: str, tower_name: str) -> Tuple[str, bool]:
        target_units = self.unified_graph.semantic_map.memory_units
        if original_uid not in target_units and original_uid not in self._uid_sources:
            return original_uid, False

        base = f"{tower_name}::{sample_id}::{original_uid}"
        candidate = base
        suffix = 2
        while candidate in target_units or candidate in self._uid_sources:
            candidate = f"{base}::{suffix}"
            suffix += 1
        return candidate, True

    def _clone_unit(
        self,
        unit: MemoryUnit,
        new_uid: str,
        sample_id: str,
        tower_name: str,
        graph_dir: Path,
        uid_collided: bool,
    ) -> MemoryUnit:
        metadata = copy.deepcopy(unit.metadata) if unit.metadata else {}
        metadata["_unified_source"] = {
            "sample_id": sample_id,
            "tower": tower_name,
            "source_uid": unit.uid,
            "source_path": str(graph_dir),
            "uid_collided": uid_collided,
        }

        new_unit = MemoryUnit(
            uid=new_uid,
            raw_data=copy.deepcopy(unit.raw_data) if unit.raw_data else {},
            metadata=metadata,
            embedding=self._clone_dense_embedding(unit.embedding),
            sparse_embedding=self._clone_sparse_embedding(unit),
        )

        for attr_name, attr_value in self._iter_extra_unit_attrs(unit):
            if attr_name in {"uid", "raw_data", "metadata", "embedding", "sparse_embedding"}:
                continue
            setattr(new_unit, attr_name, copy.deepcopy(attr_value))
        return new_unit

    def _iter_extra_unit_attrs(self, unit: MemoryUnit):
        unit_dict = getattr(unit, "__dict__", None)
        if isinstance(unit_dict, dict):
            yield from unit_dict.items()

        slots = getattr(type(unit), "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)

        for attr_name in slots:
            if attr_name in {"__dict__", "__weakref__", "uid", "_raw_data", "raw_data", "metadata", "embedding", "sparse_embedding", "text_cached"}:
                continue
            if hasattr(unit, attr_name):
                yield attr_name, getattr(unit, attr_name)

    def _clone_dense_embedding(self, embedding: Any) -> Any:
        if isinstance(embedding, np.ndarray):
            return np.array(embedding, copy=True)
        return copy.deepcopy(embedding)

    def _clone_sparse_embedding(self, unit: MemoryUnit) -> Any:
        sparse_embedding = unit.sparse_embedding
        if sparse_embedding is None and isinstance(unit.raw_data, dict):
            sparse_embedding = unit.raw_data.get("splade")

        if sparse_embedding is None:
            return None
        if isinstance(sparse_embedding, dict):
            try:
                return {int(token_id): float(weight) for token_id, weight in sparse_embedding.items()}
            except (TypeError, ValueError):
                return copy.deepcopy(sparse_embedding)
        if isinstance(sparse_embedding, np.ndarray):
            return np.array(sparse_embedding, copy=True)
        return copy.deepcopy(sparse_embedding)

    def _is_valid_dense_embedding(self, embedding: Any) -> bool:
        return (
            isinstance(embedding, np.ndarray)
            and embedding.ndim == 1
            and embedding.shape[0] == self.unified_graph.semantic_map.embedding_dim
        )

    def _source_node_attrs(self, graph: SemanticGraph, uid: str) -> Optional[Dict[str, Any]]:
        node_idx = graph._uid_to_index.get(uid)
        if node_idx is None:
            return None
        try:
            payload = graph.rx_graph[node_idx]
        except Exception:
            return None
        return copy.deepcopy(payload) if isinstance(payload, dict) else None

    def _ensure_graph_node(
        self,
        uid: str,
        unit: Optional[MemoryUnit],
        spaces: Optional[Sequence[str]],
        attrs: Optional[Dict[str, Any]],
        sample_id: str,
        tower_name: str,
        source_uid: str,
    ) -> int:
        if uid in self.unified_graph._uid_to_index:
            return self.unified_graph._uid_to_index[uid]

        node_attrs = attrs.copy() if attrs else {}
        node_attrs["uid"] = uid
        if unit is not None:
            node_attrs.setdefault("type", "memory_unit")
            node_attrs.setdefault("spaces", list(spaces or []))
            if isinstance(unit.raw_data, dict):
                for key, value in unit.raw_data.items():
                    if key not in node_attrs and isinstance(value, (str, int, float, bool)):
                        node_attrs[key] = value
        node_attrs.setdefault("created", str(datetime.now()))
        node_attrs["source_uid"] = source_uid
        node_attrs["source_sample_id"] = sample_id
        node_attrs["source_tower"] = tower_name

        idx = self.unified_graph.rx_graph.add_node(node_attrs)
        self.unified_graph._uid_to_index[uid] = idx
        self.unified_graph._index_to_uid[idx] = uid
        self.unified_graph._nodes_access_counts.setdefault(uid, 0)
        self.unified_graph._nodes_last_accessed.setdefault(uid, datetime.now().timestamp())
        self.unified_graph._modified_units.add(uid)
        return idx

    def _copy_non_unit_graph_nodes(
        self,
        source_graph: SemanticGraph,
        uid_mapping: Dict[str, str],
    ) -> Dict[str, str]:
        graph_node_mapping: Dict[str, str] = {}
        source_map = source_graph.semantic_map

        for source_idx, source_node_id in source_graph._index_to_uid.items():
            mapped_id = uid_mapping.get(source_node_id)
            if mapped_id is None:
                mapped_id = self._map_non_unit_node_id(source_node_id, source_map.memory_spaces)
            if mapped_id is None:
                continue

            graph_node_mapping[source_node_id] = mapped_id
            if mapped_id in self.unified_graph._uid_to_index:
                continue

            try:
                attrs = source_graph.rx_graph[source_idx]
            except Exception:
                attrs = {}
            node_attrs = copy.deepcopy(attrs) if isinstance(attrs, dict) else {}

            if mapped_id.startswith("ms:"):
                space_name = mapped_id[3:]
                self.unified_graph.semantic_map.create_memory_space(space_name)
                node_attrs.setdefault("type", "memory_space")
                node_attrs.setdefault("name", space_name)
            else:
                node_attrs.setdefault("uid", mapped_id)

            node_idx = self.unified_graph.rx_graph.add_node(node_attrs)
            self.unified_graph._uid_to_index[mapped_id] = node_idx
            self.unified_graph._index_to_uid[node_idx] = mapped_id

        return graph_node_mapping

    def _map_non_unit_node_id(self, node_id: str, source_spaces: Dict[str, Any]) -> Optional[str]:
        if node_id.startswith("ms:"):
            space_name = node_id[3:]
            if space_name in source_spaces:
                return node_id
        if node_id in source_spaces:
            return f"ms:{node_id}"
        return None

    def _copy_edges(self, source_graph: SemanticGraph, graph_node_mapping: Dict[str, str]) -> int:
        copied = 0
        for edge_idx in source_graph.rx_graph.edge_indices():
            try:
                src_idx, tgt_idx = source_graph.rx_graph.get_edge_endpoints_by_index(edge_idx)
                source_id = source_graph._index_to_uid.get(src_idx)
                target_id = source_graph._index_to_uid.get(tgt_idx)
                if source_id is None or target_id is None:
                    continue

                mapped_source = graph_node_mapping.get(source_id)
                mapped_target = graph_node_mapping.get(target_id)
                if mapped_source is None or mapped_target is None:
                    continue
                if mapped_source not in self.unified_graph._uid_to_index:
                    continue
                if mapped_target not in self.unified_graph._uid_to_index:
                    continue

                source_target_idx = self.unified_graph._uid_to_index[mapped_source]
                target_target_idx = self.unified_graph._uid_to_index[mapped_target]
                edge_data = source_graph.rx_graph.get_edge_data_by_index(edge_idx) or {}
                edge_payload = copy.deepcopy(edge_data) if isinstance(edge_data, dict) else {"payload": str(edge_data)}
                self.unified_graph.rx_graph.add_edge(source_target_idx, target_target_idx, edge_payload)

                rel_type = edge_payload.get("type", "RELATED_TO")
                rel_key = (mapped_source, mapped_target, rel_type)
                self.unified_graph._modified_relationships.add(rel_key)
                self.unified_graph._relationship_cache[rel_key] = {
                    key: self._json_safe(value)
                    for key, value in edge_payload.items()
                    if key not in {"type", "created", "updated"}
                }
                copied += 1
            except Exception as exc:
                self.logger.warning("Failed to copy edge %s: %s", edge_idx, exc)
        return copied

    def _json_safe(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(key): self._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._json_safe(item) for item in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        return str(value)

    def _rebuild_indexes_once(self) -> None:
        self.logger.info("Rebuilding per-sample FAISS index")
        self.unified_graph.semantic_map.build_index()

        self.logger.info("Rebuilding per-sample BM25/SPLADE indexes")
        multi_retriever = self._configure_retrievers()
        build_result = multi_retriever.build_all_indexes(
            methods_to_build=[RetrievalMethod.BM25, RetrievalMethod.SPLADE],
            force_rebuild=True,
        )
        self.logger.info("Retriever index rebuild result: %s", build_result)

    def _configure_retrievers(self) -> MultiRetriever:
        multi_retriever = self.unified_graph.get_multi_retriever()
        multi_retriever.add_retriever(BM25Retriever(self.unified_graph.semantic_map))
        multi_retriever.add_retriever(
            SPLADERetriever(
                self.unified_graph.semantic_map,
                model_name=self.config.splade_model,
            )
        )
        return multi_retriever

    def _save_unified_graph(self, output_dir: Path) -> Dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        with self._disable_sparse_autobuild_during_save():
            self.unified_graph.save_graph(str(output_dir), freeze_retrievers=self.config.freeze_retrievers)
        return self._load_saved_retrieval_state(output_dir)

    def _load_saved_retrieval_state(self, output_dir: Path) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "enabled": self.config.freeze_retrievers,
            "frozen_matrices_saved": {},
            "indices_saved": {},
        }
        graph_state_path = output_dir / "graph_state.json"
        try:
            graph_state = json.loads(graph_state_path.read_text(encoding="utf-8"))
            retrieval = graph_state.get("retrieval", {})
            state["frozen_matrices_saved"] = retrieval.get("frozen_matrices_saved", {})
            state["indices_saved"] = retrieval.get("indices_saved", {})
        except Exception as exc:
            state["error"] = str(exc)
            self.logger.warning("Failed to read saved retrieval state from %s: %s", graph_state_path, exc)
        return state

    def _write_sample_metadata(self, stats: SampleBuildStats, output_dir: Path) -> None:
        metadata_path = output_dir / "unified_graph_manifest.json"
        manifest = {
            "created_at": datetime.now().isoformat(),
            "sample_id": stats.sample_id,
            "embedding_model": self.config.embedding_model,
            "splade_model": self.config.splade_model,
            "image_embedding_model": self.config.image_embedding_model,
            "faiss_index_type": self.config.faiss_index_type,
            "freeze_retrievers": self.config.freeze_retrievers,
            "retrieval_state": stats.retrieval_state,
            "graph_statistics": {
                "memory_units": len(self.unified_graph.semantic_map.memory_units),
                "memory_spaces": len(self.unified_graph.semantic_map.memory_spaces),
                "rustworkx_nodes": self.unified_graph.rx_graph.num_nodes(),
                "rustworkx_edges": self.unified_graph.rx_graph.num_edges(),
            },
            "sample_build_stats": asdict(stats),
        }
        metadata_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        self.logger.info("Manifest written to %s", metadata_path)

        sample_report_path = output_dir / "unified_sample_build_report.json"
        sample_report_path.write_text(json.dumps(asdict(stats), ensure_ascii=False, indent=2), encoding="utf-8")
        self.logger.info("Sample build report written to %s", sample_report_path)

    @contextmanager
    def _disable_sparse_autobuild_during_save(self):
        semantic_map = self.unified_graph.semantic_map
        original = getattr(semantic_map, "build_sparse_embeddings", None)

        def _skip_sparse_build(_semantic_map, *args, **kwargs):
            self.logger.warning(
                "Skipping SemanticMap.build_sparse_embeddings during save; "
                "the unified graph only preserves sparse vectors already present in source graphs."
            )
            return {"status": "skipped", "reason": "disabled_by_build_unified_graph"}

        if original is not None:
            semantic_map.build_sparse_embeddings = types.MethodType(_skip_sparse_build, semantic_map)
        try:
            yield
        finally:
            if original is not None:
                semantic_map.build_sparse_embeddings = original

    def _write_report(self, stats: BuildStats) -> None:
        report_path = self.config.output_dir / "unified_build_report.json"
        report_path.write_text(json.dumps(asdict(stats), ensure_ascii=False, indent=2), encoding="utf-8")
        self.logger.info("Aggregate build report written to %s", report_path)

    def _log_summary(self, stats: BuildStats) -> None:
        self.logger.info("Per-sample unified graph build finished")
        self.logger.info("Samples with data: %d/%d", stats.samples_with_any_tower, stats.samples_requested)
        self.logger.info("Towers loaded/skipped: %d/%d", stats.towers_loaded, stats.towers_skipped)
        self.logger.info("Merged units/spaces/edges: %d/%d/%d", stats.total_units, stats.total_spaces, stats.total_edges)
        self.logger.info("UID collisions resolved: %d", stats.uid_collisions)
        self.logger.info(
            "Missing dense/invalid dense/missing sparse: %d/%d/%d",
            stats.missing_dense_embeddings,
            stats.invalid_dense_embeddings,
            stats.missing_sparse_embeddings,
        )
        self.logger.info("Index rebuild: %.2fs, save: %.2fs, total: %.2fs", stats.rebuild_seconds, stats.save_seconds, stats.total_seconds)

    def _cleanup_source_graph(self, graph: SemanticGraph) -> None:
        try:
            duckdb_connection = getattr(graph, "_duckdb_connection", None)
            if duckdb_connection is not None and getattr(duckdb_connection, "is_connected", False):
                duckdb_connection.close()
            graph._duckdb_connection = None
            graph.semantic_map._external_storage = None

            if getattr(graph.semantic_map, "faiss_index", None) is not None:
                try:
                    graph.semantic_map.faiss_index.reset()
                except Exception:
                    pass
                graph.semantic_map.faiss_index = None

            graph._multi_retriever = None
            graph.semantic_map._multi_retriever = None
            graph.semantic_map.memory_units.clear()
            graph.semantic_map.memory_spaces.clear()
            graph.rx_graph = rx.PyDiGraph(multigraph=True)
            graph._uid_to_index.clear()
            graph._index_to_uid.clear()
        except Exception as exc:
            self.logger.warning("Source graph cleanup failed: %s", exc)
        finally:
            gc.collect()
            self._empty_cuda_cache()

    def _empty_cuda_cache(self) -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            return


def parse_sample_ids(values: Optional[Sequence[str]]) -> Optional[List[str]]:
    if not values:
        return None

    sample_ids: List[str] = []
    for value in values:
        for token in value.replace(",", " ").split():
            stripped = token.strip()
            if stripped:
                sample_ids.append(stripped)

    if not sample_ids or any(token.lower() == "all" for token in sample_ids):
        return None

    deduped: Dict[str, None] = {}
    for sample_id in sample_ids:
        deduped.setdefault(sample_id, None)
    return list(deduped.keys())


def parse_args(argv: Optional[Sequence[str]] = None) -> BuilderConfig:
    parser = argparse.ArgumentParser(
        description="Build one unified SemanticGraph per sample from episodic, entity, and hierarchical sample graphs.",
    )
    parser.add_argument("--episodic-dir", type=Path, required=True, help="Directory containing episodic per-sample SemanticGraph folders.")
    parser.add_argument("--entity-dir", type=Path, required=True, help="Directory containing entity-relation per-sample SemanticGraph folders.")
    parser.add_argument("--hierarchical-dir", type=Path, required=True, help="Directory containing hierarchical per-sample SemanticGraph folders.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output root directory. Each sample is saved under output-dir/<sample-id>.")
    parser.add_argument("--sample-ids", nargs="+", required=True, help="Sample ids to merge. Use 'all' to discover graph folders from the three input roots. Commas are accepted.")
    parser.add_argument("--embedding-model", required=True, help="Text embedding model name used to initialize the target SemanticMap.")
    parser.add_argument("--splade-model", required=True, help="SPLADE model name used by the saved SPLADE retriever.")
    parser.add_argument("--image-embedding-model", default="clip-ViT-B-32", help="Image embedding model name for SemanticMap initialization.")
    parser.add_argument("--embedding-dim", type=int, default=None, help="Optional embedding dimension for custom embedding models.")
    parser.add_argument("--faiss-index-type", default="IDMap,Flat", help="FAISS index type for the unified SemanticMap.")
    parser.add_argument("--no-freeze-retrievers", action="store_true", help="Disable saving prebuilt static BM25/SPLADE matrices.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging verbosity.")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    return BuilderConfig(
        episodic_dir=args.episodic_dir,
        entity_dir=args.entity_dir,
        hierarchical_dir=args.hierarchical_dir,
        output_dir=args.output_dir,
        sample_ids=parse_sample_ids(args.sample_ids),
        embedding_model=args.embedding_model,
        splade_model=args.splade_model,
        image_embedding_model=args.image_embedding_model,
        embedding_dim=args.embedding_dim,
        faiss_index_type=args.faiss_index_type,
        freeze_retrievers=not args.no_freeze_retrievers,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    config = parse_args(argv)
    builder = UnifiedGraphBuilder(config)
    builder.build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())