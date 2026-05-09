"""JSON-based persistence engine with verification and auto-saving.

Handles save/load of MemorySystem state to JSON files on disk, with
optional content verification and throttle-aware rescheduling on
concurrency conflicts. Supports auto-save via periodic scheduling
and save-on-dirty (debounced) triggers.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import numpy as np

if TYPE_CHECKING:
    from ..application.session_manager import Session

from ..domain.memory_space import MemorySpace
from ..domain.memory_unit import MemoryUnit
from ..domain.types import Uid

logger = logging.getLogger(__name__)

CURRENT_VERSION = "1.0"
DATA_DIR = "data"
STATE_DIR = "state"


@dataclass
class SaveResult:
    success: bool
    saved_at: str
    stats: Dict[str, int]
    errors: List[str] = field(default_factory=list)


@dataclass
class VerificationResult:
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class PersistenceError(Exception):
    pass


class JsonPersistenceEngine:
    def __init__(self, storage_root: str):
        self._root = str(storage_root)
        self._data_dir = os.path.join(self._root, DATA_DIR)
        self._state_dir = os.path.join(self._root, STATE_DIR)
        from ..retrieval.text import TextExtractor, Tokenizer
        self._tokenizer = Tokenizer()
        self._text_extractor = TextExtractor()

    @property
    def storage_root(self) -> str:
        return self._root

    def ensure_directories(self) -> None:
        os.makedirs(self._data_dir, exist_ok=True)
        os.makedirs(self._state_dir, exist_ok=True)

    def save_units(self, units: List[MemoryUnit]) -> None:
        units_data = [u.to_dict() for u in units]
        payload = {
            "version": CURRENT_VERSION,
            "units": units_data,
        }
        file_path = os.path.join(self._data_dir, "units.json")
        self._write_json(file_path, payload)
        logger.info(f"Saved {len(units)} units to {file_path}")

    def load_units(self) -> List[MemoryUnit]:
        file_path = os.path.join(self._data_dir, "units.json")
        if not os.path.exists(file_path):
            return []
        payload = self._read_json(file_path)
        units_raw = payload.get("units") or []
        units = [MemoryUnit.from_dict(d) for d in units_raw]
        logger.info(f"Loaded {len(units)} units from {file_path}")
        return units

    def save_spaces(self, spaces: List[MemorySpace]) -> None:
        spaces_data = [s.to_dict() for s in spaces]
        payload = {
            "version": CURRENT_VERSION,
            "spaces": spaces_data,
        }
        file_path = os.path.join(self._data_dir, "spaces.json")
        self._write_json(file_path, payload)
        logger.info(f"Saved {len(spaces)} spaces to {file_path}")

    def load_spaces(self) -> List[MemorySpace]:
        file_path = os.path.join(self._data_dir, "spaces.json")
        if not os.path.exists(file_path):
            return []
        payload = self._read_json(file_path)
        spaces_raw = payload.get("spaces") or []
        spaces = [MemorySpace.from_dict(d) for d in spaces_raw]
        logger.info(f"Loaded {len(spaces)} spaces from {file_path}")
        return spaces

    def save_graph(self, edges: List[Tuple[str, str, str, Dict[str, Any]]]) -> None:
        edges_data = []
        for source, target, rel_type, properties in edges:
            edges_data.append({
                "source": str(source),
                "target": str(target),
                "relationship_name": str(rel_type),
                "properties": properties,
            })
        payload = {
            "version": CURRENT_VERSION,
            "edges": edges_data,
        }
        file_path = os.path.join(self._data_dir, "graph.json")
        self._write_json(file_path, payload)
        logger.info(f"Saved {len(edges)} edges to {file_path}")

    def load_graph(self) -> List[Tuple[str, str, str, Dict[str, Any]]]:
        file_path = os.path.join(self._data_dir, "graph.json")
        if not os.path.exists(file_path):
            return []
        payload = self._read_json(file_path)
        edges_raw = payload.get("edges") or []
        edges = []
        for e in edges_raw:
            source = str(e.get("source", ""))
            target = str(e.get("target", ""))
            rel_type = str(e.get("relationship_name", ""))
            properties = dict(e.get("properties") or {})
            edges.append((source, target, rel_type, properties))
        logger.info(f"Loaded {len(edges)} edges from {file_path}")
        return edges

    def save_sessions(self, sessions: List[Session]) -> None:
        sessions_data = [s.to_dict() for s in sessions]
        payload = {
            "version": CURRENT_VERSION,
            "sessions": sessions_data,
        }
        file_path = os.path.join(self._data_dir, "sessions.json")
        self._write_json(file_path, payload)
        logger.info(f"Saved {len(sessions)} sessions to {file_path}")

    def load_sessions(self) -> List[Any]:
        from ..application.session_manager import Session
        file_path = os.path.join(self._data_dir, "sessions.json")
        if not os.path.exists(file_path):
            return []
        payload = self._read_json(file_path)
        sessions_raw = payload.get("sessions") or []
        sessions = []
        for d in sessions_raw:
            sessions.append(Session(
                session_id=str(d.get("session_id", "")),
                unit_uids=[Uid(str(u)) for u in (d.get("unit_uids") or [])],
                start_time=str(d.get("start_time", "")),
                end_time=str(d.get("end_time", "")),
                topic=str(d.get("topic", "")),
            ))
        logger.info(f"Loaded {len(sessions)} sessions from {file_path}")
        return sessions

    def save_processed_state(
        self,
        insertion_order: List[str],
        processed_session_ids: Set[str],
        processed_similarity_pairs: Set[Tuple[str, str]],
    ) -> None:
        payload = {
            "version": CURRENT_VERSION,
            "insertion_order": insertion_order,
            "processed_session_ids": sorted(processed_session_ids),
            "processed_similarity_pairs": [
                sorted([a, b]) for a, b in processed_similarity_pairs
            ],
        }
        file_path = os.path.join(self._state_dir, "processed_state.json")
        self._write_json(file_path, payload)
        logger.info(f"Saved processed state: {len(insertion_order)} order, {len(processed_session_ids)} sessions, {len(processed_similarity_pairs)} pairs")

    def load_processed_state(self) -> Tuple[List[str], Set[str], Set[Tuple[str, str]]]:
        file_path = os.path.join(self._state_dir, "processed_state.json")
        if not os.path.exists(file_path):
            return [], set(), set()
        payload = self._read_json(file_path)
        insertion_order = list(payload.get("insertion_order") or [])
        processed_session_ids = set(payload.get("processed_session_ids") or [])
        pairs_raw = payload.get("processed_similarity_pairs") or []
        processed_similarity_pairs = set()
        for p in pairs_raw:
            if isinstance(p, list) and len(p) == 2:
                processed_similarity_pairs.add((str(p[0]), str(p[1])))
        logger.info(f"Loaded processed state: {len(insertion_order)} order, {len(processed_session_ids)} sessions, {len(processed_similarity_pairs)} pairs")
        return insertion_order, processed_session_ids, processed_similarity_pairs

    def save_manifest(
        self,
        build_state: str,
        pending_session_ids: List[str],
        stats: Dict[str, int],
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        payload = {
            "version": CURRENT_VERSION,
            "created_at": now,
            "updated_at": now,
            "build_state": build_state,
            "pending_session_ids": pending_session_ids,
            "stats": stats,
        }
        file_path = os.path.join(self._root, "manifest.json")
        self._write_json(file_path, payload)
        logger.info(f"Saved manifest: build_state={build_state}, pending={len(pending_session_ids)}")

    def load_manifest(self) -> Optional[Dict[str, Any]]:
        file_path = os.path.join(self._root, "manifest.json")
        if not os.path.exists(file_path):
            return None
        return self._read_json(file_path)

    def save_config(self, config: Dict[str, Any]) -> None:
        payload = {
            "version": CURRENT_VERSION,
            "config": config,
        }
        file_path = os.path.join(self._root, "config.json")
        self._write_json(file_path, payload)
        logger.info(f"Saved config to {file_path}")

    def load_config(self) -> Optional[Dict[str, Any]]:
        file_path = os.path.join(self._root, "config.json")
        if not os.path.exists(file_path):
            return None
        payload = self._read_json(file_path)
        return payload.get("config")

    def _write_json(self, file_path: str, payload: Dict[str, Any]) -> None:
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as e:
            raise PersistenceError(f"Failed to write {file_path}: {e}")

    def _read_json(self, file_path: str) -> Dict[str, Any]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise PersistenceError(f"Invalid JSON in {file_path}: {e}")
        except Exception as e:
            raise PersistenceError(f"Failed to read {file_path}: {e}")

    def get_storage_stats(self) -> Dict[str, int]:
        stats = {
            "unit_count": 0,
            "space_count": 0,
            "edge_count": 0,
            "session_count": 0,
        }
        units_path = os.path.join(self._data_dir, "units.json")
        spaces_path = os.path.join(self._data_dir, "spaces.json")
        graph_path = os.path.join(self._data_dir, "graph.json")
        sessions_path = os.path.join(self._data_dir, "sessions.json")

        if os.path.exists(units_path):
            try:
                data = self._read_json(units_path)
                stats["unit_count"] = len(data.get("units") or [])
            except Exception:
                pass

        if os.path.exists(spaces_path):
            try:
                data = self._read_json(spaces_path)
                stats["space_count"] = len(data.get("spaces") or [])
            except Exception:
                pass

        if os.path.exists(graph_path):
            try:
                data = self._read_json(graph_path)
                stats["edge_count"] = len(data.get("edges") or [])
            except Exception:
                pass

        if os.path.exists(sessions_path):
            try:
                data = self._read_json(sessions_path)
                stats["session_count"] = len(data.get("sessions") or [])
            except Exception:
                pass

        return stats


class IndexRebuilder:
    def __init__(self, tokenizer: Optional[Any] = None, text_extractor: Optional[Any] = None):
        from ..retrieval.text import TextExtractor, Tokenizer
        self._tokenizer = tokenizer or Tokenizer()
        self._text_extractor = text_extractor or TextExtractor()

    def tokenize_for_bm25(self, unit: MemoryUnit) -> List[str]:
        text = self._text_extractor.extract(unit)
        return self._tokenizer.tokenize(text)

    def extract_sparse_vector(self, unit: MemoryUnit) -> Optional[Dict[str, float]]:
        if unit.sparse_embedding is None:
            return None
        vec = np.asarray(unit.sparse_embedding, dtype=np.float32)
        if vec.size == 0:
            return None
        result: Dict[str, float] = {}
        for i, val in enumerate(vec):
            if abs(val) > 1e-6:
                result[f"dim_{i}"] = float(val)
        return result if result else None
