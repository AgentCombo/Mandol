"""RocksDB-backed payload persistence for the paper artifact.

The store owns serialized ``MemoryUnit`` payloads only. Retrieval indexes,
UID mappings, MemorySpace membership, and graph topology remain in memory and
are persisted by their existing checkpoint paths.
"""

from __future__ import annotations

import os
import pickle
import shutil
import threading
from typing import Any, Dict, List, Optional

from rocksdict import Options, Rdict, WriteBatch

from ..core.memory_unit import MemoryUnit
from ..utils.logging_config import create_module_logger

logger = create_module_logger("storage.rocksdb_payload_store")


class RocksDBPayloadStore:
    """Persist Mandol memory payloads in an embedded RocksDB database.

    Keys and values use a small versioned binary namespace. A payload record
    contains the complete ``MemoryUnit`` needed to materialize a cold retrieval
    result, while all retrieval and graph state remains outside RocksDB.
    """

    FORMAT_VERSION = 1
    _UNIT_PREFIX = b"unit:"
    _FORMAT_KEY = b"meta:format_version"

    def __init__(self, db_path: str) -> None:
        """Open or create a RocksDB payload store.

        Args:
            db_path: Directory containing the RocksDB files.
        """
        if not db_path or db_path == ":memory:":
            raise ValueError("RocksDB requires a filesystem directory path.")
        self.db_path = os.path.abspath(db_path)
        self._lock = threading.RLock()
        self._db: Optional[Rdict] = None
        self._open()

    @property
    def is_connected(self) -> bool:
        """Return whether the RocksDB handle is open."""
        return self._db is not None

    def _open(self) -> None:
        os.makedirs(self.db_path, exist_ok=True)
        options = Options(raw_mode=True)
        self._db = Rdict(self.db_path, options=options)
        if self._db.get(self._FORMAT_KEY) is None:
            self._db[self._FORMAT_KEY] = str(self.FORMAT_VERSION).encode("ascii")

    @classmethod
    def _unit_key(cls, uid: str) -> bytes:
        return cls._UNIT_PREFIX + str(uid).encode("utf-8")

    @staticmethod
    def _serialize_unit(unit: MemoryUnit) -> bytes:
        return pickle.dumps(unit, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _deserialize_unit(payload: Optional[bytes]) -> Optional[MemoryUnit]:
        if payload is None:
            return None
        unit = pickle.loads(payload)
        if not isinstance(unit, MemoryUnit):
            raise TypeError("RocksDB payload is not a MemoryUnit.")
        return unit

    def add_unit(
        self,
        unit: MemoryUnit,
        space_names: Optional[List[str]] = None,
        **_: Any,
    ) -> bool:
        """Insert or replace one payload.

        ``space_names`` is accepted for API compatibility. MemorySpace
        membership remains resident in ``SemanticMap`` and is not duplicated in
        the payload store.
        """
        del space_names
        with self._lock:
            if self._db is None:
                return False
            self._db[self._unit_key(unit.uid)] = self._serialize_unit(unit)
        return True

    def add_units_batch(
        self,
        units: List[MemoryUnit],
        space_names: Optional[List[str]] = None,
        **_: Any,
    ) -> int:
        """Insert or replace payloads with one RocksDB write batch."""
        del space_names
        if not units:
            return 0
        batch = WriteBatch(raw_mode=True)
        for unit in units:
            batch.put(self._unit_key(unit.uid), self._serialize_unit(unit))
        with self._lock:
            if self._db is None:
                return 0
            self._db.write(batch)
        return len(units)

    def get_unit(self, unit_id: str) -> Optional[MemoryUnit]:
        """Materialize one payload by public UID."""
        with self._lock:
            if self._db is None:
                return None
            payload = self._db.get(self._unit_key(unit_id))
        return self._deserialize_unit(payload)

    def get_units_batch(self, unit_ids: List[str]) -> List[MemoryUnit]:
        """Materialize available payloads in caller-supplied UID order."""
        return [
            unit
            for unit_id in unit_ids
            if (unit := self.get_unit(unit_id)) is not None
        ]

    def unit_exists(self, unit_id: str) -> bool:
        """Return whether a payload exists for the UID."""
        with self._lock:
            return bool(
                self._db is not None
                and self._db.get(self._unit_key(unit_id)) is not None
            )

    def delete_unit(self, unit_id: str, **_: Any) -> bool:
        """Delete one payload if present."""
        key = self._unit_key(unit_id)
        with self._lock:
            if self._db is None:
                return False
            if self._db.get(key) is None:
                return False
            self._db.delete(key)
        return True

    def list_uids(self) -> List[str]:
        """Return all persisted payload UIDs."""
        with self._lock:
            if self._db is None:
                return []
            keys = list(self._db.keys())
        return [
            key[len(self._UNIT_PREFIX):].decode("utf-8")
            for key in keys
            if isinstance(key, bytes) and key.startswith(self._UNIT_PREFIX)
        ]

    def count_units(self) -> int:
        """Return the number of persisted payload records."""
        return len(self.list_uids())

    def swap_out(self, uids: List[str], l1_data: Dict[str, Any]) -> int:
        """Persist the selected resident payloads before cache eviction."""
        by_uid = {
            unit.uid: unit
            for unit in (l1_data.get("units") or [])
            if isinstance(unit, MemoryUnit)
        }
        units = [by_uid[uid] for uid in uids if uid in by_uid]
        return self.add_units_batch(units)

    def swap_in(self, uids: List[str]) -> List[MemoryUnit]:
        """Materialize payloads without modifying resident retrieval state."""
        return self.get_units_batch(uids)

    def flush(self) -> None:
        """Flush pending RocksDB writes."""
        with self._lock:
            if self._db is not None:
                self._db.flush()

    def copy_to(self, destination: str) -> str:
        """Copy a closed, flushed RocksDB directory into a checkpoint."""
        destination = os.path.abspath(destination)
        source = self.db_path
        if source == destination:
            self.flush()
            return destination

        with self._lock:
            self.close()
            try:
                if os.path.exists(destination):
                    shutil.rmtree(destination)
                os.makedirs(os.path.dirname(destination), exist_ok=True)
                shutil.copytree(source, destination)
            finally:
                self._open()
        return destination

    def clear_database(self, confirm: bool = False) -> bool:
        """Remove all payload records when explicitly confirmed."""
        if not confirm:
            logger.warning("Clearing the payload store requires confirm=True.")
            return False
        for uid in self.list_uids():
            self.delete_unit(uid)
        self.flush()
        return True

    def close(self) -> None:
        """Close the RocksDB handle."""
        with self._lock:
            if self._db is not None:
                self._db.flush()
                self._db.close()
                self._db = None

    def reopen(self) -> None:
        """Reopen a previously closed store."""
        with self._lock:
            if self._db is None:
                self._open()

    def __enter__(self) -> "RocksDBPayloadStore":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"RocksDBPayloadStore(db_path={self.db_path!r}, "
            f"connected={self.is_connected})"
        )
