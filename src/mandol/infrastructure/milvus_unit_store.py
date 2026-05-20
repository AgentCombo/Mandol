"""Milvus-backed implementation of the UnitStore port.

Stores MemoryUnits in a Milvus vector database collection with HNSW
indexing for fast ANN search. MemorySpaces are kept in local memory.
Uses a write-back buffer (dirty units / deleted units) that is flushed
on explicit flush() calls.
"""

from __future__ import annotations

import json
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from ..domain.memory_space import MemorySpace
from ..domain.memory_unit import MemoryUnit
from ..domain.types import SpaceName, Uid
from ..ports.unit_store import UnitStore
from .config import MilvusConfig


class MilvusUnitStore(UnitStore):
    """Unit store backed by a Milvus vector database.

    Units are persisted in a Milvus collection with an HNSW index on
    the embedding field. Spaces are kept in local memory. Write operations
    are buffered and flushed explicitly via flush().

    Attributes:
        _cfg: Milvus connection configuration.
        _dim: Expected embedding dimensionality.
        _client: Lazily-initialized MilvusClient instance.
        _dirty_units: Write-back buffer for pending upserts.
        _deleted_units: Set of Uids marked for deletion.
        _spaces: In-memory mapping from SpaceName to MemorySpace.
    """

    def __init__(
        self,
        *,
        config: Optional[MilvusConfig] = None,
        embedding_dim: int,
        auto_create_collection: bool = True,
    ):
        self._cfg = config or MilvusConfig()
        self._dim = int(embedding_dim)
        self._auto_create = bool(auto_create_collection)

        self._client = None
        self._dirty_units: Dict[Uid, MemoryUnit] = {}
        self._deleted_units: set[Uid] = set()
        self._spaces: Dict[SpaceName, MemorySpace] = {}

        self._ensure_client_and_collection()

    def _ensure_client_and_collection(self) -> None:
        """Lazily connect to Milvus and create the collection if needed.

        Authentication credentials and database name are forwarded from
        the MilvusConfig. Collection creation respects *auto_create*.
        """
        if self._client is not None:
            return

        from pymilvus import MilvusClient

        kwargs = {}
        if self._cfg.user:
            kwargs["user"] = self._cfg.user
        if self._cfg.password:
            kwargs["password"] = self._cfg.password
        if self._cfg.db_name:
            kwargs["db_name"] = self._cfg.db_name

        self._client = MilvusClient(uri=self._cfg.uri, **kwargs)

        if self._auto_create:
            self._create_collection_if_missing()

    def _create_collection_if_missing(self) -> None:
        """Create the Milvus collection with HNSW index if it does not exist.

        Schema: uid (VARCHAR, 512, primary key) + embedding (FLOAT_VECTOR).
        HNSW uses Inner Product (IP) for cosine similarity on normalized vectors.
        """
        assert self._client is not None
        from pymilvus import DataType

        if self._client.has_collection(self._cfg.collection):
            return

        schema = self._client.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field(field_name="uid", datatype=DataType.VARCHAR, is_primary=True, max_length=512)
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=self._dim)

        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="HNSW",
            metric_type="IP",
            params={"M": 16, "efConstruction": 200},  # HNSW tuning: M = connectivity, efConstruction = build quality
        )

        self._client.create_collection(
            collection_name=self._cfg.collection,
            schema=schema,
            index_params=index_params,
            consistency_level="Strong",
        )

    def upsert_units(self, units: Sequence[MemoryUnit]) -> None:
        for u in units:
            self._dirty_units[Uid(str(u.uid))] = u
            self._deleted_units.discard(Uid(str(u.uid)))

    def delete_units(self, uids: Iterable[Uid]) -> None:
        for uid in uids:
            u = Uid(str(uid))
            self._dirty_units.pop(u, None)
            self._deleted_units.add(u)

    def get_unit(self, uid: Uid) -> Optional[MemoryUnit]:
        self._ensure_client_and_collection()
        assert self._client is not None
        u = Uid(str(uid))
        if u in self._dirty_units:
            return self._dirty_units[u]

        res = self._client.get(collection_name=self._cfg.collection, ids=[str(u)])
        if not res:
            return None
        row = res[0]
        return _row_to_unit(row)

    def get_units(self, uids: Sequence[Uid]) -> List[MemoryUnit]:
        self._ensure_client_and_collection()
        assert self._client is not None
        out: List[MemoryUnit] = []

        ids: List[str] = []
        for uid in uids:
            u = Uid(str(uid))
            if u in self._dirty_units:
                out.append(self._dirty_units[u])
            else:
                ids.append(str(u))

        if ids:
            res = self._client.get(collection_name=self._cfg.collection, ids=ids)
            for row in res or []:
                out.append(_row_to_unit(row))
        return out

    def list_units(self) -> List[MemoryUnit]:
        self._ensure_client_and_collection()
        assert self._client is not None
        # Milvus does not support efficient full scans; return only buffered dirty units
        return list(self._dirty_units.values())

    def upsert_spaces(self, spaces: Sequence[MemorySpace]) -> None:
        for s in spaces:
            self._spaces[SpaceName(str(s.name))] = s

    def get_space(self, name: SpaceName) -> Optional[MemorySpace]:
        return self._spaces.get(SpaceName(str(name)))

    def list_spaces(self) -> List[MemorySpace]:
        return list(self._spaces.values())

    def flush(self) -> None:
        """Persist all buffered writes and deletes to Milvus.

        Deletions are issued first via a filter expression on the uid field.
        Then dirty units with valid embeddings are upserted in batch.
        """
        self._ensure_client_and_collection()
        assert self._client is not None

        if self._deleted_units:
            expr = "uid in [" + ",".join([json.dumps(str(u)) for u in self._deleted_units]) + "]"
            self._client.delete(collection_name=self._cfg.collection, filter=expr)
            self._deleted_units.clear()

        if self._dirty_units:
            rows = []
            for u in self._dirty_units.values():
                emb = u.embedding
                if emb is None:
                    continue
                v = np.asarray(emb, dtype=np.float32).reshape(-1)
                if v.shape[0] != self._dim:
                    continue
                row = {
                    "uid": str(u.uid),
                    "embedding": v.tolist(),
                    "raw_data": json.dumps(u.raw_data, ensure_ascii=False),
                    "metadata": json.dumps(u.metadata, ensure_ascii=False),
                }
                rows.append(row)
            if rows:
                self._client.upsert(collection_name=self._cfg.collection, data=rows)
            self._dirty_units.clear()


def _row_to_unit(row: dict) -> MemoryUnit:
    """Deserialize a Milvus query result row back to a MemoryUnit.

    JSON-encoded raw_data and metadata fields are decoded; the embedding
    is converted to a float32 numpy array.

    Args:
        row: Raw row dict from a Milvus query.

    Returns:
        A reconstructed MemoryUnit.
    """
    uid = row.get("uid")
    raw_data = row.get("raw_data")
    metadata = row.get("metadata")
    embedding = row.get("embedding")

    if isinstance(raw_data, str):
        try:
            raw_data = json.loads(raw_data)
        except json.JSONDecodeError:
            raw_data = {"raw": raw_data}
    if not isinstance(raw_data, dict):
        raw_data = {"raw": raw_data}

    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            metadata = {}
    if not isinstance(metadata, dict):
        metadata = {}

    emb = None
    if embedding is not None:
        emb = np.asarray(embedding, dtype=np.float32)

    return MemoryUnit(uid=str(uid), raw_data=raw_data, metadata=metadata, embedding=emb)
