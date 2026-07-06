# duckdb_operator.py
"""Schema: memory_nodes( uid, raw_data JSON, metadata JSON, content_type, dense_embedding FLOAT[dim], splade_indices UBIGINT[], splade_values FLOAT[], memory_spaces VARCHAR[], embedding_model, created_at, updated_at, access_count ) memory_edge."""

import json
import os
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Any, Union, Generator

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from ..core.memory_unit import MemoryUnit
from ..core.memory_space import MemorySpace
from ..utils.logging_config import create_module_logger

logger = create_module_logger("duckdb_operator")

try:
    from scipy.sparse import csr_matrix as scipy_csr_matrix
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import orjson
    _json_loads = orjson.loads
    def _json_dumps(value: Any) -> str:
        return orjson.dumps(value if value is not None else {}, option=orjson.OPT_SERIALIZE_NUMPY).decode("utf-8")
except ImportError:
    _json_loads = json.loads
    def _json_dumps(value: Any) -> str:
        return json.dumps(value if value is not None else {}, ensure_ascii=False, default=str)


class DuckDBOperator:

    def __init__(
        self,
        db_path: str = ":memory:",
        read_only: bool = False,
        embedding_dim: int = 1024,
        max_retry_attempts: int = 3,
    ):
        self.db_path = db_path
        self.read_only = read_only
        self.embedding_dim = embedding_dim
        self.max_retry_attempts = max_retry_attempts
        self.is_connected = False
        self.con: Optional[duckdb.DuckDBPyConnection] = None

        self._vss_loaded = False
        self._pgq_loaded = False

        # Storage operations must preserve transactional consistency.
        self._transaction_depth: int = 0

        self.query_stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "total_query_time": 0.0,
        }

        self._connect()
        if self.is_connected:
            self._load_extensions()
            self._initialize_schema()

    
    

    def _connect(self) -> bool:
        """Run connect."""
        try:
            self.con = duckdb.connect(
                database=self.db_path,
                read_only=self.read_only,
            )
            self.is_connected = True
            logger.info(f"DuckDB connected: {self.db_path}")
            return True
        except Exception as e:
            logger.error(f"DuckDB connection failed: {e}")
            self.is_connected = False
            return False

    def _load_extensions(self):
        """Load extensions."""
        
        try:
            self.con.execute("INSTALL vss;")
            self.con.execute("LOAD vss;")
            
            self.con.execute("SET hnsw_enable_experimental_persistence=true;")
            self._vss_loaded = True
            logger.info("VSS extension loaded; HNSW dense-vector retrieval is available")
        except Exception as e:
            self._vss_loaded = False
            logger.warning(f"VSS extension loading failed; dense-vector retrieval falls back to brute-force scan: {e}")

        try:
            self.con.execute("INSTALL duckpgq FROM community;")
            self.con.execute("LOAD duckpgq;")
            self._pgq_loaded = True
            logger.info("DuckPGQ extension loaded; MATCH graph traversal is available")
        except Exception as e:
            self._pgq_loaded = False
            logger.info(
                f"DuckPGQ extension unavailable (v{duckdb.__version__}); graph traversal falls back to recursive CTE: {e}"
            )

    def _initialize_schema(self):
        """Initialize schema."""
        dim = self.embedding_dim
        try:
            
            self.con.execute(f"""
                CREATE TABLE IF NOT EXISTS memory_nodes (
                    uid              VARCHAR PRIMARY KEY,
                    raw_data         JSON,
                    metadata         JSON,
                    content_type     VARCHAR DEFAULT 'mixed',
                    -- Dense vector.
                    dense_embedding  FLOAT[{dim}],
                    -- SPLADE sparse vectors as paired arrays.
                    splade_indices   UBIGINT[],
                    splade_values    FLOAT[],
                    -- Per-node BM25 term-frequency columns bound to memory nodes.
                    bm25_terms       VARCHAR[],
                    bm25_tfs         INTEGER[],
                    -- Namespace.
                    memory_spaces    VARCHAR[],
                    embedding_model  VARCHAR DEFAULT 'default',
                    -- Timestamps.
                    created_at       TIMESTAMP DEFAULT current_timestamp,
                    updated_at       TIMESTAMP DEFAULT current_timestamp,
                    access_count     BIGINT DEFAULT 0
                );
            """)

            self.con.execute("""
                CREATE TABLE IF NOT EXISTS memory_edges (
                    source        VARCHAR NOT NULL,
                    target        VARCHAR NOT NULL,
                    relation_type VARCHAR NOT NULL,
                    properties    JSON DEFAULT '{}',
                    created_at    TIMESTAMP DEFAULT current_timestamp,
                    updated_at    TIMESTAMP DEFAULT current_timestamp,
                    PRIMARY KEY (source, target, relation_type)
                );
            """)

            
            self.con.execute(
                "CREATE INDEX IF NOT EXISTS idx_edges_source ON memory_edges (source);"
            )
            self.con.execute(
                "CREATE INDEX IF NOT EXISTS idx_edges_target ON memory_edges (target);"
            )
            self.con.execute(
                "CREATE INDEX IF NOT EXISTS idx_edges_rel_type ON memory_edges (relation_type);"
            )
            self.con.execute(
                "CREATE INDEX IF NOT EXISTS idx_nodes_content_type ON memory_nodes (content_type);"
            )

            
            if self._vss_loaded:
                try:
                    self.con.execute(f"""
                        CREATE INDEX IF NOT EXISTS idx_dense_hnsw
                        ON memory_nodes USING HNSW (dense_embedding)
                        WITH (metric = 'cosine');
                    """)
                    logger.info("HNSW vector index created")
                except Exception as e:
                    logger.warning(f"HNSW index creation failed; it may already exist or contain only NULL vectors: {e}")

            
            self.con.execute("""
                CREATE TABLE IF NOT EXISTS bm25_inverted_index (
                    collection_id  VARCHAR NOT NULL,
                    term           VARCHAR NOT NULL,
                    uid            VARCHAR NOT NULL,
                    tf             INTEGER NOT NULL,
                    PRIMARY KEY (collection_id, term, uid)
                );
            """)
            self.con.execute("""
                CREATE TABLE IF NOT EXISTS bm25_dfs (
                    collection_id  VARCHAR NOT NULL,
                    term           VARCHAR NOT NULL,
                    df             INTEGER NOT NULL,
                    PRIMARY KEY (collection_id, term)
                );
            """)
            self.con.execute("""
                CREATE TABLE IF NOT EXISTS bm25_doc_lengths (
                    collection_id  VARCHAR NOT NULL,
                    uid            VARCHAR NOT NULL,
                    doc_len        INTEGER NOT NULL,
                    PRIMARY KEY (collection_id, uid)
                );
            """)
            
            self.con.execute("""
                CREATE TABLE IF NOT EXISTS bm25_global_stats (
                    term  VARCHAR PRIMARY KEY,
                    df    INTEGER NOT NULL DEFAULT 0
                );
            """)
            
            self.con.execute(
                "CREATE INDEX IF NOT EXISTS idx_bm25_inv_coll_term "
                "ON bm25_inverted_index (collection_id, term);"
            )
            self.con.execute(
                "CREATE INDEX IF NOT EXISTS idx_bm25_dfs_coll "
                "ON bm25_dfs (collection_id);"
            )
            self.con.execute(
                "CREATE INDEX IF NOT EXISTS idx_bm25_doclen_coll "
                "ON bm25_doc_lengths (collection_id);"
            )

            if self._pgq_loaded:
                try:
                    self.con.execute("""
                        CREATE PROPERTY GRAPH IF NOT EXISTS hippo_graph
                        VERTEX TABLES (memory_nodes)
                        EDGE TABLES (
                            memory_edges
                                SOURCE KEY (source) REFERENCES memory_nodes (uid)
                                DESTINATION KEY (target) REFERENCES memory_nodes (uid)
                        );
                    """)
                    logger.info("Property Graph 'hippo_graph' created")
                except Exception as e:
                    logger.warning(f"Property Graph creation failed: {e}")
                    self._pgq_loaded = False

            logger.info("DuckDB unified schema initialized")
        except Exception as e:
            logger.error(f"DuckDB schema initialization failed: {e}")

    
    # Storage operations must preserve transactional consistency.
    

    @contextmanager
    def transaction(self) -> Generator[None, None, None]:
        """Run transaction."""
        is_outermost = self._transaction_depth == 0
        self._transaction_depth += 1
        try:
            if is_outermost:
                self.con.execute("BEGIN TRANSACTION")
            yield
            if is_outermost:
                self.con.execute("COMMIT")
        except Exception:
            if is_outermost:
                try:
                    self.con.execute("ROLLBACK")
                except Exception:
                    pass
            raise
        finally:
            self._transaction_depth -= 1

    @property
    def in_transaction(self) -> bool:
        """Run in transaction."""
        return self._transaction_depth > 0

    
    

    def _execute(
        self,
        query: str,
        parameters: Optional[list] = None,
    ) -> Optional[duckdb.DuckDBPyRelation]:
        """Execute."""
        if not self.is_connected:
            logger.error("DuckDB is not connected")
            return None

        start_time = time.time()
        for attempt in range(self.max_retry_attempts):
            try:
                if parameters:
                    result = self.con.execute(query, parameters)
                else:
                    result = self.con.execute(query)
                elapsed = time.time() - start_time
                self.query_stats["total_queries"] += 1
                self.query_stats["successful_queries"] += 1
                self.query_stats["total_query_time"] += elapsed
                return result
            except Exception as e:
                if attempt < self.max_retry_attempts - 1:
                    logger.warning(f"Query retry {attempt+1}/{self.max_retry_attempts}: {e}")
                    time.sleep(0.1 * (2 ** attempt))
                else:
                    logger.error(f"Query execution failed: {e}\nSQL: {query[:200]}")
                    self.query_stats["total_queries"] += 1
                    self.query_stats["failed_queries"] += 1
        return None

    
    

    def add_unit(
        self,
        unit: MemoryUnit,
        space_names: Optional[List[str]] = None,
        content_type: str = "mixed",
        embedding_model: str = "default",
        labels: Optional[List[str]] = None,
        properties: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Add unit."""
        try:
            raw_json = json.dumps(unit.raw_data, ensure_ascii=False)
            meta_json = json.dumps(unit.metadata or {}, ensure_ascii=False)
            ct = content_type or self._infer_content_type(unit.raw_data)
            now = datetime.now().isoformat()

            
            dense = unit.embedding.tolist() if unit.embedding is not None else None

            
            splade_idx, splade_val = None, None
            if properties:
                splade_idx = properties.get("splade_indices")
                splade_val = properties.get("splade_values")

            spaces = space_names or []

            self._execute(
                f"""
                INSERT OR REPLACE INTO memory_nodes
                    (uid, raw_data, metadata, content_type,
                     dense_embedding,
                     splade_indices, splade_values,
                     memory_spaces, embedding_model,
                     created_at, updated_at, access_count)
                VALUES (?, ?, ?, ?,
                        ?::FLOAT[{self.embedding_dim}],
                        ?, ?,
                        ?, ?,
                        ?, ?, 0)
                """,
                [
                    unit.uid, raw_json, meta_json, ct,
                    dense,
                    splade_idx, splade_val,
                    spaces, embedding_model,
                    now, now,
                ],
            )
            logger.debug(f"Node written: {unit.uid}")
            return True
        except Exception as e:
            logger.error(f"Failed to add node {unit.uid}: {e}")
            return False

    def add_units_batch(
        self,
        units: List[MemoryUnit],
        space_names: Optional[List[str]] = None,
        content_type: str = "mixed",
        embedding_model: str = "default",
    ) -> int:
        """Add units batch."""
        if not units:
            return 0
        success = 0
        now = datetime.now().isoformat()
        spaces = space_names or []
        try:
            self.con.execute("BEGIN TRANSACTION")
            for unit in units:
                raw_json = json.dumps(unit.raw_data, ensure_ascii=False)
                meta_json = json.dumps(unit.metadata or {}, ensure_ascii=False)
                ct = content_type or self._infer_content_type(unit.raw_data)
                dense = unit.embedding.tolist() if unit.embedding is not None else None

                self.con.execute(
                    f"""INSERT OR REPLACE INTO memory_nodes
                        (uid, raw_data, metadata, content_type,
                         dense_embedding, memory_spaces, embedding_model,
                         created_at, updated_at)
                        VALUES (?, ?, ?, ?,
                                ?::FLOAT[{self.embedding_dim}], ?, ?,
                                ?, ?)""",
                    [unit.uid, raw_json, meta_json, ct,
                     dense, spaces, embedding_model, now, now],
                )
                success += 1
            self.con.execute("COMMIT")
            logger.info(f"Batch wrote {success} nodes")
        except Exception as e:
            self.con.execute("ROLLBACK")
            logger.error(f"Batch node write failed: {e}")
        return success

    def get_unit(self, unit_id: str) -> Optional[MemoryUnit]:
        """Return unit."""
        result = self._execute(
            "SELECT uid, raw_data, metadata, dense_embedding FROM memory_nodes WHERE uid = ?",
            [unit_id],
        )
        if result is None:
            return None
        rows = result.fetchall()
        if not rows:
            return None
        return self._row_to_memory_unit(rows[0])

    def get_units_batch(self, unit_ids: List[str]) -> List[MemoryUnit]:
        """Return units batch."""
        if not unit_ids:
            return []
        placeholders = ", ".join(["?"] * len(unit_ids))
        result = self._execute(
            f"SELECT uid, raw_data, metadata, dense_embedding FROM memory_nodes WHERE uid IN ({placeholders})",
            unit_ids,
        )
        if result is None:
            return []
        units = []
        for row in result.fetchall():
            unit = self._row_to_memory_unit(row)
            if unit:
                units.append(unit)
        return units

    def delete_unit(self, unit_id: str, delete_relationships: bool = True) -> bool:
        """Remove unit."""
        try:
            if delete_relationships:
                self._execute(
                    "DELETE FROM memory_edges WHERE source = ? OR target = ?",
                    [unit_id, unit_id],
                )
            self._execute("DELETE FROM memory_nodes WHERE uid = ?", [unit_id])
            logger.debug(f"Node deleted: {unit_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete node {unit_id}: {e}")
            return False

    def unit_exists(self, unit_id: str) -> bool:
        result = self._execute(
            "SELECT 1 FROM memory_nodes WHERE uid = ? LIMIT 1", [unit_id]
        )
        if result is None:
            return False
        return len(result.fetchall()) > 0

    def ensure_node_exists(self, node_id: str, node_type: str = "MemoryUnit") -> bool:
        """Ensure node exists."""
        if self.unit_exists(node_id):
            return True
        try:
            now = datetime.now().isoformat()
            self._execute(
                """INSERT OR IGNORE INTO memory_nodes
                   (uid, raw_data, metadata, content_type, created_at, updated_at)
                   VALUES (?, '{}', '{}', ?, ?, ?)""",
                [node_id, node_type, now, now],
            )
            return True
        except Exception as e:
            logger.error(f"Failed to ensure node exists: {e}")
            return False

    
    

    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None,
        source_labels: Optional[List[str]] = None,
        target_labels: Optional[List[str]] = None,
    ) -> bool:
        try:
            props_json = json.dumps(properties or {}, ensure_ascii=False)
            now = datetime.now().isoformat()
            self._execute(
                """INSERT OR REPLACE INTO memory_edges
                   (source, target, relation_type, properties, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                [source_id, target_id, relationship_type, props_json, now, now],
            )
            logger.debug(f"Relationship written: ({source_id})-[{relationship_type}]->({target_id})")
            return True
        except Exception as e:
            logger.error(f"Failed to add relationship: {e}")
            return False

    def add_relationships_batch(
        self,
        relationships: List[Tuple[str, str, str, Optional[Dict[str, Any]]]],
    ) -> int:
        """Add relationships batch."""
        if not relationships:
            return 0
        success = 0
        now = datetime.now().isoformat()
        try:
            self.con.execute("BEGIN TRANSACTION")
            for rel in relationships:
                src, tgt, rtype = rel[0], rel[1], rel[2]
                props = json.dumps(
                    rel[3] if len(rel) > 3 and rel[3] else {}, ensure_ascii=False
                )
                self.con.execute(
                    """INSERT OR REPLACE INTO memory_edges
                       (source, target, relation_type, properties, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    [src, tgt, rtype, props, now, now],
                )
                success += 1
            self.con.execute("COMMIT")
        except Exception as e:
            self.con.execute("ROLLBACK")
            logger.error(f"Batch relationship insert failed: {e}")
        return success

    def get_relationships(
        self,
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
        relationship_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        conditions, params = [], []
        if source_id:
            conditions.append("source = ?")
            params.append(source_id)
        if target_id:
            conditions.append("target = ?")
            params.append(target_id)
        if relationship_type:
            conditions.append("relation_type = ?")
            params.append(relationship_type)
        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)
        result = self._execute(
            f"SELECT source, target, relation_type, properties FROM memory_edges{where} LIMIT ?",
            params,
        )
        if result is None:
            return []
        return [
            {
                "source": r[0],
                "target": r[1],
                "type": r[2],
                "properties": json.loads(r[3]) if r[3] else {},
            }
            for r in result.fetchall()
        ]

    def delete_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: Optional[str] = None,
    ) -> bool:
        try:
            if relationship_type:
                self._execute(
                    "DELETE FROM memory_edges WHERE source = ? AND target = ? AND relation_type = ?",
                    [source_id, target_id, relationship_type],
                )
            else:
                self._execute(
                    "DELETE FROM memory_edges WHERE source = ? AND target = ?",
                    [source_id, target_id],
                )
            return True
        except Exception as e:
            logger.error(f"Failed to delete relationship: {e}")
            return False

    def update_relationship_properties(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        properties: Dict[str, Any],
        merge: bool = True,
    ) -> bool:
        try:
            if merge:
                result = self._execute(
                    "SELECT properties FROM memory_edges WHERE source = ? AND target = ? AND relation_type = ?",
                    [source_id, target_id, relationship_type],
                )
                if result:
                    rows = result.fetchall()
                    if rows:
                        existing = json.loads(rows[0][0]) if rows[0][0] else {}
                        existing.update(properties)
                        properties = existing
            props_json = json.dumps(properties, ensure_ascii=False)
            now = datetime.now().isoformat()
            self._execute(
                "UPDATE memory_edges SET properties = ?, updated_at = ? WHERE source = ? AND target = ? AND relation_type = ?",
                [props_json, now, source_id, target_id, relationship_type],
            )
            return True
        except Exception as e:
            logger.error(f"Failed to update relationship properties: {e}")
            return False

    
    
    

    def search_dense(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        space_names: Optional[List[str]] = None,
        filter_uids: Optional[List[str]] = None,
    ) -> List[Tuple[str, float]]:
        """Retrieve dense."""
        query_list = query_embedding.tolist()

        conditions = ["dense_embedding IS NOT NULL"]
        params: list = []

        if space_names:
            space_placeholders = ", ".join(["?"] * len(space_names))
            conditions.append(
                f"list_has_any(memory_spaces, [{space_placeholders}]::VARCHAR[])"
            )
            params.extend(space_names)

        if filter_uids:
            uid_ph = ", ".join(["?"] * len(filter_uids))
            conditions.append(f"uid IN ({uid_ph})")
            params.extend(filter_uids)

        where = " AND ".join(conditions)

        query = f"""
            SELECT uid,
                   array_cosine_similarity(
                       dense_embedding,
                       ?::FLOAT[{self.embedding_dim}]
                   ) AS score
            FROM memory_nodes
            WHERE {where}
            ORDER BY array_distance(
                dense_embedding,
                ?::FLOAT[{self.embedding_dim}]
            )
            LIMIT ?
        """
        params = [query_list] + params + [query_list, k]

        result = self._execute(query, params)
        if result is None:
            return []
        return [(r[0], float(r[1])) for r in result.fetchall()]

    def search_dense_units(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        space_names: Optional[List[str]] = None,
    ) -> List[Tuple[MemoryUnit, float]]:
        """Retrieve dense units."""
        hits = self.search_dense(query_embedding, k, space_names)
        if not hits:
            return []
        uids = [h[0] for h in hits]
        score_map = {h[0]: h[1] for h in hits}
        units = self.get_units_batch(uids)
        return [(u, score_map.get(u.uid, 0.0)) for u in units]

    
    
    

    def update_sparse_vectors(
        self,
        uid: str,
        splade_indices: Optional[List[int]] = None,
        splade_values: Optional[List[float]] = None,
    ) -> bool:
        """Update sparse vectors."""
        try:
            sets, params = [], []
            now = datetime.now().isoformat()

            if splade_indices is not None:
                sets.append("splade_indices = ?")
                sets.append("splade_values = ?")
                params.extend([splade_indices, splade_values])

            if not sets:
                return True

            sets.append("updated_at = ?")
            params.append(now)
            params.append(uid)

            self._execute(
                f"UPDATE memory_nodes SET {', '.join(sets)} WHERE uid = ?",
                params,
            )
            return True
        except Exception as e:
            logger.error(f"Failed to update sparse vector ({uid}): {e}")
            return False

    def update_sparse_vectors_batch(
        self,
        updates: List[Dict[str, Any]],
    ) -> int:
        """Args: updates: [{"uid": str, "splade_indices": [...], "splade_values": [...]}, ...]."""
        if not updates:
            return 0
        success = 0
        now = datetime.now().isoformat()
        try:
            self.con.execute("BEGIN TRANSACTION")
            for u in updates:
                uid = u["uid"]
                sets, params = [], []
                if "splade_indices" in u and u["splade_indices"] is not None:
                    sets.append("splade_indices = ?")
                    sets.append("splade_values = ?")
                    params.extend([u["splade_indices"], u["splade_values"]])
                if not sets:
                    continue
                sets.append("updated_at = ?")
                params.append(now)
                params.append(uid)
                self.con.execute(
                    f"UPDATE memory_nodes SET {', '.join(sets)} WHERE uid = ?",
                    params,
                )
                success += 1
            self.con.execute("COMMIT")
            logger.info(f"Updated sparse vectors in batch: {success} rows")
        except Exception as e:
            self.con.execute("ROLLBACK")
            logger.error(f"Batch sparse-vector update failed: {e}")
        return success

    def read_sparse_vectors_batch(
        self,
        uids: List[str],
    ) -> List[Tuple[str, List[int], List[float]]]:
        """Read stored SPLADE vectors for a batch of memory-node UIDs.

        Args:
            uids: Memory-node UIDs to fetch from DuckDB.

        Returns:
            Tuples of ``(uid, splade_indices, splade_values)`` for available
            sparse vectors.
        """
        if not uids:
            return []
        try:
            uid_arrow = pa.table({"uid": pa.array(list(uids), type=pa.utf8())})
            self.con.register("_splade_swap_uids", uid_arrow)
            try:
                result = self.con.execute(
                    "SELECT n.uid, n.splade_indices, n.splade_values "
                    "FROM memory_nodes n "
                    "JOIN _splade_swap_uids u ON n.uid = u.uid"
                )
                rows = result.fetchall() if result else []
            finally:
                self.con.unregister("_splade_swap_uids")

            out: List[Tuple[str, List[int], List[float]]] = []
            for row in rows:
                uid = row[0]
                indices = list(row[1]) if row[1] else []
                values = list(row[2]) if row[2] else []
                out.append((uid, indices, values))
            return out
        except Exception as e:
            logger.error(f"read_sparse_vectors_batch failed: {e}")
            return []

    
    

    def graph_traverse(
        self,
        start_uids: List[str],
        max_hops: int = 2,
        relationship_types: Optional[List[str]] = None,
        direction: str = "both",
        node_limit: int = 500,
    ) -> Dict[str, Any]:
        """Args: direction: "out" / "in" / "both" Returns: {"nodes": [{"uid", "raw_data", "metadata"}...], "edges": [{"source", "target", "type", "properties"}...]}."""
        if not start_uids:
            return {"nodes": [], "edges": []}

        if self._pgq_loaded:
            return self._graph_traverse_pgq(
                start_uids, max_hops, relationship_types, direction, node_limit
            )
        return self._graph_traverse_cte(
            start_uids, max_hops, relationship_types, direction, node_limit
        )

    def _graph_traverse_pgq(
        self,
        start_uids: List[str],
        max_hops: int,
        relationship_types: Optional[List[str]],
        direction: str,
        node_limit: int,
    ) -> Dict[str, Any]:
        """Run graph traverse PGQ."""
        if direction == "in":
            return self._graph_traverse_cte(
                start_uids, max_hops, relationship_types, direction, node_limit
            )

        graph_name = "hippo_graph"
        edge_label = "memory_edges"
        tmp_cleanup = False

        
        try:
            self._execute("CREATE OR REPLACE TEMP TABLE _pgq_start_uids (uid VARCHAR)")
            for uid in start_uids:
                self._execute(
                    "INSERT INTO _pgq_start_uids VALUES (?)", [uid]
                )
        except Exception as e:
            logger.warning(f"PGQ temporary table creation failed; falling back to CTE traversal: {e}")
            return self._graph_traverse_cte(
                start_uids, max_hops, relationship_types, direction, node_limit
            )

        if relationship_types:
            try:
                rp = ", ".join(["?"] * len(relationship_types))
                self._execute(
                    f"CREATE OR REPLACE TEMP TABLE _pgq_filtered_edges AS "
                    f"SELECT * FROM memory_edges WHERE relation_type IN ({rp})",
                    list(relationship_types),
                )
                try:
                    self._execute("DROP PROPERTY GRAPH IF EXISTS _pgq_tmp_graph")
                except Exception:
                    pass
                self._execute("""
                    CREATE PROPERTY GRAPH _pgq_tmp_graph
                    VERTEX TABLES (memory_nodes)
                    EDGE TABLES (
                        _pgq_filtered_edges
                            SOURCE KEY (source) REFERENCES memory_nodes (uid)
                            DESTINATION KEY (target) REFERENCES memory_nodes (uid)
                    )
                """)
                graph_name = "_pgq_tmp_graph"
                edge_label = "_pgq_filtered_edges"
                tmp_cleanup = True
            except Exception as e:
                logger.warning(f"PGQ temporary property graph creation failed; falling back to CTE traversal: {e}")
                self._execute("DROP TABLE IF EXISTS _pgq_start_uids")
                return self._graph_traverse_cte(
                    start_uids, max_hops, relationship_types, direction, node_limit
                )

        if direction == "out":
            pattern = f"(a:memory_nodes)-[e:{edge_label}]->{{{1},{max_hops}}}(b:memory_nodes)"
        else:  # both
            pattern = f"(a:memory_nodes)-[e:{edge_label}]-{{{1},{max_hops}}}(b:memory_nodes)"

        query = f"""
            FROM GRAPH_TABLE ({graph_name}
                MATCH {pattern}
                WHERE a.uid IN (SELECT uid FROM _pgq_start_uids)
                COLUMNS (a.uid AS src_uid, b.uid AS dst_uid)
            )
            LIMIT {int(node_limit)}
        """

        try:
            result = self._execute(query)
            # Storage operations must preserve transactional consistency.
            rows = result.fetchall() if result else []
        except Exception as e:
            logger.warning(f"PGQ MATCH failed; falling back to CTE: {e}")
            rows = None
        finally:
            try:
                self._execute("DROP TABLE IF EXISTS _pgq_start_uids")
            except Exception:
                pass
            if tmp_cleanup:
                try:
                    self._execute("DROP PROPERTY GRAPH IF EXISTS _pgq_tmp_graph")
                    self._execute("DROP TABLE IF EXISTS _pgq_filtered_edges")
                except Exception:
                    pass

        if rows is None:
            logger.warning("PGQ MATCH failed; falling back to recursive CTE")
            return self._graph_traverse_cte(
                start_uids, max_hops, relationship_types, direction, node_limit
            )

        all_uids: Set[str] = set(start_uids)
        for r in rows:
            all_uids.add(r[0])
            all_uids.add(r[1])

        if not all_uids:
            return {"nodes": [], "edges": []}

        return self._collect_subgraph_full(all_uids)

    def _graph_traverse_cte(
        self,
        start_uids: List[str],
        max_hops: int,
        relationship_types: Optional[List[str]],
        direction: str,
        node_limit: int,
    ) -> Dict[str, Any]:
        """Run graph traverse cte."""
        uid_ph = ", ".join(["?"] * len(start_uids))
        params: list = list(start_uids)

        rel_filter = ""
        if relationship_types:
            rp = ", ".join(["?"] * len(relationship_types))
            rel_filter = f"AND e.relation_type IN ({rp})"
            params.extend(relationship_types)

        if direction == "out":
            edge_join = "e.source = x.uid"
            next_node = "e.target"
        elif direction == "in":
            edge_join = "e.target = x.uid"
            next_node = "e.source"
        else:
            edge_join = "(e.source = x.uid OR e.target = x.uid)"
            next_node = "CASE WHEN e.source = x.uid THEN e.target ELSE e.source END"

        query = f"""
            WITH RECURSIVE expand(uid, depth) AS (
                SELECT uid, 0
                FROM memory_nodes
                WHERE uid IN ({uid_ph})

                UNION

                SELECT {next_node}, x.depth + 1
                FROM expand x
                JOIN memory_edges e ON {edge_join} {rel_filter}
                WHERE x.depth < ?
            )
            SELECT DISTINCT uid FROM expand LIMIT ?
        """
        params += [max_hops, node_limit]
        result = self._execute(query, params)
        if result is None:
            return {"nodes": [], "edges": []}

        node_uids = {r[0] for r in result.fetchall()}
        if not node_uids:
            return {"nodes": [], "edges": []}

        return self._collect_subgraph_full(node_uids)

    def _collect_subgraph(
        self, node_uids: Set[str], edge_tuples: List[Tuple[str, str, str]]
    ) -> Dict[str, Any]:
        """Run collect subgraph."""
        uid_list = list(node_uids)
        np_str = ", ".join(["?"] * len(uid_list))
        nodes_result = self._execute(
            f"SELECT uid, raw_data, metadata FROM memory_nodes WHERE uid IN ({np_str})",
            uid_list,
        )
        nodes = []
        if nodes_result:
            for r in nodes_result.fetchall():
                nodes.append({
                    "uid": r[0],
                    "raw_data": json.loads(r[1]) if r[1] else {},
                    "metadata": json.loads(r[2]) if r[2] else {},
                })
        edges = [{"source": s, "target": t, "type": rel, "properties": {}} for s, t, rel in edge_tuples]
        return {"nodes": nodes, "edges": edges}

    def _collect_subgraph_full(self, node_uids: Set[str]) -> Dict[str, Any]:
        """Run collect subgraph full."""
        uid_list = list(node_uids)
        np_str = ", ".join(["?"] * len(uid_list))
        nodes_result = self._execute(
            f"SELECT uid, raw_data, metadata FROM memory_nodes WHERE uid IN ({np_str})",
            uid_list,
        )
        nodes = []
        if nodes_result:
            for r in nodes_result.fetchall():
                nodes.append({
                    "uid": r[0],
                    "raw_data": json.loads(r[1]) if r[1] else {},
                    "metadata": json.loads(r[2]) if r[2] else {},
                })
        edges_result = self._execute(
            f"""SELECT source, target, relation_type, properties
                FROM memory_edges
                WHERE source IN ({np_str}) AND target IN ({np_str})""",
            uid_list + uid_list,
        )
        edges = []
        if edges_result:
            for r in edges_result.fetchall():
                edges.append({
                    "source": r[0], "target": r[1], "type": r[2],
                    "properties": json.loads(r[3]) if r[3] else {},
                })
        return {"nodes": nodes, "edges": edges}

    
    def get_neighbors(
        self,
        node_id: str,
        direction: str = "both",
        relationship_types: Optional[List[str]] = None,
        depth: int = 1,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Return neighbors."""
        sub = self.graph_traverse(
            [node_id], max_hops=depth,
            relationship_types=relationship_types,
            direction=direction, node_limit=limit,
        )
        return [n for n in sub["nodes"] if n["uid"] != node_id]

    def find_shortest_path(
        self,
        source_id: str,
        target_id: str,
        relationship_types: Optional[List[str]] = None,
        max_depth: int = 6,
    ) -> Optional[Dict[str, Any]]:
        """Find shortest path."""
        rel_filter = ""
        params: list = [source_id, source_id]
        if relationship_types:
            placeholders = ", ".join(["?"] * len(relationship_types))
            rel_filter = f"AND e.relation_type IN ({placeholders})"
            params.extend(relationship_types)
        query = f"""
            WITH RECURSIVE paths(node, depth, path) AS (
                SELECT ?, 0, ARRAY[?::VARCHAR]
                UNION ALL
                SELECT e.target, p.depth + 1, array_append(p.path, e.target)
                FROM paths p
                JOIN memory_edges e ON e.source = p.node {rel_filter}
                WHERE p.depth < ?
                  AND NOT array_contains(p.path, e.target)
            )
            SELECT path, depth
            FROM paths
            WHERE node = ?
            ORDER BY depth
            LIMIT 1
        """
        params += [max_depth, target_id]
        result = self._execute(query, params)
        if result is None:
            return None
        rows = result.fetchall()
        if not rows:
            return None
        return {"path": rows[0][0], "length": rows[0][1]}

    def get_subgraph(
        self,
        center_nodes: List[str],
        depth: int = 2,
        relationship_types: Optional[List[str]] = None,
        node_limit: int = 500,
    ) -> Dict[str, Any]:
        """Return subgraph."""
        return self.graph_traverse(
            center_nodes, max_hops=depth,
            relationship_types=relationship_types,
            node_limit=node_limit,
        )

    
    

    def get_subgraph_arrow(
        self,
        center_nodes: List[str],
        depth: int = 2,
        relationship_types: Optional[List[str]] = None,
        node_limit: int = 500,
    ) -> Tuple[pa.Table, pa.Table]:
        """Returns: (nodes_arrow, edges_arrow)."""
        empty_nodes = pa.table({"uid": [], "raw_data": [], "metadata": []})
        empty_edges = pa.table({"source": [], "target": [], "relation_type": [], "properties": []})
        if not center_nodes:
            return empty_nodes, empty_edges

        sub = self.graph_traverse(
            center_nodes, max_hops=depth,
            relationship_types=relationship_types,
            node_limit=node_limit,
        )
        node_uids = [n["uid"] for n in sub["nodes"]]
        if not node_uids:
            return empty_nodes, empty_edges

        np_str = ", ".join(["?"] * len(node_uids))
        nodes_result = self._execute(
            f"SELECT uid, raw_data, metadata FROM memory_nodes WHERE uid IN ({np_str})",
            node_uids,
        )
        nodes_table = nodes_result.fetch_arrow_table() if nodes_result else empty_nodes

        edges_result = self._execute(
            f"""SELECT source, target, relation_type, properties
                FROM memory_edges
                WHERE source IN ({np_str}) AND target IN ({np_str})""",
            node_uids + node_uids,
        )
        edges_table = edges_result.fetch_arrow_table() if edges_result else empty_edges
        return nodes_table, edges_table

    def get_all_nodes_arrow(self) -> pa.Table:
        result = self._execute(
            "SELECT uid, raw_data, metadata, content_type FROM memory_nodes"
        )
        if result is None:
            return pa.table({"uid": [], "raw_data": [], "metadata": [], "content_type": []})
        return result.fetch_arrow_table()

    def get_all_edges_arrow(self) -> pa.Table:
        result = self._execute(
            "SELECT source, target, relation_type, properties FROM memory_edges"
        )
        if result is None:
            return pa.table({"source": [], "target": [], "relation_type": [], "properties": []})
        return result.fetch_arrow_table()

    
    # L1/L2 Swap In / Swap Out
    

    def swap_in(self, uids: List[str], include_splade_csr: bool = True) -> Dict[str, Any]:
        """Returns: { "dense_matrix": np.ndarray, # (N, dim) float32 "splade_values": Arrow ChunkedArray, }."""
        empty = {
            "units": [], "dense_matrix": np.empty((0, self.embedding_dim), dtype=np.float32),
            "splade_csr": None, "uid_order": [], "edges": [],
        }
        if not uids:
            return empty

        uid_arrow = pa.table({"uid": pa.array(uids, type=pa.utf8())})
        self.con.register("_swap_in_uids", uid_arrow)

        try:
            
            
            result = self.con.execute(
                """SELECT m.uid, m.raw_data, m.metadata, m.dense_embedding,
                          COALESCE(m.splade_indices, []::UBIGINT[]) AS splade_indices,
                          COALESCE(m.splade_values, []::FLOAT[]) AS splade_values
                   FROM memory_nodes m
                   JOIN _swap_in_uids q ON m.uid = q.uid"""
            )
            if result is None:
                return empty

            arrow_table = result.fetch_arrow_table()
        finally:
            self.con.unregister("_swap_in_uids")

        n = arrow_table.num_rows
        if n == 0:
            return empty

        
        dense_arr = arrow_table.column("dense_embedding").combine_chunks()
        dense_null_mask = None
        if dense_arr.null_count > 0:
            dense_null_mask = dense_arr.is_null().to_numpy(zero_copy_only=False)
            
            try:
                zero_scalar = pa.scalar(
                    [0.0] * self.embedding_dim, type=dense_arr.type
                )
                dense_arr = pc.fill_null(dense_arr, zero_scalar)
            except Exception:
                pass

        flat = dense_arr.values.to_numpy(zero_copy_only=False)
        dense_matrix = flat.astype(np.float32, copy=False).reshape(n, self.embedding_dim)

        if dense_null_mask is not None and dense_null_mask.any():
            if not dense_matrix.flags.writeable:
                dense_matrix = dense_matrix.copy()
            dense_matrix[dense_null_mask] = 0.0

        # ==== 2. MemoryUnit: orjson + list comprehension ====
        uid_order = arrow_table.column("uid").to_pylist()
        raw_col = arrow_table.column("raw_data").to_pylist()
        meta_col = arrow_table.column("metadata").to_pylist()

        units = [
            MemoryUnit(
                uid=uid_order[i],
                raw_data=_json_loads(raw_col[i]) if raw_col[i] else {},
                metadata=_json_loads(meta_col[i]) if meta_col[i] else {},
                embedding=(
                    None if (dense_null_mask is not None and dense_null_mask[i])
                    else dense_matrix[i]
                ),
            )
            for i in range(n)
        ]

        splade_csr = None
        if include_splade_csr and SCIPY_AVAILABLE:
            splade_csr = self._build_csr_from_arrow_lists(
                n,
                arrow_table.column("splade_indices"),
                arrow_table.column("splade_values"),
            )
        elif include_splade_csr:
            logger.warning("scipy is unavailable; swap_in will not return a CSR matrix")

        # ==== 4. Edges: Arrow + orjson ====
        uid_arrow_edges = pa.table({"uid": pa.array(uids, type=pa.utf8())})
        self.con.register("_swap_in_edge_uids", uid_arrow_edges)
        try:
            edges_result = self.con.execute(
                """SELECT e.source, e.target, e.relation_type, e.properties
                   FROM memory_edges e
                   WHERE EXISTS (SELECT 1 FROM _swap_in_edge_uids q WHERE e.source = q.uid)
                      OR EXISTS (SELECT 1 FROM _swap_in_edge_uids q WHERE e.target = q.uid)"""
            )
            edges: List[Dict] = []
            if edges_result:
                edge_arrow = edges_result.fetch_arrow_table()
                if edge_arrow.num_rows > 0:
                    e_src = edge_arrow.column("source").to_pylist()
                    e_tgt = edge_arrow.column("target").to_pylist()
                    e_rel = edge_arrow.column("relation_type").to_pylist()
                    e_props = edge_arrow.column("properties").to_pylist()
                    edges = [
                        {"source": e_src[j], "target": e_tgt[j], "type": e_rel[j],
                         "properties": _json_loads(e_props[j]) if e_props[j] else {}}
                        for j in range(edge_arrow.num_rows)
                    ]
        finally:
            self.con.unregister("_swap_in_edge_uids")

        logger.info(
            f"swap_in: {n} nodes, dense={dense_matrix.shape}, "
            f"splade_csr={'None' if splade_csr is None else splade_csr.shape}, "
            f"edges={len(edges)}"
        )
        return {
            "units": units,
            "dense_matrix": dense_matrix,
            "splade_csr": splade_csr,
            "splade_indices": arrow_table.column("splade_indices"),
            "splade_values": arrow_table.column("splade_values"),
            "uid_order": uid_order,
            "edges": edges,
        }

    @staticmethod
    def _build_csr_from_arrow_lists(
        n_rows: int,
        idx_chunked: pa.ChunkedArray,
        val_chunked: pa.ChunkedArray,
    ):
        """Build csr from arrow lists."""
        idx_arr = idx_chunked.combine_chunks()
        val_arr = val_chunked.combine_chunks()

        total_nnz = len(idx_arr.values)
        if total_nnz == 0:
            return None

        indptr = idx_arr.offsets.to_numpy(zero_copy_only=False).astype(np.int32)

        # Arrow flat values → CSR indices & data
        indices = idx_arr.values.to_numpy(zero_copy_only=False).astype(np.int32)
        data = val_arr.values.to_numpy(zero_copy_only=False).astype(np.float32)

        
        n_cols = int(np.max(indices)) + 1

        return scipy_csr_matrix(
            (data, indices, indptr), shape=(n_rows, n_cols)
        )

    def swap_out(self, uids: List[str], l1_data: Dict[str, Any]) -> int:
        """Swap out."""
        if not uids:
            return 0

        nodes_arrow = l1_data.get("nodes_arrow")
        node_columns = l1_data.get("node_columns")
        if nodes_arrow is None and node_columns:
            nodes_arrow = pa.Table.from_pydict(
                node_columns,
                schema=self._memory_nodes_arrow_schema(),
            )
        if nodes_arrow is None:
            nodes_arrow = self._build_legacy_swap_out_nodes_arrow(uids, l1_data)

        n = int(nodes_arrow.num_rows)
        edges = l1_data.get("edges", [])

        success = 0
        try:
            with self.transaction():
                self.con.register("_swap_out_nodes", nodes_arrow)
                try:
                    self.con.execute(
                        "INSERT OR REPLACE INTO memory_nodes SELECT * FROM _swap_out_nodes"
                    )
                    success = n
                finally:
                    self.con.unregister("_swap_out_nodes")

                if edges:
                    edges_arrow = self._build_swap_out_edges_arrow(edges)
                    if edges_arrow.num_rows > 0:
                        self.con.register("_swap_out_edges", edges_arrow)
                        try:
                            self.con.execute(
                                "INSERT OR REPLACE INTO memory_edges SELECT * FROM _swap_out_edges"
                            )
                        finally:
                            self.con.unregister("_swap_out_edges")

            logger.info(f"swap_out: flushed {success} nodes and {len(edges)} edges to L2")
        except Exception as e:
            logger.error(f"swap_out failed: {e}")
            raise
        return success

    @staticmethod
    def _memory_nodes_arrow_schema() -> pa.Schema:
        return pa.schema([
            ("uid", pa.utf8()),
            ("raw_data", pa.utf8()),
            ("metadata", pa.utf8()),
            ("content_type", pa.utf8()),
            ("dense_embedding", pa.list_(pa.float32())),
            ("splade_indices", pa.list_(pa.uint64())),
            ("splade_values", pa.list_(pa.float32())),
            ("bm25_terms", pa.list_(pa.utf8())),
            ("bm25_tfs", pa.list_(pa.int32())),
            ("memory_spaces", pa.list_(pa.utf8())),
            ("embedding_model", pa.utf8()),
            ("created_at", pa.timestamp("us")),
            ("updated_at", pa.timestamp("us")),
            ("access_count", pa.int64()),
        ])

    def _build_legacy_swap_out_nodes_arrow(self, uids: List[str], l1_data: Dict[str, Any]) -> pa.Table:
        units = l1_data.get("units", [])
        dense_matrix = l1_data.get("dense_matrix")
        splade_csr = l1_data.get("splade_csr")
        uid_order = l1_data.get("uid_order", [])
        bm25_per_uid: Dict[str, Tuple[List[str], List[int]]] = l1_data.get("bm25_per_uid", {})

        uid_to_row = {u: i for i, u in enumerate(uid_order)} if uid_order else {}
        uid_to_unit = {u.uid: u for u in units} if units else {}

        now = datetime.now()

        col_uid: List[str] = []
        col_raw: List[str] = []
        col_meta: List[str] = []
        col_ct: List[str] = []
        col_dense: List[Optional[list]] = []
        col_splade_idx: List[Optional[list]] = []
        col_splade_val: List[Optional[list]] = []
        col_bm25_terms: List[Optional[List[str]]] = []
        col_bm25_tfs: List[Optional[List[int]]] = []

        for uid in uids:
            row_idx = uid_to_row.get(uid)
            unit = uid_to_unit.get(uid)

            col_uid.append(uid)
            col_raw.append(_json_dumps(unit.raw_data) if unit else '{}')
            col_meta.append(_json_dumps(unit.metadata or {}) if unit else '{}')
            col_ct.append(self._infer_content_type(unit.raw_data) if unit else 'mixed')

            
            dense = None
            if dense_matrix is not None and row_idx is not None and row_idx < dense_matrix.shape[0]:
                dense = dense_matrix[row_idx].tolist()
            elif unit is not None and unit.embedding is not None:
                dense = unit.embedding.tolist()
            col_dense.append(dense)

            # SPLADE paired arrays
            splade_idx_list, splade_val_list = None, None
            if splade_csr is not None and row_idx is not None and row_idx < splade_csr.shape[0]:
                row_slice = splade_csr.getrow(row_idx)
                if row_slice.nnz > 0:
                    splade_idx_list = row_slice.indices.astype(np.uint64).tolist()
                    splade_val_list = row_slice.data.astype(np.float32).tolist()
            col_splade_idx.append(splade_idx_list)
            col_splade_val.append(splade_val_list)

            
            bm25_data = bm25_per_uid.get(uid)
            if bm25_data is not None:
                col_bm25_terms.append(bm25_data[0])
                col_bm25_tfs.append(bm25_data[1])
            else:
                col_bm25_terms.append(None)
                col_bm25_tfs.append(None)

        n = len(col_uid)
        nodes_dict = {
            "uid": pa.array(col_uid, type=pa.utf8()),
            "raw_data": pa.array(col_raw, type=pa.utf8()),
            "metadata": pa.array(col_meta, type=pa.utf8()),
            "content_type": pa.array(col_ct, type=pa.utf8()),
            "dense_embedding": pa.array(col_dense, type=pa.list_(pa.float32())),
            "splade_indices": pa.array(col_splade_idx, type=pa.list_(pa.uint64())),
            "splade_values": pa.array(col_splade_val, type=pa.list_(pa.float32())),
            "bm25_terms": pa.array(col_bm25_terms, type=pa.list_(pa.utf8())),
            "bm25_tfs": pa.array(col_bm25_tfs, type=pa.list_(pa.int32())),
            "memory_spaces": pa.array([None] * n, type=pa.list_(pa.utf8())),
            "embedding_model": pa.array(["default"] * n, type=pa.utf8()),
            "created_at": pa.array([now] * n, type=pa.timestamp("us")),
            "updated_at": pa.array([now] * n, type=pa.timestamp("us")),
            "access_count": pa.array([0] * n, type=pa.int64()),
        }
        return pa.Table.from_pydict(nodes_dict, schema=self._memory_nodes_arrow_schema())

    def _build_swap_out_edges_arrow(self, edges: List[Any]) -> pa.Table:
        now = datetime.now()
        e_src, e_tgt, e_rel, e_props = [], [], [], []
        for edge in edges:
            if isinstance(edge, dict):
                src = edge.get("source", edge.get("src"))
                tgt = edge.get("target", edge.get("tgt"))
                rel = edge.get("type", edge.get("relation_type", "RELATED"))
                props = _json_dumps(edge.get("properties", {}))
            elif isinstance(edge, (list, tuple)):
                src, tgt = edge[0], edge[1]
                rel = edge[2] if len(edge) > 2 else "RELATED"
                props = _json_dumps(edge[3] if len(edge) > 3 and edge[3] else {})
            else:
                continue
            e_src.append(src)
            e_tgt.append(tgt)
            e_rel.append(rel)
            e_props.append(props)

        return pa.Table.from_pydict({
            "source": pa.array(e_src, type=pa.utf8()),
            "target": pa.array(e_tgt, type=pa.utf8()),
            "relation_type": pa.array(e_rel, type=pa.utf8()),
            "properties": pa.array(e_props, type=pa.utf8()),
            "created_at": pa.array([now] * len(e_src), type=pa.timestamp("us")),
            "updated_at": pa.array([now] * len(e_src), type=pa.timestamp("us")),
        })

    
    

    def add_unit_to_spaces(
        self, unit_id: str, space_names: List[str], operation: str = "add"
    ) -> bool:
        """Add unit to spaces."""
        try:
            result = self._execute(
                "SELECT memory_spaces FROM memory_nodes WHERE uid = ?", [unit_id]
            )
            if result is None:
                return False
            rows = result.fetchall()
            if not rows:
                return False
            current = list(rows[0][0]) if rows[0][0] else []
            if operation == "add":
                updated = list(set(current) | set(space_names))
            elif operation == "remove":
                updated = [s for s in current if s not in space_names]
            else:
                updated = space_names
            self._execute(
                "UPDATE memory_nodes SET memory_spaces = ? WHERE uid = ?",
                [updated, unit_id],
            )
            return True
        except Exception as e:
            logger.error(f"Failed to update memory_spaces ({unit_id}): {e}")
            return False

    def count_units(self, space_names: Optional[List[str]] = None) -> int:
        """Count units."""
        if space_names:
            ph = ", ".join(["?"] * len(space_names))
            result = self._execute(
                f"SELECT count(*) FROM memory_nodes WHERE list_has_any(memory_spaces, [{ph}]::VARCHAR[])",
                space_names,
            )
        else:
            result = self._execute("SELECT count(*) FROM memory_nodes")
        if result is None:
            return 0
        return result.fetchone()[0]

    
    

    def get_node_count(self) -> int:
        result = self._execute("SELECT count(*) FROM memory_nodes")
        return result.fetchone()[0] if result else 0

    def get_edge_count(self) -> int:
        result = self._execute("SELECT count(*) FROM memory_edges")
        return result.fetchone()[0] if result else 0

    def get_relationship_statistics(self) -> Dict[str, Any]:
        result = self._execute(
            "SELECT relation_type, count(*) AS cnt FROM memory_edges GROUP BY relation_type ORDER BY cnt DESC"
        )
        return {r[0]: r[1] for r in result.fetchall()} if result else {}

    def get_node_statistics(self) -> Dict[str, Any]:
        result = self._execute(
            "SELECT content_type, count(*) AS cnt FROM memory_nodes GROUP BY content_type"
        )
        return {r[0]: r[1] for r in result.fetchall()} if result else {}

    def health_check(self) -> Dict[str, Any]:
        return {
            "connected": self.is_connected,
            "db_path": self.db_path,
            "embedding_dim": self.embedding_dim,
            "vss_loaded": self._vss_loaded,
            "pgq_loaded": self._pgq_loaded,
            "node_count": self.get_node_count(),
            "edge_count": self.get_edge_count(),
            "query_stats": self.query_stats.copy(),
        }

    
    

    def clear_database(self, confirm: bool = False) -> bool:
        if not confirm:
            logger.warning("Clearing the database requires confirm=True")
            return False
        try:
            self._execute("DELETE FROM memory_edges")
            self._execute("DELETE FROM memory_nodes")
            logger.info("DuckDB database cleared")
            return True
        except Exception as e:
            logger.error(f"Failed to clear database: {e}")
            return False

    def export_graph_data(self, output_file: str, format: str = "json") -> bool:
        try:
            nodes = self._execute("SELECT uid, raw_data, metadata FROM memory_nodes")
            edges = self._execute("SELECT source, target, relation_type, properties FROM memory_edges")
            data = {
                "nodes": [
                    {"uid": r[0], "raw_data": json.loads(r[1]) if r[1] else {}, "metadata": json.loads(r[2]) if r[2] else {}}
                    for r in (nodes.fetchall() if nodes else [])
                ],
                "edges": [
                    {"source": r[0], "target": r[1], "type": r[2], "properties": json.loads(r[3]) if r[3] else {}}
                    for r in (edges.fetchall() if edges else [])
                ],
            }
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to export graph data: {e}")
            return False

    def import_graph_data(self, input_file: str, clear_existing: bool = False) -> bool:
        try:
            with open(input_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if clear_existing:
                self.clear_database(confirm=True)
            now = datetime.now().isoformat()
            for node in data.get("nodes", []):
                raw_json = json.dumps(node.get("raw_data", {}), ensure_ascii=False)
                meta_json = json.dumps(node.get("metadata", {}), ensure_ascii=False)
                self._execute(
                    "INSERT OR REPLACE INTO memory_nodes (uid, raw_data, metadata, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                    [node["uid"], raw_json, meta_json, now, now],
                )
            for edge in data.get("edges", []):
                props_json = json.dumps(edge.get("properties", {}), ensure_ascii=False)
                self._execute(
                    "INSERT OR REPLACE INTO memory_edges (source, target, relation_type, properties, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                    [edge["source"], edge["target"], edge["type"], props_json, now, now],
                )
            return True
        except Exception as e:
            logger.error(f"Failed to import graph data: {e}")
            return False

    
    

    def _row_to_memory_unit(self, row: tuple) -> Optional[MemoryUnit]:
        """Run row to memory unit."""
        try:
            uid = row[0]
            raw_data = json.loads(row[1]) if row[1] else {}
            metadata = json.loads(row[2]) if row[2] else {}
            embedding = None
            if len(row) > 3 and row[3] is not None:
                embedding = np.array(row[3], dtype=np.float32)
            return MemoryUnit(uid=uid, raw_data=raw_data, metadata=metadata, embedding=embedding)
        except Exception as e:
            logger.error(f"Failed to convert row data to MemoryUnit: {e}")
            return None

    def _parse_neighbor_rows(self, rows: list) -> List[Dict[str, Any]]:
        results = []
        for r in rows:
            entry = {"uid": r[0]}
            if len(r) > 1 and r[1]:
                entry["raw_data"] = json.loads(r[1]) if isinstance(r[1], str) else r[1]
            if len(r) > 2 and r[2]:
                entry["metadata"] = json.loads(r[2]) if isinstance(r[2], str) else r[2]
            if len(r) > 3:
                entry["relation_type"] = r[3]
            if len(r) > 4 and r[4]:
                entry["edge_properties"] = json.loads(r[4]) if isinstance(r[4], str) else r[4]
            results.append(entry)
        return results

    def _infer_content_type(self, raw_data: Dict[str, Any]) -> str:
        if "text_content" in raw_data:
            return "text"
        elif "image_path" in raw_data:
            return "image"
        return "mixed"

    
    
    

    def swap_out_bm25(
        self,
        collection_id: str,
        inverted_index: Dict[str, Dict[str, int]],
        dfs: Dict[str, int],
        doc_lengths: Dict[str, int],
        target_uids: Optional[Set[str]] = None,
    ) -> int:
        """Args: inverted_index: term -> {uid: tf} dfs: term -> document frequency doc_lengths: uid -> doc_len Returns:."""
        if not doc_lengths:
            return 0

        uids_to_swap: Set[str] = target_uids if target_uids else set(doc_lengths.keys())
        if not uids_to_swap:
            return 0

        inv_coll: List[str] = []
        inv_term: List[str] = []
        inv_uid: List[str] = []
        inv_tf: List[int] = []

        for term, postings in inverted_index.items():
            for uid, tf in postings.items():
                if uid in uids_to_swap:
                    inv_coll.append(collection_id)
                    inv_term.append(term)
                    inv_uid.append(uid)
                    inv_tf.append(tf)

        
        related_terms: Set[str] = set()
        for term, postings in inverted_index.items():
            if uids_to_swap.intersection(postings.keys()):
                related_terms.add(term)

        dfs_coll: List[str] = []
        dfs_term: List[str] = []
        dfs_df: List[int] = []
        for term in related_terms:
            if term in dfs:
                dfs_coll.append(collection_id)
                dfs_term.append(term)
                dfs_df.append(dfs[term])

        dl_coll: List[str] = []
        dl_uid: List[str] = []
        dl_len: List[int] = []
        for uid in uids_to_swap:
            if uid in doc_lengths:
                dl_coll.append(collection_id)
                dl_uid.append(uid)
                dl_len.append(doc_lengths[uid])

        # Storage operations must preserve transactional consistency.
        try:
            with self.transaction():
                
                uid_arrow = pa.table({"uid": pa.array(list(uids_to_swap), type=pa.utf8())})
                self.con.register("_bm25_swap_uids", uid_arrow)
                try:
                    self.con.execute(
                        "DELETE FROM bm25_inverted_index "
                        "WHERE collection_id = ? AND uid IN (SELECT uid FROM _bm25_swap_uids)",
                        [collection_id],
                    )
                    self.con.execute(
                        "DELETE FROM bm25_doc_lengths "
                        "WHERE collection_id = ? AND uid IN (SELECT uid FROM _bm25_swap_uids)",
                        [collection_id],
                    )
                finally:
                    self.con.unregister("_bm25_swap_uids")

                if dfs_coll:
                    dfs_arrow = pa.table({
                        "collection_id": pa.array(dfs_coll, type=pa.utf8()),
                        "term": pa.array(dfs_term, type=pa.utf8()),
                        "df": pa.array(dfs_df, type=pa.int32()),
                    })
                    self.con.register("_bm25_swap_dfs", dfs_arrow)
                    try:
                        self.con.execute(
                            "INSERT OR REPLACE INTO bm25_dfs SELECT * FROM _bm25_swap_dfs"
                        )
                    finally:
                        self.con.unregister("_bm25_swap_dfs")

                if inv_coll:
                    inv_arrow = pa.table({
                        "collection_id": pa.array(inv_coll, type=pa.utf8()),
                        "term": pa.array(inv_term, type=pa.utf8()),
                        "uid": pa.array(inv_uid, type=pa.utf8()),
                        "tf": pa.array(inv_tf, type=pa.int32()),
                    })
                    self.con.register("_bm25_swap_inv", inv_arrow)
                    try:
                        self.con.execute(
                            "INSERT OR REPLACE INTO bm25_inverted_index SELECT * FROM _bm25_swap_inv"
                        )
                    finally:
                        self.con.unregister("_bm25_swap_inv")

                if dl_coll:
                    dl_arrow = pa.table({
                        "collection_id": pa.array(dl_coll, type=pa.utf8()),
                        "uid": pa.array(dl_uid, type=pa.utf8()),
                        "doc_len": pa.array(dl_len, type=pa.int32()),
                    })
                    self.con.register("_bm25_swap_dl", dl_arrow)
                    try:
                        self.con.execute(
                            "INSERT OR REPLACE INTO bm25_doc_lengths SELECT * FROM _bm25_swap_dl"
                        )
                    finally:
                        self.con.unregister("_bm25_swap_dl")

            logger.info(
                f"swap_out_bm25: collection={collection_id}, "
                f"docs={len(dl_coll)}, postings={len(inv_coll)}, terms(dfs)={len(dfs_coll)}"
            )
            return len(dl_coll)

        except Exception as e:
            logger.error(f"swap_out_bm25 failed: {e}")
            raise

    def swap_in_bm25(self, collection_id: str) -> Dict[str, Any]:
        """Load BM25 postings and collection statistics from DuckDB.

        Args:
            collection_id: Logical BM25 collection identifier.

        Returns:
            Dictionary containing ``inverted_index``, ``dfs``,
            ``doc_lengths``, ``total_docs``, and ``total_doc_length``.
        """
        empty: Dict[str, Any] = {
            "inverted_index": {},
            "dfs": {},
            "doc_lengths": {},
            "total_docs": 0,
            "total_doc_length": 0,
        }

        try:
            # ---- 1. doc_lengths ----
            dl_result = self._execute(
                "SELECT uid, doc_len FROM bm25_doc_lengths WHERE collection_id = ?",
                [collection_id],
            )
            if dl_result is None:
                return empty
            dl_rows = dl_result.fetchall()
            doc_lengths: Dict[str, int] = {row[0]: row[1] for row in dl_rows}

            if not doc_lengths:
                logger.info(f"swap_in_bm25: collection={collection_id} has no data")
                return empty

            total_docs = len(doc_lengths)
            total_doc_length = sum(doc_lengths.values())

            # ---- 2. dfs ----
            dfs_result = self._execute(
                "SELECT term, df FROM bm25_dfs WHERE collection_id = ?",
                [collection_id],
            )
            dfs: Dict[str, int] = {}
            if dfs_result:
                for row in dfs_result.fetchall():
                    dfs[row[0]] = row[1]

            # ---- 3. inverted_index ----
            inv_result = self._execute(
                "SELECT term, uid, tf FROM bm25_inverted_index WHERE collection_id = ?",
                [collection_id],
            )
            inverted_index: Dict[str, Dict[str, int]] = {}
            if inv_result:
                for row in inv_result.fetchall():
                    term, uid, tf = row[0], row[1], row[2]
                    if term not in inverted_index:
                        inverted_index[term] = {}
                    inverted_index[term][uid] = tf

            logger.info(
                f"swap_in_bm25: collection={collection_id}, "
                f"docs={total_docs}, vocab={len(inverted_index)}, "
                f"total_doc_length={total_doc_length}"
            )
            return {
                "inverted_index": inverted_index,
                "dfs": dfs,
                "doc_lengths": doc_lengths,
                "total_docs": total_docs,
                "total_doc_length": total_doc_length,
            }

        except Exception as e:
            logger.error(f"swap_in_bm25 failed: {e}")
            return empty

    def delete_bm25_collection(self, collection_id: str) -> bool:
        """Remove bm25 collection."""
        try:
            self.con.execute("BEGIN TRANSACTION")
            self._execute(
                "DELETE FROM bm25_inverted_index WHERE collection_id = ?",
                [collection_id],
            )
            self._execute(
                "DELETE FROM bm25_dfs WHERE collection_id = ?",
                [collection_id],
            )
            self._execute(
                "DELETE FROM bm25_doc_lengths WHERE collection_id = ?",
                [collection_id],
            )
            self.con.execute("COMMIT")
            logger.info(f"delete_bm25_collection: {collection_id} cleared")
            return True
        except Exception as e:
            try:
                self.con.execute("ROLLBACK")
            except Exception:
                pass
            logger.error(f"delete_bm25_collection failed: {e}")
            return False

    
    
    

    def update_bm25_global_stats(self, df_deltas: Dict[str, int]) -> None:
        """Update bm25 global stats."""
        if not df_deltas:
            return

        terms = list(df_deltas.keys())
        deltas = [df_deltas[t] for t in terms]

        delta_arrow = pa.table({
            "term": pa.array(terms, type=pa.utf8()),
            "delta": pa.array(deltas, type=pa.int32()),
        })

        with self.transaction():
            self.con.register("_bm25_df_deltas", delta_arrow)
            try:
                self.con.execute("""
                    INSERT INTO bm25_global_stats (term, df)
                    SELECT term, delta FROM _bm25_df_deltas
                    ON CONFLICT (term) DO UPDATE SET df = bm25_global_stats.df + EXCLUDED.df
                """)
                self.con.execute("DELETE FROM bm25_global_stats WHERE df <= 0")
            finally:
                self.con.unregister("_bm25_df_deltas")

    def get_bm25_node_data(self, unit_ids: List[str]) -> Dict[str, Tuple[List[str], List[int]]]:
        """Return bm25 node data."""
        if not unit_ids:
            return {}

        placeholders = ", ".join(["?"] * len(unit_ids))
        result = self._execute(
            f"SELECT uid, bm25_terms, bm25_tfs FROM memory_nodes "
            f"WHERE uid IN ({placeholders}) AND bm25_terms IS NOT NULL",
            unit_ids,
        )
        if result is None:
            return {}

        bm25_data: Dict[str, Tuple[List[str], List[int]]] = {}
        for row in result.fetchall():
            uid, terms, tfs = row[0], row[1], row[2]
            if terms is not None and tfs is not None and len(terms) == len(tfs):
                bm25_data[uid] = (list(terms), list(tfs))
        return bm25_data

    def get_bm25_global_stats(self) -> Dict[str, int]:
        """Return bm25 global stats."""
        result = self._execute("SELECT term, df FROM bm25_global_stats")
        if result is None:
            return {}
        return {row[0]: row[1] for row in result.fetchall()}

    
    

    def close(self):
        if self.con:
            try:
                self.con.close()
            except Exception:
                pass
            self.con = None
            self.is_connected = False
            logger.info("DuckDB connection closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __repr__(self):
        return (
            f"DuckDBOperator(db_path='{self.db_path}', dim={self.embedding_dim}, "
            f"vss={self._vss_loaded}, pgq={self._pgq_loaded}, "
            f"nodes={self.get_node_count()}, edges={self.get_edge_count()})"
        )

    def __repr__(self):
        return (
            f"DuckDBOperator(db_path='{self.db_path}', "
            f"connected={self.is_connected}, "
            f"nodes={self.get_node_count() if self.is_connected else '?'}, "
            f"edges={self.get_edge_count() if self.is_connected else '?'})"
        )
