"""Neo4j-backed implementation of the GraphStore port.

Stores directed relationships as Neo4j graph edges with batched
write-behind buffering. Nodes are automatically created by MERGE.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase

from ..domain.types import Uid
from ..ports.graph_store import GraphStore
from .config import Neo4jConfig


class Neo4jGraphStore(GraphStore):
    """Graph store backed by a Neo4j graph database.

    Relationships are buffered in pending lists and flushed to Neo4j
    on explicit flush() calls. Nodes are named by their Uid; edges
    carry a type label and a properties dictionary.

    Attributes:
        _cfg: Neo4j connection configuration.
        _driver: Neo4j Python driver instance.
        _pending_upserts: Batched edge inserts (source, target, type, props).
        _pending_deletes: Batched edge deletes (source, target, type).
    """

    def __init__(self, *, config: Optional[Neo4jConfig] = None):
        self._cfg = config or Neo4jConfig()
        self._driver = GraphDatabase.driver(
            self._cfg.uri, auth=(self._cfg.user, self._cfg.password)
        )
        self._pending_upserts: List[tuple[Uid, Uid, str, Dict[str, Any]]] = []
        self._pending_deletes: List[tuple[Uid, Uid, Optional[str]]] = []

    def upsert_relationship(
        self, source: Uid, target: Uid, rel_type: str, properties: Dict[str, Any]
    ) -> None:
        """Buffer an edge upsert for the next flush.

        Args:
            source: Source node Uid.
            target: Target node Uid.
            rel_type: Relationship type string.
            properties: Key-value edge metadata.
        """
        self._pending_upserts.append(
            (Uid(str(source)), Uid(str(target)), str(rel_type), dict(properties))
        )

    def delete_relationship(
        self, source: Uid, target: Uid, rel_type: Optional[str] = None
    ) -> None:
        """Buffer an edge deletion for the next flush.

        Args:
            source: Source node Uid.
            target: Target node Uid.
            rel_type: If provided, only delete edges of this type.
        """
        self._pending_deletes.append((Uid(str(source)), Uid(str(target)), rel_type))

    def get_relationship(
        self, source: Uid, target: Uid, rel_type: str
    ) -> Optional[Dict[str, Any]]:
        """Query a specific relationship directly from Neo4j.

        Args:
            source: Source node Uid.
            target: Target node Uid.
            rel_type: Relationship type to look up.

        Returns:
            Edge properties dict if found, or None.
        """
        s = str(source)
        t = str(target)
        rt = str(rel_type)
        query = (
            "MATCH (a {uid: $s})-[r]->(b {uid: $t}) "
            "WHERE type(r) = $rt RETURN properties(r) AS props LIMIT 1"
        )
        with self._driver.session(database=self._cfg.database) as sess:
            rec = sess.run(query, s=s, t=t, rt=rt).single()
            if not rec:
                return None
            props = rec.get("props")
            return dict(props) if isinstance(props, dict) else {}

    def get_neighbors(
        self, uid: Uid, *, rel_type: Optional[str] = None, direction: str = "out"
    ) -> List[Uid]:
        """Return neighbors reachable via outgoing or incoming edges.

        Args:
            uid: The node whose neighbors are requested.
            rel_type: Optional filter for edge type.
            direction: \"out\" for successors, \"in\" for predecessors.

        Returns:
            List of neighbor Uids.

        Raises:
            ValueError: If direction is not \"out\" or \"in\".
        """
        u = str(uid)
        if direction not in {"out", "in"}:
            raise ValueError("direction must be 'out' or 'in'")

        if direction == "out":
            pat = "(a {uid: $u})-[r]->(b)"
            ret = "b.uid as uid"
        else:
            pat = "(a)-[r]->(b {uid: $u})"
            ret = "a.uid as uid"

        where = "" if rel_type is None else "WHERE type(r) = $rt"
        query = f"MATCH {pat} {where} RETURN {ret}"
        params = {"u": u}
        if rel_type is not None:
            params["rt"] = str(rel_type)

        out: List[Uid] = []
        with self._driver.session(database=self._cfg.database) as sess:
            for rec in sess.run(query, **params):
                v = rec.get("uid")
                if v is not None:
                    out.append(Uid(str(v)))
        return out

    def flush(self) -> None:
        """Write all buffered upserts and deletes to Neo4j in a single session."""
        if not self._pending_upserts and not self._pending_deletes:
            return

        with self._driver.session(database=self._cfg.database) as sess:
            for s, t, rel_type in self._pending_deletes:
                if rel_type is None:
                    q = (
                        "MATCH (a {uid: $s})-[r]->(b {uid: $t}) DELETE r"
                    )
                    sess.run(q, s=str(s), t=str(t))
                else:
                    q = (
                        "MATCH (a {uid: $s})-[r]->(b {uid: $t}) "
                        "WHERE type(r) = $rt DELETE r"
                    )
                    sess.run(q, s=str(s), t=str(t), rt=str(rel_type))

            for s, t, rt, props in self._pending_upserts:
                q = (
                    "MERGE (a {uid: $s}) "
                    "MERGE (b {uid: $t}) "
                    f"MERGE (a)-[r:{rt}]->(b) "
                    "SET r += $props"
                )
                sess.run(q, s=str(s), t=str(t), props=props)

        self._pending_upserts.clear()
        self._pending_deletes.clear()
