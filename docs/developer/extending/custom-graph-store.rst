自定义 GraphStore
======================

实现 ``GraphStore`` 接口，可替换内存存储为 Neo4j 等。

.. code-block:: python

   from mandol.ports.graph_store import GraphStore
   from mandol.domain.types import Uid

   class Neo4jGraphStore(GraphStore):
       def __init__(self, uri, user, password):
           self.driver = GraphDatabase.driver(uri, auth=(user, password))

       def get_neighbors(self, uid, rel_type=None, direction="both"):
           with self.driver.session() as session:
               return session.execute_read(self._get_neighbors, uid, rel_type, direction)

       def upsert_relationship(self, source, target, rel_type, props=None):
           with self.driver.session() as session:
               session.execute_write(self._upsert_rel, source, target, rel_type, props)

       def delete_relationship(self, source, target, rel_type=None):
           with self.driver.session() as session:
               session.execute_write(self._delete_rel, source, target, rel_type)

       def get_relationships(self, uid, rel_type=None):
           with self.driver.session() as session:
               return session.execute_read(self._get_rels, uid, rel_type)

       def get_all_relationships(self):
           with self.driver.session() as session:
               return session.execute_read(self._get_all_rels)

       def flush(self):
           pass  # Neo4j 自带持久化

注入
----

.. code-block:: python

   from mandol.application.semantic_graph import SemanticGraphService

   graph_service = SemanticGraphService(store=Neo4jGraphStore(uri, user, password))
