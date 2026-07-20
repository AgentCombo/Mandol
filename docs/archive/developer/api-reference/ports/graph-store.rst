GraphStore
==================

.. code-block:: python

   class GraphStore(ABC):
       def get_neighbors(
           self, uid: Uid, rel_type=None, direction="both"
       ) -> list[Uid]: ...

       def upsert_relationship(
           self, source: Uid, target: Uid, rel_type: str, props=None
       ) -> None: ...

       def delete_relationship(
           self, source: Uid, target: Uid, rel_type=None
       ) -> None: ...

       def get_relationships(
           self, uid: Uid, rel_type=None
       ) -> list[dict]: ...

       def get_all_relationships(self) -> list[dict]: ...

       def flush(self) -> None: ...

实现：``InMemoryGraphStore``。
