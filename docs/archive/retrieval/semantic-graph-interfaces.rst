SemanticGraph 检索接口
==========================

以下接口由 ``SemanticGraph`` 提供，用于基于图关系的检索。

适用于 SemanticGraph 的接口
---------------------------

以下接口由 ``SemanticGraph`` 提供，用于基于图关系的检索。

get_explicit_neighbors
^^^^^^^^^^^^^^^^^^^^^^

获取指定单元的显式关系邻居（如实体关系、事件因果等）。

**签名**：

.. code-block:: python

   def get_explicit_neighbors(
       uids: List[str],
       rel_type: Optional[str] = None,
       direction: str = "successors"
   ) -> List[MemoryUnit]

**参数**：

- ``uids``：源单元 UID 列表
- ``rel_type``：关系类型过滤（如 ``"CAUSES"``、``"RELATED_TO"``）
- ``direction``：检索方向，可选 ``"successors"``、``"predecessors"``、``"all"``

**使用示例**：

.. code-block:: python

   # 获取因果关系的后继邻居
   causes = system.graph.get_explicit_neighbors(
       [event_uid],
       rel_type="CAUSES",
       direction="successors"
   )

get_implicit_neighbors
^^^^^^^^^^^^^^^^^^^^^^

获取指定单元的隐式语义邻居（基于向量相似度）。

**签名**：

.. code-block:: python

   def get_implicit_neighbors(
       uids: List[str],
       top_k: int = 5,
       ms_names: Optional[List[str]] = None
   ) -> List[MemoryUnit]

**使用示例**：

.. code-block:: python

   # 获取语义相似的邻居
   neighbors = system.graph.get_implicit_neighbors([unit_uid], top_k=5)

get_edges_of_unit
^^^^^^^^^^^^^^^^^

获取指定节点的所有关系边，可按类型和方向筛选。

**签名**：

.. code-block:: python

   def get_edges_of_unit(
       uid: str,
       rel_type: Optional[str] = None,
       direction: str = "all"
   ) -> List[dict]

**使用示例**：

.. code-block:: python

   # 获取某实体的所有关系
   edges = system.graph.get_edges_of_unit("entity_001")

   # 仅获取特定类型的关系
   rel_edges = system.graph.get_edges_of_unit(
       "entity_001", rel_type="RELATED_TO"
   )

search_graph_relations
^^^^^^^^^^^^^^^^^^^^^^

搜索图中的关系边，支持以种子节点做 BFS 扩展或遍历全图。

**签名**：

.. code-block:: python

   def search_graph_relations(
       seed_nodes: Optional[List[str]] = None,
       relation_types: Optional[List[str]] = None,
       max_depth: int = 2,
       limit: int = 50
   ) -> List[Tuple[str, str, Dict[str, Any]]]

**使用示例**：

.. code-block:: python

   # 从种子节点扩展搜索关系
   relations = system.graph.search_graph_relations(
       seed_nodes=["entity_001"],
       relation_types=["RELATED_TO", "CAUSES"],
       max_depth=2
   )

get_node_neighbors
^^^^^^^^^^^^^^^^^^

获取节点的综合邻居（语义邻居 + 结构邻居）。

**签名**：

.. code-block:: python

   def get_node_neighbors(
       node_uid: str,
       max_depth: int = 2,
       include_semantic: bool = True,
       include_structural: bool = True,
       similarity_threshold: float = 0.7
   ) -> Dict[str, List[MemoryUnit]]

.. _retrieval-multi-view:
