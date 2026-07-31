SemanticGraphService 完整参考
=====================================

负责记忆单元间的关系建模和图遍历。

构造
----

通过 ``system.semantic_graph`` 访问。

全部公开方法
------------

节点管理：

- ``add_unit(uid: Uid) -> None``
- ``delete_unit(uid: Uid) -> None``

关系管理：

- ``add_relationship(source: Uid, target: Uid, rel_type: str, props=None) -> None``
- ``get_relationship(source: Uid, target: Uid, rel_type: str) -> dict | None``
- ``delete_relationship(source: Uid, target: Uid, rel_type: str | None = None) -> None``
- ``list_relationships(uid: Uid, direction: str = "both") -> list[dict]``

查询：

- ``get_explicit_neighbors(uid: Uid, rel_type=None, direction="both") -> list[Uid]``
- ``get_implicit_neighbors(uid: Uid) -> list[Uid]``
- ``get_units_in_spaces(space_names: list[str]) -> list[Uid]``

图遍历：

- ``bfs_expand_units(seeds: list[Uid], per_seed=3, hops=1) -> list[Uid]``

维护：

- ``flush() -> None``
- ``get_graph_store()``

高级：

- ``get_connected_components() -> list[list[str]]``
- ``get_subgraph(uids: list[Uid], hops=1) -> dict``

使用示例
--------

.. code-block:: python

   system.semantic_graph.add_relationship(
       Uid("msg_1"), Uid("msg_2"), "RELATED_TO"
   )

   neighbors = system.semantic_graph.get_explicit_neighbors(Uid("msg_1"))
   expanded = system.semantic_graph.bfs_expand_units([Uid("msg_1")])

图结构组成
----------

SemanticGraph 由显式关系和隐式语义关系两部分组成：

显式关系
~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 40 42

   * - 边类型
     - 说明
     - 子类型
   * - ``RELATED_TO``
     - 实体关系边
     - hometown、lives_in、works_at、located_in、part_of
   * - ``CAUSES``
     - 事件因果关系（A 导致 B）
     -
   * - ``CAUSED_BY``
     - 事件因果关系（B 被 A 导致）
     -
   * - ``INVOLVES``
     - 事件-实体边
     - participant、location、organizer、victim
   * - ``EVIDENCED_BY``
     - 溯源边（指向原始对话单元）
     -
   * - ``COREF``
     - 共指边（对话单元 → 全局实体的指代关系）
     -
   * - ``ALIAS_OF``
     - 别名边（实体别名关系）
     -
   * - ``PRECEDES``
     - 时序边（对话/事件的时间先后顺序）
     -
   * - ``FOLLOWS``
     - 时序边（对话/事件的时间先后顺序）
     -

隐式语义关系
~~~~~~~~~~~~

- 基于向量相似度构建的 ``SEMANTIC_SIMILAR`` 边
- 在 ``add()`` 时通过 ``_build_immediate_similarity_edges()`` 增量构建
- 边的权重为向量余弦相似度

各边类型在高阶记忆中的具体使用方式，见 :doc:`/shared/memory-pipeline/detailed-flow` 中的「多视角记忆表示」章节。
