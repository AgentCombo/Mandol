内部检索接口
==================

以下接口为系统内部调用，不暴露给最终用户。

.. caution:: 🔧 内部接口 — 以下接口为系统内部调用，不暴露给最终用户。


内部检索接口（下划线前缀，不暴露给用户）
------------------------------------------

以下接口是系统内部调用的复杂检索方法，使用下划线前缀表示。

_bfs_expand_units
^^^^^^^^^^^^^^^^^

从种子单元开始 BFS 扩展，用于发现关联记忆。

**签名**：

.. code-block:: python

   def _bfs_expand_units(
       seeds: List[Union[MemoryUnit, str]],
       hops: int = 1,
       rel_type: Optional[str] = None
   ) -> List[MemoryUnit]

_rrf_fusion
^^^^^^^^^^^

倒数排名融合，合并多路检索结果。

**签名**：

.. code-block:: python

   def _rrf_fusion(
       result_lists: List[List[Tuple[MemoryUnit, float]]],
       k: int = 60,
       top_k: Optional[int] = None
   ) -> List[Tuple[MemoryUnit, float]]

_triple_retrieval_rrf
^^^^^^^^^^^^^^^^^^^^^

三路召回（Dense + BM25 + Sparse）→ RRF 融合。

**签名**：

.. code-block:: python

   def _triple_retrieval_rrf(
       memory_space: MemorySpace,
       query: str,
       top_k: int,
       per_method_k_multiplier: int = 3,
       rrf_k: int = 60,
       semantic_map: Optional[SemanticMap] = None
   ) -> List[Tuple[MemoryUnit, float]]

_triple_retrieval_rrf_bfs_rerank
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

三路召回 → RRF 融合 → BFS 扩展 → 重排序。

**签名**：

.. code-block:: python

   def _triple_retrieval_rrf_bfs_rerank(
       memory_space: MemorySpace,
       query: str,
       top_k: int,
       graph: SemanticGraph,
       per_method_k_multiplier: int = 3,
       rrf_k: int = 60,
       bfs_seed_n: int = 5,
       bfs_per_seed: int = 3,
       bfs_hops: int = 1,
       rerank_device: str = "remote",
       semantic_map: Optional[SemanticMap] = None
   ) -> List[Tuple[MemoryUnit, float]]

.. _retrieval-holistic:
