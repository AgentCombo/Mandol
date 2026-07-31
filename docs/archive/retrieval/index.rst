检索接口总览
============

本节介绍 Mandol 记忆系统的检索接口分类，从底层数据结构到多视角记忆，再到全记忆统一检索，按层级递进说明。

检索接口分类
------------

Mandol 的检索接口按层级分为以下几类：

1. **MemoryUnit 接口**：单个记忆单元的访问与管理
2. **SemanticMap 接口**：向量空间的语义检索
3. **SemanticGraph 接口**：基于图关系的检索
4. **多视角记忆接口**：面向对话数据集的多维度检索
5. **全记忆统一检索**：跨空间、跨视角的综合检索
6. **检索模块公开接口**：可扩展的检索器框架

接口命名约定
------------

- **公开接口**：无前缀，面向用户直接调用（如 ``get_unit``、``holistic_retrieve``）
- **内部接口**：下划线前缀 ``_``，系统内部调用，不暴露给用户（如 ``_bfs_expand_units``）

.. toctree::
   :maxdepth: 2

   memory-unit-interfaces
   semantic-map-interfaces
   semantic-graph-interfaces
   multi-view-interfaces
   internal-interfaces
   holistic-retrieve
   retrieval-module
