ADR-002: SemanticMapService 命名
=======================================

状态
----

已采纳 (Accepted)

日期
----

2024-08

上下文
------

核心服务类 ``SemanticMapService`` 命名曾引发讨论：为什么不叫 ``SemanticMap``？为什么不合并到 ``MemorySystem``？

决策
----

- ``SemanticMapService`` 而非 ``SemanticMap``：``SemanticMap`` 是领域概念（类似 DDD 的 Entity），``Service`` 后缀表示它是应用层服务
- 独立于 ``MemorySystem``：遵循单一职责原则。MemorySystem 编排，SemanticMapService 管理单元和索引，SemanticGraphService 管理关系

后果
----

- 三个类（MemorySystem, SemanticMapService, SemanticGraphService）的职责边界清晰
- 对外暴露复杂度适中：基础用户只用 MemorySystem，高级用户可访问 SemanticMapService 和 SemanticGraphService
