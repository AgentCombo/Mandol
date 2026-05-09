ADR-001: 六边形架构选择
============================

状态
----

已采纳 (Accepted)

日期
----

2024-06

上下文
------

Mandol 需要支持多种基础设施组件：
- 向量索引：FAISS, Milvus, Elasticsearch
- 嵌入模型：OpenAI, SentenceTransformers, 自定义
- 存储：内存, SQLite, 远程数据库
- 图存储：内存, Neo4j

决策
----

采用六边形（端口-适配器）架构。核心业务逻辑定义在 ``application/`` 中，只依赖 ``ports/`` 中定义的抽象接口。具体实现放在 ``infrastructure/`` 中。

后果
----

**正向**：

- 可替换性：添加新的存储/索引/模型无需修改业务逻辑
- 可测试性：端口可以在测试中用 mock 实现
- 依赖规则清晰：新开发者容易理解依赖方向

**负向**：

- 间接性：代码跳转路径变长（application → ports → infrastructure）
- 文件数增多：每个端口一个抽象文件 + 一个实现文件
