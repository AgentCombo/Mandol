ADR-001: Hexagonal Architecture Choice
==========================================

Status
-------

Accepted

Date
-----

2024-06

Context
--------

Mandol needs to support multiple infrastructure components:
- Vector indexes: FAISS, Milvus, Elasticsearch
- Embedding models: OpenAI, SentenceTransformers, Custom
- Storage: In-memory, SQLite, Remote databases
- Graph stores: In-memory, Neo4j

Decision
---------

Adopt a hexagonal (ports-and-adapters) architecture. Core business logic is defined in ``application/``, depending only on abstract interfaces defined in ``ports/``. Concrete implementations are placed in ``infrastructure/``.

Consequences
-------------

**Positive**:

- Replaceability: Adding new storage/index/model doesn't require modifying business logic
- Testability: Ports can be mocked in tests
- Clear dependency rules: New developers can easily understand dependency direction

**Negative**:

- Indirection: Code navigation path is longer (application → ports → infrastructure)
- More files: One abstract file + one implementation file per port
