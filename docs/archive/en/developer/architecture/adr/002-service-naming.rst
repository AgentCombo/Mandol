ADR-002: SemanticMapService Naming
========================================

Status
-------

Accepted

Date
-----

2024-08

Context
--------

The naming of the core service class ``SemanticMapService`` sparked discussion: Why not ``SemanticMap``? Why not merge it into ``MemorySystem``?

Decision
---------

- ``SemanticMapService`` instead of ``SemanticMap``: ``SemanticMap`` is a domain concept (similar to DDD Entity), the ``Service`` suffix indicates it's an application layer service
- Independent from ``MemorySystem``: Follows the single responsibility principle. MemorySystem orchestrates, SemanticMapService manages units and indexes, SemanticGraphService manages relationships

Consequences
-------------

- Clear responsibility boundaries for the three classes (MemorySystem, SemanticMapService, SemanticGraphService)
- Moderate external complexity: Basic users only use MemorySystem, advanced users can access SemanticMapService and SemanticGraphService
