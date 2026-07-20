Extension Guide
===============

.. note::

   This document has been migrated to :doc:`/developer/extending/index`. This page will be removed in a future version, please update your bookmarks.

This section describes how to extend the memory system, including custom components and adding new features.

Custom Embedding Provider
-------------------------

To use a custom Embedding model, implement the ``EmbeddingProvider`` interface.

Interface Definition
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from mandol.ports.embedding_provider import EmbeddingProvider
   import numpy as np
   from typing import List

   class MyEmbedder(EmbeddingProvider):
       def embedding_dim(self) -> int:
           """Return vector dimension"""
           return 768

       def embed_text(self, texts: List[str]) -> List[np.ndarray]:
           """Convert text list to vector list"""
           pass

       def embed_image_paths(self, paths: List[str]) -> List[np.ndarray]:
           """Convert image path list to vector list (optional)"""
           raise NotImplementedError("Image embedding not supported")

Using Custom Provider
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from mandol.application.memory_system import MemorySystem, ServiceConfig
   from mandol.infrastructure.config import AppConfig

   config = AppConfig(
       embedder_model="my_custom_embedder",
       embedder_device="cuda",
       embedder_dim=768,
   )

   system = MemorySystem(config=config)

Custom Graph Store
------------------

To switch graph storage from in-memory to Neo4j or other graph databases, implement the ``GraphStore`` interface.

Interface Definition
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from mandol.ports.graph_store import GraphStore
   from mandol.domain.types import Uid
   from typing import List, Dict, Any, Optional

   class MyGraphStore(GraphStore):
       def get_neighbors(
           self,
           uid: Uid,
           rel_type: Optional[str] = None,
           direction: str = "both"
       ) -> List[Uid]:
           """Get neighbor nodes"""
           pass

       def upsert_relationship(
           self,
           source: Uid,
           target: Uid,
           rel_type: str,
           props: Optional[Dict[str, Any]] = None
       ) -> None:
           """Add or update relationship"""
           pass

       def delete_relationship(
           self,
           source: Uid,
           target: Uid,
           rel_type: str
       ) -> None:
           """Delete relationship"""
           pass

       def get_relationships(
           self,
           uid: Uid,
           rel_type: Optional[str] = None
       ) -> List[Dict[str, Any]]:
           """Get relationship list"""
           pass

       def get_all_relationships(self) -> List[Dict[str, Any]]:
           """Get all relationships"""
           pass

Adding New Dimension Builders
-----------------------------

To add a new memory dimension, implement the ``DimensionBuilder`` interface and register it in the construction pipeline.

Interface Definition
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from mandol.application.multidim_semantic_graph import (
       DimensionBuilder,
       MultiDimBuildContext,
   )

   class MyCustomDimension(DimensionBuilder):
       name = "my_custom_dim"

       def build(self, ctx: MultiDimBuildContext) -> None:
           """Dimension build logic"""
           session_idx = ctx.session_idx
           naming_policy = ctx.naming_policy
           space_index = ctx.space_index
           unit_store = ctx.unit_store
           graph_store = ctx.graph_store
           embedder = ctx.embedder
           llm = ctx.llm
           pass

Registering Dimension Builder
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from mandol.application.multidim_semantic_graph import MultiDimSemanticGraph

   builder = MultiDimSemanticGraph(
       graph_service=system.graph,
       space_naming_policy=space_naming_policy,
       dimension_builders=[
           LayoutNormalizationDimension(),
           SemanticSimilarityDimension(),
           HighLevelSummaryApplicatorDimension(),
           EventCausalApplicatorDimension(),
           EntityRelationApplicatorDimension(),
           MyCustomDimension(),
       ],
   )

Custom Reranker
---------------

To customize the reranking model, implement the ``Reranker`` interface.

Interface Definition
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from mandol.ports.reranker import Reranker
   from mandol.domain.memory_unit import MemoryUnit
   from mandol.retrieval.types import SearchHit
   from typing import List

   class MyReranker(Reranker):
       def rerank(
           self,
           query: str,
           results: List[SearchHit]
       ) -> List[SearchHit]:
           """Rerank retrieval results"""
           pass

Custom LLM Provider
-------------------

To use a custom LLM service, implement the ``LLMProvider`` interface.

Interface Definition
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from mandol.ports.llm_provider import LLMProvider, ChatMessage
   from typing import List, Optional

   class MyLLMProvider(LLMProvider):
       def chat_completion(
           self,
           messages: List[ChatMessage],
           max_tokens: Optional[int] = None,
           temperature: Optional[float] = None,
           **kwargs
       ) -> str:
           """Complete dialogue generation"""
           pass

Extension Tips
--------------

- **Test First**: All custom components should have corresponding unit tests
- **Interface Compatibility**: Ensure all interface methods are implemented to avoid runtime errors
- **Performance Considerations**: For large-scale data, consider using batch operations and async processing
- **Logging**: Add logging at key operations for debugging and monitoring
