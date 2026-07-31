Developer Guide
=================

This section is for developers who need to understand the architecture, extend the system, or contribute code.

Module Map
----------

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - I want to...
     - Description
     - Go to
   * - Understand system architecture
     - Hexagonal architecture layers, component relationships, ADRs
     - :doc:`architecture/index`
   * - Browse API reference
     - Complete signatures and examples for all public interfaces
     - :doc:`api-reference/index`
   * - Extend the system
     - Custom Embedder/Reranker/LLM/GraphStore/Dimension
     - :doc:`extending/index`
   * - Contribute code
     - Development environment setup / testing / PR process
     - :doc:`contributing/index`
   * - Debug and diagnose
     - Trace retrieval pipelines / Profiling / Common pitfalls
     - :doc:`debugging/index`
   * - Deep-dive examples
     - LoCoMo wrapper layer / LongMemEval pipeline
     - :doc:`examples/locomo-architecture`
   * - Multimodal evaluation
     - Multimodal capability assessment report
     - :doc:`multimodal/capability-assessment`

.. toctree::
   :maxdepth: 2
   :hidden:

   architecture/index
   api-reference/index
   extending/index
   contributing/index
   debugging/index
   examples/locomo-architecture
   examples/longmemeval-pipeline
   multimodal/capability-assessment
