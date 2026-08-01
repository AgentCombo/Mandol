Mandol Paper Artifact Documentation
===================================

This documentation tracks the frozen ``paper-repro`` checkout used for the
experiments reported in the Mandol paper. Its public surface is based on
``MemoryUnit``, ``MemorySpace``, ``SemanticMap``, ``SemanticGraph``,
``MultiRetriever`` and the retrieval-facing subpackages in this branch.

Current runtime development and the hosted documentation at
https://agentcombo.github.io/Mandol/docs/ are generated from ``main``. Do not
substitute main-branch commands or defaults when reproducing the paper tables.

Historical pages from earlier architecture experiments are preserved under
``docs/archive/`` for design provenance. They are excluded from the Sphinx
build and are not an API contract for the current package.

.. toctree::
   :maxdepth: 2
   :caption: Current Code Guide

   current/installation
   current/quickstart
   current/data-structures
   current/retrieval
   current/persistence
   current/configuration
   current/reproduction
   current/api-reference

Indices
-------

* :ref:`genindex`
* :ref:`search`
