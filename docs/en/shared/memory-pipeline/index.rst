Memory Building and Retrieval Pipeline
========================================

Mandol's core pipeline has only three steps: **add memory → build high-level semantics → retrieve**. No matter what level of user you are, you need to understand this pipeline. This chapter provides three levels of pipeline descriptions — choose the depth that suits your needs.

.. toctree::
   :maxdepth: 1

   basic-flow
   detailed-flow
   architecture-flow

Pipeline Overview
-----------------

::

   add()                       build_high_level()              holistic_retrieve()
   ─────                       ──────────────────              ───────────────────
   Raw dialogue                Session segmentation            Four group recalls
     ↓                           ↓                               (BASE/ENTITY
   Chunking (split long text)  Space layout                      /EVENT/SUMMARY)
     ↓                           ↓                                  ↓
   Vectorization + Storage     4-category summary Map-Reduce    Three-path retrieval
     ↓                           (Episodic/Knowledge             (Dense/BM25/Sparse)
   Similarity edges               /Emotional/Procedural)          ↓
     ↓                           ↓                              RRF fusion
   Pending queue                Entity/Event/Relation             ↓
                                  extraction                    BFS graph expansion
                                  ↓                               ↓
                               Insight distillation             Rerank
                                  ↓                               ↓
                               Cross-session merging            SearchHit[]
                                  ↓
                               Global insight accumulation

.. note::

   - **Basic users**: Read :doc:`basic-flow` for the three-step pipeline and what ``build_high_level()`` does internally
   - **Advanced users**: Read :doc:`detailed-flow` for the complete mechanism of each sub-stage and tunable parameters
   - **Developers**: Read :doc:`architecture-flow` for architecture layers and extension/customization points at each stage
