Reproduction and Benchmarks
===========================

Branch roles
------------

The ``main`` branch contains the current maintained Mandol Python runtime and
documentation. The results reported in the paper were produced with the frozen
``paper-repro`` branch. Use that branch, rather than the evolving main-branch
workflows, for exact comparison with the published tables.

* `LoCoMo paper reproduction <https://github.com/AgentCombo/Mandol/blob/paper-repro/benchmark_locomo/REPRODUCE.md>`_
* `LongMemEval paper reproduction <https://github.com/AgentCombo/Mandol/blob/paper-repro/benchmark_longmemeval/REPRODUCE.md>`_

The ``legacy/original`` branch preserves the implementation that occupied
``main`` before the repository migration. It is historical reference material,
not the current API and not the paper-reproduction entry point.

Main-branch workflows
---------------------

The main checkout retains ``benchmark_locomo`` and ``benchmark_longmemeval``
for integration with the current package. Their primary router + quantification
entry points are:

.. code-block:: text

   benchmark_locomo/task_eval/locomo_triple_router_quantification.py
   benchmark_longmemeval/task_eval/benchmark_triple_router_quantification.py

The ``benchmark_self_host`` directory contains development workflows that use
Mandol's high-level memory builders directly:

.. code-block:: text

   benchmark_self_host/locomo10/
   benchmark_self_host/longmemeval/

These self-host workflows support smoke testing, integration validation and
continued workflow development. They are not the frozen pipelines used to
produce the paper tables.

Performance scope
-----------------

The public LoCoMo performance entry points are:

.. code-block:: text

   benchmark_locomo/task_eval/locomo_triple_input_speed.py
   benchmark_locomo/task_eval/locomo_triple_smart_search_qps.py

The insertion benchmark times the ``SemanticGraph.add_unit`` body, including
dense and realtime SPLADE embedding generation plus incremental index updates.
Scheduling sleep, graph initialization, warmup and report writing are outside
that interval. When RocksDB tiered paging is enabled, candidate selection and
eviction scheduling occur in the add path, while payload persistence and
resident-cache removal may finish asynchronously.

The smart-search benchmark times retrieval, fusion and configured reranking
after graph loading and warmup. Fixed-QPS scheduling sleep and report writing
are excluded. Consult the frozen ``paper-repro`` guides for the exact commands,
models, datasets and parameters used for the reported results.
